"""
LUMOS Assistive AI — Environment Logic
========================================
Architecture: POMDP (Partially Observable Markov Decision Process)

Hidden world state (WorldState) is richer than agent observations.
Observations are noisy, partial projections of the world state.
Actions trigger explicit state transitions with defined rules.
Multiple valid trajectories exist to achieve success.
Latency is penalised — agent must be efficient, not exhaustive.

State transition rules are deterministic + stochastic:
  - Deterministic: action type determines what aspect of world changes
  - Stochastic: clarity gains, noise, ASL confusion are randomly sampled
    from defined distributions, seeded per-episode for reproducibility.
"""
from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# models.py lives at root level (matching OpenEnv reference pattern)
try:
    from models import Action, Observation  # when run as package from root
    from server.scene_generator import SceneGenerator
except ImportError:
    from server.models import Action, Observation  # fallback for direct execution
    from server.scene_generator import SceneGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 10                   # generous budget, but latency penalty applies
CLARITY_THRESHOLD = 0.45         # min per-object clarity to "see" it clearly
SCORE_MIN = 0.01
SCORE_MAX = 0.99
LATENCY_PENALTY = 0.015          # deducted every step (time cost to user)

# ASL confusion matrix: visually similar letter pairs under noise
ASL_CONFUSION_PAIRS: Dict[str, str] = {
    "A": "M", "M": "A",
    "E": "S", "S": "E",
    "U": "V", "V": "U",
    "R": "U",
    "G": "H", "H": "G",
    "I": "J",
    "K": "P", "P": "K",
    "N": "M", "T": "D", "D": "T",
}

# ---------------------------------------------------------------------------
# Scenario Bank
# ---------------------------------------------------------------------------

VALID_ACTIONS: Dict[str, List[str]] = {
    "blind_navigate": [
        "scan", "focus_object", "alert_danger",
        "describe_path", "read_text", "report_failure",
    ],
    "deaf_relay": [
        "listen", "relay_text", "ask_speaker_repeat",
        "relay_partial", "flag_emergency",
    ],
    "asl_translate": [
        "observe_letter", "confirm_letter", "request_repeat",
        "speak_word", "speak_partial",
    ],
}


# ---------------------------------------------------------------------------
# Hidden World State
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    scenario: dict

    # ── Perceptual state ─────────────────────────────────────────────────────
    scene_clarity: float = 0.0
    noise_level: float = 0.0
    is_failure_scenario: bool = False

    # ── Blind navigate ───────────────────────────────────────────────────────
    all_objects: List[str] = field(default_factory=list)
    active_hazards: List[str] = field(default_factory=list)
    alerted_hazards: List[str] = field(default_factory=list)
    focus_clarity: Dict[str, float] = field(default_factory=dict)
    focused_object: Optional[str] = None

    # ── Deaf relay ───────────────────────────────────────────────────────────
    full_transcript: str = ""
    key_terms: List[str] = field(default_factory=list)
    transcript_words: List[str] = field(default_factory=list)
    chunk_size: int = 6
    chunks_delivered: int = 0
    relayed_text: str = ""
    noise_positions: List[bool] = field(default_factory=list)  # True = corrupted

    # ── ASL translate ────────────────────────────────────────────────────────
    target_word: str = ""
    asl_frames: List[str] = field(default_factory=list)
    current_frame_idx: int = 0
    confusion_map: Dict[str, str] = field(default_factory=dict)
    confirmed_letters: List[str] = field(default_factory=list)
    frame_noise_boost: Dict[int, float] = field(default_factory=dict)  # frame_idx → noise reduction

    # ── Interrupt ────────────────────────────────────────────────────────────
    pending_interrupt: Optional[dict] = None
    interrupt_delivered: bool = False
    interrupt_handled: bool = False
    interrupt_inject_step: int = -1

    # ── Episode tracking ─────────────────────────────────────────────────────
    done: bool = False
    success: bool = False
    failure_reason: Optional[str] = None
    trajectory_rewards: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    last_feedback: str = "Episode started. Choose your first action."
    last_reward: float = 0.0
    partial_credits: Dict[str, float] = field(default_factory=dict)

    # ── Real-world System Constraints ────────────────────────────────────────
    battery_level: float = 100.0

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LumosEnv:
    """
    LUMOS Assistive AI environment.
    Implements reset() / step() / state() following the OpenEnv API.
    """

    def __init__(self) -> None:
        self._world: Optional[WorldState] = None

    # ── Initialisation ───────────────────────────────────────────────────────

    def _new_world(self, task_id: str) -> WorldState:
        scenario = SceneGenerator.generate(task_id)
        failure_mode = scenario.get("failure_mode")
        is_failure = failure_mode is not None

        # Randomise noise level from distribution
        if failure_mode == "dark":
            noise_level = random.uniform(0.72, 0.92)
            base_clarity = random.uniform(0.05, 0.18)
        elif failure_mode == "heavy_noise":
            noise_level = random.uniform(0.82, 0.92)
            base_clarity = random.uniform(0.20, 0.40)
        else:
            noise_level = random.uniform(0.20, 0.65)
            base_clarity = random.uniform(0.15, 0.42)

        w = WorldState(
            task_id=task_id,
            episode_id=str(uuid.uuid4())[:8],
            step=0,
            max_steps=MAX_STEPS,
            scenario=scenario,
            scene_clarity=base_clarity,
            noise_level=noise_level,
            is_failure_scenario=is_failure,
        )

        if task_id == "blind_navigate":
            w.all_objects = list(scenario["all_objects"])
            w.active_hazards = list(scenario["hazards"])
            for obj in w.all_objects:
                w.focus_clarity[obj] = random.uniform(0.08, 0.30)

        elif task_id == "deaf_relay":
            w.full_transcript = scenario["full_transcript"]
            w.key_terms = list(scenario["key_terms"])
            words = w.full_transcript.split()
            w.transcript_words = words
            w.chunk_size = scenario.get("chunk_size", 6)
            # Corrupt word positions according to noise level
            w.noise_positions = [
                random.random() < noise_level * 0.55
                for _ in words
            ]

        elif task_id == "asl_translate":
            w.target_word = scenario["target_word"]
            w.asl_frames = list(scenario["frames"])
            w.current_frame_idx = 0
            difficulty = scenario.get("difficulty_profile", "medium")
            pairs = list(ASL_CONFUSION_PAIRS.items())
            if difficulty == "easy":
                active_pairs = random.sample(pairs, min(3, len(pairs)))
            elif difficulty == "medium":
                active_pairs = random.sample(pairs, min(7, len(pairs)))
            elif difficulty == "hard":
                active_pairs = pairs
            else:  # extreme
                active_pairs = pairs
                w.noise_level = min(0.95, w.noise_level + 0.25)
            w.confusion_map = dict(active_pairs)

        # Set up interrupt
        ev = scenario.get("interrupt_event")
        if ev:
            w.pending_interrupt = dict(ev)
            w.interrupt_inject_step = random.randint(2, 5)

        return w

    # ── Observation builder ──────────────────────────────────────────────────

    def _build_obs(self, w: WorldState) -> Observation:
        interrupt = None
        interrupt_urgency = None
        if (w.pending_interrupt and not w.interrupt_delivered
                and w.step >= w.interrupt_inject_step):
            interrupt = w.pending_interrupt.get("content")
            interrupt_urgency = w.pending_interrupt.get("urgency")
            w.interrupt_delivered = True

        task = w.task_id
        
        # Inject Fake Interrupts occasionally (stochastically tied to noise level)
        if not w.pending_interrupt and random.random() < (0.05 + 0.1 * w.noise_level) and w.step >= 3 and not w.interrupt_delivered:
            fake_interrupts = [
                "System update available. Install now?",
                "Bluetooth connection weak.",
                "Calibrating sensors. Please hold still for 5 seconds."
            ]
            interrupt = random.choice(fake_interrupts)
            interrupt_urgency = "low"
            w.pending_interrupt = {
                "type": "false_alarm",
                "content": interrupt,
                "urgency": "low"
            }
            w.interrupt_delivered = True

        if task == "blind_navigate":
            # Natural clarity degradation over time
            if w.scene_clarity > 0.15:
                w.scene_clarity = max(0.10, w.scene_clarity - round(random.uniform(0.01, 0.04), 3))
            for obj in w.all_objects:
                if obj in w.focus_clarity and w.focus_clarity[obj] > 0.20:
                    w.focus_clarity[obj] = max(0.15, w.focus_clarity[obj] - 0.02)
                    
            visible = [obj for obj in w.all_objects
                       if w.focus_clarity.get(obj, 0) > CLARITY_THRESHOLD]

            if w.scene_clarity < 0.25:
                cam = (
                    f"Scene is very dark or obstructed. "
                    f"Only vague shapes discernible: {', '.join(w.all_objects[:2]) or 'nothing'}. "
                    f"Clarity: {w.scene_clarity:.2f}."
                )
            elif w.scene_clarity < 0.45:
                cam = (
                    f"Dim scene. Outlines present but details unclear. "
                    f"Objects sensed nearby: {', '.join(w.all_objects)}. "
                    f"Clarity: {w.scene_clarity:.2f}. Use scan or focus_object."
                )
            else:
                clara = [o for o in visible]
                dim = [o for o in w.all_objects if o not in visible]
                parts = []
                if clara:
                    parts.append(f"Clearly visible: {', '.join(clara)}")
                if dim:
                    parts.append(f"Still dim: {', '.join(dim)}")
                cam = ". ".join(parts) + f". Clarity: {w.scene_clarity:.2f}."

            focused = w.focused_object
            if focused and w.focus_clarity.get(focused, 0) > 0.50:
                if focused in w.active_hazards:
                    cam += (
                        f" ⚠ FOCUSED [{focused.replace('_', ' ')}]: "
                        f"DANGER — hazard confirmed."
                    )
                elif focused in w.scenario.get("text_objects", []):
                    txt = w.scenario.get("text_content", {}).get(focused, "[text unclear]")
                    cam += f" 📖 FOCUSED [{focused.replace('_', ' ')}]: Text reads: '{txt}'"
                else:
                    cam += (
                        f" FOCUSED [{focused.replace('_', ' ')}]: "
                        f"Examined — no immediate hazard."
                    )

            return Observation(
                task_id=task, episode_id=w.episode_id,
                step_number=w.step, steps_remaining=w.steps_remaining,
                camera_feed=cam, scene_clarity=round(w.scene_clarity, 3),
                visible_objects=visible,
                noise_level=round(w.noise_level, 3),
                interrupt=interrupt, interrupt_urgency=interrupt_urgency,
                last_action_feedback=w.last_feedback + f" [Battery: {w.battery_level:.1f}%]",
                last_reward=round(w.last_reward, 4),
                user_context=w.scenario.get("user_context", ""),
            )

        elif task == "deaf_relay":
            words = w.transcript_words
            cs = w.chunk_size
            start = w.chunks_delivered * cs
            end = min(start + cs, len(words))
            total_chunks = max(1, -(-len(words) // cs))   # ceiling division

            if start < len(words):
                chunk_words = []
                for i, word in enumerate(words[start:end]):
                    wi = start + i
                    if wi < len(w.noise_positions) and w.noise_positions[wi]:
                        chunk_words.append("[CORRUPTED]")
                    else:
                        chunk_words.append(word)
                        
                    # Inject occasional environmental noise (tied to noise level)
                    if random.random() < min(0.08, 0.05 + 0.1 * w.noise_level):
                        noise_tags = ["[COUGH]", "[SIREN PASSING]", "[STATIC]", "[HEAVY BREATHING]"]
                        chunk_words.append(random.choice(noise_tags))
                        
                audio = " ".join(chunk_words)
                stream_status = f"Chunk {w.chunks_delivered + 1}/{total_chunks}"
            else:
                audio = "[END OF AUDIO STREAM]"
                stream_status = "All chunks delivered"

            return Observation(
                task_id=task, episode_id=w.episode_id,
                step_number=w.step, steps_remaining=w.steps_remaining,
                audio_stream=audio, noise_level=round(w.noise_level, 3),
                interrupt=interrupt, interrupt_urgency=interrupt_urgency,
                last_action_feedback=w.last_feedback + f" [{stream_status}] [Battery: {w.battery_level:.1f}%]",
                last_reward=round(w.last_reward, 4),
                user_context=w.scenario.get("user_context", ""),
            )

        else:  # asl_translate
            idx = w.current_frame_idx
            frames = w.asl_frames

            if idx < len(frames):
                actual = frames[idx]
                noise_reduction = w.frame_noise_boost.get(idx, 0.0)
                eff_noise = max(0.0, w.noise_level - noise_reduction)
                r = random.random()
                confusion_prob = eff_noise * 0.55

                if r < confusion_prob * 0.4 and actual in w.confusion_map:
                    # True Override: output the wrong letter entirely with high confidence
                    confused = w.confusion_map[actual]
                    asl_obs = confused
                    conf = round(random.uniform(0.78, 0.92), 2)
                elif r < confusion_prob * 0.7 and actual in w.confusion_map:
                    confused = w.confusion_map[actual]
                    asl_obs = f"ambiguous: {actual}/{confused}"
                    conf = round(random.uniform(0.32, 0.54), 2)
                elif r < confusion_prob:
                    asl_obs = "unclear"
                    conf = round(random.uniform(0.12, 0.28), 2)
                else:
                    asl_obs = actual
                    conf = round(random.uniform(0.68, 0.96), 2)
            else:
                asl_obs = "[ALL FRAMES SHOWN]"
                conf = 1.0

            confirmed_str = "".join(w.confirmed_letters) if w.confirmed_letters else None

            return Observation(
                task_id=task, episode_id=w.episode_id,
                step_number=w.step, steps_remaining=w.steps_remaining,
                noise_level=round(w.noise_level, 3),
                asl_observation=asl_obs, asl_confidence=conf,
                confirmed_so_far=confirmed_str,
                interrupt=interrupt, interrupt_urgency=interrupt_urgency,
                last_action_feedback=w.last_feedback + f" [Battery: {w.battery_level:.1f}%]",
                last_reward=round(w.last_reward, 4),
                user_context=w.scenario.get("user_context", ""),
            )

    # ── State transitions ────────────────────────────────────────────────────

    def _transition_blind(
        self, w: WorldState, action: Action
    ) -> Tuple[float, str, bool]:
        if w.battery_level < 5.0:
            w.partial_credits = {"failure": -1.0}
            return -1.0, "Hardware shutdown: Battery depleted.", False
        
        # System constraints: battery drops, affecting base latency penalty
        w.battery_level = max(0.0, w.battery_level - random.uniform(1.5, 3.5))
        effective_latency = LATENCY_PENALTY * (2.0 if w.battery_level < 20 else 1.0)
        
        reward = -effective_latency
        credits: Dict[str, float] = {"latency": -effective_latency}
        feedback = ""
        success = False

        a = action.action_type
        target = (action.target or "").lower().strip()
        content = (action.content or "").lower()
        
        # Repetition parsing
        recent_actions = [prev for prev in w.action_history[-3:] if prev.startswith(a)]
        if len(recent_actions) >= 3:
            credits["repetition"] = -0.04
            reward -= 0.04
            feedback += "Penalty: Repeating same action excessively. "

        if a == "scan":
            gain = random.uniform(0.08, 0.18)
            w.scene_clarity = min(0.88, w.scene_clarity + gain)
            if w.is_failure_scenario:
                w.scene_clarity = min(0.28, w.scene_clarity)  # dark cap
            for obj in w.all_objects:
                w.focus_clarity[obj] = min(0.85, w.focus_clarity[obj] + gain * 0.45)
            visible_count = sum(
                1 for o in w.all_objects if w.focus_clarity.get(o, 0) > CLARITY_THRESHOLD
            )
            scan_reward = round(0.04 + gain * 0.25, 4)
            credits["scan_gain"] = scan_reward
            reward += scan_reward
            feedback = (
                f"Scan complete. Global clarity → {w.scene_clarity:.2f}. "
                f"{visible_count}/{len(w.all_objects)} objects now clearly visible."
            )

        elif a == "focus_object":
            if not target:
                credits["invalid"] = -0.06
                reward -= 0.06
                feedback = "focus_object requires target=<object_name>."
            else:
                matched = next(
                    (o for o in w.all_objects
                     if target in o.lower() or o.lower() in target),
                    None,
                )
                if matched:
                    old_c = w.focus_clarity.get(matched, 0.0)
                    gain = random.uniform(0.28, 0.48)
                    w.focus_clarity[matched] = min(0.95, old_c + gain)
                    w.focused_object = matched
                    if old_c < CLARITY_THRESHOLD:
                        credits["useful_focus"] = round(0.06 + gain * 0.12, 4)
                        reward += credits["useful_focus"]
                    else:
                        credits["redundant_focus"] = 0.01
                        reward += 0.01
                    feedback = (
                        f"Focused on '{matched}'. "
                        f"Per-object clarity: {old_c:.2f} → {w.focus_clarity[matched]:.2f}."
                    )
                else:
                    credits["wrong_target"] = -0.12
                    reward -= 0.12
                    feedback = (
                        f"No object matching '{target}' found in this scene. "
                        f"Step wasted. Known objects: {w.all_objects}"
                    )

        elif a == "alert_danger":
            if not content:
                credits["empty"] = -0.08
                reward -= 0.08
                feedback = "alert_danger requires content=<description_of_hazard>."
            else:
                content_lower = content.lower()
                matched = [
                    h for h in w.active_hazards
                    if h.replace("_", " ") in content_lower
                    or any(p in content_lower for p in h.lower().split("_"))
                ]
                non_hazards = [o for o in w.all_objects if o not in w.active_hazards]
                false_alarm = any(
                    o.replace("_", " ") in content_lower for o in non_hazards
                )
                if matched:
                    per_reward = 0.28 * len(matched)
                    credits["hazards"] = round(per_reward, 4)
                    reward += per_reward
                    for h in matched:
                        w.active_hazards.remove(h)
                        w.alerted_hazards.append(h)
                    timeliness = round(0.07 * w.steps_remaining / w.max_steps, 4)
                    credits["timeliness"] = timeliness
                    reward += timeliness
                    if not w.active_hazards:
                        success = True
                    feedback = (
                        f"Alert issued for: {matched}. "
                        f"Remaining hazards: {w.active_hazards or 'none'}."
                    )
                else:
                    credits["missed"] = -0.06
                    reward -= 0.06
                    feedback = (
                        f"Alert did not match any active hazard. "
                        f"Active hazards: {w.active_hazards}"
                    )
                if false_alarm:
                    credits["false_alarm"] = -0.15
                    reward -= 0.15
                    feedback += " ⚠ False alarm penalty: non-hazard flagged."

        elif a == "describe_path":
            if not content:
                credits["empty"] = -0.05
                reward -= 0.05
                feedback = "describe_path requires content=<guidance text>."
            elif w.active_hazards:
                success = False
                reward -= 0.10
                feedback = "✗ Path is blocked by unseen hazards. Safe navigation impossible."
            else:
                clearly_visible = [
                    o for o in w.all_objects if w.focus_clarity.get(o, 0) > CLARITY_THRESHOLD
                ]
                mentioned = sum(
                    1 for o in clearly_visible if o.replace("_", " ") in content
                )
                frac = mentioned / max(1, len(clearly_visible))
                unclear_objs = [
                    o for o in w.all_objects if w.focus_clarity.get(o, 0) <= CLARITY_THRESHOLD
                ]
                hallucinations = sum(
                    1 for o in unclear_objs if o.replace("_", " ") in content
                )
                credits["path_accuracy"] = round(0.16 * frac, 4)
                credits["hallucination_penalty"] = round(-0.09 * hallucinations, 4)
                reward += credits["path_accuracy"] + credits["hallucination_penalty"]
                if frac >= 0.8 and hallucinations == 0 and not w.active_hazards:
                    success = True
                feedback = (
                    f"Path described. {mentioned}/{len(clearly_visible)} verified objects. "
                    f"{hallucinations} hallucinations detected."
                )

        elif a == "read_text":
            text_objs = w.scenario.get("text_objects", [])
            focused = w.focused_object
            if focused and focused in text_objs and w.focus_clarity.get(focused, 0) > 0.50:
                actual_txt = w.scenario.get("text_content", {}).get(focused, "")
                c_lower = content
                kws = [kw for kw in actual_txt.lower().split() if len(kw) > 4]
                hits = sum(1 for kw in kws if kw in c_lower)
                frac = hits / max(1, len(kws))
                credits["read_accuracy"] = round(0.38 * frac, 4)
                reward += credits["read_accuracy"]
                if frac >= 0.70:
                    success = True
                feedback = (
                    f"OCR from '{focused}'. {hits}/{len(kws)} key terms captured. "
                    f"Accuracy: {frac:.0%}."
                )
            elif not text_objs:
                credits["no_text"] = -0.05
                reward -= 0.05
                feedback = "No text objects in this scene."
            else:
                credits["not_focused"] = -0.08
                reward -= 0.08
                feedback = (
                    f"Must focus on a text object before reading. "
                    f"Text objects: {text_objs}"
                )

        elif a == "report_failure":
            if w.is_failure_scenario:
                credits["correct_failure"] = 0.45
                reward += 0.45
                success = True
                feedback = (
                    "✓ Correct: scene is genuinely unnavigable. "
                    "Appropriate failure handling — user alerted."
                )
            else:
                credits["wrong_failure"] = -0.28
                reward -= 0.28
                feedback = "✗ Scene IS navigable. Premature failure declaration causes harm."

        else:
            credits["invalid_action"] = -0.10
            reward -= 0.10
            valid = VALID_ACTIONS["blind_navigate"]
            feedback = f"Unknown action '{a}'. Valid: {valid}"

        # Interrupt logic
        if w.interrupt_delivered and not w.interrupt_handled:
            urgency = (w.pending_interrupt or {}).get("urgency", "low")
            pending_type = (w.pending_interrupt or {}).get("type", "none")
            if action.priority == "handle_interrupt":
                w.interrupt_handled = True
                bonus = 0.15 if urgency == "critical" else 0.06
                credits["interrupt_handled"] = round(bonus, 4)
                reward += bonus
                feedback += " ✓ Interrupt handled."
            elif urgency == "critical":
                credits["critical_ignored"] = -0.20
                reward -= 0.20
                feedback += " ✗ CRITICAL interrupt ignored!"
            elif pending_type == "false_alarm":
                credits["false_interrupt_penalty"] = -0.08
                reward -= 0.08
                feedback += " ⚠ Wasted time on false system alert!"

        w.partial_credits = credits
        return round(reward, 4), feedback, success

    def _transition_deaf(
        self, w: WorldState, action: Action
    ) -> Tuple[float, str, bool]:
        if w.battery_level < 5.0:
            w.partial_credits = {"failure": -1.0}
            return -1.0, "Hardware shutdown: Battery depleted.", False

        w.battery_level = max(0.0, w.battery_level - random.uniform(1.2, 2.5))
        effective_latency = LATENCY_PENALTY * (2.0 if w.battery_level < 20 else 1.0)
        
        reward = -effective_latency
        credits: Dict[str, float] = {"latency": -effective_latency}
        feedback = ""
        success = False

        a = action.action_type
        content = action.content or ""
        content_lower = content.lower()
        
        # Repetition parsing
        if a != "listen":
            recent_actions = [prev for prev in w.action_history[-3:] if prev.startswith(a)]
            if len(recent_actions) >= 3:
                credits["repetition"] = -0.04
                reward -= 0.04
                feedback += "Penalty: Repeating same action excessively. "

        if a == "listen":
            total_chunks = max(1, -(-len(w.transcript_words) // w.chunk_size))
            if w.chunks_delivered < total_chunks:
                w.chunks_delivered += 1
                credits["listen"] = 0.03
                reward += 0.03
                feedback = (
                    f"Listening. Received chunk {w.chunks_delivered}/{total_chunks}."
                )
                if w.chunks_delivered >= total_chunks:
                    feedback += " Full stream received."
            else:
                credits["listen_done"] = 0.0
                feedback = "Audio stream already fully received."

        elif a == "relay_text":
            if not content:
                credits["empty"] = -0.08
                reward -= 0.08
                feedback = "relay_text requires content."
            else:
                # Key term coverage
                hits = sum(1 for kt in w.key_terms if kt.lower() in content_lower)
                frac = hits / max(1, len(w.key_terms))
                # Hallucination: content words not in full transcript
                transcript_vocab = set(w.full_transcript.lower().split())
                content_words = [cw for cw in content_lower.split() if len(cw) > 4]
                hallucinated = sum(
                    1 for cw in content_words if cw not in transcript_vocab
                )
                hall_rate = hallucinated / max(1, len(content_words))

                credits["coverage"] = round(0.55 * frac, 4)
                credits["anti_hallucination"] = round(-0.22 * hall_rate, 4)
                reward += credits["coverage"] + credits["anti_hallucination"]
                w.relayed_text = content

                if frac >= 1.0:
                    credits["complete"] = 0.15
                    reward += 0.15
                    success = True
                elif frac >= 0.5:
                    credits["partial_credit"] = round(0.07 * frac, 4)
                    reward += credits["partial_credit"]

                feedback = (
                    f"Relayed. Coverage: {hits}/{len(w.key_terms)} key terms "
                    f"({frac:.0%}). Hallucination rate: {hall_rate:.0%}."
                )

        elif a == "ask_speaker_repeat":
            # Clean up corruption for the CURRENT (not yet delivered) chunk
            cs = w.chunk_size
            total_chunks = max(1, -(-len(w.transcript_words) // cs))
            chunk_idx = w.chunks_delivered  # next chunk to be delivered
            if chunk_idx < total_chunks:
                start = chunk_idx * cs
                end = min(start + cs, len(w.transcript_words))
                cleared = 0
                for i in range(start, end):
                    if i < len(w.noise_positions) and w.noise_positions[i]:
                        w.noise_positions[i] = False
                        cleared += 1
                credits["clarify"] = 0.0
                feedback = (
                    f"Requested repeat for chunk {chunk_idx + 1}. "
                    f"{cleared} corrupted word(s) will be clear on next listen."
                )
            else:
                credits["useless_clarify"] = -0.05
                reward -= 0.05
                feedback = "Nothing left to clarify — audio stream ended."

        elif a == "relay_partial":
            if not content:
                credits["empty"] = -0.05
                reward -= 0.05
                feedback = "relay_partial requires content."
            else:
                hits = sum(1 for kt in w.key_terms if kt.lower() in content_lower)
                frac = hits / max(1, len(w.key_terms))
                credits["partial_relay"] = round(0.22 * frac, 4)
                reward += credits["partial_relay"]
                feedback = (
                    f"Partial relay sent. Coverage: {hits}/{len(w.key_terms)} key terms."
                )

        elif a == "flag_emergency":
            urgent_kws = ["emergency", "evacuation", "immediate", "critical",
                          "penalty", "escalated", "shutdown", "final"]
            is_urgent = any(kw in w.full_transcript.lower() for kw in urgent_kws)
            if is_urgent:
                credits["correct_flag"] = 0.12
                reward += 0.12
                feedback = "✓ Emergency flag correct. Critical content identified."
            else:
                credits["false_flag"] = -0.10
                reward -= 0.10
                feedback = "✗ False emergency flag. Content is not urgent."

        else:
            credits["invalid"] = -0.08
            reward -= 0.08
            feedback = f"Unknown action '{a}'. Valid: {VALID_ACTIONS['deaf_relay']}"

        # Interrupt logic
        if w.interrupt_delivered and not w.interrupt_handled:
            urgency = (w.pending_interrupt or {}).get("urgency", "low")
            pending_type = (w.pending_interrupt or {}).get("type", "")
            if action.priority == "handle_interrupt":
                w.interrupt_handled = True
                bonus = 0.15 if urgency == "critical" else 0.06
                credits["interrupt"] = round(bonus, 4)
                reward += bonus
                feedback += " ✓ Interrupt acknowledged."
            elif urgency == "critical":
                credits["critical_ignored"] = -0.20
                reward -= 0.20
                feedback += " ✗ CRITICAL interrupt ignored!"
            elif pending_type == "false_alarm":
                credits["false_interrupt_penalty"] = -0.08
                reward -= 0.08
                feedback += " ⚠ Wasted time on false system alert!"

        w.partial_credits = credits
        return round(reward, 4), feedback, success

    def _transition_asl(
        self, w: WorldState, action: Action
    ) -> Tuple[float, str, bool]:
        if w.battery_level < 5.0:
            w.partial_credits = {"failure": -1.0}
            return -1.0, "Hardware shutdown: Battery depleted.", False

        w.battery_level = max(0.0, w.battery_level - random.uniform(1.5, 3.0))
        effective_latency = LATENCY_PENALTY * (2.0 if w.battery_level < 20 else 1.0)
        
        reward = -effective_latency
        credits: Dict[str, float] = {"latency": -effective_latency}
        feedback = ""
        success = False

        a = action.action_type
        target = (action.target or "").upper().strip()
        content = (action.content or "").upper().strip()
        frames = w.asl_frames
        
        # Repetition penalty
        recent_actions = [prev for prev in w.action_history[-3:] if prev.startswith(a)]
        if len(recent_actions) == 3:
            reward -= 0.04
            credits["repetition_penalty"] = -0.04

        if a == "observe_letter":
            if w.current_frame_idx < len(frames):
                w.current_frame_idx += 1
                credits["observe"] = 0.02
                reward += 0.02
                feedback = (
                    f"Observed frame {w.current_frame_idx}/{len(frames)}. "
                    f"Check asl_observation and confidence."
                )
            else:
                credits["over_observe"] = -0.03
                reward -= 0.03
                feedback = "All frames shown. Use speak_word or speak_partial."

        elif a == "confirm_letter":
            if not target or len(target) != 1:
                credits["invalid"] = -0.05
                reward -= 0.05
                feedback = "confirm_letter requires target=<single letter A-Z>."
            else:
                idx = len(w.confirmed_letters)
                if idx < len(frames):
                    actual = frames[idx]
                    if target == actual:
                        # Noise bonus: harder to get right under high noise
                        correct_reward = round(0.10 * (1 + 0.20 * w.noise_level), 4)
                        credits["correct_letter"] = correct_reward
                        reward += correct_reward
                        feedback = f"✓ Letter '{target}' confirmed correctly."
                    else:
                        credits["wrong_letter"] = -0.15
                        reward -= 0.15
                        feedback = (
                            f"✗ Letter '{target}' incorrect (actual was '{actual}'). "
                            f"Irreversible commitment penalty."
                        )
                    w.confirmed_letters.append(target)
                else:
                    credits["excess"] = -0.03
                    reward -= 0.03
                    feedback = "All letters already confirmed."

        elif a == "request_repeat":
            idx = w.current_frame_idx
            if idx < len(frames):
                current_boost = w.frame_noise_boost.get(idx, 0.0)
                if current_boost < 0.45:
                    w.frame_noise_boost[idx] = current_boost + 0.30
                    credits["repeat"] = 0.0
                    feedback = (
                        f"Repeat requested for frame {idx + 1}. "
                        f"Signal improved (+0.30 noise reduction). Observe again."
                    )
                else:
                    credits["redundant_repeat"] = -0.03
                    reward -= 0.03
                    feedback = "Frame already maxed out on repeats. No further improvement."
            else:
                credits["no_frame"] = -0.03
                reward -= 0.03
                feedback = "All frames shown. No frame to repeat."

        elif a == "speak_word":
            if not content:
                credits["empty"] = -0.08
                reward -= 0.08
                feedback = "speak_word requires content=<word>."
            else:
                target_word = w.target_word
                if content == target_word:
                    timeliness = round(0.08 * w.steps_remaining / w.max_steps, 4)
                    credits["exact_match"] = 0.50
                    credits["timeliness"] = timeliness
                    reward += 0.50 + timeliness
                    success = True
                    feedback = f"✓ Correct word '{content}' spoken! User assisted."
                else:
                    matches = sum(a == b for a, b in zip(content, target_word))
                    partial_frac = matches / max(1, len(target_word))
                    credits["partial_word"] = round(0.14 * partial_frac, 4)
                    credits["wrong_word"] = -0.12
                    reward += credits["partial_word"] - 0.12
                    feedback = (
                        f"✗ '{content}' incorrect. "
                        f"Partial match: {matches}/{len(target_word)} letters. "
                        f"Target: {len(target_word)} letters."
                    )

        elif a == "speak_partial":
            if not content:
                credits["empty"] = -0.05
                reward -= 0.05
                feedback = "speak_partial requires content."
            else:
                partial_target = w.target_word[:w.current_frame_idx]
                matches = sum(a == b for a, b in zip(content, partial_target))
                frac = matches / max(1, len(partial_target))
                credits["speak_partial"] = round(0.10 * frac, 4)
                reward += credits["speak_partial"]
                feedback = (
                    f"Partial output: '{content}'. "
                    f"Match with seen letters: {matches}/{len(partial_target)}."
                )

        else:
            credits["invalid"] = -0.08
            reward -= 0.08
            feedback = f"Unknown action '{a}'. Valid: {VALID_ACTIONS['asl_translate']}"

        # Interrupt (rare in mute mode but possible)
        if w.interrupt_delivered and not w.interrupt_handled:
            if action.priority == "handle_interrupt":
                w.interrupt_handled = True
                credits["interrupt"] = 0.05
                reward += 0.05

        w.partial_credits = credits
        return reward, feedback, success

    # ── Grader ───────────────────────────────────────────────────────────────

    def _grader_score(self, w: WorldState) -> float:
        traj = w.trajectory_rewards
        if w.success:
            efficiency = max(0.0, 1.0 - (w.step - 1) / max(1, w.max_steps - 1))
            base = 0.74 + 0.25 * efficiency   # 0.74 … 0.99
            return round(float(np.clip(base, SCORE_MIN, SCORE_MAX)), 4)
        if not traj:
            return SCORE_MIN
        raw = float(np.mean(traj))
        normed = (raw + 1.0) / 2.0
        scaled = SCORE_MIN + normed * 0.70
        return round(float(np.clip(scaled, SCORE_MIN, 0.73)), 4)

    # ── Public OpenEnv API ───────────────────────────────────────────────────

    def reset(self, task_id: str = "blind_navigate") -> Observation:
        self._world = self._new_world(task_id)
        return self._build_obs(self._world)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        w = self._world
        if w is None:
            raise RuntimeError("Call reset() before step().")
        if w.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        w.step += 1
        w.action_history.append(action.action_type)

        if w.task_id == "blind_navigate":
            reward, feedback, success = self._transition_blind(w, action)
        elif w.task_id == "deaf_relay":
            reward, feedback, success = self._transition_deaf(w, action)
        else:
            reward, feedback, success = self._transition_asl(w, action)

        reward = float(np.clip(reward, -1.0, 1.0))
        w.trajectory_rewards.append(reward)
        w.last_feedback = feedback
        w.last_reward = reward

        if success:
            # Critical interrupt tightness check
            if w.pending_interrupt and w.pending_interrupt.get("urgency") == "critical" and not w.interrupt_handled:
                w.success = False
                feedback += " [FAIL: Critical interrupt ignored]"
                w.last_feedback = feedback
            else:
                w.success = True
                
        w.done = w.done or w.success or (w.step >= w.max_steps)

        gs = self._grader_score(w)
        info = {
            "grader_score": gs,
            "partial_credits": w.partial_credits,
            "success": w.success,
            "episode_id": w.episode_id,
            "action_history": list(w.action_history),
            "active_hazards": w.active_hazards if w.task_id == "blind_navigate" else None,
        }
        return self._build_obs(w), reward, w.done, info

    def get_state(self, debug: bool = False) -> dict:
        w = self._world
        if w is None:
            return {"status": "not_initialized"}
            
        base_state = {
            "episode_id": w.episode_id,
            "task_id": w.task_id,
            "step": w.step,
            "steps_remaining": w.steps_remaining,
            "done": w.done,
            "success": w.success,
            "noise_level": round(w.noise_level, 4),
            "scene_clarity": (
                round(w.scene_clarity, 4) if w.task_id == "blind_navigate" else None
            ),
            "chunks_delivered": w.chunks_delivered if w.task_id == "deaf_relay" else None,
            "total_chunks": (
                max(1, -(-len(w.transcript_words) // w.chunk_size))
                if w.task_id == "deaf_relay" else None
            ),
            "current_frame": w.current_frame_idx if w.task_id == "asl_translate" else None,
            "total_frames": len(w.asl_frames) if w.task_id == "asl_translate" else None,
            "confirmed_letters": (
                "".join(w.confirmed_letters) if w.task_id == "asl_translate" else None
            ),
            "interrupt_delivered": w.interrupt_delivered,
            "interrupt_handled": w.interrupt_handled,
            "trajectory_rewards": [round(r, 4) for r in w.trajectory_rewards],
            "partial_credits": w.partial_credits,
            "grader_score": self._grader_score(w),
        }
        
        if debug:
            base_state.update({
                "focus_clarity": (
                    {k: round(v, 4) for k, v in w.focus_clarity.items()}
                    if w.task_id == "blind_navigate" else None
                ),
                "active_hazards": w.active_hazards,
                "target_word": w.target_word,
                "key_terms": w.scenario.get("key_terms", []),
                "full_transcript": getattr(w, "full_transcript", None),
                "noise_positions": getattr(w, "noise_positions", [])
            })
            
        return base_state

    def grader_score(self) -> float:
        if self._world is None:
            return SCORE_MIN
        return self._grader_score(self._world)
