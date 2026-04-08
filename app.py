"""
LUMOS Assistive AI — OpenEnv Environment
=========================================
Real-world assistive technology environment for people with
visual, hearing, and speech impairments.

Tasks:
  Task 1 (easy)   — Blind Mode   : Scene understanding + danger detection
  Task 2 (medium) — Deaf Mode    : Speech-to-text relay on OLED display
  Task 3 (hard)   — Mute Mode    : ASL finger-spelling → speech output

Endpoints: /reset  /step  /state  /tasks  /grader  /baseline

SCORING FIX: All grader scores are strictly in (0.0, 1.0) exclusive.
  - Success score capped at 0.99 (not 1.0)
  - Failure floor at 0.01 (not 0.0)
  - Step rewards clipped to (0.01, 0.99)
"""

from __future__ import annotations

import os
import random
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Score bounds — STRICTLY between 0 and 1 (Phase 2 requirement)
# ---------------------------------------------------------------------------
SCORE_MIN = 0.01   # floor: never return 0.0
SCORE_MAX = 0.99   # ceiling: never return 1.0


def clip_score(score: float) -> float:
    """Clip score to strictly open interval (0, 1)."""
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


# ---------------------------------------------------------------------------
# Pydantic Models (OpenEnv spec)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id: str
    camera_feed: str
    microphone: str
    voice_command: str
    oled_display: Optional[str] = None
    audio_output: Optional[str] = None
    asl_letter: Optional[str] = None
    spelled_word: Optional[str] = None
    step_number: int
    hint: str = ""


class Action(BaseModel):
    decision: str = Field(
        ...,
        description=(
            "One of: describe_scene | ocr_read | alert_danger | "
            "speech_to_text | sign_speech"
        )
    )
    output_text: str = Field(default="", description="Agent's generated output")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    partial_credits: Dict[str, float]
    done: bool
    feedback: str


# ---------------------------------------------------------------------------
# Scenario Data
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, List[dict]] = {

    # ── BLIND MODE (easy) ──────────────────────────────────────────────────
    "blind_mode": [
        {
            "camera": "Kitchen counter. Gas burner alight on the left. Cleaver balanced near the edge.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["burner", "cleaver"],
            "dangers": ["burner", "cleaver"],
            "difficulty": "easy",
        },
        {
            "camera": "Busy arterial road. Lorry reversing without warning. Pavement edge immediately ahead.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["lorry", "pavement"],
            "dangers": ["lorry"],
            "difficulty": "easy",
        },
        {
            "camera": "Living room. Chesterfield sofa ahead, sconce lamp to the left, unobstructed egress to hallway.",
            "voice": "LUMOS",
            "expect_decision": "describe_scene",
            "key_objects": ["sofa", "lamp", "hallway"],
            "dangers": [],
            "difficulty": "easy",
        },
        {
            "camera": "Open pharmacopoeia. Page reads: CONTRAINDICATED WITH MAOI INHIBITORS – CONSULT PHYSICIAN",
            "voice": "read label",
            "expect_decision": "ocr_read",
            "key_objects": ["pharmacopoeia", "contraindicated"],
            "dangers": [],
            "difficulty": "easy",
        },
        {
            "camera": "Stairwell. Effluent pooling at the base. Corroded handrail on the right.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["stairwell", "effluent", "handrail"],
            "dangers": ["stairwell", "effluent"],
            "difficulty": "easy",
        },
        {
            "camera": "Medicine cabinet ajar. Unlabelled vials scattered across the vanity.",
            "voice": "read label",
            "expect_decision": "ocr_read",
            "key_objects": ["vials", "cabinet"],
            "dangers": [],
            "difficulty": "easy",
        },
        {
            "camera": "Municipal park. Wrought-iron bench to the left, ornamental fountain ahead, clear ambulatory path.",
            "voice": "LUMOS",
            "expect_decision": "describe_scene",
            "key_objects": ["bench", "fountain", "path"],
            "dangers": [],
            "difficulty": "easy",
        },
        {
            "camera": "Excavation site. Pneumatic jackhammer operating nearby. Perimeter caution tape breached.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["jackhammer", "caution tape"],
            "dangers": ["jackhammer"],
            "difficulty": "easy",
        },
        {
            "camera": "Laboratory. Autoclave venting superheated steam. Biohazard waste container overflowing.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["autoclave", "biohazard"],
            "dangers": ["autoclave", "biohazard"],
            "difficulty": "easy",
        },
        {
            "camera": "Corridor. Emergency evacuation placard on the wall reads: ASSEMBLY POINT – WEST ANNEXE",
            "voice": "read sign",
            "expect_decision": "ocr_read",
            "key_objects": ["placard", "assembly", "annexe"],
            "dangers": [],
            "difficulty": "easy",
        },
    ],

    # ── DEAF MODE (medium) ─────────────────────────────────────────────────
    "deaf_mode": [
        {
            "microphone": "Your rheumatology consultation has been rescheduled to the haematology wing at fourteen hundred hours",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["rheumatology", "haematology", "fourteen hundred"],
            "difficulty": "medium",
        },
        {
            "microphone": "Immediate evacuation required — noxious fume ingress detected in the subterranean car park",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["evacuation", "noxious", "subterranean"],
            "difficulty": "medium",
        },
        {
            "microphone": "Your cryogenic specimen consignment has arrived at the pathology intake dock",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["cryogenic", "consignment", "pathology"],
            "difficulty": "medium",
        },
        {
            "microphone": "Tarmac closure on runway two seven — all inbound aeronautical traffic rerouted to taxiway delta",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["tarmac", "inbound", "taxiway"],
            "difficulty": "medium",
        },
        {
            "microphone": "Defibrillator malfunction reported in the cardiac catheterisation suite — standby technician en route",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["defibrillator", "catheterisation", "technician"],
            "difficulty": "medium",
        },
        {
            "microphone": "Your statutory redundancy entitlement has been recalculated per the latest actuarial assessment",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["redundancy", "entitlement", "actuarial"],
            "difficulty": "medium",
        },
        {
            "microphone": "Hyperbaric chamber pressure differential exceeds safe threshold — initiate controlled decompression immediately",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["hyperbaric", "differential", "decompression"],
            "difficulty": "medium",
        },
        {
            "microphone": "Intravenous acetylcysteine infusion is commencing in bay four — monitor for anaphylactoid reaction",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["acetylcysteine", "infusion", "anaphylactoid"],
            "difficulty": "medium",
        },
    ],

    # ── MUTE MODE (hard) ───────────────────────────────────────────────────
    "mute_mode": [
        {
            "camera_frames": ["hand_Q", "hand_U", "hand_A", "hand_R", "hand_R", "hand_E", "hand_L"],
            "target_word": "QUARREL",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_S", "hand_T", "hand_O", "hand_I", "hand_C"],
            "target_word": "STOIC",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_C", "hand_H", "hand_A", "hand_S", "hand_M"],
            "target_word": "CHASM",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_P", "hand_L", "hand_U", "hand_M", "hand_B"],
            "target_word": "PLUMB",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_F", "hand_R", "hand_O", "hand_N", "hand_D"],
            "target_word": "FROND",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_S", "hand_W", "hand_A", "hand_T", "hand_H", "hand_E"],
            "target_word": "SWATHE",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_K", "hand_N", "hand_A", "hand_V", "hand_E"],
            "target_word": "KNAVE",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_G", "hand_A", "hand_U", "hand_N", "hand_T"],
            "target_word": "GAUNT",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
    ],
}

TASK_METADATA = {
    "blind_mode": {
        "description": "Agent interprets camera feed and produces correct audio output for a blind user.",
        "difficulty": "easy",
        "valid_decisions": ["describe_scene", "ocr_read", "alert_danger"],
        "action_schema": {"decision": "string", "output_text": "string", "confidence": "float [0,1]"},
    },
    "deaf_mode": {
        "description": "Agent relays spoken words as clear text for display on the OLED screen.",
        "difficulty": "medium",
        "valid_decisions": ["speech_to_text"],
        "action_schema": {"decision": "string", "output_text": "string", "confidence": "float [0,1]"},
    },
    "mute_mode": {
        "description": "Agent recognises ASL finger-spelling frames and produces speech output.",
        "difficulty": "hard",
        "valid_decisions": ["sign_speech"],
        "action_schema": {"decision": "string", "output_text": "string", "confidence": "float [0,1]"},
    },
}

# ---------------------------------------------------------------------------
# ASL Letter Detector (deterministic)
# ---------------------------------------------------------------------------

class ASLLetterDetector:
    ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def predict(self, frame_desc: str) -> str:
        parts = frame_desc.upper().split("_")
        for part in reversed(parts):
            if len(part) == 1 and part in self.ALPHABET:
                return part
        return "?"

    def spell(self, frames: List[str]) -> str:
        return "".join(self.predict(f) for f in frames)


asl_detector = ASLLetterDetector()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

MAX_STEPS = 8


class LumosEnv:
    def __init__(self):
        self._s: dict = {}
        self._init_state("blind_mode")

    def _init_state(self, task_id: str):
        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task_id '{task_id}'.")
        scenario = random.choice(SCENARIOS[task_id])
        self._s = {
            "episode_id": str(uuid.uuid4())[:8],
            "task_id": task_id,
            "step": 0,
            "scenario": scenario,
            "trajectory_rewards": [],
            "success": False,
            "partial_scores": {},
            "letters_seen": [],
            "done": False,
        }

    def _build_obs(self, extra: dict | None = None) -> Observation:
        s = self._s
        data = s["scenario"]
        task = s["task_id"]
        extra = extra or {}

        if task == "mute_mode":
            frames = data["camera_frames"]
            idx = max(0, min(s["step"] - 1, len(frames) - 1))
            camera = frames[idx] if s["step"] > 0 else frames[0]
            asl_letter = asl_detector.predict(camera)
            spelled = asl_detector.spell(frames[: s["step"]]) if s["step"] > 0 else ""
            hint = ""
        else:
            camera = data.get("camera", "")
            asl_letter = None
            spelled = None
            hint = self._make_hint()

        return Observation(
            task_id=task,
            camera_feed=camera,
            microphone=data.get("microphone", ""),
            voice_command=data.get("voice", ""),
            oled_display=extra.get("oled_display"),
            audio_output=extra.get("audio_output"),
            asl_letter=asl_letter,
            spelled_word=spelled,
            step_number=s["step"],
            hint=hint,
        )

    def _make_hint(self) -> str:
        s = self._s
        task = s["task_id"]
        data = s["scenario"]
        step = s["step"]

        if task == "blind_mode":
            dangers = data.get("dangers", [])
            if dangers:
                return f"Hazard objects detected: {', '.join(dangers)}"
            return f"Scene objects: {', '.join(data.get('key_objects', []))}"
        elif task == "deaf_mode":
            words = data["microphone"].split()
            visible = words[: max(1, step)]
            return f"Heard so far: '{' '.join(visible)}'"
        return ""

    def _compute_reward(self, action: Action) -> Tuple[float, dict]:
        """
        Compute per-step reward, clipped to (SCORE_MIN, SCORE_MAX).
        Raw reward is computed then clipped so it never equals 0.0 or 1.0.
        """
        s = self._s
        task = s["task_id"]
        data = s["scenario"]
        credits: dict = {}
        reward = 0.0

        credits["confidence"] = round(0.03 * action.confidence, 4)
        reward += credits["confidence"]

        if task == "blind_mode":
            correct_decision = action.decision == data["expect_decision"]
            credits["correct_decision"] = 0.35 if correct_decision else -0.10
            reward += credits["correct_decision"]

            output_lower = action.output_text.lower()
            key_hits = sum(1 for k in data.get("key_objects", []) if k in output_lower)
            total_keys = max(1, len(data.get("key_objects", ["x"])))
            frac = key_hits / total_keys
            credits["key_objects_mentioned"] = round(0.32 * frac, 4)
            reward += credits["key_objects_mentioned"]

            dangers = data.get("dangers", [])
            if dangers:
                danger_hits = sum(1 for d in dangers if d in output_lower)
                dfrac = danger_hits / len(dangers)
                credits["danger_flagged"] = round(0.30 * dfrac, 4)
                reward += credits["danger_flagged"]
                if danger_hits == len(dangers) and correct_decision:
                    s["success"] = True
            else:
                credits["danger_flagged"] = 0.0
                if correct_decision and frac >= 0.67:
                    s["success"] = True

        elif task == "deaf_mode":
            correct_decision = action.decision == "speech_to_text"
            credits["correct_decision"] = 0.25 if correct_decision else -0.15
            reward += credits["correct_decision"]

            key_words = data.get("key_words", [])
            output_lower = action.output_text.lower()
            hits = sum(1 for kw in key_words if kw.lower() in output_lower)
            frac = hits / max(1, len(key_words))
            credits["key_words_relayed"] = round(0.60 * frac, 4)
            reward += credits["key_words_relayed"]

            if correct_decision and frac == 1.0:
                s["success"] = True

        elif task == "mute_mode":
            correct_decision = action.decision == "sign_speech"
            credits["correct_decision"] = 0.15 if correct_decision else -0.15
            reward += credits["correct_decision"]

            frames = data["camera_frames"]
            target = data["target_word"]
            spelled_so_far = asl_detector.spell(frames[: s["step"]])
            target_so_far = target[: s["step"]]

            matches = sum(a == b for a, b in zip(spelled_so_far, target_so_far))
            letter_acc = matches / max(1, len(target_so_far))
            credits["letter_accuracy"] = round(0.45 * letter_acc, 4)
            reward += credits["letter_accuracy"]

            if target.lower() == action.output_text.strip().lower():
                credits["word_recognised"] = 0.37
                reward += 0.37
                s["success"] = True
            else:
                credits["word_recognised"] = 0.0

            wrong = [c for c in action.output_text.upper() if c.isalpha() and c not in target]
            if wrong:
                penalty = min(0.20, 0.04 * len(wrong))
                credits["wrong_letter_penalty"] = -round(penalty, 4)
                reward -= penalty

        # ── CRITICAL FIX: clip to strictly open interval (0, 1) ──────────
        reward = clip_score(reward)
        return reward, credits

    def _compute_grader_score(self) -> float:
        """
        Compute grader score strictly in (0, 1) — never 0.0 or 1.0.

        SUCCESS → 0.99 (not 1.0)
        PARTIAL → mean of trajectory rewards, floored at 0.01
        NO STEPS → 0.01 (not 0.0)
        """
        traj = self._s.get("trajectory_rewards", [])

        if self._s.get("success"):
            return SCORE_MAX  # 0.99

        if not traj:
            return SCORE_MIN  # 0.01

        raw = float(np.mean(traj))
        return clip_score(raw)

    def reset(self, task_id: str = "blind_mode") -> Observation:
        self._init_state(task_id)
        return self._build_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self._s.get("done"):
            raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

        self._s["step"] += 1
        reward, credits = self._compute_reward(action)
        self._s["trajectory_rewards"].append(reward)
        self._s["partial_scores"] = credits

        done = self._s["step"] >= MAX_STEPS or self._s["success"]
        self._s["done"] = done

        extra = {}
        task = self._s["task_id"]
        data = self._s["scenario"]

        if task == "blind_mode" and self._s["success"]:
            extra["audio_output"] = f"TTS: {data.get('camera', '')[:80]}"
        elif task == "deaf_mode" and self._s["success"]:
            extra["oled_display"] = data["microphone"]
        elif task == "mute_mode" and self._s["success"]:
            extra["audio_output"] = f"TTS: {data['target_word']}"

        obs = self._build_obs(extra)
        grader_score = self._compute_grader_score()

        info = {
            "grader_score": grader_score,
            "partial_credits": credits,
            "success": self._s["success"],
            "episode_id": self._s["episode_id"],
        }
        return obs, reward, done, info

    def get_state(self) -> dict:
        return {
            "episode_id": self._s.get("episode_id"),
            "task_id": self._s.get("task_id"),
            "step": self._s.get("step"),
            "done": self._s.get("done"),
            "success": self._s.get("success"),
            "trajectory_rewards": self._s.get("trajectory_rewards", []),
            "partial_scores": self._s.get("partial_scores", {}),
        }

    def grader_score(self) -> float:
        return self._compute_grader_score()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LUMOS Assistive AI — OpenEnv",
    description=(
        "OpenEnv environment simulating real-world assistive AI tasks "
        "for people with visual, hearing, and speech impairments."
    ),
    version="1.0.0",
)

env = LumosEnv()


@app.get("/")
def root():
    return {
        "name": "lumos-assistive-ai",
        "version": "1.0.0",
        "tasks": list(SCENARIOS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
        "score_range": f"({SCORE_MIN}, {SCORE_MAX}) exclusive",
    }


@app.post("/reset")
def reset_endpoint(task_id: str = "blind_mode", fixed_idx: int = -1):
    global env
    import uuid as _uuid
    env = LumosEnv()
    if fixed_idx >= 0 and task_id in SCENARIOS:
        idx = fixed_idx % len(SCENARIOS[task_id])
        env._s = {
            "episode_id": str(_uuid.uuid4())[:8],
            "task_id": task_id,
            "step": 0,
            "scenario": SCENARIOS[task_id][idx],
            "trajectory_rewards": [],
            "success": False,
            "partial_scores": {},
            "done": False,
        }
        return env._build_obs().model_dump()
    obs = env.reset(task_id)
    return obs.model_dump()


@app.post("/step")
def step_endpoint(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state_endpoint():
    return env.get_state()


@app.get("/tasks")
def tasks_endpoint():
    return [
        {
            "id": task_id,
            "description": meta["description"],
            "difficulty": meta["difficulty"],
            "valid_decisions": meta["valid_decisions"],
            "action_schema": meta["action_schema"],
            "scenario_count": len(SCENARIOS[task_id]),
        }
        for task_id, meta in TASK_METADATA.items()
    ]


@app.post("/grader")
def grader_endpoint():
    score = env.grader_score()
    return {
        "grader_score": score,
        "success": env._s.get("success", False),
        "steps_taken": env._s.get("step", 0),
        "episode_id": env._s.get("episode_id"),
    }


@app.get("/baseline")
def baseline_endpoint():
    import json
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    if not api_key:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set.")

    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base_url, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI client error: {e}")

    FIXED_IDX = {"blind_mode": 0, "deaf_mode": 0, "mute_mode": 2}

    SYSTEM_PROMPT = """You are an AI agent controlling LUMOS Assistive Glasses for people with disabilities.

Rules:
- blind_mode → decision: describe_scene | ocr_read | alert_danger
  * alert_danger: hazards present (burner, cleaver, lorry, jackhammer, autoclave, effluent, biohazard)
  * ocr_read: voice_command contains 'read' or camera shows text/signage
  * describe_scene: safe navigation without hazards
- deaf_mode → decision: speech_to_text. Relay ALL content verbatim — including rare technical terms.
- mute_mode → decision: sign_speech. Output the COMPLETE word spelled by all frames seen so far.

Respond with valid JSON only:
{"decision": "<decision>", "output_text": "<output>", "confidence": <0.0-1.0>}"""

    results = {}

    for task_id in ["blind_mode", "deaf_mode", "mute_mode"]:
        fixed_idx = FIXED_IDX[task_id]
        import uuid as _uuid
        test_env = LumosEnv()
        test_env._s = {
            "episode_id": str(_uuid.uuid4())[:8],
            "task_id": task_id,
            "step": 0,
            "scenario": SCENARIOS[task_id][fixed_idx],
            "trajectory_rewards": [],
            "success": False,
            "partial_scores": {},
            "done": False,
        }
        obs = test_env._build_obs()
        final_info: dict = {}

        for _ in range(MAX_STEPS):
            user_msg = (
                f"Task: {obs.task_id}\n"
                f"Camera: {obs.camera_feed}\n"
                f"Microphone: {obs.microphone}\n"
                f"Voice command: {obs.voice_command}\n"
                f"ASL letter: {obs.asl_letter}\n"
                f"Spelled so far: {obs.spelled_word}\n"
                f"Hint: {obs.hint}\n"
                f"Step: {obs.step_number}"
            )
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=120,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                clean = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
                parsed = json.loads(clean)
                action = Action(
                    decision=str(parsed.get("decision", "describe_scene")),
                    output_text=str(parsed.get("output_text", "")),
                    confidence=float(parsed.get("confidence", 0.8)),
                )
            except Exception:
                action = Action(decision="describe_scene", output_text="", confidence=0.5)

            obs, reward, done, final_info = test_env.step(action)
            if done:
                break

        results[task_id] = {
            "grader_score": round(final_info.get("grader_score", SCORE_MIN), 4),
            "success": final_info.get("success", False),
        }

    avg = round(sum(r["grader_score"] for r in results.values()) / 3, 4)
    return {
        "model": model_name,
        "temperature": 0,
        "reproducible": True,
        "scores": results,
        "average_score": avg,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
