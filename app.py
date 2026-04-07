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
# Pydantic Models (OpenEnv spec)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id: str
    camera_feed: str          # Scene description / current ASL frame
    microphone: str           # Raw speech input (deaf mode)
    voice_command: str        # Trigger word / command ("LUMOS", "read book")
    oled_display: Optional[str] = None   # Text shown on transparent OLED
    audio_output: Optional[str] = None  # TTS string sent to speaker
    asl_letter: Optional[str] = None    # Current detected ASL letter
    spelled_word: Optional[str] = None  # Complete word spelled so far
    step_number: int
    hint: str = ""            # Partial-progress hint for agent


class Action(BaseModel):
    decision: str = Field(
        ...,
        description=(
            "One of: describe_scene | ocr_read | alert_danger | "
            "speech_to_text | sign_speech"
        )
    )
    output_text: str = Field(
        default="",
        description="The agent's generated output (scene description, relayed speech, etc.)"
    )
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
            "camera": "A kitchen counter. Hot stove on the left. Knife near the edge.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["stove", "knife"],
            "dangers": ["stove", "knife"],
            "difficulty": "easy",
        },
        {
            "camera": "Busy street. Car approaching fast from the right. Clear path ahead.",
            "voice": "LUMOS",
            "expect_decision": "alert_danger",
            "key_objects": ["car", "street"],
            "dangers": ["car"],
            "difficulty": "easy",
        },
        {
            "camera": "Living room. Couch ahead, lamp to the left, clear path to door.",
            "voice": "LUMOS",
            "expect_decision": "describe_scene",
            "key_objects": ["couch", "lamp", "door"],
            "dangers": [],
            "difficulty": "easy",
        },
        {
            "camera": "Open book. Page reads: EMERGENCY EXIT THIS WAY ARROW RIGHT",
            "voice": "read book",
            "expect_decision": "ocr_read",
            "key_objects": ["book", "exit"],
            "dangers": [],
            "difficulty": "easy",
        },
    ],

    # ── DEAF MODE (medium) ─────────────────────────────────────────────────
    "deaf_mode": [
        {
            "microphone": "Doctor appointment at 3 PM in Conference Room B",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["doctor", "3", "conference"],
            "difficulty": "medium",
        },
        {
            "microphone": "Watch out there is a car coming from behind you",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["watch out", "car", "behind"],
            "difficulty": "medium",
        },
        {
            "microphone": "The package you ordered has arrived at the front desk",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["package", "arrived", "front desk"],
            "difficulty": "medium",
        },
        {
            "microphone": "Fire alarm will be tested at noon today please do not panic",
            "voice": "",
            "expect_decision": "speech_to_text",
            "key_words": ["fire alarm", "noon", "today"],
            "difficulty": "medium",
        },
    ],

    # ── MUTE MODE (hard) ───────────────────────────────────────────────────
    "mute_mode": [
        {
            "camera_frames": ["hand_H", "hand_E", "hand_L", "hand_L", "hand_O"],
            "target_word": "HELLO",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_H", "hand_E", "hand_L", "hand_P"],
            "target_word": "HELP",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_T", "hand_H", "hand_A", "hand_N", "hand_K", "hand_S"],
            "target_word": "THANKS",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
        {
            "camera_frames": ["hand_W", "hand_A", "hand_T", "hand_E", "hand_R"],
            "target_word": "WATER",
            "voice": "",
            "expect_decision": "sign_speech",
            "difficulty": "hard",
        },
    ],
}

TASK_METADATA = {
    "blind_mode": {
        "description": "Agent must interpret camera feed and produce correct audio output for a blind user. Includes scene description, OCR reading, and danger alerts.",
        "difficulty": "easy",
        "valid_decisions": ["describe_scene", "ocr_read", "alert_danger"],
        "action_schema": {"decision": "string", "output_text": "string", "confidence": "float [0,1]"},
    },
    "deaf_mode": {
        "description": "Agent must relay spoken words as clear text for display on the OLED screen worn by a deaf user.",
        "difficulty": "medium",
        "valid_decisions": ["speech_to_text"],
        "action_schema": {"decision": "string", "output_text": "string", "confidence": "float [0,1]"},
    },
    "mute_mode": {
        "description": "Agent must recognise ASL finger-spelling frames one at a time, accumulate the word, and produce TTS speech output for a mute user.",
        "difficulty": "hard",
        "valid_decisions": ["sign_speech"],
        "action_schema": {"decision": "string", "output_text": "string", "confidence": "float [0,1]"},
    },
}

# ---------------------------------------------------------------------------
# ASL Letter Detector (deterministic — no model weight needed)
# ---------------------------------------------------------------------------

class ASLLetterDetector:
    """
    Deterministic letter extractor from frame description strings.
    Frame format: "hand_X"  where X is A-Z.
    Falls back gracefully for noisy descriptions.
    """
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
        self._s: dict = {}   # internal state (not named 'state' to avoid clash)
        self._init_state("blind_mode")

    # ── private ────────────────────────────────────────────────────────────

    def _init_state(self, task_id: str):
        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(SCENARIOS.keys())}")
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
            idx = min(s["step"], len(frames) - 1)
            camera = frames[idx]
            asl_letter = asl_detector.predict(camera)
            spelled = asl_detector.spell(frames[: s["step"] + 1])
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
        """Partial-progress hints to give the agent signal."""
        s = self._s
        task = s["task_id"]
        data = s["scenario"]
        step = s["step"]

        if task == "blind_mode":
            dangers = data.get("dangers", [])
            if dangers:
                return f"Danger objects present: {', '.join(dangers)}"
            return f"Key objects: {', '.join(data.get('key_objects', []))}"
        elif task == "deaf_mode":
            words = data["microphone"].split()
            visible = words[: max(1, step)]
            return f"Heard so far: '{' '.join(visible)}'"
        elif task == "mute_mode":
            seen = asl_detector.spell(data["camera_frames"][: step + 1])
            return f"Letters detected so far: {seen}"
        return ""

    # ── reward shaping ─────────────────────────────────────────────────────

    def _compute_reward(self, action: Action) -> Tuple[float, dict]:
        s = self._s
        task = s["task_id"]
        data = s["scenario"]
        credits: dict = {}
        reward = 0.0

        # confidence bonus (always, small)
        credits["confidence"] = round(0.05 * action.confidence, 4)
        reward += credits["confidence"]

        if task == "blind_mode":
            correct_decision = action.decision == data["expect_decision"]
            credits["correct_decision"] = 0.40 if correct_decision else 0.0
            reward += credits["correct_decision"]

            # partial: did agent mention key objects in output?
            output_lower = action.output_text.lower()
            key_hits = sum(1 for k in data.get("key_objects", []) if k in output_lower)
            frac = key_hits / max(1, len(data.get("key_objects", ["x"])))
            credits["key_objects_mentioned"] = round(0.30 * frac, 4)
            reward += credits["key_objects_mentioned"]

            # danger flag
            dangers = data.get("dangers", [])
            if dangers:
                danger_hits = sum(1 for d in dangers if d in output_lower)
                credits["danger_flagged"] = round(0.25 * danger_hits / len(dangers), 4)
                reward += credits["danger_flagged"]
                if danger_hits == len(dangers):
                    s["success"] = True
            else:
                credits["danger_flagged"] = 0.0
                if correct_decision and frac >= 0.5:
                    s["success"] = True

        elif task == "deaf_mode":
            correct_decision = action.decision == "speech_to_text"
            credits["correct_decision"] = 0.30 if correct_decision else 0.0
            reward += credits["correct_decision"]

            key_words = data.get("key_words", [])
            output_lower = action.output_text.lower()
            hits = sum(1 for kw in key_words if kw.lower() in output_lower)
            frac = hits / max(1, len(key_words))
            credits["key_words_relayed"] = round(0.55 * frac, 4)
            reward += credits["key_words_relayed"]

            if correct_decision and frac >= 0.67:
                s["success"] = True

        elif task == "mute_mode":
            correct_decision = action.decision == "sign_speech"
            credits["correct_decision"] = 0.20 if correct_decision else 0.0
            reward += credits["correct_decision"]

            frames = data["camera_frames"]
            target = data["target_word"]
            spelled_so_far = asl_detector.spell(frames[: s["step"] + 1])
            target_so_far = target[: s["step"] + 1]

            # letter-level accuracy
            matches = sum(a == b for a, b in zip(spelled_so_far, target_so_far))
            letter_acc = matches / max(1, len(target_so_far))
            credits["letter_accuracy"] = round(0.40 * letter_acc, 4)
            reward += credits["letter_accuracy"]

            # word-level output check (final steps)
            if target.lower() in action.output_text.lower():
                credits["word_recognised"] = 0.35
                reward += 0.35
                s["success"] = True
            else:
                credits["word_recognised"] = 0.0

            # penalty: wrong letter in output
            wrong = [c for c in action.output_text.upper() if c.isalpha() and c not in target]
            if wrong:
                penalty = min(0.15, 0.03 * len(wrong))
                credits["wrong_letter_penalty"] = -round(penalty, 4)
                reward -= penalty

        reward = float(np.clip(reward, 0.0, 1.0))
        return reward, credits

    # ── public API ─────────────────────────────────────────────────────────

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

        traj = self._s["trajectory_rewards"]
        grader_score = 1.0 if self._s["success"] else float(np.mean(traj))
        grader_score = float(np.clip(grader_score, 0.0, 1.0))

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
        traj = self._s.get("trajectory_rewards", [0.0])
        if self._s.get("success"):
            return 1.0
        score = float(np.mean(traj)) if traj else 0.0
        return float(np.clip(score, 0.0, 1.0))


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

# One environment instance per server process (single-user HF Space)
env = LumosEnv()


@app.get("/")
def root():
    return {
        "name": "lumos-assistive-ai",
        "version": "1.0.0",
        "tasks": list(SCENARIOS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }


@app.post("/reset")
def reset_endpoint(task_id: str = "blind_mode", fixed_idx: int = -1):
    """Reset environment. fixed_idx>=0 pins a specific scenario for reproducibility."""
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
    """Take one step in the environment."""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state_endpoint():
    """Return current internal state."""
    return env.get_state()


@app.get("/tasks")
def tasks_endpoint():
    """Return list of tasks and their action schemas."""
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
    """Return grader score for the current episode."""
    return {
        "grader_score": env.grader_score(),
        "success": env._s.get("success", False),
        "steps_taken": env._s.get("step", 0),
        "episode_id": env._s.get("episode_id"),
    }


@app.get("/baseline")
def baseline_endpoint():
    """
    Run LLM agent against all 3 tasks. Returns reproducible scores.
    Requires HF_TOKEN (or OPENAI_API_KEY), API_BASE_URL, MODEL_NAME env vars.
    """
    import json
    api_key      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN not set. Add it as an HF Space Secret.",
        )

    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base_url, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI client error: {e}")

    # Fixed scenario index per task — same scenario every run = reproducible
    FIXED_IDX = {"blind_mode": 0, "deaf_mode": 0, "mute_mode": 2}

    SYSTEM_PROMPT = """You are an AI agent controlling LUMOS Assistive Glasses for people with disabilities.

Rules:
- blind_mode → decision: describe_scene | ocr_read | alert_danger
  * alert_danger: hazards (stove, knife, car, wet floor, machinery, traffic)
  * ocr_read: voice_command contains "read" or camera shows text
  * describe_scene: safe, general navigation
- deaf_mode → decision: speech_to_text. Relay ALL key info from microphone.
- mute_mode → decision: sign_speech. Output the word being spelled.

Respond with valid JSON only:
{"decision": "<decision>", "output_text": "<output>", "confidence": <0.0-1.0>}"""

    results = {}

    for task_id in ["blind_mode", "deaf_mode", "mute_mode"]:
        fixed_idx = FIXED_IDX[task_id]
        test_env = LumosEnv()
        # Pin to fixed scenario for reproducibility
        import uuid as _uuid
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
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=120,
                    temperature=0,
                )
                raw   = resp.choices[0].message.content.strip()
                clean = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
                parsed = json.loads(clean)
                action = Action(
                    decision    = str(parsed.get("decision", "describe_scene")),
                    output_text = str(parsed.get("output_text", "")),
                    confidence  = float(parsed.get("confidence", 0.8)),
                )
            except Exception:
                action = Action(decision="describe_scene", output_text="", confidence=0.5)

            obs, reward, done, final_info = test_env.step(action)
            if done:
                break

        results[task_id] = {
            "grader_score": round(final_info.get("grader_score", 0.0), 4),
            "success":      final_info.get("success", False),
        }

    avg = round(sum(r["grader_score"] for r in results.values()) / 3, 4)
    return {
        "model":        model_name,
        "temperature":  0,
        "reproducible": True,
        "scores":       results,
        "average_score": avg,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
