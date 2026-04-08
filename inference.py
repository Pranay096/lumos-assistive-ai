"""
inference.py — LUMOS Assistive AI OpenEnv
Mandatory stdout format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP] step=N action=... reward=0.00 done=false error=null
  [END] success=true steps=N score=0.00 rewards=0.00,...

SCORING FIX: All scores strictly in (0.01, 0.99) — never 0.0 or 1.0.
"""

import json
import os
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "lumos-assistive-ai"
MAX_STEPS    = 8

# Score bounds — strictly open interval (0, 1)
SCORE_MIN = 0.01
SCORE_MAX = 0.99

BASELINE_IDX = {"blind_mode": 0, "deaf_mode": 0, "mute_mode": 2}

SYSTEM_PROMPT = """You are an AI agent controlling LUMOS Assistive Glasses for people with disabilities.

Rules:
- blind_mode: decision must be one of: describe_scene | ocr_read | alert_danger
  * Use alert_danger when camera shows hazards (stove, knife, car, wet floor, machinery)
  * Use ocr_read when voice_command contains "read" or camera shows text/book
  * Use describe_scene for general navigation without hazards
- deaf_mode: always use speech_to_text. Relay ALL key info from the microphone field verbatim.
- mute_mode: always use sign_speech.
  * The spelled_word field shows ALL letters detected so far.
  * The asl_letter field shows the CURRENT letter being shown.
  * Output the COMPLETE word accumulated so far in output_text.
  * When you see the full word spelled out, output the entire word (e.g., "THANKS", "HELLO").
  * Do NOT output just one letter — always output the full spelled_word value.

Respond with valid JSON only:
{"decision": "...", "output_text": "...", "confidence": 0.9}"""


def clip_score(score: float) -> float:
    """Ensure score is strictly in (0, 1) — never 0.0 or 1.0."""
    return max(SCORE_MIN, min(SCORE_MAX, float(score)))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Reward also must be in valid range for logging
    reward = clip_score(reward)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # CRITICAL: score must be strictly between 0 and 1
    score = clip_score(score)
    rewards_str = ",".join(f"{clip_score(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def env_reset(task_id: str, fixed_idx: int) -> dict:
    r = requests.post(
        f"{BASE_ENV_URL}/reset",
        params={"task_id": task_id, "fixed_idx": fixed_idx},
        timeout=30
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{BASE_ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def call_llm(obs: dict, client: OpenAI) -> dict:
    task = obs.get("task_id", "")
    spelled = obs.get("spelled_word") or ""
    asl_letter = obs.get("asl_letter") or ""
    step = obs.get("step_number", 0)

    mute_hint = ""
    if task == "mute_mode" and spelled:
        mute_hint = f"\nIMPORTANT: The word spelled so far is '{spelled}'. Output this complete word in output_text."

    user_msg = (
        f"Task: {obs.get('task_id')}\n"
        f"Camera: {obs.get('camera_feed', '')}\n"
        f"Microphone: {obs.get('microphone', '')}\n"
        f"Voice command: {obs.get('voice_command', '')}\n"
        f"ASL letter (current): {asl_letter}\n"
        f"Spelled so far (complete): {spelled}\n"
        f"Hint: {obs.get('hint', '')}\n"
        f"Step: {step}"
        f"{mute_hint}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content or "{}"
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(clean)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return {"decision": "describe_scene", "output_text": "", "confidence": 0.5}


def wait_for_server(retries: int = 15, delay: float = 3.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{BASE_ENV_URL}/", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        print(f"[DEBUG] Waiting for server... attempt {i+1}/{retries}", flush=True)
        time.sleep(delay)
    return False


def run_episode(task_id: str, client: OpenAI) -> dict:
    fixed_idx = BASELINE_IDX[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score = SCORE_MIN  # default floor — never 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id, fixed_idx)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            parsed = call_llm(obs, client)
            decision    = str(parsed.get("decision", "describe_scene"))
            output_text = str(parsed.get("output_text", ""))
            confidence  = float(max(0.0, min(1.0, parsed.get("confidence", 0.8))))

            action_str = f"{decision}|{output_text[:30]}"
            result = env_step({
                "decision": decision,
                "output_text": output_text,
                "confidence": confidence
            })

            obs     = result.get("observation", obs)
            reward  = clip_score(float(result.get("reward", SCORE_MIN)))
            done    = bool(result.get("done", False))
            info    = result.get("info", {})
            error   = None
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                success = bool(info.get("success", False))
                # grader_score from server is already clipped in app.py
                raw_score = float(info.get("grader_score", SCORE_MIN))
                score = clip_score(raw_score)
                break

        if not rewards:
            score = SCORE_MIN
        elif not done:
            # Episode ended without done=True — use mean of rewards
            raw_score = float(sum(rewards) / len(rewards))
            score = clip_score(raw_score)

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = SCORE_MIN
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task_id": task_id,
        "grader_score": round(score, 4),
        "success": success,
        "steps_taken": steps_taken
    }


def main():
    if not API_KEY:
        print("[ERROR] HF_TOKEN is not set.", flush=True)
        sys.exit(1)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI client: {e}", flush=True)
        sys.exit(1)

    if not wait_for_server():
        print(f"[ERROR] Cannot reach server at {BASE_ENV_URL}", flush=True)
        sys.exit(1)

    results = [run_episode(tid, client) for tid in ["blind_mode", "deaf_mode", "mute_mode"]]

    # Average also clipped
    avg = clip_score(sum(r["grader_score"] for r in results) / len(results))

    output = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "reproducible": True,
        "scores": {
            r["task_id"]: {
                "grader_score": r["grader_score"],
                "success": r["success"]
            }
            for r in results
        },
        "average_score": round(avg, 4),
    }
    print(json.dumps(output, indent=2), flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
