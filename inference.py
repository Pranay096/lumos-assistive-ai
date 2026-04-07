"""
inference.py — LUMOS Assistive AI OpenEnv
Mandatory stdout format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP] step=N action=... reward=0.00 done=false error=null
  [END] success=true steps=N score=0.00 rewards=0.00,...
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

BASELINE_IDX = {"blind_mode": 0, "deaf_mode": 0, "mute_mode": 2}  # fixed: deaf_mode 5→0

SYSTEM_PROMPT = """You are an AI agent controlling LUMOS Assistive Glasses for people with disabilities.
- blind_mode: decision must be one of: describe_scene | ocr_read | alert_danger
- deaf_mode: always use speech_to_text and relay ALL key info from microphone field
- mute_mode: always use sign_speech and output the exact word being spelled
JSON only: {"decision":"...","output_text":"...","confidence":0.9}"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def env_reset(task_id: str, fixed_idx: int) -> dict:
    r = requests.post(f"{BASE_ENV_URL}/reset", params={"task_id": task_id, "fixed_idx": fixed_idx}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{BASE_ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def call_llm(obs: dict, client: OpenAI) -> dict:
    user_msg = (
        f"Task: {obs.get('task_id')}\n"
        f"Camera: {obs.get('camera_feed', '')}\n"
        f"Microphone: {obs.get('microphone', '')}\n"
        f"Voice command: {obs.get('voice_command', '')}\n"
        f"ASL letter: {obs.get('asl_letter')}\n"
        f"Spelled so far: {obs.get('spelled_word')}\n"
        f"Hint: {obs.get('hint', '')}\n"
        f"Step: {obs.get('step_number', 0)}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=120,
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
        time.sleep(delay)
    return False


def run_episode(task_id: str, client: OpenAI) -> dict:
    fixed_idx = BASELINE_IDX[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
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
            result = env_step({"decision": decision, "output_text": output_text, "confidence": confidence})

            obs     = result.get("observation", obs)
            reward  = float(result.get("reward", 0.0))
            done    = bool(result.get("done", False))
            info    = result.get("info", {})
            error   = None
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                success = bool(info.get("success", False))
                score   = float(info.get("grader_score", 0.0))
                break

        if not rewards:
            score = 0.0
        elif not done:
            score = float(sum(rewards) / len(rewards))

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "grader_score": round(score, 4), "success": success, "steps_taken": steps_taken}


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
    avg = round(sum(r["grader_score"] for r in results) / len(results), 4)

    output = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "reproducible": True,
        "scores": {r["task_id"]: {"grader_score": r["grader_score"], "success": r["success"]} for r in results},
        "average_score": avg,
    }
    print(json.dumps(output, indent=2), flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()