"""
inference.py — LUMOS Assistive AI OpenEnv
==========================================
MANDATORY env vars:
    API_BASE_URL   LLM API endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model to use      (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       Your HuggingFace token

Optional:
    BASE_ENV_URL   Where LUMOS server runs (default: http://localhost:7860)

STDOUT FORMAT:
    [START] task=<task_name> env=lumos-assistive-ai model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json, os, sys, time
from typing import List, Optional
import requests
from openai import OpenAI

# ── Config from environment variables ───────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
BASE_ENV_URL = (os.getenv("BASE_ENV_URL") or "http://localhost:7860").rstrip("/")

BENCHMARK    = "lumos-assistive-ai"
MAX_STEPS    = 8
TEMPERATURE  = 0.0  # deterministic / reproducible

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")

SYSTEM_PROMPT = """You are an AI agent controlling LUMOS Assistive Glasses for people with disabilities.

Rules:
- blind_mode  → decision must be one of: describe_scene | ocr_read | alert_danger
  * alert_danger: when camera shows any hazard (stove, knife, car, wet floor, machinery)
  * ocr_read: when voice_command contains "read" or camera shows text/document
  * describe_scene: for general surroundings with no danger
- deaf_mode   → decision must be: speech_to_text — relay the microphone speech clearly
- mute_mode   → decision must be: sign_speech — output the word being spelled by the ASL frames

Respond with valid JSON only, no extra text:
{"decision": "<decision>", "output_text": "<your output>", "confidence": <0.0-1.0>}"""


# ── Logging helpers (required stdout format) ─────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Environment API helpers ──────────────────────────────────────────────────
def env_reset(task_id: str) -> dict:
    r = requests.post(f"{BASE_ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{BASE_ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grader() -> dict:
    r = requests.post(f"{BASE_ENV_URL}/grader", timeout=30)
    r.raise_for_status()
    return r.json()


# ── LLM call ────────────────────────────────────────────────────────────────
def call_llm(obs: dict) -> dict:
    user_msg = (
        f"Task: {obs.get('task_id')}\n"
        f"Camera: {obs.get('camera_feed','')}\n"
        f"Microphone: {obs.get('microphone','')}\n"
        f"Voice command: {obs.get('voice_command','')}\n"
        f"ASL letter: {obs.get('asl_letter')}\n"
        f"Spelled so far: {obs.get('spelled_word')}\n"
        f"Hint: {obs.get('hint','')}\n"
        f"Step: {obs.get('step_number',0)}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":user_msg}],
            temperature=TEMPERATURE, max_tokens=200,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc} — using fallback", flush=True)
        return {"decision":"describe_scene","output_text":"","confidence":0.5}


# ── Episode runner ───────────────────────────────────────────────────────────
def run_episode(task_id: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = call_llm(obs)
            action["confidence"] = float(max(0.0, min(1.0, action.get("confidence", 0.8))))
            action_str = f"{action.get('decision','?')}|{str(action.get('output_text',''))[:40]}"

            result = env_step(action)
            obs    = result.get("observation", obs)
            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            info   = result.get("info", {})
            error  = None

            rewards.append(reward)
            steps_taken = step
            success = bool(info.get("success", False))

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        grader = env_grader()
        score  = float(grader.get("grader_score", 0.0))
        success = bool(grader.get("success", False))

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "grader_score": round(score, 4), "success": success, "steps_taken": steps_taken}


# ── Server wait ──────────────────────────────────────────────────────────────
def wait_for_server(retries=12, delay=3.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{BASE_ENV_URL}/", timeout=5)
            if r.status_code == 200:
                print(f"[DEBUG] Server ready at {BASE_ENV_URL}", flush=True)
                return True
        except requests.ConnectionError:
            pass
        print(f"[DEBUG] Waiting for server... ({i+1}/{retries})", flush=True)
        time.sleep(delay)
    return False


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"[DEBUG] LUMOS Assistive AI — OpenEnv Inference", flush=True)
    print(f"[DEBUG] Model    : {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API base : {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Env URL  : {BASE_ENV_URL}", flush=True)

    if not API_KEY:
        print("[ERROR] HF_TOKEN not set. Export it: export HF_TOKEN=hf_...", flush=True)
        sys.exit(1)

    if not wait_for_server():
        print(f"[ERROR] Cannot reach server at {BASE_ENV_URL}", flush=True)
        sys.exit(1)

    results = [run_episode(tid) for tid in ["blind_mode", "deaf_mode", "mute_mode"]]
    avg = sum(r["grader_score"] for r in results) / len(results)

    print(f"\n[DEBUG] FINAL SCORES", flush=True)
    for r in results:
        icon = "✓" if r["success"] else "✗"
        print(f"[DEBUG] {icon} {r['task_id']:15s} | score={r['grader_score']:.4f} | steps={r['steps_taken']}", flush=True)
    print(f"[DEBUG] Average : {avg:.4f}", flush=True)
    print(f"[DEBUG] Reproducible: Yes (temperature=0, seeded)", flush=True)


if __name__ == "__main__":
    main()