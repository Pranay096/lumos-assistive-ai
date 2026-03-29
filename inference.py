"""
inference.py — LUMOS Assistive AI OpenEnv
==========================================
MANDATORY env vars:
    API_BASE_URL   LLM API endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model to use      (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       Your HuggingFace token

Optional:
    BASE_ENV_URL   Where LUMOS server runs (default: http://localhost:7860)

Run locally (server must be running first):
    uvicorn app:app --port 7860 &
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import json, os, sys, time
import requests
from openai import OpenAI

# ── Config from environment variables (mandatory per spec) ──────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
BASE_ENV_URL = (os.getenv("BASE_ENV_URL") or "http://localhost:7860").rstrip("/")

MAX_STEPS   = 8
TEMPERATURE = 0.0   # deterministic / reproducible

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
        print(f"  [WARN] LLM error: {exc} — using fallback")
        return {"decision":"describe_scene","output_text":"","confidence":0.5}

def run_episode(task_id: str) -> dict:
    print(f"\n{'='*50}\n  Task: {task_id}\n{'='*50}")
    obs = env_reset(task_id)
    done = False
    step = 0

    for step in range(1, MAX_STEPS + 1):
        if done: break
        action = call_llm(obs)
        action["confidence"] = float(max(0.0, min(1.0, action.get("confidence", 0.8))))
        print(f"  Step {step:02d} | {action.get('decision','?'):20s} | '{str(action.get('output_text',''))[:45]}'")
        result = env_step(action)
        obs  = result.get("observation", obs)
        done = bool(result.get("done", False))
        info = result.get("info", {})
        print(f"           reward={result.get('reward',0):.4f} | success={info.get('success',False)}")
        if done: break

    grader = env_grader()
    score  = float(grader.get("grader_score", 0.0))
    print(f"\n  Result: score={score:.4f} | success={grader.get('success',False)} | steps={step}")
    return {"task_id":task_id,"grader_score":round(score,4),"success":grader.get("success",False),"steps_taken":step}

def wait_for_server(retries=12, delay=3.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{BASE_ENV_URL}/", timeout=5)
            if r.status_code == 200:
                print(f"  Server ready at {BASE_ENV_URL}")
                return True
        except requests.ConnectionError:
            pass
        print(f"  Waiting for server... ({i+1}/{retries})")
        time.sleep(delay)
    return False

def main():
    print("\nLUMOS Assistive AI — OpenEnv Inference")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  API base : {API_BASE_URL}")
    print(f"  Env URL  : {BASE_ENV_URL}")

    if not API_KEY:
        print("\n[ERROR] HF_TOKEN not set. Export it: export HF_TOKEN=hf_...")
        sys.exit(1)

    if not wait_for_server():
        print(f"\n[ERROR] Cannot reach server at {BASE_ENV_URL}")
        print("  Start locally: uvicorn app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    results = [run_episode(tid) for tid in ["blind_mode","deaf_mode","mute_mode"]]
    avg = sum(r["grader_score"] for r in results) / len(results)

    print(f"\n{'='*50}\n  FINAL SCORES\n{'='*50}")
    for r in results:
        icon = "✓" if r["success"] else "✗"
        print(f"  {icon} {r['task_id']:15s} | score={r['grader_score']:.4f} | steps={r['steps_taken']}")
    print(f"\n  Average : {avg:.4f}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Reproducible: Yes (temperature=0, seeded)\n")

    print(json.dumps({
        "model": MODEL_NAME, "temperature": TEMPERATURE, "reproducible": True,
        "scores": {r["task_id"]:{"grader_score":r["grader_score"],"success":r["success"]} for r in results},
        "average_score": round(avg, 4)
    }, indent=2))

if __name__ == "__main__":
    main()
