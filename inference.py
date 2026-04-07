"""
inference.py — LUMOS Assistive AI OpenEnv
==========================================
MANDATORY environment variables (set as HF Space Secrets):
    API_BASE_URL   LLM API endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier   (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       HuggingFace token

Optional:
    BASE_ENV_URL   Where LUMOS server runs (default: http://localhost:7860)

Run:
    uvicorn app:app --host 0.0.0.0 --port 7860
    python inference.py
"""

import json
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Config — read ONLY from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:7860").rstrip("/")

TEMPERATURE = 0.0   # deterministic / reproducible

# Fixed scenario indices — same scenario every run = reproducible scores
BASELINE_IDX = {"blind_mode": 0, "deaf_mode": 5, "mute_mode": 2}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI agent controlling LUMOS Assistive Glasses for people with disabilities.

RULES:
- blind_mode → decision must be one of: describe_scene | ocr_read | alert_danger
  * alert_danger: any hazard (stove, knife, car, wet floor, machinery, traffic, stairs)
  * ocr_read: when voice_command contains "read" or camera shows text/document
  * describe_scene: safe environment, general navigation
- deaf_mode → decision: speech_to_text. Relay ALL key info from microphone field.
- mute_mode → decision: sign_speech.
  * asl_letter shows the current frame letter (may be ambiguous, e.g. "M or N?")
  * spelled_word shows letters accumulated so far
  * hint tells you frames remaining
  * DO NOT guess the full word until hint says "All frames shown"
  * When all frames shown, output ONLY the exact word — no extra text

Respond ONLY with valid JSON, no markdown:
{"decision": "<decision>", "output_text": "<your output>", "confidence": <0.0-1.0>}"""

# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, fixed_idx: int) -> dict:
    r = requests.post(
        f"{BASE_ENV_URL}/reset",
        params={"task_id": task_id, "fixed_idx": fixed_idx},
        timeout=30,
    )
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


# ---------------------------------------------------------------------------
# LLM call — client is created inside, not at module level
# ---------------------------------------------------------------------------

def call_llm(obs: dict, client) -> dict:
    user_msg = (
        f"Task: {obs.get('task_id')}\n"
        f"Camera: {obs.get('camera_feed', '')}\n"
        f"Microphone: {obs.get('microphone', '')}\n"
        f"Voice command: {obs.get('voice_command', '')}\n"
        f"ASL letter (may be ambiguous): {obs.get('asl_letter')}\n"
        f"Spelled so far: {obs.get('spelled_word')}\n"
        f"Hint: {obs.get('hint', '')}\n"
        f"Step: {obs.get('step_number', 0)}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=120,
        )
        raw   = resp.choices[0].message.content or "{}"
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(clean)
    except Exception as exc:
        print(f"  [WARN] LLM error: {exc}", flush=True)
        return {"decision": "describe_scene", "output_text": "", "confidence": 0.5}


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client) -> dict:
    MAX_STEPS = 12
    print(f"\n{'='*50}", flush=True)
    print(f"  Task: {task_id.upper()}", flush=True)
    print(f"{'='*50}", flush=True)

    fixed_idx = BASELINE_IDX[task_id]
    obs  = env_reset(task_id, fixed_idx)
    done = False
    step = 0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action = call_llm(obs, client)
        decision    = str(action.get("decision", "describe_scene"))
        output_text = str(action.get("output_text", ""))
        confidence  = float(max(0.0, min(1.0, action.get("confidence", 0.8))))

        print(f"  Step {step:02d} | {decision:20s} | '{output_text[:45]}'", flush=True)

        result = env_step({
            "decision":    decision,
            "output_text": output_text,
            "confidence":  confidence,
        })

        obs   = result.get("observation", obs)
        done  = bool(result.get("done", False))
        info  = result.get("info", {})
        reward = float(result.get("reward", 0.0))

        print(
            f"           reward={reward:.4f} | "
            f"grader={info.get('grader_score', 0):.4f} | "
            f"success={info.get('success', False)}",
            flush=True,
        )

        if done:
            break

    grader = env_grader()
    score  = float(grader.get("grader_score", 0.0))
    print(f"\n  Result: score={score:.4f} | success={grader.get('success', False)} | steps={step}", flush=True)

    return {
        "task_id":      task_id,
        "grader_score": round(score, 4),
        "success":      grader.get("success", False),
        "steps_taken":  step,
    }


# ---------------------------------------------------------------------------
# Server readiness check
# ---------------------------------------------------------------------------

def wait_for_server(retries: int = 15, delay: float = 3.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{BASE_ENV_URL}/", timeout=5)
            if r.status_code == 200:
                print(f"  Server ready at {BASE_ENV_URL}", flush=True)
                return True
        except requests.ConnectionError:
            pass
        print(f"  Waiting for server... ({i + 1}/{retries})", flush=True)
        time.sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Main — ALL setup happens here, nothing at module level
# ---------------------------------------------------------------------------

def main():
    print("\nLUMOS Assistive AI — OpenEnv Inference", flush=True)
    print(f"  Model    : {MODEL_NAME}", flush=True)
    print(f"  API base : {API_BASE_URL}", flush=True)
    print(f"  Env URL  : {BASE_ENV_URL}", flush=True)

    # Validate API key BEFORE creating client
    if not API_KEY:
        print("\n[ERROR] HF_TOKEN is not set.", flush=True)
        print("  Windows: $env:HF_TOKEN='hf_...'", flush=True)
        print("  Linux:   export HF_TOKEN=hf_...", flush=True)
        sys.exit(1)

    # Create OpenAI client INSIDE main(), after validation
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"\n[ERROR] Failed to create OpenAI client: {e}", flush=True)
        sys.exit(1)

    # Wait for the environment server to be ready
    if not wait_for_server():
        print(f"\n[ERROR] Cannot reach server at {BASE_ENV_URL}", flush=True)
        sys.exit(1)

    # Run all 3 tasks
    results = [run_episode(tid, client) for tid in ["blind_mode", "deaf_mode", "mute_mode"]]
    avg = round(sum(r["grader_score"] for r in results) / len(results), 4)

    print(f"\n{'='*50}", flush=True)
    print("  FINAL SCORES", flush=True)
    print(f"{'='*50}", flush=True)
    for r in results:
        icon = "✓" if r["success"] else "✗"
        print(
            f"  {icon} {r['task_id']:15s} | "
            f"score={r['grader_score']:.4f} | steps={r['steps_taken']}",
            flush=True,
        )
    print(f"\n  Average : {avg:.4f}", flush=True)
    print(f"  Model   : {MODEL_NAME}", flush=True)
    print(f"  Temp    : {TEMPERATURE} (deterministic)", flush=True)
    print(f"  Reproducible: Yes (fixed scenario index + temperature=0)\n", flush=True)

    output = {
        "model":        MODEL_NAME,
        "temperature":  TEMPERATURE,
        "reproducible": True,
        "scores": {
            r["task_id"]: {
                "grader_score": r["grader_score"],
                "success":      r["success"],
            }
            for r in results
        },
        "average_score": avg,
    }
    print(json.dumps(output, indent=2), flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Saved to baseline_scores.json", flush=True)


if __name__ == "__main__":
    main()
