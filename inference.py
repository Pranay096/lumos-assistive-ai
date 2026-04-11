"""
inference.py — LUMOS Assistive AI OpenEnv Baseline
====================================================
MANDATORY stdout format (strictly):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  — LLM endpoint  (default: HF router)
  MODEL_NAME    — model id      (default: meta-llama/llama-4-scout-17b-16e-instruct)
  API_KEY       — API key
  BASE_ENV_URL  — env server    (default: http://localhost:7860)

Agent strategy:
  - Task: blind_navigate → scan first, then focus on objects, alert dangers seen
  - Task: deaf_relay     → listen each chunk, ask_speaker_repeat if noisy, relay_text
  - Task: asl_translate  → observe_letter each frame, request_repeat if ambiguous,
                            confirm each letter, speak_word when all frames shown

The agent uses FREE CHOICE — it reasons about each observation independently.
No hardcoded action sequences.
"""

import ast
import json
import os
import sys
import textwrap
import time
from typing import List, Optional

import re
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or ""
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "lumos-assistive-ai"
MAX_STEPS    = 10
TASKS        = ["blind_navigate", "deaf_relay", "asl_translate"]

# Credits-exhausted: switch to rule-based fallback for rest of episode
_llm_exhausted = False

# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are the AI agent inside LUMOS Assistive Glasses for people with
    visual, hearing, or speech impairments. You receive a JSON observation
    and must choose ONE action. Respond ONLY with valid JSON.

    ╔══════════════════════════════════════════════════════════╗
    ║  FREE CHOICE — no fixed sequence. Reason from the observation.  ║
    ╚══════════════════════════════════════════════════════════╝

    ─── TASK: blind_navigate ─────────────────────────────────────
    Valid: scan | focus_object | alert_danger | describe_path | read_text | report_failure
    Strategy:
      1. scene_clarity < 0.45 → scan (repeatable, raises clarity)
      2. Objects visible → focus_object(target=<name>) for detail
      3. If focused object shows DANGER → alert_danger(content=<description>)
      4. All hazards alerted and path clear → describe_path(content=<guidance>)
      5. Text object in focus → read_text(content=<text you see>)
      6. Scene truly dark/unnavigable → report_failure
    ⚠ NEVER call alert_danger without focusing first (false alarm = −0.15)

    ─── TASK: deaf_relay ─────────────────────────────────────────
    Valid: listen | relay_text | ask_speaker_repeat
    Strategy:
      1. If the current audio chunk contains "[CORRUPTED]", you MUST call ask_speaker_repeat to clean it.
      2. If the current audio chunk is clean and does NOT say "[END OF AUDIO STREAM]", you MUST call listen to get the next chunk.
      3. ONLY when the audio explicitly says "[END OF AUDIO STREAM]" or "All chunks delivered", you MUST call relay_text. Do NOT call it early.
      4. relay_text content = paste ALL words you heard verbatim across all chunks combined! Note the [AUDIO HEARD SO FAR] helper in your prompt.

    ─── TASK: asl_translate ─────────────────────────────────────────
    Valid: observe_letter | confirm_letter | request_repeat | speak_word | speak_partial

    ★ CRITICAL MECHANIC (read carefully):
      - The observation shows the CURRENT unconfirmed letter in asl_observation.
      - confirm_letter(target=X) locks in letter X as the NEXT letter of the word.
        It does NOT advance the frame. Call confirm_letter for the letter you just SAW.
      - observe_letter advances the frame to the next letter.
      - Correct flow per letter:
          (1) Read asl_observation → see a letter (e.g. "S")
          (2) If confidence ≥ 0.55 → confirm_letter(target="S")
          (3) Then observe_letter to advance to next frame
          (4) Repeat until [ALL FRAMES SHOWN] → speak_word(content=<full word>)
      - ⚠ DO NOT AUTO-CORRECT THE FINAL WORD. The ASL words are often random phonetic syllables (e.g. 'DJOE'). They are NOT real dictionary words. Speak EXACTLY the letters you confirmed.
      - request_repeat: use ONLY when asl_observation is "ambiguous" or "unclear".
        It re-shows the same frame with less noise. Then observe_letter again.
      - NEVER confirm a letter you did not see in the current observation.

    JSON format (no markdown, no extra text):
    {{"action_type": "...", "target": null, "content": null, "confidence": 0.85, "priority": null}}

    Examples:
    {{"action_type": "scan", "target": null, "content": null, "confidence": 0.9, "priority": null}}
    {{"action_type": "focus_object", "target": "wet_floor_sign", "content": null, "confidence": 0.85, "priority": null}}
    {{"action_type": "alert_danger", "target": null, "content": "wet floor sign — slip hazard", "confidence": 0.9, "priority": null}}
    {{"action_type": "listen", "target": null, "content": null, "confidence": 0.9, "priority": null}}
    {{"action_type": "relay_text", "target": null, "content": "ibuprofen 400mg every 8 hours. fever above 38.5 seek help. follow-up 24th.", "confidence": 0.9, "priority": null}}
    {{"action_type": "confirm_letter", "target": "S", "content": null, "confidence": 0.85, "priority": null}}
    {{"action_type": "observe_letter", "target": null, "content": null, "confidence": 0.9, "priority": null}}
    {{"action_type": "speak_word", "target": null, "content": "WATER", "confidence": 0.9, "priority": null}}
""").strip()


# ---------------------------------------------------------------------------
# Logging (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def wait_for_server(retries: int = 20, delay: float = 3.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{BASE_ENV_URL}/", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        print(f"[DEBUG] Waiting for env server... attempt {i + 1}/{retries}", flush=True)
        time.sleep(delay)
    return False


def env_reset(task_id: str) -> dict:
    r = requests.post(
        f"{BASE_ENV_URL}/reset",
        params={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{BASE_ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Smart rule-based fallback (used when LLM credits are exhausted)
# ---------------------------------------------------------------------------

def _get_last_obs(history: List[dict]) -> dict:
    """Extract the most recent observation dict from conversation history."""
    for msg in reversed(history):
        if msg["role"] == "user":
            try:
                raw = msg["content"].split("observation:\n", 1)[1]
                return json.loads(raw)
            except Exception:
                pass
    return {}


def smart_fallback(task_id: str, history: List[dict], step: int) -> dict:
    """
    Rule-based agent used when LLM credits are exhausted.
    Reads past observations from history to make meaningful decisions.
    Much better than blindly repeating the same action.
    """
    obs = _get_last_obs(history)

    if task_id == "blind_navigate":
        clarity = obs.get("scene_clarity", 0.0) or 0.0
        visible = obs.get("visible_objects") or []
        feedback = obs.get("last_action_feedback", "")

        # Count focus actions already done
        focus_done = sum(
            1 for m in history
            if m["role"] == "assistant" and "focus_object" in m["content"]
        )
        alerts_done = sum(
            1 for m in history
            if m["role"] == "assistant" and "alert_danger" in m["content"]
        )

        if clarity < 0.45 and step <= 3:
            return {"action_type": "scan", "target": None, "content": None, "confidence": 0.8}
        if visible and focus_done < len(visible):
            target_obj = visible[min(focus_done, len(visible) - 1)]
            return {"action_type": "focus_object", "target": target_obj, "content": None, "confidence": 0.75}
        if "DANGER" in feedback or "hazard" in feedback.lower():
            obj = obs.get("camera_feed", "")
            return {"action_type": "alert_danger", "target": None, "content": f"hazard detected: {obj[:60]}", "confidence": 0.7}
        return {"action_type": "describe_path", "target": None, "content": " ".join(visible) + " path clear ahead", "confidence": 0.65}

    elif task_id == "deaf_relay":
        audio = obs.get("audio_stream", "") or ""
        listens_done = sum(
            1 for m in history
            if m["role"] == "assistant" and "listen" in m["content"]
        )

        if "[END OF AUDIO" in audio or listens_done >= 4:
            # Collect all clean audio seen across history
            heard: List[str] = []
            for m in history:
                if m["role"] == "user":
                    try:
                        o = json.loads(m["content"].split("observation:\n", 1)[1])
                        a = (o.get("audio_stream") or "").replace("[CORRUPTED]", "").strip()
                        a = re.sub(r'\[.*?\]', '', a).strip()
                        if a and "[END" not in a:
                            heard.append(a)
                    except Exception:
                        pass
            relay = " ".join(heard).strip() or "content received from audio stream"
            return {"action_type": "relay_text", "target": None, "content": relay, "confidence": 0.7}

        if "[CORRUPTED]" in audio and listens_done <= 2:
            return {"action_type": "ask_speaker_repeat", "target": None, "content": None, "confidence": 0.7}
        return {"action_type": "listen", "target": None, "content": None, "confidence": 0.8}

    else:  # asl_translate
        asl_obs = obs.get("asl_observation", "") or ""
        conf = obs.get("asl_confidence") or 0.0
        confirmed = obs.get("confirmed_so_far") or ""

        if asl_obs == "[ALL FRAMES SHOWN]":
            word = confirmed if confirmed else "WATER"
            return {"action_type": "speak_word", "target": None, "content": word, "confidence": 0.65}

        # Figure out what the last assistant action was
        last_action = ""
        for m in reversed(history):
            if m["role"] == "assistant":
                try:
                    last_action = json.loads(m["content"]).get("action_type", "")
                except Exception:
                    pass
                break

        # After confirm_letter or request_repeat, we must observe_letter to advance frame
        if last_action in ("confirm_letter", "request_repeat", ""):
            return {"action_type": "observe_letter", "target": None, "content": None, "confidence": 0.8}

        # Now we just observed — decide whether to confirm, repeat, or observe again
        if "ambiguous" in asl_obs or asl_obs == "unclear" or conf < 0.4:
            return {"action_type": "request_repeat", "target": None, "content": None, "confidence": 0.7}
        if len(asl_obs) == 1 and conf >= 0.55:
            return {"action_type": "confirm_letter", "target": asl_obs, "content": None, "confidence": conf}
        return {"action_type": "observe_letter", "target": None, "content": None, "confidence": 0.8}


# Max conversation turns to send to LLM per call (saves tokens)
# Full history is still kept locally for smart_fallback to read
MAX_HISTORY_TURNS = 8  # increased from 6 — deaf_relay needs more audio context


def _build_llm_messages(history: List[dict], task_id: str) -> List[dict]:
    """
    Build the message list to send to the LLM.
    - Truncates to last MAX_HISTORY_TURNS messages to save tokens.
    - For deaf_relay: prepends a running audio summary so the LLM
      remembers all chunks heard even after truncation.
    """
    # For deaf_relay: extract all audio heard so far
    prefix = ""
    if task_id == "deaf_relay":
        heard = []
        for m in history:
            if m["role"] == "user":
                try:
                    o = json.loads(m["content"].split("observation:\n", 1)[1])
                    a = (o.get("audio_stream") or "").strip()
                    a = re.sub(r'\[.*?\]', '', a).strip()
                    a = a.replace("[END OF AUDIO STREAM]", "").strip()
                    if a:
                        heard.append(a)
                except Exception:
                    pass
        if heard:
            prefix = (
                f"[AUDIO HEARD SO FAR across all listen steps]: "
                f"{' '.join(heard)}\n"
                f"Use THESE EXACT WORDS when you relay_text.\n\n"
            )

    # Truncate to last N messages
    trimmed = history[-MAX_HISTORY_TURNS:] if len(history) > MAX_HISTORY_TURNS else history

    # Inject audio summary into first user message if deaf_relay
    if prefix and trimmed:
        first = dict(trimmed[0])
        first["content"] = prefix + first["content"]
        trimmed = [first] + list(trimmed[1:])

    return trimmed


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(history: List[dict], client: OpenAI, task_id: str, step: int) -> dict:
    """
    Call the LLM with truncated conversation history.
    Falls back to smart_fallback on 402 credit-exhausted errors.
    """
    global _llm_exhausted
    if _llm_exhausted:
        return smart_fallback(task_id, history, step)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _build_llm_messages(history, task_id)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("Empty response from LLM")
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end != -1:
            clean = clean[start:end+1]
        
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            py_clean = clean.replace("null", "None").replace("true", "True").replace("false", "False")
            m = re.search(r'\{[^{}]*action_type[^{}]*\}', py_clean)
            if m:
                py_clean = m.group(0)
            parsed = ast.literal_eval(py_clean)
            
        if not isinstance(parsed, dict) or "action_type" not in parsed:
            raise ValueError("Missing action_type")
        _llm_exhausted = False
        return parsed
    except Exception as exc:
        exc_str = str(exc)
        if "402" in exc_str or "depleted" in exc_str.lower() or "credits" in exc_str.lower():
            print("[DEBUG] LLM credits exhausted — switching to rule-based fallback.", flush=True)
            _llm_exhausted = True
            return smart_fallback(task_id, history, step)
        if "429" in exc_str or "rate" in exc_str.lower():
            if "on tokens per day" in exc_str or "TPD" in exc_str:
                print("[DEBUG] Daily token limit reached — switching to rule-based fallback.", flush=True)
                _llm_exhausted = True
                return smart_fallback(task_id, history, step)
            
            print("[DEBUG] Rate limited — waiting 5s and retrying once.", flush=True)
            time.sleep(5)
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages,
                    temperature=0.0, max_tokens=300,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    parsed = json.loads(raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip())
                    if "action_type" in parsed:
                        return parsed
            except Exception:
                pass
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return smart_fallback(task_id, history, step)



# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client: OpenAI) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False
    error_msg: Optional[str] = None

    # Reset credit-exhausted flag at the start of each episode
    global _llm_exhausted
    _llm_exhausted = False

    # Conversation history — lets LLM remember all past audio/ASL observations
    history: List[dict] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build user message from current observation
            obs_str = json.dumps(obs, indent=2)
            user_msg = f"Step {step} observation:\n{obs_str}"
            history.append({"role": "user", "content": user_msg})

            parsed = call_llm(history, client, task_id, step)
            action_type = str(parsed.get("action_type", "scan"))
            target      = parsed.get("target")
            content     = parsed.get("content")
            confidence  = float(max(0.0, min(1.0, parsed.get("confidence", 0.8))))
            priority    = parsed.get("priority")

            # Add assistant response to history
            history.append({"role": "assistant", "content": json.dumps(parsed)})

            # Build concise action string for log
            action_log = action_type
            if target:
                action_log += f"({target})"
            if content:
                action_log += f"='{content[:25]}'"

            result = env_step({
                "action_type": action_type,
                "target": target,
                "content": content,
                "confidence": confidence,
                "priority": priority,
            })

            obs      = result.get("observation", obs)
            reward   = float(result.get("reward", 0.0))
            done     = bool(result.get("done", False))
            info     = result.get("info", {})
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=action_log, reward=reward, done=done, error=error_msg)

            if done:
                success = bool(info.get("success", False))
                score   = float(info.get("grader_score", 0.01))
                break

        if not done and rewards:
            score = max(0.01, min(0.73, sum(rewards) / len(rewards)))

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = 0.01
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task_id": task_id,
        "grader_score": round(score, 4),
        "success": success,
        "steps_taken": steps_taken,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("[ERROR] API_KEY is not set.", flush=True)
        sys.exit(1)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI client: {e}", flush=True)
        sys.exit(1)

    if not wait_for_server():
        print(f"[ERROR] Cannot reach env server at {BASE_ENV_URL}", flush=True)
        sys.exit(1)

    results = [run_episode(tid, client) for tid in TASKS]
    avg = round(sum(r["grader_score"] for r in results) / len(results), 4)

    summary = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "scores": {
            r["task_id"]: {
                "grader_score": r["grader_score"],
                "success": r["success"],
            }
            for r in results
        },
        "average_score": avg,
    }

    print(json.dumps(summary, indent=2), flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
