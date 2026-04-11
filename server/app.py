"""
LUMOS Assistive AI — FastAPI Application (thin routing layer)
=============================================================
All business logic lives in env.py and models.py.
This module ONLY wires HTTP endpoints to the LumosEnv API.
"""
from __future__ import annotations

import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from server.lumos_environment import LumosEnv, VALID_ACTIONS
try:
    from models import Action, Observation  # root-level models.py
except ImportError:
    from server.models import Action, Observation

app = FastAPI(
    title="LUMOS Assistive AI — OpenEnv",
    description=(
        "A POMDP-based RL environment simulating an AI-powered wearable glasses system "
        "for blind, deaf, and mute users. "
        "The agent must navigate partial observability, real-time noise, "
        "dynamic world state, and safety-critical decisions across three tasks."
    ),
    version="2.0.0",
)

env = LumosEnv()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "lumos-assistive-ai",
        "version": "2.0.0",
        "description": (
            "POMDP RL environment: AI agent operating wearable smart glasses "
            "for people with visual, hearing, and speech impairments."
        ),
        "tasks": ["blind_mode", "deaf_mode", "mute_mode"],
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
        "key_features": [
            "Partial observability (POMDP)",
            "Stochastic noise (scene clarity, audio corruption, ASL confusion)",
            "Free-choice actions — no required sequence",
            "State evolves based on agent actions",
            "Latency penalty per step",
            "Dynamic interrupt events",
            "Multiple valid success trajectories",
            "Genuine failure scenarios (dark/obstructed scenes)",
        ],
    }


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset_endpoint(
    task_id: str = Query(
        default="blind_mode",
        description="Task to run: blind_mode | deaf_mode | mute_mode",
    )
):
    """
    Start a new episode. Episode state is fully reset.
    Returns initial observation (partial view of the world).
    """
    global env
    try:
        env = LumosEnv()
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

@app.post("/step")
def step_endpoint(action: Action):
    """
    Take one action. Returns updated observation, reward, done flag, and info.

    The world state changes based on the action taken.
    Choose any valid action — no required sequence.
    See /tasks for valid action types per task.
    """
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": round(reward, 4),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

@app.get("/state")
def state_endpoint(debug: bool = False):
    """
    Return current hidden world state (for debugging / visualisation).
    Includes ground truth not visible to the agent if debug=True is passed.
    """
    return env.get_state(debug=debug)


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
def tasks_endpoint():
    """
    List all tasks, their valid actions, and difficulty metadata.
    """
    task_info = {
        "blind_mode": {
            "description": (
                "Agent controls wearable glasses for a blind user. "
                "Scene starts partially visible (low clarity). "
                "Agent must scan, focus on objects, and respond appropriately: "
                "alert dangers, describe safe paths, or read text — in any order."
            ),
            "difficulty": "easy",
            "valid_actions": VALID_ACTIONS["blind_mode"],
            "state_evolves_on": {
                "scan": "Global clarity +0.08–0.18; all objects improve slightly",
                "focus_object": "Per-object clarity +0.28–0.48; reveals hazard/text details",
                "alert_danger": "Removes alerted hazard from active hazard list",
            },
            "multiple_valid_paths": [
                "scan → focus_object → alert_danger (recommended)",
                "focus_object → alert_danger (risky if clarity too low for first focus)",
                "alert_danger immediately (very risky: likely miss without clarity)",
                "scan → describe_path (valid if no hazards present)",
            ],
            "failure_scenario": "Dark parking lot scene — correct action is report_failure",
        },
        "deaf_mode": {
            "description": (
                "Agent relays speech to a deaf user's OLED display. "
                "Audio arrives in chunks per listen() call, some words [CORRUPTED]. "
                "Agent must decide when to listen, when to clarify, and when to relay."
            ),
            "difficulty": "medium",
            "valid_actions": VALID_ACTIONS["deaf_mode"],
            "state_evolves_on": {
                "listen": "Advances audio stream to next chunk",
                "ask_speaker_repeat": "Clears corruption for next chunk",
            },
            "multiple_valid_paths": [
                "listen × N → relay_text (accumulate all, relay once)",
                "listen → ask_speaker_repeat → listen → relay_text (targeted denoising)",
                "listen → relay_partial → listen → relay_text (incremental relay)",
            ],
            "failure_scenario": "Heavy construction noise — partial relay expected",
        },
        "mute_mode": {
            "description": (
                "Agent translates ASL finger-spelling for a mute user. "
                "Letters arrive one-by-one with noise-based confusion (ambiguous: E/S). "
                "Agent must observe, optionally repeat-observe to improve confidence, "
                "confirm letters, and finally speak the word."
            ),
            "difficulty": "hard",
            "valid_actions": VALID_ACTIONS["mute_mode"],
            "state_evolves_on": {
                "observe_letter": "Advances frame index; returns noisy observation",
                "request_repeat": "Reduces noise on current frame; goes back one frame",
                "confirm_letter": "Irreversibly commits to letter interpretation",
            },
            "multiple_valid_paths": [
                "observe × N → speak_word (fast, risky under high noise)",
                "observe → confirm_letter (per letter, careful and deliberate)",
                "observe → [if ambiguous] request_repeat → observe → confirm_letter",
            ],
            "failure_scenario": "Extreme confusion scenario — high partial reward expected",
        },
    }
    return [
        {"id": tid, **meta}
        for tid, meta in task_info.items()
    ]


# ---------------------------------------------------------------------------
# /grader
# ---------------------------------------------------------------------------

@app.post("/grader")
def grader_endpoint():
    """
    Return final grader score for the current episode.
    Score is based on success + efficiency (steps used vs max).
    Non-success episodes receive partial scores based on trajectory rewards.
    """
    return {
        "grader_score": env.grader_score(),
        "success": env.get_state().get("success", False),
        "steps_taken": env.get_state().get("step", 0),
        "episode_id": env.get_state().get("episode_id"),
    }


# ---------------------------------------------------------------------------
# /baseline
# ---------------------------------------------------------------------------

@app.get("/baseline")
def baseline_endpoint():
    """
    Run a baseline LLM agent across all three tasks.
    Scores are representative of frontier model performance on this environment.
    Expected: ~0.55–0.75 (environment is genuinely challenging).
    """
    import json
    import textwrap

    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")

    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY not set.")

    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI client error: {e}")

    SYSTEM = textwrap.dedent("""
        You are an AI agent embedded inside LUMOS Assistive Glasses for people with disabilities.
        At each step, you receive an observation and must choose one action.
        You have FREE CHOICE — there is NO required action sequence.
        Reason about what the current observation tells you, then pick the most useful action.

        BLIND NAVIGATE: scan → improves clarity | focus_object → reveals detail | alert_danger → removes hazard
        DEAF RELAY: listen → gets audio chunk | ask_speaker_repeat → denoises | relay_text → sends to display
        ASL TRANSLATE: observe_letter → reveals letter | request_repeat → improves signal | speak_word → output

        Key rules:
        - Do NOT alert a danger you haven't focused on (false alarm penalty)
        - relay_text only after listening to enough chunks
        - speak_word only after observing all frames
        - If interrupt_urgency='critical', set priority='handle_interrupt'

        Respond with ONLY valid JSON:
        {"action_type": "...", "target": null, "content": null, "confidence": 0.8, "priority": null}
    """).strip()

    results = {}
    TASKS = ["blind_mode", "deaf_mode", "mute_mode"]

    for task_id in TASKS:
        test_env = LumosEnv()
        obs = test_env.reset(task_id=task_id)
        final_info: dict = {}

        for _ in range(10):
            user_msg = json.dumps(obs.model_dump(), indent=2)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=150,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content or "{}"
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                parsed = json.loads(clean)
                action = Action(
                    action_type=str(parsed.get("action_type", "scan")),
                    target=parsed.get("target"),
                    content=parsed.get("content"),
                    confidence=float(parsed.get("confidence", 0.8)),
                    priority=parsed.get("priority"),
                )
            except Exception:
                action = Action(action_type="scan")

            obs, reward, done, final_info = test_env.step(action)
            if done:
                break

        results[task_id] = {
            "grader_score": round(final_info.get("grader_score", 0.01), 4),
            "success": final_info.get("success", False),
        }

    avg = round(sum(r["grader_score"] for r in results.values()) / len(results), 4)
    return {
        "model": model,
        "temperature": 0.0,
        "reproducible": False,
        "note": (
            "Scores vary slightly due to stochastic noise in environment. "
            "Run inference.py with fixed random seed for reproducibility."
        ),
        "scores": results,
        "average_score": avg,
    }


# ---------------------------------------------------------------------------
# Entry point (openenv validate + direct run)
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
