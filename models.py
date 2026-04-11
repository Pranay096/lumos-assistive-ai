"""
LUMOS Assistive AI — Pydantic Models (OpenEnv spec)
====================================================
Observation is a PARTIAL, NOISY view of the hidden world state.
Action is FREE CHOICE — no required sequence, any action valid any step.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — partial, noisy view of the world
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    What the agent perceives at each step.
    This is a PARTIAL, NOISY projection of the true hidden world state.
    The agent must use actions to improve its situational awareness
    before committing to an output action.
    """
    task_id: str = Field(description="Current task: blind_mode | deaf_mode | mute_mode")
    episode_id: str = Field(description="Unique episode identifier")
    step_number: int = Field(description="Current step (1-indexed)")
    steps_remaining: int = Field(
        description=(
            "Steps left before forced episode end. "
            "Each unused step is wasted time for the user — act efficiently."
        )
    )

    # ── Visual channel ──────────────────────────────────────────────────────
    camera_feed: str = Field(
        default="",
        description=(
            "Visual input. Quality degrades with low scene_clarity. "
            "At clarity < 0.25: only shapes visible. "
            "At clarity < 0.45: objects present but details unclear. "
            "After focus_object: reveals hazard/text details for that object."
        ),
    )
    scene_clarity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "[0,1] Global visual clarity. Starts low (0.15–0.45). "
            "Improved by scan (+0.08–0.18) and focus_object (+0.30–0.50 per object). "
            "Use scan first, then focus for details."
        ),
    )
    visible_objects: List[str] = Field(
        default_factory=list,
        description=(
            "Objects with per-object clarity > 0.45. "
            "Subset of all scene objects. Use focus_object to add more."
        ),
    )

    # ── Audio channel ───────────────────────────────────────────────────────
    audio_stream: str = Field(
        default="",
        description=(
            "Current audio chunk. May contain [CORRUPTED] tokens where noise "
            "has destroyed intelligibility. Use ask_speaker_repeat to clean a chunk. "
            "Use listen to advance to the next chunk."
        ),
    )
    noise_level: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "[0,1] Current noise intensity. Affects both audio corruption rate "
            "and ASL recognition confidence. High noise = more [CORRUPTED] tokens "
            "and more 'ambiguous' ASL observations."
        ),
    )

    # ── ASL channel ─────────────────────────────────────────────────────────
    asl_observation: Optional[str] = Field(
        default=None,
        description=(
            "Current ASL frame observation (mute mode only). "
            "Values: letter (e.g. 'A'), 'ambiguous: E/S' (confusion pair), "
            "'unclear' (too noisy), '[ALL FRAMES SHOWN]' (done). "
            "Use request_repeat if confidence is low."
        ),
    )
    asl_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "[0,1] Confidence in the current ASL observation. "
            "< 0.4 = consider request_repeat. "
            "> 0.7 = safe to confirm_letter."
        ),
    )
    confirmed_so_far: Optional[str] = Field(
        default=None,
        description=(
            "Letters the agent has locked in via confirm_letter. "
            "May include errors if agent confirmed incorrectly. "
            "Use speak_word when word is complete."
        ),
    )

    # ── Interrupt channel ────────────────────────────────────────────────────
    interrupt: Optional[str] = Field(
        default=None,
        description=(
            "A new dynamic event that has occurred mid-episode. "
            "May be None (no interrupt) or a description of the event. "
            "Set action.priority='handle_interrupt' to respond to it."
        ),
    )
    interrupt_urgency: Optional[str] = Field(
        default=None,
        description=(
            "'low' — can finish current task first. "
            "'high' — handle within 2 steps. "
            "'critical' — MUST handle immediately or face -0.20 penalty."
        ),
    )

    # ── Feedback from last action ────────────────────────────────────────────
    last_action_feedback: str = Field(
        default="",
        description=(
            "Human-readable result of the previous action. "
            "Includes what changed in world state, any penalties, and progress signals."
        ),
    )
    last_reward: float = Field(
        default=0.0,
        description="Reward received at the previous step.",
    )

    # ── User context ─────────────────────────────────────────────────────────
    user_context: str = Field(
        default="",
        description=(
            "What the user is trying to accomplish. "
            "Grounds the task semantically — use this to understand "
            "what information is most critical to relay or act on."
        ),
    )


# ---------------------------------------------------------------------------
# Action — free choice, no required sequence
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Agent action. ANY valid action may be taken at ANY step.
    There is no required sequence. The agent must reason about which
    action maximises reward given current observations.

    BLIND NAVIGATE actions:
      scan                  — broad view update, improves global clarity (+0.08-0.18)
      focus_object          — zoom in on specific object (target required), reveals hazard/text detail
      alert_danger          — warn user of identified hazard (content required)
      describe_path         — navigation guidance (content required), only credit clearly visible objects
      read_text             — OCR output (must have focused a text object first, content required)
      report_failure        — declare scene unnavigable (correct in dark/obstructed scenarios)

    DEAF RELAY actions:
      listen                — advance audio stream, receive next chunk
      relay_text            — send transcription to OLED display (content required)
      ask_speaker_repeat    — request cleaner re-delivery of current chunk
      relay_partial         — send partial understanding (content required), lower reward
      flag_emergency        — mark speech as urgent (target=urgency level)

    ASL TRANSLATE actions:
      observe_letter        — observe current frame (advances frame index)
      confirm_letter        — lock in letter interpretation (target=letter), irreversible
      request_repeat        — re-observe current frame with less noise (goes back one frame)
      speak_word            — output full word (content required), triggers grading
      speak_partial         — output word assembled so far (content required)
    """
    action_type: str = Field(
        description="Action type string. See docstring for full list per task."
    )
    target: Optional[str] = Field(
        default=None,
        description=(
            "For focus_object: exact object name from visible_objects or scene. "
            "For confirm_letter: the letter (A-Z). "
            "For flag_emergency: urgency level string."
        ),
    )
    content: Optional[str] = Field(
        default=None,
        description=(
            "For output actions (alert_danger, describe_path, read_text, "
            "relay_text, relay_partial, speak_word, speak_partial): "
            "the text the agent is generating for the user."
        ),
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description=(
            "Agent's self-reported confidence [0,1]. "
            "Used in reward weighting for output actions."
        ),
    )
    priority: Optional[str] = Field(
        default=None,
        description=(
            "When interrupt is present: "
            "'current_task' = continue current objective (fine for low urgency). "
            "'handle_interrupt' = respond to the interrupt event. "
            "Not setting this when urgency='critical' incurs -0.20 penalty."
        ),
    )


# ---------------------------------------------------------------------------
# Reward breakdown (returned in info dict)
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    total: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    feedback: str


# ---------------------------------------------------------------------------
# Step result (returned by /step)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict
