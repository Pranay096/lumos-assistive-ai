---
title: LUMOS Assistive AI OpenEnv
emoji: 👓
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# LUMOS Assistive AI — OpenEnv Environment

> *"Train AI agents to see for the blind, hear for the deaf, and speak for the mute."*

LUMOS is a real-world OpenEnv environment built around an assistive AI glasses project designed for people with visual, hearing, and speech impairments. It provides three tasks of increasing difficulty, each modelling a real hardware feature of a wearable device.

---

## Motivation

**There is no existing RL/agent benchmark for assistive technology.** LUMOS fills that gap. Every point of improvement in agent performance maps directly to better quality of life for people with disabilities:

| Population | Scale |
|---|---|
| People with visual impairments | **2.2 billion** worldwide |
| People with hearing impairments | **1.5 billion** worldwide |
| People relying on AAC devices | **Millions** globally |

---

## What Makes LUMOS Challenging

Unlike toy environments, LUMOS is designed to genuinely stress-test agent capabilities:

- **Domain-specific vocabulary** — blind mode uses terms like *pharmacopoeia*, *autoclave*, *effluent*; deaf mode uses *haematology*, *acetylcysteine*, *anaphylactoid*
- **Strict graders** — wrong decision type incurs a penalty; all key objects/words must be present for full credit; deaf mode requires *all* key words for success
- **Hintless mute mode** — unlike blind/deaf modes, mute mode provides zero hints; the agent must infer the target word purely from the ASL letter sequence
- **Exact-match word recognition** — mute mode requires the agent to output the *exact* target word (not a substring or partial match)
- **Heavier wrong-letter penalties** — mute mode penalises hallucinated letters more harshly than earlier versions

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `blind_mode` | Easy | Interpret camera feed → produce audio output (scene description, OCR reading, danger alert) |
| `deaf_mode` | Medium | Relay spoken speech → display as text on OLED screen, preserving all key information including rare vocabulary |
| `mute_mode` | Hard | Recognise ASL finger-spelling frame-by-frame → accumulate uncommon word → produce exact speech output. **No hints provided.** |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `camera_feed` | string | Scene description or current ASL frame |
| `microphone` | string | Raw speech input (deaf mode) |
| `voice_command` | string | Trigger word or command |
| `oled_display` | string \| null | Text on the transparent OLED display |
| `audio_output` | string \| null | TTS string sent to speaker |
| `asl_letter` | string \| null | Currently detected ASL letter |
| `spelled_word` | string \| null | Letters accumulated so far (mute mode only) |
| `step_number` | int | Current step index |
| `hint` | string | Partial progress signal (blind/deaf only; **empty for mute mode**) |

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `decision` | string | One of: `describe_scene`, `ocr_read`, `alert_danger`, `speech_to_text`, `sign_speech` |
| `output_text` | string | The agent's generated output |
| `confidence` | float [0,1] | Agent's self-reported confidence |

---

## Reward Design

Rewards are **shaped** (not sparse) — partial credit at every step:

| Component | Task | Description |
|---|---|---|
| `confidence_bonus` | all | Small reward proportional to confidence (max 0.03) |
| `correct_decision` | all | +0.35 / +0.25 / +0.15 for correct decision; **penalty for wrong decision** |
| `key_objects_mentioned` | blind | Partial credit for mentioning relevant scene objects |
| `danger_flagged` | blind | Reward for correctly identifying *all* hazards |
| `key_words_relayed` | deaf | Partial credit for preserving rare speech content |
| `letter_accuracy` | mute | Per-letter ASL matching credit |
| `word_recognised` | mute | Full reward (+0.37) only on **exact** word match |
| `wrong_letter_penalty` | mute | Penalty for hallucinated letters (up to -0.20) |

All rewards clipped to [0.0, 1.0].

**Key strictness notes:**
- Blind mode: wrong `decision` type gives **-0.10** penalty
- Deaf mode: wrong decision gives **-0.15**; success requires **all** key words present
- Mute mode: wrong decision gives **-0.15**; only *exact* full-word match triggers success

---

## API Endpoints

```
POST /reset?task_id=blind_mode    # Start new episode
POST /step                        # Take one step
GET  /state                       # Current internal state
GET  /tasks                       # Task list + action schemas
POST /grader                      # Grader score for current episode
GET  /baseline                    # Run LLM agent, return scores
```

---

## Setup & Usage

### Local

```bash
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t lumos-openenv .
docker run -e HF_TOKEN=hf_... -p 7860:7860 lumos-openenv
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
export BASE_ENV_URL="http://localhost:7860"
python inference.py
```

### Quick API Test

```python
import requests

BASE = "http://localhost:7860"

# Reset to deaf mode
obs = requests.post(f"{BASE}/reset?task_id=deaf_mode").json()
print(obs)

# Take a step — must relay rare vocabulary verbatim
action = {
    "decision": "speech_to_text",
    "output_text": "Defibrillator malfunction in cardiac catheterisation suite — technician en route",
    "confidence": 0.9
}
result = requests.post(f"{BASE}/step", json=action).json()
print(result)
```

---

## Baseline Scores

Scores produced by `inference.py` using `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router.  
Reproducible: `temperature=0`, fixed scenario index per task, no randomness.

| Task | Model | Score | Success |
|------|-------|-------|---------|
| `blind_mode` | Llama-3.3-70B-Instruct (T=0) | 1.0000 | ✓ |
| `deaf_mode` | Llama-3.3-70B-Instruct (T=0) | 1.0000 | ✓ |
| `mute_mode` | Llama-3.3-70B-Instruct (T=0) | 1.0000 | ✓ |
| **Average** | | **1.0000** | |

*(Run `python inference.py` twice — scores are identical both times)*

---

## Real Hardware

This environment simulates the **LUMOS AI Glasses** — a physical wearable built with:

| Component | Purpose |
|---|---|
| Raspberry Pi Zero 2W + Arducam IMX219 8MP | Camera capture |
| Waveshare transparent OLED display | Text overlay for deaf users |
| Gemini Vision API | Scene understanding |
| OpenAI Whisper | Speech recognition |
| Piper TTS (offline) | Text-to-speech for blind users |
| EfficientNet-B3 (99.89% val accuracy, 223,074 images, 29 classes) | ASL recognition |
| MediaPipe | Hand/finger tracking |

---

## Project Structure

```
lumos-assistive-ai/
├── app.py                  # Root-level FastAPI app (direct uvicorn entry)
├── server/
│   └── app.py              # Server module (openenv validate entry point)
├── inference.py            # Baseline inference script
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── requirements_base.txt   # Pinned deps for Docker build
└── README.md               # This file
```

> **Note on two `app.py` files:** `server/app.py` is required by `openenv validate` (referenced in `pyproject.toml` as `server.app:main`). The root `app.py` is the direct entry point used by Dockerfile/uvicorn. Both files contain identical logic; only the `if __name__ == "__main__"` block differs.

> **Note on two requirements files:** `requirements_base.txt` contains pinned versions used by the Dockerfile for reproducible container builds. `requirements.txt` is the general development requirements file.
