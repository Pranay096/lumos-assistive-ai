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

LUMOS is a real-world OpenEnv environment built around an assistive AI glasses project
designed for people with visual, hearing, and speech impairments. It provides three
tasks of increasing difficulty, each corresponding to a real hardware feature.

---

## Motivation

There is **no existing RL/agent benchmark for assistive technology**. This environment
fills that gap. Every point of improvement on agent performance here maps directly to
better quality of life for people with disabilities:

- **2.2 billion** people with visual impairments worldwide
- **1.5 billion** people with hearing impairments worldwide
- **Millions** with speech/motor impairments who rely on AAC devices

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `blind_mode` | Easy | Interpret camera feed → produce audio output (scene description, OCR reading, danger alert) |
| `deaf_mode` | Medium | Relay spoken speech → display as text on OLED screen, preserving key information |
| `mute_mode` | Hard | Recognise ASL finger-spelling frame-by-frame → accumulate word → produce speech output |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task |
| `camera_feed` | string | Scene description or current ASL frame |
| `microphone` | string | Raw speech input |
| `voice_command` | string | Trigger word or command |
| `oled_display` | string \| null | Text on the transparent OLED display |
| `audio_output` | string \| null | TTS string sent to speaker |
| `asl_letter` | string \| null | Currently detected ASL letter |
| `spelled_word` | string \| null | Word accumulated so far |
| `step_number` | int | Current step |
| `hint` | string | Partial progress signal for agent |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `decision` | string | One of: `describe_scene`, `ocr_read`, `alert_danger`, `speech_to_text`, `sign_speech` |
| `output_text` | string | The agent's generated output |
| `confidence` | float [0,1] | Agent's confidence |

---

## Reward Design

Rewards are **shaped** (not sparse) — partial credit at every step:

- `confidence_bonus` — small reward proportional to confidence
- `correct_decision` — choosing the right action type
- `key_objects_mentioned` — mentioning relevant scene objects (blind mode)
- `danger_flagged` — correctly identifying hazards (blind mode)
- `key_words_relayed` — preserving key speech content (deaf mode)
- `letter_accuracy` — per-letter ASL matching (mute mode)
- `word_recognised` — full reward for correct final word (mute mode)
- `wrong_letter_penalty` — penalty for hallucinated letters (mute mode)

All rewards clipped to [0.0, 1.0].

---

## API Endpoints

```
POST /reset?task_id=blind_mode    # Start new episode
POST /step                        # Take one step
GET  /state                       # Current internal state
GET  /tasks                       # Task list + action schemas
POST /grader                      # Grader score for current episode
GET  /baseline                    # Run GPT-4o-mini agent, return scores
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
docker run -e OPENAI_API_KEY=sk-... -p 7860:7860 lumos-openenv
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
export BASE_ENV_URL="http://localhost:7860"   # or your HF Space URL
python inference.py
```

### Quick Test

```python
import requests

BASE = "http://localhost:7860"

# Reset to blind mode
obs = requests.post(f"{BASE}/reset?task_id=blind_mode").json()
print(obs)

# Take a step
action = {"decision": "alert_danger", "output_text": "Caution: stove and knife ahead", "confidence": 0.9}
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

This environment simulates the LUMOS AI Glasses — a physical wearable built with:
- Raspberry Pi Zero 2W + Arducam IMX219 8MP camera
- Waveshare transparent OLED display
- Gemini Vision API (scene understanding)
- OpenAI Whisper (speech recognition)
- Piper TTS (offline text-to-speech)
- Custom-trained EfficientNet-B3 ASL model (99.89% val accuracy, 223,074 images, 29 classes)
- MediaPipe (hand/finger tracking for Mute Mode)
