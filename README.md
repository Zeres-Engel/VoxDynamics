# VoxDynamics

## Real-Time Speech Emotion Recognition System

A real-time speech emotion recognition (SER) system that captures live microphone input and identifies the speaker's emotional state with minimal latency. **Language-agnostic** — works with any language without speech-to-text conversion.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Real-Time Streaming** | Live microphone capture with < 500ms latency |
| 🌍 **Multi-Language** | Acoustic-based — no STT required, works with any language |
| 📊 **8 Emotion Classes** | Happy, Angry, Sad, Neutral, Fear, Surprise, Disgust, Calm |
| 📈 **Dimensional Tracking** | Arousal, Dominance, Valence (continuous 0-1 scale) |
| 🔄 **EMA Smoothing** | Exponential Moving Average for stable, non-jittery output |
| 🗣️ **Voice Activity Detection** | Silero VAD filters noise, only processes speech |
| 💾 **Session Logging** | PostgreSQL stores full emotion history per session |
| 🐳 **One-Click Deploy** | `docker-compose up --build` — done |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VoxDynamics                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Gradio  │───▶│  Audio       │───▶│  Silero VAD           │  │
│  │  UI      │    │  Processor   │    │  (Voice Detection)    │  │
│  │  :7860   │    │  Ring Buffer │    └──────────┬────────────┘  │
│  └──────────┘    │  + EMA       │               │               │
│                  └──────┬───────┘     ┌─────────▼────────────┐  │
│                         │             │  Wav2Vec2 Emotion     │  │
│  ┌──────────┐           │             │  Model (audeering)    │  │
│  │ FastAPI  │───────────┘             │  → A, D, V dims      │  │
│  │ WS :8000 │                         └──────────┬───────────┘  │
│  └──────────┘                                    │              │
│                                        ┌─────────▼───────────┐  │
│                                        │  Emotion Mapping     │  │
│                                        │  (Centroid-based)    │  │
│                                        └─────────┬───────────┘  │
│                                                  │              │
│                                        ┌─────────▼───────────┐  │
│                                        │  PostgreSQL :5432    │  │
│                                        │  (Async Logging)     │  │
│                                        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4 GB RAM (model ~1 GB + inference overhead)

### One-Click Deploy

```bash
# Clone the repository
git clone https://github.com/<your-username>/VoxDynamics.git
cd VoxDynamics

# Launch everything
docker-compose up --build
```

### Access

| Service | URL |
|---------|-----|
| **Gradio Dashboard** | [http://localhost:7860](http://localhost:7860) |
| **FastAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) |
| **WebSocket** | `ws://localhost:8000/stream` |

---

## 🎯 Usage

### 1. Gradio Dashboard (Recommended)
1. Open [http://localhost:7860](http://localhost:7860)
2. Click the microphone button
3. Start speaking — watch emotions update in real-time!

### 2. WebSocket API (for custom clients)

```python
import asyncio
import websockets
import numpy as np

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        # Generate or capture audio (16kHz, float32, mono)
        chunk = np.zeros(16000, dtype=np.float32)  # 1 second
        await ws.send(chunk.tobytes())
        result = await ws.recv()
        print(result)  # JSON with emotion_label, arousal, dominance, valence

asyncio.run(stream_audio())
```

### 3. REST API

```bash
# Get emotion history for a session
curl http://localhost:8000/api/emotions/{session_id}

# List all sessions
curl http://localhost:8000/api/sessions
```

---

## 🧠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI Model** | `wav2vec2-large-robust-12-ft-emotion-msp-dim` | Dimensional emotion prediction (A/D/V) |
| **VAD** | Silero VAD | Voice activity detection, noise filtering |
| **Backend** | FastAPI + WebSocket | Real-time streaming API |
| **Database** | PostgreSQL + async SQLAlchemy | Session emotion logging |
| **UI** | Gradio | Real-time dashboard |
| **Ops** | Docker Compose | One-click deployment |

---

## 📁 Project Structure

```
VoxDynamics/
├── app/
│   ├── core/
│   │   ├── vad.py              # Silero VAD integration
│   │   ├── emotion_model.py    # Wav2Vec2 emotion predictor
│   │   └── processor.py        # AudioProcessor pipeline
│   ├── api/
│   │   └── websocket.py        # WebSocket stream handler
│   ├── db/
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── database.py         # Async DB connection
│   ├── ui/
│   │   └── dashboard.py        # Gradio real-time dashboard
│   ├── config.py               # Pydantic settings
│   └── main.py                 # FastAPI entry point
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── REPORT.md                   # Comprehensive technical report
└── README.md
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
