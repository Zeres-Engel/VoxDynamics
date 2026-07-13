# 🏗️ System Architecture

> **Full architectural deep-dive of the VoxDynamics Speech Emotion Recognition system.**
> This document covers the pipeline, data flow, component design, and rationale behind every engineering decision.

---

## 📡 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VoxDynamics System                             │
│                                                                          │
│  ┌──────────────────────┐      ┌─────────────────────────────────────┐   │
│  │   Browser (SPA)      │      │       FastAPI Backend (Python)       │   │
│  │  ┌────────────────┐  │      │  ┌───────────────────────────────┐  │   │
│  │  │ Audio Upload    │──┼─HTTP─┼─▶│  POST /api/analyze           │  │   │
│  │  │ WaveSurfer.js   │  │      │  │  GET  /api/sessions          │  │   │
│  │  │ Plotly.js Charts│  │      │  │  GET  /api/emotions/{uuid}   │  │   │
│  │  └────────────────┘  │      │  │  GET  /health                │  │   │
│  └──────────────────────┘      │  │  WS   /stream                │  │   │
│                                 │  └───────────┬───────────────────┘  │   │
│                                 │              │                       │   │
│  ┌──────────────────────┐      │  ┌────────────▼──────────────────┐  │   │
│  │   PostgreSQL 15      │◄─ASYNC─┼─│  SQLAlchemy + asyncpg ORM    │  │   │
│  │  ┌────────────────┐  │      │  │  ┌──────────┐ ┌───────────┐  │  │   │
│  │  │ sessions       │  │      │  │  │ Session  │ │EmotionLog │  │  │   │
│  │  │ emotion_logs   │  │      │  │  └──────────┘ └───────────┘  │  │   │
│  │  └────────────────┘  │      │  └──────────────────────────────┘  │   │
│  └──────────────────────┘      └─────────────────────────────────────┘   │
│                                                                          │
│                          AI INFERENCE PIPELINE                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  AudioProcessor (processor.py)                                      │  │
│  │                                                                     │  │
│  │  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │  │
│  │  │  VAD     │───▶│  Speech      │───▶│  CNN Emotion Predictor  │   │  │
│  │  │(Silero)  │    │  Islands     │    │  (Deep 1D-CNN, 97.25%)  │   │  │
│  │  └──────────┘    └──────────────┘    └─────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧬 AI Inference Pipeline (Detailed)

### Mode 1: File Analysis (`process_file`)

```
Audio File (any SR, any format)
        │
        ├── [Global Normalization] ── Max amplitude norm (preserves relative loudness)
        │
        ├── [VAD Path ─ 16kHz]
        │    librosa.resample(orig_sr, 16000)
        │    → Silero VAD scan (0.1s step)
        │    → Binary speech/non-speech array
        │    → Cluster into speech islands
        │      • Merge silence gaps < 0.8s (handles mid-utterance pauses)
        │      • Filter segments < 0.5s (removes noise artifacts)
        │      • Merge segments closer than 1.0s
        │    → Returns list of (start_time, end_time) tuples
        │
        └── [CNN Path ─ Original SR]
             For each speech island:
             → Extract segment + 200ms silence buffer (prevents phoneme clipping)
             → Resample to 22,050 Hz
             → fix_length(2.5s = 55,125 samples) — LEFT-ALIGNED padding
             → Feature extraction:
               • Zero Crossing Rate (ZCR): 108 frames
               • RMS Energy: 108 frames
               • MFCC (20 coefficients × 108 frames): 2,160 values
               → Total: 2,376 features
             → StandardScaler transform (fitted on 48k training samples)
             → Reshape to (2376, 1)
             → 1D-CNN forward pass
             → Softmax → 7 emotion probabilities
             → Map to emotion label + confidence
             → Heuristic Arousal/Dominance/Valence calculation
```

### Mode 2: Real-Time Streaming (`process_chunk`)

```
Audio Chunk (16kHz PCM float32)
        │
        ├── Append to Ring Buffer (deque, maxlen = 16kHz × 2.5s = 40,000 samples)
        │
        ├── Silero VAD on chunk → is_speech?
        │
        ├── If speech:
        │    → CNN inference on full ring buffer
        │    → EMA smoothing on scores: score = α·new + (1-α)·old (α=0.4)
        │    → Return structured result
        │
        └── If no speech:
             → Return last known result (marked as is_speech=false)
             → Or default neutral if no prior result
```

---

## 🏛️ Component Design

### 1. AudioProcessor (`app/core/processor.py`)

| Property | Value |
|----------|-------|
| **Class** | `AudioProcessor` |
| **Sample Rate** | 16 kHz (configurable) |
| **Buffer Duration** | 2.5 seconds (fixed to CNN input) |
| **EMA Alpha** | 0.4 (configurable) |
| **VAD Threshold** | 0.5 (configurable) |

**Key methods:**
- `load_models()` — Load VAD + CNN models (call once at startup)
- `process_chunk(chunk)` — Real-time single-chunk processing
- `process_file(waveform, sr)` — Full file analysis with intelligent segmentation
- `reset()` — Clear buffer + EMA state for new session
- `_apply_ema_scores(new_scores)` — Exponential Moving Average smoothing

### 2. CNNEmotionPredictor (`app/core/cnn_predictor.py`)

| Property | Value |
|----------|-------|
| **Class** | `CNNEmotionPredictor` |
| **Framework** | TensorFlow 2.10 / Keras |
| **Architecture** | Deep 1D-CNN (5 Conv blocks) |
| **Input Shape** | (2376, 1) |
| **Output Classes** | 7 (angry, disgust, fear, happy, neutral, sad, surprise) |
| **Benchmark** | 97.25% validation accuracy |
| **Weights** | `models/best_model1_weights.h5` |
| **Scaler** | `models/scaler2.pickle` |

**CNN Architecture:**
```
Input (2376, 1)
    │
    ├─ Conv1D(512, k=5) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(512, k=5) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(256, k=5) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(256, k=3) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(128, k=3) → BatchNorm → MaxPool1D(3) → Dropout(0.2)
    │
    ├─ Flatten
    ├─ Dense(512, relu) → BatchNorm
    └─ Dense(7, softmax) → Emotion Label + Confidence %
```

### 3. VoiceActivityDetector (`app/core/vad.py`)

| Property | Value |
|----------|-------|
| **Model** | Silero VAD v4 |
| **Framework** | PyTorch |
| **Sample Rate** | 16 kHz mandatory |
| **Frame Size** | 512 samples |
| **Threshold** | 0.5 (configurable) |

---

## 🗄️ Database Schema

### Table: `sessions`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment primary key |
| `session_uuid` | VARCHAR(64) | UUID string exposed to frontend |
| `start_time` | DATETIME | Session start timestamp |
| `end_time` | DATETIME | Session end timestamp |

### Table: `emotion_logs`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment primary key |
| `session_id` | INTEGER (FK) | References `sessions.id` |
| `timestamp` | DATETIME | Prediction timestamp |
| `emotion_label` | VARCHAR(32) | Detected emotion (angry, happy, etc.) |
| `arousal` | FLOAT | Arousal dimension (0.0–1.0) |
| `dominance` | FLOAT | Dominance dimension (0.0–1.0) |
| `valence` | FLOAT | Valence dimension (0.0–1.0) |
| `confidence` | FLOAT | Prediction confidence (0.0–1.0) |
| `duration_s` | FLOAT | Duration of audio segment |
| `offset_s` | FLOAT | Time offset in original audio |
| `scores_json` | VARCHAR(512) | Full probability distribution (JSON) |
| `latency_ms` | FLOAT | Inference latency in milliseconds |

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/api/analyze` | Upload audio for full analysis |
| `GET` | `/api/sessions` | List all historical sessions |
| `GET` | `/api/emotions/{uuid}` | Get detailed segment data for a session |
| `WS` | `/stream` | Real-time audio streaming pipeline |

> Full API documentation available at [API.md](./API.md)

---

## 📂 Project Structure

```
VoxDynamics/
├── app/                              # Production backend
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point + routes
│   ├── config.py                     # Pydantic settings (env vars)
│   ├── api/
│   │   └── websocket.py              # WebSocket handler
│   ├── core/
│   │   ├── cnn_predictor.py          # 1D-CNN: feature extraction + inference
│   │   ├── processor.py              # Pipeline orchestrator (VAD → CNN)
│   │   └── vad.py                    # Silero VAD wrapper
│   ├── db/
│   │   ├── database.py               # Async engine + session factory
│   │   └── models.py                 # SQLAlchemy ORM models
│   └── frontend/
│       ├── static/
│       │   ├── css/style.css         # Dark glassmorphism theme
│       │   └── js/
│       │       ├── app.js            # SPA logic + API calls
│       │       └── charts.js         # 5 Plotly interactive charts
│       └── template/
│           └── index.html            # Main dashboard
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # ← This file
│   ├── API.md                        # API reference
│   ├── SETUP.md                      # Detailed setup guide
│   ├── DEPLOYMENT.md                 # Deployment guide
│   ├── METHOD.md                     # Preprocessing methodology
│   └── benchmark/
│       ├── BASELINE_REPORT.md        # Wav2Vec2 baseline (25.40%)
│       ├── CALIBRATION_REPORT.md     # Centroid calibration (34.70%)
│       └── CNN_REPORT.md             # CNN pipeline tuning (→ 97.25%)
├── models/
│   ├── best_model1_weights.h5        # Pre-trained CNN weights
│   ├── scaler2.pickle                # StandardScaler (48k samples)
│   └── encoder.pkl                   # LabelEncoder
├── src/                              # Offline training scripts
├── docker-compose.yml                # App + PostgreSQL
├── Dockerfile                        # Python 3.10-slim
└── README.md                         # Project overview
```

---

## 🔬 Research Progression

The architecture evolved through 6 experiments:

| Experiment | Approach | Accuracy | Key Insight |
|:-----------|:---------|:--------:|:------------|
| Exp 1 | Wav2Vec2 + Fixed Centroids | 25.40% | Centroid–embedding mismatch; 3 classes at 0% |
| Exp 2 | Wav2Vec2 + Calibrated Centroids | 34.70% | +9.3pp but hard ceiling reached |
| Exp 3 | 1D-CNN, per-segment normalization | 23.56% | Normalization destroyed loudness info |
| Exp 4 | 1D-CNN + Global Normalization | ~55% | Preserving relative energy recovered Happy/Angry |
| Exp 5 | 1D-CNN + Left-Aligned Padding | ~75% | Matching training data distribution |
| **Exp 6** | **1D-CNN + Full Pipeline** | **80%** | 4/5 segments correct on real mix-file |
| **Benchmark** | **CNN on RAVDESS** | **97.25%** | Model's true potential on clean data |

> **Core Discovery:** The CNN required *identical preprocessing* to its training conditions. The jump from 23% → 80% came entirely from fixing the preprocessing pipeline, not from changing the model architecture.

---

## 📊 Performance Characteristics

| Metric | Value |
|:-------|:-----:|
| CNN Benchmark Accuracy | 97.25% (RAVDESS/CREMA-D) |
| Multi-Utterance Accuracy | 80% (5-emotion mix file) |
| Avg Confidence (correct) | 96.5% |
| Inference Speed | ~0.26s / segment (CPU) |
| Supported Emotions | 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| Min Audio Duration | Any (VAD auto-detects) |
| Max Upload Size | 50 MB |
| Supported Formats | .wav, .mp3, .flac |

---

### Preprocessing That Matters

| Technique | Impact | Rationale |
|:----------|:------:|:----------|
| Global normalization (once per file) | +30% | Preserves relative loudness between utterances |
| Left-aligned padding (fix_length) | +20% | Matches training data — speech starts at onset |
| 200ms silence buffer | +5% | Prevents phoneme clipping at boundaries |
| Dual-path SR (16k VAD, orig SR CNN) | Quality | Avoids aliasing artifacts in CNN features |
| No per-segment normalization | Critical | Single biggest accuracy jump (23% → 55%) |

---

*Documentation maintained with the system. Last updated: July 2026.*
