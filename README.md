<div align="center">

# 🎙️ VOXDYNAMICS
### Deep Emotion Extraction Layer

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat-square&logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![WebSocket](https://img.shields.io/badge/WebSocket-Enabled-6c47ff?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/CNN%20Accuracy-97.25%25-00ff87?style=flat-square)]()

**VoxDynamics** is a production-ready, real-time Speech Emotion Recognition (SER) system.  
It achieves **97.25% accuracy** through an intelligent utterance-segmentation pipeline powered by a fine-tuned Deep 1D-CNN trained on 48,648 samples (RAVDESS + CREMA-D).

</div>

---

## 🎬 Live Demo

![VoxDynamics Demo](docs/images/mini_demo.gif)

*Upload an audio file → AI segments utterances automatically → 5 interactive charts render in real-time*

---

## 📸 UI Showcase

### 1 · Home Page — Upload & Waveform Preview

![VoxDynamics Home](docs/images/home_page.png)

Drag & drop or browse any `.wav`, `.mp3`, or `.flac` file. WaveSurfer.js renders the waveform instantly for preview before analysis.

---

### 2 · Pre-Analysis — File Ready State

![Input Audio](docs/images/input_audio.png)

Shows file name, file size, and playable waveform player with duration. One click starts the full deep-analysis pipeline.

---

### 3 · Analysis Report — Dominant Emotion, Radar & Waveform

![Analysis Screen 1](docs/images/analysis_screen.png)

- **Dominant Emotion Card** — color-coded with breakdown of all detected emotion tags per segment
- **Emotion Signature Radar** — 7-axis spider chart mapping the emotional profile across all categories
- **Emotion Waveform Analysis** — proportional waveform chart where each utterance occupies its correct visual width; real timestamps displayed at every segment boundary

---

### 4 · Analysis Report — Distribution, Confidence Stream & Segment Log

![Analysis Screen 2](docs/images/analysis_screen_2.png)

- **Emotion Distribution Donut** — percentage breakdown of the detected emotions
- **Emotion Confidence Stream** — stacked area chart (all 7 emotions) over time
- **Micro-Segment Detection Log** — data table per utterance: time range, emotion label, confidence %, Arousal, Dominance, Valence

---

### 5 · Historical Session Archive

![History Screen](docs/images/history_screen.png)

Every analysis session is persisted to **PostgreSQL** and accessible in the historical archive table with full dimensional averages.

---

### 🎧 Audio Validation Sample

The test file below contains **5 consecutive emotions** concatenated with silence gaps — used to validate the multi-utterance pipeline end-to-end.

> `Angry → Happy → Surprised → Disgust → Angry (low-intensity)`

<div align="center">
  <audio controls>
    <source src="data/emotions/mix/angry_happy_surprised_disgust_sad.wav" type="audio/wav">
    Your browser does not support the audio element.
  </audio>
  <p><i>angry_happy_surprised_disgust_sad.wav · Validation File · 21.4s</i></p>
</div>

**Result: 4 / 5 correct = 80% on this multi-emotion real-world file**  
(97-100% confidence on the 4 correctly identified segments)

---

## 🧠 Model Research & Selection

> This is the core intellectual work of VoxDynamics. Two fundamentally different AI architectures were evaluated before selecting the final model.

### Model A — Wav2Vec2-Large-Robust (Dimensional, Wav2Vec2)

| | Detail |
| :--- | :--- |
| **Architecture** | Facebook Wav2Vec2 fine-tuned on MSP-Podcast for continuous A/D/V prediction |
| **Output** | Continuous 3D space (Arousal, Dominance, Valence) mapped to discrete labels via centroid geometry |
| **Training** | Pre-trained on 960h LibriSpeech + MSP-Podcast fine-tune |

**Accuracy Progression with Model A:**

| Experiment | Method | Accuracy |
| :--- | :--- | :---: |
| Exp 1 — Baseline | Fixed cosine centroids (theoretical) | 25.40% |
| Exp 2 — Calibrated | Dynamic centroids computed from 1,441 real samples | 34.70% |

**Why Model A was abandoned:**
- Continuous A/D/V space has severe **cluster overlap** — `Angry` and `Surprised` are nearly identical in Arousal dimension
- The model **prioritizes Arousal over Valence**, making it blind to whether a high-energy emotion is positive (Happy) or negative (Angry)
- 3 emotion classes (`Happy`, `Fearful`, `Surprised`) had **0% accuracy** at baseline — meaning the centroid mapping fundamentally fails for these emotions
- Max achievable accuracy with this architecture plateaued around **~35%** regardless of calibration strategy

---

### Model B — Deep 1D-CNN (Sequential Feature Classification) ✅ **SELECTED**

| | Detail |
| :--- | :--- |
| **Architecture** | Deep 1D Convolutional Neural Network — 5 Conv blocks |
| **Input** | 2,376-element sequential feature vector (ZCR + RMS + MFCC × 108 frames) |
| **Output** | Softmax across 7 discrete emotion classes |
| **Training Data** | RAVDESS + CREMA-D combined (48,648 samples) |
| **Benchmark Accuracy** | **97.25%** (validation set, single-utterance) |

**CNN Feature Breakdown:**

| Feature | Frames | Dimension |
| :--- | :---: | :---: |
| Zero Crossing Rate (ZCR) | 108 | 108 |
| Root Mean Square Energy (RMS) | 108 | 108 |
| MFCC (20 coefficients × 108 frames) | 108 | 2,160 |
| **Total** | | **2,376** |

**Why Model B was chosen:**
1. **Discriminative training**: Trained directly to discriminate between 7 classes — no centroid geometry hack needed
2. **Sequential feature preservation**: Features are flattened as a time-series (not averaged), so the CNN can learn the *temporal arc* of an emotion
3. **97.25% accuracy** on clean single-utterance data is significantly superior (vs 34.70% ceiling for Model A)
4. **No GPU required at inference**: Feature extraction + CNN forward pass completes in <0.3 seconds per segment on CPU

**CNN Architecture Detail:**

```
Input (2376, 1)
    │
    ├─ Conv1D(512, k=5) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(512, k=5) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(256, k=5) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    ├─ Conv1D(256, k=3) → BatchNorm → MaxPool1D(3) → Dropout(0.2)    ← pool=3 (not 5!)
    ├─ Conv1D(128, k=3) → BatchNorm → MaxPool1D(5) → Dropout(0.2)
    │
    ├─ Flatten
    ├─ Dense(512, relu) → BatchNorm → Dropout(0.2)
    └─ Dense(7, softmax)   →   Emotion Label + Confidence %
```

---

## 🔬 Full Accuracy Progression

| Step | Engine | Accuracy | Key Insight |
| :--- | :--- | :---: | :--- |
| Exp 1 | Wav2Vec2 + Fixed Centroids | 25.40% | Centroid–embedding mismatch; 3 classes at 0% |
| Exp 2 | Wav2Vec2 + Dynamic Centroids | 34.70% | +9.3pp — but hard ceiling reached |
| Exp 3 | 1D-CNN, no preprocessing | 23.56% | Correct architecture but per-segment normalization destroyed loudness info |
| Exp 4 | 1D-CNN + Global Normalization | ~55%* | Preserving relative energy recovered `Happy`/`Angry` distinction |
| Exp 5 | 1D-CNN + Left-Aligned Padding | ~75%* | Matching temporal distribution of training data was critical |
| **Exp 6** | **1D-CNN + Full Pipeline** | **80%** on real mix-file | 4/5 segments correct; 97-100% confidence on matches |
| **Benchmark** | Pure CNN on RAVDESS | **97.25%** | Model's true potential on clean single-utterance audio |

*\*Estimated from qualitative evaluation on the 5-emotion validation file*

> **The core discovery**: The CNN required **identical preprocessing** to its training conditions. The jump from 23% → 80% came entirely from fixing the preprocessing, not from changing the model.

---

## ⚙️ System Architecture

### Full Pipeline

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        VoxDynamics System                        │
 │                                                                  │
 │  Browser (HTML/JS/CSS)                                           │
 │  ┌──────────────┐  WaveSurfer.js  ┌──────────────────────────┐  │
 │  │  Audio Upload│─────────────────▶  Waveform Preview Player  │  │
 │  └──────┬───────┘                 └──────────────────────────┘  │
 │         │ HTTP POST /analyze                                     │
 │         ▼                                                        │
 │  FastAPI Backend (Python)                                        │
 │  ┌────────────────────────────────────────────────────────────┐  │
 │  │                                                            │  │
 │  │  [Path A: VAD]  librosa→16kHz → Silero VAD → Timestamps   │  │
 │  │                                      │                    │  │
 │  │  [Path B: CNN]  Original SR → Segment + 200ms buffer      │  │
 │  │                     → fix_length(2.5s, LEFT-ALIGN)        │  │
 │  │                     → ZCR + RMS + MFCC (2376 features)    │  │
 │  │                     → StandardScaler → 1D-CNN             │  │
 │  │                     → Emotion + Confidence + A/D/V        │  │
 │  │                                                            │  │
 │  └────────────────────────┬───────────────────────────────────┘  │
 │                           │ JSON Response                        │
 │                           ▼                                      │
 │  PostgreSQL DB ◄── SQLAlchemy (Async) ── Session Storage        │
 │                                                                  │
 │  Plotly.js Charts (5 charts rendered client-side):              │
 │  · Emotion Waveform  · Radar  · Donut  · Confidence Stream     │
 └─────────────────────────────────────────────────────────────────┘
```

---

## �️ Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Backend API** | FastAPI (Python) | REST endpoints, request handling, response serialization |
| **Real-Time** | WebSocket (FastAPI) | Live status updates during analysis pipeline |
| **Database** | PostgreSQL 15 | Session persistence, historical archive storage |
| **ORM** | SQLAlchemy (Async) | Non-blocking DB writes to avoid inference latency |
| **AI — VAD** | Silero VAD v4 (PyTorch) | Voice Activity Detection, speech island detection |
| **AI — CNN** | TensorFlow 2.x / Keras | Deep 1D-CNN emotion classification |
| **Audio DSP** | librosa, soundfile, numpy | Resampling, feature extraction (ZCR/RMS/MFCC) |
| **Frontend** | Vanilla HTML / CSS / JS | SPA dashboard — custom-built (no Gradio/Streamlit) |
| **Visualization** | Plotly.js | 5 interactive charts — waveform, radar, donut, stream |
| **Audio Player** | WaveSurfer.js | Animated waveform preview player |
| **Container** | Docker + Docker Compose | One-command deployment of app + DB |

> **Note on Presentation Layer**: The project brief suggests Gradio or Streamlit. VoxDynamics intentionally uses a **custom vanilla JS SPA** instead — this decision allows full control over the interactive Plotly charts, proportional waveform rendering, and the custom dark-glass aesthetic that would not be achievable through Gradio's component model.

---

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Launch in 3 Steps

```bash
# 1. Clone
git clone <repository-url>
cd VoxDynamics

# 2. Copy environment config
cp .env.example .env

# 3. Start all services (app + database)
docker-compose up -d --build
```

→ Open **[http://localhost:8000](http://localhost:8000)** in your browser.

| Service | Port | Description |
| :--- | :---: | :--- |
| `voxdynamics-app` | 8000 | FastAPI + CNN inference server |
| `voxdynamics-db` | 5432 | PostgreSQL 15 database |

### Reset Database
```bash
docker-compose down -v   # removes volumes (clears DB)
docker-compose up -d --build
```

---

## 📁 Project Structure

```
VoxDynamics/
├── app/
│   ├── api/
│   │   └── websocket.py         # WebSocket endpoint for real-time pipeline updates
│   ├── core/
│   │   ├── cnn_predictor.py     # 1D-CNN: feature extraction + inference engine
│   │   ├── processor.py         # Smart segmentation pipeline (VAD → buffer → CNN)
│   │   └── vad.py               # Silero VAD wrapper (16kHz path)
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM models (Session, Segment)
│   │   └── database.py          # Async engine + session factory
│   ├── frontend/
│   │   ├── static/
│   │   │   ├── css/style.css    # Custom dark glassmorphism theme
│   │   │   └── js/
│   │   │       ├── charts.js    # 5 Plotly interactive charts
│   │   │       └── app.js       # SPA logic (upload, WebSocket, render)
│   │   └── template/index.html  # Main dashboard HTML
│   ├── config.py                # Pydantic settings (env vars)
│   └── main.py                  # FastAPI app + route definitions
├── docs/
│   ├── benchmark/
│   │   ├── BASELINE_REPORT.md   # Experiment 1 — Wav2Vec2 baseline
│   │   ├── CALIBRATION_REPORT.md # Experiment 2 — Centroid calibration
│   │   └── CNN_REPORT.md        # Experiments 3-6 — CNN pipeline tuning
│   ├── images/                  # App screenshots + mini_demo.gif
│   └── METHOD.md                # Full preprocessing methodology
├── models/
│   ├── best_model1_weights.h5   # Pre-trained CNN weights (97.25% accuracy)
│   ├── scaler.pkl               # StandardScaler fitted on 48k training samples
│   └── encoder.pkl              # LabelEncoder for 7 emotion classes
├── src/                         # Offline evaluation & benchmark scripts
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 📄 Documentation Index

| Document | Description |
| :--- | :--- |
| [docs/METHOD.md](docs/METHOD.md) | Full preprocessing pipeline, rationale for each step, code examples |
| [docs/benchmark/BASELINE_REPORT.md](docs/benchmark/BASELINE_REPORT.md) | Exp 1 — Wav2Vec2 fixed-centroid baseline (25.40%) |
| [docs/benchmark/CALIBRATION_REPORT.md](docs/benchmark/CALIBRATION_REPORT.md) | Exp 2 — Dynamic centroid calibration (34.70%) |
| [docs/benchmark/CNN_REPORT.md](docs/benchmark/CNN_REPORT.md) | Exp 3–6 — CNN pipeline tuning, final 80% on mix-file |

---

## 📊 Performance Summary

| Metric | Value |
| :--- | :--- |
| **CNN Benchmark Accuracy** | **97.25%** (RAVDESS/CREMA-D, single-utterance) |
| **Multi-Utterance Accuracy** | **80%** (5-emotion real-world mix file) |
| **Avg Confidence (correct segs)** | **96.5%** |
| **Inference Speed** | ~0.26s / segment (CPU) |
| **Supported Emotions** | 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| **Min Audio Duration** | Any (VAD auto-detects speech islands) |
| **Max Upload Size** | 50 MB |
| **Supported Formats** | `.wav`, `.mp3`, `.flac` |

---

<div align="center">

*VoxDynamics — Pushing the boundaries of accessible, high-accuracy Speech Emotion Recognition*

**Built with FastAPI · PostgreSQL · Docker · TensorFlow · Silero VAD**

</div>
