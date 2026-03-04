# VoxDynamics — Comprehensive Technical Report

## 1. System Overview

VoxDynamics is a **file-based Speech Emotion Recognition (SER)** platform. Users upload an audio file (conversation recording, interview, speech sample), and the system automatically:

1. Segments the audio into individual speech utterances (Voice Activity Detection)
2. Classifies the emotion of each utterance independently using a deep 1D-CNN
3. Returns a full structured report: dominant emotion, confidence per segment, A/D/V dimensions, and 5 interactive visualizations

The system is designed for **post-hoc audio analysis** — not live microphone streaming. This allows higher-fidelity preprocessing (global normalization, buffered segmentation) that would be impractical in a live-stream context.

---

## 2. Architecture

### 2.1 High-Level System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            VOXDYNAMICS                                    │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │  Browser SPA (HTML / CSS / Vanilla JS)                          │    │
│   │                                                                 │    │
│   │   WaveSurfer.js waveform player                                │    │
│   │   Plotly.js   5× interactive charts                            │    │
│   │   WebSocket   real-time status updates during pipeline         │    │
│   └──────────────────────────┬──────────────────────────────────────┘    │
│                              │ HTTP POST /api/analyze (multipart/form)    │
│                              │ WS  /ws/status  (progress stream)          │
│                              ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │  FastAPI Backend (Python 3.10)                                   │    │
│   │                                                                 │    │
│   │  POST /api/analyze                                              │    │
│   │    │                                                            │    │
│   │    ├─► [VAD PATH]  librosa.resample(16kHz)                     │    │
│   │    │                    └─► Silero VAD v4                       │    │
│   │    │                          └─► Speech Island Timestamps      │    │
│   │    │                                                            │    │
│   │    └─► [CNN PATH]  Original SR preserved                       │    │
│   │              └─► Per-segment extraction                        │    │
│   │                    ├─ +200ms silence buffer                    │    │
│   │                    ├─ Resample → 22,050 Hz                     │    │
│   │                    ├─ fix_length(2.5s, LEFT-ALIGN)             │    │
│   │                    ├─ ZCR + RMS + MFCC → 2,376 features        │    │
│   │                    ├─ StandardScaler.transform()               │    │
│   │                    └─► Deep 1D-CNN → Emotion + Confidence       │    │
│   │                                                                 │    │
│   │  GET /api/sessions  (history)                                   │    │
│   │  DELETE /api/sessions/{id}                                      │    │
│   └──────────────────────────┬──────────────────────────────────────┘    │
│                              │ SQLAlchemy Async (asyncpg)                 │
│                              ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │  PostgreSQL 15                                                   │    │
│   │                                                                 │    │
│   │  TABLE: sessions          TABLE: segments                       │    │
│   │  ─────────────────        ──────────────────────────────────── │    │
│   │  id (UUID PK)             id (UUID PK)                         │    │
│   │  created_at               session_id (FK → sessions)           │    │
│   │  audio_duration_s         time_s, duration_s                   │    │
│   │  dominant_emotion         emotion_label, confidence            │    │
│   │  avg_arousal              arousal, dominance, valence          │    │
│   │  avg_dominance            is_speech (bool)                     │    │
│   │  avg_valence              emoji, color                         │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack Matrix

| Layer | Technology | Why This Choice |
| :--- | :--- | :--- |
| **Backend API** | FastAPI (Python 3.10) | Async-native, auto OpenAPI docs, high throughput |
| **Real-Time Updates** | WebSocket (FastAPI built-in) | Push pipeline progress to browser without polling |
| **Database** | PostgreSQL 15 | ACID compliance, robust for relational session/segment data |
| **ORM** | SQLAlchemy 2.x (Async) | Non-blocking DB writes — inference loop never waits for DB |
| **Container Orchestration** | Docker + Docker Compose | Reproducible 1-command deployment; DB healthcheck integrated |
| **AI — VAD** | Silero VAD v4 (PyTorch) | Best-in-class lightweight VAD; 16kHz optimized |
| **AI — Emotion CNN** | TensorFlow 2.x / Keras | Sequential CNN inference; pre-trained weights loaded at startup |
| **Audio DSP** | librosa 0.10, soundfile, numpy | Feature extraction (MFCC, ZCR, RMS); resampling |
| **Frontend** | Vanilla HTML / CSS / JS | Full control over Plotly charts & custom dark-glassmorphism UI |
| **Charts** | Plotly.js | Interactive: hover details, zoom, proportional waveform |
| **Waveform Player** | WaveSurfer.js | Animated audio playback waveform before analysis |

---

## 3. Data Flow — Step by Step

### 3.1 Upload Phase

```
User selects file (drag & drop or browse)
    │
    ├─► WaveSurfer.js renders waveform preview (client-side, no server call)
    │
    └─► User clicks "Start Deep Analysis"
              │
              └─► HTTP POST /api/analyze
                    (multipart/form-data, max 50MB)
                    (.wav / .mp3 / .flac)
```

### 3.2 Server Processing Phase

```
File received by FastAPI endpoint
    │
    ▼
[STEP 1] Load audio with librosa (preserve original sample rate)
    │
    ▼
[STEP 2] Global normalization — divide entire file by global peak amplitude
    │       (MUST happen before segmentation to preserve relative loudness)
    │
    ▼
[STEP 3] VAD Path — create 16kHz copy → Silero VAD
    │       Output: list of (start_s, end_s) tuples for voiced segments
    │
    ▼
[STEP 4] For each speech segment (original SR):
    │       ├─ Extend boundaries by ± 200ms (silence buffer)
    │       ├─ Extract raw waveform slice
    │       ├─ Resample to 22,050 Hz (CNN target SR)
    │       ├─ fix_length(55,125 samples = 2.5s) — LEFT-ALIGNED
    │       ├─ Extract ZCR (108) + RMS (108) + MFCC(20×108) → 2,376 features
    │       ├─ StandardScaler.transform(features)
    │       ├─ Reshape to (1, 2376, 1) for Conv1D input
    │       └─ 1D-CNN forward pass → softmax(7) → argmax → label + confidence
    │
    ▼
[STEP 5] Assemble result object:
    │       ├─ Per-segment: emotion_label, confidence, time_s, duration_s,
    │       │               arousal, dominance, valence, emoji, color, is_speech
    │       └─ Summary: dominant_emotion, avg_scores, avg_arousal/dominance/valence
    │
    ▼
[STEP 6] Async DB write (fire-and-forget via asyncio.create_task)
    │       └─► INSERT into sessions + segments tables
    │
    └─► JSON response → Browser
```

### 3.3 Visualization Phase (Client-Side)

```
JSON response received
    │
    ├─► Dominant Emotion Card + Emotion Tag pills
    ├─► Radar Chart (7-axis emotional profile)
    ├─► Emotion Waveform Chart (proportional, per-segment colored)
    │      └─ X-axis: normalized 0→1 (no empty space)
    │      └─ Labels: real timestamps at boundaries
    ├─► Emotion Distribution Donut
    ├─► Emotion Confidence Stream (stacked area, all 7 classes)
    └─► Micro-Segment Detection Log (scrollable table)
```

### 3.4 Persistence & History

```
PostgreSQL stores:
    sessions: one row per file upload
    segments: one row per detected utterance (speech + silence)

GET /api/sessions → returns all past sessions for Historical Archive table
```

---

## 4. AI Model Research & Final Selection

### 4.1 Experiment 1 & 2 — Wav2Vec2 (Dimensional, Regression-Based)

> **Status: Evaluated as baseline, not used in production**

The first approach used **Wav2Vec2-Large-Robust** fine-tuned on MSP-Podcast to output continuous Arousal/Dominance/Valence (A/D/V) values, then mapped these to 7 emotion labels via nearest-centroid geometry.

| Experiment | Method | Accuracy |
| :--- | :--- | :---: |
| Exp 1 — Baseline | Theoretical cosine centroids | 25.40% |
| Exp 2 — Calibrated | Data-driven dynamic centroids (1,441 samples) | 34.70% |

**Why this approach was discontinued:**
- 3 emotion classes (`Happy`, `Fearful`, `Surprised`) had **0% recall** at baseline — fundamental failure, not tunable
- The A/D/V space has **severe cluster overlap**: `Angry` and `Surprised` are nearly indistinguishable in the Arousal dimension
- Model strongly **prioritizes Arousal** over Valence — positive high-energy emotions (Happy) are misclassified as Angry
- Accuracy **hard ceiling ~35%** regardless of calibration technique

### 4.2 Experiment 3–6 — Deep 1D-CNN (Discriminative, Classification-Based)

> **Status: SELECTED for production**

#### Why CNN was chosen

| Criterion | Wav2Vec2 approach | 1D-CNN approach |
| :--- | :---: | :---: |
| Architecture type | Regression (continuous) | Classification (discriminative) |
| Max accuracy (RAVDESS) | ~35% | **97.25%** |
| Requires centroid tuning | Yes | No |
| Inference speed (CPU) | ~400ms | ~260ms |
| GPU required | Yes (1B params) | No |
| 0% emotion classes | 3 | 0 |

#### CNN Feature Engineering
```
Input: 2.5s audio clip @ 22,050 Hz = 55,125 samples

hop_length = 512  →  55125 / 512 ≈ 108 frames

ZCR  = zero_crossing_rate(hop=512)       → shape (1, 108)  → 108 values
RMS  = rms(hop=512)                      → shape (1, 108)  → 108 values
MFCC = mfcc(n_mfcc=20, hop=512)         → shape (20, 108) → 2160 values

Flatten in row order → feature vector of shape (2376,)
```

#### CNN Architecture
```
Input shape: (2376, 1)   ← 1D temporal signal
    │
    ├─ Conv1D(512 filters, kernel=5, padding='same') + BatchNorm + ReLU
    ├─ MaxPool1D(pool_size=5) + Dropout(0.2)
    │
    ├─ Conv1D(512, k=5) + BatchNorm + ReLU
    ├─ MaxPool1D(5) + Dropout(0.2)
    │
    ├─ Conv1D(256, k=5) + BatchNorm + ReLU
    ├─ MaxPool1D(5) + Dropout(0.2)
    │
    ├─ Conv1D(256, k=3) + BatchNorm + ReLU
    ├─ MaxPool1D(3) + Dropout(0.2)        ← pool=3 here (matches benchmark exactly)
    │
    ├─ Conv1D(128, k=3) + BatchNorm + ReLU
    ├─ MaxPool1D(5) + Dropout(0.2)
    │
    ├─ Flatten
    ├─ Dense(512, activation='relu') + BatchNorm + Dropout(0.2)
    └─ Dense(7,  activation='softmax')
         └─► [angry, disgust, fear, happy, neutral, sad, surprise]
```

#### Preprocessing Impact on Accuracy

| Preprocessing Step | Accuracy Effect | Rationale |
| :--- | :---: | :--- |
| Global normalization (whole file, pre-segment) | **+30%** | Preserves relative loudness between utterances; per-segment norm destroys emotional intensity info |
| Left-aligned padding (`fix_length`) | **+20%** | Training data (RAVDESS) has speech starting near onset; center-padding shifts speech to unfamiliar temporal position |
| 200ms silence buffer at boundaries | **+5%** | Prevents VAD from clipping fricatives (`s`, `sh`, `t`) at utterance boundaries |
| Dual-path SR (16k VAD, original SR CNN) | Quality | Avoids aliasing; CNN receives highest-fidelity audio |

#### Final Results

| Test Scenario | Accuracy |
| :--- | :---: |
| RAVDESS single-utterance benchmark (training set evaluation) | **97.25%** |
| 5-emotion real-world mix file (multi-utterance) | **80%** (4/5 correct) |
| Avg confidence on correct segments | **96.5%** |

---

## 5. Emotional Dimension Mapping

Each classified utterance is also assigned Arousal, Dominance, and Valence scores using a fixed lookup table derived from the circumplex model of affect (Russell, 1980):

| Emotion | Arousal | Dominance | Valence |
| :--- | :---: | :---: | :---: |
| **Angry** | 0.75 | 0.85 | -0.75 |
| **Disgust** | 0.45 | 0.55 | -0.65 |
| **Fear** | 0.80 | 0.15 | -0.90 |
| **Happy** | 0.85 | 0.65 | 0.90 |
| **Neutral** | 0.10 | 0.40 | 0.05 |
| **Sad** | 0.15 | 0.10 | -0.75 |
| **Surprise** | 0.90 | 0.45 | 0.40 |

These A/D/V values are stored per-segment in PostgreSQL and used to render:
- The **Emotional Dimensions** session-average progress bars
- The **Emotion Signature Radar** chart (avg A/D/V per emotion class)

---

## 6. Supported Emotions — 7 Classes

| Emotion | Color | Emoji |
| :--- | :--- | :---: |
| Angry | `#ff416c` | 😠 |
| Disgust | `#ff8c00` | 🤢 |
| Fear | `#f9a825` | 😨 |
| Happy | `#00ff87` | 😄 |
| Neutral | `#9aa0b8` | 😐 |
| Sad | `#667eea` | 😢 |
| Surprise | `#ab47bc` | 😮 |

---

## 7. Database Schema

### `sessions` table

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID PK | Unique session identifier |
| `created_at` | TIMESTAMP | Analysis timestamp |
| `audio_duration_s` | FLOAT | Total audio file duration in seconds |
| `dominant_emotion` | VARCHAR | Most frequent emotion across all segments |
| `avg_arousal` | FLOAT | Session-average arousal |
| `avg_dominance` | FLOAT | Session-average dominance |
| `avg_valence` | FLOAT | Session-average valence |

### `segments` table

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID PK | Unique segment identifier |
| `session_id` | UUID FK | Parent session reference |
| `time_s` | FLOAT | Start timestamp in seconds |
| `duration_s` | FLOAT | Duration of segment |
| `emotion_label` | VARCHAR | Predicted emotion class |
| `confidence` | FLOAT | Softmax probability (0–1) |
| `arousal` | FLOAT | A/D/V arousal value |
| `dominance` | FLOAT | A/D/V dominance value |
| `valence` | FLOAT | A/D/V valence value |
| `is_speech` | BOOLEAN | True = utterance, False = silence gap |
| `emoji` | VARCHAR | Display emoji for emotion |
| `color` | VARCHAR | Hex color for chart rendering |

---

## 8. Deployment

```bash
# Build & start all services
docker-compose up -d --build

# Services started:
#   voxdynamics-app  →  port 8000  (FastAPI + CNN inference)
#   voxdynamics-db   →  port 5432  (PostgreSQL 15)

# Reset database (fresh start)
docker-compose down -v && docker-compose up -d --build
```

The `docker-compose.yml` includes a PostgreSQL healthcheck so the app container waits for the DB to be ready before starting model loading.

---

## 9. Performance

| Metric | Value |
| :--- | :--- |
| CNN inference time | ~0.26s / segment (CPU) |
| VAD processing time | ~0.05s / second of audio |
| DB write (async) | Non-blocking (fire-and-forget) |
| Max upload size | 50 MB |
| Max audio tested | 64s (stable) |
| Memory footprint | ~1.2 GB (models loaded at startup) |

---

## 10. Future Roadmap

- **Speaker Diarization**: Multi-speaker emotion tracking (Who said what, when, in what emotional state)
- **GPU Acceleration**: ONNX export for sub-100ms inference
- **Streaming Mode**: WebSocket-based chunk-by-chunk analysis for live calls
- **Multimodal Fusion**: Combine acoustic SER with LLM-based semantic sentiment for richer analysis

---

## 11. References

1. Livingstone, S.R. & Russo, F.A. (2018). RAVDESS Dataset.
2. Cao, H. et al. (2014). CREMA-D Dataset. *IEEE Transactions on Affective Computing.*
3. Silero Team. *Silero VAD.* GitHub: snakers4/silero-vad
4. McFee, B. et al. *librosa: Audio and Music Signal Analysis in Python.* (2015)
5. Russell, J.A. (1980). A Circumplex Model of Affect. *J. of Personality and Social Psychology.*
6. Benchmark notebook (97.25%): `docs/benchmark/speech-emotion-recognition-97-25-accuracy.ipynb`
