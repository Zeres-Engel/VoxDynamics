# VoxDynamics — Comprehensive Technical Report

## 1. System Design

### 1.1 Overview

VoxDynamics is a **real-time Speech Emotion Recognition (SER)** system designed to continuously capture live microphone input, identify the speaker's emotional state, and track emotional changes throughout an ongoing conversation — all with minimal latency and support for multiple languages.

### 1.2 Architecture

```
┌───────────────┐
│   Client      │
│  (Mic Input)  │
└──────┬────────┘
       │ Audio Chunks (PCM float32, 16kHz)
       ▼
┌──────────────────────────────────────────────────────────┐
│                   VoxDynamics Server                      │
│                                                          │
│  ┌─────────────┐     ┌─────────────────────────────────┐ │
│  │ Gradio UI   │────▶│       AudioProcessor            │ │
│  │ (:7860)     │     │                                 │ │
│  └─────────────┘     │  1. Ring Buffer (3s window)     │ │
│                      │  2. Silero VAD                  │ │
│  ┌─────────────┐     │  3. Wav2Vec2 Inference          │ │
│  │ FastAPI WS  │────▶│  4. EMA Smoothing (α=0.3)      │ │
│  │ (:8000)     │     │  5. Centroid-based Mapping      │ │
│  └─────────────┘     └──────────────┬──────────────────┘ │
│                                     │                    │
│                      ┌──────────────▼──────────────────┐ │
│                      │     PostgreSQL (async)          │ │
│                      │     emotion_logs table          │ │
│                      └─────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow

1. **Audio Capture**: Captures microphone input at native sample rate.
2. **Resampling**: Audio is downsampled to 16 kHz mono.
3. **Ring Buffer**: Samples appended to a sliding window buffer (3 seconds).
4. **Voice Activity Detection**: Silero VAD checks each chunk for speech.
5. **Emotion Inference**: If speech detected, Wav2Vec2 predicts [A, D, V].
6. **EMA Smoothing**: Stabilizes dimensional outputs across time.
7. **Label Mapping**: Maps (A, D, V) to the nearest emotion centroid.
8. **Result Delivery**: Result sent via WebSocket to clients.
9. **Async Logging**: Async DB write via `asyncio.create_task`.

---

## 2. Methodology & Logic

For a detailed technical breakdown of the audio processing pipeline and AI models, please refer to the dedicated methodology document:

👉 **[Technical Methodology (docs/METHOD.md)](docs/METHOD.md)**

### 2.1 Why Wav2Vec2 (Acoustic-Based)?

Emotions are primarily conveyed through **how** we speak (prosody, pitch contour, speaking rate, energy), not **what** we say. The Wav2Vec2 model captures these acoustic features directly from the raw waveform, making it:
- **Language-Agnostic**: Works across cultures without configuration.
- **Low Latency**: Single-pass inference (~200-400ms).
- **Nuanced**: Detects paralinguistic cues (sarcasm, irony) that text-based sentiment misses.

---

## 3. Implementation Details

### 3.1 Async Database Logging
- DB writes happen via `asyncio.create_task()` (fire-and-forget).
- Immediate client response; background persistence.

### 3.2 Deployment
- **Dockerized**: Postgres health checks and automated model loading.
- **Async Data Layer**: Non-blocking ingestion ensures zero latency in the audio loop.

---

## 4. Performance Characteristics
- **Latency**: < 500ms (CPU).
- **Model Size**: ~1 GB.
- **Memory**: ~2.5 GB.

---

## 5. Future Work
- **GPU Acceleration**: Migrating to ONNX for <100ms latency.
- **Speaker Diarization**: Tracking emotions for multiple speakers.
- **Multimodal Fusion**: Combining acoustic emotion with semantic sentiment (STT + LLM).

---

## 6. References
1. Wagner, J., et al. (2023). "Dawn of the Transformer Era in Speech Emotion Recognition."
2. Silero Team. "Silero VAD." GitHub: snakers4/silero-vad
3. Russell, J.A. (1980). "A Circumplex Model of Affect."
