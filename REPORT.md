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

1. **Audio Capture**: Gradio's `gr.Audio(streaming=True)` captures microphone input at native sample rate
2. **Resampling**: Audio is downsampled to 16 kHz mono if needed
3. **Ring Buffer**: Samples appended to a sliding window buffer (3 seconds capacity)
4. **Voice Activity Detection**: Silero VAD checks each chunk for speech (≥ 30ms, < 1ms inference)
5. **Emotion Inference**: If speech detected, the full buffer window runs through Wav2Vec2 → RegressionHead → [A, D, V]
6. **EMA Smoothing**: Exponential Moving Average smooths dimensional outputs across time
7. **Label Mapping**: Smoothed (A, D, V) point mapped to nearest emotion centroid (Euclidean distance)
8. **Result Delivery**: JSON result sent to UI (Gradio state update) and WebSocket clients
9. **Async Logging**: Non-blocking `asyncio.create_task` writes the result to PostgreSQL

---

## 2. Research Thinking

### 2.1 Why Wav2Vec2 (Acoustic-Based) Instead of STT + LLM?

| Criteria | Acoustic-Based (Wav2Vec2) | STT + LLM Approach |
|----------|--------------------------|---------------------|
| **Language Support** | ✅ Language-agnostic (works on any language without configuration) | ❌ Requires STT model per language |
| **Latency** | ✅ Single-pass inference (~200-400ms) | ❌ STT + LLM = 2+ inference steps (>1s) |
| **Paralinguistic Cues** | ✅ Captures tone, pitch, rhythm, prosody directly | ❌ Loses all acoustic information after STT |
| **Emotional Nuance** | ✅ Detects sarcasm, irony through vocal patterns | ❌ Text-only misses vocal subtlety |
| **Resource Usage** | ✅ Single model (~1GB) | ❌ STT model + LLM = heavy resource usage |
| **Robustness** | ✅ Works in noisy environments (trained on diverse data) | ❌ STT fails with noise → cascading errors |

**Key Insight**: Emotions are primarily conveyed through **how** we speak (prosody, pitch contour, speaking rate, energy), not **what** we say. The Wav2Vec2 model captures these acoustic features directly from the raw waveform, making it fundamentally more suitable for SER tasks, especially when multi-language support is a requirement.

### 2.2 The audeering Model

The `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` model:

- **Base**: Wav2Vec2-Large-Robust (pre-trained on 63k hours of diverse audio)
- **Architecture**: Pruned from 24 → 12 transformer layers (faster inference)
- **Fine-tuned on**: MSP-Podcast v1.7 (naturalistic emotional speech from podcast recordings)
- **Output**: 3 continuous dimensions (Arousal, Dominance, Valence) via a `RegressionHead`
- **Why MSP-Podcast?**: Unlike acted emotion datasets (RAVDESS, IEMOCAP), MSP-Podcast contains **natural, spontaneous** emotional speech — leading to better generalization to real-world scenarios

### 2.3 Voice Activity Detection (Silero VAD)

**Purpose**: Gate the emotion model — only run inference when speech is actually present.

**Why it matters**:
- Prevents false predictions on silence/background noise
- Saves compute (emotion model runs ~200ms per inference — no point wasting it on noise)
- Silero VAD processes 30ms chunks in <1ms on CPU — negligible overhead

**Configuration**:
- Threshold: 0.5 (default, tuned for balanced precision/recall)
- Sample rate: 16 kHz (matching the emotion model)

### 2.4 Sliding Window Approach

Rather than processing each tiny chunk independently, we maintain a **ring buffer** (3-second sliding window):

```
Time →
[----chunk1----][----chunk2----][----chunk3----][----chunk4----]
                          └───────── Buffer Window ──────────┘
                          (emotion model sees this context)
```

**Benefits**:
- More audio context → more accurate emotion prediction
- The emotion model was trained on utterances (1-10s), not 30ms fragments
- Ring buffer (`collections.deque(maxlen=N)`) automatically drops oldest samples — constant memory usage

### 2.5 EMA Smoothing

**Problem**: Raw model outputs fluctuate frame-to-frame, causing "jittery" emotion labels in the UI.

**Solution**: Exponential Moving Average (EMA):

```
EMA_t = α × x_t + (1 - α) × EMA_{t-1}
```

Where `α = 0.3` (tuned for responsiveness while maintaining stability):
- Higher α (0.5-1.0) = responsive but jittery
- Lower α (0.1-0.2) = smooth but sluggish
- `α = 0.3` = good balance for real-time emotion tracking

This is applied independently to Arousal, Dominance, and Valence before mapping to discrete labels.

### 2.6 Emotion Centroid Mapping

We define **emotion centroids** in 3D (A, D, V) space based on psychological research (Russell's Circumplex Model extended to 3D):

| Emotion | Arousal | Dominance | Valence |
|---------|---------|-----------|---------|
| Happy | 0.75 | 0.60 | 0.85 |
| Angry | 0.85 | 0.80 | 0.20 |
| Sad | 0.25 | 0.25 | 0.20 |
| Neutral | 0.45 | 0.50 | 0.50 |
| Fear | 0.80 | 0.20 | 0.20 |
| Surprise | 0.80 | 0.50 | 0.75 |
| Disgust | 0.55 | 0.70 | 0.20 |
| Calm | 0.20 | 0.45 | 0.65 |

**Classification**: Euclidean distance from the observed (A, D, V) point to each centroid. Nearest centroid = predicted emotion.

**Confidence**: `1 / (1 + distance)` — closer to centroid = higher confidence.

---

## 3. Implementation Details

### 3.1 Async Database Logging

Database writes happen via `asyncio.create_task()` — fire-and-forget pattern:
- The emotion result is sent to the client **immediately**
- DB write happens in background — never blocks the streaming loop
- Failed DB writes are caught and logged, not propagated to the user

### 3.2 Dual Interface Design

| Interface | Protocol | Use Case |
|-----------|----------|----------|
| **Gradio** | HTTP (streaming callback) | Quick demo, browser-based |
| **WebSocket** | `ws://` binary frames | Production integration, custom clients |

Both share the same `AudioProcessor` core, ensuring consistent behavior.

### 3.3 Docker Deployment

```yaml
postgres:  health check → pg_isready every 5s
app:       depends_on postgres (healthy) → models loaded on startup
```

Model weights are cached in a Docker volume (`model_cache`) — subsequent starts are instant.

---

## 4. Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| **End-to-end Latency** | < 500ms | VAD (~1ms) + Model (~200-400ms on CPU) |
| **Model Size** | ~1 GB | Wav2Vec2-Large-Robust pruned to 12 layers |
| **Memory Usage** | ~2-3 GB | Model + audio buffer + inference overhead |
| **Supported Languages** | All | Acoustic-based, no language dependence |
| **Audio Requirements** | 16 kHz, mono, float32 | Standard speech processing format |

---

## 5. Limitations & Future Work

### Current Limitations
- **CPU-only inference**: ~200-400ms latency. GPU would bring this to ~50ms.
- **No speaker diarization**: Cannot distinguish between multiple speakers in the same session.
- **Emotion centroids are static**: Could be personalized per user over time.

### Future Improvements
- **GPU acceleration** with ONNX Runtime for faster inference
- **Speaker diarization** using pyannote.audio to track per-speaker emotions
- **Adaptive centroids**: Learn user-specific emotion patterns over multiple sessions
- **Multi-modal fusion**: Combine audio SER with text sentiment (from STT) for higher accuracy
- **WebRTC integration** via `fastrtc` for lower-latency browser streaming

---

## 6. References

1. Wagner, J., et al. (2023). "Dawn of the Transformer Era in Speech Emotion Recognition." *arXiv: 2203.07378*
2. Silero Team. "Silero VAD: Pre-trained Voice Activity Detector." GitHub: snakers4/silero-vad
3. Russell, J.A. (1980). "A Circumplex Model of Affect." *Journal of Personality and Social Psychology*
4. Lotfian, R. & Busso, C. (2019). "Building Naturalistic Emotionally Balanced Speech Corpus." *IEEE TAFFC* (MSP-Podcast)
