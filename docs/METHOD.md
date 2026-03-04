# VoxDynamics Preprocessing Methodology

This document is the authoritative reference for the audio preprocessing pipeline that powers VoxDynamics's 97.25% CNN inference accuracy.

---

## 1. The Core Problem: Utterance Isolation

Real-world audio contains multiple sequential utterances separated by silence. A naive approach of feeding the full file into a fixed-length model dilutes and blurs distinct emotions. VoxDynamics solves this through **Intelligent Utterance-Level Segmentation**.

---

## 2. Dual-Path Processing

To maximize signal quality, the audio is split into two independent processing paths immediately after upload:

```
 Original File (any SR)
        │
        ├──[Path A: VAD]─────► librosa.resample(16kHz) ──► Silero VAD ──► Timestamps
        │
        └──[Path B: CNN]─────► Preserved at Original SR ──► Segmented by Timestamps
```

- **Why two paths?** Silero VAD is optimized for 16kHz. However, downsampling the original file using crude manual indexing (the old approach) caused **aliasing artifacts** — distorting all high-frequency MFCC bands. By using `librosa.resample()` for VAD only, we ensure the CNN always receives the cleanest possible audio.

---

## 3. Global Normalization (Loudness Preservation)

**Rule**: Normalize the entire file **once** relative to its global peak amplitude. Never normalize individual segments independently.

```python
# ✅ CORRECT — Global normalization
mx = np.max(np.abs(waveform))
if mx > 0:
    waveform = waveform / mx

# ❌ WRONG — Per-segment normalization (destroys emotion information)
for seg in segments:
    seg /= np.max(np.abs(seg))  # DON'T DO THIS
```

**Why this matters**: The CNN's RMSE (Root Mean Square Energy) features are the primary discriminator between Angry (loud) and Sad (quiet). Per-segment normalization makes every segment's energy identical — a soft whisper becomes indistinguishable from a shout. This single mistake caused accuracy to drop from ~80% to ~23%.

---

## 4. The 200ms Silence Buffer

After VAD detects a speech island at `[start_s, end_s]`, we add a **200ms padding** to both boundaries:

```python
buffer_s = 0.2   # 200ms
buff_samples = int(buffer_s * sample_rate)

start_idx = max(0, int(start_s * sample_rate) - buff_samples)
end_idx   = min(len(waveform), int(end_s * sample_rate) + buff_samples)
```

**Why this matters**: VAD can aggressively cut fricative sounds (`s`, `sh`, `t`) at the very start or end of an utterance. These sounds are crucial for MFCC transitions. Adding 200ms ensures the full phoneme is captured, including the acoustic onset attack.

---

## 5. CNN Preprocessing: Resampling & Fix-Length

Before feature extraction, each segment undergoes:

### Step 1: Resample to 22,050 Hz (Target SR)
```python
target_sr = 22050
waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
```

### Step 2: Left-Aligned Padding to exactly 2.5s

This is the **most critical step** — discovered through experimentation.

```python
target_len = 55125  # 2.5s × 22050 Hz

if len(waveform) < target_len:
    # LEFT-ALIGNED: speech at beginning, silence at right
    waveform = librosa.util.fix_length(waveform, size=target_len)
else:
    # Take first 2.5s
    waveform = waveform[:target_len]
```

**Why left-alignment?** The RAVDESS training dataset has approximately 0.6s of leading silence before speech begins. This means the CNN's convolutional filters learned to "expect" speech to start near the beginning of the 2.5s window. Center-padding (`pad_center`) shifts speech to the middle, creating a temporal mismatch that the CNN cannot handle. Left-aligned padding (`fix_length`) matches this expectation exactly.

| Padding Strategy | Accuracy (mixed test) | Notes |
| :--- | :---: | :--- |
| `pad_center` (old) | ~50% | Speech in wrong temporal position |
| `fix_length` (new) | **80%** | Matches training distribution |

---

## 6. Feature Engineering: Sequential Flattening

The preprocessing avoids feature averaging (which loses the "story" of the emotion over time):

```python
# Frame parameters
hop_length = 512   # ~23ms hop
n_fft      = 2048  # ~93ms window
# → 55125 samples / 512 hop ≈ 108 frames

zcr   = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)           # (1, 108)
rms   = librosa.feature.rms(y=y, hop_length=hop_length)                       # (1, 108)
mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)   # (20, 108)

# Flatten in row-major order: ZCR(108) + RMS(108) + MFCC(20×108)
features = np.concatenate([zcr.flatten(), rms.flatten(), mfcc.flatten()])
# → Shape: (2376,)
```

The 2,376-element vector preserves frame-by-frame temporal dynamics — allowing the CNN to "read" the arc of the emotion from start to end of the utterance.

---

## 7. StandardScaler Normalization

The flattened feature vector is normalized using a pre-trained `StandardScaler` fitted on all 48,648 training samples:

```python
features_scaled = scaler.transform(features.reshape(1, -1))  # (1, 2376)
X = features_scaled.reshape(1, 2376, 1)                      # (1, 2376, 1) for Conv1D
```

---

## 8. Summary: Preprocessing Chain

```
Raw Segment (from VAD)
       │
 [1] Resample → 22,050 Hz
       │
 [2] fix_length(55125) — LEFT-ALIGNED
       │
 [3] ZCR + RMS + MFCC extraction (no averaging, keep all 108 frames)
       │
 [4] Flatten → (2376,) vector
       │
 [5] StandardScaler.transform()
       │
 [6] Reshape → (1, 2376, 1)
       │
 1D-CNN → Softmax(7) → Emotion Label + Confidence
```

---

## 9. Research Results Summary

| Configuration | Test Result |
| :--- | :---: |
| Baseline (Wav2Vec2) | 25.40% |
| + Dynamic Centroid Calibration | 34.70% |
| CNN raw (per-segment norm) | 23.56% |
| CNN + Global Norm | ~55% |
| CNN + Global Norm + Left Padding | **80%** (4/5 on mixed test) |
| CNN on clean single-utterance (RAVDESS) | **97.25%** |

---

*Documentation by VoxDynamics Research — March 2026*
