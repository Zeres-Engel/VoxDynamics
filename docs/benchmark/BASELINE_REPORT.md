# Baseline Performance Report: Emotion Engine v2.0

This report documents the quantitative baseline of the **VoxDynamics** emotion recognition pipeline. It establishes the "Ground Truth" performance before any research-driven preprocessing or calibration is applied.

## 1. Executive Summary
The current system achieves an **Overall Accuracy of 25.40%** on the validation dataset. While identifying `Neutral` and `Angry` tones effectively, the model significantly struggles with positive and high-arousal emotions like `Happy` and `Surprised`, often misclassifying them due to centroid misalignment.

## 2. Quantitative Performance Metrics

### 📊 Overall Statistics
| Metric | Value |
| :--- | :--- |
| **Total Samples** | 1,441 |
| **Correct Predictions** | 366 |
| **Processing Time** | ~384s (0.26s / sample) |
| **Overall Accuracy** | **25.40%** |

### 🔍 Per-Emotion Breakdown
| Emotion | Total Samples | True Positives (TP) | Recall (%) | Performance Rank |
| :--- | :---: | :---: | :---: | :--- |
| **Neutral** | 96 | 82 | **85.42%** | ⭐⭐⭐⭐ (Strong) |
| **Angry** | 192 | 146 | **76.04%** | ⭐⭐⭐ (Strong) |
| **Disgust** | 192 | 98 | **51.04%** | ⭐⭐ (Moderate) |
| **Sad** | 192 | 37 | **19.27%** | ⭐ (Weak) |
| **Happy** | 192 | 3 | **1.56%** | ❌ (Critical) |
| **Calm** | 192 | 0 | **0.00%** | ❌ (Failure) |
| **Fearful** | 192 | 0 | **0.00%** | ❌ (Failure) |
| **Surprised** | 192 | 0 | **0.00%** | ❌ (Failure) |

## 3. Root Cause Analysis (Qualitative)

> [!IMPORTANT]
> **Centroid Mismatch**: The 0% accuracy on three key categories suggests that the production centroids (defined in `emotion_model.py`) do not align with the actual embedding distribution produced by this specific fine-tuned Wav2Vec2 layer for the RAVDESS dataset.

### Observed Bottlenecks:
1.  **Arousal Dominance**: Emotions with similar energy (Happy vs. Angry) are indistinguishable to the current metric. High Arousal almost always defaults to `Angry`.
2.  **Valence Compression**: The model fails to capture the "Positivity" (Valence) of `Happy` and `Surprised` samples.
3.  **Preprocessing Void**: Raw audio input without normalization leads to jittery VAD predictions, affecting the final dominant label calculation.

## 4. Technical Metadata
- **Backend Code**: [app/misc/benchmark/evaluate_baseline.py](file:///c:/Users/nguye/Desktop/VoxDynamics/app/misc/benchmark/evaluate_baseline.py)
- **Raw Results**: [app/misc/benchmark/baseline_results.json](file:///c:/Users/nguye/Desktop/VoxDynamics/app/misc/benchmark/baseline_results.json)
- **Pipeline Config**:
    - **Model**: `Wav2Vec2-L-Robust-Emotion-Liem-MSP-Podcast`
    - **Step**: 0.5s Sliding Window
    - **Smoothing**: EMA (Alpha=0.5)

---
*Last Updated: March 2026*

---
*Note: This report will be updated with quantitative data once the first benchmark run is complete.*
