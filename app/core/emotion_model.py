# ============================================================
# VoxDynamics — Wav2Vec2 Emotion Model
# ============================================================
"""
Speech Emotion Recognition using audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.

The model uses a custom RegressionHead on top of Wav2Vec2 and outputs
three dimensional values:
  - Arousal  (activation / energy level)
  - Dominance (control / power)
  - Valence  (positivity / negativity)

Values are in a ~0-1 range. We map these dimensions to discrete
emotion labels using geometric proximity to emotion centroids.

Reference: https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
"""

import time
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from typing import Dict, Tuple, Optional


# ── Custom Model Classes (from HuggingFace model card) ───────

class RegressionHead(nn.Module):
    """Regression head for dimensional emotion prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Speech emotion classifier using Wav2Vec2 backbone."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# ── Emotion Centroids in (Arousal, Dominance, Valence) space ─────
# Normal conversational speech typically outputs A:~0.5, D:~0.5, V:~0.45 with this specific model.
# These centroids are re-calibrated so that standard speech does not default to 'disgust'.
EMOTION_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
    "happy":    (0.70, 0.60, 0.75),
    "angry":    (0.75, 0.75, 0.25),
    "sad":      (0.30, 0.30, 0.30),
    "neutral":  (0.50, 0.50, 0.45),
    "fear":     (0.65, 0.35, 0.30),
    "surprise": (0.70, 0.55, 0.65),
    "disgust":  (0.55, 0.65, 0.25),
    "calm":     (0.35, 0.40, 0.60),
}

EMOTION_EMOJI: Dict[str, str] = {
    "happy":    "😊",
    "angry":    "😠",
    "sad":      "😢",
    "neutral":  "😐",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "calm":     "😌",
}

EMOTION_COLORS: Dict[str, str] = {
    "happy":    "#FFD700",
    "angry":    "#FF4444",
    "sad":      "#4169E1",
    "neutral":  "#808080",
    "fear":     "#9B59B6",
    "surprise": "#FF8C00",
    "disgust":  "#2ECC71",
    "calm":     "#00CED1",
}


class EmotionPredictor:
    """
    High-level wrapper for the audeering Wav2Vec2 emotion model.

    Usage:
        predictor = EmotionPredictor()
        predictor.load()
        result = predictor.predict(waveform_np_array)
    """

    MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def __init__(self):
        self._processor: Optional[Wav2Vec2Processor] = None
        self._model: Optional[EmotionModel] = None
        self._device = "cpu"

    def load(self) -> None:
        """Load model and processor from HuggingFace Hub."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
        self._model = EmotionModel.from_pretrained(self.MODEL_NAME).to(self._device)
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _process_audio(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Run inference on raw audio, returning [arousal, dominance, valence].

        Args:
            waveform: 1-D or 2-D float32 numpy array.
            sample_rate: Sampling rate (16 kHz expected).

        Returns:
            numpy array of shape (1, 3) with [arousal, dominance, valence].
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Process through Wav2Vec2 processor
        y = self._processor(waveform, sampling_rate=sample_rate)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self._device)

        with torch.no_grad():
            # index 1 = logits (regression output)
            y = self._model(y)[1]

        return y.detach().cpu().numpy()

    def predict(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> Dict:
        """
        Full prediction: dimensions + discrete label + confidence.

        Returns:
            dict with: emotion_label, arousal, dominance, valence,
                       confidence, emoji, color, latency_ms
        """
        t0 = time.perf_counter()

        result = self._process_audio(waveform, sample_rate)
        # result shape: (1, 3) → [arousal, dominance, valence]
        arousal = float(np.clip(result[0][0], 0, 1))
        dominance = float(np.clip(result[0][1], 0, 1))
        valence = float(np.clip(result[0][2], 0, 1))

        label, confidence = self._map_to_label(arousal, dominance, valence)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "emotion_label": label,
            "arousal": arousal,
            "dominance": dominance,
            "valence": valence,
            "confidence": confidence,
            "emoji": EMOTION_EMOJI.get(label, "❓"),
            "color": EMOTION_COLORS.get(label, "#808080"),
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _map_to_label(
        arousal: float, dominance: float, valence: float
    ) -> Tuple[str, float]:
        """
        Map (A, D, V) coordinates to the nearest emotion centroid.

        Returns:
            (emotion_label, confidence) where confidence = 1 / (1 + distance).
        """
        point = np.array([arousal, dominance, valence])
        best_label = "neutral"
        best_dist = float("inf")

        for label, centroid in EMOTION_CENTROIDS.items():
            dist = np.linalg.norm(point - np.array(centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label

        confidence = 1.0 / (1.0 + best_dist)
        return best_label, float(confidence)
