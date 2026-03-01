# ============================================================
# VoxDynamics — Silero VAD Integration
# ============================================================
"""Voice Activity Detection using Silero VAD model."""

import torch
import numpy as np
from typing import Tuple


class VoiceActivityDetector:
    """Wraps Silero VAD for real-time speech detection."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._utils = None

    def load(self) -> None:
        """Load the Silero VAD model from torch.hub."""
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self._model = model
        self._utils = utils

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def reset_state(self) -> None:
        """Reset the model's internal hidden states."""
        if self._model is not None:
            self._model.reset_states()

    def detect_speech(
        self, audio_chunk: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[bool, float]:
        """
        Check whether the audio chunk contains speech.

        Args:
            audio_chunk: 1-D float32 numpy array (values in [-1, 1]).
            sample_rate: Must be 8000 or 16000.

        Returns:
            (is_speech, confidence) tuple.
        """
        if not self.is_loaded:
            raise RuntimeError("VAD model not loaded. Call load() first.")

        # Ensure float32 tensor
        if isinstance(audio_chunk, np.ndarray):
            tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        else:
            tensor = torch.as_tensor(audio_chunk, dtype=torch.float32)

        # Silero VAD expects 1-D tensor
        if tensor.dim() > 1:
            tensor = tensor.squeeze()

        with torch.no_grad():
            confidence = self._model(tensor, sample_rate).item()

        is_speech = confidence >= self.threshold
        return is_speech, confidence
