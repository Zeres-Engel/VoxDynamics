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
        Handles arbitrary chunk sizes by dividing into valid VAD frames.

        Args:
            audio_chunk: 1-D float32 numpy array.
            sample_rate: 16000.

        Returns:
            (is_speech, max_confidence) tuple.
        """
        if not self.is_loaded:
            raise RuntimeError("VAD model not loaded. Call load() first.")

        # Silero VAD expects 512, 1024, or 1536 samples for 16kHz
        frame_size = 512 if sample_rate == 16000 else 256
        
        # Ensure float32 tensor
        if isinstance(audio_chunk, np.ndarray):
            tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        else:
            tensor = torch.as_tensor(audio_chunk, dtype=torch.float32)

        if tensor.dim() > 1:
            tensor = tensor.flatten()

        # If chunk is too small, pad it
        if len(tensor) < frame_size:
            pad = torch.zeros(frame_size - len(tensor))
            tensor = torch.cat([tensor, pad])

        # If chunk is large, we process in frames and return max confidence
        max_confidence = 0.0
        
        with torch.no_grad():
            for i in range(0, len(tensor) - frame_size + 1, frame_size):
                frame = tensor[i : i + frame_size]
                conf = self._model(frame, sample_rate).item()
                max_confidence = max(max_confidence, conf)

        is_speech = max_confidence >= self.threshold
        return is_speech, max_confidence
