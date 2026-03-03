# ============================================================
# VoxDynamics — Audio Processor (Core Pipeline)
# ============================================================
"""
Central audio processing pipeline:
  Audio chunk → Ring Buffer → VAD → Wav2Vec2 → EMA Smoothing → Result

Implements a sliding window approach with exponential moving average
for smooth, non-jittery emotion tracking.
"""

import time
import uuid
import numpy as np
from collections import deque
from typing import Dict, Optional

from app.core.vad import VoiceActivityDetector
from app.core.emotion_model import EmotionPredictor, EMOTION_EMOJI, EMOTION_COLORS
from app.config import settings


class AudioProcessor:
    """
    Real-time audio processing pipeline with ring buffer and EMA smoothing.

    Flow:
        1. Receive raw audio chunk (PCM float32, 16 kHz)
        2. Append to ring buffer (sliding window ≈ 3 seconds)
        3. Run VAD on latest chunk
        4. If speech detected → run emotion model on full buffer window
        5. Apply EMA smoothing to dimensional outputs
        6. Map smoothed dimensions to discrete label
        7. Return structured result
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration_s: float = 3.0,
        ema_alpha: float = 0.5,   # was 0.3 — faster response to actual speech
        vad_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.buffer_duration_s = buffer_duration_s
        self.ema_alpha = ema_alpha

        # Ring buffer — stores raw samples (capacity = sr * duration)
        self._buffer_max_samples = int(sample_rate * buffer_duration_s)
        self._ring_buffer: deque = deque(maxlen=self._buffer_max_samples)

        # Models
        self._vad = VoiceActivityDetector(threshold=vad_threshold)
        self._emotion = EmotionPredictor()

        # EMA state
        self._ema_arousal: Optional[float] = None
        self._ema_dominance: Optional[float] = None
        self._ema_valence: Optional[float] = None

        # Session
        self.session_id = str(uuid.uuid4())[:8]

        # Latest result cache
        self._last_result: Optional[Dict] = None

    def load_models(self) -> None:
        """Load VAD and emotion models. Call once at startup."""
        self._vad.load()
        self._emotion.load()

    @property
    def models_loaded(self) -> bool:
        return self._vad.is_loaded and self._emotion.is_loaded

    def reset(self) -> None:
        """Clear buffer and EMA state for a new session."""
        self._ring_buffer.clear()
        self._vad.reset_state()
        self._ema_arousal = None
        self._ema_dominance = None
        self._ema_valence = None
        self._last_result = None
        self.session_id = str(uuid.uuid4())[:8]

    def process_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """
        Process a single audio chunk and return emotion prediction.

        Args:
            audio_chunk: 1-D float32 numpy array, 16 kHz.

        Returns:
            dict: {
                emotion_label, arousal, dominance, valence,
                confidence, emoji, color, is_speech,
                latency_ms, session_id, buffer_seconds
            }
        """
        t0 = time.perf_counter()

        # Ensure float32, mono
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=-1)

        # Normalize to [-1, 1] if needed
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        # 1. Append to ring buffer
        self._ring_buffer.extend(audio_chunk.tolist())

        # 2. Run VAD on the latest chunk
        is_speech, vad_confidence = self._vad.detect_speech(
            audio_chunk, self.sample_rate
        )

        # 3. If speech → run emotion inference on buffered window
        if is_speech and len(self._ring_buffer) >= self.sample_rate * 0.5:
            buffer_array = np.array(self._ring_buffer, dtype=np.float32)
            raw_result = self._emotion.predict(buffer_array, self.sample_rate)

            # 4. Apply EMA smoothing
            smoothed = self._apply_ema(
                raw_result["arousal"],
                raw_result["dominance"],
                raw_result["valence"],
            )

            # 5. Re-map smoothed dimensions to label
            label, confidence = EmotionPredictor._map_to_label(
                smoothed["arousal"], smoothed["dominance"], smoothed["valence"]
            )

            latency_ms = (time.perf_counter() - t0) * 1000

            self._last_result = {
                "emotion_label": label,
                "arousal": round(smoothed["arousal"], 4),
                "dominance": round(smoothed["dominance"], 4),
                "valence": round(smoothed["valence"], 4),
                "confidence": round(confidence, 4),
                "emoji": EMOTION_EMOJI.get(label, "❓"),
                "color": EMOTION_COLORS.get(label, "#808080"),
                "is_speech": True,
                "latency_ms": round(latency_ms, 2),
                "session_id": self.session_id,
                "buffer_seconds": round(
                    len(self._ring_buffer) / self.sample_rate, 2
                ),
            }
        else:
            # No speech — return last known state or neutral
            latency_ms = (time.perf_counter() - t0) * 1000
            if self._last_result:
                self._last_result = {
                    **self._last_result,
                    "is_speech": False,
                    "latency_ms": round(latency_ms, 2),
                    "buffer_seconds": round(
                        len(self._ring_buffer) / self.sample_rate, 2
                    ),
                }
            else:
                self._last_result = {
                    "emotion_label": "neutral",
                    "arousal": 0.45,
                    "dominance": 0.50,
                    "valence": 0.50,
                    "confidence": 0.0,
                    "emoji": "😐",
                    "color": "#808080",
                    "is_speech": False,
                    "latency_ms": round(latency_ms, 2),
                    "session_id": self.session_id,
                    "buffer_seconds": round(
                        len(self._ring_buffer) / self.sample_rate, 2
                    ),
                }

        return self._last_result

    def _apply_ema(
        self, arousal: float, dominance: float, valence: float
    ) -> Dict[str, float]:
        """
        Exponential Moving Average smoothing.
        EMA_t = α * x_t + (1 - α) * EMA_{t-1}
        """
        α = self.ema_alpha

        if self._ema_arousal is None:
            # First observation — initialize
            self._ema_arousal = arousal
            self._ema_dominance = dominance
            self._ema_valence = valence
        else:
            self._ema_arousal = α * arousal + (1 - α) * self._ema_arousal
            self._ema_dominance = α * dominance + (1 - α) * self._ema_dominance
            self._ema_valence = α * valence + (1 - α) * self._ema_valence

        return {
            "arousal": self._ema_arousal,
            "dominance": self._ema_dominance,
            "valence": self._ema_valence,
        }

    def process_file(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        window_s: float = 2.0,
        hop_s: float = 0.5,
    ) -> list:
        """
        Analyze a full audio file by sliding a window across it.

        Args:
            waveform: 1-D float32 numpy array (entire file).
            sample_rate: Input sample rate (will resample to 16 kHz).
            window_s: Window size in seconds for each analysis frame.
            hop_s: Hop (step) size in seconds between frames.

        Returns:
            List of dicts, one per frame:
            [{ time_s, emotion_label, arousal, dominance, valence,
               confidence, emoji, color, is_speech }, ...]
        """
        # Ensure float32 mono
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        # Normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # Resample to 16 kHz if needed
        if sample_rate != self.sample_rate:
            ratio = sample_rate / self.sample_rate
            indices = np.arange(0, len(waveform), ratio).astype(int)
            indices = indices[indices < len(waveform)]
            waveform = waveform[indices]

        window_samples = int(self.sample_rate * window_s)
        hop_samples = int(self.sample_rate * hop_s)
        total_samples = len(waveform)

        results = []

        for start_idx in range(0, total_samples - window_samples + 1, hop_samples):
            chunk = waveform[start_idx : start_idx + window_samples]
            time_s = round(start_idx / self.sample_rate, 2)

            # VAD check
            is_speech, _ = self._vad.detect_speech(chunk, self.sample_rate)

            if is_speech:
                raw = self._emotion.predict(chunk, self.sample_rate)
                results.append({
                    "time_s": time_s,
                    "emotion_label": raw["emotion_label"],
                    "arousal": round(raw["arousal"], 4),
                    "dominance": round(raw["dominance"], 4),
                    "valence": round(raw["valence"], 4),
                    "confidence": round(raw["confidence"], 4),
                    "emoji": raw["emoji"],
                    "color": raw["color"],
                    "is_speech": True,
                })
            else:
                results.append({
                    "time_s": time_s,
                    "emotion_label": "silence",
                    "arousal": 0.0,
                    "dominance": 0.0,
                    "valence": 0.0,
                    "confidence": 0.0,
                    "emoji": "🔇",
                    "color": "#333333",
                    "is_speech": False,
                })

        return results
