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
import librosa
from collections import deque
from typing import Dict, Optional

from app.core.vad import VoiceActivityDetector
from app.core.vad import VoiceActivityDetector
from app.core.cnn_predictor import CNNEmotionPredictor, EMOTION_EMOJI, EMOTION_COLORS
from app.config import settings


class AudioProcessor:
    """
    Real-time audio processing pipeline using the high-accuracy CNN engine.

    Flow:
        1. Receive raw audio chunk (PCM float32, 16 kHz)
        2. Append to ring buffer (sliding window = 2.5s for the CNN)
        3. Run VAD on latest chunk
        4. If speech detected → run high-accuracy CNN inference on full buffer
        5. Return structured result
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration_s: float = 2.5,   # Fixed to match CNN input requirement
        ema_alpha: float = 0.4,           # 0.4 for smooth transitions between windows
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
        self._emotion = CNNEmotionPredictor()

        # EMA state for all scores
        self._ema_scores: Dict[str, float] = {}

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
        self._ema_scores = {}
        self._last_result = None
        self.session_id = str(uuid.uuid4())[:8]

    def process_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """
        Process a single audio chunk and return emotion prediction.

        Args:
            audio_chunk: 1-D float32 numpy array, 16 kHz.

        Returns:
            dict: { ... }
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

            # 4. Apply EMA smoothing on probabilities
            smoothed_scores = self._apply_ema_scores(raw_result["scores"])
            
            # Determine new label from smoothed scores
            label = max(smoothed_scores, key=smoothed_scores.get)
            confidence = smoothed_scores[label]

            latency_ms = (time.perf_counter() - t0) * 1000

            self._last_result = {
                "emotion_label": label,
                "arousal": round(raw_result["arousal"], 4),
                "dominance": round(raw_result["dominance"], 4),
                "valence": round(raw_result["valence"], 4),
                "confidence": round(confidence, 4),
                "emoji": raw_result["emoji"],
                "color": raw_result["color"],
                "scores": smoothed_scores,
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
                default_scores = {l: 0.0 for l in self._emotion.labels}
                default_scores["neutral"] = 1.0
                self._last_result = {
                    "emotion_label": "neutral",
                    "arousal": 0.5,
                    "dominance": 0.5,
                    "valence": 0.5,
                    "confidence": 0.0,
                    "emoji": "😐",
                    "color": "#808080",
                    "scores": default_scores,
                    "is_speech": False,
                    "latency_ms": round(latency_ms, 2),
                    "session_id": self.session_id,
                    "buffer_seconds": round(
                        len(self._ring_buffer) / self.sample_rate, 2
                    ),
                }

        return self._last_result

    def _apply_ema_scores(self, new_scores: Dict[str, float]) -> Dict[str, float]:
        alpha = self.ema_alpha
        if not self._ema_scores:
            self._ema_scores = new_scores.copy()
        else:
            for l, val in new_scores.items():
                self._ema_scores[l] = alpha * val + (1 - alpha) * self._ema_scores.get(l, 0)
        return self._ema_scores

    def process_file(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        window_s: float = 2.5,
        hop_s: float = 0.5, # Ignored in smart mode, but kept for signature
    ) -> list:
        """
        Analyze a full audio file using Intelligent Utterance Segmentation.
        Finds islands of speech, processes each as a whole segment.
        """
        # Ensure float32 mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # 1. Global Normalization (Preserve relative intensity of segments)
        max_val = np.max(np.abs(waveform))
        if max_val > 0.001: # Avoid noise floor amplification
            waveform = waveform / max_val

        # 2. VAD Resampling path (Silero needs 16kHz)
        # We keep the high-quality 'waveform' for CNN, and 'waveform_vad' for detection
        if sample_rate != 16000:
            waveform_vad = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            vad_sr = 16000
        else:
            waveform_vad = waveform
            vad_sr = 16000

        self.reset()
        
        # 3. Detailed VAD scan (0.1s steps)
        step_s = 0.1
        step_samples = int(vad_sr * step_s)
        is_speech_states = []
        
        for i in range(0, len(waveform_vad), step_samples):
            chunk = waveform_vad[i : i + step_samples]
            if len(chunk) < vad_sr * 0.03: break
            active, _ = self._vad.detect_speech(chunk, vad_sr)
            is_speech_states.append(active)

        # 2. Cluster VAD states into speech islands
        #    - Merge gaps < 0.8s (handles brief VAD dropouts mid-utterance)
        #    - Minimum segment duration: 0.5s (filters noise/artifacts)
        raw_segments = []  # List of (start_s, end_s)
        in_speech = False
        start_time = 0
        merge_gap_frames = 8  # 0.8s — tolerates brief pauses within utterances
        silence_counter = 0

        for i, active in enumerate(is_speech_states):
            curr_time = i * step_s
            if active:
                if not in_speech:
                    start_time = curr_time
                    in_speech = True
                silence_counter = 0
            else:
                if in_speech:
                    silence_counter += 1
                    if silence_counter > merge_gap_frames:
                        # End of utterance (accounting for the merged gap)
                        end_t = curr_time - (merge_gap_frames * step_s)
                        if end_t > start_time + 0.3:
                            raw_segments.append((start_time, end_t))
                        in_speech = False

        if in_speech:
            raw_segments.append((start_time, len(is_speech_states) * step_s))

        # Post-processing: merge segments that are very close (<1.0s gap)
        # and filter out very short segments (<0.5s)
        segments = []
        for s, e in raw_segments:
            dur = e - s
            if dur < 0.5:
                # Too short — try to merge with previous segment
                if segments:
                    prev_s, prev_e = segments[-1]
                    if s - prev_e < 1.5:
                        segments[-1] = (prev_s, e)
                continue
            
            # Merge with previous if gap is small
            if segments:
                prev_s, prev_e = segments[-1]
                gap = s - prev_e
                if gap < 1.0:
                    segments[-1] = (prev_s, e)
                    continue
            
            segments.append((s, e))

        # 3. Predict for each detected segment
        results = []
        last_t = 0.0
        
        # Add a placeholder for initial silence if needed
        default_scores = {l: 0.0 for l in self._emotion.labels}; default_scores["neutral"] = 1.0

        for start_s, end_s in segments:
            # Fill silence gap before this segment
            if start_s > last_t + 0.2:
                results.append({
                    "time_s": round(last_t, 2),
                    "duration_s": round(start_s - last_t, 2),
                    "emotion_label": "silence",
                    "confidence": 0.0, "arousal": 0.5, "dominance": 0.5, "valence": 0.5,
                    "emoji": "🔇", "color": "#333333", "scores": dict(default_scores),
                    "is_speech": False
                })

            # Process Speech Island with a 200ms buffer (silence) for better context
            buffer_s = 0.2
            buff_samples = int(buffer_s * sample_rate)
            
            start_idx = max(0, int(start_s * sample_rate) - buff_samples)
            end_idx = min(len(waveform), int(end_s * sample_rate) + buff_samples)
            utterance_audio = waveform[start_idx:end_idx]
            
            # Prediction — NO EMA for file analysis (each utterance is independent)
            # Using original source signal (waveform) at original sample_rate
            raw = self._emotion.predict(utterance_audio, sample_rate)
            
            results.append({
                "time_s": round(start_s, 2),
                "duration_s": round(end_s - start_s, 2),
                "emotion_label": raw["emotion_label"],
                "arousal": round(raw["arousal"], 4),
                "dominance": round(raw["dominance"], 4),
                "valence": round(raw["valence"], 4),
                "confidence": round(raw["confidence"], 4),
                "emoji": raw["emoji"],
                "color": raw["color"],
                "scores": dict(raw["scores"]),  # Copy to avoid shared reference
                "is_speech": True
            })
            last_t = end_s

        # Final silence
        total_dur = len(waveform) / self.sample_rate
        if total_dur > last_t + 0.1:
            results.append({
                "time_s": round(last_t, 2),
                "duration_s": round(total_dur - last_t, 2),
                "emotion_label": "silence",
                "confidence": 0.0, "arousal": 0.5, "dominance": 0.5, "valence": 0.5,
                "emoji": "🔇", "color": "#333333", "scores": default_scores,
                "is_speech": False
            })

        return results
