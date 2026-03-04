
import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPool1D, Dropout, Flatten, Dense
import tensorflow.keras.layers as L
import librosa
from typing import Dict, Tuple, Optional

# Constants for UI consistency
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

class CNNEmotionPredictor:
    """
    CNN-based Speech Emotion Recognition using 2376 sequential features.
    
    Architecture: Deep 1D-CNN with BatchNormalization.
    Target: 97%+ Accuracy methodology.
    """

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self._device = "/CPU:0"
        
    def load(self) -> None:
        """Load weights and metadata."""
        # 1. Device check
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self._device = "/GPU:0"
            
        # 2. Load Scaler
        scaler_path = os.path.join(self.model_dir, 'scaler2.pickle')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # 3. Build Model
        self.model = self._build_architecture()
        
        # 4. Load Weights
        weights_path = os.path.join(self.model_dir, 'best_model1_weights.h5')
        self.model.load_weights(weights_path)
        print(f"[CNN] Weights loaded into {self._device} architecture.")

    def _build_architecture(self) -> Sequential:
        input_shape = (2376, 1)
        model = Sequential([
            L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding='same'),
            
            L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding='same'),
            Dropout(0.2),
            
            L.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding='same'),
            
            L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding='same'),
            Dropout(0.2),
            
            L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=3, strides=2, padding='same'),
            Dropout(0.2),
            
            L.Flatten(),
            L.Dense(512, activation='relu'),
            L.BatchNormalization(),
            L.Dense(len(self.labels), activation='softmax')
        ])
        return model

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _extract_features(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Sequential feature extraction matching the CNN training logic."""
        # 1. Resample to 22050 (Target for 97%+ models)
        target_sr = 22050
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            
        # 2. Fix length to exactly 2.5s (55125 samples at 22050 sr)
        target_len = 55125
        if len(waveform) < target_len:
            # Pad with zeros at the end to align speech to the left (matching benchmark offset logic)
            waveform = librosa.util.fix_length(waveform, size=target_len)
        else:
            # For long segments, take the first 2.5s
            waveform = waveform[:target_len]

        # 4. Feature Extraction
        # Frame params: 2048 window, 512 hop => ~108 frames for 55125 samples
        fl = 2048
        hl = 512
        
        # Zero Crossing Rate (108 frames)
        zcr = np.squeeze(librosa.feature.zero_crossing_rate(waveform, frame_length=fl, hop_length=hl))
        
        # RMS Energy (108 frames)
        rms = np.squeeze(librosa.feature.rms(y=waveform, frame_length=fl, hop_length=hl))
        
        # MFCC (20 coefficients * 108 frames = 2160 values)
        mfcc = librosa.feature.mfcc(y=waveform, sr=target_sr, n_mfcc=20, n_fft=fl, hop_length=hl)
        mfcc_flat = np.ravel(mfcc.T) # (Frames, Coeffs) -> Flatten
        
        # 5. Concatenate: [108] + [108] + [2160] = 2376
        feat = np.hstack((zcr, rms, mfcc_flat))
        
        # Safety slice/pad to 2376
        if len(feat) != 2376:
            if len(feat) < 2376:
                feat = np.pad(feat, (0, 2376 - len(feat)), 'constant')
            else:
                feat = feat[:2376]
            
        return feat

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Run full high-accuracy CNN inference."""
        t0 = time.perf_counter()
        
        # 1. Extract 2376 features
        feat = self._extract_features(waveform, sample_rate)
        
        # 2. Scale
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        
        # 3. Reshape for CNN (Samples, Length, Channels)
        feat_input = np.expand_dims(feat_scaled, axis=2)
        
        # 4. Infer
        with tf.device(self._device):
            probs = self.model.predict(feat_input, verbose=0)[0]
            
        idx = np.argmax(probs)
        label = self.labels[idx]
        confidence = float(probs[idx])
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        # Map label names (e.g. 'fear' -> 'fearful' for DB/UI consistency)
        ui_label = label
        if label == 'fear': ui_label = 'fearful'
        if label == 'surprise': ui_label = 'surprised'

        # Full scores for smoothing/charting
        scores = {}
        for i, l in enumerate(self.labels):
            l_name = l
            if l == 'fear': l_name = 'fearful'
            if l == 'surprise': l_name = 'surprised'
            scores[l_name] = float(probs[i])

        return {
            "emotion_label": ui_label,
            "arousal": scores.get('happy', 0.5) - scores.get('sad', 0.1), # Heuristic mapping
            "dominance": scores.get('angry', 0.5), 
            "valence": scores.get('happy', 0.5) - scores.get('angry', 0.2),
            "confidence": confidence,
            "emoji": EMOTION_EMOJI.get(ui_label, "❓"),
            "color": EMOTION_COLORS.get(ui_label, "#808080"),
            "latency_ms": latency_ms,
            "engine": "CNN-Deep",
            "scores": scores
        }
