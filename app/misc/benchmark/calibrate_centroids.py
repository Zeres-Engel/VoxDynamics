import os
import sys
import json
import torch
import numpy as np
import librosa
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.core.processor import AudioProcessor
from app.core.emotion_model import EMOTION_EMOJI

class CentroidCalibrator:
    def __init__(self):
        print("[*] Initializing Emotion Engine for Calibration...")
        self.processor = AudioProcessor()
        self.processor.load_models()
        self.raw_data = defaultdict(list) # emotion -> list of [V, A, D]

    def collect_coordinates(self, data_dir: str):
        data_path = Path(data_dir)
        folders = [f for f in data_path.iterdir() if f.is_dir()]
        
        for folder in folders:
            label = folder.name.lower()
            audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))
            print(f"[*] Extracting VAD coordinates for: {label.upper()} ({len(audio_files)} files)")
            
            for audio_file in audio_files:
                try:
                    waveform, sr = librosa.load(str(audio_file), sr=16000)
                    
                    # We bypass the classifier and get raw dimensional outputs from the predictor
                    # We use process_file to get segment-level dims, then average them for the file
                    # Or we can just get the raw dims from the predictor directly
                    
                    # Simple approach: use the processor's process_file but look at the raw dims
                    segments = self.processor.process_file(waveform, sample_rate=sr)
                    speech_segs = [s for s in segments if s.get("is_speech")]
                    
                    if speech_segs:
                        v_vals = [s["valence"] for s in speech_segs]
                        a_vals = [s["arousal"] for s in speech_segs]
                        d_vals = [s["dominance"] for s in speech_segs]
                        
                        self.raw_data[label].append([
                            float(np.mean(v_vals)),
                            float(np.mean(a_vals)),
                            float(np.mean(d_vals))
                        ])
                except Exception as e:
                    print(f"    [!] Skip {audio_file.name}: {e}")

    def compute_centroids(self):
        calibrated = {}
        print("\n[+] Calibration Results:")
        print(f"{'Emotion':<12} | {'Valence':<8} | {'Arousal':<8} | {'Dominance':<8} | {'Count':<6}")
        print("-" * 55)
        
        for emotion, coords in self.raw_data.items():
            arr = np.array(coords)
            mean_v = np.mean(arr[:, 0])
            mean_a = np.mean(arr[:, 1])
            mean_d = np.mean(arr[:, 2])
            
            calibrated[emotion] = {
                "v": round(float(mean_v), 4),
                "a": round(float(mean_a), 4),
                "d": round(float(mean_d), 4)
            }
            print(f"{emotion:<12} | {mean_v:<8.4f} | {mean_a:<8.4f} | {mean_d:<8.4f} | {len(coords):<6}")
            
        return calibrated

    def save_calibration(self, calibrated_data: dict, output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(calibrated_data, f, indent=4)
        print(f"\n[+] Calibrated centroids saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate_centroids.py <dataset_path>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    output_path = os.path.join(os.path.dirname(__file__), "calibrated_centroids.json")
    
    calibrator = CentroidCalibrator()
    calibrator.collect_coordinates(dataset_path)
    new_centroids = calibrator.compute_centroids()
    calibrator.save_calibration(new_centroids, output_path)
