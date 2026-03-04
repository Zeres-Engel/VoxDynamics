import os
import sys
import time
import json
import torch
import numpy as np
import librosa
from typing import List, Dict
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.core.processor import AudioProcessor
from app.config import settings

class BenchmarkPipeline:
    def __init__(self):
        print("[*] Initializing Emotion Engine for Baseline Benchmark...")
        # Note: AudioProcessor needs explicit load_models() call
        self.processor = AudioProcessor()
        self.processor.load_models()
        print("[*] Models loaded successfully.")
        self.results = []
        
    def run(self, data_dir: str):
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"[!] Error: Data directory {data_dir} not found.")
            return
        
        folders = [f for f in data_path.iterdir() if f.is_dir()]
        print(f"[*] Found {len(folders)} emotion categories.")
        
        all_metrics = {
            "total_samples": 0,
            "correct_predictions": 0,
            "start_time": time.time(),
            "per_emotion": {}
        }

        for folder in folders:
            label = folder.name.lower()
            all_metrics["per_emotion"][label] = {"tp": 0, "total": 0}
            
            audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))
            print(f"[*] Processing category: {label.upper()} ({len(audio_files)} files)")
            
            for audio_file in audio_files:
                try:
                    # Map folder names to model labels
                    label_map = {
                        "fearful": "fear",
                        "surprised": "surprise"
                    }
                    ground_truth = label_map.get(label, label)
                    
                    # Load audio
                    waveform, sr = librosa.load(str(audio_file), sr=16000)
                    
                    # Process using the production pipeline method
                    segments = self.processor.process_file(waveform, sample_rate=sr)
                    speech_segs = [s for s in segments if s.get("is_speech")]
                    
                    if not speech_segs:
                        predicted = "neutral"  # or silent
                        avg_conf = 0.0
                    else:
                        # Logic from main.py
                        labels = [s["emotion_label"] for s in speech_segs]
                        predicted = max(set(labels), key=labels.count)
                        avg_conf = float(np.mean([s["confidence"] for s in speech_segs]))
                    
                    # Track metrics
                    all_metrics["total_samples"] += 1
                    all_metrics["per_emotion"][label]["total"] += 1
                    
                    if predicted == ground_truth:
                        all_metrics["correct_predictions"] += 1
                        all_metrics["per_emotion"][label]["tp"] += 1
                    
                    # Store record
                    self.results.append({
                        "filename": audio_file.name,
                        "ground_truth": ground_truth,
                        "predicted": predicted,
                        "confidence": avg_conf
                    })
                    
                except Exception as e:
                    print(f"    [!] Failed to process {audio_file.name}: {str(e)}")

        all_metrics["end_time"] = time.time()
        all_metrics["total_duration"] = all_metrics["end_time"] - all_metrics["start_time"]
        all_metrics["overall_accuracy"] = (all_metrics["correct_predictions"] / all_metrics["total_samples"]) if all_metrics["total_samples"] > 0 else 0
        
        return all_metrics

    def save_report(self, metrics: Dict, output_file: str):
        report = {
            "benchmark_type": "Baseline (Current Production)",
            "metrics": metrics,
            "details": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n[+] Benchmark Complete!")
        print(f"    Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"    Total Samples: {metrics['total_samples']}")
        print(f"    Report saved to: {output_file}")

if __name__ == "__main__":
    # Usage: python evaluate_baseline.py <dataset_path>
    if len(sys.argv) < 2:
        print("Usage: python evaluate_baseline.py <dataset_path>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    output_path = os.path.join(os.path.dirname(__file__), "calibrated_results.json")
    
    pipeline = BenchmarkPipeline()
    metrics = pipeline.run(dataset_path)
    if metrics:
        pipeline.save_report(metrics, output_path)
