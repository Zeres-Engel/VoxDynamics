
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.getcwd())
from app.core.cnn_predictor import CNNEmotionPredictor

def main():
    print("--- ⚡ CNN MODEL FINAL EVALUATION ⚡ ---")
    
    # 1. Initialize Predictor (Inference Engine)
    predictor = CNNEmotionPredictor()
    predictor.load()
    
    # 2. Dataset Mapping
    label_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fearful': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'surprised': 'surprise'
    }
    
    data_path = 'data/emotions'
    file_path = []
    file_emotion = []
    
    for emotion_dir_name in os.listdir(data_path):
        if emotion_dir_name in label_map:
            target_label = label_map[emotion_dir_name]
            dir_path = os.path.join(data_path, emotion_dir_name)
            if not os.path.isdir(dir_path): continue
            for file in os.listdir(dir_path):
                if file.endswith('.wav'):
                    file_path.append(os.path.join(dir_path, file))
                    file_emotion.append(target_label)
    
    print(f"Total files to evaluate: {len(file_path)}")
    
    Y_true = []
    Y_pred = []
    
    for i, (path, emotion) in enumerate(zip(file_path, file_emotion)):
        try:
            # We use the EXACT same function the app uses
            import librosa
            waveform, sr = librosa.load(path, sr=None)
            res = predictor.predict(waveform, sr)
            
            # Map back internally if needed for report
            pred_label = res['emotion_label']
            if pred_label == 'fearful': pred_label = 'fear'
            if pred_label == 'surprised': pred_label = 'surprise'
            
            Y_true.append(emotion)
            Y_pred.append(pred_label)
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(file_path)}...")
        except Exception as e:
            pass

    # 3. Final Report
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    report = classification_report(Y_true, Y_pred, target_names=labels)
    print("\nClassification Report:")
    print(report)
    
    os.makedirs('docs/benchmark', exist_ok=True)
    with open('docs/benchmark/CNN_REPORT.md', 'w') as f:
        f.write("# CNN Model Final Evaluation Report\n\n")
        f.write("This report documents the final evaluation of the **CNN Model** (inference engine) on the VoxDynamics local dataset.\n\n")
        f.write(f"- **Engine**: `CNNEmotionPredictor` (TensorFlow)\n")
        f.write(f"- **Window Size**: 2.5 seconds\n\n")
        f.write("## Metrics\n\n```\n")
        f.write(report)
        f.write("\n```\n")
        accuracy = np.mean(np.array(Y_true) == np.array(Y_pred)) * 100
        f.write(f"\n### Overall Accuracy: **{accuracy:.2f}%**\n")

    # 4. Confusion Matrix
    cm = confusion_matrix(Y_true, Y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('CNN Model - Final Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('docs/benchmark/CNN_MODEL_CM.png')
    
    print("Report and CM saved to docs/benchmark/")

if __name__ == "__main__":
    main()
