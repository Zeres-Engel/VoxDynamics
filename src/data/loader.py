import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.features.extractor import get_features

class EmotionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # PyTorch Conv1D expects (Channels, Length)
        # So we return (1, 2376)
        return self.X[idx].unsqueeze(0), self.Y[idx]

class EmotionDataLoader:
    def __init__(self, data_path='data/emotions', test_size=0.2, batch_size=64):
        self.data_path = data_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

    def load_metadata(self):
        """Scans the data directory and creates a DataFrame of file paths and emotions."""
        file_emotion = []
        file_path = []
        
        for emotion in os.listdir(self.data_path):
            emotion_dir = os.path.join(self.data_path, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for file in os.listdir(emotion_dir):
                if file.endswith('.wav'):
                    file_path.append(os.path.join(emotion_dir, file))
                    file_emotion.append(emotion)
                    
        return pd.DataFrame({'Path': file_path, 'Emotions': file_emotion})

    def _extract_features_worker(self, df):
        """Internal method to extract features from a DataFrame of paths."""
        X, Y = [], []
        print(f"Extracting features for {len(df)} files...")
        for i, (path, emotion) in enumerate(zip(df.Path, df.Emotions)):
            try:
                # get_features returns 4 variants (Augmentation 4x)
                features = get_features(path)
                for variant in features:
                    X.append(variant)
                    Y.append(emotion)
                if (i+1) % 100 == 0:
                    print(f"Processed {i+1}/{len(df)} files...")
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)
        return X, Y

    def prepare(self):
        """Prepares PyTorch DataLoaders."""
        df = self.load_metadata()
        X_raw, Y_raw = self._extract_features_worker(df)
        
        # Fit and transform labels to class indices for CrossEntropyLoss
        self.encoder.fit(Y_raw)
        Y_indices = np.argmax(self.encoder.transform(Y_raw), axis=1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Split
        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, Y_indices, test_size=self.test_size, random_state=42, shuffle=True
        )
        
        train_ds = EmotionDataset(x_train, y_train)
        test_ds = EmotionDataset(x_test, y_test)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def save_state(self, model_dir='models'):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'encoder.pkl'), 'wb') as f:
            pickle.dump(self.encoder, f)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

    def get_encoder(self):
        return self.encoder
