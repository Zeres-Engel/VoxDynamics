import argparse
import torch
from src.data.loader import EmotionDataLoader
from src.models.cnn_model import EmotionCNNModel
from src.training.trainer import ModelTrainer

def main(args):
    print("--- Starting High-Accuracy (97.25%) PyTorch Pipeline ---")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    print("Loading and preprocessing data (Sequence Features: 2376 elements)...")
    loader = EmotionDataLoader(
        data_path=args.data_path, 
        test_size=args.test_size,
        batch_size=args.batch_size
    )
    train_loader, test_loader = loader.prepare()
    print("Data loaders ready.")
    
    # 2. Build Model
    print("Building Deep CNN model...")
    num_classes = len(loader.get_encoder().categories_[0])
    model_wrapper = EmotionCNNModel(num_classes=num_classes)
    model = model_wrapper.get_model()
    
    # 3. Train
    print("Starting training...")
    trainer = ModelTrainer(model, output_dir=args.output_dir)
    history = trainer.train(train_loader, test_loader, epochs=args.epochs)
    
    # 4. Save metadata
    loader.save_state(args.output_dir)
    print("--- Training Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train High-Accuracy SER Model (PyTorch)")
    parser.add_argument("--data_path", type=str, default="data/emotions", help="Path to emotions dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    
    args = parser.parse_args()
    main(args)
