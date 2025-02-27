from ultralytics import YOLO
from pathlib import Path
import os
import torch

def train_model():
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # load pretrained model
    
    # Get absolute path to config file
    config_path = os.path.abspath('config/data.yaml')
    
    # Train the model
    print("Starting training...")
    print(f"Using config file: {config_path}")
    
    # Set number of threads for CPU optimization
    torch.set_num_threads(os.cpu_count())
    print(f"Optimizing CPU performance with {os.cpu_count()} threads")
    
    results = model.train(
        data=config_path,
        epochs=50,     # reduced number of epochs for faster training
        imgsz=640,     # image size
        batch=8,       # reduced batch size for CPU training
        patience=10,   # reduced early stopping patience
        device='cpu',  # use CPU with optimized threading
        cache=True,    # cache images for faster training
        workers=os.cpu_count()  # maximize CPU utilization
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.best}")

if __name__ == "__main__":
    train_model()
