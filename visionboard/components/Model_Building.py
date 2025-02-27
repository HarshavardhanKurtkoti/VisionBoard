from ultralytics import YOLO
import yaml
import os
from typing import Dict, Optional
import torch
from pathlib import Path

class ModelBuilder:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the ModelBuilder
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_data_yaml(self) -> str:
        """
        Create data.yaml file for YOLOv8 training
        Returns:
            Path to created data.yaml file
        """
        data_yaml = {
            'train': str(Path(self.config['data_dir']) / 'train'),
            'val': str(Path(self.config['data_dir']) / 'valid'),
            'test': str(Path(self.config['data_dir']) / 'test'),
            'nc': 1,
            'names': ['SignBoard']
        }
        
        yaml_path = 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
            
        return yaml_path
    
    def build_model(self, model_size: str = 'm') -> None:
        """
        Build YOLOv8 model
        Args:
            model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x')
        """
        self.model = YOLO(f'yolov8{model_size}.yaml')
    
    def load_pretrained(self, weights_path: Optional[str] = None) -> None:
        """
        Load pretrained weights
        Args:
            weights_path: Path to weights file
        """
        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
        else:
            print("No pretrained weights found, using default initialization")
    
    def train(self, 
             epochs: int = 50,
             batch_size: int = 32,
             image_size: int = 640,
             save_period: int = 1) -> None:
        """
        Train the model
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            image_size: Input image size
            save_period: Save checkpoint every N epochs
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first")
            
        data_yaml = self.create_data_yaml()
        
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            save=True,
            save_period=save_period,
            pretrained=True,
            optimizer="Adam",
            lr0=0.0001,
            lrf=0.01,
            device=self.device,
            val=True,
            verbose=True
        )
    
    def export_model(self, format: str = 'onnx') -> None:
        """
        Export the trained model
        Args:
            format: Export format ('onnx', 'torchscript', etc.)
        """
        if self.model is None:
            raise ValueError("No model to export")
            
        self.model.export(format=format)

if __name__ == "__main__":
    # Example usage
    builder = ModelBuilder()
    builder.build_model()
    builder.train()