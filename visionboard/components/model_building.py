import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional, Any

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging

class ModelBuilder:
    """
    Builder utility for creating, initializing, and exporting YOLOv8 models
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelBuilder
        Args:
            config_path: Path to configuration file
        """
        try:
            self.config = self._load_config(config_path) if config_path else {}
            self.model = None
            self.device = 'cpu'
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                pass
        except Exception as e:
            logging.error(f"Error initializing ModelBuilder: {str(e)}")
            raise VisionBoardException(e, sys)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def build_model(self, model_size: str = 'n') -> Any:
        """
        Build YOLOv8 model
        Args:
            model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x')
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'yolov8{model_size}.pt')
            return self.model
        except ImportError:
            logging.warning("ultralytics not installed. ModelBuilder initialized in mock mode.")
            return None
    
    def load_pretrained(self, weights_path: Optional[str] = None) -> Any:
        """
        Load pretrained weights
        """
        try:
            from ultralytics import YOLO
            path = weights_path if weights_path and os.path.exists(weights_path) else 'yolov8n.pt'
            self.model = YOLO(path)
            return self.model
        except ImportError:
            logging.warning("ultralytics not installed.")
            return None
    
    def export_model(self, format: str = 'onnx') -> Optional[str]:
        """
        Export the trained model
        """
        if self.model is None:
            raise ValueError("No model loaded to export")
        return str(self.model.export(format=format))