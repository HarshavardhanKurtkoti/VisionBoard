import os
import sys
from typing import List, Dict, Any, Optional
import torch
from ultralytics import YOLO

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelTrainerConfig
from visionboard.utils.ml_utils.metric.classification_metric import DetectionMetrics

class YOLOModel:
    """
    Wrapper class for YOLOv8 model operations
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLOv8 model
        Args:
            model_path: Path to model weights, if None uses default YOLOv8n
        """
        try:
            logging.info(f"Initializing YOLOv8 model from {model_path if model_path else 'pretrained weights'}")
            self.model = YOLO(model_path) if model_path else YOLO('yolov8n.pt')
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logging.info(f"Model initialized on device: {self.device}")
            
        except Exception as e:
            logging.error(f"Error initializing YOLOv8 model: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def train(
        self,
        config: ModelTrainerConfig,
        train_data: str,
        val_data: Optional[str] = None
    ) -> str:
        """
        Train the YOLOv8 model
        Args:
            config: Training configuration
            train_data: Path to training data YAML
            val_data: Path to validation data YAML
        Returns:
            str: Path to best model weights
        """
        try:
            logging.info("Starting model training")
            
            # Set training arguments
            args = {
                "data": train_data,
                "epochs": config.epochs,
                "imgsz": config.img_size,
                "batch": config.batch_size,
                "device": config.device,
                "save": True,
                "save_period": -1,  # Save only best and last
                "cache": True,
                "exist_ok": True,
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "seed": 42
            }
            
            if val_data:
                args["val"] = val_data
            
            # Train model
            results = self.model.train(**args)
            best_model_path = str(results.best)
            
            logging.info(f"Training completed. Best model saved at: {best_model_path}")
            return best_model_path
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict(
        self,
        image_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an image
        Args:
            image_path: Path to image
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        Returns:
            List[Dict]: List of predictions with boxes, scores, and classes
        """
        try:
            logging.info(f"Running inference on image: {image_path}")
            
            # Run inference
            results = self.model.predict(
                image_path,
                conf=conf_thres,
                iou=iou_thres,
                device=self.device
            )
            
            # Process results
            predictions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    pred = {
                        "box": box.xyxy[0].tolist(),  # Convert to list for JSON serialization
                        "confidence": float(box.conf),
                        "class_id": int(box.cls),
                        "class_name": result.names[int(box.cls)]
                    }
                    predictions.append(pred)
            
            logging.info(f"Found {len(predictions)} objects in image")
            return predictions
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def evaluate(
        self,
        val_data: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45
    ) -> DetectionMetrics:
        """
        Evaluate model on validation dataset
        Args:
            val_data: Path to validation data YAML
            conf_thres: Confidence threshold
            iou_thres: IoU threshold
        Returns:
            DetectionMetrics: Object containing evaluation metrics
        """
        try:
            logging.info("Starting model evaluation")
            
            # Run validation
            results = self.model.val(
                data=val_data,
                conf=conf_thres,
                iou=iou_thres,
                device=self.device
            )
            
            # Extract metrics
            metrics = DetectionMetrics(
                precision=float(results.results_dict['metrics/precision']),
                recall=float(results.results_dict['metrics/recall']),
                f1_score=float(results.results_dict['metrics/F1']),
                map50=float(results.results_dict['metrics/mAP50']),
                map75=float(results.results_dict['metrics/mAP75']),
                map50_95=float(results.results_dict['metrics/mAP50-95'])
            )
            
            logging.info(f"Evaluation completed with metrics: {metrics.to_dict()}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def export(self, format: str = "onnx") -> str:
        """
        Export model to different formats
        Args:
            format: Format to export to (onnx, tflite, etc.)
        Returns:
            str: Path to exported model
        """
        try:
            logging.info(f"Exporting model to {format} format")
            
            path = self.model.export(format=format)
            logging.info(f"Model exported to: {path}")
            
            return str(path)
            
        except Exception as e:
            logging.error(f"Error exporting model: {str(e)}")
            raise VisionBoardException(e, sys)