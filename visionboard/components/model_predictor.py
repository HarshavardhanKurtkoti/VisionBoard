import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelPredictorConfig
from visionboard.entity.artifact_entity import ModelPredictorArtifact
from visionboard.utils.ml_utils.model.estimator import YOLOModel
from visionboard.ocr.text_recognition import SignboardTextReader
from visionboard.utils.main_utils.utils import create_directories
from visionboard.utils.main_utils.image_utils import read_image, save_image, draw_box_and_label

class ModelPredictor:
    """
    Component for handling model inference, bounding box visualization, and OCR text extraction
    """
    
    def __init__(self, config: Optional[ModelPredictorConfig] = None):
        """
        Initialize ModelPredictor
        Args:
            config: Configuration for model prediction
        """
        try:
            logging.info(f"{'='*20}Model Predictor component started.{'='*20}")
            self.config = config or ModelPredictorConfig()
            self.model = YOLOModel(self.config.model_path)
            self.ocr_reader = SignboardTextReader() if self.config.enable_ocr else None
            
        except Exception as e:
            logging.error(f"Error in ModelPredictor.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and OCR text on image
        """
        try:
            vis_image = image.copy()
            
            for pred in predictions:
                box = pred['box']
                conf = pred.get('confidence', 0.0)
                class_name = pred.get('class_name', 'signboard')
                text = pred.get('text', '')
                
                label = f"{class_name} {conf:.2f}"
                if text:
                    label += f" | {text[:20]}"
                
                vis_image = draw_box_and_label(vis_image, box, label=label, color=(0, 255, 0))
            
            if save_path:
                save_image(save_path, vis_image)
                logging.info(f"Saved visualization to: {save_path}")
                
            return vis_image
            
        except Exception as e:
            logging.error(f"Error visualizing predictions: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict_image(
        self,
        image_path: str,
        save_visualization: bool = True,
        extract_text: Optional[bool] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Run detection and optional OCR on a single image
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
                
            image = read_image(image_path)
            if image is None:
                raise ValueError(f"Failed to read image at {image_path}")
                
            # Run YOLO inference
            raw_predictions = self.model.predict(
                image_path=image_path,
                conf_thres=self.config.conf_threshold,
                iou_thres=self.config.iou_threshold
            )
            
            should_extract = extract_text if extract_text is not None else self.config.enable_ocr
            processed_predictions = []
            
            for pred in raw_predictions:
                box = pred['box']
                text = ""
                if should_extract and self.ocr_reader:
                    text = self.ocr_reader.extract_text(image, bbox=box)
                    
                processed_predictions.append({
                    "box": box,
                    "confidence": pred['confidence'],
                    "class_id": pred['class_id'],
                    "class_name": pred['class_name'],
                    "text": text
                })
            
            vis_path = None
            if save_visualization:
                create_directories([self.config.visualization_dir])
                vis_filename = f"pred_{os.path.basename(image_path)}"
                vis_path = os.path.join(self.config.visualization_dir, vis_filename)
                self.visualize_predictions(image, processed_predictions, save_path=vis_path)
                
            return processed_predictions, vis_path
            
        except Exception as e:
            logging.error(f"Error predicting image {image_path}: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict_batch(
        self,
        image_dir: str,
        save_visualization: bool = True,
        extract_text: Optional[bool] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run detection on all images in a directory
        """
        try:
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"Directory not found: {image_dir}")
                
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
            
            results = {}
            for img_path in image_files:
                preds, _ = self.predict_image(
                    img_path,
                    save_visualization=save_visualization,
                    extract_text=extract_text
                )
                results[img_path] = preds
                
            return results
            
        except Exception as e:
            logging.error(f"Error during batch prediction: {str(e)}")
            raise VisionBoardException(e, sys)
