import os
import sys
from typing import List, Dict, Any, Optional, Tuple

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelPredictorConfig
from visionboard.components.model_predictor import ModelPredictor

class PredictionPipeline:
    """
    Orchestration pipeline for object detection and OCR prediction
    """
    
    def __init__(self, config: Optional[ModelPredictorConfig] = None):
        """
        Initialize PredictionPipeline
        Args:
            config: Optional ModelPredictorConfig
        """
        try:
            self.config = config or ModelPredictorConfig()
            self.predictor = ModelPredictor(config=self.config)
            logging.info("PredictionPipeline initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing PredictionPipeline: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict_single(
        self,
        image_path: str,
        save_visualization: bool = True,
        extract_text: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Run prediction on a single image file
        Args:
            image_path: Path to image file
            save_visualization: Whether to save annotated image
            extract_text: Whether to perform OCR on detections
        Returns:
            List[Dict]: List of detected objects with bounding boxes, scores, and text
        """
        try:
            logging.info(f"PredictionPipeline: Predicting on single image {image_path}")
            predictions, vis_path = self.predictor.predict_image(
                image_path=image_path,
                save_visualization=save_visualization,
                extract_text=extract_text
            )
            return predictions
        except Exception as e:
            logging.error(f"Error in predict_single: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict_batch(
        self,
        image_dir: str,
        save_visualization: bool = True,
        extract_text: Optional[bool] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run prediction on a directory of images
        Args:
            image_dir: Path to directory containing images
            save_visualization: Whether to save annotated images
            extract_text: Whether to perform OCR on detections
        Returns:
            Dict: Mapping of image file paths to detected objects
        """
        try:
            logging.info(f"PredictionPipeline: Predicting on batch directory {image_dir}")
            return self.predictor.predict_batch(
                image_dir=image_dir,
                save_visualization=save_visualization,
                extract_text=extract_text
            )
        except Exception as e:
            logging.error(f"Error in predict_batch: {str(e)}")
            raise VisionBoardException(e, sys)
