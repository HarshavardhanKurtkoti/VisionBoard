import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelPredictorConfig
from visionboard.utils.ml_utils.model.estimator import YOLOModel

class ModelPredictor:
    """
    Class for handling model prediction operations
    """
    
    def __init__(self, config: ModelPredictorConfig):
        """
        Initialize predictor with configuration
        Args:
            config: Configuration for model prediction
        """
        try:
            logging.info(f"{'='*20}Model Prediction log started.{'='*20}")
            self.config = config
            
            # Initialize model
            self.model = YOLOModel(config.model_path)
            logging.info("Model initialized successfully")
            
        except Exception as e:
            logging.error(f"Error in ModelPredictor.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for inference
        Args:
            image_path: Path to input image
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            logging.info(f"Preprocessing image: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(
                image,
                (self.config.img_size, self.config.img_size),
                interpolation=cv2.INTER_LINEAR
            )
            
            logging.info("Image preprocessing completed")
            return image
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def postprocess_predictions(
        self,
        predictions: List[Dict[str, Any]],
        original_size: tuple
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model predictions
        Args:
            predictions: Raw model predictions
            original_size: Original image size (height, width)
        Returns:
            List[Dict]: Processed predictions
        """
        try:
            logging.info("Postprocessing predictions")
            
            processed_predictions = []
            for pred in predictions:
                # Scale bounding box coordinates back to original size
                box = pred['box']
                scaled_box = [
                    box[0] * original_size[1],  # x1
                    box[1] * original_size[0],  # y1
                    box[2] * original_size[1],  # x2
                    box[3] * original_size[0]   # y2
                ]
                
                # Create processed prediction
                processed_pred = {
                    'box': scaled_box,
                    'confidence': pred['confidence'],
                    'class_id': pred['class_id'],
                    'class_name': pred['class_name']
                }
                processed_predictions.append(processed_pred)
            
            logging.info(f"Processed {len(processed_predictions)} predictions")
            return processed_predictions
            
        except Exception as e:
            logging.error(f"Error postprocessing predictions: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize predictions on image
        Args:
            image: Input image
            predictions: Model predictions
            save_path: Path to save visualization (optional)
        Returns:
            np.ndarray: Image with visualized predictions
        """
        try:
            logging.info("Visualizing predictions")
            
            # Make a copy of the image
            vis_image = image.copy()
            
            # Draw predictions
            for pred in predictions:
                box = pred['box']
                conf = pred['confidence']
                label = f"{pred['class_name']} {conf:.2f}"
                
                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(
                    vis_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),  # Green color
                    2
                )
                
                # Draw label
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Save visualization if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_image)
                logging.info(f"Saved visualization to: {save_path}")
            
            logging.info("Visualization completed")
            return vis_image
            
        except Exception as e:
            logging.error(f"Error visualizing predictions: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict(
        self,
        image_path: str,
        save_visualization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an image
        Args:
            image_path: Path to input image
            save_visualization: Whether to save visualization
        Returns:
            List[Dict]: Predictions with bounding boxes and classes
        """
        try:
            logging.info(f"Running prediction on: {image_path}")
            
            # Read and preprocess image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            original_size = original_image.shape[:2]  # (height, width)
            preprocessed_image = self.preprocess_image(image_path)
            
            # Run inference
            predictions = self.model.predict(
                image_path,
                conf_thres=self.config.conf_threshold,
                iou_thres=self.config.iou_threshold
            )
            
            # Postprocess predictions
            processed_predictions = self.postprocess_predictions(
                predictions,
                original_size
            )
            
            # Visualize if requested
            if save_visualization:
                vis_path = os.path.join(
                    self.config.visualization_dir,
                    os.path.basename(image_path)
                )
                self.visualize_predictions(
                    original_image,
                    processed_predictions,
                    vis_path
                )
            
            logging.info("Prediction completed successfully")
            return processed_predictions
            
        except Exception as e:
            logging.error(f"Error running prediction: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def predict_batch(
        self,
        image_dir: str,
        save_visualization: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run inference on a batch of images
        Args:
            image_dir: Directory containing images
            save_visualization: Whether to save visualizations
        Returns:
            Dict: Mapping of image paths to predictions
        """
        try:
            logging.info(f"Running batch prediction on directory: {image_dir}")
            
            # Get list of images
            image_paths = [
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            # Run predictions
            results = {}
            for image_path in image_paths:
                predictions = self.predict(
                    image_path,
                    save_visualization
                )
                results[image_path] = predictions
            
            logging.info(f"Completed batch prediction on {len(results)} images")
            return results
            
        except Exception as e:
            logging.error(f"Error in batch prediction: {str(e)}")
            raise VisionBoardException(e, sys)
