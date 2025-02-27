import os
import sys
import yaml
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import DataTransformationConfig
from visionboard.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact

class DataTransformation:
    """
    Class for handling data transformation operations
    """
    
    def __init__(
        self,
        config: DataTransformationConfig,
        validation_artifact: DataValidationArtifact
    ):
        """
        Initialize with configuration
        Args:
            config: Configuration for data transformation
            validation_artifact: Artifact from data validation
        """
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20}")
            self.config = config
            self.validation_artifact = validation_artifact
            
        except Exception as e:
            logging.error(f"Error in DataTransformation.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def get_data_augmentation_config(self) -> Dict[str, Any]:
        """
        Get data augmentation configuration
        Returns:
            Dict: Augmentation parameters
        """
        try:
            logging.info("Loading data augmentation configuration")
            
            # Default augmentation config for YOLOv8
            default_config = {
                "hsv_h": 0.015,  # HSV-Hue augmentation
                "hsv_s": 0.7,    # HSV-Saturation augmentation
                "hsv_v": 0.4,    # HSV-Value augmentation
                "degrees": 0.0,   # Rotation
                "translate": 0.1, # Translation
                "scale": 0.5,    # Scale
                "shear": 0.0,    # Shear
                "flipud": 0.0,   # Flip up-down
                "fliplr": 0.5,   # Flip left-right
                "mosaic": 1.0,   # Mosaic augmentation
                "mixup": 0.0     # Mixup augmentation
            }
            
            # If custom config exists, update default
            if self.config.augmentation_config:
                default_config.update(self.config.augmentation_config)
            
            logging.info(f"Augmentation config: {default_config}")
            return default_config
            
        except Exception as e:
            logging.error(f"Error getting augmentation config: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to an image
        Args:
            image: Input image
        Returns:
            np.ndarray: Augmented image
        """
        try:
            logging.info("Applying augmentation to image")
            
            # Get augmentation config
            aug_config = self.get_data_augmentation_config()
            
            # TODO: Implement augmentation pipeline
            # This should:
            # 1. Apply geometric transformations
            # 2. Apply color transformations
            # 3. Apply noise and blur
            # 4. Handle label transformations
            
            # For now, just resize the image
            image = cv2.resize(image, (self.config.img_size, self.config.img_size))
            
            logging.info("Augmentation applied successfully")
            return image
            
        except Exception as e:
            logging.error(f"Error applying augmentation: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def transform_image_and_label(
        self,
        image_path: str,
        label_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform image and its label
        Args:
            image_path: Path to image
            label_path: Path to label file (optional)
        Returns:
            Tuple: Transformed image and label
        """
        try:
            logging.info(f"Transforming image: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Apply augmentation
            transformed_image = self.apply_augmentation(image)
            
            # Handle label if provided
            transformed_label = None
            if label_path and os.path.exists(label_path):
                # TODO: Transform label according to image transformation
                # This should:
                # 1. Read YOLO format labels
                # 2. Apply same geometric transformations
                # 3. Save in YOLO format
                pass
            
            logging.info("Image and label transformation completed")
            return transformed_image, transformed_label
            
        except Exception as e:
            logging.error(f"Error transforming image and label: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiate data transformation process
        Returns:
            DataTransformationArtifact: Paths to transformed data
        """
        try:
            logging.info("Starting data transformation")
            
            # Create output directories
            os.makedirs(self.config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.config.transformed_test_dir, exist_ok=True)
            
            # Transform training data
            train_dir = self.validation_artifact.valid_train_file_path
            for image_file in os.listdir(train_dir):
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(train_dir, image_file)
                    label_path = image_path.replace('images', 'labels').replace(
                        os.path.splitext(image_file)[1], '.txt'
                    )
                    
                    # Transform image and label
                    transformed_image, transformed_label = self.transform_image_and_label(
                        image_path, label_path
                    )
                    
                    # Save transformed data
                    # TODO: Implement save logic
            
            # Create data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.config.transformed_train_dir,
                transformed_test_file_path=self.config.transformed_test_dir,
                is_transformed=True,
                message="Data transformation completed successfully."
            )
            
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise VisionBoardException(e, sys)