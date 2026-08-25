import os
import sys
import shutil
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import DataTransformationConfig
from visionboard.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from visionboard.utils.main_utils.utils import create_directories, write_yaml_file
from visionboard.utils.main_utils.image_utils import read_image, save_image, resize_image

class DataTransformation:
    """
    Component for handling data transformation, resizing, augmentation config, and data.yaml generation
    """
    
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        """
        Initialize DataTransformation
        Args:
            data_validation_artifact: Artifact from data validation
            data_transformation_config: Configuration for data transformation
        """
        try:
            logging.info(f"{'='*20}Data Transformation component started.{'='*20}")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            
        except Exception as e:
            logging.error(f"Error in DataTransformation.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def get_default_augmentation_config(self) -> Dict[str, Any]:
        """
        Get default YOLOv8 augmentation hyperparameter dictionary
        """
        default_config = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0
        }
        if self.data_transformation_config.augmentation_config:
            default_config.update(self.data_transformation_config.augmentation_config)
        return default_config
    
    def transform_split(self, src_split_dir: str, dst_split_dir: str) -> None:
        """
        Transform and copy images and labels for a specific split
        """
        try:
            src_imgs = os.path.join(src_split_dir, "images")
            src_lbls = os.path.join(src_split_dir, "labels")
            dst_imgs = os.path.join(dst_split_dir, "images")
            dst_lbls = os.path.join(dst_split_dir, "labels")
            
            create_directories([dst_imgs, dst_lbls])
            
            if not os.path.exists(src_imgs):
                return
                
            for file_name in os.listdir(src_imgs):
                src_img_path = os.path.join(src_imgs, file_name)
                dst_img_path = os.path.join(dst_imgs, file_name)
                
                if not os.path.isfile(src_img_path):
                    continue
                    
                # Copy image (or resize if requested)
                img = read_image(src_img_path)
                if img is not None:
                    if self.data_transformation_config.img_size:
                        target_size = (self.data_transformation_config.img_size, self.data_transformation_config.img_size)
                        if img.shape[0] != target_size[1] or img.shape[1] != target_size[0]:
                            img = resize_image(img, target_size)
                    save_image(dst_img_path, img)
                else:
                    shutil.copy2(src_img_path, dst_img_path)
                
                # Copy corresponding label
                lbl_name = os.path.splitext(file_name)[0] + ".txt"
                src_lbl_path = os.path.join(src_lbls, lbl_name)
                dst_lbl_path = os.path.join(dst_lbls, lbl_name)
                if os.path.exists(src_lbl_path):
                    shutil.copy2(src_lbl_path, dst_lbl_path)
                    
        except Exception as e:
            logging.error(f"Error transforming split {src_split_dir}: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def generate_data_yaml(self) -> str:
        """
        Generate data.yaml file with absolute path for training
        """
        try:
            yaml_path = self.data_transformation_config.transformed_data_yaml_path
            
            val_dir = self.data_transformation_config.transformed_valid_dir
            train_dir = self.data_transformation_config.transformed_train_dir
            test_dir = self.data_transformation_config.transformed_test_dir
            
            yaml_content = {
                "path": os.path.abspath(self.data_transformation_config.data_transformation_dir),
                "train": "train/images",
                "val": "valid/images" if os.path.exists(os.path.join(val_dir, "images")) and os.listdir(os.path.join(val_dir, "images")) else "train/images",
                "test": "test/images" if os.path.exists(os.path.join(test_dir, "images")) and os.listdir(os.path.join(test_dir, "images")) else "train/images",
                "nc": 1,
                "names": ["signboard"]
            }
            
            write_yaml_file(yaml_path, yaml_content, replace=True)
            logging.info(f"Created transformed data YAML at {yaml_path}")
            return yaml_path
            
        except Exception as e:
            logging.error(f"Error generating data YAML: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Run complete data transformation stage
        """
        try:
            logging.info("Initiating Data Transformation stage")
            
            train_dst = self.data_transformation_config.transformed_train_dir
            valid_dst = self.data_transformation_config.transformed_valid_dir
            test_dst = self.data_transformation_config.transformed_test_dir
            
            create_directories([train_dst, valid_dst, test_dst])
            
            if self.data_validation_artifact.valid_train_file_path:
                self.transform_split(self.data_validation_artifact.valid_train_file_path, train_dst)
            if self.data_validation_artifact.valid_val_file_path:
                self.transform_split(self.data_validation_artifact.valid_val_file_path, valid_dst)
            if self.data_validation_artifact.valid_test_file_path:
                self.transform_split(self.data_validation_artifact.valid_test_file_path, test_dst)
            
            data_yaml_path = self.generate_data_yaml()
            
            artifact = DataTransformationArtifact(
                transformed_train_file_path=train_dst,
                transformed_test_file_path=test_dst,
                transformed_val_file_path=valid_dst,
                transformed_data_yaml_path=data_yaml_path,
                is_transformed=True,
                message="Data transformation completed successfully."
            )
            
            logging.info(f"Data transformation completed with artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {str(e)}")
            raise VisionBoardException(e, sys)