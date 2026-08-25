import os
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import DataValidationConfig
from visionboard.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from visionboard.utils.main_utils.utils import create_directories, write_yaml_file
from visionboard.utils.main_utils.image_utils import read_image

class DataValidation:
    """
    Component for validating dataset integrity, bounding box annotations, and image health
    """
    
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        """
        Initialize DataValidation
        Args:
            data_ingestion_artifact: Artifact from data ingestion
            data_validation_config: Configuration for data validation
        """
        try:
            logging.info(f"{'='*20}Data Validation component started.{'='*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
        except Exception as e:
            logging.error(f"Error in DataValidation.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def validate_image_file(self, image_path: str) -> bool:
        """
        Validate image readability, format, and dimensions
        """
        try:
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.data_validation_config.allowed_extensions:
                return False
            
            image = read_image(image_path)
            if image is None or image.size == 0:
                return False
            
            h, w = image.shape[:2]
            min_w, min_h = self.data_validation_config.min_image_size
            if w < min_w or h < min_h:
                return False
            
            return True
        except Exception:
            return False
    
    def validate_label_file(self, label_path: str) -> bool:
        """
        Validate YOLO annotation format (class_idx x_center y_center width height all normalized in [0, 1])
        """
        try:
            if not os.path.exists(label_path):
                # An image with no objects can have no label file or empty file
                return True
                
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        return False
                    
                    class_id = int(float(parts[0]))
                    if class_id < 0:
                        return False
                        
                    coords = [float(p) for p in parts[1:5]]
                    for c in coords:
                        if c < 0.0 or c > 1.0:
                            return False
                            
            return True
        except Exception:
            return False
    
    def validate_split_directory(self, split_dir: str) -> Dict[str, Any]:
        """
        Validate all images and labels in a given split directory
        """
        report = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "total_labels": 0,
            "valid_labels": 0,
            "invalid_labels": 0,
            "missing_labels": 0,
            "status": True
        }
        
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(images_dir):
            report["status"] = False
            return report
            
        for img_file in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_file)
            if not os.path.isfile(img_path):
                continue
                
            report["total_images"] += 1
            if self.validate_image_file(img_path):
                report["valid_images"] += 1
            else:
                report["invalid_images"] += 1
            
            # Check corresponding label
            lbl_name = os.path.splitext(img_file)[0] + ".txt"
            lbl_path = os.path.join(labels_dir, lbl_name)
            if os.path.exists(lbl_path):
                report["total_labels"] += 1
                if self.validate_label_file(lbl_path):
                    report["valid_labels"] += 1
                else:
                    report["invalid_labels"] += 1
            else:
                report["missing_labels"] += 1
        
        report["status"] = (report["invalid_images"] == 0 and report["invalid_labels"] == 0)
        return report
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Run complete data validation stage
        """
        try:
            logging.info("Initiating Data Validation stage")
            create_directories([
                self.data_validation_config.valid_data_dir,
                self.data_validation_config.invalid_data_dir
            ])
            
            splits = {
                "train": self.data_ingestion_artifact.train_file_path,
                "test": self.data_ingestion_artifact.test_file_path,
                "valid": self.data_ingestion_artifact.valid_file_path
            }
            
            validation_report = {}
            overall_status = True
            
            for split_name, split_path in splits.items():
                if split_path and os.path.exists(split_path):
                    split_report = self.validate_split_directory(split_path)
                    validation_report[split_name] = split_report
                    if not split_report["status"]:
                        overall_status = False
                else:
                    validation_report[split_name] = {"status": True, "message": "Split not provided"}
            
            # Save validation report
            write_yaml_file(
                self.data_validation_config.report_file_path,
                validation_report,
                replace=True
            )
            
            artifact = DataValidationArtifact(
                validation_status=overall_status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                valid_val_file_path=self.data_ingestion_artifact.valid_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                message="Data validation completed successfully." if overall_status else "Data validation found issues."
            )
            
            logging.info(f"Data validation completed with artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in initiate_data_validation: {str(e)}")
            raise VisionBoardException(e, sys)
