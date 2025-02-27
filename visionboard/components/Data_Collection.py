import os
import sys
import yaml
import shutil
from typing import Tuple, Optional
from pathlib import Path

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import DataIngestionConfig
from visionboard.entity.artifact_entity import DataIngestionArtifact
from visionboard.utils.main_utils.utils import create_directories

class DataIngestion:
    """
    Class for handling data ingestion operations
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize with configuration
        Args:
            config: Configuration for data ingestion
        """
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20}")
            self.config = config
            
        except Exception as e:
            logging.error(f"Error in DataIngestion.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def download_data(self) -> None:
        """Download dataset from source"""
        try:
            logging.info("Downloading dataset")
            
            # Create raw data directory
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            
            # TODO: Implement dataset download logic
            # This could be from:
            # 1. S3 bucket
            # 2. Public dataset URL
            # 3. Local storage
            
            logging.info("Dataset download completed")
            
        except Exception as e:
            logging.error(f"Error downloading dataset: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def split_data(self) -> Tuple[str, str]:
        """
        Split data into train and test sets
        Returns:
            Tuple[str, str]: Paths to train and test directories
        """
        try:
            logging.info("Splitting data into train and test sets")
            
            # Create train and test directories
            os.makedirs(self.config.train_dir, exist_ok=True)
            os.makedirs(self.config.test_dir, exist_ok=True)
            
            # TODO: Implement data splitting logic
            # This should:
            # 1. Split images and labels maintaining the relationship
            # 2. Create train/test YAML files for YOLOv8
            # 3. Verify split ratio and class distribution
            
            logging.info("Data splitting completed")
            return self.config.train_dir, self.config.test_dir
            
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def create_data_yaml(self, save_path: str, data_dir: str, split: str = "train") -> None:
        """
        Create YAML file for YOLOv8 training
        Args:
            save_path: Path to save YAML file
            data_dir: Path to data directory
            split: Data split (train/test)
        """
        try:
            logging.info(f"Creating {split} YAML file")
            
            # TODO: Get these from your dataset
            num_classes = 1  # Number of classes in your dataset
            class_names = ["signboard"]  # List of class names
            
            yaml_content = {
                "path": os.path.abspath(data_dir),  # Dataset root dir
                "train": "images/train",  # Train images relative to path
                "val": "images/val",      # Val images relative to path
                "test": "images/test",    # Test images relative to path
                
                "nc": num_classes,        # Number of classes
                "names": class_names      # Class names
            }
            
            # Save YAML file
            with open(save_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
            
            logging.info(f"Created YAML file at: {save_path}")
            
        except Exception as e:
            logging.error(f"Error creating YAML file: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiate data ingestion process
        Returns:
            DataIngestionArtifact: Paths to ingested data
        """
        try:
            logging.info("Starting data ingestion")
            
            # Download data
            self.download_data()
            
            # Split data
            train_dir, test_dir = self.split_data()
            
            # Create data artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_dir,
                test_file_path=test_dir,
                is_ingested=True,
                message="Data ingestion completed successfully."
            )
            
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise VisionBoardException(e, sys)