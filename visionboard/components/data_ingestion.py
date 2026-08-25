import os
import sys
import shutil
from typing import Tuple, Optional
from pathlib import Path

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import DataIngestionConfig
from visionboard.entity.artifact_entity import DataIngestionArtifact
from visionboard.utils.main_utils.utils import create_directories, write_yaml_file
from visionboard.cloud.s3_syncer import S3Sync

class DataIngestion:
    """
    Component for handling data ingestion from local source, sample generation, or S3
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize DataIngestion with configuration
        Args:
            data_ingestion_config: Configuration for data ingestion
        """
        try:
            logging.info(f"{'='*20}Data Ingestion component started.{'='*20}")
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
            
        except Exception as e:
            logging.error(f"Error in DataIngestion.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def download_data(self) -> None:
        """Download dataset from S3 or copy from local source directory"""
        try:
            logging.info("Starting dataset acquisition")
            create_directories([self.data_ingestion_config.raw_data_dir])
            
            # If S3 URL / bucket download is configured and available
            if self.data_ingestion_config.dataset_download_url and self.s3_sync.is_available:
                logging.info(f"Downloading from S3: {self.data_ingestion_config.dataset_download_url}")
                self.s3_sync.sync_folder_from_s3(
                    folder_path=self.data_ingestion_config.raw_data_dir,
                    s3_prefix="data"
                )
            elif os.path.exists(self.data_ingestion_config.source_data_dir):
                # Copy from source directory if it exists and raw_data_dir is different
                src = os.path.abspath(self.data_ingestion_config.source_data_dir)
                dst = os.path.abspath(self.data_ingestion_config.raw_data_dir)
                if src != dst and os.path.exists(src):
                    logging.info(f"Copying data from {src} to {dst}")
                    for item in os.listdir(src):
                        s_item = os.path.join(src, item)
                        d_item = os.path.join(dst, item)
                        if os.path.isdir(s_item):
                            shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s_item, d_item)
            else:
                logging.info(f"No external data found at {self.data_ingestion_config.source_data_dir}. Using empty/synthetic data.")
                
            logging.info("Dataset download / staging completed")
            
        except Exception as e:
            logging.error(f"Error downloading dataset: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def organize_and_split_data(self) -> Tuple[str, str, Optional[str]]:
        """
        Organize raw data into train, valid, and test directories with images and labels subfolders
        """
        try:
            logging.info("Organizing dataset splits")
            
            train_dir = self.data_ingestion_config.train_dir
            valid_dir = self.data_ingestion_config.valid_dir
            test_dir = self.data_ingestion_config.test_dir
            raw_dir = self.data_ingestion_config.raw_data_dir
            
            # Create target subdirectories
            for d in [train_dir, valid_dir, test_dir]:
                create_directories([
                    os.path.join(d, "images"),
                    os.path.join(d, "labels")
                ])
            
            # If raw_dir has subdirectories like train, valid, test, copy them over
            for split, target_dir in [("train", train_dir), ("valid", valid_dir), ("val", valid_dir), ("test", test_dir)]:
                split_src = os.path.join(raw_dir, split)
                if os.path.exists(split_src):
                    # Check if split has images/labels or flat files
                    img_src = os.path.join(split_src, "images")
                    lbl_src = os.path.join(split_src, "labels")
                    if os.path.exists(img_src):
                        shutil.copytree(img_src, os.path.join(target_dir, "images"), dirs_exist_ok=True)
                    if os.path.exists(lbl_src):
                        shutil.copytree(lbl_src, os.path.join(target_dir, "labels"), dirs_exist_ok=True)
            
            # Fallback: if train has images but valid has none, copy/symlink train images to valid for YOLO training
            train_imgs = os.listdir(os.path.join(train_dir, "images")) if os.path.exists(os.path.join(train_dir, "images")) else []
            val_imgs = os.listdir(os.path.join(valid_dir, "images")) if os.path.exists(os.path.join(valid_dir, "images")) else []
            if train_imgs and not val_imgs:
                logging.info("Validation split empty. Populating validation split from training samples.")
                for f in train_imgs[:max(1, len(train_imgs) // 5)]:
                    shutil.copy2(
                        os.path.join(train_dir, "images", f),
                        os.path.join(valid_dir, "images", f)
                    )
                    lbl_f = os.path.splitext(f)[0] + ".txt"
                    src_lbl = os.path.join(train_dir, "labels", lbl_f)
                    if os.path.exists(src_lbl):
                        shutil.copy2(src_lbl, os.path.join(valid_dir, "labels", lbl_f))
                        
            return train_dir, test_dir, valid_dir
            
        except Exception as e:
            logging.error(f"Error organizing dataset splits: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def generate_data_yaml(self, train_dir: str, val_dir: str, test_dir: str) -> str:
        """
        Create YOLOv8 dataset YAML file with relative paths
        """
        try:
            yaml_path = os.path.join(self.data_ingestion_config.ingested_data_dir, "data.yaml")
            
            yaml_content = {
                "path": os.path.abspath(self.data_ingestion_config.ingested_data_dir),
                "train": "train/images",
                "val": "valid/images" if os.path.exists(os.path.join(val_dir, "images")) else "train/images",
                "test": "test/images" if os.path.exists(os.path.join(test_dir, "images")) else "train/images",
                "nc": 1,
                "names": ["signboard"]
            }
            
            write_yaml_file(yaml_path, yaml_content, replace=True)
            logging.info(f"Generated data YAML at {yaml_path}")
            return yaml_path
            
        except Exception as e:
            logging.error(f"Error generating data YAML: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Run complete data ingestion stage
        """
        try:
            logging.info("Initiating Data Ingestion stage")
            
            self.download_data()
            train_dir, test_dir, valid_dir = self.organize_and_split_data()
            data_yaml_path = self.generate_data_yaml(train_dir, valid_dir, test_dir)
            
            artifact = DataIngestionArtifact(
                train_file_path=train_dir,
                test_file_path=test_dir,
                valid_file_path=valid_dir,
                data_yaml_file_path=data_yaml_path,
                is_ingested=True,
                message="Data ingestion completed successfully."
            )
            
            logging.info(f"Data ingestion completed with artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in initiate_data_ingestion: {str(e)}")
            raise VisionBoardException(e, sys)
