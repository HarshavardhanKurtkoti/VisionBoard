import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from visionboard.constant.training_pipeline import *

@dataclass
class TrainingPipelineConfig:
    """Configuration for the overall training pipeline"""
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component"""
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
    dataset_download_url: str = DATASET_DOWNLOAD_URL
    raw_data_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_RAW_DATA_DIR)
    ingested_data_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
    train_dir: str = os.path.join(ingested_data_dir, TRAIN_DIR_NAME)
    test_dir: str = os.path.join(ingested_data_dir, TEST_DIR_NAME)

@dataclass
class DataValidationConfig:
    """Configuration for data validation component"""
    data_validation_dir: str = os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DIR_NAME)
    valid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_VALID_DIR)
    invalid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_INVALID_DIR)
    valid_data_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_VALID_FILE)
    invalid_data_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_INVALID_FILE)
    required_file_list: List[str] = None
    min_image_size: tuple = (640, 640)  # YOLOv8 default input size

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation component"""
    data_transformation_dir: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_dir: str = os.path.join(data_transformation_dir, TRANSFORMED_TRAIN_DIR_NAME)
    transformed_test_dir: str = os.path.join(data_transformation_dir, TRANSFORMED_TEST_DIR_NAME)
    augmentation_config: dict = None  # Will be loaded from yaml
    img_size: int = 640  # YOLOv8 default
    batch_size: int = 16

@dataclass
class ModelTrainerConfig:
    """Configuration for model training component"""
    model_trainer_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME)
    trained_model_dir: str = os.path.join(model_trainer_dir, TRAINED_MODEL_DIR)
    trained_model_path: str = os.path.join(trained_model_dir, MODEL_FILE_NAME)
    base_model: str = "yolov8n.pt"  # YOLOv8 nano model
    num_classes: int = None
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    device: str = "cuda"  # or "cpu"
    conf_thres: float = 0.25
    iou_thres: float = 0.45

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation component"""
    model_evaluation_dir: str = os.path.join(ARTIFACT_DIR, MODEL_EVALUATION_DIR_NAME)
    evaluation_report_file_path: str = os.path.join(model_evaluation_dir, EVALUATION_REPORT_FILE)
    test_data_path: str = None
    model_path: str = None
    tokenizer_path: str = None
    metric_list: List[str] = None
    min_accuracy: float = 0.6