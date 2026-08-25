import os
from pathlib import Path
from datetime import datetime

"""
Defining common constant variables for training pipeline
"""
PIPELINE_NAME: str = "VisionBoard"
ARTIFACT_DIR: str = "Artifacts"
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Base directories
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR: str = "VisionBoard_Data"
TRAIN_DIR_NAME: str = "train"
VALID_DIR_NAME: str = "valid"
TEST_DIR_NAME: str = "test"
IMAGES_DIR_NAME: str = "images"
LABELS_DIR_NAME: str = "labels"

# Pretrained model constants
MODEL_FILE_NAME: str = "best.pt"
BASE_MODEL_NAME: str = "yolov8n.pt"
SAVED_MODEL_DIR: str = "saved_models"

# Detection & Dataset constants
CLASSES: list = ["signboard"]
NUM_CLASSES: int = len(CLASSES)
IMAGE_SIZE: int = 640
BATCH_SIZE: int = 16
EPOCHS: int = 50

# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR: str = "raw_data"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATASET_DOWNLOAD_URL: str = ""

# Data validation constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_REPORT_FILE: str = "validation_report.yaml"

# Data transformation constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
TRANSFORMED_TRAIN_DIR_NAME: str = "train"
TRANSFORMED_VALID_DIR_NAME: str = "valid"
TRANSFORMED_TEST_DIR_NAME: str = "test"

# Model training constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
TRAINED_MODEL_DIR: str = "trained_model"

# Model evaluation constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
EVALUATION_REPORT_FILE: str = "evaluation_report.yaml"

# Model prediction constants
MODEL_PREDICTION_DIR_NAME: str = "model_prediction"
VISUALIZATION_DIR_NAME: str = "visualizations"

# S3 sync constants
S3_BUCKET_NAME: str = "visionboard-data"
S3_MODEL_DIR: str = "models"
S3_DATA_DIR: str = "data"

# YAML config file paths
DATA_YAML_FILE: str = str(ROOT_DIR / "config" / "data.yaml")
MODEL_CONFIG_FILE: str = str(ROOT_DIR / "config" / "model_config.yaml")