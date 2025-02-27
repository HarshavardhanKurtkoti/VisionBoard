import os
from pathlib import Path

"""
Defining common constant variables for training pipeline
"""
PIPELINE_NAME: str = "VisionBoard"
ARTIFACT_DIR: str = "Artifacts"

# Data related constants
DATA_DIR: str = "data"
TRAIN_DIR: str = "train"
VALID_DIR: str = "valid"
TEST_DIR: str = "test"
IMAGES_DIR: str = "images"
LABELS_DIR: str = "labels"

# Model related constants
MODEL_DIR: str = "models"
SAVED_MODEL_DIR: str = "saved_models"
MODEL_NAME: str = "yolov8m.pt"
BEST_MODEL_NAME: str = "best.pt"

# Training related constants
CLASSES: list = ["SignBoard"]
NUM_CLASSES: int = len(CLASSES)
IMAGE_SIZE: int = 640
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4
EPOCHS: int = 50

# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_PROCESSED_DIR: str = "processed"
DATA_INGESTION_TRAIN_DIR: str = os.path.join(DATA_INGESTION_PROCESSED_DIR, TRAIN_DIR)
DATA_INGESTION_VALID_DIR: str = os.path.join(DATA_INGESTION_PROCESSED_DIR, VALID_DIR)
DATA_INGESTION_TEST_DIR: str = os.path.join(DATA_INGESTION_PROCESSED_DIR, TEST_DIR)

# Data validation constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_REPORT_DIR: str = "validation_reports"

# Data transformation constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRAIN_DIR: str = "transformed_train"
DATA_TRANSFORMATION_VALID_DIR: str = "transformed_valid"
DATA_TRANSFORMATION_TEST_DIR: str = "transformed_test"

# Model training constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pt"
MODEL_TRAINER_CONFIG_FILE_NAME: str = "config.yaml"

# Model evaluation constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_NAME: str = "evaluation_report.yaml"
MODEL_EVALUATION_RESULTS_DIR: str = "evaluation_results"

# S3 sync constants
S3_BUCKET_NAME: str = "visionboard-data"
S3_MODEL_DIR: str = "models"
S3_DATA_DIR: str = "data"

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent

ARTIFACTS_DIR = os.path.join(ROOT_DIR, ARTIFACT_DIR)
DATA_INGESTION_ARTIFACT_DIR = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_DIR_NAME)
DATA_VALIDATION_ARTIFACT_DIR = os.path.join(ARTIFACTS_DIR, DATA_VALIDATION_DIR_NAME)
DATA_TRANSFORMATION_ARTIFACT_DIR = os.path.join(ARTIFACTS_DIR, DATA_TRANSFORMATION_DIR_NAME)
MODEL_TRAINER_ARTIFACT_DIR = os.path.join(ARTIFACTS_DIR, MODEL_TRAINER_DIR_NAME)
MODEL_EVALUATION_ARTIFACT_DIR = os.path.join(ARTIFACTS_DIR, MODEL_EVALUATION_DIR_NAME)

# YAML configs
DATA_YAML_FILE = os.path.join(ROOT_DIR, "data.yaml")
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "model_config.yaml")