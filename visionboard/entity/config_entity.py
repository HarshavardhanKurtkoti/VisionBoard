import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from visionboard.constant.training_pipeline import (
    PIPELINE_NAME,
    ARTIFACT_DIR,
    TIMESTAMP,
    DATA_DIR,
    TRAIN_DIR_NAME,
    VALID_DIR_NAME,
    TEST_DIR_NAME,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_RAW_DATA_DIR,
    DATA_INGESTION_INGESTED_DIR,
    DATASET_DOWNLOAD_URL,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_VALID_DIR,
    DATA_VALIDATION_INVALID_DIR,
    DATA_VALIDATION_REPORT_FILE,
    DATA_TRANSFORMATION_DIR_NAME,
    TRANSFORMED_TRAIN_DIR_NAME,
    TRANSFORMED_VALID_DIR_NAME,
    TRANSFORMED_TEST_DIR_NAME,
    MODEL_TRAINER_DIR_NAME,
    TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
    BASE_MODEL_NAME,
    MODEL_EVALUATION_DIR_NAME,
    EVALUATION_REPORT_FILE,
    MODEL_PREDICTION_DIR_NAME,
    VISUALIZATION_DIR_NAME,
    CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS
)
from visionboard.utils.main_utils.utils import read_yaml_file

@dataclass
class TrainingPipelineConfig:
    """Configuration for the overall training pipeline"""
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "TrainingPipelineConfig":
        if yaml_path and os.path.exists(yaml_path):
            data = read_yaml_file(yaml_path)
            pipeline_name = data.get("logging", {}).get("project", PIPELINE_NAME)
            artifact_base = data.get("logging", {}).get("artifact_path", ARTIFACT_DIR)
            ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            return cls(
                pipeline_name=pipeline_name,
                artifact_dir=os.path.join(artifact_base, ts),
                timestamp=ts
            )
        return cls()

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component"""
    training_pipeline_config: Optional[TrainingPipelineConfig] = None
    data_ingestion_dir: str = field(init=False)
    dataset_download_url: str = DATASET_DOWNLOAD_URL
    source_data_dir: str = DATA_DIR
    raw_data_dir: str = field(init=False)
    ingested_data_dir: str = field(init=False)
    train_dir: str = field(init=False)
    valid_dir: str = field(init=False)
    test_dir: str = field(init=False)

    def __post_init__(self):
        base_dir = (
            self.training_pipeline_config.artifact_dir
            if self.training_pipeline_config
            else os.path.join(ARTIFACT_DIR, TIMESTAMP)
        )
        self.data_ingestion_dir = os.path.join(base_dir, DATA_INGESTION_DIR_NAME)
        self.raw_data_dir = os.path.join(self.data_ingestion_dir, DATA_INGESTION_RAW_DATA_DIR)
        self.ingested_data_dir = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
        self.train_dir = os.path.join(self.ingested_data_dir, TRAIN_DIR_NAME)
        self.valid_dir = os.path.join(self.ingested_data_dir, VALID_DIR_NAME)
        self.test_dir = os.path.join(self.ingested_data_dir, TEST_DIR_NAME)

@dataclass
class DataValidationConfig:
    """Configuration for data validation component"""
    training_pipeline_config: Optional[TrainingPipelineConfig] = None
    data_validation_dir: str = field(init=False)
    valid_data_dir: str = field(init=False)
    invalid_data_dir: str = field(init=False)
    report_file_path: str = field(init=False)
    required_file_list: List[str] = field(default_factory=lambda: [TRAIN_DIR_NAME, TEST_DIR_NAME])
    min_image_size: tuple = (32, 32)
    allowed_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])

    def __post_init__(self):
        base_dir = (
            self.training_pipeline_config.artifact_dir
            if self.training_pipeline_config
            else os.path.join(ARTIFACT_DIR, TIMESTAMP)
        )
        self.data_validation_dir = os.path.join(base_dir, DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir = os.path.join(self.data_validation_dir, DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir = os.path.join(self.data_validation_dir, DATA_VALIDATION_INVALID_DIR)
        self.report_file_path = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE)

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation component"""
    training_pipeline_config: Optional[TrainingPipelineConfig] = None
    data_transformation_dir: str = field(init=False)
    transformed_train_dir: str = field(init=False)
    transformed_valid_dir: str = field(init=False)
    transformed_test_dir: str = field(init=False)
    transformed_data_yaml_path: str = field(init=False)
    augmentation_config: Optional[Dict[str, Any]] = None
    img_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE

    def __post_init__(self):
        base_dir = (
            self.training_pipeline_config.artifact_dir
            if self.training_pipeline_config
            else os.path.join(ARTIFACT_DIR, TIMESTAMP)
        )
        self.data_transformation_dir = os.path.join(base_dir, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_dir = os.path.join(self.data_transformation_dir, TRANSFORMED_TRAIN_DIR_NAME)
        self.transformed_valid_dir = os.path.join(self.data_transformation_dir, TRANSFORMED_VALID_DIR_NAME)
        self.transformed_test_dir = os.path.join(self.data_transformation_dir, TRANSFORMED_TEST_DIR_NAME)
        self.transformed_data_yaml_path = os.path.join(self.data_transformation_dir, "data.yaml")

@dataclass
class ModelTrainerConfig:
    """Configuration for model training component"""
    training_pipeline_config: Optional[TrainingPipelineConfig] = None
    model_trainer_dir: str = field(init=False)
    trained_model_dir: str = field(init=False)
    trained_model_path: str = field(init=False)
    base_model: str = BASE_MODEL_NAME
    pretrained_model_path: str = BASE_MODEL_NAME
    num_classes: int = NUM_CLASSES
    class_names: List[str] = field(default_factory=lambda: list(CLASSES))
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    img_size: int = IMAGE_SIZE
    device: str = "cpu"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    model_acceptance_threshold: float = 0.20

    def __post_init__(self):
        base_dir = (
            self.training_pipeline_config.artifact_dir
            if self.training_pipeline_config
            else os.path.join(ARTIFACT_DIR, TIMESTAMP)
        )
        self.model_trainer_dir = os.path.join(base_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_dir = os.path.join(self.model_trainer_dir, TRAINED_MODEL_DIR)
        self.trained_model_path = os.path.join(self.trained_model_dir, MODEL_FILE_NAME)

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation component"""
    training_pipeline_config: Optional[TrainingPipelineConfig] = None
    model_evaluation_dir: str = field(init=False)
    evaluation_report_file_path: str = field(init=False)
    test_data_path: Optional[str] = None
    model_path: Optional[str] = None
    min_accuracy: float = 0.20
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    def __post_init__(self):
        base_dir = (
            self.training_pipeline_config.artifact_dir
            if self.training_pipeline_config
            else os.path.join(ARTIFACT_DIR, TIMESTAMP)
        )
        self.model_evaluation_dir = os.path.join(base_dir, MODEL_EVALUATION_DIR_NAME)
        self.evaluation_report_file_path = os.path.join(self.model_evaluation_dir, EVALUATION_REPORT_FILE)

@dataclass
class ModelPredictorConfig:
    """Configuration for model prediction / inference"""
    model_path: str = os.getenv("MODEL_PATH", BASE_MODEL_NAME)
    img_size: int = IMAGE_SIZE
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    save_visualization: bool = True
    visualization_dir: str = os.path.join("runs", "predict_vis")
    enable_ocr: bool = False
    class_names: List[str] = field(default_factory=lambda: list(CLASSES))

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "ModelPredictorConfig":
        if yaml_path and os.path.exists(yaml_path):
            data = read_yaml_file(yaml_path)
            inf_cfg = data.get("inference", {})
            model_cfg = data.get("model", {})
            return cls(
                model_path=model_cfg.get("name", BASE_MODEL_NAME),
                img_size=model_cfg.get("img_size", IMAGE_SIZE),
                conf_threshold=inf_cfg.get("conf_thres", 0.25),
                iou_threshold=inf_cfg.get("iou_thres", 0.45),
                device=model_cfg.get("device", "cpu"),
                save_visualization=inf_cfg.get("visualize", True),
                enable_ocr=data.get("ocr", {}).get("enabled", False)
            )
        return cls()