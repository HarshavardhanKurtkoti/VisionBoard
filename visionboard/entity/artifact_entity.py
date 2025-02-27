from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class DataIngestionArtifact:
    """Artifact produced by data ingestion component"""
    train_file_path: str
    test_file_path: str
    is_ingested: bool
    message: str

@dataclass
class DataValidationArtifact:
    """Artifact produced by data validation component"""
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    message: str

@dataclass
class DataTransformationArtifact:
    """Artifact produced by data transformation component"""
    transformed_train_file_path: str
    transformed_test_file_path: str
    is_transformed: bool
    message: str

@dataclass
class DetectionMetricArtifact:
    """Metrics for object detection model"""
    precision: float
    recall: float
    f1_score: float
    map50: float  # mAP at IoU=0.5
    map75: float  # mAP at IoU=0.75
    map50_95: float  # mAP at IoU=0.5:0.95
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mAP@0.5": self.map50,
            "mAP@0.75": self.map75,
            "mAP@0.5:0.95": self.map50_95
        }

@dataclass
class ModelTrainerArtifact:
    """Artifact produced by model trainer component"""
    trained_model_file_path: str
    train_metric_artifact: DetectionMetricArtifact
    test_metric_artifact: DetectionMetricArtifact
    is_trained: bool
    message: str

@dataclass
class ModelEvaluationArtifact:
    """Artifact produced by model evaluation component"""
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: DetectionMetricArtifact
    best_model_metric_artifact: DetectionMetricArtifact
    message: str