from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DataIngestionArtifact:
    """Artifact produced by data ingestion component"""
    train_file_path: str
    test_file_path: str
    valid_file_path: Optional[str] = None
    data_yaml_file_path: Optional[str] = None
    is_ingested: bool = True
    message: str = ""

@dataclass
class DataValidationArtifact:
    """Artifact produced by data validation component"""
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    valid_val_file_path: Optional[str] = None
    invalid_train_file_path: Optional[str] = None
    invalid_test_file_path: Optional[str] = None
    report_file_path: Optional[str] = None
    message: str = ""

@dataclass
class DataTransformationArtifact:
    """Artifact produced by data transformation component"""
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_val_file_path: Optional[str] = None
    transformed_data_yaml_path: Optional[str] = None
    is_transformed: bool = True
    message: str = ""

@dataclass
class DetectionMetricArtifact:
    """Metrics for object detection model"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map50: float = 0.0   # mAP at IoU=0.5
    map75: float = 0.0   # mAP at IoU=0.75
    map50_95: float = 0.0 # mAP at IoU=0.5:0.95
    
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

# Alias for backwards compatibility
ClassificationMetricArtifact = DetectionMetricArtifact

@dataclass
class ModelTrainerArtifact:
    """Artifact produced by model trainer component"""
    trained_model_file_path: str
    train_metric_artifact: Optional[DetectionMetricArtifact] = None
    test_metric_artifact: Optional[DetectionMetricArtifact] = None
    is_trained: bool = True
    is_model_accepted: bool = True
    message: str = ""

@dataclass
class ModelEvaluationArtifact:
    """Artifact produced by model evaluation component"""
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: Optional[str] = None
    trained_model_path: Optional[str] = None
    train_model_metric_artifact: Optional[DetectionMetricArtifact] = None
    best_model_metric_artifact: Optional[DetectionMetricArtifact] = None
    evaluation_report_path: Optional[str] = None
    message: str = ""

@dataclass
class ModelPredictorArtifact:
    """Artifact produced by model predictor component"""
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    visualization_paths: List[str] = field(default_factory=list)
    message: str = ""