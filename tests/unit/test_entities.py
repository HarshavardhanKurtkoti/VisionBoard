import os
import pytest
from visionboard.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPredictorConfig
)
from visionboard.entity.artifact_entity import (
    DetectionMetricArtifact,
    ClassificationMetricArtifact,
    DataIngestionArtifact
)

def test_training_pipeline_config():
    cfg = TrainingPipelineConfig()
    assert cfg.pipeline_name == "VisionBoard"
    assert cfg.artifact_dir is not None
    assert cfg.timestamp is not None

def test_config_from_yaml(temp_test_dir):
    yaml_path = os.path.join(temp_test_dir, "model_config.yaml")
    with open(yaml_path, "w") as f:
        f.write("""
model:
  name: "yolov8m.pt"
  img_size: 720
  device: "cpu"
inference:
  conf_thres: 0.35
  iou_thres: 0.50
  visualize: true
logging:
  project: "CustomProject"
  artifact_path: "CustomArtifacts"
""")
    
    pipe_cfg = TrainingPipelineConfig.from_yaml(yaml_path)
    assert pipe_cfg.pipeline_name == "CustomProject"
    assert "CustomArtifacts" in pipe_cfg.artifact_dir
    
    pred_cfg = ModelPredictorConfig.from_yaml(yaml_path)
    assert pred_cfg.model_path == "yolov8m.pt"
    assert pred_cfg.img_size == 720
    assert pred_cfg.conf_threshold == 0.35
    assert pred_cfg.iou_threshold == 0.50

def test_detection_metric_artifact_dict():
    metric = DetectionMetricArtifact(
        precision=0.9,
        recall=0.8,
        f1_score=0.85,
        map50=0.88,
        map75=0.70,
        map50_95=0.65
    )
    d = metric.to_dict()
    assert d["precision"] == 0.9
    assert d["mAP@0.5"] == 0.88
