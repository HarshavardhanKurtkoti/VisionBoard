import os
import pytest
from visionboard.entity.config_entity import ModelPredictorConfig
from visionboard.pipeline.prediction_pipeline import PredictionPipeline

def test_prediction_pipeline_single_image(sample_image_path, temp_test_dir):
    cfg = ModelPredictorConfig(
        model_path="yolov8n.pt",
        visualization_dir=os.path.join(temp_test_dir, "vis"),
        save_visualization=True,
        enable_ocr=False
    )
    pipeline = PredictionPipeline(config=cfg)
    predictions = pipeline.predict_single(sample_image_path, save_visualization=True)
    assert isinstance(predictions, list)

def test_prediction_pipeline_batch(sample_dataset_dir, temp_test_dir):
    cfg = ModelPredictorConfig(
        model_path="yolov8n.pt",
        visualization_dir=os.path.join(temp_test_dir, "vis_batch"),
        save_visualization=False
    )
    pipeline = PredictionPipeline(config=cfg)
    test_imgs_dir = os.path.join(sample_dataset_dir, "test", "images")
    batch_results = pipeline.predict_batch(test_imgs_dir)
    assert len(batch_results) == 2
