import os
import pytest
from visionboard.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from visionboard.pipeline.training_pipeline import TrainingPipeline

def test_training_pipeline_orchestration(sample_dataset_dir, temp_test_dir):
    pipe_cfg = TrainingPipelineConfig(
        pipeline_name="TestVisionBoard",
        artifact_dir=temp_test_dir,
        timestamp="test_ts"
    )
    
    # Point default data dir to sample_dataset_dir
    os.environ["DATA_DIR"] = sample_dataset_dir
    
    pipeline = TrainingPipeline(config=pipe_cfg)
    
    # 1. Ingestion
    ingestion_artifact = pipeline.start_data_ingestion()
    assert ingestion_artifact.is_ingested is True
    
    # 2. Validation
    validation_artifact = pipeline.start_data_validation(ingestion_artifact)
    assert validation_artifact.validation_status is True
    
    # 3. Transformation
    transformation_artifact = pipeline.start_data_transformation(validation_artifact)
    assert transformation_artifact.is_transformed is True
    
    # 4. Model Training
    trainer_artifact = pipeline.start_model_trainer(transformation_artifact)
    assert trainer_artifact.is_trained is True
    
    # 5. Model Evaluation
    eval_artifact = pipeline.start_model_evaluation(transformation_artifact, trainer_artifact)
    assert eval_artifact is not None
