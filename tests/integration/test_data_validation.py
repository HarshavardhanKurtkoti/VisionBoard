import os
import pytest
from visionboard.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from visionboard.entity.artifact_entity import DataIngestionArtifact
from visionboard.components.data_validation import DataValidation

def test_data_validation_flow(sample_dataset_dir, temp_test_dir):
    pipe_cfg = TrainingPipelineConfig(artifact_dir=temp_test_dir)
    val_cfg = DataValidationConfig(training_pipeline_config=pipe_cfg)
    
    ingest_artifact = DataIngestionArtifact(
        train_file_path=os.path.join(sample_dataset_dir, "train"),
        valid_file_path=os.path.join(sample_dataset_dir, "valid"),
        test_file_path=os.path.join(sample_dataset_dir, "test"),
        is_ingested=True
    )
    
    validator = DataValidation(
        data_ingestion_artifact=ingest_artifact,
        data_validation_config=val_cfg
    )
    val_artifact = validator.initiate_data_validation()
    
    assert val_artifact.validation_status is True
    assert os.path.exists(val_artifact.report_file_path)
