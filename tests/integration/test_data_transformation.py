import os
import pytest
from visionboard.entity.config_entity import TrainingPipelineConfig, DataTransformationConfig
from visionboard.entity.artifact_entity import DataValidationArtifact
from visionboard.components.data_transformation import DataTransformation

def test_data_transformation_flow(sample_dataset_dir, temp_test_dir):
    pipe_cfg = TrainingPipelineConfig(artifact_dir=temp_test_dir)
    trans_cfg = DataTransformationConfig(training_pipeline_config=pipe_cfg)
    
    val_artifact = DataValidationArtifact(
        validation_status=True,
        valid_train_file_path=os.path.join(sample_dataset_dir, "train"),
        valid_val_file_path=os.path.join(sample_dataset_dir, "valid"),
        valid_test_file_path=os.path.join(sample_dataset_dir, "test"),
        message="Valid"
    )
    
    transformer = DataTransformation(
        data_validation_artifact=val_artifact,
        data_transformation_config=trans_cfg
    )
    trans_artifact = transformer.initiate_data_transformation()
    
    assert trans_artifact.is_transformed is True
    assert os.path.exists(trans_artifact.transformed_data_yaml_path)
    assert os.path.isdir(trans_artifact.transformed_train_file_path)
