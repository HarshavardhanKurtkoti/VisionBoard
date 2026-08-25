import os
import sys
from typing import Optional

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.cloud.s3_syncer import S3Sync

from visionboard.components.data_ingestion import DataIngestion
from visionboard.components.data_validation import DataValidation
from visionboard.components.data_transformation import DataTransformation
from visionboard.components.model_trainer import ModelTrainer
from visionboard.components.model_evaluation import ModelEvaluation

from visionboard.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

from visionboard.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from visionboard.constant.training_pipeline import (
    S3_BUCKET_NAME,
    S3_MODEL_DIR,
    SAVED_MODEL_DIR
)

class TrainingPipeline:
    """
    Complete orchestration pipeline for training, validating, and evaluating VisionBoard models
    """
    
    def __init__(self, config: Optional[TrainingPipelineConfig] = None):
        """
        Initialize TrainingPipeline with configuration
        Args:
            config: Optional TrainingPipelineConfig
        """
        self.training_pipeline_config = config or TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Start data ingestion component"""
        try:
            logging.info("Starting data ingestion stage")
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise VisionBoardException(e, sys)
        
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Start data validation component"""
        try:
            logging.info("Starting data validation stage")
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise VisionBoardException(e, sys)
        
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """Start data transformation component"""
        try:
            logging.info("Starting data transformation stage")
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise VisionBoardException(e, sys)
        
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Start model trainer component"""
        try:
            logging.info("Starting model training stage")
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise VisionBoardException(e, sys)

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        """Start model evaluation component"""
        try:
            logging.info("Starting model evaluation stage")
            model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=model_evaluation_config
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise VisionBoardException(e, sys)

    def sync_artifacts(self) -> None:
        """Sync training artifacts and models to S3 if available"""
        try:
            if not self.s3_sync.is_available:
                logging.info("S3 is not configured. Skipping artifact sync to cloud.")
                return

            logging.info("Syncing artifacts to S3")
            self.s3_sync.sync_folder_to_s3(
                folder_path=self.training_pipeline_config.artifact_dir,
                s3_prefix=f"artifacts/{self.training_pipeline_config.timestamp}"
            )
            
            saved_model_path = os.path.join(SAVED_MODEL_DIR)
            if os.path.exists(saved_model_path):
                self.s3_sync.sync_folder_to_s3(
                    folder_path=saved_model_path,
                    s3_prefix=S3_MODEL_DIR
                )
            logging.info("Successfully synced artifacts to S3")
        except Exception as e:
            logging.warning(f"S3 artifact sync warning: {str(e)}")

    def run_pipeline(self) -> ModelEvaluationArtifact:
        """
        Execute full training pipeline lifecycle
        """
        try:
            logging.info(f"{'='*30} Training Pipeline Started {'='*30}")
            
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact,
                model_trainer_artifact
            )
            
            self.sync_artifacts()
            
            logging.info(f"{'='*30} Training Pipeline Completed Successfully {'='*30}")
            return model_evaluation_artifact
            
        except Exception as e:
            logging.error(f"Training pipeline execution failed: {str(e)}")
            raise VisionBoardException(e, sys)

    def start(self) -> ModelEvaluationArtifact:
        """Alias for run_pipeline"""
        return self.run_pipeline()

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise