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
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start data ingestion component of training pipeline
        Returns:
            DataIngestionArtifact
        """
        try:
            logging.info("Starting data ingestion")
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise VisionBoardException(e, sys)
        
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Start data validation component of training pipeline
        Args:
            data_ingestion_artifact: Output from data ingestion stage
        Returns:
            DataValidationArtifact
        """
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise VisionBoardException(e, sys)
        
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Start data transformation component of training pipeline
        Args:
            data_validation_artifact: Output from data validation stage
        Returns:
            DataTransformationArtifact
        """
        try:
            logging.info("Starting data transformation")
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise VisionBoardException(e, sys)
        
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        Start model trainer component of training pipeline
        Args:
            data_transformation_artifact: Output from data transformation stage
        Returns:
            ModelTrainerArtifact
        """
        try:
            logging.info("Starting model training")
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_training()
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise VisionBoardException(e, sys)

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        """
        Start model evaluation component of training pipeline
        Args:
            data_transformation_artifact: Output from data transformation stage
            model_trainer_artifact: Output from model trainer stage
        Returns:
            ModelEvaluationArtifact
        """
        try:
            logging.info("Starting model evaluation")
            model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=model_evaluation_config
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        
        except Exception as e:
            raise VisionBoardException(e, sys)

    def sync_artifacts(self):
        """Sync all artifacts to S3"""
        try:
            logging.info("Syncing artifacts to S3")
            # Sync artifact directory
            self.s3_sync.sync_folder_to_s3(
                folder_path=self.training_pipeline_config.artifact_dir,
                s3_prefix=f"artifacts/{self.training_pipeline_config.timestamp}"
            )
            
            # Sync saved model directory if it exists
            saved_model_path = os.path.join(SAVED_MODEL_DIR)
            if os.path.exists(saved_model_path):
                self.s3_sync.sync_folder_to_s3(
                    folder_path=saved_model_path,
                    s3_prefix=S3_MODEL_DIR
                )
            logging.info("Successfully synced artifacts to S3")
        
        except Exception as e:
            raise VisionBoardException(e, sys)

    def run_pipeline(self) -> Optional[ModelEvaluationArtifact]:
        """
        Run the complete training pipeline
        Returns:
            Optional[ModelEvaluationArtifact]: Evaluation artifact if successful
        """
        try:
            logging.info("Starting training pipeline")
            
            # Data ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Data validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            
            # Data transformation
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            
            # Model training
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Model evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            
            # Sync artifacts to S3
            self.sync_artifacts()
            
            logging.info("Training pipeline completed successfully")
            return model_evaluation_artifact
        
        except Exception as e:
            raise VisionBoardException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise e