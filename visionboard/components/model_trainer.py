import os
import sys
import shutil
from typing import Optional

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelTrainerConfig
from visionboard.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    DetectionMetricArtifact
)
from visionboard.utils.ml_utils.model.estimator import YOLOModel
from visionboard.utils.main_utils.utils import create_directories

class ModelTrainer:
    """
    Component for handling YOLOv8 model training and checkpoint management
    """
    
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """
        Initialize ModelTrainer
        Args:
            data_transformation_artifact: Artifact from data transformation stage
            model_trainer_config: Configuration for model training
        """
        try:
            logging.info(f"{'='*20}Model Trainer component started.{'='*20}")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            
        except Exception as e:
            logging.error(f"Error in ModelTrainer.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def train(self) -> str:
        """
        Execute training loop via YOLOModel estimator
        """
        try:
            logging.info("Starting model training execution")
            create_directories([
                self.model_trainer_config.model_trainer_dir,
                self.model_trainer_config.trained_model_dir
            ])
            
            data_yaml = self.data_transformation_artifact.transformed_data_yaml_path
            model = YOLOModel(self.model_trainer_config.pretrained_model_path)
            
            trained_model_weights = model.train(
                config=self.model_trainer_config,
                train_data=data_yaml
            )
            
            # Copy or save to standard trained_model_path
            dst_path = self.model_trainer_config.trained_model_path
            if os.path.exists(trained_model_weights) and os.path.abspath(trained_model_weights) != os.path.abspath(dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(trained_model_weights, dst_path)
            
            logging.info(f"Model trained and saved to: {dst_path}")
            return dst_path
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def evaluate_trained_model(self, model_path: str) -> DetectionMetricArtifact:
        """
        Evaluate newly trained model weights on validation split
        """
        try:
            logging.info(f"Evaluating trained model at {model_path}")
            data_yaml = self.data_transformation_artifact.transformed_data_yaml_path
            model = YOLOModel(model_path)
            
            metrics = model.evaluate(
                val_data=data_yaml,
                conf_thres=self.model_trainer_config.conf_threshold,
                iou_thres=self.model_trainer_config.iou_threshold
            )
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating trained model: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate complete model training component
        """
        try:
            logging.info("Initiating Model Trainer stage")
            
            trained_model_path = self.train()
            val_metrics = self.evaluate_trained_model(trained_model_path)
            
            is_accepted = val_metrics.map50 >= self.model_trainer_config.model_acceptance_threshold
            
            artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_path,
                train_metric_artifact=val_metrics,
                test_metric_artifact=val_metrics,
                is_trained=True,
                is_model_accepted=is_accepted,
                message="Model training completed successfully."
            )
            
            logging.info(f"Model training completed with artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise VisionBoardException(e, sys)
    
    # Method alias for backward compatibility
    initiate_model_training = initiate_model_trainer
