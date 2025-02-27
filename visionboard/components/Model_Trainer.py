import os
import sys
from dotenv import load_dotenv
from visionboard.utils.env_utils import get_model_config
from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelTrainerConfig
from visionboard.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact
)
from visionboard.utils.ml_utils.model.estimator import YOLOModel

# Load environment variables
load_dotenv()

class ModelTrainer:
    """
    Class for handling model training operations
    """
    
    def __init__(
        self,
        config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact
    ):
        """
        Initialize with configuration
        Args:
            config: Configuration for model training
            transformation_artifact: Artifact from data transformation
        """
        try:
            logging.info(f"{'='*20}Model Training log started.{'='*20}")
            self.config = config
            self.transformation_artifact = transformation_artifact
            
        except Exception as e:
            logging.error(f"Error in ModelTrainer.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def create_data_yaml(self) -> str:
        """
        Create YAML file for YOLOv8 training
        Returns:
            str: Path to created YAML file
        """
        try:
            logging.info("Creating data YAML file")
            
            yaml_path = os.path.join(self.config.model_trainer_dir, "data.yaml")
            
            yaml_content = {
                "path": os.path.abspath(self.transformation_artifact.transformed_train_file_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": self.config.num_classes,
                "names": self.config.class_names
            }
            
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
            
            logging.info(f"Created data YAML at: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            logging.error(f"Error creating data YAML: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def train_model(self, data_yaml_path: str) -> str:
        """
        Train YOLOv8 model
        Args:
            data_yaml_path: Path to data YAML file
        Returns:
            str: Path to trained model weights
        """
        try:
            logging.info("Starting model training")
            
            # Initialize model
            model = YOLOModel(self.config.pretrained_model_path)
            
            # Train model
            best_model_path = model.train(
                config=self.config,
                train_data=data_yaml_path,
                val_data=None  # Using validation split from data.yaml
            )
            
            logging.info(f"Model training completed. Best model: {best_model_path}")
            return best_model_path
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def validate_trained_model(
        self,
        model_path: str,
        data_yaml_path: str
    ) -> float:
        """
        Validate trained model
        Args:
            model_path: Path to model weights
            data_yaml_path: Path to data YAML
        Returns:
            float: Model accuracy (mAP50)
        """
        try:
            logging.info("Starting model validation")
            
            # Initialize model with trained weights
            model = YOLOModel(model_path)
            
            # Evaluate model
            metrics = model.evaluate(
                val_data=data_yaml_path,
                conf_thres=self.config.conf_threshold,
                iou_thres=self.config.iou_threshold
            )
            
            logging.info(f"Model validation completed. Metrics: {metrics.to_dict()}")
            return metrics.map50
            
        except Exception as e:
            logging.error(f"Error validating model: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate model training process
        Returns:
            ModelTrainerArtifact: Model training results
        """
        try:
            logging.info("Starting model training pipeline")
            
            # Create data YAML
            data_yaml_path = self.create_data_yaml()
            
            # Train model
            trained_model_path = self.train_model(data_yaml_path)
            
            # Validate model
            model_accuracy = self.validate_trained_model(
                trained_model_path,
                data_yaml_path
            )
            
            # Check if model meets acceptance criteria
            is_model_accepted = model_accuracy >= self.config.model_acceptance_threshold
            
            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model training completed.",
                trained_model_file_path=trained_model_path,
                train_metric_artifact=None,  # TODO: Add training metrics
                test_metric_artifact=None,   # TODO: Add test metrics
                is_model_accepted=is_model_accepted
            )
            
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise VisionBoardException(e, sys)
