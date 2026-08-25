import os
import sys
from typing import Dict, Any, Optional

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.entity.config_entity import ModelEvaluationConfig
from visionboard.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact,
    DetectionMetricArtifact
)
from visionboard.utils.ml_utils.model.estimator import YOLOModel
from visionboard.utils.main_utils.utils import create_directories, write_yaml_file

class ModelEvaluation:
    """
    Component for evaluating trained model against thresholds and baseline models
    """
    
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig
    ):
        """
        Initialize ModelEvaluation
        Args:
            data_transformation_artifact: Artifact from data transformation
            model_trainer_artifact: Artifact from model trainer
            model_evaluation_config: Configuration for model evaluation
        """
        try:
            logging.info(f"{'='*20}Model Evaluation component started.{'='*20}")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
            
        except Exception as e:
            logging.error(f"Error in ModelEvaluation.__init__: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiate complete model evaluation stage
        """
        try:
            logging.info("Initiating Model Evaluation stage")
            create_directories([self.model_evaluation_config.model_evaluation_dir])
            
            data_yaml = self.data_transformation_artifact.transformed_data_yaml_path
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            
            # Evaluate trained model
            model = YOLOModel(trained_model_path)
            trained_metrics = model.evaluate(
                val_data=data_yaml,
                conf_thres=self.model_evaluation_config.conf_threshold,
                iou_thres=self.model_evaluation_config.iou_threshold
            )
            
            # Acceptance criteria
            min_acc = self.model_evaluation_config.min_accuracy
            is_model_accepted = trained_metrics.map50 >= min_acc
            improved_accuracy = float(trained_metrics.map50 - min_acc)
            
            report = {
                "trained_model_path": trained_model_path,
                "is_model_accepted": is_model_accepted,
                "min_accuracy_threshold": min_acc,
                "metrics": trained_metrics.to_dict(),
                "improved_accuracy": improved_accuracy
            }
            
            report_path = self.model_evaluation_config.evaluation_report_file_path
            write_yaml_file(report_path, report, replace=True)
            
            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=trained_model_path if is_model_accepted else None,
                trained_model_path=trained_model_path,
                train_model_metric_artifact=trained_metrics,
                best_model_metric_artifact=trained_metrics if is_model_accepted else None,
                evaluation_report_path=report_path,
                message="Model accepted for deployment." if is_model_accepted else "Model did not meet accuracy threshold."
            )
            
            logging.info(f"Model evaluation completed with artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in initiate_model_evaluation: {str(e)}")
            raise VisionBoardException(e, sys)

# Backward-compatibility wrapper class
class ModelEvaluator:
    def __init__(self, weights_path: str):
        self.model = YOLOModel(weights_path)
        
    def validate(self, data_yaml: str, **kwargs) -> Dict:
        return self.model.evaluate(data_yaml, **kwargs).to_dict()
        
    def predict(self, image_path: str, **kwargs):
        return self.model.predict(image_path, **kwargs)
