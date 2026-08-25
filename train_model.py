import os
import sys
import argparse
from visionboard.pipeline.training_pipeline import TrainingPipeline
from visionboard.entity.config_entity import TrainingPipelineConfig

def train_model(config_path: str = "config/model_config.yaml"):
    """
    Run VisionBoard model training pipeline
    """
    print(f"Starting VisionBoard training with config: {config_path}")
    pipeline_config = TrainingPipelineConfig.from_yaml(config_path) if os.path.exists(config_path) else TrainingPipelineConfig()
    pipeline = TrainingPipeline(config=pipeline_config)
    artifact = pipeline.run_pipeline()
    print("\nTraining completed successfully!")
    print(f"Artifact status: {artifact.message}")
    if artifact.trained_model_path:
        print(f"Trained model saved at: {artifact.trained_model_path}")
    return artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionBoard model")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to config yaml")
    args = parser.parse_args()
    train_model(args.config)
