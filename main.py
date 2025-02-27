import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.pipeline.training_pipeline import TrainingPipeline
from visionboard.pipeline.prediction_pipeline import PredictionPipeline
from visionboard.entity.config_entity import (
    TrainingPipelineConfig,
    ModelPredictorConfig
)

# Load environment variables
load_dotenv()

class VisionBoardApp:
    """
    Main application class for VisionBoard
    """
    
    def __init__(self):
        """Initialize application"""
        try:
            logging.info(f"{'='*20}VisionBoard Application Started{'='*20}")
            
        except Exception as e:
            logging.error(f"Error initializing VisionBoard app: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def start_training(self, config_path: Optional[str] = None) -> None:
        """
        Start model training pipeline
        Args:
            config_path: Path to custom configuration file (optional)
        """
        try:
            logging.info("Starting training pipeline")
            
            # Initialize training pipeline
            pipeline_config = TrainingPipelineConfig.from_yaml(config_path) if config_path else TrainingPipelineConfig()
            pipeline = TrainingPipeline(config=pipeline_config)
            
            # Run pipeline
            pipeline.start()
            logging.info("Training pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def start_prediction(
        self,
        input_path: str,
        config_path: Optional[str] = None,
        save_visualization: bool = True
    ) -> None:
        """
        Start prediction pipeline
        Args:
            input_path: Path to input image or directory
            config_path: Path to custom configuration file (optional)
            save_visualization: Whether to save prediction visualizations
        """
        try:
            logging.info("Starting prediction pipeline")
            
            # Initialize prediction pipeline
            predictor_config = ModelPredictorConfig.from_yaml(config_path) if config_path else ModelPredictorConfig()
            pipeline = PredictionPipeline(config=predictor_config)
            
            # Run predictions
            if os.path.isfile(input_path):
                # Single image prediction
                predictions = pipeline.predict_single(
                    image_path=input_path,
                    save_visualization=save_visualization
                )
                logging.info(f"Predictions for {input_path}: {predictions}")
                
            elif os.path.isdir(input_path):
                # Batch prediction
                predictions = pipeline.predict_batch(
                    image_dir=input_path,
                    save_visualization=save_visualization
                )
                logging.info(f"Completed batch prediction on {len(predictions)} images")
                
            else:
                raise ValueError(f"Invalid input path: {input_path}")
            
            logging.info("Prediction pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise VisionBoardException(e, sys)

def main():
    """Main entry point"""
    try:
        # Initialize app
        app = VisionBoardApp()
        
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="VisionBoard - YOLOv8 based object detection")
        
        # Add subparsers for different modes
        subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
        
        # Training mode parser
        train_parser = subparsers.add_parser("train", help="Train model")
        train_parser.add_argument(
            "--config",
            type=str,
            help="Path to training configuration file"
        )
        
        # Prediction mode parser
        predict_parser = subparsers.add_parser("predict", help="Run predictions")
        predict_parser.add_argument(
            "input",
            type=str,
            help="Path to input image or directory"
        )
        predict_parser.add_argument(
            "--config",
            type=str,
            help="Path to prediction configuration file"
        )
        predict_parser.add_argument(
            "--no-vis",
            action="store_true",
            help="Disable prediction visualization"
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Execute requested operation
        if args.mode == "train":
            app.start_training(config_path=args.config)
            
        elif args.mode == "predict":
            app.start_prediction(
                input_path=args.input,
                config_path=args.config,
                save_visualization=not args.no_vis
            )
            
        else:
            parser.print_help()
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise VisionBoardException(e, sys)

if __name__ == "__main__":
    main()