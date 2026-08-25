import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.pipeline.training_pipeline import TrainingPipeline
from visionboard.pipeline.prediction_pipeline import PredictionPipeline
from visionboard.entity.config_entity import (
    TrainingPipelineConfig,
    ModelPredictorConfig,
    DataIngestionConfig,
    DataValidationConfig,
    ModelEvaluationConfig
)
from visionboard.components.data_ingestion import DataIngestion
from visionboard.components.data_validation import DataValidation
from visionboard.components.model_evaluation import ModelEvaluation

# Load environment variables
load_dotenv()

class VisionBoardApp:
    """
    Main application controller for VisionBoard
    """
    
    def __init__(self):
        """Initialize application"""
        try:
            logging.info(f"{'='*20} VisionBoard Application Started {'='*20}")
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
            pipeline_config = TrainingPipelineConfig.from_yaml(config_path) if config_path else TrainingPipelineConfig()
            pipeline = TrainingPipeline(config=pipeline_config)
            artifact = pipeline.start()
            logging.info("Training pipeline completed successfully.")
            print(f"\n[OK] Training completed successfully!")
            print(f"     Artifact Dir: {pipeline_config.artifact_dir}")
            if artifact.trained_model_path:
                print(f"     Trained Model: {artifact.trained_model_path}")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def start_prediction(
        self,
        input_path: str,
        config_path: Optional[str] = None,
        save_visualization: bool = True,
        enable_ocr: bool = False
    ) -> None:
        """
        Start prediction pipeline
        Args:
            input_path: Path to input image or directory
            config_path: Path to custom configuration file (optional)
            save_visualization: Whether to save prediction visualizations
            enable_ocr: Whether to run OCR on detected bounding boxes
        """
        try:
            logging.info("Starting prediction pipeline")
            
            predictor_config = ModelPredictorConfig.from_yaml(config_path) if config_path else ModelPredictorConfig()
            if enable_ocr:
                predictor_config.enable_ocr = True
            predictor_config.save_visualization = save_visualization
            
            pipeline = PredictionPipeline(config=predictor_config)
            
            if os.path.isfile(input_path):
                predictions = pipeline.predict_single(
                    image_path=input_path,
                    save_visualization=save_visualization,
                    extract_text=enable_ocr
                )
                print(f"\n[OK] Found {len(predictions)} detections in {input_path}:")
                for i, p in enumerate(predictions, 1):
                    ocr_info = f" | Text: '{p['text']}'" if p.get('text') else ""
                    print(f"  [{i}] Class: {p['class_name']} ({p['confidence']:.2f}) | Box: {p['box']}{ocr_info}")
                    
            elif os.path.isdir(input_path):
                batch_predictions = pipeline.predict_batch(
                    image_dir=input_path,
                    save_visualization=save_visualization,
                    extract_text=enable_ocr
                )
                print(f"\n[OK] Completed batch prediction on {len(batch_predictions)} images.")
                for img_p, preds in batch_predictions.items():
                    print(f"  - {os.path.basename(img_p)}: {len(preds)} detections")
            else:
                raise ValueError(f"Invalid input path: {input_path}")
                
            logging.info("Prediction pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise VisionBoardException(e, sys)

def main():
    """Main CLI entry point"""
    try:
        app = VisionBoardApp()
        
        parser = argparse.ArgumentParser(
            description="VisionBoard - YOLOv8 based object detection & OCR system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
        
        # 1. Training mode
        train_parser = subparsers.add_parser("train", help="Train YOLOv8 model")
        train_parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to config YAML")
        
        # 2. Prediction mode
        predict_parser = subparsers.add_parser("predict", help="Run detection and OCR")
        predict_parser.add_argument("input", type=str, help="Path to input image or directory")
        predict_parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to config YAML")
        predict_parser.add_argument("--no-vis", action="store_true", help="Disable prediction visualization")
        predict_parser.add_argument("--ocr", action="store_true", help="Enable OCR text recognition on detected signboards")
        
        # 3. Create Sample Dataset mode
        data_parser = subparsers.add_parser("create-dataset", help="Generate synthetic signboard dataset")
        data_parser.add_argument("--output", default="VisionBoard_Data", help="Target output folder")
        data_parser.add_argument("--count", type=int, default=6, help="Number of training samples to generate")
        
        # 4. Check Environment mode
        subparsers.add_parser("check-env", help="Run system diagnostics and verify dependencies")
        
        args = parser.parse_args()
        
        if args.mode == "train":
            app.start_training(config_path=args.config)
            
        elif args.mode == "predict":
            app.start_prediction(
                input_path=args.input,
                config_path=args.config,
                save_visualization=not args.no_vis,
                enable_ocr=args.ocr
            )
            
        elif args.mode == "create-dataset":
            from create_sample_dataset import create_dataset
            create_dataset(
                base_path=args.output,
                counts={"train": args.count, "valid": max(2, args.count // 2), "test": max(2, args.count // 2)}
            )
            
        elif args.mode == "check-env":
            from test_setup import test_setup
            test_setup()
            
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()