import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from visionboard.entity.config_entity import ModelPredictorConfig
from visionboard.components.model_predictor import ModelPredictor
from visionboard.logging.logger import logging

class SignboardDetectorReader:
    """
    Convenience wrapper for signboard detection and text recognition
    """
    def __init__(self, model_path: str = "yolov8n.pt", enable_ocr: bool = True):
        self.config = ModelPredictorConfig(
            model_path=model_path,
            enable_ocr=enable_ocr,
            save_visualization=True,
            visualization_dir="output_images"
        )
        self.predictor = ModelPredictor(config=self.config)
        
    def process_image(self, image_path: str, conf_threshold: float = 0.25) -> Tuple[List[Dict[str, Any]], str]:
        """
        Detect signboards and extract text
        """
        self.config.conf_threshold = conf_threshold
        predictions, vis_path = self.predictor.predict_image(
            image_path=image_path,
            save_visualization=True,
            extract_text=True
        )
        return predictions, str(vis_path)

def main():
    parser = argparse.ArgumentParser(description="Signboard Detection and Text Recognition")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    detector = SignboardDetectorReader(model_path=args.model, enable_ocr=True)
    
    target_image = args.image
    if not target_image:
        # Check sample dataset
        candidates = list(Path("VisionBoard_Data").glob("**/*.jpg"))
        if candidates:
            target_image = str(candidates[0])
            print(f"No image provided. Using sample: {target_image}")
        else:
            print("Please specify an image with --image <path>")
            return
            
    try:
        print(f"Processing image: {target_image}")
        detections, output_file = detector.process_image(target_image, conf_threshold=args.conf)
        
        print(f"\nFound {len(detections)} signboards:")
        for i, det in enumerate(detections, 1):
            print(f"\nSignboard {i}:")
            print(f"  Location: {det['box']}")
            print(f"  Confidence: {det['confidence']:.2f}")
            print(f"  Class: {det['class_name']}")
            print(f"  Text: {det.get('text', '')}")
            print("-" * 40)
            
        print(f"\nAnnotated visualization saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
