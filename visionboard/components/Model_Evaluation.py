from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import torch
from tensorboard import program
import webbrowser

class ModelEvaluator:
    def __init__(self, weights_path: str):
        """
        Initialize ModelEvaluator
        Args:
            weights_path: Path to model weights
        """
        self.model = YOLO(weights_path)
        self.class_names = ["SignBoard"]
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        
    def validate(self, 
                data_yaml: str,
                image_size: int = 720,
                batch_size: int = 1,
                conf_threshold: float = 0.4,
                iou_threshold: float = 0.5) -> Dict:
        """
        Validate model performance
        Args:
            data_yaml: Path to data.yaml file
            image_size: Input image size
            batch_size: Batch size for validation
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        Returns:
            Dictionary containing validation metrics
        """
        results = self.model.val(
            data=data_yaml,
            imgsz=image_size,
            batch=batch_size,
            conf=conf_threshold,
            iou=iou_threshold
        )
        return results
    
    def predict(self, 
               image_path: str,
               conf_threshold: float = 0.3,
               iou_threshold: float = 0.55) -> List[Dict]:
        """
        Make predictions on a single image
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        Returns:
            List of predictions
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=True
        )
        return results
    
    def visualize_predictions(self, 
                            image: np.ndarray,
                            predictions: List[Dict],
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize predictions on image
        Args:
            image: Input image
            predictions: Model predictions
            save_path: Optional path to save visualization
        Returns:
            Image with visualized predictions
        """
        image_with_boxes = image.copy()
        
        for pred in predictions:
            boxes = pred.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = self.colors[int(cls)]
                
                cv2.rectangle(
                    image_with_boxes,
                    (x1, y1), (x2, y2),
                    color=color,
                    thickness=2
                )
                
                label = f"{self.class_names[int(cls)]} {conf:.2f}"
                cv2.putText(
                    image_with_boxes,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        if save_path:
            cv2.imwrite(save_path, image_with_boxes)
            
        return image_with_boxes
    
    def start_tensorboard(self, logdir: str = 'runs/detect') -> None:
        """
        Start TensorBoard server
        Args:
            logdir: Directory containing TensorBoard logs
        """
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir])
        url = tb.launch()
        webbrowser.open(url)
    
    def plot_results(self, results_dir: str, num_samples: int = 4) -> None:
        """
        Plot and visualize detection results
        Args:
            results_dir: Directory containing result images
            num_samples: Number of samples to visualize
        """
        plt.figure(figsize=(20, 12))
        image_paths = list(Path(results_dir).glob('*.jpg'))
        np.random.shuffle(image_paths)
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            image = plt.imread(str(image_path))
            plt.subplot(2, 2, i+1)
            plt.imshow(image)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator('runs/detect/train/weights/best.pt')
    evaluator.validate('data.yaml')
    # Test predictions and visualization
