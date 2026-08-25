import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from visionboard.utils.main_utils.image_utils import save_image, draw_box_and_label

def create_synthetic_signboard(
    text: str = "STOP",
    size: Tuple[int, int] = (640, 640),
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    board_color: Tuple[int, int, int] = (0, 0, 200)
) -> Tuple[np.ndarray, str]:
    """
    Create a synthetic image containing a colored signboard with text and return YOLO label
    Args:
        text: Text to write on the signboard
        size: (width, height)
        bg_color: Background color (BGR)
        board_color: Signboard color (BGR)
    Returns:
        Tuple of (image array, YOLO format label string)
    """
    w, h = size
    image = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    # Signboard coordinates
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.25)
    x1, y1 = margin_x, margin_y
    x2, y2 = w - margin_x, h - margin_y
    
    # Draw signboard rectangle
    image = draw_box_and_label(image, [x1, y1, x2, y2], label=text, color=board_color)
    
    # Calculate YOLO format annotations: class_id x_center y_center width height (normalized)
    x_center = (x1 + x2) / (2.0 * w)
    y_center = (y1 + y2) / (2.0 * h)
    box_w = (x2 - x1) / float(w)
    box_h = (y2 - y1) / float(h)
    
    label = f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
    return image, label

def create_dataset(
    base_path: str = "VisionBoard_Data",
    counts: Optional[Dict[str, int]] = None
) -> None:
    """
    Generate sample dataset with train, valid, and test splits
    """
    if counts is None:
        counts = {"train": 6, "valid": 3, "test": 3}
        
    sample_texts = ["SPEED 50", "STOP", "CAUTION", "EXIT", "PARKING", "WAY OUT", "ONE WAY", "NO ENTRY"]
    base = Path(base_path)
    print(f"\nGenerating sample VisionBoard dataset at: {base.resolve()}")
    
    for split, count in counts.items():
        images_dir = base / split / "images"
        labels_dir = base / split / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            text = sample_texts[(i + len(split)) % len(sample_texts)]
            image, label = create_synthetic_signboard(text=text)
            
            img_file = images_dir / f"signboard_{split}_{i+1:03d}.jpg"
            lbl_file = labels_dir / f"signboard_{split}_{i+1:03d}.txt"
            
            save_image(str(img_file), image)
            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write(label + "\n")
                
        print(f"  Generated {count} samples for {split} split")
        
    print(f"\nDataset creation completed successfully at {base}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic VisionBoard dataset for training/testing")
    parser.add_argument("--output", default="VisionBoard_Data", help="Base directory for dataset")
    parser.add_argument("--train-count", type=int, default=6, help="Number of train samples")
    parser.add_argument("--valid-count", type=int, default=3, help="Number of valid samples")
    parser.add_argument("--test-count", type=int, default=3, help="Number of test samples")
    
    args = parser.parse_args()
    create_dataset(
        base_path=args.output,
        counts={"train": args.train_count, "valid": args.valid_count, "test": args.test_count}
    )
