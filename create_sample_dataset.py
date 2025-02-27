import os
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_signboard(size=(640, 640)):
    """Create a synthetic signboard image with a white background and black text"""
    # Create white background
    image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Create a gray signboard
    x1, y1 = size[0]//4, size[1]//4
    x2, y2 = 3*size[0]//4, 3*size[1]//4
    cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # Add some text
    text = "SIGN"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (x1 + x2 - text_size[0]) // 2
    text_y = (y1 + y2 + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Calculate YOLO format annotations (class_id, x_center, y_center, width, height)
    x_center = (x1 + x2) / (2 * size[0])
    y_center = (y1 + y2) / (2 * size[1])
    width = (x2 - x1) / size[0]
    height = (y2 - y1) / size[1]
    
    return image, f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def create_dataset(base_path, num_images={"train": 5, "valid": 2, "test": 2}):
    """Create a sample dataset with the specified number of images per split"""
    base_path = Path(base_path)
    print("\nCreating sample dataset...")
    
    for split, count in num_images.items():
        print(f"\nGenerating {split} set:")
        images_dir = base_path / split / "images"
        labels_dir = base_path / split / "labels"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images and labels
        for i in range(count):
            image_path = images_dir / f"signboard_{i+1:03d}.jpg"
            label_path = labels_dir / f"signboard_{i+1:03d}.txt"
            
            # Create synthetic image and label
            image, label = create_synthetic_signboard()
            
            # Save image and label
            cv2.imwrite(str(image_path), image)
            with open(label_path, 'w') as f:
                f.write(label)
            
            print(f"  Created sample {i+1}/{count}")

if __name__ == "__main__":
    # Create the sample dataset
    create_dataset("VisionBoard_Data")
    print("\nDataset creation complete! Structure:")
    print("VisionBoard_Data/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── valid/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── test/")
    print("    ├── images/")
    print("    └── labels/")
