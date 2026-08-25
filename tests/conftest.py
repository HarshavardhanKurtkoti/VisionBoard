import os
import sys
import shutil
import pytest
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visionboard.utils.main_utils.image_utils import save_image, draw_box_and_label

@pytest.fixture
def temp_test_dir(tmp_path):
    """Provide a clean temporary directory for test artifacts"""
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield str(test_dir)
    if test_dir.exists():
        shutil.rmtree(str(test_dir), ignore_errors=True)

@pytest.fixture
def sample_dataset_dir(temp_test_dir):
    """Generate a valid synthetic dataset structure with images and labels"""
    data_dir = os.path.join(temp_test_dir, "VisionBoard_Data")
    
    for split in ["train", "valid", "test"]:
        imgs_dir = os.path.join(data_dir, split, "images")
        lbls_dir = os.path.join(data_dir, split, "labels")
        os.makedirs(imgs_dir, exist_ok=True)
        os.makedirs(lbls_dir, exist_ok=True)
        
        # Create 2 sample images and labels per split
        for i in range(2):
            img = np.full((320, 320, 3), (200, 200, 200), dtype=np.uint8)
            img = draw_box_and_label(img, [50, 50, 250, 250], label="TEST", color=(0, 0, 255))
            
            img_path = os.path.join(imgs_dir, f"sample_{split}_{i}.jpg")
            lbl_path = os.path.join(lbls_dir, f"sample_{split}_{i}.txt")
            
            save_image(img_path, img)
            # YOLO format: 0 x_center y_center width height
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("0 0.468750 0.468750 0.625000 0.625000\n")
                
    return data_dir

@pytest.fixture
def sample_image_path(temp_test_dir):
    """Generate a single sample image file"""
    img = np.full((320, 320, 3), (240, 240, 240), dtype=np.uint8)
    img = draw_box_and_label(img, [40, 40, 280, 280], label="SIGN", color=(0, 128, 255))
    
    img_path = os.path.join(temp_test_dir, "test_sign.jpg")
    save_image(img_path, img)
    return img_path
