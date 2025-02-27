import os
from dotenv import load_dotenv
from ultralytics import YOLO
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_setup():
    """Test if the setup is working correctly"""
    try:
        print("\n=== Testing VisionBoard Setup ===\n")
        
        # Load environment variables
        load_dotenv()
        print(" Environment variables loaded")
        
        # Get base directory
        base_dir = Path(__file__).parent
        
        # Test model loading
        model_path = os.path.join(base_dir, os.getenv("MODEL_PATH"))
        print(f"\nChecking model:")
        print(f"- Model path: {model_path}")
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(" Model loaded successfully")
        else:
            print(" Model file not found!")
            print(f"  Checking alternative path: {os.path.join(base_dir, 'visionboard/models/yolov8m.pt')}")
            alt_path = os.path.join(base_dir, 'visionboard/models/yolov8m.pt')
            if os.path.exists(alt_path):
                model = YOLO(alt_path)
                print(" Model loaded successfully from alternative path")
            else:
                print(" Model not found in alternative path either!")
        
        # Test data paths
        data_dir = os.path.join(base_dir, os.getenv("DATA_DIR"))
        train_dir = os.path.join(data_dir, os.getenv("TRAIN_DIR"))
        test_dir = os.path.join(data_dir, os.getenv("TEST_DIR"))
        
        print(f"\nChecking directories:")
        print(f"- Data directory: {data_dir}")
        print(f"- Train directory: {train_dir}")
        print(f"- Test directory: {test_dir}")
        
        # Check if directories exist and their contents
        if os.path.exists(data_dir):
            print(" Data directory exists")
            # Check train directory
            if os.path.exists(train_dir):
                train_images = os.path.join(train_dir, "images")
                train_labels = os.path.join(train_dir, "labels")
                print(f"  Train directory exists")
                print(f"    - Images dir: {'' if os.path.exists(train_images) else ''}")
                print(f"    - Labels dir: {'' if os.path.exists(train_labels) else ''}")
            else:
                print(" Train directory not found!")
                
            # Check test directory
            if os.path.exists(test_dir):
                test_images = os.path.join(test_dir, "images")
                test_labels = os.path.join(test_dir, "labels")
                print(f"  Test directory exists")
                print(f"    - Images dir: {'' if os.path.exists(test_images) else ''}")
                print(f"    - Labels dir: {'' if os.path.exists(test_labels) else ''}")
            else:
                print(" Test directory not found!")
        else:
            print(" Data directory not found!")
        
        print("\n=== Setup Check Complete ===")
        
    except Exception as e:
        print(f"\n Error during setup test: {str(e)}")
        raise

if __name__ == "__main__":
    test_setup()
