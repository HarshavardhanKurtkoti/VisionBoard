import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_setup():
    """Comprehensive environment and dependency diagnostics for VisionBoard"""
    print("\n" + "="*50)
    print("        VisionBoard Setup & Environment Diagnostics")
    print("="*50 + "\n")
    
    # 1. Python runtime
    print(f"[x] Python Version: {sys.version.split()[0]}")
    base_dir = Path(__file__).parent.resolve()
    print(f"[x] Project Root: {base_dir}")
    
    # 2. Key Package Checks
    print("\n--- Checking Dependencies ---")
    packages = [
        ("numpy", "numpy"),
        ("OpenCV", "cv2"),
        ("PyYAML", "yaml"),
        ("Pillow", "PIL"),
        ("PyTorch", "torch"),
        ("Ultralytics YOLO", "ultralytics"),
        ("PyTesseract", "pytesseract"),
        ("Boto3", "boto3"),
        ("Pandas", "pandas"),
        ("Scikit-Learn", "sklearn")
    ]
    
    for name, module_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  [OK] {name:<20}: {ver}")
        except ImportError:
            print(f"  [--] {name:<20}: Not Installed (Optional/Install via pip)")
            
    # 3. Hardware & Acceleration
    print("\n--- Checking Hardware Acceleration ---")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  Device Count: {torch.cuda.device_count()}")
        else:
            print(f"  Running on CPU with {os.cpu_count()} logical cores")
    except Exception as e:
        print(f"  PyTorch check skipped: {e}")
        
    # 4. OCR Engine
    print("\n--- Checking OCR Engine (Tesseract) ---")
    try:
        from visionboard.ocr.text_recognition import SignboardTextReader
        reader = SignboardTextReader()
        if reader.tesseract_available:
            print("  [OK] Tesseract OCR is available and ready.")
        else:
            print("  [--] Tesseract OCR binary not detected (OCR text extraction will be skipped).")
    except Exception as e:
        print(f"  OCR check error: {e}")
        
    # 5. Pretrained Weights
    print("\n--- Checking Model Weights ---")
    model_paths = [
        "yolov8n.pt",
        "visionboard/models/yolov8m.pt"
    ]
    for p in model_paths:
        full_p = base_dir / p
        if full_p.exists():
            size_mb = full_p.stat().st_size / (1024 * 1024)
            print(f"  [OK] Found model weight: {p} ({size_mb:.2f} MB)")
        else:
            print(f"  [--] Model weight not found: {p}")
            
    # 6. Data Directory Structure
    print("\n--- Checking Data Directory Structure ---")
    data_dir = base_dir / "VisionBoard_Data"
    if data_dir.exists():
        print(f"  [OK] Data directory found: {data_dir}")
        for split in ["train", "valid", "test"]:
            split_dir = data_dir / split
            if split_dir.exists():
                imgs = len(list((split_dir / "images").glob("*.*"))) if (split_dir / "images").exists() else 0
                lbls = len(list((split_dir / "labels").glob("*.txt"))) if (split_dir / "labels").exists() else 0
                print(f"       - {split:<6}: {imgs} images, {lbls} labels")
            else:
                print(f"       - {split:<6}: missing")
    else:
        print(f"  [--] Data directory {data_dir} does not exist yet (create with 'python main.py create-dataset')")
        
    print("\n" + "="*50)
    print("Diagnostics check complete.")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_setup()
