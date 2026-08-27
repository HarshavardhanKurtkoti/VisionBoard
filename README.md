# VisionBoard: Signboard Detection & OCR System

VisionBoard is a production-ready computer vision and MLOps system built for automated signboard object detection (via YOLOv8) and text recognition (via Tesseract OCR).

---

## Key Features

- **End-to-End MLOps Pipeline**:
  1. **Data Ingestion**: Robust local and AWS S3 dataset synchronization.
  2. **Data Validation**: Automated schema validation, corrupted image checks, and normalized coordinate integrity checking with YAML reporting.
  3. **Data Transformation**: Image augmentations and YOLO dataset configuration generator.
  4. **Model Training**: Multi-device training (CPU / CUDA GPU auto-detection) with YOLOv8 checkpointing and artifact tracking.
  5. **Model Evaluation**: Comprehensive mAP@0.5, mAP@0.5:0.95, precision, and recall calculation with acceptance thresholds.
  6. **Model Prediction & OCR**: Single-image and batch inference pipeline with bounding box annotation and Tesseract text recognition.
- **Production-Grade Reliability**:
  - Robust exception handling and dual console/file logging.
  - Portable relative and environment-based path resolution (no hardcoded machine paths).
  - Graceful fallbacks for AWS credentials and OCR binaries.
- **Testing & CI/CD**:
  - Full `pytest` unit and integration test suite with synthetic fixtures.
  - Multi-stage Docker container support.
  - GitHub Actions CI workflow for automated testing on push and pull request.

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/HarshavardhanKurtkoti/VisionBoard.git
cd VisionBoard

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory (optional, defaults provided):

```env
# Model Configuration
MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
IMG_SIZE=640

# Data Configuration
DATA_DIR=VisionBoard_Data
TRAIN_DIR=train
TEST_DIR=test

# Optional OCR / AWS Configuration
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your_bucket
```

### 3. Check System Diagnostics

```bash
python main.py check-env
```

### 4. Generate Sample Dataset

```bash
python main.py create-dataset --count 10
```

### 5. Run Training Pipeline

```bash
python main.py train --config config/model_config.yaml
```

### 6. Run Predictions (with optional OCR)

```bash
# Single image prediction with OCR
python main.py predict VisionBoard_Data/test/images/signboard_test_001.jpg --ocr

# Batch directory prediction
python main.py predict VisionBoard_Data/test/images/ --ocr
```

---

## Project Structure

```
VisionBoard/
├── visionboard/
│   ├── cloud/               # AWS S3 synchronization
│   ├── components/          # Pipeline components (Ingestion, Validation, Transformation, Trainer, Evaluation, Predictor)
│   ├── constant/            # Pipeline constants and defaults
│   ├── entity/              # Typed config and artifact dataclasses
│   ├── exception/           # Custom exception handling
│   ├── logging/             # Dual console/file logger
│   ├── ocr/                 # Tesseract OCR signboard text extraction
│   ├── pipeline/            # Training and Prediction pipeline orchestrators
│   └── utils/               # IO, YAML, array, and metric utilities
├── config/                  # data.yaml & model_config.yaml
├── tests/                   # Pytest unit and integration test suite
├── create_sample_dataset.py # Synthetic dataset generator
├── test_setup.py            # Environment & hardware diagnostics
├── main.py                  # Production CLI entry point
├── requirements.txt         # Project dependencies
├── Dockerfile               # Production container image
├── render.yaml              # Render Deployment Blueprint
└── README.md
```

---

## Running Tests

```bash
# Run all unit and integration tests
pytest -v tests/

# Run with test coverage
pytest -v --cov=visionboard tests/
```

---

## Docker Support

```bash
# Build image
docker build -t visionboard .

# Run diagnostics
docker run --rm visionboard python main.py check-env

# Run training
docker run --rm -v $(pwd)/VisionBoard_Data:/app/VisionBoard_Data visionboard python main.py train
```

---

## License

MIT License
