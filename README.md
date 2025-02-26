# VisionBoard

A YOLOv8-based object detection system for signboard detection and analysis.

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/HarshavardhanKurtkoti/VisionBoard.git
cd VisionBoard
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r Requirements.txt
```

4. **Environment Setup**
Create a `.env` file in the root directory with the following variables:
```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
S3_BUCKET_NAME=your_bucket_name

# Model Configuration
MODEL_PATH=path/to/model
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
IMG_SIZE=640

# Data Configuration
DATA_DIR=VisionBoard_Data
TRAIN_DIR=train
TEST_DIR=test
```

5. **Run the Application**

For training:
```bash
python main.py train --config path/to/config.yaml
```

For prediction:
```bash
python main.py predict input/path --config path/to/config.yaml
```

## Project Structure

```
VisionBoard/
├── visionboard/              # Main package directory
│   ├── components/          # Core components
│   ├── entity/             # Data entities and configurations
│   ├── exception/          # Custom exceptions
│   ├── logging/           # Logging configuration
│   ├── pipeline/          # Training and prediction pipelines
│   └── utils/             # Utility functions
├── VisionBoard_Data/       # Data directory
├── notebooks/             # Jupyter notebooks
├── tests/                # Test files
├── .env                  # Environment variables
├── Requirements.txt      # Project dependencies
├── setup.py             # Package setup
└── main.py              # Application entry point
```

## Docker Support

Build and run using Docker:
```bash
# Build image
docker build -t visionboard .

# Run training
docker run -v $(pwd)/VisionBoard_Data:/app/VisionBoard_Data visionboard python main.py train

# Run prediction
docker run -v $(pwd)/VisionBoard_Data:/app/VisionBoard_Data visionboard python main.py predict /app/VisionBoard_Data/test
```

## License

MIT License
