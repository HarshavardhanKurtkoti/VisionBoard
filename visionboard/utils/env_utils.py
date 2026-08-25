import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get environment variable with error handling
    Args:
        key: Environment variable key
        default: Default value if key not found
        required: Whether to raise an error if key not found
    Returns:
        str: Environment variable value
    """
    value = os.getenv(key, default)
    if value is None and required:
        raise ValueError(f"Required environment variable '{key}' is missing.")
    return value or ""

def get_aws_config() -> Dict[str, str]:
    """Get AWS configuration from environment variables"""
    return {
        "aws_access_key_id": get_env_var("AWS_ACCESS_KEY_ID", ""),
        "aws_secret_access_key": get_env_var("AWS_SECRET_ACCESS_KEY", ""),
        "region_name": get_env_var("AWS_DEFAULT_REGION", "us-east-1"),
        "bucket_name": get_env_var("S3_BUCKET_NAME", "visionboard-data")
    }

def get_model_config() -> Dict[str, Any]:
    """Get model configuration from environment variables"""
    return {
        "model_path": get_env_var("MODEL_PATH", "yolov8n.pt"),
        "confidence_threshold": float(get_env_var("CONFIDENCE_THRESHOLD", "0.25")),
        "iou_threshold": float(get_env_var("IOU_THRESHOLD", "0.45")),
        "img_size": int(get_env_var("IMG_SIZE", "640"))
    }

def get_data_config() -> Dict[str, str]:
    """Get data configuration from environment variables"""
    return {
        "data_dir": get_env_var("DATA_DIR", "VisionBoard_Data"),
        "train_dir": get_env_var("TRAIN_DIR", "train"),
        "test_dir": get_env_var("TEST_DIR", "test")
    }
