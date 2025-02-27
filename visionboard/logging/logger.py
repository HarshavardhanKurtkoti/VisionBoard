import os
import sys
import logging
from datetime import datetime
from typing import Optional

LOG_DIR = "logs"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

os.makedirs(LOG_DIR, exist_ok=True)

def get_log_file_name() -> str:
    """Get the current log file name"""
    return LOG_FILE_NAME

def get_log_file_path() -> str:
    """Get the full path to the current log file"""
    return LOG_FILE_PATH

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)