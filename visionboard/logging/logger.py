import os
import sys
import logging as _logging
from datetime import datetime

LOG_DIR = "logs"
CURRENT_TIMESTAMP = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
LOG_FILE_NAME = f"log_{CURRENT_TIMESTAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

os.makedirs(LOG_DIR, exist_ok=True)

def get_log_file_name() -> str:
    """Get the current log file name"""
    return LOG_FILE_NAME

def get_log_file_path() -> str:
    """Get the full path to the current log file"""
    return LOG_FILE_PATH

def _setup_logger():
    logger = _logging.getLogger("visionboard")
    logger.setLevel(_logging.INFO)
    
    formatter = _logging.Formatter("[ %(asctime)s ] %(levelname)s [%(name)s:%(lineno)d] - %(message)s")
    
    # Check if FileHandler is already attached
    if not any(isinstance(h, _logging.FileHandler) for h in logger.handlers):
        fh = _logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(_logging.INFO)
        logger.addHandler(fh)
        
    if not any(isinstance(h, _logging.StreamHandler) and not isinstance(h, _logging.FileHandler) for h in logger.handlers):
        sh = _logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.setLevel(_logging.INFO)
        logger.addHandler(sh)
        
    return logger

logging = _setup_logger()