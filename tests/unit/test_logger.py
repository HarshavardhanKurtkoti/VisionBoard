import os
import pytest
from visionboard.logging.logger import logging, get_log_file_path, get_log_file_name

def test_logger_file_creation():
    """Verify log file exists and receives log messages"""
    log_path = get_log_file_path()
    assert os.path.exists(os.path.dirname(log_path))
    
    test_msg = "Test diagnostic log message 12345"
    logging.info(test_msg)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert test_msg in content
