import os
import sys
import yaml
import json
import dill
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging

def read_yaml_file(file_path: str) -> Dict:
    """
    Read a YAML file and return its contents as a dictionary
    Args:
        file_path: Path to YAML file
    Returns:
        Dict: Contents of YAML file
    """
    try:
        logging.info(f"Reading YAML file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at {file_path}")
            
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
            logging.info(f"Successfully read YAML file: {file_path}")
            return content
            
    except Exception as e:
        logging.error(f"Error reading YAML file {file_path}: {str(e)}")
        raise VisionBoardException(e, sys)

def write_yaml_file(file_path: str, content: Dict, replace: bool = False) -> None:
    """
    Write content to a YAML file
    Args:
        file_path: Path to YAML file
        content: Content to write
        replace: Whether to replace existing file
    """
    try:
        logging.info(f"Writing YAML file: {file_path}")
        if os.path.exists(file_path) and not replace:
            raise FileExistsError(f"File already exists at {file_path} and replace=False")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(content, f, default_flow_style=False)
            logging.info(f"Successfully wrote YAML file: {file_path}")
            
    except Exception as e:
        logging.error(f"Error writing YAML file {file_path}: {str(e)}")
        raise VisionBoardException(e, sys)

def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array to file
    Args:
        file_path: Path to save file
        array: Numpy array to save
    """
    try:
        logging.info(f"Saving numpy array to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            np.save(f, array)
            logging.info(f"Successfully saved numpy array to {file_path}")
            
    except Exception as e:
        logging.error(f"Error saving numpy array to {file_path}: {str(e)}")
        raise VisionBoardException(e, sys)

def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load numpy array from file
    Args:
        file_path: Path to numpy file
    Returns:
        np.ndarray: Loaded numpy array
    """
    try:
        logging.info(f"Loading numpy array from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
            
        with open(file_path, 'rb') as f:
            array = np.load(f)
            logging.info(f"Successfully loaded numpy array from {file_path}")
            return array
            
    except Exception as e:
        logging.error(f"Error loading numpy array from {file_path}: {str(e)}")
        raise VisionBoardException(e, sys)

def save_object(file_path: str, obj: Any) -> None:
    """
    Save Python object to file using dill
    Args:
        file_path: Path to save file
        obj: Python object to save
    """
    try:
        logging.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
            logging.info(f"Successfully saved object to {file_path}")
            
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {str(e)}")
        raise VisionBoardException(e, sys)

def load_object(file_path: str) -> Any:
    """
    Load Python object from file using dill
    Args:
        file_path: Path to object file
    Returns:
        Any: Loaded Python object
    """
    try:
        logging.info(f"Loading object from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
            
        with open(file_path, 'rb') as f:
            obj = dill.load(f)
            logging.info(f"Successfully loaded object from {file_path}")
            return obj
            
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {str(e)}")
        raise VisionBoardException(e, sys)

def create_directories(directories: List[str], exist_ok: bool = True) -> None:
    """
    Create directories if they don't exist
    Args:
        directories: List of directory paths
        exist_ok: Whether to ignore if directory exists
    """
    try:
        logging.info(f"Creating directories: {directories}")
        for directory in directories:
            os.makedirs(directory, exist_ok=exist_ok)
            logging.info(f"Created directory: {directory}")
            
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        raise VisionBoardException(e, sys)

def get_size(path: str) -> str:
    """
    Get size of file or directory in human readable format
    Args:
        path: Path to file or directory
    Returns:
        str: Size in human readable format
    """
    try:
        logging.info(f"Getting size of {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
            
        size_bytes = os.path.getsize(path) if os.path.isfile(path) else get_dir_size(path)
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
            
    except Exception as e:
        logging.error(f"Error getting size of {path}: {str(e)}")
        raise VisionBoardException(e, sys)

def get_dir_size(path: str) -> int:
    """
    Get total size of directory in bytes
    Args:
        path: Path to directory
    Returns:
        int: Size in bytes
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size
        
    except Exception as e:
        logging.error(f"Error getting directory size of {path}: {str(e)}")
        raise VisionBoardException(e, sys)