import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.utils.main_utils.image_utils import read_image, resize_image

class SignboardTextReader:
    """
    Robust OCR reader for extracting text from detected signboards
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR text reader with auto-detection of Tesseract binary
        """
        self.tesseract_available = False
        self.pytesseract = None
        
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            resolved_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
            if not resolved_cmd:
                which_cmd = shutil.which("tesseract")
                if which_cmd:
                    resolved_cmd = which_cmd
                elif os.name == 'nt':
                    candidate_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
                    ]
                    for path in candidate_paths:
                        if os.path.exists(path):
                            resolved_cmd = path
                            break
            
            if resolved_cmd and os.path.exists(resolved_cmd):
                pytesseract.pytesseract.tesseract_cmd = resolved_cmd
                self.tesseract_available = True
                logging.info(f"Tesseract OCR configured at: {resolved_cmd}")
            else:
                try:
                    pytesseract.get_tesseract_version()
                    self.tesseract_available = True
                    logging.info("Tesseract OCR found in system PATH")
                except Exception:
                    self.tesseract_available = False
                    logging.warning("Tesseract binary not found. OCR features will return empty strings.")
                    
        except ImportError:
            logging.warning("pytesseract package not installed. OCR features disabled.")
            self.tesseract_available = False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image region to improve OCR accuracy
        """
        try:
            if image is None or image.size == 0:
                return image
                
            # Convert to grayscale using standard RGB/BGR luminance coefficients
            if len(image.shape) == 3:
                # Assuming BGR
                gray = (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.uint8)
            else:
                gray = image
            
            # Resize if region is small
            h, w = gray.shape[:2]
            if h < 30 or w < 60:
                scale = max(2.0, 60.0 / max(h, 1))
                gray = resize_image(gray, (int(w * scale), int(h * scale)))
            
            # Simple Otsu-like thresholding in numpy
            mean_val = np.mean(gray)
            binary = np.where(gray > mean_val, 255, 0).astype(np.uint8)
            return binary
            
        except Exception as e:
            logging.warning(f"Error in OCR preprocessing: {str(e)}")
            return image
    
    def extract_text(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> str:
        """
        Extract text from an image or bounding box region
        """
        if not self.tesseract_available or self.pytesseract is None:
            return ""
        
        try:
            region = image
            if bbox is not None and len(bbox) == 4:
                h, w = image.shape[:2]
                if all(0.0 <= v <= 1.0 for v in bbox):
                    xc, yc, bw, bh = bbox
                    x1 = max(0, int((xc - bw / 2) * w))
                    y1 = max(0, int((yc - bh / 2) * h))
                    x2 = min(w, int((xc + bw / 2) * w))
                    y2 = min(h, int((yc + bh / 2) * h))
                else:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    region = image[y1:y2, x1:x2]
                else:
                    return ""
            
            if region is None or region.size == 0:
                return ""
                
            processed = self.preprocess_image(region)
            custom_config = r'--oem 3 --psm 6'
            text = self.pytesseract.image_to_string(processed, lang='eng', config=custom_config)
            return text.strip()
            
        except Exception as e:
            logging.warning(f"Error during OCR extraction: {str(e)}")
            return ""
    
    def process_image_with_labels(self, image_path: str, label_path: str) -> List[Dict[str, Any]]:
        """
        Process an image and its corresponding label file
        """
        try:
            image = read_image(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            results = []
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            text = self.extract_text(image, bbox)
                            results.append({
                                'class_id': class_id,
                                'bbox': bbox,
                                'text': text
                            })
            return results
        except Exception as e:
            logging.error(f"Error processing image with labels: {str(e)}")
            raise VisionBoardException(e, sys)
