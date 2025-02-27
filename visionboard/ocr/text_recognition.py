import cv2
import numpy as np
import pytesseract
from pathlib import Path
import os

class SignboardTextReader:
    def __init__(self):
        # Initialize pytesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get black and white image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised
    
    def extract_text(self, image, bbox):
        """
        Extract text from a signboard region
        """
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Convert normalized bbox to pixel coordinates
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract the signboard region
        signboard_region = image[y1:y2, x1:x2]
        
        # Save the region for debugging
        debug_dir = Path("debug_regions")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"region_{x1}_{y1}.jpg"), signboard_region)
        
        # Preprocess the region
        processed_region = self.preprocess_image(signboard_region)
        cv2.imwrite(str(debug_dir / f"processed_{x1}_{y1}.jpg"), processed_region)
        
        # Perform OCR with additional configurations
        try:
            # Configure tesseract parameters
            custom_config = r'--oem 3 --psm 6'  # Assume uniform text block
            text = pytesseract.image_to_string(processed_region, lang='eng', config=custom_config)
            return text.strip()
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""
    
    def process_image_with_labels(self, image_path, label_path):
        """
        Process an image and its corresponding label file
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Read label file
        results = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bbox = [x_center, y_center, width, height]
                
                # Extract text from this bbox
                text = self.extract_text(image, bbox)
                
                results.append({
                    'bbox': bbox,
                    'text': text
                })
        
        return results

def main():
    # Initialize the reader
    reader = SignboardTextReader()
    
    # Get the base directory
    base_dir = Path("d:/Harsha/proj/VisionBoard/VisionBoard_Data/train")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # Process first 5 images
    for i, image_file in enumerate(sorted(images_dir.glob("*.jpg"))[:5]):
        label_file = labels_dir / f"{image_file.stem}.txt"
        
        if image_file.exists() and label_file.exists():
            print(f"\nProcessing image: {image_file.name}")
            try:
                results = reader.process_image_with_labels(image_file, label_file)
                for j, result in enumerate(results, 1):
                    print(f"Signboard {j}:")
                    print(f"Location: {result['bbox']}")
                    print(f"Text: {result['text']}")
                    print("-" * 50)
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")

if __name__ == "__main__":
    main()
