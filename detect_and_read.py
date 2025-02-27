import cv2
import torch
import pytesseract
from pathlib import Path
import numpy as np
from ultralytics import YOLO

class SignboardDetectorReader:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('runs/train/exp/weights/best.pt')
        
        # Initialize Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def preprocess_for_ocr(self, image):
        """Preprocess image region for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised
    
    def read_text(self, image_region):
        """Extract text from image region using OCR"""
        if image_region.size == 0:
            return ""
            
        # Preprocess the region
        processed = self.preprocess_for_ocr(image_region)
        
        # OCR configuration
        custom_config = r'--oem 3 --psm 6'
        
        try:
            # Perform OCR
            text = pytesseract.image_to_string(processed, lang='eng', config=custom_config)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def process_image(self, image_path, conf_threshold=0.25):
        """
        Detect signboards and read text from them
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detection (0-1)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Make detection
        results = self.model(image)[0]
        
        # Process results
        detections = []
        
        # Get detection boxes
        for det in results.boxes.data:  # det: (x1, y1, x2, y2, confidence, class)
            if det[4] >= conf_threshold:  # Check confidence threshold
                x1, y1, x2, y2 = map(int, det[:4])
                conf = float(det[4])
                
                # Extract region
                signboard_region = image[y1:y2, x1:x2]
                
                # Read text
                text = self.read_text(signboard_region)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'text': text
                })
                
                # Draw rectangle and text on image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if text:
                    # Draw text with background for better visibility
                    text_size = cv2.getTextSize(text[:30], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (x1, y1-20), (x1 + text_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(image, text[:30], (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save annotated image
        output_path = Path('output_images')
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_file), image)
        
        return detections, output_file

def main():
    # Initialize detector
    detector = SignboardDetectorReader()
    
    # Process specific image
    image_path = r"D:\Harsha\proj\VisionBoard\VisionBoard_Data\train\images\56.png"
    
    try:
        # Process image
        print("Processing image...")
        detections, output_file = detector.process_image(image_path)
        
        # Print results
        print(f"\nFound {len(detections)} signboards:")
        for i, det in enumerate(detections, 1):
            print(f"\nSignboard {i}:")
            print(f"Location: {det['bbox']}")
            print(f"Confidence: {det['confidence']:.2f}")
            print(f"Text: {det['text']}")
            print("-" * 50)
        
        print(f"\nAnnotated image saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
