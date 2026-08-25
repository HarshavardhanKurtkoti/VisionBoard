import numpy as np
import pytest
from visionboard.ocr.text_recognition import SignboardTextReader

def test_ocr_reader_initialization():
    reader = SignboardTextReader()
    assert isinstance(reader.tesseract_available, bool)

def test_ocr_preprocess_image():
    reader = SignboardTextReader()
    dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    processed = reader.preprocess_image(dummy_img)
    assert processed is not None
    assert len(processed.shape) == 2  # Binary single-channel

def test_ocr_extract_text_empty_safe():
    reader = SignboardTextReader()
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    text = reader.extract_text(empty_img)
    assert text == ""
