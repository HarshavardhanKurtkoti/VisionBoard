import os
from typing import Tuple, Optional, List, Union
import numpy as np

# Try importing cv2, otherwise provide PIL-based fallbacks
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

def read_image(image_path: str) -> Optional[np.ndarray]:
    """
    Read image from file as BGR numpy array
    """
    if not os.path.exists(image_path):
        return None
        
    if HAS_CV2:
        return cv2.imread(str(image_path))
    elif HAS_PIL:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            rgb_arr = np.array(pil_img)
            # Convert RGB to BGR for consistency
            bgr_arr = rgb_arr[:, :, ::-1].copy()
            return bgr_arr
        except Exception:
            return None
    else:
        # Fallback dummy array
        return np.zeros((320, 320, 3), dtype=np.uint8)

def save_image(image_path: str, image: np.ndarray) -> bool:
    """
    Save BGR numpy array image to file
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
        if HAS_CV2:
            return bool(cv2.imwrite(str(image_path), image))
        elif HAS_PIL:
            if len(image.shape) == 3:
                rgb_arr = image[:, :, ::-1].copy()
                pil_img = Image.fromarray(rgb_arr)
            else:
                pil_img = Image.fromarray(image)
            pil_img.save(image_path)
            return True
        return False
    except Exception:
        return False

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image (width, height)
    """
    w, h = size
    if HAS_CV2:
        return cv2.resize(image, (w, h))
    elif HAS_PIL:
        if len(image.shape) == 3:
            rgb_arr = image[:, :, ::-1].copy()
            pil_img = Image.fromarray(rgb_arr).resize((w, h))
            return np.array(pil_img)[:, :, ::-1].copy()
        else:
            pil_img = Image.fromarray(image).resize((w, h))
            return np.array(pil_img)
    return image

def draw_box_and_label(
    image: np.ndarray,
    box: List[Union[int, float]],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding box and label on BGR image
    """
    vis_img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    
    if HAS_CV2:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        if label:
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            cv2.rectangle(vis_img, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 6, max(0, y1)), color, -1)
            cv2.putText(vis_img, label, (x1 + 3, max(text_h + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    elif HAS_PIL:
        rgb_arr = vis_img[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb_arr)
        draw = ImageDraw.Draw(pil_img)
        rgb_color = (color[2], color[1], color[0])
        draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=2)
        if label:
            draw.text((x1 + 4, max(0, y1 - 15)), label, fill=rgb_color)
        vis_img = np.array(pil_img)[:, :, ::-1].copy()
        
    return vis_img
