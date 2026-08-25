"""
Convert Pascal VOC XML annotations to YOLO format and split into train/val/test.
Dataset: data/annotations/*.xml + data/images/*.png
Classes: crosswalk(0), speedlimit(1), stop(2), trafficlight(3)
"""
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

SEED = 42
random.seed(SEED)

# Class mapping (alphabetical order to match lb.pickle)
CLASSES = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

SRC_IMAGES = Path("data/images")
SRC_ANNOTS = Path("data/annotations")
OUT_BASE = Path("datasets/roadsigns")

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}


def convert_voc_to_yolo(xml_path: str) -> list:
    """Convert a single Pascal VOC XML to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = int(root.find('.//size/width').text)
    h = int(root.find('.//size/height').text)

    if w == 0 or h == 0:
        return []

    yolo_lines = []
    for obj in root.findall('.//object'):
        name = obj.find('name').text.strip().lower()
        if name not in CLASS_TO_ID:
            print(f"  [SKIP] Unknown class '{name}' in {xml_path}")
            continue

        class_id = CLASS_TO_ID[name]
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Clamp to image boundaries
        xmin = max(0, min(xmin, w))
        ymin = max(0, min(ymin, h))
        xmax = max(0, min(xmax, w))
        ymax = max(0, min(ymax, h))

        # Convert to YOLO format: x_center, y_center, width, height (normalized)
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return yolo_lines


def main():
    # Collect all annotation files
    xml_files = sorted([f for f in os.listdir(SRC_ANNOTS) if f.endswith('.xml')])
    print(f"Found {len(xml_files)} annotation files")

    # Match with images
    pairs = []
    for xml_file in xml_files:
        tree = ET.parse(SRC_ANNOTS / xml_file)
        root = tree.getroot()
        img_filename = root.find('.//filename').text
        img_path = SRC_IMAGES / img_filename

        if not img_path.exists():
            # Try alternative extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                alt = SRC_IMAGES / (Path(img_filename).stem + ext)
                if alt.exists():
                    img_path = alt
                    break

        if img_path.exists():
            pairs.append((str(img_path), str(SRC_ANNOTS / xml_file)))
        else:
            print(f"  [WARN] Image not found for {xml_file}: {img_filename}")

    print(f"Matched {len(pairs)} image-annotation pairs")

    # Shuffle and split
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }

    # Create output directories and copy files
    for split_name, split_pairs in splits.items():
        img_dir = OUT_BASE / split_name / "images"
        lbl_dir = OUT_BASE / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0
        for img_path, xml_path in split_pairs:
            yolo_lines = convert_voc_to_yolo(xml_path)
            if not yolo_lines:
                skipped += 1
                continue

            # Copy image
            img_name = Path(img_path).name
            shutil.copy2(img_path, img_dir / img_name)

            # Write YOLO label
            label_name = Path(img_name).stem + ".txt"
            with open(lbl_dir / label_name, "w") as f:
                f.write("\n".join(yolo_lines))

            converted += 1

        print(f"  {split_name}: {converted} converted, {skipped} skipped")

    # Create data.yaml
    data_yaml = OUT_BASE / "data.yaml"
    abs_base = str(OUT_BASE.resolve()).replace("\\", "/")
    with open(data_yaml, "w") as f:
        f.write(f"path: {abs_base}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(CLASSES)}\n\n")
        f.write("names:\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"  {i}: '{cls}'\n")

    print(f"\ndata.yaml written to: {data_yaml}")
    print(f"Classes: {CLASSES}")
    print(f"Total pairs: {len(pairs)}")
    for split_name, split_pairs in splits.items():
        print(f"  {split_name}: {len(split_pairs)}")


if __name__ == "__main__":
    main()
