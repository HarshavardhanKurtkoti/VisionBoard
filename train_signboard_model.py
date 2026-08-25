"""
Train YOLOv8 on the 877-image road sign dataset.
4 classes: crosswalk, speedlimit, stop, trafficlight
"""
from ultralytics import YOLO
import pickle
from sklearn.preprocessing import LabelBinarizer

CLASSES = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']

def train():
    # 1. Initialize YOLOv8m with pretrained weights
    model = YOLO('yolov8n.pt')  # start from nano for faster training

    # 2. Train
    results = model.train(
        data='datasets/roadsigns/data.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        device='cpu',  # use 'cuda' if GPU available
        save=True,
        save_period=5,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        val=True,
        verbose=True,
        project='models',
        name='roadsigns_yolov8',
        exist_ok=True
    )

    # 3. Save LabelBinarizer (matching the Kaggle notebook pattern)
    lb = LabelBinarizer()
    lb.fit(CLASSES)
    with open('lb.pickle', 'wb') as f:
        pickle.dump(lb, f)
    print(f"\nlb.pickle saved with classes: {lb.classes_}")

    # 4. Export best weights path
    best_path = 'models/roadsigns_yolov8/weights/best.pt'
    print(f"Best model weights: {best_path}")

    return results

if __name__ == '__main__':
    train()
