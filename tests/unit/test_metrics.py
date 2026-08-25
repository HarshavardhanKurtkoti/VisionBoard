import numpy as np
import pytest
from visionboard.utils.ml_utils.metric.classification_metric import (
    calculate_iou,
    calculate_precision_recall,
    calculate_map,
    evaluate_detection_metrics
)

def test_calculate_iou_perfect_overlap():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([0, 0, 10, 10])
    assert calculate_iou(box1, box2) == 1.0

def test_calculate_iou_no_overlap():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([20, 20, 30, 30])
    assert calculate_iou(box1, box2) == 0.0

def test_calculate_iou_partial_overlap():
    box1 = np.array([0, 0, 10, 10])  # Area 100
    box2 = np.array([5, 0, 15, 10])  # Area 100, Inter: 5*10=50, Union: 150
    iou = calculate_iou(box1, box2)
    assert abs(iou - (50.0 / 150.0)) < 1e-5

def test_precision_recall_and_map():
    pred_boxes = [np.array([0, 0, 10, 10]), np.array([50, 50, 60, 60])]
    true_boxes = [np.array([0, 0, 10, 10])]
    
    prec, rec, f1 = calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5)
    assert prec == 0.5  # 1 TP, 1 FP -> 1/2
    assert rec == 1.0   # 1 TP, 0 FN -> 1/1
    
    metrics = evaluate_detection_metrics(pred_boxes, true_boxes)
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
