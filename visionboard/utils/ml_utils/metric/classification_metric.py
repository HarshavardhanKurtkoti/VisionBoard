import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from visionboard.entity.artifact_entity import DetectionMetricArtifact, ClassificationMetricArtifact
from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging

DetectionMetrics = DetectionMetricArtifact

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]
    Returns:
        float: IoU value
    """
    try:
        box1 = np.asarray(box1, dtype=float)
        box2 = np.asarray(box2, dtype=float)
        
        # Get coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        
        # Calculate union area
        box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0.0
        return float(iou)
        
    except Exception as e:
        logging.error(f"Error calculating IoU: {str(e)}")
        raise VisionBoardException(e, sys)

def calculate_precision_recall(
    pred_boxes: List[np.ndarray],
    true_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall and F1 score for object detection
    Args:
        pred_boxes: List of predicted bounding boxes
        true_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for considering a detection as correct
    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score
    """
    try:
        if not pred_boxes and not true_boxes:
            return 1.0, 1.0, 1.0
        if not pred_boxes:
            return 0.0, 0.0, 0.0
        if not true_boxes:
            return 0.0, 0.0, 0.0
            
        true_positives = 0
        false_positives = 0
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = None
            
            for i, gt_box in enumerate(true_boxes):
                if i in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and best_gt_idx is not None:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives = len(true_boxes) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return float(precision), float(recall), float(f1_score)
        
    except Exception as e:
        logging.error(f"Error calculating precision and recall: {str(e)}")
        raise VisionBoardException(e, sys)

def calculate_map(
    pred_boxes: List[np.ndarray],
    true_boxes: List[np.ndarray],
    iou_thresholds: Optional[List[float]] = None
) -> Tuple[float, float, float]:
    """
    Calculate mean Average Precision (mAP) at different IoU thresholds
    """
    try:
        if iou_thresholds is None:
            iou_thresholds = list(np.arange(0.5, 1.0, 0.05))
        
        if not pred_boxes and not true_boxes:
            return 1.0, 1.0, 1.0
        if not pred_boxes or not true_boxes:
            return 0.0, 0.0, 0.0
        
        aps = []
        for iou_threshold in iou_thresholds:
            precision, _, _ = calculate_precision_recall(
                pred_boxes, true_boxes, iou_threshold
            )
            aps.append(precision)
        
        map50, _, _ = calculate_precision_recall(pred_boxes, true_boxes, 0.5)
        map75, _, _ = calculate_precision_recall(pred_boxes, true_boxes, 0.75)
        map50_95 = float(np.mean(aps)) if aps else 0.0
        
        return float(map50), float(map75), float(map50_95)
        
    except Exception as e:
        logging.error(f"Error calculating mAP: {str(e)}")
        raise VisionBoardException(e, sys)

def evaluate_detection_metrics(
    pred_boxes: List[np.ndarray],
    true_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> DetectionMetricArtifact:
    """
    Calculate all detection metrics and return typed artifact
    """
    try:
        precision, recall, f1_score = calculate_precision_recall(
            pred_boxes, true_boxes, iou_threshold
        )
        map50, map75, map50_95 = calculate_map(pred_boxes, true_boxes)
        
        return DetectionMetricArtifact(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            map50=map50,
            map75=map75,
            map50_95=map50_95
        )
    except Exception as e:
        logging.error(f"Error evaluating detection metrics: {str(e)}")
        raise VisionBoardException(e, sys)

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """Helper for sklearn-compatible metric calculation if needed"""
    try:
        from sklearn.metrics import f1_score as sk_f1, precision_score as sk_prec, recall_score as sk_rec
        model_f1 = float(sk_f1(y_true, y_pred, average='weighted', zero_division=0))
        model_rec = float(sk_rec(y_true, y_pred, average='weighted', zero_division=0))
        model_prec = float(sk_prec(y_true, y_pred, average='weighted', zero_division=0))
        return ClassificationMetricArtifact(f1_score=model_f1, precision=model_prec, recall=model_rec)
    except Exception as e:
        raise VisionBoardException(e, sys)