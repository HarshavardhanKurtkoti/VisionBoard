from visionboard.entity.artifact_entity import ClassificationMetricArtifact
from visionboard.exception.exception import VisionBoardException
from sklearn.metrics import f1_score,precision_score,recall_score
import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from visionboard.logging.logger import logging

@dataclass
class DetectionMetrics:
    """Class for storing object detection metrics"""
    precision: float
    recall: float
    f1_score: float
    map50: float  # mAP at IoU=0.5
    map75: float  # mAP at IoU=0.75
    map50_95: float  # mAP at IoU=0.5:0.95
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mAP@0.5": self.map50,
            "mAP@0.75": self.map75,
            "mAP@0.5:0.95": self.map50_95
        }

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
        logging.info("Calculating IoU between bounding boxes")
        
        # Get coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0
        logging.info(f"Calculated IoU: {iou}")
        return iou
        
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
        logging.info("Calculating precision and recall metrics")
        
        if not pred_boxes or not true_boxes:
            logging.warning("Empty prediction or ground truth boxes")
            return 0.0, 0.0, 0.0
            
        true_positives = 0
        false_positives = 0
        
        # Track which ground truth boxes have been matched
        matched_gt = set()
        
        # For each predicted box
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = None
            
            # Find the best matching ground truth box
            for i, gt_box in enumerate(true_boxes):
                if i in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # If we found a match above the threshold
            if best_iou >= iou_threshold and best_gt_idx is not None:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives = len(true_boxes) - true_positives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logging.info(f"Calculated metrics - Precision: {precision}, Recall: {recall}, F1: {f1_score}")
        return precision, recall, f1_score
        
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
    Args:
        pred_boxes: List of predicted bounding boxes
        true_boxes: List of ground truth bounding boxes
        iou_thresholds: List of IoU thresholds to calculate mAP
    Returns:
        Tuple[float, float, float]: mAP@0.5, mAP@0.75, mAP@0.5:0.95
    """
    try:
        logging.info("Calculating mAP metrics")
        
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))
        
        if not pred_boxes or not true_boxes:
            logging.warning("Empty prediction or ground truth boxes")
            return 0.0, 0.0, 0.0
        
        # Calculate AP for each IoU threshold
        aps = []
        for iou_threshold in iou_thresholds:
            precision, _, _ = calculate_precision_recall(
                pred_boxes, true_boxes, iou_threshold
            )
            aps.append(precision)
        
        # Calculate mAP at different thresholds
        map50 = aps[0]  # mAP at IoU=0.5
        map75 = aps[1]  # mAP at IoU=0.75
        map50_95 = np.mean(aps)  # mAP at IoU=0.5:0.95
        
        logging.info(f"Calculated mAP metrics - mAP@0.5: {map50}, mAP@0.75: {map75}, mAP@0.5:0.95: {map50_95}")
        return map50, map75, map50_95
        
    except Exception as e:
        logging.error(f"Error calculating mAP: {str(e)}")
        raise VisionBoardException(e, sys)

def evaluate_detection_metrics(
    pred_boxes: List[np.ndarray],
    true_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> DetectionMetrics:
    """
    Calculate all detection metrics
    Args:
        pred_boxes: List of predicted bounding boxes
        true_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for precision/recall calculation
    Returns:
        DetectionMetrics: Object containing all metrics
    """
    try:
        logging.info("Evaluating all detection metrics")
        
        # Calculate precision, recall, F1
        precision, recall, f1_score = calculate_precision_recall(
            pred_boxes, true_boxes, iou_threshold
        )
        
        # Calculate mAP metrics
        map50, map75, map50_95 = calculate_map(pred_boxes, true_boxes)
        
        metrics = DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            map50=map50,
            map75=map75,
            map50_95=map50_95
        )
        
        logging.info(f"Completed metrics evaluation: {metrics.to_dict()}")
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating detection metrics: {str(e)}")
        raise VisionBoardException(e, sys)

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
            
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise VisionBoardException(e,sys)