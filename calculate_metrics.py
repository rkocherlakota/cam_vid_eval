import json
import numpy as np
import cv2
import yaml

def points_to_mask(points, img_shape):
    """Converts segmentation points to a mask.
    Args:
        points: A list of (x, y) coordinates.
        img_shape: The shape of the image (height, width).
    Returns:
        A numpy array representing the mask.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def calculate_iou(mask1, mask2):
    """Calculates the Intersection over Union (IoU) between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_precision(mask1, mask2):
    """Calculates Precision between two masks."""
    true_positive = np.logical_and(mask1, mask2).sum()
    predicted_positive = mask2.sum()
    precision = true_positive / predicted_positive if predicted_positive != 0 else 0
    return precision

def calculate_recall(mask1, mask2):
    """Calculates Recall between two masks."""
    true_positive = np.logical_and(mask1, mask2).sum()
    actual_positive = mask1.sum()
    recall = true_positive / actual_positive if actual_positive != 0 else 0
    return recall

def calculate_f1(precision, recall):
    """Calculates F1 Score from Precision and Recall."""
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1

def load_annotations(json_path):
    """Loads the annotations from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data.get('annotations', [])

def evaluate_metrics(config):
    """Evaluates IoU, Precision, Recall, and F1 Score between ground truth and predicted annotations."""
    ground_truth_json = config['input']['ground_truth_coco_output']
    predicted_json = config['input']['prediction_coco_output']
    img_shape = (config['image']['height'], config['image']['width'])
    
    gt_annotations = load_annotations(ground_truth_json)
    pred_annotations = load_annotations(predicted_json)
    
    # Create empty masks for the ground truth and predictions
    gt_combined_mask = np.zeros(img_shape, dtype=np.uint8)
    pred_combined_mask = np.zeros(img_shape, dtype=np.uint8)
    
    # Fill the ground truth mask
    for gt in gt_annotations:
        gt_points = np.array(gt['segmentation'][0]).reshape(-1, 2)
        gt_mask = points_to_mask(gt_points, img_shape)
        gt_combined_mask = np.maximum(gt_combined_mask, gt_mask)
    
    # Fill the prediction mask
    for pred in pred_annotations:
        pred_points = np.array(pred['segmentation'][0]).reshape(-1, 2)
        pred_mask = points_to_mask(pred_points, img_shape)
        pred_combined_mask = np.maximum(pred_combined_mask, pred_mask)
    
    # Calculate metrics
    iou = calculate_iou(gt_combined_mask, pred_combined_mask)
    precision = calculate_precision(gt_combined_mask, pred_combined_mask)
    recall = calculate_recall(gt_combined_mask, pred_combined_mask)
    f1_score = calculate_f1(precision, recall)
    
    return iou, precision, recall, f1_score