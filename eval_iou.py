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
    """Calculates the Intersection over Union (IoU) between two masks.
    Args:
        mask1: The first mask.
        mask2: The second mask.
    Returns:
        The IoU value.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def load_annotations(json_path):
    """Loads the annotations from a JSON file.
    Args:
        json_path: Path to the JSON file.
    Returns:
        A list of annotations, each containing the segmentation points.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data.get('annotations', [])

def evaluate_overall_iou(config):
    """Evaluates the overall IoU between ground truth and predicted annotations.
    Args:
        config: The configuration dictionary loaded from the YAML file.
    Returns:
        The overall IoU value.
    """
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
    
    # Calculate overall IoU
    iou = calculate_iou(gt_combined_mask, pred_combined_mask)
    
    return iou

def main():
    # Load configuration
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Evaluate overall IoU
    iou = evaluate_overall_iou(config)
    
    # Print the overall IoU
    print(f"Overall IoU: {iou:.4f}")

if __name__ == "__main__":
    main()
