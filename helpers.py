import os
import json
from yolo_to_labelme import process_folders  # Import the function
from json_to_coco import labelme_to_coco  # Import the function
from shapely.geometry import Polygon, Point
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import numpy as np

class MaskGenerator:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def segmentation_to_mask(self, segmentation):
        polygon = Polygon(np.array(segmentation).reshape(-1, 2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Create a grid of coordinates
        y, x = np.indices((self.height, self.width))
        coords = np.stack((x, y), axis=-1)

        # Flatten the array for easier iteration
        points = coords.reshape(-1, 2)

        for point in points:
            if polygon.contains(Point(point)):
                mask[point[1], point[0]] = 1  # Point(x, y)

        return mask.flatten()


class MetricsCalculator:
    def __init__(self, image_dims):
        self.image_dims = image_dims
        self.mask_generator = MaskGenerator(*image_dims)

    def calculate_metrics(self, gt_annotations, pred_annotations):
        gt_masks = []
        pred_masks = []

        # Assuming COCO-style structure: iterate over 'annotations' key
        gt_annotations_list = gt_annotations.get('annotations', [])
        pred_annotations_list = pred_annotations.get('annotations', [])

        for gt, pred in zip(gt_annotations_list, pred_annotations_list):
            #print(f"Ground Truth Annotation: {gt}")
            #print(f"Prediction Annotation: {pred}")

            # Ensure we're accessing the segmentation key
            if isinstance(gt, dict) and 'segmentation' in gt and isinstance(gt['segmentation'], list):
                gt_mask = self.mask_generator.segmentation_to_mask(gt['segmentation'][0])  # Assuming first segmentation
            else:
                raise ValueError(f"The 'segmentation' key is missing or invalid in the ground truth annotation: {gt}")

            if isinstance(pred, dict) and 'segmentation' in pred and isinstance(pred['segmentation'], list):
                pred_mask = self.mask_generator.segmentation_to_mask(pred['segmentation'][0])  # Assuming first segmentation
            else:
                raise ValueError(f"The 'segmentation' key is missing or invalid in the prediction annotation: {pred}")

            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)

        gt_masks = np.concatenate(gt_masks)
        pred_masks = np.concatenate(pred_masks)

        precision = precision_score(gt_masks, pred_masks, average='binary')
        recall = recall_score(gt_masks, pred_masks, average='binary')
        f1 = f1_score(gt_masks, pred_masks, average='binary')
        iou = jaccard_score(gt_masks, pred_masks, average='binary')

        return precision, recall, f1, iou




class SegmentationEvaluator:
    def __init__(self, config):
        self.ground_truth_file = config['input']['ground_truth_coco_output']
        self.predicted_annotations_file = config['input']['prediction_coco_output']
        self.image_dims = (config['image']['height'], config['image']['width'])

    def load_annotations(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def evaluate(self):
        ground_truth_annotations = self.load_annotations(self.ground_truth_file)
        predicted_annotations = self.load_annotations(self.predicted_annotations_file)

        calculator = MetricsCalculator(self.image_dims)
        precision, recall, f1, iou = calculator.calculate_metrics(
            ground_truth_annotations, predicted_annotations
        )

        return precision, recall, f1, iou

def process_images(config, ground_truth=True):
    if ground_truth:
        yolo_dir = config['input']['ground_truth_yolo_dir']  # YOLO annotations directory
        image_dir = config['input']['ground_truth_image_dir']  # Image directory
        json_output_dir = config['input']['ground_truth_json_dir']  # Output JSON directory
    else:
        yolo_dir = config['input']['prediction_yolo_dir']  # YOLO annotations directory
        image_dir = config['input']['prediction_image_dir']  # Image directory
        json_output_dir = config['input']['prediction_json_dir']  # Output JSON directory
    
    # Convert YOLO annotations and images to LabelMe JSON format
    process_folders(yolo_dir, image_dir, json_output_dir)


def convert_to_coco(config, ground_truth=True):
    if ground_truth:
        json_dir = config['input']['ground_truth_json_dir']
        coco_output = config['input']['ground_truth_coco_output']
    else:
        json_dir = config['input']['prediction_json_dir']
        coco_output = config['input']['prediction_coco_output']

    # Convert JSON to COCO format
    labelme_to_coco(json_dir, coco_output)  # Using the correct function