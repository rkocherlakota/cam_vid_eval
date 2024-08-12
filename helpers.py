import json
import numpy as np
from yolo_to_labelme import process_folders
from json_to_coco import labelme_to_coco

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