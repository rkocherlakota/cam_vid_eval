import os
import json
from tqdm import tqdm
from collections import defaultdict

def labelme_to_coco(labelme_json_dir, output_coco_json_path):
    # Initialize COCO format dictionaries
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_mapping = {}
    annotation_id = 1
    image_id = 1

    # Check if the output file exists, if not, create it
    if not os.path.exists(output_coco_json_path):
        with open(output_coco_json_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"Created new COCO JSON file at {output_coco_json_path}")
    else:
        print(f"File {output_coco_json_path} already exists. Appending to existing file.")

    # Load and process each LabelMe JSON file
    labelme_json_files = [f for f in os.listdir(labelme_json_dir) if f.endswith('.json')]
    for json_file in tqdm(labelme_json_files, desc="Processing LabelMe files"):
        with open(os.path.join(labelme_json_dir, json_file), 'r') as f:
            labelme_data = json.load(f)
            
            # Extract image info
            image_info = {
                "id": image_id,
                "file_name": labelme_data["imagePath"],
                "width": labelme_data["imageWidth"],
                "height": labelme_data["imageHeight"]
            }
            coco_data["images"].append(image_info)
            
            # Process each shape
            for shape in labelme_data["shapes"]:
                label = shape["label"]
                points = shape["points"]
                category_id = category_mapping.setdefault(label, len(category_mapping) + 1)
                
                # Add category if not present
                if category_id == len(coco_data["categories"]) + 1:
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": label
                    })
                
                # Convert polygon points to COCO format
                segmentation = [p for point in points for p in point]  # Flatten the points list
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": 0,  # Area calculation can be added if needed
                    "bbox": [],  # Bounding box calculation can be added if needed
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
            image_id += 1

    # Save COCO data to file
    with open(output_coco_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"Conversion complete. COCO JSON saved at {output_coco_json_path}")