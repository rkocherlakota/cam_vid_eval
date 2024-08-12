import json
import os
from PIL import Image

def yolo_to_labelme(yolo_data, image_path):
    """
    Converts YOLO format annotations to LabelMe format.
    
    Parameters:
        yolo_data (str): String containing YOLO format data.
        image_path (str): Path to the corresponding image.
    
    Returns:
        dict: A dictionary representing the LabelMe JSON format.
    """
    yolo_lines = yolo_data.strip().split('\n')
    
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Initialize LabelMe JSON structure
    labelme_format = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    for line in yolo_lines:
        parts = line.split()
        class_id = int(parts[0])
        points = [float(coord) for coord in parts[1:]]
        
        shape = {
            "label": str(class_id),
            "points": [],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }

        # Process points (x_center, y_center, width, height) to LabelMe format
        for i in range(0, len(points), 2):
            x = points[i] * width
            y = points[i + 1] * height
            shape["points"].append([x, y])

        labelme_format["shapes"].append(shape)

    return labelme_format

def process_folders(yolo_folder_path, image_folder_path, output_folder_path):
    """
    Processes all YOLO annotation files and corresponding images from separate folders,
    and saves each converted annotation as a JSON file in the output folder.
    
    Parameters:
        yolo_folder_path (str): Path to the folder containing YOLO annotations.
        image_folder_path (str): Path to the folder containing images.
        output_folder_path (str): Path to the folder where output JSON files will be saved.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in os.listdir(yolo_folder_path):
        if file_name.endswith('.txt'):
            # Read YOLO annotation file
            yolo_file_path = os.path.join(yolo_folder_path, file_name)
            with open(yolo_file_path, 'r') as f:
                yolo_data = f.read()

            # Corresponding image file
            image_file_name_jpg = file_name.replace('.txt', '.jpg')
            image_file_name_png = file_name.replace('.txt', '.png')
            image_path_jpg = os.path.join(image_folder_path, image_file_name_jpg)
            image_path_png = os.path.join(image_folder_path, image_file_name_png)

            if os.path.exists(image_path_jpg):
                image_path = image_path_jpg
            elif os.path.exists(image_path_png):
                image_path = image_path_png
            else:
                print(f"Image file not found for annotation {file_name}")
                continue

            # Output JSON file path
            output_file_name = file_name.replace('.txt', '.json')
            output_path = os.path.join(output_folder_path, output_file_name)

            # Convert and save
            try:
                labelme_format = yolo_to_labelme(yolo_data, image_path)
                with open(output_path, 'w') as f:
                    json.dump(labelme_format, f, indent=4)
                print(f"Converted {file_name} to {output_file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")