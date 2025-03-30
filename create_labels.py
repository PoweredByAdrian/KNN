import json
import os
import cv2

labels_file="filtered_export.json"

images_folder="dataset/images"
labels_folder="dataset/labels"

with open(labels_file,"r",encoding="utf-8") as f:
    labels=json.load(f)

class_names = ["popis u obrázku"]  # only one class
CLASS_ID = 0  # Index for "popis u obrázku"

# Function to convert bbox from percentage to YOLO format
def convert_to_yolo_format(x_percent, y_percent, width_percent, height_percent, img_width, img_height):
    # Convert percentage to coordinates (normalize to [0, 1] relative to image size)
    x_center = (x_percent * img_width / 100 + width_percent * img_width / 200) / img_width
    y_center = (y_percent * img_height / 100 + height_percent * img_height / 200) / img_height
    width_norm = width_percent / 100
    height_norm = height_percent / 100
    
    return f"{CLASS_ID} {x_center} {y_center} {width_norm} {height_norm}"

os.makedirs(labels_folder, exist_ok=True)
for task in labels:
    id = task['id']
    file_path = os.path.join(images_folder,f"{id}.jpg") 

 # Extract annotations (assuming annotations are under 'annotations' key, adjust if needed)
    annotations = task.get("annotations", [])
    
    # Create a YOLO annotation file for each image
    yolo_annotation_file = os.path.join(labels_folder, f"{id}.txt")
    with open(yolo_annotation_file, "w") as yolo_file:
        # For each annotation, convert it to YOLO format and write it to the file
        for annotation in annotations:
            for result in annotation["result"]:
                if result.get("type") == "rectanglelabels" and "Popis u obrázku" in result.get("value", {}).get("rectanglelabels", []):
                    data = result["value"]

                    # Read the image
                    image = cv2.imread(os.path.join(images_folder,f'{id}.jpg'))

                    # Get dimensions
                    height, width = image.shape[:2]

                    yolo_format = convert_to_yolo_format(data['x'],data['y'],data['width'],data['height'],width,height)
                    yolo_file.write(yolo_format+"\n")
