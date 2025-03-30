import cv2
import numpy as np

def draw_yolo_boxes(image_path, annotations_path, output_path):
    """
    Draw YOLO bounding boxes on images and save results
    
    Args:
        image_path: Path to input image
        annotations_path: Path to YOLO annotation file (.txt)
        output_path: Path to save output image
        class_names: Optional list of class names for labels
    """
    # Read image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Read annotations
    with open(annotations_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]
    
    # Draw boxes
    for i,annotation in enumerate(annotations):
        if len(annotation) < 5:
            continue
            
        class_id = int(annotation[0])
        center_x = float(annotation[1])
        center_y = float(annotation[2])
        box_width = float(annotation[3])
        box_height = float(annotation[4])
        
        # Convert normalized coordinates to pixels
        x = int((center_x - box_width/2) * width)
        y = int((center_y - box_height/2) * height)
        w = int(box_width * width)
        h = int(box_height * height)
        
        # Draw box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        label = str(i)
        cv2.putText(img, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output
    cv2.imwrite(output_path, img)

# Example usage
if __name__ == "__main__":
    image_path = "dataset/images/72220.jpg"
    annotations_path = "dataset/labels/72220.txt"
    output_path = "tmp/72220.jpg"
    
    draw_yolo_boxes(image_path, annotations_path, output_path)
