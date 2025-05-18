import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from collections import defaultdict

def process_detections(detection, base_path, image, json_path="detections.json", viz_path="visualization.png"):
    """
    Process YOLO detections for a single image, match descriptions to images,
    save visualization and JSON results.
    
    Args:
        detection: YOLO detection result for a single image
        base_path: Base path for saving outputs
        image: Original image as numpy array
        json_path: Path to save JSON results
        viz_path: Path to save visualization image, or None to skip visualization
        
    Returns:
        Tuple of (results_dict, json_path, visualization_path)
    """
    # Lists to store detections by class
    images = []      # Class 1
    descriptions = []  # Class 0
    
    # Process the boxes in this detection
    for j in range(len(detection.boxes)):
        box = detection.boxes[j]
        try:
            # Extract values safely
            cls = int(box.cls[0].item())  # class ID
            conf = float(box.conf[0].item())  # confidence score
            xyxy = box.xyxy[0].tolist()    # [x1, y1, x2, y2]
            
            # Calculate center point of the box
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            
            # Store detection with its metadata
            detection_info = {
                'id': f"{j}",  # Create unique ID
                'cls': cls,
                'conf': conf,
                'xyxy': xyxy,
                'center': (center_x, center_y),
                'width': xyxy[2] - xyxy[0],
                'height': xyxy[3] - xyxy[1]
            }
            
            # Separate by class
            if cls == 1:  # Image
                images.append(detection_info)
            elif cls == 0:  # Description
                descriptions.append(detection_info)
                
        except Exception as e:
            print(f"Error processing box {j}: {e}")
            continue
    
    # Sort by confidence (highest first) to prioritize more confident detections
    images.sort(key=lambda x: x['conf'], reverse=True)
    descriptions.sort(key=lambda x: x['conf'], reverse=True)
    
    print(f"Found {len(images)} images and {len(descriptions)} descriptions")
    
    # If no images found, don't save JSON and return
    if not images:
        print("No images found in detection, skipping JSON output")
        return {}, None, viz_path

    # Create results for all images, regardless of whether they have matching descriptions
    results = {}
    assigned_descriptions = set()
    
    # For each image, find the closest unassigned description (if any are available)
    for img in images:
        img_center = img['center']
        closest_desc = None
        min_distance = float('inf')
        
        # Only try to find a description if there are any available
        if descriptions:
            for desc in descriptions:
                if desc['id'] in assigned_descriptions:
                    continue  # Skip already assigned descriptions
                    
                desc_center = desc['center']
                
                # Calculate Euclidean distance between centers
                distance = ((img_center[0] - desc_center[0]) ** 2 + 
                            (img_center[1] - desc_center[1]) ** 2) ** 0.5
                
                # Update closest if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_desc = desc
        
        # Always add the image to results, with or without a matching description
        if closest_desc:
            assigned_descriptions.add(closest_desc['id'])
            results[img['id']] = {
                'image': img,
                'description': closest_desc,
                'distance': min_distance,
                'has_match': True
            }
        else:
            # Explicitly handle the case with no matching description
            results[img['id']] = {
                'image': img,
                'description': None,
                'distance': None,
                'has_match': False
            }
            print(f"Image {img['id']} has no matching description")
    
    # Check for unassigned descriptions
    unassigned_descriptions = [desc for desc in descriptions 
                              if desc['id'] not in assigned_descriptions]
    
    if unassigned_descriptions:
        print(f"Warning: {len(unassigned_descriptions)} descriptions were not assigned to any image")
    
    # Save to JSON in Label Studio format
    actual_json_path = save_matches_to_json(results, json_path, base_path)
    
    # Create visualization only if viz_path is provided
    if viz_path:
        create_visualization(image, results, viz_path)
    
    return results, actual_json_path, viz_path

def save_matches_to_json(results, output_path, base_path):
    """
    Save matching information to a JSON file in Label Studio format.
    Only saves if at least one image ("Obrázek") is detected.
    
    Args:
        results: Dictionary with detection results
        output_path: Path to save the JSON file
        
    Returns:
        Output path if saved, None if no images found
    """
    # Check if we have any image detections
    if not results:
        print(f"No image detections found, skipping JSON output for {output_path}")
        return None
    
    # Get file basename for the id
    file_id = os.path.basename(output_path).split('_')[0]  # Extract ID from filename
    try:
        file_id = int(file_id)  # Try to convert to int if numeric
    except ValueError:
        file_id = hash(file_id) % 100000  # Use hash if not numeric
    
    # Create the base structure matching Label Studio format
    output_data = {
        "id": file_id,
        "annotations": [
            {
                "id": file_id + 10000,  # Create a unique annotation ID
                "completed_by": 1,  # Default user ID
                "result": [],  # Will be filled with rectangle labels
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "lead_time": 0.0,
                "prediction": {},
                "result_count": 0,
                "task": file_id
            }
        ],
        "data": {
            "name": f"{base_path}.jpg",
            "image": f"/data/local-files/?d=images/{base_path}.jpg"
        },
        "meta": {},
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "project": 1
    }
    
    # Track the next available result ID number for relation connections
    next_relation_id = 0
    relations = []
    
    # Get image dimensions from the first result (if available)
    original_width = 1500   # Default values
    original_height = 2309  # Default values
    
    # Find any image to get original dimensions if possible
    for img_id, match in results.items():
        img_info = match['image']
        img_bbox = img_info['xyxy']
        if img_bbox:
            # Use the maximum x and y values as a proxy for image dimensions if needed
            # This is an approximation - ideally we'd have the actual image dimensions
            original_width = max(original_width, int(img_bbox[2]))
            original_height = max(original_height, int(img_bbox[3]))
            break
    
    # Process each image and its potential description
    for img_id, match in results.items():
        img_info = match['image']
        img_bbox = img_info['xyxy']
        
        # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] in percentages
        x_percent = (img_bbox[0] / original_width) * 100
        y_percent = (img_bbox[1] / original_height) * 100
        width_percent = ((img_bbox[2] - img_bbox[0]) / original_width) * 100
        height_percent = ((img_bbox[3] - img_bbox[1]) / original_height) * 100
        
        # Add image rectangle with "Obrázek" label
        image_result = {
            "id": f"result{next_relation_id}",  # Use incrementing IDs for results
            "type": "rectanglelabels",
            "value": {
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rotation": 0,
                "rectanglelabels": ["Obrázek"]
            },
            "origin": "prediction",
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": original_width,
            "original_height": original_height
        }
        
        output_data["annotations"][0]["result"].append(image_result)
        image_result_id = f"result{next_relation_id}"
        next_relation_id += 1
        
        # If there's a matched description, add it too
        if match['description']:
            desc_info = match['description']
            desc_bbox = desc_info['xyxy']
            
            # Convert description bbox to percentages
            desc_x_percent = (desc_bbox[0] / original_width) * 100
            desc_y_percent = (desc_bbox[1] / original_height) * 100
            desc_width_percent = ((desc_bbox[2] - desc_bbox[0]) / original_width) * 100
            desc_height_percent = ((desc_bbox[3] - desc_bbox[1]) / original_height) * 100
            
            # Generate a unique ID for the description
            desc_id = f"desc{next_relation_id}"
            next_relation_id += 1
            
            # Add description rectangle with "Popis u obrázku" label
            desc_result = {
                "id": desc_id,
                "type": "rectanglelabels",
                "value": {
                    "x": desc_x_percent,
                    "y": desc_y_percent,
                    "width": desc_width_percent,
                    "height": desc_height_percent,
                    "rotation": 0,
                    "rectanglelabels": ["Popis u obrázku"]
                },
                "origin": "prediction",
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0,
                "original_width": original_width,
                "original_height": original_height
            }
            
            output_data["annotations"][0]["result"].append(desc_result)
            
            # Add relation between image and description
            relation = {
                "type": "relation",
                "from_id": desc_id,
                "to_id": image_result_id,
                "direction": "right"
            }
            relations.append(relation)
    
    # Add all relations after all rectangles have been added
    for relation in relations:
        output_data["annotations"][0]["result"].append(relation)
    
    # Update the result count
    output_data["annotations"][0]["result_count"] = len(output_data["annotations"][0]["result"])
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Label Studio format results saved to {output_path}")
    
    return output_path

def save_empty_json(output_path, reason, num_descriptions=0):
    """
    Save an empty JSON file in Label Studio format when no images are detected.
    
    Args:
        output_path: Path to save the JSON file
        reason: Reason for empty results
        num_descriptions: Number of descriptions found (even if no images)
    """
    # Get file basename for the id
    file_id = os.path.basename(output_path).split('_')[0]  # Extract ID from filename
    try:
        file_id = int(file_id)  # Try to convert to int if numeric
    except ValueError:
        file_id = hash(file_id) % 100000  # Use hash if not numeric
    
    # Create the base structure matching Label Studio format
    output_data = {
        "id": file_id,
        "annotations": [
            {
                "id": file_id + 10000,  # Create a unique annotation ID
                "completed_by": 1,  # Default user ID
                "result": [],  # Empty results
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "lead_time": 0.0,
                "prediction": {},
                "result_count": 0,
                "task": file_id
            }
        ],
        "data": {
            "name": f"{file_id}.jpg",
            "image": f"/data/local-files/?d=images/{file_id}.jpg"
        },
        "meta": {
            "status": "error",
            "reason": reason,
            "descriptions_found": num_descriptions
        },
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "project": 1
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Empty Label Studio format results saved to {output_path}: {reason}")
    
    return output_path

def create_visualization(image, results, output_path):
    """
    Visualize the matches between images and descriptions on the original image.
    """
    # Create a copy of the image for drawing
    img_vis = image.copy()
    
    # Colors for visualization
    IMAGE_COLOR = (0, 255, 0)      # Green for images
    DESC_COLOR = (0, 0, 255)       # Red for descriptions
    LINE_COLOR = (255, 0, 0)       # Blue for connections
    NO_MATCH_COLOR = (255, 165, 0) # Orange for images without matches
    
    # Draw all matches
    for result_id, match in results.items():
        img_box = match['image']['xyxy']
        
        # Use different color for images without descriptions
        box_color = IMAGE_COLOR if match['description'] else NO_MATCH_COLOR
        
        # Draw image box
        cv2.rectangle(img_vis, 
                     (int(img_box[0]), int(img_box[1])), 
                     (int(img_box[2]), int(img_box[3])), 
                     box_color, 2)
        
        # Draw text for image
        label = f"Image {result_id}" + (" (No match)" if not match['description'] else "")
        cv2.putText(img_vis, label, 
                   (int(img_box[0]), int(img_box[1] - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # If there's a matched description
        if match['description']:
            desc_box = match['description']['xyxy']
            
            # Draw description box
            cv2.rectangle(img_vis, 
                         (int(desc_box[0]), int(desc_box[1])), 
                         (int(desc_box[2]), int(desc_box[3])), 
                         DESC_COLOR, 2)
            
            # Draw text for description
            cv2.putText(img_vis, f"Desc {match['description']['id']}", 
                       (int(desc_box[0]), int(desc_box[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, DESC_COLOR, 2)
            
            # Draw a line connecting the image and its description
            img_center = match['image']['center']
            desc_center = match['description']['center']
            
            cv2.line(img_vis, 
                    (int(img_center[0]), int(img_center[1])), 
                    (int(desc_center[0]), int(desc_center[1])), 
                    LINE_COLOR, 2)
            
            # Draw the distance
            mid_x = (img_center[0] + desc_center[0]) / 2
            mid_y = (img_center[1] + desc_center[1]) / 2
            
            cv2.putText(img_vis, f"{match['distance']:.1f}px", 
                       (int(mid_x), int(mid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 2)
    
    # Save the image directly
    cv2.imwrite(output_path, img_vis)
    print(f"Visualization saved to {output_path}")
    
    return output_path

def process_image_with_model(model, image_path, output_dir=None, create_visualization=True):
    """
    Process an image with a pre-initialized YOLO model.
    
    Args:
        model: Pre-initialized YOLO model
        image_path: Path to the image file
        output_dir: Directory to save results (defaults to same dir as image)
        create_visualization: Whether to create and save visualization during processing
        
    Returns:
        Tuple of (results_dict, json_path, viz_path) where json_path is None if no "Obrázek" was found
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None, None, None
    
    # Create output paths
    if output_dir is None:
        # Use the same directory as the image
        base_path = os.path.splitext(image_path)[0]
        parent_dir = os.path.dirname(image_path)
    else:
        # Use the specified output directory with the image filename
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(os.path.splitext(image_path)[0])
        base_path = os.path.join(output_dir, base_name)
        parent_dir = os.path.dirname(output_dir)
    
    # Create visualization directory at the same level as output_dir
    viz_dir = os.path.join(parent_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Use just the ID as filename without the _matches suffix
    json_path = f"{base_path}.json"
    viz_path = os.path.join(viz_dir, f"{os.path.basename(base_path)}.jpg")
    
    try:
        # Run detection
        detection_result = model(image)[0]  # Get first detection result
        
        # Process the detection and get results
        if create_visualization:
            results, actual_json_path, _ = process_detections(
                detection=detection_result,
                base_path=base_path,
                image=image,
                json_path=json_path,
                viz_path=viz_path
            )
        else:
            # When skipping visualization during processing
            results, actual_json_path, _ = process_detections(
                detection=detection_result,
                base_path=base_path,
                image=image,
                json_path=json_path,
                viz_path=None  # Skip visualization during processing
            )
            
            # But create visualization separately at the end if we have results
            if results:
                viz_path = create_visualization(image, results, viz_path)
                print(f"Visualization created separately: {viz_path}")
        
        # Print summary
        if results:
            print("\nMatching Results:")
            for img_id, match in results.items():
                if match['description']:
                    print(f"Image {img_id} matched with Description {match['description']['id']}")
                else:
                    print(f"Image {img_id} has no matching description")
        else:
            print("No matches found")
        
        return results, actual_json_path, viz_path
        
    except Exception as e:
        # Handle any exceptions
        error_message = f"Error processing image {image_path}: {str(e)}"
        print(error_message)
        return {}, None, None

def example_usage():
    """Example of how to use this script with an existing YOLO model."""
    from ultralytics import YOLO
    
    # Load your model - replace with your actual model path or use a standard one
    model = YOLO("yolov8n.pt")  
    
    # Path to your image
    image_path = "test_image.jpg"
    
    # Process the image with the model
    results, json_path, viz_path = process_image_with_model(model, image_path)

# You can run this directly to see an example
if __name__ == "__main__":
    example_usage()