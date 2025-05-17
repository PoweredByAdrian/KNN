import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from collections import defaultdict

def process_detections(detection, image, json_path="detections.json", viz_path="visualization.png"):
    """
    Process YOLO detections for a single image, match descriptions to images,
    save visualization and JSON results.
    
    Args:
        detection: YOLO detection result for a single image
        image: Original image as numpy array
        json_path: Path to save JSON results
        viz_path: Path to save visualization image
        
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
    
    # If no images found, return empty results
    if not images:
        print("No images found in detection")
        return {}, json_path, viz_path
        
    # Match descriptions to images based on proximity
    results = {}
    assigned_descriptions = set()
    
    # For each image, find the closest unassigned description
    for img in images:
        img_center = img['center']
        closest_desc = None
        min_distance = float('inf')
        
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
        
        # Assign the closest description to this image
        if closest_desc:
            assigned_descriptions.add(closest_desc['id'])
            results[img['id']] = {
                'image': img,
                'description': closest_desc,
                'distance': min_distance
            }
        else:
            results[img['id']] = {
                'image': img,
                'description': None,
                'distance': None
            }
    
    # Check for unassigned descriptions
    unassigned_descriptions = [desc for desc in descriptions 
                              if desc['id'] not in assigned_descriptions]
    
    if unassigned_descriptions:
        print(f"Warning: {len(unassigned_descriptions)} descriptions were not assigned to any image")
    
    # Save to JSON
    save_matches_to_json(results, json_path)
    
    # Create visualization
    create_visualization(image, results, viz_path)
    
    return results, json_path, viz_path

def save_matches_to_json(results, output_path):
    """
    Save matching information to a JSON file.
    """
    # Create a structured output format
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(results),
        "matches": []
    }
    
    # Process each match
    for img_id, match in results.items():
        img_info = match['image']
        
        # Create image entry
        image_entry = {
            "image_id": img_id,
            "confidence": float(img_info['conf']),
            "bbox": img_info['xyxy'],
            "center": list(img_info['center']),
            "size": {
                "width": float(img_info['width']),
                "height": float(img_info['height'])
            }
        }
        
        # Add description info if available
        if match['description']:
            desc_info = match['description']
            image_entry["matched_description"] = {
                "description_id": desc_info['id'],
                "confidence": float(desc_info['conf']),
                "bbox": desc_info['xyxy'],
                "center": list(desc_info['center']),
                "size": {
                    "width": float(desc_info['width']),
                    "height": float(desc_info['height'])
                },
                "matching_distance": float(match['distance'])
            }
        else:
            image_entry["matched_description"] = None
            
        output_data["matches"].append(image_entry)
    
    # Add summary statistics
    matched_count = sum(1 for match in output_data["matches"] if match["matched_description"] is not None)
    output_data["summary"] = {
        "total_images": len(output_data["matches"]),
        "matched_images": matched_count,
        "unmatched_images": len(output_data["matches"]) - matched_count,
        "match_rate": matched_count / len(output_data["matches"]) if output_data["matches"] else 0
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Matches saved to {output_path}")
    
    return output_path

def create_visualization(image, results, output_path):
    """
    Visualize the matches between images and descriptions on the original image.
    """
    # Create a copy of the image for drawing
    img_vis = image.copy()
    
    # Colors for visualization
    IMAGE_COLOR = (0, 255, 0)  # Green for images
    DESC_COLOR = (0, 0, 255)   # Red for descriptions
    LINE_COLOR = (255, 0, 0)   # Blue for connections
    
    # Draw all matches
    for result_id, match in results.items():
        img_box = match['image']['xyxy']
        
        # Draw image box
        cv2.rectangle(img_vis, 
                     (int(img_box[0]), int(img_box[1])), 
                     (int(img_box[2]), int(img_box[3])), 
                     IMAGE_COLOR, 2)
        
        # Draw text for image
        cv2.putText(img_vis, f"Image {result_id}", 
                   (int(img_box[0]), int(img_box[1] - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, IMAGE_COLOR, 2)
        
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

def example_usage():
    """Example of how to use this simplified script with an existing YOLO model."""
    from ultralytics import YOLO
    
    # Load your model - replace with your actual model path or use a standard one
    model = YOLO("yolov8n.pt")  
    
    # Path to your image
    image_path = "test_image.jpg"
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Run detection
    detection_result = model(image)[0]  # Get first detection result
    
    # Process the detection and get results
    results, json_path, viz_path = process_detections(
        detection=detection_result,
        image=image,
        json_path=f"{os.path.splitext(image_path)[0]}_matches.json",
        viz_path=f"{os.path.splitext(image_path)[0]}_visualization.png"
    )
    
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

# You can run this directly to see an example
if __name__ == "__main__":
    example_usage()