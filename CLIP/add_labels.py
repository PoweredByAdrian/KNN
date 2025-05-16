#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import argparse
import logging
from PIL import Image, ImageDraw, ImageFont
import random

# Setup logging
def setup_logging(debug_mode=False):
    """Configures the logging format and level."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logging.info(f"Log level set to: {'DEBUG' if debug_mode else 'INFO'}")
    if debug_mode:
        logging.debug("DEBUG logging enabled.")

def generate_random_color():
    """Generate a random RGB color tuple."""
    r = random.randint(50, 255)  # Avoid too dark colors
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    return (r, g, b)

def draw_rectangle_labels(image_path, json_path, output_path):
    """
    Draw rectangle labels from JSON file onto the image and save as a new file.
    
    Args:
        image_path (str): Path to the input image file
        json_path (str): Path to the JSON file containing rectangle label data
        output_path (str): Path to save the labeled image
    """
    try:
        # Load the image
        logging.info(f"Loading image: {image_path}")
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)  # Increased font size
        except IOError:
            try:
                font = ImageFont.truetype("Arial.ttf", 24)  # Increased font size
            except IOError:
                font = ImageFont.load_default()
                logging.warning("Using default font as TrueType fonts were not found")
        
        # Load the JSON data
        logging.info(f"Loading JSON data: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Track if any rectangles were found
        rectangles_found = False
        
        # Dictionary to store colors for each label type
        label_colors = {}
        
        # Extract annotations from the JSON data
        annotations = data.get("annotations", [])
        if not annotations:
            logging.warning("No annotations found in JSON file")
        
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            
            results = annotation.get("result", [])
            if not isinstance(results, list):
                continue
            
            # Process each result item
            for item in results:
                if not isinstance(item, dict):
                    continue
                
                if item.get("type") == "rectanglelabels":
                    value = item.get("value", {})
                    if not isinstance(value, dict):
                        continue
                    
                    # Get rectangle coordinates
                    coords = value.get("x", 0), value.get("y", 0), value.get("width", 0), value.get("height", 0)
                    if not all(isinstance(c, (int, float)) for c in coords):
                        continue
                    
                    # Convert relative coordinates (percentages) to absolute pixel coordinates
                    x_rel, y_rel, width_rel, height_rel = coords
                    img_width, img_height = img.size
                    
                    x1 = int(x_rel * img_width / 100)
                    y1 = int(y_rel * img_height / 100)
                    x2 = int((x_rel + width_rel) * img_width / 100)
                    y2 = int((y_rel + height_rel) * img_height / 100)
                    
                    # Get labels
                    labels = value.get("rectanglelabels", [])
                    if not labels or not isinstance(labels, list):
                        continue
                    
                    label_text = ", ".join(labels)
                    
                    # Get or create color for this label
                    if label_text not in label_colors:
                        # Generate more vibrant colors
                        r = random.randint(150, 255)  # Brighter colors
                        g = random.randint(150, 255)
                        b = random.randint(150, 255)
                        # Ensure at least one color component is fully bright
                        max_component = random.choice([0, 1, 2])
                        if max_component == 0:
                            r = 255
                        elif max_component == 1:
                            g = 255
                        else:
                            b = 255
                        label_colors[label_text] = (r, g, b)
                    
                    color = label_colors[label_text]
                    
                    # Draw rectangle with increased width (4 pixels instead of 2)
                    line_width = 5  # Increased from default 1 or 2
                    
                    # Draw multiple rectangles to create a thicker outline
                    for i in range(line_width):
                        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
                    
                    # Draw label background and text
                    text_size = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_size[2] - text_size[0]
                    text_height = text_size[3] - text_size[1]
                    
                    # Draw larger background for text
                    padding = 4  # Padding around text
                    draw.rectangle((x1-padding, y1-text_height-padding*2, 
                                   x1+text_width+padding, y1), 
                                   fill=color)
                    
                    # Draw text
                    draw.text((x1, y1-text_height-padding), label_text, fill="white", font=font)
                    
                    logging.debug(f"Drew rectangle for label '{label_text}' at {x1},{y1},{x2},{y2}")
                    rectangles_found = True
        
        if not rectangles_found:
            logging.warning("No rectangle labels found in the JSON data")
            return False
        
        # Save the labeled image
        img.save(output_path)
        logging.info(f"Labeled image saved to: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Draw rectangle labels from JSON onto the corresponding image")
    parser.add_argument("id", help="ID of the image and JSON file (without extension)")
    parser.add_argument("--images-dir", default="images", help="Directory containing the images")
    parser.add_argument("--jsons-dir", default="jsons", help="Directory containing the JSON files")
    parser.add_argument("--output-dir", default=".", help="Directory to save labeled images")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use relative paths from the script directory by default
    if not os.path.isabs(args.images_dir):
        images_dir = os.path.join(script_dir, args.images_dir)
    else:
        images_dir = args.images_dir
        
    if not os.path.isabs(args.jsons_dir):
        jsons_dir = os.path.join(script_dir, args.jsons_dir)
    else:
        jsons_dir = args.jsons_dir
    
    # For output, use current directory if specified as "."
    if args.output_dir == ".":
        output_dir = os.getcwd()
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Form file paths
    image_path = os.path.join(images_dir, f"{args.id}.jpg")
    json_path = os.path.join(jsons_dir, f"{args.id}.json")
    output_path = os.path.join(output_dir, f"{args.id}_labeled.jpg")
    
    # Check if files exist
    if not os.path.isfile(image_path):
        logging.error(f"Image file not found: {image_path}")
        sys.exit(1)
    
    if not os.path.isfile(json_path):
        logging.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    
    # Process the image
    success = draw_rectangle_labels(image_path, json_path, output_path)
    
    if success:
        print(f"Successfully created labeled image: {output_path}")
    else:
        print(f"Failed to create labeled image. See log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()