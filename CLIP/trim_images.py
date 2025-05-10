#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import argparse
import re
import time
from PIL import Image, ImageOps # Import Pillow

# --- Configuration ---
DEFAULT_JSONS_DIR_NAME = "jsons"       # Input dir for individual JSON files
DEFAULT_IMAGES_DIR_NAME = "images"     # Input dir for original images
DEFAULT_OUTPUT_DIR_NAME = "exported_images" # Output dir for cropped images
TARGET_LABEL = "Obrázek"               # The label to trigger cropping
LOG_LEVEL = logging.INFO               # Logging level
UUID_PATTERN = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'] # Common image extensions

# --- Logging Setup ---
def setup_logging(debug_mode=False):
    """Configures the logging format and level."""
    log_level = logging.DEBUG if debug_mode else LOG_LEVEL
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        force=True)
    logging.info(f"Log level set to: {'DEBUG' if debug_mode else 'INFO'}")
    if debug_mode:
        logging.debug("DEBUG logging enabled.")
# --- End Logging Setup ---

def find_image_file(image_dir, base_uuid):
    """Tries to find an image file matching the UUID with common extensions."""
    base_uuid_lower = base_uuid.lower()
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        filename = f"{base_uuid_lower}{ext}"
        filepath = os.path.join(image_dir, filename)
        if os.path.isfile(filepath):
            logging.debug(f"Found matching image file: {filepath}")
            return filepath
        # Check uppercase extension just in case
        filename_upper = f"{base_uuid_lower}{ext.upper()}"
        filepath_upper = os.path.join(image_dir, filename_upper)
        if os.path.isfile(filepath_upper):
            logging.debug(f"Found matching image file (uppercase ext): {filepath_upper}")
            return filepath_upper

    logging.warning(f"Could not find image file for UUID '{base_uuid}' in directory '{image_dir}' with extensions {ALLOWED_IMAGE_EXTENSIONS}")
    return None

def crop_and_save_image(input_image_path, output_dir, output_base_name, crop_box, index):
    """Opens, crops, and saves the image."""
    try:
        with Image.open(input_image_path) as img:
            logging.debug(f"Opened image: {input_image_path}")

            # Apply EXIF orientation correction BEFORE cropping
            try:
                img = ImageOps.exif_transpose(img)
                logging.debug("Applied EXIF transpose correction.")
            except Exception as exif_err:
                 logging.warning(f"Could not apply EXIF transpose to {input_image_path}: {exif_err} - proceeding without.")


            # Validate crop box against actual image dimensions after potential transpose
            img_width, img_height = img.size
            left, top, right, bottom = map(int, crop_box) # Ensure integers

            # Clamp coordinates to image boundaries
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)

            # Check if crop box is valid after clamping
            if left >= right or top >= bottom:
                logging.error(f"Invalid crop box dimensions after clamping for {input_image_path}: ({left},{top},{right},{bottom}). Skipping crop {index+1}.")
                return False

            logging.debug(f"Cropping to box: ({left}, {top}, {right}, {bottom})")
            cropped_img = img.crop((left, top, right, bottom))

            # Determine output filename and path
            img_ext = os.path.splitext(input_image_path)[1] # Get extension from original
            output_filename = f"{output_base_name}_{index + 1}{img_ext}" if index > 0 else f"{output_base_name}{img_ext}"
            output_filepath = os.path.join(output_dir, output_filename)

            # Save the cropped image
            cropped_img.save(output_filepath)
            logging.info(f"Saved cropped image: {output_filepath}")
            return True

    except FileNotFoundError:
        logging.error(f"Image file not found during open/crop: {input_image_path}")
        return False
    except IOError as e:
        logging.error(f"IOError processing/saving image {input_image_path} to {output_filepath}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error cropping image {input_image_path}: {e}", exc_info=logging.DEBUG)
        return False


def process_json_file(json_filepath, image_input_dir, image_output_dir):
    """Processes a single JSON file to find and crop 'Obrázek' labels."""
    logging.debug(f"Processing JSON file: {json_filepath}")
    crops_created_for_file = 0
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        if not isinstance(task_data, dict):
            logging.warning(f"Skipping invalid JSON file (not dict): {json_filepath}")
            return 0 # Return 0 crops created

        # 1. Find UUID and corresponding image path
        data_field = task_data.get("data", {})
        image_path_ls = data_field.get("image") # Path from Label Studio JSON

        if not image_path_ls or not isinstance(image_path_ls, str):
             logging.warning(f"Skipping JSON: No valid 'data.image' path found in {json_filepath}")
             return 0

        match = re.search(UUID_PATTERN, image_path_ls)
        if not match:
            logging.warning(f"Skipping JSON: No UUID found in 'data.image' path '{image_path_ls}' in {json_filepath}")
            return 0
        base_uuid = match.group(0)
        logging.debug(f"Extracted UUID '{base_uuid}' from {json_filepath}")

        input_image_path = find_image_file(image_input_dir, base_uuid)
        if not input_image_path:
            logging.error(f"Skipping JSON: Could not find matching image for UUID '{base_uuid}' in '{image_input_dir}' mentioned in {json_filepath}")
            return 0 # Cannot proceed without the image

        # 2. Iterate through annotations and results
        annotations = task_data.get("annotations", [])
        if not isinstance(annotations, list):
            logging.warning(f"Invalid 'annotations' format (not a list) in {json_filepath}")
            return 0

        crop_index = 0 # For appending _1, _2 etc. if multiple crops

        for annotation in annotations:
            if not isinstance(annotation, dict): continue
            results = annotation.get("result", [])
            if not isinstance(results, list): continue

            for item in results:
                if not isinstance(item, dict): continue

                # Check for the target label
                if item.get("type") == "rectanglelabels":
                    value_dict = item.get("value", {})
                    if not isinstance(value_dict, dict): continue

                    labels_list = value_dict.get("rectanglelabels")
                    if isinstance(labels_list, list) and TARGET_LABEL in labels_list:
                        logging.debug(f"Found target label '{TARGET_LABEL}' in {json_filepath}")

                        # Extract coordinates and original dimensions
                        try:
                            x = float(value_dict['x'])
                            y = float(value_dict['y'])
                            w = float(value_dict['width'])
                            h = float(value_dict['height'])
                            orig_w = int(item['original_width'])
                            orig_h = int(item['original_height'])

                            if w <= 0 or h <= 0 or orig_w <= 0 or orig_h <= 0:
                                 raise ValueError("Dimensions must be positive")

                        except (KeyError, ValueError, TypeError) as e:
                            logging.error(f"Missing or invalid coordinate/dimension data for '{TARGET_LABEL}' in {json_filepath}: {e}. Skipping this box.")
                            continue # Skip this specific crop

                        # Calculate absolute pixel coordinates
                        left = x / 100.0 * orig_w
                        top = y / 100.0 * orig_h
                        right = (x + w) / 100.0 * orig_w
                        bottom = (y + h) / 100.0 * orig_h

                        crop_box = (left, top, right, bottom)

                        # Perform the crop and save
                        success = crop_and_save_image(
                            input_image_path,
                            image_output_dir,
                            base_uuid, # Use UUID as the base name
                            crop_box,
                            crop_index # Pass the current index
                        )
                        if success:
                            crops_created_for_file += 1
                            crop_index += 1 # Increment index ONLY on successful crop

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file {json_filepath}: {e}")
    except IOError as e:
        logging.error(f"Error reading JSON file {json_filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing JSON {json_filepath}: {e}", exc_info=logging.DEBUG)

    return crops_created_for_file


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Crop images based on '{TARGET_LABEL}' annotations in individual JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "jsons_dir",
        nargs='?',
        default=DEFAULT_JSONS_DIR_NAME,
        help="Directory containing the individual task JSON files."
    )
    parser.add_argument(
        "--image-dir", # Changed from jsons_dir for clarity
        default=DEFAULT_IMAGES_DIR_NAME,
        help="Directory containing the original source image files."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Directory where cropped images will be saved."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed DEBUG logging output."
    )

    args = parser.parse_args()
    setup_logging(args.debug)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logging.debug("__file__ not defined, using CWD as script base: %s", script_dir)

    # Calculate absolute paths
    jsons_dir_abs = os.path.abspath(os.path.join(script_dir, args.jsons_dir))
    image_dir_abs = os.path.abspath(os.path.join(script_dir, args.image_dir))
    output_dir_abs = os.path.abspath(os.path.join(script_dir, args.output_dir))

    logging.info(f"Starting image cropping process.")
    logging.info(f"Input JSON directory: {jsons_dir_abs}")
    logging.info(f"Input Image directory: {image_dir_abs}")
    logging.info(f"Output directory for crops: {output_dir_abs}")
    logging.info(f"Target label for cropping: '{TARGET_LABEL}'")

    # Validate input directories
    if not os.path.isdir(jsons_dir_abs):
        logging.critical(f"Input JSON directory does not exist: {jsons_dir_abs}")
        sys.exit(1)
    if not os.path.isdir(image_dir_abs):
        logging.critical(f"Input Image directory does not exist: {image_dir_abs}")
        sys.exit(1)

    # Create output directory
    try:
        os.makedirs(output_dir_abs, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir_abs}")
    except OSError as e:
        logging.critical(f"Could not create output directory '{output_dir_abs}': {e}")
        sys.exit(1)

    # Process files
    total_json_files_processed = 0
    total_crops_created = 0
    total_json_files_found = 0
    json_files_with_errors = 0

    start_time = time.time()

    try:
        json_filenames = [f for f in os.listdir(jsons_dir_abs) if f.lower().endswith('.json') and os.path.isfile(os.path.join(jsons_dir_abs, f))]
        total_json_files_found = len(json_filenames)
        logging.info(f"Found {total_json_files_found} JSON files to check.")

        for filename in json_filenames:
            json_filepath = os.path.join(jsons_dir_abs, filename)
            try:
                 crops = process_json_file(json_filepath, image_dir_abs, output_dir_abs)
                 total_crops_created += crops
                 total_json_files_processed += 1
            except Exception: # Catch errors within the processing loop if process_json_file raises something unexpected
                 logging.error(f"Critical error during processing of {filename}", exc_info=True)
                 json_files_with_errors +=1


    except OSError as e:
         logging.critical(f"Error listing files in JSON directory {jsons_dir_abs}: {e}")
         sys.exit(1)

    elapsed_time = time.time() - start_time

    # Final Summary
    logging.info("\n--- Cropping Summary ---")
    logging.info(f"Processing completed in: {elapsed_time:.2f} seconds")
    logging.info(f"Total JSON files found: {total_json_files_found}")
    logging.info(f"JSON files processed: {total_json_files_processed}")
    logging.info(f"JSON files with processing errors: {json_files_with_errors}")
    logging.info(f"Total '{TARGET_LABEL}' crops created: {total_crops_created}")
    logging.info("------------------------")
    logging.info("Image cropping script finished.")