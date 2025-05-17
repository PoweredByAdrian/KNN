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
from typing import Dict, List, Tuple, Optional, Any, Union

# --- Configuration ---
SCRIPT_VERSION = "1.1"
DEFAULT_JSONS_DIR_NAME = "jsons"       # Input dir for individual JSON files
DEFAULT_IMAGES_DIR_NAME = "images"     # Input dir for original images
DEFAULT_OUTPUT_DIR_NAME = "exported_images" # Output dir for cropped images
TARGET_LABEL = "Obrázek"               # The label to trigger cropping
LOG_LEVEL = logging.INFO               # Logging level
UUID_PATTERN = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'] # Common image extensions

# Track if logging has been set up
_logging_setup_done = False

# --- Logging Setup ---
def setup_logging(debug_mode=False, use_notebook=False):
    """
    Configures the logging format and level.
    
    Args:
        debug_mode: Enable debug logging
        use_notebook: Set to True when running in a Jupyter notebook
    """
    global _logging_setup_done
    
    log_level = logging.DEBUG if debug_mode else LOG_LEVEL
    
    # Configure log format based on environment
    if use_notebook:
        # Simpler format for notebooks with no timestamps (cleaner output)
        log_format = '%(levelname)s: %(message)s'
    else:
        # More detailed format for scripts
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # If logging is already configured, reset handlers
    if _logging_setup_done:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Setup basic configuration
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True
    )
    
    # Set flag that logging has been configured
    _logging_setup_done = True
    
    logging.info(f"Log level set to: {'DEBUG' if debug_mode else 'INFO'}")
    if debug_mode:
        logging.debug("DEBUG logging enabled.")
# --- End Logging Setup ---

def find_image_file(image_dir, base_uuid):
    """
    Tries to find an image file matching the UUID with common extensions.
    
    Args:
        image_dir: Directory containing images
        base_uuid: The UUID to search for
        
    Returns:
        Path to the image file if found, None otherwise
    """
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
    """
    Opens, crops, and saves the image.
    
    Args:
        input_image_path: Path to the input image
        output_dir: Directory to save cropped images
        output_base_name: Base name for the output file
        crop_box: Tuple of (left, top, right, bottom)
        index: Index for multiple crops of the same image
        
    Returns:
        True if successful, False otherwise
    """
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
            logging.debug(f"Saved cropped image: {output_filepath}")
            return True

    except FileNotFoundError:
        logging.error(f"Image file not found during open/crop: {input_image_path}")
        return False
    except IOError as e:
        logging.error(f"IOError processing/saving image {input_image_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error cropping image {input_image_path}: {e}", exc_info=logging.DEBUG)
        return False


def process_json_file(json_filepath, image_input_dir, image_output_dir, target_label=TARGET_LABEL):
    """
    Processes a single JSON file to find and crop labels.
    
    Args:
        json_filepath: Path to the JSON file
        image_input_dir: Directory containing source images
        image_output_dir: Directory to save cropped images
        target_label: Label to look for in rectanglelabels (default: TARGET_LABEL)
        
    Returns:
        Number of crops created
    """
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
                    if isinstance(labels_list, list) and target_label in labels_list:
                        logging.debug(f"Found target label '{target_label}' in {json_filepath}")

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
                            logging.error(f"Missing or invalid coordinate/dimension data for '{target_label}' in {json_filepath}: {e}. Skipping this box.")
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


def run_crop_images(
    jsons_dir: str = DEFAULT_JSONS_DIR_NAME,
    images_dir: str = DEFAULT_IMAGES_DIR_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR_NAME,
    target_label: str = TARGET_LABEL,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Main function to crop images based on JSON files with rectangle labels.
    Can be called programmatically from other modules.
    
    Args:
        jsons_dir: Directory containing JSON files
        images_dir: Directory containing source images
        output_dir: Directory to save cropped images
        target_label: Label to filter by
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with statistics and results
    """
    # Calculate absolute paths
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logging.debug("__file__ not defined, using CWD as script base")
    
    jsons_dir_abs = os.path.abspath(jsons_dir)
    images_dir_abs = os.path.abspath(images_dir)
    output_dir_abs = os.path.abspath(output_dir)
    
    logging.info(f"Starting image cropping process")
    logging.info(f"Input JSON directory: {jsons_dir_abs}")
    logging.info(f"Input Image directory: {images_dir_abs}")
    logging.info(f"Output directory for crops: {output_dir_abs}")
    logging.info(f"Target label for cropping: '{target_label}'")
    
    # Validate input directories
    if not os.path.isdir(jsons_dir_abs):
        logging.error(f"Input JSON directory does not exist: {jsons_dir_abs}")
        return {"error": f"JSON directory not found: {jsons_dir_abs}"}
        
    if not os.path.isdir(images_dir_abs):
        logging.error(f"Input Image directory does not exist: {images_dir_abs}")
        return {"error": f"Images directory not found: {images_dir_abs}"}
    
    # Create output directory
    try:
        os.makedirs(output_dir_abs, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir_abs}")
    except OSError as e:
        logging.error(f"Could not create output directory '{output_dir_abs}': {e}")
        return {"error": f"Failed to create output directory: {str(e)}"}
    
    # Process files
    total_json_files_processed = 0
    total_crops_created = 0
    total_json_files_found = 0
    json_files_with_errors = 0
    
    start_time = time.time()
    
    try:
        # Get list of JSON files
        json_filenames = [f for f in os.listdir(jsons_dir_abs) 
                          if f.lower().endswith('.json') and os.path.isfile(os.path.join(jsons_dir_abs, f))]
        total_json_files_found = len(json_filenames)
        logging.info(f"Found {total_json_files_found} JSON files to process")
        
        # Create progress iterator if requested
        if show_progress:
            try:
                # Try to import tqdm.notebook for Jupyter environments
                try:
                    from tqdm.notebook import tqdm
                    pbar = tqdm(json_filenames, desc="Cropping images", unit="file")
                    logging.debug("Using tqdm.notebook for progress bars")
                except ImportError:
                    # Fall back to standard tqdm for terminal environments
                    from tqdm import tqdm
                    pbar = tqdm(json_filenames, desc="Cropping images", unit="file")
                    logging.debug("Using standard tqdm for progress bars")
                
                # Use the progress bar iterator
                file_iterator = pbar
            except ImportError:
                # If tqdm isn't available at all, fall back to regular iteration with log messages
                logging.warning("tqdm not available, falling back to log-based progress")
                file_iterator = json_filenames
                # Will use regular logging for progress
        else:
            # No progress display requested
            file_iterator = json_filenames
        
        # Process each JSON file
        for i, filename in enumerate(file_iterator):
            json_filepath = os.path.join(jsons_dir_abs, filename)
            
            try:
                # Process the file
                crops = process_json_file(json_filepath, images_dir_abs, output_dir_abs, target_label)
                total_crops_created += crops
                total_json_files_processed += 1
                
                # Update progress description if using tqdm
                if show_progress and 'pbar' in locals():
                    pbar.set_postfix(crops=total_crops_created)
                    
                # Traditional progress logging as fallback
                elif show_progress and (i+1) % 50 == 0:
                    progress = ((i+1) / total_json_files_found) * 100
                    logging.info(f"Progress: {progress:.1f}% ({i+1}/{total_json_files_found}) - {total_crops_created} crops created")
                    
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                json_files_with_errors += 1
        
        # Close progress bar if it exists
        if show_progress and 'pbar' in locals():
            pbar.close()
            
    except OSError as e:
        logging.error(f"Error listing files in JSON directory {jsons_dir_abs}: {e}")
        return {"error": f"Failed to list JSON files: {str(e)}"}
    
    elapsed_time = time.time() - start_time
    
    # Return the results
    result = {
        "total_files_found": total_json_files_found,
        "files_processed": total_json_files_processed,
        "files_with_errors": json_files_with_errors,
        "crops_created": total_crops_created,
        "target_label": target_label,
        "elapsed_time": elapsed_time,
        "jsons_dir": jsons_dir_abs,
        "images_dir": images_dir_abs,
        "output_dir": output_dir_abs
    }
    
    logging.info(f"Processing completed in: {elapsed_time:.2f} seconds")
    logging.info(f"Total JSON files found: {total_json_files_found}")
    logging.info(f"JSON files processed: {total_json_files_processed}")
    logging.info(f"JSON files with processing errors: {json_files_with_errors}")
    logging.info(f"Total '{target_label}' crops created: {total_crops_created}")
    
    return result


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Crop images based on rectangle label annotations in JSON files (v{SCRIPT_VERSION}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "jsons_dir",
        nargs='?',
        default=DEFAULT_JSONS_DIR_NAME,
        help="Directory containing the individual task JSON files."
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGES_DIR_NAME,
        help="Directory containing the original source image files."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Directory where cropped images will be saved."
    )
    parser.add_argument(
        "--label",
        default=TARGET_LABEL,
        help=f"Label to search for in rectangle annotations (default: '{TARGET_LABEL}')"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed DEBUG logging output."
    )

    args = parser.parse_args()
    
    # Set up logging for command-line use
    setup_logging(args.debug, use_notebook=False)
    
    # Run the main function
    result = run_crop_images(
        jsons_dir=args.jsons_dir,
        images_dir=args.image_dir,
        output_dir=args.output_dir,
        target_label=args.label,
        show_progress=True
    )
    
    # Print final summary
    if "error" in result:
        logging.critical(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print("\n--- Cropping Summary ---")
        print(f"Processing completed in: {result['elapsed_time']:.2f} seconds")
        print(f"Total JSON files found: {result['total_files_found']}")
        print(f"JSON files processed: {result['files_processed']}")
        print(f"JSON files with errors: {result['files_with_errors']}")
        print(f"Total '{result['target_label']}' crops created: {result['crops_created']}")
        print("------------------------")
        logging.info("Image cropping script finished.")