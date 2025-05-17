#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import logging
import re
import time
from typing import Optional, Tuple, List, Dict, Any
import sys
import shutil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

# --- Configuration ---
SCRIPT_VERSION = "1.1"
DEFAULT_JSON_DIR = "jsons"
DEFAULT_IMAGES_DIR = "exported_images"
DEFAULT_TEXTS_DIR = "filtered_texts"
DEFAULT_OUTPUT_DIR = "output_context"
DEFAULT_SIMILARITY_THRESHOLD = 0.25
DEFAULT_MAX_LINES_CONTEXT = 3
DEFAULT_MAX_IMAGE_SUFFIX = 20
DEFAULT_MODEL_NAME = "ViT-B/32"

# Constants
UUID_PATTERN = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
TARGET_LABEL = "Obrázek"

# Track if logging has been set up
_logging_setup_done = False

# Try importing CLIP and other required modules
try:
    import clip
    from cut_text import OCRDocument, TextBlock, draw_blocks_on_image
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure clip and cut_text modules are installed/available")
    sys.exit(1)

# --- Logging Setup ---
def setup_logging(debug_mode=False, use_notebook=False, log_to_file=True):
    """
    Configure logging format and level.
    
    Args:
        debug_mode: Enable debug logging
        use_notebook: Set to True when running in a Jupyter notebook
        log_to_file: Whether to also log to a file
    """
    global _logging_setup_done
    
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure log format based on environment
    if use_notebook:
        # Simpler format for notebooks with no timestamps
        log_format = '%(levelname)s: %(message)s'
    else:
        # More detailed format for scripts
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # If logging is already configured, reset handlers
    if _logging_setup_done:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create a logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    
    # Use a more visible format for console in terminals that support colors
    if sys.stdout.isatty() and not use_notebook:
        # Colors for different log levels
        class ColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[94m',    # Blue
                'INFO': '\033[92m',     # Green
                'WARNING': '\033[93m',  # Yellow
                'ERROR': '\033[91m',    # Red
                'CRITICAL': '\033[41m', # Red background
                'ENDC': '\033[0m'       # Reset color
            }
            
            def format(self, record):
                levelname = record.levelname
                if levelname in self.COLORS:
                    levelname_color = self.COLORS[levelname] + levelname + self.COLORS['ENDC']
                    record.levelname = levelname_color
                return super().format(record)
        
        formatter = ColorFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)
    
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    
    # File handler (optional)
    if log_to_file and not use_notebook:
        try:
            file_handler = logging.FileHandler('process_descriptions.log', mode='w')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s'))
            root_logger.addHandler(file_handler)
            logging.info("File logging enabled to process_descriptions.log")
        except Exception as e:
            logging.warning(f"Could not set up file logging: {e}")
    
    # Set flag that logging has been configured
    _logging_setup_done = True
    
    if debug_mode:
        logging.debug("DEBUG logging enabled.")
    else:
        logging.info("INFO logging enabled - use debug=True for more details")
# --- End Logging Setup ---

def has_obrazek_label(json_filepath):
    """Check if the JSON file contains 'Obrázek' label."""
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
            
        if not isinstance(task_data, dict):
            return False
            
        annotations = task_data.get("annotations", [])
        if not isinstance(annotations, list):
            return False
            
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
                
            results = annotation.get("result", [])
            if not isinstance(results, list):
                continue
                
            for item in results:
                if not isinstance(item, dict):
                    continue
                    
                if item.get("type") == "rectanglelabels":
                    value_dict = item.get("value", {})
                    if not isinstance(value_dict, dict):
                        continue
                        
                    labels_list = value_dict.get("rectanglelabels")
                    if isinstance(labels_list, list) and TARGET_LABEL in labels_list:
                        # Found the target label
                        return True
                        
        return False
    except Exception as e:
        logging.error(f"Error checking for '{TARGET_LABEL}' in {json_filepath}: {e}")
        return False

def extract_id_from_json(json_filepath):
    """Extract UUID from JSON file path or content."""
    # First try to extract from filename
    filename = os.path.basename(json_filepath)
    match = re.search(UUID_PATTERN, filename)
    if match:
        return match.group(0)
        
    # If not found in filename, try to extract from file content
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
            
        if not isinstance(task_data, dict):
            return None
            
        # Try to find UUID in data.image path
        data_field = task_data.get("data", {})
        image_path = data_field.get("image", "")
        if isinstance(image_path, str):
            match = re.search(UUID_PATTERN, image_path)
            if match:
                return match.group(0)
    except Exception as e:
        logging.error(f"Error extracting ID from {json_filepath}: {e}")
        
    return None

def get_obrazek_json_ids(json_dir):
    """Find all JSON files with 'Obrázek' label and return their IDs."""
    obrazek_ids = []
    
    try:
        json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
        logging.info(f"Found {len(json_files)} total JSON files in {json_dir}")
        
        for filename in tqdm(json_files, desc="Checking JSONs for Obrázek label"):
            json_filepath = os.path.join(json_dir, filename)
            
            if has_obrazek_label(json_filepath):
                id_value = extract_id_from_json(json_filepath)
                if id_value:
                    obrazek_ids.append(id_value)
                    logging.debug(f"Found 'Obrázek' label in {filename}, ID: {id_value}")
                    
        logging.info(f"Found {len(obrazek_ids)} JSONs with 'Obrázek' label")
        return obrazek_ids
        
    except Exception as e:
        logging.error(f"Error getting JSON IDs with 'Obrázek' label: {e}")
        return []

def find_available_images(base_image_path, max_suffix=20):
    """
    Find all available image files by checking the base name and suffixed versions.
    
    Args:
        base_image_path: Base path without suffix (e.g., "exported_images/ID.jpg")
        max_suffix: Maximum suffix number to check
        
    Returns:
        List of paths to all available image files for this ID
    """
    available_images = []
    
    # Check if the base image exists
    if os.path.exists(base_image_path):
        available_images.append(base_image_path)
        
    # Check for images with suffixes _2, _3, etc.
    base_name, extension = os.path.splitext(base_image_path)
    
    for suffix in range(2, max_suffix + 1):
        suffixed_path = f"{base_name}_{suffix}{extension}"
        if os.path.exists(suffixed_path):
            available_images.append(suffixed_path)
        else:
            break
            
    if available_images:
        logging.debug(f"Found {len(available_images)} image(s): {', '.join(os.path.basename(img) for img in available_images)}")
    else:
        logging.warning(f"No images found for base path: {os.path.basename(base_image_path)}")
        
    return available_images

def find_best_matching_block(
    image_path: str,
    xml_path: str,
    model,
    preprocess,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[Tuple[TextBlock, float, List[float], List[float], OCRDocument]]:
    """
    Given an image and OCR XML, returns the block whose text best matches the image
    based on cosine similarity.
    """
    # Check if files exist
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return None
        
    if not os.path.exists(xml_path):
        logging.error(f"XML file not found: {xml_path}")
        return None

    # 1. Load OCR blocks
    doc = OCRDocument(xml_path)
    blocks = doc.generate_blocks(lines_per_block=1, overlap=0)
    if not blocks:
        logging.warning(f"No text blocks found in {xml_path}")
        return None

    # 2. Preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

    # 3. Tokenize block texts
    texts = [block.get_text() for block in blocks]
    text_tokens = clip.tokenize(texts, truncate=True).to(device)

    # 4. CLIP encodings - only calculate cosine similarities, not softmax
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Only calculate cosine similarities
        image_features_norm = F.normalize(image_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        similarities = torch.matmul(image_features_norm, text_features_norm.T)[0].cpu().tolist()

    # Find best match based on cosine similarity
    best_idx = int(torch.tensor(similarities).argmax())
    best_similarity = similarities[best_idx]

    # We'll still return a list in the probability position to maintain function signature compatibility
    # but it will contain the same similarities, not softmax probabilities
    return blocks[best_idx], best_similarity, similarities, similarities, doc

def process_id(
    id_value: str, 
    model, 
    preprocess,
    images_dir: str = DEFAULT_IMAGES_DIR,
    texts_dir: str = DEFAULT_TEXTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_lines_context: int = DEFAULT_MAX_LINES_CONTEXT,
    max_image_suffix: int = DEFAULT_MAX_IMAGE_SUFFIX,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    best_only: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single ID with the pre-loaded CLIP model.
    """
    start_time = time.time()
    
    # Track statistics
    images_processed = 0
    images_below_threshold = 0
    
    # Construct paths
    base_image_path = os.path.join(images_dir, f"{id_value}.jpg")
    xml_path = os.path.join(texts_dir, f"{id_value}.xml")
    
    # Check if XML exists - moved to debug level
    if not os.path.exists(xml_path):
        logging.error(f"XML file not found: {xml_path}")
        return {
            "id": id_value,
            "success": False,
            "error": f"XML file not found: {xml_path}",
            "time": time.time() - start_time
        }

    # Find all available image files
    available_images = find_available_images(base_image_path, max_image_suffix)
    if not available_images:
        logging.error(f"No images found for ID {id_value}")
        return {
            "id": id_value,
            "success": False,
            "error": f"No images found for ID {id_value}",
            "time": time.time() - start_time
        }

    # Reduced verbosity - moved to debug level
    logging.debug(f"Processing ID: {id_value}")
    logging.debug(f"XML path: {xml_path}")
    logging.debug(f"Found {len(available_images)} images")
    
    # Store all context blocks from all images
    all_context_blocks = []
    
    # Process each image
    for idx, image_path in enumerate(available_images):
        # Reduced to debug level
        logging.debug(f"Processing image {idx+1}/{len(available_images)}: {image_path}")
        images_processed += 1
        
        # Find best matching block for this image
        result = find_best_matching_block(
            image_path=image_path,
            xml_path=xml_path,
            model=model,
            preprocess=preprocess,
            device=device
        )

        if result is None:
            logging.warning(f"Skipping image {image_path} due to error")
            continue

        # Extract the actual image ID from the found image path
        image_filename = os.path.basename(image_path)
        image_id_with_suffix = os.path.splitext(image_filename)[0]  # Remove extension
        
        # Now the result contains best_block, best_similarity, all_similarities, _, doc
        best_block, best_similarity, all_similarities, _, doc = result
        # Moved to debug level
        logging.debug(f"Matched block with {best_similarity:.4f} cosine similarity: \"{best_block.get_text()[:100]}...\"")

        # Build contextual block around the best matching text
        logging.debug("Building context block around best match...")  # Changed to debug

        # Instead of just building context around the best match,
        # build context based on the selected approach (all above threshold or best only)
        above_threshold_blocks = []
        
        # Find blocks to include based on chosen approach
        if best_only:
            # Only use the best match for this image (if it's above threshold)
            best_idx = int(torch.tensor(all_similarities).argmax())
            if all_similarities[best_idx] > similarity_threshold:
                above_threshold_blocks = [best_idx]
                logging.debug(f"Using best match only (similarity: {all_similarities[best_idx]:.4f})") # Changed to debug
            else:
                logging.debug(f"Best match similarity ({all_similarities[best_idx]:.4f}) below threshold ({similarity_threshold})") # Changed to debug
                images_below_threshold += 1
        else:
            # Use all blocks above threshold (original behavior)
            for idx, similarity in enumerate(all_similarities):
                if similarity > similarity_threshold:
                    above_threshold_blocks.append(idx)
            
            if above_threshold_blocks:
                logging.debug(f"Found {len(above_threshold_blocks)} blocks above threshold {similarity_threshold}") # Changed to debug
            else:
                logging.debug(f"No blocks above threshold {similarity_threshold} found") # Changed to debug
                images_below_threshold += 1
        
        if above_threshold_blocks:
            # We'll reuse the blocks generated earlier
            blocks = doc.generate_blocks(lines_per_block=1, overlap=0)
            
            # Process each above-threshold block
            all_added_blocks = set()  # To track blocks we've already added
            
            for block_idx in above_threshold_blocks:
                # Skip if we've already added this block through another context
                if block_idx in all_added_blocks:
                    continue
                    
                # Start with this block
                context_blocks = [blocks[block_idx]]
                all_added_blocks.add(block_idx)
                context_text = blocks[block_idx].get_text()
                
                # Add blocks above if they meet similarity threshold
                for i in range(1, max_lines_context + 1):
                    check_idx = block_idx - i
                    if check_idx >= 0 and all_similarities[check_idx] > similarity_threshold:
                        if check_idx not in all_added_blocks:
                            context_blocks.insert(0, blocks[check_idx])
                            all_added_blocks.add(check_idx)
                            context_text = blocks[check_idx].get_text() + "\n" + context_text
                    else:
                        break
                
                # Add blocks below if they meet similarity threshold
                for i in range(1, max_lines_context + 1):
                    check_idx = block_idx + i
                    if check_idx < len(blocks) and all_similarities[check_idx] > similarity_threshold:
                        if check_idx not in all_added_blocks:
                            context_blocks.append(blocks[check_idx])
                            all_added_blocks.add(check_idx)
                            context_text += "\n" + blocks[check_idx].get_text()
                    else:
                        break
                
                # Add these context blocks to our collection
                all_context_blocks.extend(context_blocks)
            
            if best_only:
                logging.debug(f"Built context around best match with {len(all_context_blocks)} total blocks") # Changed to debug
            else:
                logging.debug(f"Built context blocks around {len(above_threshold_blocks)} matches above threshold") # Changed to debug
            
            logging.debug(f"Total blocks added to context: {len(all_context_blocks)}") # Changed to debug
            
            if verbose and logging.getLogger().level <= logging.DEBUG:  # Only show details if we're in debug mode
                for idx in above_threshold_blocks:
                    logging.debug(f"  Block {idx}: {all_similarities[idx]:.4f} - {blocks[idx].get_text()[:50]}...")
    
    # After processing all images, visualize all accumulated context blocks on the original image
    if all_context_blocks:
        # Get original image path (use the ID without any suffix)
        original_image_path = os.path.join("images", f"{id_value}.jpg")
        
        # If original image doesn't exist, use the first available image we found earlier
        if not os.path.exists(original_image_path):
            original_image_path = available_images[0]
            logging.debug(f"Original image not found at {original_image_path}, using {available_images[0]} instead") # Changed to debug
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize all context blocks on the image
        try:
            # Call the function without the output_name parameter
            context_image = draw_blocks_on_image(original_image_path, all_context_blocks)
            
            # Define output filename
            output_file = f"{id_value}_all_contexts.jpg"
            output_path = os.path.join(output_dir, output_file)
            
            # Copy or move the generated image to the output directory
            if os.path.exists(context_image):
                shutil.copy(context_image, output_path)
                logging.info(f"✓ Context visualization saved: {output_file}")  # Keep this as INFO but formatted better
                
                # Clean up original if it's not the final destination
                if context_image != output_path and os.path.exists(context_image):
                    os.remove(context_image)
                
                processing_time = time.time() - start_time
                return {
                    "id": id_value,
                    "success": True,
                    "context_blocks": len(all_context_blocks),
                    "images_processed": images_processed,
                    "images_below_threshold": images_below_threshold,
                    "output_file": output_file,
                    "time": processing_time
                }
        except Exception as e:
            logging.error(f"Error visualizing combined context blocks: {e}")
            return {
                "id": id_value,
                "success": False,
                "error": f"Error visualizing context: {str(e)}",
                "time": time.time() - start_time
            }
    else:
        logging.warning(f"No context blocks found for ID {id_value}")  # Keep warning but simplified
        return {
            "id": id_value,
            "success": False,
            "error": "No context blocks found",
            "images_processed": images_processed,
            "images_below_threshold": images_below_threshold,
            "time": time.time() - start_time
        }

def run_process_descriptions(
    json_dir: str = DEFAULT_JSON_DIR,
    images_dir: str = DEFAULT_IMAGES_DIR, 
    texts_dir: str = DEFAULT_TEXTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_lines_context: int = DEFAULT_MAX_LINES_CONTEXT,
    max_image_suffix: int = DEFAULT_MAX_IMAGE_SUFFIX,
    max_ids: int = 0,
    model_name: str = DEFAULT_MODEL_NAME,
    process_all: bool = False,
    specific_id: str = None,
    best_only: bool = False,
    verbose: bool = False,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Main function to process descriptions using CLIP model.
    Can be called programmatically from other modules.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP model - ONLY ONCE!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    try:
        logging.info(f"Loading CLIP model: {model_name}")
        model_load_start = time.time()
        model, preprocess = clip.load(model_name, device=device)
        model_load_time = time.time() - model_load_start
        logging.info(f"Model loaded in {model_load_time:.2f} seconds")
    except Exception as e:
        error_msg = f"Error loading CLIP model: {e}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    logging.info(f"Starting processing with {model_name} model")
    
    # Determine which IDs to process
    ids_to_process = []
    
    if specific_id:
        # Process a single specific ID
        ids_to_process = [specific_id]
        logging.debug(f"Processing single ID: {specific_id}")  # Changed to debug
    elif not process_all:  # Now we check if NOT process_all (Obrazek is default)
        # Get IDs from JSON files with "Obrázek" label
        logging.info("Scanning for JSONs with 'Obrázek' label...")
        
        # Set up progress tracking for JSON scanning
        json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
        
        if show_progress:
            try:
                # Try notebook tqdm first, fall back to regular tqdm
                try:
                    from tqdm.notebook import tqdm as notebook_tqdm
                    json_iterator = notebook_tqdm(json_files, desc="Checking JSONs for Obrázek label")
                    logging.debug("Using tqdm.notebook for JSON scanning progress")
                except ImportError:
                    # Fall back to regular tqdm
                    json_iterator = tqdm(json_files, desc="Checking JSONs for Obrázek label")
                    logging.debug("Using regular tqdm for JSON scanning progress")
            except ImportError:
                # If all fails, use regular list
                json_iterator = json_files
                logging.warning("tqdm not available, progress bars disabled")
        else:
            json_iterator = json_files
            
        for filename in json_iterator:
            json_filepath = os.path.join(json_dir, filename)
            
            if has_obrazek_label(json_filepath):
                id_value = extract_id_from_json(json_filepath)
                if id_value:
                    ids_to_process.append(id_value)
                    logging.debug(f"Found 'Obrázek' label in {filename}, ID: {id_value}")
    else:
        # Get all IDs from JSON files
        logging.info("Getting all JSON IDs...")
        json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
        
        if show_progress:
            try:
                try:
                    from tqdm.notebook import tqdm as notebook_tqdm
                    files_iterator = notebook_tqdm(json_files, desc="Extracting IDs")
                except ImportError:
                    files_iterator = tqdm(json_files, desc="Extracting IDs")
            except ImportError:
                files_iterator = json_files
        else:
            files_iterator = json_files
            
        for filename in files_iterator:
            id_value = extract_id_from_json(os.path.join(json_dir, filename))
            if id_value and id_value not in ids_to_process:
                ids_to_process.append(id_value)
    
    if not ids_to_process:
        error_msg = "No valid IDs found to process."
        logging.error(error_msg)
        return {"error": error_msg}
    
    if max_ids > 0 and max_ids < len(ids_to_process):
        logging.info(f"Limiting to {max_ids} IDs out of {len(ids_to_process)} total")
        ids_to_process = ids_to_process[:max_ids]
    
    logging.info(f"Found {len(ids_to_process)} IDs to process")
    
    # Reduced logging here - removed separator lines
    logging.info(f"Config: threshold={similarity_threshold}, model={model_name}")

    # Process each ID
    results = []
    successful_contexts = 0
    total_images_processed = 0
    total_images_below_threshold = 0
    
    total_start_time = time.time()
    
    # Set up progress tracking
    if show_progress:
        try:
            # Try notebook tqdm first, fall back to regular tqdm
            try:
                from tqdm.notebook import tqdm as notebook_tqdm
                id_iterator = notebook_tqdm(
                    ids_to_process, 
                    desc="Processing IDs", 
                    unit="ID",
                    leave=True
                )
                logging.debug("Using tqdm.notebook for main processing progress")
            except ImportError:
                # Fall back to regular tqdm
                id_iterator = tqdm(
                    ids_to_process, 
                    desc="Processing IDs", 
                    unit="ID",
                    leave=True
                )
                logging.debug("Using regular tqdm for main processing progress")
        except ImportError:
            # If all fails, use regular list
            id_iterator = ids_to_process
            logging.warning("tqdm not available, progress bars disabled")
    else:
        id_iterator = ids_to_process
        
    for idx, id_value in enumerate(id_iterator):
        # Removed duplicate logging - let the progress bar show this info
        # Only keep debug level logging for individual ID processing
        logging.debug(f"Processing ID {idx+1}/{len(ids_to_process)}: {id_value}")
        
        result = process_id(
            id_value=id_value,
            model=model,
            preprocess=preprocess,
            device=device,
            images_dir=images_dir,
            texts_dir=texts_dir,
            similarity_threshold=similarity_threshold,
            max_lines_context=max_lines_context,
            max_image_suffix=max_image_suffix,
            output_dir=output_dir,
            best_only=best_only,
            verbose=verbose
        )
        
        results.append(result)
        
        if result.get("success", False):
            successful_contexts += 1
        
        total_images_processed += result.get("images_processed", 0)
        total_images_below_threshold += result.get("images_below_threshold", 0)
        
        # Update progress bar with additional info if using tqdm
        if show_progress and ('tqdm' in str(type(id_iterator)) or 'notebook.tqdm' in str(type(id_iterator))):
            success_rate = (successful_contexts/(idx+1))*100
            id_iterator.set_postfix({
                'success_rate': f"{success_rate:.1f}%",
                'successful': successful_contexts,
                'images': total_images_processed
            })
        
        # Remove verbose output - keep it only if debug is enabled
        if verbose and logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"Detailed info for ID {id_value}")
        
        # Removed periodic logging here as it's redundant with progress bar
    
    # Make sure to close the progress bar if it's a tqdm instance
    if show_progress and ('tqdm' in str(type(id_iterator)) or 'notebook.tqdm' in str(type(id_iterator))):
        id_iterator.close()
    
    total_time = time.time() - total_start_time
    
    # Summarize results - but make it more concise
    logging.info("PROCESSING COMPLETE")
    logging.info(f"Results: {successful_contexts}/{len(ids_to_process)} successful ({(successful_contexts / len(ids_to_process)) * 100:.1f}%)")
    logging.info(f"Time: {total_time:.2f}s total, {total_time / len(ids_to_process):.2f}s per ID")
    
    # Create results dictionary
    return_results = {
        "summary": {
            "total_ids": len(ids_to_process),
            "successful_ids": successful_contexts,
            "total_images_processed": total_images_processed,
            "images_below_threshold": total_images_below_threshold,
            "success_rate": (successful_contexts / len(ids_to_process)) * 100 if ids_to_process else 0,
            "elapsed_time": total_time,
            "average_time_per_id": total_time / len(ids_to_process) if ids_to_process else 0
        },
        "config": {
            "model": model_name,
            "device": device,
            "similarity_threshold": similarity_threshold,
            "max_lines_context": max_lines_context,
            "process_all": process_all,
            "best_only": best_only
        },
        "details": results
    }
    
    # Write detailed results to a JSON file
    results_file = os.path.join(output_dir, "processing_results.json")
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(return_results, f, indent=2)
        logging.debug(f"Detailed results saved to {results_file}")  # Changed to debug
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")
    
    return return_results

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description=f"Process images with CLIP to find matching text descriptions (v{SCRIPT_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help="Directory containing JSON files")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory containing exported images")
    parser.add_argument("--texts-dir", default=DEFAULT_TEXTS_DIR, help="Directory containing filtered XML texts")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store output context images")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Similarity threshold for context building")
    parser.add_argument("--max-lines-context", type=int, default=DEFAULT_MAX_LINES_CONTEXT, help="Maximum lines to check above and below")
    parser.add_argument("--max-image-suffix", type=int, default=DEFAULT_MAX_IMAGE_SUFFIX, help="Maximum suffix for alternative images")
    parser.add_argument("--max-ids", type=int, default=0, help="Maximum number of IDs to process (0 for all)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="CLIP model to use")
    parser.add_argument("--all", action="store_true", 
                        help="Process all IDs, not just ones with 'Obrázek' label")
    parser.add_argument("--id", help="Process a single specific ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed output")
    parser.add_argument("--best-only", action="store_true", 
                        help="Only use the best matching context for each image instead of all above threshold")
    parser.add_argument("--no-log-file", action="store_true", help="Disable logging to file")
    
    args = parser.parse_args()
    
    # Set up logging for command-line use
    setup_logging(args.debug, use_notebook=False, log_to_file=not args.no_log_file)
    
    # Run the main processing function
    result = run_process_descriptions(
        json_dir=args.json_dir,
        images_dir=args.images_dir,
        texts_dir=args.texts_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        max_lines_context=args.max_lines_context,
        max_image_suffix=args.max_image_suffix,
        max_ids=args.max_ids,
        model_name=args.model_name,
        process_all=args.all,
        specific_id=args.id,
        best_only=args.best_only,
        verbose=args.verbose,
        show_progress=True
    )
    
    if "error" in result:
        logging.critical(f"Processing failed: {result['error']}")
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())