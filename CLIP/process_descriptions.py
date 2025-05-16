#!/usr/bin/env python3
# filepath: /home/adrian/school/KNN/CLIP/process_descriptions.py

import os
import argparse
import json
import logging
import re
import time
from typing import Optional, Tuple, List, Dict
import sys
import shutil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

# Import these modules - make sure they're available in your environment
try:
    import clip
    from cut_text import OCRDocument, TextBlock, draw_blocks_on_image
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure clip and cut_text modules are installed/available")
    sys.exit(1)

# Constants
UUID_PATTERN = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
TARGET_LABEL = "Obrázek"

def setup_logging(debug_mode=False):
    """Configure logging format and level."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Clear any existing handlers (important for multiple runs)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up file logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='process_descriptions.log',
        filemode='w'
    )
    
    # Add console handler with colored output for better visibility
    console = logging.StreamHandler()
    console.setLevel(log_level)
    
    # Use a more visible format for console
    if sys.stdout.isatty():  # Check if running in terminal that supports colors
        # Colors for different log levels
        class ColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[94m',  # Blue
                'INFO': '\033[92m',   # Green
                'WARNING': '\033[93m', # Yellow
                'ERROR': '\033[91m',  # Red
                'CRITICAL': '\033[41m', # Red background
                'ENDC': '\033[0m'     # Reset color
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
    logging.getLogger('').addHandler(console)
    
    # Test that logging is working
    logging.info("Logging system initialized")

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
    device: str,
    args,  # Add this parameter
    images_dir: str = "exported_images",
    texts_dir: str = "filtered_texts",
    similarity_threshold: float = 0.25,
    max_lines_context: int = 3,
    max_image_suffix: int = 10,
    output_dir: str = "output_context"
) -> Dict:
    """Process a single ID with the pre-loaded CLIP model."""
    
    start_time = time.time()
    
    # Track statistics
    images_processed = 0
    images_below_threshold = 0
    
    # Construct paths
    base_image_path = os.path.join(images_dir, f"{id_value}.jpg")
    xml_path = os.path.join(texts_dir, f"filtered_{id_value}.xml")
    
    # Check if XML exists
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

    logging.info(f"Processing ID: {id_value}")
    logging.debug(f"XML path: {xml_path}")
    logging.debug(f"Found {len(available_images)} images")
    
    # Store all context blocks from all images
    all_context_blocks = []
    
    # Process each image
    for idx, image_path in enumerate(available_images):
        logging.info(f"Processing image {idx+1}/{len(available_images)}: {image_path}")
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
        logging.info(f"Matched block with {best_similarity:.4f} cosine similarity: \"{best_block.get_text()[:100]}...\"")

        # Build contextual block around the best matching text
        logging.info("Building context block around best match...")

        # Instead of just building context around the best match,
        # build context based on the selected approach (all above threshold or best only)
        above_threshold_blocks = []
        
        # Find blocks to include based on chosen approach
        if args.best_only:
            # Only use the best match for this image (if it's above threshold)
            best_idx = int(torch.tensor(all_similarities).argmax())
            if all_similarities[best_idx] > similarity_threshold:
                above_threshold_blocks = [best_idx]
                logging.info(f"Using best match only (similarity: {all_similarities[best_idx]:.4f})")
            else:
                logging.warning(f"Best match similarity ({all_similarities[best_idx]:.4f}) below threshold ({similarity_threshold})")
                images_below_threshold += 1
        else:
            # Use all blocks above threshold (original behavior)
            for idx, similarity in enumerate(all_similarities):
                if similarity > similarity_threshold:
                    above_threshold_blocks.append(idx)
            
            if above_threshold_blocks:
                logging.info(f"Found {len(above_threshold_blocks)} blocks above threshold {similarity_threshold}")
            else:
                logging.warning(f"No blocks above threshold {similarity_threshold} found")
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
            
            if args.best_only:
                logging.info(f"Built context around best match with {len(all_context_blocks)} total blocks")
            else:
                logging.info(f"Built context blocks around {len(above_threshold_blocks)} matches above threshold")
            
            logging.info(f"Total blocks added to context: {len(all_context_blocks)}")
            
            if args.verbose:
                for idx in above_threshold_blocks:
                    logging.info(f"  Block {idx}: {all_similarities[idx]:.4f} - {blocks[idx].get_text()[:50]}...")
    
    # After processing all images, visualize all accumulated context blocks on the original image
    if all_context_blocks:
        # Get original image path (use the ID without any suffix)
        original_image_path = os.path.join("images", f"{id_value}.jpg")
        
        # If original image doesn't exist, use the first available image we found earlier
        if not os.path.exists(original_image_path):
            original_image_path = available_images[0]
            logging.info(f"Original image not found at {original_image_path}, using {available_images[0]} instead")
        
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
                logging.info(f"All context blocks visualization saved to '{output_path}'")
                
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
        logging.warning(f"No context blocks were found across any images for ID {id_value}")
        return {
            "id": id_value,
            "success": False,
            "error": "No context blocks found",
            "images_processed": images_processed,
            "images_below_threshold": images_below_threshold,
            "time": time.time() - start_time
        }

def main():
    parser = argparse.ArgumentParser(description="Process images with CLIP to find matching text descriptions")
    parser.add_argument("--json-dir", default="jsons", help="Directory containing JSON files")
    parser.add_argument("--images-dir", default="exported_images", help="Directory containing exported images")
    parser.add_argument("--texts-dir", default="filtered_texts", help="Directory containing filtered XML texts")
    parser.add_argument("--output-dir", default="output_context", help="Directory to store output context images")
    parser.add_argument("--similarity-threshold", type=float, default=0.25, help="Similarity threshold for context building")
    parser.add_argument("--max-lines-context", type=int, default=3, help="Maximum lines to check above and below")
    parser.add_argument("--max-image-suffix", type=int, default=20, help="Maximum suffix for alternative images")
    parser.add_argument("--max-ids", type=int, default=0, help="Maximum number of IDs to process (0 for all)")
    parser.add_argument("--model-name", default="ViT-B/32", help="CLIP model to use")
    parser.add_argument("--all", action="store_true", 
                        help="Process all IDs, not just ones with 'Obrázek' label")
    parser.add_argument("--id", help="Process a single specific ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed output")
    parser.add_argument("--best-only", action="store_true", 
                        help="Only use the best matching context for each image instead of all above threshold")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CLIP model - ONLY ONCE!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    try:
        logging.info(f"Loading CLIP model: {args.model_name}")
        model_load_start = time.time()
        model, preprocess = clip.load(args.model_name, device=device)
        model_load_time = time.time() - model_load_start
        logging.info(f"Model loaded in {model_load_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error loading CLIP model: {e}")
        return 1
    
    print(f"\n{'='*60}\nStarting processing with {args.model_name} model\n{'='*60}")
    
    # Determine which IDs to process
    ids_to_process = []
    
    if args.id:
        # Process a single specific ID
        ids_to_process = [args.id]
        logging.info(f"Processing single ID: {args.id}")
    elif not args.all:  # Now we check if NOT args.all (Obrazek is default)
        # Get IDs from JSON files with "Obrázek" label
        logging.info("Scanning for JSONs containing 'Obrázek' label...")
        ids_to_process = get_obrazek_json_ids(args.json_dir)
    else:
        # Get all IDs from JSON files
        logging.info("Getting all JSON IDs...")
        json_files = [f for f in os.listdir(args.json_dir) if f.lower().endswith('.json')]
        ids_to_process = []
        for filename in json_files:
            id_value = extract_id_from_json(os.path.join(args.json_dir, filename))
            if id_value and id_value not in ids_to_process:
                ids_to_process.append(id_value)
    
    if not ids_to_process:
        logging.error("No valid IDs found to process. Exiting.")
        return 1
    
    if args.max_ids > 0 and args.max_ids < len(ids_to_process):
        logging.info(f"Limiting to {args.max_ids} IDs out of {len(ids_to_process)} total")
        ids_to_process = ids_to_process[:args.max_ids]
    
    logging.info(f"Found {len(ids_to_process)} IDs to process")
    
    logging.info("=" * 60)
    logging.info("STARTING PROCESSING")
    logging.info("=" * 60)
    logging.info(f"Configuration: threshold={args.similarity_threshold}, model={args.model_name}")
    logging.info(f"Directories: json={args.json_dir}, images={args.images_dir}, texts={args.texts_dir}")

    # Process each ID
    results = []
    successful_contexts = 0
    total_images_processed = 0
    total_images_below_threshold = 0
    
    total_start_time = time.time()
    
    for idx, id_value in enumerate(tqdm(ids_to_process, desc="Processing IDs")):
        logging.info(f"Processing ID {idx+1}/{len(ids_to_process)}: {id_value}")
        
        result = process_id(
            id_value=id_value,
            model=model,
            preprocess=preprocess,
            device=device,
            args=args,  # Add this argument
            images_dir=args.images_dir,
            texts_dir=args.texts_dir,
            similarity_threshold=args.similarity_threshold,
            max_lines_context=args.max_lines_context,
            max_image_suffix=args.max_image_suffix,
            output_dir=args.output_dir
        )
        
        results.append(result)
        
        if result["success"]:
            successful_contexts += 1
        
        total_images_processed += result.get("images_processed", 0)
        total_images_below_threshold += result.get("images_below_threshold", 0)
        
        if args.verbose:
            logging.info(f"Detailed info for ID {id_value}: Found {len(available_images)} images")
            for idx, img in enumerate(available_images):
                logging.info(f"  Image {idx+1}: {os.path.basename(img)}")
        
        print(f"Processed {idx+1}/{len(ids_to_process)} IDs, success rate: {(successful_contexts/(idx+1))*100:.1f}%")
    
    total_time = time.time() - total_start_time
    
    # Summarize results
    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total IDs processed: {len(ids_to_process)}")
    logging.info(f"Successful context images created: {successful_contexts}")
    logging.info(f"Total images processed: {total_images_processed}")
    logging.info(f"Images below threshold: {total_images_below_threshold}")
    logging.info(f"Success rate: {(successful_contexts / len(ids_to_process)) * 100:.2f}%")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Average time per ID: {total_time / len(ids_to_process):.2f} seconds")
    
    # Write detailed results to a JSON file
    results_file = "all_results.json" if args.all else "obrazek_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
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
                "model": args.model_name,
                "device": device,
                "similarity_threshold": args.similarity_threshold,
                "max_lines_context": args.max_lines_context
            },
            "details": results
        }, f, indent=2)
    
    logging.info(f"Detailed results saved to {results_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())