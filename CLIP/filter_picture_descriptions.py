#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import argparse
import time
import logging
from pero_ocr.core.layout import PageLayout
from typing import Dict, List, Tuple, Optional, Any, Union
import xml.etree.ElementTree as ET

# --- Configuration ---
SCRIPT_VERSION = "1.1"
DEFAULT_JSON_DIR = "filtered_jsons"
DEFAULT_XML_DIR = "filtered_texts"
DEFAULT_OUTPUT_DIR = "filtered_texts_no_desc"
DEFAULT_IOU_THRESHOLD = 0.00005
LOG_LEVEL = logging.INFO

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

def load_json_annotation(json_filepath: str) -> List[Dict]:
    """Load JSON annotation file and extract text regions with 'Popis u obrázku' label"""
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_regions = []
        
        # Find all annotations with type "rectanglelabels"
        for annotation in data.get("annotations", []):
            for result in annotation.get("result", []):
                if (result.get("type") == "rectanglelabels" and 
                    "Popis u obrázku" in result.get("value", {}).get("rectanglelabels", [])):
                    
                    value = result.get("value", {})
                    original_width = result.get("original_width", 0)
                    original_height = result.get("original_height", 0)
                    
                    # Convert percentage to absolute coordinates
                    x = value.get("x", 0) * original_width / 100
                    y = value.get("y", 0) * original_height / 100
                    width = value.get("width", 0) * original_width / 100
                    height = value.get("height", 0) * original_height / 100
                    
                    text_regions.append({
                        "id": result.get("id"),
                        "bbox": (x, y, x + width, y + height),
                        "original_width": original_width,
                        "original_height": original_height
                    })
        
        logging.debug(f"Loaded {len(text_regions)} 'Popis u obrázku' regions from {json_filepath}")
        return text_regions
    
    except Exception as e:
        logging.error(f"Error loading JSON file {json_filepath}: {e}")
        return []

def load_xml_text_regions(xml_filepath: str) -> List[Dict]:
    """Load XML file and extract text regions with their coordinates"""
    try:
        page_layout = PageLayout(file=xml_filepath)
        text_regions = []
        
        # Extract text regions from the PageLayout
        for region in page_layout.regions:
            if hasattr(region, 'polygon'):
                # Create bounding box from polygon points
                xs = [pt[0] for pt in region.polygon]
                ys = [pt[1] for pt in region.polygon]
                
                # Get text from all lines in the region
                text = ""
                for line in region.lines:
                    text += line.transcription + " "
                
                text_regions.append({
                    "id": region.id,
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "polygon": region.polygon,
                    "text": text.strip()
                })
        
        logging.debug(f"Loaded {len(text_regions)} text regions from {xml_filepath}")
        return text_regions
    
    except Exception as e:
        logging.error(f"Error loading XML file {xml_filepath}: {e}")
        return []

def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union of two bounding boxes"""
    # Each box is (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def find_matching_regions(json_regions: List[Dict], xml_regions: List[Dict], 
                         iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> List[Tuple[Dict, Dict, float]]:
    """Find matches between JSON and XML regions based on IoU"""
    matches = []
    
    for json_region in json_regions:
        best_match = None
        best_iou = 0
        
        for xml_region in xml_regions:
            iou = calculate_iou(json_region["bbox"], xml_region["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = xml_region
        
        if best_iou >= iou_threshold:
            matches.append((json_region, best_match, best_iou))
    
    logging.debug(f"Found {len(matches)} matching regions with IoU threshold of {iou_threshold}")
    return matches

def remove_matched_regions_from_xml(matches: List[Tuple[Dict, Dict, float]], 
                                   xml_filepath: str, output_filepath: Optional[str] = None) -> int:
    """Remove matched text regions from XML file and save to a new file
    
    Args:
        matches: List of tuples containing (json_region, xml_region, iou)
        xml_filepath: Path to the original XML file
        output_filepath: Path to save the modified XML file. If None, will overwrite the original file.
        
    Returns:
        Number of regions removed
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        
        # Store IDs of regions to remove
        region_ids_to_remove = [match[1]["id"] for match in matches]
        removed_count = 0
        
        # Find all parent elements that might contain TextRegion elements
        for parent in root.findall('.//*'):
            regions_to_remove = []
            for child in list(parent):
                # Check if it's a TextRegion element with matching ID
                if child.tag.endswith('TextRegion'):
                    region_id = child.get('id')
                    if region_id in region_ids_to_remove:
                        regions_to_remove.append(child)
            
            # Remove the identified TextRegion elements
            for region in regions_to_remove:
                parent.remove(region)
                removed_count += 1
        
        # If no output path is provided, overwrite the original file
        if output_filepath is None:
            output_filepath = xml_filepath
        
        # Save the modified XML
        tree.write(output_filepath, encoding='utf-8', xml_declaration=True)
        logging.info(f"Removed {removed_count} matching regions from {xml_filepath}")
        
        return removed_count
        
    except Exception as e:
        logging.error(f"Error removing regions from XML file {xml_filepath}: {e}")
        return 0

def run_filter_descriptions(
    json_dir: str = DEFAULT_JSON_DIR,
    xml_dir: str = DEFAULT_XML_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Main function to filter picture descriptions from XML files.
    Can be called programmatically from other modules.
    
    Args:
        json_dir: Directory containing JSON files with annotations
        xml_dir: Directory containing XML OCR files
        output_dir: Directory to save filtered XML files
        iou_threshold: Threshold for IoU matching
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with statistics and results
    """
    start_time = time.time()
    
    # Calculate absolute paths
    json_dir_abs = os.path.abspath(json_dir)
    xml_dir_abs = os.path.abspath(xml_dir)
    output_dir_abs = os.path.abspath(output_dir)
    
    logging.info(f"Starting picture description filtering process")
    logging.debug(f"JSON directory: {json_dir_abs}")
    logging.debug(f"XML directory: {xml_dir_abs}")
    logging.debug(f"Output directory: {output_dir_abs}")
    logging.debug(f"IoU threshold: {iou_threshold}")
    
    # Validate input directories
    if not os.path.isdir(json_dir_abs):
        error_msg = f"JSON directory does not exist: {json_dir_abs}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    if not os.path.isdir(xml_dir_abs):
        error_msg = f"XML directory does not exist: {xml_dir_abs}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    # Create output directory
    try:
        os.makedirs(output_dir_abs, exist_ok=True)
        logging.debug(f"Ensured output directory exists: {output_dir_abs}")
    except OSError as e:
        error_msg = f"Failed to create output directory: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    # Get file lists
    try:
        json_files = [f for f in os.listdir(json_dir_abs) if f.endswith('.json')]
        xml_files = [f for f in os.listdir(xml_dir_abs) if f.endswith('.xml')]
        
        logging.info(f"Found {len(json_files)} JSON files and {len(xml_files)} XML files")
    except Exception as e:
        error_msg = f"Error listing files in directories: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    # Track statistics
    total_matches = 0
    total_json_regions = 0
    processed_files = 0
    matches_removed_count = 0
    files_with_matches = 0
    copied_without_filtering = 0
    
    # Set up progress tracking with tqdm
    pbar = None
    
    if show_progress:
        try:
            # Try to use tqdm.notebook first (for Jupyter environments)
            try:
                from tqdm.notebook import tqdm
                pbar = tqdm(total=len(json_files), desc="Filtering descriptions", unit="file")
                logging.debug("Using tqdm.notebook for progress display")
            except ImportError:
                # Fall back to standard tqdm (for terminal environments)
                from tqdm import tqdm
                pbar = tqdm(total=len(json_files), desc="Filtering descriptions", unit="file")
                logging.debug("Using standard tqdm for progress display")
        except ImportError:
            # If tqdm is completely unavailable
            logging.warning("tqdm not available, falling back to log-based progress")
            # Will use regular logging for progress
    
    # Process only JSON files with matching XML files
    for i, json_file in enumerate(json_files):
        # Update progress bar if using tqdm
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                'matches': total_matches, 
                'files_with_matches': files_with_matches,
                'removed': matches_removed_count
            })
        # Fall back to periodic log messages if not using tqdm
        elif show_progress and i % 10 == 0 and len(json_files) > 0:
            progress = (i / len(json_files)) * 100
            logging.info(f"Progress: {progress:.1f}% ({i}/{len(json_files)} files)")
        
        base_name = os.path.splitext(json_file)[0]
        matching_xml = next((x for x in xml_files if os.path.splitext(x)[0] == base_name), None)
        
        if not matching_xml:
            logging.debug(f"No matching XML file found for {json_file}")
            continue
            
        json_path = os.path.join(json_dir_abs, json_file)
        xml_path = os.path.join(xml_dir_abs, matching_xml)
        
        logging.debug(f"Processing file pair: {json_file} and {matching_xml}")
        
        # Load data
        json_regions = load_json_annotation(json_path)
        
        # Even if no "Popis u obrázku" regions found, still copy the XML
        if not json_regions:
            logging.debug(f"No 'Popis u obrázku' regions found in {json_file}")
            logging.debug(f"Copying XML file without filtering")
            try:
                tree = ET.parse(xml_path)
                output_path = os.path.join(output_dir_abs, matching_xml)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
                logging.debug(f"Copied {matching_xml} to {output_path}")
                processed_files += 1
                copied_without_filtering += 1
            except Exception as e:
                logging.error(f"Error copying XML file {matching_xml}: {e}")
            continue
            
        # Load XML regions
        xml_regions = load_xml_text_regions(xml_path)
        
        logging.debug(f"Loaded {len(json_regions)} text regions from JSON file")
        logging.debug(f"Loaded {len(xml_regions)} text regions from XML file")
        
        # Find matches
        matches = find_matching_regions(json_regions, xml_regions, iou_threshold)
        
        # Process results
        if matches:
            logging.debug(f"Found {len(matches)} matching text regions")
            if json_regions:
                match_percentage = len(matches) / len(json_regions) * 100
                logging.debug(f"Match percentage: {match_percentage:.2f}% ({len(matches)}/{len(json_regions)} regions matched)")
                
            # Log sample of matches (limit to first 3 for clarity)
            for i, (json_region, xml_region, iou) in enumerate(matches[:3]):
                logging.debug(f"Match {i+1}: JSON ID: {json_region['id']}, XML ID: {xml_region['id']}, IoU: {iou:.4f}")
                if 'text' in xml_region:
                    short_text = xml_region['text'][:50] + ('...' if len(xml_region['text']) > 50 else '')
                    logging.debug(f"Text sample: {short_text}")
            
            total_matches += len(matches)
            total_json_regions += len(json_regions)
            files_with_matches += 1
            
            # Save filtered XML
            output_path = os.path.join(output_dir_abs, matching_xml)
            removed_count = remove_matched_regions_from_xml(matches, xml_path, output_filepath=output_path)
            logging.debug(f"Filtered XML saved to {output_path}")
            matches_removed_count += removed_count
        else:
            logging.debug("No matching text regions found")
            logging.debug(f"Copying XML file without filtering")
            try:
                tree = ET.parse(xml_path)
                output_path = os.path.join(output_dir_abs, matching_xml)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
                logging.debug(f"Copied {matching_xml} to {output_path}")
                copied_without_filtering += 1
            except Exception as e:
                logging.error(f"Error copying XML file {matching_xml}: {e}")
        
        processed_files += 1
    
    # Close progress bar if it exists
    if pbar is not None:
        pbar.close()
    
    elapsed_time = time.time() - start_time
    
    # Prepare result dictionary
    results = {
        "processed_files": processed_files,
        "total_json_regions": total_json_regions,
        "total_matches": total_matches,
        "files_with_matches": files_with_matches,
        "copied_without_filtering": copied_without_filtering,
        "regions_removed": matches_removed_count,
        "iou_threshold": iou_threshold,
        "json_dir": json_dir_abs,
        "xml_dir": xml_dir_abs,
        "output_dir": output_dir_abs,
        "elapsed_time": elapsed_time
    }
    
    # Add match percentage if there were regions to match
    if total_json_regions > 0:
        results["match_percentage"] = (total_matches / total_json_regions) * 100
    
    # Log summary
    logging.info(f"\nProcessing completed in: {elapsed_time:.2f} seconds")
    logging.info(f"Summary: Processed {processed_files} file pairs")
    logging.info(f"Files with matches: {files_with_matches}")
    logging.info(f"Files copied without filtering: {copied_without_filtering}")
    if total_json_regions > 0:
        logging.info(f"Total match percentage: {results['match_percentage']:.2f}% ({total_matches}/{total_json_regions} regions matched)")
    logging.info(f"Total regions removed from XML: {matches_removed_count}")
    
    return results


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Filter picture descriptions from XML OCR data (v{SCRIPT_VERSION}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help="Directory containing JSON annotation files"
    )
    parser.add_argument(
        "--xml-dir",
        default=DEFAULT_XML_DIR,
        help="Directory containing XML OCR files"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save filtered XML files"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for matching"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed DEBUG logging output"
    )

    args = parser.parse_args()
    
    # Set up logging for command-line use
    setup_logging(args.debug, use_notebook=False)
    
    # Run the filtering process
    result = run_filter_descriptions(
        json_dir=args.json_dir,
        xml_dir=args.xml_dir,
        output_dir=args.output_dir,
        iou_threshold=args.threshold,
        show_progress=True
    )
    
    # Print final summary
    if "error" in result:
        logging.critical(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print("\n--- Filtering Summary ---")
        print(f"Processing completed in: {result['elapsed_time']:.2f} seconds")
        print(f"Processed {result['processed_files']} file pairs")
        print(f"Found {result['total_matches']} matching regions in {result['files_with_matches']} files")
        
        if result.get('match_percentage') is not None:
            print(f"Match percentage: {result['match_percentage']:.2f}% ({result['total_matches']}/{result['total_json_regions']} regions matched)")
            
        print(f"Removed {result['regions_removed']} regions from XML files")
        print(f"Copied {result['copied_without_filtering']} files without filtering")
        print("------------------------")