#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# --- Configuration ---
SCRIPT_VERSION = "1.1"
DEFAULT_JSONS_DIR = "splitted_jsons"
DEFAULT_FILTERED_JSONS_DIR = "filtered_jsons"
DEFAULT_IMAGES_DIR = "downloaded_images" 
DEFAULT_FILTERED_IMAGES_DIR = "filtered_images"
DEFAULT_TEXTS_DIR = "downloaded_texts"
DEFAULT_FILTERED_TEXTS_DIR = "filtered_texts"
DEFAULT_LABEL_TO_FILTER = "Obrázek" # Note: Czech spelling with diacritics

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
    
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure log format based on environment
    if use_notebook:
        # Simpler format for notebooks with no timestamps (cleaner output)
        log_format = '%(levelname)s: %(message)s'
    else:
        # More detailed format for scripts
        log_format = '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    
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

# --- Helper Functions ---
def extract_uuid_from_filename(filename: str) -> Optional[str]:
    """Extract UUID from filename by removing extension."""
    if not filename:
        return None
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return base_name

def has_label_in_json(json_file_path: str, target_label: str, case_sensitive: bool = False) -> bool:
    """
    Check if the JSON file contains the specified label in any rectanglelabels.
    
    Args:
        json_file_path: Path to the JSON file
        target_label: Label to search for
        case_sensitive: Whether to do case-sensitive matching
    
    Returns:
        True if the label is found, False otherwise
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            logging.warning(f"Skipping file {json_file_path}: Content is not a JSON object")
            return False
            
        # Prepare the target label for comparison
        if not case_sensitive:
            search_label = target_label.lower()
        else:
            search_label = target_label
            
        # Look through annotations
        annotations = data.get("annotations", [])
        if not isinstance(annotations, list):
            return False
            
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
                
            results = annotation.get("result", [])
            if not isinstance(results, list):
                continue
                
            # Check each result item
            for item in results:
                if not isinstance(item, dict):
                    continue
                    
                if item.get("type") == "rectanglelabels":
                    value_dict = item.get("value", {})
                    if not isinstance(value_dict, dict):
                        continue
                        
                    labels_list = value_dict.get("rectanglelabels", [])
                    if not isinstance(labels_list, list):
                        continue
                        
                    # Check each label
                    for label in labels_list:
                        if not isinstance(label, str):
                            continue
                            
                        if case_sensitive:
                            if label == search_label:
                                return True
                        else:
                            if label.lower() == search_label:
                                return True
        
        return False
        
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON in file {json_file_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error processing file {json_file_path}: {e}")
        return False

def filter_jsons_by_label(
    jsons_dir: str, 
    output_dir: str, 
    target_label: str,
    copy_instead_of_move: bool = False,
    case_sensitive: bool = False,
    show_progress: bool = True
) -> Tuple[List[str], int, int]:
    """
    Filter JSON files by checking if they contain the target label.
    
    Args:
        jsons_dir: Directory containing JSON files
        output_dir: Directory to move/copy matching files
        target_label: Label to filter by
        copy_instead_of_move: If True, copy files instead of moving them
        case_sensitive: Whether to do case-sensitive label matching
        show_progress: Whether to display a progress bar
    
    Returns:
        Tuple containing:
        - List of UUIDs of matching files
        - Count of matching files
        - Count of non-matching files
    """
    if not os.path.isdir(jsons_dir):
        logging.error(f"Input directory not found: {jsons_dir}")
        return [], 0, 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Get all JSON files
    json_files = [f for f in os.listdir(jsons_dir) if f.lower().endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        logging.warning(f"No JSON files found in {jsons_dir}")
        return [], 0, 0
    
    logging.info(f"Scanning {total_files} JSON files for label: '{target_label}'")
    
    # Track results
    matching_uuids = []
    match_count = 0
    non_match_count = 0
    
    # Set up progress tracking
    if show_progress:
        try:
            # Try notebook tqdm first (for Jupyter environments)
            try:
                from tqdm.notebook import tqdm
                pbar = tqdm(json_files, desc=f"Filtering JSONs for '{target_label}'", unit="file")
                logging.debug("Using tqdm.notebook for progress display")
            except ImportError:
                # Fall back to standard tqdm (for terminal environments)
                from tqdm import tqdm
                pbar = tqdm(json_files, desc=f"Filtering JSONs for '{target_label}'", unit="file")
                logging.debug("Using standard tqdm for progress display")
            
            # Use the progress bar iterator
            files_iterator = pbar
        except ImportError:
            # If tqdm isn't available at all
            logging.warning("tqdm not available, falling back to log-based progress")
            files_iterator = json_files
    else:
        # No progress display requested
        files_iterator = json_files
    
    # Process each file
    for i, filename in enumerate(files_iterator):
        json_path = os.path.join(jsons_dir, filename)
        
        # Check if file has the target label
        if has_label_in_json(json_path, target_label, case_sensitive):
            match_count += 1
            uuid = extract_uuid_from_filename(filename)
            if uuid:
                matching_uuids.append(uuid)
            
            # Move or copy the file
            output_path = os.path.join(output_dir, filename)
            try:
                if copy_instead_of_move:
                    shutil.copy2(json_path, output_path)
                    logging.debug(f"Copied {filename} to {output_dir}")
                else:
                    shutil.move(json_path, output_path)
                    logging.debug(f"Moved {filename} to {output_dir}")
            except Exception as e:
                logging.error(f"Error {'copying' if copy_instead_of_move else 'moving'} {filename}: {e}")
        else:
            non_match_count += 1
            logging.debug(f"File {filename} does not contain label '{target_label}'")
        
        # Update progress bar with additional info if using tqdm
        if show_progress and 'pbar' in locals():
            pbar.set_postfix({
                'matches': match_count,
                'non-matches': non_match_count
            })
        # Log progress periodically if not using tqdm
        elif show_progress and i % 100 == 0 and not 'pbar' in locals():
            progress_pct = (i + 1) / total_files * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({i + 1}/{total_files}) - Found {match_count} matches")
    
    # Close progress bar if it exists
    if show_progress and 'pbar' in locals():
        pbar.close()
    
    logging.info(f"Filtering complete: {match_count} files with '{target_label}', {non_match_count} files without")
    return matching_uuids, match_count, non_match_count

def sync_files_with_uuids(
    source_dir: str, 
    uuids: List[str],
    output_dir: str = None,
    copy_instead_of_move: bool = False,
    file_extensions: List[str] = None,
    show_progress: bool = True
) -> Tuple[int, int]:
    """
    Sync files to match the filtered JSON files based on UUIDs.
    
    Args:
        source_dir: Directory containing files to sync
        uuids: List of UUIDs to keep
        output_dir: Directory to move/copy matching files (if None, remove non-matches)
        copy_instead_of_move: If True, copy files instead of moving them
        file_extensions: List of file extensions to match (e.g., ['.jpg', '.png'])
        show_progress: Whether to display a progress bar
    
    Returns:
        Tuple containing:
        - Count of matched files
        - Count of removed/skipped files
    """
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory not found: {source_dir}")
        return 0, 0
    
    # Create uuid set for faster lookups
    uuid_set = set(uuids)
    logging.info(f"Syncing files with {len(uuid_set)} UUIDs")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")
    
    # Get all matching files
    if file_extensions:
        files = [f for f in os.listdir(source_dir) 
                if any(f.lower().endswith(ext.lower()) for ext in file_extensions)]
        file_type = "/".join(ext.strip('.') for ext in file_extensions)
    else:
        files = os.listdir(source_dir)
        file_type = "files"
    
    total_files = len(files)
    if total_files == 0:
        logging.warning(f"No matching files found in {source_dir}")
        return 0, 0
    
    logging.info(f"Processing {total_files} {file_type} files")
    
    # Track results
    match_count = 0
    non_match_count = 0
    
    # Set up progress tracking
    if show_progress:
        try:
            # Try notebook tqdm first (for Jupyter environments)
            try:
                from tqdm.notebook import tqdm
                pbar = tqdm(files, desc=f"Syncing {file_type}", unit="file")
                logging.debug("Using tqdm.notebook for progress display")
            except ImportError:
                # Fall back to standard tqdm (for terminal environments)
                from tqdm import tqdm
                pbar = tqdm(files, desc=f"Syncing {file_type}", unit="file")
                logging.debug("Using standard tqdm for progress display")
            
            # Use the progress bar iterator
            files_iterator = pbar
        except ImportError:
            # If tqdm isn't available at all
            logging.warning("tqdm not available, falling back to log-based progress")
            files_iterator = files
    else:
        # No progress display requested
        files_iterator = files
    
    # Process each file
    for i, filename in enumerate(files_iterator):
        file_path = os.path.join(source_dir, filename)
        uuid = extract_uuid_from_filename(filename)
        
        if not uuid:
            logging.warning(f"Could not extract UUID from filename: {filename}")
            non_match_count += 1
            continue
        
        # Check if the UUID matches any in our filtered list
        if uuid.lower() in (u.lower() for u in uuid_set):
            match_count += 1
            
            # If output dir specified, move/copy the matching file
            if output_dir:
                output_path = os.path.join(output_dir, filename)
                try:
                    if copy_instead_of_move:
                        shutil.copy2(file_path, output_path)
                        logging.debug(f"Copied {filename} to {output_dir}")
                    else:
                        shutil.move(file_path, output_path)
                        logging.debug(f"Moved {filename} to {output_dir}")
                except Exception as e:
                    logging.error(f"Error {'copying' if copy_instead_of_move else 'moving'} {filename}: {e}")
        else:
            non_match_count += 1
            
            # If no output dir specified, remove non-matching files
            if not output_dir:
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed {filename} (UUID not in filtered JSONs)")
                except Exception as e:
                    logging.error(f"Error removing {filename}: {e}")
        
        # Update progress bar with additional info if using tqdm
        if show_progress and 'pbar' in locals():
            pbar.set_postfix({
                'matches': match_count,
                'non-matches': non_match_count
            })
        # Log progress periodically if not using tqdm
        elif show_progress and i % 100 == 0 and not 'pbar' in locals():
            progress_pct = (i + 1) / total_files * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({i + 1}/{total_files}) - Matched {match_count} files")
    
    # Close progress bar if it exists
    if show_progress and 'pbar' in locals():
        pbar.close()
    
    action = "Moved" if output_dir and not copy_instead_of_move else "Copied" if output_dir else "Kept"
    removed_action = "Skipped" if output_dir else "Removed"
    
    logging.info(f"Sync complete: {action} {match_count} files, {removed_action} {non_match_count} files")
    return match_count, non_match_count

def run_filter_by_label(
    jsons_dir: str = DEFAULT_JSONS_DIR,
    images_dir: str = DEFAULT_IMAGES_DIR,
    texts_dir: str = DEFAULT_TEXTS_DIR,
    filtered_jsons_dir: str = DEFAULT_FILTERED_JSONS_DIR,
    filtered_images_dir: str = DEFAULT_FILTERED_IMAGES_DIR,
    filtered_texts_dir: str = DEFAULT_FILTERED_TEXTS_DIR,
    label: str = DEFAULT_LABEL_TO_FILTER,
    copy_files: bool = True,
    case_sensitive: bool = False,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Main function to filter JSONs by label and sync images and XML files.
    Can be called programmatically from other modules.
    
    Args:
        jsons_dir: Directory containing JSON files
        images_dir: Directory containing image files
        texts_dir: Directory containing XML text files
        filtered_jsons_dir: Directory to store filtered JSON files
        filtered_images_dir: Directory to store filtered image files
        filtered_texts_dir: Directory to store filtered XML files
        label: Label to filter by
        copy_files: If True, copy files instead of moving them
        case_sensitive: Whether to do case-sensitive label matching
        show_progress: Whether to display progress bars
        
    Returns:
        Dictionary with statistics and results
    """
    logging.info(f"Starting filter and sync process for label: '{label}'")
    logging.info(f"JSON source directory: {jsons_dir}")
    logging.info(f"Images source directory: {images_dir}")
    logging.info(f"Texts source directory: {texts_dir}")
    
    # 1. Filter JSON files
    matching_uuids, json_match_count, json_non_match_count = filter_jsons_by_label(
        jsons_dir=jsons_dir,
        output_dir=filtered_jsons_dir,
        target_label=label,
        copy_instead_of_move=copy_files,
        case_sensitive=case_sensitive,
        show_progress=show_progress
    )
    
    # 2. Sync images based on filtered JSONs
    img_match_count, img_non_match_count = sync_files_with_uuids(
        source_dir=images_dir,
        uuids=matching_uuids,
        output_dir=filtered_images_dir,
        copy_instead_of_move=copy_files,
        file_extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'],
        show_progress=show_progress
    )
    
    # 3. Sync XML text files based on filtered JSONs
    text_match_count, text_non_match_count = sync_files_with_uuids(
        source_dir=texts_dir,
        uuids=matching_uuids,
        output_dir=filtered_texts_dir,
        copy_instead_of_move=copy_files,
        file_extensions=['.xml', '.txt'],
        show_progress=show_progress
    )
    
    # 4. Return results
    return {
        "label_filtered": label,
        "json_matches": json_match_count,
        "json_non_matches": json_non_match_count,
        "image_matches": img_match_count,
        "image_non_matches": img_non_match_count,
        "text_matches": text_match_count,
        "text_non_matches": text_non_match_count,
        "matching_uuids": matching_uuids,
        "total_matching_pairs": min(json_match_count, img_match_count, text_match_count),
        "file_operation": "Copied" if copy_files else "Moved"
    }

# --- Main Execution ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=f"Filter JSON files by label and synchronize matching images and texts (v{SCRIPT_VERSION}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--jsons",
        default=DEFAULT_JSONS_DIR,
        help="Directory containing JSON files to filter"
    )
    
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing image files to sync"
    )
    
    parser.add_argument(
        "--texts",
        default=DEFAULT_TEXTS_DIR,
        help="Directory containing XML text files to sync"
    )
    
    parser.add_argument(
        "--filtered-jsons",
        default=DEFAULT_FILTERED_JSONS_DIR,
        help="Directory to store filtered JSON files"
    )
    
    parser.add_argument(
        "--filtered-images",
        default=DEFAULT_FILTERED_IMAGES_DIR,
        help="Directory to store filtered image files"
    )
    
    parser.add_argument(
        "--filtered-texts",
        default=DEFAULT_FILTERED_TEXTS_DIR,
        help="Directory to store filtered XML text files"
    )
    
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL_TO_FILTER,
        help="Label to filter by (case-insensitive by default)"
    )
    
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them"
    )
    
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive label matching"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed DEBUG logging output"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Run the filter process
    result = run_filter_by_label(
        jsons_dir=args.jsons,
        images_dir=args.images,
        texts_dir=args.texts,
        filtered_jsons_dir=args.filtered_jsons,
        filtered_images_dir=args.filtered_images,
        filtered_texts_dir=args.filtered_texts,
        label=args.label,
        copy_files=not args.move,
        case_sensitive=args.case_sensitive,
        show_progress=not args.no_progress
    )
    
    # Print summary
    print("\n--- Filter and Sync Summary ---")
    print(f"Label filtered: '{result['label_filtered']}'")
    print(f"JSON files with label: {result['json_matches']}")
    print(f"JSON files without label: {result['json_non_matches']}")
    print(f"Image files matched: {result['image_matches']}")
    print(f"Image files not matched: {result['image_non_matches']}")
    print(f"Text files matched: {result['text_matches']}")
    print(f"Text files not matched: {result['text_non_matches']}")
    print(f"Total matching triplets (JSON+image+text): {result['total_matching_pairs']}")
    print(f"Files were {result['file_operation'].lower()}")
    print("-----------------------------")