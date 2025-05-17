#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from collections import Counter # Use Counter for easy counting
import argparse
from typing import Dict, Tuple, Set, List, Optional, Union, Any

# --- Configuration ---
DEFAULT_JSONS_DIR_NAME = "jsons" # Directory containing the split JSON files
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more verbose output
SCRIPT_VERSION = "1.1"

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

def count_rectangle_labels(
    jsons_dir_path: str, 
    label_pair_to_check: Optional[List[str]] = None,
    show_progress: bool = True
) -> Tuple[Counter, int, int, int, int, int]:
    """
    Scans JSON files in a directory, counts rectangle labels,
    and optionally counts co-occurrence of a specific label pair.
    
    Args:
        jsons_dir_path: Path to directory containing JSON files
        label_pair_to_check: Optional pair of labels to check co-occurrence
        show_progress: Whether to show progress bars
        
    Returns:
        Tuple containing:
        - Counter with label counts
        - Pair count (if a pair was specified)
        - Number of processed files
        - Number of skipped files
        - Number of files with parsing errors
        - Number of tasks with no rectangle labels
    """
    label_counts = Counter()
    pair_count = 0 # Counter for the specific pair co-occurrence
    processed_files = 0
    skipped_files = 0
    errors_parsing = 0
    tasks_with_no_rect_labels = 0

    # Normalize the pair to check (lowercase) if provided
    target_label1 = label_pair_to_check[0].lower() if label_pair_to_check else None
    target_label2 = label_pair_to_check[1].lower() if label_pair_to_check else None
    if label_pair_to_check:
         logging.info(f"Additionally checking for co-occurrence of labels: '{label_pair_to_check[0]}' AND '{label_pair_to_check[1]}'")

    if not os.path.isdir(jsons_dir_path):
        logging.error(f"Input directory not found: {jsons_dir_path}")
        return None, 0, processed_files, skipped_files, errors_parsing, tasks_with_no_rect_labels

    logging.info(f"Scanning for JSON files in: {jsons_dir_path}")
    filenames = [f for f in os.listdir(jsons_dir_path) if f.lower().endswith('.json') and os.path.isfile(os.path.join(jsons_dir_path, f))]

    if not filenames:
        logging.warning(f"No JSON files found in directory: {jsons_dir_path}")
        return label_counts, pair_count, processed_files, skipped_files, errors_parsing, tasks_with_no_rect_labels

    logging.info(f"Found {len(filenames)} JSON files to process.")
    
    # Set up progress tracking
    if show_progress:
        try:
            # Try notebook tqdm first (for Jupyter environments)
            try:
                from tqdm.notebook import tqdm
                file_iterator = tqdm(filenames, desc="Analyzing JSONs", unit="file")
                logging.debug("Using tqdm.notebook for progress display")
            except ImportError:
                # Fall back to standard tqdm (for terminal environments)
                from tqdm import tqdm
                file_iterator = tqdm(filenames, desc="Analyzing JSONs", unit="file")
                logging.debug("Using standard tqdm for progress display")
        except ImportError:
            # If tqdm isn't available at all
            logging.warning("tqdm not available, progress bars disabled")
            file_iterator = filenames
    else:
        file_iterator = filenames

    for i, filename in enumerate(file_iterator):
        file_path = os.path.join(jsons_dir_path, filename)
        logging.debug(f"Processing file: {filename}")
        task_found_rect_label = False
        # Use a set to store labels found *within this file* for pair checking
        labels_in_this_file = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                task_data = json.load(f)

            processed_files += 1

            if not isinstance(task_data, dict):
                logging.warning(f"Skipping file {filename}: Content is not a JSON object.")
                skipped_files += 1
                continue

            annotations = task_data.get("annotations", [])
            if not isinstance(annotations, list):
                logging.warning(f"Skipping file {filename}: 'annotations' is not a list.")
                skipped_files += 1
                continue

            # Iterate through annotations
            for annotation in annotations:
                if not isinstance(annotation, dict): continue

                results = annotation.get("result", [])
                if not isinstance(results, list): continue

                # Iterate through result items
                for item in results:
                    if not isinstance(item, dict): continue

                    if item.get("type") == "rectanglelabels":
                        value_dict = item.get("value", {})
                        if not isinstance(value_dict, dict): continue

                        labels_list = value_dict.get("rectanglelabels")
                        if isinstance(labels_list, list) and labels_list:
                            task_found_rect_label = True
                            for label_name in labels_list:
                                if isinstance(label_name, str) and label_name.strip():
                                    clean_label_name = label_name.strip()
                                    # Add to overall counts
                                    label_counts[clean_label_name] += 1
                                    # Add to set for this file (use lowercase for case-insensitive check)
                                    labels_in_this_file.add(clean_label_name.lower())
                                    logging.debug(f"  Counted label '{clean_label_name}' from {filename}")
                                else:
                                    logging.warning(f"Invalid label found in {filename}: {label_name}")

            # After processing all annotations in the file, check for the pair
            if label_pair_to_check:
                # Check if both target labels (lowercase) are in the set for this file
                if target_label1 in labels_in_this_file and target_label2 in labels_in_this_file:
                    pair_count += 1
                    logging.debug(f"  Found co-occurring pair '{label_pair_to_check[0]}' and '{label_pair_to_check[1]}' in {filename}")

            if not task_found_rect_label:
                 logging.debug(f"File {filename}: Processed but found no rectangle label results.")
                 tasks_with_no_rect_labels += 1

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON in file {filename}: {e}")
            errors_parsing += 1
            skipped_files += 1
        except IOError as e:
            logging.error(f"Error reading file {filename}: {e}")
            errors_parsing += 1
            skipped_files += 1
        except Exception as e:
             logging.error(f"Unexpected error processing file {filename}: {e}", exc_info=logging.DEBUG)
             errors_parsing += 1
             skipped_files += 1
             
        # Update progress bar with additional info if using tqdm
        if show_progress and hasattr(file_iterator, 'set_postfix'):
            file_iterator.set_postfix({
                'labels': len(label_counts), 
                'processed': processed_files,
                'pairs': pair_count if label_pair_to_check else 0
            })
            
    # Close progress bar if it exists
    if show_progress and hasattr(file_iterator, 'close'):
        file_iterator.close()

    # Return the new pair_count along with others
    return label_counts, pair_count, processed_files, skipped_files, errors_parsing, tasks_with_no_rect_labels


def print_summary_table(label_counts: Counter) -> None:
    """
    Prints the counted labels in a formatted table.
    
    Args:
        label_counts: Counter object with labels and their counts
    """
    if not label_counts:
        logging.info("No rectangle labels were found to summarize.")
        return

    logging.info("\n--- Individual Rectangle Label Counts ---")
    sorted_items = sorted(label_counts.items())
    try:
        max_label_len = max(len(label) for label, count in sorted_items) + 2
    except ValueError: max_label_len = len("Label Name") + 2
    count_header = "Count"
    max_count_len = len(count_header)
    for label, count in sorted_items: max_count_len = max(max_count_len, len(str(count)))
    max_count_len += 2

    header = f"{'Label Name'.ljust(max_label_len)}{count_header.rjust(max_count_len)}"
    separator = "-" * len(header)
    print(header)
    print(separator)
    total_labels = 0
    for label, count in sorted_items:
        print(f"{label.ljust(max_label_len)}{str(count).rjust(max_count_len)}")
        total_labels += count
    print(separator)
    print(f"{'Total Instances'.ljust(max_label_len)}{str(total_labels).rjust(max_count_len)}")
    print("-------------------------------------")


def run_count_labels(
    jsons_dir: str = DEFAULT_JSONS_DIR_NAME, 
    pair_to_check: Optional[List[str]] = None, 
    print_table: bool = True
) -> Dict[str, Any]:
    """
    Main function to run the label counting process.
    Can be called programmatically from other modules.
    
    Args:
        jsons_dir: Directory containing JSON files
        pair_to_check: Optional pair of labels to check co-occurrence
        print_table: Whether to print the table output (disable for programmatic use)
        
    Returns:
        Dictionary with statistics and results
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logging.debug("__file__ not defined, using CWD as script base: %s", script_dir)

    jsons_dir_abs = os.path.abspath(os.path.join(script_dir, jsons_dir))
    logging.info(f"Starting label count process for directory: {jsons_dir_abs}")

    # Pass the pair to the function
    label_counts, pair_count_result, processed_count, skipped_count, errors_count, no_labels_count = count_rectangle_labels(
        jsons_dir_abs,
        pair_to_check
    )

    results = {
        "label_counts": dict(label_counts) if label_counts else {},
        "total_files": processed_count + skipped_count + errors_count,
        "processed_files": processed_count,
        "skipped_files": skipped_count,
        "error_files": errors_count,
        "no_labels_files": no_labels_count,
        "total_labels": sum(label_counts.values()) if label_counts else 0,
        "unique_labels": len(label_counts) if label_counts else 0
    }
    
    # Add pair count results if applicable
    if pair_to_check:
        results["pair_checked"] = pair_to_check
        results["pair_count"] = pair_count_result

    # Print the summary table if requested
    if print_table and label_counts is not None:
        print_summary_table(label_counts)

        # Print the pair count result if a pair was requested
        if pair_to_check:
            logging.info("\n--- Label Pair Co-occurrence ---")
            logging.info(f"Files containing BOTH '{pair_to_check[0]}' AND '{pair_to_check[1]}': {pair_count_result}")
            logging.info("------------------------------")

        # Print processing summary
        logging.info("\n--- Processing Summary ---")
        logging.info(f"Total JSON files found: {results['total_files']}")
        logging.info(f"Files successfully processed: {results['processed_files']}")
        logging.info(f"Files skipped (invalid format/error/etc.): {results['skipped_files']}")
        logging.info(f"Tasks processed with no rectangle labels found: {results['no_labels_files']}")
        logging.info(f"Files with parsing/reading errors: {results['error_files']}")
        logging.info("--------------------------")

    return results


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Count occurrences of 'rectanglelabels' within annotation JSON files (v{SCRIPT_VERSION}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "jsons_dir",
        nargs='?',
        default=DEFAULT_JSONS_DIR_NAME,
        help="Directory containing the individual task JSON files."
    )
    parser.add_argument(
        "--pair",
        nargs=2, # Expect exactly two arguments after --pair
        metavar=('LABEL1', 'LABEL2'), # Help text placeholder names
        default=None, # Default is None, meaning don't check pairs
        help="Specify two label names to count how many files contain BOTH labels."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed DEBUG logging output."
    )

    args = parser.parse_args()

    # Setup normal logging (not notebook mode) when run as script
    setup_logging(args.debug, use_notebook=False)
    
    # Run the main function
    run_count_labels(args.jsons_dir, args.pair, print_table=True)
    
    logging.info("Label counting script finished.")
