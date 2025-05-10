#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from collections import Counter # Use Counter for easy counting
import argparse

# --- Configuration ---
DEFAULT_JSONS_DIR_NAME = "jsons" # Directory containing the split JSON files
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more verbose output

# --- Logging Setup ---
def setup_logging(debug_mode=False):
    """Configures the logging format and level."""
    log_level = logging.DEBUG if debug_mode else LOG_LEVEL # Use global LOG_LEVEL default
    log_format = '%(asctime)s - %(levelname)s - %(message)s' # Simplified format for this script
    logging.basicConfig(level=log_level,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        force=True) # force=True helps if run interactively
    logging.info(f"Log level set to: {'DEBUG' if debug_mode else 'INFO'}")
    if debug_mode:
        logging.debug("DEBUG logging enabled.")
# --- End Logging Setup ---

# <<< Modified function signature and added pair counting >>>
def count_rectangle_labels(jsons_dir_path, label_pair_to_check=None):
    """
    Scans JSON files in a directory, counts rectangle labels,
    and optionally counts co-occurrence of a specific label pair.
    Returns a tuple: (label_counts, pair_count, processed_files, skipped_files, errors_parsing, tasks_with_no_rect_labels)
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

    for filename in filenames:
        file_path = os.path.join(jsons_dir_path, filename)
        logging.debug(f"Processing file: {filename}")
        task_found_rect_label = False
        # <<< Use a set to store labels found *within this file* for pair checking >>>
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
                        # else: # Debug for empty label lists if needed
                        #     logging.debug(f"Rectlabel item with empty labels list in {filename}")

            # <<< After processing all annotations in the file, check for the pair >>>
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

    # <<< Return the new pair_count along with others >>>
    return label_counts, pair_count, processed_files, skipped_files, errors_parsing, tasks_with_no_rect_labels


def print_summary_table(label_counts):
    """Prints the counted labels in a formatted table."""
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


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count occurrences of 'rectanglelabels' within annotation JSON files. Optionally count co-occurrence of a specific pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "jsons_dir",
        nargs='?',
        default=DEFAULT_JSONS_DIR_NAME,
        help="Directory containing the individual task JSON files."
    )
    # <<< New argument for the pair >>>
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

    setup_logging(args.debug)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logging.debug("__file__ not defined, using CWD as script base: %s", script_dir)

    jsons_dir_abs = os.path.abspath(os.path.join(script_dir, args.jsons_dir))
    logging.info(f"Starting label count process for directory: {jsons_dir_abs}")

    # <<< Pass the pair to the function >>>
    label_counts, pair_count_result, processed_count, skipped_count, errors_count, no_labels_count = count_rectangle_labels(
        jsons_dir_abs,
        args.pair # Pass the list of two labels, or None if not provided
    )

    # Print the summary table if counting was successful
    if label_counts is not None:
        print_summary_table(label_counts)

        # <<< Print the pair count result if a pair was requested >>>
        if args.pair:
            logging.info("\n--- Label Pair Co-occurrence ---")
            logging.info(f"Files containing BOTH '{args.pair[0]}' AND '{args.pair[1]}': {pair_count_result}")
            logging.info("------------------------------")


        # Print processing summary
        logging.info("\n--- Processing Summary ---")
        logging.info(f"Total JSON files found: {processed_count + skipped_count + errors_count}") # Approximation
        logging.info(f"Files successfully processed: {processed_count}")
        logging.info(f"Files skipped (invalid format/error/etc.): {skipped_count + errors_count}")
        logging.info(f"Tasks processed with no rectangle labels found: {no_labels_count}")
        logging.info(f"Files with parsing/reading errors: {errors_count}")
        logging.info("--------------------------")

    logging.info("Label counting script finished.")