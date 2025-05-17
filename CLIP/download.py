#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /home/adrian/school/KNN/CLIP/download.py

import json
import os
import requests
import warnings
from dotenv import load_dotenv
import sys
import logging
import re
import time
import argparse # Import argparse for command-line arguments

# --- Global Variables & Constants ---
SCRIPT_VERSION = "2.3" # Incremented version
# --- Default Configuration (can be overridden by .env file) ---
DEFAULT_TEXTS_DIR_NAME = os.getenv("TEXTS_DIR", "texts")
DEFAULT_IMAGES_DIR_NAME = os.getenv("IMAGE_SAVE_DIR", "images")
DEFAULT_JSONS_DIR_NAME = os.getenv("JSONS_DIR", "jsons")
DEFAULT_LABELS_FILE = os.getenv("LABELS_FILE", "export.json")
DEFAULT_PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "16")
DEFAULT_LS_URL = os.getenv("LABEL_STUDIO_URL", "https://label-studio.semant.cz")
DEFAULT_PROGRESS_INTERVAL = int(os.getenv("PROGRESS_UPDATE_INTERVAL", 10))
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN") # Load token once
# --- End Configuration ---

UUID_PATTERN = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'

# Load .env file variables early
load_dotenv()
logging.debug("Attempted to load environment variables from .env file.")

# Suppress InsecureRequestWarning early
try:
    from urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
except ImportError:
    print("Warning: Could not import InsecureRequestWarning from urllib3.", file=sys.stderr)

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
        # Detailed format for scripts with timestamps and function names
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
def get_base_filenames(directory_path):
    """
    Lists files in a directory and returns a set of their base names
    (filename without extension), converted to lowercase. Handles errors.
    """
    base_names = set()
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        return None

    logging.debug(f"Scanning directory: {directory_path}")
    try:
        for filename in os.listdir(directory_path):
            full_path = os.path.join(directory_path, filename)
            if filename.startswith('.'): continue # Skip hidden
            if os.path.isfile(full_path):
                base_name, ext = os.path.splitext(filename)
                if base_name:
                     # Optional: Check if base_name looks like a UUID for robustness
                     # if re.fullmatch(UUID_PATTERN, base_name, re.IGNORECASE):
                    base_names.add(base_name.lower())
                    logging.debug(f"  Found file: {filename} -> Base: {base_name.lower()}")
                     # else:
                     #      logging.debug(f"  Skipping non-UUID base name: {base_name} from file {filename}")
            else:
                logging.debug(f"  Skipping non-file entry: {filename}")
    except OSError as e:
        logging.error(f"Error listing files in {directory_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error scanning {directory_path}: {e}")
        return None

    logging.info(f"Found {len(base_names)} unique base filenames in {directory_path}")
    return base_names

def load_labels_from_file(labels_file):
    """Loads labels strictly from an existing file."""
    if not os.path.exists(labels_file):
        logging.error(f"Labels file '{labels_file}' not found.")
        return None
    if not os.path.isfile(labels_file):
         logging.error(f"Path '{labels_file}' exists but is not a file.")
         return None

    logging.info(f"Loading labels from file: '{labels_file}'")
    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            labels = json.load(f)
        # Basic validation
        if not isinstance(labels, list):
             logging.error(f"Labels data in '{labels_file}' is not a list (type: {type(labels)}).")
             return None
        logging.info(f"Successfully loaded {len(labels)} labels from '{labels_file}'.")
        return labels
    except json.JSONDecodeError as e:
        logging.error(f"Error reading JSON from '{labels_file}': {e}.")
    except Exception as e:
        logging.error(f"Unexpected error reading '{labels_file}': {e}.")
    return None


def load_or_download_labels(labels_url, headers, labels_file):
    """Loads labels from file or downloads them if file doesn't exist/fails."""
    labels = load_labels_from_file(labels_file)
    if labels is not None:
        return labels # Successfully loaded from file

    # Proceed to download only if loading failed or file didn't exist
    logging.info(f"Labels file not found or failed to load. Downloading from Label Studio: {labels_url}")

    try:
        response = requests.get(labels_url, headers=headers, verify=False, timeout=120)
        logging.debug(f"Received response from Label Studio API. Status Code: {response.status_code}")
        response.raise_for_status()

        with open(labels_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        logging.info(f"Export successful! Saved as '{labels_file}'")
        labels = response.json()

        # Basic validation after download
        if not isinstance(labels, list):
             logging.error(f"Downloaded labels data is not a list (type: {type(labels)}). Cannot use.")
             # Optionally remove the invalid downloaded file
             # try: os.remove(labels_file) except OSError: pass
             return None

        logging.info(f"Successfully parsed {len(labels)} labels from server.")
        return labels

    except requests.exceptions.Timeout:
        logging.critical(f"Failed to export labels: Request timed out. URL: {labels_url}")
    except requests.exceptions.HTTPError as e:
         logging.critical(f"Failed to export labels (HTTP Error): {e} - Status Code: {response.status_code}. URL: {labels_url}")
         logging.debug("Response text: %s", response.text[:500])
    except requests.exceptions.RequestException as e:
        logging.critical(f"Failed to export labels (Request Exception): {e}. URL: {labels_url}")
    except json.JSONDecodeError as e:
        logging.critical(f"Failed to parse JSON response from server: {e}")
        try:
            logging.error("Response text (first 500 chars): %s", response.text[:500])
        except NameError:
             logging.error("Response object not available to show text.")

    logging.error("Label download failed.")
    return None

def download_export_json(output_file="export.json", token=None, base_url=None, project_id=None):
    """
    Downloads just the export.json file from Label Studio.
    
    Args:
        output_file: Path to save the exported JSON (default: "export.json")
        token: Label Studio token (overrides env var)
        base_url: Label Studio URL (overrides default)
        project_id: Label Studio project ID (overrides default)
        
    Returns:
        Dictionary with the loaded labels data or error information
    """
    logging.info("--- Downloading Export JSON ---")
    
    # --- Configuration Setup ---
    base_label_studio_url = (base_url or DEFAULT_LS_URL).rstrip('/')
    project_id = project_id or DEFAULT_PROJECT_ID
    label_studio_token = token or LABEL_STUDIO_TOKEN or os.getenv("LABEL_STUDIO_TOKEN")
    
    logging.info(f"Using Label Studio URL: {base_label_studio_url}")
    logging.info(f"Using Project ID: {project_id}")
    logging.info(f"Output file: {output_file}")
    
    # --- Check if token exists ---
    if not label_studio_token:
        error_msg = "LABEL_STUDIO_TOKEN not found in environment variables or parameters."
        logging.critical(error_msg)
        return {"error": error_msg}
    
    # --- Set up headers ---
    headers = {
        "Authorization": f"Token {label_studio_token}",
        "User-Agent": f"LabelStudioTool/{SCRIPT_VERSION}-ExportJSON"
    }
    
    # --- Check if file already exists ---
    if os.path.exists(output_file):
        logging.info(f"Export file '{output_file}' already exists, loading from disk.")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                labels = json.load(f)
                
            # Basic validation
            if not isinstance(labels, list):
                logging.warning(f"Loaded data in '{output_file}' is not a list (type: {type(labels)}). Will redownload.")
            else:
                logging.info(f"Successfully loaded {len(labels)} tasks from existing file.")
                return {"labels": labels, "source": "file"}
                
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing existing JSON file '{output_file}': {e}. Will redownload.")
        except Exception as e:
            logging.error(f"Error reading existing file '{output_file}': {e}. Will redownload.")
    
    # --- Download from Label Studio ---
    logging.info("Downloading export.json from Label Studio...")
    labels_url = f"{base_label_studio_url}/api/projects/{project_id}/export?exportType=JSON"
    
    try:
        # Disable SSL warnings for the request
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(labels_url, headers=headers, verify=False, timeout=120)
        
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the response to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        logging.info(f"Export successful! Saved to '{output_file}'")
        
        # Parse the response
        labels = response.json()
        
        # Basic validation
        if not isinstance(labels, list):
            error_msg = f"Downloaded data is not a list (type: {type(labels)})."
            logging.error(error_msg)
            return {"error": error_msg}
        
        logging.info(f"Successfully downloaded {len(labels)} tasks.")
        return {"labels": labels, "source": "download"}
        
    except requests.exceptions.Timeout:
        error_msg = f"Request timed out: {labels_url}"
        logging.error(error_msg)
        return {"error": error_msg}
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e}, Status Code: {response.status_code}"
        logging.error(error_msg)
        logging.debug("Response text: %s", response.text[:500])
        return {"error": error_msg}
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {e}"
        logging.error(error_msg)
        return {"error": error_msg}
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON response: {e}"
        logging.error(error_msg)
        try:
            logging.debug("Response text (first 500 chars): %s", response.text[:500])
        except NameError:
            pass
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logging.error(error_msg)
        return {"error": error_msg}


# --- Core Action Functions ---
def run_download(script_dir=None, show_progress=True, token=None, 
                texts_dir=None, images_dir=None, labels_file=None, 
                base_url=None, project_id=None, progress_interval=None):
    """
    Handles the image download process.
    
    Args:
        script_dir: Base directory for paths. If None, uses current directory.
        show_progress: Whether to show progress bars (disable in notebooks sometimes)
        token: Label Studio token (overrides env var)
        texts_dir: Directory name for texts (overrides default)
        images_dir: Directory name for images (overrides default)
        labels_file: Path to labels file (overrides default)
        base_url: Label Studio URL (overrides default)
        project_id: Label Studio project ID (overrides default)
        progress_interval: Update interval for progress display (overrides default)
        
    Returns:
        Dictionary with download statistics
    """
    logging.info("--- Running Download Mode ---")
    
    # Setup base directory
    if script_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
    
    # --- Configuration & Path Setup using parameters or defaults ---
    texts_dir = os.path.abspath(os.path.join(script_dir, texts_dir or DEFAULT_TEXTS_DIR_NAME))
    image_save_dir = os.path.abspath(os.path.join(script_dir, images_dir or DEFAULT_IMAGES_DIR_NAME))
    labels_file_path = labels_file or os.path.abspath(os.path.join(script_dir, DEFAULT_LABELS_FILE))
    base_label_studio_url = (base_url or DEFAULT_LS_URL).rstrip('/')
    project_id = project_id or DEFAULT_PROJECT_ID
    progress_interval = progress_interval or DEFAULT_PROGRESS_INTERVAL
    label_studio_token = token or LABEL_STUDIO_TOKEN

    logging.info(f"Using Labels file: {labels_file_path}")
    logging.info(f"Using Label Studio Project ID: {project_id}")
    logging.info(f"Using Label Studio Base URL: {base_label_studio_url}")
    logging.info(f"Using Image save directory: {image_save_dir}")
    logging.info(f"Using XML check directory: {texts_dir}")

    if not os.path.isdir(texts_dir):
        logging.warning(f"Texts directory '{texts_dir}' does not exist. No XML files will be found, downloads will likely be skipped.")
    try:
        os.makedirs(image_save_dir, exist_ok=True)
        logging.info(f"Ensured image save directory exists: '{image_save_dir}'")
    except OSError as e:
        logging.critical(f"Could not create image save directory '{image_save_dir}': {e}. Exiting.")
        return {"error": f"Could not create directory: {e}"}

    # --- Authorization ---
    if not label_studio_token:
        error_msg = "LABEL_STUDIO_TOKEN not found in environment variables or parameters."
        logging.critical(error_msg)
        return {"error": error_msg}
    else:
        logging.info("Label Studio token loaded successfully.")

    headers = {
        "Authorization": f"Token {label_studio_token}",
        "User-Agent": f"LabelStudioTool/{SCRIPT_VERSION}-Download"
    }
    logging.debug("Authorization headers prepared.")

    # --- Load or Download Labels ---
    labels_url = f"{base_label_studio_url}/api/projects/{project_id}/export?exportType=JSON"
    labels = load_or_download_labels(labels_url, headers, labels_file_path)

    if labels is None:
        error_msg = "Failed to load or download labels. Cannot proceed."
        logging.critical(error_msg)
        return {"error": error_msg}

    total_tasks = len(labels)
    if total_tasks == 0:
        logging.warning(f"No tasks found in the labels file '{labels_file_path}'. Nothing to process.")
        return {
            "total_tasks": 0,
            "downloaded": 0,
            "skipped_exists": 0,
            "skipped_no_uuid": 0,
            "skipped_no_path": 0,
            "skipped_no_xml": 0,
            "failed": 0,
            "processed": 0
        }

    logging.info("--- Starting Image Download Process ---")
    logging.info(f"Processing {total_tasks} tasks from labels file...")

    # --- Initialize Counters ---
    download_count = 0
    skip_count_exists = 0
    skip_count_no_uuid = 0
    skip_count_no_path = 0
    skip_count_no_xml = 0
    fail_count = 0
    processed_count = 0
    start_time = time.time()

    # --- Set up Progress Bar ---
    tasks_iterator = labels
    if show_progress:
        try:
            # Try notebook tqdm first (for Jupyter environments)
            try:
                from tqdm.notebook import tqdm
                tasks_iterator = tqdm(labels, desc="Downloading images", unit="task")
                logging.debug("Using tqdm.notebook for progress display")
            except ImportError:
                # Fall back to standard tqdm (for terminal environments)
                from tqdm import tqdm
                tasks_iterator = tqdm(labels, desc="Downloading images", unit="task")
                logging.debug("Using standard tqdm for progress display")
        except ImportError:
            # If tqdm isn't available at all
            logging.warning("tqdm not available, falling back to log-based progress")
            # Will use regular progress updates
    
    # --- Process Tasks Loop ---
    for i, task in enumerate(tasks_iterator):
        processed_count += 1
        task_id_for_log = task.get("id", "N/A")

        if not isinstance(task, dict):
            logging.warning(f"Skipping item {i+1}: Invalid task format {type(task)}")
            continue

        task_data = task.get("data", {})
        if not task_data:
            logging.warning(f"Skipping task ID {task_id_for_log}: No 'data' field.")
            continue

        image_path = task_data.get("image")
        if not image_path or not isinstance(image_path, str):
            logging.debug(f"Skipping task ID {task_id_for_log}: 'image' path missing/invalid.")
            skip_count_no_path += 1
            continue

        match = re.search(UUID_PATTERN, image_path)
        if not match:
            logging.debug(f"Skipping task ID {task_id_for_log}: No UUID in path '{image_path}'")
            skip_count_no_uuid += 1
            continue
        unique_id_str = match.group(0)

        xml_filename = f"{unique_id_str}.xml"
        xml_filepath = os.path.join(texts_dir, xml_filename)
        if not os.path.exists(xml_filepath):
            logging.debug(f"Skipping task ID {task_id_for_log} (UUID: {unique_id_str}): XML not found '{xml_filepath}'")
            skip_count_no_xml += 1
            continue

        if not image_path.startswith("/data/local-files/"):
            logging.debug(f"Skipping task ID {task_id_for_log} (UUID: {unique_id_str}): Non-local path '{image_path}'")
            skip_count_no_path += 1
            continue

        image_url = base_label_studio_url + image_path
        base_name, orig_ext = os.path.splitext(os.path.basename(image_path))
        image_extension = orig_ext if orig_ext else '.jpg'
        image_name = f"{unique_id_str}{image_extension}"
        save_path = os.path.join(image_save_dir, image_name)

        if os.path.exists(save_path):
            logging.debug(f"Skipping download: Image exists '{save_path}'.")
            skip_count_exists += 1
            continue

        try:
            response = requests.get(url=image_url, headers=headers, verify=False, timeout=45)
            response.raise_for_status()
            with open(save_path, "wb") as img_file:
                img_file.write(response.content)
            logging.debug(f"Downloaded: {image_path} -> {image_name} (Task ID: {task_id_for_log})")
            download_count += 1
        except requests.exceptions.Timeout:
             logging.error(f"Failed download (Timeout): {image_url} (Task ID: {task_id_for_log}, UUID: {unique_id_str})")
             fail_count += 1
        except requests.exceptions.HTTPError as e:
             logging.error(f"Failed download (HTTP Error {response.status_code}): {image_url} (Task ID: {task_id_for_log}, UUID: {unique_id_str}). Error: {e}")
             fail_count += 1
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed download (Request Error): {image_url} (Task ID: {task_id_for_log}, UUID: {unique_id_str}). Error: {e}")
            fail_count += 1
        except IOError as e:
            logging.error(f"Failed to save image: {save_path} (Task ID: {task_id_for_log}, UUID: {unique_id_str}). Error: {e}")
            fail_count += 1
            if os.path.exists(save_path):
                 try: os.remove(save_path); logging.warning(f"Removed partial file: {save_path}")
                 except OSError as rm_err: logging.error(f"Could not remove partial file '{save_path}': {rm_err}")
        
        # Update the progress bar with additional info if using tqdm
        if show_progress and hasattr(tasks_iterator, 'set_postfix'):
            tasks_iterator.set_postfix({
                'downloaded': download_count, 
                'skipped': skip_count_exists + skip_count_no_xml + skip_count_no_uuid + skip_count_no_path,
                'failed': fail_count
            })
        # Fall back to periodic log messages if not using tqdm
        elif show_progress and not hasattr(tasks_iterator, 'set_postfix') and (i + 1) % progress_interval == 0:
            percent = ((i + 1) / total_tasks) * 100
            logging.info(f"Progress: {percent:.1f}% ({i + 1}/{total_tasks}) - Downloaded: {download_count}, Skipped: {skip_count_exists + skip_count_no_xml + skip_count_no_uuid + skip_count_no_path}, Failed: {fail_count}")

    # Close the progress bar if it exists
    if show_progress and hasattr(tasks_iterator, 'close'):
        tasks_iterator.close()

    elapsed_time = time.time() - start_time
    
    # --- Download Summary ---
    logging.info("\n--- Download Summary ---")
    logging.info(f"Processing completed in: {elapsed_time:.2f} seconds")
    logging.info(f"Total tasks in labels file: {total_tasks}")
    logging.info(f"Tasks processed: {processed_count}")
    logging.info(f"Successfully downloaded images: {download_count}")
    logging.info(f"Skipped (image already exists): {skip_count_exists}")
    logging.info(f"Skipped (no corresponding XML found): {skip_count_no_xml}")
    logging.info(f"Skipped (no UUID extracted): {skip_count_no_uuid}")
    logging.info(f"Skipped (no/invalid/non-local image path): {skip_count_no_path}")
    logging.info(f"Failed downloads/saves: {fail_count}")
    logging.info("------------------------")
    if fail_count > 0:
         logging.warning("There were download/save failures. Check logs above.")
         
    # Return statistics dictionary for use when imported as a module
    return {
        "total_tasks": total_tasks,
        "downloaded": download_count,
        "skipped_exists": skip_count_exists,
        "skipped_no_uuid": skip_count_no_uuid,
        "skipped_no_path": skip_count_no_path,
        "skipped_no_xml": skip_count_no_xml,
        "failed": fail_count,
        "processed": processed_count,
        "elapsed_time": elapsed_time
    }

def run_compare(script_dir=None, texts_dir=None, images_dir=None):
    """
    Handles the directory comparison process.
    
    Args:
        script_dir: Base directory for paths. If None, uses current directory.
        texts_dir: Directory name for texts (overrides default)
        images_dir: Directory name for images (overrides default)
        
    Returns:
        Dictionary with comparison statistics and boolean match result
    """
    logging.info("--- Running Compare Mode ---")
    
    # Setup base directory
    if script_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
    
    # --- Path Setup using parameters or defaults ---
    texts_compare_dir = os.path.abspath(os.path.join(script_dir, texts_dir or DEFAULT_TEXTS_DIR_NAME))
    images_compare_dir = os.path.abspath(os.path.join(script_dir, images_dir or DEFAULT_IMAGES_DIR_NAME))

    logging.info("Starting comparison between directories:")
    logging.info(f"  Texts directory: {texts_compare_dir}")
    logging.info(f"  Images directory: {images_compare_dir}")

    texts_bases = get_base_filenames(texts_compare_dir)
    images_bases = get_base_filenames(images_compare_dir)

    if texts_bases is None or images_bases is None:
        error_msg = "Failed to scan one or both directories."
        logging.critical(error_msg)
        return {"error": error_msg, "match": False}

    texts_only = texts_bases - images_bases
    images_only = images_bases - texts_bases
    common_bases = texts_bases.intersection(images_bases)

    logging.info("\n--- Comparison Results ---")
    match_count = len(common_bases)
    texts_only_count = len(texts_only)
    images_only_count = len(images_only)

    logging.info(f"Matching base filenames found in both: {match_count}")

    texts_only_list = []
    if texts_only_count > 0:
        logging.warning(f"Base filenames ONLY in '{texts_dir or DEFAULT_TEXTS_DIR_NAME}' ({texts_only_count}):")
        limit = 20; count = 0
        for base_name in sorted(list(texts_only)):
            texts_only_list.append(base_name)
            logging.warning(f"  - {base_name}.xml (Image missing)")
            count += 1
            if count >= limit: 
                logging.warning(f"  ... ({texts_only_count - limit} more)")
                break
    else:
        logging.info(f"No base filenames found only in '{texts_dir or DEFAULT_TEXTS_DIR_NAME}'.")

    images_only_list = []
    if images_only_count > 0:
        logging.warning(f"Base filenames ONLY in '{images_dir or DEFAULT_IMAGES_DIR_NAME}' ({images_only_count}):")
        limit = 20; count = 0
        for base_name in sorted(list(images_only)):
            images_only_list.append(base_name)
            logging.warning(f"  - {base_name}.* (XML missing)")
            count += 1
            if count >= limit: 
                logging.warning(f"  ... ({images_only_count - limit} more)")
                break
    else:
        logging.info(f"No base filenames found only in '{images_dir or DEFAULT_IMAGES_DIR_NAME}'.")

    logging.info("--------------------------")

    match_result = False
    if texts_only_count == 0 and images_only_count == 0:
        if match_count > 0 or (len(texts_bases) == 0 and len(images_bases) == 0):
             logging.info("Success: Directories match or are empty.")
             match_result = True
        else:
             logging.warning("Result: No common files, but not empty? Check logic.")
             match_result = True
    else:
        logging.warning("Mismatch Found: Directories do not match.")
        match_result = False
        
    # Return statistics dictionary for use when imported as a module
    return {
        "match": match_result,
        "matching_count": match_count,
        "texts_only_count": texts_only_count,
        "images_only_count": images_only_count,
        "texts_only": texts_only_list[:20],  # Limit to first 20
        "images_only": images_only_list[:20], # Limit to first 20
        "has_more_texts": texts_only_count > 20,
        "has_more_images": images_only_count > 20
    }

def run_split_json(script_dir=None, show_progress=True, labels_file=None, jsons_dir=None, progress_interval=None):
    """
    Splits the main labels JSON into individual files.
    
    Args:
        script_dir: Base directory for paths. If None, uses current directory.
        show_progress: Whether to show progress bars (disable in notebooks sometimes)
        labels_file: Path to labels file (overrides default)
        jsons_dir: Directory name for output JSON files (overrides default)
        progress_interval: Update interval for progress display (overrides default)
        
    Returns:
        Dictionary with split statistics
    """
    logging.info("--- Running Split JSON Mode ---")
    
    # Setup base directory
    if script_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

    # --- Path Setup using parameters or defaults ---
    labels_file_path = labels_file or os.path.abspath(os.path.join(script_dir, DEFAULT_LABELS_FILE))
    jsons_save_dir = os.path.abspath(os.path.join(script_dir, jsons_dir or DEFAULT_JSONS_DIR_NAME))
    progress_interval = progress_interval or DEFAULT_PROGRESS_INTERVAL

    logging.info(f"Input labels file: {labels_file_path}")
    logging.info(f"Output directory for split JSONs: {jsons_save_dir}")

    try:
        os.makedirs(jsons_save_dir, exist_ok=True)
        logging.info(f"Ensured JSON save directory exists: '{jsons_save_dir}'")
    except OSError as e:
        error_msg = f"Could not create JSON save directory '{jsons_save_dir}': {e}."
        logging.critical(error_msg)
        return {"error": error_msg}

    labels = load_labels_from_file(labels_file_path) # Split requires existing file

    if labels is None:
        error_msg = f"Failed to load labels from '{labels_file_path}'. Please ensure the file exists and is valid."
        logging.critical(error_msg)
        return {"error": error_msg}

    total_tasks = len(labels)
    if total_tasks == 0:
        logging.warning(f"No tasks found in the labels file '{labels_file_path}'. Nothing to split.")
        return {
            "total_tasks": 0,
            "json_created": 0,
            "skipped_no_uuid": 0,
            "failed_writes": 0,
            "processed": 0
        }

    logging.info(f"Processing {total_tasks} tasks to split from labels file...")

    # --- Initialize Counters ---
    json_created_count = 0
    skip_count_no_uuid_split = 0
    fail_count_write = 0
    processed_count_split = 0
    start_time = time.time()

    # --- Set up Progress Bar ---
    tasks_iterator = labels
    if show_progress:
        try:
            # Try notebook tqdm first (for Jupyter environments)
            try:
                from tqdm.notebook import tqdm
                tasks_iterator = tqdm(labels, desc="Splitting JSONs", unit="file")
                logging.debug("Using tqdm.notebook for progress display")
            except ImportError:
                # Fall back to standard tqdm (for terminal environments)
                from tqdm import tqdm
                tasks_iterator = tqdm(labels, desc="Splitting JSONs", unit="file")
                logging.debug("Using standard tqdm for progress display")
        except ImportError:
            # If tqdm isn't available at all
            logging.warning("tqdm not available, falling back to log-based progress")
            # Will use regular progress updates

    # --- Process Tasks Loop for Splitting ---
    for i, task in enumerate(tasks_iterator):
        processed_count_split += 1
        task_id_for_log = task.get("id", "N/A")

        if not isinstance(task, dict):
            logging.warning(f"Skipping item {i+1}: Invalid task format {type(task)}")
            continue

        task_data = task.get("data", {})
        image_path = task_data.get("image")

        unique_id_str = None
        if image_path and isinstance(image_path, str):
            match = re.search(UUID_PATTERN, image_path)
            if match: unique_id_str = match.group(0)

        if not unique_id_str:
            logging.warning(f"Skipping task ID {task_id_for_log}: No UUID in path '{image_path}'")
            skip_count_no_uuid_split += 1
            continue

        output_filename = f"{unique_id_str}.json"
        output_filepath = os.path.join(jsons_save_dir, output_filename)

        try:
            with open(output_filepath, "w", encoding="utf-8") as f_out:
                json.dump(task, f_out, ensure_ascii=False, indent=4)
            logging.debug(f"Created JSON: {output_filepath} (Task ID: {task_id_for_log})")
            json_created_count += 1
        except IOError as e:
            logging.error(f"Failed write: {output_filepath} (Task ID: {task_id_for_log}). Error: {e}")
            fail_count_write += 1
        except Exception as e:
             logging.error(f"Unexpected error writing {output_filepath} (Task ID: {task_id_for_log}): {e}")
             fail_count_write += 1

        # Update the progress bar with additional info if using tqdm
        if show_progress and hasattr(tasks_iterator, 'set_postfix'):
            tasks_iterator.set_postfix({
                'created': json_created_count, 
                'skipped': skip_count_no_uuid_split,
                'failed': fail_count_write
            })
        # Fall back to periodic log messages if not using tqdm
        elif show_progress and not hasattr(tasks_iterator, 'set_postfix') and (i + 1) % progress_interval == 0:
            percent = ((i + 1) / total_tasks) * 100
            logging.info(f"Splitting Progress: {percent:.1f}% ({i + 1}/{total_tasks}) - Created: {json_created_count}, Skipped: {skip_count_no_uuid_split}, Failed: {fail_count_write}")

    # Close the progress bar if it exists
    if show_progress and hasattr(tasks_iterator, 'close'):
        tasks_iterator.close()

    elapsed_time = time.time() - start_time
    
    # --- Split Summary ---
    logging.info("\n--- Split JSON Summary ---")
    logging.info(f"Processing completed in: {elapsed_time:.2f} seconds")
    logging.info(f"Total tasks in input file: {total_tasks}")
    logging.info(f"Tasks processed: {processed_count_split}")
    logging.info(f"Successfully created JSON files: {json_created_count}")
    logging.info(f"Skipped (no UUID extracted): {skip_count_no_uuid_split}")
    logging.info(f"Failed JSON writes: {fail_count_write}")
    logging.info("--------------------------")
    if fail_count_write > 0:
         logging.warning("There were errors writing some JSON files. Check logs above.")
         
    # Return statistics dictionary for use when imported as a module
    return {
        "total_tasks": total_tasks,
        "json_created": json_created_count,
        "skipped_no_uuid": skip_count_no_uuid_split,
        "failed_writes": fail_count_write,
        "processed": processed_count_split,
        "elapsed_time": elapsed_time
    }


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(
        description=f"Label Studio Tool: Download, Compare, or Split JSON (v{SCRIPT_VERSION}).\n"
                    "Uses configuration from .env file or script defaults.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Mode Selection ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--download",
        action="store_true",
        help="Run image download (requires network & LS access)."
    )
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare base filenames in texts and images dirs."
    )
    mode_group.add_argument(
        "--split-json",
        action="store_true",
        help="Split the main labels JSON into individual task JSON files."
    )

    # --- Common Arguments ---
    parser.add_argument(
        "--debug",
        action="store_true",
        # Reads directly from env var if present, otherwise False
        default=os.getenv('DEBUG_LOGGING', 'false').lower() == 'true',
        help="Enable detailed DEBUG logging output (Env: DEBUG_LOGGING=true)."
    )

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Setup ---
    # Setup logging based on --debug flag
    setup_logging(args.debug)
    logging.info(f"Running Label Studio Tool v{SCRIPT_VERSION}")

    # Determine script directory robustly
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logging.debug("__file__ not defined, using CWD as script base: %s", script_dir)
    logging.debug(f"Script base directory set to: {script_dir}")

    # --- Execute Selected Mode ---
    exit_code = 0
    try:
        if args.download:
            run_download(script_dir)
        elif args.compare:
            result = run_compare(script_dir)
            if not result["match"]: exit_code = 1 # Indicate mismatch
        elif args.split_json:
            run_split_json(script_dir)
        else:
            # Should not happen
            logging.error("Internal Error: No mode selected.")
            parser.print_help()
            exit_code = 1

    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user (Ctrl+C). Exiting.")
        exit_code = 130
    except Exception as e:
        logging.critical("An unexpected error occurred during execution:", exc_info=True)
        exit_code = 1
    finally:
        logging.info(f"Script finished with exit code {exit_code}.")
        sys.exit(exit_code)