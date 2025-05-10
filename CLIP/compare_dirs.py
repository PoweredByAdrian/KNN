import os
import sys
import logging

# --- Configuration ---
# Assumes this script is in the same parent directory as 'texts' and 'images'
try:
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
     # Fallback for interactive environments
    script_parent_dir = os.getcwd()
    # logging is not configured yet, so use print for this potential warning
    print(f"Warning: __file__ not defined, using current working directory: {script_parent_dir}")

TEXTS_DIR_NAME = "texts"
IMAGES_DIR_NAME = "images"
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more verbose output (like listing all files)

# Construct full paths
texts_dir = os.path.join(script_parent_dir, TEXTS_DIR_NAME)
images_dir = os.path.join(script_parent_dir, IMAGES_DIR_NAME)
# --- End Configuration ---

# --- Basic Logging Setup ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
# --- End Logging Setup ---


def get_base_filenames(directory_path):
    """
    Lists files in a directory and returns a set of their base names
    (filename without extension), converted to lowercase.
    """
    base_names = set()
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        return None # Indicate error

    logging.debug(f"Scanning directory: {directory_path}")
    try:
        for filename in os.listdir(directory_path):
            full_path = os.path.join(directory_path, filename)
            if os.path.isfile(full_path):
                base_name, ext = os.path.splitext(filename)
                if base_name: # Ignore files with no base name (like '. M ACOSX')
                    base_names.add(base_name.lower()) # Use lowercase for case-insensitive comparison
                    logging.debug(f"  Found file: {filename} -> Base: {base_name.lower()}")
            else:
                logging.debug(f"  Skipping directory entry: {filename}")
    except OSError as e:
        logging.error(f"Error listing files in {directory_path}: {e}")
        return None

    logging.info(f"Found {len(base_names)} unique base filenames in {directory_path}")
    return base_names

# --- Main Execution ---
logging.info("Starting comparison between directories:")
logging.info(f"  Texts directory: {texts_dir}")
logging.info(f"  Images directory: {images_dir}")

# Get base filenames from both directories
texts_bases = get_base_filenames(texts_dir)
images_bases = get_base_filenames(images_dir)

# Check if directory scanning failed
if texts_bases is None or images_bases is None:
    logging.critical("Failed to scan one or both directories. Aborting comparison.")
    sys.exit(1)

# Perform comparisons using set operations
texts_only = texts_bases - images_bases
images_only = images_bases - texts_bases
common_bases = texts_bases.intersection(images_bases) # Files present in both

# --- Report Results ---
logging.info("\n--- Comparison Results ---")
match_count = len(common_bases)
texts_only_count = len(texts_only)
images_only_count = len(images_only)

logging.info(f"Matching base filenames found in both directories: {match_count}")

if texts_only_count > 0:
    logging.warning(f"Base filenames found ONLY in '{TEXTS_DIR_NAME}' directory ({texts_only_count}):")
    # Log only a few examples if the list is very long, or log all if needed
    limit = 20
    count = 0
    for base_name in sorted(list(texts_only)):
         logging.warning(f"  - {base_name}.xml (Expected)")
         count += 1
         if count >= limit:
              logging.warning(f"  ... (and {texts_only_count - limit} more)")
              break
else:
    logging.info(f"No base filenames found only in '{TEXTS_DIR_NAME}'.")


if images_only_count > 0:
    logging.warning(f"Base filenames found ONLY in '{IMAGES_DIR_NAME}' directory ({images_only_count}):")
    limit = 20
    count = 0
    # It's harder to know the exact extension for images-only, so just list bases
    for base_name in sorted(list(images_only)):
         logging.warning(f"  - {base_name}.[image_extension] (Expected XML missing)")
         count += 1
         if count >= limit:
              logging.warning(f"  ... (and {images_only_count - limit} more)")
              break
else:
    logging.info(f"No base filenames found only in '{IMAGES_DIR_NAME}'.")


logging.info("--------------------------")

# Final conclusion
if texts_only_count == 0 and images_only_count == 0:
    if match_count > 0:
        logging.info("Success: The base filenames in both directories perfectly match!")
    else:
        logging.warning("Result: Both directories appear to be empty or contain no matching base filenames.")
else:
    logging.warning("Mismatch Found: The base filenames in the directories do not perfectly match. See details above.")

logging.info("Comparison script finished.")