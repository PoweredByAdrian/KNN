import json
import os
import sys
import pprint # Import pprint for pretty printing

# --- Configuration ---
labels_file = "export.json"
# --- End Configuration ---

print(f"Attempting to read the first task's 'data' field from: {labels_file}")

# Check if the file exists
if not os.path.exists(labels_file):
    print(f"Error: Labels file '{labels_file}' not found in the current directory ({os.getcwd()}).")
    print("Please ensure the file exists and the script is run from the correct directory.")
    sys.exit(1) # Exit if the file doesn't exist

# Try to open and parse the JSON file
try:
    with open(labels_file, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    print(f"Successfully loaded '{labels_file}'.")
except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON from '{labels_file}'. It might be corrupted or not valid JSON.")
    print(f"Details: {e}")
    sys.exit(1) # Exit on JSON error
except Exception as e:
    print(f"An unexpected error occurred while reading '{labels_file}': {e}")
    sys.exit(1) # Exit on other file reading errors

# Validate the structure (expecting a list of tasks)
if not isinstance(all_tasks, list):
    print(f"Error: The content of '{labels_file}' is not a JSON list as expected.")
    print(f"Instead, found type: {type(all_tasks)}")
    print("The main download script expects a list of task objects.")
    sys.exit(1)

if not all_tasks: # Check if the list is empty
    print(f"Error: The JSON list in '{labels_file}' is empty. No tasks found to inspect.")
    sys.exit(1)

# Get the first task
first_task = all_tasks[0]
print("Successfully accessed the first task object.")

if not isinstance(first_task, dict):
    print(f"Error: The first item in the JSON list is not a dictionary (task object).")
    print(f"Found type: {type(first_task)}")
    sys.exit(1)

# Get the 'data' field from the first task
# Using .get() is safer as it returns None if 'data' doesn't exist
first_task_data = first_task.get("data")

if first_task_data is None:
    print("\nError: The first task in the JSON file does NOT contain a 'data' field.")
    print("\n--- Full content of the FIRST task (for inspection) ---")
    pprint.pprint(first_task) # Print the whole task if 'data' is missing
    print("------------------------------------------------------")
    sys.exit(1)

# Check if 'data' is a dictionary as expected
if not isinstance(first_task_data, dict):
     print(f"\nWarning: The 'data' field in the first task is not a dictionary.")
     print(f"Found type: {type(first_task_data)}")
     print("The download script expects 'data' to be a dictionary containing keys like 'image' and the unique ID.")

# Print the content of the 'data' field nicely using pprint
print(f"\n--- Content of the 'data' field from the FIRST task (Task ID: {first_task.get('id', 'N/A')}) ---")
pprint.pprint(first_task_data)
print("-------------------------------------------------------------")

print("\n>> Action Required <<")
print("1. Examine the output above carefully.")
print("2. Identify the actual key name that holds the unique identifier you want to use for filenames (e.g., 'filename', 'id', 'name', 'uuid', etc.).")
print("3. Identify the actual key name that holds the image path (usually 'image').")
print("4. Update the main download script (download.py) to use these EXACT key names in the `task_data.get(...)` calls.")