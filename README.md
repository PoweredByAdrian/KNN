# Advanced Image and Text Processing & YOLOv8 Training Toolkit

This project provides a comprehensive suite of tools for advanced image analysis, annotation processing, semantic image-text matching, and custom YOLOv8 model training. It leverages state-of-the-art deep learning models like YOLO for object detection and CLIP for image-text similarity. The project is organized around Jupyter Notebook pipelines located in the `CLIP/` directory, supported by a collection of custom Python scripts for various tasks including data preparation, model training, and inference.

## Features

*   **Custom YOLOv8 Model Training:**
    *   Scripts for data preparation (`split_data.py`, `create_labels.py`).
    *   Configuration via `data.yaml`.
    *   Training (`train.py`) and testing/validation (`test.py`) of YOLOv8 models.
*   **Object Detection (YOLO Inference):**
    *   Utilizes custom-trained or pre-trained YOLO models to detect objects within images.
*   **Annotation Processing:**
    *   Filters and refines XML annotations (e.g., Pascal VOC format) based on YOLO detections.
    *   Removes specific labeled regions (e.g., picture descriptions) from XML files.
*   **Image Processing:**
    *   Crops images based on detected object regions.
*   **Semantic Image-Text Matching (CLIP):**
    *   Employs CLIP (or M-CLIP) models to find semantically relevant text passages for images.
    *   Generates visualizations of matched images with their corresponding text contexts.
*   **Data Management & Acquisition (Label Studio Integration):**
    *   Downloads datasets (images, JSON annotations) from Label Studio projects (`download.py`).
    *   Splits master JSON exports into individual task files.
    *   Analyzes label distributions within datasets.
    *   Filters datasets based on specific labels.
*   **Intersection Analysis:** Compares AI-generated (CLIP) text matches with manually annotated text descriptions.
*   **Modular Design:** Relies on a set of reusable Python scripts, some potentially housed in the `utils/` directory.

## Pipelines & Core Workflows

The project features two primary Jupyter Notebooks within the `CLIP/` directory, orchestrating distinct workflows:

1.  **`CLIP/pipeline.ipynb` (YOLO-Driven Processing Pipeline):**
    *   **Focus:** A streamlined pipeline for processing a pre-existing set of images and XML annotations using a trained YOLO model (e.g., one trained via `train.py`, output likely in `runs/` or `last_train/`).
    *   **Key Steps:** YOLO inference, XML filtering, image trimming, and CLIP-based description matching.
    *   **Input Data:**
        *   Trained YOLO model (e.g., `best.pt` from a training run).
        *   Source XML annotation files (e.g., in `datasets/texts/`).
        *   Source image files (e.g., in `datasets/downloaded_images/`).
    *   **Output Data:** YOLO JSONs, filtered XMLs, cropped images, and final image-text match outputs. (See `DIRS` in the notebook for specific sub-paths, often created under `CLIP/pipeline/test/...`).

2.  **`CLIP/KNN.ipynb` (Comprehensive Label Studio & CLIP Pipeline):**
    *   **Focus:** An end-to-end pipeline starting from data acquisition from Label Studio, through analysis, filtering, processing, and finally CLIP-based matching and evaluation.
    *   **Key Steps:** Label Studio data download, data analysis, filtering, image cropping, XML description filtering, CLIP matching, and intersection analysis.
    *   **Input Data:**
        *   Label Studio API Token.
        *   OCR XML files (e.g., in `datasets/texts/`).
    *   **Output Data:** Downloaded data, filtered datasets, processed images/XMLs, CLIP matching outputs, and comparison visualizations. (See `DIRS` in the notebook for specific sub-paths like `filtered_jsons/`, `cropped_images/`, `output_context/`, typically created relative to the project root or within `CLIP/`).

## Core Scripts & Files

This project utilizes several Python scripts and configuration files:

*   **Notebooks (in `CLIP/`):**
    *   `pipeline.ipynb`: YOLO-driven processing.
    *   `KNN.ipynb`: Label Studio & CLIP end-to-end pipeline.
*   **Data Preparation & Training (Root Directory):**
    *   `create_labels.py`: Likely for converting annotation formats or preparing YOLO labels.
    *   `split_data.py`: Splits dataset into training, validation, (and test) sets.
    *   `train.py`: Script to train YOLOv8 models.
    *   `test.py`: Script to evaluate/test trained YOLOv8 models.
    *   `data.yaml`: YOLOv8 dataset configuration file (specifies paths to train/val images, number of classes, class names).
*   **Data Acquisition (Root Directory):**
    *   `download.py`: Manages data download from Label Studio and splits JSON exports.
*   **Pipeline Utility Scripts (Imported by Notebooks in `CLIP/`):**
    *   `process_yolo.py`: Handles YOLO model inference.
    *   `filter_picture_descriptions.py`: Filters XMLs based on YOLO/JSON data.
    *   `trim_images.py`: Crops images based on bounding boxes.
    *   `process_descriptions.py`: Core script for CLIP-based image-text matching.
    *   `cut_text.py`: Helper for `process_descriptions.py`, likely for text segmentation.
    *   `count_json.py`: Analyzes label occurrences in annotation JSONs.
    *   `filter_by_label.py`: Filters datasets based on labels.
    *   `find_label_description.py`: Identifies items with description-related labels.
*   **Other Important Files (Root Directory):**
    *   `export.json` / `filtered_export.json`: Example Label Studio export files.
    *   `requirements.txt`: List of Python dependencies.
    *   `KNN_checkpoint.pdf`: Likely a report or presentation related to the `KNN.ipynb` pipeline.
*   **Directories:**
    *   `CLIP/`: Contains the main Jupyter Notebook pipelines and some of their helper scripts.
for storing raw and prepared datasets (e.g., `datasets/images/`, `datasets/labels/`, `datasets/texts/`, `datasets/downloaded_images/`).
    *   `last_train/`: Typically created by YOLO to store results of the most recent training.
    *   `runs/`: Typically created by YOLO to store results of training, validation, and prediction runs.
    *   `utils/`: `draw.py`

## Prerequisites

*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   Required Python packages: Install using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `ultralytics`, `torch`, `pandas`, `opencv-python`, `transformers`, `clip`, `multilingual-clip`, `lxml`, etc.

## Setup

1.  **Clone the repository (if applicable) or set up your project directory.**
2.  **Install Prerequisites:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Preparation:**
    *   Organize your source images and XML/annotation files. It's recommended to use a structure like:
        *   `datasets/images/` (for YOLO training images)
        *   `datasets/labels/` (for YOLO training labels)
        *   `CLIP/texts/` (for source XML files used by pipelines)
        *   `CLIP/downloaded_images/` (for images used by pipelines, can be same as `datasets/images/` or downloaded via `download.py`)
    *   Update `data.yaml` to point to your training and validation image directories within `datasets/`, and define your class names and number of classes.
    *   Use `split_data.py` and `create_labels.py` as needed to prepare your data for YOLO training.
4.  **Label Studio Token (for `CLIP/KNN.ipynb`):**
    *   Create a file `~/.config/label_studio_config.json` with the content:
        ```json
        {"token": "your_label_studio_api_token_here"}
        ```
    *   Alternatively, modify how `label_studio_token` is loaded in `CLIP/KNN.ipynb`.
5.  **Verify Directory Paths in Notebooks:**
    *   Before running the notebooks in `CLIP/`, review the `DIRS` dictionary defined at the beginning of each.
    *   Ensure paths for source data (e.g., `DIRS["xml_src_dir"]`, `DIRS["images_src_dir"]`) correctly point to your data (e.g., to subfolders within `datasets/`).
    *   Output directories are generally created by the notebooks if they don't exist.

## Running Workflows

### 1. Training a Custom YOLO Model

1.  Prepare your dataset and configure `data.yaml`.
2.  Run the training script:
    ```bash
    python train.py
    ```
    (Adjust parameters as needed). Trained models and results will be saved in `runs/train/` or `last_train/`.

### 2. Running the Jupyter Notebook Pipelines

1.  Launch Jupyter Notebook or JupyterLab from the project's root directory.
2.  Navigate into the `CLIP/` directory.
3.  Open either `pipeline.ipynb` or `KNN.ipynb`.
4.  **For `CLIP/pipeline.ipynb`:** Ensure the `model_path` variable points to your trained YOLO model (e.g., `runs/train/exp/weights/best.pt`).
5.  Execute the cells sequentially.
    *   Markdown cells provide explanations for each step.
    *   The first few code cells usually handle setup, directory creation, and data sampling/download.
    *   Pay attention to cells that might clear existing output directories if you want to preserve previous runs.
6.  Monitor the output, especially for progress bars and summary statistics.

## Configuration

Key configuration points:

*   **`data.yaml`:** For YOLO dataset paths and class definitions.
*   **Command-line arguments for `train.py`, `test.py`**.
*   **In Jupyter Notebooks (`CLIP/*.ipynb`):**
    *   `DIRS` dictionary: Defines input, intermediate, and output paths for pipeline stages.
    *   `model_path` (in `CLIP/pipeline.ipynb`): Path to the YOLO model file.
    *   `label_studio_token` (in `CLIP/KNN.ipynb`): For Label Studio API.

## Troubleshooting

*   **`ModuleNotFoundError`**: Ensure all custom `.py` scripts are in their expected locations (root, `CLIP/`) and are importable. Verify all packages from `requirements.txt` are installed.
*   **File Not Found Errors**: Double-check paths in `data.yaml` and the `DIRS` dictionary in notebooks. Ensure your source data is correctly placed, likely within the `datasets/` folder.
*   **CUDA/GPU Issues (for YOLO/CLIP/PyTorch):** Ensure PyTorch is installed with the correct CUDA version if you intend to use a GPU. If not, models will run on CPU (slower). The `requirements.txt` specifies `torch==2.7.0` and `torchvision==0.22.0`; ensure these are compatible with your GPU drivers or install CPU-only versions if needed.