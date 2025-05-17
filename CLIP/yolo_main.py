import os
import cv2
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Import your processing function from the existing script
from process_yolo import process_detections

def process_image_directory(
    model_path,
    image_dir,
    output_base_dir="YOLO",
    max_images=None,
    image_extensions=(".jpg", ".jpeg", ".png", ".bmp")
):
    """
    Process multiple images from a directory with a YOLO model and save results.
    
    Args:
        model_path: Path to YOLO model
        image_dir: Directory containing images to process
        output_base_dir: Base directory for outputs
        max_images: Maximum number of images to process (None = process all)
        image_extensions: Tuple of valid image extensions to process
    
    Returns:
        Dict with counts of processed images and any errors
    """
    # Create output directories
    json_dir = os.path.join(output_base_dir, "jsons")
    viz_dir = os.path.join(output_base_dir, "images")
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Output directories created: {json_dir}, {viz_dir}")
    
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": "Model loading failed", "processed": 0}
    
    # Get list of image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
    
    total_images = len(image_files)
    print(f"Found {total_images} images in {image_dir}")
    
    if max_images is not None and max_images < total_images:
        print(f"Limiting processing to {max_images} images")
        image_files = image_files[:max_images]
    
    # Process images
    results_summary = {
        "total": len(image_files),
        "processed": 0,
        "success": 0,
        "errors": 0,
        "files": []
    }
    
    # Process each image with progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get base filename without extension
            img_name = img_path.stem
            
            # Define output paths
            json_path = os.path.join(json_dir, f"{img_name}.json")
            viz_path = os.path.join(viz_dir, f"{img_name}_viz.png")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Error: Could not load image {img_path}")
                results_summary["errors"] += 1
                continue
            
            # Run detection
            detection_result = model(image)[0]  # Get first detection result
            
            # Process detection
            results, json_output, viz_output = process_detections(
                detection=detection_result,
                image=image,
                json_path=json_path,
                viz_path=viz_path
            )
            
            # Add to results summary
            results_summary["processed"] += 1
            results_summary["success"] += 1
            results_summary["files"].append({
                "image": str(img_path),
                "json": json_output,
                "visualization": viz_output,
                "detections": {
                    "images": sum(1 for match in results.values() if match["image"]["cls"] == 1),
                    "descriptions": sum(1 for match in results.values() 
                                     if match["description"] and match["description"]["cls"] == 0),
                }
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results_summary["errors"] += 1
            results_summary["files"].append({
                "image": str(img_path),
                "error": str(e)
            })
    
    # Save overall summary
    import json
    summary_path = os.path.join(output_base_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Processed {results_summary['processed']} images")
    print(f"Successful: {results_summary['success']}, Errors: {results_summary['errors']}")
    print(f"Results saved to {output_base_dir}")
    print(f"Summary saved to {summary_path}")
    
    return results_summary

def main():
    """Command-line interface for the batch processing script."""
    parser = argparse.ArgumentParser(description="Process multiple images with YOLO and create visualizations")
    
    parser.add_argument("--model", "-m", required=True, help="Path to YOLO model")
    parser.add_argument("--input", "-i", required=True, help="Directory containing input images")
    parser.add_argument("--output", "-o", default="YOLO", help="Base directory for output files")
    parser.add_argument("--max", "-n", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--extensions", "-e", default=".jpg,.jpeg,.png,.bmp", 
                       help="Comma-separated list of image extensions to process")
    
    args = parser.parse_args()
    
    # Process image extensions
    extensions = tuple(args.extensions.split(","))
    
    # Run processing
    process_image_directory(
        model_path=args.model,
        image_dir=args.input,
        output_base_dir=args.output,
        max_images=args.max,
        image_extensions=extensions
    )

if __name__ == "__main__":
    main()
    
# python3 batch_process.py --model path/to/model.pt --input path/to/images --output YOLO --max 100