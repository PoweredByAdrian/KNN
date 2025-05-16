from typing import Optional, Tuple, List
from PIL import Image
import torch
import clip
from cut_text import OCRDocument, TextBlock, draw_blocks_on_image
import torch.nn.functional as F
import argparse
import os
import sys
import shutil

def find_best_matching_block(
    image_path: str,
    xml_path: str,
    model,
    preprocess,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[Tuple["TextBlock", float, List[float], List[float], "OCRDocument"]]:
    """
    Given a cropped image and OCR XML, returns the block whose text best matches the image,
    along with probabilities, cosine similarities, and the OCRDocument.

    Args:
        image_path: Path to the cropped image.
        xml_path: Path to the OCR XML.
        model: Loaded CLIP model (from clip.load()).
        preprocess: CLIP preprocessing transform (from clip.load()).
        device: CUDA or CPU.

    Returns:
        Tuple of:
            - best matching TextBlock
            - softmax probability of best match
            - list of all probabilities
            - list of all cosine similarities
            - OCRDocument instance
    """
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        return None

    # 1. Load OCR blocks
    doc = OCRDocument(xml_path)
    blocks = doc.generate_blocks(lines_per_block=1, overlap=0)
    if not blocks:
        print(f"No text blocks found in {xml_path}")
        return None

    # 2. Preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

    # 3. Tokenize block texts
    texts = [block.get_text() for block in blocks]
    text_tokens = clip.tokenize(texts).to(device)

    # 4. CLIP encodings and similarity scores
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Softmax probabilities
        logits_per_image, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0].tolist()

        # Cosine similarities
        image_features_norm = F.normalize(image_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        similarities = torch.matmul(image_features_norm, text_features_norm.T)[0].cpu().tolist()

    best_idx = int(torch.tensor(probs).argmax())
    best_prob = probs[best_idx]

    return blocks[best_idx], best_prob, probs, similarities, doc


def find_available_image(base_image_path, max_suffix=20):
    """
    Try to find an available image file by checking the base name and suffixed versions.
    
    Args:
        base_image_path: Base path without suffix (e.g., "exported_images/ID.jpg")
        max_suffix: Maximum suffix number to check
        
    Returns:
        Path to the first available image file, or None if none found
    """
    # First check if the base image exists
    if os.path.exists(base_image_path):
        return base_image_path
        
    # If not, try with suffixes _2, _3, etc.
    base_name, extension = os.path.splitext(base_image_path)
    
    for suffix in range(2, max_suffix + 1):
        suffixed_path = f"{base_name}_{suffix}{extension}"
        if os.path.exists(suffixed_path):
            print(f"Found alternative image: {suffixed_path}")
            return suffixed_path
            
    return None


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
        print(f"Found {len(available_images)} image(s): {', '.join(os.path.basename(img) for img in available_images)}")
    else:
        print(f"No images found for base path: {os.path.basename(base_image_path)}")
        
    return available_images


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Find text description in an image using CLIP model"
    )
    parser.add_argument(
        "id", 
        help="ID of the image and XML file (e.g., '7f3229b2-71c7-11e1-a313-005056a60003')"
    )
    parser.add_argument(
        "--images-dir", 
        default="exported_images",
        help="Directory containing exported images"
    )
    parser.add_argument(
        "--texts-dir", 
        default="filtered_texts",
        help="Directory containing filtered XML texts"
    )
    parser.add_argument(
        "--similarity-threshold", 
        type=float, 
        default=0.25,
        help="Minimum similarity to include in context"
    )
    parser.add_argument(
        "--max-lines-context", 
        type=int, 
        default=3,
        help="Maximum lines to check above and below"
    )
    parser.add_argument(
        "--model", 
        default="ViT-B/32",
        help="CLIP model to use"
    )
    parser.add_argument(
        "--max-image-suffix",
        type=int,
        default=10,
        help="Maximum suffix number to check for alternative images (e.g., ID_2.jpg, ID_3.jpg)"
    )

    args = parser.parse_args()

    # Construct base image and XML paths based on ID
    base_image_path = os.path.join(args.images_dir, f"{args.id}.jpg")
    xml_path = os.path.join(args.texts_dir, f"filtered_{args.id}.xml")

    # Check if XML exists
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        return 1

    # Find all available image files
    available_images = find_available_images(base_image_path, args.max_image_suffix)
    if not available_images:
        print(f"Error: No images found for ID {args.id}. Checked {base_image_path} and alternatives with suffixes _2 to _{args.max_image_suffix}")
        return 1

    print(f"Processing ID: {args.id}")
    print(f"XML path: {xml_path}")
    
    # Initialize CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model, preprocess = clip.load(args.model, device=device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return 1
        
    # Store all context blocks from all images
    all_context_blocks = []
    
    # Process each image
    for idx, image_path in enumerate(available_images):
        print(f"\nProcessing image {idx+1}/{len(available_images)}: {image_path}")
        
        # Find best matching block for this image
        result = find_best_matching_block(
            image_path=image_path,
            xml_path=xml_path,
            model=model,
            preprocess=preprocess,
            device=device
        )

        if result is None:
            print(f"Skipping image {image_path} due to error")
            continue

        # Extract the actual image ID from the found image path
        image_filename = os.path.basename(image_path)
        image_id_with_suffix = os.path.splitext(image_filename)[0]  # Remove extension
        
        best_block, prob, all_probs, all_similarities, doc = result
        print(f"Matched block with {prob:.2%} probability:\n{best_block.get_text()}")
        print("\nAll probabilities:")
        for i, p in enumerate(all_probs):
            if i < len(doc.blocks):  # Ensure we don't go out of bounds
                print(f"  Block {i}: {p:.4f} Text: {doc.blocks[i].get_text()}")

        print("\nAll cosine similarities:")
        for i, s in enumerate(all_similarities):
            print(f"  Block {i}: {s:.4f}")

        # Build contextual block around the best matching text
        print("\nBuilding context block around best match...")

        # Parameters for context building
        similarity_threshold = args.similarity_threshold
        max_lines_context = args.max_lines_context

        # Find the index of the highest probability
        best_idx = all_probs.index(max(all_probs))

        # Only build context if best match exceeds threshold
        if all_similarities[best_idx] > similarity_threshold:
            # We'll reuse the blocks generated earlier
            blocks = doc.generate_blocks(lines_per_block=1, overlap=0)
            
            # Start with best matching block
            context_blocks = [blocks[best_idx]]
            context_text = blocks[best_idx].get_text()
            
            # Add blocks above if they meet similarity threshold
            for i in range(1, max_lines_context + 1):
                if best_idx - i >= 0 and all_similarities[best_idx - i] > similarity_threshold:
                    context_blocks.insert(0, blocks[best_idx - i])
                    context_text = blocks[best_idx - i].get_text() + "\n" + context_text
                else:
                    break
            
            # Add blocks below if they meet similarity threshold
            for i in range(1, max_lines_context + 1):
                if best_idx + i < len(blocks) and all_similarities[best_idx + i] > similarity_threshold:
                    context_blocks.append(blocks[best_idx + i])
                    context_text += "\n" + blocks[best_idx + i].get_text()
                else:
                    break
            
            print(f"\nBuilt context block with {len(context_blocks)} lines:")
            print(context_text)
            
            # Add these context blocks to our collection
            all_context_blocks.extend(context_blocks)
            
            print(f"Added {len(context_blocks)} blocks to overall context (total now: {len(all_context_blocks)})")
        else:
            print(f"Best match similarity ({all_similarities[best_idx]:.4f}) below threshold ({similarity_threshold}). No context created for this image.")
    
    # After processing all images, visualize all accumulated context blocks on the original image
    if all_context_blocks:
        # Get original image path (use the ID without any suffix)
        original_image_path = os.path.join("images", f"{args.id}.jpg")
        
        # If original image doesn't exist, use the first available image we found earlier
        if not os.path.exists(original_image_path):
            original_image_path = available_images[0]
            print(f"Original image not found at {original_image_path}, using {available_images[0]} instead")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "output_context")
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize all context blocks on the image
        try:
            # Call the function without the output_name parameter
            context_image = draw_blocks_on_image(original_image_path, all_context_blocks)
            
            # Define output filename
            output_name = os.path.join(output_dir, f"{args.id}_all_contexts.jpg")
            
            # Copy or move the generated image to the output directory
            if os.path.exists(context_image):
                shutil.copy(context_image, output_name)
                print(f"\nAll context blocks visualization saved to '{output_name}'")
        except Exception as e:
            print(f"Error visualizing combined context blocks: {e}")
    else:
        print("\nNo context blocks were found across any images.")

    return 0


if __name__ == "__main__":
    sys.exit(main())