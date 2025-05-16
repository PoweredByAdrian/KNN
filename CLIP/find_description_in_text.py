from typing import Optional, Tuple, List
from PIL import Image
import torch
import clip
from cut_text import OCRDocument,TextBlock,draw_blocks_on_image
import torch.nn.functional as F

from typing import Optional, Tuple, List
from PIL import Image
import torch
import torch.nn.functional as F
import clip  # Assuming already imported

# Also assuming OCRDocument and TextBlock are defined elsewhere

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
    # 1. Load OCR blocks
    doc = OCRDocument(xml_path)
    blocks = doc.generate_blocks(lines_per_block=1, overlap=0)
    if not blocks:
        return None

    # 2. Preprocess image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

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




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

result = find_best_matching_block(
    image_path="exported_images/7f3229b2-71c7-11e1-a313-005056a60003.jpg",
    xml_path="filtered_texts/filtered_7f3229b2-71c7-11e1-a313-005056a60003.xml",
    model=model,
    preprocess=preprocess,
    device=device
)

best_block, prob, all_probs, all_similarities,doc = result
print(f"Matched block with {prob:.2%} probability:\n{best_block.get_text()}")
print("\nAll probabilities:")
for i, p in enumerate(all_probs):
    print(f"  Block {i}: {p:.4f} Text: {doc.blocks[i].get_text()}")
print("\nAll cosine similarities:")
for i, s in enumerate(all_similarities):
    print(f"  Block {i}: {s:.4f}")


# Build contextual block around the best matching text
print("\nBuilding context block around best match...")

# Parameters for context building - adjust these as needed
similarity_threshold = 0.25  # Minimum similarity to include in context
max_lines_context = 3      # Maximum lines to check above and below

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
    print(context_blocks)
    
    # Visualize the context block on the image
    try:
        context_image = draw_blocks_on_image("images/7f3229b2-71c7-11e1-a313-005056a60003.jpg", context_blocks)
        
        print(f"Context block visualization saved to '{context_image}'")
    except Exception as e:
        print(f"Error visualizing context block: {e}")
else:
    print(f"Best match similarity ({all_similarities[best_idx]:.4f}) below threshold ({similarity_threshold}). No context created.")