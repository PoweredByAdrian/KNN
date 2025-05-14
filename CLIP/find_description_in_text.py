from typing import Optional, Tuple, List
from PIL import Image
import torch
import clip
from cut_text import OCRDocument,TextBlock,draw_block_on_image
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
    image_path="exported_images\\ffbfac5d-e23e-11e6-932d-001999480be2.jpg",
    xml_path="texts\\ffbfac5d-e23e-11e6-932d-001999480be2.xml",
    model=model,
    preprocess=preprocess,
    device=device
)

best_block, prob, all_probs, all_similarities,doc = result
print(f"Matched block with {prob:.2%} probability:\n{best_block.get_text()}")
print("\nAll probabilities:")
for i, p in enumerate(all_probs):
    print(f"  Block {i}: {p:.4f}")
print("\nAll cosine similarities:")
for i, s in enumerate(all_similarities):
    print(f"  Block {i}: {s:.4f}")

draw_block_on_image("images\\ffbfac5d-e23e-11e6-932d-001999480be2.jpg",doc.blocks[best_block.id],"out.png",draw_lines=False)

