import os
from typing import List, Tuple, Union
from dataclasses import dataclass
from pero_ocr.core.layout import PageLayout
from PIL import Image,ImageDraw

@dataclass
class OCRLine:
    id: str
    text: str
    polygon: List[Tuple[int, int]]  # list of (x, y) tuples


@dataclass
class TextBlock:
    id: int
    lines: List[OCRLine]

    def get_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def bounding_box(self) -> Tuple[int, int, int, int]:
        # Compute overall bounding box of all line polygons
        all_points = [pt for line in self.lines for pt in line.polygon]
        xs, ys = zip(*all_points)
        return min(xs), min(ys), max(xs), max(ys)


class OCRDocument:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.lines = self._parse_lines()
        self.blocks = None

    def _parse_lines(self) -> List[OCRLine]:
        page_layout = PageLayout(file=self.xml_path)
        ocr_lines = []
        for line in page_layout.lines_iterator():
            polygon = [(pt[0], pt[1]) for pt in line.polygon]  # Convert to (x, y)
            ocr_lines.append(OCRLine(id=line.id, text=line.transcription, polygon=polygon))
        return ocr_lines

    def generate_blocks(
        self,
        lines_per_block: int = 15,
        overlap: int = 10,
        max_x_shift: int = 300,
        max_y_gap: int = 300
    ) -> List[TextBlock]:
        blocks = []
        i = 0
        block_id = 0

        while i < len(self.lines):
            block_lines = []
            start_line = self.lines[i]
            block_lines.append(start_line)

            prev_line = start_line

            j = i + 1
            while j < len(self.lines) and len(block_lines) < lines_per_block:
                line = self.lines[j]

                prev_x = min(p[0] for p in prev_line.polygon)
                prev_y = min(p[1] for p in prev_line.polygon)

                x = min(p[0] for p in line.polygon)
                y = min(p[1] for p in line.polygon)

                x_diff = abs(x - prev_x)
                y_diff = abs(y - prev_y)

                if x_diff > max_x_shift or y_diff > max_y_gap:
                    break

                block_lines.append(line)
                prev_line = line  # 🔄 update for next comparison
                j += 1

            blocks.append(TextBlock(id=block_id, lines=block_lines))
            block_id += 1

            i = j if j > i + (lines_per_block - overlap) else i + (lines_per_block - overlap)

        self.blocks = blocks
        return blocks


def draw_blocks_on_image(
    image_path: str,
    blocks: List[TextBlock],
    output_path: Union[str, None] = None,
    draw_lines: bool = True
) -> str:
    """
    Draws multiple text blocks on an image and saves the result.

    Args:
        image_path: Path to the original image.
        blocks: List of TextBlock objects to draw.
        output_path: Path to save the output image. If None, auto-generates.
        draw_lines: Whether to draw each line's polygon (in red).

    Returns:
        The path to the saved image.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for block in blocks:
        # Optionally draw each line's polygon
        if draw_lines:
            for line in block.lines:
                draw.polygon(line.polygon, outline="red", width=2)

        # Always draw the block's bounding box (in blue)
        bbox = block.bounding_box()
        draw.rectangle(bbox, outline="blue", width=2)

    if output_path is None:
        block_ids = "_".join([str(block.id) for block in blocks])
        filename = f"blocks_{block_ids}_bbox.png"
        output_path = os.path.join(os.getcwd(), filename)

    image.save(output_path)
    return output_path