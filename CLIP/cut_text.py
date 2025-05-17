"""
OCR Text Processing Module

This module provides classes and functions for:
1. Parsing XML files containing OCR text data
2. Organizing OCR lines into logical text blocks
3. Visualizing text blocks on the original images
"""

import os
from typing import List, Tuple, Union
from dataclasses import dataclass
from pero_ocr.core.layout import PageLayout
from PIL import Image, ImageDraw


# ------------------- Data Classes for OCR Text Elements -------------------

@dataclass
class OCRLine:
    """Represents a single line of OCR text with its polygon coordinates."""
    id: str
    text: str
    polygon: List[Tuple[int, int]]  # list of (x, y) tuples


@dataclass
class TextBlock:
    """Represents a block of text composed of multiple OCR lines."""
    id: int
    lines: List[OCRLine]

    def get_text(self) -> str:
        """Returns all text in this block joined by newlines."""
        return "\n".join(line.text for line in self.lines)

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Calculates the overall bounding box that contains all lines in this block.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        # Flatten all points from all line polygons
        all_points = [pt for line in self.lines for pt in line.polygon]
        
        # Extract x and y coordinates
        xs, ys = zip(*all_points)
        
        # Return min/max coordinates as bounding box
        return min(xs), min(ys), max(xs), max(ys)


# ------------------- Document Class for Processing OCR Data -------------------

class OCRDocument:
    """
    Handles OCR document processing from XML files.
    
    This class parses XML files with OCR data and creates logical
    text blocks from the contained lines.
    """
    
    def __init__(self, xml_path: str):
        """
        Initialize an OCR document from an XML file.
        
        Args:
            xml_path: Path to the XML file containing OCR data
        """
        self.xml_path = xml_path
        self.lines = self._parse_lines()
        self.blocks = None

    def _parse_lines(self) -> List[OCRLine]:
        """
        Parse the XML file to extract OCR lines.
        
        Returns:
            List of OCRLine objects
        """
        page_layout = PageLayout(file=self.xml_path)
        ocr_lines = []
        
        for line in page_layout.lines_iterator():
            # Convert polygon points to (x, y) format
            polygon = [(pt[0], pt[1]) for pt in line.polygon]
            
            # Create OCRLine object
            ocr_lines.append(OCRLine(
                id=line.id, 
                text=line.transcription, 
                polygon=polygon
            ))
            
        return ocr_lines

    def generate_blocks(
        self,
        lines_per_block: int = 15,
        overlap: int = 10,
        max_x_shift: int = 300,
        max_y_gap: int = 300
    ) -> List[TextBlock]:
        """
        Group OCR lines into logical text blocks based on spatial relationship.
        
        Args:
            lines_per_block: Maximum number of lines per text block
            overlap: Number of lines that can overlap between blocks
            max_x_shift: Maximum horizontal distance (pixels) to include lines in same block
            max_y_gap: Maximum vertical distance (pixels) to include lines in same block
            
        Returns:
            List of TextBlock objects
        """
        blocks = []
        i = 0
        block_id = 0

        # Process all lines
        while i < len(self.lines):
            block_lines = []
            
            # Start a new block with the current line
            start_line = self.lines[i]
            block_lines.append(start_line)
            prev_line = start_line

            # Try to add more lines to this block
            j = i + 1
            while j < len(self.lines) and len(block_lines) < lines_per_block:
                line = self.lines[j]

                # Get top-left coordinates of previous and current line
                prev_x = min(p[0] for p in prev_line.polygon)
                prev_y = min(p[1] for p in prev_line.polygon)
                x = min(p[0] for p in line.polygon)
                y = min(p[1] for p in line.polygon)

                # Calculate positioning differences
                x_diff = abs(x - prev_x)
                y_diff = abs(y - prev_y)

                # If too far away, consider it part of a different block
                if x_diff > max_x_shift or y_diff > max_y_gap:
                    break

                # Add line to the current block
                block_lines.append(line)
                prev_line = line  # Update for next comparison
                j += 1

            # Create a new block with the collected lines
            blocks.append(TextBlock(id=block_id, lines=block_lines))
            block_id += 1

            # Move index forward, considering overlap
            # If we didn't add any lines (j = i+1), move to next line
            # Otherwise, move by lines_per_block - overlap
            i = j if j > i + (lines_per_block - overlap) else i + (lines_per_block - overlap)

        # Store blocks for later use and return
        self.blocks = blocks
        return blocks


# ------------------- Visualization Functions -------------------

def draw_blocks_on_image(
    image_path: str,
    blocks: List[TextBlock],
    output_path: Union[str, None] = None,
    draw_lines: bool = True
) -> str:
    """
    Visualizes text blocks on an image by drawing polygons and bounding boxes.

    Args:
        image_path: Path to the original image
        blocks: List of TextBlock objects to visualize
        output_path: Path to save the output image (auto-generated if None)
        draw_lines: Whether to draw individual line polygons in red

    Returns:
        Path to the saved output image
    """
    # Open and prepare the image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw each text block
    for block in blocks:
        # Optionally draw individual line polygons in red
        if draw_lines:
            for line in block.lines:
                draw.polygon(line.polygon, outline="red", width=2)

        # Always draw the block's bounding box in blue
        bbox = block.bounding_box()
        draw.rectangle(bbox, outline="blue", width=2)

    # Generate output path if not provided
    if output_path is None:
        # Extract the base image name without extension
        base_image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create filename with block count
        filename = f"{base_image_name}_blocks_{len(blocks)}.png"
        output_path = os.path.join(os.getcwd(), filename)

    # Save and return
    image.save(output_path)
    return output_path