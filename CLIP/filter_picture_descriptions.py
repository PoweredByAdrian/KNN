import os
import json
import re
import argparse
from pero_ocr.core.layout import PageLayout
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

def load_json_annotation(json_filepath: str) -> List[Dict]:
    """Load JSON annotation file and extract text regions"""
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_regions = []
        
        # Find all annotations with type "rectanglelabels"
        for annotation in data.get("annotations", []):
            for result in annotation.get("result", []):
                if (result.get("type") == "rectanglelabels" and 
                    "Popis u obrázku" in result.get("value", {}).get("rectanglelabels", [])):
                    
                    value = result.get("value", {})
                    original_width = result.get("original_width", 0)
                    original_height = result.get("original_height", 0)
                    
                    # Convert percentage to absolute coordinates
                    x = value.get("x", 0) * original_width / 100
                    y = value.get("y", 0) * original_height / 100
                    width = value.get("width", 0) * original_width / 100
                    height = value.get("height", 0) * original_height / 100
                    
                    text_regions.append({
                        "id": result.get("id"),
                        "bbox": (x, y, x + width, y + height),
                        "original_width": original_width,
                        "original_height": original_height
                    })
        
        return text_regions
    
    except Exception as e:
        print(f"ERROR: Error loading JSON file {json_filepath}: {e}")
        return []

def load_xml_text_regions(xml_filepath: str) -> List[Dict]:
    """Load XML file and extract text regions with their coordinates"""
    try:
        page_layout = PageLayout(file=xml_filepath)
        text_regions = []
        
        # Extract text regions from the PageLayout
        for region in page_layout.regions:
            if hasattr(region, 'polygon'):
                # Create bounding box from polygon points
                xs = [pt[0] for pt in region.polygon]
                ys = [pt[1] for pt in region.polygon]
                
                # Get text from all lines in the region
                text = ""
                for line in region.lines:
                    text += line.transcription + " "
                
                text_regions.append({
                    "id": region.id,
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "polygon": region.polygon,
                    "text": text.strip()
                })
        
        return text_regions
    
    except Exception as e:
        print(f"ERROR: Error loading XML file {xml_filepath}: {e}")
        return []

def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union of two bounding boxes"""
    # Each box is (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def find_matching_regions(json_regions: List[Dict], xml_regions: List[Dict], 
                         iou_threshold: float = 0.3) -> List[Tuple[Dict, Dict, float]]:
    """Find matches between JSON and XML regions based on IoU"""
    matches = []
    
    for json_region in json_regions:
        best_match = None
        best_iou = 0
        
        for xml_region in xml_regions:
            iou = calculate_iou(json_region["bbox"], xml_region["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = xml_region
        
        if best_iou >= iou_threshold:
            matches.append((json_region, best_match, best_iou))
    
    return matches

def remove_matched_regions_from_xml(matches: List[Tuple[Dict, Dict, float]], 
                                       xml_filepath: str, output_filepath: Optional[str] = None) -> int:
        """Remove matched text regions from XML file and save to a new file
        
        Args:
            matches: List of tuples containing (json_region, xml_region, iou)
            xml_filepath: Path to the original XML file
            output_filepath: Path to save the modified XML file. If None, will overwrite the original file.
            
        Returns:
            Number of regions removed
        """
        try:
            # Parse the XML file
            tree = ET.parse(xml_filepath)
            root = tree.getroot()
            
            # Store IDs of regions to remove
            region_ids_to_remove = [match[1]["id"] for match in matches]
            removed_count = 0
            
            # Find all parent elements that might contain TextRegion elements
            for parent in root.findall('.//*'):
                regions_to_remove = []
                for child in list(parent):
                    # Check if it's a TextRegion element with matching ID
                    if child.tag.endswith('TextRegion'):
                        region_id = child.get('id')
                        if region_id in region_ids_to_remove:
                            regions_to_remove.append(child)
                
                # Remove the identified TextRegion elements
                for region in regions_to_remove:
                    parent.remove(region)
                    removed_count += 1
            
            # If no output path is provided, overwrite the original file
            if output_filepath is None:
                output_filepath = xml_filepath
            
            # Save the modified XML
            tree.write(output_filepath, encoding='utf-8', xml_declaration=True)
            print(f"Removed {removed_count} matching regions from {xml_filepath}")
            
            return removed_count
            
        except Exception as e:
            print(f"ERROR: Error removing regions from XML file {xml_filepath}: {e}")
            return 0

def main():
    print("Starting text region comparison...")
    parser = argparse.ArgumentParser(description="Compare text regions between JSON annotations and XML OCR data")
    parser.add_argument("json_dir", help="Directory containing JSON annotation files")
    parser.add_argument("xml_dir", help="Directory containing XML OCR files")
    parser.add_argument("--threshold", type=float, default=0.00005, help="IoU threshold for matching")
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.json_dir):
        print(f"Error: JSON directory '{args.json_dir}' does not exist")
        return
    
    if not os.path.isdir(args.xml_dir):
        print(f"Error: XML directory '{args.xml_dir}' does not exist")
        return
    
    # Get file lists
    json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
    xml_files = [f for f in os.listdir(args.xml_dir) if f.endswith('.xml')]
    
    print(f"Found {len(json_files)} JSON files and {len(xml_files)} XML files")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "filtered_texts")
    os.makedirs(output_dir, exist_ok=True)
    
    total_matches = 0
    total_json_regions = 0
    processed_files = 0
    matches_removed_count = 0
    
    # Process only JSON files with matching XML files
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        matching_xml = next((x for x in xml_files if os.path.splitext(x)[0] == base_name), None)
        
        if not matching_xml:
            print(f"No matching XML file found for {json_file}")
            continue
            
        json_path = os.path.join(args.json_dir, json_file)
        xml_path = os.path.join(args.xml_dir, matching_xml)
        
        print(f"\nProcessing file pair: {json_file} and {matching_xml}")
        
        # Load data
        json_regions = load_json_annotation(json_path)
        
        # Even if no "Popis u obrázku" regions found, still copy the XML
        if not json_regions:
            print(f"  No 'Popis u obrázku' regions found in {json_file}")
            print(f"  Copying XML file without filtering")
            try:
                tree = ET.parse(xml_path)
                output_path = os.path.join(output_dir, f"filtered_{matching_xml}")
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
                print(f"  Copied {matching_xml} to {output_path}")
                processed_files += 1
            except Exception as e:
                print(f"  Error copying XML file {matching_xml}: {e}")
            continue
            
        # Load XML regions
        xml_regions = load_xml_text_regions(xml_path)
        
        print(f"  Loaded {len(json_regions)} text regions from JSON file")
        print(f"  Loaded {len(xml_regions)} text regions from XML file")
        
        # Find matches
        matches = find_matching_regions(json_regions, xml_regions, args.threshold)
        
        # Print results
        if matches:
            print(f"  Found {len(matches)} matching text regions:")
            if json_regions:
                match_percentage = len(matches) / len(json_regions) * 100
                print(f"      Match percentage: {match_percentage:.2f}% ({len(matches)}/{len(json_regions)} regions matched)")
            for i, (json_region, xml_region, iou) in enumerate(matches):
                print(f"    Match {i+1}:")
                print(f"      JSON ID: {json_region['id']}")
                print(f"      XML ID: {xml_region['id']}")
                print(f"      IoU: {iou:.4f}")
                print(f"      Text: {xml_region['text'][:100]}..." if 'text' in xml_region else "      Text: N/A")
            
            total_matches += len(matches)
            total_json_regions += len(json_regions)
            
            # Save filtered XML
            output_path = os.path.join(output_dir, f"filtered_{matching_xml}")
            removed_count = remove_matched_regions_from_xml(matches, xml_path, output_filepath=output_path)
            print(f"  Filtered XML saved to {output_path}")
            matches_removed_count += removed_count
        else:
            print("  No matching text regions found")
            print(f"  Copying XML file without filtering")
            try:
                tree = ET.parse(xml_path)
                output_path = os.path.join(output_dir, f"filtered_{matching_xml}")
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
                print(f"  Copied {matching_xml} to {output_path}")
            except Exception as e:
                print(f"  Error copying XML file {matching_xml}: {e}")
        
        processed_files += 1
    
    # Print summary
    if total_json_regions > 0:
        total_match_percentage = total_matches / total_json_regions * 100
        print(f"\nSummary: Processed {processed_files} file pairs, found {total_matches} total matching regions")
        print(f"Total match percentage: {total_match_percentage:.2f}% ({total_matches}/{total_json_regions} regions matched)")
        print(f"Total regions removed from XML: {matches_removed_count}")
    else:
        print(f"\nSummary: Processed {processed_files} file pairs, no matching regions found")
    
    print("Finished processing all files.")

if __name__ == "__main__":
    main()