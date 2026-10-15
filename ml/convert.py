"""
RoadSense AI - Pascal-VOC XML to YOLO Format Converter
Converts RDD2022 Pascal-VOC XML annotations to standard normalized YOLO format.
Applies canonical 4-class taxonomy, typo resolution, non-damage filtering,
and coordinate validity checks with verified round-trip reconstruction.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from core.logger import get_logger

logger = get_logger("roadsense.ml.convert")

# Strict class mapping
CLASS_MAP_TO_ID: Dict[str, int] = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}


def map_raw_class_name(raw_name: str) -> Optional[int]:
    """
    Resolve raw XML class name to canonical class ID (0..3).
    Returns None if the class is explicitly filtered out (e.g. crosswalk blur, manholes).
    """
    name = raw_name.strip()
    # Check aliases (e.g. D0w0 -> D00, D01 -> D00, D11 -> D10)
    name = config.RAW_CLASS_ALIASES.get(name, name)

    # Filter non-damage classes
    if name in config.FILTERED_CLASSES:
        return None

    # Map to canonical ID
    return CLASS_MAP_TO_ID.get(name, None)


def voc_box_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, img_width: float, img_height: float
) -> Tuple[float, float, float, float]:
    """
    Convert absolute Pascal-VOC bounding box to normalized YOLO format [xc, yc, w, h].
    All values are clamped to [0.0, 1.0].
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

    # Clip to image boundaries
    xmin = max(0.0, min(float(xmin), float(img_width)))
    xmax = max(0.0, min(float(xmax), float(img_width)))
    ymin = max(0.0, min(float(ymin), float(img_height)))
    ymax = max(0.0, min(float(ymax), float(img_height)))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")

    box_w = xmax - xmin
    box_h = ymax - ymin
    xc = xmin + (box_w / 2.0)
    yc = ymin + (box_h / 2.0)

    # Normalize to [0.0, 1.0]
    norm_xc = max(0.0, min(1.0, xc / float(img_width)))
    norm_yc = max(0.0, min(1.0, yc / float(img_height)))
    norm_w = max(0.0, min(1.0, box_w / float(img_width)))
    norm_h = max(0.0, min(1.0, box_h / float(img_height)))

    return norm_xc, norm_yc, norm_w, norm_h


def yolo_to_voc_box(
    norm_xc: float, norm_yc: float, norm_w: float, norm_h: float, img_width: float, img_height: float
) -> Tuple[float, float, float, float]:
    """
    Reconstruct absolute Pascal-VOC bounding box from normalized YOLO format.
    Used for round-trip verification tests.
    """
    box_w = norm_w * img_width
    box_h = norm_h * img_height
    xc = norm_xc * img_width
    yc = norm_yc * img_height

    xmin = xc - (box_w / 2.0)
    ymin = yc - (box_h / 2.0)
    xmax = xc + (box_w / 2.0)
    ymax = yc + (box_h / 2.0)

    return xmin, ymin, xmax, ymax


def convert_voc_xml_to_yolo(xml_path: Path, output_txt_path: Path) -> int:
    """
    Parse single Pascal-VOC XML file and write normalized YOLO darknet annotation file.
    Returns: Number of valid bounding boxes written.
    If the image has 0 objects (or only filtered objects), an empty file is created
    to signal a negative background image.
    """
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is None:
        raise ValueError(f"Missing <size> tag in {xml_path}")

    img_w = float(size_elem.find("width").text)
    img_h = float(size_elem.find("height").text)

    yolo_lines: List[str] = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue

        raw_name = name_elem.text.strip()
        cls_id = map_raw_class_name(raw_name)
        if cls_id is None:
            # Filtered or non-damage class
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            xc, yc, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        except ValueError as e:
            logger.warning(f"Skipping invalid box in {xml_path.name}: {e}")

    # Write lines (or empty file for negative sample)
    with open(output_txt_path, "w") as f:
        if yolo_lines:
            f.write("\n".join(yolo_lines) + "\n")

    return len(yolo_lines)


def batch_convert_xml_to_yolo(xml_dir: Path, output_labels_dir: Path) -> Dict[str, int]:
    """Batch convert directory of XMLs to YOLO labels."""
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    xml_files = list(xml_dir.glob("*.xml"))
    total_boxes = 0
    empty_count = 0

    for x_path in xml_files:
        txt_name = x_path.stem + ".txt"
        out_txt = output_labels_dir / txt_name
        n_boxes = convert_voc_xml_to_yolo(x_path, out_txt)
        total_boxes += n_boxes
        if n_boxes == 0:
            empty_count += 1

    return {
        "total_files_converted": len(xml_files),
        "total_boxes_written": total_boxes,
        "empty_background_files": empty_count,
    }
