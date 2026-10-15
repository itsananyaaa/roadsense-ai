"""
RoadSense AI - Unit Tests for VOC to YOLO Annotation Converter
Tests coordinate conversions, round-trip fidelity, class remapping, and negative sampling.
"""

from pathlib import Path
import pytest

from ml.convert import (
    voc_box_to_yolo,
    yolo_to_voc_box,
    map_raw_class_name,
    convert_voc_xml_to_yolo,
)


def test_voc_box_to_yolo_and_round_trip():
    """Verify coordinate normalization and exact pixel round-trip reconstruction."""
    img_w, img_h = 720, 720
    xmin, ymin, xmax, ymax = 120.0, 240.0, 360.0, 480.0

    xc, yc, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

    # Expected values
    assert xc == pytest.approx(240.0 / 720.0, 1e-5)
    assert yc == pytest.approx(360.0 / 720.0, 1e-5)
    assert w == pytest.approx(240.0 / 720.0, 1e-5)
    assert h == pytest.approx(240.0 / 720.0, 1e-5)

    # Reconstruct
    rx1, ry1, rx2, ry2 = yolo_to_voc_box(xc, yc, w, h, img_w, img_h)
    assert rx1 == pytest.approx(xmin, abs=0.1)
    assert ry1 == pytest.approx(ymin, abs=0.1)
    assert rx2 == pytest.approx(xmax, abs=0.1)
    assert ry2 == pytest.approx(ymax, abs=0.1)


def test_class_mapping_rules():
    """Verify canonical IDs, typo fixes, subtype merges, and non-damage filtering."""
    # Canonical classes
    assert map_raw_class_name("D00") == 0
    assert map_raw_class_name("D10") == 1
    assert map_raw_class_name("D20") == 2
    assert map_raw_class_name("D40") == 3

    # Ground-truth typos and sub-types
    assert map_raw_class_name("D0w0") == 0
    assert map_raw_class_name("D01") == 0
    assert map_raw_class_name("D11") == 1

    # Filtered non-damage categories
    assert map_raw_class_name("D43") is None
    assert map_raw_class_name("D44") is None
    assert map_raw_class_name("D50") is None
    assert map_raw_class_name("Repair") is None


def test_degenerate_box_handling():
    """Inverted coordinates must raise ValueError."""
    with pytest.raises(ValueError):
        voc_box_to_yolo(xmin=200, ymin=100, xmax=100, ymax=200, img_width=640, img_height=640)


def test_xml_conversion_with_negative_sample(tmp_path: Path):
    """XML with only filtered non-damage classes should create an empty negative text file."""
    xml_content = """<annotation>
        <size>
            <width>640</width>
            <height>640</height>
            <depth>3</depth>
        </size>
        <object>
            <name>D44</name>
            <bndbox>
                <xmin>50</xmin>
                <ymin>50</ymin>
                <xmax>200</xmax>
                <ymax>200</ymax>
            </bndbox>
        </object>
    </annotation>"""
    xml_file = tmp_path / "test_negative.xml"
    txt_file = tmp_path / "test_negative.txt"
    xml_file.write_text(xml_content)

    boxes_written = convert_voc_xml_to_yolo(xml_file, txt_file)
    assert boxes_written == 0
    assert txt_file.exists()
    assert txt_file.read_text().strip() == ""
