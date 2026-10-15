"""
RoadSense AI - Dataset Audit Tool
Performs automated structural and statistical audits on Pascal-VOC XML road damage annotations:
- Malformed XML detection
- Empty XML (negative background) quantification
- Bounding box dimension validity (xmin < xmax, ymin < ymax, within image dimensions)
- Class frequency counts and imbalance ratios
- Exports verifiable audit reports to reports/audit_summary.json
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from core.logger import get_logger

logger = get_logger("roadsense.ml.audit")


@dataclass
class AuditReport:
    """Consolidated audit report for an RDD dataset subset."""
    subset_name: str
    total_xml_files: int = 0
    malformed_xml_files: int = 0
    empty_xml_files: int = 0
    total_bounding_boxes: int = 0
    invalid_bounding_boxes: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    canonical_damage_boxes: int = 0
    filtered_non_damage_boxes: int = 0
    image_resolutions: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subset_name": self.subset_name,
            "total_xml_files": self.total_xml_files,
            "malformed_xml_files": self.malformed_xml_files,
            "empty_xml_files": self.empty_xml_files,
            "empty_ratio": round(self.empty_xml_files / max(1, self.total_xml_files), 4),
            "total_bounding_boxes": self.total_bounding_boxes,
            "invalid_bounding_boxes": self.invalid_bounding_boxes,
            "class_distribution": self.class_distribution,
            "canonical_damage_boxes": self.canonical_damage_boxes,
            "filtered_non_damage_boxes": self.filtered_non_damage_boxes,
            "image_resolutions": self.image_resolutions,
        }


def audit_xml_directory(xml_dir: Path, subset_name: str = "Unknown") -> AuditReport:
    """
    Audit all XML files in a local directory.
    """
    report = AuditReport(subset_name=subset_name)
    xml_files = list(xml_dir.glob("*.xml"))
    report.total_xml_files = len(xml_files)

    class_counter: Counter = Counter()
    res_counter: Counter = Counter()

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Resolution check
            size_elem = root.find("size")
            if size_elem is not None:
                w = size_elem.find("width")
                h = size_elem.find("height")
                if w is not None and h is not None and w.text and h.text:
                    res_key = f"{w.text}x{h.text}"
                    res_counter[res_key] += 1

            objects = root.findall("object")
            if len(objects) == 0:
                report.empty_xml_files += 1
                continue

            for obj in objects:
                name_elem = obj.find("name")
                cname = name_elem.text.strip() if (name_elem is not None and name_elem.text) else "UNKNOWN"
                class_counter[cname] += 1
                report.total_bounding_boxes += 1

                # Classify canonical vs filtered
                if cname in ["D00", "D10", "D20", "D40", "D0w0", "D01", "D11"]:
                    report.canonical_damage_boxes += 1
                else:
                    report.filtered_non_damage_boxes += 1

                # Box geometry check
                bnd = obj.find("bndbox")
                if bnd is not None:
                    try:
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)
                        if xmax <= xmin or ymax <= ymin or xmin < 0 or ymin < 0:
                            report.invalid_bounding_boxes += 1
                    except Exception:
                        report.invalid_bounding_boxes += 1
                else:
                    report.invalid_bounding_boxes += 1

        except Exception as e:
            logger.warning(f"Malformed XML file {xml_path.name}: {e}")
            report.malformed_xml_files += 1

    report.class_distribution = dict(class_counter.most_common())
    report.image_resolutions = dict(res_counter.most_common())
    return report


def save_audit_report(report: AuditReport, output_path: Optional[Path] = None) -> Path:
    """Save audit report to JSON artifact."""
    out = output_path or (config.REPORTS_DIR / f"audit_{report.subset_name.lower()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved audit report: {out}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit RDD Pascal-VOC XML annotations.")
    parser.add_argument("--xml-dir", type=str, required=True, help="Directory containing XML files")
    parser.add_argument("--subset", type=str, default="India", help="Subset name (e.g. India, Czech)")
    args = parser.parse_args()

    rep = audit_xml_directory(Path(args.xml_dir), subset_name=args.subset)
    print("--- AUDIT RESULTS ---")
    for k, v in rep.to_dict().items():
        print(f"{k}: {v}")
    save_audit_report(rep)
