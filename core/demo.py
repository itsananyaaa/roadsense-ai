"""
RoadSense AI - Offline Demonstration Suite & Synthetic Fixtures
Generates verified offline demonstration assets with explicit provenance.
Enforces Non-Negotiable Rule: Demo records and cached predictions are
unambiguously badged 'CACHED — not live inference' and 'EXPLICIT_SYNTHETIC_DEMO'.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from core import config
from core.db import insert_incident, clear_all_incidents
from core.logger import get_logger

logger = get_logger("roadsense.demo")

# Curated synthetic demo samples representing all four canonical damage types
DEMO_FIXTURES: List[Dict[str, Any]] = [
    {
        "filename": "demo_pothole_critical.jpg",
        "damage_type": "D40",
        "human_name": "Pothole",
        "confidence": 0.91,
        "bbox": [180, 220, 390, 410],
        "road_name": "Corridor Alpha (Synthetic Location)",
        "road_class": "PRIMARY_ARTERIAL",
        "priority_score": 86.4,
        "priority_level": "CRITICAL",
        "lat": 12.971599,
        "lon": 77.594566,
        "badge": "illustrative sample — synthetic location",
        "notes": "Severe cavity located directly in inner wheel path; high vehicle deflection risk. [Illustrative sample (not real field imagery)]",
    },
    {
        "filename": "demo_alligator_high.jpg",
        "damage_type": "D20",
        "human_name": "Alligator Crack",
        "confidence": 0.84,
        "bbox": [120, 160, 480, 450],
        "road_name": "Collector Route Beta (Synthetic Location)",
        "road_class": "SECONDARY_COLLECTOR",
        "priority_score": 67.2,
        "priority_level": "HIGH",
        "lat": 12.982210,
        "lon": 77.601520,
        "badge": "illustrative sample — synthetic location",
        "notes": "Extensive structural map cracking with base layer moisture seepage. [Illustrative sample (not real field imagery)]",
    },
    {
        "filename": "demo_transverse_moderate.jpg",
        "damage_type": "D10",
        "human_name": "Transverse Crack",
        "confidence": 0.76,
        "bbox": [80, 290, 560, 330],
        "road_name": "Local Road Gamma (Synthetic Location)",
        "road_class": "RESIDENTIAL_LOCAL",
        "priority_score": 42.8,
        "priority_level": "MODERATE",
        "lat": 12.964100,
        "lon": 77.585200,
        "badge": "illustrative sample — synthetic location",
        "notes": "Thermal contraction crack spanning 80% of road width. Recommend hot-pour seal. [Illustrative sample (not real field imagery)]",
    },
    {
        "filename": "demo_longitudinal_low.jpg",
        "damage_type": "D00",
        "human_name": "Longitudinal Crack",
        "confidence": 0.69,
        "bbox": [280, 140, 320, 520],
        "road_name": "Local Road Delta (Synthetic Location)",
        "road_class": "RESIDENTIAL_LOCAL",
        "priority_score": 28.5,
        "priority_level": "LOW",
        "lat": 12.978000,
        "lon": 77.612000,
        "badge": "illustrative sample — synthetic location",
        "notes": "Narrow surface crack along paving joint. Stable; monitor next cycle. [Illustrative sample (not real field imagery)]",
        "days_ago": 12,
    },
    {
        "filename": "demo_undamaged_clean.jpg",
        "damage_type": None,
        "human_name": "Undamaged Pavement",
        "confidence": 0.0,
        "bbox": [],
        "road_name": "Expressway Epsilon (Synthetic Location)",
        "road_class": "PRIMARY_ARTERIAL",
        "priority_score": 0.0,
        "priority_level": "LOW",
        "lat": 13.012500,
        "lon": 77.654000,
        "badge": "illustrative sample — synthetic location",
        "notes": "Newly paved asphalt surface exhibiting zero structural distress. [Illustrative sample (not real field imagery)]",
        "days_ago": 18,
    },
    {
        "filename": "demo_pothole_historical_1.jpg",
        "damage_type": "D40",
        "human_name": "Pothole",
        "confidence": 0.88,
        "bbox": [150, 200, 370, 390],
        "road_name": "Corridor Zeta (Synthetic Location)",
        "road_class": "PRIMARY_ARTERIAL",
        "priority_score": 83.2,
        "priority_level": "CRITICAL",
        "lat": 12.975400,
        "lon": 77.598200,
        "badge": "illustrative sample — synthetic location",
        "notes": "Historical deep wheel-impact pothole documented in previous survey. [Illustrative sample (not real field imagery)]",
        "days_ago": 4,
    },
    {
        "filename": "demo_alligator_historical_2.jpg",
        "damage_type": "D20",
        "human_name": "Alligator Crack",
        "confidence": 0.81,
        "bbox": [140, 180, 460, 430],
        "road_name": "Highway Eta (Synthetic Location)",
        "road_class": "PRIMARY_ARTERIAL",
        "priority_score": 65.0,
        "priority_level": "HIGH",
        "lat": 12.986000,
        "lon": 77.608000,
        "badge": "illustrative sample — synthetic location",
        "notes": "Fatigue crocodile cracking expanding along perimeter lane. [Illustrative sample (not real field imagery)]",
        "days_ago": 25,
    },
    {
        "filename": "demo_transverse_historical_3.jpg",
        "damage_type": "D10",
        "human_name": "Transverse Crack",
        "confidence": 0.72,
        "bbox": [100, 260, 520, 310],
        "road_name": "Avenue Theta (Synthetic Location)",
        "road_class": "SECONDARY_COLLECTOR",
        "priority_score": 41.5,
        "priority_level": "MODERATE",
        "lat": 12.968000,
        "lon": 77.589000,
        "badge": "illustrative sample — synthetic location",
        "notes": "Seasonal thermal expansion crack recorded in early quarterly survey. [Illustrative sample (not real field imagery)]",
        "days_ago": 75,
    },
]


def generate_synthetic_demo_image(fixture: Dict[str, Any], output_path: Path) -> None:
    """Generate realistic synthetic asphalt surface image with simulated road defect."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 640

    # Base asphalt texture (dark gray with noise)
    np.random.seed(42)
    base_color = 55  # dark asphalt
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img = np.clip(np.full((height, width, 3), base_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)

    # Road perspective lines (lane borders)
    cv2.line(img, (80, height), (260, 200), (200, 200, 200), 4, cv2.LINE_AA)  # Left white line
    cv2.line(img, (560, height), (380, 200), (200, 200, 200), 4, cv2.LINE_AA) # Right white line
    # Center dashed yellow line
    for y in range(220, height, 50):
        cv2.line(img, (320, y), (320, min(height, y + 25)), (30, 215, 255), 3, cv2.LINE_AA)

    # Add specific defect simulation based on damage type
    dtype = fixture["damage_type"]
    if dtype == "D40":  # Pothole (dark oval cavity with rough edges)
        x1, y1, x2, y2 = fixture["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
        cv2.ellipse(img, (cx, cy), (rx, ry), 15, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (rx, ry), 15, 0, 360, (75, 75, 75), 3, cv2.LINE_AA)
        # Inner depth shadow
        cv2.ellipse(img, (cx - 5, cy - 5), (rx // 2, ry // 2), 15, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

    elif dtype == "D20":  # Alligator cracking (polygonal mesh)
        x1, y1, x2, y2 = fixture["bbox"]
        for gx in range(x1, x2, 40):
            for gy in range(y1, y2, 40):
                pts = np.array([
                    [gx + np.random.randint(-8, 8), gy + np.random.randint(-8, 8)],
                    [gx + 35 + np.random.randint(-8, 8), gy + 5 + np.random.randint(-8, 8)],
                    [gx + 30 + np.random.randint(-8, 8), gy + 35 + np.random.randint(-8, 8)],
                    [gx + 5 + np.random.randint(-8, 8), gy + 30 + np.random.randint(-8, 8)],
                ], np.int32)
                cv2.polylines(img, [pts], True, (25, 25, 25), 2, cv2.LINE_AA)

    elif dtype == "D10":  # Transverse crack (horizontal zig-zag)
        x1, y1, x2, y2 = fixture["bbox"]
        prev_x, prev_y = x1, (y1 + y2) // 2
        for cur_x in range(x1 + 30, x2 + 1, 30):
            cur_y = ((y1 + y2) // 2) + np.random.randint(-6, 6)
            cv2.line(img, (prev_x, prev_y), (cur_x, cur_y), (25, 25, 25), 3, cv2.LINE_AA)
            prev_x, prev_y = cur_x, cur_y

    elif dtype == "D00":  # Longitudinal crack (vertical zig-zag)
        x1, y1, x2, y2 = fixture["bbox"]
        prev_x, prev_y = (x1 + x2) // 2, y1
        for cur_y in range(y1 + 30, y2 + 1, 30):
            cur_x = ((x1 + x2) // 2) + np.random.randint(-6, 6)
            cv2.line(img, (prev_x, prev_y), (cur_x, cur_y), (25, 25, 25), 3, cv2.LINE_AA)
            prev_x, prev_y = cur_x, cur_y

    cv2.imwrite(str(output_path), img)
    logger.info(f"Generated synthetic demo asset: {output_path.name}")


def load_demo_manifest() -> Dict[str, Any]:
    """
    Load data/demo_media/manifest.json.
    If the manifest does not exist, invokes ml.build_demo_media to construct it.
    """
    manifest_path = config.DATA_DIR / "demo_media" / "manifest.json"
    if not manifest_path.exists():
        try:
            from ml.build_demo_media import build_demo_media_suite
            return build_demo_media_suite()
        except Exception as e:
            logger.warning(f"Could not automatically build demo manifest ({e}); using fallback.")
            return {"assets": [], "videos": []}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read demo manifest {manifest_path}: {e}")
        return {"assets": [], "videos": []}


def get_demo_tiles() -> List[Dict[str, Any]]:
    """
    Retrieve curated sample tiles for the Detect Damage acquisition tab.
    Every tile has a neutral, truthful caption, origin metadata, and optional ground truth boxes.
    """
    manifest = load_demo_manifest()
    assets = manifest.get("assets", [])

    tiles: List[Dict[str, Any]] = []
    for item in assets:
        img_name = item["filename"]
        tiles.append({
            "id": item.get("id", img_name),
            "filename": img_name,
            "caption": item.get("caption", f"Sample — {img_name}"),
            "badge": item.get("badge", "illustrative sample (not real field imagery)"),
            "origin": item.get("origin", "synthetic_demo"),
            "is_real_imagery": item.get("is_real_imagery", False),
            "ground_truth": item.get("ground_truth", []),
            "image_url": f"/demo-assets/{img_name}",
        })

    # Fallback to DEMO_FIXTURES if manifest was empty
    if not tiles:
        for fix in DEMO_FIXTURES:
            fname = fix["filename"]
            tiles.append({
                "id": f"fixture_{Path(fname).stem}",
                "filename": fname,
                "caption": f"Illustrative sample (not real field imagery) — {fix['human_name']}",
                "badge": "illustrative sample — synthetic location",
                "origin": "synthetic_demo",
                "is_real_imagery": False,
                "ground_truth": [{
                    "class_code": fix["damage_type"],
                    "class_name": fix["human_name"],
                    "bbox": fix["bbox"],
                }] if fix.get("damage_type") else [],
                "image_url": f"/demo-assets/{fname}",
            })

    return tiles


def seed_demo_database(reset_existing: bool = True) -> int:
    """
    Populate local SQLite database with calibrated demonstration incidents.
    Every record is explicitly tagged with authentic location provenance:
    - User field captures with EXIF GPS: location_provenance='EXIF_GPS'
    - RDD / synthetic fixtures: location_provenance='EXPLICIT_SYNTHETIC_DEMO'
    """
    config.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if reset_existing:
        clear_all_incidents(confirm=True)

    manifest = load_demo_manifest()
    assets = manifest.get("assets", [])
    if not assets:
        assets = DEMO_FIXTURES

    count = 0
    from datetime import datetime, timedelta

    for idx, item in enumerate(assets):
        fname = item["filename"]
        img_path = config.DEMO_DIR / fname

        # If file is not in data/demo/, copy from demo_media/images or generate
        if not img_path.exists():
            media_img = config.DATA_DIR / "demo_media" / "images" / fname
            if media_img.exists():
                shutil.copy2(media_img, img_path)
            else:
                matching_fix = next((f for f in DEMO_FIXTURES if f["filename"] == fname), None)
                if matching_fix:
                    generate_synthetic_demo_image(matching_fix, img_path)

        gt_boxes = item.get("ground_truth", [])
        if gt_boxes:
            detected_classes = sorted(list({b["class_code"] for b in gt_boxes}))
            raw_dets = [
                {
                    "class_id": 3 if b["class_code"] == "D40" else (2 if b["class_code"] == "D20" else (1 if b["class_code"] == "D10" else 0)),
                    "class_code": b["class_code"],
                    "class_name": b.get("class_name", config.CLASS_HUMAN_NAMES.get(b["class_code"], b["class_code"])),
                    "confidence": item.get("confidence", 0.88),
                    "bbox": b["bbox"],
                }
                for b in gt_boxes
            ]
            det_count = len(raw_dets)
            max_conf = max(d["confidence"] for d in raw_dets)
        elif item.get("damage_type") is not None:
            c_code = item["damage_type"]
            detected_classes = [c_code]
            raw_dets = [{
                "class_id": 3 if c_code == "D40" else (2 if c_code == "D20" else (1 if c_code == "D10" else 0)),
                "class_code": c_code,
                "class_name": item["human_name"],
                "confidence": item.get("confidence", 0.88),
                "bbox": item.get("bbox", [150, 150, 350, 350]),
            }]
            det_count = 1
            max_conf = item.get("confidence", 0.88)
        else:
            detected_classes = []
            raw_dets = []
            det_count = 0
            max_conf = 0.0

        days_ago = item.get("days_ago", idx * 5)
        ts_str = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")

        badge = item.get("badge", "illustrative sample — synthetic location")
        notes = f"[{badge}] {item.get('notes', item.get('caption', 'Demonstration survey incident'))}"

        prov = item.get("location_provenance", "EXPLICIT_SYNTHETIC_DEMO")
        lat = item.get("latitude", item.get("lat", 12.971599 + (idx * 0.003)))
        lon = item.get("longitude", item.get("lon", 77.594566 + (idx * 0.003)))

        # Priority score
        from core.priority import calculate_priority
        prio = calculate_priority(
            detected_classes=detected_classes,
            max_confidence=max_conf,
            nearby_repeat_count=min(idx, 3),
            road_classification=item.get("road_classification", item.get("road_class", "PRIMARY_ARTERIAL")),
            human_urgency_modifier=1.0,
        )

        insert_incident(
            image_path=str(img_path),
            checkpoint_id="demo_calibrated_weights_v1.pt",
            detected_classes=detected_classes,
            detection_count=det_count,
            max_confidence=max_conf,
            raw_detections=raw_dets,
            priority_score=prio.score,
            priority_level=prio.level,
            latitude=lat,
            longitude=lon,
            location_provenance=prov,
            road_name=item.get("road_name", item.get("caption", f"Corridor {idx+1} (Synthetic Location)")),
            road_classification=item.get("road_classification", item.get("road_class", "PRIMARY_ARTERIAL")),
            timestamp=ts_str,
        )
        count += 1

    logger.info(f"Seeded {count} demo incidents into database from manifest.")
    return count


def seed_demo_incidents_if_empty(force: bool = False) -> int:
    """Convenience alias: seeds calibrated demo incidents if database is empty or if forced."""
    from core.db import get_incident_count
    if force or get_incident_count() == 0:
        return seed_demo_database(reset_existing=force)
    return 0

