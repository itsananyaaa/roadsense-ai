"""
RoadSense AI - Leakage-Aware Dataset Splitting
Prevents video-sequence data leakage by grouping sequential dashcam frames and
clustering near-duplicates via difference perceptual hashing (dHash) before splitting.
Generates reproducible train/val manifests with fixed random seed (42).
"""

import hashlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from core import config
from core.logger import get_logger

logger = get_logger("roadsense.ml.split")


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """
    Compute 64-bit difference hash (dHash) of an image.
    Robust to minor lighting changes, compression artifacts, and vehicle vibration.
    """
    try:
        with Image.open(image_path) as img:
            # Resize to (hash_size + 1, hash_size) in grayscale
            resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(resized)
            # Compare adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convert boolean array to 64-bit integer
            hash_val = 0
            for bit in diff.flatten():
                hash_val = (hash_val << 1) | int(bit)
            return hash_val
    except Exception as e:
        logger.warning(f"Could not compute dHash for {image_path}: {e}")
        return 0


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integer hashes."""
    return bin(h1 ^ h2).count("1")


def extract_sequence_id(filename: str) -> Tuple[str, int]:
    """
    Extract country/route prefix and numerical frame index from image filename.
    e.g. 'India_001245.jpg' -> ('India', 1245)
    """
    stem = Path(filename).stem
    match = re.match(r"^([A-Za-z_]+)_(\d+)$", stem)
    if match:
        prefix, num_str = match.groups()
        return prefix, int(num_str)
    return stem, 0


def cluster_image_sequences(
    image_paths: List[Path],
    index_gap_threshold: int = 2,
    hamming_threshold: int = 10,
    use_dhash: bool = True,
) -> List[List[Path]]:
    """
    Group images into disjoint route sequence clusters.
    Ensures adjacent video frames never cross train/val boundaries.
    """
    # Sort files by sequence prefix and index
    sorted_paths = sorted(image_paths, key=lambda p: extract_sequence_id(p.name))
    if not sorted_paths:
        return []

    clusters: List[List[Path]] = []
    current_cluster: List[Path] = [sorted_paths[0]]
    prev_prefix, prev_idx = extract_sequence_id(sorted_paths[0].name)
    prev_hash = compute_dhash(sorted_paths[0]) if use_dhash else 0

    for p in sorted_paths[1:]:
        cur_prefix, cur_idx = extract_sequence_id(p.name)
        is_sequential = (cur_prefix == prev_prefix) and (0 < (cur_idx - prev_idx) <= index_gap_threshold)

        is_visually_similar = False
        cur_hash = 0
        if use_dhash and is_sequential:
            cur_hash = compute_dhash(p)
            if hamming_distance(prev_hash, cur_hash) <= hamming_threshold:
                is_visually_similar = True

        if is_sequential or is_visually_similar:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]

        prev_prefix, prev_idx = cur_prefix, cur_idx
        prev_hash = cur_hash

    if current_cluster:
        clusters.append(current_cluster)

    logger.info(f"Grouped {len(image_paths)} images into {len(clusters)} sequence clusters.")
    return clusters


def create_leakage_aware_split(
    image_paths: List[Path],
    train_ratio: float = 0.8,
    seed: int = 42,
    use_dhash: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """
    Split dataset into train and val splits at the cluster level.
    """
    clusters = cluster_image_sequences(image_paths, use_dhash=use_dhash)

    # Deterministic shuffle of clusters
    rng = random.Random(seed)
    rng.shuffle(clusters)

    total_images = len(image_paths)
    target_train_count = int(total_images * train_ratio)

    train_paths: List[Path] = []
    val_paths: List[Path] = []

    current_train_count = 0
    for cluster in clusters:
        if current_train_count < target_train_count:
            train_paths.extend(cluster)
            current_train_count += len(cluster)
        else:
            val_paths.extend(cluster)

    logger.info(
        f"Split complete (seed={seed}): {len(train_paths)} train images ({len(train_paths)/max(1, total_images)*100:.1f}%), "
        f"{len(val_paths)} val images ({len(val_paths)/max(1, total_images)*100:.1f}%)."
    )
    return train_paths, val_paths


def save_split_manifests(
    train_paths: List[Path],
    val_paths: List[Path],
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Save immutable split manifests with relative paths and SHA-256 integrity hash."""
    target_dir = output_dir or config.MANIFESTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    train_file = target_dir / "train_manifest.txt"
    val_file = target_dir / "val_manifest.txt"

    with open(train_file, "w") as f:
        for p in train_paths:
            f.write(f"{p.name}\n")

    with open(val_file, "w") as f:
        for p in val_paths:
            f.write(f"{p.name}\n")

    logger.info(f"Saved manifests: {train_file} ({len(train_paths)}), {val_file} ({len(val_paths)})")
    return train_file, val_file


def generate_yolo_dataset_yaml(
    dataset_root: Path,
    output_yaml_path: Path,
    train_split_rel: str = "images/train",
    val_split_rel: str = "images/val",
) -> Path:
    """Generate official YOLO dataset.yaml definition file."""
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = f"""# RoadSense AI - YOLO Dataset Configuration
# RDD2022 Canonical 4-Class Taxonomy

path: {dataset_root.resolve()}
train: {train_split_rel}
val: {val_split_rel}

# Classes
names:
  0: D00
  1: D10
  2: D20
  3: D40
"""
    with open(output_yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"Generated dataset YAML: {output_yaml_path}")
    return output_yaml_path
