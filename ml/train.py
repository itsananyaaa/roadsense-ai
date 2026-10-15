"""
RoadSense AI - Model Training Engine
Supports:
1. Smoke Mode (--mode smoke): Quick 1-2 epoch fine-tuning on miniature sample via Mac MPS.
   Outputs models/smoke_test.pt with explicit metadata 'smoke-test checkpoint, not evaluated'.
2. Full Mode (--mode full): Production training for Google Colab GPU with early stopping,
   checkpointing, and resumability from last.pt.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core import config
from core.logger import get_logger
from ml.split import generate_yolo_dataset_yaml

logger = get_logger("roadsense.ml.train")


def create_smoke_dataset(smoke_dir: Path) -> Path:
    """
    Generate a miniature calibrated dataset for local smoke-test training on Mac.
    Creates 12 training samples and 4 validation samples covering all canonical classes.
    """
    smoke_dir.mkdir(parents=True, exist_ok=True)
    img_train_dir = smoke_dir / "images" / "train"
    img_val_dir = smoke_dir / "images" / "val"
    lbl_train_dir = smoke_dir / "labels" / "train"
    lbl_val_dir = smoke_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Class definitions: 0: D00, 1: D10, 2: D20, 3: D40
    smoke_specs = [
        {"name": "smoke_d40_01", "cls": 3, "xc": 0.5, "yc": 0.5, "w": 0.25, "h": 0.25, "split": "train"},
        {"name": "smoke_d40_02", "cls": 3, "xc": 0.4, "yc": 0.6, "w": 0.30, "h": 0.20, "split": "train"},
        {"name": "smoke_d20_01", "cls": 2, "xc": 0.5, "yc": 0.4, "w": 0.35, "h": 0.35, "split": "train"},
        {"name": "smoke_d20_02", "cls": 2, "xc": 0.6, "yc": 0.5, "w": 0.40, "h": 0.30, "split": "train"},
        {"name": "smoke_d10_01", "cls": 1, "xc": 0.5, "yc": 0.5, "w": 0.60, "h": 0.10, "split": "train"},
        {"name": "smoke_d10_02", "cls": 1, "xc": 0.5, "yc": 0.7, "w": 0.50, "h": 0.08, "split": "train"},
        {"name": "smoke_d00_01", "cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.08, "h": 0.60, "split": "train"},
        {"name": "smoke_d00_02", "cls": 0, "xc": 0.3, "yc": 0.5, "w": 0.10, "h": 0.50, "split": "train"},
        {"name": "smoke_clean_01", "cls": None, "split": "train"},
        {"name": "smoke_clean_02", "cls": None, "split": "train"},
        # Val split
        {"name": "smoke_d40_val", "cls": 3, "xc": 0.5, "yc": 0.5, "w": 0.28, "h": 0.22, "split": "val"},
        {"name": "smoke_d20_val", "cls": 2, "xc": 0.5, "yc": 0.5, "w": 0.35, "h": 0.35, "split": "val"},
        {"name": "smoke_d00_val", "cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.10, "h": 0.55, "split": "val"},
        {"name": "smoke_clean_val", "cls": None, "split": "val"},
    ]

    for spec in smoke_specs:
        s_type = spec["split"]
        img_out = (img_train_dir if s_type == "train" else img_val_dir) / f"{spec['name']}.jpg"
        lbl_out = (lbl_train_dir if s_type == "train" else lbl_val_dir) / f"{spec['name']}.txt"

        # Generate textured asphalt image
        img = np.full((640, 640, 3), 60, dtype=np.uint8)
        # Road lane lines
        cv2.line(img, (80, 640), (280, 200), (220, 220, 220), 4)
        cv2.line(img, (560, 640), (360, 200), (220, 220, 220), 4)

        if spec["cls"] is not None:
            c = spec["cls"]
            xc, yc, w, h = spec["xc"], spec["yc"], spec["w"], spec["h"]
            x1 = int((xc - w/2) * 640)
            y1 = int((yc - h/2) * 640)
            x2 = int((xc + w/2) * 640)
            y2 = int((yc + h/2) * 640)
            if c == 3:  # Pothole
                cv2.ellipse(img, ((x1+x2)//2, (y1+y2)//2), ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, (20, 20, 20), -1)
            elif c == 2:  # Alligator
                cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
            elif c == 1:  # Transverse
                cv2.line(img, (x1, (y1+y2)//2), (x2, (y1+y2)//2), (25, 25, 25), 3)
            elif c == 0:  # Longitudinal
                cv2.line(img, ((x1+x2)//2, y1), ((x1+x2)//2, y2), (25, 25, 25), 3)

            # Write YOLO label
            lbl_out.write_text(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        else:
            # Empty negative background label
            lbl_out.write_text("")

        cv2.imwrite(str(img_out), img)

    # Generate dataset.yaml
    yaml_path = smoke_dir / "dataset.yaml"
    generate_yolo_dataset_yaml(smoke_dir, yaml_path)
    logger.info(f"Smoke test dataset initialized at {smoke_dir}")
    return yaml_path


def run_training(
    mode: str = "smoke",
    device: Optional[str] = None,
    epochs: int = 2,
    batch_size: int = 8,
    model_name: str = "yolo26n.pt",
    data_yaml: Optional[Path] = None,
    resume: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """Execute training pipeline in smoke or full mode."""
    from ultralytics import YOLO

    device = device or config.DEFAULT_DEVICE
    out_dir = output_dir or config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        smoke_dir = config.SMOKE_DIR
        yaml_path = create_smoke_dataset(smoke_dir)
        epochs = min(epochs, 3)

        logger.info(f"Starting SMOKE training (1-2 epochs on Mac {device})...")
        model = YOLO(model_name)
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=config.DEFAULT_IMG_SIZE,
            device=device,
            verbose=False,
            project=str(smoke_dir / "runs"),
            name="smoke_run",
            exist_ok=True,
        )

        # Copy smoke checkpoint to models/smoke_test.pt
        best_pt = smoke_dir / "runs" / "smoke_run" / "weights" / "best.pt"
        target_pt = out_dir / "smoke_test.pt"
        if best_pt.exists():
            shutil.copy(best_pt, target_pt)
        else:
            # Fallback to last.pt
            last_pt = smoke_dir / "runs" / "smoke_run" / "weights" / "last.pt"
            if last_pt.exists():
                shutil.copy(last_pt, target_pt)

        # Write metadata JSON explicitly stating this is a smoke-test checkpoint
        meta_path = out_dir / "smoke_test_metadata.json"
        meta = {
            "checkpoint_name": "smoke_test.pt",
            "pedigree": "smoke-test checkpoint, not evaluated",
            "epochs_trained": epochs,
            "device_used": device,
            "architecture": model_name,
            "disclaimer": "This checkpoint was trained on a miniature smoke fixture to verify pipeline execution. It is NOT evaluated for production accuracy.",
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved smoke checkpoint: {target_pt}")
        return target_pt

    else:
        # Full training mode
        if data_yaml is None or not Path(data_yaml).exists():
            raise FileNotFoundError(f"data_yaml required for full training mode: {data_yaml}")

        logger.info(f"Starting FULL training ({epochs} epochs on {device}, resume={resume})...")
        model = YOLO(model_name)
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=config.DEFAULT_IMG_SIZE,
            device=device,
            patience=10,
            save=True,
            resume=resume,
            project=str(out_dir / "runs"),
            name="rdd_full_run",
            exist_ok=True,
        )

        best_pt = out_dir / "runs" / "rdd_full_run" / "weights" / "best.pt"
        final_target = out_dir / "best.pt"
        if best_pt.exists():
            shutil.copy(best_pt, final_target)
            logger.info(f"Saved production checkpoint: {final_target}")
        return final_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSense AI - Model Training Engine")
    parser.add_argument("--mode", type=str, choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--data-yaml", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    pt = run_training(
        mode=args.mode,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch,
        model_name=args.model,
        data_yaml=Path(args.data_yaml) if args.data_yaml else None,
        resume=args.resume,
    )
    print(f"Training completed successfully. Checkpoint saved: {pt}")
