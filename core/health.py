"""
RoadSense AI - System Health & Diagnostics
Evaluates local hardware capabilities, active model checkpoint status,
database accessibility, and filesystem permissions.
"""

import os
import platform
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import config
from core.logger import get_logger

logger = get_logger("roadsense.health")


@dataclass
class HealthReport:
    """Consolidated status report of the local RoadSense AI platform."""
    is_operational: bool
    python_version: str
    os_info: str
    torch_version: str
    accelerator: str
    ultralytics_version: str
    active_checkpoint_path: str
    checkpoint_status: str  # 'EVALUATED_TRAINED', 'SMOKE_CHECKPOINT', 'PRETRAINED_BASELINE', 'MISSING'
    database_status: str    # 'READY', 'CREATED_EMPTY', 'UNREACHABLE'
    incident_count: int
    writable_paths: Dict[str, bool] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_operational": self.is_operational,
            "python_version": self.python_version,
            "os_info": self.os_info,
            "torch_version": self.torch_version,
            "accelerator": self.accelerator,
            "ultralytics_version": self.ultralytics_version,
            "active_checkpoint_path": self.active_checkpoint_path,
            "checkpoint_status": self.checkpoint_status,
            "database_status": self.database_status,
            "incident_count": self.incident_count,
            "writable_paths": self.writable_paths,
            "diagnostics": self.diagnostics,
        }


def check_checkpoint_status() -> tuple[str, str]:
    """
    Determine which checkpoint is available and its pedigree.
    Returns: (resolved_path, status_label)
    """
    if config.PRIMARY_MODEL_PATH.exists():
        return str(config.PRIMARY_MODEL_PATH), "EVALUATED_TRAINED"
    if config.SMOKE_MODEL_PATH.exists():
        return str(config.SMOKE_MODEL_PATH), "SMOKE_CHECKPOINT"
    if config.YOLO26_PRETRAINED_PATH.exists():
        return str(config.YOLO26_PRETRAINED_PATH), "PRETRAINED_BASELINE"
    if config.YOLO11_PRETRAINED_PATH.exists():
        return str(config.YOLO11_PRETRAINED_PATH), "PRETRAINED_BASELINE"
    return "None", "MISSING"


def check_database() -> tuple[str, int]:
    """Check SQLite database reachability and record count."""
    try:
        conn = sqlite3.connect(str(config.DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='incidents'")
        table_exists = cursor.fetchone()[0] > 0
        if not table_exists:
            conn.close()
            return "CREATED_EMPTY", 0
        cursor.execute("SELECT count(*) FROM incidents")
        count = cursor.fetchone()[0]
        conn.close()
        return "READY", count
    except Exception as e:
        logger.warning(f"Database check failed: {e}")
        return "UNREACHABLE", 0


def run_health_check() -> HealthReport:
    """Execute complete system diagnostic check."""
    diagnostics: List[str] = []

    # 1. Environment & Hardware
    py_ver = platform.python_version()
    os_str = f"{platform.system()} {platform.release()} ({platform.machine()})"

    # 2. PyTorch & Acceleration
    try:
        import torch
        torch_ver = torch.__version__
        if torch.backends.mps.is_available():
            accel = "Apple Metal Performance Shaders (MPS)"
        elif torch.cuda.is_available():
            accel = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
        else:
            accel = "CPU"
    except Exception as e:
        torch_ver = f"Error: {e}"
        accel = "Unavailable"
        diagnostics.append(f"PyTorch error: {e}")

    # 3. Ultralytics
    try:
        import ultralytics
        ultra_ver = ultralytics.__version__
    except Exception as e:
        ultra_ver = "Not Installed"
        diagnostics.append(f"Ultralytics error: {e}")

    # 4. Checkpoints
    ckpt_path, ckpt_status = check_checkpoint_status()
    if ckpt_status == "MISSING":
        diagnostics.append("No local model weights found in models/. Demo mode available.")
    elif ckpt_status == "PRETRAINED_BASELINE":
        diagnostics.append("Pretrained COCO baseline detected. Not fine-tuned on road damage.")
    elif ckpt_status == "SMOKE_CHECKPOINT":
        diagnostics.append("Smoke-test checkpoint active (1-2 epochs on tiny sample; not fully evaluated).")

    # 5. Database
    db_status, inc_count = check_database()
    if db_status == "UNREACHABLE":
        diagnostics.append("SQLite database unreachable.")

    # 6. Writable paths
    writable: Dict[str, bool] = {}
    for name, p in [
        ("data", config.DATA_DIR),
        ("evidence", config.EVIDENCE_DIR),
        ("models", config.MODELS_DIR),
        ("reports", config.REPORTS_DIR),
    ]:
        writable[name] = os.access(str(p), os.W_OK)
        if not writable[name]:
            diagnostics.append(f"Directory {name} ({p}) is not writable.")

    is_operational = (
        "Error" not in torch_ver
        and ultra_ver != "Not Installed"
        and all(writable.values())
    )

    return HealthReport(
        is_operational=is_operational,
        python_version=py_ver,
        os_info=os_str,
        torch_version=torch_ver,
        accelerator=accel,
        ultralytics_version=ultra_ver,
        active_checkpoint_path=ckpt_path,
        checkpoint_status=ckpt_status,
        database_status=db_status,
        incident_count=inc_count,
        writable_paths=writable,
        diagnostics=diagnostics,
    )


if __name__ == "__main__":
    report = run_health_check()
    print("--- ROADSENSE AI HEALTH CHECK ---")
    for k, v in report.to_dict().items():
        print(f"{k}: {v}")
