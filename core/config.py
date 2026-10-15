"""
RoadSense AI - Centralized Application Configuration
Provides single-source-of-truth configuration constants, path definitions,
damage taxonomy maps, priority formula parameters, and hardware settings.
"""

from pathlib import Path
from typing import Dict, List, Any

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = BASE_DIR / "core"
APP_DIR = BASE_DIR / "app"
ML_DIR = BASE_DIR / "ml"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
DEMO_DIR = DATA_DIR / "demo"
MANIFESTS_DIR = DATA_DIR / "manifests"
SMOKE_DIR = DATA_DIR / "smoke"
DOCS_DIR = BASE_DIR / "docs"

# Database
DB_PATH = DATA_DIR / "roadsense.db"

# Checkpoint paths
PRIMARY_MODEL_PATH = MODELS_DIR / "best.pt"
SMOKE_MODEL_PATH = MODELS_DIR / "smoke_test.pt"
YOLO26_PRETRAINED_PATH = BASE_DIR / "yolo26n.pt"
YOLO11_PRETRAINED_PATH = BASE_DIR / "yolo11n.pt"


def get_active_model_path() -> Path:
    """Return the highest-precedence available model checkpoint path."""
    if PRIMARY_MODEL_PATH.exists():
        return PRIMARY_MODEL_PATH
    if SMOKE_MODEL_PATH.exists():
        return SMOKE_MODEL_PATH
    if YOLO26_PRETRAINED_PATH.exists():
        return YOLO26_PRETRAINED_PATH
    if YOLO11_PRETRAINED_PATH.exists():
        return YOLO11_PRETRAINED_PATH
    return PRIMARY_MODEL_PATH


# Ensure runtime directories exist
for _path in [DATA_DIR, EVIDENCE_DIR, DEMO_DIR, MANIFESTS_DIR, MODELS_DIR, REPORTS_DIR]:
    _path.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. CANONICAL DAMAGE TAXONOMY (RDD2022 / CRDDC Benchmark)
# ==============================================================================
# Strict 4-class canonical mapping (classes 0..3)
CLASS_NAMES: Dict[int, str] = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
}

CLASS_HUMAN_NAMES: Dict[str, str] = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "D00": "Linear cracks running parallel to vehicle travel direction, primarily along wheel paths.",
    "D10": "Linear cracks running perpendicular to the pavement center line, typically thermal or reflective.",
    "D20": "Series of interconnected polygonal cracks resembling reptile skin, indicating structural fatigue.",
    "D40": "Bowl-shaped holes or severe depressions in the pavement surface exceeding 25mm in depth.",
}

# Hex colors for UI badges and visualizations (WCAG contrast compliant)
CLASS_COLORS_HEX: Dict[str, str] = {
    "D00": "#3b82f6",  # Blue
    "D10": "#8b5cf6",  # Purple
    "D20": "#f59e0b",  # Amber / Orange
    "D40": "#ef4444",  # Red / Hazard
}

# BGR colors for OpenCV bounding box rendering
CLASS_COLORS_BGR: Dict[str, tuple] = {
    "D00": (246, 130, 59),   # #3b82f6
    "D10": (246, 92, 139),   # #8b5cf6
    "D20": (11, 158, 245),   # #f59e0b
    "D40": (68, 68, 239),    # #ef4444
}

# Ground-truth raw XML mappings:
# - Map human typos and sub-types
# - Filter out non-structural road markings or utility covers
RAW_CLASS_ALIASES: Dict[str, str] = {
    "D0w0": "D00",  # Ground-truth typo in RDD2022 India
    "D01": "D00",   # Longitudinal crack sub-type
    "D11": "D10",   # Transverse crack sub-type
}

# Classes explicitly filtered out during XML conversion (become background)
FILTERED_CLASSES: List[str] = [
    "D43",     # Crosswalk blur (paint deterioration)
    "D44",     # White line blur (paint deterioration)
    "D50",     # Manhole cover (utility fixture)
    "Repair",  # Asphalt maintenance patch
]

# Base damage severity weight w_class in priority ranking (0.0 - 1.0)
CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
    "D40": 1.0,  # Potholes represent immediate structural & tire impact hazard
    "D20": 0.8,  # Alligator cracking indicates deep sub-base fatigue
    "D10": 0.5,  # Transverse cracks require crack sealing
    "D00": 0.4,  # Longitudinal cracks are early-stage surface defects
}


# ==============================================================================
# 3. PRIORITY ENGINE CONFIGURATION (Formula v1.0)
# ==============================================================================
PRIORITY_FORMULA_VERSION = "1.0.0"

PRIORITY_SCALING: Dict[str, float] = {
    "class_confidence_scale": 40.0,  # Maximum 40 points from damage class * confidence
    "cluster_repeat_scale": 6.0,     # Up to 6 points per repeat detection within radius
    "cluster_repeat_cap": 5.0,       # Maximum 5 repeat detections counted (30 points max)
    "road_class_scale": 15.0,        # Up to 15 points based on road importance
    "human_urgency_scale": 15.0,     # Up to 15 points based on field reviewer assessment
}

# Spatial clustering radius for counting repeat observations of same defect
CLUSTER_RADIUS_METERS = 25.0

ROAD_CLASS_WEIGHTS: Dict[str, float] = {
    "PRIMARY_ARTERIAL": 1.0,      # Major thoroughfare / highway (heavy traffic, high speed)
    "SECONDARY_COLLECTOR": 0.7,   # Urban connector / bus route
    "RESIDENTIAL_LOCAL": 0.4,     # Neighborhood street (low speed)
    "UNKNOWN": 0.5,               # Default baseline when unverified
}

PRIORITY_LEVELS: Dict[str, Dict[str, Any]] = {
    "CRITICAL": {
        "min_score": 75.0,
        "label": "Critical",
        "description": "Immediate inspection required within 24–48 hours",
        "color_hex": "#ef4444",
        "badge_bg": "#fee2e2",
        "badge_fg": "#991b1b",
    },
    "HIGH": {
        "min_score": 55.0,
        "label": "High",
        "description": "Scheduled inspection within 7 days",
        "color_hex": "#f97316",
        "badge_bg": "#ffedd5",
        "badge_fg": "#9a3412",
    },
    "MODERATE": {
        "min_score": 35.0,
        "label": "Moderate",
        "description": "Routine maintenance queue (30 days)",
        "color_hex": "#f59e0b",
        "badge_bg": "#fef3c7",
        "badge_fg": "#92400e",
    },
    "LOW": {
        "min_score": 0.0,
        "label": "Low",
        "description": "Monitor on subsequent survey cycle",
        "color_hex": "#10b981",
        "badge_bg": "#d1fae5",
        "badge_fg": "#065f46",
    },
}


# ==============================================================================
# 4. INFERENCE & HARDWARE CONFIGURATION
# ==============================================================================
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640

# Hardware acceleration fallback order: mps -> cuda -> cpu
def detect_default_device() -> str:
    """Determine the optimal PyTorch compute device for local host."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEFAULT_DEVICE = detect_default_device()


# ==============================================================================
# 5. INPUT VALIDATION & LIMITS
# ==============================================================================
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024       # 25 MB
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_VIDEO_SIZE_BYTES = 150 * 1024 * 1024      # 150 MB
MAX_VIDEO_DURATION_SECONDS = 180.0            # 3 minutes
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".m4v"}

# Video sampling defaults
DEFAULT_VIDEO_SAMPLE_FPS = 1.0  # Analyze 1 frame per second
MIN_VIDEO_SAMPLE_FPS = 0.5
MAX_VIDEO_SAMPLE_FPS = 5.0


# ==============================================================================
# 6. LEGAL & EPISTEMIC DISCLAIMER
# ==============================================================================
CIVIL_SAFETY_NOTICE = (
    "Model confidence is not physical damage severity. RoadSense AI provides automated "
    "inspection prioritization only and does not replace certified civil engineering "
    "structural evaluations or professional road-safety assessments."
)
