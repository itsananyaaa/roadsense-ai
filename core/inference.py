"""
RoadSense AI - Object Detection Inference Service
Wraps Ultralytics YOLO inference with input validation, Apple MPS acceleration,
custom bounding-box annotation styling, and transparent checkpoint status tracking.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from core import config
from core.health import check_checkpoint_status
from core.logger import get_logger

logger = get_logger("roadsense.inference")

# Enforce explicit image decompression bomb limit (100 million pixels)
Image.MAX_IMAGE_PIXELS = 100_000_000


class ModelWeightsNotFoundError(FileNotFoundError):
    """Raised when an inference request is made but no valid checkpoint exists."""
    pass


class InvalidImageInputError(ValueError):
    """Raised when the uploaded or provided image is corrupt or unreadable."""
    pass


@dataclass
class Detection:
    """Single bounding-box detection item."""
    class_id: int
    class_code: str
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2] integer pixels strictly clipped to image dimensions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
        }


@dataclass
class DetectionResult:
    """Consolidated inference result for an image."""
    annotated_image_bgr: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    detected_classes: List[str] = field(default_factory=list)
    detection_count: int = 0
    max_confidence: float = 0.0
    inference_latency_ms: float = 0.0
    checkpoint_id: str = "Unknown"
    checkpoint_status: str = "MISSING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_classes": self.detected_classes,
            "detection_count": self.detection_count,
            "max_confidence": round(self.max_confidence, 4),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_status": self.checkpoint_status,
            "detections": [d.to_dict() for d in self.detections],
        }


class DamageDetector:
    """
    Road damage detector instance managing model lifecycle and predictions.
    Thread-safe under concurrent FastAPI requests via threading.RLock.
    """
    _instance: Optional["DamageDetector"] = None
    _loaded_model_path: Optional[str] = None
    _model = None
    _lock = threading.RLock()

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.device = device or config.DEFAULT_DEVICE
        self.checkpoint_path, self.checkpoint_status = self._resolve_checkpoint(checkpoint_path)
        self.load_model()

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[Union[str, Path]] = None, device: Optional[str] = None) -> "DamageDetector":
        """Singleton accessor to prevent reloading weights into memory per interaction."""
        with cls._lock:
            resolved_path, _ = cls._resolve_checkpoint_static(checkpoint_path)
            if cls._instance is None or cls._loaded_model_path != resolved_path:
                cls._instance = cls(checkpoint_path=resolved_path, device=device)
                cls._loaded_model_path = resolved_path
            return cls._instance

    @staticmethod
    def _resolve_checkpoint_static(checkpoint_path: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
        if checkpoint_path and Path(checkpoint_path).exists():
            p_str = str(checkpoint_path)
            if "smoke" in p_str:
                return p_str, "SMOKE_CHECKPOINT"
            if "best" in p_str:
                return p_str, "EVALUATED_TRAINED"
            return p_str, "PRETRAINED_BASELINE"
        return check_checkpoint_status()

    def _resolve_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
        return self._resolve_checkpoint_static(checkpoint_path)

    def load_model(self) -> None:
        """Load YOLO model into memory on configured acceleration device."""
        with self._lock:
            if self.checkpoint_status == "MISSING" or self.checkpoint_path == "None":
                logger.warning("No checkpoint available to load.")
                self._model = None
                return

            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLO model from {self.checkpoint_path} on {self.device}...")
                self._model = YOLO(self.checkpoint_path)
                logger.info(f"YOLO model loaded. Pedigree: {self.checkpoint_status}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self._model = None
                raise ModelWeightsNotFoundError(f"Failed to load weights from {self.checkpoint_path}: {e}")

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
    ) -> DetectionResult:
        """
        Execute road damage detection on provided image input with thread serialization
        and safe device fallback.
        """
        with self._lock:
            # Validate and prepare image as BGR numpy array
            img_bgr = self._validate_image_input(image_input)
            orig_h, orig_w = img_bgr.shape[:2]

            if self._model is None:
                # Check if weights can be discovered
                self.checkpoint_path, self.checkpoint_status = self._resolve_checkpoint()
                if self.checkpoint_status == "MISSING":
                    raise ModelWeightsNotFoundError(
                        "No model checkpoint found. Please download best.pt into models/ or use Guided Demo Mode."
                    )
                self.load_model()

            # Run inference under torch.inference_mode with device fallback
            with torch.inference_mode():
                t0 = time.perf_counter()
                try:
                    results = self._model.predict(
                        source=img_bgr,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=self.device,
                        imgsz=config.DEFAULT_IMG_SIZE,
                        verbose=False,
                    )
                except Exception as dev_err:
                    if self.device != "cpu":
                        logger.warning(f"Inference on device '{self.device}' failed ({dev_err}); falling back to CPU.")
                        results = self._model.predict(
                            source=img_bgr,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            device="cpu",
                            imgsz=config.DEFAULT_IMG_SIZE,
                            verbose=False,
                        )
                    else:
                        raise dev_err

                # Synchronize device if needed to measure accurate latency
                if self.device == "mps" and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

            detections: List[Detection] = []
            annotated_bgr = img_bgr.copy()

            if results and len(results) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    raw_xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Strict boundary clipping to image dimensions
                    x1 = max(0, min(orig_w, int(round(raw_xyxy[0]))))
                    y1 = max(0, min(orig_h, int(round(raw_xyxy[1]))))
                    x2 = max(0, min(orig_w, int(round(raw_xyxy[2]))))
                    y2 = max(0, min(orig_h, int(round(raw_xyxy[3]))))
                    xyxy = [x1, y1, x2, y2]

                    # Resolve class code and display name
                    if self.checkpoint_status in ["EVALUATED_TRAINED", "SMOKE_CHECKPOINT"]:
                        cls_code = config.CLASS_NAMES.get(cls_id, f"CLASS_{cls_id}")
                    else:
                        # In pretrained baseline (COCO), report raw COCO class name
                        coco_names = getattr(self._model, "names", {})
                        cls_code = coco_names.get(cls_id, f"COCO_{cls_id}")

                    cls_human = config.CLASS_HUMAN_NAMES.get(cls_code, cls_code)

                    det = Detection(
                        class_id=cls_id,
                        class_code=cls_code,
                        class_name=cls_human,
                        confidence=conf,
                        bbox=xyxy,
                    )
                    detections.append(det)

                    # Draw bounding box and label
                    self._draw_detection_box(annotated_bgr, det)

            # Aggregate summary
            detected_classes = sorted(list({d.class_code for d in detections}))
            max_conf = max([d.confidence for d in detections], default=0.0)

            ckpt_id = Path(self.checkpoint_path).name if self.checkpoint_path != "None" else "None"

            return DetectionResult(
                annotated_image_bgr=annotated_bgr,
                detections=detections,
                detected_classes=detected_classes,
                detection_count=len(detections),
                max_confidence=max_conf,
                inference_latency_ms=latency_ms,
                checkpoint_id=ckpt_id,
                checkpoint_status=self.checkpoint_status,
            )

    def _draw_detection_box(self, img_bgr: np.ndarray, det: Detection) -> None:
        """Render high-contrast, professional bounding box on image."""
        x1, y1, x2, y2 = det.bbox
        color = config.CLASS_COLORS_BGR.get(det.class_code, (0, 165, 255))

        # Main bounding rectangle with 2px stroke
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Corner accents for sleek tech command aesthetic
        corner_len = min(15, max(5, int((x2 - x1) * 0.15)))
        cv2.line(img_bgr, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(img_bgr, (x1, y1), (x1, y1 + corner_len), color, 4)
        cv2.line(img_bgr, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(img_bgr, (x2, y2), (x2, y2 - corner_len), color, 4)

        # Label tag
        label_text = f"{det.class_code}: {det.confidence*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

        # Tag background pill
        tag_y1 = max(0, y1 - txt_h - 8)
        tag_y2 = y1
        cv2.rectangle(img_bgr, (x1, tag_y1), (x1 + txt_w + 10, tag_y2), color, -1)
        # Label text in white / dark contrast
        text_color = (255, 255, 255) if det.class_code in ["D40", "D00", "D10"] else (20, 20, 20)
        cv2.putText(img_bgr, label_text, (x1 + 5, tag_y2 - 4), font, font_scale, text_color, font_thick, cv2.LINE_AA)

    def _validate_image_input(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Validate format, dimensions, readability, and return BGR numpy array with EXIF orientation corrected."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise InvalidImageInputError(f"Image file does not exist: {p}")
            if p.stat().st_size == 0:
                raise InvalidImageInputError(f"Image file is empty (0 bytes): {p}")
            if p.stat().st_size > config.MAX_IMAGE_SIZE_BYTES:
                raise InvalidImageInputError(
                    f"File size ({p.stat().st_size / 1e6:.1f} MB) exceeds maximum allowed limit ({config.MAX_IMAGE_SIZE_BYTES / 1e6:.0f} MB)."
                )
            try:
                with Image.open(p) as pil_img:
                    if pil_img.size[0] * pil_img.size[1] > Image.MAX_IMAGE_PIXELS:
                        raise InvalidImageInputError(f"Image dimensions ({pil_img.size[0]}x{pil_img.size[1]}) exceed safety limits.")
                    pil_img = ImageOps.exif_transpose(pil_img)
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
                    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Image.DecompressionBombError as dbe:
                raise InvalidImageInputError(f"Image exceeds decompression-bomb limit: {dbe}")
            except InvalidImageInputError:
                raise
            except Exception:
                img = cv2.imread(str(p))
                if img is None:
                    raise InvalidImageInputError(f"File could not be decoded as an image: {p}")
                return img

        if isinstance(image_input, Image.Image):
            if image_input.size[0] * image_input.size[1] > Image.MAX_IMAGE_PIXELS:
                raise InvalidImageInputError(f"Image dimensions ({image_input.size[0]}x{image_input.size[1]}) exceed safety limits.")
            pil_img = ImageOps.exif_transpose(image_input)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if len(image_input.shape) == 3 and image_input.shape[2] in [3, 4]:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
                return image_input
            raise InvalidImageInputError(f"Invalid numpy array image shape: {image_input.shape}")

        raise InvalidImageInputError(f"Unsupported image input type: {type(image_input)}")

