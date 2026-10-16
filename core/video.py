"""
RoadSense AI - Video Intelligence Engine
Processes dashcam video streams (MP4, AVI, MOV) for road distress detection.
Features:
- Configurable frame sampling rate (e.g. 1 frame/sec, 2 fps)
- Sequential frame extraction with progress callback
- Temporal de-duplication: prevents auto-inflating duplicate incidents for the same physical pothole/crack visible across multiple consecutive frames
- Keyframe extraction and bounding box visualization
- Synthetic demo dashcam video generator for offline testing
"""

import math
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from core import config
from core.inference import DamageDetector, Detection, DetectionResult
from core.logger import get_logger
from core.priority import calculate_priority

logger = get_logger("roadsense.video")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class KeyframeEvent:
    """Represents a significant road distress detected in a video keyframe."""
    frame_index: int
    timestamp_sec: float
    timestamp_formatted: str
    detections: List[Detection]
    keyframe_path: str
    max_severity_class: str
    max_confidence: float
    priority_score: float
    priority_level: str


@dataclass
class VideoAnalysisReport:
    """Consolidated summary of video processing."""
    video_name: str
    total_frames: int
    fps: float
    duration_sec: float
    sampled_frames_count: int
    sample_fps: float
    processing_time_sec: float
    total_detections_count: int
    distinct_incidents_count: int
    damage_distribution: Dict[str, int]
    keyframe_events: List[KeyframeEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_sec": round(self.duration_sec, 2),
            "sampled_frames_count": self.sampled_frames_count,
            "sample_fps": self.sample_fps,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "total_detections_count": self.total_detections_count,
            "distinct_incidents_count": self.distinct_incidents_count,
            "damage_distribution": self.damage_distribution,
            "keyframes_count": len(self.keyframe_events)
        }


def format_timestamp(seconds: float) -> str:
    """Format floating seconds into mm:ss.ms."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 100)
    return f"{mins:02d}:{secs:02d}.{millis:02d}"


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class VideoProcessor:
    """Engine for ingesting, sampling, and analyzing dashcam footage."""

    def __init__(self, inference_engine: Optional[DamageDetector] = None):
        self.engine = inference_engine or DamageDetector.get_instance()
        self.keyframes_dir = config.EVIDENCE_DIR / "video_keyframes"
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

    def analyze_video(
        self,
        video_path: Path,
        sample_fps: float = 1.0,
        conf_threshold: float = 0.35,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_duration_sec: Optional[float] = 60.0
    ) -> VideoAnalysisReport:
        """
        Analyzes a video file at the requested sampling rate.
        Applies temporal de-duplication to prevent auto-inflating duplicate incidents.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.stat().st_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {video_path}")

        if video_path.stat().st_size > config.MAX_VIDEO_SIZE_BYTES:
            raise ValueError(
                f"Video file size ({video_path.stat().st_size / 1e6:.1f} MB) exceeds maximum allowed limit ({config.MAX_VIDEO_SIZE_BYTES / 1e6:.0f} MB)."
            )

        ext = video_path.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format '{ext}'. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            orig_fps = float(raw_fps) if (raw_fps and not math.isnan(raw_fps) and raw_fps > 0) else 25.0
            duration_sec = total_frames / orig_fps if orig_fps > 0 else 0.0

            if max_duration_sec and duration_sec > max_duration_sec:
                total_frames = int(max_duration_sec * orig_fps)
                duration_sec = max_duration_sec

            # Calculate sampling step
            frame_step = max(1, int(orig_fps / sample_fps))
            logger.info(f"Analyzing {video_path.name}: {total_frames} frames @ {orig_fps:.1f} fps (~{duration_sec:.1f}s), sampling every {frame_step} frames ({sample_fps} FPS)")

            t_start = time.perf_counter()
            sampled_count = 0
            damage_dist: Dict[str, int] = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
            total_detections = 0
            keyframe_events: List[KeyframeEvent] = []

            # Tracking active distress tracks to de-duplicate consecutive frames
            active_tracks: List[Dict[str, Any]] = []  # list of {class_code, box, last_seen_sec, incident_id}
            distinct_incidents_count = 0

            current_frame_idx = 0
            while current_frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                sampled_count += 1
                timestamp_sec = current_frame_idx / orig_fps

                if progress_callback:
                    progress = min(1.0, current_frame_idx / total_frames)
                    progress_callback(progress, f"Analyzing frame {current_frame_idx}/{total_frames} ({timestamp_sec:.1f}s)")

                # Run inference on sampled frame
                det_result = self.engine.predict(frame, conf_threshold=conf_threshold)
                detections = det_result.detections

                if detections:
                    total_detections += len(detections)
                    is_new_incident_frame = False

                    for det in detections:
                        damage_dist[det.class_code] = damage_dist.get(det.class_code, 0) + 1

                        # Check temporal de-duplication:
                        # If this detection matches an active track of same class within 2.5 seconds and spatial proximity, group it.
                        matched_track = None
                        for track in active_tracks:
                            if track["class_code"] == det.class_code and (timestamp_sec - track["last_seen_sec"]) <= 2.5:
                                iou = calculate_iou(track["box"], det.bbox)
                                c1 = ((track["box"][0] + track["box"][2]) / 2, (track["box"][1] + track["box"][3]) / 2)
                                c2 = ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
                                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                                if iou > 0.15 or dist < (frame.shape[1] * 0.20):
                                    matched_track = track
                                    break

                        if matched_track:
                            # Update track
                            matched_track["last_seen_sec"] = timestamp_sec
                            matched_track["box"] = det.bbox
                        else:
                            # New physical distress incident discovered
                            distinct_incidents_count += 1
                            is_new_incident_frame = True
                            active_tracks.append({
                                "class_code": det.class_code,
                                "box": det.bbox,
                                "last_seen_sec": timestamp_sec,
                                "incident_id": f"inc_{distinct_incidents_count}"
                            })

                    # Save keyframe if it represents a new incident or high severity
                    if is_new_incident_frame:
                        annotated_frame = det_result.annotated_image_bgr
                        keyframe_filename = f"keyframe_{int(timestamp_sec*100):05d}_{uuid.uuid4().hex[:6]}.jpg"
                        keyframe_dest = self.keyframes_dir / keyframe_filename
                        cv2.imwrite(str(keyframe_dest), annotated_frame)

                        # Compute priority for keyframe with correct function signature
                        top_det = max(detections, key=lambda d: (config.CLASS_SEVERITY_WEIGHTS.get(d.class_code, 0.4), d.confidence))
                        prio_res = calculate_priority(
                            detected_classes=[top_det.class_code],
                            max_confidence=top_det.confidence,
                            nearby_repeat_count=1,
                            road_classification="PRIMARY_ARTERIAL",
                        )
                        p_score = prio_res.score
                        p_level = prio_res.level

                        keyframe_events.append(KeyframeEvent(
                            frame_index=current_frame_idx,
                            timestamp_sec=timestamp_sec,
                            timestamp_formatted=format_timestamp(timestamp_sec),
                            detections=detections,
                            keyframe_path=str(keyframe_dest),
                            max_severity_class=top_det.class_code,
                            max_confidence=round(top_det.confidence, 3),
                            priority_score=round(p_score, 1),
                            priority_level=p_level
                        ))

                # Prune stale tracks older than 3 seconds
                active_tracks = [t for t in active_tracks if (timestamp_sec - t["last_seen_sec"]) <= 3.0]

                current_frame_idx += frame_step
        finally:
            cap.release()
        t_elapsed = time.perf_counter() - t_start

        if progress_callback:
            progress_callback(1.0, f"Analysis complete ({sampled_count} frames analyzed in {t_elapsed:.1f}s)")

        return VideoAnalysisReport(
            video_name=video_path.name,
            total_frames=total_frames,
            fps=orig_fps,
            duration_sec=duration_sec,
            sampled_frames_count=sampled_count,
            sample_fps=sample_fps,
            processing_time_sec=t_elapsed,
            total_detections_count=total_detections,
            distinct_incidents_count=distinct_incidents_count,
            damage_distribution=damage_dist,
            keyframe_events=keyframe_events
        )


def transcode_to_h264(input_path: Path, output_path: Path) -> bool:
    """
    Transcode video to browser-playable H.264 (avc1) MP4 with yuv420p pixel format
    and faststart moov atom header placement using imageio-ffmpeg.
    """
    try:
        import subprocess
        # pyrefly: ignore [missing-import]
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", str(input_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except Exception as e:
        logger.warning(f"FFmpeg H.264 transcode failed: {e}")
        return False


def generate_demo_dashcam_video(output_path: Path, duration_sec: int = 5, fps: int = 15) -> Path:
    """
    Generates a realistic animated dashcam demonstration MP4 video for offline expo testing.
    Produces a browser-playable H.264 (avc1) MP4 with yuv420p and faststart.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    total_frames = duration_sec * fps

    raw_temp_path = output_path.parent / f"_raw_temp_{output_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(raw_temp_path), fourcc, fps, (w, h))

    try:
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            # Create base asphalt texture
            base_gray = np.random.randint(65, 85, (h, w), dtype=np.uint8)
            frame = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

            # Draw perspective road lanes
            lane_color = (200, 200, 200)
            cv2.line(frame, (int(w * 0.45), int(h * 0.35)), (int(w * 0.10), h), lane_color, 4)
            cv2.line(frame, (int(w * 0.55), int(h * 0.35)), (int(w * 0.90), h), lane_color, 4)
            cv2.line(frame, (int(w * 0.50), int(h * 0.35)), (int(w * 0.50), h), (40, 180, 220), 2)  # Center dashed

            # Simulate approaching distresses
            # Pothole (D40) approaching between t=1.0 and t=3.0
            if 1.0 <= t <= 3.2:
                progress = (t - 1.0) / 2.2
                y = int(h * 0.45 + progress * (h * 0.45))
                x = int(w * 0.40 - progress * (w * 0.05))
                radius = int(8 + progress * 32)
                cv2.circle(frame, (x, y), radius, (25, 25, 30), -1)
                cv2.ellipse(frame, (x, y), (radius, int(radius * 0.6)), 0, 0, 360, (15, 15, 20), -1)
                cv2.circle(frame, (x + 2, y + 2), int(radius * 0.8), (40, 40, 45), 2)

            # Alligator crack (D20) approaching between t=2.5 and t=4.8
            if 2.5 <= t <= 4.8:
                progress = (t - 2.5) / 2.3
                y = int(h * 0.40 + progress * (h * 0.48))
                x = int(w * 0.60 + progress * (w * 0.08))
                sz = int(12 + progress * 40)
                cv2.rectangle(frame, (x - sz, y - int(sz * 0.6)), (x + sz, y + int(sz * 0.6)), (35, 35, 40), -1)
                # Web cracks
                for dx, dy in [(-sz, 0), (sz, 0), (0, -int(sz*0.6)), (0, int(sz*0.6))]:
                    cv2.line(frame, (x, y), (x + dx, y + dy), (15, 15, 18), 2)

            # Dashcam HUD Overlay
            cv2.putText(frame, f"ROADSENSE AI DASHCAM 01 | {time.strftime('%Y-%m-%d')} {format_timestamp(t)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"SPEED: 42 KM/H | GPS: 19.0760 N, 72.8777 E", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(frame)
    finally:
        out.release()

    # Transcode raw OpenCV output to browser-compliant H.264
    success = transcode_to_h264(raw_temp_path, output_path)
    if success:
        # Also preserve original mp4v master
        master_copy = output_path.parent / f"{output_path.stem}_orig_mp4v.mp4"
        if not master_copy.exists():
            import shutil
            shutil.copy2(str(raw_temp_path), str(master_copy))
        if raw_temp_path.exists():
            raw_temp_path.unlink()
        logger.info(f"Generated browser-playable H.264 demo dashcam video: {output_path} ({total_frames} frames, {duration_sec}s)")
    else:
        # Fallback to raw output if ffmpeg is unavailable
        if raw_temp_path.exists():
            raw_temp_path.rename(output_path)
        logger.warning(f"Generated un-transcoded demo video at {output_path}")

    return output_path
