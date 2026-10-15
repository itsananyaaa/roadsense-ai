"""
RoadSense AI - Video Intelligence Engine
Interactive dashcam footage analysis with configurable sampling rate,
temporal de-duplication, keyframe extraction, and batch incident logging.
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import streamlit as st
from PIL import Image

from app import theme
from core import config
from core.db import insert_incident, get_incident_count
from core.geo import LocationProvenance
from core.health import check_checkpoint_status
from core.inference import DamageDetector
from core.logger import get_logger
from core.video import VideoProcessor, generate_demo_dashcam_video, format_timestamp

logger = get_logger("roadsense.ui.video")

# Apply consistent dark slate / safety amber theme
theme.setup_page(
    page_title="Video Intelligence | RoadSense AI",
    page_icon="🎬",
    layout="wide"
)
theme.apply_theme()

st.title("🎬 Dashcam Video Intelligence Engine")
st.caption(
    "Automated frame sampling, temporal distress de-duplication, and municipal keyframe triage for continuous road inspection streams."
)

# Render Civil Engineering Disclaimer
theme.render_safety_banner()

# Global Model & Hardware Status
model_path, model_status = check_checkpoint_status()
theme.render_status_badge(model_status, extra_info=f"Active: {Path(model_path).name}")

st.markdown("---")

# Layout: Configuration Sidebar / Column and Main View
col_cfg, col_main = st.columns([1, 2], gap="large")

with col_cfg:
    st.subheader("⚙️ Stream Sampling Configuration")
    sample_fps = st.slider(
        "Sampling Rate (Frames per Second)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Rate at which video frames are extracted for inference. 1.0 FPS samples 1 frame every second."
    )

    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.15,
        max_value=0.85,
        value=0.35,
        step=0.05,
        help="Minimum confidence threshold required to record a pavement distress event."
    )

    max_duration = st.slider(
        "Max Video Duration to Analyze (Seconds)",
        min_value=3,
        max_value=60,
        value=15,
        step=3,
        help="Limits processing length for responsive interactive demonstration."
    )

    road_classification = st.selectbox(
        "Survey Road Classification",
        options=["PRIMARY_ARTERIAL", "SECONDARY_COLLECTOR", "RESIDENTIAL_LOCAL"],
        index=0,
        help="Road classification used in deterministic priority ranking."
    )

    st.markdown("---")
    st.subheader("📹 Video Source")
    video_source_mode = st.radio(
        "Select Video Input Mode",
        options=["Use Calibrated Demo Dashcam Stream", "Upload Dashcam MP4/AVI"],
        index=0
    )

    uploaded_video = None
    if video_source_mode == "Upload Dashcam MP4/AVI":
        uploaded_video = st.file_uploader(
            "Upload dashcam footage (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov"],
            help="Maximum video file size: 50 MB"
        )
    else:
        st.info("💡 **Calibrated Demo Stream:** Generates a synthetic vehicle route with approaching potholes and cracks for instant local demonstration.")

with col_main:
    st.subheader("📊 Analysis Execution & Results")

    # Prepare video file
    video_to_process: Optional[Path] = None

    if video_source_mode == "Use Calibrated Demo Dashcam Stream":
        demo_vid_path = config.DEMO_DIR / "dashcam_inspection_demo.mp4"
        if st.button("🚀 Generate & Analyze Demo Stream", type="primary", use_container_width=True):
            with st.spinner("Generating calibrated dashcam footage..."):
                generate_demo_dashcam_video(demo_vid_path, duration_sec=max_duration, fps=15)
                video_to_process = demo_vid_path
    elif uploaded_video is not None:
        if st.button("🚀 Run Video Intelligence Analysis", type="primary", use_container_width=True):
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
            tfile.write(uploaded_video.read())
            tfile.flush()
            video_to_process = Path(tfile.name)

    if video_to_process:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(prog: float, msg: str):
            progress_bar.progress(prog)
            status_text.text(f"⏳ {msg}")

        try:
            vp = VideoProcessor()
            report = vp.analyze_video(
                video_path=video_to_process,
                sample_fps=sample_fps,
                conf_threshold=conf_threshold,
                progress_callback=update_progress,
                max_duration_sec=float(max_duration)
            )

            status_text.success(f"✅ Video processing complete in {report.processing_time_sec:.2f} seconds!")
            progress_bar.progress(1.0)

            # Store report in session state
            st.session_state["latest_video_report"] = report

        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            status_text.error(f"❌ Video analysis failed: {str(e)}")

    # Display Report if available in session state
    if "latest_video_report" in st.session_state:
        rep = st.session_state["latest_video_report"]

        st.markdown("### 📈 Survey Stream Summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Video Time", f"{rep.duration_sec:.1f}s")
        with m2:
            st.metric("Sampled Frames", f"{rep.sampled_frames_count}")
        with m3:
            st.metric("Processing Speed", f"{(rep.sampled_frames_count / rep.processing_time_sec):.1f} FPS")
        with m4:
            st.metric("Distinct Incidents", f"{rep.distinct_incidents_count}")

        st.caption(
            "ℹ️ **Temporal De-Duplication Active:** Multiple consecutive frames observing the same defect are grouped into a single physical incident to prevent duplicate work order inflation."
        )

        # Damage breakdown
        st.markdown("#### 🔍 Detected Distress Breakdown")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Longitudinal (D00)", rep.damage_distribution.get("D00", 0))
        with b2:
            st.metric("Transverse (D10)", rep.damage_distribution.get("D10", 0))
        with b3:
            st.metric("Alligator (D20)", rep.damage_distribution.get("D20", 0))
        with b4:
            st.metric("Potholes (D40)", rep.damage_distribution.get("D40", 0))

        # Keyframe inspection timeline
        st.markdown("---")
        st.markdown(f"### 🖼️ Keyframe Evidence Gallery ({len(rep.keyframe_events)} keyframes)")

        if rep.keyframe_events:
            for idx, kf in enumerate(rep.keyframe_events):
                with st.expander(f"Keyframe #{idx+1} @ {kf.timestamp_formatted} — {config.CLASS_HUMAN_NAMES.get(kf.max_severity_class, kf.max_severity_class)} ({kf.priority_level})", expanded=(idx == 0)):
                    k_col1, k_col2 = st.columns([1, 1])
                    with k_col1:
                        if Path(kf.keyframe_path).exists():
                            st.image(kf.keyframe_path, caption=f"Keyframe @ {kf.timestamp_formatted}", use_container_width=True)
                    with k_col2:
                        st.markdown(f"**Primary Distress:** `{kf.max_severity_class}` ({config.CLASS_HUMAN_NAMES.get(kf.max_severity_class, kf.max_severity_class)})")
                        st.markdown(f"**Peak Confidence:** `{kf.max_confidence:.1%}`")
                        st.markdown(f"**Deterministic Priority Score:** `{kf.priority_score} / 100`")
                        st.markdown(f"**Priority Tier:** `{kf.priority_level}`")
                        st.markdown(f"**Total Detections in Frame:** `{len(kf.detections)}`")

            # Batch save button
            st.markdown("---")
            if st.button("💾 Commit All Detected Incidents to Incident Database", type="secondary", use_container_width=True):
                saved_count = 0
                for kf in rep.keyframe_events:
                    try:
                        # Synthetic calibrated GPS for demo stream
                        lat = 19.0760 + (kf.timestamp_sec * 0.0001)
                        lon = 72.8777 + (kf.timestamp_sec * 0.0001)

                        insert_incident(
                            image_path=kf.keyframe_path,
                            checkpoint_id=model_path,
                            detected_classes=[kf.max_severity_class],
                            detection_count=len(kf.detections),
                            max_confidence=kf.max_confidence,
                            raw_detections=[d.to_dict() for d in kf.detections],
                            priority_score=kf.priority_score,
                            priority_level=kf.priority_level,
                            latitude=lat,
                            longitude=lon,
                            location_provenance=LocationProvenance.EXPLICIT_SYNTHETIC_DEMO.value if video_source_mode == "Use Calibrated Demo Dashcam Stream" else LocationProvenance.UNKNOWN.value,
                            road_name="Dashcam Survey Route",
                            road_classification=road_classification,
                        )
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to persist keyframe incident: {e}")

                st.success(f"🎉 Successfully committed {saved_count} distinct road incidents into SQLite database! View on the Incident Map or Priority Center.")
        else:
            st.info("No significant pavement distress detected above the confidence threshold.")
