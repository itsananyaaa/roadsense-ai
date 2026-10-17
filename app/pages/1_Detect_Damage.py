"""
RoadSense AI - Phase 3 Vertical Slice: Damage Detection
Workflow: Upload Image -> Validate -> Real Inference -> Annotated Image ->
          Detection Table -> Save Incident to SQLite -> Show Saved Record.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from core import config
from core.db import init_db, insert_incident, get_incident_by_uuid, count_nearby_incidents
from core.geo import extract_exif_gps, format_coordinates
from core.health import run_health_check
from core.inference import DamageDetector, ModelWeightsNotFoundError, InvalidImageInputError
from core.priority import calculate_priority
from app.theme import apply_theme, render_brand_header, render_safety_notice


def init_page() -> None:
    st.set_page_config(
        page_title="Detect Damage — RoadSense AI",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    render_brand_header()
    init_db()


def main() -> None:
    init_page()

    # Health & model status
    health = run_health_check()
    ckpt_status = health.checkpoint_status

    st.title("Detect Road Damage")
    st.markdown(
        "Upload a road surface photo or select a verified test fixture. "
        "The system executes real object detection, extracts spatial provenance, "
        "computes deterministic inspection priority, and commits an auditable record to SQLite."
    )

    # Active Model Status Banner
    col_stat1, col_stat2 = st.columns([3, 1])
    with col_stat1:
        if ckpt_status == "EVALUATED_TRAINED":
            st.info(f"🟢 **Active Model:** Fine-Tuned Road Damage Detector (`{Path(health.active_checkpoint_path).name}`) | Device: `{health.accelerator}`")
        elif ckpt_status == "SMOKE_CHECKPOINT":
            st.warning(f"🟡 **Active Model:** Smoke-Test Checkpoint (`{Path(health.active_checkpoint_path).name}`, 1-2 epochs; not fully evaluated) | Device: `{health.accelerator}`")
        elif ckpt_status == "PRETRAINED_BASELINE":
            st.warning(f"🟠 **Active Model:** Pretrained Baseline (`{Path(health.active_checkpoint_path).name}`, COCO weights; not road-fine-tuned) | Device: `{health.accelerator}`")
        else:
            st.error("🔴 **Active Model:** No model checkpoint found. Demo mode available.")

    render_safety_notice()

    # Sidebar Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Inference Settings")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum model confidence required to accept a bounding box detection.",
    )

    # Image Acquisition: File Upload or Sample Selector
    st.markdown("### 1. Image Acquisition")
    tab_upload, tab_sample = st.tabs(["📁 Upload Image File", "🧪 Select Benchmark Sample"])

    selected_image: Image.Image = None
    image_source_name: str = "Uploaded Image"
    image_temp_path: Path = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a road photo (JPEG, PNG, WebP — max 25 MB):",
            type=["jpg", "jpeg", "png", "webp"],
            key="file_uploader_detect",
        )
        if uploaded_file is not None:
            try:
                # Validation: file size
                if uploaded_file.size > config.MAX_IMAGE_SIZE_BYTES:
                    st.error(f"File size exceeds maximum {config.MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB limit.")
                else:
                    selected_image = Image.open(uploaded_file)
                    image_source_name = uploaded_file.name
                    # Save temporary copy for EXIF extraction
                    image_temp_path = config.EVIDENCE_DIR / f"temp_{uploaded_file.name}"
                    uploaded_file.seek(0)
                    with open(image_temp_path, "wb") as f:
                        f.write(uploaded_file.read())
            except Exception as e:
                st.error(f"Invalid or corrupted image file: {e}")

    with tab_sample:
        from core.demo import get_demo_tiles
        tiles = get_demo_tiles()
        if not tiles:
            st.caption("No sample images in manifest. Run `python ml/build_demo_media.py` first.")
        else:
            tile_options = {f"{t['caption']} ({t['filename']})": t for t in tiles}
            chosen_label = st.selectbox("Choose benchmark fixture:", list(tile_options.keys()), key="sel_sample")
            chosen_tile = tile_options[chosen_label]
            fname = chosen_tile["filename"]

            cand1 = config.DEMO_DIR / fname
            cand2 = config.DATA_DIR / "demo_media" / "images" / fname
            image_temp_path = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)

            if image_temp_path and image_temp_path.exists():
                selected_image = Image.open(image_temp_path)
                image_source_name = fname
                st.info(f"🏷️ **Provenance:** `{chosen_tile.get('badge')}` | Origin: `{chosen_tile.get('origin')}`")

                gt_boxes = chosen_tile.get("ground_truth_boxes", [])
                if gt_boxes:
                    show_gt = st.checkbox(f"Show Ground-Truth Annotations ({len(gt_boxes)} boxes from RDD2022 XML)", value=False, key="cb_show_gt")
                    st.session_state["show_gt"] = show_gt
                    st.session_state["gt_boxes"] = gt_boxes
                else:
                    st.session_state["show_gt"] = False
                    st.session_state["gt_boxes"] = []

    if selected_image is None:
        st.info("👆 Please upload an image or select a benchmark sample above to begin analysis.")
        return

    # Metadata & Location Provenance
    st.markdown("### 2. Location & Road Context")
    lat_exif, lon_exif, prov_exif = (None, None, "UNKNOWN")
    if image_temp_path and image_temp_path.exists():
        lat_exif, lon_exif, prov_exif = extract_exif_gps(str(image_temp_path))

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        if prov_exif == "EXIF_GPS":
            st.success(f"📍 GPS Extracted: {lat_exif:.5f}, {lon_exif:.5f} (EXIF)")
            use_exif = st.checkbox("Use EXIF GPS", value=True)
            active_lat = lat_exif if use_exif else None
            active_lon = lon_exif if use_exif else None
            active_prov = "EXIF_GPS" if use_exif else "UNKNOWN"
        else:
            st.caption("No EXIF GPS metadata found in file.")
            manual_coords = st.text_input("Enter GPS (lat, lon) or leave empty:", placeholder="e.g. 12.9716, 77.5946")
            active_lat, active_lon, active_prov = None, None, "UNKNOWN"
            if manual_coords and "," in manual_coords:
                try:
                    parts = manual_coords.split(",")
                    active_lat = float(parts[0].strip())
                    active_lon = float(parts[1].strip())
                    active_prov = "USER_ENTERED"
                    st.caption(f"Location Provenance: **USER_ENTERED**")
                except ValueError:
                    st.warning("Invalid coordinate format. Ignored.")

    with col_meta2:
        road_name = st.text_input("Road / Street Name:", value="Main Arterial Corridor")

    with col_meta3:
        road_class = st.selectbox(
            "Road Classification:",
            ["PRIMARY_ARTERIAL", "SECONDARY_COLLECTOR", "RESIDENTIAL_LOCAL", "UNKNOWN"],
            index=0,
            help="Higher classification carries greater traffic volume and higher priority impact.",
        )

    # 3. Real Inference Execution
    st.markdown("### 3. Real Object Detection Inference")
    if st.button("🚀 Analyze Road Surface", type="primary", use_container_width=True, key="btn_run_inference"):
        with st.spinner("Executing YOLO detection model..."):
            try:
                detector = DamageDetector.get_instance()
                detection_result = detector.predict(selected_image, conf_threshold=conf_thresh)
                st.session_state["last_result"] = detection_result
                st.session_state["analyzed_image"] = selected_image
                st.session_state["image_source_name"] = image_source_name
                st.session_state["active_lat"] = active_lat
                st.session_state["active_lon"] = active_lon
                st.session_state["active_prov"] = active_prov
                st.session_state["road_name"] = road_name
                st.session_state["road_class"] = road_class
            except ModelWeightsNotFoundError as e:
                st.error(f"Inference Blocked: {e}")
                return
            except Exception as e:
                st.error(f"Inference Execution Error: {e}")
                return

    # Display Results if Available
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        img_orig = st.session_state["analyzed_image"]
        a_lat = st.session_state.get("active_lat")
        a_lon = st.session_state.get("active_lon")
        a_prov = st.session_state.get("active_prov", "UNKNOWN")
        r_name = st.session_state.get("road_name", "Unknown")
        r_class = st.session_state.get("road_class", "UNKNOWN")

        # Side-by-side image comparison
        st.markdown("#### Visual Analysis")
        gt_boxes = st.session_state.get("gt_boxes", [])
        if st.session_state.get("show_gt") and gt_boxes:
            c_orig, c_gt, c_anno = st.columns(3)
            # Render GT annotated copy
            gt_img = np.array(img_orig).copy()
            if len(gt_img.shape) == 2:
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
            elif gt_img.shape[2] == 4:
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGBA2RGB)
            for box in gt_boxes:
                b = box.get("box", [])
                c = box.get("class", "GT")
                if len(b) == 4:
                    cv2.rectangle(gt_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
                    cv2.putText(gt_img, f"GT: {c}", (int(b[0]), max(15, int(b[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            with c_orig:
                st.image(img_orig, caption=f"Original Input: {st.session_state['image_source_name']}", use_container_width=True)
            with c_gt:
                st.image(gt_img, caption=f"Ground-Truth Annotations ({len(gt_boxes)} boxes)", use_container_width=True)
            with c_anno:
                anno_rgb = cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB)
                st.image(anno_rgb, caption=f"Model Prediction ({res.checkpoint_id} | {res.inference_latency_ms:.1f} ms)", use_container_width=True)
        else:
            c_orig, c_anno = st.columns(2)
            with c_orig:
                st.image(img_orig, caption=f"Original Input: {st.session_state['image_source_name']}", use_container_width=True)
            with c_anno:
                anno_rgb = cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB)
                st.image(anno_rgb, caption=f"Annotated Output (Model: {res.checkpoint_id} | {res.inference_latency_ms:.1f} ms)", use_container_width=True)

        # Detection Summary
        st.markdown("#### Detection Findings")
        if res.detection_count == 0:
            st.success("✅ **No Road Distress Detected.** Pavement surface appears undamaged above selected confidence threshold.")
        else:
            det_rows = []
            for d in res.detections:
                det_rows.append({
                    "Class Code": d.class_code,
                    "Damage Type": d.class_name,
                    "Confidence": f"{d.confidence * 100:.1f}%",
                    "Bounding Box [x1, y1, x2, y2]": str(d.bbox),
                })
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True)

        # Calculate Priority Score
        repeat_count = 0
        if a_lat is not None and a_lon is not None:
            repeat_count = count_nearby_incidents(a_lat, a_lon, radius_meters=config.CLUSTER_RADIUS_METERS)

        prio_result = calculate_priority(
            detected_classes=res.detected_classes,
            max_confidence=res.max_confidence,
            nearby_repeat_count=repeat_count,
            road_classification=r_class,
            human_urgency_modifier=1.0,
        )

        st.markdown("#### Inspection Priority Assessment")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            lvl_color = config.PRIORITY_LEVELS.get(prio_result.level, {}).get("color_hex", "#475569")
            st.markdown(
                f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-value" style="color: {lvl_color};">{prio_result.score:.1f} / 100</div>
                    <div class="rs-kpi-label">Priority Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col2:
            st.markdown(
                f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-value" style="color: {lvl_color};">{prio_result.level}</div>
                    <div class="rs-kpi-label">Proposed Action Band</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col3:
            st.markdown(
                f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-value">{res.inference_latency_ms:.1f} ms</div>
                    <div class="rs-kpi-label">Inference Latency</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col4:
            st.markdown(
                f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-value">{repeat_count}</div>
                    <div class="rs-kpi-label">Repeat Defect Density (25m)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(f"**Triage Rationale:** {prio_result.rationale}")

        # 4. Save to Incident Log (SQLite Persistence)
        st.markdown("### 4. Commit to Incident Evidence Store")
        if st.button("💾 Save to Incident Log (SQLite)", type="primary", use_container_width=True, key="btn_save_incident"):
            # Save annotated evidence image to disk
            evidence_filename = f"evidence_{int(cv2.getTickCount())}.jpg"
            evidence_path = config.EVIDENCE_DIR / evidence_filename
            cv2.imwrite(str(evidence_path), res.annotated_image_bgr)

            # Insert into SQLite
            raw_dets = [d.to_dict() for d in res.detections]
            inc_uuid = insert_incident(
                image_path=str(evidence_path),
                checkpoint_id=res.checkpoint_id,
                detected_classes=res.detected_classes,
                detection_count=res.detection_count,
                max_confidence=res.max_confidence,
                raw_detections=raw_dets,
                priority_score=prio_result.score,
                priority_level=prio_result.level,
                latitude=a_lat,
                longitude=a_lon,
                location_provenance=a_prov,
                road_name=r_name,
                road_classification=r_class,
            )

            st.session_state["saved_uuid"] = inc_uuid
            st.success(f"🎉 **Incident Committed Successfully:** `{inc_uuid}`")

        # 5. Show Saved Incident Record
        if "saved_uuid" in st.session_state:
            s_uuid = st.session_state["saved_uuid"]
            record = get_incident_by_uuid(s_uuid)
            if record:
                st.markdown("### 5. Persisted Incident Record")
                st.markdown(
                    f"""
                    <div class="rs-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                            <strong style="font-size: 1.1rem; color: #0f172a;">Incident ID: {record['incident_uuid']}</strong>
                            <span class="rs-badge rs-badge-{record['priority_level'].lower()}">{record['priority_level']} PRIORITY</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 0.88rem; color: #334155;">
                            <div><strong>Timestamp:</strong> {record['timestamp']}</div>
                            <div><strong>Road:</strong> {record['road_name']} ({record['road_classification']})</div>
                            <div><strong>Location:</strong> {format_coordinates(record['latitude'], record['longitude'], record['location_provenance'])}</div>
                            <div><strong>Checkpoint:</strong> {record['checkpoint_id']}</div>
                            <div><strong>Damage Classes:</strong> {', '.join(record['detected_classes']) or 'None (Clean)'}</div>
                            <div><strong>Max Confidence:</strong> {record['max_confidence']*100:.1f}%</div>
                            <div><strong>Review Status:</strong> <code>{record['review_status']}</code></div>
                            <div><strong>Evidence File:</strong> <code>{Path(record['image_path']).name}</code></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
