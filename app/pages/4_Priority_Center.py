"""
RoadSense AI - Priority Center & Work Order Dispatch
Deterministic inspection priority ranking v1.0, interactive weight configurator,
human-in-the-loop verification, and municipal work order export.
"""

import io
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from datetime import datetime

from app import theme
from core import config
from core.db import get_all_incidents, update_review_status
from core.demo import seed_demo_incidents_if_empty
from core.logger import get_logger
from core.priority import calculate_priority, get_priority_level
from core.reports_pdf import build_work_orders_pdf

logger = get_logger("roadsense.ui.priority")

theme.setup_page(
    page_title="Priority Center | RoadSense AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Priority Center & Work Order Dispatch")
st.caption("Deterministic ranking engine v1.0, configurable municipal policy weights, and human-in-the-loop maintenance dispatch.")

# Civil Engineering Disclaimer
theme.render_safety_banner()

# Fetch incidents
raw_incidents = get_all_incidents()
if not raw_incidents:
    st.info("The incident database is empty. Click below to load calibrated survey incidents.")
    if st.button("🌱 Load Calibrated Demo Fleet Incidents", type="primary"):
        seed_demo_incidents_if_empty(force=True)
        st.success("Seeded calibrated incidents.")
        st.rerun()
    st.stop()

# Priority Formula Math Reference Card in Expander
with st.expander("📐 Deterministic Priority Formula Reference (v1.0)", expanded=False):
    st.latex(
        r"S = \min\left(100, \, (w_{\text{class}} \cdot \bar{c} \cdot 40) + (\min(N_{\text{cluster}}, 5) \cdot 6) + (w_{\text{road}} \cdot 15) + (w_{\text{human}} \cdot 15)\right)"
    )
    st.markdown("""
    - $w_{\\text{class}}$: Class severity weight (`D40` Pothole: 1.0, `D20` Alligator: 0.8, `D10` Transverse: 0.5, `D00` Longitudinal: 0.4).
    - $\\bar{c}$: AI detection confidence score ($c \\in [0, 1]$).
    - $N_{\\text{cluster}}$: Verified repeat detections within a 25-meter spatial radius (frequency multiplier, capped at 5).
    - $w_{\\text{road}}$: Road classification weight (`PRIMARY_ARTERIAL`: 1.0, `SECONDARY_COLLECTOR`: 0.7, `RESIDENTIAL_LOCAL`: 0.4).
    - $w_{\\text{human}}$: Civil inspector urgency rating ($w_{\\text{human}} \\in [0.5, 1.5]$, default 1.0).
    """)

# Two Column Layout: Left = Policy Weights Configurator, Right = Triage Queue & Dispatch
col_cfg, col_queue = st.columns([1, 2], gap="large")

with col_cfg:
    st.subheader("🛠️ Municipal Policy Tuner")
    st.caption("Adjust policy weights to simulate real-time queue re-ranking.")

    st.markdown("**Distress Severity Weights ($w_{\\text{class}}$)**")
    w_d40 = st.slider("Pothole (D40)", 0.5, 1.5, 1.0, 0.05)
    w_d20 = st.slider("Alligator Crack (D20)", 0.4, 1.2, 0.8, 0.05)
    w_d10 = st.slider("Transverse Crack (D10)", 0.2, 0.8, 0.5, 0.05)
    w_d00 = st.slider("Longitudinal Crack (D00)", 0.2, 0.8, 0.4, 0.05)

    custom_class_weights = {
        "D40": w_d40,
        "D20": w_d20,
        "D10": w_d10,
        "D00": w_d00,
    }

    st.markdown("**Road Classification Weights ($w_{\\text{road}}$)**")
    w_art = st.slider("Primary Arterial", 0.6, 1.5, 1.0, 0.1)
    w_col = st.slider("Secondary Collector", 0.4, 1.0, 0.7, 0.05)
    w_res = st.slider("Residential Local", 0.2, 0.8, 0.4, 0.05)

    custom_road_weights = {
        "PRIMARY_ARTERIAL": w_art,
        "SECONDARY_COLLECTOR": w_col,
        "RESIDENTIAL_LOCAL": w_res,
        "UNKNOWN": 0.5,
    }

    simulate_policy = st.toggle("Simulate Custom Policy Re-Ranking", value=False)

with col_queue:
    st.subheader("📋 Triage Queue")

    # Build queue dataframe
    queue_data = []
    for inc in raw_incidents:
        orig_score = float(inc.get("priority_score", 0.0))
        orig_level = str(inc.get("priority_level", "LOW")).upper()
        det_classes = inc.get("detected_classes", [])
        primary_class = det_classes[0] if det_classes else "UNKNOWN"
        conf = float(inc.get("max_confidence", 0.5))
        road_cls = inc.get("road_classification", "UNKNOWN")

        if simulate_policy:
            sim_res = calculate_priority(
                detected_classes=det_classes or [primary_class],
                max_confidence=conf,
                nearby_repeat_count=1,
                road_classification=road_cls,
                human_urgency_modifier=1.0,
            )
            score_to_use = sim_res.score
            level_to_use = sim_res.level
        else:
            score_to_use = orig_score
            level_to_use = orig_level

        queue_data.append({
            "UUID": inc.get("incident_uuid", ""),
            "Class": primary_class,
            "Description": config.CLASS_HUMAN_NAMES.get(primary_class, primary_class),
            "Confidence": f"{conf:.1%}",
            "Road": road_cls,
            "Score": round(score_to_use, 1),
            "Priority": level_to_use,
            "Status": inc.get("review_status", "PENDING"),
            "Latitude": f"{inc.get('latitude'):.5f}" if inc.get("latitude") else "N/A",
            "Longitude": f"{inc.get('longitude'):.5f}" if inc.get("longitude") else "N/A",
            "_raw_incident": inc
        })

    # Sort queue descending by score
    queue_data.sort(key=lambda x: x["Score"], reverse=True)

    df_queue = pd.DataFrame([{
        "Rank": i + 1,
        "Score": item["Score"],
        "Priority": item["Priority"],
        "Class": item["Class"],
        "Description": item["Description"],
        "Road": item["Road"],
        "Confidence": item["Confidence"],
        "Status": item["Status"],
        "UUID": item["UUID"][:8]
    } for i, item in enumerate(queue_data)])

    st.dataframe(
        df_queue,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Priority Score",
                help="Deterministic score (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100
            ),
        }
    )

    st.markdown("---")
    st.subheader("👷 Inspector Action on Top Item")
    if queue_data:
        top_item = queue_data[0]
        raw_inc = top_item["_raw_incident"]
        inc_uuid = raw_inc.get("incident_uuid", "")

        st.markdown(
            f"**Currently Focused:** Rank #1 — `{top_item['Class']}` ({top_item['Description']}) "
            f"on `{top_item['Road']}` with Priority **{top_item['Priority']}** ({top_item['Score']}/100)"
        )

        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            if st.button("✅ Verify & Approve Dispatch", type="primary", use_container_width=True):
                update_review_status(inc_uuid, "VERIFIED", reviewer_notes="Priority verified by engineering team.")
                st.success(f"Incident {inc_uuid[:8]} marked as VERIFIED.")
                st.rerun()

        with h_col2:
            if st.button("⏳ Keep Pending", use_container_width=True):
                st.info("Kept in pending queue.")

        with h_col3:
            if st.button("❌ Dismiss Defect", use_container_width=True):
                update_review_status(inc_uuid, "REJECTED", reviewer_notes="Dismissed by engineering inspector.")
                st.warning(f"Incident {inc_uuid[:8]} rejected.")
                st.rerun()

    st.markdown("---")
    st.subheader("📦 Municipal Work Order Dispatch Export")
    st.caption("Generate a formatted work order package for road maintenance crews.")

    SLA_MAP = {
        "CRITICAL": "24 to 48 Hours (Immediate Hazard)",
        "HIGH": "7 Days (Scheduled Repair)",
        "MODERATE": "30 Days (Routine Backlog)",
        "LOW": "Subsequent Survey Cycle (Monitor)"
    }

    ACTION_MAP = {
        "D40": "Cold/Hot Asphalt Patch Compaction",
        "D20": "Milling and Asphalt Overlay",
        "D10": "Hot-Pour Bituminous Crack Sealing",
        "D00": "Linear Crack Sealant Application"
    }

    work_orders = []
    for item in queue_data:
        if item["Status"] != "REJECTED":
            work_orders.append({
                "Work_Order_ID": f"WO-{item['UUID'][:8].upper()}",
                "Incident_UUID": item["UUID"],
                "Priority_Tier": item["Priority"],
                "Priority_Score": item["Score"],
                "Pavement_Distress": item["Description"],
                "Distress_Code": item["Class"],
                "Prescribed_Action": ACTION_MAP.get(item["Class"], "Pavement Assessment"),
                "Response_SLA": SLA_MAP.get(item["Priority"], "Routine Maintenance"),
                "Road_Classification": item["Road"],
                "Latitude": item["Latitude"],
                "Longitude": item["Longitude"],
                "Review_Status": item["Status"]
            })

    df_wo = pd.DataFrame(work_orders)
    csv_buffer = io.StringIO()
    df_wo.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode("utf-8")

    # Generate genuine Landscape A4 enterprise PDF
    raw_wo_incidents = [item["_raw_incident"] for item in queue_data if item["Status"] != "REJECTED"]
    pdf_bytes = build_work_orders_pdf(
        raw_wo_incidents,
        filter_desc=f"Active Triage Queue ({len(raw_wo_incidents)} orders, Non-Rejected)"
    )

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_btn_pdf, col_btn_csv = st.columns([1.4, 1.0])
    with col_btn_pdf:
        st.download_button(
            label=f"📄 Dispatch Work Orders (PDF) ({len(work_orders)} active jobs)",
            data=pdf_bytes,
            file_name=f"roadsense_work_orders_{now_str}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn_csv:
        st.download_button(
            label=f"📊 Export data (CSV) ({len(work_orders)} jobs)",
            data=csv_data,
            file_name=f"roadsense_work_orders_{now_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
