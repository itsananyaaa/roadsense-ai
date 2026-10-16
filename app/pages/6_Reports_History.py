"""
RoadSense AI - Reports & Audit History
Municipal defect reporting, compliance audit log, and multi-format data export (CSV, GeoJSON).
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from datetime import datetime

from app import theme
from core import config
from core.db import get_all_incidents
from core.demo import seed_demo_incidents_if_empty
from core.logger import get_logger
from core.reports_pdf import build_incident_report_pdf

logger = get_logger("roadsense.ui.reports")

theme.setup_page(
    page_title="Reports & History | RoadSense AI",
    page_icon="📑",
    layout="wide"
)

st.title("📑 Reports, Inventory & Audit History")
st.caption("Municipal asset management records, GIS spatial exports, and compliance audit tracking.")

# Civil Engineering Disclaimer
theme.render_safety_banner()

incidents = get_all_incidents()
if not incidents:
    st.info("The incident database is empty. Load calibrated survey records below:")
    if st.button("🌱 Load Calibrated Survey Incidents", type="primary"):
        seed_demo_incidents_if_empty(force=True)
        st.success("Seeded calibrated survey incidents.")
        st.rerun()
    st.stop()

# 1. High-Level Summary Metric Cards
total_count = len(incidents)
critical_count = sum(1 for inc in incidents if str(inc.get("priority_level", "")).upper() == "CRITICAL")
verified_count = sum(1 for inc in incidents if str(inc.get("review_status", "")).upper() == "VERIFIED")
pending_count = sum(1 for inc in incidents if str(inc.get("review_status", "")).upper() == "PENDING")

st.markdown("### 📊 Network Asset Summary")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Recorded Incidents", f"{total_count}")
with m2:
    st.metric("Critical Hazards (Score ≥75)", f"{critical_count}")
with m3:
    st.metric("Engineering Verified", f"{verified_count}")
with m4:
    st.metric("Pending Triage", f"{pending_count}")

st.markdown("---")

# 2. Interactive Search & Filterable Table
st.subheader("📋 Pavement Distress Inventory")

search_term = st.text_input("🔎 Search by UUID, Road Name, or Distress Class:", "")

table_records = []
for inc in incidents:
    uuid_str = inc.get("incident_uuid", "")
    det_classes = inc.get("detected_classes", [])
    primary_class = det_classes[0] if det_classes else "UNKNOWN"
    road_cls = inc.get("road_classification", "UNKNOWN")
    notes_str = inc.get("reviewer_notes", "") or ""

    if search_term:
        term = search_term.lower()
        if (term not in uuid_str.lower() and
            term not in primary_class.lower() and
            term not in road_cls.lower() and
            term not in notes_str.lower()):
            continue

    conf = float(inc.get("max_confidence", 0.0))
    lat = inc.get("latitude")
    lon = inc.get("longitude")

    table_records.append({
        "Incident UUID": uuid_str,
        "Distress Class": primary_class,
        "Human Name": config.CLASS_HUMAN_NAMES.get(primary_class, primary_class),
        "Confidence": f"{conf:.1%}",
        "Priority Tier": inc.get("priority_level", "LOW"),
        "Priority Score": round(float(inc.get("priority_score", 0.0)), 1),
        "Road Classification": road_cls,
        "Latitude": f"{lat:.5f}" if lat is not None else "N/A",
        "Longitude": f"{lon:.5f}" if lon is not None else "N/A",
        "Location Provenance": inc.get("location_provenance", "UNKNOWN"),
        "Review Status": inc.get("review_status", "PENDING"),
        "Recorded At": inc.get("timestamp", ""),
        "Inspector Notes": notes_str
    })

df_table = pd.DataFrame(table_records)

st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Priority Score": st.column_config.ProgressColumn(
            "Priority Score",
            min_value=0,
            max_value=100,
            format="%.1f"
        ),
    }
)

st.markdown("---")

# 3. Export Formats (PDF Formal Report, CSV & GeoJSON Data)
st.subheader("📦 Data Export Center")
st.caption("Human-readable municipal audit reports (PDF) and machine-readable GIS datasets (CSV, GeoJSON).")

exp_col1, exp_col2, exp_col3 = st.columns(3)

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

with exp_col1:
    theme.render_card_start(title="Export Audit Report (PDF)")
    st.markdown("Formal landscape A4 engineering report with summary KPIs, repeated headers, and civil safety notices.")

    pdf_bytes = build_incident_report_pdf(
        incidents,
        filter_desc=f"Full Network Ledger ({len(incidents)} total defects)"
    )

    st.download_button(
        label=f"📄 Download Formal Audit PDF ({len(incidents)} items)",
        data=pdf_bytes,
        file_name=f"roadsense_incident_report_{now_str}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )
    theme.render_card_end()

with exp_col2:
    theme.render_card_start(title="Export Ledger (CSV) (data)")
    st.markdown("Full incident inventory with coordinates, provenance tags, bounding boxes, and formula scores.")

    csv_buffer = io.StringIO()
    df_table.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label=f"📊 Download Incident CSV ({len(df_table)} records)",
        data=csv_bytes,
        file_name=f"roadsense_inventory_{now_str}.csv",
        mime="text/csv",
        use_container_width=True
    )
    theme.render_card_end()

with exp_col3:
    theme.render_card_start(title="Export GeoJSON (GIS) (data)")
    st.markdown("RFC 7946 compliant GeoJSON FeatureCollection with Point geometry for ArcGIS, QGIS, or Mapbox.")

    geojson_features = []
    for inc in incidents:
        lat = inc.get("latitude")
        lon = inc.get("longitude")
        prov = inc.get("location_provenance", "UNKNOWN")

        if lat is not None and lon is not None and prov != "UNKNOWN":
            det_classes = inc.get("detected_classes", [])
            primary_class = det_classes[0] if det_classes else "UNKNOWN"
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                },
                "properties": {
                    "id": inc.get("incident_uuid", ""),
                    "damage_class": primary_class,
                    "human_name": config.CLASS_HUMAN_NAMES.get(primary_class, primary_class),
                    "confidence": float(inc.get("max_confidence", 0.0)),
                    "priority_score": float(inc.get("priority_score", 0.0)),
                    "priority_level": inc.get("priority_level", "LOW"),
                    "road_classification": inc.get("road_classification", "UNKNOWN"),
                    "location_provenance": prov,
                    "review_status": inc.get("review_status", "PENDING"),
                    "timestamp": inc.get("timestamp", ""),
                    "notes": inc.get("reviewer_notes", "")
                }
            }
            geojson_features.append(feature)

    geojson_doc = {
        "type": "FeatureCollection",
        "features": geojson_features,
        "metadata": {
            "generator": "RoadSense AI Municipal Engine",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_count": len(geojson_features)
        }
    }

    geojson_str = json.dumps(geojson_doc, indent=2).encode("utf-8")

    st.download_button(
        label=f"🗺️ Download GeoJSON ({len(geojson_features)} points)",
        data=geojson_str,
        file_name=f"roadsense_spatial_{now_str}.geojson",
        mime="application/geo+json",
        use_container_width=True
    )
    theme.render_card_end()
