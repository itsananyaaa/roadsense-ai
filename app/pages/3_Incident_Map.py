"""
RoadSense AI - Geospatial Incident Map & Explorer
Interactive Folium geospatial map with priority markers, evidence drawer,
strict location provenance enforcement, and graceful offline fallback.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from app import theme
from core import config
from core.db import get_all_incidents, update_review_status, get_incident_by_uuid
from core.demo import seed_demo_incidents_if_empty
from core.geo import LocationProvenance
from core.logger import get_logger

logger = get_logger("roadsense.ui.map")

# Setup page and theme
theme.setup_page(
    page_title="Incident Map | RoadSense AI",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Geospatial Incident Map & Explorer")
st.caption("Municipal spatial triage, defect clustering, and evidence inspection across urban road networks.")

# Civil Engineering Disclaimer
theme.render_safety_banner()

# Fetch incidents from database
raw_incidents = get_all_incidents()

if not raw_incidents:
    st.info("ℹ️ The incident database is currently empty. Click below to load calibrated municipal fleet survey incidents.")
    if st.button("🌱 Load Calibrated Demo Fleet Incidents", type="primary"):
        seed_demo_incidents_if_empty(force=True)
        st.success("Seeded calibrated survey incidents with verified location coordinates!")
        st.rerun()
    st.stop()

# Filter Controls Bar
st.markdown("### 🔍 Spatial Filter Controls")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    class_filter = st.selectbox(
        "Damage Category",
        options=["ALL", "D00 (Longitudinal)", "D10 (Transverse)", "D20 (Alligator)", "D40 (Pothole)"],
        index=0
    )

with f_col2:
    priority_filter = st.selectbox(
        "Priority Tier",
        options=["ALL", "CRITICAL", "HIGH", "MODERATE", "LOW"],
        index=0
    )

with f_col3:
    status_filter = st.selectbox(
        "Review Status",
        options=["ALL", "PENDING", "VERIFIED", "DISMISSED"],
        index=0
    )

with f_col4:
    map_engine = st.selectbox(
        "Map Display Engine",
        options=["Interactive Leaflet/Folium", "Offline Plotly Scatter (No Tile Dependency)"],
        index=0,
        help="Switch to Offline Plotly mode if external map tiles are unreachable or for low-bandwidth expo presentations."
    )

# Process and apply filters
filtered_incidents = []
unlocated_count = 0

for item in raw_incidents:
    det_classes = item.get("detected_classes", [])
    primary_class = det_classes[0] if det_classes else "UNKNOWN"
    prio_level = item.get("priority_level", "LOW").upper()
    rev_status = item.get("review_status", "PENDING").upper()
    lat = item.get("latitude")
    lon = item.get("longitude")
    prov = item.get("location_provenance", "UNKNOWN")

    # Filter class
    if class_filter != "ALL":
        code = class_filter.split()[0]
        if primary_class != code:
            continue

    # Filter priority
    if priority_filter != "ALL" and prio_level != priority_filter:
        continue

    # Filter status
    if status_filter != "ALL" and rev_status != status_filter:
        continue

    # Strict provenance check: Do not map incidents lacking valid coordinates
    if lat is None or lon is None or prov == LocationProvenance.UNKNOWN.value:
        unlocated_count += 1
        continue

    filtered_incidents.append({
        "id": item.get("incident_uuid", ""),
        "primary_class": primary_class,
        "priority_level": prio_level,
        "priority_score": float(item.get("priority_score", 0.0)),
        "confidence": float(item.get("max_confidence", 0.0)),
        "road_classification": item.get("road_classification", "UNKNOWN"),
        "review_status": rev_status,
        "latitude": float(lat),
        "longitude": float(lon),
        "location_provenance": prov,
        "timestamp": item.get("timestamp", ""),
        "image_path": item.get("image_path", ""),
        "_raw": item
    })

if unlocated_count > 0:
    st.warning(
        f"⚠️ **Strict Location Provenance Active:** {unlocated_count} recorded incident(s) lack verified GPS coordinates and are strictly excluded from map rendering (never fabricating estimated coordinates)."
    )

st.markdown("---")

# Main Content Layout: Map on left (70%), Evidence Drawer on right (30%)
col_map, col_drawer = st.columns([7, 3], gap="medium")

# Color map for priorities
PRIORITY_COLORS = {
    "CRITICAL": "#ef4444",   # Hazard Red
    "HIGH": "#f59e0b",       # Amber
    "MODERATE": "#fb923c",   # Orange
    "LOW": "#38bdf8",        # Sky Blue
}

FOLIUM_ICON_COLORS = {
    "CRITICAL": "red",
    "HIGH": "orange",
    "MODERATE": "beige",
    "LOW": "blue",
}

selected_incident_id = st.session_state.get("selected_map_incident_id", None)

with col_map:
    st.subheader(f"📍 Network Damage Distribution ({len(filtered_incidents)} Mapped Defects)")

    if not filtered_incidents:
        st.info("No incidents match the active filter criteria. Adjust the filters above.")
    elif map_engine == "Interactive Leaflet/Folium":
        avg_lat = sum(inc["latitude"] for inc in filtered_incidents) / len(filtered_incidents)
        avg_lon = sum(inc["longitude"] for inc in filtered_incidents) / len(filtered_incidents)

        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=13,
            tiles="OpenStreetMap"
        )

        cluster = MarkerCluster(name="Damage Cluster").add_to(m)

        for inc in filtered_incidents:
            c_name = config.CLASS_HUMAN_NAMES.get(inc["primary_class"], inc["primary_class"])
            color = FOLIUM_ICON_COLORS.get(inc["priority_level"], "blue")
            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 13px; line-height: 1.4;">
                <b style="color: #0f172a;">{inc['primary_class']} — {c_name}</b><br/>
                <b>Priority:</b> <span style="color: {PRIORITY_COLORS.get(inc['priority_level'], '#000')};">{inc['priority_level']} ({inc['priority_score']:.1f}/100)</span><br/>
                <b>Confidence:</b> {inc['confidence']:.1%}<br/>
                <b>Road:</b> {inc['road_classification']}<br/>
                <b>Status:</b> {inc['review_status']}<br/>
                <small>UUID: {inc['id'][:8]}</small>
            </div>
            """
            folium.Marker(
                location=[inc["latitude"], inc["longitude"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{inc['primary_class']} ({inc['priority_level']})",
                icon=folium.Icon(color=color, icon="exclamation-circle", prefix="fa")
            ).add_to(cluster)

        _ = st_folium(m, width="100%", height=560, returned_objects=[])

    else:
        # Offline Plotly Mapbox / Scatter Fallback
        import pandas as pd
        df = pd.DataFrame([{
            "ID": inc["id"][:8],
            "Latitude": inc["latitude"],
            "Longitude": inc["longitude"],
            "Class": inc["primary_class"],
            "Name": config.CLASS_HUMAN_NAMES.get(inc["primary_class"], inc["primary_class"]),
            "Priority": inc["priority_level"],
            "Score": inc["priority_score"],
            "Status": inc["review_status"],
        } for inc in filtered_incidents])

        fig = px.scatter(
            df,
            x="Longitude",
            y="Latitude",
            color="Priority",
            color_discrete_map=PRIORITY_COLORS,
            hover_name="Name",
            hover_data={"Class": True, "Score": True, "Status": True, "ID": True},
            size=[14] * len(df),
            title="Offline Geospatial Grid Coordinates"
        )
        fig.update_layout(
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            height=560,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Incident Selection dropdown below map
    if filtered_incidents:
        options = {f"[{inc['priority_level']}] {inc['primary_class']} - {inc['id'][:8]} ({inc['road_classification']})": inc['id'] for inc in filtered_incidents}
        sel_label = st.selectbox(
            "Select Incident to Inspect in Evidence Drawer:",
            options=list(options.keys()),
            index=0
        )
        selected_incident_id = options[sel_label]
        st.session_state["selected_map_incident_id"] = selected_incident_id

with col_drawer:
    st.subheader("📋 Evidence Drawer")

    active_inc = None
    if selected_incident_id:
        for inc in filtered_incidents:
            if inc["id"] == selected_incident_id:
                active_inc = inc
                break

    if not active_inc and filtered_incidents:
        active_inc = filtered_incidents[0]

    if active_inc:
        theme.render_card_start(title=f"Defect {active_inc['primary_class']}")
        st.markdown(f"**Human Classification:** `{config.CLASS_HUMAN_NAMES.get(active_inc['primary_class'], active_inc['primary_class'])}`")
        st.markdown(f"**Incident UUID:** `{active_inc['id']}`")

        p_color = PRIORITY_COLORS.get(active_inc["priority_level"], "#38bdf8")
        st.markdown(
            f"<div style='background: {p_color}22; border-left: 4px solid {p_color}; padding: 8px 12px; border-radius: 4px; margin: 8px 0;'>"
            f"<b>Priority Score:</b> {active_inc['priority_score']:.1f} / 100 &nbsp;|&nbsp; <b>Tier:</b> {active_inc['priority_level']}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"**Detection Confidence:** `{active_inc['confidence']:.1%}`")
        st.markdown(f"**Road Classification:** `{active_inc['road_classification']}`")
        st.markdown(f"**GPS Coordinates:** `{active_inc['latitude']:.5f}, {active_inc['longitude']:.5f}`")
        st.markdown(f"**Location Provenance:** `{active_inc['location_provenance']}`")
        st.markdown(f"**Survey Timestamp:** `{active_inc['timestamp']}`")

        # Photo evidence
        img_p = active_inc.get("image_path")
        if img_p and Path(img_p).exists():
            st.image(img_p, caption=f"Evidence Frame: {Path(img_p).name}", use_container_width=True)
        else:
            st.info("No local image frame cached for this record.")

        # Human Review Actions
        st.markdown("#### 👷 Civil Engineering Triage")
        st.markdown(f"**Current Status:** `{active_inc['review_status']}`")

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("✅ Mark Verified", key=f"verify_{active_inc['id']}", use_container_width=True):
                update_review_status(active_inc['id'], "VERIFIED", reviewer_notes="Verified by municipal inspector.")
                st.success("Status updated to VERIFIED.")
                st.rerun()

        with act_col2:
            if st.button("❌ Dismiss Alarm", key=f"dismiss_{active_inc['id']}", use_container_width=True):
                update_review_status(active_inc['id'], "REJECTED", reviewer_notes="Dismissed as non-structural feature.")
                st.warning("Status updated to REJECTED.")
                st.rerun()

        theme.render_card_end()
    else:
        st.info("Select an incident to view its photo evidence and engineering review options.")
