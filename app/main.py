"""
RoadSense AI - Main Application Entrypoint
"See the Damage. Understand the Risk. Prioritize the Repair."
An AI-powered road-condition intelligence platform.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core import config
from core.db import init_db, get_summary_statistics
from core.demo import seed_demo_database
from core.health import run_health_check
from app.theme import apply_theme, render_brand_header, render_safety_notice


def init_session_state() -> None:
    """Initialize application session variables."""
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state["db_initialized"] = True
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = False
    if "health_report" not in st.session_state:
        st.session_state["health_report"] = run_health_check()


def main() -> None:
    st.set_page_config(
        page_title="RoadSense AI — Road Intelligence Console",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_theme()
    render_brand_header()

    health = st.session_state["health_report"]

    # Model status badge in sidebar
    ckpt_status = health.checkpoint_status
    if ckpt_status == "EVALUATED_TRAINED":
        badge_html = "<span class='rs-badge rs-badge-low'>Model: Fine-Tuned (Evaluated)</span>"
    elif ckpt_status == "SMOKE_CHECKPOINT":
        badge_html = "<span class='rs-badge rs-badge-amber'>Model: Smoke-Test Checkpoint</span>"
    elif ckpt_status == "PRETRAINED_BASELINE":
        badge_html = "<span class='rs-badge rs-badge-high'>Model: Pretrained Baseline (COCO)</span>"
    else:
        badge_html = "<span class='rs-badge rs-badge-critical'>Model: Missing (Demo Available)</span>"

    st.sidebar.markdown(f"<div style='margin-bottom: 0.8rem;'>{badge_html}</div>", unsafe_allow_html=True)
    st.sidebar.caption(f"Device: **{health.accelerator}**")

    # Demo mode toggle in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo Controls")
    if st.sidebar.button("Load Demo Dataset (5 Records)", use_container_width=True, key="btn_load_demo"):
        seeded = seed_demo_database(reset_existing=True)
        st.session_state["demo_mode"] = True
        st.session_state["health_report"] = run_health_check()
        st.sidebar.success(f"Seeded {seeded} demo records!")
        st.rerun()

    render_safety_notice()

    # Main Page Content: Overview & Health Center
    st.title("Road Intelligence Console")
    st.markdown("*See the Damage. Understand the Risk. Prioritize the Repair.*")

    stats = get_summary_statistics()

    # KPI Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="rs-kpi">
                <div class="rs-kpi-value">{stats['total_incidents']}</div>
                <div class="rs-kpi-label">Logged Incidents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        critical_count = stats["by_priority"].get("CRITICAL", 0)
        st.markdown(
            f"""
            <div class="rs-kpi">
                <div class="rs-kpi-value" style="color: #ef4444;">{critical_count}</div>
                <div class="rs-kpi-label">Critical Priority</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        pothole_count = stats["by_class"].get("D40", 0)
        st.markdown(
            f"""
            <div class="rs-kpi">
                <div class="rs-kpi-value">{pothole_count}</div>
                <div class="rs-kpi-label">Potholes (D40)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        pending_review = stats["by_review"].get("PENDING", 0)
        st.markdown(
            f"""
            <div class="rs-kpi">
                <div class="rs-kpi-value">{pending_review}</div>
                <div class="rs-kpi-label">Pending Verification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Mission & System Architecture Card
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown(
            """
            <div class="rs-card">
                <h3 style="margin-top: 0; color: #0f172a; font-size: 1.2rem;">Municipal Road Condition Management</h3>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    RoadSense AI replaces subjective manual road surveys with an empirical computer vision pipeline.
                    Incoming dashcam imagery and mobile video are analyzed using deep object detection trained on multi-national
                    pavement distress benchmarks (RDD2022). Each detected defect is cataloged with GPS provenance, visual evidence,
                    and a mathematically deterministic inspection dispatch score.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem;">
                    <span class="rs-badge rs-badge-amber">CRDDC Benchmark</span>
                    <span class="rs-badge rs-badge-amber">4 Canonical Distress Types</span>
                    <span class="rs-badge rs-badge-amber">Zero GPS Hallucination</span>
                    <span class="rs-badge rs-badge-amber">Evidence-Linked Audit Trail</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            f"""
            <div class="rs-card">
                <h3 style="margin-top: 0; color: #0f172a; font-size: 1.2rem;">System Health & Diagnostics</h3>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.8;">
                    <strong>Host:</strong> {health.os_info}<br>
                    <strong>Accelerator:</strong> {health.accelerator}<br>
                    <strong>PyTorch:</strong> {health.torch_version}<br>
                    <strong>Ultralytics:</strong> {health.ultralytics_version}<br>
                    <strong>Active Weights:</strong> <code>{Path(health.active_checkpoint_path).name}</code><br>
                    <strong>Database:</strong> SQLite ({health.database_status})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Canonical Classes Reference Card
    st.markdown(
        """
        <div class="rs-card">
            <h4 style="margin-top: 0; color: #0f172a;">Canonical Damage Taxonomy (RDD2022 Standard)</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 0.8rem;">
                <div style="padding: 10px; border-left: 3px solid #3b82f6; background: #f8fafc; border-radius: 4px;">
                    <strong style="color: #1e3a8a;">D00 — Longitudinal Crack</strong><br>
                    <span style="font-size: 0.82rem; color: #64748b;">Linear cracks parallel to traffic direction along wheel paths.</span>
                </div>
                <div style="padding: 10px; border-left: 3px solid #8b5cf6; background: #f8fafc; border-radius: 4px;">
                    <strong style="color: #4c1d95;">D10 — Transverse Crack</strong><br>
                    <span style="font-size: 0.82rem; color: #64748b;">Thermal cracks running perpendicular to the pavement center line.</span>
                </div>
                <div style="padding: 10px; border-left: 3px solid #f59e0b; background: #f8fafc; border-radius: 4px;">
                    <strong style="color: #78350f;">D20 — Alligator Crack</strong><br>
                    <span style="font-size: 0.82rem; color: #64748b;">Interconnected polygonal map cracking from sub-base fatigue failure.</span>
                </div>
                <div style="padding: 10px; border-left: 3px solid #ef4444; background: #f8fafc; border-radius: 4px;">
                    <strong style="color: #7f1d1d;">D40 — Pothole</strong><br>
                    <span style="font-size: 0.82rem; color: #64748b;">Bowl-shaped surface cavities posing direct vehicle impact hazards.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
