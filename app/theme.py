"""
RoadSense AI - UI Theme & Visual Design System ("Road Intelligence Console")
Implements the civil engineering command-center visual language:
- Midnight slate shell (#0f172a)
- Crisp off-white analytical card surfaces (#ffffff / #f8fafc)
- Restrained road-safety amber accent (#f59e0b)
- Alert red (#ef4444) strictly for critical defects
- WCAG-compliant typography and contrast ratios
"""

import streamlit as st
from typing import Optional

# SVG Brand Icon (Road contour with detection reticle)
ROADSENSE_LOGO_SVG = """
<svg width="36" height="36" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Pavement Perspective Trapeze -->
  <path d="M12 42L20 10H28L36 42H12Z" fill="#1e293b" stroke="#334155" stroke-width="2"/>
  <!-- Center Line Markings -->
  <line x1="24" y1="12" x2="24" y2="18" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="24" y1="24" x2="24" y2="31" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>
  <line x1="24" y1="36" x2="24" y2="42" stroke="#f59e0b" stroke-width="3.5" stroke-linecap="round"/>
  <!-- Target Detection Reticle -->
  <circle cx="24" cy="24" r="14" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3"/>
  <path d="M16 19H19V16" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
  <path d="M32 19H29V16" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
  <path d="M16 29H19V32" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
  <path d="M32 29H29V32" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

CUSTOM_CSS = """
<style>
/* ==========================================================================
   RoadSense AI - Design Tokens & Scoped Layout
   ========================================================================== */

:root {
  --rs-bg: #0f172a;
  --rs-surface: #ffffff;
  --rs-surface-alt: #f8fafc;
  --rs-border: #e2e8f0;
  --rs-border-subtle: #cbd5e1;
  --rs-text-primary: #0f172a;
  --rs-text-secondary: #475569;
  --rs-text-muted: #64748b;
  --rs-amber: #f59e0b;
  --rs-amber-hover: #d97706;
  --rs-amber-light: #fef3c7;
  --rs-red: #ef4444;
  --rs-red-light: #fee2e2;
  --rs-green: #10b981;
  --rs-green-light: #d1fae5;
  --rs-radius: 8px;
  --rs-shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --rs-shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

/* Global Font & Header Tuning */
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  letter-spacing: -0.01em;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
  background-color: #0b1120;
  border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
  padding-top: 1rem;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {
  color: #94a3b8;
  font-weight: 500;
  font-size: 0.95rem;
}

/* Cards & Surface Containers */
div.rs-card {
  background-color: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
}

div.rs-card-dark {
  background-color: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  color: #f8fafc;
}

/* Pill Badges */
span.rs-badge {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  font-size: 0.75rem;
  font-weight: 600;
  border-radius: 9999px;
  letter-spacing: 0.025em;
  text-transform: uppercase;
}

span.rs-badge-critical {
  background-color: #fee2e2;
  color: #991b1b;
  border: 1px solid #fca5a5;
}

span.rs-badge-high {
  background-color: #ffedd5;
  color: #9a3412;
  border: 1px solid #fdba74;
}

span.rs-badge-moderate {
  background-color: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
}

span.rs-badge-low {
  background-color: #d1fae5;
  color: #065f46;
  border: 1px solid #a7f3d0;
}

span.rs-badge-amber {
  background-color: #fef3c7;
  color: #b45309;
  border: 1px solid #fcd34d;
}

/* Metric KPI Styling */
div.rs-kpi {
  text-align: left;
  padding: 0.75rem 1rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
}

div.rs-kpi-value {
  font-size: 1.75rem;
  font-weight: 700;
  color: #0f172a;
  line-height: 1.2;
}

div.rs-kpi-label {
  font-size: 0.8rem;
  color: #64748b;
  font-weight: 500;
  text-transform: uppercase;
  margin-top: 0.2rem;
}

/* Primary Button Styling */
button[kind="primary"] {
  background-color: #f59e0b !important;
  border-color: #d97706 !important;
  color: #0f172a !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
}

button[kind="primary"]:hover {
  background-color: #d97706 !important;
  border-color: #b45309 !important;
  color: #ffffff !important;
}

/* Clean Tables */
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  overflow: hidden;
}
</style>
"""


def apply_theme() -> None:
    """Inject scoped CSS and design system styles into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_brand_header() -> None:
    """Render the RoadSense AI brand banner in the sidebar."""
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px solid #1e293b;">
            {ROADSENSE_LOGO_SVG}
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">RoadSense AI</div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">Road Intelligence Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_safety_notice() -> None:
    """Display the mandatory civil engineering safety notice in footer/sidebar."""
    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748b; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; margin-top: 1.5rem; line-height: 1.4;">
            <strong>Civil Engineering Notice:</strong><br>
            Model detection confidence is not structural severity. RoadSense AI assists municipal inspection prioritization only and does not replace certified professional civil engineering road safety audits.
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_page(page_title: str, page_icon: str = "🛣️", layout: str = "wide") -> None:
    """Safely configure page metadata and inject theme and brand headers."""
    try:
        st.set_page_config(
            page_title=page_title,
            page_icon=page_icon,
            layout=layout,
            initial_sidebar_state="expanded"
        )
    except Exception:
        # Ignore if page config was already set in session
        pass
    apply_theme()
    render_brand_header()
    render_safety_notice()


def render_safety_banner() -> None:
    """Render the high-visibility civil engineering notice at the top of an analytical page."""
    st.markdown(
        """
        <div style="background: #1e293b; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin-bottom: 1.2rem; font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
            <strong style="color: #f59e0b;">Civil Engineering Advisory:</strong> AI detection confidence is an empirical perception metric, not an in-situ structural severity rating. RoadSense AI assists municipal road maintenance prioritization and does not replace certified on-site geotechnical engineering audits.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str, extra_info: str = "") -> None:
    """Render a standardized model status badge with optional extra info."""
    badge_classes = {
        "EVALUATED_TRAINED": "rs-badge rs-badge-low",
        "SMOKE_CHECKPOINT": "rs-badge rs-badge-amber",
        "PRETRAINED_BASELINE": "rs-badge rs-badge-high",
        "MISSING": "rs-badge rs-badge-critical",
    }
    badge_labels = {
        "EVALUATED_TRAINED": "Fine-Tuned & Evaluated Checkpoint",
        "SMOKE_CHECKPOINT": "Smoke-Test Checkpoint (Local MPS)",
        "PRETRAINED_BASELINE": "Pretrained COCO Baseline",
        "MISSING": "Checkpoint Missing (Demo Mode Available)",
    }
    cls = badge_classes.get(status, "rs-badge rs-badge-amber")
    label = badge_labels.get(status, status)
    extra_str = f" &nbsp;|&nbsp; <span>{extra_info}</span>" if extra_info else ""

    st.markdown(
        f"<div style='margin-bottom: 0.8rem;'><span class='{cls}'>{label}</span>{extra_str}</div>",
        unsafe_allow_html=True
    )


def render_card_start(title: str = "") -> None:
    """Open a styled analytical card container."""
    title_html = f"<h4 style='margin-top: 0; color: #0f172a;'>{title}</h4>" if title else ""
    st.markdown(
        f"<div class='rs-card'>{title_html}",
        unsafe_allow_html=True
    )


def render_card_end() -> None:
    """Close an analytical card container."""
    st.markdown("</div>", unsafe_allow_html=True)

