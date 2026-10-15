"""
RoadSense AI - About & Civil Engineering Methodology
Academic citations, open-source dataset licenses, damage taxonomy specifications,
priority formula mathematics, and municipal safety disclaimers.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app import theme
from core import config

theme.setup_page(
    page_title="About & Methodology | RoadSense AI",
    page_icon="📖",
    layout="wide"
)
theme.apply_theme()

st.title("📖 About & Civil Engineering Methodology")
st.caption("Scientific foundations, dataset attribution, damage classification standards, and deterministic ranking mathematics.")

# Prominent Civil Safety Disclaimer
theme.render_safety_banner()

st.markdown("---")

# 1. Dataset Citation & Open License
st.subheader("🏛️ Dataset Attribution & Academic Citation")
theme.render_card_start(title="Road Damage Dataset 2022 (RDD2022)")
st.markdown("""
RoadSense AI is trained and benchmarked using the **Road Damage Dataset 2022 (RDD2022)**, released as part of the **IEEE BigData Cup / Global Road Damage Detection Challenge (CRDDC 2022)**.

- **Title:** *RDD2022: Multi-National Road Damage Dataset*
- **Authors:** Deeksha Arya, Hiroya Maeda, Sanjay Kumar Ghosh, Durga Toshniwal, Alexander Mraz, Takehiro Kashiyama, Yoshihide Sekimoto
- **Published In:** *Big Data Mining and Analytics*, IEEE / Tsinghua University Press
- **Dataset DOI:** [10.6084/m9.figshare.21431547.v1](https://doi.org/10.6084/m9.figshare.21431547.v1)
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  - *You are free to share and adapt the material for any purpose, with appropriate credit and indication of modifications made.*
""")
theme.render_card_end()

st.markdown("---")

# 2. Canonical Pavement Distress Taxonomy (IEEE BigData Cup)
st.subheader("🔬 Canonical Pavement Distress Taxonomy")
st.caption("Standardized distress classifications defined by the CRDDC benchmark committee.")

t_col1, t_col2 = st.columns(2)

with t_col1:
    theme.render_card_start(title="D00 — Longitudinal Crack")
    st.markdown("""
    - **Physical Description:** Linear cracks running parallel to vehicle travel direction, predominantly situated along wheel paths.
    - **Causative Factors:** Repetitive axle load fatigue, longitudinal joint construction weaknesses, or lane widening settlement.
    - **Severity Weight:** `w = 0.4`
    - **Prescribed Maintenance:** High-pressure routing and hot-pour elastomer crack sealant.
    """)
    theme.render_card_end()

    theme.render_card_start(title="D20 — Alligator / Fatigue Crack")
    st.markdown("""
    - **Physical Description:** Interconnected networks of polygonal fissures resembling reptilian hide, indicating deep structural failure.
    - **Causative Factors:** Structural inadequacy of the road base or subgrade, poor drainage, and repeated heavy vehicular overloading.
    - **Severity Weight:** `w = 0.8`
    - **Prescribed Maintenance:** Cold milling of failed asphalt layer and structural binder/wearing course overlay.
    """)
    theme.render_card_end()

with t_col2:
    theme.render_card_start(title="D10 — Transverse Crack")
    st.markdown("""
    - **Physical Description:** Linear cracks propagating perpendicular to the road centerline across the traffic lane.
    - **Causative Factors:** Thermal contraction during diurnal temperature swings, or reflection cracking propagating from underlying concrete base slabs.
    - **Severity Weight:** `w = 0.5`
    - **Prescribed Maintenance:** Bituminous crack sealing to prevent water infiltration into subgrade.
    """)
    theme.render_card_end()

    theme.render_card_start(title="D40 — Pothole (Structural Cavity)")
    st.markdown("""
    - **Physical Description:** Bowl-shaped cavities in the pavement surface exceeding 25 mm depth and 150 mm width with abrupt edges.
    - **Causative Factors:** Freeze-thaw hydro-dynamic pumping and traffic impact displacing unbonded asphalt aggregate. Immediate safety and tire hazard.
    - **Severity Weight:** `w = 1.0` (Maximum Hazard)
    - **Prescribed Maintenance:** Immediate 24–48h emergency cold/hot asphalt patch compaction.
    """)
    theme.render_card_end()

st.markdown("---")

# 3. Deterministic Priority Ranking Formula v1.0
st.subheader("📐 Priority Ranking Engine Formula v1.0")
st.caption("Mathematical specification of the deterministic score calculation.")

st.latex(
    r"S = \min\left(100, \, \left(w_{\text{class}} \cdot \bar{c} \cdot 40\right) + \left(\min(N_{\text{cluster}}, 5) \cdot 6\right) + \left(w_{\text{road}} \cdot 15\right) + \left(w_{\text{human}} \cdot 15\right)\right)"
)

st.markdown("""
### Variable Breakdown:
1. **$w_{\\text{class}} \\cdot \\bar{c} \\cdot 40$ (Distress Severity & Confidence Component — up to 40 pts):**
   - Combines the highest severity weight of the detected defect (`1.0` for Potholes down to `0.4` for Longitudinal Cracks) with the model's confidence $\\bar{c} \\in [0, 1]$.
2. **$\\min(N_{\\text{cluster}}, 5) \\cdot 6$ (Spatial Repetition Multiplier — up to 30 pts):**
   - Counts verified distinct detections within a 25-meter spatial radius using the Haversine formula. Repeat reports from multiple survey vehicles increase priority up to a hard cap of 5 detections (30 points).
3. **$w_{\\text{road}} \\cdot 15$ (Roadway Importance Weight — up to 15 pts):**
   - High-speed, heavy-traffic arterials receive maximum priority (`1.0`), while local residential streets receive lower weights (`0.4`).
4. **$w_{\\text{human}} \\cdot 15$ (Human-in-the-Loop Multiplier — up to 15 pts):**
   - Certified municipal engineers can apply an urgency modifier ($w_{\\text{human}} \\in [0.5, 1.5]$) based on site-specific factors like proximity to school zones or hospital emergency corridors.
""")

st.markdown("---")

# 4. Anti-Leakage Route Splitting Methodology
st.subheader("🛡️ Leakage-Aware Splitting Methodology")
st.markdown("""
In the RDD2022 dataset, consecutive frames (e.g. `India_001240.jpg` and `India_001241.jpg`) were recorded by dashcam video streams at ~1 Hz.
- **The Problem:** Random image splitting distributes adjacent video frames across train and validation sets, allowing deep models to artificially achieve high validation scores by memorizing static background landmarks.
- **The Solution:** RoadSense AI clusters consecutive image indices and verifies visual overlap via **64-bit perceptual difference hashing (dHash)**. Contiguous route sequences are atomically assigned to either `train` (80%) or `val` (20%), ensuring **100% zero temporal leakage**.
""")
