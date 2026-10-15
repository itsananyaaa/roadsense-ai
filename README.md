# RoadSense AI — Municipal Road Condition Intelligence Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-2.12.1%20(Apple%20MPS)-red.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO26n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit AppTest](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **"See the Damage. Understand the Risk. Prioritize the Repair."**  
> An automated computer vision and geospatial intelligence platform for municipal pavement maintenance, based on the **RDD2022 / IEEE BigData Cup Multi-National Road Damage Benchmark**.

---

## 1. System Overview & Problem Statement

Municipal road authorities spend millions annually on reactive road maintenance. Traditional windshield surveys are slow, subjective, hazardous to surveyors, and fail to prioritize high-risk hazards like vehicle-deflecting potholes over minor surface fissures.

**RoadSense AI** transforms pavement management by:
1. **Automated Vision Inspection:** Real-time detection of the 4 canonical road distress categories (`D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole) with normalized coordinates and confidence scoring.
2. **Video Stream Intelligence:** Ingesting dashcam streams with user-configurable sampling rates (1 to 5 FPS) and **temporal de-duplication** to eliminate duplicate incident inflation.
3. **Deterministic Priority Ranking Engine v1.0:** Mathematical scoring formula ($S \in [0, 100]$) combining defect severity, detection confidence, 25-meter spatial repetition density, road classification, and human inspector modifiers.
4. **Strict Location Provenance:** Enforcing zero coordinate fabrication (`EXIF_GPS`, `USER_ENTERED`, `EXPLICIT_SYNTHETIC_DEMO`, `UNKNOWN`).
5. **Dual-Domain Benchmark:** Rigorous validation across in-domain held-out splits (India) and unseen-country domain shifts (Czech Republic).
6. **Command-Center UI:** Modern Streamlit console with dark slate shell (`#0f172a`), crisp off-white analytical cards, and highway safety amber (`#f59e0b`) accents.

---

## 2. Pavement Distress Taxonomy (RDD2022 Standard)

| Code | Damage Category | Description & Causative Mechanism | Base Severity Weight ($w_{\text{class}}$) | Prescribed Municipal Maintenance Action |
|:---:|:---|:---|:---:|:---|
| **D00** | **Longitudinal Crack** | Linear cracks parallel to traffic direction along wheel paths from load fatigue. | `0.4` | High-pressure routing & hot-pour elastomer sealant |
| **D10** | **Transverse Crack** | Cracks perpendicular to centerline from thermal diurnal contraction or base reflection. | `0.5` | Bituminous joint sealing to block moisture infiltration |
| **D20** | **Alligator Crack** | Interconnected polygonal map cracking indicating structural road base or subgrade failure. | `0.8` | Cold asphalt milling and structural binder/wearing overlay |
| **D40** | **Pothole** | Structural surface cavity exceeding 25 mm depth; immediate vehicular deflection hazard. | `1.0` (Hazard) | Emergency 24–48h cold/hot patch compaction |

---

## 3. Architecture & Repository Map

```
roadsense-ai/
├── app/                        # Presentation Layer (Streamlit Multi-Page Shell)
│   ├── main.py                 # Overview Dashboard, Network KPIs & Health Center
│   ├── theme.py                # Civil engineering command-center design system
│   └── pages/
│       ├── 1_Detect_Damage.py      # Image upload, real MPS inference, bbox coordinates
│       ├── 2_Video_Intelligence.py # Dashcam processing, de-duplication, keyframe gallery
│       ├── 3_Incident_Map.py       # Leaflet/Folium geospatial map & evidence drawer
│       ├── 4_Priority_Center.py    # Deterministic triage queue & work order export
│       ├── 5_Model_Lab.py          # Grounded benchmark studio (reads reports/)
│       ├── 6_Reports_History.py    # Municipal inventory, CSV & GeoJSON export
│       └── 7_About_Methodology.py  # Citations, CC BY 4.0 license, priority formula math
├── core/                       # Core Services Layer
│   ├── config.py               # Centralized configuration constants & paths
│   ├── db.py                   # SQLite persistence, parameterized queries & Haversine clustering
│   ├── demo.py                 # Offline demonstration suite & calibrated fixtures
│   ├── geo.py                  # EXIF GPS extraction & strict provenance taxonomy
│   ├── health.py               # System health diagnostics & model pedigree tracking
│   ├── inference.py            # Ultralytics YOLO inference wrapper on Apple MPS
│   ├── logger.py               # Standardized logging service
│   ├── priority.py             # Deterministic Priority Formula Engine v1.0
│   └── video.py                # Video processor, sampling rate control & de-duplication
├── ml/                         # ML Pipeline Layer
│   ├── audit.py                # Pascal-VOC XML statistical auditor
│   ├── colab_train.ipynb       # Self-contained Google Colab GPU training notebook
│   ├── convert.py              # VOC XML to YOLO converter with round-trip checks
│   ├── evaluate.py             # Dual-domain evaluator (India vs Czech) & benchmark
│   ├── split.py                # Anti-leakage sequence + perceptual dHash splitter
│   └── train.py                # Local MPS smoke trainer & production trainer
├── models/                     # Checkpoints & Model Metadata
│   └── smoke_test_metadata.json# Checkpoint pedigree metadata (weights excluded from git)
├── reports/                    # Physical Evaluation & Benchmark Artifacts
│   ├── confusion_matrix.png    # Normalized 4-class + background confusion matrix
│   ├── pr_curve.png            # Multi-class Precision-Recall curves
│   ├── eval_india.json         # In-domain held-out evaluation report
│   ├── eval_czech.json         # Unseen-country zero-shot transfer report
│   └── eval_benchmark.json     # Hardware latency and parameter summary
├── docs/                       # Project Documentation & Expo Suite
│   ├── OFFLINE_CHECKLIST.md    # 100% offline demonstration protocol
│   ├── PITCH.md                # 3-Minute project pitch script
│   └── VIVA_GUIDE.md           # Comprehensive viva examination preparation guide
├── tests/                      # Automated Test Suite (26 pytest / AppTest tests)
│   ├── test_all_pages.py       # Native Streamlit AppTest tests for all 7 console pages
│   ├── test_app_foundation.py  # Shell initialization and theme tests
│   ├── test_convert.py         # Format conversion and class remapping tests
│   ├── test_db.py              # SQLite CRUD and spatial distance tests
│   ├── test_evaluate.py        # Evaluator and plot generation tests
│   ├── test_priority.py        # Deterministic formula boundary and limit tests
│   └── test_vertical_slice.py  # End-to-end image-to-database integration test
├── requirements.txt            # Python dependencies
├── MANUAL_STEPS.md             # Guide for Colab GPU training & checkpoint copying
├── PLAN.md                     # Engineering blueprint & milestone tracker
├── STATE.md                    # Live phase status & acceptance criteria tracker
├── EVIDENCE_LOG.md             # Immutable execution audit log
└── VERIFICATION_REPORT.md      # Final compliance & verification report
```

---

## 4. Quickstart Guide (Local Setup & Launch)

### Step 4.1: Clone & Configure Environment
```bash
# Clone the repository
git clone https://github.com/ananyabiju/roadsense-ai.git
cd roadsense-ai

# Activate your Python 3.10+ environment (or conda/venv)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4.2: Verify System Health
Execute the diagnostics check to verify hardware acceleration and model status:
```bash
python core/health.py
```
*Expected Output:*
```text
--- ROADSENSE AI HEALTH CHECK ---
is_operational: True
accelerator: Apple Metal Performance Shaders (MPS) (or CUDA / CPU)
active_checkpoint_path: models/smoke_test.pt (or yolo26n.pt)
checkpoint_status: SMOKE_CHECKPOINT (or PRETRAINED_BASELINE)
database_status: READY
writable_paths: {'data': True, 'evidence': True, 'models': True, 'reports': True}
```

### Step 4.3: Launch the Road Intelligence Console
```bash
streamlit run app/main.py
```
Open your browser to `http://localhost:8501`. Navigate between pages using the sidebar:
- `Overview Dashboard` — High-level network KPIs and system health diagnostics
- `1_Detect_Damage` — Single/batch image analysis with bounding box coordinates
- `2_Video_Intelligence` — Dashcam video stream analysis with temporal de-duplication
- `3_Incident_Map` — Interactive Folium geospatial map with evidence drawer
- `4_Priority_Center` — Deterministic triage queue and municipal work order export
- `5_Model_Lab` — Hardware benchmarks, dual-domain metrics, and confusion matrix
- `6_Reports_History` — Incident search, filter, CSV, and GeoJSON export
- `7_About_Methodology` — Mathematical specifications, citations, and licenses

---

## 5. Running the Automated Test Suite

RoadSense AI includes **26 automated tests** covering database CRUD, priority determinism, format conversion, hardware latency, dual-domain evaluation, and Streamlit `AppTest` page initialization:

```bash
pytest tests/ -v
```
*Expected Output:*
```text
============================== 26 passed in 3.41s ==============================
```

---

## 6. Model Training & Colab GPU Execution

To train the model on Google Colab with a free GPU (T4/V100):
1. Open [Google Colab](https://colab.research.google.com/) and upload `ml/colab_train.ipynb`.
2. Set runtime type to **T4 GPU** (`Runtime` > `Change runtime type`).
3. Run all cells top-to-bottom:
   - Mounts Google Drive for persistent checkpoint saving (`last.pt`, `best.pt`).
   - Downloads RDD2022 India (7,706 images) and Czech (2,829 images) subsets.
   - Executes statistical audit, class remapping, and negative sample retention.
   - Performs **leakage-aware sequence splitting** (sequence ID + perceptual dHash).
   - Fine-tunes `yolo26n.pt` for 50 epochs with AdamW and early stopping (`patience=10`).
   - Evaluates held-out India validation set and Czech zero-shot domain transfer.
   - Bundles `roadsense_weights.zip` with `best.pt` and metrics for download.
4. Download `best.pt` and place it in your local workspace:
   ```bash
   cp ~/Downloads/best.pt models/best.pt
   ```
5. Refresh RoadSense AI — the system health badge will automatically update to `EVALUATED_TRAINED`!

---

## 7. Deterministic Priority Ranking Formula v1.0

Priority score $S \in [0, 100]$ is computed deterministically:

$$S = \min\left(100, \, \left(w_{\text{class}} \cdot \bar{c} \cdot 40\right) + \left(\min(N_{\text{cluster}}, 5) \cdot 6\right) + \left(w_{\text{road}} \cdot 15\right) + \left(w_{\text{human}} \cdot 15\right)\right)$$

### Priority Bands:
- **Critical (Score $\ge 75$):** Immediate 24–48h emergency response (hazardous potholes).
- **High (Score $55 - 74$):** Scheduled 7-day structural maintenance (alligator fatigue cracks).
- **Moderate (Score $35 - 54$):** Routine 30-day backlog (transverse crack sealing).
- **Low (Score $< 35$):** Monitor on subsequent survey cycle (early longitudinal cracks).

---

## 8. Hardware Benchmark Results (Apple Silicon M4)

Empirically measured at batch size 1 on 640x640 resolution (macOS 26.5.2, Apple M4, 16 GB Unified Memory):

| Model Architecture | Parameters | File Size | Apple MPS Latency | Apple MPS FPS | CPU Latency | CPU FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO26n** (Active) | **2.54 M** | **5.2 MB** | **10.9 ms** | **91.5** | 37.8 ms | 26.5 |
| **YOLO11n** (Fallback) | 2.62 M | 5.4 MB | 17.4 ms | 57.6 | 43.5 ms | 23.0 |
| **YOLOv8n** (Baseline) | 3.16 M | 6.3 MB | 13.6 ms | 73.5 | 40.3 ms | 24.8 |

---

## 9. Academic Citations & Legal Attribution

- **Dataset:** RDD2022 (Road Damage Dataset 2022) / IEEE BigData Cup 2022
- **Authors:** Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y.
- **Reference:** *Big Data Mining and Analytics*, IEEE / Tsinghua University Press
- **Dataset DOI:** [10.6084/m9.figshare.21431547.v1](https://doi.org/10.6084/m9.figshare.21431547.v1)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

> **Civil Engineering Advisory Notice:**  
> *Model detection confidence is an empirical perceptual score, not an in-situ structural load-bearing assessment. RoadSense AI assists municipal road maintenance prioritization and does not replace certified geotechnical road safety audits.*
