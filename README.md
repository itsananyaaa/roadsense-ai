# RoadSense AI — Municipal Road Condition Intelligence Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-2.12.1%20(Apple%20MPS)-red.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO26n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Test Suite](https://img.shields.io/badge/Automated%20Tests-91%20Passed-brightgreen.svg)](https://pytest.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **"See the Damage. Understand the Risk. Prioritize the Repair."**  
> An automated computer vision, geospatial intelligence, and deterministic triage platform for municipal pavement maintenance, based on the **RDD2022 / IEEE BigData Cup Multi-National Road Damage Benchmark**.

---

## 1. System Overview & Problem Statement

Municipal road authorities spend millions annually on reactive road maintenance. Traditional windshield surveys are slow, subjective, hazardous to surveyors, and fail to prioritize high-risk hazards like vehicle-deflecting potholes over minor surface fissures.

**RoadSense AI** transforms pavement management by:
1. **Automated Vision Inspection:** Real-time detection of the 4 canonical road distress categories (`D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole) with normalized coordinates and confidence scoring.
2. **Video Stream Intelligence:** Ingests dashcam video at configurable sampling rates (1–5 FPS) with **temporal de-duplication** (2.5s window, IoU $\ge 0.30$ or centroid displacement $\le 0.15$), consolidating multi-frame detections into a single keyframe incident (93.3% deduplication rate) with HTTP 206 byte-range seeking.
3. **Deterministic Priority Ranking Engine v1.0:** Mathematical scoring formula ($S \in [0, 100]$) combining defect severity, detection confidence, 25-meter spatial repetition density, functional road classification, and human inspector modifiers.
4. **Strict Location Provenance:** Enforces zero coordinate fabrication via a 4-tier provenance enum (`EXIF_GPS`, `USER_ENTERED`, `EXPLICIT_SYNTHETIC_DEMO`, `UNKNOWN`) and automatic Null Island `(0, 0)` rejection.
5. **Dual-Domain Benchmark:** Rigorous validation across in-domain held-out splits (India mAP@0.5: `0.584`) and unseen-country domain shifts (Czech Republic mAP@0.5: `0.438`, $\Delta\text{mAP} = -25.0\%$).
6. **Enterprise Security & Offline Autonomy:** Fail-closed authentication with Argon2id password hashing, constant-time verification, HMAC-SHA256 session cookies, per-IP rate limiting, keyless Esri World Dark Gray Canvas basemaps, and 100% offline operation with zero external CDN dependencies.
7. **Dual Frontends:**
   - **Next-Gen Obsidian Web Console (`http://localhost:8000`):** Ultra-premium dark slate interface (`#0c0d14`), glowing neon splines, interactive Catmull-Rom incident frequency chart, 3D tilted hazard cards, animated 4-member SVG character crew, and offline rule-based AI Copilot.
   - **Legacy Streamlit Analytics Shell (`http://localhost:8501`):** Multi-page scientific data console across 7 modular pages.

---

## 2. Pavement Distress Taxonomy (RDD2022 Standard)

| Code | Damage Category | Description & Causative Mechanism | Base Severity Weight ($w_{\text{class}}$) | Prescribed Municipal Maintenance Action | SLA Response Target |
|:---:|:---|:---|:---:|:---|:---:|
| **D00** | **Longitudinal Crack** | Linear cracks parallel to traffic direction along wheel paths from load fatigue. Early moisture entry path. | `0.4` | High-pressure routing & linear hot-pour elastomer sealant | Subsequent survey cycle / 30-day monitor |
| **D10** | **Transverse Crack** | Cracks perpendicular to centerline from thermal diurnal contraction or base reflection. | `0.5` | Hot-pour bituminous crack sealing to arrest moisture infiltration into base | 30 Days (Routine Backlog) |
| **D20** | **Alligator Crack** | Interconnected polygonal map cracking indicating structural road base or subgrade failure. | `0.8` | Cold asphalt milling and structural binder/wearing overlay | 7 Days (Scheduled Structural) |
| **D40** | **Pothole** | Structural surface cavity exceeding 25 mm depth; immediate vehicular deflection hazard. | `1.0` (Hazard) | Emergency cold/hot asphalt patch compaction with mechanical tamping | 24–48 Hours (Emergency Triage) |

*Explicitly Filtered Classes:* `D43` & `D44` (lane/crosswalk paint wear), `D50` (manhole utility fixtures), and `Repair` (completed asphalt patches) are deliberately filtered to background.

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
│   ├── db.py                   # SQLite persistence, WAL mode & Haversine clustering
│   ├── demo.py                 # Offline demonstration suite & calibrated fixtures
│   ├── geo.py                  # EXIF GPS extraction & strict provenance taxonomy
│   ├── health.py               # System health diagnostics & model pedigree tracking
│   ├── inference.py            # Ultralytics YOLO inference wrapper with MPS lock
│   ├── logger.py               # Standardized logging service
│   ├── priority.py             # Deterministic Priority Formula Engine v1.0
│   ├── reports_pdf.py          # Landscape A4 PDF export engine (ReportLab + DejaVu fonts)
│   └── video.py                # Video processor, sampling rate control & de-duplication
├── ml/                         # ML Pipeline Layer
│   ├── audit.py                # Pascal-VOC XML statistical auditor
│   ├── build_demo_media.py     # Deterministic demo media builder & provenance registrar
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
│   ├── eval_benchmark.json     # Hardware latency and parameter summary
│   └── demo_media_check.json   # Live checkpoint verification on demo media suite
├── docs/                       # Project Documentation & Expo Suite
│   ├── MEDIA_AUDIT.md          # Complete audit of all shipped and generated media assets
│   ├── MEDIA_PROVENANCE.md     # Verified media origin, licensing, and field capture rules
│   ├── OFFLINE_CHECKLIST.md    # 100% offline demonstration protocol
│   ├── PITCH.md                # 3-Minute project pitch script
│   └── VIVA_GUIDE.md           # Comprehensive viva examination preparation guide
├── web/                        # Next-Gen Obsidian Web Console (FastAPI + HTML5/CSS/JS)
│   ├── backend.py              # High-performance FastAPI REST API wrapping core services
│   ├── run.py                  # Standalone Uvicorn runner daemon
│   └── static/
│       ├── index.html          # Resq.io dark obsidian single-page application shell
│       ├── css/style.css       # Obsidian glassmorphic design system & glow animations
│       ├── js/app.js           # Reactive SPA controller, Leaflet dark map & AI copilot
│       └── assets/leaflet/     # Bundled local Leaflet JS/CSS (100% offline compliance)
├── tests/                      # Automated Test Suite (91 pytest & AppTest tests)
│   ├── test_all_pages.py       # Native Streamlit AppTest tests for all 7 console pages
│   ├── test_app_foundation.py  # Shell initialization and theme tests
│   ├── test_audit_hardening.py # Audit hardening tests (methodology analytics, export integrity)
│   ├── test_auth.py            # Argon2id authentication and session rate limiting tests
│   ├── test_convert.py         # Format conversion and class remapping tests
│   ├── test_db.py              # SQLite CRUD, spatial clustering, and frequency aggregation
│   ├── test_evaluate.py        # Evaluator and plot generation tests
│   ├── test_frequency_chart.py # Incident frequency chart structure and live data tests
│   ├── test_pdf_export.py      # Landscape A4 PDF export tests and 2000-row stress test
│   ├── test_priority.py        # Deterministic formula v1.0 boundary and limit tests
│   ├── test_stress_matrix.py   # Comprehensive stress and negative matrix tests
│   ├── test_vertical_slice.py  # End-to-end image-to-database integration test
│   ├── test_video_stream.py    # HTTP 206 byte-range seeking and H.264 video streaming
│   └── test_web_api.py         # FastAPI endpoint tests (health, overview, triage, inference, exports)
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration template
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
git clone https://github.com/itsananyaaa/roadsense-ai.git
cd roadsense-ai

# Activate your Python 3.10+ environment (or conda/venv)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure authentication credentials (optional, defaults to admin/roadsense)
cp .env.example .env
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

### Step 4.3: Launch Options

#### Option A: Next-Gen Obsidian Web Console (Primary Interface)
```bash
python web/run.py
```
Open your browser to `http://localhost:8000`. Login with the operator credentials configured in your `.env` file (see `.env.example`):
- **Overview Dashboard:** Dynamic Catmull-Rom incident frequency curve, 3D tilted hazard cards, statistics circular bubbles, and rule-based AI Copilot.
- **Detect Damage:** Interactive Vision Studio with real-time bounding box annotations and normalized coordinates.
- **Video Intelligence:** Continuous video stream processing, sampling rate scrubber (1–5 FPS), temporal de-duplication, and HTTP 206 byte-range seeking.
- **Incident Map:** Geospatial Leaflet map powered by keyless Esri World Dark Gray Canvas with zero watermarks.
- **Priority Center:** Deterministic triage queue (Formula v1.0 descending) with one-click verify/reject controls and Landscape A4 PDF work order dispatch.
- **Model Lab:** Hardware latency analysis on Apple MPS, in-domain vs. cross-domain benchmark matrices.
- **Reports Ledger:** Searchable database inventory with instant Landscape A4 PDF, formula-injection-safe CSV (`utf-8-sig`), and RFC 7946 GeoJSON exports.

#### Option B: Legacy Streamlit Analytics Shell (Secondary Interface)
```bash
streamlit run app/main.py
```
Open your browser to `http://localhost:8501` for the multi-page analytical console.

---

## 5. Running the Automated Test Suite

RoadSense AI includes **91 automated tests** covering database CRUD, priority determinism, format conversion, hardware latency, dual-domain evaluation, FastAPI REST endpoints, Argon2id auth, landscape A4 PDF reports, CSV formula injection defense, stress/negative matrices, and Streamlit `AppTest` page initialization:

```bash
pytest tests/ -v
```
*Verified Output:*
```text
======================= 91 passed, 3 warnings in 10.71s ========================
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

$$S = \min\left(100.0, \, \max\left(0.0, \, \left(w_{\text{class}} \cdot \bar{c} \cdot 40.0\right) + \left(\min(N_{25\text{m}}, 5) \cdot 6.0\right) + \left(w_{\text{road}} \cdot 15.0\right) + \left(\frac{M_{\text{human}}}{1.5} \cdot 15.0\right)\right)\right)$$

### Priority Bands & Response SLAs:
- **Critical (Score $\ge 75$):** Immediate 24–48h emergency response (hazardous potholes).
- **High (Score $55 - 74$):** Scheduled 7-day structural maintenance (alligator fatigue cracks).
- **Moderate (Score $35 - 54$):** Routine 30-day backlog (transverse crack sealing).
- **Low (Score $< 35$):** Monitor on subsequent survey cycle (early longitudinal cracks).
- **Insufficient Info (Score $= 0$):** Undamaged road segment survey logging.

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
- **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Image Credits & Provenance:** See [docs/MEDIA_PROVENANCE.md](docs/MEDIA_PROVENANCE.md) and [docs/MEDIA_AUDIT.md](docs/MEDIA_AUDIT.md) for complete provenance records. Any test fixture that is not real field imagery is visibly labelled "Illustrative sample (not real field imagery)". Coordinates without verified EXIF GPS are strictly labelled "real image — synthetic location" or "UNKNOWN".

> **Civil Engineering Advisory Notice:**  
> *Model detection confidence is an empirical perceptual score, not an in-situ structural load-bearing assessment. RoadSense AI assists municipal road maintenance prioritization and does not replace certified geotechnical road safety audits.*

---

## 10. Security & Fail-Closed Credentials

RoadSense AI enforces a fail-closed security architecture:
- Copy `.env.example` to `.env` and configure your own secure credentials (`ROADSENSE_ADMIN_USER`, `ROADSENSE_ADMIN_PASS`, and a 32-byte hex `ROADSENSE_SESSION_SECRET` generated via `openssl rand -hex 32`).
- **Never commit `.env` to version control.** It is excluded via `.gitignore`.
- If credentials are unset, all authenticated endpoints immediately reject requests with `HTTP 401 Unauthorized`.
- Passwords are encrypted with memory-hard `Argon2id` and verified using constant-time evaluation to prevent timing side-channel attacks.
- For vulnerability reporting, see [SECURITY.md](SECURITY.md).

---

## 11. Repository Contents & Excluded Artifacts

To guarantee clean, reproducible execution while strictly protecting private data, large binary blobs and mutable local state are kept out of git:
- **Included:** Source code, web frontend, core algorithms, 91 automated tests, documentation, ReportLab TrueType fonts (`DejaVuSans`), offline Leaflet GIS assets, and curated RDD2022 demo media (`data/demo_media/`).
- **Excluded:** Local SQLite runtime databases (`data/roadsense.db`), runtime uploaded evidence (`data/evidence/`), unredacted private field media (`data/field_samples/`), multi-gigabyte raw datasets, and temporary test scratchpads.
- For the full ledger of excluded artifacts and reproduction steps, see [EXCLUDED_FROM_GIT.md](EXCLUDED_FROM_GIT.md).

---

## 12. In-Depth Project Architecture

For a comprehensive deep-dive into both the technical (YOLO26n architecture, anti-leakage dHash splitting, temporal de-duplication, Haversine clustering) and non-technical (AASHTO PCI deterioration economics, stakeholder workflows, municipal triage governance) dimensions of RoadSense AI, see [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).

---

## 13. License & Legal Notices

RoadSense AI is open-source software licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for the complete terms.

Third-party software licenses and dataset attributions (including Ultralytics AGPL-3.0 and RDD2022 CC BY 4.0) are itemized in [NOTICE](NOTICE).


