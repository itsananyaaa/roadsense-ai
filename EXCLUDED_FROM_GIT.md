# RoadSense AI — Excluded Artifacts Ledger

This document accounts for all local artifacts, data directories, model checkpoints, runtime databases, and sensitive configuration files that are deliberately excluded from git version control, the rationale for their exclusion, and exact instructions on how to regenerate or obtain them locally.

---

## 1. Inventory of Excluded Items

| Excluded Path / Pattern | Classification | Reason for Exclusion from Git | How to Regenerate or Obtain Locally |
|:---|:---|:---|:---|
| `.env`, `.env.*` (except `.env.example`) | Secrets & Credentials | Contains operator passwords, session keys, and deployment credentials. Must never be published. | Copy `.env.example` to `.env` and set local values (`ROADSENSE_ADMIN_USER`, `ROADSENSE_ADMIN_PASS`, `ROADSENSE_SESSION_SECRET`). |
| `data/roadsense.db`, `*.db-wal`, `*.db-shm` | Runtime Database | SQLite operational database containing mutable municipal records and local triage history. | Regenerated automatically on first launch or seeded via `core/demo.py:seed_demo_database()` / web console "Seed Demo Fixtures". |
| `data/evidence/` | Runtime Media Uploads | Stores user-uploaded inspection imagery, transient bounding-box renders, and temporary files created during inference. | Populated dynamically as users upload images or process videos via the Detect Damage or Video Intelligence consoles. |
| `data/field_samples/` | Private Field Imagery | Raw smartphone or dashcam video/photo field captures that may contain unredacted GPS, vehicle license plates, or bystander faces. | Place your own field photos or dashcam videos in `data/field_samples/` for local testing. |
| `data/smoke/`, `scratch/` | Temporary Test Scratchpad | Temporary files, transcode test videos, and test suites' temporary scratch directories. | Generated dynamically by automated pytest runs or video generation benchmarks. |
| `models/smoke_test.pt` | Transient Smoke Weights | Miniature 1-epoch local verification checkpoint (5.2 MB) used for test assertion. | Generated locally by running `python ml/train.py --mode smoke`. |
| Base weights: `yolo26n.pt`, `yolo11n.pt`, `yolov8n.pt` | Upstream Base Models | Standard Ultralytics pretrained baseline weights (5.2–6.5 MB). | Downloaded automatically on demand by the `ultralytics` package when loading base models. |
| Production checkpoint: `models/best.pt` (> 50 MB if full) | Production ML Weights | Full fine-tuned production weights trained on Google Colab GPU. | Follow [MANUAL_STEPS.md](MANUAL_STEPS.md) to train on Google Colab using `ml/colab_train.ipynb`, then copy `best.pt` to `models/best.pt`. |
| Raw RDD2022 dataset archives (`RDD2022/`, `*.zip`, `*.tar.gz`) | External Training Data | Multi-gigabyte raw benchmark archives (India: 7,706 images; Czech: 2,829 images; Japan: 10,506 images; Figshare DOI 10.6084/m9.figshare.21431547.v1). | Downloaded automatically in Colab via `ml/colab_train.ipynb` or directly from Figshare. |
| `__pycache__/`, `*.pyc`, `*.so` | Python Bytecode | Compiled Python bytecode cache files. | Generated automatically by the Python runtime. |
| `.pytest_cache/`, `.coverage`, `htmlcov/` | Test Artifacts | Pytest execution state and coverage measurement files. | Generated dynamically by running `pytest`. |
| `venv/`, `.venv/`, `env/`, `miniconda/` | Python Environments | Local virtual environments and package installations. | Created via `python3 -m venv venv && pip install -r requirements.txt`. |
| `.DS_Store`, `._*`, `.vscode/`, `.idea/` | OS & Editor Metadata | macOS Finder metadata and local IDE workspace settings. | OS-specific cache files. |

---

## 2. Retained & Published Fixtures

To ensure the repository is 100% runnable, reproducible, and verifiable immediately upon fresh clone, the following curated fixtures are **included** in git:

1. **`data/demo_media/`:**
   - 8 canonical distress reference images (`demo_alligator_high.jpg`, `demo_pothole_critical.jpg`, `demo_longitudinal_low.jpg`, `demo_transverse_moderate.jpg`, etc.) derived from RDD2022 under CC BY 4.0.
   - All EXIF GPS metadata stripped from image headers to prevent location confusion.
   - Ground-truth manifest `data/demo_media/manifest.json` with SHA-256 checksums and dataset citations.
2. **`reports/`:**
   - Grounded benchmark results (`eval_india.json`, `eval_czech.json`, `eval_benchmark.json`).
   - Confusion matrix (`reports/confusion_matrix.png`) and Multi-class Precision-Recall curve (`reports/pr_curve.png`).
   - Demo media live checkpoint audit (`reports/demo_media_check.json`).
3. **`core/fonts/`:**
   - Unicode TrueType fonts (`DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`) for 100% offline ReportLab PDF report generation.
4. **`web/static/assets/leaflet/`:**
   - Bundled local Leaflet CSS and JavaScript for 100% offline geospatial GIS mapping.
