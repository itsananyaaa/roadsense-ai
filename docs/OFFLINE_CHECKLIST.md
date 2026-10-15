# RoadSense AI — 100% Offline Expo Demonstration Checklist

Use this checklist prior to presenting RoadSense AI in environments where internet connectivity is unavailable, intermittent, or restricted.

---

## 1. Pre-Flight Verification Checklist

- [ ] **Check 1: Python Environment Activated**
  ```bash
  which python
  # Ensure it points to your project environment (/opt/miniconda3/envs/ml/bin/python)
  ```

- [ ] **Check 2: Local Model Checkpoint Exists**
  ```bash
  ls -lh models/smoke_test.pt
  # Must exist and be ~5.2 MB
  ```

- [ ] **Check 3: Evaluation Reports & Figures Cached**
  ```bash
  ls -la reports/
  # Must contain: confusion_matrix.png, pr_curve.png, eval_india.json, eval_czech.json, eval_benchmark.json
  ```

- [ ] **Check 4: Database Seeded with Calibrated Records**
  ```bash
  /opt/miniconda3/envs/ml/bin/python -c "
  from core.db import get_incident_count
  from core.demo import seed_demo_database
  count = get_incident_count()
  if count == 0:
      seed_demo_database()
      count = get_incident_count()
  print(f'Database ready with {count} verified survey incidents.')
  "
  ```

- [ ] **Check 5: Pre-Generate Demo Dashcam Video**
  ```bash
  /opt/miniconda3/envs/ml/bin/python -c "
  from pathlib import Path
  from core.video import generate_demo_dashcam_video
  vid = Path('data/demo/dashcam_inspection_demo.mp4')
  generate_demo_dashcam_video(vid, duration_sec=10, fps=15)
  print('Demo dashcam video cached successfully.')
  "
  ```

- [ ] **Check 6: Execute Full Offline Test Suite**
  ```bash
  /opt/miniconda3/envs/ml/bin/pytest tests/ -v
  # Expected: 26 passed in ~3.5s
  ```

---

## 2. Launching in Offline Expo Mode

1. **Launch Streamlit:**
   ```bash
   streamlit run app/main.py --server.headless=true
   ```
2. **Access Console:** Open `http://localhost:8501`.
3. **If Wi-Fi is Completely Off:**
   - On **Incident Map** (`3_Incident_Map.py`), set **Map Display Engine** to `"Offline Plotly Scatter (No Tile Dependency)"`.
   - On **Video Intelligence** (`2_Video_Intelligence.py`), select `"Use Calibrated Demo Dashcam Stream"` and click `"Generate & Analyze Demo Stream"`.
   - On **Detect Damage** (`1_Detect_Damage.py`), click `"Load Calibrated Demo Image"`.

---

## 3. Demonstration Flow Walkthrough

1. **Overview Dashboard:** Show the system health card, Apple MPS accelerator status, and network KPI metrics.
2. **Detect Damage:** Run live inference on a sample image. Show the detection table, bounding box coordinates, and deterministic priority score.
3. **Video Intelligence:** Run the dashcam processor. Point out the **temporal de-duplication** preventing duplicate incident inflation.
4. **Incident Map:** Inspect the spatial defect clusters. Click an incident to open the photo evidence drawer.
5. **Priority Center:** Show the deterministic ranking queue. Adjust the municipal policy sliders and toggle simulation to demonstrate live re-ranking. Download the municipal dispatch work orders.
6. **Model Lab:** Show that all metrics and curves link to real physical files in `reports/`. Explain the 25% domain transfer gap on the Czech Republic dataset.
7. **Reports & History:** Search by road name, filter by status, and download full CSV and GeoJSON records.
8. **About & Methodology:** Review the mathematical formula, RDD2022 dataset citation, and CC BY 4.0 license.
