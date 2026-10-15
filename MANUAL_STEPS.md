# RoadSense AI — Manual Operator Steps Guide

This document lists all operations that must be performed by the user/operator outside of automated local testing (e.g. running full cloud GPU training on Google Colab, copying checkpoints, running offline hardware tests).

---

## 1. Cloud Model Training on Google Colab (Free or Pro T4/V100/A100 GPU)

Because full model training across the entire RDD2022 dataset requires significant GPU compute (~1-2 hours on GPU vs 10+ hours on a laptop CPU), we provide an end-to-end self-contained Jupyter notebook: `ml/colab_train.ipynb`.

### Step 1.1: Open Notebook in Google Colab
1. Navigate to [Google Colaboratory](https://colab.research.google.com/).
2. Select **Upload** and upload `ml/colab_train.ipynb` from this project repository.
3. In Colab, go to **Runtime** > **Change runtime type**, select **T4 GPU** (or available GPU), and click **Save**.

### Step 1.2: Execute Top-to-Bottom
The notebook contains self-contained cells that execute in sequence:
1. **GPU Check:** Verifies `nvidia-smi` and PyTorch CUDA availability.
2. **Mount Google Drive:** Mounts `/content/drive/MyDrive/RoadSense_AI` so training checkpoints are never lost if the Colab session disconnects.
3. **Download Dataset:** Downloads `RDD2022_India.zip` (526 MB) and `RDD2022_Czech.zip` (257 MB) from the official Figshare repository.
4. **Audit & Conversion:** Runs the leakage-aware sequence split and converts Pascal-VOC XML annotations to standard YOLO format.
5. **Model Training:** Trains `yolo26n` (or `yolo11n`) for 50 epochs with early stopping (`patience=10`), auto-saving `last.pt` and `best.pt` directly to your Google Drive.
   - *Resumability:* If interrupted, re-running the cell automatically passes `resume=True` pointing to `last.pt`.
6. **Evaluation & Artifact Generation:** Computes:
   - In-domain validation metrics on India held-out set.
   - Domain-shift zero-shot metrics on Czech held-out set.
   - Confusion matrices, PR curves, and per-class mAP@0.5 and mAP@0.5:0.95.
7. **Package Results:** Bundles `best.pt`, `eval_india.json`, `eval_czech.json`, confusion matrices, and sample TP/FP/FN imagery into `roadsense_artifacts.zip` in Google Drive.

### Step 1.3: Expected Result
- A trained checkpoint file `best.pt` (~5.3 MB).
- Evaluation reports `eval_metrics.json` showing per-class precision, recall, and mAP.
- Packaging completes with a download link or Drive file path.

---

## 2. Deploying Trained Checkpoint to Local Workspace

Once Colab finishes training:
1. Download `roadsense_artifacts.zip` from your Google Drive.
2. Unzip the contents into your local workspace root `/Users/ananyabiju/Desktop/roadsense-ai`:
   ```bash
   cp ~/Downloads/best.pt /Users/ananyabiju/Desktop/roadsense-ai/models/best.pt
   cp ~/Downloads/eval_*.json /Users/ananyabiju/Desktop/roadsense-ai/reports/
   cp ~/Downloads/*.png /Users/ananyabiju/Desktop/roadsense-ai/reports/
   ```
3. Verify local loading:
   ```bash
   /opt/miniconda3/envs/ml/bin/python -c "
   from ultralytics import YOLO
   model = YOLO('models/best.pt')
   print('Model loaded successfully. Classes:', model.names)
   "
   ```
4. Expected Output:
   ```text
   Model loaded successfully. Classes: {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}
   ```

---

## 3. Running Local Smoke-Test Training (On-Device Apple Silicon M4)

To verify the training and evaluation pipeline on your local Mac without downloading gigabytes of imagery, we provide a local smoke-test script:
```bash
/opt/miniconda3/envs/ml/bin/python ml/train.py --mode smoke --device mps --epochs 2 --batch 8
```
- **Expected Execution Time:** ~45-90 seconds.
- **Expected Output:**
  - Creates a miniature dataset slice in `data/smoke/`.
  - Runs 2 epochs of fine-tuning on Apple Metal Performance Shaders (`mps`).
  - Outputs `models/smoke_test.pt` with metadata indicating `"smoke-test checkpoint, not evaluated"`.
  - Runs validation and writes `reports/smoke_eval.json`.

---

## 4. Launching the RoadSense AI Platform

To launch the web interface locally:
```bash
/opt/miniconda3/envs/ml/bin/streamlit run app/main.py
```
- **Expected Output:**
  ```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
  ```
- The application will automatically detect whether `models/best.pt` is present. If missing, it displays an informational status banner and offers the one-click **Interactive Guided Demo Mode** with pre-cached sample images.

---

## 5. Offline Project-Expo Demonstration Verification

To test that the application runs completely offline (crucial for university project expos with unreliable Wi-Fi):
1. Disconnect your Mac from Wi-Fi.
2. Launch the app:
   ```bash
   /opt/miniconda3/envs/ml/bin/streamlit run app/main.py
   ```
3. Open `http://localhost:8501` in your browser.
4. Click **"Load Demo Dataset"** in the sidebar.
5. Verify that:
   - Sample images and video frames render from local disk assets.
   - Incident records load from local SQLite.
   - Leaflet/Folium map displays graceful offline fallback tile graphics if tile servers are unreachable.
   - Priority calculations, incident filters, and report exports (CSV/JSON) execute instantaneously.
