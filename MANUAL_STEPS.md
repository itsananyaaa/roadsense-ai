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
2. Unzip the contents into your local repository workspace root:
   ```bash
   cp ~/Downloads/best.pt models/best.pt
   cp ~/Downloads/eval_*.json reports/
   cp ~/Downloads/*.png reports/
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

---

## 6. Adding Real User-Captured Field Photos

Operators can capture real road distress photographs with smartphones or handheld cameras and register them into RoadSense AI:

### Step 6.1: Image Format & EXIF Preparation
1. **Supported Formats:** JPEG (`.jpg`, `.jpeg`) or PNG (`.png`).
2. **HEIC Notice:** iOS devices capture `.heic` by default. Convert HEIC to JPEG before adding:
   ```bash
   # macOS built-in sips tool:
   sips -s format jpeg input.heic --out data/field_samples/my_pothole.jpg
   ```
3. **EXIF GPS Geotagging:** Ensure Location Services are enabled on the camera so standard EXIF GPS tags are embedded. RoadSense AI's `core/geo.py` automatically reads latitude/longitude and assigns `EXIF_GPS` provenance. If GPS is missing or privacy-stripped, the system truthfully assigns `UNKNOWN` provenance without fabricating coordinates.
4. **Orientation:** EXIF orientation tags are automatically handled upright via `PIL.ImageOps.exif_transpose`.

### Step 6.2: Copy and Register
Place your photos in `data/field_samples/` and run the media builder:
```bash
cp /path/to/my_photo.jpg data/field_samples/
/opt/miniconda3/envs/ml/bin/python ml/build_demo_media.py
```
- **Result:**
  - Registers the asset into `data/demo_media/manifest.json` with `origin = "user_field_capture"`.
  - Computes SHA-256 checksum for immutable traceability.
  - Automatically loads the field photo as a sample tile in the web console (`Detect Damage` tab) with live model inference.

---

## 7. Recording and Adding a Real Dashcam Video

The video intelligence pipeline analyzes phone/dashcam road survey recordings with temporal de-duplication:

### Step 7.1: Recording Guidelines
1. **Mounting:** Secure phone or dashcam to windshield or dashboard facing directly forward along traffic direction.
2. **Resolution & Duration:** Recommended 1080p (`1920x1080`) at 30 FPS, duration between 10 and 60 seconds (max file size: ~100 MB).
3. **Lighting:** Record in daylight; avoid direct head-on sun glare or heavy downpours.

### Step 7.2: Codec & Transcoding
Browsers require standard H.264 video with AAC audio for web playback. If recorded in HEVC (`.mov`), transcode with `ffmpeg`:
```bash
ffmpeg -i input.mov -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k data/field_samples/video/dashcam_survey_real.mp4
```

### Step 7.3: Register Video in Manifest
Run the media builder:
```bash
/opt/miniconda3/envs/ml/bin/python ml/build_demo_media.py
```
- **Result:**
  - The video is registered in `data/demo_media/manifest.json` under `videos` with `origin = "user_field_capture"`, resolution, duration, and privacy disclaimer.
  - The video is processed through `core/video.py` for live frame extraction and de-duplication.
  - *Note on GPS:* RoadSense AI does **not** infer or hallucinate GPS coordinates from video streams. Video incidents receive `UNKNOWN` location provenance unless explicit external spatial telemetry is supplied.
