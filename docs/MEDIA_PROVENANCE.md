# RoadSense AI — Media Provenance & Attribution

This document records the provenance, origin classification, licensing, and attribution terms for all image and video assets used within RoadSense AI, complying with the **Zero-Fabrication & Strict Evidence-Over-Assertion Integrity Mandate**.

---

## 1. Primary Benchmark Dataset: RDD2022

RoadSense AI evaluates road damage classification and detection architectures using the **Road Damage Dataset 2022 (RDD2022)** released through the IEEE BigData Cup / Global Road Damage Detection Challenge (CRDDC 2022).

- **Full Citation:**  
  Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2022). *Road Damage Dataset 2022 (RDD2022)*. figshare.  
  DOI: [10.6084/m9.figshare.21431547.v1](https://doi.org/10.6084/m9.figshare.21431547.v1)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Permitted Uses:** Sharing, adaptation, and commercial/non-commercial research with appropriate credit and indication of modifications.
- **Classes Utilized:**
  - `D00`: Wheel mark / Longitudinal Crack
  - `D10`: Equal interval / Transverse Crack
  - `D20`: Partial / Alligator (Fatigue) Crack
  - `D40`: Pothole / Rutting / Bump
- **Ground Truth Format:** Pascal-VOC XML annotations with pixel bounding box coordinates `[xmin, ymin, xmax, ymax]`.
- **Location Status:** RDD2022 camera frames do not include authentic GPS coordinates. In RoadSense AI, any demonstration incidents linked to RDD2022 images are tagged strictly as `location_provenance = 'EXPLICIT_SYNTHETIC_DEMO'` and visibly badged **"real image — synthetic location"**.

---

## 2. User Field Captures (`data/field_samples/`)

RoadSense AI supports genuine field imagery captured by municipal inspectors and mobile survey teams.

- **Storage Location:** `data/field_samples/` (images) and `data/field_samples/video/` (dashcam clips).
- **Registration Tool:** `ml/build_demo_media.py` automatically indexes all valid captures into `data/demo_media/manifest.json`.
- **Origin Classification:** `user_field_capture`
- **Location Provenance Rules:**
  - If valid exchangeable image file format (EXIF) GPS coordinates are embedded in the JPEG metadata, the incident is tagged with `location_provenance = 'EXIF_GPS'`.
  - If GPS metadata is absent, the incident is assigned `location_provenance = 'UNKNOWN'`. Coordinates are **never** invented or halluncinated.
- **Privacy & Consent Standards:**
  - Field photos must focus on municipal asphalt surface distress.
  - Photos must not contain identifiable faces or private vehicle license plates.
  - Images placed in `data/field_samples/` are registered with a consent/privacy record confirming compliance.
- **Format Requirements:**
  - Standard JPEG (`.jpg`, `.jpeg`) or PNG (`.png`).
  - HEIC format (Apple iPhone default) is rejected with clear instructions to export as JPEG (`sips -s format jpeg input.heic --out output.jpg`) to avoid unverified binary codec dependencies.
  - EXIF orientation tags (1–8) are automatically normalized to upright orientation prior to inference.

---

## 3. Shipped Illustrative Demonstration Fixtures

When the 13.26 GB RDD2022 dataset archive is not present on the local file system, RoadSense AI operates with verified test fixtures located in `data/demo/`:

- **Classification:** `synthetic_demo`
- **Mandatory Display Label:**  
  **"Illustrative sample (not real field imagery)"**
- **Characteristics:**
  - Procedurally generated synthetic asphalt textures with simulated geometric cracking patterns.
  - Video stream `dashcam_inspection_demo.mp4` is an illustrative rendered vehicle route simulating continuous road inspection.
  - Coordinates associated with synthetic fixtures are explicitly tagged `EXPLICIT_SYNTHETIC_DEMO`.
- **Purpose:** Ensure 100% offline test suite reproducibility (82+ pytest unit/integration tests) and interface verification without requiring multi-gigabyte dataset downloads.

---

## 4. Image Credits Summary

| Resource | Origin & Author | License | Usage in RoadSense AI |
| :--- | :--- | :--- | :--- |
| **RDD2022 Dataset** | Arya et al. / Univ. of Tokyo & IIT Roorkee | CC BY 4.0 (DOI: 10.6084/m9.figshare.21431547.v1) | Benchmark evaluation and held-out demo samples |
| **User Field Photos** | Municipal survey team | Contributor consent / Apache 2.0 | Real field verification with EXIF GPS |
| **Illustrative Fixtures** | RoadSense AI Engine | Apache 2.0 (Synthetic procedurally rendered) | Offline test execution & interface verification |
| **Leaflet Map Engine** | Vladimir Agafonkin / Leaflet contributors | BSD-2-Clause | Local vendored interactive GIS display |
