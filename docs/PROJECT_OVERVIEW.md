# RoadSense AI — Comprehensive Project Architecture & Technical Overview

> **Audience:** AI Engineering Agents, Academic Evaluators, Civil Infrastructure Directors, and Full-Stack Machine Learning Engineers.  
> **Repository:** `itsananyaaa/roadsense-ai`  
> **Platform Classification:** Automated Municipal Pavement Distress Detection, Geospatial Intelligence, and Priority Ranking Engine  
> **Active Environment:** Python 3.10.20 | PyTorch 2.12.1 (Apple MPS Acceleration) | Ultralytics 8.4.177 | FastAPI 0.115+ | Streamlit 1.58.0  
> **Verification Status:** 91 of 91 Automated Tests Passing (100% Pass Rate)

---

## 1. Executive & Non-Technical Summary

### 1.1 The Real-World Municipal Problem
Municipal departments of public works, highway authorities, and county engineers manage thousands of lane-kilometers of roadway. Traditionally, road condition assessment relies on **manual windshield surveys**: two inspectors drive slowly down roads, visually assessing distress and recording notes on clipboards or tablets.

This legacy approach suffers from four critical failures:
1. **Extreme Subjectivity:** Inspection scores depend on individual inspector bias, fatigue, and weather conditions. Two crews evaluating the same stretch often report vastly different severity ratings.
2. **Occupational Hazard & Traffic Bottlenecks:** Low-speed survey vehicles in active traffic create severe rear-end collision hazards and disrupt traffic flow.
3. **Prohibitive Cost & Lag:** Comprehensive network surveys take months and cost hundreds of thousands of dollars, meaning municipal road asset management databases are obsolete before repairs are even scheduled.
4. **Reactive Crisis Spending:** Without automated triage, cities repair roads only after citizen complaints or damage claims occur (e.g., blown tires, bent rims, vehicle accidents).

### 1.2 The RoadSense AI Solution
**RoadSense AI** transforms municipal pavement management from a slow, subjective, reactive process into an **autonomous, continuous, deterministic intelligence pipeline**:
- **Vehicle-Mounted Fleet Ingestion:** Everyday municipal vehicles (sanitation trucks, transit buses, inspector pickup trucks) mount low-cost dashcams or smartphones running continuous video capture.
- **Edge Computer Vision:** Deep convolutional backbones inspect pavement frames in real time (91.5 FPS on Apple Silicon), automatically detecting, classifying, and bounding surface distress.
- **Temporal & Spatial De-Duplication:** The platform tracks defects across consecutive video frames and physical GPS coordinates, ensuring a single physical pothole visible across 15 video frames generates **exactly one** municipal work order rather than inflating the ticket queue.
- **Deterministic Priority Ranking Engine v1.0:** Replaces black-box guesswork with an auditable mathematical formula ($S \in [0, 100]$) combining defect severity, model confidence, spatial repeat density, functional road classification, and human inspector judgment.
- **Explainable Work Orders & GIS Dispatch:** Dispatches repair crews through interactive geospatial GIS heatmaps and landscape A4 PDF work orders containing explicit maintenance actions (e.g., *Cold/Hot Asphalt Patch Compaction* vs. *Milling & Structural Overlay*).

### 1.3 Key User Personas
| Persona | Primary Goal | System Interaction |
|:---|:---|:---|
| **Public Works Director** | Network-wide health visibility, budget allocation, ROI tracking | Uses Overview Dashboard to track Network Pavement Condition Index (PCI), incident frequency trends, and budget preservation models. |
| **Pavement Asset Manager** | Triage queue prioritization, maintenance scheduling, work order dispatch | Uses Priority Center & Geospatial Incident Map to filter by severity band (Critical, High, Moderate, Low) and export verified PDF/GeoJSON work orders. |
| **Field Maintenance Crew** | Rapid site location, repair execution, verification | Receives landscape A4 dispatch sheets with GPS coordinates, defect visual crops, and exact civil engineering repair recipes. |
| **Quality Audit / Legal Counsel** | Citizen claim defense, audit trail compliance | Reviews strict location provenance (`EXIF_GPS` vs. `UNKNOWN`) and deterministic formula logs to defend municipal maintenance decisions in court. |

### 1.4 Economic & Civil Engineering Modeling
*(Disclaimer: Figures below represent theoretical planning models and illustrative assumptions, not measured field results)*

Pavement deterioration follows an exponential curve defined by the **AASHTO Pavement Condition Index (PCI)**:
- Roads spend 75% of their structural lifespan in good condition (PCI 70–100), during which routine preventive maintenance (crack sealing) costs substantially less than post-failure rebuilding.
- Once pavement drops below the "deterioration cliff" (PCI < 40), base moisture intrusion causes catastrophic fatigue failure (alligator cracking and deep potholes), requiring full-depth milling and reconstruction.
- By detecting `D00`/`D10` linear cracks early and identifying `D40` pothole clusters before sub-base collapse occurs, theoretical preservation models demonstrate substantial municipal life-cycle cost savings over reactive reconstruction.

---

## 2. Civil Engineering & Pavement Distress Science

RoadSense AI strictly adopts the international canonical distress classification established by the **RDD2022 / IEEE BigData Cup Multi-National Road Damage Benchmark**:

```
                       ┌───────────────────────────────┐
                       │     Road Surface Distress     │
                       └──────────────┬────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
       Linear Cracking                               Structural Cavities & Fatigue
      ┌───────┴───────┐                               ┌───────┴───────┐
      ▼               ▼                               ▼               ▼
     D00             D10                             D20             D40
Longitudinal      Transverse                      Alligator        Pothole
 (Wheel Path)    (Contraction)                  (Base Fatigue)  (Depth > 25mm)
 $w_{class}=0.4$ $w_{class}=0.5$                $w_{class}=0.8$ $w_{class}=1.0$
```

### 2.1 The 4 Canonical Distress Categories

| Code | Damage Category | Civil Engineering Causative Mechanism | Base Severity Weight ($w_{\text{class}}$) | Prescribed Municipal Maintenance Action | SLA Response Target |
|:---:|:---|:---|:---:|:---|:---:|
| **D00** | **Longitudinal Crack** | Linear cracks parallel to the centerline along wheel paths. Caused by heavy traffic load fatigue and weak paving joints. Early moisture entry path. | `0.4` | High-pressure routing & linear hot-pour elastomer sealant application | Subsequent survey cycle / 30-day monitor |
| **D10** | **Transverse Crack** | Cracks perpendicular to the centerline. Driven by diurnal thermal expansion/contraction cycles or reflection cracking from underlying cementitious bases. | `0.5` | Hot-pour bituminous crack sealing to arrest moisture infiltration into the base | 30 Days (Routine Backlog) |
| **D20** | **Alligator Crack** | Interconnected polygonal "crocodile skin" cracking. Represents severe structural fatigue failure of the underlying base or subgrade layers. | `0.8` | Cold asphalt milling and structural binder/wearing course overlay | 7 Days (Scheduled Structural) |
| **D40** | **Pothole** | Structural surface cavity exceeding 25 mm in depth. Formed when base water freeze-thaws or washes away under traffic, deflecting tires and risking severe accidents. | `1.0` (Hazard) | Emergency cold/hot asphalt patch compaction with mechanical tamping | 24–48 Hours (Emergency Triage) |

### 2.2 Explicitly Filtered Non-Distress Classes
The raw RDD2022 dataset contains ancillary markings that are deliberately filtered during XML ingestion:
- **`D43` & `D44` (Crosswalk & Lane Line Blur):** Surface paint wear. While important for traffic safety, paint fading does not represent structural asphalt pavement distress.
- **`D50` (Manhole Covers):** Cast iron utility infrastructure fixtures. Flagging them as potholes is a common failure mode in naive models; RoadSense AI filters them to background.
- **`Repair` (Pavement Patches):** Repaired asphalt patches indicate completed maintenance, not active defects.

---

## 3. Machine Learning & Computer Vision Pipeline

```
                                  ML PIPELINE ARCHITECTURE
                                  
  ┌───────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ RDD2022 VOC XML Files │ ──▶ │ Class Remapping &    │ ──▶ │ Anti-Leakage Sequence  │
  │ (India & Czech Sets)  │     │ Ground-Truth Cleansing│     │ & Perceptual dHash Split│
  └───────────────────────┘     └──────────────────────┘     └───────────┬────────────┘
                                                                         │
                                       ┌─────────────────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │   YOLO26n Nano Architecture   │
                       │   2.54M Params | 5.2 MB Size  │
                       └───────────────┬───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
      ┌───────────────────────────┐         ┌───────────────────────────┐
      │  In-Domain Evaluation     │         │  Zero-Shot Transfer       │
      │  (Held-Out India Split)   │         │  (Czech Republic Split)   │
      │  mAP@0.5: 0.584           │         │  mAP@0.5: 0.438 (-25.0%)  │
      └───────────────────────────┘         └───────────────────────────┘
```

### 3.1 Model Architecture Selection: Why YOLO26n?
In municipal operations, deployment takes place on vehicle edge devices (e.g., Jetson, iPad, in-vehicle NUC) and local municipal web servers without cloud GPU dependencies.
- **Parameter Footprint:** **2.54 Million parameters** with a compiled size of only **5.2 MB** (vs. 40M+ params for Faster R-CNN or 80M+ for DETR).
- **Inference Latency:** **10.9 ms per frame (91.5 FPS)** on Apple Silicon M4 (MPS acceleration).
- **Spatial Resolution & Feature Pyramids:** Road cracks possess extreme aspect ratios (hundreds of pixels long, 2–3 pixels wide). YOLO26n leverages a Path Aggregation Network (PAN-FPN) combining high-resolution spatial features for thin cracks with deep semantic features for broad potholes.

### 3.2 Anti-Leakage Data Splitting (`ml/split.py`)
In RDD2022, images were captured by smartphones mounted on survey vehicles driving continuously at ~1 FPS. Consecutive frames show the exact same pavement crack 5 meters down the road with identical background trees and road geometry.

If naive `train_test_split(random_state=42)` is applied, adjacent frames bleed across training and validation sets, allowing the network to memorize background scenery rather than learning genuine pavement distress features.

**RoadSense AI Leakage-Free Split Strategy:**
1. **Sequence Identifier Parsing:** Groups adjacent frames into route sequences based on filename timestamps and run intervals.
2. **64-Bit Perceptual Difference Hashing (dHash):** Computes difference gradient hashes across consecutive frames. If the normalized Hamming distance satisfies $H(h_i, h_j) \le 10$, the frames are clustered into the same atomic route group.
3. **Cluster-Level Partitioning:** Entire route clusters are assigned atomically to either the training set (80%) or validation set (20%), guaranteeing zero temporal or visual overlap.

### 3.3 Hardware Acceleration & Concurrency Architecture (`core/inference.py`)
- **PyTorch MPS Acceleration:** Native support for Apple Silicon Metal Performance Shaders (`torch.device("mps")`) with automatic fallback to CUDA or CPU.
- **Re-Entrant Thread Lock (`threading.RLock()`):** Prevents Apple Metal context races and buffer corruptions under concurrent asynchronous FastAPI worker threads.
- **Inference Context:** Enforces `torch.inference_mode()` to eliminate computational graph retention and prevent memory leaks.
- **Safety Limits:** Sets `Image.MAX_IMAGE_PIXELS = 100_000_000` to defend against image decompression bomb attacks.

### 3.4 Dual-Domain Evaluation & Generalization Benchmarks
Empirical artifacts generated and stored in `reports/`:

| Split / Benchmark Domain | Images Evaluated | mAP@0.5 | Pothole (`D40`) AP | Alligator (`D20`) AP | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| **In-Domain Held-Out (India)** | 1,541 | **0.584** | 0.642 | 0.591 | 10.9 ms |
| **Cross-Country Transfer (Czech)**| 2,829 | **0.438** | 0.491 | 0.412 | 11.2 ms |
| **Domain Transfer Delta** | — | **-25.0%** | -23.5% | -30.3% | — |

*Why does the domain transfer drop occur?*
Central European roads exhibit different bitumen aggregates (darker, tighter stone matrix), distinct yellow/white thermoplastic paint lines, European curb gutter geometries, and localized crack repair mastic methods that do not appear in Indian rural/urban road datasets.

---

## 4. Video Stream Intelligence & Temporal De-Duplication

Processing continuous video streams presents a major challenge: an inspection vehicle driving at 40 km/h approaches a pothole over 3 seconds. At 5 FPS, the model detects the same pothole in **15 successive frames**. Naive insertion creates 15 separate database incidents, flooding the repair queue.

```
       Frame t=0.0s             Frame t=1.0s             Frame t=2.0s
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │                  │     │                  │     │   ┌──────┐       │
   │    ┌──┐          │ ──▶ │     ┌───┐        │ ──▶ │   │ D40  │       │
   │    └──┘ (D40)    │     │     └───┘ (D40)  │     │   └──────┘ (D40) │
   │                  │     │                  │     │                  │
   └──────────────────┘     └──────────────────┘     └──────────────────┘
       Det #1 (Far)             Det #2 (Mid)             Det #3 (Near)
       
   ═══════════════════════════════════════════════════════════════════════
   TEMPORAL DE-DUPLICATION ENGINE:
   Window = 2.5s | IoU >= 0.30 OR Centroid Distance <= 0.15 | Same Class
   
   Result: 3 Raw Detections  ──▶  1 Consolidated Incident (Keyframe Selected)
```

### 4.1 De-Duplication Logic (`core/video.py`)
1. **Configurable Frame Sampling:** Analyzes frames at user-specified rates (1.0 to 5.0 FPS) using OpenCV `cv2.VideoCapture`.
2. **Temporal Window Clustering:** Maintains an active buffer of incidents within $t_{\text{window}} = 2.5\text{ seconds}$.
3. **Spatial Continuity Verification:** Two detections across successive sampled frames belong to the same physical distress if:
   $$\text{class}_A == \text{class}_B \quad \text{AND} \quad \left(\text{IoU}(A, B) \ge 0.30 \quad \text{OR} \quad \|\mathbf{c}_A - \mathbf{c}_B\| \le 0.15\right)$$
   where $\mathbf{c} = \left(\frac{x_1 + x_2}{2W}, \frac{y_1 + y_2}{2H}\right)$ is the normalized bounding box centroid.
4. **Keyframe Persistence:** The system selects the frame with the **highest confidence score and largest bounding box** (the clearest, closest vehicle view), saving this single image as the official evidence asset.
5. **Demonstrated Efficiency:** In benchmark dashcam runs, 15 raw frame detections are de-duplicated into **1 distinct physical defect (a 93.3% deduplication rate)**.
6. **Streaming & Byte-Range Seeking:** The FastAPI backend serves dashcam video via `/api/video/stream` supporting **HTTP 206 Partial Content (Range: bytes=start-end)**, allowing zero-latency timeline scrubbing in the browser.

---

## 5. Deterministic Priority Ranking Engine v1.0

Unlike opaque black-box neural networks, municipal public infrastructure spending requires **reproducible, explainable, and legally defensible prioritization**.

```
                        PRIORITY SCORING ENGINE v1.0
                        
   ┌─────────────────────────────────────────────────────────────────┐
   │ S = clamp( Class_Pts + Cluster_Pts + Road_Pts + Human_Pts, 0, 100)│
   └────────────────────────────────┬────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
     Class & Conf             Repeat Density            Road & Human
      (Max 40 pts)             (Max 30 pts)             (Max 30 pts)
   w_class * conf * 40    min(N_25m, 5) * 6       w_road*15 + w_human*15
```

### 5.1 The Mathematical Formula
The priority score $S \in [0, 100]$ is computed deterministically:

$$S = \min\left(100.0, \, \max\left(0.0, \, P_{\text{class}} + P_{\text{cluster}} + P_{\text{road}} + P_{\text{human}}\right)\right)$$

Where:
1. **Damage Class & Confidence Points ($P_{\text{class}} \le 40.0$):**
   $$P_{\text{class}} = w_{\text{class}} \cdot \bar{c} \cdot 40.0$$
   - $w_{\text{class}} \in \{0.4 (\text{D00}), \, 0.5 (\text{D10}), \, 0.8 (\text{D20}), \, 1.0 (\text{D40})\}$
   - $\bar{c} \in [0.0, 1.0]$: Maximum detection confidence for that defect class.
2. **Spatial Repeat Observation Density ($P_{\text{cluster}} \le 30.0$):**
   $$P_{\text{cluster}} = \min(N_{25\text{m}}, \, 5) \cdot 6.0$$
   - $N_{25\text{m}}$: Number of other confirmed distress incidents recorded within a 25-meter radius (calculated via Haversine great-circle distance).
   - Captures localized structural road failure (a pothole surrounded by cracks is far more dangerous than an isolated crack).
3. **Functional Road Classification Points ($P_{\text{road}} \le 15.0$):**
   $$P_{\text{road}} = w_{\text{road}} \cdot 15.0$$
   - $w_{\text{road}} \in \{1.0 (\text{Primary Arterial}), \, 0.7 (\text{Secondary Collector}), \, 0.4 (\text{Residential Local}), \, 0.5 (\text{Unknown})\}$.
4. **Human Reviewer Urgency Modifier ($P_{\text{human}} \le 15.0$):**
   $$P_{\text{human}} = \left(\frac{M_{\text{human}}}{1.5}\right) \cdot 15.0$$
   - $M_{\text{human}} \in [0.5, 1.5]$: Field engineer inspection override multiplier.

### 5.2 Priority Bands & Municipal Action
| Score Band | Priority Level | Action & Operational SLA | UI Indicator |
|:---:|:---:|:---|:---:|
| **$\ge 75.0$** | **CRITICAL** | Emergency 24–48 hour response; dispatch asphalt patch crew | Neon Pink / Crimson (`#ef4444`) |
| **$55.0 - 74.9$** | **HIGH** | Scheduled 7-day repair; assign structural milling/overlay | Neon Amber (`#f97316`) |
| **$35.0 - 54.9$** | **MODERATE** | Routine 30-day maintenance queue; joint/crack sealant | Deep Orange (`#f59e0b`) |
| **$< 35.0$** | **LOW** | Monitor on subsequent survey cycle (6–12 months) | Emerald Green (`#10b981`) |
| **$0.0$** | **INSUFFICIENT_INFO** | No damage detections present; road segment logged clean | Slate Gray (`#64748b`) |

---

## 6. Geospatial Intelligence & Strict Provenance Guarantee

### 6.1 The Zero-Hallucination GPS Principle
RoadSense AI enforces a strict 4-tier geospatial provenance enum (`core/geo.py`):
```python
class LocationProvenance(str, Enum):
    EXIF_GPS = "EXIF_GPS"                              # Extracted from hardware camera EXIF
    USER_ENTERED = "USER_ENTERED"                      # Confirmed by municipal engineer
    EXPLICIT_SYNTHETIC_DEMO = "EXPLICIT_SYNTHETIC_DEMO"# Seeded offline benchmark fixture
    UNKNOWN = "UNKNOWN"                                # No coordinates; excluded from map
```

- **Null Island Defense:** Coordinates within $10^{-5}$ degrees of `(0.0, 0.0)` are automatically rejected as invalid sensor defaults.
- **Map Exclusion:** Any record marked `UNKNOWN` is omitted from the geospatial incident map and routed to an "Unlocated Evidence Queue" requiring manual field attribution.

### 6.2 Haversine Geospatial Clustering (`core/db.py`)
To compute $N_{25\text{m}}$, the database layer uses the Haversine great-circle formula:
$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
where $R = 6,371,000\text{ m}$. Queries use pre-indexed latitude/longitude bounding boxes before applying exact distance math, achieving sub-millisecond execution over thousands of incidents.

---

## 7. Dual Frontend Architecture & Design Systems

RoadSense AI features a **dual-frontend architecture** serving both modern executive stakeholders and data science evaluators:

### 7.1 Next-Gen Obsidian Web Console (`http://localhost:8000`)
Built with vanilla HTML5, custom CSS3, and modern JavaScript (zero heavyweight Node/React dependencies):
- **Obsidian Dark Slate Theme:** Deep charcoal `#0c0d14` background with glassmorphic cards (`rgba(25, 27, 44, 0.6)`), 1px subtle borders (`rgba(255, 255, 255, 0.08)`), and high-contrast neon accents (hot pink `#ff2a85`, electric cobalt `#3b82f6`).
- **Interactive Catmull-Rom Spline:** Real-time incident frequency chart plotting historical trends across 1-week, 30-day, and 12-month windows with cursor-tracking tooltips.
- **3D Tilted Hazard Cards:** Perspective-projected cards displaying critical pothole alerts.
- **Keyless Dark GIS Basemap:** Custom Leaflet setup using free, keyless **Esri World Dark Gray Canvas** (`Canvas/World_Dark_Gray_Base` + `World_Dark_Gray_Reference`), eliminating all watermark clutter.
- **Animated SVG Character Crew:** An interactive 4-member mascot crew (Pylon, Sensor Box, Dome LiDAR, Road Stud) rendered in inline SVG, reacting dynamically to user focus, password masking (shy privacy mode), login failure, and success via a unified `requestAnimationFrame` lerp loop.
- **AI Copilot:** An offline rule-based municipal assistant answering natural-language queries about triage backlog, critical potholes, and maintenance budgets.

### 7.2 Classic Streamlit Analytics Console (`http://localhost:8501`)
A multi-page scientific evaluation suite across 7 dedicated pages (`app/pages/`):
1. `1_Detect_Damage.py`: Single-image upload, live MPS inference, bounding box visualizer.
2. `2_Video_Intelligence.py`: Dashcam video processor, FPS scrubber, keyframe gallery.
3. `3_Incident_Map.py`: Full-screen Leaflet geospatial map with evidence drawers.
4. `4_Priority_Center.py`: Sortable triage queue with human review buttons (Verify, Modify, Reject).
5. `5_Model_Lab.py`: Precision-Recall curves, confusion matrices, and latency tables.
6. `6_Reports_History.py`: Database ledger with PDF, CSV, and GeoJSON export.
7. `7_About_Methodology.py`: Mathematical formulas, citations, and CC BY 4.0 license details.

---

## 8. Export Integrity, Security & Governance

### 8.1 Unified PDF Engine (`core/reports_pdf.py`)
Produces publication-grade landscape A4 audit reports using ReportLab:
- **Two-Pass Numbered Canvas:** Dynamic "Page X of Y" page numbering.
- **Bundled TrueType Fonts:** Shipped with `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf`, guaranteeing zero font missing errors regardless of host OS.
- **Export Documents:**
  1. *Dispatch Work Orders (PDF):* Itemized repair orders with GPS, defect crops, and maintenance actions.
  2. *Incident Audit Ledger (PDF):* Comprehensive municipal incident inventory.
  3. *Methodology & Scientific Report (PDF):* 4-page whitepaper containing dataset splits, confusion matrices, PR curves, and formula derivations.

### 8.2 CSV & GeoJSON Export Hardening
- **CSV Formula Injection Defense:** Sanitizes all string fields by prepending an apostrophe (`'`) to any value starting with `=`, `+`, `-`, or `@`, preventing spreadsheet remote execution exploits. Encoded with UTF-8 BOM (`utf-8-sig`) for Excel compatibility.
- **RFC 7946 GeoJSON Compliance:** Strictly formats coordinates in standard `[longitude, latitude]` order (rejecting inverted latitude-first formatting).

### 8.3 Security & Fail-Closed Authentication
- **Fail-Closed Credentials:** Reads `ROADSENSE_ADMIN_USER` and `ROADSENSE_ADMIN_PASS` from environment variables. If missing, the backend fails closed and refuses access.
- **Argon2id Hashing:** Password hashed at startup with memory-hard Argon2id (`time_cost=2, memory_cost=65536, parallelism=2`).
- **Constant-Time Verification:** Uses precomputed dummy hashes on unknown usernames to defend against side-channel timing attacks.
- **HMAC-SHA256 Ephemeral Sessions:** Cookie-based `roadsense_session` (`httpOnly=True`, `SameSite=Lax`, `Secure` when HTTPS) valid for 24 hours.
- **Sliding-Window Rate Limiting:** Limits failed logins to 5 attempts per 60 seconds per IP address (returning HTTP 429).
