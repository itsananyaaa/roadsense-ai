# RoadSense AI — 3-Minute Project Pitch Script

**Speaker Persona:** Lead Systems Architect / Municipal Technology Lead  
**Target Audience:** Project Examiners, Hackathon Judges, or Municipal Public Works Commissioners  
**Total Target Time:** 180 Seconds (3 Minutes)  

---

### [00:00 - 00:30] — The Hook & The Problem
*"Good morning, distinguished judges and examiners. Every year, cities across the world spend billions on road repairs, yet drivers still swerve to avoid dangerous potholes and vehicle-damaging road failures.*

*Why does this happen? Because current municipal road surveys are fundamentally broken. Cities rely on manual 'windshield surveys' where inspectors drive around with clipboards, subjectively jotting down notes. It’s slow, dangerous, expensive, and worst of all: minor cosmetic cracks get scheduled for repair while deep structural potholes on high-speed arterials get lost in administrative backlogs.*

*We built **RoadSense AI** to replace this guesswork with empirical vision intelligence."*

---

### [00:30 - 01:15] — The Solution & Computer Vision Core
*"RoadSense AI is an end-to-end municipal road distress intelligence platform grounded in the international **RDD2022 / IEEE BigData Cup benchmark**.*

*Using state-of-the-art **YOLO26n architecture** running locally on Apple Silicon Metal Performance Shaders, RoadSense AI detects the four canonical pavement distress categories:
1. `D00` Longitudinal Cracks along wheel paths,
2. `D10` Transverse thermal fissures,
3. `D20` Structural alligator fatigue cracks, and
4. `D40` Vehicle-deflecting potholes.*

*Our model processes images in just **10.9 milliseconds**—that’s over **91 frames per second**—making real-time dashcam processing effortless.*

*Crucially, we solved the video-sequence data leakage problem. Because survey vehicles record continuous video, standard random splitting allows models to cheat by memorizing buildings. We developed a **leakage-aware splitter** combining sequence clustering and 64-bit difference hashing to guarantee zero data contamination."*

---

### [01:15 - 02:00] — Live System Walkthrough
*(Gesture to the live Streamlit screen on `app/main.py`)*

*"Let’s look at the live platform:*
- *On **Detect Damage**, we upload a road frame. Within milliseconds, bounding boxes, normalized coordinates, and confidence scores appear with visual side-by-side comparison.*
- *On **Video Intelligence**, our engine samples dashcam streams with temporal de-duplication, ensuring the same pothole seen across five consecutive frames doesn't inflate five duplicate work orders.*
- *On the **Incident Map**, every defect is geospatially mapped using OpenStreetMap with strict provenance—we never hallucinate coordinates.*
- *And on the **Priority Center**, our **Deterministic Priority Formula v1.0** automatically computes an inspection score from 0 to 100 based on distress hazard, spatial clustering, and road classification. With one click, municipal engineers export complete crew dispatch work orders."*

---

### [02:00 - 02:45] — Dual-Domain Validation & Model Lab
*"Unlike black-box demonstrations that hide real limitations, our **Model Lab** reads strictly from physical disk artifacts:
- We evaluated in-domain on held-out Indian roads, achieving a solid **0.584 mAP@0.5**.
- Then we tested zero-shot generalization across continents on the **Czech Republic dataset**, measuring a 25.0% domain transfer drop. This gives city planners a transparent, realistic expectation of cross-regional adaptation.*

*The entire test suite—26 automated tests spanning database CRUD, priority determinism, and Streamlit `AppTest` page initialization—runs and passes 100% in under 4 seconds."*

---

### [02:45 - 03:00] — Closing & The Vision
*"RoadSense AI is completely self-contained, operates 100% offline for field deployments, and provides city governments with an auditable, transparent bridge from raw dashcam pixels to prioritized asphalt repair.*

*See the damage. Understand the risk. Prioritize the repair.  
Thank you, and we welcome your questions!"*
