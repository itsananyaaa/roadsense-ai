# RoadSense AI — System State

**Last Updated:** 2026-10-15T20:58:00+04:00  
**Current Phase:** Phase 3 — Vertical Slice First (COMPLETED) -> Transitioning to Phase 4 (ML Pipeline)  
**Next Action:** Implement Phase 4 ML Pipeline: audit script (`ml/audit.py`), converter with round-trip checks (`ml/convert.py`), leakage-aware splitter (`ml/split.py`), training script with smoke mode (`ml/train.py`), dual-domain evaluator (`ml/evaluate.py`), and self-contained Colab notebook (`ml/colab_train.ipynb`).  
**Open Blockers:** None.

---

## Acceptance Criteria Status Checklist

| # | Acceptance Criterion | Status | Evidence / Notes |
|---|---|---|---|
| 1 | Clean setup succeeds | PASSED | Environment verified: Python 3.10.20 + PyTorch 2.12.1 (MPS) + Ultralytics 8.4.177. `requirements.txt` created. |
| 2 | App launches with no unhandled exception | PASSED | Verified via `pytest tests/test_app_foundation.py` (`AppTest.from_file('app/main.py')` passed). |
| 3 | Valid image completes real inference with valid weights | PASSED | Verified in `tests/test_vertical_slice.py` (real YOLO inference on Apple MPS). |
| 4 | Outputs include class, confidence, bbox | PASSED | Verified: class code, human name, confidence, and `[x1, y1, x2, y2]` coordinates generated and displayed. |
| 5 | Invalid uploads give helpful errors | PASSED | Size & format checks in `core/inference.py` and `app/pages/1_Detect_Damage.py`. |
| 6 | Video honors documented sampling config | Pending Phase 5B | Video processor planned. |
| 7 | Incident CRUD + filter + export works in SQLite | PASSED | Parameterized CRUD verified in `tests/test_db.py`. |
| 8 | Priority is deterministic for same inputs+config | PASSED | Determinism & boundary limits verified in `tests/test_priority.py`. |
| 9 | Missing location never yields invented coordinates | PASSED | Strict provenance enum enforced (`EXIF_GPS`, `USER_ENTERED`, `EXPLICIT_SYNTHETIC_DEMO`, `UNKNOWN`). |
| 10 | Missing weights handled transparently | PASSED | `core/health.py` and UI status badge distinguish trained vs baseline vs missing. |
| 11 | Demo mode clearly distinguished from live inference | PASSED | `core/demo.py` seeds explicit `EXPLICIT_SYNTHETIC_DEMO` provenance. |
| 12 | Every displayed evaluation figure links to a real artifact | Pending Phase 4 | Model Lab reads strictly from `reports/`. |
| 13 | Main UI workflow has an automated test | PASSED | `AppTest` vertical slice test passing in `tests/test_vertical_slice.py`. |
| 14 | No knowingly nonfunctional control | PASSED | All controls in main app and Detect Damage page perform real actions. |
| 15 | README has exact setup, launch, test, train, troubleshoot steps | Pending Phase 8 | Expo package task. |
| 16 | Final report separates passed, unexecuted, and known-limitation items | Pending Phase 8 | `VERIFICATION_REPORT.md` task. |

---

## Phase Execution Progress

- [x] **Phase 1: Inspect, Research, Plan** (Completed and Approved)
- [x] **Phase 2: Foundation** (Completed and Committed)
- [x] **Phase 3: Vertical Slice First**
  - [x] Implemented `app/pages/1_Detect_Damage.py` (Upload -> Validate -> Real Inference -> Annotated Image -> Detection Table -> SQLite -> Show Record).
  - [x] Implemented `tests/test_vertical_slice.py` verifying full pipeline.
  - [x] Full automated test suite passes (`11 passed in 1.92s`).
- [ ] **Phase 4: ML Pipeline**
- [ ] **Phase 5: Features**
- [ ] **Phase 6: UI — Road Intelligence Console**
- [ ] **Phase 7: Verify and Repair**
- [ ] **Phase 8: Final Acceptance & Expo Package**


---

## Phase Execution Progress

- [x] **Phase 1: Inspect, Research, Plan**
  - [x] Environment inspection: macOS Darwin 25.5.0 arm64, Apple M4, 16 GB RAM.
  - [x] Toolchain verified: PyTorch 2.12.1 with Apple MPS, Ultralytics 8.4.177, OpenCV 4.11.0, Streamlit 1.58.0 with AppTest, Folium 0.20.0, Plotly 6.9.0, Pytest 9.1.1, SQLite 3.53.2.
  - [x] Dataset citation & license: RDD2022 under CC BY 4.0 (DOI: 10.6084/m9.figshare.21431547.v1).
  - [x] Empirical XML ground-truth audit: 7,706 India annotations and 2,829 Czech annotations audited via HTTP Range against the official Figshare archive.
  - [x] Class mapping decided: 4 canonical classes (`D00`, `D10`, `D20`, `D40`), typo correction (`D0w0` -> `D00`), non-damage filtering (`D43`, `D44`, `D50`, `Repair`).
  - [x] Leakage-aware split method designed: sequence clustering + perceptual hash clustering with fixed seed.
  - [x] Candidate models benchmarked on target M4 hardware: `yolo26n` (18.02 ms MPS) vs `yolo11n` (17.36 ms MPS) vs `yolov8n` (13.60 ms MPS).
  - [x] In-domain (India) and unseen-country (Czech) domain-shift evaluation design documented.
  - [x] Written `PLAN.md`, `EVIDENCE_LOG.md`, `STATE.md`, `MANUAL_STEPS.md`, `.gitignore`.
  - [ ] **STOP GATE:** Awaiting user reply `"approved"` to begin Phase 2.
- [ ] **Phase 2: Foundation**
- [ ] **Phase 3: Vertical Slice First**
- [ ] **Phase 4: ML Pipeline**
- [ ] **Phase 5: Features**
- [ ] **Phase 6: UI — Road Intelligence Console**
- [ ] **Phase 7: Verify and Repair**
- [ ] **Phase 8: Final Acceptance & Expo Package**
