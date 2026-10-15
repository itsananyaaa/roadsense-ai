# RoadSense AI — System State

**Last Updated:** 2026-10-15T20:42:00+04:00  
**Current Phase:** Phase 1 — Inspect, Research, Plan (**PAUSED AT STOP GATE**)  
**Next Action:** Await user reply `"approved"` to proceed with Phase 2 (Foundation).  
**Open Blockers:** None.

---

## Acceptance Criteria Status Checklist

| # | Acceptance Criterion | Status | Evidence / Notes |
|---|---|---|---|
| 1 | Clean setup succeeds | In Progress | Python 3.10.20 + PyTorch 2.12.1 (MPS) + Ultralytics 8.4.177 verified in miniconda environment. |
| 2 | App launches with no unhandled exception | Pending Phase 2 | Phase 2 task. |
| 3 | Valid image completes real inference with valid weights | Pending Phase 3 | Models verified locally (`yolo26n.pt`, `yolo11n.pt`). |
| 4 | Outputs include class, confidence, bbox | Pending Phase 3 | Schema planned in `PLAN.md`. |
| 5 | Invalid uploads give helpful errors | Pending Phase 5A | Format/dimension validation logic designed. |
| 6 | Video honors documented sampling config | Pending Phase 5B | Video processor designed with explicit sampling rate. |
| 7 | Incident CRUD + filter + export works in SQLite | Pending Phase 5C | Database schema and parameterized queries planned. |
| 8 | Priority is deterministic for same inputs+config | Pending Phase 5E | Mathematical formula v1.0 defined in `PLAN.md`. |
| 9 | Missing location never yields invented coordinates | Pending Phase 5D | Strict provenance rule: EXIF, User, or Explicit Synthetic only. |
| 10 | Missing weights handled transparently | Pending Phase 2/3 | Explicit UI status badge & graceful fallback planned. |
| 11 | Demo mode clearly distinguished from live inference | Pending Phase 5G | "CACHED - not live inference" badge requirement embedded. |
| 12 | Every displayed evaluation figure links to a real artifact | Pending Phase 4 | Model Lab will read strictly from `reports/` JSON/CSV files. |
| 13 | Main UI workflow has an automated test | Pending Phase 7 | Streamlit `AppTest` + `pytest` suite planned. |
| 14 | No knowingly nonfunctional control | Pending Phase 6/7 | All UI buttons mapped to concrete functions. |
| 15 | README has exact setup, launch, test, train, troubleshoot steps | Pending Phase 8 | Expo package task. |
| 16 | Final report separates passed, unexecuted, and known-limitation items | Pending Phase 8 | `VERIFICATION_REPORT.md` template prepared. |

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
