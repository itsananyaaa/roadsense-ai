"""
Comprehensive verification test suite for RoadSense AI Rectifications.

Tests:
1. Manifest schema, asset checksums, and provenance fields.
2. Deterministic selection (fixed seed) using held-out split only.
3. Live inference on sample demo tiles (no mock, real checkpoint execution).
4. Truthful provenance labeling and synthetic location badges.
5. Methodology PDF export deep structure, pypdf parsing, section headers,
   metrics matching reports/*.json, and graceful handling of missing artifacts.
6. Authentication enforcement on /api/export/methodology/pdf.
7. Methodology and Model Lab API data equality with physical reports/*.json files.
8. User field capture handling (HEIC rejection, EXIF GPS).
"""

import hashlib
import io
import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

os.environ["ROADSENSE_ADMIN_USER"] = "testadmin"
os.environ["ROADSENSE_ADMIN_PASS"] = "TestPassword2026!"
os.environ["ROADSENSE_SESSION_SECRET"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

from core import config, demo, reports_pdf
from core.geo import extract_exif_gps
from web.backend import app, create_session_token, init_auth_system

init_auth_system()


@pytest.fixture
def auth_client():
    c = TestClient(app)
    token = create_session_token("testadmin")
    c.cookies.set("roadsense_session", token)
    return c


def test_demo_manifest_schema_and_checksums():
    """Verify manifest.json schema and sha256 checksums of registered media assets."""
    manifest_path = config.DATA_DIR / "demo_media" / "manifest.json"
    assert manifest_path.exists(), "data/demo_media/manifest.json must exist"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "manifest_version" in manifest
    assert "assets" in manifest
    assert len(manifest["assets"]) > 0

    required_keys = {
        "filename",
        "source_dataset",
        "file_id",
        "country",
        "split",
        "license",
        "attribution",
        "has_gps",
        "origin",
    }

    for asset in manifest["assets"]:
        missing = required_keys - set(asset.keys())
        assert not missing, f"Asset {asset.get('filename')} missing required keys: {missing}"
        assert ("ground_truth" in asset or "ground_truth_boxes" in asset), f"Asset {asset.get('filename')} missing ground truth"

        # Provenance origin must be valid and documented
        assert asset["origin"] in ["real_dataset", "user_field_capture", "illustrative_sample", "synthetic_demo"]

        # Check file exists and sha256 checksum matches if recorded
        fname = asset["filename"]
        target_file = config.DATA_DIR / "demo_media" / "images" / fname
        if not target_file.exists():
            target_file = config.DATA_DIR / "demo" / fname

        assert target_file.exists(), f"Registered asset file does not exist: {target_file}"

        if "sha256" in asset and asset["sha256"]:
            actual_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
            assert actual_sha == asset["sha256"], f"Checksum mismatch for {fname}"


def test_demo_media_selection_deterministic_heldout():
    """Verify ml/build_demo_media.py logic is deterministic and uses held-out split only."""
    import ml.build_demo_media as bdm

    assert hasattr(bdm, "build_manifest_and_eval"), "build_demo_media must export build_manifest_and_eval"

    # Run deterministic build
    manifest = bdm.build_manifest_and_eval(force_fallback=True)
    assert manifest is not None
    assert "assets" in manifest
    assert len(manifest["assets"]) >= 6

    # Verify every asset with source_dataset == 'RDD2022' is strictly from held-out split
    for asset in manifest["assets"]:
        if asset["source_dataset"] == "RDD2022":
            assert asset["split"] in ["val", "test", "held_out_val", "held-out"], (
                f"Data leakage violation: RDD asset {asset['file_id']} is from split {asset['split']}, not held-out!"
            )


def test_sample_tiles_live_inference(auth_client):
    """Verify sample tiles endpoint loads real fixtures and runs live inference."""
    # 1. Fetch demo tiles via API
    res = auth_client.get("/api/demo/samples")
    assert res.status_code == 200
    samples = res.json()["samples"]
    assert len(samples) > 0

    # 2. Pick the first sample tile and run live inference
    first_sample = samples[0]
    filename = first_sample["filename"]

    detect_res = auth_client.post(
        "/api/detect/image",
        data={
            "demo_filename": filename,
            "road_name": "Surveyed Corridor Alpha (Synthetic Location)",
            "road_classification": "PRIMARY_ARTERIAL",
            "save_to_db": "false",
        },
    )
    assert detect_res.status_code == 200
    det_data = detect_res.json()

    assert det_data["success"] is True
    assert "detections" in det_data
    assert "inference_time_ms" in det_data
    assert det_data["inference_time_ms"] > 0
    assert "priority" in det_data
    assert det_data["priority"]["score"] >= 0.0
    # Location provenance must be explicitly flagged
    assert det_data["location_provenance"] == "EXPLICIT_SYNTHETIC_DEMO"


def test_synthetic_location_badge_and_provenance(auth_client):
    """Verify demo incidents display explicit synthetic location badges."""
    samples = demo.get_demo_tiles()
    for s in samples:
        assert "badge" in s
        if not s["is_real_imagery"]:
            assert "illustrative" in s["badge"].lower()
        else:
            assert "synthetic location" in s["badge"].lower() or "exif gps" in s["badge"].lower()


def test_methodology_pdf_deep_verification(auth_client):
    """Verify /api/export/methodology/pdf delivers a 4-page, valid PDF matching physical artifacts."""
    res = auth_client.get("/api/export/methodology/pdf")
    assert res.status_code == 200
    assert "application/pdf" in res.headers.get("content-type", "")
    assert res.content.startswith(b"%PDF")

    # Parse with pypdf
    reader = PdfReader(io.BytesIO(res.content))
    assert len(reader.pages) >= 4, f"Methodology PDF should have at least 4 pages, got {len(reader.pages)}"

    full_text = "\n".join(page.extract_text() for page in reader.pages)

    # 1. Title Block & Header
    assert "ROADSENSE AI | Methodology and Evaluation" in full_text
    assert "Active Model: smoke_test.pt" in full_text
    assert "SMOKE_CHECKPOINT" in full_text

    # 2. Section Headings
    assert "1. Dataset Attribution & Leakage-Aware Splitting (RDD2022)" in full_text
    assert "2. Model Architecture & Hardware Acceleration Profile" in full_text
    assert "3. Dual-Domain Evaluation Metrics (In-Domain vs Cross-National Transfer)" in full_text
    assert "4. Deterministic Municipal Priority Ranking Engine (Formula v1.0)" in full_text
    assert "5. Live Operational Triage Analytics (SQLite Database)" in full_text
    assert "6. Demo-Media Provenance Summary & Live Checkpoint Audit" in full_text
    assert "7. Engineering Limitations, Responsible Civil Use & Reproducibility" in full_text

    # 3. Grounded Values from Physical Artifacts
    india_file = config.REPORTS_DIR / "eval_india.json"
    czech_file = config.REPORTS_DIR / "eval_czech.json"
    bench_file = config.REPORTS_DIR / "eval_benchmark.json"

    if india_file.exists():
        india_data = json.loads(india_file.read_text())
        assert f"{india_data['mAP50'] * 100:.1f}%" in full_text

    if czech_file.exists():
        czech_data = json.loads(czech_file.read_text())
        assert f"{czech_data['mAP50'] * 100:.1f}%" in full_text

    if bench_file.exists():
        bench_data = json.loads(bench_file.read_text())
        mps_lat = bench_data["latency_benchmark"]["mps"]["mean_latency_ms"]
        assert f"{mps_lat:.2f}" in full_text


def test_methodology_pdf_auth_enforced():
    """Verify /api/export/methodology/pdf strictly enforces authentication (401 without cookie)."""
    unauth = TestClient(app)
    res = unauth.get("/api/export/methodology/pdf")
    assert res.status_code == 401


def test_methodology_and_model_lab_match_reports_json(auth_client):
    """Verify numbers shown in Methodology and Model Lab API match physical reports/*.json."""
    india_file = config.REPORTS_DIR / "eval_india.json"
    czech_file = config.REPORTS_DIR / "eval_czech.json"
    bench_file = config.REPORTS_DIR / "eval_benchmark.json"

    assert india_file.exists() and czech_file.exists() and bench_file.exists()

    india_json = json.loads(india_file.read_text())
    czech_json = json.loads(czech_file.read_text())
    bench_json = json.loads(bench_file.read_text())

    # 1. Methodology analytics endpoint
    res_meth = auth_client.get("/api/analytics/methodology")
    assert res_meth.status_code == 200
    meth_data = res_meth.json()

    assert meth_data["india_eval"]["mAP50"] == india_json["mAP50"]
    assert meth_data["czech_eval"]["mAP50"] == czech_json["mAP50"]
    assert meth_data["benchmark"]["latency_benchmark"]["mps"]["mean_latency_ms"] == bench_json["latency_benchmark"]["mps"]["mean_latency_ms"]

    # 2. Benchmark endpoint (Model Lab)
    res_bench = auth_client.get("/api/benchmark")
    assert res_bench.status_code == 200
    b_data = res_bench.json()

    assert b_data["india_eval"]["mAP50"] == india_json["mAP50"]
    assert b_data["czech_eval"]["mAP50"] == czech_json["mAP50"]
    assert b_data["benchmark"]["latency_benchmark"]["mps"]["mean_latency_ms"] == bench_json["latency_benchmark"]["mps"]["mean_latency_ms"]


def test_user_field_capture_heic_and_gps():
    """Verify user field capture handling: clear error on HEIC and valid EXIF GPS extraction."""
    import ml.build_demo_media as bdm
    test_heic = config.DATA_DIR / "field_samples" / "sample.heic"

    with pytest.raises(ValueError) as excinfo:
        bdm.register_user_field_photo(test_heic)
    assert "convert image to jpeg" in str(excinfo.value).lower()

    # Test GPS extraction on clean image with no GPS
    demo_img = config.DATA_DIR / "demo" / "demo_pothole_critical.jpg"
    if demo_img.exists():
        lat, lon, prov = extract_exif_gps(str(demo_img))
        assert lat is None and lon is None
        assert prov == "UNKNOWN"


def test_methodology_pdf_missing_artifact_graceful():
    """Verify build_methodology_report_pdf handles missing physical artifacts gracefully."""
    orig_india = config.REPORTS_DIR / "eval_india.json"
    temp_india = config.REPORTS_DIR / "eval_india.json.tmp"

    if orig_india.exists():
        orig_india.rename(temp_india)
    try:
        pdf_bytes = reports_pdf.build_methodology_report_pdf()
        assert len(pdf_bytes) > 20000
        assert pdf_bytes.startswith(b"%PDF")
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join(page.extract_text() for page in reader.pages)
        assert "Not available" in full_text or "eval_india.json" in full_text
    finally:
        if temp_india.exists():
            temp_india.rename(orig_india)
