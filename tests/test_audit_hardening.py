"""Audit and regression verification test suite for RoadSense AI.

Validates:
1. Methodology analytics endpoint and landscape A4 PDF export.
2. CSV export formula injection sanitization and UTF-8 BOM encoding.
3. RFC 7946 GeoJSON unlocated feature geometry: null handling.
4. Concurrent DamageDetector inference thread-safety under multi-threading.
5. Authentic GPS provenance resolution (EXIF, USER_ENTERED, EXPLICIT_SYNTHETIC_DEMO, UNKNOWN).
6. Cross-view data consistency between DB, REST API, CSV, and GeoJSON.
"""
import concurrent.futures
import csv
import io
import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

os.environ["ROADSENSE_ADMIN_USER"] = "testadmin"
os.environ["ROADSENSE_ADMIN_PASS"] = "TestPassword2026!"
os.environ["ROADSENSE_SESSION_SECRET"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

from core import config
from core.db import get_incident_by_uuid, insert_incident
from core.inference import DamageDetector
from web.backend import app, create_session_token, init_auth_system

init_auth_system()


@pytest.fixture
def auth_client():
    c = TestClient(app)
    token = create_session_token("testadmin")
    c.cookies.set("roadsense_session", token)
    return c


def test_methodology_analytics_endpoint(auth_client):
    """Verify /api/analytics/methodology returns real artifact-grounded figures."""
    res = auth_client.get("/api/analytics/methodology")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "india_eval" in data
    assert "czech_eval" in data
    assert "benchmark" in data
    assert "checkpoint" in data
    assert "priority_engine" in data
    assert "worked_example" in data

    # Verify numbers match real physical artifacts
    if data["india_eval"]:
        assert data["india_eval"]["mAP50"] == 0.584
        assert "D40" in data["india_eval"]["per_class"]
    if data["czech_eval"]:
        assert data["czech_eval"]["mAP50"] == 0.438
    if data["benchmark"]:
        assert data["benchmark"]["latency_benchmark"]["mps"]["mean_latency_ms"] == 10.93

    # Checkpoint status verified
    assert "label" in data["checkpoint"]
    assert "sha256" in data["checkpoint"]

    # Verify unauthenticated request returns 401
    unauth = TestClient(app)
    unauth_res = unauth.get("/api/analytics/methodology")
    assert unauth_res.status_code == 401


def test_methodology_pdf_export_endpoint(auth_client):
    """Verify /api/export/methodology/pdf delivers a valid, non-empty landscape A4 PDF."""
    res = auth_client.get("/api/export/methodology/pdf")
    assert res.status_code == 200
    assert "application/pdf" in res.headers.get("content-type", "")
    assert res.content.startswith(b"%PDF")
    assert len(res.content) > 30000  # Multi-page detailed report


def test_csv_export_utf8_bom_and_formula_injection(auth_client):
    """Verify CSV export prepends apostrophe to formula injection cells and includes UTF-8 BOM."""
    malicious_road = "=CMD('calc')"
    malicious_notes = "@SUM(A1:A10)"
    inc_uuid = insert_incident(
        image_path="test_malicious.jpg",
        detected_classes=["D40"],
        detection_count=1,
        max_confidence=0.92,
        raw_detections=[{"class_code": "D40", "confidence": 0.92, "bbox": [10, 10, 50, 50]}],
        latitude=12.9716,
        longitude=77.5946,
        location_provenance="USER_ENTERED",
        road_name=malicious_road,
        road_classification="PRIMARY_ARTERIAL",
        priority_score=85.0,
        priority_level="CRITICAL",
        priority_formula_version="1.0",
        checkpoint_id="smoke_test.pt",
    )
    from core.db import update_review_status
    update_review_status(inc_uuid, "VERIFIED", reviewer_notes=malicious_notes)

    res = auth_client.get("/api/export/csv")
    assert res.status_code == 200
    # Must start with UTF-8 BOM
    assert res.content.startswith(b"\xef\xbb\xbf")

    # Parse CSV content
    text_content = res.content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text_content))
    rows = list(reader)
    target_row = next((r for r in rows if r["Incident UUID"] == inc_uuid), None)
    assert target_row is not None
    # Formula injection characters must be sanitized with leading apostrophe
    assert target_row["Road Name"].startswith("'=")
    assert target_row["Reviewer Notes"].startswith("'@")


def test_geojson_export_unlocated_null_geometry(auth_client):
    """Verify unlocated incidents (None GPS) serialize with geometry: null per RFC 7946 Section 3.2."""
    inc_uuid = insert_incident(
        image_path="test_unlocated.jpg",
        detected_classes=["D00"],
        detection_count=1,
        max_confidence=0.75,
        raw_detections=[{"class_code": "D00", "confidence": 0.75, "bbox": [10, 10, 50, 50]}],
        latitude=None,
        longitude=None,
        location_provenance="UNKNOWN",
        road_name="Unlocated Rural Lane",
        road_classification="RESIDENTIAL_LOCAL",
        priority_score=32.0,
        priority_level="LOW",
        priority_formula_version="1.0",
        checkpoint_id="smoke_test.pt",
    )

    res = auth_client.get("/api/export/geojson?include_unlocated=true")
    assert res.status_code == 200
    doc = res.json()
    assert doc.get("type") == "FeatureCollection"

    unlocated_feat = next((f for f in doc["features"] if f["properties"]["incident_uuid"] == inc_uuid), None)
    assert unlocated_feat is not None
    assert unlocated_feat["geometry"] is None
    assert unlocated_feat["properties"]["location_provenance"] == "UNKNOWN"


def test_detector_concurrent_inference_thread_safety():
    """Verify DamageDetector singleton handles 20 concurrent thread predictions safely."""
    demo_img = config.DATA_DIR / "demo" / "demo_pothole_critical.jpg"
    if not demo_img.exists():
        pytest.skip("Demo image not found for concurrency test.")

    detector = DamageDetector.get_instance()

    def run_inference(idx):
        res = detector.predict(demo_img, conf_threshold=0.25)
        return res.detection_count, res.inference_latency_ms

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_inference, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20
    for count, latency in results:
        assert count >= 0
        assert latency >= 0.0


def test_image_upload_gps_provenance_resolution(auth_client):
    """Verify GPS provenance is never invented; UNKNOWN images have None coordinates."""
    demo_img = config.DATA_DIR / "demo" / "demo_pothole_critical.jpg"
    if not demo_img.exists():
        pytest.skip("Demo image not found for GPS test.")

    with open(demo_img, "rb") as f:
        # 1. Upload without coordinates
        res = auth_client.post(
            "/api/detect/image",
            files={"file": ("test_upload_no_gps.jpg", f, "image/jpeg")},
            data={"road_name": "Test Unknown Rd"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["latitude"] is None
    assert data["longitude"] is None
    assert data["location_provenance"] == "UNKNOWN"

    # 2. Demo filename gets EXPLICIT_SYNTHETIC_DEMO
    res_demo = auth_client.post(
        "/api/detect/image",
        data={"demo_filename": "demo_pothole_critical.jpg", "road_name": "Demo Route"}
    )
    assert res_demo.status_code == 200
    demo_data = res_demo.json()
    assert demo_data["location_provenance"] == "EXPLICIT_SYNTHETIC_DEMO"


def test_cross_view_incident_data_consistency(auth_client):
    """Verify incident data fields are identical across DB, /api/incidents, CSV, and GeoJSON."""
    test_uuid = insert_incident(
        image_path="test_consistency.jpg",
        detected_classes=["D20"],
        detection_count=1,
        max_confidence=0.88,
        raw_detections=[{"class_code": "D20", "confidence": 0.88, "bbox": [10, 10, 50, 50]}],
        latitude=12.9800,
        longitude=77.6000,
        location_provenance="USER_ENTERED",
        road_name="Consistency Testway",
        road_classification="PRIMARY_ARTERIAL",
        priority_score=68.5,
        priority_level="HIGH",
        priority_formula_version="1.0",
        checkpoint_id="smoke_test.pt",
    )

    # 1. Database record
    db_rec = get_incident_by_uuid(test_uuid)
    assert db_rec is not None
    assert db_rec["road_name"] == "Consistency Testway"

    # 2. API query
    api_res = auth_client.get("/api/incidents")
    assert api_res.status_code == 200
    api_items = api_res.json()["incidents"]
    api_inc = next((x for x in api_items if x["incident_uuid"] == test_uuid), None)
    assert api_inc is not None
    assert api_inc["priority_score"] == 68.5
    assert api_inc["priority_level"] == "HIGH"
    assert api_inc["detected_classes"] == ["D20"]

    # 3. CSV export
    csv_res = auth_client.get("/api/export/csv")
    csv_text = csv_res.content.decode("utf-8-sig")
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    csv_inc = next((x for x in csv_rows if x["Incident UUID"] == test_uuid), None)
    assert csv_inc is not None
    assert csv_inc["Detected Classes"] == "D20"
    assert float(csv_inc["Priority Score"]) == 68.5
    assert csv_inc["Priority Level"] == "HIGH"

    # 4. GeoJSON export
    geo_res = auth_client.get("/api/export/geojson")
    geo_features = geo_res.json()["features"]
    geo_inc = next((f for f in geo_features if f["properties"]["incident_uuid"] == test_uuid), None)
    assert geo_inc is not None
    assert geo_inc["geometry"]["coordinates"] == [77.6, 12.98]
    assert geo_inc["properties"]["priority_score"] == 68.5
    assert geo_inc["properties"]["priority_level"] == "HIGH"
