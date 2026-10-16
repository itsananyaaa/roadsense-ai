"""Unit tests for RoadSense AI Next-Gen Web API (FastAPI).

Tests verification of all endpoints powering the modern obsidian web console.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import os
from fastapi.testclient import TestClient

os.environ["ROADSENSE_ADMIN_USER"] = "testadmin"
os.environ["ROADSENSE_ADMIN_PASS"] = "TestPassword2026!"
os.environ["ROADSENSE_SESSION_SECRET"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

from web.backend import app, create_session_token, init_auth_system
init_auth_system()


@pytest.fixture
def client():
    """Create authenticated test client for RoadSense web API."""
    c = TestClient(app)
    token = create_session_token("testadmin")
    c.cookies.set("roadsense_session", token)
    return c


def test_api_health(client):
    """Test health check and system status endpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["is_operational"] is True
    assert "database_status" in data
    assert data["database_status"] in ("READY", "CREATED_EMPTY")
    assert "accelerator" in data
    assert "active_checkpoint_path" in data


def test_api_overview(client):
    """Test overview dashboard data endpoint matching Resq.io mockup."""
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert "critical_issues" in data["kpis"]
    assert "routes_surveyed" in data["kpis"]
    assert "total_defects" in data["kpis"]
    assert "progress_pills" in data
    assert "summary" in data
    assert "trendline" in data
    assert "hazard_highlight" in data
    assert "active_incidents" in data
    assert "statistics" in data


def test_api_incidents(client):
    """Test incident listing and review workflow."""
    res = client.get("/api/incidents")
    assert res.status_code == 200
    data = res.json()
    assert "incidents" in data
    incidents = data["incidents"]
    assert isinstance(incidents, list)

    if incidents:
        test_uuid = incidents[0]["incident_uuid"]
        # Test review verification using PATCH
        rev_res = client.patch(
            f"/api/incidents/{test_uuid}/review",
            json={"review_status": "VERIFIED", "reviewer_notes": "Automated verification test."}
        )
        assert rev_res.status_code == 200
        rev_data = rev_res.json()
        assert rev_data["success"] is True
        assert rev_data["incident"]["review_status"] == "VERIFIED"


def test_api_detect_image(client):
    """Test image detection endpoint with sample demo asset."""
    demo_img = Path("data/demo/demo_pothole_critical.jpg")
    if not demo_img.exists():
        pytest.skip(f"Demo image asset not found at {demo_img}.")

    with open(demo_img, "rb") as f:
        res = client.post(
            "/api/detect/image",
            files={"file": ("demo_pothole_critical.jpg", f, "image/jpeg")},
            data={"road_name": "Test Highway", "speed_kmh": "60", "traffic_level": "HIGH"}
        )
    assert res.status_code == 200
    data = res.json()
    assert "incident_uuid" in data
    assert "detections" in data
    assert "priority" in data
    assert "score" in data["priority"]
    assert "inference_time_ms" in data
    assert "annotated_image_url" in data


def test_api_copilot(client):
    """Test AI assistant natural language query engine."""
    queries = [
        "What are the critical potholes needing repair?",
        "How does priority score work?",
        "What hardware is running inference?",
        "Export recent reports"
    ]
    for q in queries:
        res = client.post("/api/copilot", json={"query": q})
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert len(data["answer"]) > 10


def test_api_exports(client):
    """Test CSV and RFC 7946 GeoJSON export endpoints."""
    csv_res = client.get("/api/export/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers.get("content-type", "")

    geojson_res = client.get("/api/export/geojson")
    assert geojson_res.status_code == 200
    geo_data = geojson_res.json()
    assert geo_data.get("type") == "FeatureCollection"
    assert "features" in geo_data


def test_api_benchmark(client):
    """Test benchmark results endpoint for Model Lab."""
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark" in data
    assert "india_eval" in data
    assert "czech_eval" in data
    assert "confusion_matrix_url" in data
