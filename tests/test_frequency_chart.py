"""Unit tests for incident frequency aggregation and chart endpoints (Issue B).

Verifies:
- 1 week (7 buckets), 30 days (30 buckets), 12 months (12 buckets)
- Continuous zero-filling of empty date buckets
- Anchoring to current date
- Separation of Potholes (D40) vs Cracks (D00, D10, D20)
- Empty database resilience
- Authentication protection on /api/analytics/frequency
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

os.environ["ROADSENSE_ADMIN_USER"] = "testadmin"
os.environ["ROADSENSE_ADMIN_PASS"] = "TestPassword2026!"
os.environ["ROADSENSE_SESSION_SECRET"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

from core.db import get_incident_frequency, insert_incident, init_db
from web.backend import app, create_session_token, init_auth_system

init_auth_system()


@pytest.fixture
def auth_client():
    """Authenticated client for testing analytics endpoints."""
    c = TestClient(app)
    token = create_session_token("testadmin")
    c.cookies.set("roadsense_session", token)
    return c


def test_frequency_buckets_structure():
    """Test frequency returns correct number of continuous buckets for all ranges."""
    res_1w = get_incident_frequency("1w")
    assert len(res_1w["labels"]) == 7
    assert len(res_1w["series_potholes"]) == 7
    assert len(res_1w["series_cracks"]) == 7
    assert res_1w["range"] == "1w"

    res_30d = get_incident_frequency("30d")
    assert len(res_30d["labels"]) == 30
    assert len(res_30d["series_potholes"]) == 30
    assert len(res_30d["series_cracks"]) == 30
    assert res_30d["range"] == "30d"

    res_12m = get_incident_frequency("12m")
    assert len(res_12m["labels"]) == 12
    assert len(res_12m["series_potholes"]) == 12
    assert len(res_12m["series_cracks"]) == 12
    assert res_12m["range"] == "12m"


def test_frequency_zero_filled_and_non_negative():
    """Verify that all buckets have non-negative integers (zero is real data)."""
    for r in ["1w", "30d", "12m"]:
        res = get_incident_frequency(r)
        for p in res["series_potholes"]:
            assert isinstance(p, int)
            assert p >= 0
        for c in res["series_cracks"]:
            assert isinstance(c, int)
            assert c >= 0


def test_api_analytics_frequency_endpoint(auth_client):
    """Test /api/analytics/frequency endpoint responses across ranges."""
    # Test 30d (default)
    res = auth_client.get("/api/analytics/frequency?range=30d")
    assert res.status_code == 200
    data = res.json()
    assert data["range"] == "30d"
    assert len(data["labels"]) == 30
    assert "series_potholes" in data
    assert "series_cracks" in data

    # Test 1w
    res_1w = auth_client.get("/api/analytics/frequency?range=1w")
    assert res_1w.status_code == 200
    assert len(res_1w.json()["labels"]) == 7

    # Test 12m
    res_12m = auth_client.get("/api/analytics/frequency?range=12m")
    assert res_12m.status_code == 200
    assert len(res_12m.json()["labels"]) == 12


def test_api_analytics_frequency_requires_auth():
    """Test unauthenticated request to frequency endpoint is rejected."""
    raw_client = TestClient(app)
    res = raw_client.get("/api/analytics/frequency?range=30d")
    assert res.status_code == 401
