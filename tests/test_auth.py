"""Unit tests for RoadSense AI Authentication, Session Tokens, and Route Protection."""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from web.backend import (
    FAILED_LOGIN_ATTEMPTS,
    app,
    clear_failed_attempts,
    create_session_token,
    init_auth_system,
)


@pytest.fixture(autouse=True)
def setup_test_auth_env():
    """Configure predictable environment variables for auth testing."""
    os.environ["ROADSENSE_ADMIN_USER"] = "testadmin"
    os.environ["ROADSENSE_ADMIN_PASS"] = "SecureMunicipal2026!"
    os.environ["ROADSENSE_SESSION_SECRET"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    init_auth_system()
    FAILED_LOGIN_ATTEMPTS.clear()
    yield
    FAILED_LOGIN_ATTEMPTS.clear()


@pytest.fixture
def raw_client():
    """Unauthenticated client."""
    return TestClient(app)


@pytest.fixture
def auth_client():
    """Authenticated client with pre-set session cookie."""
    client = TestClient(app)
    token = create_session_token("testadmin")
    client.cookies.set("roadsense_session", token)
    return client


def test_login_success(raw_client):
    """Test successful authentication sets cookie and returns username."""
    res = raw_client.post(
        "/api/auth/login",
        json={"username": "testadmin", "password": "SecureMunicipal2026!"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["authenticated"] is True
    assert data["username"] == "testadmin"
    assert "roadsense_session" in res.cookies


def test_login_wrong_password(raw_client):
    """Wrong password returns 401 with generic error message."""
    res = raw_client.post(
        "/api/auth/login",
        json={"username": "testadmin", "password": "WrongPassword999!"},
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Incorrect username or password"


def test_login_wrong_username(raw_client):
    """Wrong username returns identical 401 generic error message."""
    res = raw_client.post(
        "/api/auth/login",
        json={"username": "impostor", "password": "SecureMunicipal2026!"},
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Incorrect username or password"


def test_login_fails_closed_when_env_missing(raw_client, monkeypatch):
    """When credentials env vars are absent, all logins fail closed."""
    monkeypatch.delenv("ROADSENSE_ADMIN_USER", raising=False)
    monkeypatch.delenv("ROADSENSE_ADMIN_PASS", raising=False)
    init_auth_system()

    res = raw_client.post(
        "/api/auth/login",
        json={"username": "testadmin", "password": "SecureMunicipal2026!"},
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Incorrect username or password"


def test_auth_me_and_logout_flow(raw_client):
    """Test session inspection, logout, and post-logout access denial."""
    # 1. Unauthenticated /api/auth/me returns 401
    res = raw_client.get("/api/auth/me")
    assert res.status_code == 401

    # 2. Login
    login_res = raw_client.post(
        "/api/auth/login",
        json={"username": "testadmin", "password": "SecureMunicipal2026!"},
    )
    assert login_res.status_code == 200

    # 3. Authenticated /api/auth/me returns 200 with username
    me_res = raw_client.get("/api/auth/me")
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "testadmin"

    # 4. Logout clears session cookie
    logout_res = raw_client.post("/api/auth/logout")
    assert logout_res.status_code == 200
    assert logout_res.json()["authenticated"] is False

    # 5. Subsequent /api/auth/me returns 401
    me_after = raw_client.get("/api/auth/me")
    assert me_after.status_code == 401


def test_protected_route_without_session_returns_401(raw_client):
    """Protected routes reject unauthenticated requests."""
    res = raw_client.get("/api/overview")
    assert res.status_code == 401
    assert res.json()["detail"] == "Authentication required"


def test_protected_route_with_session_returns_200(auth_client):
    """Protected routes allow access when a valid session token cookie is supplied."""
    res = auth_client.get("/api/overview")
    assert res.status_code == 200
    assert "kpis" in res.json()


def test_rate_limiting_triggers(raw_client):
    """Rate limiter allows 5 failed attempts, then triggers 429 on the 6th attempt."""
    # 5 failed attempts
    for _ in range(5):
        res = raw_client.post(
            "/api/auth/login",
            json={"username": "testadmin", "password": "wrong"},
        )
        assert res.status_code == 401

    # 6th attempt triggers 429 Too Many Requests
    res_blocked = raw_client.post(
        "/api/auth/login",
        json={"username": "testadmin", "password": "wrong"},
    )
    assert res_blocked.status_code == 429
    assert "Too many failed login attempts" in res_blocked.json()["detail"]
