"""Unit and integration tests for dashcam video stream route and HTTP 206 Range seeking (Issue C).

Verifies:
- 200 OK for full video requests
- 206 Partial Content for byte-range requests (e.g. Range: bytes=0-1023)
- Accurate Content-Range, Accept-Ranges, and Content-Length headers
- Correct MIME type: video/mp4
- 404 for missing video files
- Path traversal protection (e.g. ../../etc/passwd)
"""

import os
import sys
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

from web.backend import app, init_auth_system

init_auth_system()


@pytest.fixture
def client():
    return TestClient(app)


def test_video_stream_full_request(client):
    """Test full GET video request returns 200 with Accept-Ranges and video/mp4."""
    res = client.get("/api/video/stream?file=dashcam_inspection_demo.mp4")
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"
    assert res.headers.get("accept-ranges") == "bytes"
    assert int(res.headers.get("content-length", 0)) > 10000


def test_video_stream_range_request_partial_content(client):
    """Test byte range request returns 206 Partial Content with Content-Range."""
    headers = {"Range": "bytes=0-1023"}
    res = client.get("/api/video/stream?file=dashcam_inspection_demo.mp4", headers=headers)
    assert res.status_code == 206
    assert res.headers["content-type"] == "video/mp4"
    assert "content-range" in res.headers
    assert res.headers["content-range"].startswith("bytes 0-1023/")
    assert len(res.content) == 1024


def test_video_stream_suffix_range_request(client):
    """Test seeking to end of video (Range: bytes=5000-) returns 206 Partial Content."""
    headers = {"Range": "bytes=1000-"}
    res = client.get("/api/video/stream?file=dashcam_inspection_demo.mp4", headers=headers)
    assert res.status_code == 206
    assert res.headers["content-type"] == "video/mp4"
    assert "content-range" in res.headers
    assert res.headers["content-range"].startswith("bytes 1000-")


def test_video_stream_missing_file_404(client):
    """Test requesting a nonexistent video file returns 404."""
    res = client.get("/api/video/stream?file=nonexistent_video_file.mp4")
    assert res.status_code == 404


def test_video_stream_path_traversal_protection(client):
    """Test path traversal attempts are blocked."""
    res = client.get("/api/video/stream?file=../../../../etc/passwd")
    assert res.status_code in (400, 403, 404)
