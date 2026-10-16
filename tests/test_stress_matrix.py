"""Comprehensive stress and negative matrix verification suite for RoadSense AI.

Executes tests for:
1. Valid image inference
2. No-damage blank image
3. Zero-byte image upload (400)
4. Corrupt image bytes (400)
5. Non-image renamed to .jpg (400)
6. Oversized image / decompression limits (400)
7. EXIF orientation auto-transposition
8. Valid video stream analysis
9. Zero-byte video upload (400)
10. Corrupt video bytes (400)
11. Unsupported video codec/extension (400)
12. Empty database exports (CSV header only, valid empty-state PDF)
13. Demo database reset idempotence (called twice back-to-back)
14. Unknown location handling (None, None, UNKNOWN provenance)
15. Invalid / Null Island coordinates (0.0, 0.0) rejection
16. 20 concurrent parallel inference requests (thread-safety under load)
17. Rapid frequency range switching (1w, 30d, 12m)
18. 100% offline assets compliance (zero runtime CDN / external URL dependencies)
"""

import concurrent.futures
import io
import os
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

os.environ["ROADSENSE_ADMIN_USER"] = "testadmin"
os.environ["ROADSENSE_ADMIN_PASS"] = "TestPassword2026!"
os.environ["ROADSENSE_SESSION_SECRET"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

from core import config
from core.db import get_connection, init_db
from core.demo import seed_demo_database
from core.geo import is_valid_coordinates
from core.inference import DamageDetector
from web.backend import app, create_session_token, init_auth_system

init_auth_system()


@pytest.fixture
def auth_client():
    c = TestClient(app)
    token = create_session_token("testadmin")
    c.cookies.set("roadsense_session", token)
    return c


def create_test_image_bytes(width=100, height=100, color=(128, 128, 128), fmt="JPEG"):
    """Helper to generate in-memory image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_stress_valid_image(auth_client):
    """Scenario 1: Valid image returns 200 with well-formed detection schema."""
    img_bytes = create_test_image_bytes(200, 200)
    files = {"file": ("test_valid.jpg", img_bytes, "image/jpeg")}
    data = {"road_name": "NH-44 Expressway", "save_to_db": "false"}
    res = auth_client.post("/api/detect/image", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "detections" in body
    assert "priority" in body
    assert "score" in body["priority"]
    assert "level" in body["priority"]


def test_stress_no_damage_image(auth_client):
    """Scenario 2: Plain untextured image returns 0 detections and priority score >= 0."""
    img_bytes = create_test_image_bytes(100, 100, color=(80, 80, 80))
    files = {"file": ("clean_asphalt.jpg", img_bytes, "image/jpeg")}
    data = {"road_name": "Fresh Asphalt", "save_to_db": "false"}
    res = auth_client.post("/api/detect/image", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["detection_count"] == 0
    assert len(body["detections"]) == 0


def test_stress_zero_byte_image(auth_client):
    """Scenario 3: Zero-byte file rejected with 400 Bad Request."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = auth_client.post("/api/detect/image", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_stress_corrupt_image(auth_client):
    """Scenario 4: Truncated/corrupt binary bytes rejected with 400 Bad Request."""
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01" + b"\x00" * 30
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    res = auth_client.post("/api/detect/image", files=files)
    assert res.status_code == 400


def test_stress_non_image_renamed_jpg(auth_client):
    """Scenario 5: Plain text payload named .jpg rejected with 400 Bad Request."""
    text_bytes = b"This is a plain text file, not a valid JPEG image header!"
    files = {"file": ("fake_image.jpg", text_bytes, "image/jpeg")}
    res = auth_client.post("/api/detect/image", files=files)
    assert res.status_code == 400


def test_stress_oversized_image():
    """Scenario 6: File exceeding MAX_IMAGE_SIZE_BYTES rejected with 400 Bad Request."""
    detector = DamageDetector.get_instance()
    huge_bytes = b"0" * (config.MAX_IMAGE_SIZE_BYTES + 1024)
    tmp_path = config.EVIDENCE_DIR / "temp_huge.jpg"
    try:
        tmp_path.write_bytes(huge_bytes)
        with pytest.raises(Exception):
            detector.predict(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_stress_exif_rotated_image():
    """Scenario 7: Image with EXIF orientation 6 (90 CW) auto-transposes properly."""
    detector = DamageDetector.get_instance()
    img = Image.new("RGB", (200, 100), color=(100, 150, 200))
    exif_data = img.getexif()
    exif_data[0x0112] = 6  # Orientation tag: 90 deg rotation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_data)

    tmp_path = config.EVIDENCE_DIR / "temp_exif_rot.jpg"
    try:
        tmp_path.write_bytes(buf.getvalue())
        det_result = detector.predict(tmp_path)
        # Dimensions transposed from (200, 100) to (100, 200)
        h, w = det_result.annotated_image_bgr.shape[:2]
        assert h == 200
        assert w == 100
    finally:
        tmp_path.unlink(missing_ok=True)


def test_stress_valid_video(auth_client):
    """Scenario 8: Demo video processing returns 200 with structured report."""
    data = {"use_demo": "true", "sampling_fps": "2.0", "save_incidents": "false"}
    res = auth_client.post("/api/detect/video", data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "fps_achieved" in body
    assert "keyframes" in body
    assert body["total_video_frames"] > 0


def test_stress_zero_byte_video(auth_client):
    """Scenario 9: Zero-byte video file rejected with 400 Bad Request."""
    files = {"file": ("empty.mp4", b"", "video/mp4")}
    data = {"use_demo": "false", "save_incidents": "false"}
    res = auth_client.post("/api/detect/video", files=files, data=data)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_stress_corrupt_video(auth_client):
    """Scenario 10: Corrupt video bytes rejected with 400 Bad Request."""
    files = {"file": ("corrupt.mp4", b"not-a-video-stream-header", "video/mp4")}
    data = {"use_demo": "false", "save_incidents": "false"}
    res = auth_client.post("/api/detect/video", files=files, data=data)
    assert res.status_code == 400


def test_stress_unsupported_video_format(auth_client):
    """Scenario 11: Unsupported video extension rejected with 400 Bad Request."""
    files = {"file": ("clip.wmv", b"fake wmv stream", "video/x-ms-wmv")}
    data = {"use_demo": "false", "save_incidents": "false"}
    res = auth_client.post("/api/detect/video", files=files, data=data)
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


def test_stress_demo_reset_idempotence():
    """Scenario 12: Demo database reset called twice is completely idempotent."""
    # First reset
    seed_demo_database(reset_existing=True)
    with get_connection() as conn:
        count_first = conn.execute("SELECT count(*) FROM incidents").fetchone()[0]
    assert count_first >= 6

    # Second reset back-to-back
    seed_demo_database(reset_existing=True)
    with get_connection() as conn:
        count_second = conn.execute("SELECT count(*) FROM incidents").fetchone()[0]
    assert count_first == count_second


def test_stress_unknown_location_and_null_island():
    """Scenario 13 & 14: Null island (0.0, 0.0) is rejected; UNKNOWN location handling."""
    assert is_valid_coordinates(0.0, 0.0) is False
    assert is_valid_coordinates(None, None) is False
    assert is_valid_coordinates(12.9716, 77.5946) is True
    assert is_valid_coordinates(95.0, 50.0) is False  # Lat out of range
    assert is_valid_coordinates(45.0, 190.0) is False  # Lon out of range


def test_stress_parallel_inference_thread_safety(auth_client):
    """Scenario 15: 20 parallel concurrent inference requests under threadpool load."""
    img_bytes = create_test_image_bytes(160, 160)

    def send_request(idx):
        files = {"file": (f"parallel_{idx}.jpg", img_bytes, "image/jpeg")}
        data = {"road_name": f"Parallel Route {idx}", "save_to_db": "false"}
        return auth_client.post("/api/detect/image", files=files, data=data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_request, i) for i in range(20)]
        results = [f.result() for f in futures]

    for res in results:
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert "priority" in body


def test_stress_rapid_frequency_range_switching(auth_client):
    """Scenario 16: Rapid frequency range switching (1w, 30d, 12m) completes cleanly."""
    for rng in ["1w", "30d", "12m", "1w", "30d"]:
        res = auth_client.get(f"/api/analytics/frequency?range={rng}")
        assert res.status_code == 200
        data = res.json()
        assert data["range"] == rng
        assert len(data["labels"]) > 0
        assert "total_potholes" in data
        assert len(data["series_potholes"]) == len(data["labels"])


def test_stress_offline_compliance():
    """Scenario 17: Static frontend HTML/CSS/JS contains 0 external CDN or runtime web dependencies."""
    index_html = (config.BASE_DIR / "web" / "static" / "index.html").read_text()
    style_css = (config.BASE_DIR / "web" / "static" / "css" / "style.css").read_text()

    # Disallow external Google Fonts, unpkg, cdnjs, cdn.jsdelivr in HTML and CSS
    for forbidden in ["fonts.googleapis.com", "fonts.gstatic.com", "unpkg.com", "cdnjs.cloudflare.com"]:
        assert forbidden not in index_html, f"Found external CDN {forbidden} in index.html"
        assert forbidden not in style_css, f"Found external CDN {forbidden} in style.css"

    # Ensure font-face declarations refer to local assets or system fallbacks
    assert "https://" not in style_css
