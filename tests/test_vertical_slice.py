"""
RoadSense AI - Automated Test for Phase 3 Vertical Slice
Validates the complete end-to-end pipeline:
Image -> Validation -> Real Inference -> Bounding Boxes -> Priority Calculation ->
SQLite Persistence -> Record Retrieval.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import config
from core.db import init_db, insert_incident, get_incident_by_uuid
from core.demo import seed_demo_database
from core.inference import DamageDetector, DetectionResult
from core.priority import calculate_priority


@pytest.fixture(scope="module")
def setup_test_environment(tmp_path_factory):
    """Ensure test demo images and database are ready."""
    init_db()
    seed_demo_database(reset_existing=False)


def test_vertical_slice_end_to_end(setup_test_environment):
    """
    Test complete programmatic workflow from real inference to SQLite storage.
    """
    # 1. Image acquisition: Load sample fixture
    demo_image_path = config.DEMO_DIR / "demo_pothole_critical.jpg"
    assert demo_image_path.exists(), "Demo fixture should exist"

    # 2. Run real inference
    detector = DamageDetector.get_instance()
    result: DetectionResult = detector.predict(demo_image_path, conf_threshold=0.20)

    # 3. Verify detection structure
    assert isinstance(result.annotated_image_bgr, np.ndarray)
    assert result.annotated_image_bgr.shape[2] == 3
    assert result.inference_latency_ms > 0.0
    assert result.checkpoint_status in ["EVALUATED_TRAINED", "SMOKE_CHECKPOINT", "PRETRAINED_BASELINE"]

    # 4. Calculate deterministic priority
    prio = calculate_priority(
        detected_classes=result.detected_classes,
        max_confidence=result.max_confidence,
        nearby_repeat_count=1,
        road_classification="PRIMARY_ARTERIAL",
    )
    assert 0.0 <= prio.score <= 100.0
    assert prio.level in ["CRITICAL", "HIGH", "MODERATE", "LOW", "INSUFFICIENT_INFO"]

    # 5. Commit to SQLite
    evidence_path = config.EVIDENCE_DIR / "test_vertical_slice_evidence.jpg"
    cv2.imwrite(str(evidence_path), result.annotated_image_bgr)

    raw_dets = [d.to_dict() for d in result.detections]
    inc_uuid = insert_incident(
        image_path=str(evidence_path),
        checkpoint_id=result.checkpoint_id,
        detected_classes=result.detected_classes,
        detection_count=result.detection_count,
        max_confidence=result.max_confidence,
        raw_detections=raw_dets,
        priority_score=prio.score,
        priority_level=prio.level,
        latitude=12.9716,
        longitude=77.5946,
        location_provenance="USER_ENTERED",
        road_name="MG Road",
        road_classification="PRIMARY_ARTERIAL",
    )

    assert inc_uuid.startswith("INC-")

    # 6. Retrieve and verify persisted record
    record = get_incident_by_uuid(inc_uuid)
    assert record is not None
    assert record["incident_uuid"] == inc_uuid
    assert record["priority_level"] == prio.level
    assert record["priority_score"] == pytest.approx(prio.score, 0.01)
    assert record["location_provenance"] == "USER_ENTERED"
    assert record["latitude"] == pytest.approx(12.9716, 0.0001)
    assert record["longitude"] == pytest.approx(77.5946, 0.0001)
    assert record["review_status"] == "PENDING"


def test_detect_damage_page_renders_cleanly():
    """Verify app/pages/1_Detect_Damage.py boots without exception using Streamlit AppTest."""
    at = AppTest.from_file("app/pages/1_Detect_Damage.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Detect Damage page raised an exception: {at.exception}"
    assert len(at.title) >= 1
    assert "Detect Road Damage" in at.title[0].value
