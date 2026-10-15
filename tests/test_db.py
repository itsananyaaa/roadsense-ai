"""
RoadSense AI - Unit Tests for Database Persistence Layer
Tests parameterized queries, incident insertion, filtering, and review status updates.
"""

from pathlib import Path
import pytest
from core.db import (
    init_db,
    insert_incident,
    get_incident_by_uuid,
    get_all_incidents,
    update_review_status,
    count_nearby_incidents,
    get_summary_statistics,
    clear_all_incidents,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provides a fresh isolated database file per test."""
    db_file = tmp_path / "test_roadsense.db"
    init_db(db_file)
    return db_file


def test_insert_and_retrieve_incident(temp_db: Path):
    """Verify incident insertion and retrieval with JSON parsing."""
    inc_uuid = insert_incident(
        image_path="/data/test.jpg",
        checkpoint_id="test_model.pt",
        detected_classes=["D40", "D20"],
        detection_count=2,
        max_confidence=0.88,
        raw_detections=[
            {"class": "D40", "confidence": 0.88, "bbox": [10, 20, 50, 60]},
            {"class": "D20", "confidence": 0.75, "bbox": [100, 120, 200, 250]},
        ],
        priority_score=78.2,
        priority_level="CRITICAL",
        latitude=12.9716,
        longitude=77.5946,
        location_provenance="USER_ENTERED",
        road_name="Brigade Road",
        road_classification="PRIMARY_ARTERIAL",
        db_path=temp_db,
    )

    record = get_incident_by_uuid(inc_uuid, db_path=temp_db)
    assert record is not None
    assert record["incident_uuid"] == inc_uuid
    assert record["detected_classes"] == ["D40", "D20"]
    assert record["detection_count"] == 2
    assert record["max_confidence"] == 0.88
    assert len(record["raw_detections"]) == 2
    assert record["priority_level"] == "CRITICAL"
    assert record["location_provenance"] == "USER_ENTERED"
    assert record["review_status"] == "PENDING"


def test_update_review_status(temp_db: Path):
    """Verify human reviewer verification workflow."""
    inc_uuid = insert_incident(
        image_path="/data/test.jpg",
        checkpoint_id="test_model.pt",
        detected_classes=["D00"],
        detection_count=1,
        max_confidence=0.70,
        raw_detections=[{"class": "D00", "confidence": 0.70, "bbox": [0, 0, 10, 10]}],
        priority_score=25.0,
        priority_level="LOW",
        db_path=temp_db,
    )

    updated = update_review_status(
        inc_uuid=inc_uuid,
        review_status="VERIFIED",
        human_verified_classes=["D00"],
        human_urgency_modifier=1.2,
        reviewer_notes="Confirmed hairline crack; monitor.",
        db_path=temp_db,
    )
    assert updated is True

    record = get_incident_by_uuid(inc_uuid, db_path=temp_db)
    assert record["review_status"] == "VERIFIED"
    assert record["human_verified_classes"] == ["D00"]
    assert record["reviewer_notes"] == "Confirmed hairline crack; monitor."


def test_spatial_clustering_haversine(temp_db: Path):
    """Verify repeat defect counting within 25m radius."""
    # Point A
    insert_incident(
        image_path="/data/a.jpg",
        checkpoint_id="m.pt",
        detected_classes=["D40"],
        detection_count=1,
        max_confidence=0.8,
        raw_detections=[],
        priority_score=50.0,
        priority_level="MODERATE",
        latitude=12.971600,
        longitude=77.594600,
        location_provenance="USER_ENTERED",
        db_path=temp_db,
    )

    # Point B (~10 meters away from Point A)
    # 0.00009 deg latitude ≈ 10 meters
    insert_incident(
        image_path="/data/b.jpg",
        checkpoint_id="m.pt",
        detected_classes=["D40"],
        detection_count=1,
        max_confidence=0.8,
        raw_detections=[],
        priority_score=50.0,
        priority_level="MODERATE",
        latitude=12.971690,
        longitude=77.594600,
        location_provenance="USER_ENTERED",
        db_path=temp_db,
    )

    # Point C (~500 meters away)
    insert_incident(
        image_path="/data/c.jpg",
        checkpoint_id="m.pt",
        detected_classes=["D40"],
        detection_count=1,
        max_confidence=0.8,
        raw_detections=[],
        priority_score=50.0,
        priority_level="MODERATE",
        latitude=12.976000,
        longitude=77.594600,
        location_provenance="USER_ENTERED",
        db_path=temp_db,
    )

    # Near Point A within 25 meters, should find 2 points (A and B)
    count = count_nearby_incidents(12.971600, 77.594600, radius_meters=25.0, db_path=temp_db)
    assert count == 2
