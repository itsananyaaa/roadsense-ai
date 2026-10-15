"""
RoadSense AI - Unit Tests for Priority Ranking Engine
Tests mathematical determinism, boundary conditions, and priority level bands.
"""

import pytest
from core.priority import calculate_priority


def test_priority_determinism():
    """Identical inputs must yield identical scores down to float precision."""
    res1 = calculate_priority(
        detected_classes=["D40"],
        max_confidence=0.88,
        nearby_repeat_count=2,
        road_classification="PRIMARY_ARTERIAL",
        human_urgency_modifier=1.2,
    )
    res2 = calculate_priority(
        detected_classes=["D40"],
        max_confidence=0.88,
        nearby_repeat_count=2,
        road_classification="PRIMARY_ARTERIAL",
        human_urgency_modifier=1.2,
    )
    assert res1.score == res2.score
    assert res1.level == res2.level
    assert res1.breakdown == res2.breakdown


def test_priority_pothole_vs_crack():
    """Potholes (D40) must receive higher priority than longitudinal cracks (D00) at same confidence."""
    pothole_res = calculate_priority(
        detected_classes=["D40"],
        max_confidence=0.75,
        nearby_repeat_count=0,
        road_classification="RESIDENTIAL_LOCAL",
    )
    crack_res = calculate_priority(
        detected_classes=["D00"],
        max_confidence=0.75,
        nearby_repeat_count=0,
        road_classification="RESIDENTIAL_LOCAL",
    )
    assert pothole_res.score > crack_res.score


def test_priority_score_boundaries():
    """Score must strictly remain within [0.0, 100.0]."""
    # Maxed-out inputs
    max_res = calculate_priority(
        detected_classes=["D40", "D20"],
        max_confidence=1.0,
        nearby_repeat_count=50,  # exceeds cap
        road_classification="PRIMARY_ARTERIAL",
        human_urgency_modifier=2.0,  # exceeds cap
    )
    assert max_res.score <= 100.0
    assert max_res.level == "CRITICAL"

    # Empty inputs
    empty_res = calculate_priority(
        detected_classes=[],
        max_confidence=0.0,
    )
    assert empty_res.score == 0.0
    assert empty_res.level == "INSUFFICIENT_INFO"


def test_repeat_count_cap():
    """Repeat detections must be capped at 5 to avoid infinite score inflation."""
    res_at_5 = calculate_priority(
        detected_classes=["D40"],
        max_confidence=0.8,
        nearby_repeat_count=5,
    )
    res_at_20 = calculate_priority(
        detected_classes=["D40"],
        max_confidence=0.8,
        nearby_repeat_count=20,
    )
    assert res_at_5.score == res_at_20.score
    assert res_at_5.breakdown["cluster_points"] == 30.0
