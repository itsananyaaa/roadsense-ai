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


def test_priority_formula_independent_math_verification():
    """Verify priority score calculation matches independent mathematical evaluation."""
    # Test case 1: Pothole on Primary Arterial with 3 cluster repeats
    w_class = 1.0  # D40
    conf = 0.85
    repeats = 3
    w_road = 1.0   # PRIMARY_ARTERIAL
    human = 1.0

    expected_class_pts = w_class * conf * 40.0  # 34.0
    expected_cluster_pts = min(repeats, 5) * 6.0  # 18.0
    expected_road_pts = w_road * 15.0            # 15.0
    expected_human_pts = (human / 1.5) * 15.0    # 10.0
    expected_score = expected_class_pts + expected_cluster_pts + expected_road_pts + expected_human_pts  # 77.0

    res = calculate_priority(
        detected_classes=["D40"],
        max_confidence=conf,
        nearby_repeat_count=repeats,
        road_classification="PRIMARY_ARTERIAL",
        human_urgency_modifier=human,
    )

    assert pytest.approx(res.score, 0.01) == expected_score
    assert res.level == "CRITICAL"  # >= 75
    assert pytest.approx(res.breakdown["class_points"], 0.01) == expected_class_pts
    assert pytest.approx(res.breakdown["cluster_points"], 0.01) == expected_cluster_pts
    assert pytest.approx(res.breakdown["road_points"], 0.01) == expected_road_pts
    assert pytest.approx(res.breakdown["human_points"], 0.01) == expected_human_pts


def test_priority_band_boundary_transitions():
    """Verify exact inclusive/exclusive band boundaries (75, 55, 35)."""
    from core.priority import get_priority_level

    # Critical boundary: >= 75
    assert get_priority_level(100.0) == "CRITICAL"
    assert get_priority_level(75.0) == "CRITICAL"
    assert get_priority_level(74.999) == "HIGH"

    # High boundary: 55 to < 75
    assert get_priority_level(55.0) == "HIGH"
    assert get_priority_level(54.999) == "MODERATE"

    # Moderate boundary: 35 to < 55
    assert get_priority_level(35.0) == "MODERATE"
    assert get_priority_level(34.999) == "LOW"

    # Low boundary: < 35
    assert get_priority_level(0.0) == "LOW"

