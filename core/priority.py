"""
RoadSense AI - Deterministic Priority Ranking Engine (Formula v1.0)
Calculates municipal inspection dispatch priority scores based on:
1. Damage class severity (D40 > D20 > D10 > D00)
2. Detection confidence
3. Spatial repeat observation density (within 25m)
4. Functional road classification (Arterial vs Local)
5. Field engineer urgency modifier
Guarantees 100% mathematical determinism for identical inputs.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from core import config


@dataclass
class PriorityResult:
    """Detailed output of the priority calculation."""
    score: float
    level: str  # 'CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'INSUFFICIENT_INFO'
    formula_version: str
    breakdown: Dict[str, float]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "formula_version": self.formula_version,
            "breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "rationale": self.rationale,
        }


def calculate_priority(
    detected_classes: List[str],
    max_confidence: float,
    nearby_repeat_count: int = 0,
    road_classification: str = "UNKNOWN",
    human_urgency_modifier: float = 1.0,
    config_weights: Optional[Dict[str, float]] = None,
) -> PriorityResult:
    """
    Compute deterministic inspection priority score S in [0, 100].

    Formula v1.0:
    S = min(100, (w_class * conf * 40) + (min(N_repeat, 5) * 6) + (w_road * 15) + (w_human * 15))
    """
    version = config.PRIORITY_FORMULA_VERSION
    scaling = config_weights or config.PRIORITY_SCALING

    # Guard: No detections or invalid confidence
    if not detected_classes or max_confidence <= 0.0:
        return PriorityResult(
            score=0.0,
            level="INSUFFICIENT_INFO",
            formula_version=version,
            breakdown={
                "class_points": 0.0,
                "cluster_points": 0.0,
                "road_points": 0.0,
                "human_points": 0.0,
            },
            rationale="No damage detections present to assess priority.",
        )

    # 1. Damage Class Severity Weight
    # Pick highest severity among all detected classes
    max_w_class = 0.0
    primary_class = "UNKNOWN"
    for c in detected_classes:
        w = config.CLASS_SEVERITY_WEIGHTS.get(c, 0.3)
        if w > max_w_class:
            max_w_class = w
            primary_class = c

    # Confined confidence in [0, 1]
    safe_conf = max(0.0, min(1.0, float(max_confidence)))
    class_pts = max_w_class * safe_conf * scaling["class_confidence_scale"]

    # 2. Spatial Repeat Count Points
    capped_repeat = min(float(scaling["cluster_repeat_cap"]), max(0.0, float(nearby_repeat_count)))
    cluster_pts = capped_repeat * scaling["cluster_repeat_scale"]

    # 3. Road Classification Points
    road_weight = config.ROAD_CLASS_WEIGHTS.get(road_classification.upper(), config.ROAD_CLASS_WEIGHTS["UNKNOWN"])
    road_pts = road_weight * scaling["road_class_scale"]

    # 4. Human Urgency Points (normalized around 1.0)
    # Range: 0.5 to 1.5 -> maps to scaling["human_urgency_scale"]
    clamped_human = max(0.5, min(1.5, float(human_urgency_modifier)))
    human_pts = (clamped_human / 1.5) * scaling["human_urgency_scale"]

    # Total Score capped at 100
    raw_total = class_pts + cluster_pts + road_pts + human_pts
    final_score = min(100.0, max(0.0, raw_total))

    # Determine Priority Level Band
    if final_score >= config.PRIORITY_LEVELS["CRITICAL"]["min_score"]:
        level = "CRITICAL"
    elif final_score >= config.PRIORITY_LEVELS["HIGH"]["min_score"]:
        level = "HIGH"
    elif final_score >= config.PRIORITY_LEVELS["MODERATE"]["min_score"]:
        level = "MODERATE"
    else:
        level = "LOW"

    class_name = config.CLASS_HUMAN_NAMES.get(primary_class, primary_class)
    rationale = (
        f"Primary defect: {class_name} ({primary_class}, base weight {max_w_class:.1f}) at "
        f"{safe_conf*100:.1f}% confidence. {nearby_repeat_count} repeat observation(s) within "
        f"{config.CLUSTER_RADIUS_METERS:.0f}m on {road_classification.replace('_', ' ').title()} road."
    )

    return PriorityResult(
        score=final_score,
        level=level,
        formula_version=version,
        breakdown={
            "class_points": class_pts,
            "cluster_points": cluster_pts,
            "road_points": road_pts,
            "human_points": human_pts,
        },
        rationale=rationale,
    )


def get_priority_level(score: float) -> str:
    """Map numeric priority score to human-readable priority tier."""
    if score >= config.PRIORITY_LEVELS["CRITICAL"]["min_score"]:
        return "CRITICAL"
    elif score >= config.PRIORITY_LEVELS["HIGH"]["min_score"]:
        return "HIGH"
    elif score >= config.PRIORITY_LEVELS["MODERATE"]["min_score"]:
        return "MODERATE"
    else:
        return "LOW"

