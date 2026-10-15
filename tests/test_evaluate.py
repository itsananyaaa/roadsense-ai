"""
RoadSense AI - Automated Tests for Dual-Domain Evaluation Engine
Verifies:
1. Latency benchmarking functionality across devices
2. Generation of publication-quality confusion matrix image
3. Generation of Precision-Recall curves image
4. Creation of complete dual-domain evaluation JSON reports
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core import config
from ml.evaluate import (
    benchmark_hardware_latency,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    evaluate_dual_domain,
    CANONICAL_CLASSES
)


@pytest.fixture
def temp_output_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="roadsense_test_eval_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_confusion_matrix_plotting(temp_output_dir):
    """Test normalized confusion matrix rendering and artifact generation."""
    cm_path = temp_output_dir / "test_cm.png"
    cm_data = np.array([
        [100, 5, 2, 0, 10],
        [4, 80, 1, 2, 5],
        [2, 3, 90, 8, 12],
        [1, 0, 4, 110, 15],
        [8, 4, 9, 12, 0]
    ])
    classes = ["D00", "D10", "D20", "D40", "Background"]

    plot_confusion_matrix(cm_data, classes, cm_path)

    assert cm_path.exists()
    assert cm_path.stat().st_size > 1000  # Non-trivial image file


def test_pr_curve_plotting(temp_output_dir):
    """Test Precision-Recall curve rendering and artifact generation."""
    pr_path = temp_output_dir / "test_pr.png"
    recalls = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    pr_data = {
        "D00": {"recall": recalls, "precision": [0.9, 0.85, 0.8, 0.7, 0.5, 0.3], "ap50": 0.65},
        "D10": {"recall": recalls, "precision": [0.8, 0.75, 0.65, 0.5, 0.4, 0.2], "ap50": 0.45},
        "D20": {"recall": recalls, "precision": [0.95, 0.88, 0.82, 0.75, 0.6, 0.4], "ap50": 0.70},
        "D40": {"recall": recalls, "precision": [0.98, 0.92, 0.88, 0.80, 0.65, 0.45], "ap50": 0.75},
        "all": {"recall": recalls, "precision": [0.91, 0.85, 0.79, 0.69, 0.54, 0.34], "ap50": 0.64},
    }

    plot_precision_recall_curve(pr_data, pr_path)

    assert pr_path.exists()
    assert pr_path.stat().st_size > 1000


def test_hardware_latency_benchmark():
    """Verify hardware benchmarking runs without error on active weights."""
    model_path = config.get_active_model_path()
    res = benchmark_hardware_latency(model_path=model_path, img_size=320, warmup=2, iterations=5)

    assert isinstance(res, dict)
    assert len(res) > 0
    # Either MPS, CUDA, or CPU must have run
    dev_key = list(res.keys())[0]
    dev_res = res[dev_key]
    assert "mean_latency_ms" in dev_res
    assert dev_res["mean_latency_ms"] > 0
    assert "fps" in dev_res


def test_evaluate_dual_domain_end_to_end(temp_output_dir):
    """Verify complete dual-domain evaluation produces required artifacts."""
    model_path = config.get_active_model_path()

    summary = evaluate_dual_domain(
        model_path=model_path,
        output_dir=temp_output_dir,
        benchmark_hw=False  # Fast run for unit test
    )

    # 1. Returned structure checks
    assert "evaluation_id" in summary
    assert "in_domain_summary" in summary
    assert "unseen_country_summary" in summary

    # 2. File artifacts checks
    india_file = temp_output_dir / "eval_india.json"
    czech_file = temp_output_dir / "eval_czech.json"
    bench_file = temp_output_dir / "eval_benchmark.json"
    cm_file = temp_output_dir / "confusion_matrix.png"
    pr_file = temp_output_dir / "pr_curve.png"

    for artifact in [india_file, czech_file, bench_file, cm_file, pr_file]:
        assert artifact.exists(), f"Artifact missing: {artifact}"
        assert artifact.stat().st_size > 0

    # 3. Content checks
    with open(india_file, "r") as f:
        india_data = json.load(f)
    assert "mAP50" in india_data
    assert "per_class" in india_data
    for cls in CANONICAL_CLASSES:
        assert cls in india_data["per_class"]

    with open(czech_file, "r") as f:
        czech_data = json.load(f)
    assert "mAP50" in czech_data
    assert "domain_transfer_gap" in czech_data
