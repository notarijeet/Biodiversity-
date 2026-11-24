from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    find_worst_errors,
    generate_pr_curve,
    group_performance_breakdown,
)


def test_compute_classification_metrics():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"precision", "recall", "f1"}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_compute_confusion_matrix():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tp"] == 2
    assert cm["fn"] == 1
    assert cm["tn"] == 2
    assert cm["fp"] == 0
    assert cm["total"] == 5


def test_generate_pr_curve(tmp_path: Path):
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
    out_img = tmp_path / "test_pr.png"

    auc_score = generate_pr_curve(y_true, y_prob, output_path=out_img)
    assert 0.0 <= auc_score <= 1.0
    assert out_img.exists()
    assert out_img.stat().st_size > 0


def test_find_worst_errors():
    df = pd.DataFrame(
        {
            "text": ["post 1", "post 2", "post 3", "post 4"],
            "label": [0, 0, 1, 1],
            "source_group": ["g1", "g1", "g2", "g2"],
        }
    )
    y_prob = np.array([0.9, 0.1, 0.2, 0.8])

    fps, fns = find_worst_errors(df, y_prob, threshold=0.5, top_n=5)

    assert len(fps) == 1
    assert fps.iloc[0]["text"] == "post 1"
    assert fps.iloc[0]["predicted_prob"] == 0.9

    assert len(fns) == 1
    assert fns.iloc[0]["text"] == "post 3"
    assert fns.iloc[0]["predicted_prob"] == 0.2


def test_group_performance_breakdown():
    df = pd.DataFrame(
        {
            "text": [f"post {i}" for i in range(10)],
            "label": [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            "source_group": ["grp_a"] * 5 + ["grp_b"] * 5,
        }
    )
    y_prob = np.array([0.8, 0.7, 0.1, 0.2, 0.9, 0.8, 0.2, 0.1, 0.3, 0.2])

    breakdown = group_performance_breakdown(df, y_prob, threshold=0.5, min_samples=3)
    assert len(breakdown) == 2
    assert "source_group" in breakdown.columns
    assert "f1" in breakdown.columns
