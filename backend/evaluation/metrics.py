import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Ensure matplotlib cache directory is writable inside sandbox/Docker
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")


def compute_classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calculate precision, recall, and F1 scores."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_confusion_matrix(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Compute true positive, false positive, true negative, false negative."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(len(y_true)),
    }


def get_full_report(y_true: Any, y_pred: Any) -> str:
    """Return text classification report."""
    return classification_report(y_true, y_pred, zero_division=0)


def generate_pr_curve(
    y_true: Any,
    y_prob: Any,
    output_path: Path,
    title: str = "Precision-Recall Curve (WLT-OSN Test Set)",
) -> float:
    """Plot and save the Precision-Recall curve, returning the PR-AUC score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="#059669",
        lw=2,
        label=f"TF-IDF Baseline (AUC = {pr_auc:.4f})",
    )
    plt.axhline(
        y=float(np.mean(y_true)),
        color="#dc2626",
        linestyle="--",
        label=f"No Skill (Prevalence = {float(np.mean(y_true)):.4f})",
    )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pr_auc


def group_performance_breakdown(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Compute evaluation metrics sliced by source_group."""
    records = []
    df_eval = df.copy()
    df_eval["prob"] = y_prob
    df_eval["pred"] = (y_prob >= threshold).astype(int)

    for group, grp_df in df_eval.groupby("source_group"):
        if len(grp_df) < min_samples:
            continue
        y_t = grp_df["label"].values
        y_p = grp_df["pred"].values
        pos_count = int(y_t.sum())
        total_count = len(grp_df)

        prec = float(precision_score(y_t, y_p, zero_division=0)) if pos_count > 0 else 1.0
        rec = float(recall_score(y_t, y_p, zero_division=0)) if pos_count > 0 else 1.0
        f1 = float(f1_score(y_t, y_p, zero_division=0)) if pos_count > 0 else 1.0

        records.append(
            {
                "source_group": group,
                "samples": total_count,
                "positives": pos_count,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )

    return pd.DataFrame(records).sort_values(by="samples", ascending=False)


def find_worst_errors(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find top N false positives and top N false negatives.
    Returns (false_positives_df, false_negatives_df).
    """
    df_eval = df.copy()
    df_eval["predicted_prob"] = y_prob
    df_eval["predicted_label"] = (y_prob >= threshold).astype(int)

    # False Positives: true label == 0, predicted == 1, sorted by highest prob
    fps = df_eval[(df_eval["label"] == 0) & (df_eval["predicted_label"] == 1)].copy()
    fps = fps.sort_values(by="predicted_prob", ascending=False).head(top_n)

    # False Negatives: true label == 1, predicted == 0, sorted by lowest prob
    fns = df_eval[(df_eval["label"] == 1) & (df_eval["predicted_label"] == 0)].copy()
    fns = fns.sort_values(by="predicted_prob", ascending=True).head(top_n)

    return fps, fns
