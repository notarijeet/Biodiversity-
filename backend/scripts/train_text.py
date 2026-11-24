import json
from pathlib import Path

import joblib

from config import settings
from data.loader import group_aware_split, load_wlt_osn
from models.text_classifier import build_tfidf_baseline, evaluate_predictions


def train_tfidf_baseline(
    data_path: Path, output_dir: Path, test_size: float = 0.2, seed: int = 0
) -> dict[str, float]:
    """Train and evaluate the TF-IDF baseline on the WLT-OSN dataset."""
    print(f"Loading data from: {data_path}")
    df = load_wlt_osn(data_path)

    print(f"Performing group-aware split (seed={seed})...")
    train_df, test_df = group_aware_split(df, test_size=test_size, seed=seed)

    print(f"Train set: {len(train_df)} rows ({train_df['label'].sum()} positives)")
    print(f"Test set:  {len(test_df)} rows ({test_df['label'].sum()} positives)")

    # Build and fit pipeline
    print("Fitting TF-IDF + LogisticRegression baseline...")
    pipeline = build_tfidf_baseline(max_features=20000, C=1.0)
    pipeline.fit(train_df["text"], train_df["label"])

    # Predict probabilities on held-out test set
    y_test = test_df["label"].values
    y_prob = pipeline.predict_proba(test_df["text"])[:, 1]

    # Evaluate
    metrics = evaluate_predictions(y_test, y_prob, threshold=0.5)
    print("\n=== Baseline Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved baseline metrics to: {metrics_path}")

    # Save fitted pipeline
    model_path = output_dir / "baseline_tfidf.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to: {model_path}")

    return metrics


if __name__ == "__main__":
    csv_file = settings.data_dir / "raw" / "share_data.csv"
    processed_dir = settings.data_dir / "processed"
    if csv_file.exists():
        train_tfidf_baseline(csv_file, processed_dir)
    else:
        print(f"Error: {csv_file} not found. Please place dataset in data/raw/")
