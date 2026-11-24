import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

from models.text_classifier import (
    BaselineTextClassifier,
    IWTDataset,
    XLMROnlineClassifier,
    build_tfidf_baseline,
    compute_eval_metrics,
    evaluate_predictions,
    get_training_args,
)


def test_build_tfidf_baseline():
    pipeline = build_tfidf_baseline(max_features=500, C=1.0)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    assert pipeline.named_steps["clf"].class_weight == "balanced"


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.85, 0.2, 0.9, 0.4, 0.7, 0.05, 0.3])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "pr_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_baseline_text_classifier_predict_proba(tmp_path: Path):
    texts = [
        "authentic elephant ivory carving for sale ivory",
        "pangolin scales dry bulk ivory sale wholesale",
        "beautiful wild sunset at national park visit",
        "family visit to wildlife sanctuary park safari",
    ]
    labels = [1, 1, 0, 0]

    pipeline = build_tfidf_baseline(max_features=100, min_df=1)
    pipeline.fit(texts, labels)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(pipeline, model_path)

    classifier = BaselineTextClassifier(model_path=model_path)
    classifier.load()

    prob_positive = classifier.predict_proba("selling carved ivory ornaments")
    prob_negative = classifier.predict_proba("photographing birds at sanctuary")

    assert 0.0 <= prob_positive <= 1.0
    assert 0.0 <= prob_negative <= 1.0
    assert prob_positive > prob_negative


def test_baseline_metrics_file_validity():
    metrics_path = (
        Path(__file__).parent.parent.parent / "data" / "processed" / "baseline_metrics.json"
    )
    if not metrics_path.exists():
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert set(data.keys()) == {"precision", "recall", "f1", "pr_auc"}
    assert all(isinstance(v, (int, float)) for v in data.values())
    assert all(0.0 <= v <= 1.0 for v in data.values())


def test_iwt_dataset():
    encodings = {
        "input_ids": torch.tensor([[101, 102], [103, 104]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }
    labels = [1, 0]
    dataset = IWTDataset(encodings, labels)

    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 1


def test_get_training_args(tmp_path: Path):
    out_dir = str(tmp_path / "training_output")
    args = get_training_args(output_dir=out_dir, num_epochs=3, batch_size=4)

    assert args.output_dir == out_dir
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 4
    # Rule 27 check: eval_strategy used in transformers 4.46+
    assert args.eval_strategy == "epoch"
    assert args.gradient_checkpointing is True


def test_compute_eval_metrics():
    # Logits: class 0 vs class 1
    logits = np.array([[-2.0, 2.0], [3.0, -3.0], [-1.5, 1.5], [2.5, -2.5]])
    labels = np.array([1, 0, 1, 0])

    metrics = compute_eval_metrics((logits, labels))
    assert set(metrics.keys()) == {"precision", "recall", "f1", "pr_auc"}
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_xlmr_classifier_lazy_loading(tmp_path: Path):
    missing_checkpoint = tmp_path / "missing_xlmr"
    classifier = XLMROnlineClassifier(checkpoint_path=missing_checkpoint)

    # Must raise FileNotFoundError on missing weights when attempting to load
    with pytest.raises(FileNotFoundError):
        classifier.load()

    with pytest.raises(FileNotFoundError):
        classifier.predict_proba("sample text")
