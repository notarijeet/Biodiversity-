from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from config import get_device, settings

# ============================================================================
# Section 1: TF-IDF + LogisticRegression Baseline
# ============================================================================


def build_tfidf_baseline(max_features: int = 20000, C: float = 1.0, min_df: int = 2) -> Pipeline:
    """
    Build TF-IDF + LogisticRegression baseline with balanced class weights.
    ngram_range=(1,2) captures key bi-grams (e.g., 'carved ivory', 'dry scales').
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=min_df,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",  # critical for ~3% positive rate
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true: Union[list, np.ndarray],
    y_prob: Union[list, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate precision, recall, F1, and PR-AUC given true labels and predicted probabilities."""
    y_prob_arr = np.asarray(y_prob)
    y_true_arr = np.asarray(y_true)
    y_pred_thresh = (y_prob_arr >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_true_arr, y_pred_thresh, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_thresh, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_thresh, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
    }


class BaselineTextClassifier:
    """Inference wrapper for the TF-IDF baseline model."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None

    def load(self) -> None:
        if self.model_path and self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict_proba(self, text: str) -> float:
        """Return probability of positive class (suspected wildlife trade)."""
        if self.pipeline is None:
            self.load()
        assert self.pipeline is not None
        probs = self.pipeline.predict_proba([text])[0]
        return float(probs[1]) if len(probs) > 1 else float(probs[0])


# ============================================================================
# Section 2: XLM-RoBERTa Transformer Components
# ============================================================================


class IWTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for tokenized text inputs."""

    def __init__(self, encodings: dict[str, Any], labels: Optional[list[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))


def get_tokenizer(model_name: Optional[str] = None):
    """Lazily load XLM-RoBERTa tokenizer."""
    from transformers import AutoTokenizer

    name = model_name or settings.text_model_name
    return AutoTokenizer.from_pretrained(name)


def tokenize_dataset(texts: list[str], tokenizer: Any, max_length: int = 128) -> dict[str, Any]:
    """Tokenize texts for sequence classification. Max length 128 keeps memory low."""
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def compute_eval_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute precision, recall, F1, and PR-AUC for HuggingFace Trainer evaluation."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return evaluate_predictions(labels, probs, threshold=0.5)


def get_training_args(
    output_dir: str, num_epochs: int = 5, batch_size: int = 8
) -> "transformers.TrainingArguments":  # noqa: F821
    """Construct TrainingArguments adhering to 4GB GTX 1650 and eval_strategy requirements."""
    import inspect

    from transformers import TrainingArguments

    use_cuda = torch.cuda.is_available()

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 16,
        "fp16": use_cuda,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "eval_strategy": "epoch",  # Rule 27: eval_strategy, not evaluation_strategy
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "logging_steps": 50,
        "warmup_steps": 50,
        "dataloader_num_workers": 0,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "activation_checkpointing" in sig.parameters:
        kwargs["activation_checkpointing"] = True
    else:
        kwargs["gradient_checkpointing"] = True

    return TrainingArguments(**kwargs)


class XLMROnlineClassifier:
    """Inference wrapper for fine-tuned XLM-RoBERTa sequence classification."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = (
            checkpoint_path
            or (Path(settings.text_model_checkpoint) if settings.text_model_checkpoint else None)
            or (settings.models_dir / "xlm-roberta-iwt")
        )
        self.tokenizer = None
        self.model = None
        self.device = get_device()

    def load(self) -> None:
        """Lazily load model and tokenizer from checkpoint (Rule 30)."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"XLM-RoBERTa checkpoint not found at: {self.checkpoint_path}.\n"
                "Train using notebooks/02_xlm_roberta_training.ipynb and place checkpoint here."
            )

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.checkpoint_path)
        ).to(self.device)
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        """Predict positive probability (suspected wildlife trade) for given text."""
        if self.model is None or self.tokenizer is None:
            self.load()

        assert self.tokenizer is not None
        assert self.model is not None

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return float(probs[0, 1].item())
