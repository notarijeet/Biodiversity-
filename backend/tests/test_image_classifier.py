import base64
import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from models.image_classifier import (
    IWT_NEGATIVE_QUERIES,
    IWT_POSITIVE_QUERIES,
    CLIPOnlineClassifier,
    classify_image,
    decode_base64_image,
)


def _create_sample_base64_image() -> str:
    """Create a minimal 10x10 RGB test image encoded in base64."""
    img = Image.new("RGB", (10, 10), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


def test_decode_base64_image_valid():
    b64_str = _create_sample_base64_image()
    image = decode_base64_image(b64_str)
    assert isinstance(image, Image.Image)
    assert image.size == (10, 10)
    assert image.mode == "RGB"


def test_decode_base64_image_invalid():
    with pytest.raises(ValueError, match="Invalid or empty base64 string"):
        decode_base64_image("")

    with pytest.raises(ValueError, match="Failed to decode"):
        decode_base64_image("data:image/png;base64,not-a-valid-base64-payload!!!")


def test_iwt_queries_integrity():
    assert len(IWT_POSITIVE_QUERIES) >= 10
    assert len(IWT_NEGATIVE_QUERIES) >= 5
    assert all(isinstance(q, str) and len(q) > 5 for q in IWT_POSITIVE_QUERIES)
    assert all(isinstance(q, str) and len(q) > 5 for q in IWT_NEGATIVE_QUERIES)


def test_classify_image_with_mock():
    # Mock image
    img = Image.new("RGB", (64, 64), color="red")

    total_queries = len(IWT_POSITIVE_QUERIES) + len(IWT_NEGATIVE_QUERIES)

    # 1. High positive logits
    mock_processor = MagicMock()
    mock_processor.return_value.to.return_value = {"mock_input": torch.zeros(1)}

    mock_outputs = MagicMock()
    # High score on first positive query
    mock_logits = torch.zeros(1, total_queries)
    mock_logits[0, 0] = 10.0
    mock_outputs.logits_per_image = mock_logits

    mock_model = MagicMock(return_value=mock_outputs)

    res_high = classify_image(
        image=img,
        model=mock_model,
        processor=mock_processor,
        threshold=0.4,
        device="cpu",
    )
    assert res_high["image_classification"] == "SUSPECTED"
    assert res_high["image_confidence"] > 0.5
    assert res_high["top_matching_query"] == IWT_POSITIVE_QUERIES[0]

    # 2. High negative logits
    mock_logits_neg = torch.zeros(1, total_queries)
    mock_logits_neg[0, len(IWT_POSITIVE_QUERIES)] = 10.0  # high on first negative query
    mock_outputs.logits_per_image = mock_logits_neg

    res_low = classify_image(
        image=img,
        model=mock_model,
        processor=mock_processor,
        threshold=0.4,
        device="cpu",
    )
    assert res_low["image_classification"] == "UNCLEAR"
    assert res_low["image_confidence"] < 0.2


def test_clip_online_classifier_lazy_initialization():
    classifier = CLIPOnlineClassifier()
    # Respect Rule 30: Model weights must not load at instantiation
    assert classifier.model is None
    assert classifier.processor is None
