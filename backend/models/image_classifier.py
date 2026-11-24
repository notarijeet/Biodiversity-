import base64
import io
from typing import Any, Optional, Union

import torch
from PIL import Image

from config import get_device, settings

CLIP_MODEL = settings.clip_model_name

# Positive query prompts — describe what IWT products look like
IWT_POSITIVE_QUERIES = [
    "pangolin scales in a bag for sale",
    "dried pangolin scales wildlife product",
    "elephant ivory tusk carved",
    "elephant ivory jewellery for sale",
    "tiger skin pelt fur",
    "tiger bone medicine",
    "rhinoceros horn carved cup",
    "monitor lizard body parts",
    "dried wildlife products packaged for sale",
    "live exotic animal in a small cage",
    "slow loris pet video",
    "wildlife product packaged for shipping",
    # India-specific queries (Task 3.2)
    "hathajodi monitor lizard dried tantric charm",
    "safed sona raw elephant ivory tusk carved",
    "khurpa dried pangolin scales illicit trade",
    "kasturi musk deer glandular pod perfume",
    "shahtoosh tibetan antelope wool fine shawl",
    "rakta chandan red sanders sandalwood timber logs",
]

# Negative query prompts — describe what legitimate content looks like
IWT_NEGATIVE_QUERIES = [
    "wildlife photographer in the field",
    "national park safari vehicle",
    "wildlife conservation documentary",
    "zoo animal in enclosure",
    "domestic pet cat dog",
    "legal antique carved bone with certificate",
]


def decode_base64_image(b64_string: str) -> Image.Image:
    """Utility: decode a base64 image string to a PIL Image in RGB format."""
    if not isinstance(b64_string, str) or not b64_string.strip():
        raise ValueError("Invalid or empty base64 string")

    if b64_string.startswith("data:image"):
        parts = b64_string.split(",", 1)
        if len(parts) > 1:
            b64_string = parts[1]

    try:
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}") from e


def load_clip_model(model_name: Optional[str] = None):
    """
    Lazily load CLIP model and processor.
    Uses device from config.get_device() (Rule 26).
    """
    from transformers import CLIPModel, CLIPProcessor

    name = model_name or settings.clip_model_name
    device = get_device()
    model = CLIPModel.from_pretrained(name).to(device)
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


def classify_image(
    image: Image.Image,
    model: Any,
    processor: Any,
    threshold: float = 0.4,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Zero-shot classify an image for IWT content using positive and negative prompt sets.
    Returns:
    - image_confidence: float in [0.0, 1.0] (sum of positive probabilities)
    - top_matching_query: top matched positive description
    - image_classification: 'SUSPECTED' if confidence >= threshold else 'UNCLEAR'
    """
    active_device = device or get_device()
    all_queries = IWT_POSITIVE_QUERIES + IWT_NEGATIVE_QUERIES

    inputs = processor(
        text=all_queries,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(active_device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # shape: [num_queries]
        probs = logits.softmax(dim=0).cpu().numpy()

    num_pos = len(IWT_POSITIVE_QUERIES)
    pos_probs = probs[:num_pos]
    pos_score = float(pos_probs.sum())
    top_idx = int(pos_probs.argmax())
    top_query = IWT_POSITIVE_QUERIES[top_idx]

    return {
        "image_confidence": round(pos_score, 4),
        "top_matching_query": top_query,
        "image_classification": "SUSPECTED" if pos_score >= threshold else "UNCLEAR",
    }


class CLIPOnlineClassifier:
    """Lazy inference wrapper for the zero-shot CLIP image classifier."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.clip_model_name
        self.model = None
        self.processor = None
        self.device = get_device()

    def load(self) -> None:
        """Load model and processor lazily (Rule 30)."""
        if self.model is None or self.processor is None:
            self.model, self.processor = load_clip_model(self.model_name)

    def predict_image(
        self, image_input: Union[Image.Image, str], threshold: float = 0.4
    ) -> dict[str, Any]:
        """Classify a PIL Image or base64 data string."""
        if self.model is None or self.processor is None:
            self.load()

        if isinstance(image_input, str):
            image = decode_base64_image(image_input)
        else:
            image = image_input

        return classify_image(
            image=image,
            model=self.model,
            processor=self.processor,
            threshold=threshold,
            device=self.device,
        )
