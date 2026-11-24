import re


def clean_text(text: str) -> str:
    """Basic text sanitization for NLP processing."""
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
