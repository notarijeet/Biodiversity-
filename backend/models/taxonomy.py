import json
from typing import Any, Optional

from api.schemas import IdentifiedSpeciesSchema, SuspiciousIndicatorSchema
from config import settings

_TAXONOMY_CACHE: Optional[list[dict[str, Any]]] = None
_INDICATORS_CACHE: Optional[dict[str, Any]] = None
_CODEWORDS_CACHE: Optional[dict[str, Any]] = None


def load_taxonomy_database() -> list[dict[str, Any]]:
    """Load CITES taxonomy database from lexicons."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        path = settings.data_dir / "lexicons" / "taxonomy_database.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _TAXONOMY_CACHE = json.load(f)
        else:
            _TAXONOMY_CACHE = []
    return _TAXONOMY_CACHE


def load_trade_indicators() -> dict[str, Any]:
    """Load trade indicators from lexicons."""
    global _INDICATORS_CACHE
    if _INDICATORS_CACHE is None:
        path = settings.data_dir / "lexicons" / "trade_indicators.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _INDICATORS_CACHE = json.load(f)
        else:
            _INDICATORS_CACHE = {"suspicious_indicators": [], "benign_indicators": []}
    return _INDICATORS_CACHE


def load_india_codewords() -> dict[str, Any]:
    """Load India-specific Hindi/Hinglish codewords from data/lexicons/india_codewords.json."""
    global _CODEWORDS_CACHE
    if _CODEWORDS_CACHE is None:
        path = settings.data_dir / "lexicons" / "india_codewords.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _CODEWORDS_CACHE = json.load(f)
        else:
            _CODEWORDS_CACHE = {}
    return _CODEWORDS_CACHE


def match_species(text: str) -> list[IdentifiedSpeciesSchema]:
    """Match species from taxonomy database and India codewords against input text."""
    text_lower = text.lower()
    matched = []

    # 1. Match from CITES taxonomy database
    taxonomy = load_taxonomy_database()
    for entry in taxonomy:
        keywords = entry.get("keywords", [])
        if any(kw.lower() in text_lower for kw in keywords):
            matched.append(
                IdentifiedSpeciesSchema(
                    commonName=entry["commonName"],
                    scientificName=entry["scientificName"],
                    citesStatus=entry["citesStatus"],
                    conservationStatus=entry["conservationStatus"],
                )
            )

    # 2. Match from India codewords dictionary
    codewords = load_india_codewords()
    for cw, meta in codewords.items():
        if cw.lower() in text_lower:
            cites = meta.get("cites", "Appendix I")
            wpa = meta.get("wpa_schedule", "Schedule I")
            matched.append(
                IdentifiedSpeciesSchema(
                    commonName=f"{meta['species']} [{cw}]",
                    scientificName=meta.get("scientific", "Unknown"),
                    citesStatus=f"CITES {cites} / WPA {wpa}",
                    conservationStatus=f"Protected ({wpa})",
                )
            )

    return matched


def match_indicators(
    text: str,
) -> tuple[list[SuspiciousIndicatorSchema], list[str], int]:
    """
    Match suspicious, benign, and India-specific trade indicators.
    Returns:
    - matched_indicators: list of SuspiciousIndicatorSchema
    - detected_keywords: list of detected keyword triggers
    - benign_hits_count: number of benign context terms detected
    """
    data = load_trade_indicators()
    text_lower = text.lower()

    detected_indicators: list[SuspiciousIndicatorSchema] = []
    detected_keywords: list[str] = []

    # 1. Generic trade indicators
    for ind in data.get("suspicious_indicators", []):
        triggers = ind.get("trigger", [])
        matched_trigs = [t for t in triggers if t.lower() in text_lower]
        if matched_trigs:
            detected_indicators.append(
                SuspiciousIndicatorSchema(
                    indicator=ind["label"],
                    severity=ind["severity"],
                    description=ind["desc"],
                )
            )
            detected_keywords.extend(matched_trigs)

    # 2. India-specific codewords
    codewords = load_india_codewords()
    for cw, meta in codewords.items():
        if cw.lower() in text_lower:
            desc = meta.get("meaning_desc", "")
            detected_indicators.append(
                SuspiciousIndicatorSchema(
                    indicator="Indian Trade Codeword Detected",
                    severity="high",
                    description=f"Matched Hindi/Hinglish codeword '{cw}': {desc}",
                )
            )
            detected_keywords.append(cw)

    # 3. Benign indicators
    benign_terms = data.get("benign_indicators", [])
    benign_count = sum(1 for b in benign_terms if b.lower() in text_lower)

    return detected_indicators, list(set(detected_keywords)), benign_count
