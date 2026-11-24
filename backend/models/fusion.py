from typing import Optional

from api.schemas import AnalysisStatus, DetectionPrediction, ThreatSeverity


def fuse_scores(
    text_prob: Optional[float] = None, image_confidence: Optional[float] = None
) -> float:
    """
    Fuse multimodal probabilities.
    - Both present: 0.6 * text_prob + 0.4 * image_confidence
    - Text only: text_prob
    - Image only: image_confidence
    - Neither: 0.0
    """
    if text_prob is not None and image_confidence is not None:
        return 0.6 * text_prob + 0.4 * image_confidence
    if text_prob is not None:
        return text_prob
    if image_confidence is not None:
        return image_confidence
    return 0.0


def classify_fused_score(
    score: float,
) -> tuple[DetectionPrediction, ThreatSeverity, AnalysisStatus]:
    """
    Apply decision thresholds:
    - >= 0.75: 'POTENTIAL ILLEGAL WILDLIFE TRADE', 'Critical', 'FLAGGED_FOR_HUMAN_ENFORCEMENT'
    - >= 0.50: 'POTENTIAL ILLEGAL WILDLIFE TRADE', 'Elevated', 'FLAGGED_FOR_HUMAN_ENFORCEMENT'
    - >= 0.30: 'REQUIRES MANUAL REVIEW', 'Elevated', 'FLAGGED_FOR_HUMAN_ENFORCEMENT'
    - below 0.30: 'NO CLEAR EVIDENCE DETECTED', 'Negligible', 'CLEARED'
    """
    if score >= 0.75:
        return (
            "POTENTIAL ILLEGAL WILDLIFE TRADE",
            "Critical",
            "FLAGGED_FOR_HUMAN_ENFORCEMENT",
        )
    if score >= 0.50:
        return (
            "POTENTIAL ILLEGAL WILDLIFE TRADE",
            "Elevated",
            "FLAGGED_FOR_HUMAN_ENFORCEMENT",
        )
    if score >= 0.30:
        return (
            "REQUIRES MANUAL REVIEW",
            "Elevated",
            "FLAGGED_FOR_HUMAN_ENFORCEMENT",
        )
    return ("NO CLEAR EVIDENCE DETECTED", "Negligible", "CLEARED")
