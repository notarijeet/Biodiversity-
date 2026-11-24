import base64
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, File, Form, UploadFile

from api.schemas import (
    AnalysisResult,
    AnalyzeRequest,
    HealthResponse,
    InputType,
    SuspiciousIndicatorSchema,
)
from config import settings
from models.fusion import classify_fused_score, fuse_scores
from models.image_classifier import CLIPOnlineClassifier
from models.taxonomy import match_indicators, match_species
from models.text_classifier import BaselineTextClassifier, XLMROnlineClassifier

router = APIRouter()

# Global lazy model instances (Rule 30: instantiated on first request)
_text_classifier: Optional[Any] = None
_image_classifier: Optional[CLIPOnlineClassifier] = None


def get_text_model() -> Optional[Any]:
    """Lazily load text classifier (XLM-RoBERTa if checkpoint exists, else TF-IDF baseline)."""
    global _text_classifier
    if _text_classifier is None:
        xlmr_path = settings.models_dir / "xlm-roberta-iwt"
        baseline_path = settings.data_dir / "processed" / "baseline_tfidf.joblib"

        if xlmr_path.exists():
            try:
                clf = XLMROnlineClassifier(checkpoint_path=xlmr_path)
                clf.load()
                _text_classifier = clf
            except Exception:
                _text_classifier = None

        if _text_classifier is None and baseline_path.exists():
            try:
                clf = BaselineTextClassifier(model_path=baseline_path)
                clf.load()
                _text_classifier = clf
            except Exception:
                _text_classifier = None

    return _text_classifier


def get_image_model() -> Optional[CLIPOnlineClassifier]:
    """Lazily load CLIP classifier on first request."""
    global _image_classifier
    if _image_classifier is None:
        _image_classifier = CLIPOnlineClassifier()
    return _image_classifier


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness check endpoint."""
    return HealthResponse(status="ok")


@router.post("/analyze", response_model=AnalysisResult)
def analyze(payload: AnalyzeRequest) -> AnalysisResult:
    """
    Multimodal analysis endpoint (JSON payload).
    Fuses text and visual signals into a unified AnalysisResult matching frontend types.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M") + " UTC"
    case_id = f"WID-{datetime.now(timezone.utc).year}-{uuid.uuid4().hex[:4].upper()}"

    raw_type = payload.type or payload.inputType or "screenshot"
    resolved_input_type: InputType = "screenshot" if raw_type == "screenshot" else "url"
    input_source = payload.inputSource or "investigator_submission"

    text = (
        payload.text
        or payload.text_content
        or payload.extractedText
        or (payload.data if resolved_input_type == "url" else "")
        or payload.url
        or ""
    ).strip()

    image_b64 = payload.image_base64
    if not image_b64 and payload.data and payload.data.startswith("data:image"):
        image_b64 = payload.data

    text_prob: Optional[float] = None
    if text:
        text_model = get_text_model()
        if text_model is not None:
            try:
                text_prob = text_model.predict_proba(text)
            except Exception:
                text_prob = None

    image_conf: Optional[float] = None
    top_image_query: Optional[str] = None
    if image_b64:
        image_model = get_image_model()
        if image_model is not None:
            try:
                img_res = image_model.predict_image(image_b64)
                image_conf = img_res["image_confidence"]
                top_image_query = img_res.get("top_matching_query")
            except Exception:
                image_conf = None

    # Multimodal score fusion
    fused_score = fuse_scores(text_prob=text_prob, image_confidence=image_conf)

    # Taxonomy and indicators lookup
    species_list = match_species(text) if text else []
    indicators_list, keywords, benign_hits = match_indicators(text) if text else ([], [], 0)

    # Add visual indicator if image confidence is elevated
    if image_conf is not None and image_conf >= 0.4 and top_image_query:
        indicators_list.append(
            SuspiciousIndicatorSchema(
                indicator="Visual Wildlife Pattern Match",
                severity="high" if image_conf >= 0.75 else "medium",
                description=(
                    f"Zero-shot visual pattern detected: '{top_image_query}' "
                    f"(confidence: {image_conf:.2f})"
                ),
            )
        )

    # Domain knowledge reinforcement: CITES taxonomy DB & Trade Indicators
    has_high_severity = any(i.severity == "high" for i in indicators_list)
    if species_list and indicators_list:
        rule_score = 0.88 if has_high_severity else 0.65
        fused_score = max(fused_score, rule_score)
    elif species_list or indicators_list:
        fused_score = max(fused_score, 0.38)

    if benign_hits >= 2 and not indicators_list and not species_list:
        fused_score = min(fused_score, 0.15)

    prediction, threat_severity, analysis_status = classify_fused_score(fused_score)

    # Construct explainable reasoning
    explanation_parts = []
    if prediction == "POTENTIAL ILLEGAL WILDLIFE TRADE":
        explanation_parts.append(f"High-confidence pattern match (fused score: {fused_score:.2f}).")
        if species_list:
            sp = species_list[0]
            explanation_parts.append(
                f"Identified protected species: {sp.commonName} ({sp.citesStatus})."
            )
        if indicators_list:
            explanation_parts.append(
                f"Triggered indicators: {', '.join(i.indicator for i in indicators_list[:2])}."
            )
        explanation_parts.append("Forwarded for priority human enforcement review.")
    elif prediction == "REQUIRES MANUAL REVIEW":
        explanation_parts.append(
            f"Moderate risk signals detected (fused score: {fused_score:.2f})."
        )
        if species_list or indicators_list:
            explanation_parts.append(
                "Content contains wildlife terms or commercial solicitation requiring review."
            )
        else:
            explanation_parts.append("Ambiguous content flagged for verification.")
    else:
        explanation_parts.append(
            "Automated multimodal analysis detected no clear evidence of illicit wildlife trade "
            f"(fused score: {fused_score:.2f})."
        )
        if benign_hits > 0:
            explanation_parts.append(
                "Context aligns with legitimate research, conservation, or domestic content."
            )

    explanation = " ".join(explanation_parts)

    return AnalysisResult(
        id=case_id,
        submissionDate=now_utc,
        inputType=resolved_input_type,
        inputSource=input_source,
        prediction=prediction,
        confidenceScore=round(fused_score, 4),
        identifiedSpecies=species_list,
        extractedText=text or "No text provided in submission",
        suspiciousKeywords=keywords,
        suspiciousIndicators=indicators_list,
        explanation=explanation,
        analysisStatus=analysis_status,
        threatSeverity=threat_severity,
        imageUrl=image_b64,
        isSample=False,
    )


@router.post("/analyze/upload", response_model=AnalysisResult)
async def analyze_upload(
    input_type: Annotated[str, Form()] = "screenshot",
    input_source: Annotated[str, Form()] = "investigator_upload",
    text_content: Annotated[Optional[str], Form()] = None,
    image: Annotated[Optional[UploadFile], File()] = None,
) -> AnalysisResult:
    """Multimodal analysis endpoint supporting direct file uploads."""
    image_b64: Optional[str] = None
    if image is not None:
        contents = await image.read()
        b64_str = base64.b64encode(contents).decode("utf-8")
        content_type = image.content_type or "image/jpeg"
        image_b64 = f"data:{content_type};base64,{b64_str}"

    payload = AnalyzeRequest(
        text=text_content,
        image_base64=image_b64,
        inputType="screenshot" if input_type == "screenshot" else "url",
        inputSource=input_source,
    )
    return analyze(payload)
