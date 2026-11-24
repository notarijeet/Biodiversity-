import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.schemas import AnalysisResult
from main import app
from models.fusion import classify_fused_score, fuse_scores

client = TestClient(app)


def test_analysis_result_keys_match_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "analysis_result_keys.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        expected_keys = set(json.load(f))

    actual_keys = set(AnalysisResult.model_fields.keys())
    assert actual_keys == expected_keys, f"Schema mismatch: diff={actual_keys ^ expected_keys}"


def test_analyze_endpoint_contract():
    payload = {
        "text": "Offering carved ivory tusks for sale. Contact via private message.",
        "inputType": "screenshot",
        "inputSource": "test_submission",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    fixture_path = Path(__file__).parent / "fixtures" / "analysis_result_keys.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        expected_keys = set(json.load(f))

    assert set(data.keys()) == expected_keys
    assert data["inputType"] in ["screenshot", "url"]
    assert data["prediction"] in [
        "POTENTIAL ILLEGAL WILDLIFE TRADE",
        "NO CLEAR EVIDENCE DETECTED",
        "REQUIRES MANUAL REVIEW",
    ]


def test_fusion_scoring_and_classification():
    # Both text and image
    score = fuse_scores(text_prob=0.8, image_confidence=0.7)
    assert round(score, 2) == 0.76
    pred, sev, status = classify_fused_score(score)
    assert pred == "POTENTIAL ILLEGAL WILDLIFE TRADE"
    assert sev == "Critical"
    assert status == "FLAGGED_FOR_HUMAN_ENFORCEMENT"

    # Low score
    low_score = fuse_scores(text_prob=0.1, image_confidence=0.2)
    assert round(low_score, 2) == 0.14
    pred, sev, status = classify_fused_score(low_score)
    assert pred == "NO CLEAR EVIDENCE DETECTED"
    assert sev == "Negligible"
    assert status == "CLEARED"
