import json

from fastapi.testclient import TestClient

from config import settings
from main import app
from models.taxonomy import match_indicators, match_species

client = TestClient(app)


def test_india_codewords_file_structure():
    file_path = settings.data_dir / "lexicons" / "india_codewords.json"
    assert file_path.exists(), f"Missing file: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Acceptance criteria: at least 10 entries
    assert len(data) >= 10, f"Expected at least 10 entries, found {len(data)}"

    # Required fields per entry
    for codeword, meta in data.items():
        assert "species" in meta, f"Missing species in {codeword}"
        assert "scientific" in meta, f"Missing scientific in {codeword}"
        assert "language" in meta, f"Missing language in {codeword}"
        assert "cites" in meta, f"Missing cites in {codeword}"


def test_taxonomy_codeword_matching():
    text = "selling original hathajodi and khurpa for prosperity rituals"
    species = match_species(text)
    indicators, keywords, _ = match_indicators(text)

    # Must detect both species and codeword triggers
    assert any("Monitor Lizard" in s.commonName for s in species)
    assert any("Pangolin" in s.commonName for s in species)
    assert "hathajodi" in keywords
    assert "khurpa" in keywords
    assert any(i.indicator == "Indian Trade Codeword Detected" for i in indicators)


def test_acceptance_criteria_hathajodi_api():
    """Acceptance criterion 1: 'hathajodi for sale dm me' -> positive classification"""
    payload = {
        "text": "hathajodi for sale dm me",
        "inputType": "screenshot",
        "inputSource": "test_verification",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == "POTENTIAL ILLEGAL WILDLIFE TRADE"
    assert data["threatSeverity"] in ["Critical", "Elevated"]
    assert data["analysisStatus"] == "FLAGGED_FOR_HUMAN_ENFORCEMENT"
    assert any("Monitor Lizard" in s["commonName"] for s in data["identifiedSpecies"])
    assert "hathajodi" in data["suspiciousKeywords"]


def test_acceptance_criteria_safed_sona_api():
    """Acceptance criterion 2: 'safed sona available contact telegram' -> positive classification"""
    payload = {
        "text": "safed sona available contact telegram",
        "inputType": "url",
        "inputSource": "test_verification",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == "POTENTIAL ILLEGAL WILDLIFE TRADE"
    assert data["threatSeverity"] in ["Critical", "Elevated"]
    assert data["analysisStatus"] == "FLAGGED_FOR_HUMAN_ENFORCEMENT"
    assert any("Elephant" in s["commonName"] for s in data["identifiedSpecies"])
    assert "safed sona" in data["suspiciousKeywords"]
