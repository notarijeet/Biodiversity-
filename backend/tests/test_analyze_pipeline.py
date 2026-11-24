import io

from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)


def test_analyze_suspicious_wildlife_text():
    payload = {
        "text": "authentic pangolin scales for sale in bulk, contact whatsapp for shipping",
        "inputType": "screenshot",
        "inputSource": "social_media_scan",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == "POTENTIAL ILLEGAL WILDLIFE TRADE"
    assert data["threatSeverity"] in ["Critical", "Elevated"]
    assert data["analysisStatus"] == "FLAGGED_FOR_HUMAN_ENFORCEMENT"
    assert len(data["identifiedSpecies"]) > 0
    assert any("Pangolin" in s["commonName"] for s in data["identifiedSpecies"])
    assert len(data["suspiciousIndicators"]) > 0


def test_analyze_benign_text():
    payload = {
        "text": "joining wwf rangers at sanctuary for camera trap conservation field research",
        "inputType": "url",
        "inputSource": "research_portal",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == "NO CLEAR EVIDENCE DETECTED"
    assert data["threatSeverity"] == "Negligible"
    assert data["analysisStatus"] == "CLEARED"


def test_analyze_upload_multipart():
    # Create tiny in-memory image
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"image": ("test.jpg", buf, "image/jpeg")}
    data = {
        "input_type": "screenshot",
        "input_source": "mobile_upload",
        "text_content": "camera trap research at national park sanctuary",
    }

    response = client.post("/analyze/upload", data=data, files=files)
    assert response.status_code == 200

    result = response.json()
    assert "id" in result
    assert result["inputType"] == "screenshot"
    assert result["inputSource"] == "mobile_upload"
    assert result["prediction"] in [
        "POTENTIAL ILLEGAL WILDLIFE TRADE",
        "NO CLEAR EVIDENCE DETECTED",
        "REQUIRES MANUAL REVIEW",
    ]


def test_analyze_empty_input():
    payload = {
        "inputType": "screenshot",
        "inputSource": "empty_test",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "NO CLEAR EVIDENCE DETECTED"
    assert data["confidenceScore"] == 0.0


def test_analyze_frontend_payload_format():
    """Verify that frontend input payload ({type, data, extractedText}) is seamlessly processed."""
    payload = {
        "type": "screenshot",
        "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "extractedText": "Original hathajodi charm root for puja. DM on whatsapp for price.",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["inputType"] == "screenshot"
    assert "hathajodi" in data["extractedText"].lower()
    assert data["prediction"] == "POTENTIAL ILLEGAL WILDLIFE TRADE"
    assert data["threatSeverity"] in ["Critical", "Elevated"]
    assert any("hathajodi" in s["commonName"].lower() for s in data["identifiedSpecies"])
