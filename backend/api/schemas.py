from typing import Literal, Optional

from pydantic import BaseModel, Field

DetectionPrediction = Literal[
    "POTENTIAL ILLEGAL WILDLIFE TRADE",
    "NO CLEAR EVIDENCE DETECTED",
    "REQUIRES MANUAL REVIEW",
]

ThreatSeverity = Literal["Critical", "Elevated", "Low", "Negligible"]
AnalysisStatus = Literal["COMPLETED", "FLAGGED_FOR_HUMAN_ENFORCEMENT", "CLEARED"]
IndicatorSeverity = Literal["high", "medium", "low"]
InputType = Literal["screenshot", "url"]


class IdentifiedSpeciesSchema(BaseModel):
    commonName: str
    scientificName: str
    citesStatus: str
    conservationStatus: str


class SuspiciousIndicatorSchema(BaseModel):
    indicator: str
    severity: IndicatorSeverity
    description: str


class AnalysisResult(BaseModel):
    id: str
    submissionDate: str
    inputType: InputType
    inputSource: str
    prediction: DetectionPrediction
    confidenceScore: float = Field(ge=0.0, le=1.0)
    identifiedSpecies: list[IdentifiedSpeciesSchema] = []
    extractedText: str
    suspiciousKeywords: list[str] = []
    suspiciousIndicators: list[SuspiciousIndicatorSchema] = []
    explanation: str
    analysisStatus: AnalysisStatus
    threatSeverity: ThreatSeverity
    imageUrl: Optional[str] = None
    isSample: Optional[bool] = False


class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    text_content: Optional[str] = None
    extractedText: Optional[str] = None
    url: Optional[str] = None
    data: Optional[str] = None
    image_base64: Optional[str] = None
    type: Optional[InputType] = None
    inputType: Optional[InputType] = "screenshot"
    inputSource: Optional[str] = "investigator_submission"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
