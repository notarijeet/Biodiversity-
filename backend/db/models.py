import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from db.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Case(Base):
    __tablename__ = "cases"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_number = Column(String(64), unique=True, index=True, nullable=False)
    submission_date = Column(String(64), nullable=False)
    input_type = Column(String(32), nullable=False)  # 'screenshot' | 'url'
    input_source = Column(String(255), nullable=False)
    prediction = Column(String(64), nullable=False)
    confidence_score = Column(Float, nullable=False)
    threat_severity = Column(String(32), nullable=False)
    analysis_status = Column(String(64), nullable=False)
    extracted_text = Column(Text, nullable=True)
    image_url = Column(String(1024), nullable=True)
    explanation = Column(Text, nullable=False)
    is_sample = Column(Boolean, default=False)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    species = relationship("IdentifiedSpecies", back_populates="case", cascade="all, delete-orphan")
    indicators = relationship(
        "SuspiciousIndicator", back_populates="case", cascade="all, delete-orphan"
    )
    audit_logs = relationship("CaseAuditLog", back_populates="case", cascade="all, delete-orphan")


class IdentifiedSpecies(Base):
    __tablename__ = "identified_species"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(
        String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    common_name = Column(String(255), nullable=False)
    scientific_name = Column(String(255), nullable=False)
    cites_status = Column(String(255), nullable=False)
    conservation_status = Column(String(128), nullable=False)

    case = relationship("Case", back_populates="species")


class SuspiciousIndicator(Base):
    __tablename__ = "suspicious_indicators"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(
        String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    indicator = Column(String(255), nullable=False)
    severity = Column(String(16), nullable=False)  # 'high' | 'medium' | 'low'
    description = Column(Text, nullable=False)

    case = relationship("Case", back_populates="indicators")


class CaseAuditLog(Base):
    __tablename__ = "case_audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(
        String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action = Column(String(64), nullable=False)
    actor = Column(String(128), nullable=False)
    notes = Column(Text, nullable=True)
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    case = relationship("Case", back_populates="audit_logs")
