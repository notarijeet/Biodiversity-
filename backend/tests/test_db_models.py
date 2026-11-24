from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Case, CaseAuditLog, IdentifiedSpecies, SuspiciousIndicator


def test_database_models_and_relationships():
    # In-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    db = session_factory()

    try:
        new_case = Case(
            case_number="CASE-TEST-0001",
            submission_date="2026-11-18 20:00 UTC",
            input_type="screenshot",
            input_source="test_source",
            prediction="POTENTIAL ILLEGAL WILDLIFE TRADE",
            confidence_score=0.92,
            threat_severity="Critical",
            analysis_status="FLAGGED_FOR_HUMAN_ENFORCEMENT",
            extracted_text="Pangolin scales for sale",
            explanation="Test explanation",
        )
        db.add(new_case)
        db.flush()

        species = IdentifiedSpecies(
            case_id=new_case.id,
            common_name="Sunda Pangolin",
            scientific_name="Manis javanica",
            cites_status="CITES Appendix I",
            conservation_status="Critically Endangered",
        )
        indicator = SuspiciousIndicator(
            case_id=new_case.id,
            indicator="Commercial Bulk Scale Trade",
            severity="high",
            description="Bulk scales offered without permits",
        )
        audit_log = CaseAuditLog(
            case_id=new_case.id,
            action="CREATED",
            actor="SYSTEM_AI",
            notes="Case created via automated detection pipeline",
        )

        db.add_all([species, indicator, audit_log])
        db.commit()

        # Query back
        fetched_case = db.query(Case).filter_by(case_number="CASE-TEST-0001").first()
        assert fetched_case is not None
        assert len(fetched_case.species) == 1
        assert fetched_case.species[0].common_name == "Sunda Pangolin"
        assert len(fetched_case.indicators) == 1
        assert len(fetched_case.audit_logs) == 1
        assert fetched_case.audit_logs[0].action == "CREATED"

    finally:
        db.close()
