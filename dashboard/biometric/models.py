# biometric/models.py
"""
SQLAlchemy model for patient biometric metrics extracted from uploaded documents.
Each row represents one health metric for one patient.
Upsert logic (match by patient_email + title) keeps the table clean — one row per metric.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text
from auth.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class BiometricMetric(Base):
    __tablename__ = "biometric_metrics"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Which patient this metric belongs to
    patient_email = Column(String, nullable=False, index=True)

    # Display info — used as the upsert key together with patient_email
    title    = Column(String, nullable=False)   # e.g. "Blood Glucose"
    subtitle = Column(String, nullable=True)    # e.g. "Fasting Plasma Glucose"

    # Values extracted from the document
    baseline = Column(String, nullable=True)    # e.g. "70–99 mg/dL"
    current  = Column(String, nullable=True)    # e.g. "126 mg/dL"
    var_text = Column(String, nullable=True)    # e.g. "27.3%"
    var_type = Column(String, nullable=True)    # "up" | "down" | "neutral"

    # Clinical interpretation
    status = Column(String, nullable=True)      # "Stable" | "Alert" | "Monitoring" | "Improved"

    # Source traceability
    source_doc_url = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
