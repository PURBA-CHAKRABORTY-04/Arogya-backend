# patient/timeline_models.py
# SQLAlchemy ORM model for the health_timeline_entries table.

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID

from auth.database import Base


class TimelineEntry(Base):
    """One row per health event logged by a patient.

    Linked to users.email via a CASCADE-delete foreign key so that
    all timeline entries are automatically removed when the user account
    is deleted.
    """

    __tablename__ = "health_timeline_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # FK to users table — cascade delete when the account is removed
    patient_email = Column(
        String(320),
        ForeignKey("users.email", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Entry metadata
    title    = Column(String(500),  nullable=False)
    details  = Column(Text,         nullable=True)
    # 'symptom' | 'document' | 'consult' | 'voice_triage'
    type     = Column(String(50),   nullable=False, default="symptom")

    # Optional URL for document entries (Supabase public URL)
    url      = Column(Text,         nullable=True)

    # Optional severity for symptom entries (1–10)
    severity = Column(Integer,      nullable=True)

    # Managed by DB
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
