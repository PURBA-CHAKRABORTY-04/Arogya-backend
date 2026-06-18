# symptom_triage/models.py
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from auth.database import Base


class TriageSession(Base):
    """ORM model mapped to the triage_sessions table in PostgreSQL.
    Stores the history of chat sessions for symptom triage.
    """

    __tablename__ = "triage_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # FK to users table — cascade delete when the account is removed
    patient_email = Column(
        String(320),
        ForeignKey("users.email", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Automatically generated title based on the first few messages
    title = Column(String(255), nullable=True)

    # The array of chat messages in OpenAI format: [{"role": "user", "content": "..."}]
    messages = Column(JSONB, nullable=False, default=list)

    # Audit timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
