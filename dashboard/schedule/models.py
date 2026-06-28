# dashboard/schedule/models.py
# SQLAlchemy ORM model for the schedule_events table.
#
# A single unified table stores all 5 event categories.
# Category-specific columns are nullable and only populated
# when the matching category is selected by the patient.
#
# Categories:
#   Medication        → till_date, times_a_day, relation_to_meal
#   Doctor Visit      → details_notes
#   Lab / Vitals Test → details_notes
#   Exercise/Physical → details_notes
#   Other Event       → details_notes

import uuid

from sqlalchemy import Column, Date, DateTime, ForeignKey, String, Text, Time, func
from sqlalchemy.dialects.postgresql import UUID

from auth.database import Base


class ScheduleEvent(Base):
    """One row per scheduled event created by a patient.

    Linked to users.email via a CASCADE-delete foreign key so that
    all schedule events are automatically removed when the user account
    is deleted.
    """

    __tablename__ = "schedule_events"

    # ── Primary key ───────────────────────────────────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ── Patient link — FK to users.email; cascade-deletes with account ────────
    patient_email = Column(
        String(320),
        ForeignKey("users.email", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Common fields (all categories) ────────────────────────────────────────
    event_title = Column(String(500), nullable=False)
    event_time  = Column(Time,        nullable=False)

    # category: "Medication" | "Doctor Visit" | "Lab / Vitals Test"
    #           | "Exercise / Physical" | "Other Event"
    category = Column(String(50), nullable=False)

    # ── Medication-only fields ─────────────────────────────────────────────────
    # till_date: the date until which the medicine must be taken
    till_date = Column(Date, nullable=True)

    # times_a_day: "Once daily" | "Twice daily" | "3x daily" | "4x daily" | "As needed"
    times_a_day = Column(String(50), nullable=True)

    # relation_to_meal: "Before meal" | "After meal" | "With meal" | "No relation"
    relation_to_meal = Column(String(30), nullable=True)

    # ── Non-medication categories (Doctor Visit / Lab / Exercise / Other) ──────
    # Free-text notes / details for the event
    details_notes = Column(Text, nullable=True)

    # ── Audit timestamp ────────────────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
