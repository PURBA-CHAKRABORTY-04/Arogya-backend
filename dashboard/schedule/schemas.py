# dashboard/schedule/schemas.py
# Pydantic schemas for the Schedule Events API.
#
# category and times_a_day are stored as plain strings so they accept
# whatever the frontend sends (e.g. "medication", "Medication", "consultation").
# The validator only checks that times_a_day is present when category is Medication.

from datetime import date, datetime, time
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


# ── CREATE ────────────────────────────────────────────────────────────────────
# Used by POST /schedule/

class ScheduleEventCreate(BaseModel):
    patient_email: EmailStr
    event_title:   str  = Field(..., min_length=1, max_length=500)
    event_time:    time                                  # "HH:MM" or "HH:MM:SS"
    # Accepts frontend lowercase values: "medication" | "consultation" | "lab" | "exercise" | "other"
    category:      str  = Field(..., max_length=50)

    # Medication-only fields
    till_date:        Optional[date] = None
    times_a_day:      Optional[str]  = Field(None, max_length=50)
    relation_to_meal: Optional[str]  = Field(None, max_length=30)

    # Non-medication categories (Doctor Visit / Lab / Exercise / Other)
    details_notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_category_fields(self) -> "ScheduleEventCreate":
        """times_a_day is required for Medication events (any case variant)."""
        if self.category.lower() == "medication":
            if not self.times_a_day:
                raise ValueError(
                    "'times_a_day' is required when category is 'Medication'."
                )
        return self


# ── UPDATE ────────────────────────────────────────────────────────────────────
# Used by PUT /schedule/{event_id} — all fields optional.

class ScheduleEventUpdate(BaseModel):
    event_title:      Optional[str]  = Field(None, min_length=1, max_length=500)
    event_time:       Optional[time] = None
    category:         Optional[str]  = Field(None, max_length=50)

    # Medication-only
    till_date:        Optional[date] = None
    times_a_day:      Optional[str]  = Field(None, max_length=50)
    relation_to_meal: Optional[str]  = Field(None, max_length=30)

    # Non-medication
    details_notes: Optional[str] = None


# ── RESPONSE ──────────────────────────────────────────────────────────────────
# Returned by GET, POST, and PUT.

class ScheduleEventResponse(BaseModel):
    id:            UUID
    patient_email: EmailStr
    event_title:   str
    event_time:    time
    category:      str
    created_at:    datetime

    # Medication-only (None for other categories)
    till_date:        Optional[date] = None
    times_a_day:      Optional[str]  = None
    relation_to_meal: Optional[str]  = None

    # Non-medication (None for Medication category)
    details_notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
