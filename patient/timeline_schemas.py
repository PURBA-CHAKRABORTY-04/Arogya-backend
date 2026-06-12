# patient/timeline_schemas.py
# Pydantic schemas for the health timeline entries API.

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── CREATE ────────────────────────────────────────────────────────────────────
# Used by POST /patient/timeline

class TimelineEntryCreate(BaseModel):
    patient_email: EmailStr
    title:    str             = Field(..., min_length=1, max_length=500)
    details:  Optional[str]  = None
    type:     str             = Field(default="symptom", max_length=50)
    url:      Optional[str]  = None        # Supabase public URL for documents
    severity: Optional[int]  = Field(None, ge=1, le=10)  # 1–10 for symptoms


# ── RESPONSE ──────────────────────────────────────────────────────────────────
# Returned by GET and POST

class TimelineEntryResponse(BaseModel):
    id:            UUID
    patient_email: EmailStr
    title:         str
    details:       Optional[str]  = None
    type:          str
    url:           Optional[str]  = None
    severity:      Optional[int]  = None
    created_at:    datetime

    model_config = ConfigDict(from_attributes=True)
