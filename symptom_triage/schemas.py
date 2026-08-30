import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from doctor_matching.schemas import DoctorMatchResponseItem


class ChatMessage(BaseModel):
    role: str  # 'user', 'assistant', 'system'
    content: str


class SymptomTriageRequest(BaseModel):
    patient_email: str
    session_id: Optional[str] = None
    messages: List[ChatMessage]
    # Optional: custom triage rules supplied by the patient/operator.
    # Set on the first message; stored in the session and reused automatically thereafter.
    triage_rules: Optional[str] = None


class TriageResponse(BaseModel):
    message: str
    chips: Optional[List[str]] = None
    isComplete: bool
    session_id: Optional[str] = None
    # ── Populated only when isComplete=True ──
    urgency_level: Optional[str] = None          # "emergency" | "urgent" | "semi-urgent" | "routine"
    recommended_specialty: Optional[str] = None  # e.g. "Cardiologist", "General Physician"
    triage_summary: Optional[str] = None         # 2-3 sentence clinical summary
    matched_doctors: Optional[List[DoctorMatchResponseItem]] = None  # Top Kolkata doctors matched to condition
    # ── Debug / transparency field: the exact system prompt sent to the LLM ──
    system_prompt: Optional[str] = None


class TriageSessionResponse(BaseModel):
    id: uuid.UUID
    patient_email: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TriageReportResponse(BaseModel):
    """Returned by GET /sessions/{id}/report once a session is complete."""
    session_id: uuid.UUID
    title: Optional[str]
    urgency_level: Optional[str]
    recommended_specialty: Optional[str]
    triage_summary: Optional[str]
    matched_doctors: Optional[List[DoctorMatchResponseItem]] = None
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True

