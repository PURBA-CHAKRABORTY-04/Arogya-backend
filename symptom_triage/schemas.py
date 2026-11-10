import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., max_length=20)  # 'user', 'assistant', 'system'
    content: str = Field(..., min_length=1, max_length=2000, description="Individual message content capped at 2000 characters to prevent token-bombing")


class SymptomTriageRequest(BaseModel):
    patient_email: Optional[str] = None
    session_id: Optional[str] = None
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=50, description="Chat history capped at 50 messages")


class TriageResponse(BaseModel):
    message: str
    chips: Optional[List[str]] = None
    isComplete: bool
    session_id: Optional[str] = None


class TriageSessionResponse(BaseModel):
    id: uuid.UUID
    patient_email: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Intelligence pipeline — POST /symptom_triage/route
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """Input for the 5-stage specialty routing engine."""
    complaint: str                              # free-form patient text (required)
    age: Optional[str] = None
    gender: Optional[str] = None
    history: Optional[str] = None
    medications: Optional[str] = None
    vitals: Optional[str] = None
    duration: Optional[str] = None
    pregnancy_status: Optional[str] = None
    mental_state: Optional[str] = None


class SpecialtyItem(BaseModel):
    name: str
    priority: int
    confidence: float


class RouteResponse(BaseModel):
    """Output of the 5-stage specialty routing engine (schema v2.0)."""
    schema_version: str
    emergency: bool
    urgency: str                                # "emergency" | "urgent" | "routine"
    specialties: List[SpecialtyItem]
    missing_info: List[str]
    pipeline_success: bool
    stage_repairs: Dict[str, List[str]]         # stage name → list of repair notes


# ---------------------------------------------------------------------------
# Unified 3-stage Chat Pipeline — POST /symptom_triage/chat
# ---------------------------------------------------------------------------

class MatchedDoctorItem(BaseModel):
    name: str
    role: Optional[str] = None
    qualification: Optional[str] = None
    location: Optional[str] = None
    match_score: Optional[int] = 85
    consultation_fee: Optional[float] = 500.0


class ChatTriageRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    complaint: Optional[str] = None
    story: Optional[str] = None
    session_id: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    age: Optional[Any] = None
    gender: Optional[str] = None


class ChatTriageResponse(BaseModel):
    message: str
    reply: str
    keywords: List[str] = Field(default_factory=list)
    negated_keywords: List[str] = Field(default_factory=list)
    tuples: Dict[str, Any] = Field(default_factory=dict)
    emergency: bool = False
    is_emergency: bool = False
    urgency: str = "routine"
    red_flags: List[str] = Field(default_factory=list)
    specialties: List[str] = Field(default_factory=list)
    specialty_details: List[SpecialtyItem] = Field(default_factory=list)
    doctors: List[MatchedDoctorItem] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    chips: List[str] = Field(default_factory=list)
    is_final: bool = True
    session_id: Optional[str] = None

