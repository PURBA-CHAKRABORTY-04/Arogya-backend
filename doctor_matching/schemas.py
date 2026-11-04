from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime


class LocationPayload(BaseModel):
    """Payload for patient location submission and session storage."""
    session_id: str = Field(..., description="Unique session ID to persist location across searches")
    lat: Optional[float] = None
    lng: Optional[float] = None
    pincode: Optional[str] = None
    address: Optional[str] = None


class LocationResponse(BaseModel):
    session_id: str
    status: str
    saved_location: Dict[str, Any]


class DoctorMatchRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID retaining location state")
    patient_email: Optional[str] = None
    symptoms: Optional[str] = None
    location: Optional[str] = "All"  # manual city override or "All"
    patient_location: Optional[LocationPayload] = None
    max_price: Optional[float] = None
    recommended_specialty: Optional[str] = None
    
    # Hard Filters
    insurance_filter: Optional[str] = None        # e.g., "Star Health", "HDFC ERGO"
    max_distance_km: Optional[float] = None         # Max radius limit in kilometers
    availability_only: Optional[bool] = False       # Filter only available doctors

    # Pagination (Issue #15)
    limit: Optional[int] = Field(50, ge=1, le=100)
    offset: Optional[int] = Field(0, ge=0)


class ScoreBreakdownSchema(BaseModel):
    raw_score: float
    final_score: int
    base_score: float
    d_score: float
    distance_km: Optional[float] = None
    distance_score: float
    distance_weighted: float
    e_bonus: float
    fee_score: float
    fee_weighted: float
    s_spec: float
    s_symp: float
    s_combined: float


class DoctorResponse(BaseModel):
    id: UUID
    email: Optional[str] = None
    name: str
    role: str
    qualification: Optional[str] = ""
    experience_years: Optional[float] = None
    city: Optional[str] = "Bangalore"
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    consultation_fee: Optional[float] = 0.0
    photo_url: Optional[str] = None
    description: Optional[str] = None
    verified: Optional[bool] = True
    is_available: Optional[bool] = True
    insurance_accepted: Optional[str] = None
    average_rating: Optional[float] = 4.5
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DoctorMatchResponseItem(BaseModel):
    doctor: DoctorResponse
    match_score: int          # percentage score from 0 to 100
    match_reason: str         # clinical justification
    score_breakdown: Optional[ScoreBreakdownSchema] = None


class DoctorMatchResponse(BaseModel):
    matches: List[DoctorMatchResponseItem]
    session_id: Optional[str] = None
    candidate_pool_count: int = 0
    limit: Optional[int] = 50
    offset: Optional[int] = 0


class InteractionLogRequest(BaseModel):
    session_id: str
    patient_email: Optional[str] = None
    search_query: Optional[str] = None
    doctor_id: UUID
    rank_position: Optional[float] = None
    raw_score: Optional[float] = None
    final_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, Any]] = None
    interaction_type: str = Field(..., description="'click' | 'booking' | 'impression'")


class InteractionLogResponse(BaseModel):
    status: str
    interaction_id: UUID


# ── Predictive ML Decision Matcher Schemas ────────────────────────────────────

class PredictiveMatchRequest(BaseModel):
    query_prompt: str = Field(..., description="User prompt query (e.g. 'Dr. Amitava Das' or 'Cardiologist for chest pain')")
    prompter_specialty: Optional[str] = Field(None, description="Inferred or requested clinical specialty")
    urgency_level: Optional[str] = Field("routine", description="'routine' | 'high' | 'critical'")
    limit: Optional[int] = Field(10, ge=1, le=50)


class PredictiveMatchItem(BaseModel):
    doctor_name: str
    specialization: str
    hospital_location: str
    educational_qualification: str
    match_score: int
    match_probability: float
    is_safe: bool
    harm_prevention_status: str
    why_reasons: List[str]
    cautions: List[str]
    model_breakdown: Dict[str, Any]


class PredictiveMatchResponse(BaseModel):
    query_prompt: str
    prompter_specialty: Optional[str]
    matches: List[PredictiveMatchItem]
    total_candidates_searched: int
    harm_prevention_enforced: bool = True
