from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class DoctorMatchRequest(BaseModel):
    patient_email: Optional[str] = None
    symptoms: Optional[str] = None
    location: Optional[str] = "All"
    max_price: Optional[float] = None
    # Optional specialty hint from completed triage session — anchors LLM matching
    recommended_specialty: Optional[str] = None


class DoctorResponse(BaseModel):
    id: UUID
    email: Optional[str] = None
    name: str
    role: str
    qualification: Optional[str] = ""
    experience_years: Optional[float] = None
    city: Optional[str] = "Kolkata"
    consultation_fee: Optional[float] = 0.0
    photo_url: Optional[str] = None
    description: Optional[str] = None
    verified: Optional[bool] = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DoctorMatchResponseItem(BaseModel):
    doctor: DoctorResponse
    match_score: int          # percentage from 0 to 100
    match_reason: str         # justification text from LLM / match criteria


class DoctorMatchResponse(BaseModel):
    matches: List[DoctorMatchResponseItem]
