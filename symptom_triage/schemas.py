import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # 'user', 'assistant', 'system'
    content: str


class SymptomTriageRequest(BaseModel):
    patient_email: str
    session_id: Optional[str] = None
    messages: List[ChatMessage]


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
