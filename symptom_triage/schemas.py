from typing import List, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str  # e.g., 'user' or 'model'
    content: str

class SymptomTriageRequest(BaseModel):
    patient_email: str
    messages: List[ChatMessage]

class TriageResponse(BaseModel):
    message: str
    chips: Optional[List[str]] = None
    isComplete: bool
