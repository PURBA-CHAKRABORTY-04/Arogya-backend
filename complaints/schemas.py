from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class ComplaintBase(BaseModel):
    category: str
    description: str
    doctor_reference: Optional[str] = None

class ComplaintCreate(ComplaintBase):
    patient_email: Optional[EmailStr] = None

class ComplaintResponse(ComplaintBase):
    id: str
    patient_email: EmailStr
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
