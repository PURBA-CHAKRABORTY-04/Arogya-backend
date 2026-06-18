from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class ComplaintBase(BaseModel):
    category: str
    description: str

class ComplaintCreate(ComplaintBase):
    patient_email: EmailStr

class ComplaintResponse(ComplaintBase):
    id: str
    patient_email: EmailStr
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
