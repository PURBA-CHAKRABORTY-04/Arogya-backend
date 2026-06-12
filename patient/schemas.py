# patient/schemas.py
from datetime import date
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, computed_field


# ── CREATE ────────────────────────────────────────────────────────────────────
# Used by POST /patient/patient-info.
# dob and blood_group are accepted here — set once, never changed via PUT.

class PatientInfoCreate(BaseModel):
    email:          EmailStr
    name:           str  = Field(..., min_length=1, max_length=255)
    dob:            date                               # uneditable after creation
    contact_number: str  = Field(..., max_length=20)
    gender:         str  = Field(..., min_length=1)
    city:           str  = Field(..., min_length=1, max_length=100)
    blood_group:    Optional[str]   = Field(None, max_length=5)  # uneditable after creation

    height_cm: Optional[float] = Field(None, gt=0)
    weight_kg: Optional[float] = Field(None, gt=0)

    chronic_conditions: Optional[List[str]] = None
    known_allergies:    Optional[List[str]] = None
    active_medications: Optional[List[str]] = None

    smoking_status:      Optional[str] = None  # "Non-Smoker" | "Tobacco User"
    alcohol_consumption: Optional[str] = None  # "Alcohol-Free" | "Consumes Alcohol"

    emergency_contact_name:         Optional[str] = Field(None, max_length=255)
    emergency_contact_phone:        Optional[str] = Field(None, max_length=20)
    emergency_contact_relationship: Optional[str] = Field(None, max_length=100)


# ── UPDATE ────────────────────────────────────────────────────────────────────
# Used by PUT /patient/patient-info/{email}.
# dob, blood_group, and email are intentionally EXCLUDED — they are immutable.

class PatientInfoUpdate(BaseModel):
    name:           Optional[str]   = Field(None, min_length=1, max_length=255)
    contact_number: Optional[str]   = Field(None, max_length=20)
    gender:         Optional[str]   = None
    city:           Optional[str]   = Field(None, min_length=1, max_length=100)

    height_cm: Optional[float] = Field(None, gt=0)
    weight_kg: Optional[float] = Field(None, gt=0)

    chronic_conditions: Optional[List[str]] = None
    known_allergies:    Optional[List[str]] = None
    active_medications: Optional[List[str]] = None

    smoking_status:      Optional[str] = None
    alcohol_consumption: Optional[str] = None

    emergency_contact_name:         Optional[str] = Field(None, max_length=255)
    emergency_contact_phone:        Optional[str] = Field(None, max_length=20)
    emergency_contact_relationship: Optional[str] = Field(None, max_length=100)


# ── RESPONSE ──────────────────────────────────────────────────────────────────
# Returned by GET, POST, and PUT.
# age and bmi are @computed_field — calculated fresh on every response, never stored.

class PatientInfoResponse(BaseModel):
    id:             UUID
    email:          EmailStr
    name:           str
    dob:            date
    contact_number: str
    gender:         str
    city:           str
    blood_group:    Optional[str]   = None

    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    chronic_conditions: Optional[List[str]] = None
    known_allergies:    Optional[List[str]] = None
    active_medications: Optional[List[str]] = None

    smoking_status:      Optional[str] = None
    alcohol_consumption: Optional[str] = None

    emergency_contact_name:         Optional[str] = None
    emergency_contact_phone:        Optional[str] = None
    emergency_contact_relationship: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def age(self) -> int:
        """Age in full years, calculated from dob."""
        today = date.today()
        return (
            today.year - self.dob.year
            - ((today.month, today.day) < (self.dob.month, self.dob.day))
        )

    @computed_field  # type: ignore[misc]
    @property
    def bmi(self) -> Optional[float]:
        """BMI = weight_kg / (height_m)^2, rounded to 1 decimal. None if either value is missing."""
        if self.height_cm and self.weight_kg and self.height_cm > 0:
            return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)
        return None

    model_config = ConfigDict(from_attributes=True)
