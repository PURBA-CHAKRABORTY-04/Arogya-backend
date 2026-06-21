# biometric/schemas.py
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class BiometricMetricResponse(BaseModel):
    id:             str
    patient_email:  str
    title:          str
    subtitle:       Optional[str] = None
    baseline:       Optional[str] = None
    current:        Optional[str] = None
    var_text:       Optional[str] = None
    var_type:       Optional[Literal["up", "down", "neutral"]] = "neutral"
    status:         Optional[Literal["Stable", "Alert", "Monitoring", "Improved"]] = "Monitoring"
    source_doc_url: Optional[str] = None
    created_at:     datetime
    updated_at:     datetime

    model_config = {"from_attributes": True}


class BiometricMetricUpdate(BaseModel):
    """Allows partial updates to any editable field."""
    subtitle:       Optional[str] = None
    baseline:       Optional[str] = None
    current:        Optional[str] = None
    var_text:       Optional[str] = None
    var_type:       Optional[Literal["up", "down", "neutral"]] = None
    status:         Optional[Literal["Stable", "Alert", "Monitoring", "Improved"]] = None
