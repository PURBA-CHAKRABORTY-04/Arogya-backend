# patient/router.py
"""
Patient profile and health timeline CRUD.

All endpoints require a valid Bearer JWT.
Identity is derived from the token — clients never send their own email in the URL.
Routes use /me to make it explicit that each patient only sees their own data.
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.dependencies import get_current_user, require_patient, require_doctor
from auth.models import User
from dashboard.patient.models import PatientInfo
from dashboard.patient.schemas import PatientInfoCreate, PatientInfoResponse, PatientInfoUpdate
from dashboard.patient.timeline_models import TimelineEntry
from dashboard.patient.timeline_schemas import TimelineEntryCreate, TimelineEntryResponse


router = APIRouter(prefix="/patient", tags=["Patient Info"])


class EmergencySosPayload(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    notes: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Patient Profile  (/patient/patient-info/me)
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/patient-info",
    response_model=PatientInfoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new patient profile",
)
def create_patient_info(
    payload: PatientInfoCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Create a new PatientInfo record for the authenticated patient.

    - Returns **403** if the account role is not ``patient``.
    - Returns **409** if a profile already exists.
    """
    # Always use the authenticated user's email — ignore any email in the payload
    email = current_user.email

    existing = db.query(PatientInfo).filter(PatientInfo.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A patient profile already exists for this account.",
        )

    data = payload.model_dump()
    data["email"] = email  # enforce server-side identity
    profile = PatientInfo(**data)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


@router.get(
    "/patient-info/me",
    response_model=PatientInfoResponse,
    summary="Fetch the authenticated patient's profile",
)
def get_patient_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Retrieve the PatientInfo record for the currently logged-in patient.

    - Returns **404** if no profile exists yet.
    """
    profile = db.query(PatientInfo).filter(PatientInfo.email == current_user.email).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patient profile found. Please complete the onboarding wizard.",
        )
    return profile


@router.put(
    "/patient-info/me",
    response_model=PatientInfoResponse,
    summary="Update the authenticated patient's profile",
)
def update_patient_info(
    payload: PatientInfoUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Partially update the authenticated patient's PatientInfo record.

    Immutable fields (``dob``, ``blood_group``, ``email``, ``name``, ``gender``)
    are excluded from the request schema and cannot be changed here.

    - Returns **404** if no profile exists.
    """
    profile = db.query(PatientInfo).filter(PatientInfo.email == current_user.email).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patient profile found.",
        )

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(profile, field, value)

    db.commit()
    db.refresh(profile)
    return profile


@router.delete(
    "/patient-info/me",
    status_code=status.HTTP_200_OK,
    summary="Delete the authenticated patient's profile",
)
def delete_patient_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Permanently delete the authenticated patient's PatientInfo record.

    - Returns **404** if no profile exists.
    """
    profile = db.query(PatientInfo).filter(PatientInfo.email == current_user.email).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patient profile found.",
        )

    db.delete(profile)
    db.commit()
    return {"message": "Patient profile deleted successfully."}


# ══════════════════════════════════════════════════════════════════════════════
# Health Timeline  (/patient/timeline/me)
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/timeline/me",
    response_model=list[TimelineEntryResponse],
    summary="Fetch all health timeline entries for the authenticated patient",
)
def get_timeline(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Returns all timeline entries for the authenticated patient, newest-first.
    """
    entries = (
        db.query(TimelineEntry)
        .filter(TimelineEntry.patient_email == current_user.email)
        .order_by(TimelineEntry.created_at.desc())
        .all()
    )
    return entries


@router.post(
    "/timeline",
    response_model=TimelineEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new health timeline entry",
)
def create_timeline_entry(
    payload: TimelineEntryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Creates a new timeline entry (symptom log, document upload record,
    voice triage summary, etc.) for the authenticated patient.
    The ``patient_email`` in the payload is overridden by the token's identity.
    """
    data = payload.model_dump()
    data["patient_email"] = current_user.email  # enforce server-side identity

    entry = TimelineEntry(**data)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.delete(
    "/timeline/{entry_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single health timeline entry",
)
def delete_timeline_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Permanently removes the timeline entry with the given UUID.
    Only the owning patient may delete their own entries.

    - Returns **404** if the entry does not exist.
    - Returns **403** if the entry does not belong to the caller.
    """
    entry = db.query(TimelineEntry).filter(TimelineEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Timeline entry not found.",
        )
    if entry.patient_email != current_user.email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this entry.",
        )
    db.delete(entry)
    db.commit()
    return {"message": "Timeline entry deleted."}


# ══════════════════════════════════════════════════════════════════════════════
# Emergency SOS & Clinical Queue
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/emergency-sos",
    response_model=TimelineEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record emergency SOS dispatch event",
)
def trigger_emergency_sos(
    payload: EmergencySosPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Issue #2: Real emergency SOS recording.
    Attaches GPS coordinates, sets critical severity (10), and adds to health timeline.
    """
    loc_parts = []
    if payload.latitude and payload.longitude:
        loc_parts.append(f"GPS: {payload.latitude:.5f}, {payload.longitude:.5f}")
    if payload.address:
        loc_parts.append(f"Address: {payload.address}")

    loc_str = f" [{' | '.join(loc_parts)}]" if loc_parts else ""
    details = f"🚨 EMERGENCY SOS DISPATCH TRIGGERED by patient.{loc_str}"
    if payload.notes:
        details += f" Notes: {payload.notes}"

    entry = TimelineEntry(
        patient_email=current_user.email,
        title="CRITICAL: Emergency SOS Dispatched",
        details=details,
        type="emergency",
        severity=10,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get(
    "/doctor-queue",
    summary="Fetch active clinical patient queue for doctors",
)
def get_doctor_patient_queue(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """
    Issue #12: Dynamic doctor queue endpoint.
    Retrieves real patient profiles and latest clinical status from Neon DB.
    """
    patients = db.query(PatientInfo).limit(20).all()
    queue = []
    for p in patients:
        latest_entry = (
            db.query(TimelineEntry)
            .filter(TimelineEntry.patient_email == p.email)
            .order_by(TimelineEntry.created_at.desc())
            .first()
        )
        is_sos = latest_entry and (latest_entry.severity == 10 or latest_entry.type == "emergency")
        queue.append({
            "email": p.email,
            "name": p.name,
            "gender": p.gender,
            "dob": p.dob.isoformat() if hasattr(p.dob, "isoformat") and p.dob else str(p.dob or ""),
            "blood_group": p.blood_group,
            "chronic_conditions": p.chronic_conditions or [],
            "known_allergies": p.known_allergies or [],
            "smoking_status": p.smoking_status,
            "alcohol_consumption": p.alcohol_consumption,
            "urgency": "CRITICAL" if is_sos else ("ELEVATED" if latest_entry and (latest_entry.severity or 0) >= 7 else "ROUTINE"),
            "latest_timeline": {
                "title": latest_entry.title,
                "details": latest_entry.details,
                "severity": latest_entry.severity,
                "created_at": latest_entry.created_at.isoformat() if latest_entry.created_at else None,
            } if latest_entry else None,
        })
    return queue
