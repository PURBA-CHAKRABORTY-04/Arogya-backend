# patient/router.py
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.models import User
from patient.models import PatientInfo
from patient.schemas import PatientInfoCreate, PatientInfoResponse, PatientInfoUpdate
from patient.timeline_models import TimelineEntry
from patient.timeline_schemas import TimelineEntryCreate, TimelineEntryResponse


# All routes in this file start with /patient and appear under the Patient Info tag in /docs.
router = APIRouter(prefix="/patient", tags=["Patient Info"])


# ── RBAC helper ───────────────────────────────────────────────────────────────

def _require_patient_role(email: str, db: Session) -> User:
    """
    Fetch the user by email and raise 403 if their account role is not 'patient'.
    Also raises 404 if the email is not registered at all.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No registered user found with this email.",
        )
    if user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patient accounts can have a patient profile.",
        )
    return user


# ── Create ────────────────────────────────────────────────────────────────────

@router.post(
    "/patient-info",
    response_model=PatientInfoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new patient profile",
)
def create_patient_info(payload: PatientInfoCreate, db: Session = Depends(get_db)):
    """
    Create a new PatientInfo record for an existing **patient** user.

    - Returns **404** if the email is not found in the users table.
    - Returns **403** if the account role is not ``patient``.
    - Returns **409** if a profile already exists for this email.
    """
    # Guard: email must exist and belong to a patient account.
    _require_patient_role(payload.email, db)

    # Guard: each user may only have one patient profile.
    existing = db.query(PatientInfo).filter(PatientInfo.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A patient profile already exists for this email.",
        )

    # Build and persist the new profile.
    profile = PatientInfo(**payload.model_dump())
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


# ── Read ──────────────────────────────────────────────────────────────────────

@router.get(
    "/patient-info/{email}",
    response_model=PatientInfoResponse,
    summary="Fetch a patient profile by email",
)
def get_patient_info(email: str, db: Session = Depends(get_db)):
    """
    Retrieve the PatientInfo record for the given email.

    **Note:** Email addresses must be percent-encoded in the URL path
    (e.g. ``user%40example.com``). The server decodes them automatically.

    - Returns **404** if no profile exists for this email.
    - Returns **403** if the account role is not ``patient``.
    """
    email = unquote(email)

    # Guard: only patient-role accounts have profiles.
    _require_patient_role(email, db)

    profile = db.query(PatientInfo).filter(PatientInfo.email == email).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patient profile found for this email.",
        )
    return profile


# ── Update ────────────────────────────────────────────────────────────────────

@router.put(
    "/patient-info/{email}",
    response_model=PatientInfoResponse,
    summary="Update editable fields of a patient profile",
)
def update_patient_info(
    email: str,
    payload: PatientInfoUpdate,
    db: Session = Depends(get_db),
):
    """
    Partially update the PatientInfo record for the given email.

    Immutable fields (``dob``, ``blood_group``, ``email``, ``name``, ``gender``)
    cannot be changed via this endpoint — they are excluded from the request schema.

    **Note:** Email addresses must be percent-encoded in the URL path.

    - Returns **404** if no profile exists for this email.
    - Returns **403** if the account role is not ``patient``.
    """
    email = unquote(email)

    # Guard: only patient-role accounts own profiles.
    _require_patient_role(email, db)

    profile = db.query(PatientInfo).filter(PatientInfo.email == email).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patient profile found for this email.",
        )

    # Apply only the fields that were explicitly sent (exclude_unset skips None defaults).
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(profile, field, value)

    db.commit()
    db.refresh(profile)
    return profile


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete(
    "/patient-info/{email}",
    status_code=status.HTTP_200_OK,
    summary="Delete a patient profile",
)
def delete_patient_info(email: str, db: Session = Depends(get_db)):
    """
    Permanently delete the PatientInfo record for the given email.

    **Note:** Email addresses must be percent-encoded in the URL path.

    - Returns **404** if no profile exists for this email.
    - Returns **403** if the account role is not ``patient``.
    """
    email = unquote(email)

    # Guard: only patient-role accounts own profiles.
    _require_patient_role(email, db)

    profile = db.query(PatientInfo).filter(PatientInfo.email == email).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patient profile found for this email.",
        )

    db.delete(profile)
    db.commit()
    return {"message": f"Patient profile for {email} deleted successfully."}


# ══════════════════════════════════════════════════════════════════════════════
# Health Timeline Entries
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/timeline/{email}",
    response_model=list[TimelineEntryResponse],
    summary="Fetch all health timeline entries for a patient",
)
def get_timeline(email: str, db: Session = Depends(get_db)):
    """
    Returns all timeline entries for the given patient email,
    ordered newest-first.

    - Returns **403** if the account is not a patient role.
    - Returns **404** if the user does not exist.
    """
    email = unquote(email)
    _require_patient_role(email, db)

    entries = (
        db.query(TimelineEntry)
        .filter(TimelineEntry.patient_email == email)
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
):
    """
    Creates a new timeline entry (symptom log, document upload record,
    voice triage summary, etc.) for the given patient.

    - Returns **403** if the account is not a patient role.
    - Returns **404** if the user does not exist.
    """
    _require_patient_role(payload.patient_email, db)

    entry = TimelineEntry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.delete(
    "/timeline/{entry_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single health timeline entry",
)
def delete_timeline_entry(entry_id: str, db: Session = Depends(get_db)):
    """
    Permanently removes the timeline entry with the given UUID.

    - Returns **404** if the entry does not exist.
    """
    entry = db.query(TimelineEntry).filter(TimelineEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Timeline entry not found.",
        )
    db.delete(entry)
    db.commit()
    return {"message": "Timeline entry deleted."}
