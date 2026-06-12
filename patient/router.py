# patient/router.py
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.models import User
from patient.models import PatientInfo
from patient.schemas import PatientInfoCreate, PatientInfoResponse, PatientInfoUpdate


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
