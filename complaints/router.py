"""
complaints/router.py
---------------------
Patient complaint/feedback submission and history.

All endpoints require a valid Bearer JWT.
Identity is derived from the token — /me replaces /{patient_email}.
"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.dependencies import get_current_user, require_patient
from auth.models import User
from complaints.models import Complaint
from complaints.schemas import ComplaintCreate, ComplaintResponse
from complaints.email_utils import send_complaint_email

router = APIRouter()


@router.post("/", response_model=ComplaintResponse)
def create_complaint(
    complaint_data: ComplaintCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Submit a new complaint or feedback ticket.
    The patient_email in the payload is overridden by the token's identity.
    """
    new_complaint = Complaint(
        patient_email=current_user.email,  # enforce server-side identity
        category=complaint_data.category,
        description=complaint_data.description,
        status="Submitted",
    )
    db.add(new_complaint)
    db.commit()
    db.refresh(new_complaint)

    background_tasks.add_task(
        send_complaint_email,
        email=current_user.email,
        category=new_complaint.category,
        description=new_complaint.description,
        name=current_user.full_name or None,
    )

    return new_complaint


@router.get("/me", response_model=List[ComplaintResponse])
def get_complaints(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Fetch all complaint tickets for the authenticated patient, newest-first.
    """
    complaints = (
        db.query(Complaint)
        .filter(Complaint.patient_email == current_user.email)
        .order_by(Complaint.created_at.desc())
        .all()
    )
    return complaints
