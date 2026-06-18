from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from auth.database import get_db
from auth.models import User
from complaints.models import Complaint
from complaints.schemas import ComplaintCreate, ComplaintResponse
from complaints.email_utils import send_complaint_email

router = APIRouter()

@router.post("/", response_model=ComplaintResponse)
def create_complaint(
    complaint_data: ComplaintCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Verify patient exists
    user = db.query(User).filter(User.email == complaint_data.patient_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Patient not found")

    new_complaint = Complaint(
        patient_email=complaint_data.patient_email,
        category=complaint_data.category,
        description=complaint_data.description,
        status="Submitted"
    )
    db.add(new_complaint)
    db.commit()
    db.refresh(new_complaint)

    # Queue email sending in background
    background_tasks.add_task(
        send_complaint_email,
        email=user.email,
        category=new_complaint.category,
        description=new_complaint.description,
        name=user.full_name if user.full_name else None
    )

    return new_complaint

@router.get("/{patient_email}", response_model=List[ComplaintResponse])
def get_complaints(patient_email: str, db: Session = Depends(get_db)):
    complaints = db.query(Complaint).filter(Complaint.patient_email == patient_email).order_by(Complaint.created_at.desc()).all()
    return complaints
