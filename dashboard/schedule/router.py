# dashboard/schedule/router.py
# FastAPI router for Schedule Events CRUD.
#
# All routes are prefixed with /schedule.
# Endpoints:
#   POST   /schedule/              — create a new schedule event
#   GET    /schedule/{email}       — list all events for a patient
#   GET    /schedule/{email}/today — list only today's events for a patient
#   PUT    /schedule/{event_id}    — update a schedule event
#   DELETE /schedule/{event_id}    — delete a schedule event

from datetime import date
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.models import User
from dashboard.schedule.models import ScheduleEvent
from dashboard.schedule.schemas import (
    ScheduleEventCreate,
    ScheduleEventResponse,
    ScheduleEventUpdate,
)


router = APIRouter(prefix="/schedule", tags=["Schedule Events"])


# ── RBAC helper ───────────────────────────────────────────────────────────────

def _require_patient(email: str, db: Session) -> User:
    """Fetch user by email; raise 404 if missing, 403 if not a patient."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No registered user found with this email.",
        )
    if user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patient accounts can manage schedule events.",
        )
    return user


# ── CREATE ────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ScheduleEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new schedule event",
)
def create_schedule_event(
    payload: ScheduleEventCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new schedule event for a patient.

    - **Medication** category requires ``times_a_day``.
      Optionally accepts ``till_date`` and ``relation_to_meal``.
    - All other categories accept an optional ``details_notes`` field.

    Returns **403** if the account is not a patient role.
    Returns **404** if the email is not registered.
    """
    _require_patient(payload.patient_email, db)

    event = ScheduleEvent(**payload.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


# ── READ — all events ─────────────────────────────────────────────────────────

@router.get(
    "/{email}",
    response_model=list[ScheduleEventResponse],
    summary="Fetch all schedule events for a patient",
)
def get_all_schedule_events(email: str, db: Session = Depends(get_db)):
    """
    Returns every schedule event for the given patient email,
    ordered by event_time ascending.

    **Note:** Percent-encode ``@`` as ``%40`` in the URL path.

    Returns **403** if the account is not a patient role.
    Returns **404** if the user does not exist.
    """
    email = unquote(email)
    _require_patient(email, db)

    events = (
        db.query(ScheduleEvent)
        .filter(ScheduleEvent.patient_email == email)
        .order_by(ScheduleEvent.event_time.asc())
        .all()
    )
    return events


# ── READ — today's events only ────────────────────────────────────────────────

@router.get(
    "/{email}/today",
    response_model=list[ScheduleEventResponse],
    summary="Fetch today's schedule events for a patient",
)
def get_today_schedule_events(email: str, db: Session = Depends(get_db)):
    """
    Returns schedule events for the current date only (using ``till_date``
    for Medication events and ``created_at`` date for others).

    Logic:
    - Medication events are included if ``till_date`` is NULL or >= today.
    - All other events are included if ``created_at`` date == today.

    **Note:** Percent-encode ``@`` as ``%40`` in the URL path.
    """
    email = unquote(email)
    _require_patient(email, db)

    today = date.today()

    events = (
        db.query(ScheduleEvent)
        .filter(ScheduleEvent.patient_email == email)
        .order_by(ScheduleEvent.event_time.asc())
        .all()
    )

    # Filter in Python so we can apply category-specific logic cleanly.
    result = []
    for ev in events:
        if ev.category == "Medication":
            # Show medication event if it has no end date or hasn't expired yet.
            if ev.till_date is None or ev.till_date >= today:
                result.append(ev)
        else:
            # Non-medication events: show if they were created today.
            if ev.created_at.date() == today:
                result.append(ev)

    return result


# ── UPDATE ────────────────────────────────────────────────────────────────────

@router.put(
    "/{event_id}",
    response_model=ScheduleEventResponse,
    summary="Update a schedule event",
)
def update_schedule_event(
    event_id: UUID,
    payload: ScheduleEventUpdate,
    db: Session = Depends(get_db),
):
    """
    Partially update a schedule event by its UUID.

    Only the fields provided in the request body are updated.

    Returns **404** if no event with that UUID exists.
    """
    event = db.query(ScheduleEvent).filter(ScheduleEvent.id == event_id).first()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule event not found.",
        )

    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(event, field, value)

    db.commit()
    db.refresh(event)
    return event


# ── DELETE ────────────────────────────────────────────────────────────────────

@router.delete(
    "/{event_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a schedule event",
)
def delete_schedule_event(event_id: UUID, db: Session = Depends(get_db)):
    """
    Permanently remove a schedule event by its UUID.

    Returns **404** if no event with that UUID exists.
    """
    event = db.query(ScheduleEvent).filter(ScheduleEvent.id == event_id).first()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule event not found.",
        )

    db.delete(event)
    db.commit()
    return {"message": "Schedule event deleted successfully."}
