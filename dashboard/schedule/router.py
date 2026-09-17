# dashboard/schedule/router.py
"""
Schedule Events CRUD — auth-gated, identity from JWT token.

All endpoints require a valid Bearer JWT.
The /me pattern replaces /{email} paths to prevent IDOR.

Endpoints:
  POST   /schedule/          — create a new schedule event
  GET    /schedule/me        — list all events for the authenticated patient
  GET    /schedule/me/today  — list only today's events
  PUT    /schedule/{event_id} — update (with ownership check)
  DELETE /schedule/{event_id} — delete (with ownership check)
"""

from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.dependencies import get_current_user, require_patient
from auth.models import User
from dashboard.schedule.models import ScheduleEvent
from dashboard.schedule.schemas import (
    ScheduleEventCreate,
    ScheduleEventResponse,
    ScheduleEventUpdate,
)

router = APIRouter(prefix="/schedule", tags=["Schedule Events"])


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
    current_user: User = Depends(require_patient),
):
    """
    Create a new schedule event for the authenticated patient.
    The ``patient_email`` in the payload is overridden by the token's identity.

    - **Medication** category requires ``times_a_day``.
    - Other categories accept an optional ``details_notes`` field.
    """
    data = payload.model_dump()
    data["patient_email"] = current_user.email  # enforce server-side identity

    event = ScheduleEvent(**data)
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


# ── READ — all events ─────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=list[ScheduleEventResponse],
    summary="Fetch all schedule events for the authenticated patient",
)
def get_all_schedule_events(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Returns every schedule event for the authenticated patient,
    ordered by event_time ascending.
    """
    events = (
        db.query(ScheduleEvent)
        .filter(ScheduleEvent.patient_email == current_user.email)
        .order_by(ScheduleEvent.event_time.asc())
        .all()
    )
    return events


# ── READ — today's events only ────────────────────────────────────────────────

@router.get(
    "/me/today",
    response_model=list[ScheduleEventResponse],
    summary="Fetch today's schedule events for the authenticated patient",
)
def get_today_schedule_events(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Returns schedule events for the current date only.

    Logic:
    - Medication events are included if ``till_date`` is NULL or >= today.
    - All other events are included if ``created_at`` date == today.
    """
    today = date.today()

    events = (
        db.query(ScheduleEvent)
        .filter(ScheduleEvent.patient_email == current_user.email)
        .order_by(ScheduleEvent.event_time.asc())
        .all()
    )

    result = []
    for ev in events:
        if ev.category == "Medication":
            if ev.till_date is None or ev.till_date >= today:
                result.append(ev)
        else:
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
    current_user: User = Depends(get_current_user),
):
    """
    Partially update a schedule event by its UUID.
    Only the owning patient may update their events.

    - Returns **404** if the event does not exist.
    - Returns **403** if the event does not belong to the caller.
    """
    event = db.query(ScheduleEvent).filter(ScheduleEvent.id == event_id).first()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule event not found.",
        )
    if event.patient_email != current_user.email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this event.",
        )

    for field, value in payload.model_dump(exclude_unset=True).items():
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
def delete_schedule_event(
    event_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Permanently remove a schedule event by its UUID.
    Only the owning patient may delete their events.

    - Returns **404** if the event does not exist.
    - Returns **403** if the event does not belong to the caller.
    """
    event = db.query(ScheduleEvent).filter(ScheduleEvent.id == event_id).first()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule event not found.",
        )
    if event.patient_email != current_user.email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this event.",
        )

    db.delete(event)
    db.commit()
    return {"message": "Schedule event deleted successfully."}
