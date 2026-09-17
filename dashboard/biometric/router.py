# biometric/router.py
"""
CRUD endpoints for patient biometric metrics.
These are populated automatically when a document is uploaded and parsed.

All endpoints require a valid Bearer JWT.
Identity is derived from the token — the /me pattern replaces /{email} paths.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.dependencies import get_current_user, require_patient
from auth.models import User
from dashboard.biometric.models import BiometricMetric
from dashboard.biometric.schemas import BiometricMetricResponse, BiometricMetricUpdate

router = APIRouter(prefix="/biometric", tags=["Biometric Metrics"])


# ── Read ───────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=list[BiometricMetricResponse],
    summary="Fetch all biometric metrics for the authenticated patient",
)
def get_biometric_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Returns all biometric metric rows for the authenticated patient,
    ordered by most recently updated first.
    """
    metrics = (
        db.query(BiometricMetric)
        .filter(BiometricMetric.patient_email == current_user.email)
        .order_by(BiometricMetric.updated_at.desc())
        .all()
    )
    return metrics


# ── Update (single metric) ─────────────────────────────────────────────────────

@router.put(
    "/me/{metric_id}",
    response_model=BiometricMetricResponse,
    summary="Update a single biometric metric",
)
def update_biometric_metric(
    metric_id: str,
    payload: BiometricMetricUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Partially update a metric. Only the authenticated owner may update their metrics.

    - Returns **404** if the metric does not exist or does not belong to the caller.
    """
    metric = (
        db.query(BiometricMetric)
        .filter(
            BiometricMetric.id == metric_id,
            BiometricMetric.patient_email == current_user.email,
        )
        .first()
    )
    if not metric:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metric not found.")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(metric, field, value)

    db.commit()
    db.refresh(metric)
    return metric


# ── Delete (single metric) ─────────────────────────────────────────────────────

@router.delete(
    "/me/{metric_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single biometric metric",
)
def delete_biometric_metric(
    metric_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_patient),
):
    """
    Permanently remove a metric. Only the authenticated owner may delete their metrics.

    - Returns **404** if the metric does not exist or does not belong to the caller.
    """
    metric = (
        db.query(BiometricMetric)
        .filter(
            BiometricMetric.id == metric_id,
            BiometricMetric.patient_email == current_user.email,
        )
        .first()
    )
    if not metric:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metric not found.")

    db.delete(metric)
    db.commit()
    return {"message": "Metric deleted."}
