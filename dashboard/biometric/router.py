# biometric/router.py
"""
CRUD endpoints for patient biometric metrics.
These are populated automatically when a document is uploaded and parsed.
"""

from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
from dashboard.biometric.models import BiometricMetric
from dashboard.biometric.schemas import BiometricMetricResponse, BiometricMetricUpdate

router = APIRouter(prefix="/biometric", tags=["Biometric Metrics"])


# ── Read ───────────────────────────────────────────────────────────────────────

@router.get(
    "/{email}",
    response_model=list[BiometricMetricResponse],
    summary="Fetch all biometric metrics for a patient",
)
def get_biometric_metrics(email: str, db: Session = Depends(get_db)):
    """
    Returns all biometric metric rows for the given patient email,
    ordered by most recently updated first.
    """
    email = unquote(email)
    metrics = (
        db.query(BiometricMetric)
        .filter(BiometricMetric.patient_email == email)
        .order_by(BiometricMetric.updated_at.desc())
        .all()
    )
    return metrics


# ── Update (single metric) ─────────────────────────────────────────────────────

@router.put(
    "/{email}/{metric_id}",
    response_model=BiometricMetricResponse,
    summary="Update a single biometric metric",
)
def update_biometric_metric(
    email: str,
    metric_id: str,
    payload: BiometricMetricUpdate,
    db: Session = Depends(get_db),
):
    email = unquote(email)
    metric = (
        db.query(BiometricMetric)
        .filter(
            BiometricMetric.id == metric_id,
            BiometricMetric.patient_email == email,
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
    "/{email}/{metric_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single biometric metric",
)
def delete_biometric_metric(
    email: str,
    metric_id: str,
    db: Session = Depends(get_db),
):
    email = unquote(email)
    metric = (
        db.query(BiometricMetric)
        .filter(
            BiometricMetric.id == metric_id,
            BiometricMetric.patient_email == email,
        )
        .first()
    )
    if not metric:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metric not found.")

    db.delete(metric)
    db.commit()
    return {"message": "Metric deleted."}
