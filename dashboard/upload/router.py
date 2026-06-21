# upload/router.py
"""
Cloudinary file-upload endpoint with full document intelligence pipeline:
  1. Validate + upload file to Cloudinary
  2. Send file to Unstructured.io for text extraction
  3. Send extracted text to GPT-4o for biometric metric extraction
  4. Upsert extracted metrics into the biometric_metrics table
"""

import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import cloudinary
import cloudinary.uploader
import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.database import get_db
from dashboard.biometric.models import BiometricMetric

# ── Configure Cloudinary ──────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

# ── Configure OpenAI ──────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Constants ─────────────────────────────────────────────────────────────────
UNSTRUCTURED_API_KEY      = os.getenv("UNSTRUCTURED_API_KEY", "")
UNSTRUCTURED_ENDPOINT     = "https://api.unstructuredapp.io/general/v0/general"

ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "application/pdf",
}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

router = APIRouter(prefix="/upload", tags=["File Upload"])


# ── Response schema ───────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    public_id:         str
    secure_url:        str
    resource_type:     str
    format:            str
    bytes:             int
    metrics_extracted: int   # how many biometric rows were upserted


# ── GPT metric extraction ─────────────────────────────────────────────────────
GPT_SYSTEM_PROMPT = """You are a clinical data extractor for a medical AI platform.
Given raw text from a medical document (lab report, prescription, diagnostic report, imaging summary, etc.), 
extract every measurable health metric or vital sign you can identify (e.g., Blood Glucose, Hemoglobin, Blood Pressure, Heart Rate/Pulse, SpO2, Temperature, Weight, Height, cholesterol, etc.).

Return ONLY a valid JSON array (no markdown, no explanation). Each item must have:
{
  "title":    "short metric name (e.g. Blood Glucose, Hemoglobin, BMI, Heart Rate)",
  "subtitle": "measurement context (e.g. Fasting Plasma Glucose, Resting HR)",
  "baseline": "reference / normal range with unit (e.g. 70-99 mg/dL, 120/80 mmHg). If not explicitly provided in the document, use standard clinical normal reference ranges.",
  "current":  "value found in this document with unit (e.g. 126 mg/dL, 135/85 mmHg)",
  "var_text": "percentage or absolute change from baseline (e.g. 27.3%, +12 bpm). Use — if unknown.",
  "var_type": "up | down | neutral",
  "status":   "Stable | Alert | Monitoring | Improved"
}

Rules:
- Even if a reference range is not explicitly mentioned in the text, you MUST extract the metric. Use standard clinical normal ranges as the baseline.
- If a metric is within normal range → status: Stable
- If a metric is outside normal range and clinically concerning → status: Alert  
- If a metric is outside range but not immediately dangerous → status: Monitoring
- If a metric improved compared to a prior value in the same doc → status: Improved
- If no numeric value is found for a field, use null.
- Return [] if no measurable metrics are found.
"""


def _extract_metrics_with_gpt(document_text: str) -> list[dict]:
    """Sends extracted document text to GPT-4o and returns a list of metric dicts."""
    if not document_text.strip():
        return []

    # Truncate to avoid token overflow (roughly 12k chars ~ 3k tokens)
    truncated = document_text[:12000]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Medical document text:\n\n{truncated}"},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if GPT wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return []


def _upsert_metrics(
    db: Session,
    patient_email: str,
    metrics: list[dict],
    source_url: str,
) -> int:
    """
    Upsert biometric metrics into the DB.
    Match key: patient_email + title (case-insensitive).
    Returns the number of metrics upserted.
    """
    now = datetime.now(timezone.utc)
    count = 0

    for m in metrics:
        title = (m.get("title") or "").strip()
        if not title:
            continue

        existing = (
            db.query(BiometricMetric)
            .filter(
                BiometricMetric.patient_email == patient_email,
                BiometricMetric.title.ilike(title),
            )
            .first()
        )

        var_type = m.get("var_type", "neutral")
        if var_type not in ("up", "down", "neutral"):
            var_type = "neutral"

        status_val = m.get("status", "Monitoring")
        if status_val not in ("Stable", "Alert", "Monitoring", "Improved"):
            status_val = "Monitoring"

        if existing:
            existing.subtitle       = m.get("subtitle") or existing.subtitle
            existing.baseline       = m.get("baseline") or existing.baseline
            existing.current        = m.get("current")  or existing.current
            existing.var_text       = m.get("var_text") or existing.var_text
            existing.var_type       = var_type
            existing.status         = status_val
            existing.source_doc_url = source_url
            existing.updated_at     = now
        else:
            db.add(BiometricMetric(
                id             = str(uuid.uuid4()),
                patient_email  = patient_email,
                title          = title,
                subtitle       = m.get("subtitle"),
                baseline       = m.get("baseline"),
                current        = m.get("current"),
                var_text       = m.get("var_text"),
                var_type       = var_type,
                status         = status_val,
                source_doc_url = source_url,
                created_at     = now,
                updated_at     = now,
            ))

        count += 1

    db.commit()
    return count


# ── Main upload endpoint ──────────────────────────────────────────────────────

@router.post(
    "/report",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a patient report to Cloudinary, parse it, and extract biometric metrics",
)
async def upload_report(
    file:          UploadFile    = File(...),
    patient_email: Optional[str] = Form(None),
    db:            Session       = Depends(get_db),
):
    """
    Full pipeline:
    1. Validate MIME type and size.
    2. Upload to Cloudinary (folder: cognocareai/patient-reports).
    3. Parse with Unstructured.io to extract text from PDF/image.
    4. Send text to GPT-4o to extract structured biometric metrics.
    5. Upsert metrics into the biometric_metrics table.
    6. Return the Cloudinary URL plus the count of metrics upserted.
    """

    # ── Validate MIME type ────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Only PDF, JPG, PNG are accepted.",
        )

    # ── Read and validate size ────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 10 MB limit.",
        )

    is_pdf        = file.content_type == "application/pdf"
    resource_type = "raw" if is_pdf else "image"

    # ── 1. Upload to Cloudinary ───────────────────────────────────────
    try:
        cld_result = cloudinary.uploader.upload(
            contents,
            folder        = "cognocareai/patient-reports",
            resource_type = resource_type,
            use_filename  = True,
            unique_filename = True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cloudinary upload failed: {exc}",
        )

    secure_url = cld_result["secure_url"]
    public_id  = cld_result["public_id"]
    metrics_extracted = 0

    # ── 2–5. Parse + extract metrics (only if we know the patient) ────
    if patient_email and UNSTRUCTURED_API_KEY:
        try:
            print(f"[PIPELINE] Starting for patient={patient_email}, file={file.filename}")

            # 2. Send file bytes to Unstructured.io
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    UNSTRUCTURED_ENDPOINT,
                    headers={"unstructured-api-key": UNSTRUCTURED_API_KEY},
                    files={
                        "files": (
                            file.filename or "document",
                            io.BytesIO(contents),
                            file.content_type,
                        )
                    },
                    data={"output_format": "application/json"},
                )

            print(f"[PIPELINE] Unstructured status={resp.status_code}")

            # 3. Join text elements from Unstructured response
            document_text = ""
            if resp.status_code == 200:
                elements = resp.json()
                if isinstance(elements, list):
                    document_text = "\n".join(
                        e.get("text", "") for e in elements if e.get("text")
                    )
                print(f"[PIPELINE] Extracted {len(document_text)} chars from {len(elements)} elements")
            else:
                print(f"[PIPELINE] Unstructured error: {resp.text[:300]}")

            # 4. Extract metrics with GPT-4o
            if document_text.strip():
                gpt_metrics = _extract_metrics_with_gpt(document_text)
                print(f"[PIPELINE] GPT returned {len(gpt_metrics)} metrics")

                # 5. Upsert into DB
                if gpt_metrics:
                    metrics_extracted = _upsert_metrics(
                        db, patient_email, gpt_metrics, secure_url
                    )
                    print(f"[PIPELINE] Upserted {metrics_extracted} metrics to DB")
            else:
                print("[PIPELINE] No text extracted — skipping GPT step")

        except Exception as exc:
            # Pipeline failure should NOT fail the upload itself
            print(f"[PIPELINE ERROR] {type(exc).__name__}: {exc}")
    elif not patient_email:
        print("[PIPELINE] Skipped — no patient_email in request")
    elif not UNSTRUCTURED_API_KEY:
        print("[PIPELINE] Skipped — UNSTRUCTURED_API_KEY not set")


    return UploadResponse(
        public_id         = public_id,
        secure_url        = secure_url,
        resource_type     = cld_result["resource_type"],
        format            = cld_result.get("format", "pdf" if is_pdf else ""),
        bytes             = cld_result.get("bytes", len(contents)),
        metrics_extracted = metrics_extracted,
    )
