import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from openai import OpenAI

from auth.database import get_db
from doctor_matching.models import DoctorProfile
from doctor_matching.router import get_matched_doctors
from dashboard.patient.models import PatientInfo
from dashboard.patient.timeline_models import TimelineEntry
from .schemas import (
    SymptomTriageRequest,
    TriageResponse,
    TriageSessionResponse,
    TriageReportResponse,
    ChatMessage,
)
from .models import TriageSession

router = APIRouter(prefix="/symptom_triage", tags=["Symptom Triage"])


def get_kolkata_specialties(db: Session) -> List[str]:
    """
    Query the doctor_profiles table for distinct specialties available in Kolkata.
    Returns a cleaned, deduplicated list of the top specialties by doctor count.
    """
    rows = (
        db.query(DoctorProfile.role, func.count(DoctorProfile.id).label("cnt"))
        .filter(DoctorProfile.city.ilike("Kolkata"))
        .group_by(DoctorProfile.role)
        .order_by(func.count(DoctorProfile.id).desc())
        .limit(50)
        .all()
    )
    # Return only the specialty names
    return [row[0] for row in rows if row[0] and row[0].strip()]


def get_system_instruction(
    context_str: str,
    triage_rules: Optional[str] = None,
    available_specialties: Optional[List[str]] = None,
) -> str:
    rules_section = ""
    if triage_rules and triage_rules.strip():
        rules_section = f"""

USER TRIAGE RULES (provided by the patient — you MUST follow these strictly throughout the entire conversation):
{triage_rules.strip()}
"""

    # Build the live Kolkata specialties section
    if available_specialties:
        specialties_list = "\n".join(f"  - {s}" for s in available_specialties)
        specialties_section = f"""

AVAILABLE DOCTOR SPECIALTIES IN KOLKATA (live from our database — you MUST choose recommended_specialty ONLY from this list):
{specialties_list}
"""
    else:
        specialties_section = ""

    return f"""You are Arogya AI, an empathetic, expert clinical intelligence symptom triage assistant for the Arogya platform.
Your goal is to carefully and methodically ask the patient relevant medical questions one by one to fully understand their symptoms.
Once you have enough clinical information to make a safe assessment, provide a short triage summary and suggest appropriate medical actions.

=== GEOGRAPHIC CONSTRAINT — CRITICAL ===
This platform ONLY serves patients in KOLKATA, West Bengal, India.
- Every doctor recommendation you make must be available in Kolkata.
- When you set recommended_specialty, you MUST pick the closest match from the "AVAILABLE DOCTOR SPECIALTIES IN KOLKATA" list below.
- Do NOT recommend specialties not present in that list. If unsure, use "General Medicine" or "Internal Medicine" as the fallback.

Patient Context:
{context_str}{specialties_section}{rules_section}

IMPORTANT INSTRUCTIONS:
- You must always respond with a valid JSON object matching this schema:
{{
  "message": "Your text response or question to the patient",
  "chips": ["Option 1", "Option 2"],
  "isComplete": false
}}

- When you set "isComplete": true, you MUST also include these additional fields in the same JSON object:
{{
  "urgency_level": "routine",
  "recommended_specialty": "<MUST be one of the specialties from the AVAILABLE DOCTOR SPECIALTIES IN KOLKATA list above>",
  "triage_summary": "2-3 sentence clinical summary of the patient's condition, key findings, and recommended action.",
  "session_title": "Brief 5-8 word title describing the chief complaint"
}}

- Maintain an empathetic and professional tone.
- Ask ONLY ONE question at a time.
- If symptoms suggest a severe or life-threatening emergency, IMMEDIATELY set isComplete to true with urgency_level "emergency" and recommend calling emergency services.

Urgency level definitions:
- "emergency": Life-threatening (chest pain with radiation, stroke symptoms, severe breathing difficulty, loss of consciousness)
- "urgent": Needs same-day or next-day care (high fever, moderate injury, uncontrolled pain)
- "semi-urgent": Needs care within 1-3 days (infection symptoms, moderate pain, worsening chronic symptoms)
- "routine": Non-urgent, can schedule standard appointment
"""


@router.get("/sessions", response_model=List[TriageSessionResponse])
def get_triage_sessions(patient_email: str, db: Session = Depends(get_db)):
    """Fetch all chat sessions for the patient, auto-deleting those older than 14 days."""
    fourteen_days_ago = datetime.now(timezone.utc) - timedelta(days=14)

    # Auto-delete sessions older than 14 days
    db.query(TriageSession).filter(
        TriageSession.patient_email == patient_email,
        TriageSession.created_at < fourteen_days_ago
    ).delete(synchronize_session=False)
    db.commit()

    sessions = (
        db.query(TriageSession)
        .filter(TriageSession.patient_email == patient_email)
        .order_by(TriageSession.updated_at.desc())
        .all()
    )
    return sessions


@router.get("/sessions/{session_id}")
def get_triage_session(session_id: str, db: Session = Depends(get_db)):
    """Fetch the chat messages of a specific session."""
    session = db.query(TriageSession).filter(TriageSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session.id, "title": session.title, "messages": session.messages}


@router.get("/sessions/{session_id}/report", response_model=TriageReportResponse)
def get_triage_report(session_id: str, db: Session = Depends(get_db)):
    """
    Fetch the structured triage report for a completed session.
    Returns urgency_level, recommended_specialty, and triage_summary.
    Only populated after isComplete=True has been returned by the triage endpoint.
    """
    session = db.query(TriageSession).filter(TriageSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.triage_summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This session has no completed triage report yet."
        )

    matched_docs = []
    try:
        matched_docs = get_matched_doctors(
            db=db,
            symptoms=session.triage_summary,
            recommended_specialty=session.recommended_specialty,
            location="Kolkata",
            limit=4
        )
    except Exception as err:
        print(f"Failed to fetch matched doctors for triage report: {err}")

    return TriageReportResponse(
        session_id=session.id,
        title=session.title,
        urgency_level=session.urgency_level,
        recommended_specialty=session.recommended_specialty,
        triage_summary=session.triage_summary,
        matched_doctors=matched_docs,
        completed_at=session.completed_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_triage_session(session_id: str, db: Session = Depends(get_db)):
    """Manually delete a specific chat session."""
    session = db.query(TriageSession).filter(TriageSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return


@router.post("/triage", response_model=TriageResponse)
def triage_symptoms(payload: SymptomTriageRequest, db: Session = Depends(get_db)):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    # ── 1. Fetch patient context ──
    patient = db.query(PatientInfo).filter(PatientInfo.email == payload.patient_email).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient profile not found. Cannot provide personalized triage."
        )

    context_parts = []
    context_parts.append(f"Patient Name: {patient.name}")
    context_parts.append(f"Gender: {patient.gender}")
    if patient.smoking_status:
        context_parts.append(f"Smoking Status: {patient.smoking_status}")
    if patient.alcohol_consumption:
        context_parts.append(f"Alcohol Consumption: {patient.alcohol_consumption}")
    if patient.chronic_conditions:
        context_parts.append(f"Chronic Conditions: {', '.join(patient.chronic_conditions)}")
    if patient.known_allergies:
        context_parts.append(f"Known Allergies: {', '.join(patient.known_allergies)}")
    if patient.active_medications:
        context_parts.append(f"Active Medications: {', '.join(patient.active_medications)}")

    timeline_entries = (
        db.query(TimelineEntry)
        .filter(TimelineEntry.patient_email == payload.patient_email)
        .order_by(TimelineEntry.created_at.desc())
        .limit(10)
        .all()
    )
    if timeline_entries:
        context_parts.append("\nRecent Health Timeline Entries (Uploaded Symptoms/Documents):")
        for entry in reversed(timeline_entries):
            severity_str = f" (Severity: {entry.severity}/10)" if entry.severity else ""
            context_parts.append(
                f"- {entry.created_at.strftime('%Y-%m-%d')}: [{entry.type.upper()}] "
                f"{entry.title}{severity_str} - {entry.details or 'No additional details'}"
            )

    context_str = "\n".join(context_parts)
    # system_prompt is built after session load so we know effective_rules — see step 3 below

    # ── 2. Session Management ──
    session = None
    if payload.session_id:
        session = db.query(TriageSession).filter(TriageSession.id == payload.session_id).first()

    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    latest_user_message = payload.messages[-1].content

    if not session:
        # Create a new session with a placeholder title
        title = latest_user_message[:50] + ("..." if len(latest_user_message) > 50 else "")
        session = TriageSession(
            patient_email=payload.patient_email,
            title=title,
            messages=[],
            # Persist any user-defined triage rules supplied on session creation
            triage_rules=payload.triage_rules.strip() if payload.triage_rules else None,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    else:
        # Allow updating triage_rules mid-session if the caller explicitly re-sends them
        if payload.triage_rules and payload.triage_rules.strip():
            session.triage_rules = payload.triage_rules.strip()

    # ── 3. Build message history for OpenAI ──
    # Resolve effective triage rules: prefer payload value, fall back to what is stored in the session
    effective_rules = (
        (payload.triage_rules or "").strip()
        or (session.triage_rules or "")
    ) or None

    # ── Fetch live Kolkata specialties from the database ──
    kolkata_specialties = get_kolkata_specialties(db)

    system_prompt = get_system_instruction(
        context_str,
        triage_rules=effective_rules,
        available_specialties=kolkata_specialties,
    )

    openai_messages = [{"role": "system", "content": system_prompt}]

    for msg in payload.messages:
        role = "assistant" if msg.role in ("model", "assistant") else "user"
        openai_messages.append({"role": role, "content": msg.content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=1024,
        )

        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)

        is_complete = response_data.get("isComplete", False)

        # ── 4. Persist messages to session ──
        updated_messages = []
        for msg in payload.messages:
            role = "assistant" if msg.role in ("model", "assistant") else "user"
            updated_messages.append({"role": role, "content": msg.content})

        updated_messages.append({
            "role": "assistant",
            "content": response_data.get("message", "")
        })

        session.messages = updated_messages

        # ── 5. Fix updated_at — JSONB mutations don't trigger SQLAlchemy onupdate ──
        session.updated_at = datetime.now(timezone.utc)

        # ── 6. Persist structured triage report on completion ──
        urgency_level = None
        recommended_specialty = None
        triage_summary = None
        matched_docs = None

        if is_complete:
            urgency_level = response_data.get("urgency_level")
            recommended_specialty = response_data.get("recommended_specialty")
            triage_summary = response_data.get("triage_summary")
            session_title = response_data.get("session_title")

            session.urgency_level = urgency_level
            session.recommended_specialty = recommended_specialty
            session.triage_summary = triage_summary
            session.completed_at = datetime.now(timezone.utc)

            # Update session title to the LLM-generated clinical title
            if session_title:
                session.title = session_title

            # Automatically match top Kolkata doctors for this assessed condition
            try:
                matched_docs = get_matched_doctors(
                    db=db,
                    symptoms=triage_summary or latest_user_message,
                    recommended_specialty=recommended_specialty,
                    location="Kolkata",
                    limit=4
                )
            except Exception as match_err:
                print(f"Failed to match doctors in triage: {match_err}")
                matched_docs = []

        db.commit()

        return TriageResponse(
            message=response_data.get("message", "I'm sorry, I couldn't process that."),
            chips=response_data.get("chips"),
            isComplete=is_complete,
            session_id=str(session.id),
            urgency_level=urgency_level,
            recommended_specialty=recommended_specialty,
            triage_summary=triage_summary,
            matched_doctors=matched_docs,
            system_prompt=system_prompt,
        )

    except Exception as e:
        error_str = str(e)
        print(f"ERROR in triage_symptoms: {error_str}")
        if "429" in error_str or "quota" in error_str.lower():
            return TriageResponse(
                message="CognoCare AI is currently experiencing high traffic. Please try again in a few moments.",
                chips=[],
                isComplete=False,
                session_id=str(session.id) if session else None
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {error_str}"
        )
