import os
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from openai import OpenAI

from auth.database import get_db
from auth.dependencies import get_current_user, require_patient, get_optional_current_user
from auth.models import User
from dashboard.patient.models import PatientInfo
from dashboard.patient.timeline_models import TimelineEntry
from .schemas import (
    SymptomTriageRequest, TriageResponse, TriageSessionResponse, ChatMessage,
    RouteRequest, RouteResponse, SpecialtyItem,
    ChatTriageRequest, ChatTriageResponse, MatchedDoctorItem,
)
from .models import TriageSession
from .triage_engine import TriageEngine
from .chat_pipeline import (
    extract_clinical_tuple, triage_story, match_doctors_for_triage, resolve_case_id
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/symptom_triage", tags=["Symptom Triage"])

# ── Issue #4: Module-level OpenAI singleton ──────────────────────────────────────
# Creating a new OpenAI() client per-request is wasteful (connection pooling
# overhead). We build it once at import time and reuse it for every request.
_openai_client: OpenAI | None = None

def get_openai_client() -> OpenAI:
    """Return the module-level OpenAI singleton, creating it if needed."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_system_instruction(context_str: str) -> str:
    return f"""
You are Arogya AI, an empathetic, expert clinical intelligence symptom triage assistant.
Your goal is to carefully and methodically ask the patient relevant medical questions one by one to fully understand their symptoms.
Once you have enough clinical information to make a safe assessment, provide a short triage summary and suggest appropriate medical actions.

Patient Context:
{context_str}

IMPORTANT INSTRUCTIONS:
- You must always respond with a valid JSON object matching this schema:
{{
  "message": "Your text response or question to the patient",
  "chips": ["Option 1", "Option 2"], // (Optional) array of suggested quick replies for the user. e.g. ["Yes", "No"]. Try to provide chips for yes/no questions or common answers to make it easier for the patient.
  "isComplete": false // Set to true ONLY when you have asked enough questions and are providing the final triage report/summary.
}}
- Maintain an empathetic and professional tone.
- Ask ONLY ONE question at a time to avoid overwhelming the patient.
- If the symptoms suggest a severe or life-threatening emergency, IMMEDIATELY set isComplete to true and recommend seeking immediate medical attention or calling emergency services.
"""


@router.get("/sessions", response_model=List[TriageSessionResponse])
def get_triage_sessions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetch chat sessions for the authenticated patient with pagination.
    - Issue #20: Mutation (auto-delete) removed from GET handler to guarantee idempotency.
    - Issue #15: Added limit/offset pagination.
    """
    sessions = (
        db.query(TriageSession)
        .filter(TriageSession.patient_email == current_user.email)
        .order_by(TriageSession.updated_at.desc())
        .offset(offset)
        .limit(min(max(limit, 1), 100))
        .all()
    )
    return sessions


@router.post("/sessions/cleanup", status_code=status.HTTP_200_OK)
def cleanup_old_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Issue #20: Explicit maintenance endpoint for purging sessions older than 14 days.
    """
    fourteen_days_ago = datetime.now(timezone.utc) - timedelta(days=14)
    deleted_count = (
        db.query(TriageSession)
        .filter(
            TriageSession.patient_email == current_user.email,
            TriageSession.created_at < fourteen_days_ago,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"message": f"Cleaned up {deleted_count} expired session(s)."}


@router.get("/sessions/{session_id}")
def get_triage_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch the chat messages of a specific session. Verifies ownership."""
    session = db.query(TriageSession).filter(TriageSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.patient_email != current_user.email:
        raise HTTPException(status_code=403, detail="You do not have access to this session.")
    return {"id": session.id, "title": session.title, "messages": session.messages}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_triage_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manually delete a specific chat session. Verifies ownership."""
    session = db.query(TriageSession).filter(TriageSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.patient_email != current_user.email:
        raise HTTPException(status_code=403, detail="You do not have access to this session.")
    db.delete(session)
    db.commit()
    return


@router.post("/triage", response_model=TriageResponse)
def triage_symptoms(
    payload: SymptomTriageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Use module-level singleton — no per-request client creation
    client = get_openai_client()

    # Derive patient email from the token — ignore payload.patient_email
    patient_email = current_user.email
    # Fetch patient context using the verified email
    patient = db.query(PatientInfo).filter(PatientInfo.email == patient_email).first()

    context_parts = []
    if patient:
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
        if hasattr(patient, "active_medications") and patient.active_medications:
            context_parts.append(f"Active Medications: {', '.join(patient.active_medications)}")
    else:
        context_parts.append("(No patient profile found — responding without medical history context.)")
    
    timeline_entries = (
        db.query(TimelineEntry)
        .filter(TimelineEntry.patient_email == patient_email)
        .order_by(TimelineEntry.created_at.desc())
        .limit(10)
        .all()
    )
    if timeline_entries:
        context_parts.append("\nRecent Health Timeline Entries (Uploaded Symptoms/Documents):")
        for entry in reversed(timeline_entries):
            severity_str = f" (Severity: {entry.severity}/10)" if entry.severity else ""
            context_parts.append(f"- {entry.created_at.strftime('%Y-%m-%d')}: [{entry.type.upper()}] {entry.title}{severity_str} - {entry.details or 'No additional details'}")

    context_str = "\n".join(context_parts)
    system_prompt = get_system_instruction(context_str)

    # Session Management
    session = None
    if payload.session_id:
        session = db.query(TriageSession).filter(TriageSession.id == payload.session_id).first()

    # Get the latest user message
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # ── Issue #10: Input Sanitization ──────────────────────────────────────────
    # Strip dangerous control characters and null bytes from incoming messages
    for msg in payload.messages:
        cleaned = "".join(ch for ch in msg.content if ch.isprintable() or ch in "\n\r\t").strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Empty or non-printable message content is not permitted.")
        msg.content = cleaned

    latest_user_message = payload.messages[-1].content
    
    if not session:
        # Create a new session
        title = latest_user_message[:50] + ("..." if len(latest_user_message) > 50 else "")
        session = TriageSession(
            patient_email=patient_email,  # always use verified token identity
            title=title,
            messages=[]
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
    # ── Issue #7: Sliding-Window History Strategy ──────────────────────────────
    # Sending full unbounded chat histories blows up token costs exponentially
    # and exhausts max_tokens context limits.
    # Best practice: Always preserve System Instructions + Initial Chief Complaint
    # + Recent Window of last 6 turns (3 Q&A rounds).
    openai_messages = [{"role": "system", "content": system_prompt}]
    
    all_msgs = payload.messages
    if len(all_msgs) <= 7:
        window_msgs = all_msgs
    else:
        initial_msg = all_msgs[0]
        recent_msgs = all_msgs[-6:]
        omitted_count = len(all_msgs) - 7
        
        # Keep chief complaint
        openai_messages.append({
            "role": "assistant" if initial_msg.role in ("model", "assistant") else "user",
            "content": f"[Patient Initial Chief Complaint]: {initial_msg.content}"
        })
        # Inject rolling transition indicator
        openai_messages.append({
            "role": "system",
            "content": f"[Note: {omitted_count} earlier dialogue turns condensed to preserve LLM token context. Active clinical conversation continues below.]"
        })
        window_msgs = recent_msgs

    for msg in window_msgs:
        role = "assistant" if msg.role in ("model", "assistant") else "user"
        openai_messages.append({"role": role, "content": msg.content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            response_format={ "type": "json_object" },
            temperature=0.4,
            max_tokens=1024,
        )
        
        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)
        
        # Save to DB
        updated_messages = []
        for msg in payload.messages:
            role = "assistant" if msg.role in ("model", "assistant") else "user"
            updated_messages.append({"role": role, "content": msg.content})
        
        updated_messages.append({
            "role": "assistant",
            "content": response_data.get("message", "")
        })
        
        session.messages = updated_messages
        session.updated_at = datetime.now(timezone.utc)
        db.commit()

        return TriageResponse(
            message=response_data.get("message", "I'm sorry, I couldn't process that."),
            chips=response_data.get("chips"),
            isComplete=response_data.get("isComplete", False),
            session_id=str(session.id)
        )
    except Exception as e:
        error_str = str(e)
        logger.error("ERROR in triage_symptoms: %s", error_str, exc_info=True)
        if "429" in error_str or "quota" in error_str.lower():
            return TriageResponse(
                message="Arogya AI is currently experiencing high traffic. Please try again in a few moments.",
                chips=[],
                isComplete=False,
                session_id=str(session.id) if session else None
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {error_str}"
        )


# ---------------------------------------------------------------------------
# POST /symptom_triage/route — 5-stage intelligence pipeline
# ---------------------------------------------------------------------------

@router.post("/route", response_model=RouteResponse)
def route_specialty(
    payload: RouteRequest,
    current_user: User = Depends(get_current_user),
):
    """Run the full 5-stage specialty routing pipeline.

    Accepts free-form patient complaint text plus optional structured fields.
    Returns validated specialty routing JSON (schema v2.0) along with a
    per-stage audit trail of any repairs made.
    """
    try:
        engine = TriageEngine(openai_api_key=None, client=get_openai_client())
        result = engine.run(payload.model_dump(exclude_none=True))
    except TypeError:
        # Older TriageEngine signature — fall back to passing key directly
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
        engine = TriageEngine(openai_api_key=api_key)
        result = engine.run(payload.model_dump(exclude_none=True))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage engine error: {exc}",
        )

    final = result.final

    # Build stage → repairs map for the frontend audit panel
    stage_repairs: dict[str, list[str]] = {}
    for s in result.stages:
        if s.issues:
            stage_repairs[s.stage] = s.issues

    return RouteResponse(
        schema_version=final.get("schema_version", "2.0"),
        emergency=final.get("emergency", False),
        urgency=final.get("urgency", "routine"),
        specialties=[
            SpecialtyItem(
                name=sp["name"],
                priority=sp["priority"],
                confidence=sp["confidence"],
            )
            for sp in final.get("specialties", [])
        ],
        missing_info=final.get("missing_info", []),
        pipeline_success=result.success,
        stage_repairs=stage_repairs,
    )


# ---------------------------------------------------------------------------
# POST /symptom_triage/chat — Unified 3-stage triage & doctor matching chat endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatTriageResponse)
def triage_chat(
    payload: ChatTriageRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Unified 3-stage triage chat endpoint:
    1. Clinical Entity & Tuple Extraction (C, A, G, D, H, M, E) + Negation Detection
    2. Triage & Urgency Evaluation with deterministic safety rules & age routing
    3. Doctor Matching & Ranking against Kolkata doctor database
    """
    # 1. Resolve patient message / story
    story_text = ""
    if payload.message:
        story_text = payload.message
    elif payload.text:
        story_text = payload.text
    elif payload.complaint:
        story_text = payload.complaint
    elif payload.story:
        story_text = payload.story
    elif payload.messages:
        for m in reversed(payload.messages):
            if m.role in ("user", "patient"):
                story_text = m.content
                break
        if not story_text and payload.messages:
            story_text = payload.messages[-1].content

    story_text = story_text.strip()
    if not story_text:
        raise HTTPException(status_code=400, detail="Patient message/story is required.")

    # 2. Stage 1: Clinical Entity & Tuple Extraction
    tuple_data = extract_clinical_tuple(story_text)
    if payload.age is not None:
        try:
            tuple_data["age"] = float(payload.age)
        except (ValueError, TypeError):
            pass
    if payload.gender:
        tuple_data["gender"] = payload.gender

    # 3. Stage 2: Triage Assessment & Red-Flag Escalation
    case_id = resolve_case_id(story_text)
    triage_data = triage_story(case_id, story_text, tuple_data)

    # 4. Stage 3: Doctor Matching from Kolkata database
    matched_doctors = match_doctors_for_triage(
        triage_data["specialties"],
        tuple_data["keywords"],
        limit=5
    )
    doctor_items = [
        MatchedDoctorItem(
            name=doc["name"],
            role=doc.get("role"),
            qualification=doc.get("qualification"),
            location=doc.get("location"),
            match_score=doc.get("match_score", 85),
            consultation_fee=doc.get("consultation_fee", 500.0),
        )
        for doc in matched_doctors
    ]

    # 5. Session tracking & persistence
    session_id = payload.session_id or str(uuid.uuid4())
    if current_user and db:
        try:
            session = db.query(TriageSession).filter(TriageSession.id == session_id).first()
            if not session:
                session = TriageSession(
                    id=uuid.UUID(session_id) if len(session_id) == 36 else uuid.uuid4(),
                    patient_email=current_user.email,
                    title=story_text[:50] + ("..." if len(story_text) > 50 else ""),
                    messages=[]
                )
                db.add(session)
            session.messages = list(session.messages or []) + [
                {"role": "user", "content": story_text},
                {"role": "assistant", "content": triage_data["reply"]}
            ]
            session.updated_at = datetime.now(timezone.utc)
            db.commit()
            session_id = str(session.id)
        except Exception as e:
            logger.warning(f"Session DB persistence note: {e}")
            db.rollback()

    # Build specialty details
    spec_items = [
        SpecialtyItem(
            name=spec,
            priority=i + 1,
            confidence=0.95 if i == 0 else 0.80
        )
        for i, spec in enumerate(triage_data["specialties"])
    ]

    return ChatTriageResponse(
        message=triage_data["reply"],
        reply=triage_data["reply"],
        keywords=tuple_data["keywords"],
        negated_keywords=tuple_data["negated_keywords"],
        tuples={
            "age": tuple_data.get("age"),
            "gender": tuple_data.get("gender"),
            "duration": tuple_data.get("duration"),
            "chief_complaints": tuple_data.get("keywords"),
            "negated_symptoms": tuple_data.get("negated_keywords"),
            "history": tuple_data.get("history"),
            "medications": tuple_data.get("medications"),
            "red_flags": triage_data.get("red_flags", []),
        },
        emergency=triage_data["emergency"],
        is_emergency=triage_data["emergency"],
        urgency=triage_data["urgency"],
        red_flags=triage_data.get("red_flags", []),
        specialties=triage_data["specialties"],
        specialty_details=spec_items,
        doctors=doctor_items,
        rules=triage_data.get("rules", []),
        chips=triage_data.get("chips", []),
        is_final=triage_data.get("is_final", True),
        session_id=session_id,
    )

