import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from openai import OpenAI

from auth.database import get_db
from patient.models import PatientInfo
from patient.timeline_models import TimelineEntry
from .schemas import SymptomTriageRequest, TriageResponse, TriageSessionResponse, ChatMessage
from .models import TriageSession

router = APIRouter(prefix="/symptom_triage", tags=["Symptom Triage"])

def get_system_instruction(context_str: str) -> str:
    return f"""
You are CognoCare AI, an empathetic, expert clinical intelligence symptom triage assistant.
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

    # Fetch patient context
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
        
    latest_user_message = payload.messages[-1].content
    
    if not session:
        # Create a new session
        title = latest_user_message[:50] + ("..." if len(latest_user_message) > 50 else "")
        session = TriageSession(
            patient_email=payload.patient_email,
            title=title,
            messages=[]
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
    # Build history for OpenAI
    openai_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in payload.messages:
        role = "assistant" if msg.role == "model" or msg.role == "assistant" else "user"
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
            role = "assistant" if msg.role == "model" or msg.role == "assistant" else "user"
            updated_messages.append({"role": role, "content": msg.content})
        
        updated_messages.append({
            "role": "assistant",
            "content": response_data.get("message", "")
        })
        
        session.messages = updated_messages
        db.commit()

        return TriageResponse(
            message=response_data.get("message", "I'm sorry, I couldn't process that."),
            chips=response_data.get("chips"),
            isComplete=response_data.get("isComplete", False),
            session_id=str(session.id)
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
