import os
import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import google.generativeai as genai

from auth.database import get_db
from patient.models import PatientInfo
from patient.timeline_models import TimelineEntry
from .schemas import SymptomTriageRequest, TriageResponse

router = APIRouter(prefix="/symptom_triage", tags=["Symptom Triage"])

generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

@router.post("/triage", response_model=TriageResponse)
def triage_symptoms(payload: SymptomTriageRequest, db: Session = Depends(get_db)):
    # 1. Configure API key on demand to ensure it picks up env vars
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    # 2. Fetch patient context
    patient = db.query(PatientInfo).filter(PatientInfo.email == payload.patient_email).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient profile not found. Cannot provide personalized triage."
        )

    # 3. Build context string
    context_parts = []
    context_parts.append(f"Patient Name: {patient.name}")
    context_parts.append(f"Gender: {patient.gender}")
    if patient.smoking_status:
        context_parts.append(f"Smoking Status: {patient.smoking_status}")
    if patient.alcohol_consumption:
        context_parts.append(f"Alcohol Consumption: {patient.alcohol_consumption}")
    if patient.chronic_conditions and len(patient.chronic_conditions) > 0:
        context_parts.append(f"Chronic Conditions: {', '.join(patient.chronic_conditions)}")
    if patient.known_allergies and len(patient.known_allergies) > 0:
        context_parts.append(f"Known Allergies: {', '.join(patient.known_allergies)}")
    
    # 4. Fetch recent health timeline entries (symptoms, documents)
    timeline_entries = (
        db.query(TimelineEntry)
        .filter(TimelineEntry.patient_email == payload.patient_email)
        .order_by(TimelineEntry.created_at.desc())
        .limit(10)
        .all()
    )
    
    if timeline_entries:
        context_parts.append("\nRecent Health Timeline Entries (Uploaded Symptoms/Documents):")
        for entry in reversed(timeline_entries): # Chronological order
            severity_str = f" (Severity: {entry.severity}/10)" if entry.severity else ""
            context_parts.append(f"- {entry.created_at.strftime('%Y-%m-%d')}: [{entry.type.upper()}] {entry.title}{severity_str} - {entry.details or 'No additional details'}")

    context_str = "\n".join(context_parts)

    system_instruction = f"""
You are CognoCare AI, a clinical intelligence symptom triage assistant.
You will ask the patient relevant medical questions one by one to understand their symptoms fully.
Once you have enough information, provide a short triage summary and suggested actions.

Patient Context:
{context_str}

IMPORTANT INSTRUCTIONS:
- You must always respond with a valid JSON object matching this schema:
{{
  "message": "Your text response or question to the patient",
  "chips": ["Option 1", "Option 2"], // (Optional) array of suggested quick replies for the user. e.g. ["Yes", "No"]. Try to provide chips for yes/no questions or common answers to make it easier for the patient.
  "isComplete": false // Set to true ONLY when you have asked enough questions and are providing the final triage report/summary.
}}
- Be concise, empathetic, and professional. Keep responses short.
- Ask one question at a time.
- If the symptoms are severe/emergency, recommend seeking immediate medical attention.
"""

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Convert incoming history to Gemini format
    history = []
    for msg in payload.messages[:-1]: # exclude the latest user message for history
        role = "user" if msg.role == "user" else "model"
        # Since we use JSON response schema, we must pass valid string back as the model's history
        history.append({"role": role, "parts": [msg.content]})

    chat = model.start_chat(history=history)
    latest_user_message = payload.messages[-1].content
    
    try:
        response = chat.send_message(latest_user_message)
        # Clean potential markdown from response
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        response_data = json.loads(text.strip())
        
        return TriageResponse(
            message=response_data.get("message", "I'm sorry, I couldn't process that."),
            chips=response_data.get("chips"),
            isComplete=response_data.get("isComplete", False)
        )
    except Exception as e:
        error_str = str(e)
        print(f"ERROR in triage_symptoms: {error_str}")
        
        # Gracefully handle Google API rate limits or quota issues
        if "429" in error_str or "Quota exceeded" in error_str:
            return TriageResponse(
                message="CognoCare AI has reached its current Free Tier usage quota for the minute. Please wait about 60 seconds and try your message again!",
                chips=[],
                isComplete=False
            )
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {error_str}"
        )
