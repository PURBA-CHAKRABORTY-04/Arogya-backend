import os
import json
import threading
import re
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import Base, get_db
from symptom_triage.models import TriageSession
from .models import DoctorProfile
from .schemas import DoctorMatchRequest, DoctorMatchResponse, DoctorMatchResponseItem, DoctorResponse

router = APIRouter(prefix="/doctor-matching", tags=["Doctor Matching"])

# Prompt designed for structured JSON matching
MATCHING_SYSTEM_PROMPT = """You are a medical AI matching engine.
Given a patient's symptoms, an optional recommended specialty from a triage assessment, and a list of candidate doctors (with their specialties and clinical descriptions), calculate an accurate, nuanced match score (0 to 100) and a clinical matching justification for each doctor.

Rules:
- If a recommended_specialty is provided, doctors whose specialty closely matches it should receive a high score (82-98).
- High match score (82-99): Doctor's specialty is directly relevant to the patient's primary symptoms or recommended specialty.
- Moderate match score (50-81): Doctor has general specialty relevance or functional/integrative medicine overlap.
- Low match score (0-49): Doctor's specialty is unrelated to the symptoms and recommended specialty.
- Granularity & Realism: Do NOT default to generic round numbers (e.g., exactly 85, 90, 95, 100) unless specifically warranted. Differentiate candidates based on years of experience, relevant sub-specialties, and clinical keyword alignment described in their profiles. Two doctors in the same broad specialty should rarely receive identical scores if one has more specific relevant experience.
- The response must be a valid JSON object matching this schema:
{
  "matches": [
    {
      "doctor_id": "string (UUID representing doctor ID)",
      "match_score": 93,
      "match_reason": "Clinical justification here"
    }
  ]
}
"""


def _run_llm_matching(symptoms: str, doctors_list: list[dict], recommended_specialty: str = None) -> dict:
    """Uses GPT-4o-mini to calculate semantic match scores and justifications for candidate doctors."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt_input = {
            "patient_symptoms": symptoms or "General consultation request",
            "available_doctors": [
                {
                    "id": str(doc["id"]),
                    "name": doc["name"],
                    "specialty": doc["role"],
                    "description": doc["description"]
                }
                for doc in doctors_list
            ]
        }

        if recommended_specialty:
            prompt_input["recommended_specialty"] = recommended_specialty

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MATCHING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Match task data:\n\n{json.dumps(prompt_input)}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2500,
            timeout=10.0,
        )

        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[MATCH ENGINE ERROR] {type(exc).__name__}: {exc}")
        return {}


def _get_specialty_keywords(specialty: str) -> list[str]:
    """Generates fuzzy root keywords for clinical specialty matching."""
    if not specialty:
        return []
    
    spec_clean = specialty.lower().strip()
    keywords = [spec_clean]
    
    # Common medical stem and alias mappings
    stem_map = {
        "cardio": ["cardio", "heart", "coronary", "vascular", "cardiac"],
        "neuro": ["neuro", "brain", "spine", "stroke", "nervous"],
        "ortho": ["ortho", "bone", "joint", "musculoskeletal", "spine"],
        "pulmon": ["pulmon", "respiratory", "chest", "lung", "asthma"],
        "gastro": ["gastro", "digest", "liver", "endoscopy", "gi "],
        "derm": ["derm", "skin", "cosmet"],
        "ent": ["ent", "ear", "nose", "throat", "head & neck", "otolaryng"],
        "ophthal": ["ophthal", "eye", "vision", "retina"],
        "paediatr": ["paediatr", "pediatr", "child"],
        "pediatr": ["paediatr", "pediatr", "child"],
        "psych": ["psych", "mental", "behavioral"],
        "endocrin": ["endocrin", "diabet", "thyroid", "metabolic"],
        "nephro": ["nephro", "kidney", "renal", "dialysis"],
        "urol": ["urol", "urinary", "bladder", "prostate"],
        "oncol": ["oncol", "cancer", "tumor"],
        "general": ["general", "internal medicine", "physician", "family"],
        "gynec": ["gynec", "gynaec", "obstetric", "women"],
        "gynaec": ["gynec", "gynaec", "obstetric", "women"],
    }
    
    for stem, aliases in stem_map.items():
        if stem in spec_clean:
            keywords.extend(aliases)
            
    # Add individual words
    words = re.findall(r'[a-zA-Z]{3,}', spec_clean)
    keywords.extend(words)
    return list(set(keywords))


def get_matched_doctors(
    db: Session,
    symptoms: str = "",
    recommended_specialty: Optional[str] = None,
    location: str = "All",
    max_price: Optional[float] = None,
    limit: int = 50,
) -> List[DoctorMatchResponseItem]:
    """
    Core doctor matching engine.
    Ranks doctors and matches them against symptoms and recommended specialties.
    """
    query = db.query(DoctorProfile)

    # 1. Apply hard filtering (Location and Price)
    if location and location.lower() != "all":
        query = query.filter(DoctorProfile.city.ilike(f"%{location.strip()}%"))

    if max_price is not None:
        query = query.filter(DoctorProfile.consultation_fee <= max_price)

    doctors = query.all()
    if not doctors:
        # If city filter had 0 results, fall back to all doctors
        doctors = db.query(DoctorProfile).all()
        if not doctors:
            return []

    # 2. Continuous Heuristic Matching
    scored_doctors = []
    spec_keywords = _get_specialty_keywords(recommended_specialty) if recommended_specialty else []
    symptoms_lower = (symptoms or "").lower()

    # Symptom-to-specialty mapping clusters for heuristic scoring
    symptom_specialty_map = [
        (["chest", "heart", "breath", "palpitation", "bp", "cardiac"], ["cardio", "heart", "vascular"], "Cardiology"),
        (["headache", "migraine", "dizziness", "numbness", "seizure", "neural", "nerve"], ["neuro", "brain", "neural", "spine"], "Neurology"),
        (["cough", "breath", "wheez", "asthma", "lung", "phlegm", "respiratory"], ["pulmon", "respiratory", "chest"], "Pulmonology"),
        (["stomach", "nausea", "vomit", "diarrhea", "acidity", "abdomen", "gastric"], ["gastro", "digest", "liver", "gi "], "Gastroenterology"),
        (["joint", "back", "knee", "hip", "bone", "fracture", "spine", "arthritis"], ["ortho", "bone", "joint", "spine"], "Orthopedics"),
        (["skin", "rash", "itch", "allergy", "acne", "eczema", "dermatitis"], ["derm", "skin", "allergy"], "Dermatology"),
        (["ear", "nose", "throat", "sinus", "tonsil", "hearing", "vocal"], ["ent", "ear", "throat", "otolaryng"], "ENT"),
        (["urinary", "bladder", "prostate", "kidney", "renal"], ["urol", "nephro", "kidney"], "Urology/Nephrology"),
        (["women", "pregnancy", "period", "pelvic", "maternity"], ["gynec", "gynaec", "obstetric"], "Gynecology"),
        (["child", "infant", "pediatric", "toddler"], ["paediatr", "pediatr", "child"], "Pediatrics")
    ]

    for doc in doctors:
        role_lower = (doc.role or "").lower()
        desc_lower = (doc.description or "").lower()

        # Base baseline score
        base_score = 62.0
        match_reason = f"Experienced {doc.role} available in {doc.city or 'your area'}."

        # Specialty keyword density
        specialty_points = 0
        if spec_keywords:
            role_matches = sum(1 for kw in spec_keywords if len(kw) > 2 and kw in role_lower)
            desc_matches = sum(1 for kw in spec_keywords if len(kw) > 3 and kw in desc_lower)
            
            if role_matches > 0:
                specialty_points = 24.0 + min(role_matches * 3.0 + desc_matches * 1.5, 10.0)
                match_reason = f"Primary specialty ({doc.role}) directly aligns with recommended {recommended_specialty} assessment."
            elif desc_matches > 0:
                specialty_points = 14.0 + min(desc_matches * 2.5, 8.0)
                match_reason = f"Clinical background includes expertise relevant to {recommended_specialty}."

        # Symptom keyword overlap
        symptom_points = 0
        if symptoms_lower:
            for sym_keywords, doc_keywords, spec_name in symptom_specialty_map:
                sym_match_count = sum(1 for sk in sym_keywords if sk in symptoms_lower)
                if sym_match_count > 0:
                    doc_has_role = any(dk in role_lower for dk in doc_keywords)
                    doc_has_desc = any(dk in desc_lower for dk in doc_keywords)
                    if doc_has_role:
                        pts = 25.0 + min(sym_match_count * 2.0, 7.0)
                        if pts > symptom_points:
                            symptom_points = pts
                            match_reason = f"{spec_name} specialist directly matches reported symptoms."
                    elif doc_has_desc:
                        pts = 15.0 + min(sym_match_count * 1.5, 5.0)
                        if pts > symptom_points:
                            symptom_points = pts
                            match_reason = f"Clinical experience includes relevant treatment for reported symptoms."

        # Experience & rating micro-bonus for non-bucketed organic variation (0 - 4 points)
        exp_years = getattr(doc, "experience_years", 0) or 0
        exp_bonus = min(exp_years * 0.15, 2.5) if exp_years else 0.0
        rating = getattr(doc, "rating", 0) or 0
        rating_bonus = min(max(rating - 4.0, 0.0) * 1.5, 1.5) if rating else 0.0

        raw_score = base_score + max(specialty_points, symptom_points) + (min(specialty_points, symptom_points) * 0.2) + exp_bonus + rating_bonus
        final_heuristic_score = int(round(min(max(raw_score, 45.0), 98.0)))

        scored_doctors.append({
            "doc": doc,
            "heuristic_score": final_heuristic_score,
            "heuristic_reason": match_reason
        })

    # Sort candidates by heuristic score
    scored_doctors.sort(key=lambda x: x["heuristic_score"], reverse=True)

    # 3. Refine top candidates with GPT-4o-mini directly (no leaky threads/race condition)
    llm_candidates = [
        {
            "id": item["doc"].id,
            "name": item["doc"].name,
            "role": item["doc"].role,
            "description": item["doc"].description or ""
        }
        for item in scored_doctors[:20]
    ]

    llm_matches = {}
    if (symptoms.strip() or recommended_specialty) and os.getenv("OPENAI_API_KEY"):
        llm_output = _run_llm_matching(symptoms, llm_candidates, recommended_specialty)
        if llm_output and "matches" in llm_output:
            for item in llm_output["matches"]:
                doc_id = item.get("doctor_id")
                if doc_id:
                    llm_matches[str(doc_id)] = {
                        "score": item.get("match_score", 50),
                        "reason": item.get("match_reason", "Recommended based on clinical profile.")
                    }

    # 4. Construct final items
    results = []
    for item in scored_doctors:
        doc = item["doc"]
        doc_str_id = str(doc.id)

        if doc_str_id in llm_matches:
            score = llm_matches[doc_str_id]["score"]
            reason = llm_matches[doc_str_id]["reason"]
        else:
            score = item["heuristic_score"]
            reason = item["heuristic_reason"]

        doc_schema = DoctorResponse.model_validate(doc)
        results.append(
            DoctorMatchResponseItem(
                doctor=doc_schema,
                match_score=score,
                match_reason=reason
            )
        )

    results.sort(key=lambda x: x.match_score, reverse=True)
    return results[:limit]


@router.post("/match", response_model=DoctorMatchResponse)
def match_doctors(payload: DoctorMatchRequest, db: Session = Depends(get_db)):
    """
    Ranks and matches available doctors against patient symptoms, location (city), and consultation fees.
    Works dynamically with datasets of any size (4,000+ doctors).
    """
    symptoms = payload.symptoms or ""
    patient_email = payload.patient_email
    location = payload.location or "All"
    max_price = payload.max_price
    recommended_specialty = payload.recommended_specialty

    # Auto-fetch symptoms and specialty from latest triage session if blank
    if patient_email and (not symptoms.strip() or not recommended_specialty):
        latest_session = (
            db.query(TriageSession)
            .filter(TriageSession.patient_email == patient_email)
            .order_by(TriageSession.updated_at.desc())
            .first()
        )
        if latest_session:
            if not recommended_specialty and latest_session.recommended_specialty:
                recommended_specialty = latest_session.recommended_specialty

            if not symptoms.strip() and latest_session.messages:
                user_texts = [
                    m.get("content")
                    for m in latest_session.messages
                    if m.get("role") == "user" and m.get("content")
                ]
                if user_texts:
                    symptoms = " | ".join(user_texts[-3:])

    matches = get_matched_doctors(
        db=db,
        symptoms=symptoms,
        recommended_specialty=recommended_specialty,
        location=location,
        max_price=max_price,
        limit=50
    )

    return DoctorMatchResponse(matches=matches)
