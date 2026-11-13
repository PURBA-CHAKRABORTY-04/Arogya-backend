import os
import json
import uuid
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from auth.database import get_db
from auth.dependencies import get_current_user, get_optional_current_user
from auth.models import User
from symptom_triage.models import TriageSession
from .models import DoctorProfile, DoctorRankingInteraction
from .schemas import (
    DoctorMatchRequest,
    DoctorMatchResponse,
    DoctorMatchResponseItem,
    DoctorResponse,
    LocationPayload,
    LocationResponse,
    ScoreBreakdownSchema,
    InteractionLogRequest,
    InteractionLogResponse,
    PredictiveMatchRequest,
    PredictiveMatchResponse,
    PredictiveMatchItem,
)
from .clinical_matcher import extract_features_for_candidate
from .ml_decision_models import TwoStageConstrainedEnsemble
from .config import DEFAULT_CONFIG, MatchingEngineConfig
from .distance_service import distance_service
from .scoring_engine import (
    compute_doctor_score,
    compute_symptom_specialty_scores,
    calculate_pool_fee_scores,
)

router = APIRouter(prefix="/doctor-matching", tags=["Doctor Matching"])

# ── Issue #5: In-memory session location store ───────────────────────────────
# WARNING: This plain dict is NOT shared across Uvicorn workers or server
# restarts. If multiple workers are running (e.g. uvicorn --workers N or
# Render auto-scale), sessions created in one worker won't be visible in another.
#
# MIGRATION PATH: Replace with Redis (or a fast DB table):
#   import redis, os, json
#   _redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
#   def set_location(sid, data): _redis.setex(sid, 3600, json.dumps(data))
#   def get_location(sid): raw = _redis.get(sid); return json.loads(raw) if raw else None
#
# Until Redis is available, set UVICORN_WORKERS=1 to guarantee correctness.
SESSION_LOCATIONS: Dict[str, Dict[str, Any]] = {}


@router.post("/location", response_model=LocationResponse)
def set_session_location(payload: LocationPayload):
    """
    Mandatory unskippable location query endpoint.
    Stores user's location (browser geolocation lat/long or pincode/address entry)
    in session state, reusable for subsequent searches without re-asking.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    SESSION_LOCATIONS[session_id] = {
        "lat": payload.lat,
        "lng": payload.lng,
        "pincode": payload.pincode,
        "address": payload.address,
    }
    return LocationResponse(
        session_id=session_id,
        status="Location updated in session state",
        saved_location=SESSION_LOCATIONS[session_id],
    )


@router.get("/location/{session_id}")
def get_session_location(session_id: str):
    """Retrieves stored location for a search session."""
    if session_id not in SESSION_LOCATIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No saved location found for this session ID.",
        )
    return {"session_id": session_id, "location": SESSION_LOCATIONS[session_id]}


def infer_specialty_from_symptoms(symptoms: str) -> Optional[str]:
    """
    Semantically map ANY patient symptom description to the appropriate medical specialty.
    Uses fast rule-based clinical keyword mapping with gpt-4o-mini fallback.
    Prevents unrelated doctors (e.g. Ortho for belly ache / neurological symptoms)
    from winning on qualification points alone.
    """
    if not symptoms or len(symptoms.strip()) < 3:
        return None

    s_low = symptoms.lower()
    if any(w in s_low for w in ["belly", "stomach", "abdomen", "abdominal", "tummy", "digestion", "diarrhea", "vomit", "acidity", "constipation", "gas", "bloat", "cramp", "gerd", "acid reflux", "heartburn", "jaundice", "liver"]):
        return "Gastroenterology"
    if any(w in s_low for w in ["chest", "heart", "palpitation", "cardiac", "angina", "hypertension", "bp"]):
        return "Cardiology"
    if any(w in s_low for w in ["breath", "cough", "lung", "asthma", "wheez", "respiratory", "phlegm", "bronchitis"]):
        return "Pulmonology"
    if any(w in s_low for w in ["headache", "migraine", "dizzy", "dizziness", "vertigo", "seizure", "numb", "nerve", "brain", "stroke", "paralysis", "faint"]):
        return "Neurology"
    if any(w in s_low for w in ["bone", "joint", "fracture", "knee", "spine", "back pain", "backache", "neck pain", "sprain", "ortho", "arthritis"]):
        return "Orthopedics"
    if any(w in s_low for w in ["skin", "rash", "itch", "acne", "pimple", "eczema", "hives", "spots", "dermatitis"]):
        return "Dermatology"
    if any(w in s_low for w in ["ear", "throat", "tonsil", "sinus", "hearing", "nasal", "nose", "voice"]):
        return "ENT"
    if any(w in s_low for w in ["urine", "urination", "kidney", "prostate", "bladder", "dysuria", "hematuria"]):
        return "Urology"
    if any(w in s_low for w in ["eye", "vision", "blur", "redness in eye", "cataract"]):
        return "Ophthalmology"
    if any(w in s_low for w in ["pregnant", "pregnancy", "period", "menstrual", "vaginal", "ovary", "pcos"]):
        return "Gynecology"
    if any(w in s_low for w in ["child", "pediatric", "infant", "baby"]):
        return "Pediatrics"
    if any(w in s_low for w in ["fever", "chills", "fatigue", "body ache", "weakness", "infection"]):
        return "General Medicine"

    # Semantic LLM inference for complex, multi-symptom, or colloquial text
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = (
                "Given a patient's symptom description, return ONLY the single most relevant medical specialty from: "
                "[Cardiology, Pulmonology, Neurology, Gastroenterology, Nephrology, Orthopedics, "
                "Dermatology, Ophthalmology, ENT, Urology, Gynecology, Pediatrics, Psychiatry, General Medicine]. "
                "Return ONLY the specialty name verbatim, no punctuation, no other text."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Symptoms: {symptoms}"}
                ],
                max_tokens=15,
                temperature=0.0,
                timeout=2.5,
            )
            specialty = resp.choices[0].message.content.strip()
            if specialty and len(specialty) < 30:
                return specialty
        except Exception as exc:
            logger.warning("Semantic specialty inference failed: %s", exc)

    return None


def get_matched_doctors(
    db: Session,
    symptoms: str = "",
    recommended_specialty: Optional[str] = None,
    location: str = "All",
    patient_origin: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    max_price: Optional[float] = None,
    insurance_filter: Optional[str] = None,
    max_distance_km: Optional[float] = None,
    availability_only: bool = False,
    limit: int = 50,
    config: Optional[MatchingEngineConfig] = None,
) -> List[DoctorMatchResponseItem]:
    """
    Core doctor matching engine with:
    1. Pre-scoring Hard Filters (Insurance, Availability, Price)
    2. Batched Distance Matrix Resolution & Session Caching
    3. Multi-Factor Weighted Scoring Engine
    4. Post-Distance Radius Filter
    """
    # Semantically infer the clinical specialty if user provided symptoms but no specialty
    if symptoms and (not recommended_specialty or recommended_specialty.lower() in ("all", "none", "")):
        inferred = infer_specialty_from_symptoms(symptoms)
        if inferred:
            recommended_specialty = inferred
            logger.info("Semantically inferred specialty '%s' for symptoms: %s", inferred, symptoms[:50])

    cfg = config or DEFAULT_CONFIG
    query = db.query(DoctorProfile)

    # --- HARD FILTERS (APPLIED BEFORE SCORING) ---
    # Filter 1: Availability
    if availability_only:
        query = query.filter(DoctorProfile.is_available.is_(True))

    # Filter 2: Max Consultation Fee
    if max_price is not None:
        query = query.filter(DoctorProfile.consultation_fee <= max_price)

    # Filter 3: City / Area match (if provided)
    if location and location.lower() != "all":
        query = query.filter(DoctorProfile.city.ilike(f"%{location.strip()}%"))

    # Filter 4: Insurance Accepted
    if insurance_filter and insurance_filter.strip():
        query = query.filter(
            DoctorProfile.insurance_accepted.ilike(f"%{insurance_filter.strip()}%")
        )

    candidates = query.all()
    if not candidates:
        # Fall back to all doctors if city filter produced 0 results
        candidates = db.query(DoctorProfile).all()
        if not candidates:
            return []

    # Prepare candidate dicts for batched distance resolution
    doc_dicts = [
        {
            "id": str(doc.id),
            "address": doc.address or doc.city,
            "latitude": doc.latitude,
            "longitude": doc.longitude,
            "city": doc.city,
        }
        for doc in candidates
    ]

    # Resolve patient origin location (from payload or session state)
    effective_session_id = session_id or "default_session"
    origin_location = patient_origin or SESSION_LOCATIONS.get(effective_session_id, {})

    # Batched Distance Matrix Calculation (cached per session)
    distances = distance_service.compute_batch_distances(
        session_id=effective_session_id,
        patient_origin=origin_location,
        doctors=doc_dicts,
    )

    # Calculate pool-relative fee scores for tiebreaking
    candidate_fees = [doc.consultation_fee or 0.0 for doc in candidates]
    fee_scores = calculate_pool_fee_scores(candidate_fees)

    results: List[DoctorMatchResponseItem] = []

    for idx, doc in enumerate(candidates):
        doc_id_str = str(doc.id)
        dist_km = distances.get(doc_id_str, 8.0)

        # Filter 5: Max Acceptable Distance Radius Filter (Post-distance calculation)
        if max_distance_km is not None and dist_km > max_distance_km:
            continue

        # Calculate S_spec and S_symp match scores
        s_spec, s_symp = compute_symptom_specialty_scores(
            symptoms=symptoms,
            recommended_specialty=recommended_specialty,
            doctor_role=doc.role,
            doctor_description=doc.description,
        )

        fee_score = fee_scores[idx] if idx < len(fee_scores) else 50.0

        # Calculate master multi-factor raw score and score breakdown
        breakdown = compute_doctor_score(
            s_spec=s_spec,
            s_symp=s_symp,
            qualification_str=doc.qualification,
            experience_years=doc.experience_years,
            distance_km=dist_km,
            fee_score=fee_score,
            config=cfg,
        )

        # Build clinical match reason justification string
        reasons = []
        if breakdown.s_spec > 20:
            reasons.append(f"Specialist in {doc.role}")
        elif breakdown.s_symp > 20:
            reasons.append(f"Expertise in treating your symptoms")
        else:
            reasons.append(f"Experienced {doc.role}")

        if dist_km <= 5.0:
            reasons.append(f"Nearby clinic ({dist_km:.1f} km)")
        elif dist_km:
            reasons.append(f"{dist_km:.1f} km away")

        if doc.experience_years:
            reasons.append(f"{int(doc.experience_years)} yrs exp")

        match_reason = " • ".join(reasons)

        doc_schema = DoctorResponse.model_validate(doc)
        breakdown_schema = ScoreBreakdownSchema(
            raw_score=breakdown.raw_score,
            final_score=breakdown.final_score,
            base_score=breakdown.base_score,
            d_score=breakdown.d_score,
            distance_km=breakdown.distance_km,
            distance_score=breakdown.distance_score,
            distance_weighted=breakdown.distance_weighted,
            e_bonus=breakdown.e_bonus,
            fee_score=breakdown.fee_score,
            fee_weighted=breakdown.fee_weighted,
            s_spec=breakdown.s_spec,
            s_symp=breakdown.s_symp,
            s_combined=breakdown.s_combined,
        )

        results.append(
            DoctorMatchResponseItem(
                doctor=doc_schema,
                match_score=breakdown.final_score,
                match_reason=match_reason,
                score_breakdown=breakdown_schema,
            )
        )

    # Rank doctors: clinically relevant doctors (s_combined > 0) strictly outrank non-matches,
    # ordered by total multi-factor raw score
    results.sort(key=lambda x: (x.score_breakdown.s_combined > 0, x.score_breakdown.raw_score), reverse=True)
    return results[:limit]


@router.post("/match", response_model=DoctorMatchResponse)
def match_doctors(
    payload: DoctorMatchRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Ranks and matches available doctors against patient symptoms, location, and pre-scoring hard filters.
    Supports both authenticated patients and guest searches.
    """
    symptoms = payload.symptoms or ""
    patient_email = current_user.email if current_user else (payload.patient_email or None)
    location = payload.location or "All"
    max_price = payload.max_price
    recommended_specialty = payload.recommended_specialty
    session_id = payload.session_id or str(uuid.uuid4())

    # Resolve patient location
    patient_origin = None
    if payload.patient_location:
        patient_origin = {
            "lat": payload.patient_location.lat,
            "lng": payload.patient_location.lng,
            "pincode": payload.patient_location.pincode,
            "address": payload.patient_location.address,
        }
        SESSION_LOCATIONS[session_id] = patient_origin
    elif session_id in SESSION_LOCATIONS:
        patient_origin = SESSION_LOCATIONS[session_id]

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
        patient_origin=patient_origin,
        session_id=session_id,
        max_price=max_price,
        insurance_filter=payload.insurance_filter,
        max_distance_km=payload.max_distance_km,
        availability_only=payload.availability_only or False,
        limit=50,
    )

    limit = payload.limit or 50
    offset = payload.offset or 0
    paginated_matches = matches[offset : offset + limit]

    return DoctorMatchResponse(
        matches=paginated_matches,
        session_id=session_id,
        candidate_pool_count=len(matches),
        limit=limit,
        offset=offset,
    )


@router.post("/interaction", response_model=InteractionLogResponse)
def log_doctor_interaction(
    payload: InteractionLogRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Logs user interactions (impressions, clicks, bookings) from doctor search results.
    patient_email in the payload is overridden by the authenticated user's identity.
    """
    try:
        interaction = DoctorRankingInteraction(
            session_id=payload.session_id,
            patient_email=current_user.email,  # always use verified identity
            search_query=payload.search_query,
            doctor_id=payload.doctor_id,
            rank_position=payload.rank_position,
            raw_score=payload.raw_score,
            final_score=payload.final_score,
            score_breakdown=json.dumps(payload.score_breakdown) if payload.score_breakdown else None,
            interaction_type=payload.interaction_type,
        )
        db.add(interaction)
        db.commit()
        db.refresh(interaction)

        return InteractionLogResponse(
            status=f"Recorded {payload.interaction_type} interaction",
            interaction_id=interaction.id,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record interaction: {str(exc)}",
        )


# ── Predictive Decision Systems Matcher (Medical Dataset & ML Model) ──────────

_cached_dataset_records: Optional[List[Dict[str, Any]]] = None
_ensemble_engine = TwoStageConstrainedEnsemble()


def _get_dataset_records() -> List[Dict[str, Any]]:
    """Loads and caches the 4,291 doctor records from the CSV dataset."""
    global _cached_dataset_records
    if _cached_dataset_records is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "doctror datset",
            "Dataset Doctors - Sheet1 (1).csv"
        )
        if os.path.exists(dataset_path):
            import csv
            with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                _cached_dataset_records = list(reader)
        else:
            _cached_dataset_records = []
    return _cached_dataset_records


@router.post("/predictive-match", response_model=PredictiveMatchResponse)
def run_predictive_doctor_match(payload: PredictiveMatchRequest):
    """
    Predictive scoring model matching user prompt & prompter specialty
    against candidate medical dataset records.

    Features:
      - Match doctor specialties and doctor names from query prompt
      - Mathematical multi-model ensemble (Bayesian GLM + Random Forest + ANN)
      - Hard clinical safety constraint: "Harm: No" invariant
      - Transparent "Why?" explainability breakdown per candidate
    """
    records = _get_dataset_records()
    if not records:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Doctor dataset is unavailable.",
        )

    results: List[PredictiveMatchItem] = []
    prompter_spec = payload.prompter_specialty or ""
    query_p = payload.query_prompt.strip()

    # Pre-filter candidates: evaluate on records matching name tokens or specialty domain
    candidates = []
    clean_q = query_p.lower()
    for r in records:
        doc_name = (r.get("Name of Doctor") or "").lower()
        doc_spec = (r.get("Specialization") or "").lower()
        
        # Candidate pre-selection for speed over 4,291 rows
        if (clean_q in doc_name or any(tok in doc_name for tok in clean_q.split() if len(tok) > 3)
            or (prompter_spec and prompter_spec.lower() in doc_spec)
            or any(kw in doc_spec for kw in ["cardio", "neuro", "ortho", "gastro", "pulmon", "general", "pediatr"] if kw in clean_q)):
            candidates.append(r)

    # Fallback to sample if pre-filter returned too few
    if len(candidates) < 15:
        candidates = records[:200]

    for row in candidates:
        feat = extract_features_for_candidate(
            query_prompt=query_p,
            prompter_specialty=prompter_spec,
            doctor_row=row,
            urgency_level=payload.urgency_level or "routine"
        )
        res = _ensemble_engine.match_candidate(row, feat)
        doc = res["doctor"]
        exp = res["explanation"]

        results.append(PredictiveMatchItem(
            doctor_name=str(doc.get("Name of Doctor") or "Unknown Doctor"),
            specialization=str(doc.get("Specialization") or "General Practice"),
            hospital_location=str(doc.get("Location/Hospital") or "Kolkata, WB"),
            educational_qualification=str(doc.get("Educational Qualification") or "MBBS"),
            match_score=res["match_score"],
            match_probability=res["match_probability"],
            is_safe=res["is_safe"],
            harm_prevention_status=exp["harm_prevention_status"],
            why_reasons=exp["why_reasons"],
            cautions=exp["cautions"],
            model_breakdown=res["model_breakdown"]
        ))

    # Sort candidates by match_score descending
    results.sort(key=lambda x: x.match_score, reverse=True)
    top_results = results[:payload.limit or 10]

    return PredictiveMatchResponse(
        query_prompt=query_p,
        prompter_specialty=prompter_spec,
        matches=top_results,
        total_candidates_searched=len(candidates),
        harm_prevention_enforced=True
    )
