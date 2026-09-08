import os
import json
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.database import get_db
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
)
from .config import DEFAULT_CONFIG, MatchingEngineConfig
from .distance_service import distance_service
from .scoring_engine import (
    compute_doctor_score,
    compute_symptom_specialty_scores,
    calculate_pool_fee_scores,
)

router = APIRouter(prefix="/doctor-matching", tags=["Doctor Matching"])

# In-memory storage for search session location state
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

    # Rank doctors by match score (highest to lowest)
    results.sort(key=lambda x: x.match_score, reverse=True)
    return results[:limit]


@router.post("/match", response_model=DoctorMatchResponse)
def match_doctors(payload: DoctorMatchRequest, db: Session = Depends(get_db)):
    """
    Ranks and matches available doctors against patient symptoms, location, and pre-scoring hard filters.
    Executes batched Distance Matrix API calls and multi-factor weighted scoring.
    """
    symptoms = payload.symptoms or ""
    patient_email = payload.patient_email
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

    return DoctorMatchResponse(
        matches=matches, session_id=session_id, candidate_pool_count=len(matches)
    )


@router.post("/interaction", response_model=InteractionLogResponse)
def log_doctor_interaction(payload: InteractionLogRequest, db: Session = Depends(get_db)):
    """
    Logs user interactions (impressions, clicks, bookings) from doctor search results
    into PostgreSQL table doctor_ranking_interactions.
    Used for offline calibration of scoring engine weights against real user behavior.
    """
    try:
        interaction = DoctorRankingInteraction(
            session_id=payload.session_id,
            patient_email=payload.patient_email,
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
