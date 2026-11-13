"""
scoring_engine.py — Core Scoring Engine for Doctor Matching & Ranking.

Implements the master multi-factor weighted scoring formula:
Raw Score = Base Score (62.0)
          + max(S_spec, S_symp) + 0.2 * min(S_spec, S_symp)
          + D_score                                          (qualifications, capped at 50)
          + Distance_score * w_dist                          (Distance_score = 100 * exp(-distance_km / k))
          + E_bonus                                          (log-scaled experience, capped at 20)
          + Fee_score * w_fee                                (w_fee ~ 0.10 - 0.15 tiebreaker)

All point values, weights, decay constants, and caps are defined as configurable constants
in config.py to allow seamless recalibration against click/booking interaction logs.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .config import DEFAULT_CONFIG, MatchingEngineConfig


@dataclass
class ScoreBreakdown:
    """Detailed score component breakdown for auditing and transparent justification."""
    raw_score: float
    final_score: int
    base_score: float
    d_score: float
    distance_km: Optional[float]
    distance_score: float
    distance_weighted: float
    e_bonus: float
    fee_score: float
    fee_weighted: float
    s_spec: float
    s_symp: float
    s_combined: float


def calculate_qualification_score(
    qualification_str: Optional[str],
    config: Optional[MatchingEngineConfig] = None
) -> float:
    """
    Tiered Qualification/Degree Scoring (with caps):
    - Base eligibility (MBBS / BAMS / BHMS): 10 points
    - Postgraduate (MD / MS / DNB): +15 points
    - Super-specialty (DM / MCh): +20 points
    - Diploma (DGO, DCH, DLO, etc.): +8 points
    - Fellowship / Certification: +5 points each, capped at 2 (max +10)
    - Total D_score capped at configured cap (default 50.0 points).
    """
    if not qualification_str:
        return 0.0

    cfg = (config or DEFAULT_CONFIG).QUALIFICATIONS
    qual_upper = qualification_str.upper()
    d_score = 0.0

    # 1. Base eligibility check (MBBS, BAMS, BHMS, BDS)
    if any(b in qual_upper for b in ["MBBS", "BAMS", "BHMS", "BDS"]):
        d_score += cfg.MBBS

    # 2. Postgraduate degrees (MD, MS, DNB)
    pg_matches = re.findall(r'\b(MD|MS|DNB)\b', qual_upper)
    if pg_matches:
        d_score += cfg.POSTGRADUATE

    # 3. Super-specialty degrees (DM, MCH)
    ss_matches = re.findall(r'\b(DM|MCH)\b', qual_upper)
    if ss_matches:
        d_score += cfg.SUPER_SPECIALTY

    # 4. Diplomas (DGO, DCH, DLO, DA, DMRT, DPM, DDV, DIP, DIPLOMA)
    diploma_matches = re.findall(r'\b(DGO|DCH|DLO|DA|DMRT|DPM|DDV|DIP|DIPLOMA)\b', qual_upper)
    if diploma_matches:
        d_score += cfg.DIPLOMA

    # 5. Fellowships / Board Certifications (FRCS, MRCP, FRCP, FACS, FACC, FIAMS, FELLOW, FELLOWSHIP)
    fellowship_terms = re.findall(
        r'\b(FRCS|MRCP|FRCP|FACS|FACC|FIAMS|FICS|FICOG|FELLOW|FELLOWSHIP|CCST)\b',
        qual_upper
    )
    fellowship_count = min(len(fellowship_terms), cfg.FELLOWSHIP_MAX_COUNT)
    d_score += fellowship_count * cfg.FELLOWSHIP_EACH

    # Cap total D_score at configured cap (default 50.0)
    return min(d_score, cfg.TOTAL_D_SCORE_CAP)


def calculate_distance_score(
    distance_km: Optional[float],
    decay_k: float = 10.0
) -> float:
    """
    Computes distance score with exponential decay:
    Distance_score = 100 * exp(-distance_km / k)
    If distance_km is None or negative, defaults to neutral score (50.0).
    """
    if distance_km is None or distance_km < 0:
        return 50.0  # neutral score when travel distance is unknown
    
    return 100.0 * math.exp(-distance_km / max(decay_k, 0.001))


def calculate_experience_bonus(
    years_experience: Optional[float],
    scale_factor: float = 6.5,
    cap: float = 20.0
) -> float:
    """
    Log-scaled Experience Scoring with diminishing returns:
    E_bonus = min(20, log(years_experience + 1) * scale_factor)
    """
    years = max(years_experience or 0.0, 0.0)
    raw_bonus = math.log(years + 1.0) * scale_factor
    return min(cap, raw_bonus)


def calculate_pool_fee_scores(
    candidate_fees: List[float]
) -> List[float]:
    """
    Computes pool-relative fee scores:
    Fee_score = 100 * (max_fee - doctor_fee) / (max_fee - min_fee)
    Higher score indicates lower consultation fee (more accessible/cheaper).
    If all candidate doctors have identical fees, returns 50.0 for all.
    """
    if not candidate_fees:
        return []

    min_fee = min(candidate_fees)
    max_fee = max(candidate_fees)

    if math.isclose(max_fee, min_fee, abs_tol=1e-5):
        return [50.0 for _ in candidate_fees]

    fee_range = max_fee - min_fee
    return [
        100.0 * (max_fee - fee) / fee_range
        for fee in candidate_fees
    ]


def compute_symptom_specialty_scores(
    symptoms: Optional[str],
    recommended_specialty: Optional[str],
    doctor_role: str,
    doctor_description: Optional[str]
) -> Tuple[float, float]:
    """
    Calculates S_spec (Specialty match score) and S_symp (Symptom overlap score).
    Issue #14: Replaced raw substring matching with word-boundary regex and
    clinically curated medical taxonomy to eliminate false positives.
    Returns (S_spec, S_symp).
    """
    role_lower = (doctor_role or "").lower()
    desc_lower = (doctor_description or "").lower()
    symptoms_lower = (symptoms or "").lower()
    spec_lower = (recommended_specialty or "").lower().strip()

    # 1. Compute S_spec (Driven by LLM triage output)
    s_spec = 0.0
    if spec_lower:
        # Match whole specialty root (e.g. cardio, neuro, ortho)
        spec_stem = spec_lower.rstrip("ology").rstrip("ologist").rstrip("ic").rstrip("ist")
        if re.search(r'\b' + re.escape(spec_lower) + r'\b', role_lower) or (len(spec_stem) >= 4 and spec_stem in role_lower):
            s_spec = 35.0
        elif any(len(p) > 3 and p in role_lower for p in spec_lower.split()):
            s_spec = 25.0
        elif spec_lower in desc_lower:
            s_spec = 15.0

    # 2. Compute S_symp using clinical domain mapping with whole-word boundaries
    clinical_domain_map = [
        # Cardiology
        (["chest pain", "angina", "myocardial", "heart attack", "palpitation", "cardiac", "arrhythmia", "tachycardia", "hypertension", "bp"],
         ["cardio", "heart", "vascular"]),
        # Neurology
        (["headache", "migraine", "dizziness", "vertigo", "syncope", "numbness", "seizure", "neuropathy", "tremor", "stroke", "nerve"],
         ["neuro", "brain", "spine"]),
        # Pulmonology
        (["cough", "shortness of breath", "dyspnea", "wheezing", "asthma", "bronchitis", "pneumonia", "sputum", "phlegm", "lung"],
         ["pulmon", "respiratory", "chest"]),
        # Gastroenterology
        (["stomach pain", "stomach ache", "belly pain", "belly ache", "belly", "tummy", "tummy ache", "abdominal pain", "abdominal ache", "abdomen pain", "nausea", "vomiting", "diarrhea", "loose motion", "loose motions", "constipation", "acid reflux", "gerd", "gastric", "acidity", "indigestion", "bloating", "cramps", "heartburn", "jaundice", "liver", "gut pain", "digestive"],
         ["gastro", "digest", "liver", "hepat"]),
        # Orthopedics
        (["joint pain", "back pain", "backache", "knee pain", "bone fracture", "bone pain", "neck pain", "shoulder pain", "arthritis", "ligament", "sprain", "orthopedic", "spine pain"],
         ["ortho", "bone", "joint", "spine"]),
        # General / Internal Medicine
        (["fever", "chills", "high temperature", "weakness", "body ache", "fatigue", "malaise", "weight loss", "exhaustion", "viral"],
         ["general", "internal medicine", "physician", "family"]),
        # Dermatology
        (["skin rash", "pruritus", "itching", "dermatitis", "eczema", "psoriasis", "hives", "acne", "skin lesion"],
         ["derm", "skin"]),
        # ENT
        (["ear pain", "tinnitus", "sinusitis", "sore throat", "tonsillitis", "pharyngitis", "hearing loss", "hoarseness"],
         ["ent", "ear", "throat", "otolaryng"]),
        # Urology / Nephrology
        (["dysuria", "hematuria", "kidney stone", "urinary infection", "prostate", "flank pain", "renal"],
         ["urol", "nephro"]),
        # Gynecology / Obstetrics
        (["pregnancy", "menstrual", "pelvic pain", "dysmenorrhea", "amenorrhea", "vaginal", "maternity"],
         ["gynec", "gynaec", "obstetric"]),
        # Pediatrics
        (["infant", "toddler", "pediatric", "child", "colic", "newborn"],
         ["paediatr", "pediatr", "child"])
    ]

    s_symp = 0.0
    if symptoms_lower:
        for sym_tokens, doc_tokens in clinical_domain_map:
            # Word-boundary matching to prevent false positives like "chest of drawers"
            overlap = 0
            for st in sym_tokens:
                pattern = r'\b' + re.escape(st) + r'\b' if " " in st or len(st) > 3 else r'\b' + re.escape(st) + r'\b'
                if re.search(pattern, symptoms_lower):
                    overlap += 1

            if overlap > 0:
                has_role = any(dt in role_lower for dt in doc_tokens)
                has_desc = any(dt in desc_lower for dt in doc_tokens)
                if has_role:
                    pts = 25.0 + min(overlap * 3.0, 10.0)
                    s_symp = max(s_symp, pts)
                elif has_desc:
                    pts = 15.0 + min(overlap * 2.0, 7.0)
                    s_symp = max(s_symp, pts)

    return s_spec, s_symp


def compute_doctor_score(
    s_spec: float,
    s_symp: float,
    qualification_str: Optional[str],
    experience_years: Optional[float],
    distance_km: Optional[float],
    fee_score: float,
    config: Optional[MatchingEngineConfig] = None
) -> ScoreBreakdown:
    """
    Computes master multi-factor raw score and score breakdown.
    
    Formula:
    Raw Score = Base Score (62.0)
              + max(S_spec, S_symp) + 0.2 * min(S_spec, S_symp)
              + D_score
              + Distance_score * w_dist
              + E_bonus
              + Fee_score * w_fee
    """
    cfg = config or DEFAULT_CONFIG

    # 1. Combined Specialty/Symptom Match Score
    s_combined = max(s_spec, s_symp) + (cfg.SYMPTOM_SPECIALTY_MIN_WEIGHT * min(s_spec, s_symp))

    # 2. Tiered Qualification Score (capped at 50)
    d_score = calculate_qualification_score(qualification_str, config=cfg)

    # 3. Exponential Decay Distance Score
    dist_score = calculate_distance_score(distance_km, decay_k=cfg.DISTANCE.DECAY_K)
    dist_weighted = dist_score * cfg.DISTANCE.WEIGHT_W_DIST

    # 4. Log-scaled Experience Bonus (capped at 20)
    e_bonus = calculate_experience_bonus(
        experience_years,
        scale_factor=cfg.EXPERIENCE.SCALE_FACTOR,
        cap=cfg.EXPERIENCE.CAP_E_BONUS
    )

    # 5. Pool-Relative Fee Score
    fee_weighted = fee_score * cfg.FEE.WEIGHT_W_FEE

    # 6. Sum total Raw Score
    raw_score = (
        cfg.BASE_SCORE
        + s_combined
        + d_score
        + dist_weighted
        + e_bonus
        + fee_weighted
    )

    # 7. Clamp final score into integer percentage range (default 45 to 98)
    final_score = int(round(min(max(raw_score, cfg.MIN_FINAL_SCORE), cfg.MAX_FINAL_SCORE)))

    return ScoreBreakdown(
        raw_score=round(raw_score, 2),
        final_score=final_score,
        base_score=cfg.BASE_SCORE,
        d_score=round(d_score, 2),
        distance_km=round(distance_km, 2) if distance_km is not None else None,
        distance_score=round(dist_score, 2),
        distance_weighted=round(dist_weighted, 2),
        e_bonus=round(e_bonus, 2),
        fee_score=round(fee_score, 2),
        fee_weighted=round(fee_weighted, 2),
        s_spec=round(s_spec, 2),
        s_symp=round(s_symp, 2),
        s_combined=round(s_combined, 2)
    )
