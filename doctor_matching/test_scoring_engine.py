"""
test_scoring_engine.py — Pytest Unit Tests for Doctor Matching & Ranking Scoring Engine.

Validates all required scoring components:
1. Qualification Tier Scoring: Only MBBS vs MBBS + MD + Fellowship, diploma, super-specialty, and score cap (50 points max).
2. Distance Exponential Decay Scoring: Close doctor vs Far doctor, k decay parameter.
3. Fee Scoring: Pool-relative fee score calculation (Cheap vs Expensive).
4. Specialty vs Symptom Match Combination: High S_spec / low S_symp vs High S_symp / low S_spec.
5. Log-scaled Experience Scoring: Diminishing returns formula & cap.
6. Master Formula Raw Score & Score Breakdown Integrity.
"""

import math
import pytest
from doctor_matching.config import MatchingEngineConfig, DEFAULT_CONFIG
from doctor_matching.scoring_engine import (
    calculate_qualification_score,
    calculate_distance_score,
    calculate_experience_bonus,
    calculate_pool_fee_scores,
    compute_symptom_specialty_scores,
    compute_doctor_score,
)


def test_qualification_scoring_mbbs_vs_md_fellowship():
    """
    Test Requirement 1: Compare doctor with only MBBS vs doctor with MBBS + MD + Fellowship.
    - MBBS only: 10 points
    - MBBS + MD + Fellowship: 10 + 15 + 5 = 30 points
    """
    cfg = DEFAULT_CONFIG

    score_mbbs_only = calculate_qualification_score("MBBS", config=cfg)
    score_full = calculate_qualification_score("MBBS, MD, Fellowship in Cardiology", config=cfg)

    assert score_mbbs_only == 10.0, f"Expected 10.0 for MBBS only, got {score_mbbs_only}"
    assert score_full == 30.0, f"Expected 30.0 for MBBS + MD + Fellowship, got {score_full}"
    assert score_full > score_mbbs_only, "Doctor with higher qualifications must score higher"


def test_qualification_scoring_diploma_and_super_specialty_and_cap():
    """
    Test qualification tiers: Super-specialty (DM/MCh) + Diplomas + Fellowship capping at 50 max.
    """
    cfg = DEFAULT_CONFIG

    # MBBS(10) + MD(15) + DM(20) + DGO(8) + 3 Fellowships(capped at 2 => +10) = 63 points -> capped at 50
    score_super = calculate_qualification_score(
        "MBBS, MD, DM (Neurology), DGO, Fellowship 1, Fellowship 2, Fellowship 3",
        config=cfg
    )
    assert score_super == 50.0, f"Expected D_score capped at 50.0, got {score_super}"


def test_distance_scoring_close_vs_far():
    """
    Test Requirement 2: Compare close doctor (e.g. 2 km) vs far doctor (e.g. 30 km).
    Distance_score = 100 * exp(-distance_km / k) with k=10
    - Close (2 km): 100 * exp(-2/10) ~ 81.87
    - Far (30 km): 100 * exp(-30/10) ~ 4.98
    """
    k = 10.0
    dist_close = 2.0
    dist_far = 30.0

    score_close = calculate_distance_score(dist_close, decay_k=k)
    score_far = calculate_distance_score(dist_far, decay_k=k)

    expected_close = 100.0 * math.exp(-2.0 / 10.0)
    expected_far = 100.0 * math.exp(-30.0 / 10.0)

    assert math.isclose(score_close, expected_close, abs_tol=1e-2)
    assert math.isclose(score_far, expected_far, abs_tol=1e-2)
    assert score_close > score_far, "Close doctor must receive significantly higher distance score"
    assert score_close > 80.0
    assert score_far < 10.0


def test_fee_scoring_cheap_vs_expensive():
    """
    Test Requirement 3: Compare cheap doctor (₹300) vs expensive doctor (₹1500) relative to candidate pool.
    Fee_score = 100 * (max_fee - doctor_fee) / (max_fee - min_fee)
    Pool: [300, 800, 1500]
    - ₹300 (min fee / cheapest): 100 * (1500 - 300) / 1200 = 100 points
    - ₹1500 (max fee / most expensive): 100 * (1500 - 1500) / 1200 = 0 points
    """
    candidate_fees = [300.0, 800.0, 1500.0]
    scores = calculate_pool_fee_scores(candidate_fees)

    assert len(scores) == 3
    assert math.isclose(scores[0], 100.0), f"Cheapest doctor should get 100 points, got {scores[0]}"
    assert math.isclose(scores[2], 0.0), f"Most expensive doctor should get 0 points, got {scores[2]}"
    assert scores[0] > scores[1] > scores[2]


def test_specialty_vs_symptom_match_scoring():
    """
    Test Requirement 4: Compare high S_spec / low S_symp vs high S_symp / low S_spec.
    Formula: max(S_spec, S_symp) + 0.2 * min(S_spec, S_symp)
    """
    # Doctor A: High S_spec (35), Low S_symp (5) -> max(35, 5) + 0.2*min(35, 5) = 35 + 1.0 = 36.0
    # Doctor B: Low S_spec (5), High S_symp (35) -> max(5, 35) + 0.2*min(5, 35) = 35 + 1.0 = 36.0
    # Doctor C: High S_spec (35), High S_symp (35) -> 35 + 0.2*35 = 42.0

    bdown_a = compute_doctor_score(s_spec=35.0, s_symp=5.0, qualification_str="MBBS", experience_years=5.0, distance_km=5.0, fee_score=50.0)
    bdown_b = compute_doctor_score(s_spec=5.0, s_symp=35.0, qualification_str="MBBS", experience_years=5.0, distance_km=5.0, fee_score=50.0)
    bdown_c = compute_doctor_score(s_spec=35.0, s_symp=35.0, qualification_str="MBBS", experience_years=5.0, distance_km=5.0, fee_score=50.0)

    assert math.isclose(bdown_a.s_combined, 36.0)
    assert math.isclose(bdown_b.s_combined, 36.0)
    assert math.isclose(bdown_c.s_combined, 42.0)
    assert bdown_c.s_combined > bdown_a.s_combined, "Doctor strong on both specialty & symptom match should rank higher"


def test_experience_bonus_log_scaling():
    """
    Test log-scaled experience bonus:
    E_bonus = min(20, log(years + 1) * scale_factor)
    Diminishing returns test: gain from 0 -> 10 years is higher than 10 -> 20 years.
    """
    scale = 6.5
    bonus_0 = calculate_experience_bonus(0.0, scale_factor=scale)
    bonus_10 = calculate_experience_bonus(10.0, scale_factor=scale)
    bonus_20 = calculate_experience_bonus(20.0, scale_factor=scale)

    assert bonus_0 == 0.0
    gain_first_10 = bonus_10 - bonus_0
    gain_next_10 = bonus_20 - bonus_10

    assert gain_first_10 > gain_next_10, "Experience curve must exhibit diminishing returns"
    assert bonus_20 <= 20.0, "Experience bonus must be capped at 20.0"


def test_full_master_raw_score_calculation():
    """
    Tests exact master raw score summation against the formula:
    Raw Score = Base Score (62.0)
              + max(S_spec, S_symp) + 0.2 * min(S_spec, S_symp)
              + D_score
              + Distance_score * w_dist
              + E_bonus
              + Fee_score * w_fee
    """
    cfg = DEFAULT_CONFIG
    # Custom known values:
    # Base: 62.0
    # S_spec: 30, S_symp: 10 -> S_combined = 30 + 0.2*10 = 32.0
    # Qual: MBBS + MD -> D_score = 10 + 15 = 25.0
    # Dist: 0 km -> Distance_score = 100.0 -> dist_weighted = 100 * 0.35 = 35.0
    # Exp: 10 yrs -> log(11)*6.5 = 15.586 -> E_bonus = 15.59
    # Fee: 100.0 -> fee_weighted = 100 * 0.12 = 12.0
    # Expected Raw Score = 62.0 + 32.0 + 25.0 + 35.0 + 15.59 + 12.0 = 181.59
    
    bdown = compute_doctor_score(
        s_spec=30.0,
        s_symp=10.0,
        qualification_str="MBBS, MD",
        experience_years=10.0,
        distance_km=0.0,
        fee_score=100.0,
        config=cfg
    )

    expected_s_combined = 32.0
    expected_d_score = 25.0
    expected_dist_weighted = 35.0
    expected_fee_weighted = 12.0
    expected_exp = min(20.0, math.log(11.0) * 6.5)

    expected_raw = 62.0 + expected_s_combined + expected_d_score + expected_dist_weighted + expected_exp + expected_fee_weighted

    assert math.isclose(bdown.raw_score, expected_raw, abs_tol=0.1)
    assert bdown.final_score == cfg.MAX_FINAL_SCORE  # Clamped to max limit (98)
