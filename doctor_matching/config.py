"""
config.py — Centralized Configuration Block for Doctor Matching & Ranking Scoring Engine.

All weights, point values, decay parameters, and caps are defined here as named, 
tunable constants to allow calibration based on real click/booking interaction logs.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class QualificationPointConfig:
    """Configurable point values for medical degrees and credentials."""
    MBBS: float = 10.0
    POSTGRADUATE: float = 15.0      # MD, MS, DNB
    SUPER_SPECIALTY: float = 20.0   # DM, MCh
    DIPLOMA: float = 8.0            # DGO, DCH, DLO, DNB Diploma
    FELLOWSHIP_EACH: float = 5.0    # Fellowships, FIAMS, FACC, etc.
    FELLOWSHIP_MAX_COUNT: int = 2   # Cap at 2 fellowships (max 10 points)
    TOTAL_D_SCORE_CAP: float = 50.0 # Maximum total qualification score


@dataclass
class DistanceScoreConfig:
    """Configurable exponential decay parameter and weight for travel distance."""
    DECAY_K: float = 10.0           # Exponential decay constant k in 100 * exp(-dist / k)
    WEIGHT_W_DIST: float = 0.35     # Weight multiplier for distance score in Raw Score


@dataclass
class ExperienceScoreConfig:
    """Configurable log-scale parameters for years of experience."""
    SCALE_FACTOR: float = 6.5       # Multiplier for log(years + 1)
    CAP_E_BONUS: float = 20.0       # Maximum experience bonus points


@dataclass
class FeeScoreConfig:
    """Configurable tiebreaker parameters for candidate pool relative consultation fees."""
    WEIGHT_W_FEE: float = 0.12      # Low weight tiebreaker weight (default 0.10 - 0.15)


@dataclass
class MatchingEngineConfig:
    """Master Configuration object for doctor matching engine."""
    BASE_SCORE: float = 62.0                      # Starting baseline constant
    QUALIFICATIONS: QualificationPointConfig = field(default_factory=QualificationPointConfig)
    DISTANCE: DistanceScoreConfig = field(default_factory=DistanceScoreConfig)
    EXPERIENCE: ExperienceScoreConfig = field(default_factory=ExperienceScoreConfig)
    FEE: FeeScoreConfig = field(default_factory=FeeScoreConfig)
    
    # Synergistic weighting for symptom vs specialty match
    SYMPTOM_SPECIALTY_MIN_WEIGHT: float = 0.2     # Weight for min(S_spec, S_symp)
    
    # Score clamping limits
    MIN_FINAL_SCORE: int = 45
    MAX_FINAL_SCORE: int = 98


# Default global instance used across the system
DEFAULT_CONFIG = MatchingEngineConfig()
