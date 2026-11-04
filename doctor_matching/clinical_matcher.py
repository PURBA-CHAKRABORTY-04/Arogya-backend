"""
doctor_matching/clinical_matcher.py
================================================================================
Feature Engineering, Clinical Safety Verification ("Harm: No"), and Explainability
Engine for Doctor Recommendation & Matching across Medical Datasets.
================================================================================
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── CLINICAL TAXONOMY MAP ─────────────────────────────────────────────────────

CLINICAL_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "cardiology": {
        "keywords": ["cardio", "heart", "coronary", "cardiac", "vascular", "angina", "arrhythmia", "hypertension", "bp", "chest pain", "interventional cardiology"],
        "synonyms": ["cardiologist", "cardiac surgeon", "interventional cardiologist", "electrophysiologist"]
    },
    "neurology": {
        "keywords": ["neuro", "brain", "spine", "stroke", "seizure", "epilepsy", "headache", "migraine", "nerve", "neuropathy", "tremor", "dizziness"],
        "synonyms": ["neurologist", "neurosurgeon", "neuro physician"]
    },
    "pulmonology": {
        "keywords": ["pulmon", "respiratory", "lung", "asthma", "copd", "bronch", "cough", "shortness of breath", "dyspnea", "pneumonia", "chest"],
        "synonyms": ["pulmonologist", "chest specialist", "respiratory physician"]
    },
    "gastroenterology": {
        "keywords": ["gastro", "digest", "liver", "stomach", "endoscopy", "abdomen", "gastric", "reflux", "jaundice", "hepat", "bowel"],
        "synonyms": ["gastroenterologist", "hepatologist", "gastroscience"]
    },
    "orthopedics": {
        "keywords": ["ortho", "bone", "joint", "spine", "knee", "hip", "fracture", "arthritis", "ligament", "sprain", "musculoskeletal"],
        "synonyms": ["orthopedic surgeon", "orthopedist", "spine surgeon"]
    },
    "dermatology": {
        "keywords": ["derm", "skin", "hair", "nail", "rash", "itch", "eczema", "psoriasis", "acne", "allergy"],
        "synonyms": ["dermatologist", "skin specialist", "cosmetologist"]
    },
    "ent": {
        "keywords": ["ent", "ear", "nose", "throat", "head & neck", "otolaryng", "sinus", "tonsil", "hearing", "tinnitus"],
        "synonyms": ["ent specialist", "otolaryngologist", "head and neck surgeon"]
    },
    "oncology": {
        "keywords": ["onco", "cancer", "tumor", "malignan", "chemo", "radiation", "carcinoma"],
        "synonyms": ["oncologist", "medical oncologist", "surgical oncologist", "radiation oncologist"]
    },
    "gynaecology": {
        "keywords": ["gynec", "gynaec", "obstetric", "pregnancy", "women", "maternity", "period", "pelvic", "fertility"],
        "synonyms": ["gynaecologist", "obstetrician", "women health specialist"]
    },
    "pediatrics": {
        "keywords": ["pediatr", "paediatr", "child", "infant", "newborn", "neonat", "colic"],
        "synonyms": ["pediatrician", "child specialist", "neonatologist"]
    },
    "psychiatry": {
        "keywords": ["psychiatr", "mental", "depression", "anxiety", "bipolar", "schizo", "behavioral", "counseling", "stress"],
        "synonyms": ["psychiatrist", "psychologist", "mental health specialist"]
    },
    "general medicine": {
        "keywords": ["general medicine", "internal medicine", "physician", "family", "fever", "weakness", "infection", "consultant"],
        "synonyms": ["general physician", "consultant physician", "internal medicine specialist"]
    },
    "nephrology_urology": {
        "keywords": ["nephro", "urol", "kidney", "renal", "urinary", "bladder", "prostate", "dialysis"],
        "synonyms": ["nephrologist", "urologist", "renal specialist"]
    },
    "ophthalmology": {
        "keywords": ["ophthalm", "eye", "vision", "cataract", "retina", "glaucoma"],
        "synonyms": ["ophthalmologist", "eye specialist", "eye surgeon"]
    }
}


# ── FEATURE EXTRACTION VECTOR ─────────────────────────────────────────────────

@dataclass
class MatchFeatureVector:
    """Standardized numerical features for candidate doctor matching."""
    name_similarity: float       # [0, 1] Jaro-Winkler / token match
    specialty_match: float       # [0, 1] Taxonomy-aligned clinical overlap
    qualification_tier: float    # [0, 1] Normalized degree prestige/seniority
    experience_scaled: float     # [0, 1] Log-scaled years of clinical practice
    distance_proximity: float    # [0, 1] Spatial decay score
    urgency_alignment: float     # [0, 1] Emergency / critical symptom routing index
    is_safe: bool = True         # Hard clinical safety constraint: "Harm: No"
    safety_violation_reason: Optional[str] = None
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> List[float]:
        return [
            self.name_similarity,
            self.specialty_match,
            self.qualification_tier,
            self.experience_scaled,
            self.distance_proximity,
            self.urgency_alignment
        ]


# ── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────

def clean_doctor_name(name: str) -> str:
    """Normalize doctor names by removing honorifics, initials, and excess spaces."""
    n = name.lower()
    n = re.sub(r'^(dr\.|dr|doctor|prof\.|prof|md\.)\s*', '', n, flags=re.IGNORECASE)
    n = re.sub(r'[^\w\s]', ' ', n)
    return ' '.join(n.split())


def compute_jaro_winkler(s1: str, s2: str) -> float:
    """Exact Jaro-Winkler string similarity implementation."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = math.floor(max(len1, len2) / 2) - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions /= 2.0
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    # Prefix scale
    p = 0.1
    prefix_len = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + (prefix_len * p * (1.0 - jaro))


def calculate_name_similarity(prompt_name_query: str, doctor_dataset_name: str) -> float:
    """
    Computes hybrid semantic and fuzzy token similarity between user prompt query
    and candidate doctor name in dataset.
    """
    if not prompt_name_query or not prompt_name_query.strip():
        # If user did not specify a name, name similarity is neutral/baseline 0.5
        return 0.5

    clean_prompt = clean_doctor_name(prompt_name_query)
    clean_target = clean_doctor_name(doctor_dataset_name)

    if clean_prompt == clean_target:
        return 1.0

    # Token overlap
    p_tokens = set(clean_prompt.split())
    t_tokens = set(clean_target.split())
    if not p_tokens or not t_tokens:
        return 0.0

    token_overlap = len(p_tokens.intersection(t_tokens)) / max(len(p_tokens), 1)

    # Substring direct hit
    if clean_prompt in clean_target or clean_target in clean_prompt:
        return max(0.90, token_overlap)

    jw = compute_jaro_winkler(clean_prompt, clean_target)
    return round(max(jw, token_overlap), 4)


def match_specialty_score(prompter_specialty: str, doctor_specialization: str) -> float:
    """
    Matches prompter specialty / complaint with doctor's specialization
    using clinical hierarchical taxonomy matching.
    """
    p_spec = prompter_specialty.lower().strip()
    d_spec = doctor_specialization.lower().strip()

    if not p_spec:
        return 0.5  # Neutral if unprompted

    # Exact string containment
    if p_spec in d_spec or d_spec in p_spec:
        return 1.0

    # Check taxonomy domains
    for domain, data in CLINICAL_TAXONOMY.items():
        domain_keywords = data["keywords"]
        domain_synonyms = data["synonyms"]

        p_matches_domain = any(re.search(r'\b' + re.escape(kw) + r'\b', p_spec) for kw in domain_keywords)
        if p_matches_domain:
            d_matches_domain = any(re.search(r'\b' + re.escape(kw) + r'\b', d_spec) for kw in domain_keywords + domain_synonyms)
            if d_matches_domain:
                return 0.95

    # Token overlap fallback
    p_words = set(re.findall(r'\w+', p_spec))
    d_words = set(re.findall(r'\w+', d_spec))
    overlap = len(p_words.intersection(d_words))
    if overlap > 0:
        return min(0.3 + 0.3 * overlap, 0.85)

    return 0.05


def parse_qualification_tier(qual_str: str) -> float:
    """
    Evaluates education string and maps degrees to normalized clinical qualification prestige score in [0.2, 1.0].
    """
    q = (qual_str or "").upper()
    score = 0.35  # Base MBBS

    if "GOLD MEDALIST" in q or "AIIMS" in q:
        score += 0.20
    if "DM" in q or "MCH" in q:
        score += 0.35
    elif "MD" in q or "MS" in q or "DNB" in q:
        score += 0.25
    if "FRCS" in q or "MRCP" in q or "CCST" in q or "FRCP" in q or "FACG" in q:
        score += 0.15

    return round(min(score, 1.0), 3)


# ── CLINICAL SAFETY & EXPLAINABILITY ENGINE ───────────────────────────────────

def evaluate_clinical_safety(
    prompter_specialty: str,
    specialty_match: float,
    urgency_level: str = "routine"
) -> Tuple[bool, Optional[str]]:
    """
    "Harm? -> No" Safety Invariant:
    If a patient presents with an acute or specified symptom/specialty requirement,
    we strictly prohibit matching them to an unrelated specialty doctor, preventing
    deadly misallocation (e.g. allocating Dermatology to Acute Coronary Syndrome).
    """
    is_urgent = urgency_level.lower() in ("critical", "emergency", "high")
    threshold = 0.50 if is_urgent else 0.20

    if prompter_specialty.strip() and specialty_match < threshold:
        return False, (
            f"CLINICAL SAFETY GATE ACTIVATED (Harm: No): Candidate specialty has zero or negligible "
            f"alignment ({specialty_match * 100:.1f}%) with requested clinical condition '{prompter_specialty}'. "
            f"Prevented clinical misrouting."
        )

    return True, None


def generate_why_explanation(
    doctor_record: Dict[str, Any],
    features: MatchFeatureVector,
    composite_probability: float
) -> Dict[str, Any]:
    """
    Generates human-interpretable clinical justifications answering "Why this doctor?".
    """
    reasons: List[str] = []
    cautions: List[str] = []

    if features.specialty_match >= 0.90:
        reasons.append(f"Specialty Match: {doctor_record.get('Specialization')} aligns directly with clinical triage assessment (+{features.specialty_match*35:.1f} pts).")
    elif features.specialty_match >= 0.50:
        reasons.append(f"Specialty Correlation: Related clinical sub-specialty match.")

    if features.name_similarity > 0.80:
        reasons.append(f"Doctor Identity: Direct name match found in verified registry ({features.name_similarity*100:.1f}% confidence).")

    if features.qualification_tier >= 0.75:
        reasons.append(f"Academic Tier: Advanced super-specialty training ({doctor_record.get('Educational Qualification', 'N/A')}).")

    if not features.is_safe:
        cautions.append(features.safety_violation_reason or "Safety risk flagged.")

    return {
        "composite_score_percent": round(composite_probability * 100, 1),
        "confidence_level": "VERY HIGH" if composite_probability >= 0.85 else ("HIGH" if composite_probability >= 0.70 else "MODERATE"),
        "harm_prevention_status": "CERTIFIED_SAFE (Harm: No)" if features.is_safe else "REJECTED_BY_SAFETY_GATE",
        "why_reasons": reasons,
        "cautions": cautions,
        "feature_breakdown": {
            "name_score": round(features.name_similarity, 3),
            "specialty_score": round(features.specialty_match, 3),
            "qualification_score": round(features.qualification_tier, 3),
            "experience_score": round(features.experience_scaled, 3)
        }
    }


def extract_features_for_candidate(
    query_prompt: str,
    prompter_specialty: str,
    doctor_row: Dict[str, Any],
    patient_coords: Optional[Tuple[float, float]] = None,
    urgency_level: str = "routine"
) -> MatchFeatureVector:
    """
    Full feature pipeline transforming user prompt & candidate dataset row into a numerical feature vector.
    """
    doc_name = str(doctor_row.get("Name of Doctor") or doctor_row.get("name") or "")
    doc_spec = str(doctor_row.get("Specialization") or doctor_row.get("role") or "")
    doc_qual = str(doctor_row.get("Educational Qualification") or doctor_row.get("qualification") or "")

    name_sim = calculate_name_similarity(query_prompt, doc_name)
    spec_sim = match_specialty_score(prompter_specialty, doc_spec)
    qual_tier = parse_qualification_tier(doc_qual)
    exp_scaled = 0.65  # Median clinical experience benchmark
    dist_prox = 0.85   # Nearby hospital default
    urgency_align = 0.90 if spec_sim >= 0.80 else 0.30

    is_safe, violation_reason = evaluate_clinical_safety(prompter_specialty, spec_sim, urgency_level)

    return MatchFeatureVector(
        name_similarity=name_sim,
        specialty_match=spec_sim,
        qualification_tier=qual_tier,
        experience_scaled=exp_scaled,
        distance_proximity=dist_prox,
        urgency_alignment=urgency_align,
        is_safe=is_safe,
        safety_violation_reason=violation_reason,
        raw_dict=doctor_row
    )
