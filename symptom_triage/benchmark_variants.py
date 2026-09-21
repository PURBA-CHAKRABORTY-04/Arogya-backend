"""
Arogya AI — Triage Routing Prompt Variant Benchmark
====================================================
Runs all 15 prompt variants against 12 clinical test cases using gpt-4o-mini.
Measures: prompt tokens, completion tokens, total tokens, latency (ms), quality score.
Outputs: benchmark_results.json  +  variant_comparison_report.md

Usage:
    cd "d:\\Congocare Ai\\Arogya-backend"
    python symptom_triage/benchmark_variants.py

Requires OPENAI_API_KEY in environment or .env in Arogya-backend/.
"""

import os
import json
import time
import statistics
from pathlib import Path

# ── Load .env if python-dotenv is available ───────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env)
except ImportError:
    pass

from openai import OpenAI

# =============================================================================
# 1.  TEST BATTERY  (12 clinical cases)
# =============================================================================

TEST_CASES = [
    {
        "id": "TC-01",
        "label": "Crushing chest pain — STEMI pattern",
        "input": {
            "name": "Rajesh Kumar", "age": 58, "gender": "male",
            "symptoms": ["crushing chest pressure", "pain radiating to left arm", "sweating", "nausea"],
            "medical_conditions": ["hypertension", "type 2 diabetes"],
            "severity": "severe", "duration": "30 minutes",
            "associated_symptoms": ["shortness of breath", "dizziness"],
        },
        "expected": {
            "emergency": True, "urgency": "emergency",
            "acceptable_p1": ["Emergency Care", "Emergency"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-02",
        "label": "Knee pain — orthopaedic, routine",
        "input": {
            "name": "Priya Sen", "age": 45, "gender": "female",
            "symptoms": ["right knee pain", "stiffness"],
            "medical_conditions": [], "severity": "mild", "duration": "3 weeks",
            "associated_symptoms": ["swelling after walking"],
        },
        "expected": {
            "emergency": False, "urgency": "routine",
            "acceptable_p1": ["Orthopedics", "Orthopaedics", "Orthopedic Surgery"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-03",
        "label": "Headache + vision changes",
        "input": {
            "name": "Ananya Das", "age": 34, "gender": "female",
            "symptoms": ["severe headache", "blurred vision", "sensitivity to light"],
            "medical_conditions": ["migraine"], "severity": "moderate", "duration": "2 days",
            "associated_symptoms": ["nausea"],
        },
        "expected": {
            "emergency": False, "urgency": "urgent",
            "acceptable_p1": ["Neurology", "Ophthalmology", "Neurology / Ophthalmology"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-04",
        "label": "Abdominal pain + fever + vomiting",
        "input": {
            "name": "Suman Roy", "age": 29, "gender": "male",
            "symptoms": ["severe abdominal pain", "fever", "vomiting"],
            "medical_conditions": [], "severity": "moderate", "duration": "12 hours",
            "associated_symptoms": ["loss of appetite", "pain worsens with movement"],
        },
        "expected": {
            "emergency": False, "urgency": "urgent",
            "acceptable_p1": ["Gastroenterology", "Surgery", "General Surgery", "Emergency Care"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-05",
        "label": "Tooth pain — dentistry only",
        "input": {
            "name": "Meera Ghosh", "age": 27, "gender": "female",
            "symptoms": ["tooth pain", "sensitivity to cold"],
            "medical_conditions": [], "severity": "mild", "duration": "2 days",
            "associated_symptoms": [],
        },
        "expected": {
            "emergency": False, "urgency": "routine",
            "acceptable_p1": ["Dentistry", "Dental"],
            "max_specialties": 2, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-06",
        "label": "Severe SOB + chest tightness + cyanosis",
        "input": {
            "name": "Arun Bose", "age": 65, "gender": "male",
            "symptoms": ["severe shortness of breath", "chest tightness"],
            "medical_conditions": ["COPD", "heart failure"], "severity": "severe", "duration": "1 hour",
            "associated_symptoms": ["wheezing", "cyanosis of lips"],
        },
        "expected": {
            "emergency": True, "urgency": "emergency",
            "acceptable_p1": ["Emergency Care", "Emergency", "Pulmonology", "Cardiology"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-07",
        "label": "Anaphylaxis — throat swelling + dyspnea",
        "input": {
            "name": "Tanvi Sharma", "age": 22, "gender": "female",
            "symptoms": ["widespread skin rash", "throat swelling", "difficulty breathing"],
            "medical_conditions": ["shellfish allergy"], "severity": "severe", "duration": "20 minutes",
            "associated_symptoms": ["tongue swelling", "hives", "dizziness"],
        },
        "expected": {
            "emergency": True, "urgency": "emergency",
            "acceptable_p1": ["Emergency Care", "Emergency", "Allergy/Immunology"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-08",
        "label": "Urinary burning — urology routine",
        "input": {
            "name": "Kavya Nair", "age": 31, "gender": "female",
            "symptoms": ["burning sensation during urination", "frequent urination"],
            "medical_conditions": [], "severity": "mild", "duration": "3 days",
            "associated_symptoms": ["lower abdominal discomfort"],
        },
        "expected": {
            "emergency": False, "urgency": "routine",
            "acceptable_p1": ["Urology", "Nephrology", "Gynecology"],
            "max_specialties": 2, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-09",
        "label": "Stroke red-flag — facial droop + slurred speech",
        "input": {
            "name": "Dilip Chatterjee", "age": 72, "gender": "male",
            "symptoms": ["sudden facial drooping on left side", "slurred speech", "arm weakness"],
            "medical_conditions": ["atrial fibrillation"], "severity": "severe", "duration": "10 minutes",
            "associated_symptoms": ["confusion", "sudden severe headache"],
        },
        "expected": {
            "emergency": True, "urgency": "emergency",
            "acceptable_p1": ["Emergency Care", "Emergency", "Neurology"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
    {
        "id": "TC-10",
        "label": "Empty symptoms — input validation",
        "input": {
            "name": "Test Patient", "age": 30, "gender": "male",
            "symptoms": [], "medical_conditions": [],
            "severity": "unknown", "duration": "",
            "associated_symptoms": [],
        },
        "expected": {
            "emergency": False, "urgency": "routine",
            "acceptable_p1": [],   # should return empty specialties
            "max_specialties": 0,  # expect [] specialties
            "missing_info_expected": True,
        },
    },
    {
        "id": "TC-11",
        "label": "Vague complaint — insufficient input",
        "input": {
            "name": "Pooja Mehta", "age": 25, "gender": "female",
            "symptoms": ["I feel unwell", "not sure what's wrong"],
            "medical_conditions": [], "severity": "mild", "duration": "1 day",
            "associated_symptoms": [],
        },
        "expected": {
            "emergency": False, "urgency": "routine",
            "acceptable_p1": ["General Medicine", "Internal Medicine", "General Practice"],
            "max_specialties": 2, "missing_info_expected": True,
        },
    },
    {
        "id": "TC-12",
        "label": "Inflammatory joint pain — rheumatology vs ortho",
        "input": {
            "name": "Soma Banerjee", "age": 48, "gender": "female",
            "symptoms": ["joint pain in hands and wrists", "morning stiffness", "joint swelling"],
            "medical_conditions": ["family history of rheumatoid arthritis"],
            "severity": "moderate", "duration": "3 months",
            "associated_symptoms": ["fatigue", "low-grade fever"],
        },
        "expected": {
            "emergency": False, "urgency": "urgent",
            "acceptable_p1": ["Rheumatology", "Orthopedics", "Orthopaedics"],
            "max_specialties": 3, "missing_info_expected": False,
        },
    },
]

# =============================================================================
# 2.  SHARED PROMPT BLOCKS
# =============================================================================

_OUTPUT_CONTRACT = """\
OUTPUT CONTRACT (JSON only, no other text):
{
  "emergency": boolean,
  "urgency": "emergency" | "urgent" | "routine",
  "specialties": [ { "name": string, "priority": number } ],
  "missing_info": [string]
}"""

_SEVERITY_RULES = """\
SEVERITY / URGENCY RULES:
If symptoms indicate a possible medical emergency (severe/crushing chest pain, chest pain with severe breathing difficulty, sudden loss of consciousness, stroke-like symptoms, severe uncontrolled bleeding, severe allergic reaction with breathing difficulty, sudden severe neurological deficit): emergency = true, urgency = "emergency".
If symptoms are concerning but not clearly emergent: urgency = "urgent".
If symptoms are mild/stable: urgency = "routine".
Do not rule out an emergency solely because the patient describes symptoms as "mild"."""

_NEGATIVE_RULES = """\
NEGATIVE RULES — Do NOT:
- recommend unrelated specialties
- recommend every specialty that could theoretically treat the symptom
- diagnose the patient
- prescribe medication
- invent doctors or test results
- assume a cause without evidence
- use patient name to determine specialty
- use gender as a shortcut unless clinically relevant"""

_MISSING_INFO_RULE = """\
MISSING INFORMATION RULE:
If information is insufficient, provide the best-supported specialty and request only information that could materially change routing. Do not ask more than a few targeted questions."""

_PRIORITY_RULES = """\
PRIORITY RULES:
Priority 1 = specialty most directly relevant to the presenting complaint.
Priority 2 = next clinically plausible specialty based on available symptoms.
Priority 3 = only when strongly justified.
Maximum recommended specialties: 3.
Never include a specialty solely to increase coverage."""


def _fmt(inp: dict, fields=None) -> str:
    if fields is None:
        fields = list(inp.keys())
    return json.dumps({k: v for k, v in inp.items() if k in fields}, indent=2)


# =============================================================================
# 3.  VARIANT BUILDERS  — each returns (system: str, user: str)
# =============================================================================

def V01(inp):
    sys = (
        "ROLE:\nYou are a medical triage and doctor-specialty routing assistant. "
        "Your task is to identify the most relevant medical specialty based on the patient's "
        "symptoms/condition and severity. You route patients to appropriate specialists; "
        "you do not diagnose or prescribe treatment.\n\n"
        "OBJECTIVE:\n"
        "- Return only specialties relevant to the patient's reported problem.\n"
        "- Rank specialties by clinical relevance, primary first.\n"
        "- Include secondary specialties only when there is a reasonable clinical basis.\n"
        "- Use severity to determine urgency.\n"
        "- Do not recommend unrelated specialties.\n"
        "- Do not invent diagnoses, doctors, or specialties.\n"
        "- If emergency symptoms are present, prioritize emergency care.\n\n"
        "CONTEXT:\nPatient information may include name, age, gender, symptoms, medical conditions, "
        "severity, duration, and associated symptoms.\nSymptoms/medical conditions and severity are "
        "the primary routing signals. Age and gender should influence routing only when clinically "
        "relevant. Patient name is identification metadata and must never influence specialty selection.\n\n"
        + _PRIORITY_RULES + "\n\n" + _SEVERITY_RULES + "\n\n"
        "INPUT BASIS:\nPrimary: symptoms, medical condition, severity.\n"
        "Secondary: duration, associated symptoms, relevant medical history, age, gender.\n"
        "Identification only (never used for routing): patient name.\n"
        "Do not infer symptoms, medical history, or severity that were not provided.\n\n"
        + _MISSING_INFO_RULE + "\n\n" + _NEGATIVE_RULES + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp)


def V02(inp):
    sys = (
        "ROLE: Medical triage and doctor-specialty router. Route based on symptoms, condition "
        "and severity. Do not diagnose or prescribe.\n\n"
        "RULES:\n"
        "1. Return only clinically relevant specialties.\n"
        "2. Rank by relevance: primary specialty first, plausible secondary causes afterward.\n"
        "3. Emergency/red-flag symptoms -> emergency evaluation first.\n"
        "4. Use symptoms/condition and severity primarily; age/gender only when clinically relevant.\n"
        "5. Never infer missing information.\n"
        "6. Never recommend unrelated specialties.\n"
        "7. If information is insufficient, request only essential missing information.\n"
        "8. Maximum 3 specialties.\n"
        "9. Output JSON only.\n\n"
        'OUTPUT SCHEMA:\n{"emergency": bool, "urgency": string, "specialties": '
        '[{"name": string, "priority": number}], "missing_info": [string]}'
    )
    return sys, _fmt(inp, ["age", "gender", "symptoms", "severity"])


def V03(inp):
    sys = (
        "ROLE: You are a medical triage and doctor-specialty routing assistant. "
        "You route patients; you do not diagnose or prescribe.\n\n"
        "Use this mapping table as your primary reference for routing decisions:\n\n"
        "| Complaint | Priority 1 | Possible Priority 2 |\n"
        "|---|---|---|\n"
        "| Chest pain | Cardiology | Gastroenterology / Pulmonology depending on symptoms |\n"
        "| Shortness of breath | Pulmonology | Cardiology |\n"
        "| Abdominal pain | Gastroenterology | Gynecology / Urology / Surgery depending on symptoms |\n"
        "| Knee pain | Orthopedics | Rheumatology if inflammatory features |\n"
        "| Skin rash | Dermatology | Allergy/Immunology if allergy-related |\n"
        "| Headache | Neurology | Ophthalmology/ENT depending on symptoms |\n"
        "| Tooth pain | Dentistry | — |\n"
        "| Ear pain | ENT | Dentistry if referred dental pain suspected |\n"
        "| Urinary symptoms | Urology | Nephrology depending on presentation |\n"
        "| Joint pain | Orthopedics/Rheumatology | — |\n\n"
        "Only deviate from this table when the patient's specific symptoms clearly justify a different "
        "specialty. Do not add a Priority 2 specialty unless the table or the symptoms explicitly support it.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["age", "gender", "symptoms", "severity", "duration", "associated_symptoms"])


def V04(inp):
    sys = (
        "ROLE: Medical triage and doctor-specialty routing assistant. Route only; do not diagnose or prescribe.\n\n"
        "Before producing the final answer, think step by step in a \"reasoning\" field:\n"
        "1. Identify the primary presenting complaint.\n"
        "2. Check for emergency red-flag symptoms.\n"
        "3. Determine the most directly relevant specialty.\n"
        "4. Determine if a secondary specialty is clinically justified (not merely theoretically possible).\n"
        "5. Determine urgency based on severity.\n"
        "6. Identify any missing information that would materially change routing.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n"
        + _PRIORITY_RULES + "\n\n"
        "OUTPUT CONTRACT (JSON only):\n"
        '{\n  "reasoning": string,\n  "emergency": boolean,\n  "urgency": string,\n'
        '  "specialties": [{"name": string, "priority": number}],\n  "missing_info": [string]\n}'
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["age", "gender", "symptoms", "severity", "duration"])


def V05(inp):
    sys = (
        "ROLE: Medical triage and doctor-specialty routing assistant. Route only; do not diagnose or prescribe.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n"
        + _PRIORITY_RULES + "\n\n"
        "EXAMPLES:\n\n"
        'Input: {"symptoms": ["chest pain"], "severity": "moderate"}\n'
        'Output: {"emergency": false, "urgency": "urgent", "specialties": [{"name": "Cardiology", "priority": 1}], '
        '"missing_info": ["duration", "shortness_of_breath", "radiation", "sweating"]}\n\n'
        'Input: {"symptoms": ["crushing chest pressure", "sweating", "pain radiating to left arm"], "severity": "severe"}\n'
        'Output: {"emergency": true, "urgency": "emergency", "specialties": [{"name": "Emergency Care", "priority": 1}, '
        '{"name": "Cardiology", "priority": 2}], "missing_info": []}\n\n'
        'Input: {"symptoms": ["tooth pain"], "severity": "moderate"}\n'
        'Output: {"emergency": false, "urgency": "routine", "specialties": [{"name": "Dentistry", "priority": 1}], '
        '"missing_info": []}\n\n'
        'Input: {"symptoms": ["chest pain", "cough", "fever"], "severity": "moderate"}\n'
        'Output: {"emergency": false, "urgency": "urgent", "specialties": [{"name": "Cardiology", "priority": 1}, '
        '{"name": "Pulmonology", "priority": 2}], "missing_info": ["duration"]}\n\n'
        "Now route the following patient using the same logic:\n\n"
        + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["age", "gender", "symptoms", "severity", "duration"])


def V06(inp):
    sys = (
        "ROLE: Medical triage router. Your single most important priority is precision over recall — "
        "it is far worse to recommend an irrelevant specialty than to omit a plausible one.\n\n"
        "RULES:\n"
        "- Only include a secondary specialty if you are highly confident (not merely 'possible') it is clinically relevant.\n"
        "- When in doubt, output only the primary specialty.\n"
        "- Never pad the specialty list to appear thorough.\n"
        "- Maximum 2 specialties unless emergency routing requires 'Emergency Care' as an additional entry.\n"
        "- Do not diagnose or prescribe.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity", "age", "gender"])


def V07(inp):
    sys = (
        "ROLE: Medical triage router. Do not diagnose or prescribe.\n\n"
        "STAGE 1 — EMERGENCY CHECK (perform this before anything else):\n"
        "Scan symptoms for red flags: severe/crushing chest pain, chest pain with severe breathing difficulty, "
        "sudden loss of consciousness, stroke-like symptoms, severe uncontrolled bleeding, "
        "severe allergic reaction with breathing difficulty, sudden severe neurological deficit.\n"
        "If any red flag is present, set emergency=true, urgency=\"emergency\", and make \"Emergency Care\" "
        "priority 1 regardless of any other rule below. Do not let secondary specialty logic dilute this.\n\n"
        "STAGE 2 — SPECIALTY ROUTING (only after Stage 1 is resolved):\n"
        "- Priority 1 = specialty most directly relevant to the presenting complaint.\n"
        "- Priority 2 = next clinically plausible specialty.\n"
        "- Maximum 3 specialties.\n"
        "- Never recommend unrelated specialties.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity", "duration", "associated_symptoms"])


def V08(inp):
    sys = (
        "ROLE: Medical triage router. Do not diagnose or prescribe.\n\n"
        "MISSING INFORMATION POLICY (strict):\n"
        "- Never ask more than 2 missing_info items, even if more could theoretically help.\n"
        "- Only ask for information that would change the Priority 1 specialty or the emergency flag "
        "— not information that would only refine Priority 2 or 3.\n"
        "- Always provide a best-effort specialty recommendation even with incomplete data.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _PRIORITY_RULES + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity"])


def V09(inp):
    sys = (
        "ROLE: Medical triage router. Do not diagnose or prescribe.\n\n"
        "DEMOGRAPHIC USAGE POLICY:\n"
        "- Age and gender may ONLY be used to select a specialty when they are directly tied to "
        "organ systems or conditions specific to that demographic (e.g., pregnancy-related symptoms -> "
        "Obstetrics/Gynecology; pediatric age -> Pediatrics as a routing layer above the specialty).\n"
        "- Do not use age or gender as a tiebreaker between otherwise equally plausible specialties.\n"
        "- Do not use gender-based statistical priors (e.g., 'chest pain is more likely cardiac in men') "
        "to change specialty selection — route on reported symptoms only.\n"
        "- Patient name must never be used for routing under any circumstance.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n"
        + _PRIORITY_RULES + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["name", "age", "gender", "symptoms", "severity"])


def V10(inp):
    sys = (
        "ROLE: Medical triage router. Do not diagnose or prescribe.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n"
        + _PRIORITY_RULES + "\n\n"
        "For each recommended specialty, include a confidence score (0.0-1.0) reflecting how strongly "
        "the reported symptoms support that specialty. Only include a specialty if confidence >= 0.4.\n\n"
        "OUTPUT CONTRACT (JSON only):\n"
        "{\n  \"emergency\": boolean,\n  \"urgency\": string,\n"
        "  \"specialties\": [{\"name\": string, \"priority\": number, \"confidence\": number}],\n"
        "  \"missing_info\": [string]\n}"
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity", "duration"])


def V11(inp):
    sys = (
        "ROLE: You are a cautious, conservative medical triage router. You would rather under-recommend "
        "than over-recommend. You are not a diagnostic tool and never speculate beyond what symptoms "
        "directly support. You do not diagnose or prescribe.\n\n"
        "OBJECTIVE:\n"
        "- Return only specialties with a strong, direct clinical link to the reported symptoms.\n"
        "- Rank specialties by clinical relevance, primary first.\n"
        "- Include secondary specialties only when clearly justified — not just theoretically possible.\n"
        "- If in doubt, output only the primary specialty.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n"
        + _PRIORITY_RULES + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity", "duration", "associated_symptoms"])


def V12(inp):
    sys = (
        "ROLE: You are a thorough medical triage router that considers all clinically plausible specialties "
        "for a presenting complaint, not just the single most likely one. You do not diagnose or prescribe.\n\n"
        "RULES:\n"
        "- Include up to 3 specialties whenever more than one plausible cause exists.\n"
        "- Favor completeness over minimalism, but never include a specialty with no clinical basis at all.\n"
        "- Rank by clinical relevance, primary first.\n\n"
        + _SEVERITY_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity", "duration"])


def V13(inp):
    sys = (
        "ROLE: Medical triage router. Do not diagnose or prescribe.\n\n"
        "INPUT VALIDATION (perform before routing):\n"
        '- If "symptoms" is empty or null, do not attempt routing. '
        'Return specialties: [] and missing_info: ["symptoms"].\n'
        '- If "symptoms" contains only non-medical text (e.g. "I don\'t know what\'s wrong"), '
        "treat as insufficient input and request clarification rather than guessing.\n"
        "- Otherwise proceed with normal routing rules.\n\n"
        + _SEVERITY_RULES + "\n\n" + _NEGATIVE_RULES + "\n\n" + _MISSING_INFO_RULE + "\n\n"
        + _PRIORITY_RULES + "\n\n" + _OUTPUT_CONTRACT
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity"])


def V14(inp):
    sys = (
        "ROLE: Medical triage router. Do not diagnose or prescribe. [R-ROLE]\n\n"
        "RULES:\n"
        "[R1] Return only clinically relevant specialties.\n"
        "[R2] Rank primary first, secondary only if clinically justified.\n"
        "[R3] Emergency red flags override all other routing — Emergency Care becomes priority 1.\n"
        "[R4] Symptoms + severity are primary signals; age/gender secondary; name never used.\n"
        "[R5] Never infer unstated information.\n"
        "[R6] Max 3 specialties.\n"
        "[R7] Request only missing info that would change routing.\n"
        "[R8] No diagnosis, no prescriptions, no invented doctors.\n\n"
        'OUTPUT CONTRACT (JSON only):\n{"emergency": bool, "urgency": string, '
        '"specialties": [{"name": string, "priority": number}], '
        '"missing_info": [string], "rules_applied": [string]}'
    )
    return sys, "PATIENT INPUT:\n" + _fmt(inp, ["symptoms", "severity", "duration"])


def V15(inp):
    sys = (
        "You are a medical triage and doctor-specialty routing assistant. "
        "You route patients based on symptoms, condition, and severity. "
        "You do not diagnose or prescribe treatment.\n\n"
        + _PRIORITY_RULES + "\n\n" + _SEVERITY_RULES + "\n\n"
        "INPUT BASIS:\nPrimary: symptoms, medical condition, severity.\n"
        "Secondary: duration, associated symptoms, relevant medical history, age, gender.\n"
        "Identification only (never used for routing): patient name.\n"
        "Do not infer symptoms, medical history, or severity that were not provided.\n\n"
        + _MISSING_INFO_RULE + "\n\n" + _NEGATIVE_RULES + "\n\n" + _OUTPUT_CONTRACT + "\n\n"
        "Respond to the user's message with JSON only. No preamble, no explanation, no markdown."
    )
    return sys, _fmt(inp, ["age", "gender", "symptoms", "severity", "duration", "associated_symptoms"])


VARIANTS = {
    "V01_full_spec":           V01,
    "V02_minimal":             V02,
    "V03_mapping_table":       V03,
    "V04_chain_of_thought":    V04,
    "V05_few_shot":            V05,
    "V06_precision_max":       V06,
    "V07_emergency_gate":      V07,
    "V08_missing_info_cap":    V08,
    "V09_demo_guardrails":     V09,
    "V10_confidence_score":    V10,
    "V11_cautious_persona":    V11,
    "V12_aggressive_coverage": V12,
    "V13_input_validation":    V13,
    "V14_rule_id_format":      V14,
    "V15_sys_user_split":      V15,
}

# =============================================================================
# 4.  QUALITY SCORING  (0-5 pts per test case)
# =============================================================================

def score_output(output: dict, expected: dict) -> dict:
    scores = {}

    # 4.1  Emergency flag correct
    scores["emergency_correct"] = int(bool(output.get("emergency", False)) == expected["emergency"])

    # 4.2  Urgency tier correct
    scores["urgency_correct"] = int(
        str(output.get("urgency", "")).lower() == expected["urgency"].lower()
    )

    # 4.3  Priority-1 specialty correct
    specialties = output.get("specialties", []) or []
    p1 = None
    for sp in specialties:
        if isinstance(sp, dict) and sp.get("priority") == 1:
            p1 = sp.get("name", "")
            break
    if p1 is None and specialties:
        first = specialties[0]
        p1 = first.get("name", "") if isinstance(first, dict) else str(first)

    acceptable = expected["acceptable_p1"]
    if acceptable:
        p1_ok = any(
            acc.lower() in (p1 or "").lower() or (p1 or "").lower() in acc.lower()
            for acc in acceptable
        )
    else:
        # TC-10: expect empty specialties list
        p1_ok = len(specialties) == 0

    scores["priority1_correct"] = int(p1_ok)

    # 4.4  No over-recommendation
    scores["no_hallucination"] = int(len(specialties) <= max(expected["max_specialties"], 0))

    # 4.5  Missing info appropriate
    missing = output.get("missing_info", []) or []
    if expected["missing_info_expected"]:
        scores["missing_info_ok"] = int(len(missing) > 0)
    else:
        scores["missing_info_ok"] = int(len(missing) <= 3)

    scores["total"] = sum(v for k, v in scores.items() if k != "total")
    scores["p1_name"] = p1 or ""
    scores["n_specialties"] = len(specialties)
    return scores


# =============================================================================
# 5.  RUNNER
# =============================================================================

def call_api(client: OpenAI, system_p: str, user_p: str) -> dict:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_p},
            {"role": "user",   "content": user_p},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    lat_ms = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"_parse_error": raw[:300]}

    return {
        "output": parsed,
        "raw": raw,
        "prompt_tokens":     resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens":      resp.usage.total_tokens,
        "latency_ms":        round(lat_ms, 1),
    }


def run_benchmark(api_key: str) -> list:
    client  = OpenAI(api_key=api_key)
    results = []
    total   = len(VARIANTS) * len(TEST_CASES)

    print(f"\n{'='*65}")
    print(f"  Arogya AI — Triage Prompt Variant Benchmark")
    print(f"  {len(VARIANTS)} variants x {len(TEST_CASES)} test cases = {total} API calls")
    print(f"{'='*65}\n")

    for v_name, builder in VARIANTS.items():
        v_records = []
        print(f"VARIANT: {v_name}")
        for tc in TEST_CASES:
            sys_p, usr_p = builder(tc["input"])
            try:
                data    = call_api(client, sys_p, usr_p)
                quality = score_output(data["output"], tc["expected"])
                data["quality"] = quality
                data["error"]   = None
            except Exception as exc:
                data = {
                    "output": {}, "raw": "",
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "latency_ms": 0,
                    "quality": {"total": 0, "emergency_correct": 0, "urgency_correct": 0,
                                "priority1_correct": 0, "no_hallucination": 0,
                                "missing_info_ok": 0, "p1_name": "", "n_specialties": 0},
                    "error": str(exc),
                }

            rec = {"variant": v_name, "tc_id": tc["id"], "tc_label": tc["label"], **data}
            v_records.append(rec)
            results.append(rec)

            q   = data["quality"]["total"]
            ico = "✓" if q >= 4 else ("~" if q >= 2 else "✗")
            err = f"  [ERR: {data['error'][:50]}]" if data["error"] else ""
            print(
                f"  {ico} {tc['id']}  score={q}/5  "
                f"tok={data['total_tokens']}  lat={data['latency_ms']:.0f}ms{err}"
            )
            time.sleep(0.35)   # ~3 req/s — well within rate limits

        n  = len(v_records)
        aq = statistics.mean(r["quality"]["total"] for r in v_records)
        at = statistics.mean(r["total_tokens"] for r in v_records)
        al = statistics.mean(r["latency_ms"] for r in v_records)
        print(f"  -> avg score={aq:.2f}/5  avg_tok={at:.0f}  avg_lat={al:.0f}ms\n")

    return results


# =============================================================================
# 6.  STATS + REPORT
# =============================================================================

def compute_stats(results: list) -> list:
    from collections import defaultdict
    by_v: dict = defaultdict(list)
    for r in results:
        by_v[r["variant"]].append(r)

    stats = []
    for v, recs in by_v.items():
        n   = len(recs)
        def _avg(key):      return statistics.mean(r[key] for r in recs)
        def _qavg(key):     return sum(r["quality"].get(key, 0) for r in recs)
        lats = sorted(r["latency_ms"] for r in recs)

        s = {
            "variant":             v,
            "n":                   n,
            "avg_quality":         round(_avg("total_tokens") * 0 + statistics.mean(r["quality"]["total"] for r in recs), 2),
            "pct_quality":         round(sum(r["quality"]["total"] for r in recs) / (n * 5) * 100, 1),
            "avg_tokens":          round(_avg("total_tokens")),
            "avg_prompt_tok":      round(_avg("prompt_tokens")),
            "avg_completion_tok":  round(_avg("completion_tokens")),
            "avg_latency_ms":      round(_avg("latency_ms")),
            "p95_latency_ms":      round(lats[max(0, int(n * 0.95) - 1)]),
            "emergency_acc":       round(_qavg("emergency_correct") / n * 100, 1),
            "priority1_acc":       round(_qavg("priority1_correct") / n * 100, 1),
            "urgency_acc":         round(_qavg("urgency_correct")   / n * 100, 1),
            "no_halluc_pct":       round(_qavg("no_hallucination")  / n * 100, 1),
            "missing_info_acc":    round(_qavg("missing_info_ok")   / n * 100, 1),
            "errors":              sum(1 for r in recs if r.get("error")),
        }
        s["efficiency"] = round(s["avg_quality"] / max(s["avg_tokens"], 1) * 1000, 4)
        stats.append(s)

    stats.sort(key=lambda x: (-x["avg_quality"], x["avg_tokens"]))
    return stats


def _get(stats, vname, key, default="N/A"):
    for s in stats:
        if s["variant"] == vname:
            return s.get(key, default)
    return default


def generate_report(results: list, stats: list) -> str:
    best       = stats[0]
    low_tok    = min(stats, key=lambda x: x["avg_tokens"])
    low_lat    = min(stats, key=lambda x: x["avg_latency_ms"])
    best_emerg = max(stats, key=lambda x: x["emergency_acc"])
    best_p1    = max(stats, key=lambda x: x["priority1_acc"])
    best_eff   = max(stats, key=lambda x: x["efficiency"])

    # Per-TC matrix
    tc_header = "| TC | Label | " + " | ".join(s["variant"][-4:] for s in stats) + " |\n"
    tc_sep    = "|---|---" + "|---" * len(stats) + "|\n"
    by_key    = {(r["variant"], r["tc_id"]): r for r in results}
    tc_rows   = ""
    for tc in TEST_CASES:
        cells = []
        for s in stats:
            rec = by_key.get((s["variant"], tc["id"]))
            if rec:
                q = rec["quality"]["total"]
                cells.append(("✅" if q == 5 else "🟡" if q >= 3 else "❌") + str(q))
            else:
                cells.append("—")
        tc_rows += f"| {tc['id']} | {tc['label'][:38]} | " + " | ".join(cells) + " |\n"

    # Summary table
    sum_rows = ""
    for i, s in enumerate(stats, 1):
        sum_rows += (
            f"| #{i} `{s['variant']}` "
            f"| {s['avg_quality']}/5 ({s['pct_quality']}%) "
            f"| {s['avg_tokens']} "
            f"| {s['avg_latency_ms']} ms "
            f"| {s['emergency_acc']}% "
            f"| {s['priority1_acc']}% "
            f"| {s['urgency_acc']}% "
            f"| {s['no_halluc_pct']}% "
            f"| {s['efficiency']} |\n"
        )

    # Axis comparisons
    def pair(v1, v2, key, higher_is_better=True):
        a, b = _get(stats, v1, key, 0), _get(stats, v2, key, 0)
        winner = v1 if (a > b) == higher_is_better else v2
        return f"{v1}={a} vs {v2}={b} → **winner: {winner}**"

    report = f"""# Arogya AI — Triage Prompt Variant A/B Benchmark Report

> **Model**: `gpt-4o-mini` | **Temp**: 0.0 | **Variants**: {len(stats)} | **Test Cases**: {len(TEST_CASES)} | **Total Calls**: {len(stats) * len(TEST_CASES)}

---

## Executive Summary

| Metric | Winner | Value |
|---|---|---|
| 🏆 Best Overall Quality | `{best['variant']}` | {best['avg_quality']}/5 ({best['pct_quality']}%) |
| ⚡ Lowest Token Cost | `{low_tok['variant']}` | {low_tok['avg_tokens']} avg tokens |
| 🚀 Lowest Latency | `{low_lat['variant']}` | {low_lat['avg_latency_ms']} ms avg |
| 🚨 Best Emergency Detection | `{best_emerg['variant']}` | {best_emerg['emergency_acc']}% |
| 🎯 Best Priority-1 Specialty | `{best_p1['variant']}` | {best_p1['priority1_acc']}% |
| 💡 Best Quality/Token Efficiency | `{best_eff['variant']}` | {best_eff['efficiency']} |

---

## Full Ranked Comparison Table

| Rank & Variant | Avg Quality | Avg Tokens | Avg Latency | Emergency Acc | P1 Acc | Urgency Acc | No-Halluc | Efficiency |
|---|---|---|---|---|---|---|---|---|
{sum_rows}
---

## Per Test-Case Score Matrix

{tc_header}{tc_sep}{tc_rows}
> ✅ = 5/5 perfect | 🟡 = 3–4/5 partial | ❌ = 0–2/5 poor

---

## Axis-by-Axis Analysis

### V01 vs V02 — Full spec vs Minimal
{pair("V01_full_spec", "V02_minimal", "avg_quality")} (quality)
{pair("V01_full_spec", "V02_minimal", "avg_tokens", False)} (tokens — lower wins)

### V03 — Mapping Table — Hallucination Control
No-hallucination rate: **{_get(stats, "V03_mapping_table", "no_halluc_pct")}%** (V01 baseline: {_get(stats, "V01_full_spec", "no_halluc_pct")}%)

### V04 — Chain-of-Thought
Quality: **{_get(stats, "V04_chain_of_thought", "avg_quality")}** | Completion tokens: **{_get(stats, "V04_chain_of_thought", "avg_completion_tok")}** (vs V01: {_get(stats, "V01_full_spec", "avg_completion_tok")})

### V05 — Few-Shot Examples
Quality: **{_get(stats, "V05_few_shot", "avg_quality")}** | Prompt tokens: **{_get(stats, "V05_few_shot", "avg_prompt_tok")}** (vs V01: {_get(stats, "V01_full_spec", "avg_prompt_tok")})

### V06 vs V12 — Precision vs Recall
{pair("V06_precision_max", "V12_aggressive_coverage", "no_halluc_pct")} (no-hallucination — higher wins)
{pair("V06_precision_max", "V12_aggressive_coverage", "priority1_acc")} (P1 accuracy)

### V07 — Emergency Gate
Emergency acc: **{_get(stats, "V07_emergency_gate", "emergency_acc")}%** vs V01: **{_get(stats, "V01_full_spec", "emergency_acc")}%**

### V08 — Missing-Info Cap
Avg missing_info items requested (TC-01/09 only): check raw JSON.
Missing-info accuracy: **{_get(stats, "V08_missing_info_cap", "missing_info_acc")}%** vs V01: **{_get(stats, "V01_full_spec", "missing_info_acc")}%**

### V13 — Input Validation
TC-10 (empty symptoms): score = **{by_key.get(("V13_input_validation", "TC-10"), {}).get("quality", {}).get("total", "N/A")}**/5

### V15 — System/User Split
Quality: **{_get(stats, "V15_sys_user_split", "avg_quality")}** | Tokens: **{_get(stats, "V15_sys_user_split", "avg_tokens")}**

---

## Production Recommendation

### 🏆 Best Overall
`{best['variant']}` — {best['avg_quality']}/5 quality, {best['pct_quality']}% accuracy.

### ⚡ Token-Constrained Production
`{low_tok['variant']}` — {low_tok['avg_tokens']} avg tokens, quality={_get(stats, low_tok["variant"], "avg_quality")}/5.

### 💡 Recommended (Best Quality-Per-Token)
`{best_eff['variant']}` — efficiency={best_eff['efficiency']}, quality={best_eff['avg_quality']}/5, tokens={best_eff['avg_tokens']}.

### 🚨 Emergency-Critical Routing
`{best_emerg['variant']}` — {best_emerg['emergency_acc']}% emergency detection accuracy.

---

## All Variants — Quality Ranking
{chr(10).join(f"{i+1}. `{s['variant']}` — {s['avg_quality']}/5 ({s['pct_quality']}%)" for i, s in enumerate(stats))}

## All Variants — Token Efficiency Ranking (fewest tokens first)
{chr(10).join(f"{i+1}. `{s['variant']}` — {s['avg_tokens']} tokens" for i, s in enumerate(sorted(stats, key=lambda x: x["avg_tokens"])))}

## All Variants — Latency Ranking (fastest first)
{chr(10).join(f"{i+1}. `{s['variant']}` — {s['avg_latency_ms']} ms" for i, s in enumerate(sorted(stats, key=lambda x: x["avg_latency_ms"])))}

---
*Generated by `symptom_triage/benchmark_variants.py` — Arogya AI*
"""
    return report


# =============================================================================
# 7.  MAIN
# =============================================================================

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "\n[ERROR] OPENAI_API_KEY not found.\n"
            "Set it in Arogya-backend/.env or export it as an environment variable.\n"
        )

    results = run_benchmark(api_key)

    out_dir     = Path(__file__).parent
    json_path   = out_dir / "benchmark_results.json"
    report_path = out_dir / "variant_comparison_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Raw results  -> {json_path}")

    stats  = compute_stats(results)
    report = generate_report(results, stats)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Report       -> {report_path}")
    print("\n" + "="*65)
    print(report[:3000])
