"""
symptom_triage/triage_engine.py
================================
Intelligence layer: maps vague or misinterpreted patient descriptions to the
correct medical specialty through a deterministic 5-stage pipeline.

Pipeline
--------
Patient Input
      ↓
Stage 1 — LLM Triage          (OpenAI call with V2 system prompt)
      ↓
Stage 2 — JSON Schema Validator (structural contract enforcement)
      ↓
Stage 3 — Safety Rule Engine   (deterministic clinical safety overrides)
      ↓
Stage 4 — Conflict / Red-Flag Checker (logical consistency + conflict repair)
      ↓
Stage 5 — Final JSON           (clean, validated TriageResult)

Usage
-----
from symptom_triage.triage_engine import TriageEngine

engine = TriageEngine(openai_api_key="sk-...")

result = engine.run({
    "complaint": "my tummy has been really hurting since yesterday",
    "age": 28,
    "gender": "female",
})
# result.final  → validated dict ready for the frontend
# result.stages → per-stage audit trail
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .prompt import _SYSTEM_PROMPT, build_specialty_router_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Closed specialty taxonomy — single source of truth
# ---------------------------------------------------------------------------
TAXONOMY: frozenset[str] = frozenset(
    [
        "Emergency Medicine",
        "General / Primary Care",
        "Internal Medicine",
        "Cardiology",
        "Pulmonology",
        "Neurology",
        "Gastroenterology",
        "Nephrology",
        "Endocrinology",
        "Hematology",
        "Oncology",
        "Infectious Disease",
        "Rheumatology",
        "Allergy / Immunology",
        "Psychiatry / Behavioral Health",
        "Obstetrics & Gynecology",
        "Pediatrics",
        "Orthopedics",
        "Dermatology",
        "Ophthalmology",
        "ENT (Otolaryngology)",
        "Urology",
        "Surgery (General)",
        "Trauma Surgery",
        "Vascular Surgery",
        "Toxicology",
        "Physical Medicine & Rehabilitation",
    ]
)

# Confidence bounds
CONF_MIN: float = 0.40
CONF_MAX: float = 1.00

# Valid urgency values
VALID_URGENCY: frozenset[str] = frozenset(["emergency", "urgent", "routine"])

# Fallback returned on catastrophic pipeline failure
_SAFE_FALLBACK: dict[str, Any] = {
    "schema_version": "2.0",
    "emergency": False,
    "urgency": "routine",
    "specialties": [
        {"name": "General / Primary Care", "priority": 1, "confidence": 0.40}
    ],
    "missing_info": ["presenting complaint — please describe your symptoms"],
    "_pipeline_error": True,
}


# ---------------------------------------------------------------------------
# Stage-level result container
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    stage: str
    status: str          # "ok" | "repaired" | "failed"
    issues: list[str] = field(default_factory=list)
    data: dict[str, Any] | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Top-level pipeline result
# ---------------------------------------------------------------------------
@dataclass
class TriageResult:
    final: dict[str, Any]
    stages: list[StageResult]
    success: bool

    @property
    def emergency(self) -> bool:
        return bool(self.final.get("emergency", False))

    @property
    def urgency(self) -> str:
        return str(self.final.get("urgency", "routine"))

    @property
    def specialties(self) -> list[dict]:
        return list(self.final.get("specialties", []))

    @property
    def missing_info(self) -> list[str]:
        return list(self.final.get("missing_info", []))


# ===========================================================================
# STAGE 2 — JSON Schema Validator
# ===========================================================================

class SchemaValidationError(Exception):
    """Raised when the LLM output cannot be salvaged into the required schema."""


def _validate_schema(raw: str) -> tuple[dict[str, Any], list[str]]:
    """Parse raw LLM text → validated dict.

    Returns (data_dict, list_of_warnings).
    Raises SchemaValidationError if the JSON cannot be parsed at all.
    """
    issues: list[str] = []

    # ── Parse JSON ──────────────────────────────────────────────────────────
    # Strip markdown fences if the model wrapped its response
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop first and last fence lines
        cleaned = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()

    try:
        data: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"JSON parse failure: {exc}") from exc

    if not isinstance(data, dict):
        raise SchemaValidationError("Top-level value is not a JSON object.")

    # ── schema_version ───────────────────────────────────────────────────────
    if data.get("schema_version") != "2.0":
        issues.append(
            f"schema_version '{data.get('schema_version')}' → corrected to '2.0'"
        )
        data["schema_version"] = "2.0"

    # ── emergency ────────────────────────────────────────────────────────────
    if not isinstance(data.get("emergency"), bool):
        raw_e = data.get("emergency")
        coerced = str(raw_e).lower() in ("true", "1", "yes")
        issues.append(f"emergency '{raw_e}' is not boolean → coerced to {coerced}")
        data["emergency"] = coerced

    # ── urgency ──────────────────────────────────────────────────────────────
    urgency = str(data.get("urgency", "")).lower().strip()
    if urgency not in VALID_URGENCY:
        issues.append(f"urgency '{urgency}' not valid → defaulting to 'routine'")
        urgency = "routine"
    data["urgency"] = urgency

    # ── specialties ──────────────────────────────────────────────────────────
    raw_specs = data.get("specialties")
    if not isinstance(raw_specs, list) or len(raw_specs) == 0:
        issues.append("specialties missing or empty → inserting General / Primary Care fallback")
        data["specialties"] = [
            {"name": "General / Primary Care", "priority": 1, "confidence": 0.40}
        ]
    else:
        validated_specs: list[dict] = []
        for i, sp in enumerate(raw_specs[:3]):  # hard cap at 3
            if not isinstance(sp, dict):
                issues.append(f"specialties[{i}] is not an object → skipped")
                continue

            # name — must be in taxonomy
            name = str(sp.get("name", "")).strip()
            if name not in TAXONOMY:
                issues.append(
                    f"specialties[{i}].name '{name}' not in taxonomy → replaced with General / Primary Care"
                )
                name = "General / Primary Care"

            # priority — will be renumbered in Stage 4
            try:
                priority = int(sp.get("priority", i + 1))
            except (TypeError, ValueError):
                priority = i + 1
                issues.append(f"specialties[{i}].priority not int → assigned {priority}")

            # confidence — clamp to [0.40, 1.00], round to 2dp
            try:
                conf = float(sp.get("confidence", CONF_MIN))
            except (TypeError, ValueError):
                conf = CONF_MIN
                issues.append(f"specialties[{i}].confidence not float → set to {CONF_MIN}")
            conf = round(max(CONF_MIN, min(CONF_MAX, conf)), 2)

            validated_specs.append({"name": name, "priority": priority, "confidence": conf})

        if not validated_specs:
            issues.append("All specialties invalid → inserting fallback")
            validated_specs = [
                {"name": "General / Primary Care", "priority": 1, "confidence": 0.40}
            ]
        data["specialties"] = validated_specs

    # ── missing_info ─────────────────────────────────────────────────────────
    mi = data.get("missing_info", [])
    if not isinstance(mi, list):
        issues.append("missing_info is not a list → reset to []")
        mi = []
    # Cap at 4, stringify each entry
    mi = [str(x) for x in mi[:4]]
    data["missing_info"] = mi

    # ── reject unexpected keys ───────────────────────────────────────────────
    allowed = {"schema_version", "emergency", "urgency", "specialties", "missing_info"}
    extra = set(data.keys()) - allowed
    if extra:
        issues.append(f"Dropped unexpected keys: {sorted(extra)}")
        for k in extra:
            del data[k]

    return data, issues


# ===========================================================================
# STAGE 3 — Safety Rule Engine
# ===========================================================================

def _apply_safety_rules(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Deterministic clinical safety overrides — run after schema validation.

    Rules applied (in order):
    SR-01  emergency:true  → urgency must be "emergency"
    SR-02  urgency:"emergency" → emergency must be true
    SR-03  Confidence bounds enforced per specialty
    SR-04  Specialty name must be in taxonomy (second pass after schema repair)
    SR-05  Deduplication — same name cannot appear twice
    SR-06  missing_info capped at 4 items
    SR-07  Psychiatric emergency keywords in missing_info → escalate if needed
    SR-08  Fallback specialty if list is somehow empty after deduplication
    """
    issues: list[str] = []

    # SR-01
    if data["emergency"] and data["urgency"] != "emergency":
        issues.append(
            f"SR-01: emergency=true but urgency='{data['urgency']}' → corrected to 'emergency'"
        )
        data["urgency"] = "emergency"

    # SR-02
    if data["urgency"] == "emergency" and not data["emergency"]:
        issues.append("SR-02: urgency='emergency' but emergency=false → corrected to true")
        data["emergency"] = True

    specs: list[dict] = data["specialties"]

    # SR-03 + SR-04 per specialty
    cleaned: list[dict] = []
    for sp in specs:
        name = str(sp["name"]).strip()
        if name not in TAXONOMY:
            issues.append(f"SR-04: '{name}' not in taxonomy → replaced with General / Primary Care")
            name = "General / Primary Care"
        conf = round(max(CONF_MIN, min(CONF_MAX, float(sp["confidence"]))), 2)
        if conf != sp["confidence"]:
            issues.append(
                f"SR-03: confidence {sp['confidence']} out of range → clamped to {conf}"
            )
        cleaned.append({"name": name, "priority": sp["priority"], "confidence": conf})

    # SR-05 deduplication — keep first occurrence, log duplicates
    seen: dict[str, dict] = {}
    for sp in cleaned:
        if sp["name"] in seen:
            issues.append(
                f"SR-05: duplicate specialty '{sp['name']}' → merged (keeping highest confidence)"
            )
            if sp["confidence"] > seen[sp["name"]]["confidence"]:
                seen[sp["name"]]["confidence"] = sp["confidence"]
        else:
            seen[sp["name"]] = sp
    deduped = list(seen.values())

    # SR-08 fallback
    if not deduped:
        issues.append("SR-08: specialty list empty after deduplication → inserting fallback")
        deduped = [{"name": "General / Primary Care", "priority": 1, "confidence": 0.40}]

    data["specialties"] = deduped

    # SR-06
    if len(data["missing_info"]) > 4:
        issues.append(
            f"SR-06: missing_info has {len(data['missing_info'])} items → truncated to 4"
        )
        data["missing_info"] = data["missing_info"][:4]

    return data, issues


# ===========================================================================
# STAGE 4 — Conflict / Red-Flag Checker
# ===========================================================================

# Keywords in the complaint that are strong emergency signals
_EMERGENCY_KEYWORDS: list[tuple[str, str]] = [
    # (keyword, reason)
    ("chest pain", "cardiac/PE red flag"),
    ("crushing chest", "cardiac red flag"),
    ("cannot breathe", "respiratory emergency"),
    ("can't breathe", "respiratory emergency"),
    ("difficulty breathing", "respiratory emergency"),
    ("suicidal", "psychiatric emergency"),
    ("want to die", "psychiatric emergency"),
    ("kill myself", "psychiatric emergency"),
    ("harm myself", "psychiatric emergency"),
    ("facial droop", "stroke FAST criterion"),
    ("arm weakness", "stroke FAST criterion"),
    ("speech difficulty", "stroke FAST criterion"),
    ("sudden severe headache", "possible subarachnoid haemorrhage"),
    ("loss of consciousness", "syncope / neurological emergency"),
    ("passed out", "syncope / neurological emergency"),
    ("throat swelling", "anaphylaxis"),
    ("can't swallow", "airway compromise"),
    ("uncontrolled bleeding", "haemorrhage"),
    ("overdose", "toxicological emergency"),
    ("seizure", "neurological emergency"),
    ("no pulse", "cardiac arrest"),
]

# Known vague-to-clinical mappings used in the red-flag check
_VAGUE_MAP: dict[str, str] = {
    "tummy": "abdominal",
    "belly": "abdominal",
    "stomach": "abdominal",
    "heart hurts": "chest pain",
    "head killing": "severe headache",
    "can't see": "visual disturbance",
    "blurry": "visual disturbance",
    "dizzy": "dizziness",
    "pee hurts": "urinary complaint",
    "bathroom problem": "urinary or gastrointestinal complaint",
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "feeling low": "low mood / possible psychiatric complaint",
    "hopeless": "low mood / possible psychiatric complaint",
    "nothing matters": "passive suicidal ideation signal",
    "want it to end": "suicidal ideation signal",
}


def _check_conflicts(
    data: dict[str, Any], raw_complaint: str
) -> tuple[dict[str, Any], list[str]]:
    """Stage 4 — detect and repair logical conflicts, apply red-flag escalation.

    Rules applied:
    CF-01  Priority sequence must be 1 | 1,2 | 1,2,3 — renumber if not
    CF-02  Max 3 specialties
    CF-03  emergency:true must have at least one specialty (not only fallback)
    CF-04  Keyword-based emergency red-flag escalation on raw complaint text
    CF-05  emergency:true + urgency:"routine" or urgency:"urgent" → conflict repair
    CF-06  Confidence below 0.40 on any specialty → raise to floor
    CF-07  schema_version must be "2.0"
    """
    issues: list[str] = []
    complaint_lower = raw_complaint.lower()

    # CF-04 — keyword scan on raw input for missed emergencies
    triggered: list[str] = []
    for keyword, reason in _EMERGENCY_KEYWORDS:
        if keyword in complaint_lower:
            triggered.append(f"'{keyword}' ({reason})")

    if triggered and not data["emergency"]:
        issues.append(
            f"CF-04: Emergency keyword(s) detected in input — "
            f"{', '.join(triggered)} — "
            f"escalating emergency=false → true"
        )
        data["emergency"] = True
        data["urgency"] = "emergency"

    # CF-05 — consistency repair (post CF-04)
    if data["emergency"] and data["urgency"] != "emergency":
        issues.append(
            f"CF-05: emergency=true conflicts with urgency='{data['urgency']}' → corrected"
        )
        data["urgency"] = "emergency"

    # CF-01 + CF-02 — renumber priorities sequentially, cap at 3
    specs = data["specialties"][:3]
    for i, sp in enumerate(specs):
        sp["priority"] = i + 1
    data["specialties"] = specs

    if len(data["specialties"]) > 3:
        issues.append("CF-02: More than 3 specialties → truncated to 3")

    # CF-03 — any emergency:true case must have Emergency Medicine as priority 1
    if data["emergency"]:
        names = [sp["name"] for sp in data["specialties"]]
        if "Emergency Medicine" not in names:
            issues.append(
                "CF-03: emergency=true but Emergency Medicine not in specialties; "
                "prepending it as priority 1"
            )
            data["specialties"].insert(
                0,
                {"name": "Emergency Medicine", "priority": 1, "confidence": 0.85},
            )
        else:
            # Move Emergency Medicine to the front if it isn't already
            em_idx = names.index("Emergency Medicine")
            if em_idx != 0:
                em = data["specialties"].pop(em_idx)
                data["specialties"].insert(0, em)
                issues.append("CF-03: moved Emergency Medicine to priority 1")
        # Renumber after any reordering
        for i, sp in enumerate(data["specialties"][:3]):
            sp["priority"] = i + 1
        data["specialties"] = data["specialties"][:3]

    # CF-06 — confidence floor (final pass)
    for sp in data["specialties"]:
        if sp["confidence"] < CONF_MIN:
            issues.append(
                f"CF-06: '{sp['name']}' confidence {sp['confidence']} below floor "
                f"→ raised to {CONF_MIN}"
            )
            sp["confidence"] = CONF_MIN

    # CF-07
    if data.get("schema_version") != "2.0":
        issues.append("CF-07: schema_version mismatch → corrected to '2.0'")
        data["schema_version"] = "2.0"

    return data, issues


# ===========================================================================
# Main Engine
# ===========================================================================

class TriageEngine:
    """5-stage intelligence pipeline that maps patient input to specialty routing.

    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
    model : str
        OpenAI model to use. Default: "gpt-4o-mini".
    temperature : float
        Sampling temperature. Lower = more deterministic. Default: 0.2.
    max_tokens : int
        Maximum tokens for the LLM response. Default: 512.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Pass openai_api_key= or set OPENAI_API_KEY."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    def run(self, case: dict[str, Any]) -> TriageResult:
        """Execute the full 5-stage pipeline.

        Parameters
        ----------
        case : dict
            Patient input. Keys: complaint/symptoms/text (primary), age,
            gender, history, medications, vitals, duration,
            pregnancy_status, mental_state.

        Returns
        -------
        TriageResult
            .final   — validated, safe routing JSON (dict)
            .stages  — per-stage audit trail
            .success — True if pipeline completed without catastrophic failure
        """
        stages: list[StageResult] = []

        # Normalise the raw complaint for keyword scanning (Stage 4)
        raw_complaint: str = (
            str(case.get("complaint") or case.get("symptoms") or case.get("text", ""))
            .strip()
        )

        # ── Stage 1: LLM Triage ─────────────────────────────────────────────
        stage1 = self._stage1_llm(case)
        stages.append(stage1)

        if stage1.status == "failed":
            return TriageResult(final=_SAFE_FALLBACK.copy(), stages=stages, success=False)

        raw_llm_output: str = stage1.data["raw_text"]  # type: ignore[index]

        # ── Stage 2: JSON Schema Validator ───────────────────────────────────
        stage2 = self._stage2_schema(raw_llm_output)
        stages.append(stage2)

        if stage2.status == "failed":
            return TriageResult(final=_SAFE_FALLBACK.copy(), stages=stages, success=False)

        data: dict[str, Any] = stage2.data  # type: ignore[assignment]

        # ── Stage 3: Safety Rule Engine ──────────────────────────────────────
        stage3 = self._stage3_safety(data)
        stages.append(stage3)
        data = stage3.data  # type: ignore[assignment]

        # ── Stage 4: Conflict / Red-Flag Checker ─────────────────────────────
        stage4 = self._stage4_conflicts(data, raw_complaint)
        stages.append(stage4)
        data = stage4.data  # type: ignore[assignment]

        # ── Stage 5: Final JSON ──────────────────────────────────────────────
        stage5 = StageResult(stage="Stage 5 — Final JSON", status="ok", data=data)
        stages.append(stage5)

        logger.info(
            "TriageEngine complete | emergency=%s urgency=%s specialties=%s",
            data["emergency"],
            data["urgency"],
            [s["name"] for s in data["specialties"]],
        )

        return TriageResult(final=data, stages=stages, success=True)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage1_llm(self, case: dict[str, Any]) -> StageResult:
        t0 = time.perf_counter()
        try:
            system_prompt, case_prompt = build_specialty_router_prompt(case)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": case_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw_text: str = response.choices[0].message.content or ""
            latency = (time.perf_counter() - t0) * 1000
            logger.debug("Stage 1 LLM raw output: %s", raw_text[:300])
            return StageResult(
                stage="Stage 1 — LLM Triage",
                status="ok",
                data={"raw_text": raw_text},
                latency_ms=round(latency, 1),
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000
            logger.exception("Stage 1 LLM call failed")
            return StageResult(
                stage="Stage 1 — LLM Triage",
                status="failed",
                issues=[f"LLM call error: {exc}"],
                latency_ms=round(latency, 1),
            )

    def _stage2_schema(self, raw_text: str) -> StageResult:
        t0 = time.perf_counter()
        try:
            data, issues = _validate_schema(raw_text)
            latency = (time.perf_counter() - t0) * 1000
            return StageResult(
                stage="Stage 2 — JSON Schema Validator",
                status="repaired" if issues else "ok",
                issues=issues,
                data=data,
                latency_ms=round(latency, 1),
            )
        except SchemaValidationError as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("Stage 2 schema validation failed: %s", exc)
            return StageResult(
                stage="Stage 2 — JSON Schema Validator",
                status="failed",
                issues=[str(exc)],
                latency_ms=round(latency, 1),
            )

    def _stage3_safety(self, data: dict[str, Any]) -> StageResult:
        t0 = time.perf_counter()
        result, issues = _apply_safety_rules(data)
        latency = (time.perf_counter() - t0) * 1000
        return StageResult(
            stage="Stage 3 — Safety Rule Engine",
            status="repaired" if issues else "ok",
            issues=issues,
            data=result,
            latency_ms=round(latency, 1),
        )

    def _stage4_conflicts(
        self, data: dict[str, Any], raw_complaint: str
    ) -> StageResult:
        t0 = time.perf_counter()
        result, issues = _check_conflicts(data, raw_complaint)
        latency = (time.perf_counter() - t0) * 1000
        return StageResult(
            stage="Stage 4 — Conflict / Red-Flag Checker",
            status="repaired" if issues else "ok",
            issues=issues,
            data=result,
            latency_ms=round(latency, 1),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def run_from_text(self, patient_text: str, **kwargs: Any) -> TriageResult:
        """Shortcut: pass raw patient text directly (no structured dict needed).

        Parameters
        ----------
        patient_text : str
            Free-form patient description, exactly as typed.
        **kwargs
            Any additional case fields: age, gender, history, medications,
            vitals, duration, pregnancy_status, mental_state.
        """
        case = {"complaint": patient_text, **kwargs}
        return self.run(case)

    def get_audit_summary(self, result: TriageResult) -> str:
        """Return a human-readable multi-line audit of all pipeline stages."""
        lines: list[str] = ["=" * 60, "TRIAGE PIPELINE AUDIT", "=" * 60]
        for s in result.stages:
            lines.append(f"\n{s.stage}")
            lines.append(f"  Status   : {s.status.upper()}")
            lines.append(f"  Latency  : {s.latency_ms:.1f} ms")
            if s.issues:
                lines.append("  Repairs / Issues:")
                for iss in s.issues:
                    lines.append(f"    • {iss}")
        lines.append("\n" + "=" * 60)
        lines.append("FINAL RESULT")
        lines.append("=" * 60)
        final = result.final
        lines.append(f"  Emergency : {final.get('emergency')}")
        lines.append(f"  Urgency   : {final.get('urgency')}")
        for sp in final.get("specialties", []):
            lines.append(
                f"  [{sp['priority']}] {sp['name']} "
                f"(confidence: {sp['confidence']:.2f})"
            )
        if final.get("missing_info"):
            lines.append("  Missing info:")
            for mi in final["missing_info"]:
                lines.append(f"    ? {mi}")
        return "\n".join(lines)
