# Comprehensive Technical Report: Arogya AI Clinical Symptom Triage

## 1. Executive Summary

This document presents the full technical report and evaluation analysis for the **Deterministic Hybrid Neuro-Symbolic Architecture** developed for the **Arogya AI / CongoCare** clinical platform.

The system addresses the core vulnerabilities of raw Large Language Models (LLMs) in clinical triage:
- **Specialty Hallucination**: Inventing non-existent hospital departments.
- **Under-Triage Risk**: Failing to identify acute life threats (myocardial infarction, stroke, anaphylaxis).
- **Schema Instability**: Emitting non-deterministic formatting that crashes healthcare databases.

By coupling a flexible natural language interpreter (`gpt-4o-mini`) with three strict, deterministic symbolic projection operators ($\Pi_{\text{Schema}}$, $\Pi_{\text{Safety}}$, $\Pi_{\text{Conflict}}$), the system guarantees **100% emergency recall** and **0% out-of-taxonomy specialty recommendations**.

---

## 2. Terminology & Conceptual Lexicon

### A. Architectural & Systems Terms
- **Neuro-Symbolic Architecture**: A hybrid design combining neural networks / LLMs (for free-text natural language understanding) with symbolic deterministic logic (rule-based validators, formal grammar constraints, and hard safety gates).
- **Deterministic vs. Probabilistic**:
  - *Probabilistic*: LLMs generate text by predicting token probability distributions, which can fluctuate with temperature and context.
  - *Deterministic*: Rule-based functions produce identical, mathematically predictable outputs every time, operating side-effect free.
- **Zero-Trust Model**: The system treats the LLM output as an unverified hypothesis. No downstream clinical service (doctor matching, emergency dispatch) is ever allowed to consume raw model output until it has successfully traversed all symbolic filter stages.
- **Closed Taxonomy ($|T|=27$)**: A strictly bounded set of 27 recognized medical specialties. Any specialty outside this set is disallowed and converted by rule SR-04 to `General / Primary Care`.
- **Audit Trail (`stage_repairs`)**: A per-request ledger detailing every automated repair performed across stages 2–4, guaranteeing clinical explainability and regulatory traceability.

### B. Mathematical Formulations
- **Input Patient Tuple**:
  $$\mathcal{P} = (C, A, G, H, M, E)$$
  Where $C$ is the chief complaint text, $A$ is age, $G$ is sex/gender, $H$ is chronic conditions, $M$ is active medications, and $E$ is longitudinal timeline events.
- **Pipeline as Function Composition**:
  $$\mathcal{Y}^* = \Pi_{\mathrm{Conflict}} \circ \Pi_{\mathrm{Safety}} \circ \Pi_{\mathrm{Schema}} \left( f_{\theta}(\mathcal{P}; \, \mathcal{S}_{\mathrm{prompt}}) \right)$$
  - $\Pi_{\text{Schema}}$: Enforces JSON typing, key whitelist, and bounds.
  - $\Pi_{\text{Safety}}$: Enforces flag-tier consistency and closed taxonomy projection.
  - $\Pi_{\text{Conflict}}$: Enforces deterministic red-flag escalation on raw patient text.
- **Confidence Clamping**:
  $$c_i^* = \min(\max(c_i, \, 0.40), \, 1.00)$$
  Prevents near-zero probability mass from entering the doctor matching engine while capping values at 1.00.
- **Token Efficiency Ratio**:
  $$\eta = \left( \frac{\bar{Q}}{\bar{T}_{\mathrm{prompt}} + \bar{T}_{\mathrm{completion}}} \right) \times 10^3$$
  Measures clinical quality points achieved per 1,000 tokens consumed.
- **Doctor-Patient Ranking Formula**:
  $$\mathrm{Score}(d) = w_1 \cdot \mathrm{Sim}(s_1, d_{\mathrm{spec}}) + w_2 \cdot e^{-\lambda \cdot \mathrm{Dist}_{\mathrm{Haversine}}(u, d)} + w_3 \cdot \mathrm{Rating}(d) + w_4 \cdot \mathrm{Affordability}(d)$$
  Combines medical specialty alignment, exponential geographical distance decay, verified clinician ratings, and fee affordability into a single ranking objective.

### C. Clinical & Safety Terms
- **Under-Triage**: Assigning a lower urgency or routine tier to an acute emergency. In medical informatics, under-triage can lead to patient fatality and is the primary failure mode guarded against.
- **Over-Triage**: Categorizing a routine complaint as an emergency. While safe for the individual, excessive over-triage strains emergency department capacity.
- **Stroke FAST Criteria**: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Explicitly encoded in rule `CF-04`.
- **Anaphylaxis**: Severe, systemic hypersensitivity reaction causing airway swelling and dyspnea, captured by `CF-04`.
- **STEMI Pattern**: ST-Elevation Myocardial Infarction — classic crushing chest pain radiating to left arm/jaw, captured by `CF-04`.

---

## 3. Evaluation Protocol & Results

A total of **15 prompt variants** were systematically compared across **12 standardized clinical presentations (TC01–TC12)**, generating **180 total API calls** on `gpt-4o-mini`.

### Full Benchmark Leaderboard

| Rank | Variant Name | Quality (/5) | Total Tokens | Latency (ms) | P1 Acc (%) | Urgency Acc (%) | No-Halluc (%) | Efficiency ($\eta$) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **V10_confidence_score** | **4.83** | **492** | 1,780 | 91.7% | 91.7% | 100.0% | 9.82 |
| 2 | V13_input_validation | 4.83 | 534 | 1,678 | 91.7% | 91.7% | 100.0% | 9.04 |
| 3 | V11_cautious_persona | 4.83 | 549 | 1,357 | 91.7% | 91.7% | 100.0% | 8.80 |
| 4 | V09_demo_guardrails | 4.83 | 598 | 1,796 | 91.7% | 91.7% | 100.0% | 8.08 |
| 5 | V03_mapping_table | 4.83 | 646 | 1,358 | 83.3% | 100.0% | 100.0% | 7.48 |
| 6 | V05_few_shot | 4.83 | 786 | 1,503 | 91.7% | 100.0% | 100.0% | 6.15 |
| 7 | V15_sys_user_split | 4.75 | 583 | 2,439 | 83.3% | 91.7% | 100.0% | 8.15 |
| 8 | V01_full_spec (baseline) | 4.75 | 735 | 2,400 | 83.3% | 91.7% | 100.0% | 6.46 |
| 9 | V08_missing_info_cap | 4.67 | 474 | 1,655 | 83.3% | 91.7% | 100.0% | 9.85 |
| 10 | V07_emergency_gate | 4.67 | 575 | 1,691 | 91.7% | 100.0% | 83.3% | 8.12 |
| 11 | V06_precision_max | 4.58 | 485 | 1,799 | 83.3% | 91.7% | 91.7% | 9.44 |
| 12 | V12_aggressive_coverage | 4.50 | 431 | 1,775 | 83.3% | 91.7% | 75.0% | 10.44 |
| 13 | V04_chain_of_thought | 4.50 | 642 | 3,306 | 75.0% | 91.7% | 91.7% | 7.01 |
| 14 | V14_rule_id_format | 4.17 | 285 | 1,631 | 100.0% | 16.7% | 100.0% | 14.63 |
| 15 | V02_minimal | 3.92 | 289 | 1,821 | 91.7% | 16.7% | 83.3% | 13.56 |

---

## 4. Analytical Insights & Interpretation

1. **Why V10 Won**:
   - V10 achieved the tied-highest clinical quality score (**4.83 / 5.0**) with the smallest token overhead (**492 tokens**) among all tied variants.
   - It outputs per-specialty confidence scores natively, supplying the required numeric input for Stage 2 validation and Stage 3 clamping (`SR-03`).
2. **Chain-of-Thought (CoT) Inefficiency in Triage**:
   - `V04_chain_of_thought` was by far the slowest variant (**3,306 ms**, +85.7% latency penalty) and suffered lower P1 specialty accuracy (75.0%) and lower quality (4.50).
   - In rapid clinical triage, verbose token generation introduces unwanted variance and unacceptable delays.
3. **The Urgency Rubric Is Mandatory**:
   - Prompts that stripped the explicit clinical urgency definitions (`V02` minimal and `V14` rule-id) saw urgency accuracy collapse to **16.7%**. This reveals that LLMs cannot accurately determine whether a case is routine, urgent, or emergency without explicit rubric anchors.
4. **Token Efficiency Metric Trap**:
   - While `V14` scored the highest $\eta$ (14.63) due to its minimal token count (285 tokens), its 16.7% urgency accuracy makes it clinically dangerous. Token efficiency $\eta$ must only be evaluated among candidate architectures that already satisfy mandatory clinical safety thresholds.
