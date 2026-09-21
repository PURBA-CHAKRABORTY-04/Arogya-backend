# Arogya AI — Triage Prompt Variant A/B Benchmark Report

> **Model**: `gpt-4o-mini` | **Temp**: 0.0 | **Variants**: 15 | **Test Cases**: 12 | **Total Calls**: 180

---

## Executive Summary

| Metric | Winner | Value |
|---|---|---|
| 🏆 Best Overall Quality | `V10_confidence_score` | 4.83/5 (96.7%) |
| ⚡ Lowest Token Cost | `V14_rule_id_format` | 285 avg tokens |
| 🚀 Lowest Latency | `V11_cautious_persona` | 1357 ms avg |
| 🚨 Best Emergency Detection | `V10_confidence_score` | 100.0% |
| 🎯 Best Priority-1 Specialty | `V14_rule_id_format` | 100.0% |
| 💡 Best Quality/Token Efficiency | `V14_rule_id_format` | 14.6316 |

---

## Full Ranked Comparison Table

| Rank & Variant | Avg Quality | Avg Tokens | Avg Latency | Emergency Acc | P1 Acc | Urgency Acc | No-Halluc | Efficiency |
|---|---|---|---|---|---|---|---|---|
| #1 `V10_confidence_score` | 4.83/5 (96.7%) | 492 | 1780 ms | 100.0% | 91.7% | 91.7% | 100.0% | 9.8171 |
| #2 `V13_input_validation` | 4.83/5 (96.7%) | 534 | 1678 ms | 100.0% | 91.7% | 91.7% | 100.0% | 9.0449 |
| #3 `V11_cautious_persona` | 4.83/5 (96.7%) | 549 | 1357 ms | 100.0% | 91.7% | 91.7% | 100.0% | 8.7978 |
| #4 `V09_demo_guardrails` | 4.83/5 (96.7%) | 598 | 1796 ms | 100.0% | 91.7% | 91.7% | 100.0% | 8.0769 |
| #5 `V03_mapping_table` | 4.83/5 (96.7%) | 646 | 1358 ms | 100.0% | 83.3% | 100.0% | 100.0% | 7.4768 |
| #6 `V05_few_shot` | 4.83/5 (96.7%) | 786 | 1503 ms | 100.0% | 91.7% | 100.0% | 100.0% | 6.145 |
| #7 `V15_sys_user_split` | 4.75/5 (95.0%) | 583 | 2439 ms | 100.0% | 83.3% | 91.7% | 100.0% | 8.1475 |
| #8 `V01_full_spec` | 4.75/5 (95.0%) | 735 | 2400 ms | 100.0% | 83.3% | 91.7% | 100.0% | 6.4626 |
| #9 `V08_missing_info_cap` | 4.67/5 (93.3%) | 474 | 1655 ms | 100.0% | 83.3% | 91.7% | 100.0% | 9.8523 |
| #10 `V07_emergency_gate` | 4.67/5 (93.3%) | 575 | 1691 ms | 100.0% | 91.7% | 100.0% | 83.3% | 8.1217 |
| #11 `V06_precision_max` | 4.58/5 (91.7%) | 485 | 1799 ms | 100.0% | 83.3% | 91.7% | 91.7% | 9.4433 |
| #12 `V12_aggressive_coverage` | 4.5/5 (90.0%) | 431 | 1775 ms | 100.0% | 83.3% | 91.7% | 75.0% | 10.4408 |
| #13 `V04_chain_of_thought` | 4.5/5 (90.0%) | 642 | 3306 ms | 100.0% | 75.0% | 91.7% | 91.7% | 7.0093 |
| #14 `V14_rule_id_format` | 4.17/5 (83.3%) | 285 | 1631 ms | 100.0% | 100.0% | 16.7% | 100.0% | 14.6316 |
| #15 `V02_minimal` | 3.92/5 (78.3%) | 289 | 1821 ms | 100.0% | 91.7% | 16.7% | 83.3% | 13.564 |

---

## Per Test-Case Score Matrix

| TC | Label | core | tion | sona | ails | able | shot | plit | spec | _cap | gate | _max | rage | ught | rmat | imal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TC-01 | Crushing chest pain — STEMI pattern | 🟡4 | 🟡4 | 🟡4 | 🟡4 | 🟡4 | ✅5 | 🟡4 | 🟡4 | 🟡4 | ✅5 | ✅5 | 🟡4 | 🟡4 | 🟡4 | 🟡3 |
| TC-02 | Knee pain — orthopaedic, routine | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | 🟡4 |
| TC-03 | Headache + vision changes | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 |
| TC-04 | Abdominal pain + fever + vomiting | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 |
| TC-05 | Tooth pain — dentistry only | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | ✅5 | 🟡4 | 🟡3 |
| TC-06 | Severe SOB + chest tightness + cyanosi | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | 🟡4 |
| TC-07 | Anaphylaxis — throat swelling + dyspne | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | ✅5 | 🟡4 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | ✅5 | 🟡4 | 🟡4 |
| TC-08 | Urinary burning — urology routine | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | ✅5 | 🟡4 | ✅5 | 🟡4 | 🟡3 |
| TC-09 | Stroke red-flag — facial droop + slurr | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡4 | 🟡4 |
| TC-10 | Empty symptoms — input validation | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡3 | ✅5 | 🟡3 | 🟡4 | 🟡4 |
| TC-11 | Vague complaint — insufficient input | ✅5 | ✅5 | ✅5 | ✅5 | ✅5 | 🟡3 | ✅5 | 🟡4 | 🟡3 | ❌2 | 🟡3 | 🟡4 | 🟡3 | 🟡4 | 🟡4 |
| TC-12 | Inflammatory joint pain — rheumatology | 🟡4 | 🟡4 | 🟡4 | 🟡4 | ✅5 | ✅5 | 🟡4 | 🟡4 | 🟡4 | ✅5 | 🟡4 | 🟡4 | 🟡4 | 🟡4 | 🟡4 |

> ✅ = 5/5 perfect | 🟡 = 3–4/5 partial | ❌ = 0–2/5 poor

---

## Axis-by-Axis Analysis

### V01 vs V02 — Full spec vs Minimal
V01_full_spec=4.75 vs V02_minimal=3.92 → **winner: V01_full_spec** (quality)
V01_full_spec=735 vs V02_minimal=289 → **winner: V02_minimal** (tokens — lower wins)

### V03 — Mapping Table — Hallucination Control
No-hallucination rate: **100.0%** (V01 baseline: 100.0%)

### V04 — Chain-of-Thought
Quality: **4.5** | Completion tokens: **150** (vs V01: 61)

### V05 — Few-Shot Examples
Quality: **4.83** | Prompt tokens: **728** (vs V01: 674)

### V06 vs V12 — Precision vs Recall
V06_precision_max=91.7 vs V12_aggressive_coverage=75.0 → **winner: V06_precision_max** (no-hallucination — higher wins)
V06_precision_max=83.3 vs V12_aggressive_coverage=83.3 → **winner: V12_aggressive_coverage** (P1 accuracy)

### V07 — Emergency Gate
Emergency acc: **100.0%** vs V01: **100.0%**

### V08 — Missing-Info Cap
Avg missing_info items requested (TC-01/09 only): check raw JSON.
Missing-info accuracy: **91.7%** vs V01: **100.0%**

### V13 — Input Validation
TC-10 (empty symptoms): score = **5**/5

### V15 — System/User Split
Quality: **4.75** | Tokens: **583**

---

## Production Recommendation

### 🏆 Best Overall
`V10_confidence_score` — 4.83/5 quality, 96.7% accuracy.

### ⚡ Token-Constrained Production
`V14_rule_id_format` — 285 avg tokens, quality=4.17/5.

### 💡 Recommended (Best Quality-Per-Token)
`V14_rule_id_format` — efficiency=14.6316, quality=4.17/5, tokens=285.

### 🚨 Emergency-Critical Routing
`V10_confidence_score` — 100.0% emergency detection accuracy.

---

## All Variants — Quality Ranking
1. `V10_confidence_score` — 4.83/5 (96.7%)
2. `V13_input_validation` — 4.83/5 (96.7%)
3. `V11_cautious_persona` — 4.83/5 (96.7%)
4. `V09_demo_guardrails` — 4.83/5 (96.7%)
5. `V03_mapping_table` — 4.83/5 (96.7%)
6. `V05_few_shot` — 4.83/5 (96.7%)
7. `V15_sys_user_split` — 4.75/5 (95.0%)
8. `V01_full_spec` — 4.75/5 (95.0%)
9. `V08_missing_info_cap` — 4.67/5 (93.3%)
10. `V07_emergency_gate` — 4.67/5 (93.3%)
11. `V06_precision_max` — 4.58/5 (91.7%)
12. `V12_aggressive_coverage` — 4.5/5 (90.0%)
13. `V04_chain_of_thought` — 4.5/5 (90.0%)
14. `V14_rule_id_format` — 4.17/5 (83.3%)
15. `V02_minimal` — 3.92/5 (78.3%)

## All Variants — Token Efficiency Ranking (fewest tokens first)
1. `V14_rule_id_format` — 285 tokens
2. `V02_minimal` — 289 tokens
3. `V12_aggressive_coverage` — 431 tokens
4. `V08_missing_info_cap` — 474 tokens
5. `V06_precision_max` — 485 tokens
6. `V10_confidence_score` — 492 tokens
7. `V13_input_validation` — 534 tokens
8. `V11_cautious_persona` — 549 tokens
9. `V07_emergency_gate` — 575 tokens
10. `V15_sys_user_split` — 583 tokens
11. `V09_demo_guardrails` — 598 tokens
12. `V04_chain_of_thought` — 642 tokens
13. `V03_mapping_table` — 646 tokens
14. `V01_full_spec` — 735 tokens
15. `V05_few_shot` — 786 tokens

## All Variants — Latency Ranking (fastest first)
1. `V11_cautious_persona` — 1357 ms
2. `V03_mapping_table` — 1358 ms
3. `V05_few_shot` — 1503 ms
4. `V14_rule_id_format` — 1631 ms
5. `V08_missing_info_cap` — 1655 ms
6. `V13_input_validation` — 1678 ms
7. `V07_emergency_gate` — 1691 ms
8. `V12_aggressive_coverage` — 1775 ms
9. `V10_confidence_score` — 1780 ms
10. `V09_demo_guardrails` — 1796 ms
11. `V06_precision_max` — 1799 ms
12. `V02_minimal` — 1821 ms
13. `V01_full_spec` — 2400 ms
14. `V15_sys_user_split` — 2439 ms
15. `V04_chain_of_thought` — 3306 ms

---
*Generated by `symptom_triage/benchmark_variants.py` — Arogya AI*
