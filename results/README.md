# Arogya AI — Triage Benchmark & Architecture Results

This directory contains the experimental results, benchmark data, and implementation audit for the research paper:

> **"A Deterministic Hybrid Neuro-Symbolic Architecture for Clinical Symptom Triage and Specialty Routing"**  
> *System Specification v2.0 — Arogya AI / CongoCare Platform*

---

## Directory Contents

| File | Description |
|---|---|
| [`benchmark_summary.json`](./benchmark_summary.json) | Complete machine-readable benchmark dataset covering all 15 prompt variants across 12 clinical presentations (180 total model evaluations). |
| [`variant_metrics.csv`](./variant_metrics.csv) | Tabular comparison of all 15 prompt architectures across Quality, Token count, Latency, Accuracies, and Token Efficiency. |
| [`test_case_evaluations.csv`](./test_case_evaluations.csv) | Detailed breakdown of the 12 clinical benchmark presentations (TC01–TC12), ground truth acuity, and deterministic pipeline behavior. |
| [`triage_benchmark_report.md`](./triage_benchmark_report.md) | In-depth technical report explaining all formulas, metrics, statistics, and clinical findings. |
| [`implementation_audit.md`](./implementation_audit.md) | Line-by-line verification matrix proving that each paper specification (5-stage pipeline, SR-01..SR-08, CF-01..CF-07, doctor matching, emergency dispatch) is implemented in this codebase. |

---

## Key Experimental Findings

1. **Winning Production Variant**: `V10_confidence_score` (Quality: **4.83 / 5.0**, Tokens: **492**, Latency: **1,780 ms**, Emergency Detection: **100.0%**, P1 Accuracy: **91.7%**, Urgency Accuracy: **91.7%**, No-Hallucination: **100.0%**).
2. **Zero-Trust Safety Guarantee**: 100% emergency recall achieved across all 15 variants via the Stage 4 deterministic red-flag lexical scanner (`CF-04`), overriding any LLM under-triage.
3. **Chain-of-Thought Failure**: Variant `V04` increased latency by **85.7%** (to 3,306 ms) while degrading clinical quality (4.50 vs 4.83) and introducing hallucinations.
4. **Necessity of Urgency Rubric**: Prompts omitting explicit rubric criteria (`V02`, `V14`) plummeted to an **unacceptable 16.7% urgency accuracy**, proving LLMs cannot self-calibrate clinical urgency.
