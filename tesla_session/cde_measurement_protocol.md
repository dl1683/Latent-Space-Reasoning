# CDE v1: Measurement Contract, Operator Portfolio, and Selector Protocol

## Status: DESIGN PHASE — Codex Review Required

## Origin
Codex Wave 1 + Wave 2 Reviews, Priority Directive:
> "The next document should be: 'Controlled Decorrelation Ensemble v1: Measurement Contract, Operator Portfolio, and Selector Protocol.'"

This document specifies the complete measurement infrastructure for CDE. Nothing is built until this measurement contract is satisfied.

---

## 1. Trace Schema

Every candidate generation MUST produce a trace record. This is the fundamental data unit of CDE.

### Trace Record Fields

```json
{
  "trace_id": "uuid4",
  "timestamp": "ISO-8601",
  "task_id": "arith_001",
  "task_domain": "arithmetic|legal|planning|logic",
  "task_text": "What is 7 × 8?",
  "task_ground_truth": "56",
  
  "operator": "prefix|temperature|nucleus|rephrase|decompose|symbolic|retrieval",
  "operator_config": {
    "type": "prefix",
    "token_count": 2,
    "rms_scale": 1.0,
    "seed": 3,
    "temperature": null,
    "top_p": null,
    "template_id": null
  },
  
  "generation": {
    "raw_output": "Let me calculate 7 × 8 = 56. The answer is 56.",
    "token_count": 42,
    "max_tokens_hit": false,
    "generation_time_ms": 1520,
    "peak_memory_mb": 14200
  },
  
  "fingerprint": {
    "first_16_token_ids": [1432, 567, ...],
    "first_32_token_ids": [1432, 567, ..., 892],
    "first_32_embedding": [0.12, -0.34, ...],
    "mean_logprob_first_32": -1.23,
    "attn_entropy_layer16_mean": 2.45,
    "fingerprint_hash": "sha256_of_first_32_tokens"
  },
  
  "evaluation": {
    "extracted_answer": "56",
    "answer_anywhere_correct": true,
    "last_integer_correct": true,
    "self_certainty_score": 3.45,
    "repetition_rate": 0.02,
    "degenerate": false
  },
  
  "selectors": {
    "exact_match": {"selected": true, "confidence": 1.0},
    "majority_vote": {"selected": true, "confidence": 0.8},
    "self_certainty": {"selected": false, "confidence": 0.6},
    "length_heuristic": {"selected": true, "confidence": 0.7}
  }
}
```

### Storage
- Primary: `experiments/cde_traces.jsonl` (append-only)
- Index: `experiments/cde_trace_index.json` (task_id → list of trace_ids)
- Summary: `experiments/cde_operator_summary.json` (per-operator aggregate metrics)

### Non-Negotiable Fields
Every trace MUST have: trace_id, task_id, operator, operator_config, raw_output, extracted_answer, last_integer_correct, first_32_token_ids, mean_logprob_first_32, generation_time_ms. Without these, the trace is INCOMPLETE and must be regenerated.

---

## 2. Operator Portfolio

### Tier 1: Core Operators (MUST measure first)

| ID | Name | Config | Surface | Cost | Notes |
|---|---|---|---|---|---|
| O1 | Greedy Baseline | do_sample=False, no prefix | - | 1.0x | The control. One output per task. |
| O2 | Random Soft Prefix | 2 tokens, RMS-matched, seeds 0-9 | Embedding | 1.0x | Our primary method. |
| O3 | Zero Soft Prefix | 2 zero-valued tokens | Embedding | 1.0x | Position-shift control. |
| O4 | Temperature 0.3 | do_sample=True, T=0.3, seeds 0-9 | Decoding | 1.0x | Low-entropy stochastic. |
| O5 | Temperature 0.6 | do_sample=True, T=0.6, seeds 0-9 | Decoding | 1.0x | Medium-entropy stochastic. |
| O6 | Temperature 1.0 | do_sample=True, T=1.0, seeds 0-9 | Decoding | 1.0x | High-entropy stochastic. |
| O7 | Nucleus 0.6/0.9 | T=0.6, top_p=0.9, seeds 0-9 | Decoding | 1.0x | Practical sampling config. |
| O8 | Prompt Rephrase | 10 hand-written templates | Tokens | 1.0x | Linguistic diversity. |

### Tier 2: Extended Operators (Measure after Tier 1)

| ID | Name | Config | Surface | Cost | Notes |
|---|---|---|---|---|---|
| O9 | Random Token Prefix | 2 random vocab tokens, seeds 0-9 | Tokens | 1.0x | Discrete version of O2. |
| O10 | Position Shift | Position IDs += k, k=1..10 | Position | 1.0x | Pure position effect. |
| O11 | DDC (Decomposed) | Decompose + solve sub-problems | Prompt | 3-5x | Architecture 9. |
| O12 | Neuro-Symbolic | Translate to Python + execute | Symbolic | 1.5x | Architecture 14. |
| O13 | RARS (Self-Retrieve) | Elicit knowledge + ground | Prompt | 3-5x | Architecture 10. |

### Tier 3: Advanced Operators (Only after evidence from Tier 1-2)

| ID | Name | Config | Surface | Cost | Notes |
|---|---|---|---|---|---|
| O14 | ALM-Guided Prefix | Basin map → targeted seed | Embedding | 1.0x* | Architecture 13. *After mapping. |
| O15 | Causal Perturbation | Bottleneck features only | Hidden | 1.1x | Architecture 11. |
| O16 | Attention Surgery | Sink-reduction masks | Attention | 1.3x | Architecture 16. |
| O17 | GGIO | Gradient-optimized prefix | Embedding | 20x | Architecture 8. |

---

## 3. Measurement Contract

### Phase 1: Operator Characterization (Tier 1)

**What**: Generate traces for all 8 Tier 1 operators on the calibration task set.

**Task set**: 25 arithmetic tasks (same as existing experiments).

**N per operator**: 10 (except O1 which produces 1 deterministic output).

**For stochastic operators (O4-O7)**: 3 independent panels of N=10 each (seeds 0-9, 10-19, 20-29) to assess variance.

**Total generations**:
- O1: 25 × 1 = 25
- O2: 25 × 10 = 250 (reuse existing data if trace schema is populated)
- O3: 25 × 10 = 250
- O4-O7: 25 × 10 × 3 panels = 750 each → 3000
- O8: 25 × 10 = 250
- **Total: 3,775 generations**
- **Estimated time**: ~6-8 hours GPU at ~6-8 sec/generation

### Per-Operator Metrics (computed from traces)

**M1: Individual Accuracy**
```
A(O) = (1 / (|tasks| × N)) × Σ_t Σ_s correct(t, s)
```
Report: mean ± bootstrap 95% CI (1000 iterations over tasks).

**M2: Oracle Accuracy**
```
oracle(O, N) = (1 / |tasks|) × Σ_t [max_s correct(t, s)]
```
Report: mean across panels (for stochastic operators), ± SD.

**M3: Within-Operator Error Correlation**
For each seed pair (i, j), compute Pearson correlation between binary error vectors (length = |tasks|):
```
rho(O) = mean_{i≠j} corr(error_i, error_j)
```
where error_s[t] = 1 if candidate s got task t wrong.
Report: mean rho ± SD across all seed pairs.

**M4: Trajectory Class Count**
Embed first-32-token outputs using cosine similarity.
Cluster using DBSCAN (eps=0.3, min_samples=2) or hierarchical clustering.
```
K(O, N) = number of clusters
```
Report: K ± SD across panels.

**M5: Effective Yield**
```
Y(O) = 1 - (truncated + degenerate) / total_candidates
```
where degenerate = repetition_rate > 0.3 OR output < 10 tokens.

**M6: Cost**
```
C(O) = mean_generation_time_ms(O) / mean_generation_time_ms(O1)
```

### Cross-Operator Metrics (computed from traces)

**M7: Correct-Set Jaccard**
For operators O_a, O_b:
```
correct_a = {t : any seed of O_a correct on t}
correct_b = {t : any seed of O_b correct on t}
J(O_a, O_b) = |correct_a ∩ correct_b| / |correct_a ∪ correct_b|
```
Report: full K×K Jaccard matrix (K = number of operators).

**M8: Cross-Operator Error Correlation**
For operator pair (O_a, O_b), randomly pair seeds and compute:
```
rho_cross(O_a, O_b) = mean_{(i,j)} corr(error_O_a_i, error_O_b_j)
```
Report: full K×K correlation matrix.

**M9: Trajectory Overlap**
For operator pair (O_a, O_b):
```
overlap(O_a, O_b) = |trajectories_a ∩ trajectories_b| / |trajectories_a ∪ trajectories_b|
```
where trajectory membership is defined by cluster ID from pooled clustering.

### Decision Gates

**Gate 1: Is multi-operator worthwhile?**
- IF max(J(O_a, O_b)) < 0.8 for any pair: YES — operators solve different problems
- IF all J > 0.9: NO — all operators solve the same problems. Use the best single operator.
- **If NO**: Report this as a finding. The paper story becomes "random prefix is as good as any other operator for our tasks."

**Gate 2: Which operators are complementary?**
- Rank operator pairs by: low Jaccard + both have A > 0.2 + both have Y > 0.7
- Top 3 pairs are candidates for the ensemble

**Gate 3: Is the selector adequate?**
- IF selector_accuracy > 0.8 × oracle_accuracy: selector is adequate → focus on diversity
- IF selector_accuracy < 0.5 × oracle_accuracy: selector is the bottleneck → invest in selector before diversity
- **This gate determines whether to pursue Pillar 1 (diversity) or Pillar 3 (selection) first**

---

## 4. Selector Protocol

### Selector Registry

| ID | Selector | Input | Cost | Domain |
|---|---|---|---|---|
| S1 | Extract + Exact Match | output text | ~0 | Arithmetic only |
| S2 | Majority Vote | all candidate outputs | ~0 | All (needs extractable answers) |
| S3 | Self-Certainty | mean logprob from trace | ~0 | All |
| S4 | Length Heuristic | output token count | ~0 | All |
| S5 | Degenerate Filter | repetition_rate, token_count | ~0 | All (pre-filter) |
| S6 | LLM Judge | full output text | 10-50x | Open-ended only |
| S7 | Symbolic Verifier | extracted expression + execution | ~0 | Arithmetic, Logic, Code |
| S8 | Constraint Checker | extracted plan + constraint set | ~0 | Planning |

### Selector Stack (Ordered Pipeline)

For each set of N candidates:
1. **S5: Degenerate Filter** — remove truncated, repetitive, empty outputs → N' candidates
2. **S7/S8: Formal Verifier** — if domain is arithmetic/logic/planning, formally verify. Pick verified-correct if any. → done or N' candidates
3. **S2: Majority Vote** — among remaining, pick the most common extracted answer → selected candidate
4. **S3: Self-Certainty** — tiebreaker if majority vote is inconclusive → selected candidate

For open-ended tasks (legal):
1. **S5: Degenerate Filter** → N' candidates
2. **S4: Length Heuristic** — remove extremely short/long outliers → N'' candidates
3. **S6: LLM Judge** — evaluate remaining candidates → selected candidate

### Selector Evaluation Protocol

On the calibration set (25 arithmetic tasks), for each operator's N=10 candidates:
1. Run each selector independently
2. Record: which candidate it selects, whether that candidate is correct
3. Compute:
   - **selector_accuracy(S)** = fraction of tasks where S picks a correct candidate (when one exists)
   - **selector_recall(S)** = fraction of tasks with any correct candidate where S finds one
   - **oracle_gap(S)** = oracle_accuracy - selector_accuracy (lower = better selector)
   - **false_positive_rate(S)** = fraction of tasks where S picks a "confident" incorrect candidate

### Pre-Registered Selector Interpretation

- **S1 (exact match)**: Expected selector_accuracy ≈ oracle_accuracy for arithmetic (near-perfect selector). This is the gold standard — how close do other selectors get?
- **S2 (majority vote)**: Expected to work well when N≥5 and individual accuracy > 0.3. If A < 0.3, majority vote may consistently pick the wrong answer (majority are wrong).
- **S3 (self-certainty)**: Unknown. May correlate with correctness for confident models, or may select confidently wrong candidates.
- **S7 (symbolic verifier)**: Should match S1 for extractable arithmetic. The advantage is for multi-step problems where intermediate verification is possible.

---

## 5. Compute Allocation Protocol

### Static Allocation (Phase 1)

Before ensemble measurement, use uniform allocation:
- For N=10: allocate to the single best Tier 1 operator (from Phase 1 characterization)
- Compare against: 50/50 split between top 2 complementary operators

### Greedy Allocation (Phase 2)

After Phase 1 data is collected, implement the greedy allocator:
```
allocation = {op: 0 for op in viable_operators}
for candidate_idx in range(budget_N):
    marginal_gain = {}
    for op in viable_operators:
        # Expected gain from adding one more candidate from op
        # Uses measured A(op), rho(op), and current coverage
        marginal_gain[op] = (1 - current_coverage) * A(op) * (1 - rho(op))
    best_op = argmax(marginal_gain)
    allocation[best_op] += 1
    current_coverage = estimate_coverage(allocation)
```

### Stopping Rules

**Stop generating more candidates when:**
1. Oracle coverage saturates: oracle(N) - oracle(N-1) < 1 task for 3 consecutive candidates
2. All new candidates duplicate existing basins: 3 consecutive fingerprint collisions
3. Selector confidence saturates: selector picks the same candidate regardless of additional options
4. Budget exhausted: total generation time exceeds wall-clock budget

**Stop adding operators when:**
1. Cross-operator Jaccard > 0.9 (new operator is redundant with existing set)
2. New operator's effective yield Y < 0.5 (too many degenerate candidates)
3. Marginal oracle gain from the new operator < 1 task

---

## 6. Domain Routing

### Arithmetic Tasks
- **Primary operators**: O2 (prefix), O4 (temp 0.3), O8 (rephrase)
- **Primary selector**: S7 (symbolic verifier) → S2 (majority vote)
- **Extended operators**: O12 (neuro-symbolic), O11 (DDC)
- **Measurement**: exact intermediate verification available

### Legal Reasoning Tasks
- **Primary operators**: O2 (prefix), O5 (temp 0.6), O8 (rephrase), O13 (RARS)
- **Primary selector**: S6 (LLM judge)
- **Extended operators**: O11 (DDC, decompose into legal elements)
- **Measurement**: LLM judge is the selector; no exact verifier. Selector audit is CRITICAL.

### Planning Tasks
- **Primary operators**: O2 (prefix), O5 (temp 0.6), O8 (rephrase)
- **Primary selector**: S8 (constraint checker) → S6 (LLM judge)
- **Extended operators**: O27 (CPH, constraint programming hybrid)
- **Measurement**: constraint satisfaction partially verifiable

### Logic/Code Tasks
- **Primary operators**: O2 (prefix), O5 (temp 0.6), O12 (neuro-symbolic)
- **Primary selector**: S7 (symbolic verifier)
- **Extended operators**: O11 (DDC)
- **Measurement**: execution-based verification available

---

## 7. N-Scaling Curves

### Required Measurement

For each Tier 1 operator and the top 2 ensembles, plot:
- X axis: N (number of candidates) = 1, 2, 4, 8, 16
- Y axis: oracle accuracy and selector accuracy

This reveals:
1. **Where oracle saturates**: the point beyond which more candidates don't help
2. **Where selector diverges from oracle**: the point beyond which the selector can't exploit additional candidates
3. **Which operator scales best**: the operator with the steepest N-scaling curve is the best diversity generator
4. **Whether ensemble beats single operator at matched N**: the core CDE thesis

### How to Generate N-Scaling Data Cheaply
- From the Phase 1 data (N=10 per operator), subsample:
  - N=1: single random seed
  - N=2: random pair of seeds
  - N=4: random 4-subset
  - N=8: random 8-subset
  - N=16: need to generate 6 more (or use all 10 and bootstrap)
- Bootstrap: for each N, sample 100 random subsets and compute mean/CI
- This gives the full N-scaling curve from EXISTING data (no new generations needed for N ≤ 10)

---

## 8. Evidence Requirements (Codex-Mandated)

From Codex Wave 2 review, 10 specific evidence requirements:

| # | Requirement | How CDE Measurement Protocol Addresses It |
|---|---|---|
| 1 | Prefix vs temp vs rephrase under equal compute | Phase 1: all Tier 1 operators at equal N=10 |
| 2 | Pairwise error-correlation matrix | Metric M8: full K×K cross-operator correlation |
| 3 | Oracle-selector gap | Selector Protocol: selector_accuracy vs oracle_accuracy |
| 4 | First-32-token clustering → correctness | Trace schema includes fingerprint; analyze cluster → correct correlation |
| 5 | N-scaling curves: N=1,2,4,8,16 | Section 7: subsampled from Phase 1 data |
| 6 | Domain split | Section 6: separate measurement per domain |
| 7 | Basin cluster stability (for ALM) | Phase 2: extend with ALM measurement after Phase 1 |
| 8 | Mechanism probes for CIR/Attention Surgery | Gated attention probe experiment (already designed) |
| 9 | Formalization error rate for neuro-symbolic | Phase 2: Tier 2 operator characterization |
| 10 | Fact hallucination rate for RARS | Phase 2: Tier 2 operator characterization |

### Phase 1 Satisfies Requirements 1-6
All core evidence can be derived from Phase 1 traces without additional experiments. This is the critical insight: the measurement protocol is designed so that ONE experimental run (3,775 generations) answers ALL fundamental questions.

### Phase 2 Satisfies Requirements 7-10
Extended operators (Tier 2) require additional generation but build on Phase 1 infrastructure.

---

## 9. Relationship to Existing Experimental Designs

### Temperature Comparison (temperature_comparison_design.md)
- IS Phase 1 of CDE measurement
- NEEDS ADDITIONS: trace schema, fingerprints, cross-operator metrics, selector audit
- Should be UPGRADED, not replaced

### Gated Attention Probe (gated_attention_probe_design.md)
- IS evidence requirement #8
- Independent of CDE Phase 1
- Can run in parallel or after

### Phase A Blueprint (phase6_blueprint.md)
- Implements operator O2 (random soft prefix) within the CDE framework
- Component 0 (probe) → validates inputs_embeds compatibility
- Component 1 (SpecGenerator) → generates prefix embeddings for operator O2
- Component 2 (Applicator) → applies prefix and generates candidates
- Blueprint IS the O2 operator implementation, now situated within CDE

---

## 10. Success Criteria for CDE v1

### Minimum Viable CDE
The framework is validated if:
1. Phase 1 measurements are complete for all Tier 1 operators
2. At least ONE operator pair has Jaccard < 0.7 (complementary)
3. 2-operator ensemble beats best single operator at N=10 by ≥ 2 tasks
4. Selector accuracy > 60% of oracle accuracy for the best selector

### Full CDE
The framework is publishable if:
1. Minimum Viable CDE criteria met
2. N-scaling curves show clear ensemble advantage
3. Domain routing produces different optimal allocations for arithmetic vs legal
4. The CDE framework reproduces all existing results (no regression from switching to the framework)

### CDE Paper Story
If all criteria are met, the paper story is:
1. We propose the Controlled Decorrelation Ensemble framework for inference-time reasoning
2. We show that operator decorrelation, not just candidate count, drives oracle improvement
3. Random soft prefix perturbation is one operator; combined with temperature and prompt rephrasing, it produces measurably complementary candidates
4. We provide the measurement contract (trace schema, metrics, decision gates) as a tool for evaluating ANY inference-time intervention
5. For our calibration tasks, the optimal allocation is [X% prefix + Y% temperature + Z% rephrase]
