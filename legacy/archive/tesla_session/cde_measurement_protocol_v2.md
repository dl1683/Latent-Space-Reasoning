# CDE v2: Measurement Contract, Operator Portfolio, and Selector Protocol

## Status: REVISED — Addressing Codex R1 Feedback

## Changes From v1 (Codex R1 Corrections Applied)

1. **Primary endpoint**: selected accuracy (not oracle) using deployable selectors
2. **Trace schema**: added run_id, candidate_set_id, panel_id, selection_trace, model identity, full generation config, operator artifact hashes
3. **Decision gates**: Fixed Gate 1 (min Jaccard, not max). Added relative viability. Primary gate on selected accuracy.
4. **Operator portfolio**: Random Token Prefix + Position Shift moved to Tier 1. Zero Soft Prefix is a control (N=1).
5. **Selector**: Operator-stratified consensus replaces raw majority vote. Separation of oracle vs deployable selectors.
6. **N-scaling**: Requires N=16+ actual generations, not subsampling. Bootstrap over tasks.
7. **Held-out split**: 25 tasks is pilot. Validation requires held-out tasks or held-out seed panels.
8. **Success criteria**: Primary = selected accuracy beats best single operator under equal compute.
9. **Allocator**: Greedy A*(1-rho) is a heuristic. Paired with empirical validation, not used as sole estimator.

---

## 1. Primary Endpoint (NON-NEGOTIABLE)

**CDE succeeds if and only if:**

> Under equal compute budget, a deployable selector chooses a better answer from a decorrelated operator mix than from the best single operator.

**Selected accuracy** = fraction of tasks where the DEPLOYABLE selector picks a correct answer.

**Oracle accuracy** = fraction of tasks where ANY candidate in the set is correct. This is a DIAGNOSTIC ceiling, not a success metric.

**Equal compute**: same total generated-token budget (not same N, because operators may differ in cost).

---

## 2. Trace Schema (Revised)

### Candidate Trace Record

```json
{
  "trace_id": "uuid4",
  "run_id": "cde_phase1_20260415",
  "candidate_set_id": "set_arith001_prefix_panel0",
  "panel_id": 0,
  "candidate_index": 3,
  "timestamp": "ISO-8601",
  
  "task": {
    "task_id": "arith_001",
    "domain": "arithmetic",
    "text": "What is 7 × 8?",
    "ground_truth": "56",
    "rendered_prompt": "What is 7 × 8?"
  },
  
  "model": {
    "name": "Qwen3-4B",
    "checkpoint": "Qwen/Qwen3-4B",
    "quantization": "Q4_K_M",
    "tokenizer_hash": "sha256:abc...",
    "library": "transformers==4.48.0",
    "git_commit": "fb139d1"
  },
  
  "operator": {
    "type": "prefix",
    "config": {
      "token_count": 2,
      "rms_scale": 1.0,
      "seed": 3
    },
    "artifact_hash": "sha256:prefix_tensor_hash",
    "artifact_rms": 1.023,
    "artifact_norm": 2.89
  },
  
  "generation_config": {
    "max_new_tokens": 1024,
    "do_sample": false,
    "temperature": null,
    "top_k": null,
    "top_p": null,
    "repetition_penalty": null,
    "seed_state": "manual_seed_3"
  },
  
  "generation": {
    "raw_output": "Let me calculate...",
    "output_token_ids": [1432, 567, ...],
    "token_count": 42,
    "max_tokens_hit": false,
    "generation_time_ms": 1520,
    "peak_memory_mb": 14200
  },
  
  "fingerprint": {
    "first_16_token_ids": [1432, 567, ...],
    "first_32_token_ids": [1432, 567, ..., 892],
    "first_32_embedding_hash": "sha256:embed_hash",
    "first_32_embedding_file": "artifacts/embeddings/arith001_prefix_s3.npy",
    "mean_logprob_first_32": -1.23,
    "attn_entropy_layer16_mean": 2.45,
    "fingerprint_hash": "sha256_of_first_32_tokens"
  },
  
  "evaluation": {
    "evaluator_version": "harness_v2.1",
    "extraction_method": "last_integer_wins",
    "extracted_answer": "56",
    "normalized_answer": "56",
    "answer_anywhere_correct": true,
    "last_integer_correct": true,
    "self_certainty_score": 3.45,
    "repetition_rate": 0.02,
    "degenerate": false,
    "is_oracle": false
  }
}
```

### Selection Trace Record (SET-LEVEL)

```json
{
  "selection_trace_id": "uuid4",
  "run_id": "cde_phase1_20260415",
  "candidate_set_id": "set_arith001_ensemble_panel0",
  "task_id": "arith_001",
  "timestamp": "ISO-8601",
  
  "candidate_traces": ["trace_001", "trace_002", ..., "trace_010"],
  "allocation": {"prefix": 6, "temperature_0.6": 4},
  
  "oracle": {
    "any_correct": true,
    "correct_trace_ids": ["trace_001", "trace_005", "trace_008"],
    "oracle_accuracy": 1.0
  },
  
  "selectors": {
    "majority_vote": {
      "type": "deployable",
      "selected_trace_id": "trace_001",
      "selected_answer": "56",
      "correct": true,
      "confidence": 0.8,
      "answer_distribution": {"56": 7, "54": 2, "": 1}
    },
    "operator_stratified_consensus": {
      "type": "deployable",
      "selected_trace_id": "trace_005",
      "selected_answer": "56",
      "correct": true,
      "confidence": 0.85,
      "operator_votes": {"prefix": "56", "temperature": "56"},
      "n_operators_agreeing": 2,
      "abstained": false
    },
    "self_certainty": {
      "type": "deployable",
      "selected_trace_id": "trace_003",
      "selected_answer": "54",
      "correct": false,
      "confidence": 0.72,
      "scores": [3.45, 3.12, 4.01, 2.89, ...]
    },
    "formal_verifier": {
      "type": "deployable_domain",
      "domain": "arithmetic",
      "selected_trace_id": "trace_001",
      "selected_answer": "56",
      "correct": true,
      "verified_count": 7,
      "verification_results": [true, false, true, ...]
    },
    "oracle_exact_match": {
      "type": "evaluation_only",
      "selected_trace_id": "trace_001",
      "correct": true,
      "note": "NOT deployable - requires ground truth"
    }
  },
  
  "primary_metric": {
    "selected_accuracy": 1.0,
    "selector_used": "operator_stratified_consensus",
    "oracle_accuracy": 1.0,
    "oracle_selector_gap": 0.0
  }
}
```

### Storage
- Candidate traces: `experiments/cde_traces.jsonl` (append-only)
- Selection traces: `experiments/cde_selections.jsonl` (append-only)
- Embeddings: `experiments/cde_artifacts/embeddings/` (numpy files, referenced by hash)
- Index: `experiments/cde_index.json`

---

## 3. Operator Portfolio (Revised)

### Tier 1: Core Operators + Required Controls

| ID | Name | Type | N | Config | Notes |
|---|---|---|---|---|---|
| O1 | Greedy Baseline | Control | 1 | do_sample=False, no prefix | Single deterministic output |
| O2 | Random Soft Prefix | Operator | 16 | 2 tokens, RMS-matched | Primary method |
| O3 | Zero Soft Prefix | Control | 1 | 2 zero-valued tokens | Position-shift control |
| O4 | Random Token Prefix | Operator | 16 | 2 random vocab tokens | Discrete version of O2 |
| O5 | Position Shift | Control | 5 | Position IDs += k, k=1..5 | Pure position effect |
| O6 | Temperature 0.6 | Operator | 16×3 panels | do_sample=True, T=0.6 | Medium stochastic |
| O7 | Nucleus 0.6/0.9 | Operator | 16×3 panels | T=0.6, top_p=0.9 | Practical sampling |
| O8 | Prompt Rephrase | Operator | 10 | 10 templates | Linguistic diversity |

**Why N=16 (not N=10)**: Codex mandated N≥16 for valid N-scaling curves. N=16 allows subsampling to 1, 2, 4, 8, 16 without extrapolation.

**Why 3 panels for stochastic**: Seeds 0-15, 16-31, 32-47. Variance assessment across panels.

**Total Phase 1 generations**:
- O1: 25 × 1 = 25
- O2: 25 × 16 = 400
- O3: 25 × 1 = 25
- O4: 25 × 16 = 400
- O5: 25 × 5 = 125
- O6: 25 × 16 × 3 = 1,200
- O7: 25 × 16 × 3 = 1,200
- O8: 25 × 10 = 250
- **Total: 3,625 generations** (~8-10 hours GPU)

### Tier 2: Extended Operators (After Phase 1)

| ID | Name | Config | Notes |
|---|---|---|---|
| O9 | Temperature 0.3 | T=0.3, N=16 | Low stochastic (if T=0.6 is best, explore lower) |
| O10 | Temperature 1.0 | T=1.0, N=16 | High stochastic (if T=0.6 is best, explore higher) |
| O11 | DDC | Decompose + solve | Architecture 9 |
| O12 | Neuro-Symbolic | Python execution | Architecture 14 |
| O13 | RARS | Elicit + ground | Architecture 10 |
| O14 | MMNE | Multi-model swap | Architecture 28 |
| O15 | PDR | Persona × perturbation | Architecture 30 |

### Tier 3: Advanced (After evidence from Tier 1-2)

| ID | Name | Config | Notes |
|---|---|---|---|
| O16 | ALM-Guided | Basin-targeted prefix | Architecture 13 |
| O17 | CIR | Causal bottleneck perturbation | Architecture 11 |
| O18 | GGIO | Gradient-optimized prefix | Architecture 8 |

---

## 4. Selector Protocol (Revised)

### Deployable vs Evaluation-Only Selectors

**CRITICAL DISTINCTION**:
- **Deployable**: can be used at inference time without ground truth
- **Evaluation-only**: requires ground truth (oracle). NEVER used as a success metric.

### Deployable Selectors

**DS1: Degenerate Filter** (pre-filter, not selector)
- Remove candidates with: repetition_rate > 0.3, token_count < 10, max_tokens_hit AND no answer extracted
- Type: pre-filter applied before all other selectors

**DS2: Formal Verifier** (domain-specific)
- Arithmetic: Python eval of extracted expression
- Logic: truth table / SAT solver
- Code: execution test
- Type: deployable where available, highest confidence

**DS3: Operator-Stratified Answer Consensus** (PRIMARY DEPLOYABLE SELECTOR)
```python
class OperatorStratifiedConsensus:
    """
    Codex-mandated selector: normalized answer clusters scored by
    operator diversity, not raw candidate count.
    """
    def select(self, candidates: List[Candidate]) -> Selection:
        # 1. Extract and normalize answers
        answer_clusters = cluster_by_normalized_answer(candidates)
        
        # 2. Score each answer cluster
        for cluster in answer_clusters:
            cluster.score = 0
            operators_supporting = set()
            for candidate in cluster.members:
                operators_supporting.add(candidate.operator)
                # Weight by operator quality prior (from Phase 1 characterization)
                cluster.score += self.operator_quality[candidate.operator]
            
            # Bonus for cross-operator agreement (decorrelation signal)
            cluster.n_operators = len(operators_supporting)
            cluster.diversity_bonus = cluster.n_operators / self.total_operators
            
            # Combined score
            cluster.final_score = cluster.score * (1 + cluster.diversity_bonus)
        
        # 3. Select highest-scoring cluster
        best_cluster = max(answer_clusters, key=lambda c: c.final_score)
        
        # 4. Abstention check
        if best_cluster.final_score < self.abstention_threshold:
            return Selection(abstained=True, fallback='greedy_baseline')
        
        # 5. Within-cluster tiebreaker: highest self-certainty
        selected = max(best_cluster.members, key=lambda c: c.self_certainty)
        
        return Selection(
            selected=selected,
            confidence=best_cluster.final_score,
            n_operators_agreeing=best_cluster.n_operators,
            abstained=False
        )
```

**DS4: Self-Certainty** (tiebreaker only)
- Mean logprob of generated tokens
- Used ONLY as within-cluster tiebreaker, not primary selector

**DS5: LLM Judge** (open-ended tasks only)
- Claude or local model evaluates candidate quality
- Expensive: 10-50x generation cost
- Used only for legal/planning where no formal verifier exists

### Evaluation-Only Selectors (Oracle)

**ES1: Extract + Exact Match** — requires ground truth. Used for measuring oracle accuracy ceiling.
**ES2: Answer-Anywhere Match** — requires ground truth. Diagnostic only.

### Selector Audit Protocol

For each deployable selector, measure on Phase 1 data:
- **Selected accuracy**: P(selected candidate is correct)
- **Precision when positive**: P(correct | selector chose this candidate AND selector didn't abstain)
- **Recall**: P(selector finds a correct candidate | oracle says one exists)
- **False positive rate**: P(selector picks incorrect candidate confidently)
- **Oracle-selector gap by N**: for N=1,2,4,8,16, plot oracle vs selected accuracy
- **Calibration**: ECE or Brier score if selector provides confidence
- **Abstention rate**: fraction of tasks where selector abstains

---

## 5. Measurement Contract (Revised)

### Phase 1 Metrics

**Primary (success/failure determination):**

**PM1: Selected Accuracy Under Equal Compute**
```
selected_acc(ensemble, S) = mean_over_tasks(selector_S_correct(ensemble_candidates))
```
For the primary deployable selector (DS3: operator-stratified consensus).

**PM2: Cost-Normalized Lift**
```
lift = selected_acc(ensemble) - selected_acc(best_single_operator)
```
Under equal total token budget. Positive lift = CDE works.

**PM3: Oracle-Selector Gap**
```
gap = oracle_acc(ensemble) - selected_acc(ensemble)
```
Small gap = selector is adequate. Large gap = selector is the bottleneck.

**Diagnostic (understanding, not success determination):**

**DM1: Individual Accuracy** A(O) per operator
**DM2: Oracle Accuracy** oracle(O, N) per operator
**DM3: Within-Operator Error Correlation** rho(O) per operator
**DM4: Cross-Operator Error Correlation** rho_cross(O_a, O_b) per operator pair
**DM5: Correct-Set Jaccard** J(O_a, O_b) per operator pair
**DM6: Marginal Unique Solves** per operator: tasks solved by O_b that O_a misses
**DM7: Trajectory Class Count** K(O, N) per operator
**DM8: Effective Yield** Y(O) per operator
**DM9: Duplicate-Basin Rate / Fingerprint Collision Rate**
**DM10: Answer-Cluster Diversity**: unique normalized answers, answer entropy
**DM11: Cluster-Correctness Association**: mutual information between trajectory cluster and correctness
**DM12: Extraction Failure/Ambiguity Rate**: disagreement between extraction methods

**Statistical requirements:**
- All CIs: bootstrap over TASKS (1000 iterations), not over candidates
- Paired comparisons: paired permutation test (same tasks)
- Report both point estimate and 95% CI

### Phase 1 Decision Gates (Revised)

**Gate 0: Are controls working?**
- O1 (greedy) produces expected baseline accuracy (~32% for Qwen3-4B Q4)
- O3 (zero prefix) produces similar or slightly different accuracy from O1
- If O3 ≈ O1: position shift alone doesn't explain the effect
- If O3 >> O1: position shift IS the mechanism → changes interpretation

**Gate 1: Do any operators complement each other?**
- Criteria: min(J(O_a, O_b)) ≤ 0.7 for at least one pair of viable operators
- "Viable" = A(O) ≥ 50% of best cheap operator AND Y(O) ≥ 0.85
- Also: rho_cross(O_a, O_b) < rho_within(O_a) for the same pair
- **If NO**: all operators solve the same problems. CDE provides no benefit.

**Gate 2: Does ensemble improve selected accuracy?**
- Ensemble of top complementary pair at equal compute vs best single operator at same compute
- Criteria: selected_acc(ensemble) > selected_acc(best_single) (point estimate)
- Stronger: 95% CI of lift excludes 0
- **This is the PRIMARY go/no-go gate.**

**Gate 3: Is the selector adequate?**
- selected_acc(DS3) ≥ 80% of oracle_acc for arithmetic
- For legal: require separate judge audit
- If NOT: selector improvement is the priority, not diversity improvement

**Gate 4: Is the effect real (not overfitting)?**
- Held-out validation: either (a) additional tasks beyond the 25, or (b) held-out seed panels
- Phase 1 panel structure enables: train on panels 0+1, validate on panel 2

---

## 6. N-Scaling Protocol (Revised)

### Actual Generation at Multiple N

For Tier 1 operators with N=16 seeds, construct N-scaling curves by:
1. **N=1**: single seed (report mean over 16 possible single-seed choices)
2. **N=2**: all C(16,2)=120 pairs, compute mean oracle and selected accuracy
3. **N=4**: 200 random 4-subsets
4. **N=8**: 200 random 8-subsets
5. **N=16**: all seeds

For ensembles: generate actual allocations at each target N (not post-hoc mixing):
- N=4 ensemble: 2 prefix + 2 temperature (pre-registered allocation)
- N=8 ensemble: 4 prefix + 4 temperature
- N=16 ensemble: 8 prefix + 8 temperature (or proportional to Phase 1 characterization)

**Bootstrap**: over TASKS as the primary uncertainty unit. Report mean ± 95% CI.

**Selector curves**: for each N value, run the deployable selector and report selected accuracy. The N-scaling curve should show BOTH oracle and selected accuracy.

### What N-Scaling Reveals

- **Oracle saturation point**: N beyond which oracle no longer improves
- **Selector saturation point**: N beyond which selector no longer improves (may be EARLIER than oracle saturation)
- **Ensemble advantage**: the N at which ensemble selected accuracy surpasses single-operator selected accuracy
- **Cost crossover**: the compute budget at which ensemble becomes worthwhile

---

## 7. Held-Out Validation (New Section)

### The Problem
25 arithmetic tasks is a PILOT. Using the same 25 tasks for characterization, allocation optimization, and success claims creates selection bias.

### Solution A: Held-Out Tasks
- Reserve 5 tasks from the 25 as held-out (tasks 21-25)
- Use tasks 1-20 for characterization and operator selection
- Validate on tasks 21-25
- **Weakness**: 5 tasks is too few for reliable validation. One task = 20pp.

### Solution B: Held-Out Panels (Preferred)
- Stochastic operators have 3 panels (seeds 0-15, 16-31, 32-47)
- Use panels 0+1 for characterization and allocation
- Validate on panel 2
- **Strength**: full 25 tasks in validation, different random seeds
- **Weakness**: only validates stochastic operators, not prefix (which is deterministic)

### Solution C: New Task Set (Best but Expensive)
- After Phase 1 characterization on 25 arithmetic tasks:
- Generate a NEW set of 25-50 tasks for validation
- Run the CDE allocation determined from Phase 1
- Report selected accuracy on the new tasks
- **Strength**: completely held-out, no data leakage
- **Weakness**: doubles the compute cost

**Recommendation**: Solution B for Phase 1 pilot (cheapest, validates the method). Solution C for any publishable claim.

---

## 8. Equal-Compute Accounting (New Section)

### Token Budget Accounting

Operators have different costs per candidate:

| Operator | Tokens per candidate | Cost multiplier |
|---|---|---|
| O2 (prefix) | ~1024 (max) | 1.0x |
| O6 (temperature) | ~1024 (max) | 1.0x |
| O8 (rephrase) | ~1024 (max) | 1.0x |
| O4 (random token) | ~1024 (max) | 1.0x |
| O11 (DDC) | ~3000 (decompose + sub-solve) | 3.0x |
| O12 (neuro-symbolic) | ~200 (formalization only) | 0.2x |

### Equal-Budget Comparison

For a total token budget B:
- Single operator O_a at N=B/cost(O_a) candidates
- Ensemble at N_a=B_a/cost(O_a) + N_b=B_b/cost(O_b) where B_a+B_b=B

Example: B = 16,384 tokens (≈16 full generations)
- Single prefix: 16 candidates
- Ensemble: 8 prefix (8,192 tokens) + 8 temperature (8,192 tokens) = 16 candidates, same budget
- DDC: 5 decomposed solutions (5 × 3,072 tokens ≈ 15,360 tokens) = 5 candidates, same budget

### Wall-Clock Accounting

Different operators have different latency:
- Temperature sampling: ~same as greedy
- Prefix perturbation: ~same as greedy
- DDC: 3-5x longer (sequential sub-problems)
- Neuro-symbolic: ~0.5x (short generation + instant execution)

Report BOTH token-budget and wall-clock comparisons.

---

## 9. Success Criteria (Revised)

### Minimum Viable CDE (Pilot on 25 Tasks)

1. Complete trace and selection records for all Tier 1 operators
2. At least ONE operator pair where:
   - J(correct_sets) ≤ 0.7 (complementary)
   - rho_cross < rho_within for both operators (cross-operator is more decorrelated)
   - Both operators viable: A ≥ 50% of best, Y ≥ 0.85
3. **PRIMARY GATE**: Ensemble selected accuracy (DS3) > best single-operator selected accuracy under equal token budget (point estimate positive)
4. Oracle improvement exists (diagnostic confirmation)
5. DS3 achieves ≥ 80% of oracle accuracy on arithmetic
6. Survives held-out panel validation (Solution B)
7. Cost-normalized lift is positive vs best temperature/nucleus baseline

### Publishable CDE (Requires Additional Evidence)

1. Minimum Viable criteria met
2. N-scaling advantage across multiple N values (not only N=16)
3. Selected accuracy CI excludes 0 lift (paired permutation test)
4. Domain routing demonstrated: different optimal allocation for arithmetic vs legal
5. Selector calibration reported (ECE or Brier)
6. Held-out task set validation (Solution C)
7. Paper claim: "decorrelation converted into selected accuracy" — with complete evidence chain

### Paper Story (If All Criteria Met)

1. We propose the Controlled Decorrelation Ensemble framework
2. We show operator decorrelation drives oracle improvement (diagnostic)
3. **Critically**: We show a deployable selector converts decorrelation into USABLE accuracy gains
4. Random soft prefix perturbation is one operator; combined with [other operators], it produces measurably decorrelated candidates with different error patterns
5. We provide the measurement contract (trace schema, metrics, decision gates) as a reusable tool
6. The optimal allocation for [domain] is [X% prefix + Y% temperature + Z% rephrase]
