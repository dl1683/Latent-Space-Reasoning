# CDE Phase 1 Implementation Preparation

## Status: READY WHEN GPU AVAILABLE

## Purpose
This document translates the CDE v2 measurement protocol into concrete implementation tasks. When GPU becomes available, this is the execution plan.

---

## 1. Code Changes Required

### 1.1 Trace Logger (NEW: experiments/cde_trace_logger.py)

Produces JSONL traces conforming to the CDE v2 schema. Every generation call must route through this logger.

Key functions:
- `create_trace(task, operator, generation_config) -> trace_id`
- `record_generation(trace_id, output, logprobs, timing) -> None`
- `record_fingerprint(trace_id, first_32_tokens, embedding) -> None`
- `record_evaluation(trace_id, extracted_answer, correct) -> None`
- `flush() -> None`

### 1.2 Operator Registry (NEW: experiments/cde_operators.py)

Unified interface for all Tier 1 operators. Each operator implements:
```python
class Operator(ABC):
    name: str
    config: dict
    
    @abstractmethod
    def prepare(self, model, tokenizer) -> None:
        """Set up the operator (e.g., compute prefix embeddings)."""
    
    @abstractmethod
    def generate(self, question_embeds, seed: int) -> GenerationResult:
        """Generate one candidate."""
    
    @abstractmethod
    def get_artifact_hash(self) -> str:
        """Unique hash of the operator's configuration/artifacts."""
```

Implementations needed:
- `GreedyBaseline` — O1
- `RandomSoftPrefix` — O2 (adapts existing src/latent_reasoning logic)
- `ZeroSoftPrefix` — O3
- `RandomTokenPrefix` — O4
- `PositionShift` — O5
- `TemperatureSampling` — O6/O7/O9/O10
- `PromptRephrase` — O8

### 1.3 Selector Stack (NEW: experiments/cde_selectors.py)

Implements the selector protocol from CDE v2:
- `DegenerateFilter` — DS1
- `FormalVerifier` — DS2 (wraps Python eval for arithmetic)
- `OperatorStratifiedConsensus` — DS3 (PRIMARY)
- `SelfCertaintyRanker` — DS4
- `SelectionTraceLogger` — produces selection trace JSONL

### 1.4 Answer Normalizer (NEW: experiments/cde_answer_norm.py)

Critical for DS3 (operator-stratified consensus). Candidates from different operators may express the same answer differently:
- "56" vs "The answer is 56" vs "= 56" vs "fifty-six"

Normalization pipeline:
1. Extract all integers from output (existing extract_answer logic)
2. Normalize: strip whitespace, convert text numbers to digits
3. Canonical form: just the number as a string

For legal/planning: use sentence embedding similarity for answer clustering (not exact match).

### 1.5 Harness Upgrade (MODIFY: experiments/harness.py)

Extend the existing harness to:
- Accept operator as a parameter
- Log traces via CDE trace logger
- Support all Tier 1 operators
- Compute fingerprints (first-32-token IDs, mean logprob)

### 1.6 Analysis Script (NEW: experiments/cde_analyze.py)

Post-generation analysis:
- Read trace JSONL
- Compute all metrics (PM1-PM3, DM1-DM12)
- Compute cross-operator Jaccard matrix
- Compute error correlation matrix
- Generate N-scaling curves
- Run selector audit
- Produce summary tables and figures

---

## 2. Prompt Templates for O8 (Pre-Registered)

10 templates for arithmetic tasks. Each template MUST:
- Preserve the exact computation (never change operands)
- Change only the framing/wording
- Be manually audited for all 25 tasks

### Templates

```python
ARITHMETIC_TEMPLATES = [
    # T0: Standard (baseline prompt format)
    "What is {expr}?",
    
    # T1: Compute framing
    "Compute {expr}.",
    
    # T2: Calculate framing
    "Calculate: {expr}",
    
    # T3: Explicit step request
    "What is {expr}? Show your work step by step.",
    
    # T4: Answer-focused
    "The answer to {expr} is",
    
    # T5: Question word swap
    "Find the value of {expr}.",
    
    # T6: Imperative
    "Solve: {expr}",
    
    # T7: Contextual
    "If you were asked to evaluate {expr}, what would the result be?",
    
    # T8: Symbol swap (where applicable)
    "{expr_alt}",  # e.g., "7 times 8" instead of "7 × 8"
    
    # T9: Reverse framing
    "What does {expr} equal?",
]
```

**Audit requirement**: Before running, manually verify all 250 rendered prompts (25 tasks × 10 templates). Ensure no template changes the mathematical operation.

---

## 3. Execution Plan

### Step 1: Infrastructure (No GPU needed)
- [ ] Implement trace logger
- [ ] Implement operator registry
- [ ] Implement selector stack
- [ ] Implement answer normalizer
- [ ] Implement analysis script
- [ ] Audit prompt templates
- [ ] Unit tests for all components (mock model)

### Step 2: Validation (GPU needed, ~30 min)
- [ ] Run probe_inputs_embeds.py on Qwen3-4B Q4 (verify inputs_embeds works)
- [ ] Generate 5 candidates with each Tier 1 operator on 1 task
- [ ] Verify traces are correctly logged
- [ ] Verify selectors produce sensible results
- [ ] Verify fingerprints are correctly computed

### Step 3: Phase 1 Full Run (GPU needed, ~8-10 hours)
- [ ] Run all Tier 1 operators on all 25 tasks
- [ ] Verify no crashes/OOM/truncation issues
- [ ] Spot-check traces for completeness

### Step 4: Analysis (No GPU needed)
- [ ] Run cde_analyze.py
- [ ] Compute all metrics
- [ ] Generate figures (N-scaling curves, Jaccard matrix, correlation matrix)
- [ ] Apply decision gates
- [ ] Write preliminary results summary

### Step 5: Codex Review
- [ ] Submit Phase 1 results to Codex for evidence gate review
- [ ] Address any Codex feedback

---

## 4. Risk Mitigation

### Risk: inputs_embeds incompatibility
**Likelihood**: Medium (known boundary_mode issue from blueprint)
**Impact**: Blocks O2, O3, O4
**Mitigation**: probe_inputs_embeds.py (Component 0 from blueprint) is run FIRST

### Risk: Memory overflow with N=16 × 25 tasks
**Likelihood**: Low (Q4 model fits in 24GB, single generation at a time)
**Impact**: Crashes during long run
**Mitigation**: Per-generation gc.collect() + empty_cache() (existing pattern). Checkpoint every 50 generations.

### Risk: Template O8 changes the computation
**Likelihood**: Low if audited
**Impact**: Invalid comparison (confound)
**Mitigation**: Manual audit of all 250 prompts before running

### Risk: Phase 1 takes too long (>12 hours)
**Likelihood**: Medium (3,625 generations at ~8 sec each = ~8 hours)
**Impact**: GPU occupied for too long
**Mitigation**: Run overnight. Checkpoint after each operator. Can resume from checkpoint.

---

## 5. Existing Code to Reuse

### From experiments/harness.py
- `extract_answer()` — answer extraction
- `grade_response()` — evaluation
- Task list and formatting

### From src/latent_reasoning/
- Model loading (with quantization)
- Prefix generation (random soft prefix)
- Generation wrapper (inputs_embeds path)

### From experiments/ (existing results)
- Qwen3-4B Q4 N=10 results (REUSABLE as partial O2 data, if trace schema is populated retroactively)
- These cover seeds 0-9. Need seeds 10-15 for CDE Phase 1 N=16 requirement.
- Can reuse seeds 0-9 and generate only 10-15 (saves 60% of O2 generation time)

---

## 6. File Structure After Phase 1

```
experiments/
  cde_traces.jsonl              # All candidate traces
  cde_selections.jsonl          # All selection traces
  cde_trace_logger.py           # Trace logging infrastructure
  cde_operators.py              # Operator registry
  cde_selectors.py              # Selector stack
  cde_answer_norm.py            # Answer normalization
  cde_analyze.py                # Analysis and metrics
  cde_phase1_runner.py          # Main Phase 1 execution script
  cde_artifacts/
    embeddings/                 # First-32-token embeddings (numpy)
    figures/                    # Generated plots
  cde_phase1_results.md         # Summary of Phase 1 findings
  cde_phase1_decision_gates.md  # Gate evaluation results
```

---

## 7. Dependencies

### Python packages (already available)
- transformers (model loading, generation)
- torch (tensor operations)
- numpy (embeddings, storage)
- scikit-learn (clustering, metrics for analysis)
- matplotlib (figures)

### Potentially needed
- hdbscan (for ALM clustering — pip install hdbscan)
- umap-learn (for ALM visualization — pip install umap-learn)

### Not needed for Phase 1
- z3-solver (only for neuro-symbolic Tier 2)
- sentence-transformers (only for legal/planning answer clustering)
