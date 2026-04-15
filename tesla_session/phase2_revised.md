# Tesla Mode Phase 2 (Revised) — Next-Gen System Design

## Revision Notes (Per Codex Phase 2 Review)

Phase 2 original was "directionally right but not ready." Codex blocked Phase 3/5. Required revisions:
1. Mark `attention_sink_mass`, 32-token classification, and 6-feature routing as **hypotheses**, not design primitives
2. Remove multi-surface interventions from mainline (keep as future research)
3. Define Observer's real tensor/cache/logit/attention contract
4. Define fallback as "validated random raw-prefix + full output validation" — NOT current package behavior
5. Require unbiased atlas logging of ALL candidates (promoted and unpromoted)
6. Add MI/AUROC/recall-of-oracle thresholds as hard gates before Router is built

**The architecture is now split into two phases:**
- **Phase A (Now — Minimum Viable System)**: Validated raw soft-prefix pipeline + candidate logging + offline MI analysis
- **Phase B (Conditional — After Gate)**: Online Observer-Router, only if MI analysis passes acceptance threshold

---

## Reframe (Refined)

**What we actually know from evidence:**
- 2 random embedding-scale tokens prepended via raw `inputs_embeds` improve Qwen3-4B arithmetic +19.6pp
- 100% oracle coverage at 2 tokens (n=10 directions)
- Direction-agnostic: random = optimized (p=1.0)
- Attention sink rescue observed in planning (14 → 650 words), but sink-mass-to-correctness correlation is NOT measured

**What we hypothesize but have not shown:**
- Early trajectory biomarkers predict final correctness
- 32 tokens of partial generation is sufficient classification window
- attention_sink_mass correlates with final incorrectness
- A router trained on early features can retain ≥90% oracle winners while promoting ≤50% candidates

**The design must not build on hypotheses. The design must test them first.**

---

## Phase A: Minimum Viable Next-Gen System

### Goal
Fix the architecture gap between the validated research mechanism and the shipped package. Ship a clean, reproducible pipeline that can be used to collect the data needed to decide whether Phase B (Observer-Router) is worth building.

### Architecture

```
Query + Model Config
    ↓
[1: Spec Generator]
  Generate N random PerturbationSpecs
  (input_prefix surface only; 2 tokens; RMS=0.022; N seeds)
    ↓
[2: Raw Soft-Prefix Applicator]
  Apply each spec via raw inputs_embeds (NOT encoder.decode())
  This IS the validated mechanism
    ↓
[3: Full Generation] × N (all candidates, no pruning)
  max_new_tokens=1024 (or 2048 for planning/legal)
  Collect: token IDs, logprobs, attentions (if available), hidden states (layer -4)
    ↓
[4: Candidate Logger] — logs EVERYTHING (not just promoted)
  Per candidate: spec, partial features at w={1,4,8,16,32,64,128} tokens,
  full output text, timing, truncation flag
    ↓
[5: Output Validator]
  Arithmetic: deterministic extractor (extract_answer → exact match)
  Planning/Legal: LLM-as-judge with structured rubric
  Returns: correct | incorrect | unknown per candidate
    ↓
[6: Atlas Logger]
  Append (spec, early_features_all_windows, full_output, correctness) to atlas
  ALL N candidates logged, not just best
    ↓
Best output (oracle selection for science; random valid for deployment baseline)
```

---

## Component Specifications (Phase A Only)

### Component 1: Spec Generator
**Responsibility**: Generate N PerturbationSpecs for the input_prefix surface only.

**Interface contract**:
```python
@dataclass
class PerturbationSpec:
    surface: Literal["input_prefix"]  # ONLY this in Phase A mainline
    token_count: int                   # default 2
    rms_scale: float                   # default 0.022
    seed: int                          # for reproducibility
    model_id: str                      # must be explicit
    quantization: str                  # must be explicit

def generate_specs(
    model_id: str,
    quantization: str,
    n_candidates: int,
    token_count: int = 2,
    rms_scale: float = 0.022,
    seed_offset: int = 0
) -> List[PerturbationSpec]
```

**Failure mode**: Raises `ValueError` for any surface other than `input_prefix` in Phase A. Never silently ignores parameters.

---

### Component 2: Raw Soft-Prefix Applicator
**Responsibility**: Apply a PerturbationSpec to a tokenized input via raw `inputs_embeds`. This replaces the broken `encoder.decode()` path.

**The validated mechanism (from `run_latent_sensitivity.py:49,135`)**:
```python
# Generate random noise at RMS scale
noise = torch.randn(1, token_count, embed_dim, device=device, dtype=dtype)
noise = noise * (rms_scale / noise.norm(dim=-1, keepdim=True).clamp(min=1e-8))

# Get input embeddings
input_embeds = model.get_input_embeddings()(input_ids)  # (1, seq_len, embed_dim)

# Prepend noise
modified_embeds = torch.cat([noise, input_embeds], dim=1)  # (1, token_count + seq_len, embed_dim)

# Extend attention mask
extended_mask = torch.cat([
    torch.ones(1, token_count, device=device, dtype=attention_mask.dtype),
    attention_mask
], dim=1)

# Generate — MUST use inputs_embeds, NOT input_ids
output = model.generate(
    inputs_embeds=modified_embeds,
    attention_mask=extended_mask,
    input_ids=None,          # CRITICAL: must be None when using inputs_embeds
    max_new_tokens=1024,
    do_sample=False,
    temperature=None,
    top_p=None,
    output_attentions=True,  # collect for biomarker logging
    output_hidden_states=True,
    return_dict_in_generate=True
)
```

**Interface contract**:
```python
@dataclass
class GenerationResult:
    output_ids: Tensor                            # generated token IDs
    output_text: str                              # decoded text
    logprobs: List[float]                         # per-token log probabilities
    attentions: Optional[List[Tensor]]            # per-layer attention at each step (if available)
    hidden_states: Optional[List[Tensor]]         # per-layer hidden states (if available)
    prefix_length: int                            # number of prefix tokens prepended
    truncated: bool                               # True if hit max_new_tokens
    generation_time_s: float

def apply_and_generate(
    model,
    tokenizer,
    input_ids: Tensor,
    attention_mask: Tensor,
    spec: PerturbationSpec,
    max_new_tokens: int = 1024
) -> GenerationResult
```

**Failure mode**: If `input_ids` is accidentally passed alongside `inputs_embeds`, raises immediately. If model doesn't support `output_attentions`, logs warning and continues with `attentions=None`.

---

### Component 3: Candidate Logger (Full — ALL Candidates)
**Responsibility**: Extract and store early trajectory features at multiple observation windows. Must log ALL candidates to avoid selection bias in atlas.

**Early features extracted at w ∈ {1, 4, 8, 16, 32, 64, 128} generated tokens:**
```python
@dataclass
class EarlyFeatures:
    window: int                          # w tokens observed
    token_ids: List[int]                 # first w generated token IDs
    mean_logprob: float                  # mean log P per token in window
    logprob_slope: float                 # linear trend (positive = improving confidence)
    token_entropy_mean: float            # mean per-step entropy in window
    token_entropy_slope: float           # trend in entropy (increasing = exploration)
    attention_sink_mass: Optional[float] # fraction attn in prefix positions (if available)
    eos_appeared: bool                   # EOS token appeared in window
    think_token_fraction: float          # fraction of think-related tokens
    repetition_rate: float               # fraction of repeated n-grams
    cumulative_logprob: float            # sum of log P up to window
    # NOTE: answer_position_forecast and trajectory_class are NOT extracted here —
    # they are hypotheses to be tested offline after MI analysis
```

**All of this is raw logging — no routing decisions made at this stage.**

**Interface contract**:
```python
def extract_early_features(
    result: GenerationResult,
    windows: List[int] = [1, 4, 8, 16, 32, 64, 128]
) -> Dict[int, EarlyFeatures]  # keyed by window size
```

---

### Component 4: Output Validator
**Responsibility**: Score the actual text output. The ONLY primary evaluation signal.

**Arithmetic**:
```python
def validate_arithmetic(output_text: str, ground_truth: int | str) -> ValidationResult:
    extracted = extract_answer(output_text)  # last integer in output
    correct = (extracted == ground_truth)
    answer_anywhere = any integer in output == ground_truth
    return ValidationResult(
        correct=correct,
        answer_anywhere_correct=answer_anywhere,
        converged=(extracted is not None),
        score=float(correct)
    )
```

**Planning/Legal (LLM-as-judge)**:
```python
def validate_qualitative(
    output_text: str,
    task_description: str,
    judge: LLMJudge,
    rubric: Dict[str, str]  # e.g., {correctness, completeness, specificity, hallucination}
) -> ValidationResult:
    # Returns per-dimension scores (no scalar collapse)
    # Judge MUST be output-space (reads actual text, not latent)
    # Fallback: if judge unavailable, return ValidationResult(score=None, abstain=True)
    # NEVER return a heuristic quality score
```

**`ValidationResult` structure**:
```python
@dataclass
class ValidationResult:
    correct: Optional[bool]               # None if unknown/qualitative
    answer_anywhere_correct: Optional[bool]  # for P decomposition
    converged: bool                        # answer reached end of output
    score: Optional[float]                 # None for qualitative until judge
    dimension_scores: Optional[Dict[str, float]]  # per-rubric-dimension
    abstain: bool = False                  # True if cannot validate
    judge_model: Optional[str] = None      # which judge was used
```

---

### Component 5: Atlas Logger
**Responsibility**: Append every candidate's data (not just winners). Prevent contamination.

**Append-only JSONL record per candidate:**
```json
{
    "timestamp": "ISO8601",
    "experiment_id": "...",
    "model_id": "...",
    "quantization": "...",
    "task_class": "arithmetic|planning|legal",
    "task_id": "...",
    "spec": { "surface": "input_prefix", "token_count": 2, "rms_scale": 0.022, "seed": 42 },
    "early_features": {
        "1": { ... EarlyFeatures ... },
        "4": { ... },
        "8": { ... },
        "16": { ... },
        "32": { ... },
        "64": { ... },
        "128": { ... }
    },
    "full_output": "...",
    "validation": { ... ValidationResult ... },
    "selected_as_best": false,
    "judge_model_version": "...",
    "prompt_template_hash": "...",
    "tokenizer_hash": "..."
}
```

**Contamination guards**:
- `judge_model_version` and `tokenizer_hash` logged per record — stale models detected by version mismatch
- `selected_as_best` is a flag, NOT a filter — all records written regardless
- Records from different model versions are not mixed in MI analysis

---

## Phase B: Observer-Router (Conditional — Gated by MI Analysis)

**This phase does NOT exist yet as an architecture.** It is a conditional future design, gated on the following acceptance criteria from the MI experiment:

### Acceptance Gate (Hard)
Run the Priority Experiment (see below). Phase B only proceeds if:
- `I(early_features_64; final_correct) > 0.1 bits` on arithmetic dataset
- A heuristic router retains **≥90% of oracle winners** while promoting **≤50% of candidates**
- This holds for both Qwen3-4B Q4 and Qwen3-8B Q8

If the gate fails: Phase B is killed. The system remains Phase A + offline oracle analysis. This is acceptable — oracle analysis is still scientifically valid.

### Hypotheses Being Tested (NOT Design Decisions)
These are hypotheses that must be confirmed before Phase B design:
- H1: `attention_sink_mass` at 32 tokens predicts final incorrectness (AUROC > 0.65)
- H2: `mean_logprob` slope at 64 tokens predicts convergence (AUROC > 0.65)
- H3: A 6-feature linear model can route with ≥90% oracle recall at ≤50% promotion
- H4: Trajectory class labels are stable after 64 tokens (transition probability < 0.2)

---

## Priority Experiment: MI Analysis (Blocks All Phase B Work)

**The Question**: At what token window do early trajectory features contain enough mutual information with final correctness to support a router?

**Thesis**: At 64 tokens, `I(features; final_correct) > 0.1 bits` and a heuristic router retains ≥90% oracle winners.

**Counter-thesis**: MI stays below 0.1 bits at 128 tokens for qualitative tasks. Early features predict length/truncation but not correctness.

**The Minimum Viable Experiment**:
1. Use existing logged outputs from arithmetic (25 tasks × 10 seeds), legal (12 tasks × 5 seeds), planning (5 tasks × 5 seeds)
2. For each output, extract `EarlyFeatures` at w ∈ {1, 4, 8, 16, 32, 64, 128} from the full generation record
3. Compute `I(features_w; correct)` via discretization or k-NN estimator
4. Compute AUROC for each feature independently at each window
5. Simulate heuristic router: rank candidates by mean_logprob, promote top-K, measure oracle winner recall

**Acceptance criteria**:
- MI > 0.1 bits at ≤64 tokens: Phase B design proceeds
- MI < 0.1 bits at 128 tokens: Phase B killed; system stays Phase A
- Oracle recall ≥ 90% at ≤50% promotion: Router architecture justified
- Oracle recall < 90% at 50% promotion: Router cannot improve on full-generation oracle

**What we're measuring**:
```
P(final_correct) = P(answer_anywhere_correct) × P(final_correct | answer_anywhere_correct)

Measured per:
- Model (Qwen3-4B Q4, Qwen3-8B Q8, DeepSeek where available)
- Task class (arithmetic, planning, legal)
- Perturbation energy E = token_count × rms_scale²
- Observation window w ∈ {1, 4, 8, 16, 32, 64, 128}
```

---

## What NOT to Build (Updated Exclusions)

- **No Observer-Router until MI gate passes** — it is a hypothesis, not an architecture
- **No multi-surface interventions in mainline** — residual/attention-mask surfaces not validated
- **No fixed W projection** — deleted as mainline; adds no value over random noise
- **No latent evolution loop** — not until landscape is shown smooth under real scorer
- **No scalar latent scorer as primary** — CLAUDE.md says scores are irrelevant; output validator is primary
- **No `encoder.decode()` path** — replace entirely with raw `inputs_embeds`
- **No heuristic quality fallback** — if judge is unavailable, abstain rather than score by length/format
- **No oracle-as-deployment** — oracle = science ceiling; Phase A uses full-N generation + output validation

---

## State Map

| State | Owner | Location | Mutability |
|---|---|---|---|
| Atlas log (all candidates) | Component 5 | `data/atlas_log.jsonl` | Append-only |
| MI analysis results | Experiment | `data/mi_analysis.json` | Written by experiment |
| Phase B gate decision | Codex review | `tesla_session/codex_phase2b_gate.md` | One-time write |
| Experiment ledger | Repo standard | `experiments/ledger.jsonl` | Append-only |
| Session config | Caller | `configs/*.yaml` | Read-only at runtime |

---

## Failure Modes (Updated)

| Failure | Consequence | Fallback |
|---|---|---|
| `encoder.decode()` called | Silent incorrect mechanism | Raise `DeprecationError`; force `apply_and_generate()` |
| Attentions unavailable | `attention_sink_mass = None` | Log as None; do not impute |
| Judge unavailable | No validation score | Return `abstain=True`; do NOT score by heuristic |
| All candidates abstained | No valid output | Return best by `mean_logprob` with `confidence=None` and `warn=True` |
| Atlas mixed model versions | Contaminated MI analysis | Filter by `model_id + tokenizer_hash + judge_model_version` |
| MI gate fails | Phase B killed | System stays Phase A; report as acceptable ceiling |

---

## Open Questions Remaining (For Phase 3 Stress Test)

1. Can early features be extracted without rerunning generation? (Use cached outputs from existing experiments?)
2. Is `I(features; correct)` different for arithmetic vs. qualitative tasks? If so, separate gates per task class?
3. Does the observation window interact with `token_count`? (2-token prefix may shift fate earlier or later)
4. What is the minimum atlas size needed for the MI estimate to be reliable? (Sample size for MI estimation)
5. If MI is above threshold for arithmetic but not legal, build a domain-specific router or kill Phase B entirely?
6. What does abstention do to system utility? Is a system that abstains 30% of the time useful?
