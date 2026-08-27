# Tesla Mode Phase 2.1 — Corrections (Per Codex Round 3)

## Status
Phase A design is architecturally correct. Two critical errors require correction before Phase 3/5.
This document addresses all 7 required changes from Round 3 review.

---

## Correction 1: RMS Scaling Bug (CRITICAL)

### What was wrong
Phase 2 revised pseudocode:
```python
# WRONG — sets per-token L2 norm, not RMS
noise = noise * (rms_scale / noise.norm(dim=-1, keepdim=True).clamp(min=1e-8))
```

For `embed_dim=2560`, this makes per-dimension RMS ≈ `0.022 / sqrt(2560) ≈ 0.00043` — roughly **50x too small** compared to real token embeddings.

### The correct mechanism (from run_latent_sensitivity.py:1157)
```python
# CORRECT — sets tensor-level RMS to match real token embedding scale
torch.manual_seed(seed)
sp = torch.randn(1, token_count, embed_dim, dtype=dtype, device=device)
current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
effective_rms = 0.022  # matches real token embedding RMS
sp = sp * (effective_rms / current_rms)
# Result: sp.square().mean().sqrt() ≈ 0.022 — correct scale
```

This is the ONLY valid normalization for Phase A. Any deviation invalidates the experiment.

**Impact**: If the wrong scaling is used, Phase A telemetry measures a different mechanism than the validated one. The MI gate would be measuring a system that doesn't match the published results.

---

## Correction 2: Existing Logs Are Pilot-Only

### What existing logs contain (from run_latent_sensitivity.py:1384)
```
response[:2000]       — truncated text
response_raw[:2000]   — truncated raw text  
correct               — boolean
generated_token_count — integer
eos_appeared          — boolean
```

### What they DO NOT contain
- Exact token IDs
- Per-token logprobs
- Per-token entropy
- Attention weights
- Hidden states
- Full untruncated outputs

**Decision**: Existing logs CAN support a crude text-only pilot (e.g., repetition rate, output length, EOS timing). They CANNOT support the hard MI gate.

**The hard MI gate requires new Phase A telemetry collection** — a dedicated data collection run using the corrected Component 2 with `output_scores=True`, `output_attentions=True`, `output_hidden_states=True`.

**Sample size required for hard gate** (per Codex):
- Pilot: ≥100 task groups × 10 candidates per model/task class
- Hard gate: 200-500 task groups × 10 candidates per model/task class
- Each label class: ≥100 positive AND ≥100 negative candidate records

---

## Correction 3: Component 2 — Add output_scores and Logprob Extraction

Updated Component 2 generation call:
```python
output = model.generate(
    inputs_embeds=modified_embeds,
    attention_mask=extended_mask,
    input_ids=None,               # CRITICAL: None when using inputs_embeds
    max_new_tokens=1024,
    do_sample=False,
    temperature=None,
    top_p=None,
    output_scores=True,           # NEW: enables per-token score extraction
    output_attentions=True,       # for attention_sink_mass (if available)
    output_hidden_states=True,    # for layer-level features (if available)
    return_dict_in_generate=True
)

# Extract per-token logprobs from scores
logprobs = []
for step_scores in output.scores:  # output.scores is List[Tensor(vocab_size)]
    probs = torch.softmax(step_scores[0], dim=-1)
    entropy = -(probs * probs.log().clamp_min(-30)).sum().item()
    chosen_id = step_scores[0].argmax().item()
    chosen_logprob = step_scores[0][chosen_id].item() - torch.logsumexp(step_scores[0], dim=-1).item()
    logprobs.append({
        "token_id": chosen_id,
        "logprob": chosen_logprob,
        "entropy": entropy,
        "top1_logit": step_scores[0].max().item()
    })
```

**Sequence-slicing test required**: Before data collection, run a unit test confirming that:
- Token IDs from `output.sequences[0][prefix_length:]` match tokens decoded from `output_text`
- Per-token logprobs at step `t` correspond to the token at position `t` in the output
- Early window slicing `logprobs[:w]` correctly captures the first `w` generated tokens (not prompt tokens)

---

## Correction 4: Task-Held-Out MI Evaluation with Bootstrap CIs

### Why candidate rows cannot be independent
If we compute `I(features; correct)` treating each candidate row as independent, we inflate MI because task difficulty leaks: hard tasks produce many incorrect candidates, easy tasks produce many correct ones. A feature that merely correlates with "this is a hard task" would score high.

### Required evaluation protocol

```
1. Split by task_id (NOT by candidate row)
   - Train split: 70% of task IDs
   - Test split: 30% of task IDs (held out during feature selection)

2. MI estimation
   - Use a pre-registered estimator (k-NN MI or discretized MI)
   - Choose estimator BEFORE seeing results — do not switch after inspection
   - Apply permutation null: shuffle correctness labels within each task, recompute MI
   - Bootstrap CI: resample task groups (not rows) with replacement, 1000 iterations
   - Report: MI ± 95% CI, p-value vs permutation null

3. Oracle recall gate (the real operational test)
   - For each query group (same task, N candidates), rank candidates by each feature
   - Promote top-K fraction (K = 10%, 20%, 30%, 40%, 50% of candidates)
   - Measure: P(oracle winner in promoted set) across held-out tasks
   - Bootstrap by task group
   - Report: recall@K curve ± 95% CI for each feature and combination

4. Acceptance criteria (updated)
   - MI > 0.1 bits with p < 0.05 vs permutation null at w=64 for arithmetic
   - Oracle winner recall ≥ 90% at ≤ 50% promotion on held-out tasks
   - Both must hold for Phase B to proceed
   - Either failing = Phase B killed
```

---

## Correction 5: Complete Atlas Contamination Fields

Updated atlas record with full contamination guards:
```json
{
    "schema_version": "1.0",
    "timestamp": "2026-04-15T10:30:00Z",
    "experiment_id": "run_20260415_001",
    "code_commit": "git SHA at time of run",
    
    "model": {
        "model_id": "Qwen/Qwen3-4B",
        "hf_revision": "commit SHA or tag",
        "quantization": "4bit",
        "quantization_config_hash": "hash of BitsAndBytesConfig",
        "tokenizer_hash": "hash of tokenizer vocab + config",
        "generation_config_hash": "hash of GenerationConfig used"
    },
    
    "task": {
        "task_id": "arith_0001",
        "task_class": "arithmetic",
        "task_hash": "hash of full task text",
        "train_test_split_id": "split_v1_test"
    },
    
    "spec": {
        "surface": "input_prefix",
        "token_count": 2,
        "rms_scale": 0.022,
        "seed": 42,
        "soft_prefix_hash": "hash of actual noise tensor applied"
    },
    
    "is_baseline": false,
    
    "early_features": {
        "1":  { "n_observed": 1,  "feature_available": {...}, "..." : "..." },
        "4":  { "n_observed": 4,  "feature_available": {...}, "..." : "..." },
        "8":  { "n_observed": 8,  "feature_available": {...}, "..." : "..." },
        "16": { "n_observed": 16, "feature_available": {...}, "..." : "..." },
        "32": { "n_observed": 32, "feature_available": {...}, "..." : "..." },
        "64": { "n_observed": 64, "feature_available": {...}, "..." : "..." },
        "128":{ "n_observed": 128,"feature_available": {...}, "..." : "..." }
    },
    
    "full_output": "complete untruncated output text",
    "full_token_ids": [1234, 5678, ...],
    "truncated": false,
    "generation_time_s": 12.3,
    
    "validation": {
        "correct": true,
        "answer_anywhere_correct": true,
        "converged": true,
        "score": null,
        "dimension_scores": null,
        "abstain": false,
        "judge_model": null,
        "judge_prompt_hash": null,
        "rubric_hash": null
    },
    
    "selected_as_best": false
}
```

**Key additions**: `hf_revision`, `quantization_config_hash`, `generation_config_hash`, `soft_prefix_hash`, `code_commit`, `task_hash`, `train_test_split_id`, `judge_prompt_hash`, `rubric_hash`, `schema_version`.

---

## Correction 6: Baseline (No-Prefix) as Required Logged Candidate

The no-prefix baseline MUST be logged as a candidate in every task group.

Why:
- Without it, oracle recall calculation has no denominator for "how much does the oracle winner actually beat baseline?"
- Group-relative features (e.g., "is this candidate better than baseline?") are undefined without it
- MI analysis cannot distinguish "feature predicts correctness" from "feature predicts better-than-baseline improvement"

**Implementation**:
```python
# Always generate one baseline candidate per task (seed=-1 convention)
baseline_spec = PerturbationSpec(
    surface="input_prefix",
    token_count=0,          # zero prefix tokens = no perturbation
    rms_scale=0.0,
    seed=-1,
    model_id=model_id,
    quantization=quantization
)
# Baseline is treated as candidate index 0 in every task group
# token_count=0 means: use raw input_ids (no inputs_embeds needed)
```

---

## Correction 7: Qualitative Labels Defined Before MI Collection

For legal and planning tasks, `correct` cannot be `None` in the MI experiment. The target label must be defined and locked in before data collection.

### Option A: Oracle-relative label (RECOMMENDED for MI experiment)
```
correct = 1 if this candidate is the oracle winner for this task group
correct = 0 otherwise
```
This makes the MI question: "do early features predict which candidate will be selected by the judge?"
This is task-held-out clean and does not require ground truth.

### Option B: Above-baseline binary
```
correct = 1 if judge score for this candidate > judge score for baseline candidate
correct = 0 otherwise
```
This requires judge scores for all candidates AND the baseline. More data-intensive.

### Option C: Top-rubric-dimension threshold
```
correct = 1 if correctness_score >= threshold (e.g., 7/10)
correct = 0 otherwise
```
Requires judge to score all candidates before MI analysis. Must lock threshold BEFORE seeing feature data.

**Decision**: Use Option A (oracle-relative) for legal/planning MI experiment. Oracle-relative is:
- Pre-registerable (no threshold judgment needed)
- Task-held-out clean
- Aligned with the operational goal (select the best candidate)

---

## Updated EarlyFeatures Contract (Per Correction 3 + 4)

```python
@dataclass
class EarlyFeatures:
    window: int                          # w generated tokens observed

    # Availability flags (True if data was collected, False if not available)
    logprobs_available: bool
    attentions_available: bool
    hidden_states_available: bool

    # Token features (always available if generation succeeded)
    token_ids: List[int]                 # first w generated token IDs (NOT prompt tokens)
    n_observed: int                      # actual tokens observed (min(w, actual_output_len))
    eos_appeared: bool                   # EOS token appeared in window
    truncated_at_window: bool            # output ended exactly at window boundary

    # Logprob features (if logprobs_available)
    mean_logprob: Optional[float]        # mean log P per token in window
    logprob_slope: Optional[float]       # linear trend coefficient (OLS on step index)
    cumulative_logprob: Optional[float]  # sum of log P up to window

    # Entropy features (if logprobs_available)
    token_entropy_mean: Optional[float]  # mean per-step entropy
    token_entropy_slope: Optional[float] # linear trend in entropy
    entropy_at_final_token: Optional[float] # entropy at token w

    # Attention features (if attentions_available)
    attention_sink_mass: Optional[float] # fraction of attn in prefix positions
                                          # SPEC: mean over layers {1,2,3,4}, heads all,
                                          # sequence positions all, attending TO tokens {0,1}
    attention_sink_layer_profile: Optional[List[float]]  # per-layer sink mass (layers 0-7)

    # Format/structure features (always available)
    think_token_fraction: Optional[float]  # fraction of <think>-related token IDs
    repetition_rate: float                  # fraction of repeated 4-grams in window
    
    # Group-relative features (computed at analysis time, not collection time)
    # These are NOT stored in atlas; derived during MI analysis from full task group
    # relative_logprob_rank: int   — rank among candidates in same task by mean_logprob
    # above_baseline: bool         — whether any feature exceeds baseline candidate value
```

**Note**: Group-relative features are computed OFFLINE from the full task group, never stored per-candidate. This prevents leakage of group information during collection.

---

## Summary: What Phase A Looks Like After 2.1 Corrections

```
Data Collection Run (Phase A):
  For each task group:
    1. Generate baseline candidate (token_count=0) → log to atlas
    2. Generate N=10 random-prefix candidates (token_count=2, RMS=0.022, correct scaling)
       Each with output_scores=True, output_attentions=True, output_hidden_states=True
    3. Extract EarlyFeatures at w={1,4,8,16,32,64,128} from generation telemetry
    4. Run Output Validator on each candidate (arithmetic: exact match; qualitative: oracle-relative)
    5. Log ALL N+1 candidates to atlas (baseline + N perturbations)

MI Analysis (offline, after data collection):
    1. Split task groups 70/30 train/test
    2. Estimate I(features_w; oracle_winner) with pre-registered k-NN estimator
    3. Compute permutation null and bootstrap CIs by task group
    4. Evaluate oracle recall@K on test split
    5. Report H1-H6 outcomes

Gate: if MI > 0.1 bits AND oracle recall ≥ 90% at ≤ 50% promotion:
    → Proceed to Phase B Observer-Router design
Gate: if either fails:
    → Phase B killed; system stays Phase A; oracle analysis is the product
```

---

## The 6 Hypotheses to Formalize in Phase 5 (Per Codex)

- **H1**: Early features at 64 tokens contain >0.1 bits of task-held-out MI about final correctness for arithmetic (Qwen3-4B Q4)
- **H2**: Early features predict convergence/truncation more strongly than answer-anywhere computation for Qwen3-4B Q4
- **H3**: Qwen3-8B Q8 shows more answer-anywhere computation signal in early features than Qwen3-4B Q4 (mirrors the mean-win decomposition)
- **H4**: `attention_sink_mass` predicts collapse/truncation better than semantic correctness (AUROC > 0.65 for truncation, AUROC < 0.65 for correctness)
- **H5**: A held-out promotion rule can retain ≥90% of oracle winners while promoting ≤50% of candidates (on arithmetic held-out tasks)
- **H6**: Legal/planning qualitative tasks have weaker early correctness MI than arithmetic, even if length/collapse MI is present
