# Tesla Mode Phase 6: The Blueprint

## Status
Approved by Codex Round 5. Phase A Blueprint only.
Phase B (Observer-Router) is conditional — Blueprint will be written only after H1 + H5 pass.

## Amendments from Round 5 (Incorporated)
- Missing stress test: generation-path confounding added (Section 10)
- MI estimator corrected: discrete-continuous, not KSG (Section 7)
- Feature normalization specified: within-task z-score (Section 7)
- H5 clarified: baseline excluded from the 10 candidates (Section 7)
- Cheaper H1 feasibility path added (Section 8)
- Phase B explicit kill/proceed rule added (Section 9)

## Amendments from Round 6 (Incorporated)
- `generation_start_index` replaced with `boundary_mode` policy — computed per-call from `combined_length` (Component 0 + 2)
- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
- Lean rerun prefix-equivalence assertion added to Component 2 contract
- MI H1 defined as scalar `routing_score` MI — not summed per-feature MI (Section 7)
- Within-task z-scoring clarified: per group, std < eps = degenerate excluded (Section 7)
- Implementation order: arithmetic validator added before Phase 1/2 (Section 8)
- Phase B kill/proceed edge cases fixed: exact thresholds, H1-significant-but-below-0.1, H5 ties, all-wrong groups (Section 9)

---

## The Single Most Important Thing to Build First

**Component 0: The `inputs_embeds` Generation Probe**

This is a standalone test script, not part of the main system. It must pass before any other Phase A component is written.

```python
# experiments/probe_inputs_embeds.py
# Purpose: Empirically determine inputs_embeds sequence boundary behavior
# for this exact model, tokenizer, transformers version, and generation config.

def run_probe(model, tokenizer, device):
    """
    Generates one candidate with inputs_embeds and asserts all alignments.
    Must pass before any Phase A data collection.
    """
    # 1. Build a known test input
    test_query = "What is 7 × 8?"
    messages = [{"role": "user", "content": test_query}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    prompt_token_count = input_ids.shape[1]

    # 2. Build soft prefix
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)                        # (1, prompt_len, embed_dim)
    embed_dim = input_embeds.shape[-1]
    token_count = 2

    torch.manual_seed(42)
    sp = torch.randn(1, token_count, embed_dim, dtype=input_embeds.dtype, device=device)
    embedding_rms = embed_layer.weight.detach().square().mean().sqrt().item()
    rms_multiplier = 1.0
    effective_rms = embedding_rms * rms_multiplier
    current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
    sp = sp * (effective_rms / current_rms)

    actual_prefix_rms = sp.square().mean().sqrt().item()
    assert abs(actual_prefix_rms - effective_rms) / effective_rms < 0.01, \
        f"RMS mismatch: {actual_prefix_rms} vs {effective_rms}"

    # 3. Build modified inputs
    combined_embeds = torch.cat([sp, input_embeds], dim=1)       # (1, token_count+prompt_len, embed_dim)
    combined_input_length = combined_embeds.shape[1]
    attention_mask = torch.ones(1, combined_input_length, dtype=torch.long, device=device)

    # 4. Generate with full telemetry
    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            input_ids=None,
            max_new_tokens=64,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index — varies per prompt length)
    # HuggingFace behavior: with inputs_embeds, output.sequences may or may not
    # include dummy prefix tokens. We detect the mode once and save it as a policy.
    sequences = output.sequences[0]          # shape: (N,) where N is unknown
    n_scores = len(output.scores)            # number of generated tokens (always correct)
    n_sequences = sequences.shape[0]

    # ASSERT: scores length must equal generated tokens
    assert n_scores <= 64, f"Too many scores: {n_scores}"

    # Detect boundary mode — saved as a string policy, NOT as a fixed integer
    if n_sequences == n_scores:
        boundary_mode = "generated_only"     # sequences contains ONLY generated tokens
    elif n_sequences == combined_input_length + n_scores:
        boundary_mode = "input_echo"         # sequences contains prefix + generated tokens
    else:
        raise ValueError(
            f"Unexpected sequences length: {n_sequences}. "
            f"n_scores={n_scores}, combined_input_length={combined_input_length}. "
            f"Cannot determine generation boundary. Check transformers version."
        )

    # Per-call index derivation (NOT a saved scalar — combined_length changes per prompt):
    # generation_start_index = 0 if boundary_mode == "generated_only" else combined_length

    # 6. Extract generated token IDs using the boundary mode
    generation_start_index = 0 if boundary_mode == "generated_only" else combined_input_length
    generated_ids = sequences[generation_start_index:]
    assert len(generated_ids) == n_scores, \
        f"Token ID slice mismatch: {len(generated_ids)} vs {n_scores}"

    # 7. Verify logprob alignment
    for t, step_scores in enumerate(output.scores):
        expected_id = generated_ids[t].item()
        max_score_id = step_scores[0].argmax().item()
        assert expected_id == max_score_id, \
            f"Logprob/token mismatch at step {t}: expected {expected_id}, scores say {max_score_id}"

    # 8. Decode and verify
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # 9. Baseline equivalence check
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False
        )
    baseline_text = tokenizer.decode(baseline_output[0][prompt_token_count:], skip_special_tokens=False)

    # NOTE: baseline and prefixed outputs will differ (that's the point of perturbation)
    # But ZERO-prefix must match baseline:
    zero_embeds = input_embeds  # no prefix
    zero_mask = torch.ones(1, prompt_token_count, dtype=torch.long, device=device)
    with torch.no_grad():
        zero_output = model.generate(
            inputs_embeds=zero_embeds,
            attention_mask=zero_mask,
            input_ids=None,
            max_new_tokens=64,
            do_sample=False,
            output_scores=True,           # needed for len(zero_output.scores)
            return_dict_in_generate=True
        )
    zero_sequences = zero_output.sequences[0]
    # Use len(output.scores) not prompt_token_count + 64 — generation may stop early
    n_zero_scores = len(zero_output.scores)
    if zero_sequences.shape[0] == n_zero_scores:
        zero_generated_ids = zero_sequences                     # generated_only mode
    elif zero_sequences.shape[0] == prompt_token_count + n_zero_scores:
        zero_generated_ids = zero_sequences[prompt_token_count:]  # input_echo mode
    else:
        raise ValueError(
            f"Zero-prefix boundary detection failed: sequences={zero_sequences.shape[0]}, "
            f"n_scores={n_zero_scores}, prompt_len={prompt_token_count}"
        )
    zero_text = tokenizer.decode(zero_generated_ids, skip_special_tokens=False)

    assert zero_text == baseline_text, \
        f"ZERO-PREFIX DOES NOT MATCH BASELINE. inputs_embeds path is broken.\n" \
        f"Baseline: {repr(baseline_text)}\nZero-prefix: {repr(zero_text)}"

    # 10. Report — save boundary_mode, NOT a fixed generation_start_index
    probe_result = {
        "prompt_token_count": prompt_token_count,
        "prefix_token_count": token_count,
        "combined_input_length": combined_input_length,
        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
        # NOTE: generation_start_index is NOT saved — it is computed per-call as:
        #   0 if boundary_mode == "generated_only" else combined_length
        "n_generated_tokens": n_scores,
        "n_sequences": n_sequences,
        "embedding_rms": embedding_rms,
        "effective_rms": effective_rms,
        "actual_prefix_rms": actual_prefix_rms,
        "generated_text_sample": generated_text[:200],
        "baseline_text_sample": baseline_text[:200],
        "zero_prefix_matches_baseline": True,   # asserted above
        "logprob_alignment_verified": True       # asserted above
    }
    return probe_result
```

**This probe must be run and pass before writing any other Phase A component.**
**Save `boundary_mode` to config. Every subsequent component computes `generation_start_index` per-call as `0 if boundary_mode == "generated_only" else combined_length`. Never use a fixed scalar.**

---

## Component 1: Spec Generator

```python
# src/latent_reasoning/phase_a/spec_generator.py

@dataclass
class PerturbationSpec:
    surface: Literal["input_prefix", "baseline"]
    token_count: int           # 0 for baseline
    rms_multiplier: float      # 1.0 = match embedding RMS
    seed: int                  # -1 for baseline
    model_id: str
    quantization: str

def generate_specs(
    model_id: str,
    quantization: str,
    n_prefix_candidates: int,
    token_count: int = 2,
    rms_multiplier: float = 1.0,
    seed_offset: int = 0
) -> List[PerturbationSpec]:
    """
    Returns: [baseline_spec] + [n_prefix_candidates × prefix_specs]
    Baseline spec always first (seed=-1, surface=baseline, token_count=0).
    """
    baseline = PerturbationSpec(
        surface="baseline", token_count=0, rms_multiplier=0.0,
        seed=-1, model_id=model_id, quantization=quantization
    )
    prefix_specs = [
        PerturbationSpec(
            surface="input_prefix", token_count=token_count,
            rms_multiplier=rms_multiplier, seed=seed_offset + i,
            model_id=model_id, quantization=quantization
        )
        for i in range(n_prefix_candidates)
    ]
    return [baseline] + prefix_specs
```

---

## Component 2: Raw Soft-Prefix Applicator

```python
# src/latent_reasoning/phase_a/applicator.py

class RawSoftPrefixApplicator:
    """
    Applies a PerturbationSpec to a model input and generates with full telemetry.
    Built on the validated mechanism from run_latent_sensitivity.py.
    """
    def __init__(self, model, tokenizer, device, boundary_mode: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" — from probe
        # Calibrate embedding RMS once at init
        self.embedding_rms = model.get_input_embeddings().weight.detach().square().mean().sqrt().item()

    def _get_start_index(self, combined_length: int) -> int:
        """Per-call computation — combined_length varies per prompt, so cannot be a fixed scalar."""
        return 0 if self.boundary_mode == "generated_only" else combined_length

    def apply_and_generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        spec: PerturbationSpec,
        max_new_tokens: int = 1024,
        max_observe_tokens: int = 128,
    ) -> GenerationResult:

        if spec.surface == "baseline":
            # MUST match direct model.generate(input_ids=...) byte-for-byte
            return self._generate_baseline(input_ids, attention_mask, max_new_tokens)

        # Generate soft prefix
        embed_layer = self.model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)                   # (1, prompt_len, embed_dim)
        embed_dim = input_embeds.shape[-1]

        gen = torch.Generator(device=self.device)
        gen.manual_seed(spec.seed)
        sp = torch.randn(
            1, spec.token_count, embed_dim,
            generator=gen, dtype=input_embeds.dtype, device=self.device
        )
        effective_rms = self.embedding_rms * spec.rms_multiplier
        current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
        sp = sp * (effective_rms / current_rms)

        # ASSERT actual RMS matches target
        actual_prefix_rms = sp.square().mean().sqrt().item()
        assert abs(actual_prefix_rms - effective_rms) / effective_rms < 0.01

        combined_embeds = torch.cat([sp, input_embeds], dim=1)
        combined_length = combined_embeds.shape[1]
        combined_mask = torch.ones(1, combined_length, dtype=torch.long, device=self.device)

        # Generation with telemetry (two-phase: observe + lean)
        # Phase 1: collect telemetry for first max_observe_tokens
        with torch.no_grad():
            observe_output = self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                input_ids=None,
                max_new_tokens=max_observe_tokens,
                do_sample=False,
                output_scores=True,
                output_attentions=True,   # only for observe phase
                output_hidden_states=False,
                return_dict_in_generate=True
            )

        # Extract telemetry from observe phase
        generation_start_index = self._get_start_index(combined_length)
        observe_ids = observe_output.sequences[0][generation_start_index:]
        observe_scores = observe_output.scores   # List[Tensor(vocab_size)] length = min(max_observe_tokens, actual)
        observe_attentions = observe_output.attentions  # available for first max_observe_tokens tokens

        # ASSERT: logprob alignment
        for t in range(len(observe_scores)):
            expected_id = observe_ids[t].item()
            max_id = observe_scores[t][0].argmax().item()
            assert expected_id == max_id, f"Logprob mismatch at step {t}"

        # Extract per-token telemetry
        token_telemetry = self._extract_token_telemetry(observe_ids, observe_scores, observe_attentions)

        # Phase 2: continue lean generation for remaining tokens
        # Use KV cache from observe phase to continue without re-running prefix
        if not observe_output.sequences[0].shape[0] >= combined_length + max_new_tokens:
            # Continue from the last observed token
            remaining_tokens = max_new_tokens - len(observe_ids)
            if remaining_tokens > 0 and not token_telemetry["eos_in_observe"]:
                with torch.no_grad():
                    lean_output = self.model.generate(
                        inputs_embeds=combined_embeds,
                        attention_mask=combined_mask,
                        input_ids=None,
                        max_new_tokens=max_new_tokens,  # generate full output fresh (simpler than cache)
                        do_sample=False,
                        output_scores=False,
                        output_attentions=False,
                        return_dict_in_generate=True
                    )
                full_ids = lean_output.sequences[0][generation_start_index:]
                # ASSERT: lean rerun prefix-equivalence
                assert torch.equal(full_ids[:len(observe_ids)], observe_ids), \
                    "Lean rerun diverged from observe run — telemetry may have altered generation"
            else:
                full_ids = observe_ids
        else:
            full_ids = observe_ids

        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
        truncated = len(full_ids) >= max_new_tokens

        # Compute soft_prefix_hash for atlas contamination detection
        sp_hash = hashlib.sha256(sp.cpu().float().numpy().tobytes()).hexdigest()[:16]

        return GenerationResult(
            output_ids=full_ids,
            output_text=full_text,
            token_telemetry=token_telemetry,  # first max_observe_tokens tokens
            truncated=truncated,
            actual_prefix_rms=actual_prefix_rms,
            effective_rms=effective_rms,
            soft_prefix_hash=sp_hash,
            generation_start_index=generation_start_index,
            prefix_token_count=spec.token_count,
            prompt_token_count=input_ids.shape[1]
        )
```

**Note on lean re-run**: The current design re-runs generation for full output instead of continuing from KV cache. This is simpler and avoids cache continuation bugs, at the cost of 2× compute for the first 128 tokens. This is acceptable for Phase A data collection (science, not production). If this is a bottleneck, implement exact KV continuation and add it as a stress test.

---

## Component 3: EarlyFeatures Extractor

Extracts features from the `token_telemetry` field at each window.

```python
@dataclass
class EarlyFeatures:
    window: int
    n_observed: int              # min(window, actual_generated_length)
    feature_extractor_version: str = "1.0"

    # Availability
    logprobs_available: bool = False
    attentions_available: bool = False

    # Token features (always)
    token_ids: List[int] = field(default_factory=list)
    eos_appeared: bool = False
    truncated_at_window: bool = False

    # Logprob features (if available)
    mean_logprob: Optional[float] = None
    logprob_slope: Optional[float] = None   # None if n_observed < 2
    cumulative_logprob: Optional[float] = None

    # Entropy features (if available)
    token_entropy_mean: Optional[float] = None
    token_entropy_slope: Optional[float] = None  # None if n_observed < 2

    # Attention features (if available)
    # Definition: mean fraction of attention in prefix positions {0,...,prefix_token_count-1}
    # Averaged over layers {1,2,3,4} (0-indexed), all heads, all query positions in window
    attention_sink_mass: Optional[float] = None
    attention_sink_mass_definition: str = "mean_over_layers_1_to_4_all_heads_attending_to_prefix_positions"

    # Format features (always)
    think_token_fraction: Optional[float] = None
    repetition_rate: float = 0.0  # fraction of repeated 4-grams (0.0 for n_observed < 4)

    # Metadata
    generation_position_ids_policy: str = "default"  # "default" or "shifted"
```

---

## Component 4: Output Validator

```python
# Arithmetic
def validate_arithmetic(output_text: str, ground_truth: int) -> ValidationResult:
    # extract_answer: search OUTSIDE <think>...</think> blocks only
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    output_no_think = think_pattern.sub('', output_text)
    last_int = extract_last_integer(output_no_think)
    correct = (last_int == ground_truth) if last_int is not None else False
    converged = last_int is not None

    # answer_anywhere: search full output including <think>
    answer_anywhere = any integer in full output == ground_truth

    return ValidationResult(
        correct=correct,
        answer_anywhere_correct=answer_anywhere,
        converged=converged,
        score=float(correct),
        answer_anywhere_search_scope="full_output_including_think",
        final_answer_search_scope="output_excluding_think"
    )

# Qualitative (legal/planning) — oracle-relative label
# Labels are assigned AFTER all N+1 candidates in a task group are judged
def assign_oracle_relative_labels(
    validation_results: List[ValidationResult],
    judge_scores: List[float],
    min_oracle_margin: float = 0.5  # winner must beat median by this amount
) -> List[ValidationResult]:
    median_score = statistics.median(judge_scores)
    oracle_idx = int(argmax(judge_scores))
    oracle_score = judge_scores[oracle_idx]
    oracle_margin = oracle_score - median_score

    for i, result in enumerate(validation_results):
        if oracle_margin >= min_oracle_margin:
            result.correct = (i == oracle_idx)
            result.oracle_margin = oracle_margin
            result.oracle_winner_idx = oracle_idx
        else:
            result.correct = None  # degenerate task group — excluded from MI
            result.oracle_margin = oracle_margin
            result.degenerate_task_group = True
    return validation_results
```

---

## Component 5: Atlas Logger

```python
# Schema version 1.0 — frozen at preregistration
ATLAS_RECORD = {
    "schema_version": "1.0",
    "timestamp": str,
    "experiment_id": str,
    "code_commit": str,                    # git SHA
    "phase_a_version": str,               # package version

    "model": {
        "model_id": str,
        "hf_revision": str,               # HuggingFace commit SHA
        "quantization": str,
        "quantization_config_hash": str,
        "tokenizer_hash": str,
        "chat_template_hash": str,
        "generation_config_hash": str,
        "torch_version": str,
        "transformers_version": str,
        "bitsandbytes_version": str,
        "cuda_version": str,
        "embedding_rms": float
    },

    "task": {
        "task_id": str,
        "task_class": str,
        "task_hash": str,                 # hash of full task input text
        "train_test_split_id": str,
        "ground_truth": str               # serialized (int for arithmetic, rubric_hash for qualitative)
    },

    "spec": {
        "surface": str,
        "token_count": int,
        "rms_multiplier": float,
        "effective_rms": float,
        "actual_prefix_rms": float,
        "seed": int,
        "soft_prefix_hash": str
    },

    "generation": {
        "generation_start_index": int,
        "prompt_token_count": int,
        "prefix_token_count": int,
        "combined_input_length": int,
        "max_new_tokens": int,
        "max_observe_tokens": int,
        "position_ids_policy": str,
        "determinism_verified": bool,
        "generation_time_s": float
    },

    "is_baseline": bool,
    "candidate_order_in_group": int,      # 0=baseline, 1..N=prefix candidates
    "candidate_shuffle_hash": str,        # hash of candidate order for this group

    "early_features": {
        "1":   EarlyFeatures,
        "4":   EarlyFeatures,
        "8":   EarlyFeatures,
        "16":  EarlyFeatures,
        "32":  EarlyFeatures,
        "64":  EarlyFeatures,
        "128": EarlyFeatures
    },

    "full_output": str,
    "full_token_ids": List[int],
    "truncated": bool,

    "validation": {
        "correct": Optional[bool],
        "answer_anywhere_correct": Optional[bool],
        "converged": bool,
        "score": Optional[float],
        "oracle_relative_label": Optional[bool],
        "oracle_margin": Optional[float],
        "degenerate_task_group": bool,
        "judge_model": Optional[str],
        "judge_model_version": Optional[str],
        "judge_prompt_hash": Optional[str],
        "rubric_hash": Optional[str],
        "judge_blinding_verified": bool,
        "answer_anywhere_search_scope": str,
        "final_answer_search_scope": str
    },

    "selected_as_best": bool,
    "missing_features_not_imputed": bool  # ALWAYS True — no imputation ever
}
```

**Atomic write policy**: Each record written as a single `json.dumps(record) + '\n'`. On crash, last partial line is discarded on resume. Atlas is append-only. Records are never modified after write.

---

## Section 7: MI Analysis Spec (Preregistered — Amended)

### Estimator
**Primary**: `sklearn.feature_selection.mutual_info_classif` with `n_neighbors=5` (mixed discrete-continuous MI for binary target). NOT KSG (which requires continuous targets).

**Why**: The target (`correct`) is binary. KSG is designed for continuous-continuous MI and produces biased estimates with binary targets. `mutual_info_classif` handles mixed types correctly.

**Important**: `mutual_info_classif` computes MI **per feature**, not joint MI over all features. Do NOT sum per-feature MI values — redundancy makes sums invalid.

**H1 definition** (pre-registered — pick one, do not change after seeing data):
> H1 is the MI of the pre-registered scalar `routing_score` against `correct`, evaluated on the held-out test split. `routing_score = mean_logprob - 0.5 × attention_sink_mass + 0.1 × (1 - token_entropy_mean)` (weights frozen from preregistration). This is a single scalar → binary target MI, estimated with `mutual_info_classif` on the routing_score column.

Per-feature MI from `mutual_info_classif` is reported descriptively to understand which features contribute, but H1 is specifically tested on `routing_score`.

**Nats → bits conversion**: divide all MI values by `ln(2)`.

### Feature Normalization
**Method**: Within-task z-scoring — for each feature, normalize by `(value - task_group_mean) / task_group_std` before MI estimation. This removes task-difficulty confounding.

**Clarification**: Z-scoring is computed **per task group separately**, using only the candidates within that group. It is NOT computed across all task groups.

**Degenerate groups**: If `task_group_std < 1e-6` for a feature (all candidates have same value), that feature is marked degenerate for that group/window. Degenerate feature values are excluded from MI computation for that group; the group is not excluded entirely.

**Implementation**: Computed at analysis time from atlas data, not stored in atlas (prevents leakage at collection time).

**Missing values**: Features with `None` values (e.g., `logprob_slope` at w=1, `attention_sink_mass` when unavailable) are excluded from that window's MI computation. No imputation. Atlas field `missing_features_not_imputed = True` always.

### Evaluation Protocol
1. Load all atlas records for a given model/quantization/task_class
2. Filter: exclude `degenerate_task_group=True` records
3. Feature normalization: within-task z-score on train split
4. **Train split** (70% of task_ids): feature selection only (which features to include in H5 routing score). Feature selection must be completed before looking at test split.
5. **Test split** (30% of task_ids): MI estimation and oracle recall. No further feature tuning.
6. MI estimation: `mutual_info_classif` on test split features vs `correct` labels
7. Permutation null: 1000 permutations, shuffle `correct` within task groups (not across tasks)
8. Bootstrap CIs: 1000 bootstrap samples by task group
9. Significance: p < 0.05 vs permutation null

### H5 Clarification
- Candidate universe for H5: **10 prefix candidates only** (baseline excluded from routing competition)
- "Promote top-5 of 10" means: rank the 10 prefix candidates by routing score, promote top-5
- Oracle winner is the best-scoring prefix candidate (baseline excluded)
- Oracle recall = P(oracle-winner prefix ∈ promoted top-5) across held-out task groups

### Multiple Comparisons
- H1-H6 are pre-registered. No correction for multiple comparisons across hypotheses (each has its own acceptance criterion).
- Within H1, if testing across 7 windows: apply Bonferroni correction for 7 tests.

---

## Section 8: Data Collection Plan

### Phase 0: Preflight Tests (Required Before Any Data Collection)

Run these in order; each must pass before proceeding:

| Test | Assert |
|---|---|
| `probe_inputs_embeds.py` passes | generation_start_index determined, logprob aligned, zero-prefix = baseline |
| RMS assertion passes | actual_prefix_rms within 1% of effective_rms |
| JSONL crash/resume | partial last line discarded correctly on reload |
| Telemetry-vs-lean equivalence | observe + lean generates same full output as single lean run |
| Determinism smoke test | same seed produces byte-identical output on 2 consecutive runs |
| Attention aggregation sanity | attention_sink_mass = 1.0 for first token (trivially attends to itself) |

### Phase 1: Feasibility Pilot (Optional — Faster Signal)

Goal: Get a non-confirmatory H1 signal in ~1 day before committing to full collection.

- 25-30 arithmetic task groups × 1 baseline + 5 prefix candidates
- `max_new_tokens=256` (arithmetic usually completes much sooner)
- `max_observe_tokens=64` (focus on first 64 tokens for feasibility check)
- Collect: token IDs, logprobs, entropy ONLY — skip attentions and hidden states
- Do NOT count toward hard MI gate
- Expected VRAM: ~4GB peak per candidate with Q4. Fast.

**Decision rule**: If pilot MI (w=64) is below 0.03 bits for arithmetic, reconsider Phase B entirely before running 100-group collection.

### Phase 2: Arithmetic Pilot (100 Groups)

- 100 arithmetic task groups × 1 baseline + 10 prefix candidates = 1,100 candidates
- Qwen3-4B Q4 first, then Qwen3-8B Q8 for H3 (separate run)
- `max_new_tokens=1024`, `max_observe_tokens=128`
- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
- Task groups: reuse existing 25 tasks + 75 new arithmetic tasks (generated fresh, not reused from published experiments — different seeds or different task difficulty levels)
- Expected runtime: ~8-12 hours per model on RTX 5090 (Qwen3-4B Q4 × 100 groups × 11 candidates × ~30s per candidate)

### Phase 3: Legal/Planning Pilots (After Arithmetic Completes)

- Legal: 100 task groups × 1 baseline + 5 prefix candidates (blinded judge required)
- Planning: 50 task groups × 1 baseline + 5 prefix candidates
- Add blinded judging protocol before running
- Task groups: 12 existing tasks + 88 new for legal; 5 existing + 45 new for planning

---

## Section 9: Phase B Kill/Proceed Rule (Explicit)

**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**

1. `I(routing_score_64; correct) >= 0.1 bits` with p < 0.05 vs permutation null (H1) — arithmetic Qwen3-4B Q4
2. Oracle winner recall ≥ 90% at ≤ 50% promotion on held-out tasks (H5) — prefix candidates only, baseline excluded

**Edge cases — explicit rules:**
- H1 = exactly 0.1 bits: **proceed** (threshold is inclusive)
- H5 = exactly 90% recall: **proceed** (threshold is inclusive)
- H1 > 0.1 bits but NOT significant (p ≥ 0.05): **extend to 200+ groups** — do not proceed or kill
- H1 MI in 0.05-0.1 bits (ambiguous): **extend to 200+ groups** — do not proceed or kill
- H5 oracle recall in 80-90% (ambiguous): **extend to 200+ groups** — do not proceed or kill
- After 200+ groups still ambiguous: **kill Phase B** — ambiguity is not a pass
- H1 MI ≤ 0.05 bits: **kill Phase B immediately** — no extension needed
- H5 recall < 80% at 50% promotion: **kill Phase B immediately**

**H5 tie handling:**
- If multiple prefix candidates are correct (≥ 1 correct prefix in promoted set): oracle recall = 1 for this group
- "Oracle winner" = any correct prefix candidate in the task group, not necessarily the highest-scoring
- If ALL 10 prefix candidates are wrong for a task group: that group is excluded from H5 (all-wrong groups have no oracle winner to retain)
- If 0 prefix candidates are correct but baseline is correct: group excluded from H5 (perturbation cannot help)

**Phase A alone ships if Phase B is killed.** See Section 5 for Phase A standalone value.

**Phase A alone ships if Phase B is killed.** Phase A is a reproducible validated candidate-generation and oracle evaluation system. It is scientifically valid and deployable as an offline oracle pipeline.

---

## Section 10: Stress Tests to Validate Before Implementation

### ST-0 (NEW): Generation-Path Confounding
**Test**: Prepending soft tokens changes position IDs for all downstream tokens under default HuggingFace behavior. For RoPE-based models (Qwen3), if `combined_embeds` has length `prefix_len + prompt_len`, the prompt tokens are assigned position IDs starting at `prefix_len`, not 0. This can change attention patterns throughout the sequence even for non-prefix positions.

**Validation**: Generate with 2-token prefix. Compare:
- Prompt attention patterns with prefix vs without prefix at the same prompt token positions
- Output text with prefix vs without prefix

If position shift causes large divergence even for semantically null prefixes (e.g., zero vectors), the mechanism may be primarily a position shift, not a content perturbation.

**Decision**: If position shift is the primary mechanism, add it to the mechanism description. If it's incidental, document it as a confound.

### ST-1: Sequence Alignment (CRITICAL — addressed in probe)
### ST-2: RMS Scaling (addressed in applicator)
### ST-3: Atlas Crash Safety (addressed in logger)
### ST-4: extract_answer Correctness (run on 10 existing outputs before use)
### ST-5: Early Feature Degeneracy (check feature variance before MI analysis)
### ST-6: Baseline Equivalence (confirmed by probe)
### ST-7: VRAM Budget (first-128 token mitigation applied)
### ST-8: Oracle Label Degeneracy (margin threshold applied)
### ST-9: CUDA Non-Determinism (smoke test in preflight)

---

## Implementation Order (What NOT to Build Until Each Gate Passes)

```
1. Component 0 (Probe) ← BUILD FIRST
   Gate: probe passes all assertions; boundary_mode saved to config

2. Component 1 (Spec Generator) ← simple, no gates
3. Component 2 (Applicator) ← depends on probe's boundary_mode
   Gate: applicator passes telemetry-vs-lean equivalence test
         (assert first len(observe_ids) tokens match lean rerun)

4. Component 3 (EarlyFeatures Extractor) ← depends on applicator
5. Component 5 (Atlas Logger) ← depends on schema finalization

6. Component 4a (Output Validator — arithmetic) ← MUST BE DONE BEFORE DATA COLLECTION
   Gate: validate on 10 existing outputs manually before use

7. Preflight test suite ← runs all ST-0 through ST-9

8. Phase 1 feasibility pilot ← 25-30 groups
   Gate: MI feasibility check (if < 0.03 bits, reconsider)

9. Phase 2 arithmetic pilot ← 100 groups
   Gate: H1 + H5 acceptance criteria

10. Component 4b (Output Validator — qualitative) ← depends on judge blinding protocol
    Gate: judge blinding protocol verified

11. Phase 3 legal/planning pilots

   IF H1 + H5 pass:
12. Phase B Blueprint (separate Tesla session)
    ELSE:
12. Package Phase A as standalone system
```

---

## What NOT to Build (Final Explicit Exclusions)

- **No Observer-Router (Phase B)** until H1 + H5 pass
- **No multi-surface interventions** until input_prefix is fully validated
- **No fixed W projection** — deleted permanently from mainline
- **No latent evolution loop** — deleted until landscape is confirmed smooth
- **No scalar latent scorer as primary** — output validator is the only primary signal
- **No `encoder.decode()` path** — deprecated; raises `DeprecationError`
- **No imputation of missing features** — `missing_features_not_imputed = True` always
- **No heuristic quality fallback** — abstain rather than score by length/format
- **No claims from old logs** — existing logs are pilot-only; no hard MI gate claims
- **No Phase B based on ambiguous MI** — 0.05-0.10 bits range requires more data, not a decision
