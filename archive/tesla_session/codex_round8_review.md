Reading additional input from stdin...
OpenAI Codex v0.118.0 (research preview)
--------
workdir: C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019d92b5-cc7a-73a0-a44c-aacd2531a307
--------
user
You are in TESLA MODE — Round 8 convergence check.

Read the file tesla_session/phase6_blueprint.md (the full Phase A Blueprint) and tesla_session/codex_final_verdict.md (your Round 7 review).

Round 7 identified 5 code-level bugs. All 5 have been fixed in the Blueprint:

1. FIXED: Component 2 applicator body now uses local variable from _get_start_index(), not self.generation_start_index. All 3 usage sites (observe, lean, return) updated.

2. FIXED: Lean rerun prefix-equivalence assertion added: assert torch.equal(full_ids[:len(observe_ids)], observe_ids)

3. FIXED: Zero-prefix probe generate() call now includes output_scores=True. Boundary check uses strict ValueError for unexpected lengths.

4. FIXED: Phase B threshold changed from '> 0.1 bits' to '>= 0.1 bits' throughout. All edge cases consistent.

5. FIXED: H1/H5 typo in edge cases corrected — 'H1 = exactly 90% recall' changed to 'H5 = exactly 90% recall'.

YOUR TASK:
1. Verify each fix is correctly applied in the Blueprint by reading the actual file.
2. Check for any remaining inconsistencies, stale references, or new issues introduced by the fixes.
3. Determine: Is the Blueprint now ready for engineering handoff? Can an engineer build from this without asking questions?
4. If YES: declare convergence explicitly. If NO: list remaining gate-blocking issues.

Be thorough. Read the actual pseudocode, not just the amendment notes.
codex
I’ll read the current blueprint and the prior verdict directly, then cross-check each Round 7 finding against the actual pseudocode and surrounding text for stale references.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Content -Raw -LiteralPath 'tesla_session/phase6_blueprint.md'" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Content -Raw -LiteralPath 'tesla_session/codex_final_verdict.md'" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 succeeded in 403ms:
# Tesla Mode Phase 6: The Blueprint

## Status
Approved by Codex Round 5. Phase A Blueprint only.
Phase B (Observer-Router) is conditional ƒ?" Blueprint will be written only after H1 + H5 pass.

## Amendments from Round 5 (Incorporated)
- Missing stress test: generation-path confounding added (Section 10)
- MI estimator corrected: discrete-continuous, not KSG (Section 7)
- Feature normalization specified: within-task z-score (Section 7)
- H5 clarified: baseline excluded from the 10 candidates (Section 7)
- Cheaper H1 feasibility path added (Section 8)
- Phase B explicit kill/proceed rule added (Section 9)

## Amendments from Round 6 (Incorporated)
- `generation_start_index` replaced with `boundary_mode` policy ƒ?" computed per-call from `combined_length` (Component 0 + 2)
- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
- Lean rerun prefix-equivalence assertion added to Component 2 contract
- MI H1 defined as scalar `routing_score` MI ƒ?" not summed per-feature MI (Section 7)
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
    test_query = "What is 7 A- 8?"
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

    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index ƒ?" varies per prompt length)
    # HuggingFace behavior: with inputs_embeds, output.sequences may or may not
    # include dummy prefix tokens. We detect the mode once and save it as a policy.
    sequences = output.sequences[0]          # shape: (N,) where N is unknown
    n_scores = len(output.scores)            # number of generated tokens (always correct)
    n_sequences = sequences.shape[0]

    # ASSERT: scores length must equal generated tokens
    assert n_scores <= 64, f"Too many scores: {n_scores}"

    # Detect boundary mode ƒ?" saved as a string policy, NOT as a fixed integer
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

    # Per-call index derivation (NOT a saved scalar ƒ?" combined_length changes per prompt):
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
    # Use len(output.scores) not prompt_token_count + 64 ƒ?" generation may stop early
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

    # 10. Report ƒ?" save boundary_mode, NOT a fixed generation_start_index
    probe_result = {
        "prompt_token_count": prompt_token_count,
        "prefix_token_count": token_count,
        "combined_input_length": combined_input_length,
        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
        # NOTE: generation_start_index is NOT saved ƒ?" it is computed per-call as:
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
    Returns: [baseline_spec] + [n_prefix_candidates A- prefix_specs]
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
        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" ƒ?" from probe
        # Calibrate embedding RMS once at init
        self.embedding_rms = model.get_input_embeddings().weight.detach().square().mean().sqrt().item()

    def _get_start_index(self, combined_length: int) -> int:
        """Per-call computation ƒ?" combined_length varies per prompt, so cannot be a fixed scalar."""
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
                    "Lean rerun diverged from observe run ƒ?" telemetry may have altered generation"
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

**Note on lean re-run**: The current design re-runs generation for full output instead of continuing from KV cache. This is simpler and avoids cache continuation bugs, at the cost of 2A- compute for the first 128 tokens. This is acceptable for Phase A data collection (science, not production). If this is a bottleneck, implement exact KV continuation and add it as a stress test.

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

# Qualitative (legal/planning) ƒ?" oracle-relative label
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
            result.correct = None  # degenerate task group ƒ?" excluded from MI
            result.oracle_margin = oracle_margin
            result.degenerate_task_group = True
    return validation_results
```

---

## Component 5: Atlas Logger

```python
# Schema version 1.0 ƒ?" frozen at preregistration
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
    "missing_features_not_imputed": bool  # ALWAYS True ƒ?" no imputation ever
}
```

**Atomic write policy**: Each record written as a single `json.dumps(record) + '\n'`. On crash, last partial line is discarded on resume. Atlas is append-only. Records are never modified after write.

---

## Section 7: MI Analysis Spec (Preregistered ƒ?" Amended)

### Estimator
**Primary**: `sklearn.feature_selection.mutual_info_classif` with `n_neighbors=5` (mixed discrete-continuous MI for binary target). NOT KSG (which requires continuous targets).

**Why**: The target (`correct`) is binary. KSG is designed for continuous-continuous MI and produces biased estimates with binary targets. `mutual_info_classif` handles mixed types correctly.

**Important**: `mutual_info_classif` computes MI **per feature**, not joint MI over all features. Do NOT sum per-feature MI values ƒ?" redundancy makes sums invalid.

**H1 definition** (pre-registered ƒ?" pick one, do not change after seeing data):
> H1 is the MI of the pre-registered scalar `routing_score` against `correct`, evaluated on the held-out test split. `routing_score = mean_logprob - 0.5 A- attention_sink_mass + 0.1 A- (1 - token_entropy_mean)` (weights frozen from preregistration). This is a single scalar ƒ+' binary target MI, estimated with `mutual_info_classif` on the routing_score column.

Per-feature MI from `mutual_info_classif` is reported descriptively to understand which features contribute, but H1 is specifically tested on `routing_score`.

**Nats ƒ+' bits conversion**: divide all MI values by `ln(2)`.

### Feature Normalization
**Method**: Within-task z-scoring ƒ?" for each feature, normalize by `(value - task_group_mean) / task_group_std` before MI estimation. This removes task-difficulty confounding.

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
- Oracle recall = P(oracle-winner prefix ƒ^^ promoted top-5) across held-out task groups

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

### Phase 1: Feasibility Pilot (Optional ƒ?" Faster Signal)

Goal: Get a non-confirmatory H1 signal in ~1 day before committing to full collection.

- 25-30 arithmetic task groups A- 1 baseline + 5 prefix candidates
- `max_new_tokens=256` (arithmetic usually completes much sooner)
- `max_observe_tokens=64` (focus on first 64 tokens for feasibility check)
- Collect: token IDs, logprobs, entropy ONLY ƒ?" skip attentions and hidden states
- Do NOT count toward hard MI gate
- Expected VRAM: ~4GB peak per candidate with Q4. Fast.

**Decision rule**: If pilot MI (w=64) is below 0.03 bits for arithmetic, reconsider Phase B entirely before running 100-group collection.

### Phase 2: Arithmetic Pilot (100 Groups)

- 100 arithmetic task groups A- 1 baseline + 10 prefix candidates = 1,100 candidates
- Qwen3-4B Q4 first, then Qwen3-8B Q8 for H3 (separate run)
- `max_new_tokens=1024`, `max_observe_tokens=128`
- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
- Task groups: reuse existing 25 tasks + 75 new arithmetic tasks (generated fresh, not reused from published experiments ƒ?" different seeds or different task difficulty levels)
- Expected runtime: ~8-12 hours per model on RTX 5090 (Qwen3-4B Q4 A- 100 groups A- 11 candidates A- ~30s per candidate)

### Phase 3: Legal/Planning Pilots (After Arithmetic Completes)

- Legal: 100 task groups A- 1 baseline + 5 prefix candidates (blinded judge required)
- Planning: 50 task groups A- 1 baseline + 5 prefix candidates
- Add blinded judging protocol before running
- Task groups: 12 existing tasks + 88 new for legal; 5 existing + 45 new for planning

---

## Section 9: Phase B Kill/Proceed Rule (Explicit)

**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**

1. `I(routing_score_64; correct) >= 0.1 bits` with p < 0.05 vs permutation null (H1) ƒ?" arithmetic Qwen3-4B Q4
2. Oracle winner recall ƒ% 90% at ƒ% 50% promotion on held-out tasks (H5) ƒ?" prefix candidates only, baseline excluded

**Edge cases ƒ?" explicit rules:**
- H1 = exactly 0.1 bits: **proceed** (threshold is inclusive)
- H5 = exactly 90% recall: **proceed** (threshold is inclusive)
- H1 > 0.1 bits but NOT significant (p ƒ% 0.05): **extend to 200+ groups** ƒ?" do not proceed or kill
- H1 MI in 0.05-0.1 bits (ambiguous): **extend to 200+ groups** ƒ?" do not proceed or kill
- H5 oracle recall in 80-90% (ambiguous): **extend to 200+ groups** ƒ?" do not proceed or kill
- After 200+ groups still ambiguous: **kill Phase B** ƒ?" ambiguity is not a pass
- H1 MI ƒ% 0.05 bits: **kill Phase B immediately** ƒ?" no extension needed
- H5 recall < 80% at 50% promotion: **kill Phase B immediately**

**H5 tie handling:**
- If multiple prefix candidates are correct (ƒ% 1 correct prefix in promoted set): oracle recall = 1 for this group
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

### ST-1: Sequence Alignment (CRITICAL ƒ?" addressed in probe)
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
1. Component 0 (Probe) ƒ+? BUILD FIRST
   Gate: probe passes all assertions; boundary_mode saved to config

2. Component 1 (Spec Generator) ƒ+? simple, no gates
3. Component 2 (Applicator) ƒ+? depends on probe's boundary_mode
   Gate: applicator passes telemetry-vs-lean equivalence test
         (assert first len(observe_ids) tokens match lean rerun)

4. Component 3 (EarlyFeatures Extractor) ƒ+? depends on applicator
5. Component 5 (Atlas Logger) ƒ+? depends on schema finalization

6. Component 4a (Output Validator ƒ?" arithmetic) ƒ+? MUST BE DONE BEFORE DATA COLLECTION
   Gate: validate on 10 existing outputs manually before use

7. Preflight test suite ƒ+? runs all ST-0 through ST-9

8. Phase 1 feasibility pilot ƒ+? 25-30 groups
   Gate: MI feasibility check (if < 0.03 bits, reconsider)

9. Phase 2 arithmetic pilot ƒ+? 100 groups
   Gate: H1 + H5 acceptance criteria

10. Component 4b (Output Validator ƒ?" qualitative) ƒ+? depends on judge blinding protocol
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
- **No fixed W projection** ƒ?" deleted permanently from mainline
- **No latent evolution loop** ƒ?" deleted until landscape is confirmed smooth
- **No scalar latent scorer as primary** ƒ?" output validator is the only primary signal
- **No `encoder.decode()` path** ƒ?" deprecated; raises `DeprecationError`
- **No imputation of missing features** ƒ?" `missing_features_not_imputed = True` always
- **No heuristic quality fallback** ƒ?" abstain rather than score by length/format
- **No claims from old logs** ƒ?" existing logs are pilot-only; no hard MI gate claims
- **No Phase B based on ambiguous MI** ƒ?" 0.05-0.10 bits range requires more data, not a decision

Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 401ms:
Reading additional input from stdin...
OpenAI Codex v0.118.0 (research preview)
--------
workdir: C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019d92b0-6cae-73f3-a29d-93ba516ce343
--------
user
Read tesla_session/phase6_blueprint.md and tesla_session/codex_phase6_review.md. You identified 7 fixes needed. Check if they were applied in the amended Blueprint. If all 7 are correct: declare 'Design is converged. Blueprint is ready for engineering handoff.' If not: state what must change. Keep response under 500 words.
codex
Iƒ?Tll read the blueprint and the review side by side, then map the seven requested fixes against the amended blueprint rather than relying on memory.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Content -Raw -LiteralPath 'tesla_session/phase6_blueprint.md'" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Content -Raw -LiteralPath 'tesla_session/codex_phase6_review.md'" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 succeeded in 313ms:
**Gate Verdict**

Not ready to hand to an engineer yet.

The architecture has converged, but the Blueprint still has one gate-blocking implementation flaw: `generation_start_index` is treated as a fixed scalar saved from the probe and reused globally. That is only valid when `output.sequences` contains generated tokens only. If the HF path returns input-echo/dummy-prefix tokens, the correct start index is `combined_input_length`, which changes per prompt and per prefix length. The probe at `phase6_blueprint.md:74-92` correctly detects a boundary mode for one example, but `phase6_blueprint.md:166-168` and the applicator at `phase6_blueprint.md:224-228`, `phase6_blueprint.md:284`, and `phase6_blueprint.md:314` turn that into a global integer. That will silently mis-slice every prompt with a different length.

The fix is simple but mandatory: save a `generation_boundary_mode`, not a fixed index. For example:

```python
if n_sequences == n_scores:
    boundary_mode = "generated_only"
elif n_sequences == combined_input_length + n_scores:
    boundary_mode = "input_echo"
else:
    raise ...
```

Then compute per call:

```python
generation_start_index = 0 if boundary_mode == "generated_only" else combined_length
```

The zero-prefix baseline check also needs the same policy. Its current slice check at `phase6_blueprint.md:136-141` compares against `prompt_token_count + 64`, which breaks when generation stops before `max_new_tokens`. Use `len(output.scores)` or the same boundary detector there too.

**Component Review**

Component 0 is directionally right but not correct as written because of the fixed-index issue above. Once changed to a boundary-mode probe, it is the right first component.

Component 2’'?Ts observe-plus-lean design is sound for Phase A because it reruns lean generation rather than trying fragile KV continuation. That is acceptable science-first engineering. But it must assert that the lean rerun’'?Ts first `len(observe_ids)` tokens exactly match the observed run. Otherwise telemetry-enabled generation could differ from lean generation and contaminate labels. The preflight test at `phase6_blueprint.md:589` says this, but the applicator contract should require the assertion.

Component 5’'?Ts Atlas schema is largely sufficient for contamination control. It now includes the important provenance: model revision, quantization hash, tokenizer/chat/generation config hashes, code commit, schema version, task hash, split ID, soft-prefix hash, candidate order hash, judge/rubric hashes, and missing-feature policy. I would still add `analysis_code_commit` or `analysis_version`, exact split manifest/hash, attention implementation, dtype/device, and driver version, but those are hardening fields, not architecture blockers.

**MI Analysis**

`sklearn.feature_selection.mutual_info_classif(n_neighbors=5)` is the right family for a binary target, and the Blueprint correctly rejects KSG for the binary-label case. But the spec must state that sklearn’'?Ts function estimates MI per feature, not joint MI for the whole feature vector. If H1 means `I(features_64; correct)` as a multivariate quantity, `mutual_info_classif` does not directly provide that. The Blueprint should define one of these before implementation:

1. H1 is the maximum/tested per-feature MI after train-split feature selection.
2. H1 is MI of the preregistered scalar `routing_score`.
3. H1 is MI of a trained train-split classifier score on held-out tasks.

Do not sum sklearn per-feature MI values; redundancy makes that invalid.

Within-task z-scoring is the right normalization because it suppresses task-difficulty confounding. The spec should clarify that z-scoring is computed separately inside each task group using only candidate features from that group, with `std < eps` treated as a degenerate feature for that group/window.

The 70/30 split by `task_id` is correct. It should be deterministic by split manifest or hash seed, and the same task split should be reused across model variants for H3-style comparisons.

**Implementation Order**

Mostly correct, but the ordering currently hides a dependency: the arithmetic Output Validator must exist before the feasibility pilot and the 100-group arithmetic pilot. `phase6_blueprint.md:687` says ’'?oComponent 4 ’'?" qualitative’'?? after Phase 2, which is fine only if the arithmetic validator is implemented earlier. Split Component 4 explicitly:

1. Arithmetic validator before Phase 1/2 data collection.
2. Qualitative judge validator before legal/planning pilots.

**Phase B Rule**

The proceed rule is directionally clear: Phase B only if H1 and H5 both pass. But the kill/extend wording at `phase6_blueprint.md:631-635` is internally ambiguous because it says ’'?okilled if’'?? and then describes ambiguous cases that require extension.

Also fix edge cases:

- Define what happens at exactly `0.1` bits and exactly `90%`.
- Define what happens when H1 is `>0.1` but not significant.
- Define final decision after extending to 200+ groups.
- Define H5 ties: if multiple prefix candidates are correct, does any promoted correct prefix count?
- Define all-wrong task groups: they should not create arbitrary ’'?ooracle winners’'?? for H5.

Without that, H5 can be gamed by tie-heavy or all-failed groups.

**Missing Items That Would Cause Questions**

The items most likely to make an engineer stop and ask are:

- Boundary mode vs fixed `generation_start_index`.
- Per-feature vs joint/scalar MI interpretation.
- Arithmetic validator placement in the implementation order.
- H5 tie/all-wrong oracle policy.
- Exact deterministic split manifest/hash.
- Lean rerun prefix-equivalence assertion.

**Convergence Check**

All substantive architectural concerns from Rounds 1-5 have been addressed: no `encoder.decode()` path, no W projection, no latent evolution, Phase B gated behind H1/H5, raw `inputs_embeds` first, calibrated RMS, atlas provenance, baseline logging, missing-feature non-imputation, task-held-out evaluation, and generation-path confounding.

There is no remaining concern that would change the architecture. The design has converged.

But the Blueprint is not yet ready for engineering handoff until the sequence-boundary artifact is changed from a saved scalar `generation_start_index` to a per-call boundary policy. That is the single remaining gate-blocking gap.
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 313ms:
# Tesla Mode Phase 6: The Blueprint

## Status
Approved by Codex Round 5. Phase A Blueprint only.
Phase B (Observer-Router) is conditional ’'?" Blueprint will be written only after H1 + H5 pass.

## Amendments from Round 5 (Incorporated)
- Missing stress test: generation-path confounding added (Section 10)
- MI estimator corrected: discrete-continuous, not KSG (Section 7)
- Feature normalization specified: within-task z-score (Section 7)
- H5 clarified: baseline excluded from the 10 candidates (Section 7)
- Cheaper H1 feasibility path added (Section 8)
- Phase B explicit kill/proceed rule added (Section 9)

## Amendments from Round 6 (Incorporated)
- `generation_start_index` replaced with `boundary_mode` policy ’'?" computed per-call from `combined_length` (Component 0 + 2)
- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
- Lean rerun prefix-equivalence assertion added to Component 2 contract
- MI H1 defined as scalar `routing_score` MI ’'?" not summed per-feature MI (Section 7)
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
    test_query = "What is 7 A- 8?"
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

    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index ’'?" varies per prompt length)
    # HuggingFace behavior: with inputs_embeds, output.sequences may or may not
    # include dummy prefix tokens. We detect the mode once and save it as a policy.
    sequences = output.sequences[0]          # shape: (N,) where N is unknown
    n_scores = len(output.scores)            # number of generated tokens (always correct)
    n_sequences = sequences.shape[0]

    # ASSERT: scores length must equal generated tokens
    assert n_scores <= 64, f"Too many scores: {n_scores}"

    # Detect boundary mode ’'?" saved as a string policy, NOT as a fixed integer
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

    # Per-call index derivation (NOT a saved scalar ’'?" combined_length changes per prompt):
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
            return_dict_in_generate=True
        )
    zero_sequences = zero_output.sequences[0]
    # Use len(output.scores) not prompt_token_count + 64 ’'?" generation may stop early
    n_zero_scores = len(zero_output.scores)
    if zero_sequences.shape[0] == n_zero_scores:
        zero_generated_ids = zero_sequences                     # generated_only mode
    else:
        zero_generated_ids = zero_sequences[prompt_token_count:]  # input_echo mode
    zero_text = tokenizer.decode(zero_generated_ids, skip_special_tokens=False)

    assert zero_text == baseline_text, \
        f"ZERO-PREFIX DOES NOT MATCH BASELINE. inputs_embeds path is broken.\n" \
        f"Baseline: {repr(baseline_text)}\nZero-prefix: {repr(zero_text)}"

    # 10. Report ’'?" save boundary_mode, NOT a fixed generation_start_index
    probe_result = {
        "prompt_token_count": prompt_token_count,
        "prefix_token_count": token_count,
        "combined_input_length": combined_input_length,
        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
        # NOTE: generation_start_index is NOT saved ’'?" it is computed per-call as:
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
    Returns: [baseline_spec] + [n_prefix_candidates A- prefix_specs]
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
        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" ’'?" from probe
        # Calibrate embedding RMS once at init
        self.embedding_rms = model.get_input_embeddings().weight.detach().square().mean().sqrt().item()

    def _get_start_index(self, combined_length: int) -> int:
        """Per-call computation ’'?" combined_length varies per prompt, so cannot be a fixed scalar."""
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
        observe_ids = observe_output.sequences[0][self.generation_start_index:]
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
                full_ids = lean_output.sequences[0][self.generation_start_index:]
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
            generation_start_index=self.generation_start_index,
            prefix_token_count=spec.token_count,
            prompt_token_count=input_ids.shape[1]
        )
```

**Note on lean re-run**: The current design re-runs generation for full output instead of continuing from KV cache. This is simpler and avoids cache continuation bugs, at the cost of 2A- compute for the first 128 tokens. This is acceptable for Phase A data collection (science, not production). If this is a bottleneck, implement exact KV continuation and add it as a stress test.

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

# Qualitative (legal/planning) ’'?" oracle-relative label
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
            result.correct = None  # degenerate task group ’'?" excluded from MI
            result.oracle_margin = oracle_margin
            result.degenerate_task_group = True
    return validation_results
```

---

## Component 5: Atlas Logger

```python
# Schema version 1.0 ’'?" frozen at preregistration
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
    "missing_features_not_imputed": bool  # ALWAYS True ’'?" no imputation ever
}
```

**Atomic write policy**: Each record written as a single `json.dumps(record) + '\n'`. On crash, last partial line is discarded on resume. Atlas is append-only. Records are never modified after write.

---

## Section 7: MI Analysis Spec (Preregistered ’'?" Amended)

### Estimator
**Primary**: `sklearn.feature_selection.mutual_info_classif` with `n_neighbors=5` (mixed discrete-continuous MI for binary target). NOT KSG (which requires continuous targets).

**Why**: The target (`correct`) is binary. KSG is designed for continuous-continuous MI and produces biased estimates with binary targets. `mutual_info_classif` handles mixed types correctly.

**Important**: `mutual_info_classif` computes MI **per feature**, not joint MI over all features. Do NOT sum per-feature MI values ’'?" redundancy makes sums invalid.

**H1 definition** (pre-registered ’'?" pick one, do not change after seeing data):
> H1 is the MI of the pre-registered scalar `routing_score` against `correct`, evaluated on the held-out test split. `routing_score = mean_logprob - 0.5 A- attention_sink_mass + 0.1 A- (1 - token_entropy_mean)` (weights frozen from preregistration). This is a single scalar ’'+' binary target MI, estimated with `mutual_info_classif` on the routing_score column.

Per-feature MI from `mutual_info_classif` is reported descriptively to understand which features contribute, but H1 is specifically tested on `routing_score`.

**Nats ’'+' bits conversion**: divide all MI values by `ln(2)`.

### Feature Normalization
**Method**: Within-task z-scoring ’'?" for each feature, normalize by `(value - task_group_mean) / task_group_std` before MI estimation. This removes task-difficulty confounding.

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
- Oracle recall = P(oracle-winner prefix ’'^^ promoted top-5) across held-out task groups

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

### Phase 1: Feasibility Pilot (Optional ’'?" Faster Signal)

Goal: Get a non-confirmatory H1 signal in ~1 day before committing to full collection.

- 25-30 arithmetic task groups A- 1 baseline + 5 prefix candidates
- `max_new_tokens=256` (arithmetic usually completes much sooner)
- `max_observe_tokens=64` (focus on first 64 tokens for feasibility check)
- Collect: token IDs, logprobs, entropy ONLY ’'?" skip attentions and hidden states
- Do NOT count toward hard MI gate
- Expected VRAM: ~4GB peak per candidate with Q4. Fast.

**Decision rule**: If pilot MI (w=64) is below 0.03 bits for arithmetic, reconsider Phase B entirely before running 100-group collection.

### Phase 2: Arithmetic Pilot (100 Groups)

- 100 arithmetic task groups A- 1 baseline + 10 prefix candidates = 1,100 candidates
- Qwen3-4B Q4 first, then Qwen3-8B Q8 for H3 (separate run)
- `max_new_tokens=1024`, `max_observe_tokens=128`
- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
- Task groups: reuse existing 25 tasks + 75 new arithmetic tasks (generated fresh, not reused from published experiments ’'?" different seeds or different task difficulty levels)
- Expected runtime: ~8-12 hours per model on RTX 5090 (Qwen3-4B Q4 A- 100 groups A- 11 candidates A- ~30s per candidate)

### Phase 3: Legal/Planning Pilots (After Arithmetic Completes)

- Legal: 100 task groups A- 1 baseline + 5 prefix candidates (blinded judge required)
- Planning: 50 task groups A- 1 baseline + 5 prefix candidates
- Add blinded judging protocol before running
- Task groups: 12 existing tasks + 88 new for legal; 5 existing + 45 new for planning

---

## Section 9: Phase B Kill/Proceed Rule (Explicit)

**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**

1. `I(routing_score_64; correct) > 0.1 bits` with p < 0.05 vs permutation null (H1) ’'?" arithmetic Qwen3-4B Q4
2. Oracle winner recall ’'%A? 90% at ’'% 50% promotion on held-out tasks (H5) ’'?" prefix candidates only, baseline excluded

**Edge cases ’'?" explicit rules:**
- H1 = exactly 0.1 bits: **proceed** (threshold is inclusive)
- H1 = exactly 90% recall: **proceed** (threshold is inclusive)
- H1 > 0.1 bits but NOT significant (p ’'%A? 0.05): **extend to 200+ groups** ’'?" do not proceed or kill
- H1 MI in 0.05-0.1 bits (ambiguous): **extend to 200+ groups** ’'?" do not proceed or kill
- H5 oracle recall in 80-90% (ambiguous): **extend to 200+ groups** ’'?" do not proceed or kill
- After 200+ groups still ambiguous: **kill Phase B** ’'?" ambiguity is not a pass
- H1 MI ’'% 0.05 bits: **kill Phase B immediately** ’'?" no extension needed
- H5 recall < 80% at 50% promotion: **kill Phase B immediately**

**H5 tie handling:**
- If multiple prefix candidates are correct (’'%A? 1 correct prefix in promoted set): oracle recall = 1 for this group
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

### ST-1: Sequence Alignment (CRITICAL ’'?" addressed in probe)
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
1. Component 0 (Probe) ’'+? BUILD FIRST
   Gate: probe passes all assertions; boundary_mode saved to config

2. Component 1 (Spec Generator) ’'+? simple, no gates
3. Component 2 (Applicator) ’'+? depends on probe's boundary_mode
   Gate: applicator passes telemetry-vs-lean equivalence test
         (assert first len(observe_ids) tokens match lean rerun)

4. Component 3 (EarlyFeatures Extractor) ’'+? depends on applicator
5. Component 5 (Atlas Logger) ’'+? depends on schema finalization

6. Component 4a (Output Validator ’'?" arithmetic) ’'+? MUST BE DONE BEFORE DATA COLLECTION
   Gate: validate on 10 existing outputs manually before use

7. Preflight test suite ’'+? runs all ST-0 through ST-9

8. Phase 1 feasibility pilot ’'+? 25-30 groups
   Gate: MI feasibility check (if < 0.03 bits, reconsider)

9. Phase 2 arithmetic pilot ’'+? 100 groups
   Gate: H1 + H5 acceptance criteria

10. Component 4b (Output Validator ’'?" qualitative) ’'+? depends on judge blinding protocol
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
- **No fixed W projection** ’'?" deleted permanently from mainline
- **No latent evolution loop** ’'?" deleted until landscape is confirmed smooth
- **No scalar latent scorer as primary** ’'?" output validator is the only primary signal
- **No `encoder.decode()` path** ’'?" deprecated; raises `DeprecationError`
- **No imputation of missing features** ’'?" `missing_features_not_imputed = True` always
- **No heuristic quality fallback** ’'?" abstain rather than score by length/format
- **No claims from old logs** ’'?" existing logs are pilot-only; no hard MI gate claims
- **No Phase B based on ambiguous MI** ’'?" 0.05-0.10 bits range requires more data, not a decision

Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

codex
I found the amendment list, but Iƒ?Tm checking the actual pseudocode now because a changelog line can say a fix landed while the component body still contradicts it.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n "generation_start_index|boundary_mode|zero_output|output_scores|lean|routing_score|Within-task|Degenerate|Component 4a|Edge cases|H5 tie|all-wrong|split manifest|split" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n "Missing Items|Boundary mode|per-feature|Within-task|Implementation Order|Phase B Rule|lean|zero-prefix|exactly|H5 ties|all-wrong" "tesla_session/codex_phase6_review.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 succeeded in 256ms:
16:- `generation_start_index` replaced with `boundary_mode` policy ƒ?" computed per-call from `combined_length` (Component 0 + 2)
19:- MI H1 defined as scalar `routing_score` MI ƒ?" not summed per-feature MI (Section 7)
20:- Within-task z-scoring clarified: per group, std < eps = degenerate excluded (Section 7)
22:- Phase B kill/proceed edge cases fixed: exact thresholds, H1-significant-but-below-0.1, H5 ties, all-wrong groups (Section 9)
79:            output_scores=True,
83:    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index ƒ?" varies per prompt length)
95:        boundary_mode = "generated_only"     # sequences contains ONLY generated tokens
97:        boundary_mode = "input_echo"         # sequences contains prefix + generated tokens
106:    # generation_start_index = 0 if boundary_mode == "generated_only" else combined_length
109:    generation_start_index = 0 if boundary_mode == "generated_only" else combined_input_length
110:    generated_ids = sequences[generation_start_index:]
138:        zero_output = model.generate(
146:    zero_sequences = zero_output.sequences[0]
148:    n_zero_scores = len(zero_output.scores)
159:    # 10. Report ƒ?" save boundary_mode, NOT a fixed generation_start_index
164:        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
165:        # NOTE: generation_start_index is NOT saved ƒ?" it is computed per-call as:
166:        #   0 if boundary_mode == "generated_only" else combined_length
181:**Save `boundary_mode` to config. Every subsequent component computes `generation_start_index` per-call as `0 if boundary_mode == "generated_only" else combined_length`. Never use a fixed scalar.**
238:    def __init__(self, model, tokenizer, device, boundary_mode: str):
242:        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" ƒ?" from probe
248:        return 0 if self.boundary_mode == "generated_only" else combined_length
286:        # Generation with telemetry (two-phase: observe + lean)
295:                output_scores=True,
302:        observe_ids = observe_output.sequences[0][self.generation_start_index:]
315:        # Phase 2: continue lean generation for remaining tokens
322:                    lean_output = self.model.generate(
328:                        output_scores=False,
332:                full_ids = lean_output.sequences[0][self.generation_start_index:]
352:            generation_start_index=self.generation_start_index,
358:**Note on lean re-run**: The current design re-runs generation for full output instead of continuing from KV cache. This is simpler and avoids cache continuation bugs, at the cost of 2A- compute for the first 128 tokens. This is acceptable for Phase A data collection (science, not production). If this is a bottleneck, implement exact KV continuation and add it as a stress test.
487:        "train_test_split_id": str,
502:        "generation_start_index": int,
567:> H1 is the MI of the pre-registered scalar `routing_score` against `correct`, evaluated on the held-out test split. `routing_score = mean_logprob - 0.5 A- attention_sink_mass + 0.1 A- (1 - token_entropy_mean)` (weights frozen from preregistration). This is a single scalar ƒ+' binary target MI, estimated with `mutual_info_classif` on the routing_score column.
569:Per-feature MI from `mutual_info_classif` is reported descriptively to understand which features contribute, but H1 is specifically tested on `routing_score`.
574:**Method**: Within-task z-scoring ƒ?" for each feature, normalize by `(value - task_group_mean) / task_group_std` before MI estimation. This removes task-difficulty confounding.
578:**Degenerate groups**: If `task_group_std < 1e-6` for a feature (all candidates have same value), that feature is marked degenerate for that group/window. Degenerate feature values are excluded from MI computation for that group; the group is not excluded entirely.
587:3. Feature normalization: within-task z-score on train split
588:4. **Train split** (70% of task_ids): feature selection only (which features to include in H5 routing score). Feature selection must be completed before looking at test split.
589:5. **Test split** (30% of task_ids): MI estimation and oracle recall. No further feature tuning.
590:6. MI estimation: `mutual_info_classif` on test split features vs `correct` labels
615:| `probe_inputs_embeds.py` passes | generation_start_index determined, logprob aligned, zero-prefix = baseline |
618:| Telemetry-vs-lean equivalence | observe + lean generates same full output as single lean run |
640:- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
655:**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**
657:1. `I(routing_score_64; correct) > 0.1 bits` with p < 0.05 vs permutation null (H1) ƒ?" arithmetic Qwen3-4B Q4
660:**Edge cases ƒ?" explicit rules:**
670:**H5 tie handling:**
673:- If ALL 10 prefix candidates are wrong for a task group: that group is excluded from H5 (all-wrong groups have no oracle winner to retain)
711:   Gate: probe passes all assertions; boundary_mode saved to config
714:3. Component 2 (Applicator) ƒ+? depends on probe's boundary_mode
715:   Gate: applicator passes telemetry-vs-lean equivalence test
716:         (assert first len(observe_ids) tokens match lean rerun)
721:6. Component 4a (Output Validator ƒ?" arithmetic) ƒ+? MUST BE DONE BEFORE DATA COLLECTION
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 257ms:
24:The zero-prefix baseline check also needs the same policy. Its current slice check at `phase6_blueprint.md:136-141` compares against `prompt_token_count + 64`, which breaks when generation stops before `max_new_tokens`. Use `len(output.scores)` or the same boundary detector there too.
30:Component 2ƒ?Ts observe-plus-lean design is sound for Phase A because it reruns lean generation rather than trying fragile KV continuation. That is acceptable science-first engineering. But it must assert that the lean rerunƒ?Ts first `len(observe_ids)` tokens exactly match the observed run. Otherwise telemetry-enabled generation could differ from lean generation and contaminate labels. The preflight test at `phase6_blueprint.md:589` says this, but the applicator contract should require the assertion.
38:1. H1 is the maximum/tested per-feature MI after train-split feature selection.
42:Do not sum sklearn per-feature MI values; redundancy makes that invalid.
44:Within-task z-scoring is the right normalization because it suppresses task-difficulty confounding. The spec should clarify that z-scoring is computed separately inside each task group using only candidate features from that group, with `std < eps` treated as a degenerate feature for that group/window.
48:**Implementation Order**
55:**Phase B Rule**
61:- Define what happens at exactly `0.1` bits and exactly `90%`.
64:- Define H5 ties: if multiple prefix candidates are correct, does any promoted correct prefix count?
65:- Define all-wrong task groups: they should not create arbitrary ƒ?ooracle winnersƒ?? for H5.
69:**Missing Items That Would Cause Questions**
73:- Boundary mode vs fixed `generation_start_index`.
76:- H5 tie/all-wrong oracle policy.
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

2026-04-15T19:49:27.950202Z ERROR codex_core::tools::router: error=`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=655; $i -le 675; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }'` rejected: blocked by policy
2026-04-15T19:49:27.950205Z ERROR codex_core::tools::router: error=`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=292; $i -le 354; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }'` rejected: blocked by policy
2026-04-15T19:49:27.950205Z ERROR codex_core::tools::router: error=`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=130; $i -le 155; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }'` rejected: blocked by policy
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=130; $i -le 155; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=292; $i -le 354; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=655; $i -le 675; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 declined in 0ms:
`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=292; $i -le 354; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=130; $i -le 155; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content -LiteralPath '"'tesla_session/phase6_blueprint.md'; for ("'$i=655; $i -le 675; $i++) { '"'{0}: {1}' -f "'$i, $lines[$i-1] }'` rejected: blocked by policy
codex
The amended blueprint does incorporate most of the requested analysis changes, but the component pseudocode still has a boundary-index contradiction. Iƒ?Tm checking whether that is isolated or paired with the zero-prefix check.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 6 "zero_output|n_zero_scores|zero_generated_ids|output_scores=True" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 6 "observe_ids|lean_output|full_ids|generation_start_index=self|_get_start_index" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 4 "H1 = exactly|Oracle winner|ALL 10|Component 4a|routing_score|Degenerate groups" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 succeeded in 275ms:
73-        output = model.generate(
74-            inputs_embeds=combined_embeds,
75-            attention_mask=attention_mask,
76-            input_ids=None,
77-            max_new_tokens=64,
78-            do_sample=False,
79:            output_scores=True,
80-            return_dict_in_generate=True
81-        )
82-
83-    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index ƒ?" varies per prompt length)
84-    # HuggingFace behavior: with inputs_embeds, output.sequences may or may not
85-    # include dummy prefix tokens. We detect the mode once and save it as a policy.
--
132-
133-    # NOTE: baseline and prefixed outputs will differ (that's the point of perturbation)
134-    # But ZERO-prefix must match baseline:
135-    zero_embeds = input_embeds  # no prefix
136-    zero_mask = torch.ones(1, prompt_token_count, dtype=torch.long, device=device)
137-    with torch.no_grad():
138:        zero_output = model.generate(
139-            inputs_embeds=zero_embeds,
140-            attention_mask=zero_mask,
141-            input_ids=None,
142-            max_new_tokens=64,
143-            do_sample=False,
144-            return_dict_in_generate=True
145-        )
146:    zero_sequences = zero_output.sequences[0]
147-    # Use len(output.scores) not prompt_token_count + 64 ƒ?" generation may stop early
148:    n_zero_scores = len(zero_output.scores)
149:    if zero_sequences.shape[0] == n_zero_scores:
150:        zero_generated_ids = zero_sequences                     # generated_only mode
151-    else:
152:        zero_generated_ids = zero_sequences[prompt_token_count:]  # input_echo mode
153:    zero_text = tokenizer.decode(zero_generated_ids, skip_special_tokens=False)
154-
155-    assert zero_text == baseline_text, \
156-        f"ZERO-PREFIX DOES NOT MATCH BASELINE. inputs_embeds path is broken.\n" \
157-        f"Baseline: {repr(baseline_text)}\nZero-prefix: {repr(zero_text)}"
158-
159-    # 10. Report ƒ?" save boundary_mode, NOT a fixed generation_start_index
--
289-            observe_output = self.model.generate(
290-                inputs_embeds=combined_embeds,
291-                attention_mask=combined_mask,
292-                input_ids=None,
293-                max_new_tokens=max_observe_tokens,
294-                do_sample=False,
295:                output_scores=True,
296-                output_attentions=True,   # only for observe phase
297-                output_hidden_states=False,
298-                return_dict_in_generate=True
299-            )
300-
301-        # Extract telemetry from observe phase
--
634-
635-### Phase 2: Arithmetic Pilot (100 Groups)
636-
637-- 100 arithmetic task groups A- 1 baseline + 10 prefix candidates = 1,100 candidates
638-- Qwen3-4B Q4 first, then Qwen3-8B Q8 for H3 (separate run)
639-- `max_new_tokens=1024`, `max_observe_tokens=128`
640:- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
641-- Task groups: reuse existing 25 tasks + 75 new arithmetic tasks (generated fresh, not reused from published experiments ƒ?" different seeds or different task difficulty levels)
642-- Expected runtime: ~8-12 hours per model on RTX 5090 (Qwen3-4B Q4 A- 100 groups A- 11 candidates A- ~30s per candidate)
643-
644-### Phase 3: Legal/Planning Pilots (After Arithmetic Completes)
645-
646-- Legal: 100 task groups A- 1 baseline + 5 prefix candidates (blinded judge required)
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 287ms:
240-        self.tokenizer = tokenizer
241-        self.device = device
242-        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" ƒ?" from probe
243-        # Calibrate embedding RMS once at init
244-        self.embedding_rms = model.get_input_embeddings().weight.detach().square().mean().sqrt().item()
245-
246:    def _get_start_index(self, combined_length: int) -> int:
247-        """Per-call computation ƒ?" combined_length varies per prompt, so cannot be a fixed scalar."""
248-        return 0 if self.boundary_mode == "generated_only" else combined_length
249-
250-    def apply_and_generate(
251-        self,
252-        input_ids: Tensor,
--
296-                output_attentions=True,   # only for observe phase
297-                output_hidden_states=False,
298-                return_dict_in_generate=True
299-            )
300-
301-        # Extract telemetry from observe phase
302:        observe_ids = observe_output.sequences[0][self.generation_start_index:]
303-        observe_scores = observe_output.scores   # List[Tensor(vocab_size)] length = min(max_observe_tokens, actual)
304-        observe_attentions = observe_output.attentions  # available for first max_observe_tokens tokens
305-
306-        # ASSERT: logprob alignment
307-        for t in range(len(observe_scores)):
308:            expected_id = observe_ids[t].item()
309-            max_id = observe_scores[t][0].argmax().item()
310-            assert expected_id == max_id, f"Logprob mismatch at step {t}"
311-
312-        # Extract per-token telemetry
313:        token_telemetry = self._extract_token_telemetry(observe_ids, observe_scores, observe_attentions)
314-
315-        # Phase 2: continue lean generation for remaining tokens
316-        # Use KV cache from observe phase to continue without re-running prefix
317-        if not observe_output.sequences[0].shape[0] >= combined_length + max_new_tokens:
318-            # Continue from the last observed token
319:            remaining_tokens = max_new_tokens - len(observe_ids)
320-            if remaining_tokens > 0 and not token_telemetry["eos_in_observe"]:
321-                with torch.no_grad():
322:                    lean_output = self.model.generate(
323-                        inputs_embeds=combined_embeds,
324-                        attention_mask=combined_mask,
325-                        input_ids=None,
326-                        max_new_tokens=max_new_tokens,  # generate full output fresh (simpler than cache)
327-                        do_sample=False,
328-                        output_scores=False,
329-                        output_attentions=False,
330-                        return_dict_in_generate=True
331-                    )
332:                full_ids = lean_output.sequences[0][self.generation_start_index:]
333-            else:
334:                full_ids = observe_ids
335-        else:
336:            full_ids = observe_ids
337-
338:        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
339:        truncated = len(full_ids) >= max_new_tokens
340-
341-        # Compute soft_prefix_hash for atlas contamination detection
342-        sp_hash = hashlib.sha256(sp.cpu().float().numpy().tobytes()).hexdigest()[:16]
343-
344-        return GenerationResult(
345:            output_ids=full_ids,
346-            output_text=full_text,
347-            token_telemetry=token_telemetry,  # first max_observe_tokens tokens
348-            truncated=truncated,
349-            actual_prefix_rms=actual_prefix_rms,
350-            effective_rms=effective_rms,
351-            soft_prefix_hash=sp_hash,
352:            generation_start_index=self.generation_start_index,
353-            prefix_token_count=spec.token_count,
354-            prompt_token_count=input_ids.shape[1]
355-        )
356-```
357-
358-**Note on lean re-run**: The current design re-runs generation for full output instead of continuing from KV cache. This is simpler and avoids cache continuation bugs, at the cost of 2A- compute for the first 128 tokens. This is acceptable for Phase A data collection (science, not production). If this is a bottleneck, implement exact KV continuation and add it as a stress test.
--
710-1. Component 0 (Probe) ƒ+? BUILD FIRST
711-   Gate: probe passes all assertions; boundary_mode saved to config
712-
713-2. Component 1 (Spec Generator) ƒ+? simple, no gates
714-3. Component 2 (Applicator) ƒ+? depends on probe's boundary_mode
715-   Gate: applicator passes telemetry-vs-lean equivalence test
716:         (assert first len(observe_ids) tokens match lean rerun)
717-
718-4. Component 3 (EarlyFeatures Extractor) ƒ+? depends on applicator
719-5. Component 5 (Atlas Logger) ƒ+? depends on schema finalization
720-
721-6. Component 4a (Output Validator ƒ?" arithmetic) ƒ+? MUST BE DONE BEFORE DATA COLLECTION
722-   Gate: validate on 10 existing outputs manually before use
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 284ms:
15-## Amendments from Round 6 (Incorporated)
16-- `generation_start_index` replaced with `boundary_mode` policy ƒ?" computed per-call from `combined_length` (Component 0 + 2)
17-- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
18-- Lean rerun prefix-equivalence assertion added to Component 2 contract
19:- MI H1 defined as scalar `routing_score` MI ƒ?" not summed per-feature MI (Section 7)
20-- Within-task z-scoring clarified: per group, std < eps = degenerate excluded (Section 7)
21-- Implementation order: arithmetic validator added before Phase 1/2 (Section 8)
22-- Phase B kill/proceed edge cases fixed: exact thresholds, H1-significant-but-below-0.1, H5 ties, all-wrong groups (Section 9)
23-
--
563-
564-**Important**: `mutual_info_classif` computes MI **per feature**, not joint MI over all features. Do NOT sum per-feature MI values ƒ?" redundancy makes sums invalid.
565-
566-**H1 definition** (pre-registered ƒ?" pick one, do not change after seeing data):
567:> H1 is the MI of the pre-registered scalar `routing_score` against `correct`, evaluated on the held-out test split. `routing_score = mean_logprob - 0.5 A- attention_sink_mass + 0.1 A- (1 - token_entropy_mean)` (weights frozen from preregistration). This is a single scalar ƒ+' binary target MI, estimated with `mutual_info_classif` on the routing_score column.
568-
569:Per-feature MI from `mutual_info_classif` is reported descriptively to understand which features contribute, but H1 is specifically tested on `routing_score`.
570-
571-**Nats ƒ+' bits conversion**: divide all MI values by `ln(2)`.
572-
573-### Feature Normalization
574-**Method**: Within-task z-scoring ƒ?" for each feature, normalize by `(value - task_group_mean) / task_group_std` before MI estimation. This removes task-difficulty confounding.
575-
576-**Clarification**: Z-scoring is computed **per task group separately**, using only the candidates within that group. It is NOT computed across all task groups.
577-
578:**Degenerate groups**: If `task_group_std < 1e-6` for a feature (all candidates have same value), that feature is marked degenerate for that group/window. Degenerate feature values are excluded from MI computation for that group; the group is not excluded entirely.
579-
580-**Implementation**: Computed at analysis time from atlas data, not stored in atlas (prevents leakage at collection time).
581-
582-**Missing values**: Features with `None` values (e.g., `logprob_slope` at w=1, `attention_sink_mass` when unavailable) are excluded from that window's MI computation. No imputation. Atlas field `missing_features_not_imputed = True` always.
--
594-
595-### H5 Clarification
596-- Candidate universe for H5: **10 prefix candidates only** (baseline excluded from routing competition)
597-- "Promote top-5 of 10" means: rank the 10 prefix candidates by routing score, promote top-5
598:- Oracle winner is the best-scoring prefix candidate (baseline excluded)
599-- Oracle recall = P(oracle-winner prefix ƒ^^ promoted top-5) across held-out task groups
600-
601-### Multiple Comparisons
602-- H1-H6 are pre-registered. No correction for multiple comparisons across hypotheses (each has its own acceptance criterion).
--
653-## Section 9: Phase B Kill/Proceed Rule (Explicit)
654-
655-**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**
656-
657:1. `I(routing_score_64; correct) > 0.1 bits` with p < 0.05 vs permutation null (H1) ƒ?" arithmetic Qwen3-4B Q4
658:2. Oracle winner recall ƒ% 90% at ƒ% 50% promotion on held-out tasks (H5) ƒ?" prefix candidates only, baseline excluded
659-
660-**Edge cases ƒ?" explicit rules:**
661:- H1 = exactly 0.1 bits: **proceed** (threshold is inclusive)
662:- H1 = exactly 90% recall: **proceed** (threshold is inclusive)
663-- H1 > 0.1 bits but NOT significant (p ƒ% 0.05): **extend to 200+ groups** ƒ?" do not proceed or kill
664-- H1 MI in 0.05-0.1 bits (ambiguous): **extend to 200+ groups** ƒ?" do not proceed or kill
665-- H5 oracle recall in 80-90% (ambiguous): **extend to 200+ groups** ƒ?" do not proceed or kill
666-- After 200+ groups still ambiguous: **kill Phase B** ƒ?" ambiguity is not a pass
--
668-- H5 recall < 80% at 50% promotion: **kill Phase B immediately**
669-
670-**H5 tie handling:**
671-- If multiple prefix candidates are correct (ƒ% 1 correct prefix in promoted set): oracle recall = 1 for this group
672:- "Oracle winner" = any correct prefix candidate in the task group, not necessarily the highest-scoring
673:- If ALL 10 prefix candidates are wrong for a task group: that group is excluded from H5 (all-wrong groups have no oracle winner to retain)
674-- If 0 prefix candidates are correct but baseline is correct: group excluded from H5 (perturbation cannot help)
675-
676-**Phase A alone ships if Phase B is killed.** See Section 5 for Phase A standalone value.
677-
--
717-
718-4. Component 3 (EarlyFeatures Extractor) ƒ+? depends on applicator
719-5. Component 5 (Atlas Logger) ƒ+? depends on schema finalization
720-
721:6. Component 4a (Output Validator ƒ?" arithmetic) ƒ+? MUST BE DONE BEFORE DATA COLLECTION
722-   Gate: validate on 10 existing outputs manually before use
723-
724-7. Preflight test suite ƒ+? runs all ST-0 through ST-9
725-
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

codex
Not all 7 fixes are correctly applied. Do **not** declare handoff yet.

Required changes:

1. **Component 2 still uses a fixed/undefined `self.generation_start_index`.**  
   The blueprint defines `_get_start_index(combined_length)`, but then slices with `self.generation_start_index` at observe, lean, and return paths. Replace those with a per-call local:
   `generation_start_index = self._get_start_index(combined_length)`.

2. **Component 2 does not actually assert lean rerun prefix-equivalence.**  
   The implementation order mentions the gate, but the applicator contract/body must assert:
   `full_ids[:len(observe_ids)] == observe_ids` after lean rerun.

3. **Zero-prefix probe uses `len(zero_output.scores)` but does not request scores.**  
   Add `output_scores=True` to the zero-prefix `model.generate(...)` call. Also make the zero-prefix boundary check strict: accept only `n_zero_scores` or `prompt_token_count + n_zero_scores`; otherwise raise.

4. **Phase B threshold wording is internally inconsistent.**  
   Main rule says `I(routing_score_64; correct) > 0.1 bits`, but edge case says exactly `0.1` proceeds. Change main rule to `>= 0.1 bits`.

5. **Phase B edge-case label typo:**  
   `H1 = exactly 90% recall` should be `H5 = exactly 90% recall`.

The other requested fixes are substantively present: scalar `routing_score` MI, no summed per-feature MI, per-task z-scoring with degenerate feature handling, arithmetic validator before data collection, H5 baseline exclusion, H5 tie/all-wrong handling, and Phase B extension/kill rules.
tokens used
21,981
Not all 7 fixes are correctly applied. Do **not** declare handoff yet.

Required changes:

1. **Component 2 still uses a fixed/undefined `self.generation_start_index`.**  
   The blueprint defines `_get_start_index(combined_length)`, but then slices with `self.generation_start_index` at observe, lean, and return paths. Replace those with a per-call local:
   `generation_start_index = self._get_start_index(combined_length)`.

2. **Component 2 does not actually assert lean rerun prefix-equivalence.**  
   The implementation order mentions the gate, but the applicator contract/body must assert:
   `full_ids[:len(observe_ids)] == observe_ids` after lean rerun.

3. **Zero-prefix probe uses `len(zero_output.scores)` but does not request scores.**  
   Add `output_scores=True` to the zero-prefix `model.generate(...)` call. Also make the zero-prefix boundary check strict: accept only `n_zero_scores` or `prompt_token_count + n_zero_scores`; otherwise raise.

4. **Phase B threshold wording is internally inconsistent.**  
   Main rule says `I(routing_score_64; correct) > 0.1 bits`, but edge case says exactly `0.1` proceeds. Change main rule to `>= 0.1 bits`.

5. **Phase B edge-case label typo:**  
   `H1 = exactly 90% recall` should be `H5 = exactly 90% recall`.

The other requested fixes are substantively present: scalar `routing_score` MI, no summed per-feature MI, per-task z-scoring with degenerate feature handling, arithmetic validator before data collection, H5 baseline exclusion, H5 tie/all-wrong handling, and Phase B extension/kill rules.

Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

codex
The current blueprint file differs from the older Round 7 transcript as expected; the first pass shows the five named fixes are present in the live blueprint. I’m now doing targeted stale-reference searches so the verdict is based on the active file, not the prior transcript embedded in the verdict file.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 3 \"self\\.generation_start_index|generation_start_index|_get_start_index|lean rerun|torch\\.equal\" \"tesla_session/phase6_blueprint.md\"" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 4 "zero_output|n_zero_scores|zero_generated_ids|output_scores=True|Zero-prefix" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 4 \"routing_score_64|exactly 0\\.1|exactly 90|> 0\\.1|>= 0\\.1|H5 = exactly|H1 = exactly 90|Oracle winner|ALL 10\" \"tesla_session/phase6_blueprint.md\"" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"generation_start_index|boundary_mode|prompt_token_count \\+ 64|len\\(zero_output\\.scores\\)|len\\(output\\.scores\\)|output_scores=True|self\\.generation_start_index|H1 = exactly 90|H5 = exactly 90|> 0\\.1 bits|>= 0\\.1 bits\" \"tesla_session/phase6_blueprint.md\"" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 succeeded in 475ms:
604-
605-### H5 Clarification
606-- Candidate universe for H5: **10 prefix candidates only** (baseline excluded from routing competition)
607-- "Promote top-5 of 10" means: rank the 10 prefix candidates by routing score, promote top-5
608:- Oracle winner is the best-scoring prefix candidate (baseline excluded)
609-- Oracle recall = P(oracle-winner prefix ∈ promoted top-5) across held-out task groups
610-
611-### Multiple Comparisons
612-- H1-H6 are pre-registered. No correction for multiple comparisons across hypotheses (each has its own acceptance criterion).
--
663-## Section 9: Phase B Kill/Proceed Rule (Explicit)
664-
665-**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**
666-
667:1. `I(routing_score_64; correct) >= 0.1 bits` with p < 0.05 vs permutation null (H1) — arithmetic Qwen3-4B Q4
668:2. Oracle winner recall ≥ 90% at ≤ 50% promotion on held-out tasks (H5) — prefix candidates only, baseline excluded
669-
670-**Edge cases — explicit rules:**
671:- H1 = exactly 0.1 bits: **proceed** (threshold is inclusive)
672:- H5 = exactly 90% recall: **proceed** (threshold is inclusive)
673:- H1 > 0.1 bits but NOT significant (p ≥ 0.05): **extend to 200+ groups** — do not proceed or kill
674-- H1 MI in 0.05-0.1 bits (ambiguous): **extend to 200+ groups** — do not proceed or kill
675-- H5 oracle recall in 80-90% (ambiguous): **extend to 200+ groups** — do not proceed or kill
676-- After 200+ groups still ambiguous: **kill Phase B** — ambiguity is not a pass
677-- H1 MI ≤ 0.05 bits: **kill Phase B immediately** — no extension needed
678-- H5 recall < 80% at 50% promotion: **kill Phase B immediately**
679-
680-**H5 tie handling:**
681-- If multiple prefix candidates are correct (≥ 1 correct prefix in promoted set): oracle recall = 1 for this group
682:- "Oracle winner" = any correct prefix candidate in the task group, not necessarily the highest-scoring
683:- If ALL 10 prefix candidates are wrong for a task group: that group is excluded from H5 (all-wrong groups have no oracle winner to retain)
684-- If 0 prefix candidates are correct but baseline is correct: group excluded from H5 (perturbation cannot help)
685-
686-**Phase A alone ships if Phase B is killed.** See Section 5 for Phase A standalone value.
687-
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 477ms:
13-- Phase B explicit kill/proceed rule added (Section 9)
14-
15-## Amendments from Round 6 (Incorporated)
16-- `generation_start_index` replaced with `boundary_mode` policy — computed per-call from `combined_length` (Component 0 + 2)
17:- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
18-- Lean rerun prefix-equivalence assertion added to Component 2 contract
19-- MI H1 defined as scalar `routing_score` MI — not summed per-feature MI (Section 7)
20-- Within-task z-scoring clarified: per group, std < eps = degenerate excluded (Section 7)
21-- Implementation order: arithmetic validator added before Phase 1/2 (Section 8)
--
75-            attention_mask=attention_mask,
76-            input_ids=None,
77-            max_new_tokens=64,
78-            do_sample=False,
79:            output_scores=True,
80-            return_dict_in_generate=True
81-        )
82-
83-    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index — varies per prompt length)
--
134-    # But ZERO-prefix must match baseline:
135-    zero_embeds = input_embeds  # no prefix
136-    zero_mask = torch.ones(1, prompt_token_count, dtype=torch.long, device=device)
137-    with torch.no_grad():
138:        zero_output = model.generate(
139-            inputs_embeds=zero_embeds,
140-            attention_mask=zero_mask,
141-            input_ids=None,
142-            max_new_tokens=64,
143-            do_sample=False,
144:            output_scores=True,           # needed for len(zero_output.scores)
145-            return_dict_in_generate=True
146-        )
147:    zero_sequences = zero_output.sequences[0]
148-    # Use len(output.scores) not prompt_token_count + 64 — generation may stop early
149:    n_zero_scores = len(zero_output.scores)
150:    if zero_sequences.shape[0] == n_zero_scores:
151:        zero_generated_ids = zero_sequences                     # generated_only mode
152:    elif zero_sequences.shape[0] == prompt_token_count + n_zero_scores:
153:        zero_generated_ids = zero_sequences[prompt_token_count:]  # input_echo mode
154-    else:
155-        raise ValueError(
156:            f"Zero-prefix boundary detection failed: sequences={zero_sequences.shape[0]}, "
157:            f"n_scores={n_zero_scores}, prompt_len={prompt_token_count}"
158-        )
159:    zero_text = tokenizer.decode(zero_generated_ids, skip_special_tokens=False)
160-
161-    assert zero_text == baseline_text, \
162-        f"ZERO-PREFIX DOES NOT MATCH BASELINE. inputs_embeds path is broken.\n" \
163:        f"Baseline: {repr(baseline_text)}\nZero-prefix: {repr(zero_text)}"
164-
165-    # 10. Report — save boundary_mode, NOT a fixed generation_start_index
166-    probe_result = {
167-        "prompt_token_count": prompt_token_count,
--
297-                attention_mask=combined_mask,
298-                input_ids=None,
299-                max_new_tokens=max_observe_tokens,
300-                do_sample=False,
301:                output_scores=True,
302-                output_attentions=True,   # only for observe phase
303-                output_hidden_states=False,
304-                return_dict_in_generate=True
305-            )
--
646-
647-- 100 arithmetic task groups × 1 baseline + 10 prefix candidates = 1,100 candidates
648-- Qwen3-4B Q4 first, then Qwen3-8B Q8 for H3 (separate run)
649-- `max_new_tokens=1024`, `max_observe_tokens=128`
650:- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
651-- Task groups: reuse existing 25 tasks + 75 new arithmetic tasks (generated fresh, not reused from published experiments — different seeds or different task difficulty levels)
652-- Expected runtime: ~8-12 hours per model on RTX 5090 (Qwen3-4B Q4 × 100 groups × 11 candidates × ~30s per candidate)
653-
654-### Phase 3: Legal/Planning Pilots (After Arithmetic Completes)
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 478ms:
13-- Phase B explicit kill/proceed rule added (Section 9)
14-
15-## Amendments from Round 6 (Incorporated)
16:- `generation_start_index` replaced with `boundary_mode` policy — computed per-call from `combined_length` (Component 0 + 2)
17-- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
18-- Lean rerun prefix-equivalence assertion added to Component 2 contract
19-- MI H1 defined as scalar `routing_score` MI — not summed per-feature MI (Section 7)
--
103-        )
104-
105-    # Per-call index derivation (NOT a saved scalar — combined_length changes per prompt):
106:    # generation_start_index = 0 if boundary_mode == "generated_only" else combined_length
107-
108-    # 6. Extract generated token IDs using the boundary mode
109:    generation_start_index = 0 if boundary_mode == "generated_only" else combined_input_length
110:    generated_ids = sequences[generation_start_index:]
111-    assert len(generated_ids) == n_scores, \
112-        f"Token ID slice mismatch: {len(generated_ids)} vs {n_scores}"
113-
--
162-        f"ZERO-PREFIX DOES NOT MATCH BASELINE. inputs_embeds path is broken.\n" \
163-        f"Baseline: {repr(baseline_text)}\nZero-prefix: {repr(zero_text)}"
164-
165:    # 10. Report — save boundary_mode, NOT a fixed generation_start_index
166-    probe_result = {
167-        "prompt_token_count": prompt_token_count,
168-        "prefix_token_count": token_count,
169-        "combined_input_length": combined_input_length,
170-        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
171:        # NOTE: generation_start_index is NOT saved — it is computed per-call as:
172-        #   0 if boundary_mode == "generated_only" else combined_length
173-        "n_generated_tokens": n_scores,
174-        "n_sequences": n_sequences,
--
184-```
185-
186-**This probe must be run and pass before writing any other Phase A component.**
187:**Save `boundary_mode` to config. Every subsequent component computes `generation_start_index` per-call as `0 if boundary_mode == "generated_only" else combined_length`. Never use a fixed scalar.**
188-
189----
190-
--
249-        # Calibrate embedding RMS once at init
250-        self.embedding_rms = model.get_input_embeddings().weight.detach().square().mean().sqrt().item()
251-
252:    def _get_start_index(self, combined_length: int) -> int:
253-        """Per-call computation — combined_length varies per prompt, so cannot be a fixed scalar."""
254-        return 0 if self.boundary_mode == "generated_only" else combined_length
255-
--
305-            )
306-
307-        # Extract telemetry from observe phase
308:        generation_start_index = self._get_start_index(combined_length)
309:        observe_ids = observe_output.sequences[0][generation_start_index:]
310-        observe_scores = observe_output.scores   # List[Tensor(vocab_size)] length = min(max_observe_tokens, actual)
311-        observe_attentions = observe_output.attentions  # available for first max_observe_tokens tokens
312-
--
336-                        output_attentions=False,
337-                        return_dict_in_generate=True
338-                    )
339:                full_ids = lean_output.sequences[0][generation_start_index:]
340:                # ASSERT: lean rerun prefix-equivalence
341:                assert torch.equal(full_ids[:len(observe_ids)], observe_ids), \
342-                    "Lean rerun diverged from observe run — telemetry may have altered generation"
343-            else:
344-                full_ids = observe_ids
--
359-            actual_prefix_rms=actual_prefix_rms,
360-            effective_rms=effective_rms,
361-            soft_prefix_hash=sp_hash,
362:            generation_start_index=generation_start_index,
363-            prefix_token_count=spec.token_count,
364-            prompt_token_count=input_ids.shape[1]
365-        )
--
509-    },
510-
511-    "generation": {
512:        "generation_start_index": int,
513-        "prompt_token_count": int,
514-        "prefix_token_count": int,
515-        "combined_input_length": int,
--
622-
623-| Test | Assert |
624-|---|---|
625:| `probe_inputs_embeds.py` passes | generation_start_index determined, logprob aligned, zero-prefix = baseline |
626-| RMS assertion passes | actual_prefix_rms within 1% of effective_rms |
627-| JSONL crash/resume | partial last line discarded correctly on reload |
628-| Telemetry-vs-lean equivalence | observe + lean generates same full output as single lean run |
--
723-2. Component 1 (Spec Generator) ← simple, no gates
724-3. Component 2 (Applicator) ← depends on probe's boundary_mode
725-   Gate: applicator passes telemetry-vs-lean equivalence test
726:         (assert first len(observe_ids) tokens match lean rerun)
727-
728-4. Component 3 (EarlyFeatures Extractor) ← depends on applicator
729-5. Component 5 (Atlas Logger) ← depends on schema finalization
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 480ms:
16:- `generation_start_index` replaced with `boundary_mode` policy — computed per-call from `combined_length` (Component 0 + 2)
17:- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
79:            output_scores=True,
83:    # 5. CRITICAL: Determine boundary_mode empirically (NOT a fixed index — varies per prompt length)
87:    n_scores = len(output.scores)            # number of generated tokens (always correct)
95:        boundary_mode = "generated_only"     # sequences contains ONLY generated tokens
97:        boundary_mode = "input_echo"         # sequences contains prefix + generated tokens
106:    # generation_start_index = 0 if boundary_mode == "generated_only" else combined_length
109:    generation_start_index = 0 if boundary_mode == "generated_only" else combined_input_length
110:    generated_ids = sequences[generation_start_index:]
144:            output_scores=True,           # needed for len(zero_output.scores)
148:    # Use len(output.scores) not prompt_token_count + 64 — generation may stop early
149:    n_zero_scores = len(zero_output.scores)
165:    # 10. Report — save boundary_mode, NOT a fixed generation_start_index
170:        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
171:        # NOTE: generation_start_index is NOT saved — it is computed per-call as:
172:        #   0 if boundary_mode == "generated_only" else combined_length
187:**Save `boundary_mode` to config. Every subsequent component computes `generation_start_index` per-call as `0 if boundary_mode == "generated_only" else combined_length`. Never use a fixed scalar.**
244:    def __init__(self, model, tokenizer, device, boundary_mode: str):
248:        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" — from probe
254:        return 0 if self.boundary_mode == "generated_only" else combined_length
301:                output_scores=True,
308:        generation_start_index = self._get_start_index(combined_length)
309:        observe_ids = observe_output.sequences[0][generation_start_index:]
339:                full_ids = lean_output.sequences[0][generation_start_index:]
362:            generation_start_index=generation_start_index,
512:        "generation_start_index": int,
625:| `probe_inputs_embeds.py` passes | generation_start_index determined, logprob aligned, zero-prefix = baseline |
650:- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
667:1. `I(routing_score_64; correct) >= 0.1 bits` with p < 0.05 vs permutation null (H1) — arithmetic Qwen3-4B Q4
672:- H5 = exactly 90% recall: **proceed** (threshold is inclusive)
673:- H1 > 0.1 bits but NOT significant (p ≥ 0.05): **extend to 200+ groups** — do not proceed or kill
721:   Gate: probe passes all assertions; boundary_mode saved to config
724:3. Component 2 (Applicator) ← depends on probe's boundary_mode
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

codex
The targeted search confirms the old `self.generation_start_index` sites are gone from the live blueprint, and the zero-prefix and threshold edits are in place. I’m doing one more pass over Component 2 and the H5/H1 language for contradictions that would still block implementation.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 90 "class RawSoftPrefixApplicator" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 60 "Section 7: MI Analysis Spec|Section 9: Phase B Kill/Proceed" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"TODO|FIXME|self\\.generation_start_index|prompt_token_count \\+ 64|> 0\\.1 bits|H1 = exactly 90|KSG|sum per-feature|encoder\\.decode|generation_boundary_mode|fixed scalar|input_echo|generated_only\" \"tesla_session/phase6_blueprint.md\"" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 succeeded in 355ms:
149-    n_zero_scores = len(zero_output.scores)
150-    if zero_sequences.shape[0] == n_zero_scores:
151-        zero_generated_ids = zero_sequences                     # generated_only mode
152-    elif zero_sequences.shape[0] == prompt_token_count + n_zero_scores:
153-        zero_generated_ids = zero_sequences[prompt_token_count:]  # input_echo mode
154-    else:
155-        raise ValueError(
156-            f"Zero-prefix boundary detection failed: sequences={zero_sequences.shape[0]}, "
157-            f"n_scores={n_zero_scores}, prompt_len={prompt_token_count}"
158-        )
159-    zero_text = tokenizer.decode(zero_generated_ids, skip_special_tokens=False)
160-
161-    assert zero_text == baseline_text, \
162-        f"ZERO-PREFIX DOES NOT MATCH BASELINE. inputs_embeds path is broken.\n" \
163-        f"Baseline: {repr(baseline_text)}\nZero-prefix: {repr(zero_text)}"
164-
165-    # 10. Report — save boundary_mode, NOT a fixed generation_start_index
166-    probe_result = {
167-        "prompt_token_count": prompt_token_count,
168-        "prefix_token_count": token_count,
169-        "combined_input_length": combined_input_length,
170-        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
171-        # NOTE: generation_start_index is NOT saved — it is computed per-call as:
172-        #   0 if boundary_mode == "generated_only" else combined_length
173-        "n_generated_tokens": n_scores,
174-        "n_sequences": n_sequences,
175-        "embedding_rms": embedding_rms,
176-        "effective_rms": effective_rms,
177-        "actual_prefix_rms": actual_prefix_rms,
178-        "generated_text_sample": generated_text[:200],
179-        "baseline_text_sample": baseline_text[:200],
180-        "zero_prefix_matches_baseline": True,   # asserted above
181-        "logprob_alignment_verified": True       # asserted above
182-    }
183-    return probe_result
184-```
185-
186-**This probe must be run and pass before writing any other Phase A component.**
187-**Save `boundary_mode` to config. Every subsequent component computes `generation_start_index` per-call as `0 if boundary_mode == "generated_only" else combined_length`. Never use a fixed scalar.**
188-
189----
190-
191-## Component 1: Spec Generator
192-
193-```python
194-# src/latent_reasoning/phase_a/spec_generator.py
195-
196-@dataclass
197-class PerturbationSpec:
198-    surface: Literal["input_prefix", "baseline"]
199-    token_count: int           # 0 for baseline
200-    rms_multiplier: float      # 1.0 = match embedding RMS
201-    seed: int                  # -1 for baseline
202-    model_id: str
203-    quantization: str
204-
205-def generate_specs(
206-    model_id: str,
207-    quantization: str,
208-    n_prefix_candidates: int,
209-    token_count: int = 2,
210-    rms_multiplier: float = 1.0,
211-    seed_offset: int = 0
212-) -> List[PerturbationSpec]:
213-    """
214-    Returns: [baseline_spec] + [n_prefix_candidates × prefix_specs]
215-    Baseline spec always first (seed=-1, surface=baseline, token_count=0).
216-    """
217-    baseline = PerturbationSpec(
218-        surface="baseline", token_count=0, rms_multiplier=0.0,
219-        seed=-1, model_id=model_id, quantization=quantization
220-    )
221-    prefix_specs = [
222-        PerturbationSpec(
223-            surface="input_prefix", token_count=token_count,
224-            rms_multiplier=rms_multiplier, seed=seed_offset + i,
225-            model_id=model_id, quantization=quantization
226-        )
227-        for i in range(n_prefix_candidates)
228-    ]
229-    return [baseline] + prefix_specs
230-```
231-
232----
233-
234-## Component 2: Raw Soft-Prefix Applicator
235-
236-```python
237-# src/latent_reasoning/phase_a/applicator.py
238-
239:class RawSoftPrefixApplicator:
240-    """
241-    Applies a PerturbationSpec to a model input and generates with full telemetry.
242-    Built on the validated mechanism from run_latent_sensitivity.py.
243-    """
244-    def __init__(self, model, tokenizer, device, boundary_mode: str):
245-        self.model = model
246-        self.tokenizer = tokenizer
247-        self.device = device
248-        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" — from probe
249-        # Calibrate embedding RMS once at init
250-        self.embedding_rms = model.get_input_embeddings().weight.detach().square().mean().sqrt().item()
251-
252-    def _get_start_index(self, combined_length: int) -> int:
253-        """Per-call computation — combined_length varies per prompt, so cannot be a fixed scalar."""
254-        return 0 if self.boundary_mode == "generated_only" else combined_length
255-
256-    def apply_and_generate(
257-        self,
258-        input_ids: Tensor,
259-        attention_mask: Tensor,
260-        spec: PerturbationSpec,
261-        max_new_tokens: int = 1024,
262-        max_observe_tokens: int = 128,
263-    ) -> GenerationResult:
264-
265-        if spec.surface == "baseline":
266-            # MUST match direct model.generate(input_ids=...) byte-for-byte
267-            return self._generate_baseline(input_ids, attention_mask, max_new_tokens)
268-
269-        # Generate soft prefix
270-        embed_layer = self.model.get_input_embeddings()
271-        input_embeds = embed_layer(input_ids)                   # (1, prompt_len, embed_dim)
272-        embed_dim = input_embeds.shape[-1]
273-
274-        gen = torch.Generator(device=self.device)
275-        gen.manual_seed(spec.seed)
276-        sp = torch.randn(
277-            1, spec.token_count, embed_dim,
278-            generator=gen, dtype=input_embeds.dtype, device=self.device
279-        )
280-        effective_rms = self.embedding_rms * spec.rms_multiplier
281-        current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
282-        sp = sp * (effective_rms / current_rms)
283-
284-        # ASSERT actual RMS matches target
285-        actual_prefix_rms = sp.square().mean().sqrt().item()
286-        assert abs(actual_prefix_rms - effective_rms) / effective_rms < 0.01
287-
288-        combined_embeds = torch.cat([sp, input_embeds], dim=1)
289-        combined_length = combined_embeds.shape[1]
290-        combined_mask = torch.ones(1, combined_length, dtype=torch.long, device=self.device)
291-
292-        # Generation with telemetry (two-phase: observe + lean)
293-        # Phase 1: collect telemetry for first max_observe_tokens
294-        with torch.no_grad():
295-            observe_output = self.model.generate(
296-                inputs_embeds=combined_embeds,
297-                attention_mask=combined_mask,
298-                input_ids=None,
299-                max_new_tokens=max_observe_tokens,
300-                do_sample=False,
301-                output_scores=True,
302-                output_attentions=True,   # only for observe phase
303-                output_hidden_states=False,
304-                return_dict_in_generate=True
305-            )
306-
307-        # Extract telemetry from observe phase
308-        generation_start_index = self._get_start_index(combined_length)
309-        observe_ids = observe_output.sequences[0][generation_start_index:]
310-        observe_scores = observe_output.scores   # List[Tensor(vocab_size)] length = min(max_observe_tokens, actual)
311-        observe_attentions = observe_output.attentions  # available for first max_observe_tokens tokens
312-
313-        # ASSERT: logprob alignment
314-        for t in range(len(observe_scores)):
315-            expected_id = observe_ids[t].item()
316-            max_id = observe_scores[t][0].argmax().item()
317-            assert expected_id == max_id, f"Logprob mismatch at step {t}"
318-
319-        # Extract per-token telemetry
320-        token_telemetry = self._extract_token_telemetry(observe_ids, observe_scores, observe_attentions)
321-
322-        # Phase 2: continue lean generation for remaining tokens
323-        # Use KV cache from observe phase to continue without re-running prefix
324-        if not observe_output.sequences[0].shape[0] >= combined_length + max_new_tokens:
325-            # Continue from the last observed token
326-            remaining_tokens = max_new_tokens - len(observe_ids)
327-            if remaining_tokens > 0 and not token_telemetry["eos_in_observe"]:
328-                with torch.no_grad():
329-                    lean_output = self.model.generate(
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 772ms:
9:- MI estimator corrected: discrete-continuous, not KSG (Section 7)
17:- Zero-prefix baseline check uses `len(output.scores)` not `prompt_token_count + 64` (Component 0)
95:        boundary_mode = "generated_only"     # sequences contains ONLY generated tokens
97:        boundary_mode = "input_echo"         # sequences contains prefix + generated tokens
106:    # generation_start_index = 0 if boundary_mode == "generated_only" else combined_length
109:    generation_start_index = 0 if boundary_mode == "generated_only" else combined_input_length
148:    # Use len(output.scores) not prompt_token_count + 64 — generation may stop early
151:        zero_generated_ids = zero_sequences                     # generated_only mode
153:        zero_generated_ids = zero_sequences[prompt_token_count:]  # input_echo mode
170:        "boundary_mode": boundary_mode,         # "generated_only" | "input_echo"
172:        #   0 if boundary_mode == "generated_only" else combined_length
187:**Save `boundary_mode` to config. Every subsequent component computes `generation_start_index` per-call as `0 if boundary_mode == "generated_only" else combined_length`. Never use a fixed scalar.**
248:        self.boundary_mode = boundary_mode  # "generated_only" | "input_echo" — from probe
253:        """Per-call computation — combined_length varies per prompt, so cannot be a fixed scalar."""
254:        return 0 if self.boundary_mode == "generated_only" else combined_length
570:**Primary**: `sklearn.feature_selection.mutual_info_classif` with `n_neighbors=5` (mixed discrete-continuous MI for binary target). NOT KSG (which requires continuous targets).
572:**Why**: The target (`correct`) is binary. KSG is designed for continuous-continuous MI and produces biased estimates with binary targets. `mutual_info_classif` handles mixed types correctly.
574:**Important**: `mutual_info_classif` computes MI **per feature**, not joint MI over all features. Do NOT sum per-feature MI values — redundancy makes sums invalid.
673:- H1 > 0.1 bits but NOT significant (p ≥ 0.05): **extend to 200+ groups** — do not proceed or kill
762:- **No `encoder.decode()` path** — deprecated; raises `DeprecationError`
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 781ms:
507-        "seed": int,
508-        "soft_prefix_hash": str
509-    },
510-
511-    "generation": {
512-        "generation_start_index": int,
513-        "prompt_token_count": int,
514-        "prefix_token_count": int,
515-        "combined_input_length": int,
516-        "max_new_tokens": int,
517-        "max_observe_tokens": int,
518-        "position_ids_policy": str,
519-        "determinism_verified": bool,
520-        "generation_time_s": float
521-    },
522-
523-    "is_baseline": bool,
524-    "candidate_order_in_group": int,      # 0=baseline, 1..N=prefix candidates
525-    "candidate_shuffle_hash": str,        # hash of candidate order for this group
526-
527-    "early_features": {
528-        "1":   EarlyFeatures,
529-        "4":   EarlyFeatures,
530-        "8":   EarlyFeatures,
531-        "16":  EarlyFeatures,
532-        "32":  EarlyFeatures,
533-        "64":  EarlyFeatures,
534-        "128": EarlyFeatures
535-    },
536-
537-    "full_output": str,
538-    "full_token_ids": List[int],
539-    "truncated": bool,
540-
541-    "validation": {
542-        "correct": Optional[bool],
543-        "answer_anywhere_correct": Optional[bool],
544-        "converged": bool,
545-        "score": Optional[float],
546-        "oracle_relative_label": Optional[bool],
547-        "oracle_margin": Optional[float],
548-        "degenerate_task_group": bool,
549-        "judge_model": Optional[str],
550-        "judge_model_version": Optional[str],
551-        "judge_prompt_hash": Optional[str],
552-        "rubric_hash": Optional[str],
553-        "judge_blinding_verified": bool,
554-        "answer_anywhere_search_scope": str,
555-        "final_answer_search_scope": str
556-    },
557-
558-    "selected_as_best": bool,
559-    "missing_features_not_imputed": bool  # ALWAYS True — no imputation ever
560-}
561-```
562-
563-**Atomic write policy**: Each record written as a single `json.dumps(record) + '\n'`. On crash, last partial line is discarded on resume. Atlas is append-only. Records are never modified after write.
564-
565----
566-
567:## Section 7: MI Analysis Spec (Preregistered — Amended)
568-
569-### Estimator
570-**Primary**: `sklearn.feature_selection.mutual_info_classif` with `n_neighbors=5` (mixed discrete-continuous MI for binary target). NOT KSG (which requires continuous targets).
571-
572-**Why**: The target (`correct`) is binary. KSG is designed for continuous-continuous MI and produces biased estimates with binary targets. `mutual_info_classif` handles mixed types correctly.
573-
574-**Important**: `mutual_info_classif` computes MI **per feature**, not joint MI over all features. Do NOT sum per-feature MI values — redundancy makes sums invalid.
575-
576-**H1 definition** (pre-registered — pick one, do not change after seeing data):
577-> H1 is the MI of the pre-registered scalar `routing_score` against `correct`, evaluated on the held-out test split. `routing_score = mean_logprob - 0.5 × attention_sink_mass + 0.1 × (1 - token_entropy_mean)` (weights frozen from preregistration). This is a single scalar → binary target MI, estimated with `mutual_info_classif` on the routing_score column.
578-
579-Per-feature MI from `mutual_info_classif` is reported descriptively to understand which features contribute, but H1 is specifically tested on `routing_score`.
580-
581-**Nats → bits conversion**: divide all MI values by `ln(2)`.
582-
583-### Feature Normalization
584-**Method**: Within-task z-scoring — for each feature, normalize by `(value - task_group_mean) / task_group_std` before MI estimation. This removes task-difficulty confounding.
585-
586-**Clarification**: Z-scoring is computed **per task group separately**, using only the candidates within that group. It is NOT computed across all task groups.
587-
588-**Degenerate groups**: If `task_group_std < 1e-6` for a feature (all candidates have same value), that feature is marked degenerate for that group/window. Degenerate feature values are excluded from MI computation for that group; the group is not excluded entirely.
589-
590-**Implementation**: Computed at analysis time from atlas data, not stored in atlas (prevents leakage at collection time).
591-
592-**Missing values**: Features with `None` values (e.g., `logprob_slope` at w=1, `attention_sink_mass` when unavailable) are excluded from that window's MI computation. No imputation. Atlas field `missing_features_not_imputed = True` always.
593-
594-### Evaluation Protocol
595-1. Load all atlas records for a given model/quantization/task_class
596-2. Filter: exclude `degenerate_task_group=True` records
597-3. Feature normalization: within-task z-score on train split
598-4. **Train split** (70% of task_ids): feature selection only (which features to include in H5 routing score). Feature selection must be completed before looking at test split.
599-5. **Test split** (30% of task_ids): MI estimation and oracle recall. No further feature tuning.
600-6. MI estimation: `mutual_info_classif` on test split features vs `correct` labels
601-7. Permutation null: 1000 permutations, shuffle `correct` within task groups (not across tasks)
602-8. Bootstrap CIs: 1000 bootstrap samples by task group
603-9. Significance: p < 0.05 vs permutation null
604-
605-### H5 Clarification
606-- Candidate universe for H5: **10 prefix candidates only** (baseline excluded from routing competition)
607-- "Promote top-5 of 10" means: rank the 10 prefix candidates by routing score, promote top-5
608-- Oracle winner is the best-scoring prefix candidate (baseline excluded)
609-- Oracle recall = P(oracle-winner prefix ∈ promoted top-5) across held-out task groups
610-
611-### Multiple Comparisons
612-- H1-H6 are pre-registered. No correction for multiple comparisons across hypotheses (each has its own acceptance criterion).
613-- Within H1, if testing across 7 windows: apply Bonferroni correction for 7 tests.
614-
615----
616-
617-## Section 8: Data Collection Plan
618-
619-### Phase 0: Preflight Tests (Required Before Any Data Collection)
620-
621-Run these in order; each must pass before proceeding:
622-
623-| Test | Assert |
624-|---|---|
625-| `probe_inputs_embeds.py` passes | generation_start_index determined, logprob aligned, zero-prefix = baseline |
626-| RMS assertion passes | actual_prefix_rms within 1% of effective_rms |
627-| JSONL crash/resume | partial last line discarded correctly on reload |
628-| Telemetry-vs-lean equivalence | observe + lean generates same full output as single lean run |
629-| Determinism smoke test | same seed produces byte-identical output on 2 consecutive runs |
630-| Attention aggregation sanity | attention_sink_mass = 1.0 for first token (trivially attends to itself) |
631-
632-### Phase 1: Feasibility Pilot (Optional — Faster Signal)
633-
634-Goal: Get a non-confirmatory H1 signal in ~1 day before committing to full collection.
635-
636-- 25-30 arithmetic task groups × 1 baseline + 5 prefix candidates
637-- `max_new_tokens=256` (arithmetic usually completes much sooner)
638-- `max_observe_tokens=64` (focus on first 64 tokens for feasibility check)
639-- Collect: token IDs, logprobs, entropy ONLY — skip attentions and hidden states
640-- Do NOT count toward hard MI gate
641-- Expected VRAM: ~4GB peak per candidate with Q4. Fast.
642-
643-**Decision rule**: If pilot MI (w=64) is below 0.03 bits for arithmetic, reconsider Phase B entirely before running 100-group collection.
644-
645-### Phase 2: Arithmetic Pilot (100 Groups)
646-
647-- 100 arithmetic task groups × 1 baseline + 10 prefix candidates = 1,100 candidates
648-- Qwen3-4B Q4 first, then Qwen3-8B Q8 for H3 (separate run)
649-- `max_new_tokens=1024`, `max_observe_tokens=128`
650-- `output_scores=True`, `output_attentions=True` (first 128 tokens only)
651-- Task groups: reuse existing 25 tasks + 75 new arithmetic tasks (generated fresh, not reused from published experiments — different seeds or different task difficulty levels)
652-- Expected runtime: ~8-12 hours per model on RTX 5090 (Qwen3-4B Q4 × 100 groups × 11 candidates × ~30s per candidate)
653-
654-### Phase 3: Legal/Planning Pilots (After Arithmetic Completes)
655-
656-- Legal: 100 task groups × 1 baseline + 5 prefix candidates (blinded judge required)
657-- Planning: 50 task groups × 1 baseline + 5 prefix candidates
658-- Add blinded judging protocol before running
659-- Task groups: 12 existing tasks + 88 new for legal; 5 existing + 45 new for planning
660-
661----
662-
663:## Section 9: Phase B Kill/Proceed Rule (Explicit)
664-
665-**Phase B (Observer-Router) proceeds ONLY if ALL of the following pass on held-out test split:**
666-
667-1. `I(routing_score_64; correct) >= 0.1 bits` with p < 0.05 vs permutation null (H1) — arithmetic Qwen3-4B Q4
668-2. Oracle winner recall ≥ 90% at ≤ 50% promotion on held-out tasks (H5) — prefix candidates only, baseline excluded
669-
670-**Edge cases — explicit rules:**
671-- H1 = exactly 0.1 bits: **proceed** (threshold is inclusive)
672-- H5 = exactly 90% recall: **proceed** (threshold is inclusive)
673-- H1 > 0.1 bits but NOT significant (p ≥ 0.05): **extend to 200+ groups** — do not proceed or kill
674-- H1 MI in 0.05-0.1 bits (ambiguous): **extend to 200+ groups** — do not proceed or kill
675-- H5 oracle recall in 80-90% (ambiguous): **extend to 200+ groups** — do not proceed or kill
676-- After 200+ groups still ambiguous: **kill Phase B** — ambiguity is not a pass
677-- H1 MI ≤ 0.05 bits: **kill Phase B immediately** — no extension needed
678-- H5 recall < 80% at 50% promotion: **kill Phase B immediately**
679-
680-**H5 tie handling:**
681-- If multiple prefix candidates are correct (≥ 1 correct prefix in promoted set): oracle recall = 1 for this group
682-- "Oracle winner" = any correct prefix candidate in the task group, not necessarily the highest-scoring
683-- If ALL 10 prefix candidates are wrong for a task group: that group is excluded from H5 (all-wrong groups have no oracle winner to retain)
684-- If 0 prefix candidates are correct but baseline is correct: group excluded from H5 (perturbation cannot help)
685-
686-**Phase A alone ships if Phase B is killed.** See Section 5 for Phase A standalone value.
687-
688-**Phase A alone ships if Phase B is killed.** Phase A is a reproducible validated candidate-generation and oracle evaluation system. It is scientifically valid and deployable as an offline oracle pipeline.
689-
690----
691-
692-## Section 10: Stress Tests to Validate Before Implementation
693-
694-### ST-0 (NEW): Generation-Path Confounding
695-**Test**: Prepending soft tokens changes position IDs for all downstream tokens under default HuggingFace behavior. For RoPE-based models (Qwen3), if `combined_embeds` has length `prefix_len + prompt_len`, the prompt tokens are assigned position IDs starting at `prefix_len`, not 0. This can change attention patterns throughout the sequence even for non-prefix positions.
696-
697-**Validation**: Generate with 2-token prefix. Compare:
698-- Prompt attention patterns with prefix vs without prefix at the same prompt token positions
699-- Output text with prefix vs without prefix
700-
701-If position shift causes large divergence even for semantically null prefixes (e.g., zero vectors), the mechanism may be primarily a position shift, not a content perturbation.
702-
703-**Decision**: If position shift is the primary mechanism, add it to the mechanism description. If it's incidental, document it as a confound.
704-
705-### ST-1: Sequence Alignment (CRITICAL — addressed in probe)
706-### ST-2: RMS Scaling (addressed in applicator)
707-### ST-3: Atlas Crash Safety (addressed in logger)
708-### ST-4: extract_answer Correctness (run on 10 existing outputs before use)
709-### ST-5: Early Feature Degeneracy (check feature variance before MI analysis)
710-### ST-6: Baseline Equivalence (confirmed by probe)
711-### ST-7: VRAM Budget (first-128 token mitigation applied)
712-### ST-8: Oracle Label Degeneracy (margin threshold applied)
713-### ST-9: CUDA Non-Determinism (smoke test in preflight)
714-
715----
716-
717-## Implementation Order (What NOT to Build Until Each Gate Passes)
718-
719-```
720-1. Component 0 (Probe) ← BUILD FIRST
721-   Gate: probe passes all assertions; boundary_mode saved to config
722-
723-2. Component 1 (Spec Generator) ← simple, no gates
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 45 \"lean_output = self\\.model\\.generate\" \"tesla_session/phase6_blueprint.md\"" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 35 "def validate_arithmetic|assign_oracle_relative_labels" "tesla_session/phase6_blueprint.md"' in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
2026-04-15T19:55:36.130492Z ERROR codex_core::tools::router: error=`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 25 \"ATLAS_RECORD|generation\\\": \\{|early_features|validation\\\": \\{\" \"tesla_session/phase6_blueprint.md\""` rejected: blocked by policy
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 25 \"ATLAS_RECORD|generation\\\": \\{|early_features|validation\\\": \\{\" \"tesla_session/phase6_blueprint.md\"" in C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning
 declined in 0ms:
`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 25 \"ATLAS_RECORD|generation\\\": \\{|early_features|validation\\\": \\{\" \"tesla_session/phase6_blueprint.md\""` rejected: blocked by policy
 succeeded in 335ms:
284-        # ASSERT actual RMS matches target
285-        actual_prefix_rms = sp.square().mean().sqrt().item()
286-        assert abs(actual_prefix_rms - effective_rms) / effective_rms < 0.01
287-
288-        combined_embeds = torch.cat([sp, input_embeds], dim=1)
289-        combined_length = combined_embeds.shape[1]
290-        combined_mask = torch.ones(1, combined_length, dtype=torch.long, device=self.device)
291-
292-        # Generation with telemetry (two-phase: observe + lean)
293-        # Phase 1: collect telemetry for first max_observe_tokens
294-        with torch.no_grad():
295-            observe_output = self.model.generate(
296-                inputs_embeds=combined_embeds,
297-                attention_mask=combined_mask,
298-                input_ids=None,
299-                max_new_tokens=max_observe_tokens,
300-                do_sample=False,
301-                output_scores=True,
302-                output_attentions=True,   # only for observe phase
303-                output_hidden_states=False,
304-                return_dict_in_generate=True
305-            )
306-
307-        # Extract telemetry from observe phase
308-        generation_start_index = self._get_start_index(combined_length)
309-        observe_ids = observe_output.sequences[0][generation_start_index:]
310-        observe_scores = observe_output.scores   # List[Tensor(vocab_size)] length = min(max_observe_tokens, actual)
311-        observe_attentions = observe_output.attentions  # available for first max_observe_tokens tokens
312-
313-        # ASSERT: logprob alignment
314-        for t in range(len(observe_scores)):
315-            expected_id = observe_ids[t].item()
316-            max_id = observe_scores[t][0].argmax().item()
317-            assert expected_id == max_id, f"Logprob mismatch at step {t}"
318-
319-        # Extract per-token telemetry
320-        token_telemetry = self._extract_token_telemetry(observe_ids, observe_scores, observe_attentions)
321-
322-        # Phase 2: continue lean generation for remaining tokens
323-        # Use KV cache from observe phase to continue without re-running prefix
324-        if not observe_output.sequences[0].shape[0] >= combined_length + max_new_tokens:
325-            # Continue from the last observed token
326-            remaining_tokens = max_new_tokens - len(observe_ids)
327-            if remaining_tokens > 0 and not token_telemetry["eos_in_observe"]:
328-                with torch.no_grad():
329:                    lean_output = self.model.generate(
330-                        inputs_embeds=combined_embeds,
331-                        attention_mask=combined_mask,
332-                        input_ids=None,
333-                        max_new_tokens=max_new_tokens,  # generate full output fresh (simpler than cache)
334-                        do_sample=False,
335-                        output_scores=False,
336-                        output_attentions=False,
337-                        return_dict_in_generate=True
338-                    )
339-                full_ids = lean_output.sequences[0][generation_start_index:]
340-                # ASSERT: lean rerun prefix-equivalence
341-                assert torch.equal(full_ids[:len(observe_ids)], observe_ids), \
342-                    "Lean rerun diverged from observe run — telemetry may have altered generation"
343-            else:
344-                full_ids = observe_ids
345-        else:
346-            full_ids = observe_ids
347-
348-        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
349-        truncated = len(full_ids) >= max_new_tokens
350-
351-        # Compute soft_prefix_hash for atlas contamination detection
352-        sp_hash = hashlib.sha256(sp.cpu().float().numpy().tobytes()).hexdigest()[:16]
353-
354-        return GenerationResult(
355-            output_ids=full_ids,
356-            output_text=full_text,
357-            token_telemetry=token_telemetry,  # first max_observe_tokens tokens
358-            truncated=truncated,
359-            actual_prefix_rms=actual_prefix_rms,
360-            effective_rms=effective_rms,
361-            soft_prefix_hash=sp_hash,
362-            generation_start_index=generation_start_index,
363-            prefix_token_count=spec.token_count,
364-            prompt_token_count=input_ids.shape[1]
365-        )
366-```
367-
368-**Note on lean re-run**: The current design re-runs generation for full output instead of continuing from KV cache. This is simpler and avoids cache continuation bugs, at the cost of 2× compute for the first 128 tokens. This is acceptable for Phase A data collection (science, not production). If this is a bottleneck, implement exact KV continuation and add it as a stress test.
369-
370----
371-
372-## Component 3: EarlyFeatures Extractor
373-
374-Extracts features from the `token_telemetry` field at each window.
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

 succeeded in 330ms:
386-
387-    # Token features (always)
388-    token_ids: List[int] = field(default_factory=list)
389-    eos_appeared: bool = False
390-    truncated_at_window: bool = False
391-
392-    # Logprob features (if available)
393-    mean_logprob: Optional[float] = None
394-    logprob_slope: Optional[float] = None   # None if n_observed < 2
395-    cumulative_logprob: Optional[float] = None
396-
397-    # Entropy features (if available)
398-    token_entropy_mean: Optional[float] = None
399-    token_entropy_slope: Optional[float] = None  # None if n_observed < 2
400-
401-    # Attention features (if available)
402-    # Definition: mean fraction of attention in prefix positions {0,...,prefix_token_count-1}
403-    # Averaged over layers {1,2,3,4} (0-indexed), all heads, all query positions in window
404-    attention_sink_mass: Optional[float] = None
405-    attention_sink_mass_definition: str = "mean_over_layers_1_to_4_all_heads_attending_to_prefix_positions"
406-
407-    # Format features (always)
408-    think_token_fraction: Optional[float] = None
409-    repetition_rate: float = 0.0  # fraction of repeated 4-grams (0.0 for n_observed < 4)
410-
411-    # Metadata
412-    generation_position_ids_policy: str = "default"  # "default" or "shifted"
413-```
414-
415----
416-
417-## Component 4: Output Validator
418-
419-```python
420-# Arithmetic
421:def validate_arithmetic(output_text: str, ground_truth: int) -> ValidationResult:
422-    # extract_answer: search OUTSIDE <think>...</think> blocks only
423-    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
424-    output_no_think = think_pattern.sub('', output_text)
425-    last_int = extract_last_integer(output_no_think)
426-    correct = (last_int == ground_truth) if last_int is not None else False
427-    converged = last_int is not None
428-
429-    # answer_anywhere: search full output including <think>
430-    answer_anywhere = any integer in full output == ground_truth
431-
432-    return ValidationResult(
433-        correct=correct,
434-        answer_anywhere_correct=answer_anywhere,
435-        converged=converged,
436-        score=float(correct),
437-        answer_anywhere_search_scope="full_output_including_think",
438-        final_answer_search_scope="output_excluding_think"
439-    )
440-
441-# Qualitative (legal/planning) — oracle-relative label
442-# Labels are assigned AFTER all N+1 candidates in a task group are judged
443:def assign_oracle_relative_labels(
444-    validation_results: List[ValidationResult],
445-    judge_scores: List[float],
446-    min_oracle_margin: float = 0.5  # winner must beat median by this amount
447-) -> List[ValidationResult]:
448-    median_score = statistics.median(judge_scores)
449-    oracle_idx = int(argmax(judge_scores))
450-    oracle_score = judge_scores[oracle_idx]
451-    oracle_margin = oracle_score - median_score
452-
453-    for i, result in enumerate(validation_results):
454-        if oracle_margin >= min_oracle_margin:
455-            result.correct = (i == oracle_idx)
456-            result.oracle_margin = oracle_margin
457-            result.oracle_winner_idx = oracle_idx
458-        else:
459-            result.correct = None  # degenerate task group — excluded from MI
460-            result.oracle_margin = oracle_margin
461-            result.degenerate_task_group = True
462-    return validation_results
463-```
464-
465----
466-
467-## Component 5: Atlas Logger
468-
469-```python
470-# Schema version 1.0 — frozen at preregistration
471-ATLAS_RECORD = {
472-    "schema_version": "1.0",
473-    "timestamp": str,
474-    "experiment_id": str,
475-    "code_commit": str,                    # git SHA
476-    "phase_a_version": str,               # package version
477-
478-    "model": {
Cannot set property. Property setting is supported only on core types in this language mode.
At line:1 char:1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [], RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 

codex
**Verdict: Design is converged. Blueprint is ready for engineering handoff.**

I verified the live [phase6_blueprint.md](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:1), not just the amendment notes or the Round 7 transcript.

**Round 7 Fix Verification**
1. Component 2 now computes `generation_start_index = self._get_start_index(combined_length)` locally and uses that local value for observe slicing, lean slicing, and the returned `GenerationResult`: lines [308](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:308), [339](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:339), and [362](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:362). No `self.generation_start_index` stale usage remains.

2. Lean rerun prefix-equivalence is now asserted directly in the applicator body with `torch.equal(full_ids[:len(observe_ids)], observe_ids)`: [340](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:340)-[342](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:342).

3. Zero-prefix probe now passes `output_scores=True` and uses strict boundary detection with `ValueError` on unexpected lengths: [138](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:138)-[158](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:158).

4. The Phase B primary H1 gate is now inclusive: `I(routing_score_64; correct) >= 0.1 bits`: [667](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:667). The exact-threshold edge case also says 0.1 proceeds: [671](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:671).

5. The H1/H5 typo is corrected: “H5 = exactly 90% recall” appears at [672](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:672).

**Remaining Notes**
I found no gate-blocking implementation issues. There are two harmless cleanup items an engineer can handle without design clarification: Component 2 still has stale comments saying “continue”/“KV cache” even though the code intentionally reruns lean generation, and Section 9 repeats the “Phase A alone ships” sentence twice. Neither changes behavior or blocks handoff.

Convergence declared.
tokens used
43,351
**Verdict: Design is converged. Blueprint is ready for engineering handoff.**

I verified the live [phase6_blueprint.md](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:1), not just the amendment notes or the Round 7 transcript.

**Round 7 Fix Verification**
1. Component 2 now computes `generation_start_index = self._get_start_index(combined_length)` locally and uses that local value for observe slicing, lean slicing, and the returned `GenerationResult`: lines [308](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:308), [339](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:339), and [362](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:362). No `self.generation_start_index` stale usage remains.

2. Lean rerun prefix-equivalence is now asserted directly in the applicator body with `torch.equal(full_ids[:len(observe_ids)], observe_ids)`: [340](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:340)-[342](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:342).

3. Zero-prefix probe now passes `output_scores=True` and uses strict boundary detection with `ValueError` on unexpected lengths: [138](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:138)-[158](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:158).

4. The Phase B primary H1 gate is now inclusive: `I(routing_score_64; correct) >= 0.1 bits`: [667](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:667). The exact-threshold edge case also says 0.1 proceeds: [671](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:671).

5. The H1/H5 typo is corrected: “H5 = exactly 90% recall” appears at [672](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase6_blueprint.md:672).

**Remaining Notes**
I found no gate-blocking implementation issues. There are two harmless cleanup items an engineer can handle without design clarification: Component 2 still has stale comments saying “continue”/“KV cache” even though the code intentionally reruns lean generation, and Section 9 repeats the “Phase A alone ships” sentence twice. Neither changes behavior or blocks handoff.

Convergence declared.
