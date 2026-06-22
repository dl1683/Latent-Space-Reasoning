# Tesla Mode Phase 2: Mental Assembly — Next-Gen System Design

## Reframe (Per Codex Phase 1 Directive)

The unit of control is NOT a latent vector. It is a **trajectory class**: the attractor the model falls into when generation begins. The mechanism is **small initial-condition energy control over autoregressive trajectories**. Everything in Phase 2 is designed around this frame.

Codex also flagged a critical architecture gap: the shipped `orchestrator.py` uses `encoder.decode()` which does NOT implement the validated mechanism (raw `inputs_embeds` soft-prefix injection). The new design must align codebase with validated mechanism.

---

## The Next-Gen System: Trajectory-Control Architecture (TCA)

### Conceptual Foundation

A language model under greedy decoding is a discrete dynamical system. It has attractors — basins of generation behavior the model falls into given initial conditions. The current system's key discovery:

**Small perturbations to initial embedding conditions (2 tokens, RMS-matched) shift basin selection.**

The non-monotonic dose response (peak at 2 tokens), direction-independence, and attention-sink rescue all follow from this: we are doing *basin hopping near criticality*, not *injecting semantic content*.

The next-gen system makes this explicit and precise. Instead of:
> "Sample random noise, generate N outputs, pick best via oracle"

We do:
> "Measure the trajectory class produced by each perturbation in the first 16-64 tokens, route by predicted correctness, generate only the high-probability candidates"

---

## Component Inventory

### Component A: Perturbation Energy Controller
**Responsibility**: Produces a calibrated intervention (not just "random noise") parametrized by energy and surface.

**Inputs**: 
- Model ID + quantization level (determines phase diagram position)
- Task class (arithmetic / planning / legal / unknown)
- Budget (N candidates allowed)
- Phase diagram lookup (if pre-built for this model/task)

**Outputs**: 
- A set of N intervention objects, each specifying:
  - `surface`: input_prefix | early_layer_residual | attention_mask | mid_layer_nudge
  - `token_count`: integer (1-8)
  - `rms_scale`: float (RMS energy level, default 0.022)
  - `position`: where in the sequence the perturbation is applied
  - `seed`: for reproducibility
- Total perturbation energy `E = token_count × rms_scale²` per intervention

**Interface contract**:
```python
PerturbationSpec = {
    "surface": str,           # one of: "input_prefix", "layer_residual", "attn_mask", "mid_nudge"
    "token_count": int,       # number of perturbation tokens
    "rms_scale": float,       # energy scale
    "layer": int | None,      # which layer (for residual/mid_nudge surfaces)
    "seed": int               # random seed for reproducibility
}

def generate_interventions(model_id, quantization, task_class, n_candidates, budget) -> List[PerturbationSpec]
```

**Failure mode**: If the phase diagram is not yet built for this model/task, falls back to `n_candidates` random input-prefix specs at RMS=0.022. Never fails silently.

**State**: Reads from `perturbation_atlas.json` (pre-built per model/quantization). Writes to atlas as new data arrives.

---

### Component B: Early Trajectory Observer
**Responsibility**: Observe the first 16-64 generated tokens and extract trajectory biomarkers. This is the KEY new component — it enables closed-loop routing before expensive full generation.

**Inputs**:
- Partial generation (first 16-64 tokens of each candidate)
- Model hidden states at each step
- Attention patterns at each step

**Outputs** (per candidate):
- `attention_sink_mass`: fraction of attention concentrated in first 2 tokens (layers 1-4 averaged)
- `answer_position_forecast`: P(answer appears at end) vs P(answer appears mid-sequence), from early token distribution
- `entropy_trajectory`: KL divergence of token distribution over time (increasing = exploration, decreasing = convergence)
- `think_depth`: proportion of `<think>...</think>` tokens vs output tokens
- `length_forecast`: estimated total output length given current trajectory
- `trajectory_class`: discrete label (converging | exploring | collapsing | diverging)

**Interface contract**:
```python
TrajectoryBiomarkers = {
    "attention_sink_mass": float,        # [0,1] — high = attention sink trap
    "answer_position_forecast": float,   # [0,1] — high = answer likely at end
    "entropy_trajectory": List[float],   # per-step token entropy
    "think_depth": float,                # [0,1] — fraction of thinking tokens
    "length_forecast": int,              # estimated total output tokens
    "trajectory_class": str              # "converging"|"exploring"|"collapsing"|"diverging"
}

def observe_early_trajectory(model, input_with_intervention, n_observe_tokens=32) -> TrajectoryBiomarkers
```

**Failure mode**: If model doesn't support attention output, falls back to token-distribution features only. Trajectory class defaults to "unknown" — all candidates are promoted to full generation.

**State**: Stateless per call. But biomarker distributions are logged to atlas to build phase diagram.

---

### Component C: Trajectory Router (Controller)
**Responsibility**: Given N sets of biomarkers, choose which candidates to fully generate. This replaces the latent scorer + evolution loop.

**Inputs**:
- N × TrajectoryBiomarkers (one per intervention candidate)
- Budget (max full generations allowed)
- Routing policy (learned | heuristic)

**Outputs**:
- List of candidate indices to promote to full generation
- Confidence scores per candidate
- Abstention signal: if no candidates look promising, signal escalation

**Routing Policy (v1 — Heuristic)**:
- Reject any candidate with `attention_sink_mass > 0.6` and `trajectory_class == "collapsing"`
- Promote top-K by `answer_position_forecast × (1 - attention_sink_mass)`
- If fewer than 1 candidate passes, promote the least-bad and flag uncertainty

**Routing Policy (v2 — Learned)**:
- Train a lightweight classifier on (biomarkers → P(final_correct)) from accumulated experiment data
- Input: 6 biomarker features per candidate
- Output: calibrated probability of final correctness
- Architecture: small MLP (6 → 32 → 1), trained on logged (biomarkers, outcome) pairs from the perturbation atlas

**Interface contract**:
```python
def route_candidates(
    biomarkers: List[TrajectoryBiomarkers],
    budget: int,
    policy: "heuristic" | "learned"
) -> RoutingDecision

RoutingDecision = {
    "promote": List[int],        # indices of candidates to fully generate
    "confidences": List[float],  # calibrated P(correct) per promoted candidate
    "abstain": bool              # True if system should escalate to verifier/larger model
}
```

**Failure mode**: Policy degrades gracefully — heuristic always produces at least 1 candidate. Learned policy falls back to heuristic if not trained.

**State**: Reads learned policy from `router_checkpoint.pt`. Stateless otherwise.

---

### Component D: Multi-Surface Applicator
**Responsibility**: Apply a PerturbationSpec to an actual model, producing a modified input for generation. This is the component that MUST replace `orchestrator.py`'s broken `encoder.decode()` path.

**Surfaces implemented**:
1. `input_prefix`: Prepend N random embedding-scale tokens to `inputs_embeds`. RMS-matched. This is the validated mechanism.
2. `early_layer_residual`: Add noise vector to residual stream at layer L (new — to be validated)
3. `attn_mask`: Modify attention mask to suppress attention sink tokens (new — to be validated)
4. `mid_layer_nudge`: Add learned steering vector to residual stream at middle layers (new — to be validated)

**Interface contract**:
```python
def apply_intervention(
    model,
    tokenized_input,
    spec: PerturbationSpec
) -> ModifiedInput

ModifiedInput = {
    "inputs_embeds": Tensor,     # modified embeddings (for input_prefix surface)
    "hidden_state_hooks": List,  # registered hooks (for residual surfaces)
    "attention_mask": Tensor,    # modified mask (for attn_mask surface)
    "input_ids": None            # always None when inputs_embeds is used
}
```

**Critical implementation note**: When surface == `input_prefix`, `input_ids` must be set to `None` and `inputs_embeds` must be used directly. The current `orchestrator.py` does not do this correctly (lines 583). The new design makes this the default and only path.

**Failure mode**: Unknown surface raises `ValueError`. RMS scale outside [0.001, 0.1] raises `ValueError`. Never silently applies a no-op.

**State**: Stateless. Hooks are registered and removed per generation call.

---

### Component E: Output-Space Validator
**Responsibility**: Evaluate the ACTUAL text outputs of promoted candidates. This replaces the scalar latent scorer as the primary signal.

**Accepts**:
- Task class + ground truth (if available)
- N decoded text outputs
- LLM judge (external, strong model: Claude Sonnet / GPT-4o)
- Domain-specific verifiers (for arithmetic: exact-match extractor; for legal: rubric-based; for planning: completeness check)

**Produces**:
- Per-candidate scores (multi-dimensional: correctness, specificity, completeness, risk)
- Best candidate selection
- Abstention signal if all candidates below threshold

**Architecture**:
- **Arithmetic**: deterministic verifier (extract_answer → compare to ground truth)
- **Planning/Legal**: LLM-as-judge with structured rubric (correctness, completeness, specificity, hallucination)
- **Unknown**: LLM-as-judge with generic quality rubric

**Interface contract**:
```python
def validate_outputs(
    outputs: List[str],
    task_class: str,
    ground_truth: str | None,
    judge: LLMJudge | None
) -> ValidationResult

ValidationResult = {
    "scores": List[Dict[str, float]],   # per-candidate, per-dimension
    "best_candidate_idx": int,
    "best_score": Dict[str, float],
    "abstain": bool                      # True if all candidates below threshold
}
```

**Failure mode**: If judge is unavailable, falls back to heuristic length + format check. Never returns random selection.

---

### Component F: Perturbation Atlas Builder
**Responsibility**: Accumulate logged (model, quantization, task_class, spec, biomarkers, outcome) tuples and build the phase diagram — the calibrated map of which interventions work for which models/tasks.

**Inputs**: Experiment runs, streaming from the Observer and Validator

**Outputs**:
- `perturbation_atlas.json`: per (model_id, quantization, task_class) → recommended specs
- `phase_diagram.json`: per (model_id, quantization) → (E, surface, position) → P(trajectory_class), P(final_correct)
- `router_training_data.jsonl`: (biomarkers, ground_truth_correct) pairs for training Component C v2

**Interface contract**:
```python
def log_experiment(
    model_id: str,
    quantization: str,
    task_class: str,
    spec: PerturbationSpec,
    biomarkers: TrajectoryBiomarkers,
    final_output: str,
    correct: bool | None
) -> None

def get_recommended_specs(model_id, quantization, task_class, n_candidates) -> List[PerturbationSpec]
```

**State**: Persists to disk. Append-only writes to `atlas_log.jsonl`.

---

## Data Flow: Primary Execution Path

```
Query + Task Class + Model ID
    ↓
[A: Perturbation Energy Controller]
  - Look up atlas for model/quantization/task_class
  - Generate N PerturbationSpecs (calibrated energy, surface, seed)
    ↓
[D: Multi-Surface Applicator] × N (parallel)
  - Apply each spec to tokenized input
  - Produce N modified inputs
    ↓
[Partial Generation] × N (first 32 tokens per candidate)
    ↓
[B: Early Trajectory Observer] × N (parallel)
  - Extract biomarkers from partial generation
  - Classify trajectory class per candidate
    ↓
[C: Trajectory Router]
  - Filter/rank by biomarkers
  - Promote top-K candidates (K ≤ budget)
  - Signal abstention if no good candidates
    ↓
[Full Generation] × K (promoted candidates only)
    ↓
[E: Output-Space Validator]
  - Multi-dimensional scoring of full outputs
  - Select best candidate
  - Log to atlas
    ↓
Best Output + Confidence + Metadata
```

---

## The Decomposition Codex Mandated

Per Codex's Priority Directive, the design must make measurable:

**`P(final_correct) = P(answer_anywhere_correct) × P(final_correct | answer_anywhere_correct)`**

As a function of:
- `E = token_count × rms_scale²` (perturbation energy)
- `surface` (which surface the intervention is applied to)
- `position` (where in the sequence)
- `quantization` (8-bit vs 4-bit)
- Early trajectory order parameters: attention_sink_mass, entropy_trajectory, answer_position_forecast

**Why this matters for the design**:
- If perturbation raises `P(answer_anywhere_correct)`: it's improving computation. The model solves the problem but couldn't before.
- If perturbation raises `P(final_correct | answer_anywhere_correct)`: it's a convergence/format controller. The model could solve it but put the answer in the wrong place.
- If perturbation mainly changes trajectory diversity without predictable correctness: it's an oracle-routing engine, not a reasoning improver. Cannot claim "improvement" — can only claim "better selection."

Component B (Trajectory Observer) feeds this decomposition by letting us measure answer_anywhere_correct from partial generation, before we even know if the final answer is correct.

---

## State: Where It Lives, Who Owns It

| State | Owner | Location | Mutability |
|---|---|---|---|
| Perturbation atlas | Component F | `data/perturbation_atlas.json` | Append on each run |
| Phase diagram | Component F | `data/phase_diagram.json` | Rebuilt from atlas |
| Router checkpoint | Component C v2 | `checkpoints/router.pt` | Written by training |
| Atlas log | Component F | `data/atlas_log.jsonl` | Append-only |
| Experiment ledger | Repo standard | `experiments/ledger.jsonl` | Append-only |
| Session config | Caller | `configs/*.yaml` | Read-only at runtime |

---

## Failure Modes and Degradation

| Failure | Consequence | Fallback |
|---|---|---|
| Atlas empty for model/task | No calibrated specs | Random N specs at RMS=0.022, 2-token |
| Attention output unavailable | Biomarkers incomplete | Token distribution features only; promote all |
| Learned router not trained | v2 policy unavailable | Heuristic policy |
| All candidates abstained | No good output | Flag escalation; return best with low confidence |
| Judge unavailable | No output validation | Length + format heuristic |
| Unknown surface in spec | Invalid intervention | Raise ValueError, do not silently no-op |

**Degraded mode = current system behavior.** Full mode = the new system. Graceful fallback ensures the old experiments still run.

---

## What NOT to Build (Explicit Exclusions)

- **No fixed W projection** — delete as mainline. It adds no value over random noise.
- **No latent evolution loop** — not until a real scorer exists and landscape is confirmed smooth.
- **No scalar latent scorer as primary** — CLAUDE.md says scores are irrelevant. Use output-space validator.
- **No orchestrator.decode() path** — this does not implement the validated mechanism. Replace entirely.
- **No "general" model-agnostic defaults** — every component is parametrized by model_id + quantization.
- **No oracle-as-deployment** — best-of-N oracle is science, not product. Router is the deployment path.

---

## Critical Open Questions (For Phase 3 Stress Test)

1. Is 32 tokens enough to classify trajectory? Or does fate get decided later?
2. Can partial generation be done efficiently without full KV cache? Or do we pay full cost?
3. Is attention_sink_mass actually predictive of final correctness? (Hypothesis, not yet validated)
4. Does multi-surface intervention compose? Or do surfaces interfere?
5. Can the heuristic router work well enough that we never need to train v2?
6. Does the abstention signal reduce quality in practice (too many abstentions = useless system)?
