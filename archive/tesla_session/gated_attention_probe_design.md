# Gated Attention Transfer Probe — Experimental Design

## Status: DESIGN PHASE (Pre-Codex Approval)

## Purpose
Test whether our perturbation mechanism survives on gated-attention architectures that eliminate attention sinks. This is the #1 priority identified by both Claude and Codex.

## Feasibility Assessment

### Hardware Constraint
RTX 5090 Laptop, ~24GB VRAM.

### Model Options (ranked by feasibility)

1. **Qwen3.5-4B** (RECOMMENDED)
   - 4B parameters, fits easily at 4-bit or 8-bit quantization
   - Hybrid architecture: mostly Gated DeltaNet (linear attention) + Gated Attention layers
   - Gated attention eliminates attention sinks by design
   - **Direct comparison**: Qwen3-4B (no gated attention) vs Qwen3.5-4B (gated attention), same parameter count, same model family
   - Available on HuggingFace: Qwen/Qwen3.5-4B

2. **1B Gated Attention Models** (from NeurIPS paper)
   - From QwQZh/gated_attention on HuggingFace
   - Three variants: 1B_baseline (no gate), 1B_gate_elementwise, 1B_gate_headwise
   - Tiny models, fast experiments
   - BUT: 1B may be too small for meaningful arithmetic reasoning
   - ALSO: These are research checkpoints, not instruction-tuned models

3. **Qwen3-Next-80B-A3B at Q4** (NOT FEASIBLE)
   - Needs ~40GB VRAM at Q4 — doesn't fit on 24GB
   - Would need aggressive quantization + CPU offloading = very slow
   - The 3B active parameters (MoE) might make it feasible with offloading
   - STRETCH GOAL only

### Recommendation
Run the probe on **Qwen3.5-4B at 4-bit** — same quantization as our Qwen3-4B baseline. This gives us:
- Same model family (Qwen)
- Same parameter count (4B)
- Same quantization (4-bit)
- The ONLY difference: Qwen3.5 has gated attention, Qwen3 does not

This is the cleanest comparison possible.

## Experimental Design

### Phase 1: Arithmetic Probe (25 tasks)

**Same 25 arithmetic tasks** used in all prior experiments.

#### Conditions (4 + optional):
1. **Baseline**: Qwen3.5-4B, no prefix, greedy decoding
2. **Zero-prefix**: Qwen3.5-4B, zero-valued 2-token prefix via inputs_embeds
3. **Random-prefix N=10**: Qwen3.5-4B, random 2-token prefix, RMS matched, seeds 0-9
4. **Position-shift control**: Qwen3.5-4B, no prefix but position_ids start at 2 (tests if positional shift alone accounts for effect)
5. *(Optional)* **Dose-response**: 1/3/8 token counts, RMS matched

#### Measurements:
- Last-integer accuracy (primary)
- Answer-anywhere accuracy
- Truncation rate
- Output length distribution
- Oracle coverage at N=10
- Attention sink mass at early layers (if output_attentions works)

### Phase 2: Planning Rescue Probe (1 task)

**The cache-debugging task** where baseline produced 14 words and all 5 perturbation seeds produced 650+ words.

#### Conditions:
1. Baseline: Qwen3.5-4B, max_new_tokens=2048
2. Random-prefix N=5: same conditions as original planning experiment

#### Expected outcomes by hypothesis:

**If sink-dependent** (worst case):
- Qwen3.5-4B baseline: no 14-word collapse (gated attention prevents sink)
- Qwen3.5-4B perturbed: ≈ baseline (nothing to rescue)
- Arithmetic: oracle coverage drops toward individual accuracy
- Conclusion: our mechanism is architecture-dependent, limited to pre-gated models

**If trajectory-diversification** (best case):
- Qwen3.5-4B baseline: may or may not collapse differently
- Qwen3.5-4B perturbed: oracle coverage remains high (diverse trajectories)
- Arithmetic: oracle still >> mean accuracy
- Conclusion: mechanism is fundamental, gated attention is orthogonal

**If position-confound**:
- Condition 4 (position-shift) reproduces much of the effect
- Conclusion: our "perturbation" is really a positional encoding shift, not embedding noise

### Phase 3: Cross-Architecture Comparison Table

If Phase 1-2 complete, compile:

| Model | Gated? | Base acc | +Prefix mean | +Prefix oracle | Planning rescue? |
|-------|--------|----------|-------------|----------------|-----------------|
| Qwen3-4B Q4 | No | 32% | 51.6% | 100% | Yes (14w→650w) |
| Qwen3.5-4B Q4 | Yes | ? | ? | ? | ? |
| Qwen3-8B Q8 | No | 16% | 28.8% | 80% | N/A |

## Interpretation Criteria (Pre-Registered)

### The effect is SINK-DEPENDENT if:
- Qwen3.5-4B oracle coverage at N=10 < 60% (vs 100% on Qwen3-4B)
- AND Qwen3.5-4B mean accuracy improvement < 5pp (vs 19.6pp on Qwen3-4B)
- AND planning task shows no rescue effect

### The effect is TRAJECTORY-DIVERSIFICATION if:
- Qwen3.5-4B oracle coverage at N=10 > 80%
- OR Qwen3.5-4B mean accuracy improvement > 10pp
- These thresholds are set BEFORE seeing data

### Ambiguous zone:
- Oracle 60-80% or mean improvement 5-10pp
- Requires additional conditions (dose-response, more tasks) to interpret

## Dependencies
- Qwen3.5-4B must support inputs_embeds generation (verify first)
- Must run the inputs_embeds probe (Component 0 from Blueprint) on Qwen3.5-4B before data collection
- Chat template may differ from Qwen3 — verify tokenizer compatibility

## Timeline
Estimated: 2-3 days for Phases 1-2 once implementation begins.
This should run BEFORE paper submission.
