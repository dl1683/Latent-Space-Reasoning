# Diffusion Moonshot Reasoning Architecture V1

This is the current build target after the span-v4 counterfactual probe gates.
It folds the local diffusion evidence into the AI Moonshots philosophy and the
adjacent meta/open-exploration research notes. The point is to stop searching for
another threshold and build the next controller around signed value, external
anchors, and consolidation.

## Source Digest

Adjacent sources reviewed for this design:

| Source | Design pressure |
| --- | --- |
| `C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\README.md` | The Moonshots constraint is geometry over scale: single-GPU experiments, first-principles structure, and cheap intelligence rather than cloud-scale brute force. |
| `C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\Eklavya\SYSTEM.md` | Distill behavior signatures under controlled probes, not decoded answers or architecture-coupled hidden states. |
| `C:\Users\devan\OneDrive\Desktop\Projects\_meta\projects\latent-space-reasoning.md` | The original LSR perturbation story is weakened by oracle-vs-mean gaps, token-cost bugs, held-out gaps, and gated-attention risk; strict mean and external falsifiers must lead. |
| `C:\Users\devan\OneDrive\Desktop\Projects\_meta\research\lsr-priors.md` | The existential test is whether mechanisms survive architectures that remove attention sinks; novelty must be framed as measured mechanism, not hype. |
| `C:\Users\devan\OneDrive\Desktop\Projects\_meta\insights\cross-domain-mechanisms.md` | The durable portfolio pattern is local-vs-integration metric split, error correction, stigmergic shared artifacts, and consolidation as a first-class phase. |
| `C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration\Grand Unified Patterns\the_seven_laws_of_intelligence.md` | Intelligence survives through error correction, attention/resource allocation, controlled noise, and dual fast/slow architecture. |
| `C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration\Attention Everywhere\attention_as_resource_allocation.md` | The controller is an allocation mechanism: spend compute only where marginal value exceeds opportunity cost. |
| `C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration\Memory Architecture\what_would_real_ai_memory_look_like.md` | Real memory needs fast binding, slow consolidation, pattern completion, salience, forgetting, and schema formation; a transcript or vector store is insufficient. |
| `C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration\Recursion and Self-Reference\self_reference_in_ai.md` | Self-improvement needs an external anchor; generator-verifier-updater loops without non-self ratified tests drift or overfit. |

## Current Evidence

The latest span-v4 evidence rejects four local controller families:

| Artifact | What failed |
| --- | --- |
| `DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md` | A zero-error local distinct-retention rule has 3 errors on a fresh planning slice. |
| `DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md` | The transfer-screened gap/span conjunction has 3 strict errors on `plan_017`-`plan_024`. |
| `DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md` | Richer nearest-prototype signatures preserve all positives but select 11 no-lift negatives under leave-slice-out. |
| `DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md` | Cost-calibrating the scalar signature score either keeps 11 false positives or misses most positives. |
| `DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md` | A two-stage threshold-fragment veto reduces false positives from 11 to 8 but introduces 5 false negatives and lowers practical-penalty utility. |

The conclusion is not "need a better threshold." It is:

1. The measurement surface is now good enough to expose value geometry.
2. The controller is too poor because it compresses that surface to binary local
   labels or a scalar threshold.
3. The next controller must predict signed realized value with uncertainty and
   cost, then be frozen before a fresh GPU slice.

## Architecture Target

Build a Signed Value Tomography Controller.

### 1. Measurement Surface

Use counterfactual span probes as Eklavya-style behavior signatures. Each task
gets a structured response surface rather than a single score:

- raw trajectory score and selected trajectory state
- measured probe value prediction
- gap visibility, realization defect visibility, span evidence, retention risk
- slot-overlap and text-fidelity features
- source quality and prompt-gap geometry
- validator status and defect tags

The controller never trusts one feature as a gate. It consumes the whole
signature.

### 2. Signed Value Head

Train or fit a small offline head that predicts:

`signed_value = realized_lift_vs_selected_trajectory - lambda * marginal_relative_cost`

This replaces binary labels like `label` or `would_probe_score`. A row can be
positive, weakly positive, cost-negative, or actively harmful. The controller
selects only when:

`E[signed_value] - uncertainty_penalty > 0`

Minimum offline features:

- `probe_signature_score`
- measured gap/span/retention/realization features
- validity booleans
- source quality
- prompt gap count
- no-lift veto diagnostics as features, not hard rules
- slice identity only for diagnostics, never as a live feature

### 3. External Anchors

The system must not self-ratify. Each training pass uses:

- leave-one-slice-out transfer as the first anchor
- a frozen fresh GPU slice as the second anchor
- a negative-control slice where the probe should not help
- cost utility at `lambda = 0.020000` as the default practical objective

No local zero-error rule can be promoted without both anchors.

### 4. Fast/Slow Loop

The controller should follow a dual architecture:

- Fast binding: append every probe run, defect, selected row, and realized lift
  into a replayable evidence packet.
- Slow consolidation: periodically convert those packets into updated signed
  value features, negative controls, and retired-rule summaries.

This mirrors the memory architecture notes: the useful artifact is not a larger
context transcript. It is a consolidated state that records what changed,
what was superseded, and which action follows.

### 5. Error Correction

Every controller candidate must produce three outputs:

- Decision: selected rows, skipped rows, expected utility, uncertainty.
- Syndrome: which failure mode it claims to detect, such as no-lift, invalid
  probe text, retention collapse, or source mismatch.
- Repair target: what new measurement would most reduce uncertainty.

This keeps the system aligned with the portfolio pattern: detect, diagnose,
repair, verify.

## Implementation Plan

### M1: Offline Signed-Value Learner

Add `experiments/fit_diffusion_span_probe_signed_value.py`.

Status: implemented as `DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md`.
The first signed-value head is a boundary, not a promotion: it improves strict
signed utility from `0.498571` to `0.582500`, preserves zero false negatives,
and reduces false positives from 11 to 9, but it does not clear the `0.625500`
promotion bar.

Inputs:

- `eval_results/diffusion_language/counterfactual_span_probe_signature_model_v4.json`
- `eval_results/diffusion_language/counterfactual_span_probe_no_lift_veto_v4.json`

Outputs:

- `eval_results/diffusion_language/counterfactual_span_probe_signed_value_v4.json`
- `DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md`

Required report sections:

- leave-one-slice-out utility at `lambda = 0.020000`
- regret versus cost-aware oracle
- false positives, false negatives, and missed lift
- per-slice transfer table
- comparison against base signature gate, utility frontier, and no-lift veto

Promotion bar:

- utility must exceed the base signature gate's `0.625500`
- false negatives must be zero, unless the missed positive has negative
  cost-adjusted signed value
- false positives must be lower than 11
- all metrics must be computed on leave-one-slice-out rows

### M2: Negative Controls

Add a control mode that deliberately withholds one feature family:

Status: implemented as
`DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_CONTROLS_V4.md`. M2 passes
as an audit: all matched-k and best-withheld feature-family controls degrade
relative to `signed_value_knn_k8_all`. This supports the distributed-signature
interpretation, but it does not override the M1 promotion failure.

- no text-fidelity features
- no source-quality features
- no retention features
- no gap/span features

The signed-value claim is only real if the winning model degrades when the
load-bearing feature family is removed. Otherwise, the head is likely learning
slice artifacts.

### M3: Frozen Fresh Slice

Only after M1 and M2 pass, run a fresh GPU slice with the signed-value head
frozen. The run must record:

- CUDA content hash
- task IDs before generation
- prompt/probe policy
- raw output path
- score path
- report path
- exact signed-value commit hash

### M4: Controller Integration

Only after the fresh slice passes, wire the signed-value head as an audit-only
spend trigger. It should shadow the existing policy first. Live spending remains
blocked until shadow decisions dominate under cost utility.

## Rejected Paths

Do not spend another turn on these without new evidence:

- hand threshold over retention, gap, span, or source quality
- scalar threshold over `probe_signature_score`
- binary no-lift veto made from one- or two-condition fragments
- oracle-only LSR framing
- local zero-error fit without leave-slice-out and fresh-slice anchors
- self-generated controller data without external score anchors

## First Engineering Step

Implement M1 as an offline evaluator. Keep it small, deterministic, and
replayable. The first version can be a ridge-free nearest-neighbor or small
piecewise learner as long as it predicts signed value directly and reports
utility, not just classification errors.

The reason to do M1 before another GPU run is strategic: the GPU should test a
frozen controller, not search for one.
