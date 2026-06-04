# Diffusion Counterfactual Micro-Probe Runner Hook V1

This file documents the first runner-facing measured counterfactual probe hook.
It is not a promoted spend gate and it does not run full repair.

## Runner Surface

Use the existing benchmark runner with:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py `
  --repair-spend-trigger counterfactual_micro_probe_v1 `
  --repair-selector candidate_aware_promotion_v1
```

The trigger records probe diagnostics in `repair_spend_gate_rows` for each
selected repair source. When frozen triage sets `would_probe=true`, it also
generates a bounded `generation_stage="counterfactual_probe"` raw record under
a strict `32` token / `16` step budget. It always returns `should_run=false`, so
the `repair_selected` arm keeps the evolved record and cannot get score credit
from probe logic.

## Recorded Fields

Each gate row keeps the normal source diagnostics and adds:

| Field | Meaning |
| --- | --- |
| `counterfactual_probe_gate` | Always `diagnostic_only`. |
| `counterfactual_probe_policy` | `deterministic_missing_constraint_probe_v1`. |
| `counterfactual_probe_cost_relative` | Current scaffold probe cost, `0.125`. |
| `counterfactual_probe_observation` | `measured_generation` when a bounded probe was generated; otherwise `deterministic_scaffold`. |
| `counterfactual_probe_text` | Bounded missing-constraint sketch with explicit `full_repair_authorized=false`. |
| `probe_feature_delta` | Gap visibility, realization-defect visibility, span evidence, and retention-risk proxy deltas. |
| `probe_value_prediction` | Deterministic proxy value prediction using the same public scaffold terms. |
| `would_probe` | Frozen triage decision for buying the cheap observation, not permission to run repair. |
| `measured_probe_feature_delta` | Present only after a measured probe record replaces the scaffold deltas. |
| `measured_probe_value_prediction` | Present only after a measured probe record replaces the scaffold value prediction. |

## Gate Semantics

The hook is intentionally weaker than the architecture gate:

- It may say `would_probe=true`.
- It must still say `should_run=false`.
- It may generate `counterfactual_probe` raw records.
- It must not generate repair candidates.
- It must not change the promoted policy from
  `denoise_phase_repairability` plus `candidate_aware_promotion_v1`.

The first offline fit found that the deterministic gap-visibility scaffold can
keep all seven profitable target rows while removing four of five no-lift rows,
but that result is still diagnostic. The measured runner hook can now replace
the scaffold deltas in fresh runs, but those rows still need to be accumulated
and refit before any spend-gated GPU run is allowed.

The first one-task CUDA smoke is recorded in
`DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_SMOKE_V1.md`.

## Next Measurement

The next GPU-backed increment should run this hook on the named counterexample
rows and refit the value-of-information policy:

1. Run `counterfactual_micro_probe_v1` on the named v5-v9 counterexamples.
2. Confirm raw `counterfactual_probe` records exist for every `would_probe`
   row.
3. Refit `DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md` against measured
   deltas.
4. Promote only if the architecture gate in
   `DIFFUSION_COUNTERFACTUAL_CONTROLLER_ARCHITECTURE_V1.md` clears.
