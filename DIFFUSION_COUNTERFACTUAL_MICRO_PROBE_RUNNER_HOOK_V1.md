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

The default `--counterfactual-probe-mode triage` generates bounded measured
probe records only for rows where frozen triage sets `would_probe=true`. Use
`--counterfactual-probe-mode all` for offline shadow fitting when negative rows
also need measured probe deltas. Both modes keep `should_run=false`.
Use `--counterfactual-probe-policy strict_tomography_probe_v1` to replace the
legacy prose probe with fixed diagnostic slots:
`MISSING_CONSTRAINT=`, `EVIDENCE_NEEDED=`, `RETENTION_RISK=`, and exact
`FULL_REPAIR_AUTHORIZED=false`.

The trigger records probe diagnostics in `repair_spend_gate_rows` for each
selected repair source. When frozen triage sets `would_probe=true`, it also
generates a bounded `generation_stage="counterfactual_probe"` raw record under
a strict `32` token / `16` step budget for the legacy policy, or `48` tokens /
`24` steps for `strict_tomography_probe_v1`. It always returns
`should_run=false`, so the `repair_selected` arm keeps the evolved record and
cannot get score credit from probe logic.

## Recorded Fields

Each gate row keeps the normal source diagnostics and adds:

| Field | Meaning |
| --- | --- |
| `counterfactual_probe_gate` | Always `diagnostic_only`. |
| `counterfactual_probe_policy` | `deterministic_missing_constraint_probe_v1`. |
| `counterfactual_probe_cost_relative` | Current scaffold probe cost, `0.125`. |
| `counterfactual_probe_observation` | `measured_generation` when a bounded probe was generated; otherwise `deterministic_scaffold`. |
| `counterfactual_probe_text` | Bounded missing-constraint sketch with explicit `full_repair_authorized=false`. |
| `counterfactual_probe_text_valid_for_stage1` | True only when measured text has exact authorization, all diagnostic slots, no placeholder/generic slot, and no known slot/sentinel typo. |
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
The first 12-row named-counterexample CUDA run is recorded in
`DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_COUNTEREXAMPLES_V1.md`.
The all-shadow 12-row measured fit is recorded in
`DIFFUSION_COUNTERFACTUAL_MEASURED_PROBE_VALUE_POLICY_V1.md`; it keeps the full
repair gate closed because measured-only Stage 1 rules still make two errors.
The probe-text fidelity audit is recorded in
`DIFFUSION_COUNTERFACTUAL_PROBE_TEXT_FIDELITY_V1.md`; it shows the current probe
text is not stable enough for promotion because malformed authorization strings
and weak diagnostic slots remain common.
The strict tomography follow-up is recorded in
`DIFFUSION_COUNTERFACTUAL_TOMOGRAPHY_PROBE_TEXT_FIDELITY_V1.md`: it fixes the
authorization sentinel but still leaves four invalid diagnostic rows and one
post-probe false positive, so it remains diagnostic-only.
The validity-required Stage 1 fit is recorded in
`DIFFUSION_COUNTERFACTUAL_VALIDATED_PROBE_STAGE1_GATE_V1.md`: invalid diagnostic
rows are treated as missing evidence. That exposes three invalid profitable
rows and a best validated rule with five errors.

## Next Measurement

The next increment should improve the measured target rows until the Stage 1
value-of-information policy can clear the controller gate:

1. Add post-probe features that are not just Stage 0 prompt-gap or `would_probe`
   rediscovery.
2. Treat `FULL_REPAIR_AUTHORIZED=false` as a hard validity sentinel and discard
   probe rows that cannot reproduce it exactly.
3. Check whether measured probe values can decide full repair after the probe,
   not just whether to buy a probe.
4. Promote only if the architecture gate in
   `DIFFUSION_COUNTERFACTUAL_CONTROLLER_ARCHITECTURE_V1.md` clears.
