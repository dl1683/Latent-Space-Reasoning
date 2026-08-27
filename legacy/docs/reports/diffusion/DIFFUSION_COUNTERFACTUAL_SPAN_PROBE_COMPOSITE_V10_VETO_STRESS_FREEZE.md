# Diffusion Span Probe Composite V10 Veto-Stress Freeze

## Decision

The first v10 replay was a useful fresh GPU boundary, but it did not stress the
trajectory-relative veto because the repair source matched the selected
trajectory on every planning row. This addendum freezes a narrow follow-up:
reuse the committed v10 task slice and label artifact, but rerun the measured
span-probe pass with `--repair-source-policy fixed` so rows where
`planning_state` selected `random_32` can produce negative
`source_task_delta_vs_trajectory`.

This is not a new promotion attempt. It is an information-channel audit for the
veto term that the previous replay failed to exercise.

## Frozen Measurement Command

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed_transfer_v10 --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-source-policy fixed --repair-spend-trigger counterfactual_micro_probe_v1 --counterfactual-probe-mode all --counterfactual-probe-policy span_tomography_probe_v4 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\span_probe_composite_v10_fixed_source_measurement_raw.jsonl --scores-output eval_results\diffusion_language\span_probe_composite_v10_fixed_source_measurement_scores.json --report-output eval_results\diffusion_language\span_probe_composite_v10_fixed_source_measurement_report.md
```

## Replay Inputs

- Measurement: `eval_results/diffusion_language/span_probe_composite_v10_fixed_source_measurement_scores.json`
- Measurement raw: `eval_results/diffusion_language/span_probe_composite_v10_fixed_source_measurement_raw.jsonl`
- Labels: `eval_results/diffusion_language/span_probe_composite_v10_label_scores.json`
- Replay script: `experiments/replay_diffusion_span_probe_composite_v10.py`

## Pass/Fail Reading

- If the fixed-source replay creates negative source-vs-trajectory rows and
  blocks no-lift rows without missing positive repairs, the trajectory-relative
  channel earns a stronger implementation target.
- If it blocks positive rows, selects the full slice, or remains below the
  frozen utility bar, the composite remains diagnostic-only.
- If no negative-delta rows appear, the source-policy change failed to exercise
  the intended channel and should not be counted as veto evidence.
