# Diffusion Counterfactual Micro-Probe Smoke V1

This is the first GPU-backed smoke test of the measured
`counterfactual_micro_probe_v1` runner hook.

## Command

```powershell
python experiments\run_diffusion_three_arm_benchmark.py `
  --families planning `
  --task-ids plan_070 `
  --candidates llada-moe-7b-a1b-instruct-hf `
  --limit-schedules 1 `
  --limit-evolved-schedules 0 `
  --limit-repair-candidates 1 `
  --repair-source-policy trajectory `
  --repair-spend-trigger counterfactual_micro_probe_v1 `
  --repair-selector candidate_aware_promotion_v1 `
  --repair-source-quality-threshold 0.99 `
  --repair-source-min-chars 40 `
  --repair-source-prompt-gap-min 0 `
  --repair-source-prompt-gap-max 8 `
  --repair-source-prompt-coverage-min 0.0 `
  --repair-source-prompt-coverage-max 1.0 `
  --device cuda `
  --dtype bfloat16 `
  --raw-output eval_results\diffusion_language\counterfactual_micro_probe_plan070_smoke_raw.jsonl `
  --scores-output eval_results\diffusion_language\counterfactual_micro_probe_plan070_smoke_scores.json `
  --report-output eval_results\diffusion_language\counterfactual_micro_probe_plan070_smoke_report.md
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Raw generations | `eval_results/diffusion_language/counterfactual_micro_probe_plan070_smoke_raw.jsonl` |
| Scores | `eval_results/diffusion_language/counterfactual_micro_probe_plan070_smoke_scores.json` |
| Report | `eval_results/diffusion_language/counterfactual_micro_probe_plan070_smoke_report.md` |

## Result

| Check | Value |
| --- | --- |
| Normal arm generations | `1` |
| Counterfactual probe generations | `1` |
| Probe generation stage | `counterfactual_probe` |
| Probe control | `counterfactual_micro_probe_v1` |
| Probe observation | `measured_generation` |
| Probe tokens | `32` |
| Gate `would_probe` | `true` |
| Gate `should_run` | `false` |
| Repair score credit | `none` |
| Measured probe value prediction | `0.010953` |

## Reading

This smoke test proves the hook can spend GPU on a bounded micro-probe and
thread the measured observation back into `repair_spend_gate_rows` without
authorizing full repair. It does not prove a deployable spend gate. The next
evidence step is to run the same measured hook across the named v5-v9
counterexample rows, then refit the value-of-information policy using measured
probe deltas instead of deterministic scaffold deltas.
