# Diffusion Generated-Repair Value Hook V19 Control Replay

## Decision

The v19 control replay keeps `generated_repair_value_v1` local rather than promoted. A permissive planning-quality repair selector with zero promotion margin matches the hook's aggregate repair-covered utility on the committed v19 raw artifact, but it does so with worse precision: it selects the no-lift/tiny-lift repair rows that the hook keeps out.

The useful reading is precision, not global dominance. The hook is a cleaner allocation policy; it is not yet a broadly promoted controller.

## Replay

- Hook result: [DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_RESULT.md](DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_RESULT.md)
- Hook raw input: `eval_results\diffusion_language\generated_repair_value_hook_v19_label_raw.jsonl`
- Control scores: `eval_results\diffusion_language\generated_repair_value_hook_v19_broad_denoise_control_scores.json`
- Control report: `eval_results\diffusion_language\generated_repair_value_hook_v19_broad_denoise_control_report.md`
- Control run ID: `diffusion-6841660796b6d7c6`
- Control selector: `planning_quality`
- Control promotion margin: `0.0`
- Replay mode: no-generation rescore over committed v19 raw candidates

Command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results\diffusion_language\generated_repair_value_hook_v19_label_raw.jsonl --task-preset lean_gpu_mixed_transfer_v19 --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-source-policy random --repair-pack constraint_span_phase_final_preserve_seeded_gated --repair-spend-trigger denoise_phase_repairability --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-phase-budget frontier --repair-selector planning_quality --repair-promotion-margin 0.0 --trajectory-selector planning_state --raw-output eval_results\diffusion_language\generated_repair_value_hook_v19_broad_denoise_control_raw.jsonl --scores-output eval_results\diffusion_language\generated_repair_value_hook_v19_broad_denoise_control_scores.json --report-output eval_results\diffusion_language\generated_repair_value_hook_v19_broad_denoise_control_report.md
```

## Comparison

| Metric | Hook | Permissive Control |
| --- | ---: | ---: |
| Repair-covered delta versus fixed | `+0.044286` | `+0.044286` |
| Repair-covered delta versus random | `+0.075777` | `+0.075777` |
| Task delta per extra generation | `+0.071` | `+0.071` |
| Oracle headroom versus repair | `0.001` | `0.001` |
| Selector regret versus repair | `0.001 over 1/8` | `0.001 over 1/8` |

## Row Accounting

- Hook-selected generated repairs: `plan_145`, `plan_149`, `plan_152`
- Permissive-control selected generated repairs: `plan_145`, `plan_149`, `plan_150`, `plan_151`, `plan_152`
- Shared substantial positives: `plan_145`, `plan_149`, `plan_152`
- Control-only no-lift row: `plan_150`
- Control-only tiny-lift row: `plan_151`

`plan_151` remains the low-margin counterexample: the raw generated repair has `+0.007000` task lift versus trajectory, but zero planning-quality delta versus source. `plan_150` remains the no-lift precision test.

## Reading

This replay does not falsify the v19 hook's local utility, because the hook and permissive control have identical aggregate repair-covered lift on the committed raw rows. It does block stronger promotion language: a simpler permissive planning-quality control can match the same aggregate score once repair candidates already exist.

The hook's advantage is allocation precision. It avoids selecting `plan_150` and `plan_151`, so it spends fewer repaired outputs as selected repairs while preserving the same task utility. The next gate should test whether that precision matters on a broader availability/control slice where unnecessary selected repairs carry explicit operational cost, or freeze a new transfer slice that tries to recover low-margin positives like `plan_151` without admitting no-lift rows like `plan_150`.
