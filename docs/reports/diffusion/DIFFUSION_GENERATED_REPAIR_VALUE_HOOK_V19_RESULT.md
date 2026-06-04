# Diffusion Generated-Repair Value Hook V19 Result

## Decision

The fresh v19 live-hook validation supports `generated_repair_value_v1` as a local generated-repair selector on this slice, with one bounded miss. The hook selected three task-positive generated repairs, rejected the no-lift generated repair, and improved repair-covered task score after repair generation cost accounting.

This remains a local validation result, not broad controller-promotion language.

## Run

- Freeze: [DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_FREEZE.md](DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_FREEZE.md)
- Run ID: `diffusion-e4db8307fba01a16`
- Task preset: `lean_gpu_mixed_transfer_v19`
- Full generations: `27`
- Selector: `generated_repair_value_v1`
- Raw output: `eval_results\diffusion_language\generated_repair_value_hook_v19_label_raw.jsonl`
- Scores: `eval_results\diffusion_language\generated_repair_value_hook_v19_label_scores.json`
- Report: `eval_results\diffusion_language\generated_repair_value_hook_v19_label_report.md`
- Target sheet: [DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_TARGETS.md](DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_TARGETS.md)

## Result

- Target rows: `5`
- Generated repair positives: `4`
- Positive tasks: `plan_145`, `plan_149`, `plan_151`, `plan_152`
- Negative tasks: `plan_150`
- Hook-selected tasks: `plan_145`, `plan_149`, `plan_152`
- Hook-rejected no-lift task: `plan_150`
- Hook miss: `plan_151`, a `+0.007000` candidate lift row with zero planning-quality delta versus source
- Hook false positives on the target sheet: `0`

## Utility

- Repair coverage: `8/11` overall, `8/9` eligible
- Repair-covered fixed task score: `0.252188`
- Repair-covered random task score: `0.220696`
- Repair-covered selected latent repair task score: `0.296473`
- Repair-covered delta versus fixed: `+0.044286`
- Repair-covered delta versus random: `+0.075777`
- Repair task delta per extra generation versus evolved: `+0.071`
- Repair wins/ties/losses versus evolved: `3/5/0`
- Oracle headroom versus repair: `0.001`
- Selector regret versus repair: `0.001 over 1/8 improvable`

## Reading

The hook transfers on the fresh v19 slice under the frozen command: it selected the three substantial generated-repair positives and rejected the no-lift generated candidate. The miss on `plan_151` is deliberately not retuned away here: it is a low-margin source-tie row where task lift is only `+0.007000` and the label-free planning-quality delta versus source is `0.000000`, so it is below the `0.02` promotion margin and outside the hook's current source-relative signal.

The strongest supported statement is local: `generated_repair_value_v1` is a runner-level selector that can convert source-relative planning-quality lift into cost-positive generated repair selection on this fresh slice. The result does not prove source preservation, does not prove broad denoise triggering is sufficient, and does not justify global promotion without more transfer slices.
