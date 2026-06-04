# Diffusion Composite Selector Runner Policy

This documents the first runner-facing implementation of the decomposed
diffusion selector.

## Implemented Trigger

`experiments/run_diffusion_three_arm_benchmark.py` now accepts:

`--repair-spend-trigger decomposed_four_head_selector`

The trigger exposes the fitted four-head selector from
[`DIFFUSION_COMPOSITE_SELECTOR_FIT.md`](DIFFUSION_COMPOSITE_SELECTOR_FIT.md) as
live benchmark behavior.

The runner also accepts the next experimental spend-transfer trigger:

`--repair-spend-trigger decomposed_spend_transfer_rule`

That trigger keeps the decomposed spend geometry and adds the fitted
source-task floor from
[`DIFFUSION_SPEND_TRANSFER_RULE_FIT.md`](DIFFUSION_SPEND_TRANSFER_RULE_FIT.md):
`current_decomposed_spend_source_task_ge_0p295357`. The earlier `0.3075`
floor was verified by transfer-preset run `diffusion-f50e82f88f59111b`, but
the corrected repair-oracle labels show that floor is too conservative because
it skips positive low-margin repair `plan_012`.

## Current Heads

| Head | Runner Behavior |
| --- | --- |
| Spend | Run repair when a repairable denoise skeleton exists inside the configured phase cap, the source sits inside the prompt-gap band, and source quality is at or below `--repair-value-proxy-source-quality-max`. |
| Source | Keep final-state source by default; history is diagnostic evidence until retention/source-advantage checks justify promotion. |
| Retention | Require the phase/final repair packs to preserve final-state context unless the phase-history anchor is explicitly retention safe. |
| Realization | Use the current compact preservation-seeded realization policy, `auto_compat_preserve_seeded`. |

The runner diagnostics record:

- `composite_selector_id`
- `spend_head_rule_id`
- `source_head_rule_id`
- `retention_head_rule_id`
- `realization_head_rule_id`
- `spend_head_prediction`
- `source_head_prediction`
- `retention_head_prediction`
- `realization_head_policy`
- `spend_head_source_task_min` for the transfer-rule trigger

## Reproduction Command

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-pack constraint_span_phase_final_preserve_seeded_gated --repair-source-policy fixed --repair-spend-trigger decomposed_four_head_selector --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-phase-budget frontier --repair-value-proxy-source-quality-max 0.301429 --repair-selector planning_quality_seed_realization_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --evolved-promotion-margin 0.015 --device cuda --dtype bfloat16
```

## Validation

The trigger is covered by
`tests/test_diffusion_three_arm_benchmark.py::test_primary_repair_gate_diagnostics_apply_decomposed_four_head_selector`.

The transfer-rule trigger is covered by
`tests/test_diffusion_three_arm_benchmark.py::test_primary_repair_gate_diagnostics_apply_decomposed_spend_transfer_rule`.

The current focused runner validation passed:

`python -m pytest tests\test_diffusion_three_arm_benchmark.py::test_lean_gpu_mixed_transfer_task_preset_selects_independent_transfer_suite tests\test_diffusion_three_arm_benchmark.py::test_primary_repair_gate_diagnostics_apply_decomposed_spend_transfer_rule -q`

Result: both selected runner tests passed inside the current focused validation batch.

## Fresh CUDA Confirmation

Fresh run:

`diffusion-62476b492c9e592c`

Artifacts:

- `eval_results/diffusion_language/llada_moe_mixed_decomposed_four_head_selector_frontier_v1_scores.json`
- `eval_results/diffusion_language/llada_moe_mixed_decomposed_four_head_selector_frontier_v1_report.md`
- `eval_results/diffusion_language/llada_moe_mixed_decomposed_four_head_selector_frontier_v1_raw.jsonl`

Result:

- selected latent repair score: `0.508705`
- relative GPU cost: `2.375000x`
- delta vs greedy/fixed: `0.096429`
- delta vs random perturbation: `0.136580`
- repaired tasks: `plan_004`, `plan_006`, `plan_007`
- repair-oracle headroom: `0.000000`

The run reproduces the lower-cost value-proxy point, not the top-score
`0.531116` frontier. Its importance is provenance: the same spend decisions are
now made through the named four-head selector trigger, and every repair-spend
gate row records the fitted selector head IDs.

## Fresh Transfer-Rule CUDA Confirmation

Fresh run:

`diffusion-f50e82f88f59111b`

Artifacts:

- `eval_results/diffusion_language/llada_moe_mixed_transfer_decomposed_spend_transfer_rule_frontier_v1_scores.json`
- `eval_results/diffusion_language/llada_moe_mixed_transfer_decomposed_spend_transfer_rule_frontier_v1_report.md`
- `eval_results/diffusion_language/llada_moe_mixed_transfer_decomposed_spend_transfer_rule_frontier_v1_raw.jsonl`

Result:

- full model generations: `14`
- transfer trigger: `decomposed_spend_transfer_rule`
- source-task floor: `0.3075`
- repair spends on independent planning rows: `0/4`
- selected latent repair score on repair-covered planning rows: `0.345268`
- delta vs greedy/fixed: `0.000000`
- delta vs random perturbation: `0.019536`
- `plan_012` skip reason: `transfer_source_task_score_low`

This confirms the strict source-task floor executes in the runner, but it is no
longer the recommended repair-availability setting. The all-repairable transfer
run shows `plan_012` has positive oracle repair lift `0.020000`; the selected
repair arm stayed unchanged because the promotion margin also held the repair
back.

## Expanded Transfer Label Confirmation

Fresh all-repairable run:

`diffusion-76fd30506cace1ee`

Artifacts:

- `eval_results/diffusion_language/llada_moe_mixed_transfer_v2_all_repairable_frontier_v1_scores.json`
- `eval_results/diffusion_language/llada_moe_mixed_transfer_v2_all_repairable_frontier_v1_report.md`
- `eval_results/diffusion_language/llada_moe_mixed_transfer_v2_all_repairable_frontier_v1_raw.jsonl`

Result:

- full model generations: `24`
- independent planning rows: `8`
- positive repair-availability rows: `1`
- positive row: `plan_012`
- decomposed spend-head errors: `0`
- single repairability errors: `1`

## Promotion-Value Confirmation

Generated report:

`DIFFUSION_TRANSFER_PROMOTION_VALUE.md`

Result:

- best promotion policy: `inherit`
- runner alias: `--repair-selector transfer_promotion_value`
- best promotion run: `diffusion-2a4bd4e3cad622a2`
- full model generations: `23`
- repair-covered planning score: `0.350938`
- delta vs trajectory on repair-covered planning rows: `0.002500`
- oracle headroom vs selected repair: `0.000000`

## Reading

This is now a promoted budget-controller confirmation, not a new top-score
frontier. The current transfer result says the controller needs two learned
heads: repair availability and promotion value. The inherited planning-state
selector is the current best promotion proxy on the expanded transfer slice,
and the runner exposes that proxy by name as
`--repair-selector transfer_promotion_value --repair-promotion-margin 0.0`.
