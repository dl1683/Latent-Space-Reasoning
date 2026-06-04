# Diffusion Low-Margin Repair V20 Result

## Decision

The v20 low-margin fallback proof obligation is negative for promotion. The fresh slice produced generated repair candidates, but none improved over the selected trajectory. The `plan_151` tiny-positive geometry did not recur, so there is no basis to implement or promote a low-margin fallback layer.

The main `generated_repair_value_v1` hook remains unchanged.

## Run

- Freeze: [DIFFUSION_LOW_MARGIN_REPAIR_V20_FREEZE.md](DIFFUSION_LOW_MARGIN_REPAIR_V20_FREEZE.md)
- Run ID: `diffusion-19d1c9173d08bc15`
- Task preset: `lean_gpu_mixed_transfer_v20`
- Full generations: `26`
- Label selector: `planning_quality`
- Promotion margin: `0.0`
- Raw output: `eval_results\diffusion_language\low_margin_repair_v20_label_raw.jsonl`
- Scores: `eval_results\diffusion_language\low_margin_repair_v20_label_scores.json`
- Report: `eval_results\diffusion_language\low_margin_repair_v20_label_report.md`
- Target sheet: [DIFFUSION_LOW_MARGIN_REPAIR_V20_TARGETS.md](DIFFUSION_LOW_MARGIN_REPAIR_V20_TARGETS.md)

## Target Rows

- Target rows: `4`
- Positive promotion rows: `0`
- Positive tasks: `none`
- Negative tasks: `plan_154`, `plan_155`, `plan_158`, `plan_160`
- Candidate-aware promotion errors: `0`

| Task | Candidate Lift | Source Lift | Planning Delta | Gap Terms | Span Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `plan_154` | `-0.043000` | `+0.020000` | `0.000000` | `9` | `2.181198` |
| `plan_155` | `-0.063000` | `0.000000` | `0.000000` | `5` | `0.000000` |
| `plan_158` | `-0.127286` | `0.000000` | `0.000000` | `6` | `0.000000` |
| `plan_160` | `-0.130000` | `+0.013929` | `+0.013929` | `9` | `2.493391` |

## Utility

- Repair task delta versus fixed: `0.000000`
- Repair task delta versus random: `+0.121009`
- Repair task delta versus evolved: `0.000000`
- Repair task delta per extra generation versus evolved: `0.000000`
- Oracle headroom versus repair: `0.000000`

## Reading

V20 does not validate a low-margin fallback. It also reinforces the T70/T71 caution that polished or high-gap/high-span repair candidates are not enough: `plan_154` and `plan_160` have source-positive or span-looking features, but their generated repair candidates are negative versus the selected trajectory.

The next useful direction is not to lower the main hook margin. The next gate should either broaden availability with the existing `generated_repair_value_v1` precision policy, or test a different source-search/candidate-diversity mechanism that creates actual positive repair candidates before promotion selection.
