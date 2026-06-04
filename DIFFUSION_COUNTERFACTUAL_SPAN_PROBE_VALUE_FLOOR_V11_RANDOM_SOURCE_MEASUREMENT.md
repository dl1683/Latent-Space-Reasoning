# Diffusion Span Probe Value Floor V11 Random-Source Measurement

## Decision

The random-source measurement pass satisfies the source-divergence gate that the
fixed-source pass failed. It produced nonzero source-vs-trajectory deltas on
`5/8` planning rows while keeping the v10 measured probe-value floor frozen.

This is still not a transfer result. It only authorizes the next step: run the
predeclared random-source label pass and replay the frozen floor without
threshold changes.

## Run

- Run ID: `diffusion-80ff9643f2b45f23`
- Content hash: `80ff9643f2b45f236f97eb010a61dfaf7f0fbb19e0ffbf4267664b03065222f1`
- Scores: `eval_results\diffusion_language\span_probe_value_floor_v11_random_source_measurement_scores.json`
- Raw: `eval_results\diffusion_language\span_probe_value_floor_v11_random_source_measurement_raw.jsonl`
- Report: `eval_results\diffusion_language\span_probe_value_floor_v11_random_source_measurement_report.md`
- Full model generations: `22`
- Counterfactual probe generations: `8`
- Frozen floor: `measured_probe_value_prediction >= 0.02891517987715706`
- Source policy: `random`

## Planning Rows

| Task | Probe Value | Source Delta vs Trajectory | Prompt Gap | Prompt Coverage | Frozen Floor Selects |
| --- | ---: | ---: | ---: | ---: | --- |
| `plan_081` | `0.029694` | `-0.176929` | `12` | `0.187500` | yes |
| `plan_082` | `0.032150` | `-0.257857` | `12` | `0.000000` | yes |
| `plan_083` | `0.018333` | `-0.256429` | `12` | `0.000000` | no |
| `plan_084` | `0.027852` | `0.000000` | `3` | `0.785714` | no |
| `plan_085` | `0.022113` | `-0.307857` | `12` | `0.000000` | no |
| `plan_086` | `0.040063` | `-0.301429` | `10` | `0.357143` | yes |
| `plan_087` | `0.025530` | `0.000000` | `5` | `0.722222` | no |
| `plan_088` | `0.017525` | `0.000000` | `4` | `0.733333` | no |

## Reading

The source-divergence stress is now real: the random-source pass creates a
negative source-vs-trajectory channel on the rows most likely to test whether
probe value alone is too permissive. The label pass is now meaningful, but the
controller remains diagnostic-only until the replay shows false positives,
false negatives, utility, and probe-cost accounting.
