# Diffusion Span Probe Value Floor V11 Measurement Boundary

## Decision

Do not run the v11 label pass as a source-divergent transfer test yet. The
frozen measurement pass completed, but it failed the predeclared source-stress
gate: every planning row has `source_task_delta_vs_trajectory = 0.000000`.

This is still useful evidence. It shows the fixed-source command alone is not
enough to force source divergence on the v11 task slice when the planning-state
trajectory selector also chooses the fixed source.

## Run

- Run ID: `diffusion-0bd575b42c734811`
- Content hash: `0bd575b42c734811be7892b80cf75aab88324c30905519062d226a81aab7cf09`
- Scores: `eval_results\diffusion_language\span_probe_value_floor_v11_measurement_scores.json`
- Raw: `eval_results\diffusion_language\span_probe_value_floor_v11_measurement_raw.jsonl`
- Report: `eval_results\diffusion_language\span_probe_value_floor_v11_measurement_report.md`
- Full model generations: `22`
- Counterfactual probe generations: `8`
- Frozen floor: `measured_probe_value_prediction >= 0.02891517987715706`

## Planning Rows

| Task | Probe Value | Source Delta vs Trajectory | Prompt Gap | Prompt Coverage | Frozen Floor Selects |
| --- | ---: | ---: | ---: | ---: | --- |
| `plan_081` | `0.023126` | `0.000000` | `4` | `0.750000` | no |
| `plan_082` | `0.000000` | `0.000000` | `0` | `1.000000` | no |
| `plan_083` | `0.020881` | `0.000000` | `11` | `0.312500` | no |
| `plan_084` | `0.027852` | `0.000000` | `3` | `0.785714` | no |
| `plan_085` | `0.000000` | `0.000000` | `7` | `0.533333` | no |
| `plan_086` | `0.040786` | `0.000000` | `1` | `0.928571` | yes |
| `plan_087` | `0.025530` | `0.000000` | `5` | `0.722222` | no |
| `plan_088` | `0.017525` | `0.000000` | `4` | `0.733333` | no |

## Reading

The measured probe-value floor did not fail on labels; it did not reach the
declared source-divergence test. Running labels now would only test another
same-source slice and would not satisfy the v11 freeze gate.

The next protocol should force divergence explicitly, for example by freezing a
trajectory source from the selected planning state while measuring a different
repair source, or by replaying multiple source policies from the same generated
base pool before labels are used.
