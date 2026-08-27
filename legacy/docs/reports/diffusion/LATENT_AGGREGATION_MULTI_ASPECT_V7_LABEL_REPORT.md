# Diffusion Schedule-Selection Benchmark Report

Full model generations: `336`
Counterfactual probe generations: `0`
Arm selections: `336`
Run ID: `diffusion-e09d816eb435b85b`
Content hash: `e09d816eb435b85b5e565bf873abed072be5788be00a1109aeb24de85853e239`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `inherit`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `False`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `False`
History mutability: `monotonic 336/336, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `True`
Repair pack: `constraint_span_phase_final_preserve_seeded_gated`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `denoise_phase_repairability`
Counterfactual probe mode: `triage`
Counterfactual probe policy: `deterministic_missing_constraint_probe_v1`
Repair source-quality threshold: `0.500`
Repair source min chars: `320`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `999`
Repair source prompt coverage band: `0.000-1.000`
Repair value-proxy source-quality max: `0.310`
Repair cost penalty lambda: `0.180`
Repair transfer source-task min: `0.2954`
Repair phase budget: `custom`
Repair denoise skeleton max step: `none`
Phase-source threshold band: `target>=0.960, text>=0.960, chars>=0.950`
Repair source controls: ``
History rescue fractions: ``
History rescue visible: `False`
History rescue trigger: `baseline`
History rescue source controls: ``
Prompt-guided rescue trigger: `off`
Prompt-guided rescue limit: `1`
Prompt-guided rescue source-quality threshold: `0.450`
Prompt-guided rescue source controls: ``
Constraint-gap rescue trigger: `off`
Constraint-gap rescue limit: `1`
Constraint-gap rescue min terms: `6`
Constraint-gap rescue source-quality band: `0.400-0.500`
Constraint-gap rescue source controls: ``
Repair selector: `generated_repair_value_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.006`
Trajectory task delta vs random: `0.017`
Trajectory wins/ties/losses vs fixed: `7/89/0`
Trajectory wins/ties/losses vs random: `17/76/3`
Oracle generation budget/task: `3.50`
Oracle task score: `0.185`
Oracle headroom vs trajectory: `0.025`
Oracle wins/ties/losses vs trajectory: `33/63/0`
Selector regret vs trajectory: `0.025 over 33/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.057`
Repair task delta vs random: `0.081`
Repair task delta vs trajectory: `0.049`
Repair task delta vs evolved: `0.049`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.024`
Repair wins/ties/losses vs evolved: `29/19/0`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `5/43/0`
Selector regret vs repair: `0.002 over 5/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.270320 | 0.000000 | 0.024122 | - | - |
| random perturbation | repair-covered tasks | 0.246198 | -0.024122 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.327170 | 0.056850 | 0.080972 | 33/15/0 | 38/9/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.154 | 0.468 | 0.233 |
| random | 96 | 1.00 | 0.142 | 0.437 | 0.216 |
| trajectory_selected | 96 | 2.50 | 0.160 | 0.473 | 0.238 |
| repair_selected | 48 | 4.00 | 0.327 | 0.651 | 0.408 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.154 | 0.468 | 0.233 |
| planning | random | 96 | 1.00 | 0.142 | 0.437 | 0.216 |
| planning | trajectory_selected | 96 | 2.50 | 0.160 | 0.473 | 0.238 |
| planning | repair_selected | 48 | 4.00 | 0.327 | 0.651 | 0.408 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_345 | random_32 | True | denoise_phase_repairable | False |  | 0.486 | 0.386 | 266 | True | 1 | 0.900 | True | True | 6.000 | 0.188 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_346 | random_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 288 | True | 4 | 0.636 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_347 | random_32 | True | denoise_phase_repairable | False |  | 0.263 | 0.223 | 278 | True | 4 | 0.667 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.285 | 0.265 | 333 | True | 6 | 0.400 | True | True | 3.000 | 0.094 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_349 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.379 | 0.281 | 321 | True | 2 | 0.800 | True | True | 4.000 | 0.125 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 342 | True | 5 | 0.556 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.375 | 0.272 | 331 | True | 3 | 0.667 | True | True | 3.000 | 0.094 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_352 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 332 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_353 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 288 | True | 1 | 0.909 | True | True | 4.000 | 0.125 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.340 | 0.260 | 322 | True | 2 | 0.778 | True | True | 3.000 | 0.094 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.250 | 0.230 | 296 | True | 5 | 0.600 | True | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_356 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 356 | True | 2 | 0.800 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_357 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 321 | True | 1 | 0.909 | True | True | 4.000 | 0.125 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_358 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 300 | True | 1 | 0.875 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.364 | 0.324 | 275 | True | 2 | 0.889 | True | True | 3.000 | 0.094 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 332 | True | 4 | 0.600 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.310 | 0.290 | 341 | True | 3 | 0.667 | True | True | 3.000 | 0.094 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_362 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.399 | 0.281 | 283 | True | 2 | 0.800 | True | True | 3.000 | 0.094 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_363 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.308 | 0.248 | 354 | True | 4 | 0.667 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_364 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 330 | True | 1 | 1.000 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_365 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 302 | True | 5 | 0.500 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_366 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 358 | True | 4 | 0.636 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_367 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 291 | True | 3 | 0.778 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_368 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 345 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_369 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 358 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_370 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 356 | True | 1 | 0.889 | True | True | 3.000 | 0.094 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_371 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.243 | 0.223 | 289 | True | 5 | 0.600 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_372 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 324 | True | 2 | 0.800 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 332 | True | 3 | 0.727 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_374 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 301 | True | 1 | 0.889 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_375 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.314 | 0.294 | 357 | True | 0 | 1.000 | True | True | 3.000 | 0.094 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 294 | True | 1 | 0.857 | True | True | 5.000 | 0.156 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.275 | 0.235 | 296 | True | 3 | 0.625 | True | True | 5.000 | 0.156 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_378 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.282 | 0.282 | 236 | True | 4 | 0.000 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_379 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.275 | 0.235 | 287 | True | 4 | 0.556 | True | True | 4.000 | 0.125 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_380 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 336 | True | 0 | 1.000 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 371 | True | 4 | 0.556 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_382 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.242 | 0.223 | 268 | True | 5 | 0.545 | True | True | 2.000 | 0.062 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_383 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.408 | 0.408 | 302 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 303 | True | 2 | 0.800 | True | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 321 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_386 | random_32 | True | denoise_phase_repairable | False |  | 0.137 | 0.117 | 98 | True | 3 | 0.667 | True | True | 5.000 | 0.156 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_387 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 349 | True | 1 | 0.900 | True | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_388 | random_32 | True | denoise_phase_repairable | False |  | 0.105 | 0.045 | 60 | True | 5 | 0.556 | True | True | 14.000 | 0.438 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.404 | 0.326 | 238 | True | 8 | 0.300 | True | True | 5.000 | 0.156 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_390 | random_32 | True | denoise_phase_repairable | False |  | 0.255 | 0.235 | 322 | True | 2 | 0.714 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_391 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.201 | 0.201 | 195 | True | 6 | 0.333 | True | True | 6.000 | 0.188 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_392 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.260 | 371 | True | 2 | 0.818 | True | True | 4.000 | 0.125 | 0.182 | 0.182 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 48 | 27 | low_confidence_32,random_32 | final | 30.9 | 0.979 | 0.021 | 0.000 | 0.029 | 0.025 | 0.039 | 0.042 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28/11/9 | 0.320 | 0.659 | 0.405 |
| history_prefix_25_repair | 48 | 2 | low_confidence_32,random_32 | history | 48.1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | -0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 14/22/12 | 0.276 | 0.676 | 0.376 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_345 | False | random_32 | 2.049 | 0.729 | 1.000 | 0.000 | 0.300 | False | Use triggers or checks to verify task freshness before and after processing claims. |
| llada-8b-instruct-hf | plan_346 | False | random_32 | 2.506 | 0.745 | 1.000 | 0.000 | 0.091 | False | Identify key tasks and objectives. |
| llada-8b-instruct-hf | plan_346 | False | random_32 | 2.543 | 0.820 | 1.000 | 0.000 | 0.091 | False | Develop and expand the existing ontology. |
| llada-8b-instruct-hf | plan_346 | False | random_32 | 3.268 | 0.745 | 1.000 | 0.000 | 0.000 | False | Analyze and report results for insights and improvement. |
| llada-8b-instruct-hf | plan_347 | False | random_32 | 2.046 | 0.673 | 1.000 | 0.000 | 0.250 | False | This allows for better tracking, collaboration, and clear accountability, ultimately im... |
| llada-8b-instruct-hf | plan_348 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Assign each phase to a team member or task. |
| llada-8b-instruct-hf | plan_348 | False | low_confidence_32 | 2.100 | 0.928 | 1.000 | 0.000 | 0.000 | False | Set milestones and checkpoints to track progress and ensure completion. |
| llada-8b-instruct-hf | plan_348 | False | low_confidence_32 | 2.829 | 0.936 | 1.000 | 0.000 | 0.100 | False | Regularly review the adjust the timeline as needed to ensure the project on track and w... |
| llada-8b-instruct-hf | plan_349 | True | low_confidence_32 | 1.845 | 0.790 | 1.000 | 0.000 | 0.300 | False | Identify potential complements in the final plan. |
| llada-8b-instruct-hf | plan_349 | True | low_confidence_32 | 2.049 | 0.865 | 1.000 | 0.000 | 0.100 | False | Evaluate each complement's potential rollback benefits vs. benefits. |
| llada-8b-instruct-hf | plan_349 | True | low_confidence_32 | 2.647 | 0.865 | 1.000 | 0.000 | 0.200 | False | Implement the chosen complement in the final plan. |
| llada-8b-instruct-hf | plan_350 | True | low_confidence_32 | 3.189 | 0.928 | 1.000 | 0.000 | 0.000 | False | Use tools like time tracking, feedback forms, and performance dashboards to gather data. |
| llada-8b-instruct-hf | plan_350 | True | low_confidence_32 | 2.184 | 1.000 | 1.000 | 0.000 | 0.333 | False | Regularly review these metrics to assess the candidate's improvement and adjust adjustm... |
| llada-8b-instruct-hf | plan_351 | True | low_confidence_32 | 2.115 | 0.981 | 1.000 | 0.000 | 0.111 | False | Then, the scope should be expanded to include the necessary elements. |
| llada-8b-instruct-hf | plan_351 | True | low_confidence_32 | 2.070 | 0.917 | 1.000 | 0.000 | 0.111 | False | Next, the boundaries should be set to ensure the scope does not exceed acceptable limits. |
| llada-8b-instruct-hf | plan_351 | True | low_confidence_32 | 2.177 | 0.981 | 1.000 | 0.000 | 0.333 | False | Finally, the policy should be developed to complement the defined scope and boundaries. |
| llada-8b-instruct-hf | plan_352 | False | low_confidence_32 | 1.308 | 0.735 | 1.000 | 0.000 | 0.400 | False | Identify the action directions recommended by both candidates. |
| llada-8b-instruct-hf | plan_352 | False | low_confidence_32 | 1.423 | 0.910 | 1.000 | 0.000 | 0.200 | False | Determine the polarity (positive or negative) of each action direction. |
| llada-8b-instruct-hf | plan_352 | False | low_confidence_32 | 2.070 | 0.803 | 1.000 | 0.000 | 0.400 | False | Compare the polarities and action directions to identify any contradictions or inconsis... |
| llada-8b-instruct-hf | plan_353 | True | low_confidence_32 | 1.297 | 0.674 | 1.000 | 0.000 | 0.273 | False | Identify the average length of verbose answers. |
| llada-8b-instruct-hf | plan_353 | True | low_confidence_32 | 1.425 | 0.910 | 1.000 | 0.000 | 0.182 | False | Normalize the length of all answers to this average length. |
| llada-8b-instruct-hf | plan_353 | True | low_confidence_32 | 2.080 | 0.803 | 1.000 | 0.000 | 0.455 | False | Compare the normalized length of answers to identify false-positive results. |
| llada-8b-instruct-hf | plan_354 | False | low_confidence_32 | 1.422 | 0.925 | 1.000 | 0.000 | 0.222 | False | analyze the overlap between aspects, classify the overlap, |
| llada-8b-instruct-hf | plan_354 | False | low_confidence_32 | 2.011 | 0.621 | 1.000 | 0.000 | 0.222 | False | prioritize the overlap, resolve the overlap, test the audit, review and validate the au... |
| llada-8b-instruct-hf | plan_355 | True | low_confidence_32 | 2.065 | 0.865 | 1.000 | 0.000 | 0.100 | False | Implement the perturbation in the model architecture and training. |
| llada-8b-instruct-hf | plan_355 | True | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Evaluate the impact on performance and robustness. |
| llada-8b-instruct-hf | plan_355 | True | low_confidence_32 | 2.788 | 0.790 | 1.000 | 0.000 | 0.000 | False | Document the process and results for reproducibility. |
| llada-8b-instruct-hf | plan_356 | False | low_confidence_32 | 1.452 | 1.000 | 1.000 | 0.000 | 0.200 | False | This ensures that the probe does not inadvertently include the missing aspect or aspect... |
| llada-8b-instruct-hf | plan_356 | False | low_confidence_32 | 2.190 | 1.000 | 1.000 | 0.000 | 0.300 | False | The boundary should be set to prevent any leakage or leakage related to the missing asp... |
| llada-8b-instruct-hf | plan_357 | False | low_confidence_32 | 2.199 | 1.000 | 1.000 | 0.000 | 0.182 | False | This will allow for a fair comparison between the two methods, as the improvements are... |
| llada-8b-instruct-hf | plan_358 | True | low_confidence_32 | 2.112 | 1.000 | 1.000 | 0.000 | 0.125 | False | Normal the CBR by dividing the total cost by the total benefit. |
| llada-8b-instruct-hf | plan_358 | True | low_confidence_32 | 2.141 | 0.893 | 1.000 | 0.000 | 0.250 | False | Compare the normalized CBRs and select the option with the lowest normalized CBR for in... |
| llada-8b-instruct-hf | plan_359 | True | low_confidence_32 | 2.601 | 0.955 | 1.000 | 0.000 | 0.111 | False | Task ID 2. |
| llada-8b-instruct-hf | plan_359 | True | low_confidence_32 | 1.786 | 0.637 | 1.000 | 0.000 | 0.222 | False | Old Ontology complement (if applicable) 3. |
| llada-8b-instruct-hf | plan_359 | True | low_confidence_32 | 1.835 | 0.731 | 1.000 | 0.000 | 0.222 | False | Owner complement (if applicable) 4. |
| llada-8b-instruct-hf | plan_360 | True | low_confidence_32 | 1.452 | 1.000 | 1.000 | 0.000 | 0.200 | False | Please provide the list of concerns, including their potential impact, relevant stakeho... |
| llada-8b-instruct-hf | plan_360 | True | low_confidence_32 | 2.202 | 1.000 | 1.000 | 0.000 | 0.200 | False | Once I have this information, I can create a detailed plan for the audit. |
| llada-8b-instruct-hf | plan_361 | False | low_confidence_32 | 1.333 | 0.873 | 1.000 | 0.000 | 0.556 | False | If the temporal-order complement is before the anchor mitigation order, prioritize the... |
| llada-8b-instruct-hf | plan_361 | False | low_confidence_32 | 2.083 | 0.873 | 1.000 | 0.000 | 0.556 | False | If the anchor mitigation order is after the temporal-order complement, prioritize the t... |
| llada-8b-instruct-hf | plan_362 | True | low_confidence_32 | 1.903 | 0.865 | 1.000 | 0.000 | 0.200 | False | The trigger must be specific and measurable. |
| llada-8b-instruct-hf | plan_362 | True | low_confidence_32 | 1.903 | 0.865 | 1.000 | 0.000 | 0.200 | False | The complement must be directly related to the trigger. |
| llada-8b-instruct-hf | plan_362 | True | low_confidence_32 | 3.315 | 0.865 | 1.000 | 0.000 | 0.100 | False | The complement must be actionable and easy to understand. |
| llada-8b-instruct-hf | plan_363 | True | low_confidence_32 | 1.449 | 0.981 | 1.000 | 0.000 | 0.222 | False | Then, evaluate the available metrics and their relevance to these objectives. |
| llada-8b-instruct-hf | plan_363 | True | low_confidence_32 | 2.122 | 1.000 | 1.000 | 0.000 | 0.111 | False | Use a scoring system to rate the relevance of each metric. |
| llada-8b-instruct-hf | plan_363 | True | low_confidence_32 | 2.147 | 0.981 | 1.000 | 0.000 | 0.444 | False | Finally, select the metrics with the highest relevance scores to ensure accurate and me... |
| llada-8b-instruct-hf | plan_364 | True | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.111 | False | This could could be a specific task, activity, or component within the project scope. |
| llada-8b-instruct-hf | plan_364 | True | low_confidence_32 | 2.187 | 1.000 | 1.000 | 0.000 | 0.444 | False | Once this aspect is identified, plan a review session to ensure all all necessary oblig... |
| llada-8b-instruct-hf | plan_365 | True | low_confidence_32 | 2.085 | 0.910 | 1.000 | 0.000 | 0.100 | False | Establish a clear communication protocol to gather input from both sources. |
| llada-8b-instruct-hf | plan_365 | True | low_confidence_32 | 2.085 | 0.910 | 1.000 | 0.000 | 0.100 | False | Implement a weighting system to balance the influence of each source. |
| llada-8b-instruct-hf | plan_365 | True | low_confidence_32 | 2.162 | 0.910 | 1.000 | 0.000 | 0.200 | False | Develop a contingency plan to address potential where- and when- conflicts in the future. |
| llada-8b-instruct-hf | plan_366 | False | low_confidence_32 | 2.147 | 0.905 | 1.000 | 0.000 | 0.273 | False | This rule should prioritize the inclusion of new aspects that will significantly improv... |
| llada-8b-instruct-hf | plan_367 | True | low_confidence_32 | 1.361 | 0.790 | 1.000 | 0.000 | 0.222 | False | Identify the dropped aspect. |
| llada-8b-instruct-hf | plan_367 | True | low_confidence_32 | 1.400 | 0.865 | 1.000 | 0.000 | 0.222 | False | Determine the impact of the dropped aspect. |
| llada-8b-instruct-hf | plan_367 | True | low_confidence_32 | 1.902 | 0.388 | 1.000 | 0.000 | 0.444 | False | Acknowledge the absence of the dropped aspect in the final answer. |
| llada-8b-instruct-hf | plan_368 | False | low_confidence_32 | 2.046 | 0.850 | 1.000 | 0.000 | 0.111 | False | Establish a reporting framework to collect and analyze data. |
| llada-8b-instruct-hf | plan_368 | False | low_confidence_32 | 2.871 | 1.000 | 1.000 | 0.000 | 0.000 | False | Schedule a regular schedule to review progress, identify issues, and implement strategi... |
| llada-8b-instruct-hf | plan_369 | False | low_confidence_32 | 2.091 | 0.856 | 1.000 | 0.000 | 0.444 | False | This implies that the routing protocols will be designed to handle high-anchor tasks mo... |
| llada-8b-instruct-hf | plan_370 | True | low_confidence_32 | 1.414 | 1.000 | 1.000 | 0.000 | 0.444 | False | Since the safety aspects are not provided, focus on the owner aspects to tailor the rep... |
| llada-8b-instruct-hf | plan_370 | True | low_confidence_32 | 2.131 | 1.000 | 1.000 | 0.000 | 0.667 | False | This will ensure that the reporting is relevant and relevant to the owner aspects of th... |
| llada-8b-instruct-hf | plan_371 | False | low_confidence_32 | 1.358 | 0.790 | 1.000 | 0.000 | 0.200 | False | Identify the perturbation source and its characteristics. |
| llada-8b-instruct-hf | plan_371 | False | low_confidence_32 | 2.021 | 0.758 | 1.000 | 0.000 | 0.000 | False | Collect a sample of original data. |
| llada-8b-instruct-hf | plan_371 | False | low_confidence_32 | 2.080 | 0.758 | 1.000 | 0.000 | 0.200 | False | Compare the perturbed data to the original data to determine the noise rate. |
| llada-8b-instruct-hf | plan_372 | True | low_confidence_32 | 2.098 | 0.968 | 1.000 | 0.000 | 0.100 | False | Set a threshold for positive thetas and reject any results below this threshold. |
| llada-8b-instruct-hf | plan_372 | True | low_confidence_32 | 2.081 | 0.825 | 1.000 | 0.000 | 0.500 | False | This will prevent the v7 result from passing coverage by selecting small but insignific... |
| llada-8b-instruct-hf | plan_373 | True | low_confidence_32 | 1.419 | 1.000 | 1.000 | 0.000 | 0.455 | False | use the candidate's concrete owner as the escalation path, |
| llada-8b-instruct-hf | plan_373 | True | low_confidence_32 | 1.435 | 1.000 | 1.000 | 0.000 | 0.364 | False | use the candidate's concrete owner as escalation, |
| llada-8b-instruct-hf | plan_373 | True | low_confidence_32 | 2.120 | 0.892 | 1.000 | 0.000 | 0.455 | False | use the candidate's concrete owner as the escalation path. |
| llada-8b-instruct-hf | plan_374 | True | low_confidence_32 | 1.380 | 0.865 | 1.000 | 0.000 | 0.333 | False | Break the sequence into partial aspects. |
| llada-8b-instruct-hf | plan_374 | True | low_confidence_32 | 1.380 | 0.865 | 1.000 | 0.000 | 0.333 | False | Gather evidence for each partial aspect. |
| llada-8b-instruct-hf | plan_374 | True | low_confidence_32 | 2.050 | 0.770 | 1.000 | 0.000 | 0.556 | False | Aggregate the evidence for each partial aspect to form a comprehensive understanding of... |
| llada-8b-instruct-hf | plan_375 | True | low_confidence_32 | 2.600 | 1.000 | 0.778 | 0.000 | 0.222 | False | This can be done by using the criteria from one candidate as a reference and comparing... |
| llada-8b-instruct-hf | plan_376 | True | low_confidence_32 | 2.118 | 0.856 | 1.000 | 0.000 | 0.286 | False | This involves referencing the relevant data, findings, or results from the v6 version t... |
| llada-8b-instruct-hf | plan_377 | True | low_confidence_32 | 1.413 | 0.981 | 1.000 | 0.000 | 0.375 | False | Train the aspect extractor on the labeled data and then test it on the unlabeled data. |
| llada-8b-instruct-hf | plan_377 | True | low_confidence_32 | 2.048 | 0.786 | 1.000 | 0.000 | 0.500 | False | Measure the performance on the unlabeled data to determine if there is any label leakag... |
| llada-8b-instruct-hf | plan_378 | False | low_confidence_32 | 1.977 | 0.643 | 1.000 | 0.000 | 0.000 | False | Set up GPU environment. |
| llada-8b-instruct-hf | plan_378 | False | low_confidence_32 | 1.977 | 0.643 | 1.000 | 0.000 | 0.000 | False | Compile and link v7 code. |
| llada-8b-instruct-hf | plan_378 | False | low_confidence_32 | 2.642 | 0.480 | 1.000 | 0.000 | 0.000 | False | Validate results for reproducibility. |
| llada-8b-instruct-hf | plan_379 | False | low_confidence_32 | 2.128 | 0.874 | 1.000 | 0.000 | 0.222 | False | We can then compare the length of the answers to determine if v v7 prioritizes longer a... |
| llada-8b-instruct-hf | plan_380 | True | low_confidence_32 | 1.889 | 1.000 | 0.444 | 0.000 | 0.556 | False | This would allow the analysis to focus solely on the measurement value of the aspect wi... |
| llada-8b-instruct-hf | plan_381 | True | low_confidence_32 | 1.890 | 0.378 | 1.000 | 0.000 | 0.333 | False | If the contradictions are significant, consider excluding the source family from the an... |
| llada-8b-instruct-hf | plan_382 | True | low_confidence_32 | 1.320 | 0.688 | 1.000 | 0.000 | 0.182 | False | Defining claim boundaries. |
| llada-8b-instruct-hf | plan_382 | True | low_confidence_32 | 1.297 | 0.688 | 1.000 | 0.000 | 0.364 | False | Planning within the claim boundaries. |
| llada-8b-instruct-hf | plan_382 | True | low_confidence_32 | 2.070 | 0.688 | 1.000 | 0.000 | 0.182 | False | Review and adjustment within the claim boundaries. |
| llada-8b-instruct-hf | plan_383 | True | low_confidence_32 | 2.519 | 0.790 | 1.000 | 0.000 | 0.111 | False | Document all changes made during the copying process. |
| llada-8b-instruct-hf | plan_383 | True | low_confidence_32 | 3.314 | 0.865 | 1.000 | 0.000 | 0.111 | False | Regularly review and update procedures to maintain controls. |
| llada-8b-instruct-hf | plan_384 | False | low_confidence_32 | 1.459 | 1.000 | 1.000 | 0.000 | 0.200 | False | This this provides a measure of the number of tasks that have been newly covered. |
| llada-8b-instruct-hf | plan_384 | False | low_confidence_32 | 2.170 | 1.000 | 1.000 | 0.000 | 0.400 | False | Answer: Count the number of old no-complement tasks that are now covered by the new ont... |
| llada-8b-instruct-hf | plan_385 | True | low_confidence_32 | 2.154 | 0.925 | 1.000 | 0.000 | 0.222 | False | Additionally, you would need to define the proof required to verify the correctness of... |
| llada-8b-instruct-hf | plan_386 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Aggregation should abstain when the number of weaker candidates exceeds that of the str... |
| llada-8b-instruct-hf | plan_387 | True | low_confidence_32 | 2.877 | 1.000 | 1.000 | 0.000 | 0.000 | False | Further analysis or additional data may be needed to confirm the reliability of the fin... |
| llada-8b-instruct-hf | plan_388 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The claim boundary is: "One new source carries all v7 lift." |
| llada-8b-instruct-hf | plan_389 | False | low_confidence_32 | 2.538 | 0.790 | 1.000 | 0.000 | 0.000 | False | Identify the key features of v6. |
| llada-8b-instruct-hf | plan_389 | False | low_confidence_32 | 2.538 | 0.790 | 1.000 | 0.000 | 0.000 | False | Analyze the performance of v6. |
| llada-8b-instruct-hf | plan_389 | False | low_confidence_32 | 2.774 | 0.770 | 1.000 | 0.000 | 0.000 | False | Discuss the factors that contributed to the failure of v6. |
| llada-8b-instruct-hf | plan_390 | False | random_32 | 1.947 | 0.626 | 1.000 | 0.000 | 0.143 | False | Create a dedicated repository for documenting artifacts. |
| llada-8b-instruct-hf | plan_390 | False | random_32 | 2.055 | 0.865 | 1.000 | 0.000 | 0.143 | False | Establish a schedule for documentation, review, and archiving. |
| llada-8b-instruct-hf | plan_390 | False | random_32 | 2.805 | 0.865 | 1.000 | 0.000 | 0.143 | False | Assign roles and responsibilities for documentation contributors. |
| llada-8b-instruct-hf | plan_391 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Define requirements for v8. |
| llada-8b-instruct-hf | plan_391 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Create design for v8. |
| llada-8b-instruct-hf | plan_391 | False | low_confidence_32 | 2.826 | 0.875 | 1.000 | 0.000 | 0.000 | False | Prepare documentation and deployment for v8. |
| llada-8b-instruct-hf | plan_392 | True | low_confidence_32 | 2.124 | 1.000 | 1.000 | 0.000 | 0.000 | False | This involves shouldering the responsibilities of each component, ensuring their compat... |
| llada-8b-instruct-hf | plan_392 | True | low_confidence_32 | 2.780 | 0.837 | 1.000 | 0.000 | 0.091 | False | Use a variety of datasets and scenarios to validate the robustness and accuracy of the... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_345 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_346 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_347 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_348 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_349 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_350 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_351 | entropy_32 | entropy_32 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_352 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_353 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_354 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_355 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_356 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_357 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_358 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_359 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_360 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_361 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_362 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_363 | entropy_32 | entropy_32 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_364 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_365 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_366 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_367 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_368 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_369 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_370 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_371 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_372 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_373 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_374 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_375 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_376 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_377 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_378 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_379 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.126 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_380 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_381 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_382 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_383 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_384 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_385 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_386 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_387 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_388 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_389 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_390 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_391 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_392 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_345 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.502 | 0.000 | 0.000 | 0.000 | 0.335 | 0.335 | 0.486 | 0.000 | 0.486 | 0.000 | 0.486 | 0.000 |
| llada-8b-instruct-hf | plan_346 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.379 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_347 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.383 | 0.000 | 0.000 | 0.000 | 0.241 | 0.241 | 0.263 | 0.000 | 0.263 | 0.000 | 0.283 | 0.020 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.321 | 0.000 | 0.000 | 0.000 | 0.285 | 0.045 | 0.285 | 0.000 | 0.285 | 0.000 | 0.285 | 0.000 |
| llada-8b-instruct-hf | plan_349 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.471 | 0.000 | 0.076 | 0.076 | 0.379 | 0.375 | 0.379 | 0.000 | 0.436 | 0.057 | 0.436 | 0.000 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.385 | 0.000 | 0.083 | 0.083 | 0.304 | 0.178 | 0.304 | 0.000 | 0.398 | 0.094 | 0.398 | 0.000 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.423 | 0.000 | 0.126 | 0.126 | 0.375 | 0.375 | 0.375 | 0.000 | 0.413 | 0.038 | 0.413 | 0.000 |
| llada-8b-instruct-hf | plan_352 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.456 | 0.000 | 0.000 | 0.000 | 0.283 | 0.303 | 0.283 | 0.000 | 0.283 | 0.000 | 0.303 | 0.020 |
| llada-8b-instruct-hf | plan_353 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.449 | 0.000 | 0.027 | 0.027 | 0.283 | 0.283 | 0.283 | 0.000 | 0.324 | 0.041 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.437 | 0.000 | 0.000 | 0.000 | 0.340 | 0.340 | 0.340 | 0.000 | 0.340 | 0.000 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.387 | 0.000 | 0.064 | 0.064 | 0.250 | 0.250 | 0.250 | 0.000 | 0.287 | 0.037 | 0.287 | 0.000 |
| llada-8b-instruct-hf | plan_356 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.423 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_357 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.434 | 0.000 | 0.000 | 0.000 | 0.260 | 0.177 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_358 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.453 | 0.000 | 0.087 | 0.087 | 0.283 | 0.283 | 0.283 | 0.000 | 0.321 | 0.039 | 0.321 | 0.000 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.480 | 0.000 | 0.110 | 0.110 | 0.364 | 0.364 | 0.364 | 0.000 | 0.444 | 0.080 | 0.444 | 0.000 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.358 | 0.000 | 0.203 | 0.203 | 0.200 | 0.045 | 0.200 | 0.000 | 0.380 | 0.180 | 0.380 | 0.000 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.423 | 0.000 | 0.000 | 0.000 | 0.310 | 0.310 | 0.310 | 0.000 | 0.310 | 0.000 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_362 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.470 | 0.000 | 0.121 | 0.121 | 0.399 | 0.399 | 0.399 | 0.000 | 0.474 | 0.075 | 0.474 | 0.000 |
| llada-8b-instruct-hf | plan_363 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.412 | 0.000 | 0.126 | 0.126 | 0.308 | 0.308 | 0.308 | 0.000 | 0.392 | 0.084 | 0.392 | 0.000 |
| llada-8b-instruct-hf | plan_364 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.424 | 0.000 | 0.207 | 0.207 | 0.261 | 0.241 | 0.261 | 0.000 | 0.422 | 0.160 | 0.422 | 0.000 |
| llada-8b-instruct-hf | plan_365 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.326 | 0.000 | 0.042 | 0.042 | 0.240 | 0.240 | 0.240 | 0.000 | 0.261 | 0.021 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_366 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.338 | 0.000 | 0.000 | 0.000 | 0.220 | 0.045 | 0.220 | 0.000 | 0.220 | 0.000 | 0.240 | 0.020 |
| llada-8b-instruct-hf | plan_367 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.419 | 0.000 | 0.049 | 0.049 | 0.261 | 0.261 | 0.261 | 0.000 | 0.279 | 0.017 | 0.279 | 0.000 |
| llada-8b-instruct-hf | plan_368 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.387 | 0.000 | 0.044 | 0.044 | 0.241 | 0.241 | 0.241 | 0.000 | 0.263 | 0.021 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_369 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.307 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_370 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.411 | 0.000 | 0.103 | 0.103 | 0.301 | 0.276 | 0.301 | 0.000 | 0.376 | 0.075 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_371 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.369 | 0.000 | 0.000 | 0.000 | 0.243 | 0.243 | 0.243 | 0.000 | 0.243 | 0.000 | 0.263 | 0.020 |
| llada-8b-instruct-hf | plan_372 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.422 | 0.000 | 0.183 | 0.183 | 0.283 | 0.283 | 0.283 | 0.000 | 0.402 | 0.119 | 0.402 | 0.000 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.400 | 0.000 | 0.157 | 0.157 | 0.221 | 0.221 | 0.221 | 0.000 | 0.339 | 0.118 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_374 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.425 | 0.000 | 0.202 | 0.202 | 0.240 | 0.240 | 0.240 | 0.000 | 0.400 | 0.160 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_375 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.520 | 0.000 | 0.216 | 0.216 | 0.314 | 0.362 | 0.314 | 0.000 | 0.474 | 0.160 | 0.474 | 0.000 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.426 | 0.000 | 0.203 | 0.203 | 0.240 | 0.240 | 0.240 | 0.000 | 0.400 | 0.160 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.391 | 0.000 | 0.027 | 0.027 | 0.275 | 0.275 | 0.275 | 0.000 | 0.299 | 0.024 | 0.299 | 0.000 |
| llada-8b-instruct-hf | plan_378 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.253 | 0.000 | 0.000 | 0.000 | 0.282 | 0.282 | 0.282 | 0.000 | 0.282 | 0.000 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_379 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.372 | 0.000 | 0.000 | 0.000 | 0.275 | 0.263 | 0.275 | 0.000 | 0.275 | 0.000 | 0.275 | 0.000 |
| llada-8b-instruct-hf | plan_380 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.452 | 0.000 | 0.099 | 0.099 | 0.261 | 0.198 | 0.261 | 0.000 | 0.333 | 0.071 | 0.333 | 0.000 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.355 | 0.000 | 0.103 | 0.103 | 0.261 | 0.261 | 0.261 | 0.000 | 0.356 | 0.095 | 0.356 | 0.000 |
| llada-8b-instruct-hf | plan_382 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.369 | 0.000 | 0.087 | 0.087 | 0.242 | 0.280 | 0.242 | 0.000 | 0.301 | 0.059 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_383 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.496 | 0.000 | 0.090 | 0.090 | 0.408 | 0.408 | 0.408 | 0.000 | 0.445 | 0.038 | 0.445 | 0.000 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.404 | 0.000 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.376 | 0.000 | 0.088 | 0.088 | 0.220 | 0.220 | 0.220 | 0.000 | 0.284 | 0.064 | 0.290 | 0.006 |
| llada-8b-instruct-hf | plan_386 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.314 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 | 0.137 | 0.000 | 0.137 | 0.000 | 0.137 | 0.000 |
| llada-8b-instruct-hf | plan_387 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.453 | 0.000 | 0.122 | 0.122 | 0.261 | 0.261 | 0.261 | 0.000 | 0.354 | 0.092 | 0.354 | 0.000 |
| llada-8b-instruct-hf | plan_388 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.256 | 0.000 | 0.000 | 0.000 | 0.105 | 0.105 | 0.105 | 0.000 | 0.105 | 0.000 | 0.105 | 0.000 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.361 | 0.000 | 0.000 | 0.000 | 0.404 | 0.045 | 0.404 | 0.000 | 0.404 | 0.000 | 0.404 | 0.000 |
| llada-8b-instruct-hf | plan_390 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.397 | 0.000 | 0.071 | 0.071 | 0.200 | 0.200 | 0.255 | 0.000 | 0.298 | 0.043 | 0.298 | 0.000 |
| llada-8b-instruct-hf | plan_391 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.292 | 0.000 | 0.000 | 0.000 | 0.201 | 0.201 | 0.201 | 0.000 | 0.201 | 0.000 | 0.201 | 0.000 |
| llada-8b-instruct-hf | plan_392 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.451 | 0.000 | 0.147 | 0.147 | 0.280 | 0.280 | 0.280 | 0.000 | 0.402 | 0.121 | 0.402 | 0.000 |
