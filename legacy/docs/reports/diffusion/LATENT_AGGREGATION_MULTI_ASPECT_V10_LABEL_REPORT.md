# Diffusion Schedule-Selection Benchmark Report

Full model generations: `334`
Counterfactual probe generations: `0`
Arm selections: `336`
Run ID: `diffusion-1b64200d29a31cf4`
Content hash: `1b64200d29a31cf4508ee336c6624f55e101fe0bcf804b7bbc88e2b8698f687a`
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
History mutability: `monotonic 334/334, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.013`
Trajectory task delta vs random: `0.026`
Trajectory wins/ties/losses vs fixed: `17/79/0`
Trajectory wins/ties/losses vs random: `29/66/1`
Oracle generation budget/task: `3.48`
Oracle task score: `0.173`
Oracle headroom vs trajectory: `0.029`
Oracle wins/ties/losses vs trajectory: `31/65/0`
Selector regret vs trajectory: `0.029 over 31/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.069`
Repair task delta vs random: `0.099`
Repair task delta vs trajectory: `0.053`
Repair task delta vs evolved: `0.053`
Repair generation budget delta vs evolved: `1.96`
Repair task delta per extra generation vs evolved: `0.027`
Repair wins/ties/losses vs evolved: `27/21/0`
Oracle headroom vs repair: `0.003`
Oracle wins/ties/losses vs repair: `2/46/0`
Selector regret vs repair: `0.003 over 2/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.253914 | 0.000000 | 0.029192 | - | - |
| random perturbation | repair-covered tasks | 0.224722 | -0.029192 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.323302 | 0.069388 | 0.098580 | 32/16/0 | 34/14/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.131 | 0.325 | 0.180 |
| random | 96 | 1.00 | 0.118 | 0.288 | 0.160 |
| trajectory_selected | 96 | 2.50 | 0.144 | 0.350 | 0.196 |
| repair_selected | 48 | 3.96 | 0.323 | 0.621 | 0.398 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.131 | 0.325 | 0.180 |
| planning | random | 96 | 1.00 | 0.118 | 0.288 | 0.160 |
| planning | trajectory_selected | 96 | 2.50 | 0.144 | 0.350 | 0.196 |
| planning | repair_selected | 48 | 3.96 | 0.323 | 0.621 | 0.398 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_393 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.275 | 0.235 | 299 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_394 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.287 | 0.247 | 366 | True | 0 | 1.000 | True | True | 3.000 | 0.094 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_395 | random_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 301 | True | 6 | 0.615 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_396 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 269 | True | 3 | 0.700 | True | True | 4.000 | 0.125 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_397 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 314 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_398 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 303 | True | 5 | 0.600 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_399 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.379 | 0.281 | 341 | True | 1 | 0.875 | True | True | 3.000 | 0.094 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_400 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.278 | 318 | True | 0 | 1.000 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_401 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 293 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_402 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 344 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_403 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 268 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_404 | random_32 | True | denoise_phase_repairable | False |  | 0.177 | 0.117 | 85 | True | 0 | 1.000 | True | True | 11.000 | 0.344 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_405 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 286 | True | 3 | 0.667 | True | True | 3.000 | 0.094 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_406 | random_32 | True | denoise_phase_repairable | False |  | 0.177 | 0.117 | 103 | True | 5 | 0.583 | True | True | 2.000 | 0.062 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_407 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 328 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_408 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 295 | True | 0 | 1.000 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_409 | random_32 | True | denoise_phase_repairable | False |  | 0.231 | 0.109 | 81 | True | 3 | 0.700 | True | True | 15.000 | 0.469 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_410 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 233 | True | 5 | 0.500 | True | True | 6.000 | 0.188 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_411 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 361 | True | 3 | 0.667 | True | True | 3.000 | 0.094 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_412 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 305 | True | 3 | 0.571 | True | True | 4.000 | 0.125 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_413 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 317 | True | 6 | 0.400 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_414 | random_32 | True | denoise_phase_repairable | False |  | 0.528 | 0.410 | 294 | True | 3 | 0.625 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_415 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 290 | True | 2 | 0.667 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_416 | random_32 | True | denoise_phase_repairable | False |  | 0.259 | 0.239 | 285 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_417 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.263 | 0.223 | 355 | True | 1 | 0.909 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_418 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 304 | True | 1 | 0.917 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_419 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 304 | True | 1 | 0.889 | True | True | 5.000 | 0.156 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_420 | random_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 282 | True | 2 | 0.667 | True | True | 2.000 | 0.062 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_421 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 314 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_422 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.375 | 0.298 | 338 | True | 3 | 0.700 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_423 | random_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 286 | True | 3 | 0.846 | True | True | 5.000 | 0.156 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_424 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 344 | True | 1 | 0.875 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_425 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 279 | True | 3 | 0.625 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_426 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.336 | 0.276 | 367 | True | 3 | 0.625 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_427 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.420 | 0.380 | 363 | True | 4 | 0.700 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_428 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.383 | 0.303 | 272 | True | 4 | 0.556 | True | True | 4.000 | 0.125 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_429 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.458 | 0.340 | 357 | True | 3 | 0.778 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_430 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.300 | 0.260 | 268 | True | 4 | 0.667 | True | True | 3.000 | 0.094 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_431 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.279 | 0.239 | 301 | True | 1 | 0.875 | True | True | 4.000 | 0.125 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_432 | low_confidence_32 | False | no_repairable_denoise_skeleton | False |  | 0.045 | 0.045 | 1 | True | 10 | 0.000 | True | False | none | none | none | 0.000 |
| llada-8b-instruct-hf | plan_433 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.398 | 0.398 | 266 | True | 1 | 0.833 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_434 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 248 | True | 6 | 0.455 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_435 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.217 | 311 | True | 1 | 0.857 | True | True | 4.000 | 0.125 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_436 | random_32 | True | denoise_phase_repairable | False |  | 0.354 | 0.294 | 282 | True | 2 | 0.750 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_437 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 298 | True | 0 | 1.000 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_438 | random_32 | True | denoise_phase_repairable | False |  | 0.198 | 0.138 | 91 | True | 1 | 0.857 | True | True | 15.000 | 0.469 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_439 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.201 | 0.201 | 339 | True | 2 | 0.818 | True | True | 4.000 | 0.125 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_440 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 361 | True | 4 | 0.714 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 47 | 27 | low_confidence_32,random_32 | final | 26.7 | 0.979 | 0.021 | 0.000 | 0.022 | 0.022 | 0.052 | 0.052 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 27/18/2 | 0.327 | 0.650 | 0.408 |
| history_prefix_25_repair | 47 | 0 | low_confidence_32,random_32 | history | 48.1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.006 | -0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/27/11 | 0.266 | 0.635 | 0.358 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_393 | True | low_confidence_32 | 2.027 | 0.707 | 1.000 | 0.000 | 0.417 | False | This will allow us to verify the effectiveness of the v9 complement-packet policy on fr... |
| llada-8b-instruct-hf | plan_394 | True | low_confidence_32 | 1.825 | 0.887 | 0.429 | 0.000 | 0.571 | False | This can be achieved by setting the sequencing boundary at the beginning of the anchor... |
| llada-8b-instruct-hf | plan_395 | False | random_32 | 2.057 | 0.820 | 1.000 | 0.000 | 0.000 | False | Implement masking to hide sensitive information. |
| llada-8b-instruct-hf | plan_395 | False | random_32 | 2.047 | 0.820 | 1.000 | 0.000 | 0.077 | False | Anonymize reports to protect identities. |
| llada-8b-instruct-hf | plan_395 | False | random_32 | 2.797 | 0.820 | 1.000 | 0.000 | 0.077 | False | Regularly audit reports for anomalies. |
| llada-8b-instruct-hf | plan_396 | True | low_confidence_32 | 2.065 | 0.865 | 1.000 | 0.000 | 0.100 | False | Capture the JSON output. |
| llada-8b-instruct-hf | plan_396 | True | low_confidence_32 | 2.026 | 0.790 | 1.000 | 0.000 | 0.100 | False | Analyze the JSON structure. |
| llada-8b-instruct-hf | plan_396 | True | low_confidence_32 | 2.788 | 0.790 | 1.000 | 0.000 | 0.000 | False | Document any discrepancies or anomalies. |
| llada-8b-instruct-hf | plan_397 | False | low_confidence_32 | 2.083 | 0.905 | 1.000 | 0.000 | 0.700 | False | However, without further information about the purpose of the packet and the specific c... |
| llada-8b-instruct-hf | plan_398 | False | low_confidence_32 | 2.864 | 1.000 | 1.000 | 0.000 | 0.100 | False | This can be done by maintaining a set of expected roles and comparing discrepancies. |
| llada-8b-instruct-hf | plan_399 | True | low_confidence_32 | 2.584 | 0.925 | 1.000 | 0.000 | 0.125 | False | Analyze the impact of the contradiction on the system and functionality. |
| llada-8b-instruct-hf | plan_399 | True | low_confidence_32 | 1.991 | 0.693 | 1.000 | 0.000 | 0.625 | False | Prior a resolution plan that outlines the necessary changes to either the v10 complemen... |
| llada-8b-instruct-hf | plan_400 | True | low_confidence_32 | 1.923 | 0.925 | 0.556 | 0.000 | 0.444 | False | This record would document the number packet samples taken and the reason for the abste... |
| llada-8b-instruct-hf | plan_401 | False | low_confidence_32 | 2.058 | 0.714 | 1.000 | 0.000 | 0.333 | False | This will ensure that the score improvement of the v10 aggregate is solely due to its o... |
| llada-8b-instruct-hf | plan_402 | True | low_confidence_32 | 2.877 | 1.000 | 1.000 | 0.000 | 0.000 | False | Further validation or additional experiments may be needed to confirm the robustness of... |
| llada-8b-instruct-hf | plan_403 | True | low_confidence_32 | 1.378 | 0.835 | 1.000 | 0.000 | 0.200 | False | Identify the v10 theme bucket. |
| llada-8b-instruct-hf | plan_403 | True | low_confidence_32 | 1.423 | 0.910 | 1.000 | 0.000 | 0.200 | False | Determine the robustness of the bucket. |
| llada-8b-instruct-hf | plan_403 | True | low_confidence_32 | 2.161 | 0.910 | 1.000 | 0.000 | 0.300 | False | Develop a plan to report the robustness of the bucket. |
| llada-8b-instruct-hf | plan_404 | False | random_32 | 1.547 | 0.000 | 0.600 | 0.000 | 0.400 | False | Plan high-leverage auditing. |
| llada-8b-instruct-hf | plan_405 | False | low_confidence_32 | 2.208 | 1.000 | 1.000 | 0.000 | 0.444 | False | This could could be a way to to reduce the size of the source or to focus focus on the... |
| llada-8b-instruct-hf | plan_406 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The family claim is that cross-latent perturbation continues to contribute unique aspec... |
| llada-8b-instruct-hf | plan_407 | True | low_confidence_32 | 1.436 | 1.000 | 1.000 | 0.000 | 0.333 | False | If this task significantly enhances the model's performance or adds substantial value,... |
| llada-8b-instruct-hf | plan_407 | True | low_confidence_32 | 2.200 | 1.000 | 1.000 | 0.000 | 0.222 | False | Otherwise, if the impact is minimal or the task is redundant, you may want to review or... |
| llada-8b-instruct-hf | plan_408 | True | low_confidence_32 | 2.813 | 1.000 | 0.889 | 0.000 | 0.111 | False | We will need to calculate the cost of each resource used and sum them up to determine t... |
| llada-8b-instruct-hf | plan_409 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Claim the boundary between the equal-budget best-of baseline and v10 aggregation. |
| llada-8b-instruct-hf | plan_410 | False | low_confidence_32 | 1.354 | 0.780 | 1.000 | 0.000 | 0.200 | False | A list of anchors ( each anchor a list of aspects) 2. |
| llada-8b-instruct-hf | plan_410 | False | low_confidence_32 | 1.423 | 0.955 | 1.000 | 0.000 | 0.200 | False | A list of missing aspects (each aspect associated with a label) Output: 1. |
| llada-8b-instruct-hf | plan_410 | False | low_confidence_32 | 2.207 | 1.000 | 1.000 | 0.000 | 0.200 | False | A list of missing aspects. |
| llada-8b-instruct-hf | plan_411 | True | low_confidence_32 | 1.411 | 0.925 | 1.000 | 0.000 | 0.222 | False | This involves identifying the specific sources or references that should be included in... |
| llada-8b-instruct-hf | plan_411 | True | low_confidence_32 | 2.163 | 1.000 | 1.000 | 0.000 | 0.444 | False | It is important to clearly define the source requirements to avoid any ambiguity or con... |
| llada-8b-instruct-hf | plan_412 | False | low_confidence_32 | 3.274 | 1.000 | 1.000 | 0.000 | 0.429 | False | Additionally, the table should also list any concepts in the old ontology that are not... |
| llada-8b-instruct-hf | plan_413 | True | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | When a anchor is received, increment its count in a data structure (e.g., a hash table). |
| llada-8b-instruct-hf | plan_413 | True | low_confidence_32 | 2.186 | 0.968 | 1.000 | 0.000 | 0.200 | False | Trigger an alert if an anchor's count exceeds a predefined threshold, indicating the pr... |
| llada-8b-instruct-hf | plan_414 | True | random_32 | 1.959 | 1.000 | 1.000 | 0.000 | 0.250 | False | Implement a fix to to improve the complement coverage. |
| llada-8b-instruct-hf | plan_414 | True | random_32 | 1.823 | 0.758 | 1.000 | 0.000 | 0.250 | False | Conduct additional testing to verify that the complement coverage has been improved. |
| llada-8b-instruct-hf | plan_414 | True | random_32 | 3.283 | 0.790 | 1.000 | 0.000 | 0.000 | False | Document the findings and fixes for future reference. |
| llada-8b-instruct-hf | plan_415 | True | low_confidence_32 | 2.078 | 0.910 | 1.000 | 0.000 | 0.167 | False | Assess the coverage of the result. |
| llada-8b-instruct-hf | plan_415 | True | low_confidence_32 | 1.398 | 0.910 | 1.000 | 0.000 | 0.333 | False | Evaluate the conditional lift of the result. |
| llada-8b-instruct-hf | plan_415 | True | low_confidence_32 | 2.000 | 0.708 | 1.000 | 0.000 | 0.667 | False | Compare the coverage and conditional lift to established standards or benchmarks in the... |
| llada-8b-instruct-hf | plan_416 | True | random_32 | 2.577 | 0.865 | 1.000 | 0.000 | 0.000 | False | Create a set of diverse, unrelated tasks. |
| llada-8b-instruct-hf | plan_416 | True | random_32 | 2.577 | 0.865 | 1.000 | 0.000 | 0.000 | False | Evaluate the model's performance on these tasks. |
| llada-8b-instruct-hf | plan_416 | True | random_32 | 2.065 | 0.696 | 1.000 | 0.000 | 0.222 | False | Analyze the results to assess transferability and potential risks. |
| llada-8b-instruct-hf | plan_417 | True | low_confidence_32 | 2.099 | 0.936 | 1.000 | 0.000 | 0.636 | False | This involves designing the proof object to accommodate the structure and constraints o... |
| llada-8b-instruct-hf | plan_418 | False | low_confidence_32 | 2.044 | 0.801 | 1.000 | 0.000 | 0.000 | False | Define the objectives and constraints. |
| llada-8b-instruct-hf | plan_418 | False | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Simulate various scenarios using historical data. |
| llada-8b-instruct-hf | plan_418 | False | low_confidence_32 | 2.817 | 0.865 | 1.000 | 0.000 | 0.083 | False | Select the best policy based on results. |
| llada-8b-instruct-hf | plan_419 | True | low_confidence_32 | 2.144 | 0.905 | 1.000 | 0.000 | 0.333 | False | Once you have the clause in detail, you can develop a reporting strategy that addresses... |
| llada-8b-instruct-hf | plan_420 | True | random_32 | 1.409 | 0.910 | 1.000 | 0.000 | 0.333 | False | Run the v10 replay for each of these realizations. |
| llada-8b-instruct-hf | plan_420 | True | random_32 | 2.047 | 0.741 | 1.000 | 0.000 | 0.500 | False | Analyze the differences in the replay's output to assess the realization sensitivity. |
| llada-8b-instruct-hf | plan_421 | False | low_confidence_32 | 2.775 | 0.790 | 1.000 | 0.000 | 0.111 | False | Document the differences and distinctions in the report. |
| llada-8b-instruct-hf | plan_422 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The cautious claim would be: "The v10 run exhibits numerous packet-shape failures, but... |
| llada-8b-instruct-hf | plan_423 | False | random_32 | 1.302 | 0.662 | 1.000 | 0.000 | 0.231 | False | Determine the score of the packet source. |
| llada-8b-instruct-hf | plan_423 | False | random_32 | 1.368 | 0.790 | 1.000 | 0.000 | 0.154 | False | Analyze the clauses and the score ratio. |
| llada-8b-instruct-hf | plan_423 | False | random_32 | 2.157 | 0.865 | 1.000 | 0.000 | 0.154 | False | Report findings to determine the balance between quality and compliance. |
| llada-8b-instruct-hf | plan_424 | True | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | This includes clear instructions for setup, data acquisition, and analysis, and any nec... |
| llada-8b-instruct-hf | plan_424 | True | low_confidence_32 | 2.883 | 1.000 | 1.000 | 0.000 | 0.000 | False | Regularly review and update the documentation to maintain consistency and reproducibility. |
| llada-8b-instruct-hf | plan_425 | True | low_confidence_32 | 2.499 | 0.726 | 1.000 | 0.000 | 0.125 | False | Assess the impact of each anchor on the result. |
| llada-8b-instruct-hf | plan_425 | True | low_confidence_32 | 2.542 | 0.820 | 1.000 | 0.000 | 0.125 | False | Determine the most dominant anchor. |
| llada-8b-instruct-hf | plan_425 | True | low_confidence_32 | 2.620 | 0.820 | 1.000 | 0.000 | 0.250 | False | Develop strategies to mitigate anchor dominance. |
| llada-8b-instruct-hf | plan_426 | True | low_confidence_32 | 1.939 | 1.000 | 1.000 | 0.000 | 0.250 | False | This involves refining the packet sources to be more specific and targeted, ensuring th... |
| llada-8b-instruct-hf | plan_426 | True | low_confidence_32 | 2.650 | 0.925 | 1.000 | 0.000 | 0.250 | False | We will work on identifying the specific requirements and adjusting the packet sources... |
| llada-8b-instruct-hf | plan_427 | True | low_confidence_32 | 1.370 | 0.917 | 1.000 | 0.000 | 0.400 | False | This test should include a baseline measurement and a subsequent measurement after impl... |
| llada-8b-instruct-hf | plan_427 | True | low_confidence_32 | 2.141 | 0.893 | 1.000 | 0.000 | 0.200 | False | The results should be compared to determine the difference in reliability between the b... |
| llada-8b-instruct-hf | plan_428 | False | low_confidence_32 | 2.002 | 0.745 | 1.000 | 0.000 | 0.111 | False | Verify owner identity. |
| llada-8b-instruct-hf | plan_428 | False | low_confidence_32 | 2.024 | 0.788 | 1.000 | 0.000 | 0.111 | False | Check for ownership and consistency. |
| llada-8b-instruct-hf | plan_428 | False | low_confidence_32 | 2.725 | 0.650 | 1.000 | 0.000 | 0.000 | False | Document findings and suggest corrections if necessary. |
| llada-8b-instruct-hf | plan_429 | True | low_confidence_32 | 1.334 | 0.808 | 1.000 | 0.000 | 0.333 | False | This involves setting up a controlled environment where the rollback process is initiat... |
| llada-8b-instruct-hf | plan_429 | True | low_confidence_32 | 2.164 | 1.000 | 1.000 | 0.000 | 0.333 | False | The goal is to ensure that the rollback functionality works as expected in the absence... |
| llada-8b-instruct-hf | plan_430 | False | low_confidence_32 | 2.827 | 0.865 | 1.000 | 0.000 | 0.000 | False | Report the result and suggest corrections if necessary. |
| llada-8b-instruct-hf | plan_431 | True | low_confidence_32 | 1.455 | 0.988 | 1.000 | 0.000 | 0.250 | False | Determine the new scope and and the changes made to the original scope. |
| llada-8b-instruct-hf | plan_431 | True | low_confidence_32 | 2.002 | 0.654 | 1.000 | 0.000 | 0.375 | False | Document the new scope, update the task plan to reflect the new scope, and validate the... |
| llada-8b-instruct-hf | plan_433 | False | low_confidence_32 | 2.519 | 0.748 | 1.000 | 0.000 | 0.000 | False | Gather relevant data and logs. |
| llada-8b-instruct-hf | plan_433 | False | low_confidence_32 | 2.414 | 0.555 | 1.000 | 0.000 | 0.000 | False | Analyze the root cause. |
| llada-8b-instruct-hf | plan_433 | False | low_confidence_32 | 2.518 | 0.745 | 1.000 | 0.000 | 0.000 | False | Document the findings and recommendations. |
| llada-8b-instruct-hf | plan_434 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To ensure a fair comparison, normalize the number of rows in each source family by a co... |
| llada-8b-instruct-hf | plan_435 | True | low_confidence_32 | 2.064 | 0.856 | 1.000 | 0.000 | 0.714 | False | This means that the transfer should be stopped as soon as it fails the safety condition... |
| llada-8b-instruct-hf | plan_436 | True | random_32 | 1.280 | 0.683 | 1.000 | 0.000 | 0.375 | False | Identify the token perturbation insight. |
| llada-8b-instruct-hf | plan_436 | True | random_32 | 2.002 | 0.750 | 1.000 | 0.000 | 0.125 | False | Implement a mechanism to perturb tokens. |
| llada-8b-instruct-hf | plan_436 | True | random_32 | 1.397 | 0.865 | 1.000 | 0.000 | 0.250 | False | Regulate the perturbation to avoid overclaiming. |
| llada-8b-instruct-hf | plan_437 | False | low_confidence_32 | 1.839 | 0.787 | 0.500 | 0.000 | 0.500 | False | The goal is to ensure that the v10 aggregate is above the best candidate. |
| llada-8b-instruct-hf | plan_438 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Plan to plan threshold discipline for deterministic extractor thresholds in the v10 res... |
| llada-8b-instruct-hf | plan_439 | True | low_confidence_32 | 2.549 | 1.000 | 1.000 | 0.000 | 0.182 | False | Could you provide additional context or information about the proof object and its comp... |
| llada-8b-instruct-hf | plan_439 | True | low_confidence_32 | 3.976 | 1.000 | 1.000 | 0.000 | 0.091 | False | This will help me create a precise and operational plan. |
| llada-8b-instruct-hf | plan_440 | True | low_confidence_32 | 2.088 | 0.925 | 1.000 | 0.000 | 0.000 | False | This will involve analyzing various datasets, identifying patterns, and evaluating thei... |
| llada-8b-instruct-hf | plan_440 | True | low_confidence_32 | 1.715 | 0.034 | 1.000 | 0.000 | 0.500 | False | By combining different methods, I aim to develop a more robust and versatile model, ess... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_393 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_394 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_395 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_396 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_397 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_398 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_399 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_400 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_401 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_402 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_403 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_404 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_405 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_406 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_407 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_408 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_409 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_410 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_411 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_412 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.113 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.142 | 0.000 | 0.000 | 0.000 | 0.142 | 0.000 |
| dream-7b-instruct-hf | plan_413 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_414 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_415 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_416 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_417 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_418 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_419 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_420 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_421 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_422 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_423 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_424 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_425 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.130 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_426 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_427 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_428 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_429 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_430 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_431 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_432 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_433 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_434 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_435 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_436 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_437 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_438 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_439 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_440 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_393 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.419 | 0.000 | 0.144 | 0.144 | 0.275 | 0.241 | 0.275 | 0.000 | 0.375 | 0.100 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_394 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.473 | 0.000 | 0.048 | 0.048 | 0.287 | 0.212 | 0.287 | 0.000 | 0.309 | 0.021 | 0.309 | 0.000 |
| llada-8b-instruct-hf | plan_395 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.338 | 0.000 | 0.000 | 0.000 | 0.220 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_396 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.389 | 0.000 | 0.073 | 0.073 | 0.221 | 0.137 | 0.221 | 0.000 | 0.280 | 0.059 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_397 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.428 | 0.000 | 0.000 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.327 | 0.107 |
| llada-8b-instruct-hf | plan_398 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.364 | 0.000 | 0.000 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.261 | 0.020 |
| llada-8b-instruct-hf | plan_399 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.472 | 0.000 | 0.259 | 0.259 | 0.379 | 0.379 | 0.379 | 0.000 | 0.582 | 0.203 | 0.582 | 0.000 |
| llada-8b-instruct-hf | plan_400 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.520 | 0.000 | 0.051 | 0.051 | 0.318 | 0.066 | 0.318 | 0.000 | 0.339 | 0.021 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_401 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.381 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_402 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.446 | 0.000 | 0.061 | 0.061 | 0.241 | 0.241 | 0.241 | 0.000 | 0.279 | 0.037 | 0.279 | 0.000 |
| llada-8b-instruct-hf | plan_403 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.435 | 0.000 | 0.180 | 0.180 | 0.240 | 0.177 | 0.240 | 0.000 | 0.379 | 0.139 | 0.379 | 0.000 |
| llada-8b-instruct-hf | plan_404 | low_confidence_32 | random_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.381 | 0.000 | 0.000 | 0.000 | 0.045 | 0.177 | 0.177 | 0.000 | 0.177 | 0.000 | 0.177 | 0.000 |
| llada-8b-instruct-hf | plan_405 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.310 | 0.000 | 0.000 | 0.000 | 0.200 | 0.065 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_406 | low_confidence_32 | random_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.298 | 0.000 | 0.000 | 0.000 | 0.045 | 0.177 | 0.177 | 0.000 | 0.177 | 0.000 | 0.177 | 0.000 |
| llada-8b-instruct-hf | plan_407 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.390 | 0.000 | 0.205 | 0.205 | 0.261 | 0.261 | 0.261 | 0.000 | 0.462 | 0.200 | 0.462 | 0.000 |
| llada-8b-instruct-hf | plan_408 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.468 | 0.000 | 0.103 | 0.103 | 0.261 | 0.260 | 0.261 | 0.000 | 0.336 | 0.075 | 0.336 | 0.000 |
| llada-8b-instruct-hf | plan_409 | low_confidence_32 | random_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.309 | 0.000 | 0.000 | 0.000 | 0.045 | 0.231 | 0.231 | 0.000 | 0.231 | 0.000 | 0.231 | 0.000 |
| llada-8b-instruct-hf | plan_410 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.330 | 0.000 | 0.000 | 0.000 | 0.200 | 0.200 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_411 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.376 | 0.000 | 0.202 | 0.202 | 0.240 | 0.137 | 0.240 | 0.000 | 0.360 | 0.120 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_412 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.364 | 0.000 | 0.000 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_413 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.317 | 0.000 | 0.043 | 0.043 | 0.261 | 0.261 | 0.261 | 0.000 | 0.282 | 0.021 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_414 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.460 | 0.000 | 0.069 | 0.069 | 0.474 | 0.474 | 0.528 | 0.000 | 0.544 | 0.016 | 0.544 | 0.000 |
| llada-8b-instruct-hf | plan_415 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.364 | 0.000 | 0.175 | 0.175 | 0.261 | 0.261 | 0.261 | 0.000 | 0.420 | 0.159 | 0.420 | 0.000 |
| llada-8b-instruct-hf | plan_416 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.379 | 0.000 | 0.047 | 0.047 | 0.258 | 0.258 | 0.259 | 0.000 | 0.300 | 0.041 | 0.300 | 0.000 |
| llada-8b-instruct-hf | plan_417 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.455 | 0.000 | 0.161 | 0.161 | 0.263 | 0.157 | 0.263 | 0.000 | 0.380 | 0.118 | 0.380 | 0.000 |
| llada-8b-instruct-hf | plan_418 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.472 | 0.000 | 0.000 | 0.000 | 0.304 | 0.200 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_419 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.431 | 0.000 | 0.203 | 0.203 | 0.260 | 0.260 | 0.260 | 0.000 | 0.400 | 0.140 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_420 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.389 | 0.000 | 0.067 | 0.067 | 0.221 | 0.241 | 0.241 | 0.000 | 0.324 | 0.083 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_421 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.375 | 0.000 | 0.000 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_422 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.428 | 0.000 | 0.000 | 0.000 | 0.375 | 0.219 | 0.375 | 0.000 | 0.375 | 0.000 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_423 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.408 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_424 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.421 | 0.000 | 0.204 | 0.204 | 0.200 | 0.200 | 0.200 | 0.000 | 0.340 | 0.140 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_425 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.364 | 0.000 | 0.148 | 0.148 | 0.220 | 0.045 | 0.220 | 0.000 | 0.358 | 0.138 | 0.358 | 0.000 |
| llada-8b-instruct-hf | plan_426 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.386 | 0.000 | 0.116 | 0.116 | 0.336 | 0.336 | 0.336 | 0.000 | 0.391 | 0.055 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_427 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.484 | 0.000 | 0.068 | 0.068 | 0.420 | 0.237 | 0.420 | 0.000 | 0.439 | 0.020 | 0.439 | 0.000 |
| llada-8b-instruct-hf | plan_428 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.408 | 0.000 | 0.000 | 0.000 | 0.383 | 0.342 | 0.383 | 0.000 | 0.383 | 0.000 | 0.383 | 0.000 |
| llada-8b-instruct-hf | plan_429 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.495 | 0.000 | 0.068 | 0.068 | 0.458 | 0.458 | 0.458 | 0.000 | 0.479 | 0.021 | 0.479 | 0.000 |
| llada-8b-instruct-hf | plan_430 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.406 | 0.000 | 0.000 | 0.000 | 0.300 | 0.300 | 0.300 | 0.000 | 0.300 | 0.000 | 0.300 | 0.000 |
| llada-8b-instruct-hf | plan_431 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.419 | 0.000 | 0.143 | 0.143 | 0.279 | 0.279 | 0.279 | 0.000 | 0.380 | 0.101 | 0.380 | 0.000 |
| llada-8b-instruct-hf | plan_432 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_433 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.519 | 0.000 | 0.000 | 0.000 | 0.398 | 0.298 | 0.398 | 0.000 | 0.398 | 0.000 | 0.398 | 0.000 |
| llada-8b-instruct-hf | plan_434 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.319 | 0.000 | 0.000 | 0.000 | 0.240 | 0.177 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_435 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.441 | 0.000 | 0.183 | 0.183 | 0.315 | 0.137 | 0.315 | 0.000 | 0.454 | 0.139 | 0.454 | 0.000 |
| llada-8b-instruct-hf | plan_436 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.431 | 0.000 | 0.137 | 0.137 | 0.263 | 0.263 | 0.354 | 0.000 | 0.443 | 0.089 | 0.443 | 0.000 |
| llada-8b-instruct-hf | plan_437 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.463 | 0.000 | 0.000 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_438 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.352 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.198 | 0.000 | 0.198 | 0.000 | 0.198 | 0.000 |
| llada-8b-instruct-hf | plan_439 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.429 | 0.000 | 0.205 | 0.205 | 0.201 | 0.201 | 0.201 | 0.000 | 0.382 | 0.180 | 0.382 | 0.000 |
| llada-8b-instruct-hf | plan_440 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.375 | 0.000 | 0.178 | 0.178 | 0.220 | 0.200 | 0.220 | 0.000 | 0.339 | 0.119 | 0.339 | 0.000 |
