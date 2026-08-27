# Diffusion Schedule-Selection Benchmark Report

Full model generations: `240`
Counterfactual probe generations: `48`
Arm selections: `336`
Run ID: `diffusion-25f7c5e249c45dee`
Content hash: `25f7c5e249c45deef898b473cd9942e63f1c168d8861beaaea6c24288967e59b`
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
History mutability: `monotonic 240/240, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `prefix`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `counterfactual_micro_probe_v1`
Counterfactual probe mode: `all`
Counterfactual probe policy: `span_tomography_probe_v4`
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
Repair selector: `planning_quality`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.007`
Trajectory task delta vs random: `0.019`
Trajectory wins/ties/losses vs fixed: `12/83/1`
Trajectory wins/ties/losses vs random: `22/71/3`
Oracle generation budget/task: `2.50`
Oracle task score: `0.150`
Oracle headroom vs trajectory: `0.002`
Oracle wins/ties/losses vs trajectory: `4/92/0`
Selector regret vs trajectory: `0.002 over 4/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.008`
Repair task delta vs random: `0.032`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/48/0`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `3/45/0`
Selector regret vs repair: `0.002 over 3/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.270320 | 0.000000 | 0.024122 | - | - |
| random perturbation | repair-covered tasks | 0.246198 | -0.024122 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.278293 | 0.007973 | 0.032095 | 5/43/0 | 16/29/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.142 | 0.392 | 0.205 |
| random | 96 | 1.00 | 0.130 | 0.350 | 0.185 |
| trajectory_selected | 96 | 2.50 | 0.149 | 0.399 | 0.211 |
| repair_selected | 48 | 2.00 | 0.278 | 0.678 | 0.378 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.142 | 0.392 | 0.205 |
| planning | random | 96 | 1.00 | 0.130 | 0.350 | 0.185 |
| planning | trajectory_selected | 96 | 2.50 | 0.149 | 0.399 | 0.211 |
| planning | repair_selected | 48 | 2.00 | 0.278 | 0.678 | 0.378 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_345 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.486 | 0.386 | 266 | True | 1 | 0.900 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_346 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.304 | 0.244 | 288 | True | 4 | 0.636 | True | True | 7.000 | 0.219 | 0.091 | 0.091 |
| llada-8b-instruct-hf | plan_347 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.263 | 0.223 | 278 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.285 | 0.265 | 333 | True | 6 | 0.400 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_349 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.379 | 0.281 | 321 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.304 | 0.244 | 342 | True | 5 | 0.556 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.375 | 0.272 | 331 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_352 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.283 | 0.223 | 332 | True | 1 | 0.900 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_353 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.283 | 0.223 | 288 | True | 1 | 0.909 | True | True | 7.000 | 0.219 | 0.636 | 0.636 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.340 | 0.260 | 322 | True | 2 | 0.778 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.250 | 0.230 | 296 | True | 5 | 0.600 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_356 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.221 | 0.201 | 356 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_357 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 321 | True | 1 | 0.909 | True | True | 7.000 | 0.219 | 0.545 | 0.545 |
| llada-8b-instruct-hf | plan_358 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.283 | 0.223 | 300 | True | 1 | 0.875 | True | True | 7.000 | 0.219 | 0.875 | 0.875 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.364 | 0.324 | 275 | True | 2 | 0.889 | True | True | 7.000 | 0.219 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.200 | 0.180 | 332 | True | 4 | 0.600 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.310 | 0.290 | 341 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_362 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.399 | 0.281 | 283 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_363 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.308 | 0.248 | 354 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_364 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 330 | True | 1 | 1.000 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_365 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 302 | True | 5 | 0.500 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_366 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.220 | 0.180 | 358 | True | 4 | 0.636 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_367 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 291 | True | 3 | 0.778 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_368 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 345 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_369 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 358 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_370 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.301 | 0.201 | 356 | True | 1 | 0.889 | True | True | 7.000 | 0.219 | 0.667 | 0.667 |
| llada-8b-instruct-hf | plan_371 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.243 | 0.223 | 289 | True | 5 | 0.600 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_372 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.283 | 0.223 | 324 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.221 | 0.201 | 332 | True | 3 | 0.727 | True | True | 7.000 | 0.219 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_374 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 301 | True | 1 | 0.889 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_375 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.314 | 0.294 | 357 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.667 | 0.667 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 294 | True | 1 | 0.857 | True | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.275 | 0.235 | 296 | True | 3 | 0.625 | True | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_378 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.282 | 0.282 | 236 | True | 4 | 0.000 | True | True | 7.000 | 0.219 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_379 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.275 | 0.235 | 287 | True | 4 | 0.556 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_380 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 336 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 371 | True | 4 | 0.556 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_382 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.242 | 0.223 | 268 | True | 5 | 0.545 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_383 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.408 | 0.408 | 302 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.283 | 0.223 | 303 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.220 | 0.180 | 321 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_386 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.137 | 0.117 | 98 | True | 3 | 0.667 | True | True | 7.000 | 0.219 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_387 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 349 | True | 1 | 0.900 | True | True | 7.000 | 0.219 | 0.700 | 0.700 |
| llada-8b-instruct-hf | plan_388 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.105 | 0.045 | 60 | True | 5 | 0.556 | True | True | 20.000 | 0.625 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.404 | 0.326 | 238 | True | 8 | 0.300 | True | True | 7.000 | 0.219 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_390 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.255 | 0.235 | 322 | True | 2 | 0.714 | True | True | 7.000 | 0.219 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_391 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.201 | 0.201 | 195 | True | 6 | 0.333 | True | True | 7.000 | 0.219 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_392 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.280 | 0.260 | 371 | True | 2 | 0.818 | True | True | 7.000 | 0.219 | 0.455 | 0.455 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_345 | entropy_32 | origin_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_346 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_347 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_348 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_349 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_350 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_351 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_352 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_353 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_354 | entropy_32 | origin_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_355 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_356 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_357 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_358 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_359 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_360 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_361 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_362 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_363 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_364 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_365 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_366 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_367 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_368 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_369 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_370 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_371 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_372 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_373 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_374 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_375 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.014 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_376 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_377 | entropy_32 | entropy_32 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_378 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_379 | entropy_32 | origin_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_380 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_381 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_382 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_383 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.014 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_384 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_385 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_386 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_387 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_388 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_389 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_390 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_391 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_392 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_345 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.502 | 0.000 | 0.386 | 0.000 | 0.335 | 0.335 | 0.486 | 0.000 | 0.486 | 0.000 | 0.486 | 0.000 |
| llada-8b-instruct-hf | plan_346 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.378 | 0.000 | 0.244 | 0.000 | 0.221 | 0.221 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_347 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.382 | 0.000 | 0.223 | 0.000 | 0.241 | 0.241 | 0.263 | 0.000 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.320 | 0.000 | 0.265 | 0.000 | 0.285 | 0.045 | 0.285 | 0.000 | 0.285 | 0.000 | 0.285 | 0.000 |
| llada-8b-instruct-hf | plan_349 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.463 | 0.000 | 0.281 | 0.000 | 0.379 | 0.375 | 0.379 | 0.000 | 0.379 | 0.000 | 0.379 | 0.000 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.389 | 0.000 | 0.244 | 0.000 | 0.304 | 0.178 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.419 | 0.000 | 0.272 | 0.000 | 0.375 | 0.375 | 0.375 | 0.000 | 0.375 | 0.000 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_352 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.459 | 0.000 | 0.223 | 0.000 | 0.283 | 0.303 | 0.283 | 0.000 | 0.283 | 0.000 | 0.303 | 0.020 |
| llada-8b-instruct-hf | plan_353 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.447 | 0.000 | 0.223 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.436 | 0.000 | 0.260 | 0.000 | 0.340 | 0.340 | 0.340 | 0.000 | 0.340 | 0.000 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.385 | 0.000 | 0.230 | 0.000 | 0.250 | 0.250 | 0.250 | 0.000 | 0.250 | 0.000 | 0.250 | 0.000 |
| llada-8b-instruct-hf | plan_356 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.420 | 0.000 | 0.201 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_357 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.439 | 0.000 | 0.180 | 0.000 | 0.260 | 0.177 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_358 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.450 | 0.000 | 0.223 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.486 | 0.000 | 0.324 | 0.000 | 0.364 | 0.364 | 0.364 | 0.000 | 0.364 | 0.000 | 0.364 | 0.000 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.362 | 0.000 | 0.180 | 0.000 | 0.200 | 0.045 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.421 | 0.000 | 0.290 | 0.000 | 0.310 | 0.310 | 0.310 | 0.000 | 0.310 | 0.000 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_362 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.463 | 0.000 | 0.281 | 0.000 | 0.399 | 0.399 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_363 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.410 | 0.000 | 0.248 | 0.000 | 0.308 | 0.308 | 0.308 | 0.000 | 0.308 | 0.000 | 0.308 | 0.000 |
| llada-8b-instruct-hf | plan_364 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.413 | 0.000 | 0.201 | 0.000 | 0.261 | 0.241 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_365 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.325 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_366 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.332 | 0.000 | 0.180 | 0.000 | 0.220 | 0.045 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_367 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.421 | 0.000 | 0.201 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_368 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.383 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_369 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.310 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_370 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.403 | 0.000 | 0.201 | 0.000 | 0.301 | 0.276 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_371 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.367 | 0.000 | 0.223 | 0.000 | 0.243 | 0.243 | 0.243 | 0.000 | 0.243 | 0.000 | 0.243 | 0.000 |
| llada-8b-instruct-hf | plan_372 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.421 | 0.000 | 0.223 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.393 | 0.000 | 0.201 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_374 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.430 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_375 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.515 | 0.000 | 0.294 | 0.000 | 0.314 | 0.362 | 0.314 | 0.000 | 0.314 | 0.000 | 0.362 | 0.048 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.430 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.388 | 0.000 | 0.235 | 0.000 | 0.275 | 0.275 | 0.275 | 0.000 | 0.275 | 0.000 | 0.275 | 0.000 |
| llada-8b-instruct-hf | plan_378 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.250 | 0.000 | 0.282 | 0.000 | 0.282 | 0.282 | 0.282 | 0.000 | 0.282 | 0.000 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_379 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.371 | 0.000 | 0.235 | 0.000 | 0.275 | 0.263 | 0.275 | 0.000 | 0.275 | 0.000 | 0.275 | 0.000 |
| llada-8b-instruct-hf | plan_380 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.440 | 0.000 | 0.201 | 0.000 | 0.261 | 0.198 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.355 | 0.000 | 0.201 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_382 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.365 | 0.000 | 0.223 | 0.000 | 0.242 | 0.280 | 0.242 | 0.000 | 0.242 | 0.000 | 0.280 | 0.038 |
| llada-8b-instruct-hf | plan_383 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.474 | 0.000 | 0.408 | 0.000 | 0.408 | 0.408 | 0.408 | 0.000 | 0.408 | 0.000 | 0.408 | 0.000 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.401 | 0.000 | 0.223 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.381 | 0.000 | 0.180 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_386 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.315 | 0.000 | 0.117 | 0.000 | 0.065 | 0.065 | 0.137 | 0.000 | 0.137 | 0.000 | 0.137 | 0.000 |
| llada-8b-instruct-hf | plan_387 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.457 | 0.000 | 0.201 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_388 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.256 | 0.000 | 0.045 | 0.000 | 0.105 | 0.105 | 0.105 | 0.000 | 0.105 | 0.000 | 0.105 | 0.000 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.365 | 0.000 | 0.326 | 0.000 | 0.404 | 0.045 | 0.404 | 0.000 | 0.404 | 0.000 | 0.404 | 0.000 |
| llada-8b-instruct-hf | plan_390 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.396 | 0.000 | 0.235 | 0.000 | 0.200 | 0.200 | 0.255 | 0.000 | 0.255 | 0.000 | 0.255 | 0.000 |
| llada-8b-instruct-hf | plan_391 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.291 | 0.000 | 0.201 | 0.000 | 0.201 | 0.201 | 0.201 | 0.000 | 0.201 | 0.000 | 0.201 | 0.000 |
| llada-8b-instruct-hf | plan_392 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.451 | 0.000 | 0.260 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
