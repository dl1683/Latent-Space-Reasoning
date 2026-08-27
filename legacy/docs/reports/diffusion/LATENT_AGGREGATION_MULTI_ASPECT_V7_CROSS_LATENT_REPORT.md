# Diffusion Schedule-Selection Benchmark Report

Full model generations: `336`
Counterfactual probe generations: `0`
Arm selections: `192`
Run ID: `diffusion-98509fda756dab06`
Content hash: `98509fda756dab06ecb60ccff1620c6ba96acaa6ff381f91147f9509ad89004d`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `inherit`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `8`
Exact verifier revision: `False`
History mutability: `monotonic 240/336, changes 0, remasks 1300, rewrites 286, mask increases 192`
History repairs included: `False`
Repair pack: `prefix`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `always`
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
Repair selector: `planning_quality`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.008`
Trajectory task delta vs random: `0.032`
Trajectory wins/ties/losses vs fixed: `5/43/0`
Trajectory wins/ties/losses vs random: `16/29/3`
Oracle generation budget/task: `7.00`
Oracle task score: `0.306`
Oracle headroom vs trajectory: `0.027`
Oracle wins/ties/losses vs trajectory: `27/21/0`
Selector regret vs trajectory: `0.027 over 27/48 improvable`
Evolved task delta vs fixed: `0.018`
Evolved task delta vs random: `0.042`
Evolved task delta vs trajectory: `0.010`
Evolved wins/ties/losses vs fixed: `19/24/5`
Evolved wins/ties/losses vs random: `27/15/6`
Evolved wins/ties/losses vs trajectory: `14/28/6`
Oracle headroom vs evolved: `0.017`
Oracle wins/ties/losses vs evolved: `21/27/0`
Selector regret vs evolved: `0.017 over 21/48 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.270 | 0.677 | 0.372 |
| random | 48 | 1.00 | 0.246 | 0.613 | 0.338 |
| trajectory_selected | 48 | 2.00 | 0.278 | 0.678 | 0.378 |
| evolved | 48 | 7.00 | 0.289 | 0.675 | 0.385 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.270 | 0.677 | 0.372 |
| planning | random | 48 | 1.00 | 0.246 | 0.613 | 0.338 |
| planning | trajectory_selected | 48 | 2.00 | 0.278 | 0.678 | 0.378 |
| planning | evolved | 48 | 7.00 | 0.289 | 0.675 | 0.385 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_345 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.502 | 0.502 | 0.000 | 0.335 | 0.335 | 0.486 | 0.486 | 0.151 | 0.151 | 0.000 | 0.486 | 0.000 |
| llada-8b-instruct-hf | plan_346 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.378 | 0.398 | 0.020 | 0.221 | 0.221 | 0.304 | 0.241 | 0.083 | 0.020 | -0.063 | 0.304 | 0.063 |
| llada-8b-instruct-hf | plan_347 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.382 | 0.382 | 0.000 | 0.241 | 0.241 | 0.263 | 0.263 | 0.021 | 0.021 | 0.000 | 0.299 | 0.036 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.320 | 0.320 | 0.000 | 0.285 | 0.045 | 0.285 | 0.285 | 0.000 | 0.000 | 0.000 | 0.285 | 0.000 |
| llada-8b-instruct-hf | plan_349 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.463 | 0.505 | 0.042 | 0.379 | 0.375 | 0.379 | 0.436 | 0.000 | 0.057 | 0.057 | 0.458 | 0.021 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.389 | 0.389 | 0.000 | 0.304 | 0.178 | 0.304 | 0.304 | 0.000 | 0.000 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.419 | 0.419 | 0.000 | 0.375 | 0.375 | 0.375 | 0.375 | 0.000 | 0.000 | 0.000 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_352 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.459 | 0.459 | 0.000 | 0.283 | 0.303 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.303 | 0.020 |
| llada-8b-instruct-hf | plan_353 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.447 | 0.447 | 0.000 | 0.283 | 0.283 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.436 | 0.436 | 0.000 | 0.340 | 0.340 | 0.340 | 0.340 | 0.000 | 0.000 | 0.000 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.385 | 0.414 | 0.029 | 0.250 | 0.250 | 0.250 | 0.309 | 0.000 | 0.059 | 0.059 | 0.309 | 0.000 |
| llada-8b-instruct-hf | plan_356 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.420 | 0.483 | 0.063 | 0.221 | 0.221 | 0.221 | 0.264 | 0.000 | 0.043 | 0.043 | 0.264 | 0.000 |
| llada-8b-instruct-hf | plan_357 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.439 | 0.439 | 0.000 | 0.260 | 0.177 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.301 | 0.041 |
| llada-8b-instruct-hf | plan_358 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.450 | 0.472 | 0.022 | 0.283 | 0.283 | 0.283 | 0.304 | 0.000 | 0.021 | 0.021 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.486 | 0.486 | 0.000 | 0.364 | 0.364 | 0.364 | 0.364 | 0.000 | 0.000 | 0.000 | 0.364 | 0.000 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.362 | 0.362 | 0.000 | 0.200 | 0.045 | 0.200 | 0.200 | 0.000 | 0.000 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.421 | 0.421 | 0.000 | 0.310 | 0.310 | 0.310 | 0.310 | 0.000 | 0.000 | 0.000 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_362 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.463 | 0.508 | 0.045 | 0.399 | 0.399 | 0.399 | 0.458 | 0.000 | 0.059 | 0.059 | 0.458 | 0.000 |
| llada-8b-instruct-hf | plan_363 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.410 | 0.410 | 0.000 | 0.308 | 0.308 | 0.308 | 0.308 | 0.000 | 0.000 | 0.000 | 0.375 | 0.068 |
| llada-8b-instruct-hf | plan_364 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.413 | 0.440 | 0.027 | 0.261 | 0.241 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.283 | 0.021 |
| llada-8b-instruct-hf | plan_365 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.325 | 0.421 | 0.097 | 0.240 | 0.240 | 0.240 | 0.261 | 0.000 | 0.021 | 0.021 | 0.349 | 0.087 |
| llada-8b-instruct-hf | plan_366 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.332 | 0.388 | 0.055 | 0.220 | 0.045 | 0.220 | 0.220 | 0.000 | 0.000 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_367 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.421 | 0.450 | 0.029 | 0.261 | 0.261 | 0.261 | 0.311 | 0.000 | 0.050 | 0.050 | 0.311 | 0.000 |
| llada-8b-instruct-hf | plan_368 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.383 | 0.416 | 0.033 | 0.241 | 0.241 | 0.241 | 0.320 | 0.000 | 0.079 | 0.079 | 0.320 | 0.000 |
| llada-8b-instruct-hf | plan_369 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.310 | 0.382 | 0.072 | 0.240 | 0.240 | 0.240 | 0.240 | 0.000 | 0.000 | 0.000 | 0.282 | 0.042 |
| llada-8b-instruct-hf | plan_370 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.403 | 0.456 | 0.052 | 0.301 | 0.276 | 0.301 | 0.301 | 0.000 | 0.000 | 0.000 | 0.344 | 0.043 |
| llada-8b-instruct-hf | plan_371 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.367 | 0.409 | 0.042 | 0.243 | 0.243 | 0.243 | 0.240 | 0.000 | -0.003 | -0.003 | 0.304 | 0.064 |
| llada-8b-instruct-hf | plan_372 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.421 | 0.462 | 0.041 | 0.283 | 0.283 | 0.283 | 0.304 | 0.000 | 0.021 | 0.021 | 0.325 | 0.021 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.393 | 0.437 | 0.044 | 0.221 | 0.221 | 0.221 | 0.243 | 0.000 | 0.021 | 0.021 | 0.243 | 0.000 |
| llada-8b-instruct-hf | plan_374 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.430 | 0.430 | 0.000 | 0.240 | 0.240 | 0.240 | 0.240 | 0.000 | 0.000 | 0.000 | 0.281 | 0.041 |
| llada-8b-instruct-hf | plan_375 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.515 | 0.515 | 0.000 | 0.314 | 0.362 | 0.314 | 0.314 | 0.000 | 0.000 | 0.000 | 0.362 | 0.048 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.430 | 0.430 | 0.000 | 0.240 | 0.240 | 0.240 | 0.240 | 0.000 | 0.000 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.388 | 0.410 | 0.021 | 0.275 | 0.275 | 0.275 | 0.261 | 0.000 | -0.014 | -0.014 | 0.275 | 0.014 |
| llada-8b-instruct-hf | plan_378 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.250 | 0.337 | 0.087 | 0.282 | 0.282 | 0.282 | 0.241 | 0.000 | -0.040 | -0.040 | 0.319 | 0.078 |
| llada-8b-instruct-hf | plan_379 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.371 | 0.403 | 0.033 | 0.275 | 0.263 | 0.275 | 0.296 | 0.000 | 0.021 | 0.021 | 0.296 | 0.000 |
| llada-8b-instruct-hf | plan_380 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.440 | 0.482 | 0.041 | 0.261 | 0.198 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.355 | 0.355 | 0.000 | 0.261 | 0.261 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_382 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.365 | 0.387 | 0.022 | 0.242 | 0.280 | 0.242 | 0.280 | 0.000 | 0.038 | 0.038 | 0.284 | 0.004 |
| llada-8b-instruct-hf | plan_383 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.474 | 0.474 | 0.000 | 0.408 | 0.408 | 0.408 | 0.408 | 0.000 | 0.000 | 0.000 | 0.408 | 0.000 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.401 | 0.401 | 0.000 | 0.283 | 0.283 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.381 | 0.381 | 0.000 | 0.220 | 0.220 | 0.220 | 0.220 | 0.000 | 0.000 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_386 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.315 | 0.315 | 0.000 | 0.065 | 0.065 | 0.137 | 0.137 | 0.072 | 0.072 | 0.000 | 0.158 | 0.021 |
| llada-8b-instruct-hf | plan_387 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.457 | 0.457 | 0.000 | 0.261 | 0.261 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.281 | 0.020 |
| llada-8b-instruct-hf | plan_388 | low_confidence_32 | random_32 | random_32 | evolved_revision_random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.256 | 0.310 | 0.054 | 0.105 | 0.105 | 0.105 | 0.177 | 0.000 | 0.072 | 0.072 | 0.177 | 0.000 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.365 | 0.432 | 0.067 | 0.404 | 0.045 | 0.404 | 0.384 | 0.000 | -0.020 | -0.020 | 0.404 | 0.020 |
| llada-8b-instruct-hf | plan_390 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.396 | 0.396 | 0.000 | 0.200 | 0.200 | 0.255 | 0.255 | 0.055 | 0.055 | 0.000 | 0.255 | 0.000 |
| llada-8b-instruct-hf | plan_391 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.291 | 0.341 | 0.050 | 0.201 | 0.201 | 0.201 | 0.312 | 0.000 | 0.111 | 0.111 | 0.312 | 0.000 |
| llada-8b-instruct-hf | plan_392 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.451 | 0.467 | 0.016 | 0.280 | 0.280 | 0.280 | 0.243 | 0.000 | -0.037 | -0.037 | 0.280 | 0.037 |
