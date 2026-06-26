# Diffusion Schedule-Selection Benchmark Report

Full model generations: `384`
Counterfactual probe generations: `0`
Arm selections: `384`
Run ID: `diffusion-5527e08851996c2b`
Content hash: `5527e08851996c2bdcab2ace42a1547bcddf07ed101f50dba500cef5e32972ae`
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
History mutability: `monotonic 384/384, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.009`
Trajectory task delta vs random: `0.018`
Trajectory wins/ties/losses vs fixed: `20/65/11`
Trajectory wins/ties/losses vs random: `27/57/12`
Oracle generation budget/task: `4.00`
Oracle task score: `0.358`
Oracle headroom vs trajectory: `0.027`
Oracle wins/ties/losses vs trajectory: `52/44/0`
Selector regret vs trajectory: `0.027 over 52/96 improvable`
Repair arm coverage: `96/96` overall
Repair eligible coverage: `96/96`
Repair task delta vs fixed: `0.031`
Repair task delta vs random: `0.040`
Repair task delta vs trajectory: `0.022`
Repair task delta vs evolved: `0.022`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.011`
Repair wins/ties/losses vs evolved: `43/51/2`
Oracle headroom vs repair: `0.006`
Oracle wins/ties/losses vs repair: `19/77/0`
Selector regret vs repair: `0.006 over 19/96 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `96/96` overall, `96/96` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.321671 | 0.000000 | 0.008920 | - | - |
| random perturbation | repair-covered tasks | 0.312751 | -0.008920 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.352863 | 0.031192 | 0.040112 | 50/38/8 | 56/33/7 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.322 | 0.655 | 0.405 |
| random | 96 | 1.00 | 0.313 | 0.634 | 0.393 |
| trajectory_selected | 96 | 2.00 | 0.331 | 0.656 | 0.412 |
| repair_selected | 96 | 4.00 | 0.353 | 0.667 | 0.431 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.322 | 0.655 | 0.405 |
| planning | random | 96 | 1.00 | 0.313 | 0.634 | 0.393 |
| planning | trajectory_selected | 96 | 2.00 | 0.331 | 0.656 | 0.412 |
| planning | repair_selected | 96 | 4.00 | 0.353 | 0.667 | 0.431 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_441 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.360 | 0.260 | 366 | True | 12 | 0.407 | True | True | 4.000 | 0.125 | 0.037 | 0.037 |
| llada-8b-instruct-hf | plan_442 | random_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.197 | 184 | True | 12 | 0.300 | True | True | 5.000 | 0.156 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_443 | random_32 | True | denoise_phase_repairable | False |  | 0.286 | 0.226 | 360 | True | 12 | 0.286 | True | True | 4.000 | 0.125 | 0.095 | 0.095 |
| llada-8b-instruct-hf | plan_444 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.235 | 353 | True | 12 | 0.444 | True | True | 4.000 | 0.125 | 0.037 | 0.037 |
| llada-8b-instruct-hf | plan_445 | random_32 | True | denoise_phase_repairable | False |  | 0.349 | 0.269 | 334 | True | 12 | 0.533 | True | True | 3.000 | 0.094 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_446 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.399 | 0.281 | 280 | True | 12 | 0.500 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_447 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.461 | 0.281 | 336 | True | 12 | 0.481 | True | True | 4.000 | 0.125 | 0.074 | 0.074 |
| llada-8b-instruct-hf | plan_448 | random_32 | True | denoise_phase_repairable | False |  | 0.326 | 0.226 | 361 | True | 12 | 0.519 | True | True | 3.000 | 0.094 | 0.037 | 0.037 |
| llada-8b-instruct-hf | plan_449 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 306 | True | 12 | 0.400 | True | True | 5.000 | 0.156 | 0.050 | 0.050 |
| llada-8b-instruct-hf | plan_450 | random_32 | True | denoise_phase_repairable | False |  | 0.451 | 0.371 | 263 | True | 12 | 0.400 | True | True | 5.000 | 0.156 | 0.040 | 0.040 |
| llada-8b-instruct-hf | plan_451 | random_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 375 | True | 12 | 0.407 | True | True | 2.000 | 0.062 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_452 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.554 | 0.411 | 353 | True | 12 | 0.423 | True | True | 4.000 | 0.125 | 0.115 | 0.115 |
| llada-8b-instruct-hf | plan_453 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 324 | True | 12 | 0.421 | True | True | 4.000 | 0.125 | 0.053 | 0.053 |
| llada-8b-instruct-hf | plan_454 | random_32 | True | denoise_phase_repairable | False |  | 0.403 | 0.323 | 307 | True | 12 | 0.462 | True | True | 4.000 | 0.125 | 0.038 | 0.038 |
| llada-8b-instruct-hf | plan_455 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 384 | True | 12 | 0.417 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_456 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.531 | 0.394 | 255 | True | 12 | 0.238 | True | True | 4.000 | 0.125 | 0.048 | 0.048 |
| llada-8b-instruct-hf | plan_457 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.336 | 0.256 | 338 | True | 12 | 0.520 | True | True | 4.000 | 0.125 | 0.080 | 0.080 |
| llada-8b-instruct-hf | plan_458 | random_32 | True | denoise_phase_repairable | False |  | 0.414 | 0.276 | 335 | True | 12 | 0.400 | True | True | 2.000 | 0.062 | 0.120 | 0.120 |
| llada-8b-instruct-hf | plan_459 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 317 | True | 12 | 0.435 | True | True | 4.000 | 0.125 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_460 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 364 | True | 12 | 0.556 | True | True | 3.000 | 0.094 | 0.074 | 0.074 |
| llada-8b-instruct-hf | plan_461 | random_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 272 | True | 12 | 0.483 | True | True | 3.000 | 0.094 | 0.034 | 0.034 |
| llada-8b-instruct-hf | plan_462 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 378 | True | 12 | 0.368 | True | True | 3.000 | 0.094 | 0.105 | 0.105 |
| llada-8b-instruct-hf | plan_463 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 390 | True | 10 | 0.680 | True | True | 4.000 | 0.125 | 0.080 | 0.080 |
| llada-8b-instruct-hf | plan_464 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 380 | True | 12 | 0.391 | True | True | 5.000 | 0.156 | 0.087 | 0.087 |
| llada-8b-instruct-hf | plan_465 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.253 | 0.193 | 316 | True | 12 | 0.440 | True | True | 4.000 | 0.125 | 0.040 | 0.040 |
| llada-8b-instruct-hf | plan_466 | random_32 | True | denoise_phase_repairable | False |  | 0.463 | 0.383 | 287 | True | 12 | 0.348 | True | True | 4.000 | 0.125 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_467 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 316 | True | 12 | 0.440 | True | True | 5.000 | 0.156 | 0.120 | 0.120 |
| llada-8b-instruct-hf | plan_468 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 347 | True | 12 | 0.346 | True | True | 3.000 | 0.094 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_469 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 370 | True | 11 | 0.615 | True | True | 3.000 | 0.094 | 0.038 | 0.038 |
| llada-8b-instruct-hf | plan_470 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 360 | True | 12 | 0.565 | True | True | 4.000 | 0.125 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_471 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 331 | True | 12 | 0.455 | True | True | 4.000 | 0.125 | 0.091 | 0.091 |
| llada-8b-instruct-hf | plan_472 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 251 | True | 7 | 0.700 | True | True | 4.000 | 0.125 | 0.050 | 0.050 |
| llada-8b-instruct-hf | plan_473 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.356 | 0.256 | 288 | True | 12 | 0.417 | True | True | 4.000 | 0.125 | 0.042 | 0.042 |
| llada-8b-instruct-hf | plan_474 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.239 | 363 | True | 12 | 0.423 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_475 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.217 | 349 | True | 12 | 0.217 | True | True | 3.000 | 0.094 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_476 | random_32 | True | denoise_phase_repairable | False |  | 0.326 | 0.226 | 253 | True | 10 | 0.591 | True | True | 3.000 | 0.094 | 0.091 | 0.091 |
| llada-8b-instruct-hf | plan_477 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 354 | True | 10 | 0.591 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_478 | random_32 | True | denoise_phase_repairable | False |  | 0.431 | 0.331 | 384 | True | 12 | 0.391 | True | True | 3.000 | 0.094 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_479 | random_32 | True | denoise_phase_repairable | False |  | 0.381 | 0.281 | 274 | True | 12 | 0.429 | True | True | 8.000 | 0.250 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_480 | random_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 286 | True | 12 | 0.250 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_481 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.314 | 0.214 | 362 | True | 12 | 0.500 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_482 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 404 | True | 12 | 0.360 | True | True | 4.000 | 0.125 | 0.040 | 0.040 |
| llada-8b-instruct-hf | plan_483 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.376 | 0.239 | 339 | True | 12 | 0.500 | True | True | 3.000 | 0.094 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_484 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.376 | 0.239 | 243 | True | 12 | 0.500 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_485 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 353 | True | 9 | 0.625 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_486 | random_32 | True | denoise_phase_repairable | False |  | 0.376 | 0.239 | 359 | True | 12 | 0.455 | True | True | 4.000 | 0.125 | 0.045 | 0.045 |
| llada-8b-instruct-hf | plan_487 | random_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.235 | 200 | True | 12 | 0.304 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_488 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 377 | True | 12 | 0.360 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_489 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 290 | True | 12 | 0.333 | True | True | 3.000 | 0.094 | 0.074 | 0.074 |
| llada-8b-instruct-hf | plan_490 | random_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 402 | True | 12 | 0.560 | True | True | 3.000 | 0.094 | 0.040 | 0.040 |
| llada-8b-instruct-hf | plan_491 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 298 | True | 12 | 0.519 | True | True | 4.000 | 0.125 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_492 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.314 | 0.214 | 317 | True | 12 | 0.500 | True | True | 5.000 | 0.156 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_493 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 383 | True | 12 | 0.435 | True | True | 3.000 | 0.094 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_494 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.274 | 0.214 | 291 | True | 12 | 0.269 | True | True | 4.000 | 0.125 | 0.038 | 0.038 |
| llada-8b-instruct-hf | plan_495 | random_32 | True | denoise_phase_repairable | False |  | 0.376 | 0.239 | 364 | True | 12 | 0.481 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_496 | random_32 | True | denoise_phase_repairable | False |  | 0.360 | 0.260 | 372 | True | 12 | 0.357 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_497 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.351 | 0.251 | 385 | True | 12 | 0.565 | True | True | 4.000 | 0.125 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_498 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.239 | 330 | True | 12 | 0.607 | True | True | 4.000 | 0.125 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_499 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 363 | True | 12 | 0.333 | True | True | 3.000 | 0.094 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_500 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.239 | 306 | True | 12 | 0.393 | True | True | 4.000 | 0.125 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_501 | random_32 | True | denoise_phase_repairable | False |  | 0.340 | 0.260 | 353 | True | 12 | 0.444 | True | True | 3.000 | 0.094 | 0.037 | 0.037 |
| llada-8b-instruct-hf | plan_502 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 264 | True | 12 | 0.483 | True | True | 4.000 | 0.125 | 0.103 | 0.103 |
| llada-8b-instruct-hf | plan_503 | random_32 | True | denoise_phase_repairable | False |  | 0.355 | 0.217 | 376 | True | 11 | 0.692 | True | True | 3.000 | 0.094 | 0.115 | 0.115 |
| llada-8b-instruct-hf | plan_504 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.299 | 0.239 | 295 | True | 12 | 0.346 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_505 | random_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.239 | 442 | True | 12 | 0.367 | True | True | 3.000 | 0.094 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_506 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.310 | 0.230 | 334 | True | 12 | 0.419 | True | True | 4.000 | 0.125 | 0.065 | 0.065 |
| llada-8b-instruct-hf | plan_507 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 336 | True | 12 | 0.556 | True | True | 4.000 | 0.125 | 0.148 | 0.148 |
| llada-8b-instruct-hf | plan_508 | random_32 | True | denoise_phase_repairable | False |  | 0.378 | 0.278 | 357 | True | 12 | 0.593 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_509 | random_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 270 | True | 12 | 0.375 | True | True | 3.000 | 0.094 | 0.094 | 0.094 |
| llada-8b-instruct-hf | plan_510 | random_32 | True | denoise_phase_repairable | False |  | 0.321 | 0.281 | 396 | True | 12 | 0.333 | True | True | 2.000 | 0.062 | 0.033 | 0.033 |
| llada-8b-instruct-hf | plan_511 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.376 | 0.239 | 293 | True | 12 | 0.440 | True | True | 6.000 | 0.188 | 0.040 | 0.040 |
| llada-8b-instruct-hf | plan_512 | random_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 290 | True | 12 | 0.393 | True | True | 4.000 | 0.125 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_513 | random_32 | True | denoise_phase_repairable | False |  | 0.335 | 0.217 | 384 | True | 12 | 0.393 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_514 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 329 | True | 12 | 0.333 | True | True | 4.000 | 0.125 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_515 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.390 | 0.273 | 354 | True | 12 | 0.419 | True | True | 3.000 | 0.094 | 0.129 | 0.129 |
| llada-8b-instruct-hf | plan_516 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.356 | 0.256 | 358 | True | 12 | 0.290 | True | True | 4.000 | 0.125 | 0.065 | 0.065 |
| llada-8b-instruct-hf | plan_517 | random_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 247 | True | 12 | 0.429 | True | True | 7.000 | 0.219 | 0.029 | 0.029 |
| llada-8b-instruct-hf | plan_518 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.355 | 0.217 | 343 | True | 12 | 0.320 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_519 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.297 | 0.217 | 384 | True | 12 | 0.645 | True | True | 5.000 | 0.156 | 0.065 | 0.065 |
| llada-8b-instruct-hf | plan_520 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 343 | True | 12 | 0.517 | True | True | 3.000 | 0.094 | 0.103 | 0.103 |
| llada-8b-instruct-hf | plan_521 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 319 | True | 12 | 0.429 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_522 | random_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.217 | 264 | True | 12 | 0.417 | True | True | 3.000 | 0.094 | 0.042 | 0.042 |
| llada-8b-instruct-hf | plan_523 | random_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 352 | True | 12 | 0.448 | True | True | 2.000 | 0.062 | 0.034 | 0.034 |
| llada-8b-instruct-hf | plan_524 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.501 | 0.364 | 315 | True | 12 | 0.194 | True | True | 4.000 | 0.125 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_525 | random_32 | True | denoise_phase_repairable | False |  | 0.238 | 0.138 | 151 | True | 12 | 0.321 | True | True | 5.000 | 0.156 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_526 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 349 | True | 12 | 0.438 | True | True | 4.000 | 0.125 | 0.062 | 0.062 |
| llada-8b-instruct-hf | plan_527 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 397 | True | 12 | 0.250 | True | True | 3.000 | 0.094 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_528 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.340 | 0.260 | 375 | True | 12 | 0.323 | True | True | 3.000 | 0.094 | 0.065 | 0.065 |
| llada-8b-instruct-hf | plan_529 | random_32 | True | denoise_phase_repairable | False |  | 0.423 | 0.323 | 361 | True | 12 | 0.433 | True | True | 2.000 | 0.062 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_530 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 361 | True | 12 | 0.484 | True | True | 3.000 | 0.094 | 0.032 | 0.032 |
| llada-8b-instruct-hf | plan_531 | random_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 302 | True | 12 | 0.520 | True | True | 2.000 | 0.062 | 0.040 | 0.040 |
| llada-8b-instruct-hf | plan_532 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.319 | 0.239 | 294 | True | 12 | 0.500 | True | True | 6.000 | 0.188 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_533 | random_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 366 | True | 12 | 0.458 | True | True | 3.000 | 0.094 | 0.042 | 0.042 |
| llada-8b-instruct-hf | plan_534 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.376 | 0.239 | 390 | True | 12 | 0.393 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_535 | random_32 | True | denoise_phase_repairable | False |  | 0.356 | 0.256 | 313 | True | 12 | 0.345 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_536 | random_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 377 | True | 12 | 0.568 | True | True | 5.000 | 0.156 | 0.054 | 0.054 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 96 | 24 | low_confidence_32,random_32 | final | 33.0 | 0.979 | 0.021 | 0.000 | 0.003 | 0.003 | -0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 31/34/31 | 0.332 | 0.675 | 0.418 |
| history_prefix_25_repair | 96 | 21 | low_confidence_32,random_32 | history | 48.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28/36/32 | 0.332 | 0.683 | 0.420 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_441 | False | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | Start with a small subset of pods and gradually increase the number of updated pods. |
| llada-8b-instruct-hf | plan_441 | False | low_confidence_32 | 2.031 | 0.805 | 1.000 | 0.000 | 0.037 | False | Monitor memory usage and adjust pod limits accordingly. |
| llada-8b-instruct-hf | plan_441 | False | low_confidence_32 | 2.095 | 0.758 | 1.000 | 0.000 | 0.148 | False | Additionally, consider using resource reservations or resource reservations to mitigate... |
| llada-8b-instruct-hf | plan_442 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Decision: Merge now Reasoning: - Benefit: Save infrastructure cost - Risk: Multi-week d... |
| llada-8b-instruct-hf | plan_443 | True | random_32 | 2.105 | 0.949 | 1.000 | 0.000 | 0.048 | False | First, onboard core hires for essential roles, ensuring minimal onboarding time. |
| llada-8b-instruct-hf | plan_443 | True | random_32 | 2.120 | 0.807 | 1.000 | 0.000 | 0.143 | False | This way, the startup can accelerate its team without delay and without compromising it... |
| llada-8b-instruct-hf | plan_444 | True | low_confidence_32 | 2.508 | 0.901 | 1.000 | 0.000 | 0.148 | False | First, implement a temporary freeze on data processing for 30 days to address the compl... |
| llada-8b-instruct-hf | plan_444 | True | low_confidence_32 | 3.326 | 1.000 | 1.000 | 0.000 | 0.074 | False | This approach ensures compliance without compromising SLAs. |
| llada-8b-instruct-hf | plan_445 | False | random_32 | 2.007 | 0.625 | 1.000 | 0.000 | 0.267 | False | If the model shows no significant improvement on the real user queries, switch the path... |
| llada-8b-instruct-hf | plan_446 | False | low_confidence_32 | 1.875 | 0.786 | 1.000 | 0.000 | 0.107 | False | Complexity of the migration process on theSQL system. |
| llada-8b-instruct-hf | plan_446 | False | low_confidence_32 | 3.322 | 0.865 | 1.000 | 0.000 | 0.036 | False | Potential impact on the performance and scalability of the migration. |
| llada-8b-instruct-hf | plan_447 | True | low_confidence_32 | 2.440 | 0.478 | 1.000 | 0.000 | 0.222 | False | **Longer-Term Fix:** Implement a dynamic rate limiter algorithm that adjusts limits bas... |
| llada-8b-instruct-hf | plan_448 | True | random_32 | 2.174 | 0.919 | 1.000 | 0.000 | 0.111 | False | This approach maximizes paper strength and provides the necessary flexibility to meet t... |
| llada-8b-instruct-hf | plan_449 | True | low_confidence_32 | 1.391 | 0.861 | 1.000 | 0.000 | 0.200 | False | Track infrastructure costs, paid conversions, and user retention. |
| llada-8b-instruct-hf | plan_449 | True | low_confidence_32 | 1.378 | 0.893 | 1.000 | 0.000 | 0.250 | False | Use a split test to ensure fairness and measure the impact of the free tier removal on... |
| llada-8b-instruct-hf | plan_449 | True | low_confidence_32 | 2.056 | 0.650 | 1.000 | 0.000 | 0.100 | False | Collect and analyze the results within 30 days. |
| llada-8b-instruct-hf | plan_450 | False | random_32 | 2.398 | 0.614 | 1.000 | 0.000 | 0.080 | False | Merge conflict rate to reduce the time spent on merge conflicts. |
| llada-8b-instruct-hf | plan_450 | False | random_32 | 2.450 | 0.743 | 1.000 | 0.000 | 0.080 | False | Pipeline time after parallelizing the test suite. |
| llada-8b-instruct-hf | plan_450 | False | random_32 | 3.184 | 0.709 | 1.000 | 0.000 | 0.120 | False | The additional engineering cost of parallelizing the test suite. |
| llada-8b-instruct-hf | plan_451 | False | random_32 | 1.879 | 0.862 | 1.000 | 0.000 | 0.185 | False | Use a trade-off or cost-benefit analysis to weigh the benefits of reduced false positiv... |
| llada-8b-instruct-hf | plan_451 | False | random_32 | 2.148 | 0.887 | 1.000 | 0.000 | 0.148 | False | Adjust the threshold to find the optimal balance that maximizes customer trust and fina... |
| llada-8b-instruct-hf | plan_452 | False | low_confidence_32 | 2.078 | 0.925 | 1.000 | 0.000 | 0.000 | False | Implement a retry mechanism to handle transient failures, ensuring that the are no retr... |
| llada-8b-instruct-hf | plan_452 | False | low_confidence_32 | 2.870 | 1.000 | 1.000 | 0.000 | 0.038 | False | Use circuit breaking to to limit retries and prevent cascading failures, maintaining th... |
| llada-8b-instruct-hf | plan_453 | False | low_confidence_32 | 1.977 | 0.531 | 1.000 | 0.000 | 0.263 | False | Additionally, analyze the underrepresentation of mobile users on slow connections to en... |
| llada-8b-instruct-hf | plan_454 | False | random_32 | 1.250 | 0.576 | 1.000 | 0.000 | 0.115 | False | This approach is the best option because: 1. the transition will take only 30 days. |
| llada-8b-instruct-hf | plan_454 | False | random_32 | 1.349 | 0.760 | 1.000 | 0.000 | 0.154 | False | 2. the cheaper alternative has fewer features. |
| llada-8b-instruct-hf | plan_454 | False | random_32 | 2.195 | 0.955 | 1.000 | 0.000 | 0.115 | False | 3. the immediate transition will be with the cheaper alternative. |
| llada-8b-instruct-hf | plan_455 | False | low_confidence_32 | 1.462 | 1.000 | 1.000 | 0.000 | 0.083 | False | This approach adjusts the drift threshold dynamically based on recent data patterns. |
| llada-8b-instruct-hf | plan_455 | False | low_confidence_32 | 2.055 | 0.731 | 1.000 | 0.000 | 0.292 | False | By increasing the window size, the drift threshold can align more closely with actual d... |
| llada-8b-instruct-hf | plan_456 | False | low_confidence_32 | 1.913 | 0.891 | 1.000 | 0.000 | 0.143 | False | Rotate the most critical secrets 6 weeks before the audit. |
| llada-8b-instruct-hf | plan_456 | False | low_confidence_32 | 1.913 | 0.891 | 1.000 | 0.000 | 0.143 | False | Rotate the remaining secrets 4 weeks before the audit. |
| llada-8b-instruct-hf | plan_456 | False | low_confidence_32 | 2.663 | 0.891 | 1.000 | 0.000 | 0.143 | False | Rotate the remaining secrets 2 weeks before the audit. |
| llada-8b-instruct-hf | plan_457 | False | low_confidence_32 | 1.851 | 0.338 | 1.000 | 0.000 | 0.360 | False | The product manager should delay the legal team's decision on the anonymization pipelin... |
| llada-8b-instruct-hf | plan_458 | True | random_32 | 2.094 | 0.799 | 1.000 | 0.000 | 0.200 | False | This ensures that the operational data center can handle the increased load in the even... |
| llada-8b-instruct-hf | plan_459 | False | low_confidence_32 | 1.330 | 0.729 | 1.000 | 0.000 | 0.174 | False | However, they should consider the impact of removing these entries on the model's perfo... |
| llada-8b-instruct-hf | plan_459 | False | low_confidence_32 | 2.173 | 0.910 | 1.000 | 0.000 | 0.087 | False | They may also need to re the distribution of the evaluation set to reflect the unique d... |
| llada-8b-instruct-hf | plan_460 | True | low_confidence_32 | 1.308 | 0.697 | 1.000 | 0.000 | 0.185 | False | Long-term resilience: Implement a manual update process to ensure the latest cache vers... |
| llada-8b-instruct-hf | plan_460 | True | low_confidence_32 | 1.435 | 0.919 | 1.000 | 0.000 | 0.111 | False | Establish communication with the library maintainer to restore releases. |
| llada-8b-instruct-hf | plan_460 | True | low_confidence_32 | 2.122 | 0.810 | 1.000 | 0.000 | 0.111 | False | Explore alternative repositories to reduce dependency on the package registry for futur... |
| llada-8b-instruct-hf | plan_461 | False | random_32 | 1.385 | 0.802 | 1.000 | 0.000 | 0.069 | False | 2. the duration of the retention increase. |
| llada-8b-instruct-hf | plan_461 | False | random_32 | 2.193 | 0.955 | 1.000 | 0.000 | 0.069 | False | 3. the impact the feature is having on the overall experience of the users. |
| llada-8b-instruct-hf | plan_462 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Use a shared communication platform to track progress and provide updates. |
| llada-8b-instruct-hf | plan_462 | False | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | Encourage regular breaks and self-care to prevent burnout. |
| llada-8b-instruct-hf | plan_463 | False | low_confidence_32 | 1.343 | 0.770 | 1.000 | 0.000 | 0.200 | False | Store sensitive data in the compliant region and store non-sensitive data in the best-p... |
| llada-8b-instruct-hf | plan_463 | False | low_confidence_32 | 1.927 | 0.443 | 1.000 | 0.000 | 0.280 | False | Use data transfer protocols to move data efficiently between regions, ensuring latency... |
| llada-8b-instruct-hf | plan_464 | True | low_confidence_32 | 2.073 | 0.893 | 1.000 | 0.000 | 0.043 | False | Measure the time taken to complete critical tasks on both systems. |
| llada-8b-instruct-hf | plan_464 | True | low_confidence_32 | 1.458 | 1.000 | 1.000 | 0.000 | 0.130 | False | If the Rust backend shows significant performance improvements, proceed with the rewrite. |
| llada-8b-instruct-hf | plan_464 | True | low_confidence_32 | 2.088 | 0.785 | 1.000 | 0.000 | 0.261 | False | If the Python backend meets the performance requirements, allocate engineering time to... |
| llada-8b-instruct-hf | plan_465 | False | low_confidence_32 | 2.064 | 0.736 | 1.000 | 0.000 | 0.280 | False | Once the ORM layer is refact, the WAF rule can be added to block the SQL injection vect... |
| llada-8b-instruct-hf | plan_466 | False | random_32 | 2.435 | 0.700 | 1.000 | 0.000 | 0.087 | False | Determine the cause of the revenue change. |
| llada-8b-instruct-hf | plan_466 | False | random_32 | 3.130 | 0.775 | 1.000 | 0.000 | 0.043 | False | Evaluate the impact on overall revenue. |
| llada-8b-instruct-hf | plan_466 | False | random_32 | 3.167 | 0.653 | 1.000 | 0.000 | 0.087 | False | Reconsider the decision based on the net revenue impact. |
| llada-8b-instruct-hf | plan_467 | True | low_confidence_32 | 1.416 | 0.905 | 1.000 | 0.000 | 0.120 | False | This can involve rewriting the query, using appropriate indexes, and ensuring that the... |
| llada-8b-instruct-hf | plan_467 | True | low_confidence_32 | 2.045 | 0.674 | 1.000 | 0.000 | 0.200 | False | Pre-aggregating data, moving to a columnar store, or reducing the dashboard scope would... |
| llada-8b-instruct-hf | plan_468 | False | low_confidence_32 | 1.940 | 0.620 | 1.000 | 0.000 | 0.038 | False | Forensics: Collect logs, analyze the breach, and identify the source. |
| llada-8b-instruct-hf | plan_468 | False | low_confidence_32 | 1.293 | 0.658 | 1.000 | 0.000 | 0.154 | False | Cost mitigation: Stop unused resources, re back unused instances, and budget for additi... |
| llada-8b-instruct-hf | plan_468 | False | low_confidence_32 | 2.125 | 0.805 | 1.000 | 0.000 | 0.077 | False | Process changes: Implement stricter access controls, regular security audits, and enhan... |
| llada-8b-instruct-hf | plan_469 | True | low_confidence_32 | 2.902 | 0.205 | 1.000 | 0.000 | 0.346 | False | This approach ensures that the results are reliable and meet the integrity required for... |
| llada-8b-instruct-hf | plan_470 | False | low_confidence_32 | 1.954 | 0.511 | 1.000 | 0.000 | 0.261 | False | This approach allows users to manage the battery drain issue without affecting the over... |
| llada-8b-instruct-hf | plan_471 | False | low_confidence_32 | 1.348 | 0.778 | 1.000 | 0.000 | 0.182 | False | This will avoid a feature freeze and allow the monolith to continue running while the m... |
| llada-8b-instruct-hf | plan_471 | False | low_confidence_32 | 2.105 | 0.778 | 1.000 | 0.000 | 0.136 | False | Additionally, use eventler to maintain transaction consistency with the rest of the mon... |
| llada-8b-instruct-hf | plan_472 | True | low_confidence_32 | 1.906 | 0.377 | 1.000 | 0.000 | 0.250 | False | This way, the total latency will be 250ms, which is well than the machine learning mode... |
| llada-8b-instruct-hf | plan_473 | False | low_confidence_32 | 1.917 | 0.912 | 1.000 | 0.000 | 0.125 | False | If slippage is significant, adjust the model and re-test the live strategy. |
| llada-8b-instruct-hf | plan_473 | False | low_confidence_32 | 1.922 | 0.423 | 1.000 | 0.000 | 0.250 | False | If the returns align match the backtest after adjusting slippage, stop the live test to... |
| llada-8b-instruct-hf | plan_474 | True | low_confidence_32 | 1.815 | 0.219 | 1.000 | 0.000 | 0.231 | False | This approach will allow them to understand the dependency graph, identify potential is... |
| llada-8b-instruct-hf | plan_475 | True | low_confidence_32 | 2.057 | 0.820 | 1.000 | 0.000 | 0.000 | False | Draft a formal letter outlining the violations and potential consequences. |
| llada-8b-instruct-hf | plan_475 | True | low_confidence_32 | 2.134 | 1.000 | 1.000 | 0.000 | 0.043 | False | If violations continue, consider legal legal action or discontin terminating the relati... |
| llada-8b-instruct-hf | plan_475 | True | low_confidence_32 | 2.094 | 0.730 | 1.000 | 0.000 | 0.130 | False | Explore alternative customer or solutions to minimize legal and financial impact. |
| llada-8b-instruct-hf | plan_476 | False | random_32 | 1.918 | 0.428 | 1.000 | 0.000 | 0.318 | False | Therefore, the system can check and process all 500K requests per second, without dropp... |
| llada-8b-instruct-hf | plan_477 | True | low_confidence_32 | 1.983 | 0.556 | 1.000 | 0.000 | 0.273 | False | The team should focus on refining and writing Approach B with the available compute and... |
| llada-8b-instruct-hf | plan_478 | False | random_32 | 2.558 | 1.000 | 1.000 | 0.000 | 0.174 | False | Apply the changes from the production database to the staging database. |
| llada-8b-instruct-hf | plan_478 | False | random_32 | 2.540 | 1.000 | 1.000 | 0.000 | 0.217 | False | Replicate the databases and apply changes from the staging database to the production d... |
| llada-8b-instruct-hf | plan_478 | False | random_32 | 2.929 | 0.247 | 1.000 | 0.000 | 0.217 | False | Ensure the reconciliation process completes before the production fix is fully deployed... |
| llada-8b-instruct-hf | plan_479 | True | random_32 | 1.857 | 0.733 | 1.000 | 0.000 | 0.071 | False | Continue operation in rain. |
| llada-8b-instruct-hf | plan_479 | True | random_32 | 1.771 | 0.548 | 1.000 | 0.000 | 0.071 | False | Minimize lost revenue. |
| llada-8b-instruct-hf | plan_479 | True | random_32 | 3.199 | 0.602 | 1.000 | 0.000 | 0.000 | False | Continuous monitoring and improvement. |
| llada-8b-instruct-hf | plan_480 | False | random_32 | 2.037 | 0.775 | 1.000 | 0.000 | 0.000 | False | Discuss potential solutions and trade-offs. |
| llada-8b-instruct-hf | plan_480 | False | random_32 | 1.996 | 0.696 | 1.000 | 0.000 | 0.036 | False | Evaluate the pros and cons of each architecture. |
| llada-8b-instruct-hf | plan_480 | False | random_32 | 2.034 | 0.589 | 1.000 | 0.000 | 0.071 | False | Make a collective decision based on data and consensus. |
| llada-8b-instruct-hf | plan_481 | False | low_confidence_32 | 1.256 | 0.590 | 1.000 | 0.000 | 0.192 | False | This involves gradually transferring the COBOL batch jobs to the cloud system, starting... |
| llada-8b-instruct-hf | plan_481 | False | low_confidence_32 | 2.010 | 0.609 | 1.000 | 0.000 | 0.231 | False | This ensures that the regulatory reports are produced on time, regardless of the full r... |
| llada-8b-instruct-hf | plan_482 | False | random_32 | 1.805 | 0.670 | 1.000 | 0.000 | 0.120 | False | Use a classification model to identify potential violations and predefined rules to fla... |
| llada-8b-instruct-hf | plan_482 | False | random_32 | 2.610 | 0.799 | 1.000 | 0.000 | 0.160 | False | Implement a feedback loop to continuously review user complaints and adjust the system'... |
| llada-8b-instruct-hf | plan_483 | False | low_confidence_32 | 2.170 | 0.968 | 1.000 | 0.000 | 0.192 | False | This decision balances out the significant cost savings against the temporary loss in m... |
| llada-8b-instruct-hf | plan_484 | False | low_confidence_32 | 1.900 | 0.372 | 1.000 | 0.000 | 0.273 | False | This will allow the AS to be shipped on time, even if it takes 1-4 weeks,, avoiding the... |
| llada-8b-instruct-hf | plan_485 | True | low_confidence_32 | 3.094 | 0.617 | 1.000 | 0.000 | 0.375 | False | This would allocate higher ad bids for more relevant products and lower ad bids for lon... |
| llada-8b-instruct-hf | plan_486 | False | random_32 | 1.900 | 0.907 | 1.000 | 0.000 | 0.227 | False | During the maintenance window, the hospital should allocate sufficient staff to apply t... |
| llada-8b-instruct-hf | plan_486 | False | random_32 | 1.997 | 0.571 | 1.000 | 0.000 | 0.227 | False | This approach minimizes the impact on the hospital's operations and reduces the risk of... |
| llada-8b-instruct-hf | plan_487 | False | random_32 | 1.945 | 0.598 | 1.000 | 0.000 | 0.043 | False | Ignore the bug before shipping: $0. |
| llada-8b-instruct-hf | plan_487 | False | random_32 | 2.773 | 0.770 | 1.000 | 0.000 | 0.043 | False | Choose the best option: Fix the desync bug. |
| llada-8b-instruct-hf | plan_488 | False | low_confidence_32 | 1.301 | 0.674 | 1.000 | 0.000 | 0.160 | False | This will allow the support team to continue handling the customer escalation without i... |
| llada-8b-instruct-hf | plan_488 | False | low_confidence_32 | 2.180 | 0.960 | 1.000 | 0.000 | 0.160 | False | Additionally, consider the use of temporary access tokens or backup systems to ensure t... |
| llada-8b-instruct-hf | plan_489 | False | low_confidence_32 | 1.310 | 0.640 | 1.000 | 0.000 | 0.074 | False | Assess the current Fortran code and hardware limitations. |
| llada-8b-instruct-hf | plan_489 | False | low_confidence_32 | 2.035 | 0.784 | 1.000 | 0.000 | 0.037 | False | Develop a roadmap to rewrite the solver in C++. |
| llada-8b-instruct-hf | plan_489 | False | low_confidence_32 | 2.113 | 0.752 | 1.000 | 0.000 | 0.074 | False | Establish a timeline6 months and regularcommissioning to ensure a the transition. |
| llada-8b-instruct-hf | plan_490 | False | random_32 | 1.459 | 1.000 | 1.000 | 0.000 | 0.120 | False | Use this feedback to adjust the algorithm's parameters and prioritize more predictable... |
| llada-8b-instruct-hf | plan_490 | False | random_32 | 2.066 | 0.714 | 1.000 | 0.000 | 0.200 | False | Additionally, consider offering additional benefits, such as better pay and rest suppor... |
| llada-8b-instruct-hf | plan_491 | False | low_confidence_32 | 2.455 | 0.801 | 1.000 | 0.000 | 0.259 | False | Evaluate the performance of the affected models using the stale features. |
| llada-8b-instruct-hf | plan_491 | False | low_confidence_32 | 3.095 | 0.578 | 1.000 | 0.000 | 0.222 | False | Use the results to make an informed decision as to whether the affected models need ret... |
| llada-8b-instruct-hf | plan_492 | False | low_confidence_32 | 1.925 | 0.441 | 1.000 | 0.000 | 0.286 | False | This process can be completed in 2 weeks, allowing the platform to onboard the supplier... |
| llada-8b-instruct-hf | plan_493 | False | low_confidence_32 | 3.330 | 0.910 | 1.000 | 0.000 | 0.043 | False | Additionally, they should experiment with different qubit configurations to understand... |
| llada-8b-instruct-hf | plan_494 | False | low_confidence_32 | 2.636 | 0.805 | 1.000 | 0.000 | 0.077 | False | This approach aims to minimize disruption and conserve engineering resources. |
| llada-8b-instruct-hf | plan_495 | False | random_32 | 2.404 | 0.396 | 1.000 | 0.000 | 0.259 | False | By isolating the segment, the team can use the honeypot to gather intelligence and trac... |
| llada-8b-instruct-hf | plan_496 | True | random_32 | 1.369 | 0.811 | 1.000 | 0.000 | 0.107 | False | Consider the cost analysis, timelines, and risks associated with parallel vs. sequentia... |
| llada-8b-instruct-hf | plan_496 | True | random_32 | 2.121 | 0.815 | 1.000 | 0.000 | 0.107 | False | Assess the potential risks and develop mitigation strategies to minimize the impact on... |
| llada-8b-instruct-hf | plan_497 | False | low_confidence_32 | 1.400 | 0.878 | 1.000 | 0.000 | 0.130 | False | Use live traffic analytics to predict peak times and adjust capacity allocation accordi... |
| llada-8b-instruct-hf | plan_497 | False | low_confidence_32 | 1.841 | 0.277 | 1.000 | 0.000 | 0.304 | False | Employ a auto-scaling mechanism that temporarily increases capacity during live events... |
| llada-8b-instruct-hf | plan_498 | False | low_confidence_32 | 1.814 | 0.695 | 1.000 | 0.000 | 0.143 | False | This involves combining GPS data with visual features to mitigate GPS drift and visual... |
| llada-8b-instruct-hf | plan_499 | False | low_confidence_32 | 3.139 | 0.648 | 1.000 | 0.000 | 0.167 | False | This would reduce the need to add more templates and improve the process of for legitim... |
| llada-8b-instruct-hf | plan_500 | False | low_confidence_32 | 1.419 | 0.905 | 1.000 | 0.000 | 0.071 | False | This a) the number of failed retries, b) the delay between retries, and c) the number o... |
| llada-8b-instruct-hf | plan_500 | False | low_confidence_32 | 1.854 | 0.300 | 1.000 | 0.000 | 0.321 | False | This approach reduces the number of failed retries and CI compute, systematically addre... |
| llada-8b-instruct-hf | plan_501 | False | random_32 | 1.809 | 0.212 | 1.000 | 0.000 | 0.259 | False | A complete recall of past results may not be necessary; instead, update the dataset to... |
| llada-8b-instruct-hf | plan_502 | False | low_confidence_32 | 1.367 | 0.757 | 1.000 | 0.000 | 0.069 | False | Utilize storage to cover 60% of the gap at sunset. |
| llada-8b-instruct-hf | plan_502 | False | low_confidence_32 | 2.059 | 0.662 | 1.000 | 0.000 | 0.103 | False | Increase additional generation sources, such as natural gas, to meet the increased heat... |
| llada-8b-instruct-hf | plan_503 | False | random_32 | 2.650 | 0.942 | 1.000 | 0.000 | 0.308 | False | By supporting customers up to 2 major versions, the vendor can maintain compatibility a... |
| llada-8b-instruct-hf | plan_504 | False | low_confidence_32 | 2.471 | 0.495 | 1.000 | 0.000 | 0.154 | False | By optimizing the algorithm, the batch process can be sped up, potentially from 7:45 AM... |
| llada-8b-instruct-hf | plan_505 | False | random_32 | 1.449 | 1.000 | 1.000 | 0.000 | 0.167 | False | This involves using a subset of the existing simulations to generate adversarial scenar... |
| llada-8b-instruct-hf | plan_505 | False | random_32 | 2.030 | 0.667 | 1.000 | 0.000 | 0.233 | False | Additionally, schedule adversarial tests to run concurrently with the regular simulatio... |
| llada-8b-instruct-hf | plan_506 | False | low_confidence_32 | 1.316 | 0.680 | 1.000 | 0.000 | 0.129 | False | Analyze work environment, management style, and job satisfaction. |
| llada-8b-instruct-hf | plan_506 | False | low_confidence_32 | 2.100 | 0.927 | 1.000 | 0.000 | 0.032 | False | Consider conducting a focus group or survey to gather more insights. |
| llada-8b-instruct-hf | plan_506 | False | low_confidence_32 | 2.122 | 0.778 | 1.000 | 0.000 | 0.065 | False | Document and report on the findings within 2 weeks. |
| llada-8b-instruct-hf | plan_507 | True | low_confidence_32 | 1.772 | 0.629 | 1.000 | 0.000 | 0.185 | False | This will allow Region B to regain write capability and resume customer transactions. |
| llada-8b-instruct-hf | plan_507 | True | low_confidence_32 | 2.676 | 1.000 | 1.000 | 0.000 | 0.296 | False | The updated replicas in Region A will be up-to-date with the write master, ensuring dat... |
| llada-8b-instruct-hf | plan_508 | False | random_32 | 2.129 | 0.839 | 1.000 | 0.000 | 0.148 | False | Additionally, it should weigh the immediate benefits against the loss of control over t... |
| llada-8b-instruct-hf | plan_509 | False | random_32 | 1.883 | 0.820 | 1.000 | 0.000 | 0.125 | False | 6 weeks: developing fraud detection system. |
| llada-8b-instruct-hf | plan_509 | False | random_32 | 2.519 | 0.748 | 1.000 | 0.000 | 0.031 | False | Disclose findings to advertisers. |
| llada-8b-instruct-hf | plan_509 | False | random_32 | 2.460 | 0.438 | 1.000 | 0.000 | 0.156 | False | 6 weeks: implementing a remediation plan to maintain revenue and trust. |
| llada-8b-instruct-hf | plan_510 | False | random_32 | 1.846 | 0.317 | 1.000 | 0.000 | 0.267 | False | This system can identify and resolve deadlocks dynamically by continuously monitoring t... |
| llada-8b-instruct-hf | plan_511 | False | low_confidence_32 | 2.110 | 0.817 | 1.000 | 0.000 | 0.160 | False | The risk of updating the satellite is lower than the risk of sensor data corruption, ma... |
| llada-8b-instruct-hf | plan_512 | False | random_32 | 2.054 | 0.698 | 1.000 | 0.000 | 0.214 | False | This will allow the ML engineers to work on new model development while still migrating... |
| llada-8b-instruct-hf | plan_513 | False | random_32 | 2.012 | 0.614 | 1.000 | 0.000 | 0.214 | False | This approach minimizes the risk of spreading misinformation and mitigates some of the... |
| llada-8b-instruct-hf | plan_514 | True | low_confidence_32 | 2.123 | 0.841 | 1.000 | 0.000 | 0.292 | False | The decision criteria should be the project of the highest impact on the company's comp... |
| llada-8b-instruct-hf | plan_515 | True | low_confidence_32 | 2.157 | 0.927 | 1.000 | 0.000 | 0.194 | False | By doing so, the system can maintain a high throughput while minimizing the risk of ser... |
| llada-8b-instruct-hf | plan_516 | True | low_confidence_32 | 1.377 | 0.828 | 1.000 | 0.000 | 0.097 | False | Stakeholder communication: Prepare a detailed report explaining the issue, its impact,... |
| llada-8b-instruct-hf | plan_516 | True | low_confidence_32 | 2.044 | 0.672 | 1.000 | 0.000 | 0.161 | False | Prevention strategy: Implement a validation step in the ETL process to ensure decimal p... |
| llada-8b-instruct-hf | plan_517 | False | random_32 | 1.447 | 0.940 | 1.000 | 0.000 | 0.057 | False | Depot cost 4. |
| llada-8b-instruct-hf | plan_517 | False | random_32 | 1.411 | 0.863 | 1.000 | 0.000 | 0.057 | False | Logistics costs 6. |
| llada-8b-instruct-hf | plan_517 | False | random_32 | 2.092 | 0.720 | 1.000 | 0.000 | 0.086 | False | Regulatory approval timelines |
| llada-8b-instruct-hf | plan_518 | False | low_confidence_32 | 2.078 | 0.714 | 1.000 | 0.000 | 0.120 | False | However, to address the issue and ensure long-term reliability, the team should invest... |
| llada-8b-instruct-hf | plan_519 | False | low_confidence_32 | 1.829 | 0.262 | 1.000 | 0.000 | 0.323 | False | This approach ensures that the model remains accurate while reducing the impact with pr... |
| llada-8b-instruct-hf | plan_520 | False | low_confidence_32 | 1.295 | 0.711 | 1.000 | 0.000 | 0.276 | False | During peak hours, relax the skill threshold to reduce queue times, but maintain a buff... |
| llada-8b-instruct-hf | plan_520 | False | low_confidence_32 | 2.102 | 0.820 | 1.000 | 0.000 | 0.241 | False | During off-peak hours, tighten the skill threshold to minimize lopsided matches and red... |
| llada-8b-instruct-hf | plan_521 | True | low_confidence_32 | 1.921 | 0.921 | 1.000 | 0.000 | 0.107 | False | This will reduce the number of false positives, allowing radiologists to focus on more... |
| llada-8b-instruct-hf | plan_521 | True | low_confidence_32 | 2.586 | 0.764 | 1.000 | 0.000 | 0.214 | False | While it may miss 5% of real tumors, the benefits of reduced false positives and improv... |
| llada-8b-instruct-hf | plan_522 | False | random_32 | 2.014 | 0.726 | 1.000 | 0.000 | 0.000 | False | Decontaminate the affected cleanroom. |
| llada-8b-instruct-hf | plan_522 | False | random_32 | 1.313 | 0.638 | 1.000 | 0.000 | 0.083 | False | Complete requalification for the line. |
| llada-8b-instruct-hf | plan_522 | False | random_32 | 1.847 | 0.217 | 1.000 | 0.000 | 0.167 | False | Monitor production closely to avoid contractual penalty clauses. |
| llada-8b-instruct-hf | plan_523 | False | random_32 | 2.871 | 1.000 | 1.000 | 0.000 | 0.034 | False | Replacing would likely make the best use of the investment and ensure better outcomes f... |
| llada-8b-instruct-hf | plan_524 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Implement a rate management strategy that a) uses a combination of rate APIs and local... |
| llada-8b-instruct-hf | plan_525 | False | random_32 | 2.034 | 0.630 | 1.000 | 0.000 | 0.214 | False | Ensure no models are deleted needed for regulatory audit trails. |
| llada-8b-instruct-hf | plan_526 | False | low_confidence_32 | 3.106 | 0.584 | 1.000 | 0.000 | 0.188 | False | Additionally, deploy additional ambulances in the affected neighborhoods to improve res... |
| llada-8b-instruct-hf | plan_527 | False | low_confidence_32 | 1.460 | 1.000 | 1.000 | 0.000 | 0.107 | False | This involves dividing the database into smaller, more manageable partitions based on t... |
| llada-8b-instruct-hf | plan_527 | False | low_confidence_32 | 2.217 | 1.000 | 1.000 | 0.000 | 0.143 | False | This will help reduce lock contention and and improve performance without violating the... |
| llada-8b-instruct-hf | plan_528 | False | low_confidence_32 | 1.299 | 0.636 | 1.000 | 0.000 | 0.097 | False | First, use the deepfake detection system to identify potential issues. |
| llada-8b-instruct-hf | plan_528 | False | low_confidence_32 | 1.433 | 0.949 | 1.000 | 0.000 | 0.097 | False | Then, conduct manual video review and cross-checking with external experts to confirm l... |
| llada-8b-instruct-hf | plan_528 | False | low_confidence_32 | 2.094 | 0.777 | 1.000 | 0.000 | 0.161 | False | Finally, publish timely retractions for falsely flagged videos to maintain transparency... |
| llada-8b-instruct-hf | plan_529 | False | random_32 | 2.076 | 0.887 | 1.000 | 0.000 | 0.433 | False | We can then use this information to adjust the alert threshold and determine the optima... |
| llada-8b-instruct-hf | plan_530 | False | low_confidence_32 | 1.705 | 0.496 | 1.000 | 0.000 | 0.226 | False | Gradually convert flat-rate plans to usage-based pricing over time months, Offer incent... |
| llada-8b-instruct-hf | plan_530 | False | low_confidence_32 | 2.611 | 0.795 | 1.000 | 0.000 | 0.129 | False | This will help minimize churn and maximize revenue while not alienating existing custom... |
| llada-8b-instruct-hf | plan_531 | False | random_32 | 2.059 | 0.736 | 1.000 | 0.000 | 0.280 | False | The closer the maneuver occurs to the approach, the greater the collision probability a... |
| llada-8b-instruct-hf | plan_532 | True | low_confidence_32 | 1.466 | 1.000 | 1.000 | 0.000 | 0.071 | False | Focus on a basic UI, a functional API, and a preliminary model. |
| llada-8b-instruct-hf | plan_532 | True | low_confidence_32 | 1.323 | 0.695 | 1.000 | 0.000 | 0.179 | False | Defer polish polish UI, robust API, and well-validated model. |
| llada-8b-instruct-hf | plan_532 | True | low_confidence_32 | 1.926 | 0.433 | 1.000 | 0.000 | 0.250 | False | Each component must meet core user requirements and basic functionality for a credible... |
| llada-8b-instruct-hf | plan_533 | False | random_32 | 1.256 | 0.599 | 1.000 | 0.000 | 0.208 | False | This approach will preserve the loss of pre-pandemic fraud signatures while avoiding th... |
| llada-8b-instruct-hf | plan_533 | False | random_32 | 2.175 | 0.912 | 1.000 | 0.000 | 0.083 | False | Implement a transfer learning strategy to adapt the features from both periods, leverag... |
| llada-8b-instruct-hf | plan_534 | False | low_confidence_32 | 1.292 | 0.664 | 1.000 | 0.000 | 0.143 | False | This system can be integrated with the fleet management system to dynamically adjust ve... |
| llada-8b-instruct-hf | plan_534 | False | low_confidence_32 | 2.063 | 0.686 | 1.000 | 0.000 | 0.143 | False | This interim solution will improve safety while minimizing the impact on ore throughput. |
| llada-8b-instruct-hf | plan_535 | True | random_32 | 1.292 | 0.603 | 1.000 | 0.000 | 0.103 | False | Patching critical and frequently used services first. |
| llada-8b-instruct-hf | plan_535 | True | random_32 | 1.980 | 0.666 | 1.000 | 0.000 | 0.034 | False | Schedule incremental patches with thorough testing. |
| llada-8b-instruct-hf | plan_535 | True | random_32 | 2.051 | 0.622 | 1.000 | 0.000 | 0.103 | False | Establish a patch window to minimize attacker exposure. |
| llada-8b-instruct-hf | plan_536 | False | random_32 | 1.369 | 0.825 | 1.000 | 0.000 | 0.135 | False | Use metrics like completion time, utilization, and resource usage per job to determine... |
| llada-8b-instruct-hf | plan_536 | False | random_32 | 2.132 | 0.866 | 1.000 | 0.000 | 0.270 | False | Use a fair preemptive scheduling algorithm to to prioritize long-running jobs while ens... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_441 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.342 | 0.000 | 0.000 | 0.000 | 0.360 | 0.360 | 0.360 | 0.000 | 0.360 | 0.000 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_442 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.285 | 0.000 | 0.065 | 0.065 | 0.197 | 0.197 | 0.315 | 0.000 | 0.376 | 0.062 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_443 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.284 | 0.000 | 0.040 | 0.040 | 0.290 | 0.286 | 0.286 | 0.000 | 0.302 | 0.016 | 0.302 | 0.000 |
| llada-8b-instruct-hf | plan_444 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.341 | 0.000 | 0.075 | 0.075 | 0.315 | 0.339 | 0.315 | 0.000 | 0.381 | 0.066 | 0.381 | 0.000 |
| llada-8b-instruct-hf | plan_445 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.369 | 0.000 | 0.000 | 0.000 | 0.273 | 0.349 | 0.349 | 0.000 | 0.349 | 0.000 | 0.349 | 0.000 |
| llada-8b-instruct-hf | plan_446 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.377 | 0.000 | 0.000 | 0.000 | 0.399 | 0.399 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_447 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.359 | 0.000 | 0.075 | 0.075 | 0.461 | 0.344 | 0.461 | 0.000 | 0.499 | 0.037 | 0.499 | 0.000 |
| llada-8b-instruct-hf | plan_448 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.347 | 0.000 | 0.064 | 0.064 | 0.260 | 0.326 | 0.326 | 0.000 | 0.364 | 0.037 | 0.364 | 0.000 |
| llada-8b-instruct-hf | plan_449 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.316 | 0.000 | 0.046 | 0.046 | 0.283 | 0.274 | 0.283 | 0.000 | 0.304 | 0.021 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_450 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.401 | 0.000 | 0.000 | 0.000 | 0.376 | 0.451 | 0.451 | 0.000 | 0.451 | 0.000 | 0.451 | 0.000 |
| llada-8b-instruct-hf | plan_451 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.322 | 0.000 | 0.000 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_452 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.408 | 0.000 | 0.000 | 0.000 | 0.554 | 0.280 | 0.554 | 0.000 | 0.554 | 0.000 | 0.554 | 0.000 |
| llada-8b-instruct-hf | plan_453 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.280 | 0.000 | 0.034 | 0.034 | 0.301 | 0.301 | 0.301 | 0.000 | 0.294 | -0.008 | 0.301 | 0.008 |
| llada-8b-instruct-hf | plan_454 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.385 | 0.000 | 0.000 | 0.000 | 0.383 | 0.383 | 0.403 | 0.000 | 0.403 | 0.000 | 0.403 | 0.000 |
| llada-8b-instruct-hf | plan_455 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.318 | 0.000 | 0.000 | 0.000 | 0.281 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_456 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.351 | 0.000 | 0.000 | 0.000 | 0.531 | 0.435 | 0.531 | 0.000 | 0.531 | 0.000 | 0.531 | 0.000 |
| llada-8b-instruct-hf | plan_457 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.362 | 0.000 | 0.000 | 0.000 | 0.336 | 0.260 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-8b-instruct-hf | plan_458 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.355 | 0.000 | 0.086 | 0.086 | 0.355 | 0.414 | 0.414 | 0.000 | 0.444 | 0.030 | 0.444 | 0.000 |
| llada-8b-instruct-hf | plan_459 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.304 | 0.000 | 0.083 | 0.083 | 0.260 | 0.301 | 0.260 | 0.000 | 0.319 | 0.059 | 0.319 | 0.000 |
| llada-8b-instruct-hf | plan_460 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.348 | 0.000 | 0.061 | 0.061 | 0.301 | 0.301 | 0.301 | 0.000 | 0.339 | 0.038 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_461 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.325 | 0.000 | 0.000 | 0.000 | 0.301 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_462 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.306 | 0.000 | 0.044 | 0.044 | 0.281 | 0.281 | 0.281 | 0.000 | 0.323 | 0.041 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_463 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.361 | 0.000 | 0.000 | 0.000 | 0.260 | 0.280 | 0.260 | 0.000 | 0.260 | 0.000 | 0.280 | 0.020 |
| llada-8b-instruct-hf | plan_464 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.340 | 0.000 | 0.072 | 0.072 | 0.344 | 0.344 | 0.344 | 0.000 | 0.387 | 0.043 | 0.387 | 0.000 |
| llada-8b-instruct-hf | plan_465 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.312 | 0.000 | 0.000 | 0.000 | 0.253 | 0.253 | 0.253 | 0.000 | 0.253 | 0.000 | 0.253 | 0.000 |
| llada-8b-instruct-hf | plan_466 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.379 | 0.000 | 0.000 | 0.000 | 0.324 | 0.324 | 0.463 | 0.000 | 0.463 | 0.000 | 0.463 | 0.000 |
| llada-8b-instruct-hf | plan_467 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.303 | 0.000 | 0.042 | 0.042 | 0.260 | 0.261 | 0.260 | 0.000 | 0.281 | 0.021 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_468 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.313 | 0.000 | 0.000 | 0.000 | 0.304 | 0.304 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_469 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.337 | 0.000 | 0.087 | 0.087 | 0.323 | 0.303 | 0.323 | 0.000 | 0.382 | 0.059 | 0.382 | 0.000 |
| llada-8b-instruct-hf | plan_470 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.337 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_471 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.308 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.301 | 0.021 |
| llada-8b-instruct-hf | plan_472 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.367 | 0.000 | 0.042 | 0.042 | 0.240 | 0.302 | 0.240 | 0.000 | 0.281 | 0.041 | 0.302 | 0.021 |
| llada-8b-instruct-hf | plan_473 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.337 | 0.000 | 0.071 | 0.071 | 0.356 | 0.378 | 0.356 | 0.000 | 0.398 | 0.041 | 0.398 | 0.000 |
| llada-8b-instruct-hf | plan_474 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.332 | 0.000 | 0.043 | 0.043 | 0.339 | 0.339 | 0.339 | 0.000 | 0.356 | 0.017 | 0.378 | 0.021 |
| llada-8b-instruct-hf | plan_475 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.255 | 0.000 | 0.063 | 0.063 | 0.318 | 0.297 | 0.318 | 0.000 | 0.355 | 0.037 | 0.355 | 0.000 |
| llada-8b-instruct-hf | plan_476 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.357 | 0.000 | 0.000 | 0.000 | 0.326 | 0.326 | 0.326 | 0.000 | 0.326 | 0.000 | 0.326 | 0.000 |
| llada-8b-instruct-hf | plan_477 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.336 | 0.000 | 0.069 | 0.069 | 0.323 | 0.323 | 0.323 | 0.000 | 0.365 | 0.042 | 0.365 | 0.000 |
| llada-8b-instruct-hf | plan_478 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.359 | 0.000 | 0.000 | 0.000 | 0.280 | 0.431 | 0.431 | 0.000 | 0.431 | 0.000 | 0.431 | 0.000 |
| llada-8b-instruct-hf | plan_479 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.350 | 0.000 | 0.074 | 0.074 | 0.261 | 0.381 | 0.381 | 0.000 | 0.419 | 0.037 | 0.419 | 0.000 |
| llada-8b-instruct-hf | plan_480 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.270 | 0.000 | 0.085 | 0.085 | 0.323 | 0.323 | 0.301 | 0.000 | 0.360 | 0.059 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_481 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.332 | 0.000 | 0.000 | 0.000 | 0.314 | 0.280 | 0.314 | 0.000 | 0.314 | 0.000 | 0.314 | 0.000 |
| llada-8b-instruct-hf | plan_482 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.282 | 0.000 | 0.032 | 0.032 | 0.280 | 0.280 | 0.260 | 0.000 | 0.273 | 0.013 | 0.280 | 0.008 |
| llada-8b-instruct-hf | plan_483 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.318 | 0.000 | 0.000 | 0.000 | 0.376 | 0.376 | 0.376 | 0.000 | 0.376 | 0.000 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_484 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.326 | 0.000 | 0.000 | 0.000 | 0.376 | 0.376 | 0.376 | 0.000 | 0.376 | 0.000 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_485 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.347 | 0.000 | 0.042 | 0.042 | 0.280 | 0.280 | 0.280 | 0.000 | 0.301 | 0.021 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_486 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.320 | 0.000 | 0.000 | 0.000 | 0.301 | 0.301 | 0.376 | 0.000 | 0.376 | 0.000 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_487 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.287 | 0.000 | 0.000 | 0.000 | 0.370 | 0.315 | 0.315 | 0.000 | 0.315 | 0.000 | 0.370 | 0.055 |
| llada-8b-instruct-hf | plan_488 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.288 | 0.000 | 0.042 | 0.042 | 0.240 | 0.240 | 0.240 | 0.000 | 0.281 | 0.041 | 0.323 | 0.041 |
| llada-8b-instruct-hf | plan_489 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.301 | 0.000 | 0.085 | 0.085 | 0.324 | 0.281 | 0.324 | 0.000 | 0.379 | 0.055 | 0.379 | 0.000 |
| llada-8b-instruct-hf | plan_490 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.325 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_491 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.345 | 0.000 | 0.000 | 0.000 | 0.323 | 0.335 | 0.323 | 0.000 | 0.323 | 0.000 | 0.335 | 0.013 |
| llada-8b-instruct-hf | plan_492 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.305 | 0.000 | 0.000 | 0.000 | 0.314 | 0.281 | 0.314 | 0.000 | 0.314 | 0.000 | 0.314 | 0.000 |
| llada-8b-instruct-hf | plan_493 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.331 | 0.000 | 0.000 | 0.000 | 0.323 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_494 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.284 | 0.000 | 0.092 | 0.092 | 0.274 | 0.274 | 0.274 | 0.000 | 0.338 | 0.064 | 0.339 | 0.001 |
| llada-8b-instruct-hf | plan_495 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.338 | 0.000 | 0.124 | 0.124 | 0.381 | 0.376 | 0.376 | 0.000 | 0.494 | 0.117 | 0.494 | 0.000 |
| llada-8b-instruct-hf | plan_496 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.317 | 0.000 | 0.049 | 0.049 | 0.301 | 0.301 | 0.360 | 0.000 | 0.424 | 0.064 | 0.424 | 0.000 |
| llada-8b-instruct-hf | plan_497 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.368 | 0.000 | 0.000 | 0.000 | 0.351 | 0.280 | 0.351 | 0.000 | 0.351 | 0.000 | 0.351 | 0.000 |
| llada-8b-instruct-hf | plan_498 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.370 | 0.000 | 0.000 | 0.000 | 0.339 | 0.339 | 0.339 | 0.000 | 0.339 | 0.000 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_499 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.278 | 0.000 | 0.000 | 0.000 | 0.280 | 0.177 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_500 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.288 | 0.000 | 0.278 | 0.278 | 0.339 | 0.323 | 0.339 | 0.000 | 0.604 | 0.265 | 0.604 | 0.000 |
| llada-8b-instruct-hf | plan_501 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.335 | 0.000 | 0.000 | 0.000 | 0.303 | 0.303 | 0.340 | 0.000 | 0.340 | 0.000 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_502 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.316 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_503 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.381 | 0.000 | 0.092 | 0.092 | 0.344 | 0.355 | 0.355 | 0.000 | 0.419 | 0.064 | 0.419 | 0.000 |
| llada-8b-instruct-hf | plan_504 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.318 | 0.000 | 0.000 | 0.000 | 0.299 | 0.065 | 0.299 | 0.000 | 0.299 | 0.000 | 0.339 | 0.040 |
| llada-8b-instruct-hf | plan_505 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.328 | 0.000 | 0.065 | 0.065 | 0.419 | 0.339 | 0.339 | 0.000 | 0.376 | 0.037 | 0.419 | 0.042 |
| llada-8b-instruct-hf | plan_506 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.332 | 0.000 | 0.000 | 0.000 | 0.310 | 0.310 | 0.310 | 0.000 | 0.310 | 0.000 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_507 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.339 | 0.000 | 0.124 | 0.124 | 0.280 | 0.280 | 0.280 | 0.000 | 0.376 | 0.096 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_508 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.392 | 0.000 | 0.000 | 0.000 | 0.323 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_509 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.300 | 0.000 | 0.044 | 0.044 | 0.355 | 0.355 | 0.301 | 0.000 | 0.323 | 0.021 | 0.355 | 0.032 |
| llada-8b-instruct-hf | plan_510 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.337 | 0.000 | 0.000 | 0.000 | 0.277 | 0.321 | 0.321 | 0.000 | 0.321 | 0.000 | 0.407 | 0.086 |
| llada-8b-instruct-hf | plan_511 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.336 | 0.000 | 0.000 | 0.000 | 0.376 | 0.292 | 0.376 | 0.000 | 0.376 | 0.000 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_512 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.293 | 0.000 | 0.042 | 0.042 | 0.301 | 0.301 | 0.280 | 0.000 | 0.301 | 0.021 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_513 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.310 | 0.000 | 0.000 | 0.000 | 0.285 | 0.285 | 0.335 | 0.000 | 0.335 | 0.000 | 0.335 | 0.000 |
| llada-8b-instruct-hf | plan_514 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.253 | 0.000 | 0.044 | 0.044 | 0.261 | 0.227 | 0.261 | 0.000 | 0.323 | 0.061 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_515 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.319 | 0.000 | 0.041 | 0.041 | 0.390 | 0.207 | 0.390 | 0.000 | 0.383 | -0.008 | 0.390 | 0.008 |
| llada-8b-instruct-hf | plan_516 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.304 | 0.000 | 0.072 | 0.072 | 0.356 | 0.356 | 0.356 | 0.000 | 0.379 | 0.023 | 0.379 | 0.000 |
| llada-8b-instruct-hf | plan_517 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.324 | 0.000 | 0.000 | 0.000 | 0.344 | 0.344 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_518 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.297 | 0.000 | 0.000 | 0.000 | 0.355 | 0.355 | 0.355 | 0.000 | 0.355 | 0.000 | 0.376 | 0.021 |
| llada-8b-instruct-hf | plan_519 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.372 | 0.000 | 0.000 | 0.000 | 0.297 | 0.318 | 0.297 | 0.000 | 0.297 | 0.000 | 0.318 | 0.020 |
| llada-8b-instruct-hf | plan_520 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.352 | 0.000 | 0.064 | 0.064 | 0.323 | 0.323 | 0.323 | 0.000 | 0.360 | 0.037 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_521 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.331 | 0.000 | 0.064 | 0.064 | 0.303 | 0.303 | 0.303 | 0.000 | 0.378 | 0.075 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_522 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.309 | 0.000 | 0.041 | 0.041 | 0.301 | 0.318 | 0.318 | 0.000 | 0.335 | 0.018 | 0.335 | 0.000 |
| llada-8b-instruct-hf | plan_523 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.302 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_524 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.354 | 0.000 | 0.085 | 0.085 | 0.501 | 0.469 | 0.501 | 0.000 | 0.524 | 0.022 | 0.524 | 0.000 |
| llada-8b-instruct-hf | plan_525 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.250 | 0.000 | 0.000 | 0.000 | 0.280 | 0.238 | 0.238 | 0.000 | 0.238 | 0.000 | 0.280 | 0.042 |
| llada-8b-instruct-hf | plan_526 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.315 | 0.000 | 0.000 | 0.000 | 0.301 | 0.240 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_527 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.237 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_528 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.317 | 0.000 | 0.000 | 0.000 | 0.340 | 0.340 | 0.340 | 0.000 | 0.340 | 0.000 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_529 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.385 | 0.000 | 0.074 | 0.074 | 0.398 | 0.423 | 0.423 | 0.000 | 0.457 | 0.034 | 0.457 | 0.000 |
| llada-8b-instruct-hf | plan_530 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.330 | 0.000 | 0.000 | 0.000 | 0.301 | 0.260 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_531 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.330 | 0.000 | 0.127 | 0.127 | 0.335 | 0.301 | 0.301 | 0.000 | 0.399 | 0.097 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_532 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.307 | 0.000 | 0.047 | 0.047 | 0.319 | 0.319 | 0.319 | 0.000 | 0.340 | 0.021 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_533 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.333 | 0.000 | 0.000 | 0.000 | 0.260 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_534 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.323 | 0.000 | 0.000 | 0.000 | 0.376 | 0.376 | 0.376 | 0.000 | 0.376 | 0.000 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_535 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.313 | 0.000 | 0.053 | 0.053 | 0.397 | 0.356 | 0.356 | 0.000 | 0.381 | 0.025 | 0.410 | 0.029 |
| llada-8b-instruct-hf | plan_536 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.316 | 0.000 | 0.000 | 0.000 | 0.301 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
