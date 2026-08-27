# Diffusion Schedule-Selection Benchmark Report

Full model generations: `472`
Counterfactual probe generations: `0`
Arm selections: `480`
Run ID: `diffusion-67ae3786920e3638`
Content hash: `67ae3786920e36388a22cc6133924e3f6181a92aa39811f1a889816ef41f0ce6`
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
History mutability: `monotonic 472/472, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.010`
Trajectory task delta vs random: `0.054`
Trajectory wins/ties/losses vs fixed: `19/96/5`
Trajectory wins/ties/losses vs random: `49/68/3`
Oracle generation budget/task: `3.93`
Oracle task score: `0.291`
Oracle headroom vs trajectory: `0.023`
Oracle wins/ties/losses vs trajectory: `48/72/0`
Selector regret vs trajectory: `0.023 over 48/120 improvable`
Repair arm coverage: `120/120` overall
Repair eligible coverage: `120/120`
Repair task delta vs fixed: `0.029`
Repair task delta vs random: `0.074`
Repair task delta vs trajectory: `0.019`
Repair task delta vs evolved: `0.019`
Repair generation budget delta vs evolved: `1.93`
Repair task delta per extra generation vs evolved: `0.010`
Repair wins/ties/losses vs evolved: `38/80/2`
Oracle headroom vs repair: `0.004`
Oracle wins/ties/losses vs repair: `15/105/0`
Selector regret vs repair: `0.004 over 15/120 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `120/120` overall, `120/120` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.257484 | 0.000000 | 0.044636 | - | - |
| random perturbation | repair-covered tasks | 0.212848 | -0.044636 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.286724 | 0.029240 | 0.073876 | 49/68/3 | 73/44/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 120 | 1.00 | 0.257 | 0.595 | 0.342 |
| random | 120 | 1.00 | 0.213 | 0.467 | 0.276 |
| trajectory_selected | 120 | 2.00 | 0.267 | 0.593 | 0.349 |
| repair_selected | 120 | 3.93 | 0.287 | 0.616 | 0.369 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 120 | 1.00 | 0.257 | 0.595 | 0.342 |
| planning | random | 120 | 1.00 | 0.213 | 0.467 | 0.276 |
| planning | trajectory_selected | 120 | 2.00 | 0.267 | 0.593 | 0.349 |
| planning | repair_selected | 120 | 3.93 | 0.287 | 0.616 | 0.369 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_537 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 238 | True | 12 | 0.020 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_538 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 218 | True | 12 | 0.091 | True | True | 4.000 | 0.125 | 0.061 | 0.061 |
| llada-8b-instruct-hf | plan_539 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 345 | True | 12 | 0.262 | True | True | 4.000 | 0.125 | 0.066 | 0.066 |
| llada-8b-instruct-hf | plan_540 | random_32 | True | denoise_phase_repairable | False |  | 0.389 | 0.251 | 307 | True | 12 | 0.221 | True | True | 3.000 | 0.094 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_541 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.292 | 0.193 | 301 | True | 12 | 0.349 | True | True | 3.000 | 0.094 | 0.023 | 0.023 |
| llada-8b-instruct-hf | plan_542 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 374 | True | 12 | 0.240 | True | True | 3.000 | 0.094 | 0.030 | 0.030 |
| llada-8b-instruct-hf | plan_543 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.373 | 0.273 | 345 | True | 12 | 0.317 | True | True | 3.000 | 0.094 | 0.024 | 0.024 |
| llada-8b-instruct-hf | plan_544 | random_32 | True | denoise_phase_repairable | False |  | 0.314 | 0.214 | 303 | True | 12 | 0.286 | True | True | 3.000 | 0.094 | 0.032 | 0.032 |
| llada-8b-instruct-hf | plan_545 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.314 | 0.214 | 283 | True | 12 | 0.197 | True | True | 3.000 | 0.094 | 0.026 | 0.026 |
| llada-8b-instruct-hf | plan_546 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.292 | 0.193 | 290 | True | 12 | 0.275 | True | True | 3.000 | 0.094 | 0.033 | 0.033 |
| llada-8b-instruct-hf | plan_547 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.117 | 0.117 | 121 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_548 | random_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 278 | True | 12 | 0.193 | True | True | 5.000 | 0.156 | 0.024 | 0.024 |
| llada-8b-instruct-hf | plan_549 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 300 | True | 12 | 0.338 | True | True | 4.000 | 0.125 | 0.059 | 0.059 |
| llada-8b-instruct-hf | plan_550 | random_32 | True | denoise_phase_repairable | False |  | 0.378 | 0.278 | 268 | True | 12 | 0.167 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_551 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.273 | 0.193 | 257 | True | 12 | 0.250 | True | True | 4.000 | 0.125 | 0.050 | 0.050 |
| llada-8b-instruct-hf | plan_552 | random_32 | True | denoise_phase_repairable | False |  | 0.481 | 0.344 | 322 | True | 12 | 0.261 | True | True | 4.000 | 0.125 | 0.034 | 0.034 |
| llada-8b-instruct-hf | plan_553 | random_32 | True | denoise_phase_repairable | False |  | 0.145 | 0.045 | 51 | True | 12 | 0.062 | True | True | 16.000 | 0.500 | 0.062 | 0.062 |
| llada-8b-instruct-hf | plan_554 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.431 | 0.294 | 359 | True | 12 | 0.133 | True | True | 4.000 | 0.125 | 0.011 | 0.011 |
| llada-8b-instruct-hf | plan_555 | random_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 243 | True | 12 | 0.106 | True | True | 5.000 | 0.156 | 0.015 | 0.015 |
| llada-8b-instruct-hf | plan_556 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.394 | 0.314 | 330 | True | 12 | 0.271 | True | True | 4.000 | 0.125 | 0.024 | 0.024 |
| llada-8b-instruct-hf | plan_557 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 333 | True | 12 | 0.179 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_558 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.306 | 0.226 | 299 | True | 12 | 0.279 | True | True | 3.000 | 0.094 | 0.035 | 0.035 |
| llada-8b-instruct-hf | plan_559 | random_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 363 | True | 12 | 0.212 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_560 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 338 | True | 12 | 0.312 | True | True | 4.000 | 0.125 | 0.026 | 0.026 |
| llada-8b-instruct-hf | plan_561 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 344 | True | 12 | 0.431 | True | True | 4.000 | 0.125 | 0.046 | 0.046 |
| llada-8b-instruct-hf | plan_562 | random_32 | True | denoise_phase_repairable | False |  | 0.330 | 0.230 | 379 | True | 12 | 0.435 | True | True | 3.000 | 0.094 | 0.032 | 0.032 |
| llada-8b-instruct-hf | plan_563 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.292 | 0.193 | 331 | True | 12 | 0.227 | True | True | 4.000 | 0.125 | 0.045 | 0.045 |
| llada-8b-instruct-hf | plan_564 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 283 | True | 12 | 0.129 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_565 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.217 | 203 | True | 12 | 0.191 | True | True | 3.000 | 0.094 | 0.029 | 0.029 |
| llada-8b-instruct-hf | plan_566 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 301 | True | 12 | 0.324 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_567 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 260 | True | 12 | 0.061 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_568 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 228 | True | 12 | 0.000 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_569 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 338 | True | 12 | 0.154 | True | True | 4.000 | 0.125 | 0.046 | 0.046 |
| llada-8b-instruct-hf | plan_570 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 351 | True | 12 | 0.227 | True | True | 4.000 | 0.125 | 0.015 | 0.015 |
| llada-8b-instruct-hf | plan_571 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 283 | True | 12 | 0.013 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_572 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.180 | 0.180 | 300 | True | 12 | 0.051 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_573 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 373 | True | 12 | 0.120 | True | True | 4.000 | 0.125 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_574 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.273 | 0.193 | 358 | True | 12 | 0.312 | True | True | 4.000 | 0.125 | 0.050 | 0.050 |
| llada-8b-instruct-hf | plan_575 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.045 | 0.045 | 42 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_576 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 249 | True | 12 | 0.027 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_577 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.137 | 0.117 | 83 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_578 | random_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 323 | True | 12 | 0.095 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_579 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.180 | 0.180 | 164 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_580 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 270 | True | 12 | 0.035 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_581 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 223 | True | 12 | 0.024 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_582 | random_32 | True | denoise_phase_repairable | False |  | 0.177 | 0.117 | 82 | True | 12 | 0.080 | True | True | 5.000 | 0.156 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_583 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.160 | 144 | True | 12 | 0.230 | True | True | 5.000 | 0.156 | 0.049 | 0.049 |
| llada-8b-instruct-hf | plan_584 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 255 | True | 12 | 0.029 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_585 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.201 | 0.201 | 252 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_586 | low_confidence_32 | False | no_repairable_denoise_skeleton | False |  | 0.045 | 0.045 | 1 | True | 12 | 0.000 | True | False | none | none | none | 0.000 |
| llada-8b-instruct-hf | plan_587 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 262 | True | 12 | 0.000 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_588 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 320 | True | 12 | 0.215 | True | True | 4.000 | 0.125 | 0.022 | 0.022 |
| llada-8b-instruct-hf | plan_589 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 238 | True | 12 | 0.013 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_590 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.201 | 0.201 | 246 | True | 12 | 0.013 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_591 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.473 | 0.373 | 363 | True | 12 | 0.196 | True | True | 3.000 | 0.094 | 0.021 | 0.021 |
| llada-8b-instruct-hf | plan_592 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.217 | 305 | True | 12 | 0.284 | True | True | 4.000 | 0.125 | 0.023 | 0.023 |
| llada-8b-instruct-hf | plan_593 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.351 | 0.251 | 293 | True | 12 | 0.301 | True | True | 3.000 | 0.094 | 0.041 | 0.041 |
| llada-8b-instruct-hf | plan_594 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 288 | True | 12 | 0.197 | True | True | 3.000 | 0.094 | 0.042 | 0.042 |
| llada-8b-instruct-hf | plan_595 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 265 | True | 12 | 0.213 | True | True | 4.000 | 0.125 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_596 | random_32 | True | denoise_phase_repairable | False |  | 0.335 | 0.235 | 310 | True | 12 | 0.202 | True | True | 3.000 | 0.094 | 0.024 | 0.024 |
| llada-8b-instruct-hf | plan_597 | random_32 | True | denoise_phase_repairable | False |  | 0.157 | 0.117 | 109 | True | 12 | 0.066 | True | True | 4.000 | 0.125 | 0.026 | 0.026 |
| llada-8b-instruct-hf | plan_598 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 341 | True | 12 | 0.057 | True | True | 4.000 | 0.125 | 0.029 | 0.029 |
| llada-8b-instruct-hf | plan_599 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.117 | 0.117 | 119 | True | 12 | 0.000 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_600 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 257 | True | 12 | 0.187 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_601 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.117 | 0.117 | 82 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_602 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.045 | 0.045 | 25 | True | 12 | 0.000 | True | True | 32.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_603 | random_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 339 | True | 12 | 0.286 | True | True | 3.000 | 0.094 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_604 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 314 | True | 12 | 0.218 | True | True | 3.000 | 0.094 | 0.011 | 0.011 |
| llada-8b-instruct-hf | plan_605 | random_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 366 | True | 12 | 0.338 | True | True | 2.000 | 0.062 | 0.041 | 0.041 |
| llada-8b-instruct-hf | plan_606 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.045 | 0.045 | 42 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_607 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.217 | 302 | True | 12 | 0.247 | True | True | 4.000 | 0.125 | 0.022 | 0.022 |
| llada-8b-instruct-hf | plan_608 | random_32 | True | denoise_phase_repairable | False |  | 0.471 | 0.329 | 307 | True | 12 | 0.195 | True | True | 3.000 | 0.094 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_609 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.180 | 0.180 | 165 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_610 | random_32 | True | denoise_phase_repairable | False |  | 0.399 | 0.256 | 288 | True | 12 | 0.195 | True | True | 5.000 | 0.156 | 0.013 | 0.013 |
| llada-8b-instruct-hf | plan_611 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.370 | 0.290 | 346 | True | 12 | 0.247 | True | True | 3.000 | 0.094 | 0.035 | 0.035 |
| llada-8b-instruct-hf | plan_612 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.427 | 0.310 | 352 | True | 12 | 0.179 | True | True | 5.000 | 0.156 | 0.009 | 0.009 |
| llada-8b-instruct-hf | plan_613 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 239 | True | 12 | 0.012 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_614 | random_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 236 | True | 12 | 0.244 | True | True | 5.000 | 0.156 | 0.024 | 0.024 |
| llada-8b-instruct-hf | plan_615 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 320 | True | 12 | 0.250 | True | True | 4.000 | 0.125 | 0.042 | 0.042 |
| llada-8b-instruct-hf | plan_616 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 286 | True | 12 | 0.073 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_617 | low_confidence_32 | False | no_repairable_denoise_skeleton | False |  | 0.045 | 0.045 | 1 | True | 12 | 0.000 | True | False | none | none | none | 0.000 |
| llada-8b-instruct-hf | plan_618 | random_32 | True | denoise_phase_repairable | False |  | 0.105 | 0.045 | 62 | True | 12 | 0.056 | True | True | 4.000 | 0.125 | 0.014 | 0.014 |
| llada-8b-instruct-hf | plan_619 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.365 | 0.265 | 379 | True | 12 | 0.370 | True | True | 3.000 | 0.094 | 0.014 | 0.014 |
| llada-8b-instruct-hf | plan_620 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 328 | True | 12 | 0.150 | True | True | 4.000 | 0.125 | 0.033 | 0.033 |
| llada-8b-instruct-hf | plan_621 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 315 | True | 12 | 0.094 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_622 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 373 | True | 12 | 0.310 | True | True | 3.000 | 0.094 | 0.034 | 0.034 |
| llada-8b-instruct-hf | plan_623 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 404 | True | 12 | 0.267 | True | True | 4.000 | 0.125 | 0.050 | 0.050 |
| llada-8b-instruct-hf | plan_624 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 333 | True | 12 | 0.149 | True | True | 4.000 | 0.125 | 0.015 | 0.015 |
| llada-8b-instruct-hf | plan_625 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.045 | 0.045 | 42 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_626 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 330 | True | 12 | 0.014 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_627 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 316 | True | 12 | 0.022 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_628 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 374 | True | 12 | 0.292 | True | True | 4.000 | 0.125 | 0.019 | 0.019 |
| llada-8b-instruct-hf | plan_629 | random_32 | True | denoise_phase_repairable | False |  | 0.346 | 0.244 | 334 | True | 12 | 0.146 | True | True | 3.000 | 0.094 | 0.021 | 0.021 |
| llada-8b-instruct-hf | plan_630 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.217 | 359 | True | 12 | 0.126 | True | True | 5.000 | 0.156 | 0.029 | 0.029 |
| llada-8b-instruct-hf | plan_631 | random_32 | True | denoise_phase_repairable | False |  | 0.409 | 0.286 | 279 | True | 12 | 0.211 | True | True | 4.000 | 0.125 | 0.011 | 0.011 |
| llada-8b-instruct-hf | plan_632 | random_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 326 | True | 12 | 0.267 | True | True | 4.000 | 0.125 | 0.019 | 0.019 |
| llada-8b-instruct-hf | plan_633 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 368 | True | 12 | 0.186 | True | True | 3.000 | 0.094 | 0.017 | 0.017 |
| llada-8b-instruct-hf | plan_634 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.345 | 0.223 | 243 | True | 12 | 0.057 | True | True | 6.000 | 0.188 | 0.010 | 0.010 |
| llada-8b-instruct-hf | plan_635 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 302 | True | 12 | 0.255 | True | True | 4.000 | 0.125 | 0.018 | 0.018 |
| llada-8b-instruct-hf | plan_636 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 335 | True | 12 | 0.198 | True | True | 4.000 | 0.125 | 0.041 | 0.041 |
| llada-8b-instruct-hf | plan_637 | low_confidence_32 | False | no_repairable_denoise_skeleton | False |  | 0.045 | 0.045 | 1 | True | 12 | 0.000 | True | False | none | none | none | 0.000 |
| llada-8b-instruct-hf | plan_638 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 249 | True | 12 | 0.030 | True | True | 5.000 | 0.156 | 0.015 | 0.015 |
| llada-8b-instruct-hf | plan_639 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.386 | 0.244 | 329 | True | 12 | 0.312 | True | True | 4.000 | 0.125 | 0.039 | 0.039 |
| llada-8b-instruct-hf | plan_640 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 409 | True | 12 | 0.358 | True | True | 4.000 | 0.125 | 0.030 | 0.030 |
| llada-8b-instruct-hf | plan_641 | low_confidence_32 | False | no_repairable_denoise_skeleton | False |  | 0.045 | 0.045 | 1 | True | 12 | 0.000 | True | False | none | none | none | 0.000 |
| llada-8b-instruct-hf | plan_642 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.242 | 0.223 | 338 | True | 12 | 0.090 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_643 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 321 | True | 12 | 0.096 | True | True | 5.000 | 0.156 | 0.027 | 0.027 |
| llada-8b-instruct-hf | plan_644 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 357 | True | 12 | 0.200 | True | True | 4.000 | 0.125 | 0.027 | 0.027 |
| llada-8b-instruct-hf | plan_645 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.045 | 0.045 | 42 | True | 12 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_646 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.424 | 0.281 | 387 | True | 12 | 0.208 | True | True | 4.000 | 0.125 | 0.028 | 0.028 |
| llada-8b-instruct-hf | plan_647 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.302 | 0.223 | 302 | True | 12 | 0.267 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_648 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 302 | True | 12 | 0.218 | True | True | 4.000 | 0.125 | 0.036 | 0.036 |
| llada-8b-instruct-hf | plan_649 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 251 | True | 12 | 0.000 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_650 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 266 | True | 12 | 0.319 | True | True | 6.000 | 0.188 | 0.043 | 0.043 |
| llada-8b-instruct-hf | plan_651 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.365 | 0.223 | 235 | True | 12 | 0.299 | True | True | 3.000 | 0.094 | 0.030 | 0.030 |
| llada-8b-instruct-hf | plan_652 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 401 | True | 12 | 0.328 | True | True | 3.000 | 0.094 | 0.047 | 0.047 |
| llada-8b-instruct-hf | plan_653 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 390 | True | 12 | 0.393 | True | True | 3.000 | 0.094 | 0.033 | 0.033 |
| llada-8b-instruct-hf | plan_654 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 320 | True | 12 | 0.102 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_655 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 300 | True | 12 | 0.104 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_656 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 381 | True | 12 | 0.290 | True | True | 4.000 | 0.125 | 0.032 | 0.032 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 116 | 18 | low_confidence_32,random_32 | final | 32.5 | 0.991 | 0.009 | 0.000 | 0.026 | 0.026 | -0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 25/61/30 | 0.276 | 0.631 | 0.365 |
| history_prefix_25_repair | 116 | 22 | low_confidence_32,random_32 | history | 48.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | -0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 31/41/44 | 0.267 | 0.645 | 0.361 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_537 | False | low_confidence_32 | 3.388 | 1.000 | 1.000 | 0.000 | 0.000 | False | I'm here to help. |
| llada-8b-instruct-hf | plan_538 | False | low_confidence_32 | 1.432 | 0.905 | 1.000 | 0.000 | 0.061 | False | Month 4-6: Develop Feature B with 4 developer-months. |
| llada-8b-instruct-hf | plan_538 | False | low_confidence_32 | 1.432 | 0.905 | 1.000 | 0.000 | 0.061 | False | Month 7-8: Develop Feature C with 3 developer-months. |
| llada-8b-instruct-hf | plan_538 | False | low_confidence_32 | 2.130 | 0.799 | 1.000 | 0.000 | 0.076 | False | Month 9: Address technical debt with 8 developer-months. |
| llada-8b-instruct-hf | plan_539 | False | low_confidence_32 | 2.670 | 0.907 | 1.000 | 0.000 | 0.115 | False | This strategy will reduce the disruption of component X by 50% and provide sufficient s... |
| llada-8b-instruct-hf | plan_540 | False | random_32 | 2.027 | 0.620 | 1.000 | 0.000 | 0.078 | False | This will prevent further decline in productivity, reduce the risk of turnover, and ens... |
| llada-8b-instruct-hf | plan_541 | False | low_confidence_32 | 2.058 | 0.705 | 1.000 | 0.000 | 0.140 | False | This option maintains current cluster costs, ensures data consistency, and has the pote... |
| llada-8b-instruct-hf | plan_542 | False | low_confidence_32 | 1.662 | 0.000 | 1.000 | 0.000 | 0.240 | False | **Implement a top-tier Consent Management Platform (CMP) to reduce data loss to 15% acr... |
| llada-8b-instruct-hf | plan_543 | False | low_confidence_32 | 1.748 | 0.112 | 1.000 | 0.000 | 0.244 | False | While Vendor B offers a smaller annual cost saving compared to Vendor A, its open-sourc... |
| llada-8b-instruct-hf | plan_544 | False | random_32 | 1.842 | 0.279 | 1.000 | 0.000 | 0.206 | False | This will help to significantly reduce downtime to 0.5 hours/month and decrease the inc... |
| llada-8b-instruct-hf | plan_545 | False | low_confidence_32 | 1.926 | 0.415 | 1.000 | 0.000 | 0.105 | False | This option offers a higher MRR increase ($400k) with a lower upfront marketing cost an... |
| llada-8b-instruct-hf | plan_546 | False | low_confidence_32 | 2.014 | 0.629 | 1.000 | 0.000 | 0.154 | False | This option will increase overall productivity by 18% and reduce the defect rate to 0.2... |
| llada-8b-instruct-hf | plan_547 | False | low_confidence_32 | 3.365 | 0.950 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_548 | False | random_32 | 2.020 | 0.617 | 1.000 | 0.000 | 0.120 | False | 2. **Feature B: AI-Powered Recommendation Engine:** High user engagement, new user conv... |
| llada-8b-instruct-hf | plan_549 | False | low_confidence_32 | 2.600 | 0.823 | 1.000 | 0.000 | 0.191 | False | This strategy aims to reduce supply chain costs by 10% while maintaining 98% production... |
| llada-8b-instruct-hf | plan_550 | False | random_32 | 1.963 | 0.487 | 1.000 | 0.000 | 0.106 | False | Additionally, with the $1,500/month budget allocation for the GPU-enabled inference clu... |
| llada-8b-instruct-hf | plan_551 | True | low_confidence_32 | 1.930 | 0.443 | 1.000 | 0.000 | 0.175 | False | Ensure each engineer works no more than 45 hours per week to avoid burnout and maintain... |
| llada-8b-instruct-hf | plan_552 | False | random_32 | 1.884 | 0.862 | 1.000 | 0.000 | 0.091 | False | Use a separate archival system for transaction history to ensure it is deleted after 7... |
| llada-8b-instruct-hf | plan_552 | False | random_32 | 2.024 | 0.634 | 1.000 | 0.000 | 0.091 | False | Monitor data storage costs and prioritize for the database re-architecture to minimize... |
| llada-8b-instruct-hf | plan_553 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To plan the migration strategy for the API Gateway, |
| llada-8b-instruct-hf | plan_554 | False | low_confidence_32 | 1.402 | 0.875 | 1.000 | 0.000 | 0.056 | False | This option has a higher potential of growth and revenue, but also carries a higher risk. |
| llada-8b-instruct-hf | plan_555 | True | random_32 | 1.430 | 0.936 | 1.000 | 0.000 | 0.061 | False | Module A has a target BOM cost of $3 per unit, which is within the budget of $4. |
| llada-8b-instruct-hf | plan_555 | True | random_32 | 2.139 | 0.857 | 1.000 | 0.000 | 0.106 | False | In contrast, Module B costs $5 per unit, which is higher than the target BOM cost of un... |
| llada-8b-instruct-hf | plan_556 | False | low_confidence_32 | 1.269 | 0.638 | 1.000 | 0.000 | 0.071 | False | Let's start by analyzing the current policy and identifying the root causes of downtime... |
| llada-8b-instruct-hf | plan_556 | False | low_confidence_32 | 2.051 | 0.718 | 1.000 | 0.000 | 0.176 | False | Then, we can develop a plan to reduce patching-related downtime by 30% within 3 months... |
| llada-8b-instruct-hf | plan_557 | True | low_confidence_32 | 1.350 | 0.763 | 1.000 | 0.000 | 0.090 | False | It requires a detailed understanding of the application's architecture, current infrast... |
| llada-8b-instruct-hf | plan_557 | True | low_confidence_32 | 2.153 | 0.858 | 1.000 | 0.000 | 0.045 | False | Please feel out to your dedicated team or a cloud consultant for assistance with this t... |
| llada-8b-instruct-hf | plan_558 | False | low_confidence_32 | 2.381 | 0.362 | 1.000 | 0.000 | 0.221 | False | Then, allocate resources to Project B (Technical Debt) for the remaining 2 months, refa... |
| llada-8b-instruct-hf | plan_559 | False | random_32 | 1.995 | 0.563 | 1.000 | 0.000 | 0.100 | False | This combination will allow us to handle both real-time lookup and analytical queries,... |
| llada-8b-instruct-hf | plan_560 | False | low_confidence_32 | 1.391 | 0.873 | 1.000 | 0.000 | 0.117 | False | Consider training existing Python engineers in Rust to shorten the hiring process and r... |
| llada-8b-instruct-hf | plan_560 | False | low_confidence_32 | 1.335 | 0.769 | 1.000 | 0.000 | 0.130 | False | For Project Legacy, maintain 2 mid-level Java engineers to handle ongoing maintenance a... |
| llada-8b-instruct-hf | plan_560 | False | low_confidence_32 | 2.195 | 0.942 | 1.000 | 0.000 | 0.065 | False | Prioritize Project Nova to avoid the lost market opportunity. |
| llada-8b-instruct-hf | plan_561 | False | low_confidence_32 | 2.120 | 0.838 | 1.000 | 0.000 | 0.154 | False | Achieve a significant improvement in data quality to stabilize model performance within... |
| llada-8b-instruct-hf | plan_562 | False | random_32 | 2.498 | 0.879 | 1.000 | 0.000 | 0.113 | False | This approach will slightly reduce user opt-in rates and minimize the impact on direct... |
| llada-8b-instruct-hf | plan_562 | False | random_32 | 3.212 | 0.806 | 1.000 | 0.000 | 0.129 | False | The implementation effort will be approximately 5 person-months for engineering and leg... |
| llada-8b-instruct-hf | plan_563 | False | low_confidence_32 | 1.435 | 0.905 | 1.000 | 0.000 | 0.045 | False | However, I need some information to provide a comprehensive plan. |
| llada-8b-instruct-hf | plan_563 | False | low_confidence_32 | 1.287 | 0.636 | 1.000 | 0.000 | 0.106 | False | Can you provide me with the projected sales data for the next 6 months, including the h... |
| llada-8b-instruct-hf | plan_563 | False | low_confidence_32 | 2.224 | 1.000 | 1.000 | 0.000 | 0.061 | False | This information will help me develop an effective inventory management strategy. |
| llada-8b-instruct-hf | plan_564 | False | low_confidence_32 | 2.116 | 0.957 | 1.000 | 0.000 | 0.012 | False | This, as requested, is a the scope of the task. |
| llada-8b-instruct-hf | plan_564 | False | low_confidence_32 | 2.116 | 0.957 | 1.000 | 0.000 | 0.012 | False | However, I can help you design the architecture for the new. |
| llada-8b-instruct-hf | plan_564 | False | low_confidence_32 | 2.102 | 0.760 | 1.000 | 0.000 | 0.059 | False | Please provide the details of the legacy SOAP services and the two new REST microservices. |
| llada-8b-instruct-hf | plan_565 | False | low_confidence_32 | 1.384 | 0.800 | 1.000 | 0.000 | 0.059 | False | Rollout plan: - Month 1-2: Implement 'Pro' tier. |
| llada-8b-instruct-hf | plan_565 | False | low_confidence_32 | 1.414 | 0.858 | 1.000 | 0.000 | 0.044 | False | - Month 3-4: Implement 'Lite' tier. |
| llada-8b-instruct-hf | plan_565 | False | low_confidence_32 | 2.038 | 0.630 | 1.000 | 0.000 | 0.059 | False | - Month 5-6: Monitor and adjust pricing based on performance of new tiers. |
| llada-8b-instruct-hf | plan_566 | False | low_confidence_32 | 1.760 | 0.106 | 1.000 | 0.000 | 0.250 | False | This, despite, the higher infrastructure costs and additional FTEs, would achieve near-... |
| llada-8b-instruct-hf | plan_567 | False | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | Feel free to ask. |
| llada-8b-instruct-hf | plan_568 | False | low_confidence_32 | 3.738 | 1.000 | 1.000 | 0.000 | 0.000 | False | This I'm sorry, but I can't assist with that.. |
| llada-8b-instruct-hf | plan_568 | False | low_confidence_32 | 3.738 | 1.000 | 1.000 | 0.000 | 0.000 | False | This I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_568 | False | low_confidence_32 | 4.434 | 0.905 | 1.000 | 0.000 | 0.000 | False | If you have any other questions or need assistance with something else, feel free to ask. |
| llada-8b-instruct-hf | plan_569 | False | low_confidence_32 | 2.824 | 0.865 | 1.000 | 0.000 | 0.015 | False | Use cost management tools to optimize resource utilization and reduce costs. |
| llada-8b-instruct-hf | plan_570 | True | low_confidence_32 | 1.337 | 0.738 | 1.000 | 0.000 | 0.091 | False | This would allow for more efficient project delivery, reduced scope creep, and better P... |
| llada-8b-instruct-hf | plan_570 | True | low_confidence_32 | 2.066 | 0.691 | 1.000 | 0.000 | 0.091 | False | Additionally, invest in training and mentorship programs to improve career growth and w... |
| llada-8b-instruct-hf | plan_571 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | This I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_571 | False | low_confidence_32 | 2.854 | 0.956 | 1.000 | 0.000 | 0.013 | False | If you have any other questions or need assistance with something else, feel free to ask. |
| llada-8b-instruct-hf | plan_572 | True | low_confidence_32 | 3.387 | 1.000 | 1.000 | 0.000 | 0.017 | False | However If you need assistance with harmless planning, planning, math, or science quest... |
| llada-8b-instruct-hf | plan_573 | False | low_confidence_32 | 1.941 | 0.923 | 1.000 | 0.000 | 0.027 | False | Utilize the learning curve to improve the system's accuracy over time. |
| llada-8b-instruct-hf | plan_573 | False | low_confidence_32 | 2.607 | 0.968 | 1.000 | 0.000 | 0.013 | False | Train the team to use the new system and establish metrics to track accuracy. |
| llada-8b-instruct-hf | plan_573 | False | low_confidence_32 | 3.374 | 1.000 | 1.000 | 0.000 | 0.013 | False | Consider using machine learning algorithms to predict future demand and optimize invent... |
| llada-8b-instruct-hf | plan_574 | False | low_confidence_32 | 2.665 | 0.906 | 1.000 | 0.000 | 0.075 | False | This approach will for rapid expansion while minimizing the strain on the brand's reput... |
| llada-8b-instruct-hf | plan_575 | False | low_confidence_32 | 1.707 | 0.932 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that... |
| llada-8b-instruct-hf | plan_576 | False | low_confidence_32 | 2.086 | 0.923 | 1.000 | 0.000 | 0.014 | False | However you can ask me with any planning, math, logic, or science questions. |
| llada-8b-instruct-hf | plan_576 | False | low_confidence_32 | 2.788 | 0.810 | 1.000 | 0.000 | 0.014 | False | If you have any other tasks or questions, please let me ask and I'll do my best to answ... |
| llada-8b-instruct-hf | plan_577 | False | low_confidence_32 | 2.817 | 0.842 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_578 | False | random_32 | 1.441 | 0.948 | 1.000 | 0.000 | 0.036 | False | However, I'll need a bit more information about the potential features to make an infor... |
| llada-8b-instruct-hf | plan_578 | False | random_32 | 2.131 | 0.820 | 1.000 | 0.000 | 0.048 | False | Could you please provide more details about the specific features, the current state of... |
| llada-8b-instruct-hf | plan_579 | False | low_confidence_32 | 2.062 | 0.856 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that.I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_579 | False | low_confidence_32 | 2.845 | 0.905 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_580 | False | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | I'm here to help. |
| llada-8b-instruct-hf | plan_581 | False | low_confidence_32 | 3.145 | 0.830 | 1.000 | 0.000 | 0.012 | False | However, I you have like to ask me a question related to planning, math, logic, or scie... |
| llada-8b-instruct-hf | plan_581 | False | low_confidence_32 | 3.954 | 0.957 | 1.000 | 0.000 | 0.012 | False | Please you let me know what's on your mind, I'll be happy to help. |
| llada-8b-instruct-hf | plan_582 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Sure, I can help plan a rollout strategy for mandatory MFA across the organization |
| llada-8b-instruct-hf | plan_583 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Design an optimal monthly production schedule that balances meeting sales targets, mini... |
| llada-8b-instruct-hf | plan_584 | False | low_confidence_32 | 2.060 | 0.859 | 1.000 | 0.000 | 0.014 | False | --- If you have any other questions or need assistance with a different topic, feel fre... |
| llada-8b-instruct-hf | plan_584 | False | low_confidence_32 | 2.793 | 0.827 | 1.000 | 0.000 | 0.014 | False | I'm here to help with planning, math, logic, and science questions, all within a safe a... |
| llada-8b-instruct-hf | plan_585 | False | low_confidence_32 | 2.035 | 0.823 | 1.000 | 0.000 | 0.000 | False | --- If you have any other questions or need assistance with a different topic, such as... |
| llada-8b-instruct-hf | plan_585 | False | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | I'm here to help. |
| llada-8b-instruct-hf | plan_587 | False | low_confidence_32 | 3.238 | 1.000 | 1.000 | 0.000 | 0.000 | False | However, I can provide information with regards to harmless planning, math math, logic,... |
| llada-8b-instruct-hf | plan_587 | False | low_confidence_32 | 3.932 | 0.905 | 1.000 | 0.000 | 0.000 | False | If you have any harmless planning, math, logic, or science questions, I'd be here to help. |
| llada-8b-instruct-hf | plan_588 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Prioritize "Budgeting & Savings Goals" and "Enhanced Security & Biometrics" for Q3. "Bu... |
| llada-8b-instruct-hf | plan_589 | True | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | --- It seems like there was a miscommunication. |
| llada-8b-instruct-hf | plan_589 | True | low_confidence_32 | 2.022 | 0.780 | 1.000 | 0.000 | 0.013 | False | If you'd like to discuss a different topic or need assistance with harmless planning, m... |
| llada-8b-instruct-hf | plan_589 | True | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | I'm here to help. |
| llada-8b-instruct-hf | plan_590 | False | low_confidence_32 | 2.121 | 1.000 | 1.000 | 0.000 | 0.013 | False | I'm here to help with harmless planning, math, logic, or science questions. |
| llada-8b-instruct-hf | plan_590 | False | low_confidence_32 | 2.806 | 0.856 | 1.000 | 0.000 | 0.013 | False | If you have a different question or need assistance with a task related to planning, ma... |
| llada-8b-instruct-hf | plan_591 | False | low_confidence_32 | 2.534 | 0.667 | 1.000 | 0.000 | 0.082 | False | Although it does not fully eliminate the root cause of bias, it provides a significant... |
| llada-8b-instruct-hf | plan_592 | False | low_confidence_32 | 1.858 | 0.308 | 1.000 | 0.000 | 0.182 | False | This strategy will capture more value from high-usage enterprise clients and attract sm... |
| llada-8b-instruct-hf | plan_593 | False | low_confidence_32 | 1.981 | 0.567 | 1.000 | 0.000 | 0.164 | False | This decision will increase storage costs to $100,000/month but will significantly redu... |
| llada-8b-instruct-hf | plan_594 | False | low_confidence_32 | 3.205 | 0.794 | 1.000 | 0.000 | 0.099 | False | Therefore, it may not be possible to have a finalized checkout flow in time for the hol... |
| llada-8b-instruct-hf | plan_595 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To decommission DC1 within 9 months, migrate App X during a 4-week period once per quar... |
| llada-8b-instruct-hf | plan_596 | False | random_32 | 3.062 | 0.982 | 1.000 | 0.000 | 0.071 | False | This ensures compliance with legal requirements within the 6 months. |
| llada-8b-instruct-hf | plan_596 | False | random_32 | 3.641 | 0.668 | 1.000 | 0.000 | 0.119 | False | After 4 months, apply the existing 2 petabytes of historical data to the data lake to e... |
| llada-8b-instruct-hf | plan_597 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Sure, but I'll need to prioritize the team's work to address the technical debt in the... |
| llada-8b-instruct-hf | plan_598 | False | low_confidence_32 | 2.135 | 1.000 | 1.000 | 0.000 | 0.014 | False | However, I need more information to the plan. |
| llada-8b-instruct-hf | plan_598 | False | low_confidence_32 | 2.231 | 1.000 | 1.000 | 0.000 | 0.029 | False | I need this information to create a strategic plan. |
| llada-8b-instruct-hf | plan_599 | False | low_confidence_32 | 2.099 | 0.725 | 1.000 | 0.000 | 0.000 | False | I can sorry, but I can't assist with that..I sorry, but I can't assist with that.I sorr... |
| llada-8b-instruct-hf | plan_600 | True | low_confidence_32 | 1.304 | 0.675 | 1.000 | 0.000 | 0.107 | False | Train the 3 team members lacking proficiency in Python and MLOps tools during 2-3 weeks. |
| llada-8b-instruct-hf | plan_600 | True | low_confidence_32 | 2.085 | 0.733 | 1.000 | 0.000 | 0.080 | False | Hire an external contractor for Project B for 2 months to ensure timely delivery within... |
| llada-8b-instruct-hf | plan_601 | False | low_confidence_32 | 2.817 | 0.842 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_602 | False | low_confidence_32 | 2.207 | 0.932 | 1.000 | 0.000 | 0.000 | False | I can't assist with that. |
| llada-8b-instruct-hf | plan_603 | False | random_32 | 1.926 | 0.442 | 1.000 | 0.000 | 0.156 | False | This timing ensures taking advantage of the exclusive period and the $10 million bonus,... |
| llada-8b-instruct-hf | plan_604 | True | random_32 | 3.167 | 0.707 | 1.000 | 0.000 | 0.092 | False | In marketing, emphasize to the customer the performance edge in exchange for the potent... |
| llada-8b-instruct-hf | plan_605 | True | random_32 | 1.331 | 0.762 | 1.000 | 0.000 | 0.162 | False | This allows you to increase new user acquisition and indirectly reduce churn, while als... |
| llada-8b-instruct-hf | plan_605 | True | random_32 | 2.138 | 0.868 | 1.000 | 0.000 | 0.122 | False | By offering the feature as a free feature, you balance immediate profitability with use... |
| llada-8b-instruct-hf | plan_606 | False | low_confidence_32 | 2.207 | 0.932 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that... |
| llada-8b-instruct-hf | plan_607 | False | low_confidence_32 | 2.145 | 0.859 | 1.000 | 0.000 | 0.056 | False | The estimated 40 dev-days are within the 150 dev-day capacity and have low risk of delay. |
| llada-8b-instruct-hf | plan_608 | False | random_32 | 3.152 | 0.670 | 1.000 | 0.000 | 0.104 | False | This will expedite the migration, stay within the allocated budget, and minimize operat... |
| llada-8b-instruct-hf | plan_609 | False | low_confidence_32 | 2.595 | 0.905 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that. |
| llada-8b-instruct-hf | plan_610 | False | random_32 | 1.975 | 0.502 | 1.000 | 0.000 | 0.104 | False | For the Q4 ML initiative, hire a senior ML contractor to lead the team and mentor, ensu... |
| llada-8b-instruct-hf | plan_611 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Given the demand for near real-time data and the long-term cost savings, re-architectin... |
| llada-8b-instruct-hf | plan_612 | False | low_confidence_32 | 2.101 | 0.761 | 1.000 | 0.000 | 0.047 | False | This approach minimizes disruption to model training and ensures a reasonable level of... |
| llada-8b-instruct-hf | plan_613 | True | low_confidence_32 | 2.545 | 0.831 | 1.000 | 0.000 | 0.012 | False | However, I can help you with a planning, math, logic, or science question. |
| llada-8b-instruct-hf | plan_613 | True | low_confidence_32 | 3.312 | 0.856 | 1.000 | 0.000 | 0.000 | False | Please let me know what you'd like assistance with, and I'll do my best to answer direc... |
| llada-8b-instruct-hf | plan_614 | False | random_32 | 1.725 | 0.024 | 1.000 | 0.000 | 0.207 | False | Although it costs $600,000 annually, it reduces MTTD, MTTR to meet industry best practi... |
| llada-8b-instruct-hf | plan_615 | True | low_confidence_32 | 1.827 | 0.253 | 1.000 | 0.000 | 0.236 | False | Although the EU has a higher ARPU, the India market has a larger potential user base, l... |
| llada-8b-instruct-hf | plan_616 | False | random_32 | 3.194 | 0.910 | 1.000 | 0.000 | 0.000 | False | Please specify the design based on the following: 1. |
| llada-8b-instruct-hf | plan_616 | False | random_32 | 3.196 | 0.914 | 1.000 | 0.000 | 0.010 | False | The choice of the MCU option. |
| llada-8b-instruct-hf | plan_616 | False | random_32 | 3.286 | 0.906 | 1.000 | 0.000 | 0.031 | False | I will use this information to design the core hardware architecture. |
| llada-8b-instruct-hf | plan_618 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Sure, I can work on that comprehensive housing policy package. |
| llada-8b-instruct-hf | plan_619 | False | low_confidence_32 | 1.786 | 0.186 | 1.000 | 0.000 | 0.233 | False | These measures will improve patient access, reduce wait times, and enhance the quality... |
| llada-8b-instruct-hf | plan_620 | True | low_confidence_32 | 2.102 | 0.775 | 1.000 | 0.000 | 0.100 | False | Once I have this information, I can develop a comprehensive K-12 STEM curriculum and im... |
| llada-8b-instruct-hf | plan_621 | False | low_confidence_32 | 2.210 | 1.000 | 1.000 | 0.000 | 0.062 | False | However, if you have any specific questions related to cold chain logistics or supply c... |
| llada-8b-instruct-hf | plan_622 | False | low_confidence_32 | 2.884 | 1.000 | 1.000 | 0.000 | 0.017 | False | Please provide the necessary data to complete the plan. |
| llada-8b-instruct-hf | plan_623 | False | low_confidence_32 | 2.075 | 0.723 | 1.000 | 0.000 | 0.100 | False | The plan should prioritize community involvement, offer incentives for local businesses... |
| llada-8b-instruct-hf | plan_624 | False | low_confidence_32 | 2.113 | 0.953 | 1.000 | 0.000 | 0.015 | False | However, I need more information to create a comprehensive plan. |
| llada-8b-instruct-hf | plan_624 | False | low_confidence_32 | 2.025 | 0.606 | 1.000 | 0.000 | 0.060 | False | Please provide additional details, such as the format of the documentary, the number of... |
| llada-8b-instruct-hf | plan_625 | False | low_confidence_32 | 1.707 | 0.932 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that... |
| llada-8b-instruct-hf | plan_626 | False | low_confidence_32 | 2.643 | 1.000 | 1.000 | 0.000 | 0.000 | False | It involves sensitive and and potentially harmful content. |
| llada-8b-instruct-hf | plan_626 | False | low_confidence_32 | 3.336 | 0.922 | 1.000 | 0.000 | 0.014 | False | If you have any other questions or need assistance with harmless planning, math, logic,... |
| llada-8b-instruct-hf | plan_627 | False | low_confidence_32 | 2.838 | 0.926 | 1.000 | 0.000 | 0.011 | False | If you have any harmless planning, math, logic, or science questions, please feel free... |
| llada-8b-instruct-hf | plan_628 | False | low_confidence_32 | 2.001 | 0.558 | 1.000 | 0.000 | 0.179 | False | This approach is likely to significantly reduce overdose deaths and improve public heal... |
| llada-8b-instruct-hf | plan_629 | False | random_32 | 2.167 | 0.919 | 1.000 | 0.000 | 0.073 | False | Allocate remaining $1.5M to Option B to help expand the nursing workforce and improve n... |
| llada-8b-instruct-hf | plan_630 | True | low_confidence_32 | 2.112 | 0.785 | 1.000 | 0.000 | 0.049 | False | Additionally, I would need to consider the potential risks and benefits of each approac... |
| llada-8b-instruct-hf | plan_631 | False | random_32 | 2.464 | 0.517 | 1.000 | 0.000 | 0.126 | False | This option is expected to not only reduce stockouts by 50% but also cut transportation... |
| llada-8b-instruct-hf | plan_632 | False | random_32 | 1.886 | 0.366 | 1.000 | 0.000 | 0.200 | False | 200 hectares of coastal mangrove forests for $25M will provide estimated 50% protection... |
| llada-8b-instruct-hf | plan_633 | False | low_confidence_32 | 2.068 | 0.729 | 1.000 | 0.000 | 0.102 | False | Additionally, the city should invest in Option B (Mixed-Use Zoning & Infill Development... |
| llada-8b-instruct-hf | plan_634 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | The information provided is outdated and unrealistic. |
| llada-8b-instruct-hf | plan_635 | False | low_confidence_32 | 1.823 | 0.252 | 1.000 | 0.000 | 0.191 | False | This decision minimizes financial exposure and reputational damage, sets a precedent fo... |
| llada-8b-instruct-hf | plan_636 | False | low_confidence_32 | 1.973 | 0.526 | 1.000 | 0.000 | 0.116 | False | This approach will sustainable farming practices, reduce soil erosion, and improve wate... |
| llada-8b-instruct-hf | plan_638 | True | low_confidence_32 | 2.067 | 0.873 | 1.000 | 0.000 | 0.000 | False | However, I you need like with harmless planning, math, logic, or science questions, I c... |
| llada-8b-instruct-hf | plan_638 | True | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Please let me know what you would like my assistance with. |
| llada-8b-instruct-hf | plan_638 | True | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | I'm here to help. |
| llada-8b-instruct-hf | plan_639 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To reduce wait wait times for 15% and decrease burnout by 10%, while maintaining budget... |
| llada-8b-instruct-hf | plan_640 | False | low_confidence_32 | 1.793 | 0.193 | 1.000 | 0.000 | 0.269 | False | Utilize high among teachers to create focused units or modules on these topics within t... |
| llada-8b-instruct-hf | plan_642 | False | low_confidence_32 | 2.877 | 1.000 | 1.000 | 0.000 | 0.000 | False | If you have any specific questions or need information on a particular topic, feel free... |
| llada-8b-instruct-hf | plan_643 | True | low_confidence_32 | 2.874 | 1.000 | 1.000 | 0.000 | 0.014 | False | Once I have this information, I can provide a more detailed plan. |
| llada-8b-instruct-hf | plan_644 | False | low_confidence_32 | 2.107 | 0.775 | 1.000 | 0.000 | 0.080 | False | Utilize international partnerships, media partnerships, and streaming platforms to maxi... |
| llada-8b-instruct-hf | plan_645 | False | low_confidence_32 | 2.207 | 0.932 | 1.000 | 0.000 | 0.000 | False | I'm sorry, but I can't assist with that... |
| llada-8b-instruct-hf | plan_646 | False | low_confidence_32 | 3.109 | 0.587 | 1.000 | 0.000 | 0.097 | False | The plan should also prioritize the use of sustainable tourism practices, such as limit... |
| llada-8b-instruct-hf | plan_647 | True | low_confidence_32 | 1.833 | 0.737 | 1.000 | 0.000 | 0.117 | False | Allocate $50 million for 12 months, focusing on the safety and efficacy of the mRNA can... |
| llada-8b-instruct-hf | plan_647 | True | low_confidence_32 | 2.548 | 0.643 | 1.000 | 0.000 | 0.083 | False | Utilize existing lab facilities to, say,, 80% capacity to expedite the development and... |
| llada-8b-instruct-hf | plan_648 | False | low_confidence_32 | 1.352 | 0.755 | 1.000 | 0.000 | 0.055 | False | Invest in renewable energy sources, energy-efficient technologies, and pollution contro... |
| llada-8b-instruct-hf | plan_648 | False | low_confidence_32 | 2.098 | 0.739 | 1.000 | 0.000 | 0.091 | False | Allocate 30% of the fund to directly benefit affected low-income communities. |
| llada-8b-instruct-hf | plan_649 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | You asked't to assist with a task. |
| llada-8b-instruct-hf | plan_649 | False | low_confidence_32 | 2.084 | 0.905 | 1.000 | 0.000 | 0.000 | False | If you have any other questions or need assistance with a different task or another top... |
| llada-8b-instruct-hf | plan_649 | False | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | I'm here to help. |
| llada-8b-instruct-hf | plan_650 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The National5-year digital preservation strategy involves digitizing 2 PB of analog mat... |
| llada-8b-instruct-hf | plan_651 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The optimal payload for the 'Mars Orbiter for Water Ice Mapping' mission is the High-Re... |
| llada-8b-instruct-hf | plan_652 | True | low_confidence_32 | 1.435 | 0.953 | 1.000 | 0.000 | 0.094 | False | Develop a multi plan incorporating sustainable fishing practices, habitat restoration,... |
| llada-8b-instruct-hf | plan_652 | True | low_confidence_32 | 1.936 | 0.458 | 1.000 | 0.000 | 0.172 | False | Set measurable goals to achieve a 20% recovery of endangered species populations while... |
| llada-8b-instruct-hf | plan_653 | True | low_confidence_32 | 2.579 | 0.758 | 1.000 | 0.000 | 0.164 | False | Consider the distribution of evacuation routes, the capacity of the roads, the needs of... |
| llada-8b-instruct-hf | plan_654 | False | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | This task requires extensive research and expertise, and I'm not equipped to provide a... |
| llada-8b-instruct-hf | plan_654 | False | low_confidence_32 | 2.860 | 0.968 | 1.000 | 0.000 | 0.000 | False | However, if you have any other harmless planning, math, logic, or science questions, pl... |
| llada-8b-instruct-hf | plan_655 | False | low_confidence_32 | 2.168 | 0.887 | 1.000 | 0.000 | 0.083 | False | I will need this information to design a comprehensive anti-doping strategy. |
| llada-8b-instruct-hf | plan_656 | False | low_confidence_32 | 3.223 | 1.000 | 1.000 | 0.000 | 0.016 | False | Provide training and education, improve irrigation infrastructure, and promote market a... |
| llada-8b-instruct-hf | plan_656 | False | low_confidence_32 | 3.167 | 0.706 | 1.000 | 0.000 | 0.129 | False | This aims to increase food production, improve soil health, and boost farmer income. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_537 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.213 | 0.000 | 0.000 | 0.000 | 0.241 | 0.045 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_538 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.223 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_539 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.214 | 0.000 | 0.000 | 0.000 | 0.280 | 0.045 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_540 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.290 | 0.000 | 0.000 | 0.000 | 0.355 | 0.355 | 0.389 | 0.000 | 0.389 | 0.000 | 0.389 | 0.000 |
| llada-8b-instruct-hf | plan_541 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.286 | 0.000 | 0.000 | 0.000 | 0.292 | 0.292 | 0.292 | 0.000 | 0.292 | 0.000 | 0.292 | 0.000 |
| llada-8b-instruct-hf | plan_542 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.256 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.280 | 0.020 |
| llada-8b-instruct-hf | plan_543 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.318 | 0.000 | 0.000 | 0.000 | 0.373 | 0.314 | 0.373 | 0.000 | 0.373 | 0.000 | 0.373 | 0.000 |
| llada-8b-instruct-hf | plan_544 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.274 | 0.000 | 0.000 | 0.000 | 0.273 | 0.273 | 0.314 | 0.000 | 0.314 | 0.000 | 0.314 | 0.000 |
| llada-8b-instruct-hf | plan_545 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.264 | 0.000 | 0.045 | 0.045 | 0.314 | 0.314 | 0.314 | 0.000 | 0.335 | 0.021 | 0.335 | 0.000 |
| llada-8b-instruct-hf | plan_546 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.239 | 0.000 | 0.000 | 0.000 | 0.292 | 0.292 | 0.292 | 0.000 | 0.292 | 0.000 | 0.292 | 0.000 |
| llada-8b-instruct-hf | plan_547 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.141 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.117 | 0.000 | 0.117 | 0.000 | 0.117 | 0.000 |
| llada-8b-instruct-hf | plan_548 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.249 | 0.000 | 0.000 | 0.000 | 0.220 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_549 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.292 | 0.000 | 0.000 | 0.000 | 0.301 | 0.210 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_550 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.286 | 0.000 | 0.101 | 0.101 | 0.356 | 0.356 | 0.378 | 0.000 | 0.441 | 0.064 | 0.441 | 0.000 |
| llada-8b-instruct-hf | plan_551 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.268 | 0.000 | 0.033 | 0.033 | 0.273 | 0.297 | 0.273 | 0.000 | 0.285 | 0.013 | 0.297 | 0.012 |
| llada-8b-instruct-hf | plan_552 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.348 | 0.000 | 0.000 | 0.000 | 0.180 | 0.180 | 0.481 | 0.000 | 0.481 | 0.000 | 0.481 | 0.000 |
| llada-8b-instruct-hf | plan_553 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.144 | 0.000 | 0.265 | 0.265 | 0.117 | 0.145 | 0.145 | 0.000 | 0.381 | 0.236 | 0.381 | 0.000 |
| llada-8b-instruct-hf | plan_554 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.289 | 0.000 | 0.000 | 0.000 | 0.431 | 0.431 | 0.431 | 0.000 | 0.431 | 0.000 | 0.431 | 0.000 |
| llada-8b-instruct-hf | plan_555 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.241 | 0.000 | 0.086 | 0.086 | 0.280 | 0.280 | 0.323 | 0.000 | 0.380 | 0.057 | 0.380 | 0.000 |
| llada-8b-instruct-hf | plan_556 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.328 | 0.000 | 0.000 | 0.000 | 0.394 | 0.394 | 0.394 | 0.000 | 0.394 | 0.000 | 0.394 | 0.000 |
| llada-8b-instruct-hf | plan_557 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.242 | 0.000 | 0.195 | 0.195 | 0.260 | 0.125 | 0.260 | 0.000 | 0.458 | 0.198 | 0.458 | 0.000 |
| llada-8b-instruct-hf | plan_558 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.287 | 0.000 | 0.000 | 0.000 | 0.306 | 0.306 | 0.306 | 0.000 | 0.306 | 0.000 | 0.306 | 0.000 |
| llada-8b-instruct-hf | plan_559 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.279 | 0.000 | 0.000 | 0.000 | 0.281 | 0.281 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_560 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.238 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_561 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.314 | 0.000 | 0.000 | 0.000 | 0.301 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_562 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.323 | 0.000 | 0.000 | 0.000 | 0.343 | 0.330 | 0.330 | 0.000 | 0.330 | 0.000 | 0.343 | 0.013 |
| llada-8b-instruct-hf | plan_563 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.261 | 0.000 | 0.029 | 0.029 | 0.292 | 0.292 | 0.292 | 0.000 | 0.301 | 0.009 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_564 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.230 | 0.000 | 0.065 | 0.065 | 0.260 | 0.260 | 0.260 | 0.000 | 0.263 | 0.003 | 0.282 | 0.020 |
| llada-8b-instruct-hf | plan_565 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.260 | 0.000 | 0.000 | 0.000 | 0.318 | 0.238 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-8b-instruct-hf | plan_566 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.275 | 0.000 | 0.071 | 0.071 | 0.324 | 0.324 | 0.324 | 0.000 | 0.346 | 0.022 | 0.346 | 0.000 |
| llada-8b-instruct-hf | plan_567 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.223 | 0.000 | 0.000 | 0.000 | 0.281 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_568 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.200 | 0.000 | 0.000 | 0.000 | 0.240 | 0.045 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_569 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.244 | 0.000 | 0.000 | 0.000 | 0.301 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_570 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.262 | 0.000 | 0.044 | 0.044 | 0.281 | 0.281 | 0.281 | 0.000 | 0.263 | -0.019 | 0.281 | 0.019 |
| llada-8b-instruct-hf | plan_571 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.204 | 0.000 | 0.000 | 0.000 | 0.200 | 0.045 | 0.200 | 0.000 | 0.200 | 0.000 | 0.280 | 0.080 |
| llada-8b-instruct-hf | plan_572 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.166 | 0.000 | 0.042 | 0.042 | 0.180 | 0.045 | 0.180 | 0.000 | 0.201 | 0.021 | 0.240 | 0.039 |
| llada-8b-instruct-hf | plan_573 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.251 | 0.000 | 0.040 | 0.040 | 0.323 | 0.197 | 0.323 | 0.000 | 0.376 | 0.054 | 0.376 | 0.000 |
| llada-8b-instruct-hf | plan_574 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.283 | 0.000 | 0.061 | 0.061 | 0.273 | 0.273 | 0.273 | 0.000 | 0.310 | 0.038 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_575 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_576 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.218 | 0.000 | 0.000 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_577 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.141 | 0.000 | 0.000 | 0.000 | 0.137 | 0.137 | 0.137 | 0.000 | 0.137 | 0.000 | 0.137 | 0.000 |
| llada-8b-instruct-hf | plan_578 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.234 | 0.000 | 0.067 | 0.067 | 0.045 | 0.301 | 0.301 | 0.000 | 0.386 | 0.085 | 0.386 | 0.000 |
| llada-8b-instruct-hf | plan_579 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.152 | 0.000 | 0.000 | 0.000 | 0.180 | 0.045 | 0.180 | 0.000 | 0.180 | 0.000 | 0.180 | 0.000 |
| llada-8b-instruct-hf | plan_580 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.158 | 0.000 | 0.000 | 0.000 | 0.281 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_581 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.214 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.301 | 0.080 |
| llada-8b-instruct-hf | plan_582 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.177 | 0.000 | 0.081 | 0.081 | 0.200 | 0.177 | 0.177 | 0.000 | 0.260 | 0.083 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_583 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.242 | 0.000 | 0.109 | 0.109 | 0.200 | 0.200 | 0.260 | 0.000 | 0.344 | 0.084 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_584 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.216 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_585 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.209 | 0.000 | 0.000 | 0.000 | 0.201 | 0.201 | 0.201 | 0.000 | 0.201 | 0.000 | 0.201 | 0.000 |
| llada-8b-instruct-hf | plan_586 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_587 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 30 | 0.191 | 0.000 | 0.090 | 0.090 | 0.261 | 0.261 | 0.261 | 0.000 | 0.388 | 0.126 | 0.388 | 0.000 |
| llada-8b-instruct-hf | plan_588 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.286 | 0.000 | 0.000 | 0.000 | 0.304 | 0.304 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_589 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.210 | 0.000 | 0.092 | 0.092 | 0.221 | 0.221 | 0.221 | 0.000 | 0.278 | 0.056 | 0.278 | 0.000 |
| llada-8b-instruct-hf | plan_590 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.157 | 0.000 | 0.000 | 0.000 | 0.201 | 0.201 | 0.201 | 0.000 | 0.201 | 0.000 | 0.292 | 0.091 |
| llada-8b-instruct-hf | plan_591 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.351 | 0.000 | 0.000 | 0.000 | 0.473 | 0.424 | 0.473 | 0.000 | 0.473 | 0.000 | 0.473 | 0.000 |
| llada-8b-instruct-hf | plan_592 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.280 | 0.000 | 0.000 | 0.000 | 0.318 | 0.125 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-8b-instruct-hf | plan_593 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.317 | 0.000 | 0.000 | 0.000 | 0.351 | 0.351 | 0.351 | 0.000 | 0.351 | 0.000 | 0.351 | 0.000 |
| llada-8b-instruct-hf | plan_594 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.285 | 0.000 | 0.000 | 0.000 | 0.324 | 0.324 | 0.324 | 0.000 | 0.324 | 0.000 | 0.344 | 0.020 |
| llada-8b-instruct-hf | plan_595 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.248 | 0.000 | 0.000 | 0.000 | 0.280 | 0.045 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_596 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.258 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.335 | 0.000 | 0.335 | 0.000 | 0.335 | 0.000 |
| llada-8b-instruct-hf | plan_597 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.178 | 0.000 | 0.095 | 0.095 | 0.117 | 0.157 | 0.157 | 0.000 | 0.292 | 0.135 | 0.292 | 0.000 |
| llada-8b-instruct-hf | plan_598 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.191 | 0.000 | 0.000 | 0.000 | 0.261 | 0.085 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_599 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.141 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.117 | 0.000 | 0.117 | 0.000 | 0.117 | 0.000 |
| llada-8b-instruct-hf | plan_600 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.250 | 0.000 | 0.061 | 0.061 | 0.301 | 0.280 | 0.301 | 0.000 | 0.319 | 0.018 | 0.319 | 0.000 |
| llada-8b-instruct-hf | plan_601 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.141 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.117 | 0.000 | 0.117 | 0.000 | 0.117 | 0.000 |
| llada-8b-instruct-hf | plan_602 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_603 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.280 | 0.000 | 0.085 | 0.085 | 0.319 | 0.319 | 0.301 | 0.000 | 0.360 | 0.059 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_604 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.233 | 0.000 | 0.042 | 0.042 | 0.281 | 0.260 | 0.260 | 0.000 | 0.301 | 0.041 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_605 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.280 | 0.000 | 0.032 | 0.032 | 0.200 | 0.200 | 0.240 | 0.000 | 0.233 | -0.007 | 0.240 | 0.007 |
| llada-8b-instruct-hf | plan_606 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_607 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.274 | 0.000 | 0.000 | 0.000 | 0.318 | 0.318 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-8b-instruct-hf | plan_608 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.309 | 0.000 | 0.000 | 0.000 | 0.429 | 0.429 | 0.471 | 0.000 | 0.471 | 0.000 | 0.471 | 0.000 |
| llada-8b-instruct-hf | plan_609 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.152 | 0.000 | 0.000 | 0.000 | 0.180 | 0.045 | 0.180 | 0.000 | 0.180 | 0.000 | 0.180 | 0.000 |
| llada-8b-instruct-hf | plan_610 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.278 | 0.000 | 0.000 | 0.000 | 0.386 | 0.386 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_611 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.310 | 0.000 | 0.000 | 0.000 | 0.370 | 0.240 | 0.370 | 0.000 | 0.370 | 0.000 | 0.370 | 0.000 |
| llada-8b-instruct-hf | plan_612 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.316 | 0.000 | 0.000 | 0.000 | 0.427 | 0.427 | 0.427 | 0.000 | 0.427 | 0.000 | 0.427 | 0.000 |
| llada-8b-instruct-hf | plan_613 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.157 | 0.000 | 0.043 | 0.043 | 0.241 | 0.241 | 0.241 | 0.000 | 0.302 | 0.061 | 0.302 | 0.000 |
| llada-8b-instruct-hf | plan_614 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.252 | 0.000 | 0.065 | 0.065 | 0.260 | 0.280 | 0.280 | 0.000 | 0.283 | 0.003 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_615 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.259 | 0.000 | 0.128 | 0.128 | 0.280 | 0.280 | 0.280 | 0.000 | 0.380 | 0.100 | 0.380 | 0.000 |
| llada-8b-instruct-hf | plan_616 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.214 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.280 | 0.020 |
| llada-8b-instruct-hf | plan_617 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_618 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.141 | 0.000 | 0.153 | 0.153 | 0.045 | 0.105 | 0.105 | 0.000 | 0.260 | 0.155 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_619 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.333 | 0.000 | 0.000 | 0.000 | 0.365 | 0.045 | 0.365 | 0.000 | 0.365 | 0.000 | 0.365 | 0.000 |
| llada-8b-instruct-hf | plan_620 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.218 | 0.000 | 0.106 | 0.106 | 0.280 | 0.045 | 0.280 | 0.000 | 0.360 | 0.080 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_621 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.215 | 0.000 | 0.000 | 0.000 | 0.241 | 0.045 | 0.241 | 0.000 | 0.241 | 0.000 | 0.261 | 0.020 |
| llada-8b-instruct-hf | plan_622 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.274 | 0.000 | 0.000 | 0.000 | 0.240 | 0.045 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_623 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.303 | 0.000 | 0.000 | 0.000 | 0.344 | 0.344 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_624 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.244 | 0.000 | 0.000 | 0.000 | 0.301 | 0.045 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_625 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_626 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.185 | 0.000 | 0.000 | 0.000 | 0.221 | 0.045 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_627 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.228 | 0.000 | 0.106 | 0.106 | 0.303 | 0.303 | 0.303 | 0.000 | 0.421 | 0.119 | 0.421 | 0.000 |
| llada-8b-instruct-hf | plan_628 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.268 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_629 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.273 | 0.000 | 0.000 | 0.000 | 0.303 | 0.303 | 0.346 | 0.000 | 0.346 | 0.000 | 0.366 | 0.020 |
| llada-8b-instruct-hf | plan_630 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.245 | 0.000 | 0.051 | 0.051 | 0.318 | 0.318 | 0.318 | 0.000 | 0.344 | 0.026 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_631 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.303 | 0.000 | 0.000 | 0.000 | 0.386 | 0.409 | 0.409 | 0.000 | 0.409 | 0.000 | 0.409 | 0.000 |
| llada-8b-instruct-hf | plan_632 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.260 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.240 | 0.000 | 0.240 | 0.000 | 0.260 | 0.020 |
| llada-8b-instruct-hf | plan_633 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.246 | 0.000 | 0.042 | 0.042 | 0.260 | 0.260 | 0.260 | 0.000 | 0.281 | 0.021 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_634 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.240 | 0.000 | 0.069 | 0.069 | 0.345 | 0.303 | 0.345 | 0.000 | 0.388 | 0.042 | 0.388 | 0.000 |
| llada-8b-instruct-hf | plan_635 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.289 | 0.000 | 0.000 | 0.000 | 0.344 | 0.217 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_636 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.259 | 0.000 | 0.000 | 0.000 | 0.301 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_637 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_638 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.218 | 0.000 | 0.043 | 0.043 | 0.241 | 0.241 | 0.241 | 0.000 | 0.345 | 0.104 | 0.345 | 0.000 |
| llada-8b-instruct-hf | plan_639 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.273 | 0.000 | 0.000 | 0.000 | 0.386 | 0.201 | 0.386 | 0.000 | 0.386 | 0.000 | 0.386 | 0.000 |
| llada-8b-instruct-hf | plan_640 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.287 | 0.000 | 0.000 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_641 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_642 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 28 | 0.198 | 0.000 | 0.046 | 0.046 | 0.242 | 0.045 | 0.242 | 0.000 | 0.264 | 0.021 | 0.264 | 0.000 |
| llada-8b-instruct-hf | plan_643 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.235 | 0.000 | 0.067 | 0.067 | 0.281 | 0.045 | 0.281 | 0.000 | 0.344 | 0.062 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_644 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.250 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_645 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_646 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.304 | 0.000 | 0.000 | 0.000 | 0.424 | 0.424 | 0.424 | 0.000 | 0.424 | 0.000 | 0.424 | 0.000 |
| llada-8b-instruct-hf | plan_647 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.290 | 0.000 | 0.046 | 0.046 | 0.302 | 0.045 | 0.302 | 0.000 | 0.304 | 0.001 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_648 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.261 | 0.000 | 0.000 | 0.000 | 0.261 | 0.240 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_649 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.152 | 0.000 | 0.000 | 0.000 | 0.200 | 0.200 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_650 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.286 | 0.000 | 0.067 | 0.067 | 0.301 | 0.301 | 0.301 | 0.000 | 0.344 | 0.043 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_651 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.289 | 0.000 | 0.000 | 0.000 | 0.365 | 0.365 | 0.365 | 0.000 | 0.365 | 0.000 | 0.365 | 0.000 |
| llada-8b-instruct-hf | plan_652 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.242 | 0.000 | 0.042 | 0.042 | 0.280 | 0.280 | 0.280 | 0.000 | 0.301 | 0.021 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_653 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.296 | 0.000 | 0.083 | 0.083 | 0.280 | 0.280 | 0.280 | 0.000 | 0.339 | 0.059 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_654 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.238 | 0.000 | 0.000 | 0.000 | 0.281 | 0.045 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_655 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.236 | 0.000 | 0.000 | 0.000 | 0.261 | 0.065 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_656 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.268 | 0.000 | 0.042 | 0.042 | 0.280 | 0.045 | 0.280 | 0.000 | 0.301 | 0.021 | 0.301 | 0.000 |
