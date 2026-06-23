# Diffusion Schedule-Selection Benchmark Report

Full model generations: `240`
Counterfactual probe generations: `48`
Arm selections: `336`
Run ID: `diffusion-fb84d1ca21badb07`
Content hash: `fb84d1ca21badb077d1cebe04509f74aa6384dd2370c8f1748eac6f7cd5e39c8`
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
Trajectory task delta vs fixed: `0.005`
Trajectory task delta vs random: `0.022`
Trajectory wins/ties/losses vs fixed: `14/77/5`
Trajectory wins/ties/losses vs random: `30/62/4`
Oracle generation budget/task: `2.50`
Oracle task score: `0.163`
Oracle headroom vs trajectory: `0.006`
Oracle wins/ties/losses vs trajectory: `11/85/0`
Selector regret vs trajectory: `0.006 over 11/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.006`
Repair task delta vs random: `0.033`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/48/0`
Oracle headroom vs repair: `0.004`
Oracle wins/ties/losses vs repair: `7/41/0`
Selector regret vs repair: `0.004 over 7/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.284007 | 0.000000 | 0.026881 | - | - |
| random perturbation | repair-covered tasks | 0.257126 | -0.026881 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.290403 | 0.006396 | 0.033277 | 6/40/2 | 17/28/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.153 | 0.410 | 0.217 |
| random | 96 | 1.00 | 0.135 | 0.347 | 0.188 |
| trajectory_selected | 96 | 2.50 | 0.157 | 0.417 | 0.222 |
| repair_selected | 48 | 2.00 | 0.290 | 0.678 | 0.387 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.153 | 0.410 | 0.217 |
| planning | random | 96 | 1.00 | 0.135 | 0.347 | 0.188 |
| planning | trajectory_selected | 96 | 2.50 | 0.157 | 0.417 | 0.222 |
| planning | repair_selected | 48 | 2.00 | 0.290 | 0.678 | 0.387 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_249 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 255 | True | 7 | 0.533 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_250 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.301 | 0.201 | 235 | True | 8 | 0.429 | True | True | 7.000 | 0.219 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_251 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.341 | 0.301 | 258 | True | 5 | 0.769 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-8b-instruct-hf | plan_252 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.391 | 0.311 | 324 | True | 2 | 0.917 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_253 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.301 | 0.201 | 305 | True | 3 | 0.750 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_254 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 331 | True | 1 | 0.923 | True | True | 7.000 | 0.219 | 0.385 | 0.385 |
| llada-8b-instruct-hf | plan_255 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.378 | 0.260 | 309 | True | 2 | 0.867 | True | True | 7.000 | 0.219 | 0.467 | 0.467 |
| llada-8b-instruct-hf | plan_256 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.200 | 0.180 | 319 | True | 7 | 0.533 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_257 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.324 | 0.244 | 398 | True | 3 | 0.750 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_258 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.065 | 0.045 | 64 | True | 4 | 0.636 | True | True | 13.000 | 0.406 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_259 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.303 | 0.223 | 354 | True | 3 | 0.750 | True | True | 7.000 | 0.219 | 0.750 | 0.750 |
| llada-8b-instruct-hf | plan_260 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.381 | 0.301 | 331 | True | 1 | 0.889 | True | True | 7.000 | 0.219 | 0.556 | 0.556 |
| llada-8b-instruct-hf | plan_261 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 271 | True | 3 | 0.769 | True | True | 7.000 | 0.219 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_262 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.418 | 0.340 | 342 | True | 5 | 0.500 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_263 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.117 | 0.117 | 136 | True | 5 | 0.500 | True | True | 7.000 | 0.219 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_264 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 381 | True | 6 | 0.571 | True | True | 7.000 | 0.219 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_265 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.280 | 0.180 | 331 | True | 3 | 0.727 | True | True | 7.000 | 0.219 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_266 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.330 | 0.230 | 336 | True | 3 | 0.727 | True | True | 7.000 | 0.219 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_267 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.301 | 0.201 | 380 | True | 1 | 0.909 | True | True | 7.000 | 0.219 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_268 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.345 | 0.285 | 345 | True | 1 | 0.929 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_269 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.220 | 0.180 | 334 | True | 2 | 0.867 | True | True | 7.000 | 0.219 | 0.467 | 0.467 |
| llada-8b-instruct-hf | plan_270 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.319 | 0.299 | 258 | True | 4 | 0.600 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_271 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 287 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_272 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.404 | 0.326 | 293 | True | 1 | 0.917 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_273 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.378 | 0.267 | 267 | True | 4 | 0.636 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_274 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.309 | 0.269 | 336 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_275 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.391 | 0.269 | 336 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_276 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 314 | True | 1 | 0.923 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-8b-instruct-hf | plan_277 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.334 | 0.294 | 414 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_278 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 348 | True | 5 | 0.583 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_279 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 348 | True | 5 | 0.545 | True | True | 7.000 | 0.219 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_280 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.273 | 0.193 | 360 | True | 3 | 0.786 | True | True | 7.000 | 0.219 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_281 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.296 | 0.256 | 387 | True | 4 | 0.600 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_282 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.255 | 0.235 | 307 | True | 6 | 0.643 | True | True | 7.000 | 0.219 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_283 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.213 | 0.193 | 399 | True | 6 | 0.500 | True | True | 7.000 | 0.219 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_284 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.366 | 0.244 | 311 | True | 3 | 0.727 | True | True | 7.000 | 0.219 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_285 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.395 | 0.272 | 339 | True | 2 | 0.778 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_286 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.234 | 0.214 | 276 | True | 6 | 0.500 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_287 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 369 | True | 1 | 0.909 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_288 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.178 | 0.138 | 126 | True | 4 | 0.500 | True | True | 7.000 | 0.219 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_289 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.324 | 0.244 | 288 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_290 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.233 | 0.193 | 342 | True | 3 | 0.750 | True | True | 7.000 | 0.219 | 0.583 | 0.583 |
| llada-8b-instruct-hf | plan_291 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.180 | 0.180 | 300 | True | 3 | 0.625 | True | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_292 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.339 | 0.299 | 323 | True | 4 | 0.600 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_293 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 326 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_294 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.311 | 0.251 | 340 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_295 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.414 | 0.374 | 328 | True | 4 | 0.667 | True | True | 1.000 | 0.031 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_296 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.350 | 0.290 | 369 | True | 2 | 0.846 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_249 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_250 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_251 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_252 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_253 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_254 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_255 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_256 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_257 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_258 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_259 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_260 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_261 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_262 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_263 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_264 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_265 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_266 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_267 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_268 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_269 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_270 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_271 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_272 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_273 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_274 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_275 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_276 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_277 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_278 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_279 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_280 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_281 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_282 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.127 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_283 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_284 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_285 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_286 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_287 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_288 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_289 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_290 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.129 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_291 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_292 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_293 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.180 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_294 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_295 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_296 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_249 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.339 | 0.000 | 0.201 | 0.000 | 0.241 | 0.045 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_250 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.297 | 0.000 | 0.201 | 0.000 | 0.281 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_251 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.442 | 0.000 | 0.301 | 0.000 | 0.341 | 0.045 | 0.341 | 0.000 | 0.341 | 0.000 | 0.341 | 0.000 |
| llada-8b-instruct-hf | plan_252 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.471 | 0.000 | 0.311 | 0.000 | 0.273 | 0.391 | 0.391 | 0.000 | 0.391 | 0.000 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_253 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.388 | 0.000 | 0.201 | 0.000 | 0.301 | 0.157 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_254 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.430 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_255 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.452 | 0.000 | 0.260 | 0.000 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_256 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.308 | 0.000 | 0.180 | 0.000 | 0.200 | 0.200 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_257 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.416 | 0.000 | 0.244 | 0.000 | 0.324 | 0.324 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_258 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.272 | 0.000 | 0.045 | 0.000 | 0.045 | 0.045 | 0.065 | 0.000 | 0.065 | 0.000 | 0.065 | 0.000 |
| llada-8b-instruct-hf | plan_259 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.420 | 0.000 | 0.223 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_260 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.486 | 0.000 | 0.301 | 0.000 | 0.381 | 0.381 | 0.381 | 0.000 | 0.381 | 0.000 | 0.381 | 0.000 |
| llada-8b-instruct-hf | plan_261 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.380 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_262 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.400 | 0.000 | 0.340 | 0.000 | 0.418 | 0.418 | 0.418 | 0.000 | 0.418 | 0.000 | 0.418 | 0.000 |
| llada-8b-instruct-hf | plan_263 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.274 | 0.000 | 0.117 | 0.000 | 0.045 | 0.045 | 0.117 | 0.000 | 0.117 | 0.000 | 0.117 | 0.000 |
| llada-8b-instruct-hf | plan_264 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.340 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_265 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.388 | 0.000 | 0.180 | 0.000 | 0.280 | 0.217 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_266 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.384 | 0.000 | 0.230 | 0.000 | 0.330 | 0.217 | 0.330 | 0.000 | 0.330 | 0.000 | 0.330 | 0.000 |
| llada-8b-instruct-hf | plan_267 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.436 | 0.000 | 0.201 | 0.000 | 0.301 | 0.280 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_268 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.475 | 0.000 | 0.285 | 0.000 | 0.345 | 0.345 | 0.345 | 0.000 | 0.345 | 0.000 | 0.345 | 0.000 |
| llada-8b-instruct-hf | plan_269 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.406 | 0.000 | 0.180 | 0.000 | 0.220 | 0.045 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_270 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.381 | 0.000 | 0.299 | 0.000 | 0.319 | 0.319 | 0.319 | 0.000 | 0.319 | 0.000 | 0.339 | 0.020 |
| llada-8b-instruct-hf | plan_271 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.389 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_272 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.489 | 0.000 | 0.326 | 0.000 | 0.404 | 0.333 | 0.404 | 0.000 | 0.404 | 0.000 | 0.404 | 0.000 |
| llada-8b-instruct-hf | plan_273 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.411 | 0.000 | 0.267 | 0.000 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_274 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.435 | 0.000 | 0.269 | 0.000 | 0.309 | 0.309 | 0.309 | 0.000 | 0.309 | 0.000 | 0.309 | 0.000 |
| llada-8b-instruct-hf | plan_275 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.428 | 0.000 | 0.269 | 0.000 | 0.391 | 0.391 | 0.391 | 0.000 | 0.391 | 0.000 | 0.429 | 0.038 |
| llada-8b-instruct-hf | plan_276 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.421 | 0.000 | 0.180 | 0.000 | 0.260 | 0.200 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_277 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.475 | 0.000 | 0.294 | 0.000 | 0.334 | 0.334 | 0.334 | 0.000 | 0.334 | 0.000 | 0.334 | 0.000 |
| llada-8b-instruct-hf | plan_278 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.355 | 0.000 | 0.201 | 0.000 | 0.241 | 0.200 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_279 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.339 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_280 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.387 | 0.000 | 0.193 | 0.000 | 0.303 | 0.303 | 0.273 | 0.000 | 0.273 | 0.000 | 0.303 | 0.030 |
| llada-8b-instruct-hf | plan_281 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.392 | 0.000 | 0.256 | 0.000 | 0.296 | 0.178 | 0.296 | 0.000 | 0.296 | 0.000 | 0.296 | 0.000 |
| llada-8b-instruct-hf | plan_282 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.377 | 0.000 | 0.235 | 0.000 | 0.293 | 0.293 | 0.255 | 0.000 | 0.255 | 0.000 | 0.293 | 0.038 |
| llada-8b-instruct-hf | plan_283 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.323 | 0.000 | 0.193 | 0.000 | 0.213 | 0.241 | 0.213 | 0.000 | 0.213 | 0.000 | 0.241 | 0.029 |
| llada-8b-instruct-hf | plan_284 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.410 | 0.000 | 0.244 | 0.000 | 0.345 | 0.366 | 0.366 | 0.000 | 0.366 | 0.000 | 0.366 | 0.000 |
| llada-8b-instruct-hf | plan_285 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.445 | 0.000 | 0.272 | 0.000 | 0.395 | 0.395 | 0.395 | 0.000 | 0.395 | 0.000 | 0.445 | 0.050 |
| llada-8b-instruct-hf | plan_286 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.348 | 0.000 | 0.214 | 0.000 | 0.234 | 0.197 | 0.234 | 0.000 | 0.234 | 0.000 | 0.234 | 0.000 |
| llada-8b-instruct-hf | plan_287 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.435 | 0.000 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_288 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.317 | 0.000 | 0.138 | 0.000 | 0.178 | 0.178 | 0.178 | 0.000 | 0.178 | 0.000 | 0.180 | 0.002 |
| llada-8b-instruct-hf | plan_289 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.397 | 0.000 | 0.244 | 0.000 | 0.324 | 0.324 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_290 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.392 | 0.000 | 0.193 | 0.000 | 0.233 | 0.233 | 0.233 | 0.000 | 0.233 | 0.000 | 0.233 | 0.000 |
| llada-8b-instruct-hf | plan_291 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.358 | 0.000 | 0.180 | 0.000 | 0.180 | 0.180 | 0.180 | 0.000 | 0.180 | 0.000 | 0.180 | 0.000 |
| llada-8b-instruct-hf | plan_292 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.413 | 0.000 | 0.299 | 0.000 | 0.339 | 0.339 | 0.339 | 0.000 | 0.339 | 0.000 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_293 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.396 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_294 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.442 | 0.000 | 0.251 | 0.000 | 0.311 | 0.190 | 0.311 | 0.000 | 0.311 | 0.000 | 0.311 | 0.000 |
| llada-8b-instruct-hf | plan_295 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.449 | 0.000 | 0.374 | 0.000 | 0.291 | 0.291 | 0.414 | 0.000 | 0.414 | 0.000 | 0.414 | 0.000 |
| llada-8b-instruct-hf | plan_296 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.449 | 0.000 | 0.290 | 0.000 | 0.350 | 0.329 | 0.350 | 0.000 | 0.350 | 0.000 | 0.350 | 0.000 |
