# Diffusion Schedule-Selection Benchmark Report

Full model generations: `288`
Counterfactual probe generations: `0`
Arm selections: `336`
Run ID: `diffusion-bf04b1ee85912aa1`
Content hash: `bf04b1ee85912aa1403abae874227ac5d5f110b594f3bc2815d9438f39f770fd`
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
Trajectory task delta vs fixed: `-0.015`
Trajectory task delta vs random: `0.014`
Trajectory wins/ties/losses vs fixed: `16/49/31`
Trajectory wins/ties/losses vs random: `24/59/13`
Oracle generation budget/task: `3.00`
Oracle task score: `0.154`
Oracle headroom vs trajectory: `0.029`
Oracle wins/ties/losses vs trajectory: `34/62/0`
Selector regret vs trajectory: `0.029 over 34/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `-0.041`
Repair task delta vs random: `0.023`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/48/0`
Oracle headroom vs repair: `0.056`
Oracle wins/ties/losses vs repair: `32/16/0`
Selector regret vs repair: `0.056 over 32/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.263757 | 0.000000 | 0.063104 | - | - |
| random perturbation | repair-covered tasks | 0.200653 | -0.063104 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.223210 | -0.040548 | 0.022557 | 6/13/29 | 15/22/11 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.140 | 0.402 | 0.205 |
| random | 96 | 1.00 | 0.111 | 0.378 | 0.177 |
| trajectory_selected | 96 | 3.00 | 0.125 | 0.424 | 0.200 |
| repair_selected | 48 | 3.00 | 0.223 | 0.699 | 0.342 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.140 | 0.402 | 0.205 |
| planning | random | 96 | 1.00 | 0.111 | 0.378 | 0.177 |
| planning | trajectory_selected | 96 | 3.00 | 0.125 | 0.424 | 0.200 |
| planning | repair_selected | 48 | 3.00 | 0.223 | 0.699 | 0.342 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_297 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.177 | 0.117 | 165 | True | 2 | 0.857 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.323 | 0.223 | 298 | True | 2 | 0.857 | True | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.282 | 0.223 | 354 | True | 4 | 0.714 | True | True | 7.000 | 0.219 | 0.357 | 0.357 |
| llada-8b-instruct-hf | plan_300 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.200 | 0.160 | 167 | True | 6 | 0.643 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.290 | 0.230 | 276 | True | 2 | 0.818 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_302 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.177 | 0.117 | 143 | True | 2 | 0.889 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_303 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.220 | 0.160 | 181 | True | 2 | 0.800 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_304 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.137 | 0.117 | 160 | True | 4 | 0.667 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_305 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.277 | 0.197 | 167 | True | 3 | 0.769 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_306 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.419 | 0.361 | 151 | True | 1 | 0.923 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_307 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.137 | 0.117 | 163 | True | 3 | 0.769 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_308 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.298 | 0.238 | 165 | True | 3 | 0.846 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_309 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.356 | 0.239 | 370 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_310 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.197 | 0.117 | 169 | True | 6 | 0.545 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_311 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.284 | 0.244 | 289 | True | 2 | 0.818 | True | True | 7.000 | 0.219 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_312 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.178 | 0.138 | 195 | True | 0 | 1.000 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_313 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.137 | 0.117 | 153 | True | 3 | 0.818 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_314 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.177 | 0.117 | 179 | True | 4 | 0.750 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.240 | 0.180 | 327 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.667 | 0.667 |
| llada-8b-instruct-hf | plan_316 | random_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.263 | 0.223 | 288 | True | 4 | 0.692 | True | True | 7.000 | 0.219 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_317 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.174 | 0.154 | 144 | True | 6 | 0.500 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_318 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.157 | 0.117 | 144 | True | 6 | 0.571 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_319 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.157 | 0.117 | 164 | True | 1 | 0.909 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_320 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.197 | 0.117 | 175 | True | 3 | 0.800 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.344 | 0.244 | 357 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.667 | 0.667 |
| llada-8b-instruct-hf | plan_322 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.257 | 0.217 | 174 | True | 4 | 0.667 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_323 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.177 | 0.117 | 162 | True | 6 | 0.583 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_324 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.137 | 0.117 | 150 | True | 5 | 0.583 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.296 | 0.256 | 364 | True | 3 | 0.727 | True | True | 7.000 | 0.219 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.284 | 0.244 | 305 | True | 1 | 0.917 | True | True | 7.000 | 0.219 | 0.583 | 0.583 |
| llada-8b-instruct-hf | plan_327 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.214 | 0.154 | 200 | True | 3 | 0.727 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_328 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.117 | 0.117 | 179 | True | 6 | 0.500 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.304 | 0.244 | 371 | True | 2 | 0.818 | True | True | 7.000 | 0.219 | 0.545 | 0.545 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.402 | 0.362 | 359 | True | 7 | 0.417 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_331 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.261 | 0.181 | 149 | True | 1 | 1.000 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_332 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.177 | 0.117 | 150 | True | 1 | 0.900 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_333 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.197 | 0.117 | 158 | True | 3 | 0.667 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_334 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.338 | 0.238 | 156 | True | 1 | 0.917 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_335 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.212 | 0.172 | 168 | True | 1 | 0.900 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_336 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.117 | 0.117 | 170 | True | 2 | 0.750 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.242 | 0.223 | 390 | True | 3 | 0.625 | True | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_338 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.137 | 0.117 | 162 | True | 2 | 0.750 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_339 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.292 | 0.272 | 233 | True | 2 | 0.778 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_340 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.130 | 0.130 | 152 | True | 5 | 0.643 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_341 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.226 | 0.206 | 151 | True | 0 | 1.000 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_342 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.117 | 0.117 | 184 | True | 4 | 0.714 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_343 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.117 | 0.117 | 185 | True | 6 | 0.500 | True | False | none | none | none | none |
| llada-8b-instruct-hf | plan_344 | counterfactual_micro_probe_v1 | False | counterfactual_probe_recorded_no_repair | True | deterministic_scaffold | 0.162 | 0.142 | 161 | True | 2 | 0.818 | True | False | none | none | none | none |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_297 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_298 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_299 | entropy_32 | entropy_32 | entropy_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_300 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_301 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_302 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_303 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_304 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_305 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_306 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.113 | 0.000 | 0.000 | 0.000 | 0.000 | 0.180 | 0.180 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_307 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_308 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_309 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_310 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_311 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_312 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_313 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_314 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_315 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_316 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_317 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_318 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_319 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_320 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_321 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.117 | 0.117 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_322 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_323 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_324 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_325 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_326 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_327 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_328 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_329 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_330 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_331 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_332 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_333 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_334 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_335 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_336 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_337 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_338 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_339 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_340 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_341 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_342 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_343 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_344 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_297 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.449 | 0.000 | 0.117 | 0.000 | 0.220 | 0.157 | 0.177 | 0.000 | 0.177 | 0.000 | 0.220 | 0.043 |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.428 | 0.000 | 0.223 | 0.000 | 0.323 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.392 | 0.000 | 0.223 | 0.000 | 0.282 | 0.157 | 0.282 | 0.000 | 0.282 | 0.000 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_300 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.426 | 0.000 | 0.160 | 0.000 | 0.263 | 0.263 | 0.200 | 0.000 | 0.200 | 0.000 | 0.263 | 0.063 |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | counterfactual_micro_probe_v1 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.423 | 0.000 | 0.230 | 0.000 | 0.290 | 0.137 | 0.290 | 0.000 | 0.290 | 0.000 | 0.290 | 0.000 |
| llada-8b-instruct-hf | plan_302 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.435 | 0.000 | 0.117 | 0.000 | 0.240 | 0.177 | 0.177 | 0.000 | 0.177 | 0.000 | 0.240 | 0.063 |
| llada-8b-instruct-hf | plan_303 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.471 | 0.000 | 0.160 | 0.000 | 0.283 | 0.283 | 0.220 | 0.000 | 0.220 | 0.000 | 0.283 | 0.063 |
| llada-8b-instruct-hf | plan_304 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.404 | 0.000 | 0.117 | 0.000 | 0.299 | 0.137 | 0.137 | 0.000 | 0.137 | 0.000 | 0.299 | 0.162 |
| llada-8b-instruct-hf | plan_305 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.471 | 0.000 | 0.197 | 0.000 | 0.360 | 0.277 | 0.277 | 0.000 | 0.277 | 0.000 | 0.360 | 0.083 |
| llada-8b-instruct-hf | plan_306 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.569 | 0.000 | 0.361 | 0.000 | 0.402 | 0.422 | 0.419 | 0.000 | 0.419 | 0.000 | 0.422 | 0.003 |
| llada-8b-instruct-hf | plan_307 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.400 | 0.000 | 0.117 | 0.000 | 0.200 | 0.065 | 0.137 | 0.000 | 0.137 | 0.000 | 0.200 | 0.063 |
| llada-8b-instruct-hf | plan_308 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.506 | 0.000 | 0.238 | 0.000 | 0.241 | 0.137 | 0.298 | 0.000 | 0.298 | 0.000 | 0.298 | 0.000 |
| llada-8b-instruct-hf | plan_309 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.450 | 0.000 | 0.239 | 0.000 | 0.356 | 0.336 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-8b-instruct-hf | plan_310 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.387 | 0.000 | 0.117 | 0.000 | 0.260 | 0.197 | 0.197 | 0.000 | 0.197 | 0.000 | 0.260 | 0.063 |
| llada-8b-instruct-hf | plan_311 | low_confidence_32 | counterfactual_micro_probe_v1 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.429 | 0.000 | 0.244 | 0.000 | 0.284 | 0.158 | 0.284 | 0.000 | 0.284 | 0.000 | 0.284 | 0.000 |
| llada-8b-instruct-hf | plan_312 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.503 | 0.000 | 0.138 | 0.000 | 0.241 | 0.178 | 0.178 | 0.000 | 0.178 | 0.000 | 0.241 | 0.063 |
| llada-8b-instruct-hf | plan_313 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.401 | 0.000 | 0.117 | 0.000 | 0.221 | 0.137 | 0.137 | 0.000 | 0.137 | 0.000 | 0.221 | 0.084 |
| llada-8b-instruct-hf | plan_314 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.439 | 0.000 | 0.117 | 0.000 | 0.240 | 0.240 | 0.177 | 0.000 | 0.177 | 0.000 | 0.240 | 0.063 |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.458 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.261 | 0.021 |
| llada-8b-instruct-hf | plan_316 | low_confidence_32 | counterfactual_micro_probe_v1 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.387 | 0.000 | 0.223 | 0.000 | 0.263 | 0.117 | 0.263 | 0.000 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_317 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.373 | 0.000 | 0.154 | 0.000 | 0.220 | 0.174 | 0.174 | 0.000 | 0.174 | 0.000 | 0.277 | 0.103 |
| llada-8b-instruct-hf | plan_318 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.369 | 0.000 | 0.117 | 0.000 | 0.200 | 0.200 | 0.157 | 0.000 | 0.157 | 0.000 | 0.200 | 0.043 |
| llada-8b-instruct-hf | plan_319 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.459 | 0.000 | 0.117 | 0.000 | 0.200 | 0.200 | 0.157 | 0.000 | 0.157 | 0.000 | 0.261 | 0.104 |
| llada-8b-instruct-hf | plan_320 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.446 | 0.000 | 0.117 | 0.000 | 0.294 | 0.260 | 0.197 | 0.000 | 0.197 | 0.000 | 0.294 | 0.097 |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.490 | 0.000 | 0.244 | 0.000 | 0.344 | 0.241 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_322 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.466 | 0.000 | 0.217 | 0.000 | 0.261 | 0.257 | 0.257 | 0.000 | 0.257 | 0.000 | 0.261 | 0.004 |
| llada-8b-instruct-hf | plan_323 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.388 | 0.000 | 0.117 | 0.000 | 0.280 | 0.177 | 0.177 | 0.000 | 0.177 | 0.000 | 0.280 | 0.103 |
| llada-8b-instruct-hf | plan_324 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.377 | 0.000 | 0.117 | 0.000 | 0.220 | 0.220 | 0.137 | 0.000 | 0.137 | 0.000 | 0.275 | 0.138 |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | counterfactual_micro_probe_v1 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.411 | 0.000 | 0.256 | 0.000 | 0.296 | 0.157 | 0.296 | 0.000 | 0.296 | 0.000 | 0.346 | 0.050 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.481 | 0.000 | 0.244 | 0.000 | 0.284 | 0.157 | 0.284 | 0.000 | 0.284 | 0.000 | 0.284 | 0.000 |
| llada-8b-instruct-hf | plan_327 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.453 | 0.000 | 0.154 | 0.000 | 0.260 | 0.214 | 0.214 | 0.000 | 0.214 | 0.000 | 0.297 | 0.083 |
| llada-8b-instruct-hf | plan_328 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.386 | 0.000 | 0.117 | 0.000 | 0.240 | 0.240 | 0.117 | 0.000 | 0.117 | 0.000 | 0.240 | 0.123 |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | counterfactual_micro_probe_v1 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.450 | 0.000 | 0.244 | 0.000 | 0.304 | 0.201 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.400 | 0.000 | 0.362 | 0.000 | 0.402 | 0.402 | 0.402 | 0.000 | 0.402 | 0.000 | 0.402 | 0.000 |
| llada-8b-instruct-hf | plan_331 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.495 | 0.000 | 0.181 | 0.000 | 0.223 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_332 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.444 | 0.000 | 0.117 | 0.000 | 0.240 | 0.177 | 0.177 | 0.000 | 0.177 | 0.000 | 0.240 | 0.063 |
| llada-8b-instruct-hf | plan_333 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.402 | 0.000 | 0.117 | 0.000 | 0.280 | 0.217 | 0.197 | 0.000 | 0.197 | 0.000 | 0.280 | 0.083 |
| llada-8b-instruct-hf | plan_334 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.512 | 0.000 | 0.238 | 0.000 | 0.260 | 0.261 | 0.338 | 0.000 | 0.338 | 0.000 | 0.338 | 0.000 |
| llada-8b-instruct-hf | plan_335 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.487 | 0.000 | 0.172 | 0.000 | 0.309 | 0.212 | 0.212 | 0.000 | 0.212 | 0.000 | 0.309 | 0.097 |
| llada-8b-instruct-hf | plan_336 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.431 | 0.000 | 0.117 | 0.000 | 0.255 | 0.117 | 0.117 | 0.000 | 0.117 | 0.000 | 0.255 | 0.138 |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.383 | 0.000 | 0.223 | 0.000 | 0.242 | 0.065 | 0.242 | 0.000 | 0.242 | 0.000 | 0.242 | 0.000 |
| llada-8b-instruct-hf | plan_338 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.423 | 0.000 | 0.117 | 0.000 | 0.315 | 0.137 | 0.137 | 0.000 | 0.137 | 0.000 | 0.315 | 0.178 |
| llada-8b-instruct-hf | plan_339 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.494 | 0.000 | 0.272 | 0.000 | 0.268 | 0.292 | 0.292 | 0.000 | 0.292 | 0.000 | 0.292 | 0.000 |
| llada-8b-instruct-hf | plan_340 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.397 | 0.000 | 0.130 | 0.000 | 0.261 | 0.130 | 0.130 | 0.000 | 0.130 | 0.000 | 0.261 | 0.132 |
| llada-8b-instruct-hf | plan_341 | low_confidence_32 | low_confidence_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.509 | 0.000 | 0.206 | 0.000 | 0.045 | 0.045 | 0.226 | 0.000 | 0.226 | 0.000 | 0.226 | 0.000 |
| llada-8b-instruct-hf | plan_342 | low_confidence_32 | counterfactual_micro_probe_v1 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.432 | 0.000 | 0.117 | 0.000 | 0.268 | 0.117 | 0.117 | 0.000 | 0.117 | 0.000 | 0.268 | 0.151 |
| llada-8b-instruct-hf | plan_343 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.387 | 0.000 | 0.117 | 0.000 | 0.180 | 0.117 | 0.117 | 0.000 | 0.117 | 0.000 | 0.180 | 0.063 |
| llada-8b-instruct-hf | plan_344 | low_confidence_32 | random_32 | counterfactual_micro_probe_v1 |  | counterfactual_micro_probe_v1 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.449 | 0.000 | 0.142 | 0.000 | 0.200 | 0.243 | 0.162 | 0.000 | 0.162 | 0.000 | 0.243 | 0.081 |
