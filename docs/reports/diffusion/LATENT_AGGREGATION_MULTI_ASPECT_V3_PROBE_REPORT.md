# Diffusion Schedule-Selection Benchmark Report

Full model generations: `120`
Counterfactual probe generations: `24`
Arm selections: `168`
Run ID: `diffusion-8f8bbe8378edf9b5`
Content hash: `8f8bbe8378edf9b5170b745d6b1059408d97ff78345820f3907f2fdf6630c90f`
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
History mutability: `monotonic 120/120, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.006`
Trajectory task delta vs random: `0.015`
Trajectory wins/ties/losses vs fixed: `7/39/2`
Trajectory wins/ties/losses vs random: `13/30/5`
Oracle generation budget/task: `2.50`
Oracle task score: `0.186`
Oracle headroom vs trajectory: `0.010`
Oracle wins/ties/losses vs trajectory: `9/39/0`
Selector regret vs trajectory: `0.010 over 9/48 improvable`
Repair arm coverage: `24/48` overall
Repair eligible coverage: `24/24`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.021`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/24/0`
Oracle headroom vs repair: `0.001`
Oracle wins/ties/losses vs repair: `3/21/0`
Selector regret vs repair: `0.001 over 3/24 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `24/48` overall, `24/24` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.313958 | 0.000000 | 0.020693 | - | - |
| random perturbation | repair-covered tasks | 0.293265 | -0.020693 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.313958 | 0.000000 | 0.020693 | 0/24/0 | 7/15/2 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.170 | 0.431 | 0.235 |
| random | 48 | 1.00 | 0.161 | 0.393 | 0.219 |
| trajectory_selected | 48 | 2.50 | 0.176 | 0.439 | 0.242 |
| repair_selected | 24 | 2.00 | 0.314 | 0.698 | 0.410 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.170 | 0.431 | 0.235 |
| planning | random | 48 | 1.00 | 0.161 | 0.393 | 0.219 |
| planning | trajectory_selected | 48 | 2.50 | 0.176 | 0.439 | 0.242 |
| planning | repair_selected | 24 | 2.00 | 0.314 | 0.698 | 0.410 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_201 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.323 | 0.223 | 306 | True | 5 | 0.706 | True | True | 7.000 | 0.219 | 0.294 | 0.294 |
| llada-8b-instruct-hf | plan_202 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.321 | 0.281 | 305 | True | 5 | 0.737 | True | True | 7.000 | 0.219 | 0.263 | 0.263 |
| llada-8b-instruct-hf | plan_203 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.280 | 0.180 | 355 | True | 1 | 0.923 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-8b-instruct-hf | plan_204 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 372 | True | 7 | 0.684 | True | True | 7.000 | 0.219 | 0.105 | 0.105 |
| llada-8b-instruct-hf | plan_205 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 305 | True | 8 | 0.500 | True | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_206 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.486 | 0.369 | 372 | True | 2 | 0.857 | True | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-8b-instruct-hf | plan_207 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.378 | 0.278 | 317 | True | 6 | 0.538 | True | True | 7.000 | 0.219 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_208 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.311 | 0.251 | 385 | True | 4 | 0.733 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_209 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.281 | 0.201 | 328 | True | 4 | 0.692 | True | True | 7.000 | 0.219 | 0.538 | 0.538 |
| llada-8b-instruct-hf | plan_210 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.315 | 0.217 | 297 | True | 6 | 0.571 | True | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-8b-instruct-hf | plan_211 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.339 | 0.239 | 364 | True | 7 | 0.533 | True | True | 7.000 | 0.219 | 0.267 | 0.267 |
| llada-8b-instruct-hf | plan_212 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.281 | 0.201 | 339 | True | 4 | 0.714 | True | True | 7.000 | 0.219 | 0.357 | 0.357 |
| llada-8b-instruct-hf | plan_213 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.274 | 0.214 | 309 | True | 4 | 0.636 | True | True | 7.000 | 0.219 | 0.545 | 0.545 |
| llada-8b-instruct-hf | plan_214 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.299 | 0.239 | 261 | True | 8 | 0.385 | True | True | 7.000 | 0.219 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_215 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.335 | 0.235 | 325 | True | 9 | 0.500 | True | True | 7.000 | 0.219 | 0.062 | 0.062 |
| llada-8b-instruct-hf | plan_216 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.391 | 0.311 | 334 | True | 1 | 0.917 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_217 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.281 | 0.201 | 292 | True | 2 | 0.818 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_218 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.281 | 0.201 | 291 | True | 7 | 0.533 | True | True | 7.000 | 0.219 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_219 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.303 | 0.223 | 309 | True | 6 | 0.625 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_220 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 362 | True | 4 | 0.692 | True | True | 7.000 | 0.219 | 0.385 | 0.385 |
| llada-8b-instruct-hf | plan_221 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.318 | 0.278 | 364 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.545 | 0.545 |
| llada-8b-instruct-hf | plan_222 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.399 | 0.289 | 329 | True | 7 | 0.562 | True | True | 7.000 | 0.219 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_223 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.315 | 0.235 | 355 | True | 2 | 0.846 | True | True | 7.000 | 0.219 | 0.462 | 0.462 |
| llada-8b-instruct-hf | plan_224 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 329 | True | 4 | 0.750 | True | True | 7.000 | 0.219 | 0.438 | 0.438 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_201 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.130 | 0.000 | 0.000 | 0.000 | 0.130 | 0.000 |
| dream-7b-instruct-hf | plan_202 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_203 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.110 | 0.000 | 0.000 | 0.000 | 0.108 | 0.000 | 0.108 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_204 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.117 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_205 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_206 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_207 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_208 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_209 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_210 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.110 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_211 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_212 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_213 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_214 | entropy_32 | origin_64 | entropy_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.119 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.066 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_215 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_216 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_217 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.127 | 0.000 | 0.000 | 0.000 | 0.180 | 0.180 | 0.045 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_218 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_219 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_220 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_221 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_222 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_223 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_224 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| llada-8b-instruct-hf | plan_201 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.393 | 0.000 | 0.223 | 0.000 | 0.323 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_202 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.418 | 0.000 | 0.281 | 0.000 | 0.321 | 0.321 | 0.321 | 0.000 | 0.321 | 0.000 | 0.341 | 0.020 |
| llada-8b-instruct-hf | plan_203 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.432 | 0.000 | 0.180 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_204 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.356 | 0.000 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_205 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.333 | 0.000 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_206 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.506 | 0.000 | 0.369 | 0.000 | 0.486 | 0.420 | 0.486 | 0.000 | 0.486 | 0.000 | 0.486 | 0.000 |
| llada-8b-instruct-hf | plan_207 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.339 | 0.000 | 0.278 | 0.000 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_208 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.436 | 0.000 | 0.251 | 0.000 | 0.311 | 0.311 | 0.311 | 0.000 | 0.311 | 0.000 | 0.311 | 0.000 |
| llada-8b-instruct-hf | plan_209 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.375 | 0.000 | 0.201 | 0.000 | 0.281 | 0.197 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_210 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.380 | 0.000 | 0.217 | 0.000 | 0.315 | 0.315 | 0.315 | 0.000 | 0.315 | 0.000 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_211 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.362 | 0.000 | 0.239 | 0.000 | 0.339 | 0.339 | 0.339 | 0.000 | 0.339 | 0.000 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_212 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.384 | 0.000 | 0.201 | 0.000 | 0.281 | 0.260 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_213 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.376 | 0.000 | 0.214 | 0.000 | 0.274 | 0.281 | 0.274 | 0.000 | 0.274 | 0.000 | 0.281 | 0.008 |
| llada-8b-instruct-hf | plan_214 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.320 | 0.000 | 0.239 | 0.000 | 0.299 | 0.214 | 0.299 | 0.000 | 0.299 | 0.000 | 0.299 | 0.000 |
| llada-8b-instruct-hf | plan_215 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.339 | 0.000 | 0.235 | 0.000 | 0.335 | 0.335 | 0.335 | 0.000 | 0.335 | 0.000 | 0.335 | 0.000 |
| llada-8b-instruct-hf | plan_216 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.497 | 0.000 | 0.311 | 0.000 | 0.391 | 0.324 | 0.391 | 0.000 | 0.391 | 0.000 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_217 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.402 | 0.000 | 0.201 | 0.000 | 0.281 | 0.277 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_218 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.325 | 0.000 | 0.201 | 0.000 | 0.281 | 0.282 | 0.281 | 0.000 | 0.281 | 0.000 | 0.282 | 0.001 |
| llada-8b-instruct-hf | plan_219 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.355 | 0.000 | 0.223 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_220 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.350 | 0.000 | 0.201 | 0.000 | 0.241 | 0.065 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_221 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.508 | 0.000 | 0.278 | 0.000 | 0.318 | 0.318 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-8b-instruct-hf | plan_222 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.405 | 0.000 | 0.289 | 0.000 | 0.399 | 0.399 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_223 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.434 | 0.000 | 0.235 | 0.000 | 0.315 | 0.315 | 0.315 | 0.000 | 0.315 | 0.000 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_224 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.392 | 0.000 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
