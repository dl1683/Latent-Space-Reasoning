# Diffusion Schedule-Selection Benchmark Report

Full model generations: `8`
Counterfactual probe generations: `8`
Arm selections: `32`
Run ID: `diffusion-83e2101f1bbedac4`
Content hash: `83e2101f1bbedac41a700f502f368685c264e4dae69a36dab5f6fb4af0b37c75`
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
History mutability: `monotonic 8/8, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `prefix`
Repair source policy: `trajectory`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `counterfactual_micro_probe_v1`
Counterfactual probe mode: `all`
Counterfactual probe policy: `span_tomography_probe_v4`
Repair source-quality threshold: `0.990`
Repair source min chars: `40`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `12`
Repair source prompt coverage band: `0.000-1.000`
Repair value-proxy source-quality max: `0.310`
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
Repair selector: `candidate_aware_promotion_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.000`
Trajectory wins/ties/losses vs fixed: `0/8/0`
Trajectory wins/ties/losses vs random: `0/8/0`
Oracle generation budget/task: `1.00`
Oracle task score: `0.398`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/8/0`
Selector regret vs trajectory: `0.000 over 0/8 improvable`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.000`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/8` overall, `8/8` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.398170 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.398170 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.398170 | 0.000000 | 0.000000 | 0/8/0 | 0/8/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.398 | 0.698 | 0.473 |
| random | 8 | 1.00 | 0.398 | 0.698 | 0.473 |
| trajectory_selected | 8 | 1.00 | 0.398 | 0.698 | 0.473 |
| repair_selected | 8 | 1.00 | 0.398 | 0.698 | 0.473 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.398 | 0.698 | 0.473 |
| planning | random | 8 | 1.00 | 0.398 | 0.698 | 0.473 |
| planning | trajectory_selected | 8 | 1.00 | 0.398 | 0.698 | 0.473 |
| planning | repair_selected | 8 | 1.00 | 0.398 | 0.698 | 0.473 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_017 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.466 | 0.366 | 364 | True | 10 | 0.444 | True | True | 7.000 | 0.219 | 0.167 | 0.167 |
| llada-moe-7b-a1b-instruct-hf | plan_018 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.348 | 0.248 | 348 | True | 8 | 0.556 | True | True | 7.000 | 0.219 | 0.167 | 0.167 |
| llada-moe-7b-a1b-instruct-hf | plan_019 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.413 | 0.333 | 360 | True | 12 | 0.235 | True | True | 7.000 | 0.219 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_020 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.260 | 0.180 | 339 | True | 6 | 0.625 | True | True | 7.000 | 0.219 | 0.125 | 0.125 |
| llada-moe-7b-a1b-instruct-hf | plan_021 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.356 | 0.256 | 317 | True | 6 | 0.600 | True | True | 7.000 | 0.219 | 0.133 | 0.133 |
| llada-moe-7b-a1b-instruct-hf | plan_022 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.446 | 0.346 | 363 | True | 12 | 0.400 | True | True | 7.000 | 0.219 | 0.150 | 0.150 |
| llada-moe-7b-a1b-instruct-hf | plan_023 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.485 | 0.320 | 341 | True | 7 | 0.562 | True | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-moe-7b-a1b-instruct-hf | plan_024 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.411 | 0.331 | 348 | True | 7 | 0.588 | True | True | 7.000 | 0.219 | 0.294 | 0.294 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_017 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.392 | 0.000 | 0.410 | 0.000 | 0.466 | 0.466 | 0.466 | 0.000 | 0.466 | 0.000 | 0.466 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_018 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.332 | 0.000 | 0.248 | 0.000 | 0.348 | 0.348 | 0.348 | 0.000 | 0.348 | 0.000 | 0.348 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_019 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.338 | 0.000 | 0.356 | 0.000 | 0.413 | 0.413 | 0.413 | 0.000 | 0.413 | 0.000 | 0.413 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_020 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.292 | 0.000 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_021 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.386 | 0.000 | 0.256 | 0.000 | 0.356 | 0.356 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_022 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.340 | 0.000 | 0.386 | 0.000 | 0.446 | 0.446 | 0.446 | 0.000 | 0.446 | 0.000 | 0.446 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_023 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.428 | 0.000 | 0.377 | 0.000 | 0.485 | 0.485 | 0.485 | 0.000 | 0.485 | 0.000 | 0.485 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_024 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.424 | 0.000 | 0.390 | 0.000 | 0.411 | 0.411 | 0.411 | 0.000 | 0.411 | 0.000 | 0.411 | 0.000 |
