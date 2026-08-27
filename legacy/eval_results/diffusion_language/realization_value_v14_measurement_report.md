# Diffusion Schedule-Selection Benchmark Report

Full model generations: `22`
Counterfactual probe generations: `8`
Arm selections: `41`
Run ID: `diffusion-22840dc9c690b7c0`
Content hash: `22840dc9c690b7c099b00bfaa16454a5ed40b5be0a0cf2f42613d0c3c9f9f56c`
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
History mutability: `monotonic 22/22, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `prefix`
Repair source policy: `random`
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
Trajectory task delta vs fixed: `-0.006`
Trajectory task delta vs random: `0.048`
Trajectory wins/ties/losses vs fixed: `0/10/1`
Trajectory wins/ties/losses vs random: `3/7/1`
Oracle generation budget/task: `2.00`
Oracle task score: `0.445`
Oracle headroom vs trajectory: `0.006`
Oracle wins/ties/losses vs trajectory: `1/10/0`
Selector regret vs trajectory: `0.006 over 1/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `-0.008`
Repair task delta vs random: `0.066`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.008`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.008 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.361696 | 0.000000 | 0.073562 | - | - |
| random perturbation | repair-covered tasks | 0.288134 | -0.073562 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.353821 | -0.007875 | 0.065687 | 0/7/1 | 3/4/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.445 | 0.537 | 0.468 |
| random | 11 | 1.00 | 0.391 | 0.436 | 0.402 |
| trajectory_selected | 11 | 2.00 | 0.439 | 0.529 | 0.462 |
| repair_selected | 8 | 2.00 | 0.354 | 0.687 | 0.437 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | random | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.038 | 0.760 |
| planning | fixed | 8 | 1.00 | 0.362 | 0.698 | 0.446 |
| planning | random | 8 | 1.00 | 0.288 | 0.559 | 0.356 |
| planning | trajectory_selected | 8 | 2.00 | 0.354 | 0.687 | 0.437 |
| planning | repair_selected | 8 | 2.00 | 0.354 | 0.687 | 0.437 |
| science | fixed | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.246 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.039 | 0.010 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_105 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.065 | 0.045 | 48 | True | 12 | 0.000 | True | True | 20.000 | 0.625 | 0.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_106 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.339 | 0.239 | 427 | True | 8 | 0.562 | True | True | 7.000 | 0.219 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_107 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.157 | 0.117 | 82 | True | 10 | 0.438 | True | True | 13.000 | 0.406 | 0.125 | 0.125 |
| llada-moe-7b-a1b-instruct-hf | plan_108 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.190 | 0.130 | 144 | True | 4 | 0.769 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-moe-7b-a1b-instruct-hf | plan_109 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.301 | 0.201 | 375 | True | 5 | 0.643 | True | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_110 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.468 | 0.408 | 348 | True | 3 | 0.800 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_111 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 397 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.353 | 0.353 |
| llada-moe-7b-a1b-instruct-hf | plan_112 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.524 | 0.389 | 407 | True | 6 | 0.571 | True | True | 7.000 | 0.219 | 0.214 | 0.214 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.038 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_105 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.518 | 0.000 | 0.290 | 0.000 | 0.390 | 0.065 | 0.390 | 0.000 | 0.390 | 0.000 | 0.390 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_106 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.380 | 0.000 | 0.239 | 0.000 | 0.339 | 0.339 | 0.339 | 0.000 | 0.339 | 0.000 | 0.339 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_107 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.427 | 0.000 | 0.217 | 0.000 | 0.318 | 0.157 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_108 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.435 | 0.000 | 0.193 | 0.000 | 0.292 | 0.190 | 0.292 | 0.000 | 0.292 | 0.000 | 0.292 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_109 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.384 | 0.000 | 0.201 | 0.000 | 0.301 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_110 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.504 | 0.000 | 0.408 | 0.000 | 0.468 | 0.468 | 0.468 | 0.000 | 0.468 | 0.000 | 0.468 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_111 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.467 | 0.000 | 0.201 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_112 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.378 | 0.000 | 0.326 | 0.000 | 0.524 | 0.524 | 0.461 | 0.000 | 0.461 | 0.000 | 0.524 | 0.063 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.246 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
