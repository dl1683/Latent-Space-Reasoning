# Diffusion Schedule-Selection Benchmark Report

Full model generations: `22`
Counterfactual probe generations: `8`
Arm selections: `41`
Run ID: `diffusion-3932b27b465b78a3`
Content hash: `3932b27b465b78a35e6adbc50f6f06c2c7105c8b65ea83fdde39237e9604ee2e`
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
Repair source policy: `fixed`
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
Trajectory task delta vs fixed: `0.003`
Trajectory task delta vs random: `0.045`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/7/1`
Oracle generation budget/task: `2.00`
Oracle task score: `0.424`
Oracle headroom vs trajectory: `0.004`
Oracle wins/ties/losses vs trajectory: `1/10/0`
Selector regret vs trajectory: `0.004 over 1/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.005`
Repair task delta vs random: `0.062`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.005`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.005 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.323929 | 0.000000 | 0.057339 | - | - |
| random perturbation | repair-covered tasks | 0.266589 | -0.057339 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.328437 | 0.004509 | 0.061848 | 1/7/0 | 3/4/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.417 | 0.537 | 0.447 |
| random | 11 | 1.00 | 0.376 | 0.439 | 0.392 |
| trajectory_selected | 11 | 2.00 | 0.421 | 0.537 | 0.450 |
| repair_selected | 8 | 2.00 | 0.328 | 0.698 | 0.421 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | random | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.038 | 0.760 |
| planning | fixed | 8 | 1.00 | 0.324 | 0.698 | 0.417 |
| planning | random | 8 | 1.00 | 0.267 | 0.564 | 0.341 |
| planning | trajectory_selected | 8 | 2.00 | 0.328 | 0.698 | 0.421 |
| planning | repair_selected | 8 | 2.00 | 0.328 | 0.698 | 0.421 |
| science | fixed | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.246 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.039 | 0.010 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_073 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.356 | 0.256 | 383 | True | 2 | 0.882 | True | True | 7.000 | 0.219 | 0.353 | 0.353 |
| llada-moe-7b-a1b-instruct-hf | plan_074 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.429 | 0.299 | 386 | True | 4 | 0.692 | True | True | 7.000 | 0.219 | 0.385 | 0.385 |
| llada-moe-7b-a1b-instruct-hf | plan_075 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.336 | 0.256 | 342 | True | 8 | 0.467 | True | True | 7.000 | 0.219 | 0.200 | 0.200 |
| llada-moe-7b-a1b-instruct-hf | plan_076 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.300 | 0.260 | 380 | True | 5 | 0.706 | True | True | 7.000 | 0.219 | 0.353 | 0.353 |
| llada-moe-7b-a1b-instruct-hf | plan_077 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.243 | 0.223 | 387 | True | 5 | 0.667 | True | True | 7.000 | 0.219 | 0.133 | 0.133 |
| llada-moe-7b-a1b-instruct-hf | plan_078 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.287 | 0.247 | 395 | True | 1 | 0.929 | True | True | 7.000 | 0.219 | 0.571 | 0.571 |
| llada-moe-7b-a1b-instruct-hf | plan_079 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.324 | 0.244 | 321 | True | 4 | 0.636 | True | True | 7.000 | 0.219 | 0.182 | 0.182 |
| llada-moe-7b-a1b-instruct-hf | plan_080 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.315 | 0.235 | 368 | True | 2 | 0.867 | True | True | 7.000 | 0.219 | 0.533 | 0.533 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.038 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_073 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.442 | 0.000 | 0.256 | 0.000 | 0.356 | 0.045 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_074 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.434 | 0.000 | 0.315 | 0.000 | 0.429 | 0.465 | 0.465 | 0.000 | 0.465 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_075 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.356 | 0.000 | 0.256 | 0.000 | 0.336 | 0.336 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_076 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.398 | 0.000 | 0.260 | 0.000 | 0.300 | 0.180 | 0.300 | 0.000 | 0.300 | 0.000 | 0.300 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_077 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.393 | 0.000 | 0.223 | 0.000 | 0.243 | 0.138 | 0.243 | 0.000 | 0.243 | 0.000 | 0.243 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_078 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.485 | 0.000 | 0.247 | 0.000 | 0.287 | 0.287 | 0.287 | 0.000 | 0.287 | 0.000 | 0.287 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_079 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.387 | 0.000 | 0.244 | 0.000 | 0.324 | 0.324 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_080 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.425 | 0.000 | 0.235 | 0.000 | 0.315 | 0.356 | 0.315 | 0.000 | 0.315 | 0.000 | 0.356 | 0.041 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.246 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
