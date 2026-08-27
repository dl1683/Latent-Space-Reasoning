# Diffusion Schedule-Selection Benchmark Report

Full model generations: `22`
Counterfactual probe generations: `8`
Arm selections: `41`
Run ID: `diffusion-0bd575b42c734811`
Content hash: `0bd575b42c734811be7892b80cf75aab88324c30905519062d226a81aab7cf09`
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
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.118`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `5/6/0`
Oracle generation budget/task: `2.00`
Oracle task score: `0.415`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/11/0`
Selector regret vs trajectory: `0.000 over 0/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.163`
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

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.320491 | 0.000000 | 0.162562 | - | - |
| random perturbation | repair-covered tasks | 0.157929 | -0.162562 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.320491 | 0.000000 | 0.162562 | 0/8/0 | 5/3/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.415 | 0.537 | 0.445 |
| random | 11 | 1.00 | 0.297 | 0.303 | 0.298 |
| trajectory_selected | 11 | 2.00 | 0.415 | 0.537 | 0.445 |
| repair_selected | 8 | 2.00 | 0.320 | 0.698 | 0.415 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | random | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.038 | 0.760 |
| planning | fixed | 8 | 1.00 | 0.320 | 0.698 | 0.415 |
| planning | random | 8 | 1.00 | 0.158 | 0.377 | 0.213 |
| planning | trajectory_selected | 8 | 2.00 | 0.320 | 0.698 | 0.415 |
| planning | repair_selected | 8 | 2.00 | 0.320 | 0.698 | 0.415 |
| science | fixed | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.246 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.039 | 0.010 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_081 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.314 | 0.214 | 305 | True | 4 | 0.750 | True | True | 7.000 | 0.219 | 0.312 | 0.312 |
| llada-moe-7b-a1b-instruct-hf | plan_082 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.303 | 0.223 | 353 | True | 0 | 1.000 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_083 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.301 | 0.201 | 333 | True | 11 | 0.312 | True | True | 7.000 | 0.219 | 0.188 | 0.188 |
| llada-moe-7b-a1b-instruct-hf | plan_084 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.383 | 0.303 | 351 | True | 3 | 0.786 | True | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_085 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.353 | 0.273 | 378 | True | 7 | 0.533 | True | True | 7.000 | 0.219 | 0.467 | 0.467 |
| llada-moe-7b-a1b-instruct-hf | plan_086 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.366 | 0.306 | 356 | True | 1 | 0.929 | True | True | 7.000 | 0.219 | 0.214 | 0.214 |
| llada-moe-7b-a1b-instruct-hf | plan_087 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.263 | 0.223 | 324 | True | 5 | 0.722 | True | True | 7.000 | 0.219 | 0.278 | 0.278 |
| llada-moe-7b-a1b-instruct-hf | plan_088 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.280 | 0.260 | 352 | True | 4 | 0.733 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.038 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_081 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.355 | 0.000 | 0.214 | 0.000 | 0.314 | 0.137 | 0.314 | 0.000 | 0.314 | 0.000 | 0.314 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_082 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.421 | 0.000 | 0.223 | 0.000 | 0.303 | 0.045 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_083 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.212 | 0.000 | 0.201 | 0.000 | 0.301 | 0.045 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_084 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.428 | 0.000 | 0.303 | 0.000 | 0.383 | 0.383 | 0.383 | 0.000 | 0.383 | 0.000 | 0.383 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_085 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.325 | 0.000 | 0.273 | 0.000 | 0.353 | 0.045 | 0.353 | 0.000 | 0.353 | 0.000 | 0.353 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_086 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.501 | 0.000 | 0.306 | 0.000 | 0.366 | 0.065 | 0.366 | 0.000 | 0.366 | 0.000 | 0.366 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_087 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.403 | 0.000 | 0.223 | 0.000 | 0.263 | 0.263 | 0.263 | 0.000 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_088 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.438 | 0.000 | 0.260 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.246 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
