# Diffusion Schedule-Selection Benchmark Report

Full model generations: `22`
Counterfactual probe generations: `8`
Arm selections: `41`
Run ID: `diffusion-36cf5bcb4eacd64e`
Content hash: `36cf5bcb4eacd64ec750326f4390b864bdf6bf9ec41d5267782863ee53795194`
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
Trajectory task delta vs fixed: `-0.011`
Trajectory task delta vs random: `0.050`
Trajectory wins/ties/losses vs fixed: `1/9/1`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `2.00`
Oracle task score: `0.450`
Oracle headroom vs trajectory: `0.019`
Oracle wins/ties/losses vs trajectory: `1/10/0`
Selector regret vs trajectory: `0.019 over 1/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `-0.015`
Repair task delta vs random: `0.068`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.026`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.026 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.357946 | 0.000000 | 0.083562 | - | - |
| random perturbation | repair-covered tasks | 0.274384 | -0.083562 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.342839 | -0.015107 | 0.068455 | 1/6/1 | 3/5/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.442 | 0.537 | 0.466 |
| random | 11 | 1.00 | 0.381 | 0.431 | 0.394 |
| trajectory_selected | 11 | 2.00 | 0.431 | 0.525 | 0.455 |
| repair_selected | 8 | 2.00 | 0.343 | 0.681 | 0.427 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | random | 1 | 1.00 | 1.000 | 0.038 | 0.760 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.038 | 0.760 |
| planning | fixed | 8 | 1.00 | 0.358 | 0.698 | 0.443 |
| planning | random | 8 | 1.00 | 0.274 | 0.552 | 0.344 |
| planning | trajectory_selected | 8 | 2.00 | 0.343 | 0.681 | 0.427 |
| planning | repair_selected | 8 | 2.00 | 0.343 | 0.681 | 0.427 |
| science | fixed | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.246 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.246 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.039 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.039 | 0.010 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_089 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.157 | 0.117 | 91 | True | 12 | 0.316 | True | True | 7.000 | 0.219 | 0.211 | 0.211 |
| llada-moe-7b-a1b-instruct-hf | plan_090 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.240 | 0.180 | 173 | True | 10 | 0.375 | True | True | 7.000 | 0.219 | 0.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_091 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.427 | 0.327 | 383 | True | 2 | 0.867 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-moe-7b-a1b-instruct-hf | plan_092 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.374 | 0.294 | 382 | True | 7 | 0.533 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-moe-7b-a1b-instruct-hf | plan_093 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.394 | 0.244 | 333 | True | 6 | 0.455 | True | True | 7.000 | 0.219 | 0.182 | 0.182 |
| llada-moe-7b-a1b-instruct-hf | plan_094 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.232 | 0.172 | 166 | True | 12 | 0.250 | True | True | 7.000 | 0.219 | 0.188 | 0.188 |
| llada-moe-7b-a1b-instruct-hf | plan_095 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.045 | 0.045 | 15 | True | 12 | 0.000 | True | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_096 | random_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.325 | 0.265 | 370 | True | 12 | 0.353 | True | True | 7.000 | 0.219 | 0.176 | 0.176 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.038 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_089 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.384 | 0.000 | 0.294 | 0.000 | 0.374 | 0.157 | 0.374 | 0.000 | 0.374 | 0.000 | 0.374 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_090 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.273 | 0.000 | 0.201 | 0.000 | 0.261 | 0.240 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_091 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.498 | 0.000 | 0.327 | 0.000 | 0.427 | 0.427 | 0.427 | 0.000 | 0.427 | 0.000 | 0.427 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_092 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.397 | 0.000 | 0.294 | 0.000 | 0.374 | 0.374 | 0.374 | 0.000 | 0.374 | 0.000 | 0.374 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_093 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.325 | 0.000 | 0.244 | 0.000 | 0.394 | 0.394 | 0.394 | 0.000 | 0.394 | 0.000 | 0.394 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_094 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.257 | 0.000 | 0.172 | 0.000 | 0.438 | 0.232 | 0.232 | 0.000 | 0.232 | 0.000 | 0.438 | 0.206 |
| llada-moe-7b-a1b-instruct-hf | plan_095 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.486 | 0.000 | 0.294 | 0.000 | 0.354 | 0.045 | 0.354 | 0.000 | 0.354 | 0.000 | 0.354 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_096 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.320 | 0.000 | 0.265 | 0.000 | 0.240 | 0.325 | 0.325 | 0.000 | 0.325 | 0.000 | 0.325 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.246 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
