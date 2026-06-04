# Diffusion Schedule-Selection Benchmark Report

Full model generations: `8`
Counterfactual probe generations: `8`
Arm selections: `32`
Run ID: `diffusion-5635caed114fc8cb`
Content hash: `5635caed114fc8cb5c035757fb3e8f82c27234da60c1b806b65d44d274c28024`
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
Oracle task score: `0.412`
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
| fixed baseline | repair-covered tasks | 0.412277 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.412277 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.412277 | 0.000000 | 0.000000 | 0/8/0 | 0/8/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| random | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| trajectory_selected | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| repair_selected | 8 | 1.00 | 0.412 | 0.698 | 0.484 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| planning | random | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| planning | trajectory_selected | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| planning | repair_selected | 8 | 1.00 | 0.412 | 0.698 | 0.484 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.465 | 0.348 | 331 | True | 9 | 0.467 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.689 | 0.559 | 263 | True | 12 | 0.278 | True | True | 7.000 | 0.219 | 0.111 | 0.111 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.422 | 0.324 | 241 | True | 6 | 0.600 | True | True | 7.000 | 0.219 | 0.200 | 0.200 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.338 | 0.278 | 373 | True | 2 | 0.882 | True | True | 7.000 | 0.219 | 0.235 | 0.235 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.421 | 0.299 | 358 | True | 10 | 0.412 | True | True | 7.000 | 0.219 | 0.176 | 0.176 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.391 | 0.301 | 351 | True | 9 | 0.438 | True | True | 7.000 | 0.219 | 0.188 | 0.188 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.307 | 0.247 | 322 | True | 8 | 0.417 | True | True | 7.000 | 0.219 | 0.167 | 0.167 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.264 | 0.244 | 241 | True | 12 | 0.062 | True | True | 7.000 | 0.219 | 0.000 | 0.000 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.426 | 0.000 | 0.395 | 0.000 | 0.465 | 0.465 | 0.465 | 0.000 | 0.465 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.441 | 0.000 | 0.586 | 0.000 | 0.689 | 0.689 | 0.689 | 0.000 | 0.689 | 0.000 | 0.689 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.424 | 0.000 | 0.384 | 0.000 | 0.422 | 0.422 | 0.422 | 0.000 | 0.422 | 0.000 | 0.422 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.467 | 0.000 | 0.278 | 0.000 | 0.338 | 0.338 | 0.338 | 0.000 | 0.338 | 0.000 | 0.338 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.330 | 0.000 | 0.299 | 0.000 | 0.421 | 0.421 | 0.421 | 0.000 | 0.421 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.372 | 0.000 | 0.345 | 0.000 | 0.391 | 0.391 | 0.391 | 0.000 | 0.391 | 0.000 | 0.391 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.331 | 0.000 | 0.247 | 0.000 | 0.307 | 0.307 | 0.307 | 0.000 | 0.307 | 0.000 | 0.307 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.181 | 0.000 | 0.244 | 0.000 | 0.264 | 0.264 | 0.264 | 0.000 | 0.264 | 0.000 | 0.264 | 0.000 |
