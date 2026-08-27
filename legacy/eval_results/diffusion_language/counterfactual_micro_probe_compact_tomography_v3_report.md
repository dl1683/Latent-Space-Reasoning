# Diffusion Schedule-Selection Benchmark Report

Full model generations: `12`
Counterfactual probe generations: `12`
Arm selections: `48`
Run ID: `diffusion-dc9528f0b4f66da5`
Content hash: `dc9528f0b4f66da5e22fc205bee2d291e3f22754770989df3a55277a412174b4`
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
History mutability: `monotonic 12/12, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Counterfactual probe policy: `compact_tomography_probe_v3`
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
Trajectory wins/ties/losses vs fixed: `0/12/0`
Trajectory wins/ties/losses vs random: `0/12/0`
Oracle generation budget/task: `1.00`
Oracle task score: `0.337`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/12/0`
Selector regret vs trajectory: `0.000 over 0/12 improvable`
Repair arm coverage: `12/12` overall
Repair eligible coverage: `12/12`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.000`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/12/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/12/0`
Selector regret vs repair: `0.000 over 0/12 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `12/12` overall, `12/12` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.336756 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.336756 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.336756 | 0.000000 | 0.000000 | 0/12/0 | 0/12/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 12 | 1.00 | 0.337 | 0.695 | 0.426 |
| random | 12 | 1.00 | 0.337 | 0.695 | 0.426 |
| trajectory_selected | 12 | 1.00 | 0.337 | 0.695 | 0.426 |
| repair_selected | 12 | 1.00 | 0.337 | 0.695 | 0.426 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 12 | 1.00 | 0.337 | 0.695 | 0.426 |
| planning | random | 12 | 1.00 | 0.337 | 0.695 | 0.426 |
| planning | trajectory_selected | 12 | 1.00 | 0.337 | 0.695 | 0.426 |
| planning | repair_selected | 12 | 1.00 | 0.337 | 0.695 | 0.426 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_034 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.434 | 0.311 | 387 | True | 4 | 0.750 | True | True | 7.000 | 0.219 | 0.188 | 0.188 |
| llada-moe-7b-a1b-instruct-hf | plan_044 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.336 | 0.256 | 380 | True | 4 | 0.733 | True | True | 7.000 | 0.219 | 0.200 | 0.200 |
| llada-moe-7b-a1b-instruct-hf | plan_045 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.389 | 0.329 | 311 | True | 10 | 0.231 | True | True | 7.000 | 0.219 | 0.077 | 0.077 |
| llada-moe-7b-a1b-instruct-hf | plan_046 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.386 | 0.244 | 388 | True | 7 | 0.571 | True | True | 7.000 | 0.219 | 0.214 | 0.214 |
| llada-moe-7b-a1b-instruct-hf | plan_050 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.342 | 0.282 | 296 | True | 9 | 0.385 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-moe-7b-a1b-instruct-hf | plan_061 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.389 | 0.286 | 332 | True | 3 | 0.700 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_063 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.429 | 0.286 | 374 | True | 6 | 0.615 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-moe-7b-a1b-instruct-hf | plan_064 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.180 | 0.180 | 168 | True | 12 | 0.154 | True | True | 7.000 | 0.219 | 0.154 | 0.154 |
| llada-moe-7b-a1b-instruct-hf | plan_069 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.358 | 0.278 | 339 | True | 12 | 0.421 | True | True | 7.000 | 0.219 | 0.316 | 0.316 |
| llada-moe-7b-a1b-instruct-hf | plan_070 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.274 | 0.214 | 383 | True | 5 | 0.765 | True | True | 7.000 | 0.219 | 0.235 | 0.235 |
| llada-moe-7b-a1b-instruct-hf | plan_071 | low_confidence_32 | False | counterfactual_probe_triage_skip_no_repair | False | measured_generation | 0.220 | 0.180 | 201 | True | 12 | 0.176 | True | True | 7.000 | 0.219 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_072 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.304 | 0.244 | 339 | True | 2 | 0.952 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_034 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.450 | 0.000 | 0.386 | 0.000 | 0.434 | 0.434 | 0.434 | 0.000 | 0.434 | 0.000 | 0.434 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_044 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.391 | 0.000 | 0.256 | 0.000 | 0.336 | 0.336 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_045 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.322 | 0.000 | 0.352 | 0.000 | 0.389 | 0.389 | 0.389 | 0.000 | 0.389 | 0.000 | 0.389 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_046 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.370 | 0.000 | 0.244 | 0.000 | 0.386 | 0.386 | 0.386 | 0.000 | 0.386 | 0.000 | 0.386 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_050 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.355 | 0.000 | 0.282 | 0.000 | 0.342 | 0.342 | 0.342 | 0.000 | 0.342 | 0.000 | 0.342 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_061 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.438 | 0.000 | 0.286 | 0.000 | 0.389 | 0.389 | 0.389 | 0.000 | 0.389 | 0.000 | 0.389 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_063 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.415 | 0.000 | 0.286 | 0.000 | 0.429 | 0.429 | 0.429 | 0.000 | 0.429 | 0.000 | 0.429 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_064 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.142 | 0.000 | 0.180 | 0.000 | 0.180 | 0.180 | 0.180 | 0.000 | 0.180 | 0.000 | 0.180 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_069 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.335 | 0.000 | 0.278 | 0.000 | 0.358 | 0.358 | 0.358 | 0.000 | 0.358 | 0.000 | 0.358 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_070 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.410 | 0.000 | 0.214 | 0.000 | 0.274 | 0.274 | 0.274 | 0.000 | 0.274 | 0.000 | 0.274 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_071 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.231 | 0.000 | 0.180 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_072 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.479 | 0.000 | 0.244 | 0.000 | 0.304 | 0.304 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
