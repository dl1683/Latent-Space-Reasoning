# Diffusion Schedule-Selection Benchmark Report

Full model generations: `1`
Counterfactual probe generations: `1`
Arm selections: `4`
Run ID: `diffusion-d12a018127a72611`
Content hash: `d12a018127a72611a0046797937f39b96f02d0e0f2fbd4ef806b2aa8150563f0`
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
History mutability: `monotonic 1/1, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Repair source-quality threshold: `0.990`
Repair source min chars: `40`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `8`
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
Trajectory wins/ties/losses vs fixed: `0/1/0`
Trajectory wins/ties/losses vs random: `0/1/0`
Oracle generation budget/task: `1.00`
Oracle task score: `0.274`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/1/0`
Selector regret vs trajectory: `0.000 over 0/1 improvable`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.000`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/1/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `1/1` overall, `1/1` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.273929 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.273929 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.273929 | 0.000000 | 0.000000 | 0/1/0 | 0/1/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.274 | 0.698 | 0.380 |
| random | 1 | 1.00 | 0.274 | 0.698 | 0.380 |
| trajectory_selected | 1 | 1.00 | 0.274 | 0.698 | 0.380 |
| repair_selected | 1 | 1.00 | 0.274 | 0.698 | 0.380 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 1 | 1.00 | 0.274 | 0.698 | 0.380 |
| planning | random | 1 | 1.00 | 0.274 | 0.698 | 0.380 |
| planning | trajectory_selected | 1 | 1.00 | 0.274 | 0.698 | 0.380 |
| planning | repair_selected | 1 | 1.00 | 0.274 | 0.698 | 0.380 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_070 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.274 | 0.214 | 383 | True | 5 | 0.765 | True | True | 7.000 | 0.219 | 0.235 | 0.235 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_070 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.410 | 0.000 | 0.214 | 0.000 | 0.274 | 0.274 | 0.274 | 0.000 | 0.274 | 0.000 | 0.274 | 0.000 |
