# Diffusion Schedule-Selection Benchmark Report

Full model generations: `18`
Arm selections: `12`
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
History mutability: `monotonic 18/18, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_gap`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `always`
Repair source-quality threshold: `0.500`
Repair source min chars: `320`
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
Repair selector: `planning_quality_delta_risk_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.000`
Trajectory wins/ties/losses vs fixed: `0/3/0`
Trajectory wins/ties/losses vs random: `0/3/0`
Oracle generation budget/task: `6.00`
Oracle task score: `0.357`
Oracle headroom vs trajectory: `0.036`
Oracle wins/ties/losses vs trajectory: `2/1/0`
Selector regret vs trajectory: `0.036 over 2/3 improvable`
Repair arm coverage: `3/3` overall
Repair eligible coverage: `3/3`
Repair task delta vs fixed: `0.036`
Repair task delta vs random: `0.036`
Repair task delta vs trajectory: `0.036`
Repair task delta vs evolved: `0.036`
Repair generation budget delta vs evolved: `5.00`
Repair task delta per extra generation vs evolved: `0.007`
Repair wins/ties/losses vs evolved: `2/1/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/3/0`
Selector regret vs repair: `0.000 over 0/3 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 3 | 1.00 | 0.322 | 0.659 | 0.406 |
| random | 3 | 1.00 | 0.322 | 0.659 | 0.406 |
| trajectory_selected | 3 | 1.00 | 0.322 | 0.659 | 0.406 |
| repair_selected | 3 | 6.00 | 0.357 | 0.678 | 0.437 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 3 | 1.00 | 0.322 | 0.659 | 0.406 |
| planning | random | 3 | 1.00 | 0.322 | 0.659 | 0.406 |
| planning | trajectory_selected | 3 | 1.00 | 0.322 | 0.659 | 0.406 |
| planning | repair_selected | 3 | 6.00 | 0.357 | 0.678 | 0.437 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_revision_anchor25_repair | 3 | 0 | final | 48.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/3/0 | 0.322 | 0.695 | 0.415 |
| constraint_gap_revision_repair | 3 | 0 | final | 64.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/3/0 | 0.322 | 0.698 | 0.416 |
| constraint_gap_span_repair | 3 | 1 | final | 37.7 | 0.000 | 0.000 | 0.015 | 0.022 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/2/0 | 0.344 | 0.663 | 0.423 |
| prefix_25_repair | 3 | 0 | final | 48.0 | 0.000 | 0.000 | -0.011 | -0.024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/1/1 | 0.297 | 0.564 | 0.364 |
| state_adaptive_history_repair | 3 | 1 | history | 42.7 | 0.000 | 0.000 | -0.011 | -0.024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/1/1 | 0.297 | 0.564 | 0.364 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.339 | 0.000 | 0.046 | 0.046 | 0.399 | 0.399 | 0.399 | 0.000 | 0.465 | 0.066 | 0.465 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_revision_anchor25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  | 0.295 | 0.000 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 22 | 0.405 | 0.000 | 0.080 | 0.080 | 0.283 | 0.283 | 0.283 | 0.000 | 0.323 | 0.040 | 0.323 | 0.000 |
