# Diffusion Schedule-Selection Benchmark Report

Full model generations: `48`
Arm selections: `32`
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
History mutability: `monotonic 48/48, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory wins/ties/losses vs fixed: `0/8/0`
Trajectory wins/ties/losses vs random: `0/8/0`
Oracle generation budget/task: `6.00`
Oracle task score: `0.471`
Oracle headroom vs trajectory: `0.059`
Oracle wins/ties/losses vs trajectory: `8/0/0`
Selector regret vs trajectory: `0.059 over 8/8 improvable`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.057`
Repair task delta vs random: `0.057`
Repair task delta vs trajectory: `0.057`
Repair task delta vs evolved: `0.057`
Repair generation budget delta vs evolved: `5.00`
Repair task delta per extra generation vs evolved: `0.011`
Repair wins/ties/losses vs evolved: `7/1/0`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.002 over 1/8 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| random | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| trajectory_selected | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| repair_selected | 8 | 6.00 | 0.470 | 0.684 | 0.523 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | trajectory_selected | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | repair_selected | 8 | 6.00 | 0.470 | 0.684 | 0.523 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_revision_anchor25_repair | 8 | 0 | final | 48.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/8/0 | 0.412 | 0.693 | 0.483 |
| constraint_gap_revision_repair | 8 | 0 | final | 64.0 | 0.000 | 0.000 | -0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/7/0 | 0.413 | 0.698 | 0.484 |
| constraint_gap_span_repair | 8 | 4 | final | 35.9 | 0.000 | 0.000 | 0.017 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/2/1 | 0.437 | 0.672 | 0.496 |
| prefix_25_repair | 8 | 1 | final | 48.0 | 0.000 | 0.000 | -0.009 | -0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/3/2 | 0.401 | 0.635 | 0.459 |
| state_adaptive_history_repair | 8 | 2 | history | 44.1 | 0.000 | 0.000 | -0.018 | -0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/4/2 | 0.390 | 0.635 | 0.451 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.339 | 0.000 | 0.046 | 0.046 | 0.399 | 0.399 | 0.399 | 0.000 | 0.465 | 0.066 | 0.465 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.422 | 0.000 | 0.073 | 0.073 | 0.604 | 0.604 | 0.604 | 0.000 | 0.695 | 0.090 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 31 | 0.500 | 0.000 | 0.021 | 0.021 | 0.443 | 0.443 | 0.443 | 0.000 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.295 | 0.000 | 0.034 | 0.034 | 0.283 | 0.283 | 0.283 | 0.000 | 0.316 | 0.034 | 0.316 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.338 | 0.000 | 0.037 | 0.037 | 0.378 | 0.378 | 0.378 | 0.000 | 0.435 | 0.057 | 0.435 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.366 | 0.000 | 0.079 | 0.079 | 0.298 | 0.298 | 0.298 | 0.000 | 0.446 | 0.149 | 0.446 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  | 0.521 | 0.000 | 0.000 | 0.000 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 | 0.623 | 0.012 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 22 | 0.405 | 0.000 | 0.080 | 0.080 | 0.283 | 0.283 | 0.283 | 0.000 | 0.323 | 0.040 | 0.323 | 0.000 |
