# Diffusion Schedule-Selection Benchmark Report

Full model generations: `49`
Arm selections: `40`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `False`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `False`
History mutability: `monotonic 49/49, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `state_adaptive`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `source_quality_or_short`
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
Constraint-gap rescue trigger: `prompt_gap`
Constraint-gap rescue limit: `3`
Constraint-gap rescue min terms: `6`
Constraint-gap rescue source-quality band: `0.400-0.500`
Constraint-gap rescue source controls: ``
Repair selector: `planning_quality_delta_risk_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `-0.000`
Trajectory task delta vs random: `0.036`
Trajectory wins/ties/losses vs fixed: `0/7/1`
Trajectory wins/ties/losses vs random: `4/1/3`
Oracle generation budget/task: `6.12`
Oracle task score: `0.482`
Oracle headroom vs trajectory: `0.070`
Oracle wins/ties/losses vs trajectory: `6/2/0`
Selector regret vs trajectory: `0.070 over 6/8 improvable`
Evolved task delta vs fixed: `0.039`
Evolved task delta vs random: `0.075`
Evolved task delta vs trajectory: `0.039`
Evolved wins/ties/losses vs fixed: `4/4/0`
Evolved wins/ties/losses vs random: `8/0/0`
Evolved wins/ties/losses vs trajectory: `4/4/0`
Oracle headroom vs evolved: `0.031`
Oracle wins/ties/losses vs evolved: `5/3/0`
Selector regret vs evolved: `0.031 over 5/8 improvable`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.070`
Repair task delta vs random: `0.106`
Repair task delta vs trajectory: `0.070`
Repair task delta vs evolved: `0.031`
Repair generation budget delta vs evolved: `2.12`
Repair task delta per extra generation vs evolved: `0.015`
Repair wins/ties/losses vs evolved: `5/3/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| random | 8 | 1.00 | 0.376 | 0.659 | 0.447 |
| trajectory_selected | 8 | 2.00 | 0.412 | 0.659 | 0.474 |
| evolved | 8 | 4.00 | 0.451 | 0.630 | 0.496 |
| repair_selected | 8 | 6.12 | 0.482 | 0.667 | 0.528 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.376 | 0.659 | 0.447 |
| planning | trajectory_selected | 8 | 2.00 | 0.412 | 0.659 | 0.474 |
| planning | evolved | 8 | 4.00 | 0.451 | 0.630 | 0.496 |
| planning | repair_selected | 8 | 6.12 | 0.482 | 0.667 | 0.528 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_revision_anchor25_repair | 1 | 1 | final | 48.0 | 0.000 | 0.000 | 0.085 | 0.085 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.614 | 0.688 | 0.633 |
| constraint_gap_revision_repair | 1 | 0 | final | 64.0 | 0.000 | 0.000 | 0.076 | 0.076 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.605 | 0.698 | 0.628 |
| constraint_gap_span_repair | 1 | 0 | final | 25.0 | 0.000 | 0.000 | -0.038 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.529 | 0.616 | 0.551 |
| prefix_25_repair | 7 | 0 | final | 48.0 | 0.000 | 0.000 | 0.020 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/1/1 | 0.453 | 0.688 | 0.512 |
| state_adaptive_history_repair | 7 | 4 | history | 48.0 | 0.000 | 0.000 | 0.020 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/1/1 | 0.453 | 0.688 | 0.512 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | constraint_gap_revision_anchor25_repair | constraint_gap_revision_anchor25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.339 | 0.483 | 0.085 | 0.085 | 0.399 | 0.473 | 0.399 | 0.529 | 0.614 | 0.085 | 0.614 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 47 | 0.431 | 0.444 | 0.023 | 0.023 | 0.604 | 0.604 | 0.602 | 0.654 | 0.695 | 0.040 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 31 | 0.500 | 0.500 | 0.021 | 0.021 | 0.443 | 0.284 | 0.443 | 0.443 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.295 | 0.369 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.347 | 0.347 | 0.000 | 0.347 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.338 | 0.382 | 0.000 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 47 | 0.366 | 0.370 | 0.021 | 0.021 | 0.298 | 0.341 | 0.298 | 0.363 | 0.424 | 0.061 | 0.424 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_spend_gate_kept_evolved_source_quality_or_short |  |  | 0.521 | 0.537 | 0.000 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 22 | 0.405 | 0.405 | 0.080 | 0.080 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
