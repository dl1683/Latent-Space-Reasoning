# Diffusion Schedule-Selection Benchmark Report

Full model generations: `44`
Arm selections: `40`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `False`
History mutability: `monotonic 44/44, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span`
Repair source policy: `evolved_and_trajectory`
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
Repair selector: `inherit`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.002`
Trajectory task delta vs random: `0.042`
Trajectory wins/ties/losses vs fixed: `1/7/0`
Trajectory wins/ties/losses vs random: `3/5/0`
Oracle generation budget/task: `5.50`
Oracle task score: `0.474`
Oracle headroom vs trajectory: `0.060`
Oracle wins/ties/losses vs trajectory: `6/2/0`
Selector regret vs trajectory: `0.060 over 6/8 improvable`
Evolved task delta vs fixed: `0.010`
Evolved task delta vs random: `0.050`
Evolved task delta vs trajectory: `0.007`
Evolved wins/ties/losses vs fixed: `3/4/1`
Evolved wins/ties/losses vs random: `4/4/0`
Evolved wins/ties/losses vs trajectory: `3/4/1`
Oracle headroom vs evolved: `0.052`
Oracle wins/ties/losses vs evolved: `7/1/0`
Selector regret vs evolved: `0.052 over 7/8 improvable`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.056`
Repair task delta vs random: `0.096`
Repair task delta vs trajectory: `0.054`
Repair task delta vs evolved: `0.047`
Repair generation budget delta vs evolved: `1.50`
Repair task delta per extra generation vs evolved: `0.031`
Repair wins/ties/losses vs evolved: `5/1/2`
Oracle headroom vs repair: `0.006`
Oracle wins/ties/losses vs repair: `2/6/0`
Selector regret vs repair: `0.006 over 2/8 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| evolved | 8 | 4.00 | 0.422 | 0.645 | 0.478 |
| repair_selected | 8 | 5.50 | 0.469 | 0.684 | 0.522 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| planning | trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| planning | evolved | 8 | 4.00 | 0.422 | 0.645 | 0.478 |
| planning | repair_selected | 8 | 5.50 | 0.469 | 0.684 | 0.522 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_repair | 12 | 7 | evolved_low_confidence_48,low_confidence_32,random_32 | final | 45.8 | 0.000 | 0.000 | 0.000 | 0.041 | 0.036 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/1/2 | 0.459 | 0.688 | 0.516 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.425 | 0.425 | 0.425 | 0.000 | 0.465 | 0.465 | 0.465 | 0.465 | 0.465 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_inherit_score_repair_pool | evolved_low_confidence_48 | final |  | 0.448 | 0.479 | 0.556 | 0.077 | 0.689 | 0.580 | 0.689 | 0.684 | 0.667 | -0.016 | 0.689 | 0.021 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_inherit_score_repair_pool | low_confidence_32 | final |  | 0.418 | 0.418 | 0.512 | 0.094 | 0.422 | 0.422 | 0.422 | 0.422 | 0.538 | 0.116 | 0.538 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_inherit_score_repair_pool | evolved_low_confidence_48 | final |  | 0.466 | 0.491 | 0.515 | 0.024 | 0.338 | 0.157 | 0.338 | 0.358 | 0.359 | 0.001 | 0.359 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_inherit_score_repair_pool | low_confidence_32 | final |  | 0.334 | 0.334 | 0.408 | 0.075 | 0.421 | 0.421 | 0.421 | 0.421 | 0.459 | 0.037 | 0.459 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_inherit_score_repair_pool | low_confidence_32 | final |  | 0.366 | 0.410 | 0.535 | 0.125 | 0.391 | 0.341 | 0.391 | 0.433 | 0.459 | 0.026 | 0.459 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_inherit_score_repair_pool | low_confidence_32 | final |  | 0.333 | 0.333 | 0.495 | 0.162 | 0.307 | 0.307 | 0.307 | 0.307 | 0.516 | 0.209 | 0.516 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_inherit_score_repair_pool | random_32 | final |  | 0.274 | 0.279 | 0.409 | 0.130 | 0.264 | 0.283 | 0.283 | 0.286 | 0.284 | -0.001 | 0.307 | 0.023 |
