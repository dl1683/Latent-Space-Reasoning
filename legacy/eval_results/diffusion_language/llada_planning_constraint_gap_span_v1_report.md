# Diffusion Schedule-Selection Benchmark Report

Full model generations: `6`
Arm selections: `4`
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
History mutability: `monotonic 6/6, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory wins/ties/losses vs fixed: `0/1/0`
Trajectory wins/ties/losses vs random: `0/1/0`
Oracle generation budget/task: `6.00`
Oracle task score: `0.399`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/1/0`
Selector regret vs trajectory: `0.000 over 0/1 improvable`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.000`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `5.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/1/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.399 | 0.659 | 0.464 |
| random | 1 | 1.00 | 0.399 | 0.659 | 0.464 |
| trajectory_selected | 1 | 1.00 | 0.399 | 0.659 | 0.464 |
| repair_selected | 1 | 6.00 | 0.399 | 0.659 | 0.464 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 1 | 1.00 | 0.399 | 0.659 | 0.464 |
| planning | random | 1 | 1.00 | 0.399 | 0.659 | 0.464 |
| planning | trajectory_selected | 1 | 1.00 | 0.399 | 0.659 | 0.464 |
| planning | repair_selected | 1 | 6.00 | 0.399 | 0.659 | 0.464 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_revision_anchor25_repair | 1 | 0 | final | 48.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.399 | 0.688 | 0.471 |
| constraint_gap_revision_repair | 1 | 0 | final | 64.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.399 | 0.698 | 0.474 |
| constraint_gap_span_repair | 1 | 0 | final | 16.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.399 | 0.672 | 0.467 |
| prefix_25_repair | 1 | 0 | final | 48.0 | 0.000 | 0.000 | -0.113 | -0.113 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/0/1 | 0.286 | 0.317 | 0.294 |
| state_adaptive_history_repair | 1 | 0 | history | 48.0 | 0.000 | 0.000 | -0.113 | -0.113 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/0/1 | 0.286 | 0.317 | 0.294 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_revision_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  | 0.339 | 0.000 | 0.000 | 0.000 | 0.399 | 0.399 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
