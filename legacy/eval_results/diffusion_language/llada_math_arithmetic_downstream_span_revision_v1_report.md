# Diffusion Schedule-Selection Benchmark Report

Full model generations: `4`
Arm selections: `4`
Exact-task trajectory policy: `proposal_history`
Trajectory selector: `planning_state`
Evolved selector: `inherit`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `False`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `True`
History mutability: `monotonic 4/4, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `prefix`
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
Oracle generation budget/task: `4.00`
Oracle task score: `1.000`
Oracle headroom vs trajectory: `1.000`
Oracle wins/ties/losses vs trajectory: `1/0/0`
Selector regret vs trajectory: `1.000 over 1/1 improvable`
Exact proposal-history sources: `trajectory_selected:fallback=1`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `1.000`
Repair task delta vs random: `1.000`
Repair task delta vs trajectory: `1.000`
Repair task delta vs evolved: `1.000`
Repair generation budget delta vs evolved: `3.00`
Repair task delta per extra generation vs evolved: `0.333`
Repair wins/ties/losses vs evolved: `1/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.000 | 0.015 | 0.004 |
| random | 1 | 1.00 | 0.000 | 0.015 | 0.004 |
| trajectory_selected | 1 | 1.00 | 0.000 | 0.015 | 0.004 |
| repair_selected | 1 | 4.00 | 1.000 | 0.578 | 0.894 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 0.000 | 0.015 | 0.004 |
| math | random | 1 | 1.00 | 0.000 | 0.015 | 0.004 |
| math | trajectory_selected | 1 | 1.00 | 0.000 | 0.015 | 0.004 |
| math | repair_selected | 1 | 4.00 | 1.000 | 0.578 | 0.894 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| arithmetic_contradiction_span_repair | 1 | 0 | final | 45.0 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 3.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.500 | 0.875 |
| arithmetic_feedback_repair | 1 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 3.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.578 | 0.894 |
| self_check_answer_repair | 1 | 0 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 3.0 | 0.000 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.000 | 0.578 | 0.144 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | math_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | arithmetic_feedback_repair | arithmetic_feedback_repair | exact_answer_proposal_history_no_match_kept_fixed |  | exact_answer_arithmetic_feedback | final |  | 0.015 | 0.000 | 0.956 | 0.956 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
