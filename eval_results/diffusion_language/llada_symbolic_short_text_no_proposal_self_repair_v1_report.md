# Diffusion Schedule-Selection Benchmark Report

Full model generations: `5`
Arm selections: `5`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
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
Oracle generation budget/task: `5.00`
Oracle task score: `0.000`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/1/0`
Selector regret vs trajectory: `0.000 over 0/1 improvable`
Evolved task delta vs fixed: `0.000`
Evolved task delta vs random: `0.000`
Evolved task delta vs trajectory: `0.000`
Evolved wins/ties/losses vs fixed: `0/1/0`
Evolved wins/ties/losses vs random: `0/1/0`
Evolved wins/ties/losses vs trajectory: `0/1/0`
Oracle headroom vs evolved: `0.000`
Oracle wins/ties/losses vs evolved: `0/1/0`
Selector regret vs evolved: `0.000 over 0/1 improvable`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.000`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/1/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.000 | 0.041 | 0.010 |
| random | 1 | 1.00 | 0.000 | 0.041 | 0.010 |
| trajectory_selected | 1 | 2.00 | 0.000 | 0.041 | 0.010 |
| evolved | 1 | 4.00 | 0.000 | 0.041 | 0.010 |
| repair_selected | 1 | 5.00 | 0.000 | 0.041 | 0.010 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.041 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.041 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.041 | 0.010 |
| symbolic | evolved | 1 | 4.00 | 0.000 | 0.041 | 0.010 |
| symbolic | repair_selected | 1 | 5.00 | 0.000 | 0.041 | 0.010 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| self_check_answer_repair | 1 | 0 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.000 | 0.654 | 0.164 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | sym_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | self_check_answer_repair | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_repair_kept_source |  |  | 0.041 | 0.041 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
