# Diffusion Schedule-Selection Benchmark Report

Full model generations: `19`
Arm selections: `18`
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
Trajectory wins/ties/losses vs fixed: `0/4/0`
Trajectory wins/ties/losses vs random: `0/4/0`
Oracle generation budget/task: `4.75`
Oracle task score: `1.000`
Oracle headroom vs trajectory: `0.500`
Oracle wins/ties/losses vs trajectory: `2/2/0`
Selector regret vs trajectory: `0.500 over 2/4 improvable`
Evolved task delta vs fixed: `0.000`
Evolved task delta vs random: `0.000`
Evolved task delta vs trajectory: `0.000`
Evolved wins/ties/losses vs fixed: `0/4/0`
Evolved wins/ties/losses vs random: `0/4/0`
Evolved wins/ties/losses vs trajectory: `0/4/0`
Oracle headroom vs evolved: `0.500`
Oracle wins/ties/losses vs evolved: `2/2/0`
Selector regret vs evolved: `0.500 over 2/4 improvable`
Repair arm coverage: `2/4` overall
Repair eligible coverage: `2/2`
Repair task delta vs fixed: `1.000`
Repair task delta vs random: `1.000`
Repair task delta vs trajectory: `1.000`
Repair task delta vs evolved: `1.000`
Repair generation budget delta vs evolved: `1.50`
Repair task delta per extra generation vs evolved: `0.667`
Repair wins/ties/losses vs evolved: `2/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/2/0`
Selector regret vs repair: `0.000 over 0/2 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 4 | 1.00 | 0.500 | 0.304 | 0.451 |
| random | 4 | 1.00 | 0.500 | 0.086 | 0.397 |
| trajectory_selected | 4 | 2.00 | 0.500 | 0.304 | 0.451 |
| evolved | 4 | 4.00 | 0.500 | 0.304 | 0.451 |
| repair_selected | 2 | 5.50 | 1.000 | 0.540 | 0.885 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 4 | 1.00 | 0.500 | 0.304 | 0.451 |
| math | random | 4 | 1.00 | 0.500 | 0.086 | 0.397 |
| math | trajectory_selected | 4 | 2.00 | 0.500 | 0.304 | 0.451 |
| math | evolved | 4 | 4.00 | 0.500 | 0.304 | 0.451 |
| math | repair_selected | 2 | 5.50 | 1.000 | 0.540 | 0.885 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| arithmetic_feedback_repair | 1 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 3.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.572 | 0.893 |
| self_check_answer_repair | 2 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 1.000 | 0.500 | 3.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 1/1/0 | 0.500 | 0.540 | 0.510 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | math_012 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.698 | 0.698 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_013 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | arithmetic_feedback_repair | arithmetic_feedback_repair | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_arithmetic_feedback | final |  | 0.040 | 0.040 | 0.956 | 0.956 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_014 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | self_check_answer_repair | self_check_answer_repair | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_self_repair_format_change | final |  | 0.440 | 0.440 | 0.955 | 0.955 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_015 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.038 | 0.038 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
