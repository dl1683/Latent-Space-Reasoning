# Diffusion Schedule-Selection Benchmark Report

Full model generations: `21`
Arm selections: `18`
Exact-task trajectory policy: `proposal_history`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `False`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `True`
History mutability: `monotonic 21/21, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `prefix`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
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
Oracle generation budget/task: `5.25`
Oracle task score: `1.000`
Oracle headroom vs trajectory: `0.500`
Oracle wins/ties/losses vs trajectory: `2/2/0`
Selector regret vs trajectory: `0.500 over 2/4 improvable`
Exact proposal-history sources: `evolved:fallback=4, trajectory_selected:fallback=4`
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
Repair generation budget delta vs evolved: `2.50`
Repair task delta per extra generation vs evolved: `0.400`
Repair wins/ties/losses vs evolved: `2/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/2/0`
Selector regret vs repair: `0.000 over 0/2 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `2/4` overall, `2/2` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.000000 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.000000 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 1.000000 | 1.000000 | 1.000000 | 2/0/0 | 2/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 4 | 1.00 | 0.500 | 0.015 | 0.379 |
| random | 4 | 1.00 | 0.500 | 0.043 | 0.386 |
| trajectory_selected | 4 | 2.00 | 0.500 | 0.015 | 0.379 |
| evolved | 4 | 4.00 | 0.500 | 0.015 | 0.379 |
| repair_selected | 2 | 6.50 | 1.000 | 0.488 | 0.872 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 3 | 1.00 | 0.667 | 0.015 | 0.504 |
| math | random | 3 | 1.00 | 0.667 | 0.015 | 0.504 |
| math | trajectory_selected | 3 | 2.00 | 0.667 | 0.015 | 0.504 |
| math | evolved | 3 | 4.00 | 0.667 | 0.015 | 0.504 |
| math | repair_selected | 1 | 7.00 | 1.000 | 0.500 | 0.875 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.125 | 0.031 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |
| symbolic | evolved | 1 | 4.00 | 0.000 | 0.016 | 0.004 |
| symbolic | repair_selected | 1 | 6.00 | 1.000 | 0.477 | 0.869 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| answer_span_repair | 2 | 0 | low_confidence_32 | final | 1.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 | 0.000 | 1.0 | 1.5 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0/2/0 | 0.000 | 0.273 | 0.068 |
| arithmetic_contradiction_span_repair | 1 | 1 | low_confidence_32 | final | 45.0 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 3.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.500 | 0.875 |
| self_check_answer_repair | 2 | 1 | low_confidence_32 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 1.000 | 0.500 | 2.5 | 0.000 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/1/0 | 0.500 | 0.527 | 0.507 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed |  |  |  |  | 0.015 | 0.015 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | arithmetic_contradiction_span_repair | arithmetic_contradiction_span_repair | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_arithmetic_span_revision | low_confidence_32 | final |  | 0.015 | 0.015 | 0.955 | 0.955 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_011 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed |  |  |  |  | 0.016 | 0.016 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_007 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | self_check_answer_repair | self_check_answer_repair | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_self_repair_format_change | low_confidence_32 | final |  | 0.016 | 0.016 | 0.955 | 0.955 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
