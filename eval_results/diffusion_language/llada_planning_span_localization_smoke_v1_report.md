# Diffusion Schedule-Selection Benchmark Report

Full model generations: `2`
Arm selections: `4`
Run ID: `diffusion-d67e0b34ba99f9af`
Content hash: `d67e0b34ba99f9af9e83a54ea9be9bf688dba7956a3baa4f92ca4bd667e35f79`
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
History mutability: `monotonic 2/2, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span`
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
Trajectory wins/ties/losses vs fixed: `0/1/0`
Trajectory wins/ties/losses vs random: `0/1/0`
Oracle generation budget/task: `2.00`
Oracle task score: `0.465`
Oracle headroom vs trajectory: `0.066`
Oracle wins/ties/losses vs trajectory: `1/0/0`
Selector regret vs trajectory: `0.066 over 1/1 improvable`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `0.066`
Repair task delta vs random: `0.066`
Repair task delta vs trajectory: `0.066`
Repair task delta vs evolved: `0.066`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.066`
Repair wins/ties/losses vs evolved: `1/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `1/1` overall, `1/1` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.398929 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.398929 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.465357 | 0.066429 | 0.066429 | 1/0/0 | 1/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.399 | 0.698 | 0.474 |
| random | 1 | 1.00 | 0.399 | 0.698 | 0.474 |
| trajectory_selected | 1 | 1.00 | 0.399 | 0.698 | 0.474 |
| repair_selected | 1 | 2.00 | 0.465 | 0.688 | 0.521 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 1 | 1.00 | 0.399 | 0.698 | 0.474 |
| planning | random | 1 | 1.00 | 0.399 | 0.698 | 0.474 |
| planning | trajectory_selected | 1 | 1.00 | 0.399 | 0.698 | 0.474 |
| planning | repair_selected | 1 | 2.00 | 0.465 | 0.688 | 0.521 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_repair | 1 | 1 | low_confidence_32 | final | 56.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.046 | 0.066 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.465 | 0.688 | 0.521 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_001 | True | low_confidence_32 | 6.161 | 0.905 | 1.000 | 0.160 | 0.200 | False | If the baseline job fails, you can still run the intervention job, but the baseline dat... |
| llada-8b-instruct-hf | plan_001 | True | low_confidence_32 | 6.685 | 0.800 | 1.000 | 0.000 | 0.267 | False | If the baseline job succeeds, you can then run the intervention job, ensuring you have... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | final |  | 0.332 | 0.000 | 0.206 | 0.206 | 0.399 | 0.399 | 0.399 | 0.000 | 0.465 | 0.066 | 0.465 | 0.000 |
