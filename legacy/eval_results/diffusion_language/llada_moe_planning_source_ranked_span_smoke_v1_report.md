# Diffusion Schedule-Selection Benchmark Report

Full model generations: `16`
Arm selections: `10`
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
History mutability: `monotonic 12/16, changes 0, remasks 64, rewrites 17, mask increases 64`
History repairs included: `False`
Repair pack: `constraint_span`
Repair source policy: `non_revision_plus_gap_trajectory`
Adaptive source gate mode: `score_max`
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
Repair selector: `planning_quality_prompt_coverage_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.079`
Trajectory wins/ties/losses vs fixed: `0/2/0`
Trajectory wins/ties/losses vs random: `2/0/0`
Oracle generation budget/task: `8.00`
Oracle task score: `0.574`
Oracle headroom vs trajectory: `0.034`
Oracle wins/ties/losses vs trajectory: `1/1/0`
Selector regret vs trajectory: `0.034 over 1/2 improvable`
Evolved task delta vs fixed: `0.018`
Evolved task delta vs random: `0.098`
Evolved task delta vs trajectory: `0.018`
Evolved wins/ties/losses vs fixed: `1/0/1`
Evolved wins/ties/losses vs random: `2/0/0`
Evolved wins/ties/losses vs trajectory: `1/0/1`
Oracle headroom vs evolved: `0.016`
Oracle wins/ties/losses vs evolved: `2/0/0`
Selector regret vs evolved: `0.016 over 2/2 improvable`
Repair arm coverage: `2/2` overall
Repair eligible coverage: `2/2`
Repair task delta vs fixed: `0.034`
Repair task delta vs random: `0.113`
Repair task delta vs trajectory: `0.034`
Repair task delta vs evolved: `0.016`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.008`
Repair wins/ties/losses vs evolved: `2/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/2/0`
Selector regret vs repair: `0.000 over 0/2 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `2/2` overall, `2/2` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.540000 | 0.000000 | 0.079464 | - | - |
| random perturbation | repair-covered tasks | 0.460536 | -0.079464 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.573750 | 0.033750 | 0.113214 | 1/1/0 | 2/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 2 | 1.00 | 0.540 | 0.659 | 0.570 |
| random | 2 | 1.00 | 0.461 | 0.574 | 0.489 |
| trajectory_selected | 2 | 2.00 | 0.540 | 0.659 | 0.570 |
| evolved | 2 | 6.00 | 0.558 | 0.631 | 0.576 |
| repair_selected | 2 | 8.00 | 0.574 | 0.692 | 0.603 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 2 | 1.00 | 0.540 | 0.659 | 0.570 |
| planning | random | 2 | 1.00 | 0.461 | 0.574 | 0.489 |
| planning | trajectory_selected | 2 | 2.00 | 0.540 | 0.659 | 0.570 |
| planning | evolved | 2 | 6.00 | 0.558 | 0.631 | 0.576 |
| planning | repair_selected | 2 | 8.00 | 0.574 | 0.692 | 0.603 |

## Adaptive Source Gate

| Candidate | Task | Add Source | Reason | Primary | Trajectory | Gap Terms | Traj PQ | Generated | Selected | Gap Term Sample |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| llada-moe-7b-a1b-instruct-hf | plan_002 | True | add | evolved_low_confidence_48 | low_confidence_32 | 12 | 0.559 | 1 | 1 | pipeline,fails,once,every,thousand,noisy,hours,customer |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | add | evolved_low_confidence_48 | low_confidence_32 | 9 | 0.301 | 1 | 1 | customer,shows,wrong,needs,today,deeper,later,plan |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_repair | 4 | 2 | evolved_low_confidence_48,low_confidence_32 | final | 51.0 | 0.000 | 0.000 | 0.000 | 0.034 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/1/1 | 0.566 | 0.690 | 0.597 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_002 | False | evolved_low_confidence_48 | 1.368 | 0.817 | 1.000 | 0.000 | 0.111 | False | Focus on the last 100 records of the failed batch to identify the likely pattern. |
| llada-moe-7b-a1b-instruct-hf | plan_002 | False | evolved_low_confidence_48 | 3.420 | 0.232 | 1.000 | 0.000 | 0.167 | False | Check the data source, transformation logic, and output validation to isolate the root... |
| llada-moe-7b-a1b-instruct-hf | plan_002 | True | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Run a stress test with a controlled dataset of 10,000 records to reproduce the failure,... |
| llada-moe-7b-a1b-instruct-hf | plan_006 | False | evolved_low_confidence_48 | 2.109 | 0.925 | 1.000 | 0.000 | 0.000 | False | Document the issue and schedule a quick meeting with the relevant team. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | False | evolved_low_confidence_48 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Once the fix is live, initiate a root-cause analysis to determine if the migration affe... |
| llada-moe-7b-a1b-instruct-hf | plan_006 | False | evolved_low_confidence_48 | 2.821 | 0.841 | 1.000 | 0.000 | 0.062 | False | Prioritize the immediate fix to minimize customer impact. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 2.109 | 0.925 | 1.000 | 0.000 | 0.000 | False | Document the issue and schedule a quick meeting with the relevant team. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Once the fix is confirmed, initiate a root-cause analysis to review the migration proce... |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 2.893 | 1.000 | 1.000 | 0.000 | 0.062 | False | Ensure the analysis is thorough and includesable to prevent future issues. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.448 | 0.479 | 0.586 | 0.025 | 0.689 | 0.580 | 0.689 | 0.684 | 0.689 | 0.005 | 0.689 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.366 | 0.410 | 0.463 | 0.090 | 0.391 | 0.341 | 0.391 | 0.433 | 0.459 | 0.026 | 0.459 | 0.000 |
