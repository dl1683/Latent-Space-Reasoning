# Diffusion Schedule-Selection Benchmark Report

Full model generations: `53`
Arm selections: `53`
Run ID: `diffusion-34e1ccb29a8754bc`
Content hash: `34e1ccb29a8754bcf41c6fccda06432c3bb4d00ad451b9429b219edc41feec2b`
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
History mutability: `monotonic 53/53, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `-0.000`
Trajectory task delta vs random: `0.026`
Trajectory wins/ties/losses vs fixed: `0/10/1`
Trajectory wins/ties/losses vs random: `4/4/3`
Oracle generation budget/task: `4.82`
Oracle task score: `0.607`
Oracle headroom vs trajectory: `0.126`
Oracle wins/ties/losses vs trajectory: `6/5/0`
Selector regret vs trajectory: `0.126 over 6/11 improvable`
Exact proposal-history sources: `evolved:fallback=1, evolved:final=2, trajectory_selected:fallback=1, trajectory_selected:final=2`
Evolved task delta vs fixed: `0.028`
Evolved task delta vs random: `0.054`
Evolved task delta vs trajectory: `0.028`
Evolved wins/ties/losses vs fixed: `4/7/0`
Evolved wins/ties/losses vs random: `8/3/0`
Evolved wins/ties/losses vs trajectory: `4/7/0`
Oracle headroom vs evolved: `0.098`
Oracle wins/ties/losses vs evolved: `3/8/0`
Selector regret vs evolved: `0.098 over 3/11 improvable`
Repair arm coverage: `9/11` overall
Repair eligible coverage: `9/9`
Repair task delta vs fixed: `0.150`
Repair task delta vs random: `0.182`
Repair task delta vs trajectory: `0.150`
Repair task delta vs evolved: `0.116`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.116`
Repair wins/ties/losses vs evolved: `2/7/0`
Oracle headroom vs repair: `0.004`
Oracle wins/ties/losses vs repair: `1/8/0`
Selector regret vs repair: `0.004 over 1/9 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `9/11` overall, `9/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.366468 | 0.000000 | 0.031984 | - | - |
| random perturbation | repair-covered tasks | 0.334484 | -0.031984 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.516468 | 0.150000 | 0.181984 | 5/4/0 | 9/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.482 | 0.525 | 0.492 |
| random | 11 | 1.00 | 0.455 | 0.535 | 0.475 |
| trajectory_selected | 11 | 2.00 | 0.481 | 0.529 | 0.493 |
| evolved | 11 | 4.00 | 0.510 | 0.526 | 0.514 |
| repair_selected | 9 | 5.00 | 0.516 | 0.610 | 0.540 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.040 | 0.760 |
| math | random | 1 | 1.00 | 1.000 | 0.040 | 0.760 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.043 | 0.761 |
| math | evolved | 1 | 4.00 | 1.000 | 0.065 | 0.766 |
| planning | fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| planning | random | 8 | 1.00 | 0.376 | 0.701 | 0.457 |
| planning | trajectory_selected | 8 | 2.00 | 0.412 | 0.698 | 0.483 |
| planning | evolved | 8 | 4.00 | 0.451 | 0.682 | 0.509 |
| planning | repair_selected | 8 | 5.00 | 0.456 | 0.681 | 0.512 |
| science | fixed | 1 | 1.00 | 1.000 | 0.109 | 0.777 |
| science | random | 1 | 1.00 | 1.000 | 0.204 | 0.801 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.109 | 0.777 |
| science | evolved | 1 | 4.00 | 1.000 | 0.109 | 0.777 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.040 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.040 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.089 | 0.022 |
| symbolic | evolved | 1 | 4.00 | 0.000 | 0.165 | 0.041 |
| symbolic | repair_selected | 1 | 5.00 | 1.000 | 0.039 | 0.760 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_repair | 8 | 1 | evolved_low_confidence_48,evolved_random_48,low_confidence_32 | final | 33.0 | 0.000 | 0.015 | 0.015 | -0.012 | -0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/4/2 | 0.440 | 0.667 | 0.497 |
| counterfactual_answer_proposal | 1 | 1 | evolved_random_48 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.039 | 0.760 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_001 | False | evolved_random_48 | 2.947 | 0.279 | 1.000 | 0.000 | 0.267 | False | The baseline provides a fallback if the intervention fails, ensuring that the overall r... |
| llada-8b-instruct-hf | plan_002 | False | evolved_low_confidence_48 | 3.605 | 0.745 | 1.000 | 0.000 | 0.056 | False | Analyze the logs for patterns or common issues. |
| llada-8b-instruct-hf | plan_002 | False | evolved_low_confidence_48 | 3.657 | 0.820 | 1.000 | 0.000 | 0.000 | False | Prioritize the issues based on potential impact. |
| llada-8b-instruct-hf | plan_002 | False | evolved_low_confidence_48 | 3.436 | 0.224 | 1.000 | 0.000 | 0.111 | False | Develop a plan to isolate and fix the issue before the demo. |
| llada-8b-instruct-hf | plan_003 | False | low_confidence_32 | 2.110 | 0.148 | 1.000 | 0.000 | 0.467 | False | Decision rule: If the improvement in offline accuracy is significant (e.g., above a cer... |
| llada-8b-instruct-hf | plan_003 | False | low_confidence_32 | 2.984 | 0.308 | 1.000 | 0.000 | 0.200 | False | Otherwise, rollback or gate the release. |
| llada-8b-instruct-hf | plan_004 | False | evolved_random_48 | 2.021 | 0.758 | 1.000 | 0.000 | 0.000 | False | Measure the model's results. |
| llada-8b-instruct-hf | plan_004 | False | evolved_random_48 | 2.008 | 0.758 | 1.000 | 0.000 | 0.059 | False | Compare the results to the original baseline. |
| llada-8b-instruct-hf | plan_004 | False | evolved_random_48 | 2.052 | 0.657 | 1.000 | 0.000 | 0.118 | False | Determine if the results are still impressive, or if the baseline has changed. |
| llada-8b-instruct-hf | plan_005 | False | evolved_low_confidence_48 | 1.303 | 0.699 | 1.000 | 0.000 | 0.176 | False | If a checkpoint fails, resume from the last successful checkpoint and complete the trai... |
| llada-8b-instruct-hf | plan_005 | False | evolved_low_confidence_48 | 2.441 | 0.500 | 1.000 | 0.000 | 0.353 | False | This ensures reproducibility by allowing for exact recovery from the last successful ch... |
| llada-8b-instruct-hf | plan_006 | True | evolved_random_48 | 2.064 | 0.825 | 1.000 | 0.000 | 0.062 | False | Apply a temporary fix to the dashboard. |
| llada-8b-instruct-hf | plan_006 | True | evolved_random_48 | 1.967 | 0.631 | 1.000 | 0.000 | 0.062 | False | Verify the fix and reach out to the customer team to ensure the issue is resolved. |
| llada-8b-instruct-hf | plan_006 | True | evolved_random_48 | 1.724 | 0.000 | 1.000 | 0.000 | 0.188 | False | Schedule a later meeting with the team to investigate the root cause. |
| llada-8b-instruct-hf | plan_007 | False | evolved_low_confidence_48 | 2.146 | 0.963 | 1.000 | 0.000 | 0.333 | False | By comparing the performance before and after the change, you can determine if the opti... |
| llada-8b-instruct-hf | plan_008 | False | low_confidence_32 | 2.109 | 0.913 | 1.000 | 0.000 | 0.625 | False | If the outputs become more and less evasive evasive as the system improves, it may indi... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 |  | evolved_low_confidence_48 | exact_answer_proposal_final_match | exact_answer_proposal_final_match |  |  |  |  | 0.043 | 0.065 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.332 | 0.479 | 0.000 | 0.000 | 0.399 | 0.473 | 0.399 | 0.529 | 0.529 | 0.000 | 0.529 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.424 | 0.443 | 0.000 | 0.000 | 0.604 | 0.604 | 0.602 | 0.654 | 0.654 | 0.000 | 0.654 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.508 | 0.508 | 0.000 | 0.000 | 0.443 | 0.284 | 0.443 | 0.443 | 0.443 | 0.000 | 0.443 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.289 | 0.366 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.347 | 0.347 | 0.000 | 0.347 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.338 | 0.376 | 0.000 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | evolved_random_48 | final |  | 0.363 | 0.365 | 0.021 | 0.021 | 0.298 | 0.341 | 0.298 | 0.363 | 0.404 | 0.041 | 0.404 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.509 | 0.532 | 0.000 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.643 | 0.032 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.410 | 0.410 | 0.000 | 0.000 | 0.283 | 0.264 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | random_32 | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed |  |  |  |  | 0.109 | 0.109 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | counterfactual_answer_proposal | counterfactual_answer_proposal | exact_answer_proposal_final_match | exact_answer_proposal_final_match | exact_answer_counterfactual_proposal_match | evolved_random_48 | final |  | 0.089 | 0.165 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
