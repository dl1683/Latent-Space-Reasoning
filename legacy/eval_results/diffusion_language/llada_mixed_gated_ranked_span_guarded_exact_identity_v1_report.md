# Diffusion Schedule-Selection Benchmark Report

Full model generations: `63`
Arm selections: `53`
Run ID: `diffusion-45da934106d48a5b`
Content hash: `45da934106d48a5bcb4514b89fc53a684f8f6012c8dac2d078d5e3df94adafce`
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
History mutability: `monotonic 63/63, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `state_adaptive`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
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
Trajectory task delta vs random: `0.026`
Trajectory wins/ties/losses vs fixed: `0/10/1`
Trajectory wins/ties/losses vs random: `4/4/3`
Oracle generation budget/task: `5.73`
Oracle task score: `0.631`
Oracle headroom vs trajectory: `0.149`
Oracle wins/ties/losses vs trajectory: `7/4/0`
Selector regret vs trajectory: `0.149 over 7/11 improvable`
Exact proposal-history sources: `evolved:fallback=1, evolved:final=2, trajectory_selected:fallback=1, trajectory_selected:final=2`
Evolved task delta vs fixed: `0.028`
Evolved task delta vs random: `0.054`
Evolved task delta vs trajectory: `0.028`
Evolved wins/ties/losses vs fixed: `4/7/0`
Evolved wins/ties/losses vs random: `8/3/0`
Evolved wins/ties/losses vs trajectory: `4/7/0`
Oracle headroom vs evolved: `0.121`
Oracle wins/ties/losses vs evolved: `7/4/0`
Selector regret vs evolved: `0.121 over 7/11 improvable`
Repair arm coverage: `9/11` overall
Repair eligible coverage: `9/9`
Repair task delta vs fixed: `0.181`
Repair task delta vs random: `0.213`
Repair task delta vs trajectory: `0.182`
Repair task delta vs evolved: `0.147`
Repair generation budget delta vs evolved: `2.11`
Repair task delta per extra generation vs evolved: `0.070`
Repair wins/ties/losses vs evolved: `7/2/0`
Oracle headroom vs repair: `0.001`
Oracle wins/ties/losses vs repair: `1/8/0`
Selector regret vs repair: `0.001 over 1/9 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `9/11` overall, `9/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.366468 | 0.000000 | 0.031984 | - | - |
| random perturbation | repair-covered tasks | 0.334484 | -0.031984 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.547778 | 0.181310 | 0.213294 | 7/2/0 | 9/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.482 | 0.525 | 0.492 |
| random | 11 | 1.00 | 0.455 | 0.535 | 0.475 |
| trajectory_selected | 11 | 2.00 | 0.481 | 0.529 | 0.493 |
| evolved | 11 | 4.00 | 0.510 | 0.526 | 0.514 |
| repair_selected | 9 | 6.11 | 0.548 | 0.618 | 0.565 |

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
| planning | repair_selected | 8 | 6.12 | 0.491 | 0.690 | 0.541 |
| science | fixed | 1 | 1.00 | 1.000 | 0.109 | 0.777 |
| science | random | 1 | 1.00 | 1.000 | 0.204 | 0.801 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.109 | 0.777 |
| science | evolved | 1 | 4.00 | 1.000 | 0.109 | 0.777 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.040 | 0.010 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.040 | 0.010 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.089 | 0.022 |
| symbolic | evolved | 1 | 4.00 | 0.000 | 0.165 | 0.041 |
| symbolic | repair_selected | 1 | 6.00 | 1.000 | 0.039 | 0.760 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| answer_span_repair | 1 | 0 | evolved_random_48 | final | 1.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | -1.000 | 0.000 | 1.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0/1/0 | 0.000 | 0.303 | 0.076 |
| constraint_gap_revision_anchor25_repair | 1 | 0 | evolved_random_48 | final | 48.0 | 0.000 | 0.180 | 0.000 | 0.085 | 0.085 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.614 | 0.688 | 0.633 |
| constraint_gap_revision_repair | 1 | 1 | evolved_random_48 | final | 64.0 | 0.000 | 0.000 | 0.000 | 0.076 | 0.076 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.605 | 0.698 | 0.628 |
| constraint_gap_span_repair | 1 | 0 | evolved_random_48 | final | 25.0 | 0.000 | 0.000 | 0.000 | -0.038 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/1/0 | 0.529 | 0.616 | 0.551 |
| counterfactual_answer_proposal | 1 | 1 | evolved_random_48 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.039 | 0.760 |
| prefix_25_repair | 7 | 1 | evolved_low_confidence_48,evolved_random_48,low_confidence_32 | final | 48.0 | 0.000 | 0.000 | 0.000 | 0.020 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/1/1 | 0.453 | 0.688 | 0.512 |
| state_adaptive_history_repair | 7 | 4 | evolved_low_confidence_48,evolved_random_48,low_confidence_32 | history | 47.1 | 0.000 | 0.000 | 0.000 | -0.003 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/1/2 | 0.431 | 0.655 | 0.487 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_001 | False | evolved_random_48 | 2.947 | 0.279 | 1.000 | 0.000 | 0.267 | False | The baseline provides a fallback if the intervention fails, ensuring that the overall r... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 |  | evolved_low_confidence_48 | exact_answer_proposal_final_match | exact_answer_proposal_final_match |  |  |  |  | 0.043 | 0.065 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | constraint_gap_revision_repair | constraint_gap_revision_anchor25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | evolved_random_48 | final |  | 0.332 | 0.479 | 0.076 | 0.076 | 0.399 | 0.473 | 0.399 | 0.529 | 0.605 | 0.076 | 0.614 | 0.009 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | evolved_low_confidence_48 | final |  | 0.424 | 0.443 | 0.023 | 0.023 | 0.604 | 0.604 | 0.602 | 0.654 | 0.695 | 0.040 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | history | 26 | 0.508 | 0.508 | 0.021 | 0.021 | 0.443 | 0.284 | 0.443 | 0.443 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | evolved_random_48 | history | 39 | 0.289 | 0.366 | 0.029 | 0.029 | 0.283 | 0.283 | 0.283 | 0.347 | 0.375 | 0.029 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.338 | 0.376 | 0.000 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | evolved_random_48 | history | 39 | 0.363 | 0.365 | 0.076 | 0.076 | 0.298 | 0.341 | 0.298 | 0.363 | 0.479 | 0.116 | 0.479 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_spend_gate_kept_evolved_source_quality_or_short |  |  |  | 0.509 | 0.532 | 0.000 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | history | 20 | 0.410 | 0.410 | 0.080 | 0.080 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | random_32 | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed |  |  |  |  | 0.109 | 0.109 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | counterfactual_answer_proposal | counterfactual_answer_proposal | exact_answer_proposal_final_match | exact_answer_proposal_final_match | exact_answer_counterfactual_proposal_match | evolved_random_48 | final |  | 0.089 | 0.165 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
