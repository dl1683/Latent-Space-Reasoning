# Diffusion Schedule-Selection Benchmark Report

Full model generations: `48`
Arm selections: `32`
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
History mutability: `monotonic 48/48, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_gap`
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
Trajectory wins/ties/losses vs fixed: `0/8/0`
Trajectory wins/ties/losses vs random: `0/8/0`
Oracle generation budget/task: `6.00`
Oracle task score: `0.467`
Oracle headroom vs trajectory: `0.055`
Oracle wins/ties/losses vs trajectory: `7/1/0`
Selector regret vs trajectory: `0.055 over 7/8 improvable`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.053`
Repair task delta vs random: `0.053`
Repair task delta vs trajectory: `0.053`
Repair task delta vs evolved: `0.053`
Repair generation budget delta vs evolved: `5.00`
Repair task delta per extra generation vs evolved: `0.011`
Repair wins/ties/losses vs evolved: `6/2/0`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.002 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/8` overall, `8/8` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.412277 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.412277 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.465313 | 0.053036 | 0.053036 | 6/2/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| random | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| trajectory_selected | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| repair_selected | 8 | 6.00 | 0.465 | 0.681 | 0.519 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | trajectory_selected | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | repair_selected | 8 | 6.00 | 0.465 | 0.681 | 0.519 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_revision_anchor25_repair | 8 | 0 | low_confidence_32 | final | 48.0 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/8/0 | 0.412 | 0.693 | 0.483 |
| constraint_gap_revision_repair | 8 | 0 | low_confidence_32 | final | 64.0 | 0.000 | 0.020 | 0.000 | -0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/7/0 | 0.413 | 0.698 | 0.484 |
| constraint_gap_span_repair | 8 | 3 | low_confidence_32 | final | 37.1 | 0.000 | 0.000 | 0.000 | 0.010 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/2/2 | 0.430 | 0.672 | 0.490 |
| prefix_25_repair | 8 | 1 | low_confidence_32 | final | 48.0 | 0.000 | 0.000 | 0.000 | -0.009 | -0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/3/2 | 0.401 | 0.635 | 0.459 |
| state_adaptive_history_repair | 8 | 2 | low_confidence_32 | history | 44.1 | 0.000 | 0.000 | 0.000 | -0.018 | -0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/4/2 | 0.390 | 0.635 | 0.451 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_001 | True | low_confidence_32 | 6.161 | 0.905 | 1.000 | 0.160 | 0.200 | False | If the baseline job fails, you can still run the intervention job, but the baseline dat... |
| llada-8b-instruct-hf | plan_001 | True | low_confidence_32 | 6.685 | 0.800 | 1.000 | 0.000 | 0.267 | False | If the baseline job succeeds, you can then run the intervention job, ensuring you have... |
| llada-8b-instruct-hf | plan_002 | False | low_confidence_32 | 3.625 | 0.790 | 1.000 | 0.000 | 0.056 | False | Analyze the logs for patterns or common issues. |
| llada-8b-instruct-hf | plan_002 | False | low_confidence_32 | 3.664 | 0.865 | 1.000 | 0.000 | 0.056 | False | Narrow down potential causes. |
| llada-8b-instruct-hf | plan_002 | False | low_confidence_32 | 3.456 | 0.269 | 1.000 | 0.000 | 0.111 | False | Develop a plan to isolate and fix the issue before the demo. |
| llada-8b-instruct-hf | plan_003 | False | low_confidence_32 | 2.110 | 0.148 | 1.000 | 0.000 | 0.467 | False | Decision rule: If the improvement in offline accuracy is significant (e.g., above a cer... |
| llada-8b-instruct-hf | plan_003 | False | low_confidence_32 | 2.984 | 0.308 | 1.000 | 0.000 | 0.200 | False | Otherwise, rollback or gate the release. |
| llada-8b-instruct-hf | plan_004 | False | low_confidence_32 | 1.357 | 0.752 | 1.000 | 0.000 | 0.118 | False | Increase the number of tokens used in the experiment. |
| llada-8b-instruct-hf | plan_004 | False | low_confidence_32 | 1.392 | 0.865 | 1.000 | 0.000 | 0.176 | False | Change the prompt format to match the baseline. |
| llada-8b-instruct-hf | plan_004 | False | low_confidence_32 | 2.747 | 0.758 | 1.000 | 0.000 | 0.059 | False | Compare the results to the original baseline to ensure the improvement is genuine. |
| llada-8b-instruct-hf | plan_005 | True | low_confidence_32 | 2.082 | 0.925 | 1.000 | 0.000 | 0.059 | False | If a checkpoint fails, resume from the last successful checkpoint. |
| llada-8b-instruct-hf | plan_005 | True | low_confidence_32 | 2.001 | 0.613 | 1.000 | 0.000 | 0.294 | False | This, combined with a rolling checkpointing that skards over failed checkpoints, ensure... |
| llada-8b-instruct-hf | plan_006 | True | low_confidence_32 | 1.348 | 0.701 | 1.000 | 0.000 | 0.188 | False | Plan the fix: Update the dashboard to reflect the correct totals. |
| llada-8b-instruct-hf | plan_006 | True | low_confidence_32 | 1.371 | 0.745 | 1.000 | 0.000 | 0.125 | False | Confirm the fix: Verify the updated totals with the customer. |
| llada-8b-instruct-hf | plan_006 | True | low_confidence_32 | 2.763 | 0.713 | 1.000 | 0.000 | 0.000 | False | Document the fix: Log the issue and solution for future reference. |
| llada-8b-instruct-hf | plan_007 | False | low_confidence_32 | 2.157 | 0.963 | 1.000 | 0.000 | 0.250 | False | By comparing the performance before and after the optimizer change, you can determine i... |
| llada-8b-instruct-hf | plan_008 | False | low_confidence_32 | 2.109 | 0.913 | 1.000 | 0.000 | 0.625 | False | If the outputs become more and less evasive evasive as the system improves, it may indi... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | final |  | 0.339 | 0.000 | 0.206 | 0.206 | 0.399 | 0.399 | 0.399 | 0.000 | 0.465 | 0.066 | 0.465 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | final |  | 0.422 | 0.000 | 0.073 | 0.073 | 0.604 | 0.604 | 0.604 | 0.000 | 0.695 | 0.090 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | history | 31 | 0.500 | 0.000 | 0.021 | 0.021 | 0.443 | 0.443 | 0.443 | 0.000 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_revision_anchor25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.295 | 0.000 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | final |  | 0.338 | 0.000 | 0.037 | 0.037 | 0.378 | 0.378 | 0.378 | 0.000 | 0.435 | 0.057 | 0.435 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | final |  | 0.366 | 0.000 | 0.079 | 0.079 | 0.298 | 0.298 | 0.298 | 0.000 | 0.446 | 0.149 | 0.446 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.521 | 0.000 | 0.000 | 0.000 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 | 0.623 | 0.012 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool |  | max_planning_quality_delta_risk_guarded_score_repair_pool | low_confidence_32 | history | 22 | 0.405 | 0.000 | 0.080 | 0.080 | 0.283 | 0.283 | 0.283 | 0.000 | 0.323 | 0.040 | 0.323 | 0.000 |
