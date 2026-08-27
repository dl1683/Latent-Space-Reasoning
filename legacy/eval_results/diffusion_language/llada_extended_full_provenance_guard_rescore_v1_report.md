# Diffusion Schedule-Selection Benchmark Report

Full model generations: `135`
Arm selections: `127`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
History repairs included: `False`
Repair pack: `state_adaptive`
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
Constraint-gap rescue limit: `1`
Constraint-gap rescue min terms: `6`
Constraint-gap rescue source-quality band: `0.400-0.500`
Constraint-gap rescue source controls: ``
Repair selector: `planning_quality_delta_risk_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `-0.000`
Trajectory task delta vs random: `0.079`
Trajectory wins/ties/losses vs fixed: `0/28/1`
Trajectory wins/ties/losses vs random: `6/20/3`
Oracle generation budget/task: `4.66`
Oracle task score: `0.860`
Oracle headroom vs trajectory: `0.125`
Oracle wins/ties/losses vs trajectory: `9/20/0`
Selector regret vs trajectory: `0.125 over 9/29 improvable`
Evolved task delta vs fixed: `0.011`
Evolved task delta vs random: `0.090`
Evolved task delta vs trajectory: `0.011`
Evolved wins/ties/losses vs fixed: `4/25/0`
Evolved wins/ties/losses vs random: `10/19/0`
Evolved wins/ties/losses vs trajectory: `4/25/0`
Oracle headroom vs evolved: `0.115`
Oracle wins/ties/losses vs evolved: `9/20/0`
Selector regret vs evolved: `0.115 over 9/29 improvable`
Repair arm coverage: `11/29` overall
Repair eligible coverage: `11/11`
Repair task delta vs fixed: `0.330`
Repair task delta vs random: `0.356`
Repair task delta vs trajectory: `0.330`
Repair task delta vs evolved: `0.302`
Repair generation budget delta vs evolved: `1.73`
Repair task delta per extra generation vs evolved: `0.175`
Repair wins/ties/losses vs evolved: `9/2/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/11/0`
Selector regret vs repair: `0.000 over 0/11 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 29 | 1.00 | 0.734 | 0.224 | 0.607 |
| random | 29 | 1.00 | 0.656 | 0.242 | 0.552 |
| trajectory_selected | 29 | 2.00 | 0.734 | 0.224 | 0.607 |
| evolved | 29 | 4.00 | 0.745 | 0.220 | 0.614 |
| repair_selected | 11 | 5.73 | 0.630 | 0.601 | 0.623 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 11 | 1.00 | 0.909 | 0.039 | 0.692 |
| math | random | 11 | 1.00 | 0.818 | 0.048 | 0.626 |
| math | trajectory_selected | 11 | 2.00 | 0.909 | 0.039 | 0.692 |
| math | evolved | 11 | 4.00 | 0.909 | 0.039 | 0.692 |
| math | repair_selected | 1 | 6.00 | 1.000 | 0.578 | 0.894 |
| planning | fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| planning | random | 8 | 1.00 | 0.376 | 0.701 | 0.457 |
| planning | trajectory_selected | 8 | 2.00 | 0.412 | 0.698 | 0.483 |
| planning | evolved | 8 | 4.00 | 0.451 | 0.682 | 0.509 |
| planning | repair_selected | 8 | 5.88 | 0.491 | 0.690 | 0.541 |
| science | fixed | 3 | 1.00 | 1.000 | 0.067 | 0.767 |
| science | random | 3 | 1.00 | 1.000 | 0.099 | 0.775 |
| science | trajectory_selected | 3 | 2.00 | 1.000 | 0.067 | 0.767 |
| science | evolved | 3 | 4.00 | 1.000 | 0.067 | 0.767 |
| symbolic | fixed | 7 | 1.00 | 0.714 | 0.042 | 0.546 |
| symbolic | random | 7 | 1.00 | 0.571 | 0.083 | 0.449 |
| symbolic | trajectory_selected | 7 | 2.00 | 0.714 | 0.042 | 0.546 |
| symbolic | evolved | 7 | 4.00 | 0.714 | 0.042 | 0.546 |
| symbolic | repair_selected | 2 | 5.00 | 1.000 | 0.258 | 0.815 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| arithmetic_feedback_repair | 1 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 3.0 | 0.000 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.578 | 0.894 |
| constraint_gap_revision_repair | 1 | 1 | final | 64.0 | 0.000 | 0.000 | 0.076 | 0.076 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.605 | 0.698 | 0.628 |
| counterfactual_answer_proposal | 1 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 1/0/0 | 1.000 | 0.039 | 0.760 |
| prefix_25_repair | 7 | 1 | final | 48.0 | 0.000 | 0.000 | 0.020 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 5/1/1 | 0.453 | 0.688 | 0.512 |
| self_check_answer_repair | 2 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 1.000 | 0.500 | 2.5 | 0.000 | 0.0 | 0.0 | 1.0 | 1/1/0 | 0.500 | 0.527 | 0.507 |
| state_adaptive_history_repair | 7 | 4 | history | 47.1 | 0.000 | 0.000 | -0.003 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 4/1/2 | 0.431 | 0.655 | 0.487 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.040 | 0.040 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_002 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.040 | 0.040 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.038 | 0.038 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_006 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.040 | 0.040 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.038 | 0.038 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | arithmetic_feedback_repair | arithmetic_feedback_repair | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_arithmetic_feedback | final |  | 0.038 | 0.038 | 0.956 | 0.956 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_011 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | constraint_gap_revision_repair | constraint_gap_revision_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.332 | 0.479 | 0.076 | 0.076 | 0.399 | 0.473 | 0.399 | 0.529 | 0.605 | 0.076 | 0.605 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | final |  | 0.424 | 0.443 | 0.023 | 0.023 | 0.604 | 0.604 | 0.602 | 0.654 | 0.695 | 0.040 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 26 | 0.508 | 0.508 | 0.021 | 0.021 | 0.443 | 0.284 | 0.443 | 0.443 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 39 | 0.289 | 0.366 | 0.029 | 0.029 | 0.283 | 0.283 | 0.283 | 0.347 | 0.375 | 0.029 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.338 | 0.376 | 0.000 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 39 | 0.363 | 0.365 | 0.076 | 0.076 | 0.298 | 0.341 | 0.298 | 0.363 | 0.479 | 0.116 | 0.479 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_spend_gate_kept_evolved_source_quality_or_short |  |  | 0.509 | 0.532 | 0.000 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 20 | 0.410 | 0.410 | 0.080 | 0.080 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.109 | 0.109 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.047 | 0.047 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sci_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.046 | 0.046 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.044 | 0.044 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | counterfactual_answer_proposal | counterfactual_answer_proposal | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_counterfactual_proposal_match | final |  | 0.040 | 0.040 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.052 | 0.052 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_004 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.041 | 0.041 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_005 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.038 | 0.038 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_006 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_007 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | self_check_answer_repair | self_check_answer_repair | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_self_repair_format_change | final |  | 0.039 | 0.039 | 0.955 | 0.955 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
