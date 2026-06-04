# Diffusion Schedule-Selection Benchmark Report

Full model generations: `69`
Arm selections: `69`
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
Trajectory task delta vs random: `0.118`
Trajectory wins/ties/losses vs fixed: `0/17/0`
Trajectory wins/ties/losses vs random: `2/15/0`
Oracle generation budget/task: `4.06`
Oracle task score: `1.000`
Oracle headroom vs trajectory: `0.059`
Oracle wins/ties/losses vs trajectory: `1/16/0`
Selector regret vs trajectory: `0.059 over 1/17 improvable`
Evolved task delta vs fixed: `0.000`
Evolved task delta vs random: `0.118`
Evolved task delta vs trajectory: `0.000`
Evolved wins/ties/losses vs fixed: `0/17/0`
Evolved wins/ties/losses vs random: `2/15/0`
Evolved wins/ties/losses vs trajectory: `0/17/0`
Oracle headroom vs evolved: `0.059`
Oracle wins/ties/losses vs evolved: `1/16/0`
Selector regret vs evolved: `0.059 over 1/17 improvable`
Repair arm coverage: `1/17` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `1.000`
Repair task delta vs random: `1.000`
Repair task delta vs trajectory: `1.000`
Repair task delta vs evolved: `1.000`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `1.000`
Repair wins/ties/losses vs evolved: `1/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 17 | 1.00 | 0.941 | 0.045 | 0.717 |
| random | 17 | 1.00 | 0.824 | 0.068 | 0.635 |
| trajectory_selected | 17 | 2.00 | 0.941 | 0.045 | 0.717 |
| evolved | 17 | 4.00 | 0.941 | 0.045 | 0.717 |
| repair_selected | 1 | 5.00 | 1.000 | 0.039 | 0.760 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 8 | 1.00 | 1.000 | 0.039 | 0.760 |
| math | random | 8 | 1.00 | 0.875 | 0.052 | 0.669 |
| math | trajectory_selected | 8 | 2.00 | 1.000 | 0.039 | 0.760 |
| math | evolved | 8 | 4.00 | 1.000 | 0.039 | 0.760 |
| science | fixed | 3 | 1.00 | 1.000 | 0.067 | 0.767 |
| science | random | 3 | 1.00 | 1.000 | 0.099 | 0.775 |
| science | trajectory_selected | 3 | 2.00 | 1.000 | 0.067 | 0.767 |
| science | evolved | 3 | 4.00 | 1.000 | 0.067 | 0.767 |
| symbolic | fixed | 6 | 1.00 | 0.833 | 0.042 | 0.636 |
| symbolic | random | 6 | 1.00 | 0.667 | 0.075 | 0.519 |
| symbolic | trajectory_selected | 6 | 2.00 | 0.833 | 0.042 | 0.636 |
| symbolic | evolved | 6 | 4.00 | 0.833 | 0.042 | 0.636 |
| symbolic | repair_selected | 1 | 5.00 | 1.000 | 0.039 | 0.760 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| counterfactual_answer_proposal | 1 | 1 | final | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1/0/0 | 1.000 | 0.039 | 0.760 |

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
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.109 | 0.109 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.047 | 0.047 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sci_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.046 | 0.046 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.044 | 0.044 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | counterfactual_answer_proposal | counterfactual_answer_proposal | fixed_exact_answer_guard | fixed_exact_answer_guard | exact_answer_counterfactual_proposal_match | final |  | 0.040 | 0.040 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.052 | 0.052 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_004 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.041 | 0.041 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_005 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.038 | 0.038 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_006 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  |  |  | 0.039 | 0.039 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
