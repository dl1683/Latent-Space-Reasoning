# Diffusion Schedule-Selection Benchmark Report

Full model generations: `15`
Arm selections: `15`
Exact-task trajectory policy: `proposal_history`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision schedules included: `True`
Revision remask fraction: `0.500`
Revision steps: `24`
History mutability: `monotonic 9/15, changes 0, remasks 96, rewrites 8, mask increases 96`
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
Trajectory wins/ties/losses vs fixed: `0/3/0`
Trajectory wins/ties/losses vs random: `0/3/0`
Oracle generation budget/task: `5.00`
Oracle task score: `1.000`
Oracle headroom vs trajectory: `1.000`
Oracle wins/ties/losses vs trajectory: `3/0/0`
Selector regret vs trajectory: `1.000 over 3/3 improvable`
Exact proposal-history sources: `evolved:fallback=3, trajectory_selected:fallback=3`
Evolved task delta vs fixed: `0.000`
Evolved task delta vs random: `0.000`
Evolved task delta vs trajectory: `0.000`
Evolved wins/ties/losses vs fixed: `0/3/0`
Evolved wins/ties/losses vs random: `0/3/0`
Evolved wins/ties/losses vs trajectory: `0/3/0`
Oracle headroom vs evolved: `1.000`
Oracle wins/ties/losses vs evolved: `3/0/0`
Selector regret vs evolved: `1.000 over 3/3 improvable`
Repair arm coverage: `3/3` overall
Repair eligible coverage: `3/3`
Repair task delta vs fixed: `1.000`
Repair task delta vs random: `1.000`
Repair task delta vs trajectory: `1.000`
Repair task delta vs evolved: `1.000`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `1.000`
Repair wins/ties/losses vs evolved: `3/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/3/0`
Selector regret vs repair: `0.000 over 0/3 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 3 | 1.00 | 0.000 | 0.044 | 0.011 |
| random | 3 | 1.00 | 0.000 | 0.063 | 0.016 |
| trajectory_selected | 3 | 2.00 | 0.000 | 0.044 | 0.011 |
| evolved | 3 | 4.00 | 0.000 | 0.044 | 0.011 |
| repair_selected | 3 | 5.00 | 1.000 | 0.066 | 0.767 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| symbolic | fixed | 3 | 1.00 | 0.000 | 0.044 | 0.011 |
| symbolic | random | 3 | 1.00 | 0.000 | 0.063 | 0.016 |
| symbolic | trajectory_selected | 3 | 2.00 | 0.000 | 0.044 | 0.011 |
| symbolic | evolved | 3 | 4.00 | 0.000 | 0.044 | 0.011 |
| symbolic | repair_selected | 3 | 5.00 | 1.000 | 0.066 | 0.767 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| counterfactual_answer_proposal | 3 | 3 | final | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/0/0 | 1.000 | 0.066 | 0.767 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | sym_008 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | counterfactual_answer_proposal | counterfactual_answer_proposal | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_counterfactual_proposal_match | final |  | 0.018 | 0.018 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_009 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | counterfactual_answer_proposal | counterfactual_answer_proposal | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_counterfactual_proposal_match | final |  | 0.020 | 0.020 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_010 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | counterfactual_answer_proposal | counterfactual_answer_proposal | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_counterfactual_proposal_match | final |  | 0.095 | 0.095 | 1.001 | 1.001 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
