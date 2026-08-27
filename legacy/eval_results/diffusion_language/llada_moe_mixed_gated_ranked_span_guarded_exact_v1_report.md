# Diffusion Schedule-Selection Benchmark Report

Full model generations: `60`
Arm selections: `52`
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
History mutability: `monotonic 60/60, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Constraint-gap rescue limit: `3`
Constraint-gap rescue min terms: `6`
Constraint-gap rescue source-quality band: `0.400-0.500`
Constraint-gap rescue source controls: ``
Repair selector: `planning_quality_delta_risk_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.002`
Trajectory task delta vs random: `0.031`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `5.45`
Oracle task score: `0.598`
Oracle headroom vs trajectory: `0.023`
Oracle wins/ties/losses vs trajectory: `4/7/0`
Selector regret vs trajectory: `0.023 over 4/11 improvable`
Exact proposal-history sources: `evolved:fallback=1, evolved:final=2, trajectory_selected:fallback=1, trajectory_selected:final=2`
Evolved task delta vs fixed: `0.007`
Evolved task delta vs random: `0.036`
Evolved task delta vs trajectory: `0.005`
Evolved wins/ties/losses vs fixed: `3/7/1`
Evolved wins/ties/losses vs random: `4/7/0`
Evolved wins/ties/losses vs trajectory: `3/7/1`
Oracle headroom vs evolved: `0.018`
Oracle wins/ties/losses vs evolved: `2/9/0`
Selector regret vs evolved: `0.018 over 2/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.034`
Repair task delta vs random: `0.074`
Repair task delta vs trajectory: `0.031`
Repair task delta vs evolved: `0.024`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.012`
Repair wins/ties/losses vs evolved: `1/7/0`
Oracle headroom vs repair: `0.001`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.001 over 1/8 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.573 | 0.565 | 0.571 |
| random | 11 | 1.00 | 0.543 | 0.517 | 0.537 |
| trajectory_selected | 11 | 2.00 | 0.574 | 0.573 | 0.574 |
| evolved | 11 | 4.00 | 0.580 | 0.571 | 0.578 |
| repair_selected | 8 | 6.00 | 0.446 | 0.693 | 0.508 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.282 | 0.820 |
| math | random | 1 | 1.00 | 1.000 | 0.282 | 0.820 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.282 | 0.820 |
| math | evolved | 1 | 4.00 | 1.000 | 0.282 | 0.820 |
| planning | fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| planning | random | 8 | 1.00 | 0.372 | 0.638 | 0.439 |
| planning | trajectory_selected | 8 | 2.00 | 0.415 | 0.698 | 0.485 |
| planning | evolved | 8 | 4.00 | 0.422 | 0.695 | 0.490 |
| planning | repair_selected | 8 | 6.00 | 0.446 | 0.693 | 0.508 |
| science | fixed | 1 | 1.00 | 1.000 | 0.310 | 0.827 |
| science | random | 1 | 1.00 | 1.000 | 0.161 | 0.790 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.310 | 0.827 |
| science | evolved | 1 | 4.00 | 1.000 | 0.310 | 0.827 |
| symbolic | fixed | 1 | 1.00 | 1.000 | 0.039 | 0.760 |
| symbolic | random | 1 | 1.00 | 1.000 | 0.135 | 0.784 |
| symbolic | trajectory_selected | 1 | 2.00 | 1.000 | 0.135 | 0.784 |
| symbolic | evolved | 1 | 4.00 | 1.000 | 0.135 | 0.784 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| prefix_25_repair | 8 | 0 | final | 48.0 | 0.000 | 0.000 | 0.000 | -0.053 | -0.068 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/2/5 | 0.354 | 0.606 | 0.417 |
| state_adaptive_history_repair | 8 | 1 | history | 42.4 | 0.000 | 0.000 | 0.000 | -0.026 | -0.041 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/3/4 | 0.381 | 0.657 | 0.450 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | exact_answer_proposal_final_match | exact_answer_proposal_final_match |  |  |  | 0.282 | 0.282 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  | 0.426 | 0.426 | 0.000 | 0.000 | 0.465 | 0.465 | 0.465 | 0.465 | 0.465 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.441 | 0.488 | 0.000 | 0.000 | 0.689 | 0.580 | 0.689 | 0.684 | 0.684 | 0.000 | 0.689 | 0.005 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  | 0.424 | 0.424 | 0.000 | 0.000 | 0.422 | 0.422 | 0.422 | 0.422 | 0.422 | 0.000 | 0.422 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.467 | 0.496 | 0.000 | 0.000 | 0.338 | 0.157 | 0.338 | 0.358 | 0.358 | 0.000 | 0.358 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  | 0.330 | 0.330 | 0.000 | 0.000 | 0.421 | 0.421 | 0.421 | 0.421 | 0.421 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.372 | 0.404 | 0.000 | 0.000 | 0.391 | 0.341 | 0.391 | 0.433 | 0.433 | 0.000 | 0.433 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_delta_risk_guarded_score_repair_pool | history | 26 | 0.331 | 0.331 | 0.121 | 0.121 | 0.307 | 0.307 | 0.307 | 0.307 | 0.499 | 0.191 | 0.499 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.273 | 0.269 | 0.000 | 0.000 | 0.264 | 0.283 | 0.283 | 0.286 | 0.286 | 0.000 | 0.286 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed |  |  |  | 0.310 | 0.310 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_002 | low_confidence_32 | random_32 | random_32 | random_32 |  | random_32 | exact_answer_proposal_final_match | exact_answer_proposal_final_match |  |  |  | 0.135 | 0.135 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
