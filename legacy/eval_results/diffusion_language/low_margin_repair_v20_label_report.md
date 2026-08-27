# Diffusion Schedule-Selection Benchmark Report

Full model generations: `26`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-19d1c9173d08bc15`
Content hash: `19d1c9173d08bc157c6aa33c9581932ee7e4308442c48fe9a3d09ba29f79ac04`
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
History mutability: `monotonic 26/26, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span_phase_final_preserve_seeded_gated`
Repair source policy: `random`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `denoise_phase_repairability`
Counterfactual probe mode: `triage`
Counterfactual probe policy: `deterministic_missing_constraint_probe_v1`
Repair source-quality threshold: `0.500`
Repair source min chars: `240`
Repair source prompt-gap min: `2`
Repair source prompt-gap max: `9`
Repair source prompt coverage band: `0.400-1.000`
Repair value-proxy source-quality max: `0.310`
Repair transfer source-task min: `0.2954`
Repair phase budget: `frontier`
Repair denoise skeleton max step: `31.000`
Phase-source threshold band: `target>=0.960, text>=0.960, chars>=0.950`
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
Repair selector: `planning_quality`
Repair promotion margin: `0.000`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.088`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `7/4/0`
Oracle generation budget/task: `2.36`
Oracle task score: `0.425`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/11/0`
Selector regret vs trajectory: `0.000 over 0/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.121`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.50`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.334643 | 0.000000 | 0.121009 | - | - |
| random perturbation | repair-covered tasks | 0.213634 | -0.121009 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.334643 | 0.000000 | 0.121009 | 0/8/0 | 7/1/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.425 | 0.504 | 0.445 |
| random | 11 | 1.00 | 0.337 | 0.332 | 0.336 |
| trajectory_selected | 11 | 2.00 | 0.425 | 0.504 | 0.445 |
| repair_selected | 8 | 2.50 | 0.335 | 0.659 | 0.416 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.335 | 0.659 | 0.416 |
| planning | random | 8 | 1.00 | 0.214 | 0.422 | 0.266 |
| planning | trajectory_selected | 8 | 2.00 | 0.335 | 0.659 | 0.416 |
| planning | repair_selected | 8 | 2.50 | 0.335 | 0.659 | 0.416 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_153 | low_confidence_32 | False | outside_repairable_band | False |  | 0.320 | 0.260 | 350 | True | 1 | 0.950 | False | True | 9.000 | 0.281 | 0.450 | 0.450 |
| llada-moe-7b-a1b-instruct-hf | plan_154 | random_32 | True | denoise_phase_repairable | False |  | 0.198 | 0.138 | 155 | True | 9 | 0.429 | True | True | 26.000 | 0.812 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_155 | random_32 | True | denoise_phase_repairable | False |  | 0.235 | 0.154 | 161 | True | 5 | 0.667 | True | True | 23.000 | 0.719 | 0.467 | 0.467 |
| llada-moe-7b-a1b-instruct-hf | plan_156 | random_32 | False | outside_repairable_band | False |  | 0.085 | 0.045 | 22 | True | 10 | 0.231 | False | False | none | none | none | 0.231 |
| llada-moe-7b-a1b-instruct-hf | plan_157 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 2 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_158 | random_32 | True | denoise_phase_repairable | False |  | 0.198 | 0.138 | 160 | True | 6 | 0.538 | True | True | 24.000 | 0.750 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_159 | random_32 | False | outside_repairable_band | False |  | 0.200 | 0.160 | 93 | True | 12 | 0.125 | False | False | none | none | none | 0.125 |
| llada-moe-7b-a1b-instruct-hf | plan_160 | random_32 | True | denoise_phase_repairable | False |  | 0.428 | 0.348 | 342 | True | 9 | 0.471 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 4 | 0 | random_32 | final | 29.2 | 1.000 | 0.000 | 0.000 | 0.060 | 0.060 | 0.003 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/2/0 | 0.273 | 0.569 | 0.347 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_154 | False | random_32 | 1.440 | 1.000 | 1.000 | 0.000 | 0.286 | False | Avoid task value with span metrics. |
| llada-moe-7b-a1b-instruct-hf | plan_154 | False | random_32 | 2.181 | 1.000 | 1.000 | 0.000 | 0.357 | False | Audit on span metrics, content, structure and task value. |
| llada-moe-7b-a1b-instruct-hf | plan_155 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | For candidates with zero source-relative planning-quality delta, the source-tie fallbac... |
| llada-moe-7b-a1b-instruct-hf | plan_158 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Test whether the absence of strong span evidence correlates with a low-margin signal us... |
| llada-moe-7b-a1b-instruct-hf | plan_160 | False | random_32 | 1.832 | 0.737 | 1.000 | 0.000 | 0.118 | False | Explain its role as a validation, correction, or recovery mechanism, not undermining th... |
| llada-moe-7b-a1b-instruct-hf | plan_160 | False | random_32 | 2.493 | 0.567 | 1.000 | 0.000 | 0.235 | False | Clarify it aligns with the main hook claim by enhancing resilience, robustness, and ada... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_153 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.424 | 0.000 | 0.260 | 0.000 | 0.320 | 0.320 | 0.320 | 0.000 | 0.320 | 0.000 | 0.320 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_154 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_planning_quality_score_repair_pool |  |  |  | 0.328 | 0.000 | 0.201 | 0.000 | 0.261 | 0.198 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_155 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_planning_quality_score_repair_pool |  |  |  | 0.468 | 0.000 | 0.217 | 0.000 | 0.297 | 0.235 | 0.297 | 0.000 | 0.297 | 0.000 | 0.297 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_156 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.357 | 0.000 | 0.214 | 0.000 | 0.314 | 0.085 | 0.314 | 0.000 | 0.314 | 0.000 | 0.314 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_157 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.364 | 0.000 | 0.180 | 0.000 | 0.240 | 0.045 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_158 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_planning_quality_score_repair_pool |  |  |  | 0.441 | 0.000 | 0.266 | 0.000 | 0.326 | 0.198 | 0.326 | 0.000 | 0.326 | 0.000 | 0.326 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_159 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.521 | 0.000 | 0.286 | 0.000 | 0.346 | 0.200 | 0.346 | 0.000 | 0.346 | 0.000 | 0.346 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_160 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_planning_quality_score_repair_pool |  |  |  | 0.451 | 0.000 | 0.492 | 0.000 | 0.572 | 0.428 | 0.572 | 0.000 | 0.572 | 0.000 | 0.572 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
