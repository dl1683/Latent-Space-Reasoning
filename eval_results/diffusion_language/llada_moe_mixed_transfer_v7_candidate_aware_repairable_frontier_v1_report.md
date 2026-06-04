# Diffusion Schedule-Selection Benchmark Report

Full model generations: `30`
Arm selections: `41`
Run ID: `diffusion-711ea5fcfd8c07e5`
Content hash: `711ea5fcfd8c07e59032f30dc406570929fc4b058fdaab95efb7bccf3a17d018`
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
History mutability: `monotonic 30/30, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span_phase_final_preserve_seeded_gated`
Repair source policy: `fixed`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `denoise_phase_repairability`
Repair source-quality threshold: `0.500`
Repair source min chars: `320`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `999`
Repair source prompt coverage band: `0.000-1.000`
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
Repair selector: `candidate_aware_promotion_v1`
Repair promotion margin: `0.000`
Trajectory task delta vs fixed: `-0.007`
Trajectory task delta vs random: `0.036`
Trajectory wins/ties/losses vs fixed: `1/9/1`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `2.73`
Oracle task score: `0.434`
Oracle headroom vs trajectory: `0.036`
Oracle wins/ties/losses vs trajectory: `3/8/0`
Selector regret vs trajectory: `0.036 over 3/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.023`
Repair task delta vs random: `0.082`
Repair task delta vs trajectory: `0.033`
Repair task delta vs evolved: `0.033`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.033`
Repair wins/ties/losses vs evolved: `2/6/0`
Oracle headroom vs repair: `0.017`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.017 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.306339 | 0.000000 | 0.059027 | - | - |
| random perturbation | repair-covered tasks | 0.247312 | -0.059027 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.329375 | 0.023036 | 0.082062 | 3/4/1 | 4/4/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.405 | 0.490 | 0.426 |
| random | 11 | 1.00 | 0.362 | 0.382 | 0.367 |
| trajectory_selected | 11 | 2.00 | 0.398 | 0.462 | 0.414 |
| repair_selected | 8 | 3.00 | 0.329 | 0.608 | 0.399 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.306 | 0.639 | 0.390 |
| planning | random | 8 | 1.00 | 0.247 | 0.491 | 0.308 |
| planning | trajectory_selected | 8 | 2.00 | 0.297 | 0.601 | 0.373 |
| planning | repair_selected | 8 | 3.00 | 0.329 | 0.608 | 0.399 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_049 | low_confidence_32 | True | denoise_phase_repairable | 0.180 | 0.180 | 135 | True | 12 | 0.077 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-moe-7b-a1b-instruct-hf | plan_050 | low_confidence_32 | True | denoise_phase_repairable | 0.342 | 0.282 | 296 | True | 9 | 0.385 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_051 | low_confidence_32 | True | denoise_phase_repairable | 0.386 | 0.244 | 372 | True | 6 | 0.667 | True | True | 4.000 | 0.125 | 0.067 | 0.067 |
| llada-moe-7b-a1b-instruct-hf | plan_052 | low_confidence_32 | True | denoise_phase_repairable | 0.318 | 0.217 | 381 | True | 6 | 0.647 | True | True | 3.000 | 0.094 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_053 | low_confidence_32 | True | denoise_phase_repairable | 0.241 | 0.201 | 342 | True | 4 | 0.750 | True | True | 4.000 | 0.125 | 0.188 | 0.188 |
| llada-moe-7b-a1b-instruct-hf | plan_054 | low_confidence_32 | True | denoise_phase_repairable | 0.327 | 0.287 | 385 | True | 12 | 0.143 | True | True | 3.000 | 0.094 | 0.071 | 0.071 |
| llada-moe-7b-a1b-instruct-hf | plan_055 | low_confidence_32 | True | denoise_phase_repairable | 0.316 | 0.256 | 351 | True | 4 | 0.714 | True | True | 3.000 | 0.094 | 0.214 | 0.214 |
| llada-moe-7b-a1b-instruct-hf | plan_056 | low_confidence_32 | True | denoise_phase_repairable | 0.340 | 0.260 | 447 | True | 8 | 0.529 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 8 | 2 | low_confidence_32 | final | 42.8 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/3/3 | 0.314 | 0.629 | 0.393 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_049 | False | low_confidence_32 | 2.893 | 1.000 | 1.000 | 0.000 | 0.077 | False | I benchmark v v7 v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v... |
| llada-moe-7b-a1b-instruct-hf | plan_050 | False | low_confidence_32 | 2.733 | 0.710 | 1.000 | 0.000 | 0.000 | False | Train the model on real data, validate on synthetic data, and use appropriate metrics (... |
| llada-moe-7b-a1b-instruct-hf | plan_051 | False | low_confidence_32 | 1.351 | 0.818 | 1.000 | 0.000 | 0.200 | False | Measure the accuracy of the predicted denoised states in identifying these constraints. |
| llada-moe-7b-a1b-instruct-hf | plan_051 | False | low_confidence_32 | 1.962 | 0.550 | 1.000 | 0.000 | 0.400 | False | If the benchmark scores low, it indicates that the controller did not correctly predict... |
| llada-moe-7b-a1b-instruct-hf | plan_052 | False | low_confidence_32 | 1.445 | 1.000 | 1.000 | 0.000 | 0.235 | False | Look for any inconsistencies or orbs of information that could indicate the judge is ob... |
| llada-moe-7b-a1b-instruct-hf | plan_052 | False | low_confidence_32 | 1.922 | 0.445 | 1.000 | 0.000 | 0.294 | False | The goal is to identify potential vulnerabilities related to the repair system's relian... |
| llada-moe-7b-a1b-instruct-hf | plan_053 | False | low_confidence_32 | 2.171 | 0.925 | 1.000 | 0.000 | 0.125 | False | Analyze the data to determine if there is a curved relationship between the values of t... |
| llada-moe-7b-a1b-instruct-hf | plan_054 | True | low_confidence_32 | 2.124 | 1.000 | 1.000 | 0.000 | 0.071 | False | Trace the candidate generation process and compare observed outcomes with with expected... |
| llada-moe-7b-a1b-instruct-hf | plan_054 | True | low_confidence_32 | 2.077 | 0.735 | 1.000 | 0.000 | 0.143 | False | Ensure alignment between the policy,, candidate, and reward functions to identify poten... |
| llada-moe-7b-a1b-instruct-hf | plan_055 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | A hardware-aware validation before switching model precision should include: (1) Benchm... |
| llada-moe-7b-a1b-instruct-hf | plan_056 | True | low_confidence_32 | 1.378 | 0.893 | 1.000 | 0.000 | 0.294 | False | Measure reasoning performance under different repair conditions, focusing on whether th... |
| llada-moe-7b-a1b-instruct-hf | plan_056 | True | low_confidence_32 | 1.759 | 0.168 | 1.000 | 0.000 | 0.471 | False | Analyze the correlation between improved reasoning outcomes and reduced task-relevant i... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_049 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.140 | 0.000 | 0.045 | 0.000 | 0.180 | 0.045 | 0.045 | 0.000 | 0.045 | 0.000 | 0.180 | 0.135 |
| llada-moe-7b-a1b-instruct-hf | plan_050 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.353 | 0.000 | 0.282 | 0.000 | 0.342 | 0.304 | 0.342 | 0.000 | 0.342 | 0.000 | 0.342 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_051 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.400 | 0.000 | 0.244 | 0.000 | 0.386 | 0.386 | 0.386 | 0.000 | 0.386 | 0.000 | 0.386 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_052 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.385 | 0.000 | 0.217 | 0.000 | 0.318 | 0.217 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_053 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.409 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_054 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.275 | 0.000 | 0.519 | 0.232 | 0.327 | 0.157 | 0.327 | 0.000 | 0.493 | 0.166 | 0.493 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_055 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.445 | 0.000 | 0.387 | 0.000 | 0.316 | 0.375 | 0.375 | 0.000 | 0.375 | 0.000 | 0.375 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_056 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.380 | 0.000 | 0.451 | 0.190 | 0.340 | 0.252 | 0.340 | 0.000 | 0.434 | 0.094 | 0.434 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
