# Diffusion Schedule-Selection Benchmark Report

Full model generations: `30`
Arm selections: `41`
Run ID: `diffusion-5ccf340cde81d101`
Content hash: `5ccf340cde81d10168820fb2d250eb7a8f18354f8f81ea9015fde9a205f1dc45`
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
Trajectory task delta vs fixed: `-0.002`
Trajectory task delta vs random: `0.059`
Trajectory wins/ties/losses vs fixed: `1/9/1`
Trajectory wins/ties/losses vs random: `4/6/1`
Oracle generation budget/task: `2.73`
Oracle task score: `0.478`
Oracle headroom vs trajectory: `0.026`
Oracle wins/ties/losses vs trajectory: `4/7/0`
Selector regret vs trajectory: `0.026 over 4/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.033`
Repair task delta vs random: `0.118`
Repair task delta vs trajectory: `0.036`
Repair task delta vs evolved: `0.036`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.036`
Repair wins/ties/losses vs evolved: `4/4/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.374598 | 0.000000 | 0.084589 | - | - |
| random perturbation | repair-covered tasks | 0.290009 | -0.084589 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.407589 | 0.032991 | 0.117580 | 4/4/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.454 | 0.501 | 0.466 |
| random | 11 | 1.00 | 0.393 | 0.383 | 0.390 |
| trajectory_selected | 11 | 2.00 | 0.452 | 0.492 | 0.462 |
| repair_selected | 8 | 3.00 | 0.408 | 0.671 | 0.473 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.375 | 0.655 | 0.445 |
| planning | random | 8 | 1.00 | 0.290 | 0.492 | 0.341 |
| planning | trajectory_selected | 8 | 2.00 | 0.371 | 0.642 | 0.439 |
| planning | repair_selected | 8 | 3.00 | 0.408 | 0.671 | 0.473 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_057 | low_confidence_32 | True | denoise_phase_repairable | 0.329 | 0.269 | 301 | True | 6 | 0.600 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_058 | low_confidence_32 | True | denoise_phase_repairable | 0.378 | 0.298 | 320 | True | 9 | 0.529 | True | True | 4.000 | 0.125 | 0.235 | 0.235 |
| llada-moe-7b-a1b-instruct-hf | plan_059 | low_confidence_32 | True | denoise_phase_repairable | 0.437 | 0.319 | 369 | True | 3 | 0.769 | True | True | 4.000 | 0.125 | 0.154 | 0.154 |
| llada-moe-7b-a1b-instruct-hf | plan_060 | low_confidence_32 | True | denoise_phase_repairable | 0.435 | 0.333 | 358 | True | 8 | 0.529 | True | True | 4.000 | 0.125 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_061 | low_confidence_32 | True | denoise_phase_repairable | 0.389 | 0.286 | 332 | True | 3 | 0.700 | True | True | 3.000 | 0.094 | 0.300 | 0.300 |
| llada-moe-7b-a1b-instruct-hf | plan_062 | low_confidence_32 | True | denoise_phase_repairable | 0.420 | 0.340 | 336 | True | 5 | 0.643 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |
| llada-moe-7b-a1b-instruct-hf | plan_063 | low_confidence_32 | True | denoise_phase_repairable | 0.429 | 0.286 | 374 | True | 6 | 0.615 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-moe-7b-a1b-instruct-hf | plan_064 | low_confidence_32 | True | denoise_phase_repairable | 0.180 | 0.180 | 168 | True | 12 | 0.154 | True | True | 4.000 | 0.125 | 0.154 | 0.154 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 8 | 4 | low_confidence_32 | final | 25.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/1/4 | 0.384 | 0.679 | 0.458 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_057 | False | low_confidence_32 | 1.866 | 0.416 | 1.000 | 0.000 | 0.533 | False | Compare the number of no-lift repairs attempts and number of missed positive repairs to... |
| llada-moe-7b-a1b-instruct-hf | plan_058 | True | low_confidence_32 | 1.721 | 0.581 | 1.000 | 0.000 | 0.412 | False | Test the promotion selector's error rate on generated candidates, and test the spend ga... |
| llada-moe-7b-a1b-instruct-hf | plan_058 | True | low_confidence_32 | 2.714 | 1.000 | 1.000 | 0.000 | 0.176 | False | This allows you to distinguish the the selector's error from the spend gate issue. |
| llada-moe-7b-a1b-instruct-hf | plan_059 | False | low_confidence_32 | 2.610 | 0.837 | 1.000 | 0.000 | 0.231 | False | This test will validate the feature's reliability and ensure consistency with source-qu... |
| llada-moe-7b-a1b-instruct-hf | plan_060 | True | low_confidence_32 | 2.704 | 1.000 | 1.000 | 0.000 | 0.294 | False | Additionally, evaluate the the of the the constraints to ensure the repair is effective... |
| llada-moe-7b-a1b-instruct-hf | plan_061 | True | low_confidence_32 | 2.876 | 1.000 | 1.000 | 0.000 | 0.100 | False | Decide based on task complexity, resource availability, and performance goals. |
| llada-moe-7b-a1b-instruct-hf | plan_062 | False | low_confidence_32 | 1.998 | 0.622 | 1.000 | 0.000 | 0.357 | False | This ensures the model is evaluated on fresh, unseen data, reducing the risk of overfit... |
| llada-moe-7b-a1b-instruct-hf | plan_063 | True | low_confidence_32 | 1.457 | 1.000 | 1.000 | 0.000 | 0.154 | False | Use a probabilistic approach: evaluate denoised history for consistency, factual accura... |
| llada-moe-7b-a1b-instruct-hf | plan_063 | True | low_confidence_32 | 2.138 | 0.873 | 1.000 | 0.000 | 0.231 | False | Prioritize denoised history that retains critical context and contextual integrity, ens... |
| llada-moe-7b-a1b-instruct-hf | plan_064 | False | low_confidence_32 | 2.214 | 1.000 | 1.000 | 0.000 | 0.154 | False | run, and both outcomes. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_057 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.392 | 0.000 | 0.269 | 0.000 | 0.329 | 0.157 | 0.329 | 0.000 | 0.329 | 0.000 | 0.329 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_058 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.369 | 0.000 | 0.431 | 0.134 | 0.378 | 0.248 | 0.378 | 0.000 | 0.391 | 0.014 | 0.391 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_059 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.473 | 0.000 | 0.396 | 0.000 | 0.437 | 0.437 | 0.437 | 0.000 | 0.437 | 0.000 | 0.437 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_060 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.356 | 0.000 | 0.525 | 0.140 | 0.435 | 0.435 | 0.435 | 0.000 | 0.569 | 0.134 | 0.569 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_061 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.455 | 0.000 | 0.286 | 0.027 | 0.389 | 0.389 | 0.342 | 0.000 | 0.389 | 0.047 | 0.389 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_062 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.438 | 0.000 | 0.405 | 0.000 | 0.420 | 0.045 | 0.420 | 0.000 | 0.420 | 0.000 | 0.420 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_063 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.420 | 0.000 | 0.483 | 0.196 | 0.429 | 0.429 | 0.429 | 0.000 | 0.525 | 0.096 | 0.525 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_064 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.304 | 0.000 | 0.180 | 0.000 | 0.180 | 0.180 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
