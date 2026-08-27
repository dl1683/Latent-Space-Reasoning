# Diffusion Schedule-Selection Benchmark Report

Full model generations: `30`
Arm selections: `41`
Run ID: `diffusion-ae7a4edd5c22ca20`
Content hash: `ae7a4edd5c22ca2027c07f41ec361f7f741a6ed5fb7e4920af68cfdc4976619f`
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
Trajectory task delta vs fixed: `0.014`
Trajectory task delta vs random: `0.042`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `2.73`
Oracle task score: `0.473`
Oracle headroom vs trajectory: `0.049`
Oracle wins/ties/losses vs trajectory: `5/6/0`
Selector regret vs trajectory: `0.049 over 5/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.087`
Repair task delta vs random: `0.126`
Repair task delta vs trajectory: `0.068`
Repair task delta vs evolved: `0.068`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.068`
Repair wins/ties/losses vs evolved: `5/3/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.313170 | 0.000000 | 0.039232 | - | - |
| random perturbation | repair-covered tasks | 0.273938 | -0.039232 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.399866 | 0.086696 | 0.125929 | 5/3/0 | 5/3/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.410 | 0.481 | 0.427 |
| random | 11 | 1.00 | 0.381 | 0.408 | 0.388 |
| trajectory_selected | 11 | 2.00 | 0.423 | 0.472 | 0.435 |
| repair_selected | 8 | 3.00 | 0.400 | 0.676 | 0.469 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.313 | 0.627 | 0.392 |
| planning | random | 8 | 1.00 | 0.274 | 0.527 | 0.337 |
| planning | trajectory_selected | 8 | 2.00 | 0.332 | 0.615 | 0.403 |
| planning | repair_selected | 8 | 3.00 | 0.400 | 0.676 | 0.469 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_041 | low_confidence_32 | True | denoise_phase_repairable | 0.291 | 0.251 | 327 | True | 12 | 0.263 | True | True | 3.000 | 0.094 | 0.105 | 0.105 |
| llada-moe-7b-a1b-instruct-hf | plan_042 | low_confidence_32 | True | denoise_phase_repairable | 0.308 | 0.248 | 364 | True | 6 | 0.571 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-moe-7b-a1b-instruct-hf | plan_043 | low_confidence_32 | True | denoise_phase_repairable | 0.358 | 0.278 | 363 | True | 7 | 0.650 | True | True | 3.000 | 0.094 | 0.050 | 0.050 |
| llada-moe-7b-a1b-instruct-hf | plan_044 | low_confidence_32 | True | denoise_phase_repairable | 0.336 | 0.256 | 380 | True | 4 | 0.733 | True | True | 4.000 | 0.125 | 0.133 | 0.133 |
| llada-moe-7b-a1b-instruct-hf | plan_045 | low_confidence_32 | True | denoise_phase_repairable | 0.389 | 0.329 | 311 | True | 10 | 0.231 | True | True | 3.000 | 0.094 | 0.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_046 | low_confidence_32 | True | denoise_phase_repairable | 0.386 | 0.244 | 388 | True | 7 | 0.571 | True | True | 3.000 | 0.094 | 0.143 | 0.143 |
| llada-moe-7b-a1b-instruct-hf | plan_047 | low_confidence_32 | True | denoise_phase_repairable | 0.391 | 0.331 | 338 | True | 9 | 0.471 | True | True | 4.000 | 0.125 | 0.176 | 0.176 |
| llada-moe-7b-a1b-instruct-hf | plan_048 | low_confidence_32 | True | denoise_phase_repairable | 0.045 | 0.045 | 95 | True | 12 | 0.250 | True | True | 3.000 | 0.094 | 0.125 | 0.125 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 8 | 5 | low_confidence_32 | final | 38.8 | 1.000 | 0.000 | 0.000 | 0.015 | 0.000 | 0.067 | 0.070 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/0/3 | 0.383 | 0.683 | 0.458 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_041 | True | low_confidence_32 | 1.463 | 1.000 | 1.000 | 0.000 | 0.158 | False | If the gate candidate is valid,, proceed the repair candidate. |
| llada-moe-7b-a1b-instruct-hf | plan_041 | True | low_confidence_32 | 1.463 | 1.000 | 1.000 | 0.000 | 0.158 | False | If not, reject the gate candidate and proceed the repair candidate. |
| llada-moe-7b-a1b-instruct-hf | plan_041 | True | low_confidence_32 | 2.213 | 1.000 | 1.000 | 0.000 | 0.158 | False | If the gate candidate is invalid, reject the repair candidate. |
| llada-moe-7b-a1b-instruct-hf | plan_042 | False | low_confidence_32 | 1.355 | 0.798 | 1.000 | 0.000 | 0.214 | False | Use the seven candidates as a training set and evaluate the selector on performance met... |
| llada-moe-7b-a1b-instruct-hf | plan_042 | False | low_confidence_32 | 2.220 | 1.000 | 1.000 | 0.000 | 0.143 | False | if needed, test it on a separate set of candidates from different different slices or d... |
| llada-moe-7b-a1b-instruct-hf | plan_043 | False | low_confidence_32 | 1.446 | 0.968 | 1.000 | 0.000 | 0.100 | False | Score each states based relevance, coherence, and logical progression toward the solution. |
| llada-moe-7b-a1b-instruct-hf | plan_043 | False | low_confidence_32 | 2.006 | 0.653 | 1.000 | 0.000 | 0.350 | False | Compare these scores to the quality and correctness of the final answer to assess the u... |
| llada-moe-7b-a1b-instruct-hf | plan_044 | True | low_confidence_32 | 2.013 | 0.667 | 1.000 | 0.000 | 0.400 | False | Conduct the test before promoting the repair policy to assess the trade-off between ans... |
| llada-moe-7b-a1b-instruct-hf | plan_045 | False | low_confidence_32 | 1.923 | 0.433 | 1.000 | 0.000 | 0.231 | False | Use additional schedules only when the marginal cost in GPU time exceeds the marginal b... |
| llada-moe-7b-a1b-instruct-hf | plan_046 | True | low_confidence_32 | 1.938 | 0.559 | 1.000 | 0.000 | 0.429 | False | These states could be tested against a benchmark dataset known to contain useful missin... |
| llada-moe-7b-a1b-instruct-hf | plan_047 | True | low_confidence_32 | 1.296 | 0.717 | 1.000 | 0.000 | 0.294 | False | Use a reward mechanism to ensure the judge's rewards are with the system's goals, Imple... |
| llada-moe-7b-a1b-instruct-hf | plan_047 | True | low_confidence_32 | 2.211 | 1.000 | 1.000 | 0.000 | 0.176 | False | Use a loss function to guide the judge judge's behavior towards correct decisions. |
| llada-moe-7b-a1b-instruct-hf | plan_048 | True | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Release the positive average lift as, ,,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_041 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.314 | 0.000 | 0.390 | 0.138 | 0.291 | 0.045 | 0.291 | 0.000 | 0.398 | 0.107 | 0.398 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_042 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.366 | 0.000 | 0.248 | 0.000 | 0.308 | 0.308 | 0.308 | 0.000 | 0.308 | 0.000 | 0.308 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_043 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.419 | 0.000 | 0.278 | 0.000 | 0.358 | 0.358 | 0.358 | 0.000 | 0.358 | 0.000 | 0.358 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_044 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.390 | 0.000 | 0.442 | 0.186 | 0.336 | 0.336 | 0.336 | 0.000 | 0.449 | 0.112 | 0.449 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_045 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.321 | 0.000 | 0.352 | 0.000 | 0.389 | 0.389 | 0.389 | 0.000 | 0.389 | 0.000 | 0.389 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_046 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.368 | 0.000 | 0.398 | 0.154 | 0.386 | 0.282 | 0.386 | 0.000 | 0.463 | 0.076 | 0.463 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_047 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.392 | 0.000 | 0.413 | 0.035 | 0.391 | 0.279 | 0.391 | 0.000 | 0.399 | 0.008 | 0.399 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_048 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.267 | 0.000 | 0.369 | 0.215 | 0.045 | 0.195 | 0.195 | 0.000 | 0.435 | 0.241 | 0.435 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
