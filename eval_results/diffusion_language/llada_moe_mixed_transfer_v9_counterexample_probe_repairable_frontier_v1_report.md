# Diffusion Schedule-Selection Benchmark Report

Full model generations: `30`
Arm selections: `41`
Run ID: `diffusion-ba2ef3f1e4dd6d47`
Content hash: `ba2ef3f1e4dd6d4751c0c5d17e69c644618e303b8c48068d92a39ceeb1922a41`
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
Trajectory task delta vs fixed: `-0.006`
Trajectory task delta vs random: `0.005`
Trajectory wins/ties/losses vs fixed: `0/10/1`
Trajectory wins/ties/losses vs random: `1/10/0`
Oracle generation budget/task: `2.73`
Oracle task score: `0.437`
Oracle headroom vs trajectory: `0.050`
Oracle wins/ties/losses vs trajectory: `5/6/0`
Selector regret vs trajectory: `0.050 over 5/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.060`
Repair task delta vs random: `0.075`
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
| fixed baseline | repair-covered tasks | 0.290893 | 0.000000 | 0.014571 | - | - |
| random perturbation | repair-covered tasks | 0.276321 | -0.014571 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.351205 | 0.060312 | 0.074884 | 5/3/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.393 | 0.504 | 0.421 |
| random | 11 | 1.00 | 0.383 | 0.492 | 0.410 |
| trajectory_selected | 11 | 2.00 | 0.388 | 0.492 | 0.414 |
| repair_selected | 8 | 3.00 | 0.351 | 0.672 | 0.431 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.291 | 0.659 | 0.383 |
| planning | random | 8 | 1.00 | 0.276 | 0.642 | 0.368 |
| planning | trajectory_selected | 8 | 2.00 | 0.283 | 0.642 | 0.373 |
| planning | repair_selected | 8 | 3.00 | 0.351 | 0.672 | 0.431 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_065 | low_confidence_32 | True | denoise_phase_repairable | 0.304 | 0.244 | 365 | True | 12 | 0.429 | True | True | 4.000 | 0.125 | 0.048 | 0.048 |
| llada-moe-7b-a1b-instruct-hf | plan_066 | low_confidence_32 | True | denoise_phase_repairable | 0.384 | 0.324 | 323 | True | 4 | 0.765 | True | True | 3.000 | 0.094 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_067 | low_confidence_32 | True | denoise_phase_repairable | 0.241 | 0.201 | 373 | True | 12 | 0.056 | True | True | 4.000 | 0.125 | 0.056 | 0.056 |
| llada-moe-7b-a1b-instruct-hf | plan_068 | low_confidence_32 | True | denoise_phase_repairable | 0.241 | 0.201 | 200 | True | 12 | 0.250 | True | True | 4.000 | 0.125 | 0.050 | 0.050 |
| llada-moe-7b-a1b-instruct-hf | plan_069 | low_confidence_32 | True | denoise_phase_repairable | 0.358 | 0.278 | 339 | True | 12 | 0.421 | True | True | 4.000 | 0.125 | 0.105 | 0.105 |
| llada-moe-7b-a1b-instruct-hf | plan_070 | low_confidence_32 | True | denoise_phase_repairable | 0.274 | 0.214 | 383 | True | 5 | 0.765 | True | True | 4.000 | 0.125 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_071 | low_confidence_32 | True | denoise_phase_repairable | 0.220 | 0.180 | 201 | True | 12 | 0.176 | True | True | 4.000 | 0.125 | 0.118 | 0.118 |
| llada-moe-7b-a1b-instruct-hf | plan_072 | low_confidence_32 | True | denoise_phase_repairable | 0.304 | 0.244 | 339 | True | 2 | 0.952 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 8 | 6 | low_confidence_32 | final | 40.1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.043 | 0.053 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/2/1 | 0.344 | 0.680 | 0.428 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_065 | True | low_confidence_32 | 1.887 | 0.389 | 1.000 | 0.000 | 0.333 | False | The probe should measure task-specific accuracy, logical consistency, and decision boun... |
| llada-moe-7b-a1b-instruct-hf | plan_066 | True | low_confidence_32 | 1.327 | 0.813 | 1.000 | 0.000 | 0.412 | False | However, the the plan may still lack an ordering of tests and rollback criteria. |
| llada-moe-7b-a1b-instruct-hf | plan_066 | True | low_confidence_32 | 2.127 | 0.925 | 1.000 | 0.000 | 0.294 | False | Therefore, the audit should focus on verifying the sequence of the tests and the presen... |
| llada-moe-7b-a1b-instruct-hf | plan_067 | True | low_confidence_32 | 2.130 | 1.000 | 1.000 | 0.000 | 0.056 | False | Scheduleitize repairs by operational urgency and impact. |
| llada-moe-7b-a1b-instruct-hf | plan_067 | True | low_confidence_32 | 2.099 | 0.925 | 1.000 | 0.000 | 0.000 | False | Conduct regular audits to identify underused or redundant resources. |
| llada-moe-7b-a1b-instruct-hf | plan_067 | True | low_confidence_32 | 2.891 | 1.000 | 1.000 | 0.000 | 0.000 | False | Automate procurement and approval workflows to ensure compliance and and reduce delays. |
| llada-moe-7b-a1b-instruct-hf | plan_068 | True | low_confidence_32 | 2.893 | 1.000 | 1.000 | 0.000 | 0.000 | False | The, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, th... |
| llada-moe-7b-a1b-instruct-hf | plan_069 | False | low_confidence_32 | 1.396 | 0.916 | 1.000 | 0.000 | 0.316 | False | Apply the proxy to a subset of candidates before repair, predict their promoability, an... |
| llada-moe-7b-a1b-instruct-hf | plan_069 | False | low_confidence_32 | 2.142 | 0.893 | 1.000 | 0.000 | 0.263 | False | This measures the proxy's predictive power while avoiding post-repair labels. |
| llada-moe-7b-a1b-instruct-hf | plan_070 | True | low_confidence_32 | 1.923 | 0.469 | 1.000 | 0.000 | 0.353 | False | This involves assessing the weak intermediate state's reliability and performance again... |
| llada-moe-7b-a1b-instruct-hf | plan_071 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The evidence table should include: GPU Cost,, Prompt ID,, Prompt ID,, Prompt ID,, Promp... |
| llada-moe-7b-a1b-instruct-hf | plan_072 | True | low_confidence_32 | 2.197 | 1.000 | 1.000 | 0.000 | 0.190 | False | This slice will evaluate the impact of these factors on repair value, ensuring the theo... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_065 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.338 | 0.000 | 0.367 | 0.122 | 0.304 | 0.304 | 0.304 | 0.000 | 0.379 | 0.075 | 0.379 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_066 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.426 | 0.000 | 0.424 | 0.024 | 0.384 | 0.384 | 0.384 | 0.000 | 0.384 | 0.000 | 0.384 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_067 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.213 | 0.000 | 0.375 | 0.236 | 0.241 | 0.178 | 0.178 | 0.000 | 0.418 | 0.239 | 0.418 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_068 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.206 | 0.000 | 0.396 | 0.195 | 0.241 | 0.241 | 0.241 | 0.000 | 0.401 | 0.160 | 0.401 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_069 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.339 | 0.000 | 0.278 | 0.000 | 0.358 | 0.304 | 0.358 | 0.000 | 0.358 | 0.000 | 0.358 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_070 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.410 | 0.000 | 0.226 | 0.013 | 0.274 | 0.274 | 0.274 | 0.000 | 0.286 | 0.013 | 0.286 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_071 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool |  |  |  | 0.226 | 0.000 | 0.180 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_072 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.476 | 0.000 | 0.403 | 0.159 | 0.304 | 0.304 | 0.304 | 0.000 | 0.363 | 0.059 | 0.363 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
