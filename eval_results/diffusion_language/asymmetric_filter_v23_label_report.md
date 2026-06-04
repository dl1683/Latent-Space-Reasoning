# Diffusion Schedule-Selection Benchmark Report

Full model generations: `36`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-27588651dd595f61`
Content hash: `27588651dd595f616d43b76db5bf33d76632e6cb0eedfc7deeabaa1ca9195106`
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
History mutability: `monotonic 36/36, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `True`
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
Repair selector: `generated_repair_value_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.003`
Trajectory task delta vs random: `0.043`
Trajectory wins/ties/losses vs fixed: `2/9/0`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `3.27`
Oracle task score: `0.474`
Oracle headroom vs trajectory: `0.042`
Oracle wins/ties/losses vs trajectory: `6/5/0`
Selector regret vs trajectory: `0.042 over 6/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.062`
Repair task delta vs random: `0.118`
Repair task delta vs trajectory: `0.058`
Repair task delta vs evolved: `0.058`
Repair generation budget delta vs evolved: `1.75`
Repair task delta per extra generation vs evolved: `0.033`
Repair wins/ties/losses vs evolved: `6/2/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.338929 | 0.000000 | 0.055688 | - | - |
| random perturbation | repair-covered tasks | 0.283241 | -0.055688 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.401071 | 0.062143 | 0.117830 | 7/1/0 | 7/1/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.428 | 0.504 | 0.447 |
| random | 11 | 1.00 | 0.388 | 0.408 | 0.393 |
| trajectory_selected | 11 | 2.00 | 0.431 | 0.459 | 0.438 |
| repair_selected | 8 | 3.75 | 0.401 | 0.625 | 0.457 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.339 | 0.659 | 0.419 |
| planning | random | 8 | 1.00 | 0.283 | 0.527 | 0.344 |
| planning | trajectory_selected | 8 | 2.00 | 0.343 | 0.597 | 0.406 |
| planning | repair_selected | 8 | 3.75 | 0.401 | 0.625 | 0.457 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_177 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.405 | 0.303 | 372 | True | 8 | 0.682 | True | True | 13.000 | 0.406 | 0.409 | 0.409 |
| llada-moe-7b-a1b-instruct-hf | plan_178 | random_32 | True | denoise_phase_repairable | False |  | 0.404 | 0.324 | 399 | True | 3 | 0.800 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_179 | random_32 | True | denoise_phase_repairable | False |  | 0.218 | 0.138 | 131 | True | 8 | 0.529 | True | True | 16.000 | 0.500 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_180 | random_32 | True | denoise_phase_repairable | False |  | 0.098 | 0.057 | 65 | True | 7 | 0.533 | True | True | 23.000 | 0.719 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_181 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 372 | True | 7 | 0.737 | True | True | 11.000 | 0.344 | 0.421 | 0.421 |
| llada-moe-7b-a1b-instruct-hf | plan_182 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.370 | 0.310 | 363 | True | 3 | 0.750 | True | True | 6.000 | 0.188 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_183 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.349 | 0.269 | 378 | True | 2 | 0.833 | True | True | 6.000 | 0.188 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_184 | random_32 | False | outside_repairable_band | False |  | 0.141 | 0.121 | 66 | True | 9 | 0.357 | False | False | none | none | none | 0.357 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 7 | 4 | low_confidence_32,random_32 | final | 30.0 | 1.000 | 0.000 | 0.000 | 0.034 | 0.034 | 0.062 | 0.065 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/3/0 | 0.368 | 0.599 | 0.426 |
| history_prefix_25_repair | 7 | 2 | low_confidence_32,random_32 | history | 48.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/1/2 | 0.307 | 0.639 | 0.390 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_177 | False | low_confidence_32 | 3.162 | 0.800 | 1.000 | 0.000 | 0.364 | False | This ensures the system rejects invalid history-prefix waste without sacrificing recall... |
| llada-moe-7b-a1b-instruct-hf | plan_178 | True | random_32 | 1.370 | 0.821 | 1.000 | 0.000 | 0.133 | False | Use this replay evidence to verify plan consistency, execution stability, and result co... |
| llada-moe-7b-a1b-instruct-hf | plan_178 | True | random_32 | 2.040 | 0.689 | 1.000 | 0.000 | 0.200 | False | Specifically, include multiple iterations of plan execution, minimal delta in planning... |
| llada-moe-7b-a1b-instruct-hf | plan_179 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Audit the history-prefix candidate's source-relative lift against actual task performan... |
| llada-moe-7b-a1b-instruct-hf | plan_180 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Plan the candidate-source routing table for the next replay gate. |
| llada-moe-7b-a1b-instruct-hf | plan_181 | True | low_confidence_32 | 1.759 | 0.135 | 1.000 | 0.000 | 0.421 | False | Identify the cost at which the policy achieves an optimal balance between fewer selecti... |
| llada-moe-7b-a1b-instruct-hf | plan_182 | True | low_confidence_32 | 1.952 | 1.000 | 1.000 | 0.000 | 0.250 | False | If there are no generated positives, the system should check for availability availabil... |
| llada-moe-7b-a1b-instruct-hf | plan_182 | True | low_confidence_32 | 2.170 | 1.000 | 1.000 | 0.000 | 0.333 | False | If there are no generated positives and there is no availability, it indicates that the... |
| llada-moe-7b-a1b-instruct-hf | plan_183 | True | low_confidence_32 | 1.255 | 0.700 | 1.000 | 0.000 | 0.583 | False | This allows you to integrate the replay selector into the runner pipeline, ensuring it... |
| llada-moe-7b-a1b-instruct-hf | plan_183 | True | low_confidence_32 | 1.927 | 0.581 | 1.000 | 0.000 | 0.667 | False | Implementation boundary: Implement the replay selector as a runner hook before altering... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_177 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.440 | 0.000 | 0.062 | 0.062 | 0.405 | 0.405 | 0.405 | 0.000 | 0.427 | 0.021 | 0.427 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_178 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.497 | 0.000 | 0.214 | 0.214 | 0.436 | 0.404 | 0.436 | 0.000 | 0.560 | 0.124 | 0.560 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_179 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.404 | 0.000 | 0.130 | 0.130 | 0.302 | 0.218 | 0.302 | 0.000 | 0.344 | 0.041 | 0.344 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_180 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.255 | 0.000 | 0.000 | 0.000 | 0.085 | 0.098 | 0.098 | 0.000 | 0.098 | 0.000 | 0.098 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_181 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.383 | 0.000 | 0.065 | 0.065 | 0.280 | 0.280 | 0.280 | 0.000 | 0.323 | 0.042 | 0.323 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_182 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.475 | 0.000 | 0.160 | 0.160 | 0.370 | 0.370 | 0.390 | 0.000 | 0.499 | 0.109 | 0.499 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_183 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.420 | 0.000 | 0.173 | 0.173 | 0.349 | 0.349 | 0.349 | 0.000 | 0.475 | 0.126 | 0.475 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_184 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.520 | 0.000 | 0.000 | 0.000 | 0.483 | 0.141 | 0.483 | 0.000 | 0.483 | 0.000 | 0.483 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
