# Diffusion Schedule-Selection Benchmark Report

Full model generations: `41`
Arm selections: `73`
Run ID: `diffusion-106f05c6dd5532ee`
Content hash: `106f05c6dd5532eea09b80ceb30da2adef517b078aa230130a5c3800dd5b20cc`
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
History mutability: `monotonic 41/41, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span_phase_final_preserve_seeded_gated`
Repair source policy: `fixed`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `trajectory_relative_decomposed_spend`
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
Repair selector: `transfer_promotion_value`
Repair promotion margin: `0.000`
Trajectory task delta vs fixed: `0.009`
Trajectory task delta vs random: `0.061`
Trajectory wins/ties/losses vs fixed: `2/17/0`
Trajectory wins/ties/losses vs random: `10/9/0`
Oracle generation budget/task: `2.16`
Oracle task score: `0.433`
Oracle headroom vs trajectory: `0.010`
Oracle wins/ties/losses vs trajectory: `3/16/0`
Selector regret vs trajectory: `0.010 over 3/19 improvable`
Repair arm coverage: `16/19` overall
Repair eligible coverage: `16/17`
Repair task delta vs fixed: `0.022`
Repair task delta vs random: `0.084`
Repair task delta vs trajectory: `0.011`
Repair task delta vs evolved: `0.011`
Repair generation budget delta vs evolved: `0.19`
Repair task delta per extra generation vs evolved: `0.060`
Repair wins/ties/losses vs evolved: `3/13/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/16/0`
Selector regret vs repair: `0.000 over 0/16 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `16/19` overall, `16/17` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.366719 | 0.000000 | 0.061661 | - | - |
| random perturbation | repair-covered tasks | 0.305058 | -0.061661 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.388638 | 0.021920 | 0.083580 | 5/11/0 | 11/5/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 19 | 1.00 | 0.414 | 0.570 | 0.453 |
| random | 19 | 1.00 | 0.362 | 0.499 | 0.396 |
| trajectory_selected | 19 | 2.00 | 0.423 | 0.570 | 0.460 |
| repair_selected | 16 | 2.19 | 0.389 | 0.667 | 0.458 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 16 | 1.00 | 0.367 | 0.659 | 0.440 |
| planning | random | 16 | 1.00 | 0.305 | 0.575 | 0.373 |
| planning | trajectory_selected | 16 | 2.00 | 0.377 | 0.659 | 0.448 |
| planning | repair_selected | 16 | 2.19 | 0.389 | 0.667 | 0.458 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | False | outside_repairable_band | 0.356 | 0.256 | 352 | True | 10 | 0.471 | False | True | 17.000 | 0.531 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | False | value_proxy_source_quality_high | 0.393 | 0.333 | 327 | True | 7 | 0.562 | True | True | 15.000 | 0.469 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | False | outside_repairable_band | 0.336 | 0.239 | 329 | True | 12 | 0.294 | False | False | none | none | none | 0.294 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | True | trajectory_relative_decomposed_spend | 0.295 | 0.235 | 309 | True | 8 | 0.529 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_013 | low_confidence_32 | False | outside_repairable_band | 0.304 | 0.244 | 348 | True | 10 | 0.444 | False | True | 32.000 | 1.000 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_014 | low_confidence_32 | False | outside_repairable_band | 0.303 | 0.223 | 329 | True | 10 | 0.412 | False | True | 25.000 | 0.781 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_015 | low_confidence_32 | False | outside_repairable_band | 0.453 | 0.335 | 308 | True | 10 | 0.333 | False | False | none | none | none | 0.333 |
| llada-moe-7b-a1b-instruct-hf | plan_016 | low_confidence_32 | False | outside_repairable_band | 0.241 | 0.201 | 288 | True | 12 | 0.250 | False | False | none | none | none | 0.250 |
| llada-moe-7b-a1b-instruct-hf | plan_017 | low_confidence_32 | False | outside_repairable_band | 0.466 | 0.366 | 364 | True | 10 | 0.444 | False | True | 23.000 | 0.719 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_018 | low_confidence_32 | True | trajectory_relative_decomposed_spend | 0.348 | 0.248 | 348 | True | 8 | 0.556 | True | True | 18.000 | 0.562 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_019 | low_confidence_32 | False | outside_repairable_band | 0.413 | 0.333 | 360 | True | 12 | 0.235 | False | False | none | none | none | 0.235 |
| llada-moe-7b-a1b-instruct-hf | plan_020 | low_confidence_32 | False | transfer_source_task_score_low | 0.260 | 0.180 | 339 | True | 6 | 0.625 | True | True | 11.000 | 0.344 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_021 | low_confidence_32 | True | trajectory_relative_decomposed_spend | 0.356 | 0.256 | 317 | True | 6 | 0.600 | True | True | 14.000 | 0.438 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_022 | low_confidence_32 | False | outside_repairable_band | 0.446 | 0.346 | 363 | True | 12 | 0.400 | False | True | 18.000 | 0.562 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_023 | low_confidence_32 | False | value_proxy_source_quality_high | 0.485 | 0.320 | 341 | True | 7 | 0.562 | True | True | 8.000 | 0.250 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_024 | low_confidence_32 | False | value_proxy_source_quality_high | 0.411 | 0.331 | 348 | True | 7 | 0.588 | True | True | 22.000 | 0.688 | 0.412 | 0.412 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 3 | 3 | low_confidence_32 | final | 23.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.060 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/0/0 | 0.394 | 0.699 | 0.470 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_012 | True | low_confidence_32 | 1.282 | 0.620 | 1.000 | 0.000 | 0.176 | False | Measure the accuracy of multi-step answers for both groups. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | True | low_confidence_32 | 1.413 | 0.887 | 1.000 | 0.000 | 0.176 | False | If group B has significantly worse answers, revert the compression. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | True | low_confidence_32 | 2.000 | 0.595 | 1.000 | 0.000 | 0.294 | False | If group B has significantly better answers, keep the compression for the next release. |
| llada-moe-7b-a1b-instruct-hf | plan_018 | True | low_confidence_32 | 3.265 | 0.785 | 1.000 | 0.000 | 0.056 | False | Compare the results of both experiments to determine which approach is more effective i... |
| llada-moe-7b-a1b-instruct-hf | plan_021 | True | low_confidence_32 | 2.840 | 0.925 | 1.000 | 0.000 | 0.067 | False | Document the process to build trust in the judge's consistency. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.361 | 0.000 | 0.361 | 0.000 | 0.356 | 0.356 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.386 | 0.000 | 0.386 | 0.000 | 0.393 | 0.393 | 0.393 | 0.000 | 0.393 | 0.000 | 0.393 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.261 | 0.000 | 0.261 | 0.000 | 0.336 | 0.296 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_transfer_promotion_value_score_repair_pool | low_confidence_32 | final |  | 0.291 | 0.000 | 0.404 | 0.113 | 0.295 | 0.257 | 0.295 | 0.000 | 0.315 | 0.020 | 0.315 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_013 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.345 | 0.000 | 0.345 | 0.000 | 0.304 | 0.157 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_014 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.329 | 0.000 | 0.329 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_015 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.457 | 0.000 | 0.457 | 0.000 | 0.453 | 0.453 | 0.558 | 0.000 | 0.558 | 0.000 | 0.558 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_016 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.248 | 0.000 | 0.248 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_017 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.391 | 0.000 | 0.391 | 0.000 | 0.466 | 0.303 | 0.466 | 0.000 | 0.466 | 0.000 | 0.466 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_018 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_transfer_promotion_value_score_repair_pool | low_confidence_32 | final |  | 0.329 | 0.000 | 0.437 | 0.108 | 0.348 | 0.207 | 0.348 | 0.000 | 0.438 | 0.090 | 0.438 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_019 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.332 | 0.000 | 0.332 | 0.000 | 0.413 | 0.413 | 0.413 | 0.000 | 0.413 | 0.000 | 0.413 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_020 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.372 | 0.000 | 0.372 | 0.000 | 0.260 | 0.260 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_021 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_transfer_promotion_value_score_repair_pool | low_confidence_32 | final |  | 0.388 | 0.000 | 0.499 | 0.111 | 0.356 | 0.356 | 0.356 | 0.000 | 0.428 | 0.071 | 0.428 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_022 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.340 | 0.000 | 0.340 | 0.000 | 0.446 | 0.383 | 0.446 | 0.000 | 0.446 | 0.000 | 0.446 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_023 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.425 | 0.000 | 0.425 | 0.000 | 0.485 | 0.283 | 0.485 | 0.000 | 0.485 | 0.000 | 0.485 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_024 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_trajectory_relative_decomposed_spend |  |  |  | 0.422 | 0.000 | 0.422 | 0.000 | 0.411 | 0.218 | 0.411 | 0.000 | 0.411 | 0.000 | 0.411 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
