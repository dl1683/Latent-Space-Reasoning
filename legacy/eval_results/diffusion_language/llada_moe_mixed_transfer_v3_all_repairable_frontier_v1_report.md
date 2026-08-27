# Diffusion Schedule-Selection Benchmark Report

Full model generations: `45`
Arm selections: `73`
Run ID: `diffusion-db9cf6afb7c371ab`
Content hash: `db9cf6afb7c371abe7a829532fc727ce6b716ca537a64ff9ab98dd394750819e`
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
History mutability: `monotonic 45/45, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Repair selector: `planning_quality_seed_realization_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.009`
Trajectory task delta vs random: `0.061`
Trajectory wins/ties/losses vs fixed: `2/17/0`
Trajectory wins/ties/losses vs random: `10/9/0`
Oracle generation budget/task: `2.37`
Oracle task score: `0.433`
Oracle headroom vs trajectory: `0.010`
Oracle wins/ties/losses vs trajectory: `3/16/0`
Selector regret vs trajectory: `0.010 over 3/19 improvable`
Repair arm coverage: `16/19` overall
Repair eligible coverage: `16/17`
Repair task delta vs fixed: `0.021`
Repair task delta vs random: `0.082`
Repair task delta vs trajectory: `0.010`
Repair task delta vs evolved: `0.010`
Repair generation budget delta vs evolved: `0.44`
Repair task delta per extra generation vs evolved: `0.023`
Repair wins/ties/losses vs evolved: `2/14/0`
Oracle headroom vs repair: `0.001`
Oracle wins/ties/losses vs repair: `1/15/0`
Selector regret vs repair: `0.001 over 1/16 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `16/19` overall, `16/17` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.366719 | 0.000000 | 0.061661 | - | - |
| random perturbation | repair-covered tasks | 0.305058 | -0.061661 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.387388 | 0.020670 | 0.082330 | 4/12/0 | 11/5/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 19 | 1.00 | 0.414 | 0.570 | 0.453 |
| random | 19 | 1.00 | 0.362 | 0.499 | 0.396 |
| trajectory_selected | 19 | 2.00 | 0.423 | 0.570 | 0.460 |
| repair_selected | 16 | 2.44 | 0.387 | 0.665 | 0.457 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 16 | 1.00 | 0.367 | 0.659 | 0.440 |
| planning | random | 16 | 1.00 | 0.305 | 0.575 | 0.373 |
| planning | trajectory_selected | 16 | 2.00 | 0.377 | 0.659 | 0.448 |
| planning | repair_selected | 16 | 2.44 | 0.387 | 0.665 | 0.457 |
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
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | True | denoise_phase_repairable | 0.393 | 0.333 | 327 | True | 7 | 0.562 | True | True | 15.000 | 0.469 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | False | outside_repairable_band | 0.336 | 0.239 | 329 | True | 12 | 0.294 | False | False | none | none | none | 0.294 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | True | denoise_phase_repairable | 0.295 | 0.235 | 309 | True | 8 | 0.529 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_013 | low_confidence_32 | False | outside_repairable_band | 0.304 | 0.244 | 348 | True | 10 | 0.444 | False | True | 32.000 | 1.000 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_014 | low_confidence_32 | False | outside_repairable_band | 0.303 | 0.223 | 329 | True | 10 | 0.412 | False | True | 25.000 | 0.781 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_015 | low_confidence_32 | False | outside_repairable_band | 0.453 | 0.335 | 308 | True | 10 | 0.333 | False | False | none | none | none | 0.333 |
| llada-moe-7b-a1b-instruct-hf | plan_016 | low_confidence_32 | False | outside_repairable_band | 0.241 | 0.201 | 288 | True | 12 | 0.250 | False | False | none | none | none | 0.250 |
| llada-moe-7b-a1b-instruct-hf | plan_017 | low_confidence_32 | False | outside_repairable_band | 0.466 | 0.366 | 364 | True | 10 | 0.444 | False | True | 23.000 | 0.719 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_018 | low_confidence_32 | True | denoise_phase_repairable | 0.348 | 0.248 | 348 | True | 8 | 0.556 | True | True | 18.000 | 0.562 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_019 | low_confidence_32 | False | outside_repairable_band | 0.413 | 0.333 | 360 | True | 12 | 0.235 | False | False | none | none | none | 0.235 |
| llada-moe-7b-a1b-instruct-hf | plan_020 | low_confidence_32 | True | denoise_phase_repairable | 0.260 | 0.180 | 339 | True | 6 | 0.625 | True | True | 11.000 | 0.344 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_021 | low_confidence_32 | True | denoise_phase_repairable | 0.356 | 0.256 | 317 | True | 6 | 0.600 | True | True | 14.000 | 0.438 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_022 | low_confidence_32 | False | outside_repairable_band | 0.446 | 0.346 | 363 | True | 12 | 0.400 | False | True | 18.000 | 0.562 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_023 | low_confidence_32 | True | denoise_phase_repairable | 0.485 | 0.320 | 341 | True | 7 | 0.562 | True | True | 8.000 | 0.250 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_024 | low_confidence_32 | True | denoise_phase_repairable | 0.411 | 0.331 | 348 | True | 7 | 0.588 | True | True | 22.000 | 0.688 | 0.412 | 0.412 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 7 | 2 | low_confidence_32 | final | 31.3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/1/2 | 0.377 | 0.693 | 0.456 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_010 | False | low_confidence_32 | 1.441 | 0.963 | 1.000 | 0.000 | 0.250 | False | If the gain is consistent across runs runs with the same seed and order, it is likely r... |
| llada-moe-7b-a1b-instruct-hf | plan_010 | False | low_confidence_32 | 2.205 | 1.000 | 1.000 | 0.000 | 0.188 | False | If not, investigate the the impact of the random seed and test order. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 1.282 | 0.620 | 1.000 | 0.000 | 0.176 | False | Measure the accuracy of multi-step answers for both groups. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 1.413 | 0.887 | 1.000 | 0.000 | 0.176 | False | If group B has significantly worse answers, revert the compression. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 2.000 | 0.595 | 1.000 | 0.000 | 0.294 | False | If group B has significantly better answers, keep the compression for the next release. |
| llada-moe-7b-a1b-instruct-hf | plan_018 | True | low_confidence_32 | 3.265 | 0.785 | 1.000 | 0.000 | 0.056 | False | Compare the results of both experiments to determine which approach is more effective i... |
| llada-moe-7b-a1b-instruct-hf | plan_020 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Use a-validation set to evaluate performance across both families. |
| llada-moe-7b-a1b-instruct-hf | plan_020 | False | low_confidence_32 | 2.079 | 0.925 | 1.000 | 0.000 | 0.062 | False | Analyze the results to ensure the schedule performs well on both domains. |
| llada-moe-7b-a1b-instruct-hf | plan_020 | False | low_confidence_32 | 2.095 | 0.762 | 1.000 | 0.000 | 0.188 | False | Adjust the parameters of the schedule accordingly to avoid overfitting to one family. |
| llada-moe-7b-a1b-instruct-hf | plan_021 | True | low_confidence_32 | 2.840 | 0.925 | 1.000 | 0.000 | 0.067 | False | Document the process to build trust in the judge's consistency. |
| llada-moe-7b-a1b-instruct-hf | plan_023 | False | low_confidence_32 | 2.136 | 1.000 | 1.000 | 0.000 | 0.000 | False | Track total GPU time used and final accuracy. |
| llada-moe-7b-a1b-instruct-hf | plan_023 | False | low_confidence_32 | 2.065 | 0.735 | 1.000 | 0.000 | 0.188 | False | Compare efficiency (e.g., time per repair) and accuracy gain to determine which model o... |
| llada-moe-7b-a1b-instruct-hf | plan_024 | False | low_confidence_32 | 2.062 | 0.869 | 1.000 | 0.000 | 0.059 | False | Monitor for any unintended inversions or harmful refusals. |
| llada-moe-7b-a1b-instruct-hf | plan_024 | False | low_confidence_32 | 1.857 | 0.291 | 1.000 | 0.000 | 0.294 | False | Adjust the model’s confidence threshold or refusal rule sensitivity based on the feedba... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.361 | 0.000 | 0.256 | 0.000 | 0.356 | 0.356 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.386 | 0.000 | 0.389 | 0.000 | 0.393 | 0.393 | 0.393 | 0.000 | 0.393 | 0.000 | 0.393 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.261 | 0.000 | 0.239 | 0.000 | 0.336 | 0.296 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.291 | 0.000 | 0.235 | 0.000 | 0.295 | 0.257 | 0.295 | 0.000 | 0.295 | 0.000 | 0.315 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | plan_013 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.345 | 0.000 | 0.244 | 0.000 | 0.304 | 0.157 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_014 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.329 | 0.000 | 0.223 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_015 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.457 | 0.000 | 0.505 | 0.000 | 0.453 | 0.453 | 0.558 | 0.000 | 0.558 | 0.000 | 0.558 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_016 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.248 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_017 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.391 | 0.000 | 0.410 | 0.000 | 0.466 | 0.303 | 0.466 | 0.000 | 0.466 | 0.000 | 0.466 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_018 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.329 | 0.000 | 0.419 | 0.171 | 0.348 | 0.207 | 0.348 | 0.000 | 0.438 | 0.090 | 0.438 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_019 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.332 | 0.000 | 0.356 | 0.000 | 0.413 | 0.413 | 0.413 | 0.000 | 0.413 | 0.000 | 0.413 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_020 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.372 | 0.000 | 0.244 | 0.000 | 0.260 | 0.260 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_021 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.388 | 0.000 | 0.401 | 0.145 | 0.356 | 0.356 | 0.356 | 0.000 | 0.428 | 0.071 | 0.428 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_022 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.340 | 0.000 | 0.386 | 0.000 | 0.446 | 0.383 | 0.446 | 0.000 | 0.446 | 0.000 | 0.446 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_023 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.425 | 0.000 | 0.377 | 0.000 | 0.485 | 0.283 | 0.485 | 0.000 | 0.485 | 0.000 | 0.485 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_024 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.422 | 0.000 | 0.390 | 0.000 | 0.411 | 0.218 | 0.411 | 0.000 | 0.411 | 0.000 | 0.411 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
