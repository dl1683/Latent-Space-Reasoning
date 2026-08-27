# Diffusion Schedule-Selection Benchmark Report

Full model generations: `25`
Arm selections: `41`
Run ID: `diffusion-4e697187918ba007`
Content hash: `4e697187918ba007011e77009f1d4872ee8798729a4a45908373e612966601da`
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
History mutability: `monotonic 25/25, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.008`
Trajectory task delta vs random: `0.054`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `2.27`
Oracle task score: `0.506`
Oracle headroom vs trajectory: `0.010`
Oracle wins/ties/losses vs trajectory: `4/7/0`
Selector regret vs trajectory: `0.010 over 4/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.018`
Repair task delta vs random: `0.082`
Repair task delta vs trajectory: `0.007`
Repair task delta vs evolved: `0.007`
Repair generation budget delta vs evolved: `0.38`
Repair task delta per extra generation vs evolved: `0.020`
Repair wins/ties/losses vs evolved: `2/6/0`
Oracle headroom vs repair: `0.007`
Oracle wins/ties/losses vs repair: `2/6/0`
Selector regret vs repair: `0.007 over 2/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.420804 | 0.000000 | 0.064330 | - | - |
| random perturbation | repair-covered tasks | 0.356473 | -0.064330 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.438661 | 0.017857 | 0.082187 | 3/5/0 | 5/3/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.488 | 0.504 | 0.492 |
| random | 11 | 1.00 | 0.441 | 0.441 | 0.441 |
| trajectory_selected | 11 | 2.00 | 0.495 | 0.497 | 0.496 |
| repair_selected | 8 | 2.38 | 0.439 | 0.653 | 0.492 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.421 | 0.659 | 0.480 |
| planning | random | 8 | 1.00 | 0.356 | 0.572 | 0.410 |
| planning | trajectory_selected | 8 | 2.00 | 0.431 | 0.649 | 0.486 |
| planning | repair_selected | 8 | 2.38 | 0.439 | 0.653 | 0.492 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_025 | low_confidence_32 | False | outside_repairable_band | 0.414 | 0.374 | 280 | True | 12 | 0.143 | False | False | none | none | none | 0.143 |
| llada-moe-7b-a1b-instruct-hf | plan_026 | low_confidence_32 | True | denoise_phase_repairable | 0.404 | 0.324 | 343 | True | 9 | 0.471 | True | True | 13.000 | 0.406 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_027 | low_confidence_32 | False | outside_repairable_band | 0.348 | 0.247 | 303 | True | 12 | 0.235 | False | False | none | none | none | 0.235 |
| llada-moe-7b-a1b-instruct-hf | plan_028 | low_confidence_32 | True | denoise_phase_repairable | 0.571 | 0.433 | 385 | True | 5 | 0.688 | True | True | 18.000 | 0.562 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_029 | low_confidence_32 | False | outside_repairable_band | 0.417 | 0.357 | 388 | True | 12 | 0.316 | False | False | none | none | none | 0.316 |
| llada-moe-7b-a1b-instruct-hf | plan_030 | low_confidence_32 | False | outside_repairable_band | 0.366 | 0.266 | 375 | True | 10 | 0.444 | False | True | 14.000 | 0.438 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_031 | low_confidence_32 | True | denoise_phase_repairable | 0.442 | 0.342 | 391 | True | 8 | 0.500 | True | True | 16.000 | 0.500 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_032 | low_confidence_32 | False | outside_repairable_band | 0.405 | 0.345 | 305 | True | 12 | 0.176 | False | False | none | none | none | 0.176 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 3 | 2 | low_confidence_32 | final | 25.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.030 | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/0/0 | 0.495 | 0.676 | 0.540 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_026 | True | low_confidence_32 | 2.129 | 0.860 | 1.000 | 0.000 | 0.294 | False | Request peer review of the judge’s responses and and conduct a blind audit of the judge... |
| llada-moe-7b-a1b-instruct-hf | plan_028 | False | low_confidence_32 | 1.278 | 0.642 | 1.000 | 0.000 | 0.188 | False | Use statistical tests to assess if compressed traces hide uncertainty or missed failures. |
| llada-moe-7b-a1b-instruct-hf | plan_028 | False | low_confidence_32 | 2.031 | 0.624 | 1.000 | 0.000 | 0.125 | False | If significant risks arise, pause deployment and request review before enabling by defa... |
| llada-moe-7b-a1b-instruct-hf | plan_031 | True | low_confidence_32 | 2.832 | 0.917 | 1.000 | 0.000 | 0.000 | False | Use A/B testing and statistical significance analysis to evaluate trade-offs before ado... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_025 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.340 | 0.000 | 0.389 | 0.000 | 0.414 | 0.414 | 0.414 | 0.000 | 0.414 | 0.000 | 0.414 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_026 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.375 | 0.000 | 0.403 | 0.032 | 0.404 | 0.404 | 0.404 | 0.000 | 0.441 | 0.037 | 0.441 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_027 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.210 | 0.000 | 0.247 | 0.000 | 0.348 | 0.348 | 0.348 | 0.000 | 0.348 | 0.000 | 0.348 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_028 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.480 | 0.000 | 0.502 | 0.000 | 0.571 | 0.515 | 0.571 | 0.000 | 0.571 | 0.000 | 0.581 | 0.010 |
| llada-moe-7b-a1b-instruct-hf | plan_029 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.353 | 0.000 | 0.388 | 0.000 | 0.417 | 0.417 | 0.417 | 0.000 | 0.417 | 0.000 | 0.462 | 0.045 |
| llada-moe-7b-a1b-instruct-hf | plan_030 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.356 | 0.000 | 0.266 | 0.000 | 0.366 | 0.045 | 0.366 | 0.000 | 0.366 | 0.000 | 0.366 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_031 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.378 | 0.000 | 0.420 | 0.028 | 0.442 | 0.304 | 0.442 | 0.000 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_032 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.369 | 0.000 | 0.429 | 0.000 | 0.405 | 0.405 | 0.489 | 0.000 | 0.489 | 0.000 | 0.489 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
