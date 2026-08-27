# Diffusion Schedule-Selection Benchmark Report

Full model generations: `29`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-1b2da43bad7a69b0`
Content hash: `1b2da43bad7a69b0eb61bb5f6e6bee7c284ca9d735a7e790e62a70644398cc11`
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
History mutability: `monotonic 29/29, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Repair selector: `candidate_aware_promotion_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.049`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `2.64`
Oracle task score: `0.417`
Oracle headroom vs trajectory: `0.015`
Oracle wins/ties/losses vs trajectory: `2/9/0`
Selector regret vs trajectory: `0.015 over 2/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.018`
Repair task delta vs random: `0.086`
Repair task delta vs trajectory: `0.018`
Repair task delta vs evolved: `0.018`
Repair generation budget delta vs evolved: `0.88`
Repair task delta per extra generation vs evolved: `0.021`
Repair wins/ties/losses vs evolved: `1/7/0`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.002 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.303973 | 0.000000 | 0.067446 | - | - |
| random perturbation | repair-covered tasks | 0.236527 | -0.067446 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.322455 | 0.018482 | 0.085929 | 1/7/0 | 5/3/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.403 | 0.504 | 0.428 |
| random | 11 | 1.00 | 0.354 | 0.423 | 0.371 |
| trajectory_selected | 11 | 2.00 | 0.403 | 0.504 | 0.428 |
| repair_selected | 8 | 2.88 | 0.322 | 0.654 | 0.405 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.304 | 0.659 | 0.393 |
| planning | random | 8 | 1.00 | 0.237 | 0.547 | 0.314 |
| planning | trajectory_selected | 8 | 2.00 | 0.304 | 0.659 | 0.393 |
| planning | repair_selected | 8 | 2.88 | 0.322 | 0.654 | 0.405 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_113 | random_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 348 | True | 4 | 0.800 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_114 | random_32 | True | denoise_phase_repairable | False |  | 0.177 | 0.117 | 130 | True | 8 | 0.529 | True | True | 16.000 | 0.500 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_115 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 352 | True | 3 | 0.824 | True | True | 7.000 | 0.219 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_116 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 225 | True | 8 | 0.467 | True | True | 26.000 | 0.812 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_117 | random_32 | True | denoise_phase_repairable | False |  | 0.197 | 0.117 | 177 | True | 6 | 0.625 | True | True | 15.000 | 0.469 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_118 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 346 | True | 8 | 0.500 | True | True | 12.000 | 0.375 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_119 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 25 | True | 12 | 0.200 | False | False | none | none | none | 0.200 |
| llada-moe-7b-a1b-instruct-hf | plan_120 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.349 | 0.269 | 409 | True | 4 | 0.692 | True | True | 9.000 | 0.281 | 0.538 | 0.538 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 7 | 1 | low_confidence_32,random_32 | final | 23.0 | 0.857 | 0.143 | 0.000 | 0.051 | 0.051 | 0.033 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/4/0 | 0.297 | 0.631 | 0.381 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_113 | False | random_32 | 2.105 | 0.875 | 1.000 | 0.000 | 0.467 | False | That is,: the positives are selected by the static band, or the positives are selected... |
| llada-moe-7b-a1b-instruct-hf | plan_114 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The controller should prioritize spending repair compute as the diagnostic probe indica... |
| llada-moe-7b-a1b-instruct-hf | plan_115 | False | low_confidence_32 | 2.211 | 1.000 | 1.000 | 0.000 | 0.118 | False | This ensures the probe's data is meaningful and not excluded by the static analysis. |
| llada-moe-7b-a1b-instruct-hf | plan_116 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | / Static / Probe / Total / /--------/--------/--------/ / Low / High / Low / / High / L... |
| llada-moe-7b-a1b-instruct-hf | plan_117 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Conclusive criteria include rows where static features and probe features disagree, no... |
| llada-moe-7b-a1b-instruct-hf | plan_118 | False | low_confidence_32 | 2.124 | 1.000 | 1.000 | 0.000 | 0.062 | False | Ensure consistent informat,, conditions, and data collection methods (Prompt vs. |
| llada-moe-7b-a1b-instruct-hf | plan_118 | False | low_confidence_32 | 2.130 | 0.910 | 1.000 | 0.000 | 0.312 | False | Collect) to demonstrate that the probe-conditioned head is not merely prompt coverage. |
| llada-moe-7b-a1b-instruct-hf | plan_120 | True | low_confidence_32 | 2.186 | 0.981 | 1.000 | 0.000 | 0.154 | False | This phase requires rigorous testing, security review, and performance validation to en... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_113 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.427 | 0.000 | 0.294 | 0.000 | 0.374 | 0.261 | 0.374 | 0.000 | 0.374 | 0.000 | 0.374 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_114 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.378 | 0.000 | 0.180 | 0.000 | 0.280 | 0.177 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_115 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.409 | 0.000 | 0.180 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_116 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.309 | 0.000 | 0.201 | 0.000 | 0.301 | 0.301 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_117 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.338 | 0.000 | 0.180 | 0.000 | 0.260 | 0.197 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_118 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.357 | 0.000 | 0.201 | 0.000 | 0.281 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.294 | 0.012 |
| llada-moe-7b-a1b-instruct-hf | plan_119 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.414 | 0.000 | 0.266 | 0.000 | 0.306 | 0.045 | 0.306 | 0.000 | 0.306 | 0.000 | 0.306 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_120 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.396 | 0.000 | 0.501 | 0.232 | 0.349 | 0.349 | 0.349 | 0.000 | 0.497 | 0.148 | 0.497 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
