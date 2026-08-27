# Diffusion Schedule-Selection Benchmark Report

Full model generations: `28`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-7c7da63c19349927`
Content hash: `7c7da63c193499278510ff802bac4f12b439cfd8aaea8b1ac35d0dd0f69e3104`
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
History mutability: `monotonic 28/28, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.004`
Trajectory task delta vs random: `0.020`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `2.55`
Oracle task score: `0.437`
Oracle headroom vs trajectory: `0.013`
Oracle wins/ties/losses vs trajectory: `3/8/0`
Selector regret vs trajectory: `0.013 over 3/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.022`
Repair task delta vs random: `0.045`
Repair task delta vs trajectory: `0.017`
Repair task delta vs evolved: `0.017`
Repair generation budget delta vs evolved: `0.75`
Repair task delta per extra generation vs evolved: `0.023`
Repair wins/ties/losses vs evolved: `3/5/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.328929 | 0.000000 | 0.022938 | - | - |
| random perturbation | repair-covered tasks | 0.305991 | -0.022938 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.351116 | 0.022187 | 0.045125 | 3/5/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.421 | 0.504 | 0.442 |
| random | 11 | 1.00 | 0.404 | 0.454 | 0.417 |
| trajectory_selected | 11 | 2.00 | 0.425 | 0.497 | 0.443 |
| repair_selected | 8 | 2.75 | 0.351 | 0.674 | 0.432 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.329 | 0.659 | 0.412 |
| planning | random | 8 | 1.00 | 0.306 | 0.590 | 0.377 |
| planning | trajectory_selected | 8 | 2.00 | 0.334 | 0.649 | 0.413 |
| planning | repair_selected | 8 | 2.75 | 0.351 | 0.674 | 0.432 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_129 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.415 | 0.315 | 357 | True | 2 | 0.846 | True | True | 9.000 | 0.281 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_130 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.451 | 0.351 | 384 | True | 3 | 0.800 | True | True | 13.000 | 0.406 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_131 | random_32 | True | denoise_phase_repairable | False |  | 0.335 | 0.235 | 228 | True | 6 | 0.647 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_132 | low_confidence_32 | False | outside_repairable_band | False |  | 0.305 | 0.205 | 344 | True | 1 | 0.938 | False | True | 5.000 | 0.156 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_133 | random_32 | True | denoise_phase_repairable | False |  | 0.157 | 0.117 | 111 | True | 6 | 0.600 | True | True | 21.000 | 0.656 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_134 | random_32 | False | late_repairable_denoise_skeleton | False |  | 0.304 | 0.244 | 258 | True | 6 | 0.455 | True | True | 32.000 | 1.000 | 0.455 | 0.455 |
| llada-moe-7b-a1b-instruct-hf | plan_135 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 367 | True | 8 | 0.556 | True | True | 9.000 | 0.281 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_136 | random_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.160 | 182 | True | 8 | 0.417 | True | True | 30.000 | 0.938 | 0.417 | 0.417 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 6 | 3 | low_confidence_32,random_32 | final | 26.8 | 1.000 | 0.000 | 0.000 | 0.020 | 0.020 | 0.023 | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/3/0 | 0.329 | 0.635 | 0.406 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_129 | True | low_confidence_32 | 1.341 | 0.860 | 1.000 | 0.000 | 0.462 | False | If the repair eliminates the advantage, proceed to the next gate with either the repair... |
| llada-moe-7b-a1b-instruct-hf | plan_129 | True | low_confidence_32 | 1.897 | 0.434 | 1.000 | 0.000 | 0.462 | False | Ensure the next gate reflects the updated state before claiming source-search value. |
| llada-moe-7b-a1b-instruct-hf | plan_130 | True | low_confidence_32 | 2.132 | 0.893 | 1.000 | 0.000 | 0.267 | False | Compare the results to determine whether preservation or generation is the bottleneck. |
| llada-moe-7b-a1b-instruct-hf | plan_131 | True | random_32 | 2.084 | 0.742 | 1.000 | 0.000 | 0.176 | False | Once the audit is complete, you can change the selector based on the policy audit findi... |
| llada-moe-7b-a1b-instruct-hf | plan_133 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The replay and label gates ensure that a source-positive row is unreliable unless repea... |
| llada-moe-7b-a1b-instruct-hf | plan_135 | False | low_confidence_32 | 1.449 | 1.000 | 1.000 | 0.000 | 0.167 | False | Use complementary metrics to task value, such as task accuracy or user feedback, to det... |
| llada-moe-7b-a1b-instruct-hf | plan_135 | False | low_confidence_32 | 2.202 | 1.000 | 1.000 | 0.000 | 0.222 | False | Implement cross-validation and sensitivity analysis to to ensure that high span scores... |
| llada-moe-7b-a1b-instruct-hf | plan_136 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Plan evidence packet, including reasoning logs, information preservation criteria, and... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_129 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.485 | 0.000 | 0.445 | 0.045 | 0.415 | 0.415 | 0.415 | 0.000 | 0.453 | 0.037 | 0.453 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_130 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.477 | 0.000 | 0.481 | 0.050 | 0.451 | 0.451 | 0.451 | 0.000 | 0.501 | 0.050 | 0.501 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_131 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | random_32 | final |  | 0.381 | 0.000 | 0.285 | 0.050 | 0.295 | 0.335 | 0.335 | 0.000 | 0.385 | 0.050 | 0.385 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_132 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.411 | 0.000 | 0.205 | 0.000 | 0.305 | 0.305 | 0.305 | 0.000 | 0.305 | 0.000 | 0.305 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_133 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.403 | 0.000 | 0.180 | 0.000 | 0.220 | 0.157 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_134 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.402 | 0.000 | 0.373 | 0.000 | 0.399 | 0.304 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_135 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.299 | 0.000 | 0.201 | 0.000 | 0.281 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_136 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.312 | 0.000 | 0.265 | 0.000 | 0.265 | 0.200 | 0.265 | 0.000 | 0.265 | 0.000 | 0.265 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
