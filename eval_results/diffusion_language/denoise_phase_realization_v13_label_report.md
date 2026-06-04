# Diffusion Schedule-Selection Benchmark Report

Full model generations: `27`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-567d84d21ee7c484`
Content hash: `567d84d21ee7c4848ec4584cea24c99e5e757e9e57e6ae0bf414408baeb213c7`
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
History mutability: `monotonic 27/27, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs random: `0.006`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `1/9/1`
Oracle generation budget/task: `2.45`
Oracle task score: `0.445`
Oracle headroom vs trajectory: `0.040`
Oracle wins/ties/losses vs trajectory: `4/7/0`
Selector regret vs trajectory: `0.040 over 4/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.040`
Repair task delta vs random: `0.048`
Repair task delta vs trajectory: `0.040`
Repair task delta vs evolved: `0.040`
Repair generation budget delta vs evolved: `0.62`
Repair task delta per extra generation vs evolved: `0.064`
Repair wins/ties/losses vs evolved: `2/6/0`
Oracle headroom vs repair: `0.015`
Oracle wins/ties/losses vs repair: `2/6/0`
Selector regret vs repair: `0.015 over 2/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.307054 | 0.000000 | 0.008304 | - | - |
| random perturbation | repair-covered tasks | 0.298750 | -0.008304 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.347143 | 0.040089 | 0.048393 | 2/6/0 | 3/4/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.405 | 0.504 | 0.430 |
| random | 11 | 1.00 | 0.399 | 0.465 | 0.416 |
| trajectory_selected | 11 | 2.00 | 0.405 | 0.504 | 0.430 |
| repair_selected | 8 | 2.62 | 0.347 | 0.666 | 0.427 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.307 | 0.659 | 0.395 |
| planning | random | 8 | 1.00 | 0.299 | 0.605 | 0.375 |
| planning | trajectory_selected | 8 | 2.00 | 0.307 | 0.659 | 0.395 |
| planning | repair_selected | 8 | 2.62 | 0.347 | 0.666 | 0.427 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_097 | random_32 | False | outside_repairable_band | False |  | 0.453 | 0.353 | 205 | True | 10 | 0.375 | False | False | none | none | none | 0.375 |
| llada-moe-7b-a1b-instruct-hf | plan_098 | random_32 | True | denoise_phase_repairable | False |  | 0.105 | 0.045 | 87 | True | 6 | 0.538 | True | True | 24.000 | 0.750 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_099 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.356 | 0.256 | 374 | True | 4 | 0.786 | True | True | 6.000 | 0.188 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_100 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.325 | 0.265 | 323 | True | 2 | 0.875 | True | True | 10.000 | 0.312 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_101 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 262 | True | 5 | 0.765 | True | True | 16.000 | 0.500 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_102 | low_confidence_32 | False | outside_repairable_band | False |  | 0.348 | 0.247 | 340 | True | 0 | 1.000 | False | True | 6.000 | 0.188 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_103 | low_confidence_32 | False | outside_repairable_band | False |  | 0.349 | 0.269 | 338 | True | 0 | 1.000 | False | True | 8.000 | 0.250 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_104 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.213 | 0.193 | 354 | True | 7 | 0.533 | True | True | 12.000 | 0.375 | 0.467 | 0.467 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 5 | 2 | low_confidence_32,random_32 | final | 29.2 | 0.800 | 0.200 | 0.000 | 0.048 | 0.048 | 0.064 | 0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/3/0 | 0.312 | 0.626 | 0.391 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_098 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | sourceAlignment:high, trajectoryQuality:moderate, denoiseModerate, promptCoverage:weak, |
| llada-moe-7b-a1b-instruct-hf | plan_099 | True | low_confidence_32 | 1.432 | 0.968 | 1.000 | 0.000 | 0.214 | False | Conduct the audit to review the selector's logic, evaluation rules, and potential biase... |
| llada-moe-7b-a1b-instruct-hf | plan_099 | True | low_confidence_32 | 1.685 | 0.000 | 1.000 | 0.000 | 0.500 | False | The goal is to identify and understand the reasons behind the selector's rejection of t... |
| llada-moe-7b-a1b-instruct-hf | plan_100 | False | low_confidence_32 | 1.925 | 1.000 | 1.000 | 0.000 | 0.375 | False | The test should evaluate whether phase evidence (from the deno phase) can be used to ov... |
| llada-moe-7b-a1b-instruct-hf | plan_100 | False | low_confidence_32 | 2.677 | 1.000 | 1.000 | 0.000 | 0.375 | False | The goal is to determine whether phase evidence should override the static gate. |
| llada-moe-7b-a1b-instruct-hf | plan_101 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | / Signal / Load-Bearing / Decision / /--------/-------------/--------/ / High-Value Fea... |
| llada-moe-7b-a1b-instruct-hf | plan_104 | True | low_confidence_32 | 2.089 | 0.780 | 1.000 | 0.000 | 0.267 | False | This boundary should be clearly enough to allow for external verification and the progr... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_097 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.398 | 0.000 | 0.244 | 0.000 | 0.344 | 0.453 | 0.344 | 0.000 | 0.344 | 0.000 | 0.453 | 0.109 |
| llada-moe-7b-a1b-instruct-hf | plan_098 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.401 | 0.000 | 0.180 | 0.000 | 0.280 | 0.105 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_099 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.443 | 0.000 | 0.474 | 0.218 | 0.356 | 0.356 | 0.356 | 0.000 | 0.495 | 0.139 | 0.495 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_100 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.432 | 0.000 | 0.265 | 0.000 | 0.325 | 0.325 | 0.325 | 0.000 | 0.325 | 0.000 | 0.325 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_101 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.356 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_102 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.496 | 0.000 | 0.247 | 0.000 | 0.348 | 0.348 | 0.348 | 0.000 | 0.348 | 0.000 | 0.348 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_103 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.455 | 0.000 | 0.269 | 0.000 | 0.349 | 0.349 | 0.349 | 0.000 | 0.349 | 0.000 | 0.358 | 0.009 |
| llada-moe-7b-a1b-instruct-hf | plan_104 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.344 | 0.000 | 0.428 | 0.235 | 0.213 | 0.213 | 0.213 | 0.000 | 0.394 | 0.182 | 0.394 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
