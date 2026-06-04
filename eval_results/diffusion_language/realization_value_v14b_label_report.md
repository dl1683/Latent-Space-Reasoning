# Diffusion Schedule-Selection Benchmark Report

Full model generations: `27`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-c9a25cfe3d8aa862`
Content hash: `c9a25cfe3d8aa8628c110f6623fb874869af593a7e256309c12d7af6f7fc5725`
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
Trajectory task delta vs fixed: `-0.006`
Trajectory task delta vs random: `0.048`
Trajectory wins/ties/losses vs fixed: `0/10/1`
Trajectory wins/ties/losses vs random: `3/7/1`
Oracle generation budget/task: `2.45`
Oracle task score: `0.458`
Oracle headroom vs trajectory: `0.018`
Oracle wins/ties/losses vs trajectory: `2/9/0`
Selector regret vs trajectory: `0.018 over 2/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.017`
Repair task delta vs random: `0.091`
Repair task delta vs trajectory: `0.025`
Repair task delta vs evolved: `0.025`
Repair generation budget delta vs evolved: `0.62`
Repair task delta per extra generation vs evolved: `0.040`
Repair wins/ties/losses vs evolved: `2/6/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.361696 | 0.000000 | 0.073562 | - | - |
| random perturbation | repair-covered tasks | 0.288134 | -0.073562 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.379063 | 0.017366 | 0.090929 | 1/7/0 | 4/4/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.445 | 0.504 | 0.460 |
| random | 11 | 1.00 | 0.391 | 0.408 | 0.396 |
| trajectory_selected | 11 | 2.00 | 0.439 | 0.496 | 0.453 |
| repair_selected | 8 | 2.62 | 0.379 | 0.664 | 0.450 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.362 | 0.659 | 0.436 |
| planning | random | 8 | 1.00 | 0.288 | 0.527 | 0.348 |
| planning | trajectory_selected | 8 | 2.00 | 0.354 | 0.648 | 0.427 |
| planning | repair_selected | 8 | 2.62 | 0.379 | 0.664 | 0.450 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_105 | random_32 | False | outside_repairable_band | False |  | 0.065 | 0.045 | 48 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_106 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.239 | 427 | True | 8 | 0.562 | True | True | 7.000 | 0.219 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_107 | random_32 | False | outside_repairable_band | False |  | 0.157 | 0.117 | 82 | True | 10 | 0.438 | False | True | 30.000 | 0.938 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_108 | random_32 | True | denoise_phase_repairable | False |  | 0.190 | 0.130 | 144 | True | 4 | 0.769 | True | True | 13.000 | 0.406 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_109 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 375 | True | 5 | 0.643 | True | True | 6.000 | 0.188 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_110 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.468 | 0.408 | 348 | True | 3 | 0.800 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_111 | low_confidence_32 | False | outside_repairable_band | False |  | 0.261 | 0.201 | 397 | True | 0 | 1.000 | False | True | 8.000 | 0.250 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_112 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.524 | 0.389 | 407 | True | 6 | 0.571 | True | True | 29.000 | 0.906 | 0.429 | 0.429 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 5 | 2 | low_confidence_32,random_32 | final | 32.4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/1/2 | 0.377 | 0.645 | 0.444 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_106 | False | low_confidence_32 | 2.193 | 0.968 | 1.000 | 0.000 | 0.125 | False | Additionally, conduct regular maintenance checks to maintain equipment functionality an... |
| llada-moe-7b-a1b-instruct-hf | plan_108 | False | random_32 | 1.685 | 0.000 | 1.000 | 0.000 | 0.615 | False | ensure the generator value is not selected by the repair selector to avoid counting the... |
| llada-moe-7b-a1b-instruct-hf | plan_109 | True | low_confidence_32 | 3.225 | 1.000 | 1.000 | 0.000 | 0.071 | False | This ensures a clear financial of the system's costs and benefits. |
| llada-moe-7b-a1b-instruct-hf | plan_109 | True | low_confidence_32 | 3.967 | 1.000 | 1.000 | 0.000 | 0.071 | False | This requires detailed data on system efficiency, energy consumption, and operational c... |
| llada-moe-7b-a1b-instruct-hf | plan_110 | False | low_confidence_32 | 2.042 | 0.716 | 1.000 | 0.000 | 0.200 | False | This the failure report should include the initial state, the expected state after the... |
| llada-moe-7b-a1b-instruct-hf | plan_112 | True | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Consolidate failed failed gates by analyzing their causes causes causes causes causes c... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_105 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.521 | 0.000 | 0.290 | 0.000 | 0.390 | 0.065 | 0.390 | 0.000 | 0.390 | 0.000 | 0.390 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_106 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.375 | 0.000 | 0.239 | 0.000 | 0.339 | 0.339 | 0.339 | 0.000 | 0.339 | 0.000 | 0.339 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_107 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.421 | 0.000 | 0.217 | 0.000 | 0.318 | 0.157 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_108 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.441 | 0.000 | 0.193 | 0.000 | 0.292 | 0.190 | 0.292 | 0.000 | 0.292 | 0.000 | 0.292 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_109 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.382 | 0.000 | 0.412 | 0.210 | 0.301 | 0.301 | 0.301 | 0.000 | 0.440 | 0.139 | 0.440 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_110 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.501 | 0.000 | 0.488 | 0.000 | 0.468 | 0.468 | 0.468 | 0.000 | 0.468 | 0.000 | 0.468 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_111 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.467 | 0.000 | 0.201 | 0.000 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_112 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.385 | 0.000 | 0.453 | 0.077 | 0.524 | 0.524 | 0.461 | 0.000 | 0.524 | 0.063 | 0.524 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
