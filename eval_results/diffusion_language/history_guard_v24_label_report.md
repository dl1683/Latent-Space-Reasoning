# Diffusion Schedule-Selection Benchmark Report

Full model generations: `32`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-12de30e468544b5d`
Content hash: `12de30e468544b5d54e93b7a4dfce4663b2f3e980d70e017d78389d53e08420c`
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
History mutability: `monotonic 32/32, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.074`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `2.91`
Oracle task score: `0.461`
Oracle headroom vs trajectory: `0.027`
Oracle wins/ties/losses vs trajectory: `3/8/0`
Selector regret vs trajectory: `0.027 over 3/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.038`
Repair task delta vs random: `0.139`
Repair task delta vs trajectory: `0.038`
Repair task delta vs evolved: `0.038`
Repair generation budget delta vs evolved: `1.25`
Repair task delta per extra generation vs evolved: `0.030`
Repair wins/ties/losses vs evolved: `3/5/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.345804 | 0.000000 | 0.101179 | - | - |
| random perturbation | repair-covered tasks | 0.244625 | -0.101179 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.383527 | 0.037723 | 0.138902 | 3/5/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.433 | 0.504 | 0.451 |
| random | 11 | 1.00 | 0.360 | 0.344 | 0.356 |
| trajectory_selected | 11 | 2.00 | 0.433 | 0.504 | 0.451 |
| repair_selected | 8 | 3.25 | 0.384 | 0.671 | 0.455 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.346 | 0.659 | 0.424 |
| planning | random | 8 | 1.00 | 0.245 | 0.438 | 0.293 |
| planning | trajectory_selected | 8 | 2.00 | 0.346 | 0.659 | 0.424 |
| planning | repair_selected | 8 | 3.25 | 0.384 | 0.671 | 0.455 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_185 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.400 | 0.278 | 375 | True | 6 | 0.625 | True | True | 8.000 | 0.250 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_186 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 48 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_187 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 348 | True | 3 | 0.800 | True | True | 16.000 | 0.500 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_188 | random_32 | True | denoise_phase_repairable | False |  | 0.257 | 0.197 | 308 | True | 9 | 0.529 | True | True | 23.000 | 0.719 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_189 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 11 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_190 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.540 | 0.438 | 347 | True | 3 | 0.812 | True | True | 11.000 | 0.344 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_191 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 382 | True | 2 | 0.875 | True | True | 13.000 | 0.406 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_192 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 4 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 5 | 2 | low_confidence_32,random_32 | final | 39.4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.036 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/1/2 | 0.400 | 0.673 | 0.469 |
| history_prefix_25_repair | 5 | 1 | low_confidence_32,random_32 | history | 48.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 | 0.014 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/2/2 | 0.378 | 0.688 | 0.456 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_185 | False | low_confidence_32 | 2.063 | 0.893 | 1.000 | 0.000 | 0.062 | False | To ensure this, verify the audit process does not alter the selection logic or introduc... |
| llada-moe-7b-a1b-instruct-hf | plan_185 | False | low_confidence_32 | 2.136 | 0.893 | 1.000 | 0.000 | 0.188 | False | Conduct controlled checks and document the audit thoroughly to preserve the integrity o... |
| llada-moe-7b-a1b-instruct-hf | plan_187 | False | low_confidence_32 | 2.034 | 0.646 | 1.000 | 0.000 | 0.333 | False | This, in with a clean selected hook, will prevent confusion confusion by accurately acc... |
| llada-moe-7b-a1b-instruct-hf | plan_188 | True | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | [history, prefix, repair, repairability, preservation, useful, scaffold, version, integ... |
| llada-moe-7b-a1b-instruct-hf | plan_190 | True | low_confidence_32 | 2.280 | 0.254 | 1.000 | 0.000 | 0.562 | False | Each row should specify: candidate-serve, candidate-source, routing decision, route typ... |
| llada-moe-7b-a1b-instruct-hf | plan_191 | False | low_confidence_32 | 1.844 | 0.290 | 1.000 | 0.000 | 0.375 | False | Use historical data to demonstrate cost efficiency, reliability improvements, and opera... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_185 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.418 | 0.000 | 0.000 | 0.000 | 0.400 | 0.400 | 0.400 | 0.000 | 0.400 | 0.000 | 0.400 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_186 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.411 | 0.000 | 0.000 | 0.000 | 0.299 | 0.045 | 0.299 | 0.000 | 0.299 | 0.000 | 0.299 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_187 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 25 | 0.361 | 0.000 | 0.210 | 0.210 | 0.280 | 0.280 | 0.280 | 0.000 | 0.409 | 0.129 | 0.409 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_188 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.409 | 0.000 | 0.122 | 0.122 | 0.295 | 0.257 | 0.295 | 0.000 | 0.350 | 0.055 | 0.350 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_189 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.438 | 0.000 | 0.000 | 0.000 | 0.258 | 0.045 | 0.258 | 0.000 | 0.258 | 0.000 | 0.258 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_190 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.512 | 0.000 | 0.229 | 0.229 | 0.540 | 0.540 | 0.540 | 0.000 | 0.658 | 0.118 | 0.658 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_191 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.447 | 0.000 | 0.000 | 0.000 | 0.344 | 0.344 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_192 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.517 | 0.000 | 0.000 | 0.000 | 0.350 | 0.045 | 0.350 | 0.000 | 0.350 | 0.000 | 0.350 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
