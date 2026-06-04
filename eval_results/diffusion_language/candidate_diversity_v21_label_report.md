# Diffusion Schedule-Selection Benchmark Report

Full model generations: `34`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-09714dd28133e7e9`
Content hash: `09714dd28133e7e9fe2d7ea15989c00c89329ca44ae678f8203717ac5e7620b8`
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
History mutability: `monotonic 34/34, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs random: `0.048`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `3.09`
Oracle task score: `0.436`
Oracle headroom vs trajectory: `0.028`
Oracle wins/ties/losses vs trajectory: `3/8/0`
Selector regret vs trajectory: `0.028 over 3/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.036`
Repair task delta vs random: `0.102`
Repair task delta vs trajectory: `0.036`
Repair task delta vs evolved: `0.036`
Repair generation budget delta vs evolved: `1.50`
Repair task delta per extra generation vs evolved: `0.024`
Repair wins/ties/losses vs evolved: `3/4/1`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.002 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.311161 | 0.000000 | 0.065795 | - | - |
| random perturbation | repair-covered tasks | 0.245366 | -0.065795 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.347634 | 0.036473 | 0.102268 | 3/4/1 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.408 | 0.504 | 0.432 |
| random | 11 | 1.00 | 0.360 | 0.380 | 0.365 |
| trajectory_selected | 11 | 2.00 | 0.408 | 0.504 | 0.432 |
| repair_selected | 8 | 3.50 | 0.348 | 0.677 | 0.430 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.311 | 0.659 | 0.398 |
| planning | random | 8 | 1.00 | 0.245 | 0.488 | 0.306 |
| planning | trajectory_selected | 8 | 2.00 | 0.311 | 0.659 | 0.398 |
| planning | repair_selected | 8 | 3.50 | 0.348 | 0.677 | 0.430 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_161 | random_32 | True | denoise_phase_repairable | False |  | 0.238 | 0.138 | 117 | True | 5 | 0.750 | True | True | 10.000 | 0.312 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_162 | random_32 | True | denoise_phase_repairable | False |  | 0.198 | 0.138 | 146 | True | 7 | 0.588 | True | True | 29.000 | 0.906 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_163 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 48 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_164 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 403 | True | 3 | 0.812 | True | True | 6.000 | 0.188 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_165 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.287 | 0.247 | 379 | True | 5 | 0.643 | True | True | 9.000 | 0.281 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_166 | low_confidence_32 | False | outside_repairable_band | False |  | 0.307 | 0.268 | 305 | True | 0 | 1.000 | False | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_167 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.374 | 0.294 | 403 | True | 3 | 0.800 | True | True | 6.000 | 0.188 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_168 | random_32 | True | denoise_phase_repairable | False |  | 0.188 | 0.168 | 71 | True | 7 | 0.400 | True | True | 25.000 | 0.781 | 0.400 | 0.400 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 6 | 2 | low_confidence_32,random_32 | final | 28.2 | 1.000 | 0.000 | 0.000 | 0.060 | 0.060 | 0.035 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/3/1 | 0.300 | 0.566 | 0.366 |
| history_prefix_25_repair | 6 | 3 | low_confidence_32,random_32 | history | 48.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.065 | 0.075 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/2/0 | 0.343 | 0.688 | 0.429 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_161 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Plan a fresh availability test that adds a history-sourced repair candidate without cha... |
| llada-moe-7b-a1b-instruct-hf | plan_162 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Design the target sheet to evaluate whether candidate diversity leads to useful generat... |
| llada-moe-7b-a1b-instruct-hf | plan_164 | True | low_confidence_32 | 1.333 | 0.771 | 1.000 | 0.000 | 0.188 | False | Compare selected outputs against baseline cost benchmarks, analyzing efficiency, accura... |
| llada-moe-7b-a1b-instruct-hf | plan_164 | True | low_confidence_32 | 2.193 | 0.968 | 1.000 | 0.000 | 0.125 | False | Track cost deviations, selection criteria, and performance metrics to ensure consistenc... |
| llada-moe-7b-a1b-instruct-hf | plan_165 | False | low_confidence_32 | 1.322 | 0.773 | 1.000 | 0.000 | 0.286 | False | This ensures the second candidate is considered only when the first is correctly identi... |
| llada-moe-7b-a1b-instruct-hf | plan_165 | False | low_confidence_32 | 2.218 | 1.000 | 1.000 | 0.000 | 0.357 | False | This: recall, candidate candidate, candidate candidate, etc., enhancing precision by mi... |
| llada-moe-7b-a1b-instruct-hf | plan_167 | True | low_confidence_32 | 1.377 | 0.893 | 1.000 | 0.000 | 0.333 | False | Ensure that repair decisions do not influence selector selection, and verify that promo... |
| llada-moe-7b-a1b-instruct-hf | plan_167 | True | low_confidence_32 | 2.164 | 0.875 | 1.000 | 0.000 | 0.133 | False | This maintains experimental integrity and prevents conf of claims. |
| llada-moe-7b-a1b-instruct-hf | plan_168 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Ensure GPU repair run frozen before labels, then replayed after labels. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_161 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.410 | 0.000 | 0.083 | 0.083 | 0.301 | 0.238 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_162 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.375 | 0.000 | 0.083 | 0.083 | 0.301 | 0.198 | 0.301 | 0.000 | 0.281 | -0.020 | 0.301 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | plan_163 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.191 | 0.000 | 0.000 | 0.000 | 0.221 | 0.045 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_164 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.443 | 0.000 | 0.139 | 0.139 | 0.324 | 0.324 | 0.324 | 0.000 | 0.420 | 0.096 | 0.420 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_165 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.332 | 0.000 | 0.000 | 0.000 | 0.287 | 0.287 | 0.287 | 0.000 | 0.287 | 0.000 | 0.287 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_166 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.509 | 0.000 | 0.000 | 0.000 | 0.307 | 0.307 | 0.307 | 0.000 | 0.307 | 0.000 | 0.307 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_167 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.471 | 0.000 | 0.192 | 0.192 | 0.374 | 0.374 | 0.374 | 0.000 | 0.513 | 0.139 | 0.513 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_168 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.537 | 0.000 | 0.269 | 0.269 | 0.371 | 0.188 | 0.371 | 0.000 | 0.448 | 0.077 | 0.448 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
