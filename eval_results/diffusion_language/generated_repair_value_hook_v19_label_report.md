# Diffusion Schedule-Selection Benchmark Report

Full model generations: `27`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-e4db8307fba01a16`
Content hash: `e4db8307fba01a16f69acba71a63d31c6912d12b38f1cfe4eb7145f448e7d149`
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
Repair selector: `generated_repair_value_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.023`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `2/8/1`
Oracle generation budget/task: `2.45`
Oracle task score: `0.398`
Oracle headroom vs trajectory: `0.033`
Oracle wins/ties/losses vs trajectory: `4/7/0`
Selector regret vs trajectory: `0.033 over 4/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.044`
Repair task delta vs random: `0.076`
Repair task delta vs trajectory: `0.044`
Repair task delta vs evolved: `0.044`
Repair generation budget delta vs evolved: `0.62`
Repair task delta per extra generation vs evolved: `0.071`
Repair wins/ties/losses vs evolved: `3/5/0`
Oracle headroom vs repair: `0.001`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.001 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.252188 | 0.000000 | 0.031491 | - | - |
| random perturbation | repair-covered tasks | 0.220696 | -0.031491 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.296473 | 0.044286 | 0.075777 | 3/5/0 | 5/2/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.365 | 0.459 | 0.389 |
| random | 11 | 1.00 | 0.342 | 0.433 | 0.365 |
| trajectory_selected | 11 | 2.00 | 0.365 | 0.459 | 0.389 |
| repair_selected | 8 | 2.62 | 0.296 | 0.590 | 0.370 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.252 | 0.597 | 0.338 |
| planning | random | 8 | 1.00 | 0.221 | 0.562 | 0.306 |
| planning | trajectory_selected | 8 | 2.00 | 0.252 | 0.597 | 0.338 |
| planning | repair_selected | 8 | 2.62 | 0.296 | 0.590 | 0.370 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_145 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 357 | True | 3 | 0.824 | True | True | 6.000 | 0.188 | 0.471 | 0.471 |
| llada-moe-7b-a1b-instruct-hf | plan_146 | random_32 | False | outside_repairable_band | False |  | 0.238 | 0.138 | 241 | True | 12 | 0.312 | False | False | none | none | none | 0.312 |
| llada-moe-7b-a1b-instruct-hf | plan_147 | low_confidence_32 | False | outside_repairable_band | False |  | 0.065 | 0.045 | 48 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_148 | random_32 | False | outside_repairable_band | False |  | 0.108 | 0.108 | 127 | True | 10 | 0.250 | False | False | none | none | none | 0.250 |
| llada-moe-7b-a1b-instruct-hf | plan_149 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 385 | True | 5 | 0.692 | True | True | 7.000 | 0.219 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_150 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 213 | True | 6 | 0.538 | True | True | 7.000 | 0.219 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_151 | random_32 | True | denoise_phase_repairable | False |  | 0.267 | 0.167 | 190 | True | 7 | 0.417 | True | True | 27.000 | 0.844 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_152 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 373 | True | 5 | 0.615 | True | True | 8.000 | 0.250 | 0.462 | 0.462 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 5 | 3 | low_confidence_32,random_32 | final | 27.2 | 1.000 | 0.000 | 0.000 | 0.024 | 0.024 | 0.071 | 0.071 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/2/0 | 0.342 | 0.647 | 0.418 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_145 | True | low_confidence_32 | 2.141 | 0.887 | 1.000 | 0.000 | 0.235 | False | This involves assessing the hook's ability to handle live rows, ensuring any potential... |
| llada-moe-7b-a1b-instruct-hf | plan_149 | True | low_confidence_32 | 2.100 | 0.968 | 1.000 | 0.000 | 0.077 | False | Include metrics such as convergence speed, solution quality, and computational cost per... |
| llada-moe-7b-a1b-instruct-hf | plan_149 | True | low_confidence_32 | 1.857 | 0.359 | 1.000 | 0.000 | 0.538 | False | Ensure the report clearly differentiates between effective adaptive repair and merely s... |
| llada-moe-7b-a1b-instruct-hf | plan_150 | False | low_confidence_32 | 3.811 | 0.640 | 1.000 | 0.000 | 0.077 | False | Ensure accurate, neutral, and complete documentation to maintain claim integrity. |
| llada-moe-7b-a1b-instruct-hf | plan_151 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Plan the source accounting mechanism to explicitly account for both source-tie and sour... |
| llada-moe-7b-a1b-instruct-hf | plan_152 | True | low_confidence_32 | 2.124 | 1.000 | 1.000 | 0.000 | 0.077 | False | This ensures transparency and avoids misleading stakeholders about the selector's readi... |
| llada-moe-7b-a1b-instruct-hf | plan_152 | True | low_confidence_32 | 2.810 | 0.860 | 1.000 | 0.000 | 0.077 | False | Document the status change to maintain clarity and trust. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_145 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.436 | 0.000 | 0.209 | 0.209 | 0.323 | 0.323 | 0.323 | 0.000 | 0.483 | 0.160 | 0.483 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_146 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.289 | 0.000 | 0.000 | 0.000 | 0.344 | 0.238 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_147 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 | 0.065 | 0.000 | 0.065 | 0.000 | 0.065 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_148 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.400 | 0.000 | 0.000 | 0.000 | 0.261 | 0.108 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_149 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.380 | 0.000 | 0.183 | 0.183 | 0.301 | 0.301 | 0.301 | 0.000 | 0.420 | 0.119 | 0.420 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_150 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.354 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_151 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.309 | 0.000 | 0.000 | 0.000 | 0.260 | 0.267 | 0.260 | 0.000 | 0.260 | 0.000 | 0.267 | 0.007 |
| llada-moe-7b-a1b-instruct-hf | plan_152 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.367 | 0.000 | 0.081 | 0.081 | 0.241 | 0.241 | 0.241 | 0.000 | 0.316 | 0.075 | 0.316 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
