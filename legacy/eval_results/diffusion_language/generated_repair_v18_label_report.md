# Diffusion Schedule-Selection Benchmark Report

Full model generations: `26`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-da860927095a60e2`
Content hash: `da860927095a60e206e4fff117c2506c5a68a3939e4774a6e66c89e13e6f4b1c`
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
History mutability: `monotonic 26/26, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `-0.000`
Trajectory task delta vs random: `0.017`
Trajectory wins/ties/losses vs fixed: `0/10/1`
Trajectory wins/ties/losses vs random: `2/9/0`
Oracle generation budget/task: `2.36`
Oracle task score: `0.421`
Oracle headroom vs trajectory: `0.026`
Oracle wins/ties/losses vs trajectory: `2/9/0`
Selector regret vs trajectory: `0.026 over 2/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.035`
Repair task delta vs random: `0.059`
Repair task delta vs trajectory: `0.035`
Repair task delta vs evolved: `0.035`
Repair generation budget delta vs evolved: `0.50`
Repair task delta per extra generation vs evolved: `0.071`
Repair wins/ties/losses vs evolved: `2/6/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.294196 | 0.000000 | 0.023429 | - | - |
| random perturbation | repair-covered tasks | 0.270768 | -0.023429 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.329420 | 0.035223 | 0.058652 | 2/6/0 | 4/4/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.396 | 0.504 | 0.423 |
| random | 11 | 1.00 | 0.379 | 0.444 | 0.395 |
| trajectory_selected | 11 | 2.00 | 0.396 | 0.504 | 0.423 |
| repair_selected | 8 | 2.50 | 0.329 | 0.665 | 0.413 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.294 | 0.659 | 0.385 |
| planning | random | 8 | 1.00 | 0.271 | 0.576 | 0.347 |
| planning | trajectory_selected | 8 | 2.00 | 0.294 | 0.659 | 0.385 |
| planning | repair_selected | 8 | 2.50 | 0.329 | 0.665 | 0.413 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_137 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 365 | True | 2 | 0.889 | True | True | 11.000 | 0.344 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_138 | low_confidence_32 | False | outside_repairable_band | False |  | 0.260 | 0.180 | 375 | True | 0 | 1.000 | False | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_139 | random_32 | True | denoise_phase_repairable | False |  | 0.279 | 0.239 | 340 | True | 5 | 0.615 | True | True | 17.000 | 0.531 | 0.462 | 0.462 |
| llada-moe-7b-a1b-instruct-hf | plan_140 | random_32 | False | outside_repairable_band | False |  | 0.197 | 0.117 | 117 | True | 11 | 0.389 | False | False | none | none | none | 0.389 |
| llada-moe-7b-a1b-instruct-hf | plan_141 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.389 | 0.289 | 358 | True | 5 | 0.692 | True | True | 9.000 | 0.281 | 0.538 | 0.538 |
| llada-moe-7b-a1b-instruct-hf | plan_142 | low_confidence_32 | False | outside_repairable_band | False |  | 0.220 | 0.180 | 239 | True | 10 | 0.167 | False | False | none | none | none | 0.167 |
| llada-moe-7b-a1b-instruct-hf | plan_143 | random_32 | False | outside_repairable_band | False |  | 0.138 | 0.138 | 81 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_144 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.360 | 0.260 | 385 | True | 8 | 0.588 | True | True | 12.000 | 0.375 | 0.471 | 0.471 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 4 | 2 | low_confidence_32,random_32 | final | 32.2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/1/1 | 0.403 | 0.678 | 0.472 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_137 | True | low_confidence_32 | 1.816 | 0.273 | 1.000 | 0.000 | 0.556 | False | This demonstrates that generation adds value beyond source selection selection by enabl... |
| llada-moe-7b-a1b-instruct-hf | plan_139 | True | random_32 | 2.155 | 0.944 | 1.000 | 0.000 | 0.308 | False | If the candidate succeeds with the current margin, it indicates no need for increasing... |
| llada-moe-7b-a1b-instruct-hf | plan_141 | False | low_confidence_32 | 1.966 | 0.625 | 1.000 | 0.000 | 0.615 | False | Specify that the is that repair mechanism assumes presence of equal-source positives, e... |
| llada-moe-7b-a1b-instruct-hf | plan_144 | False | low_confidence_32 | 2.184 | 1.000 | 1.000 | 0.000 | 0.353 | False | Ensure the replay proof provides the repair strategy, while the runner executes the con... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_137 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.420 | 0.000 | 0.483 | 0.260 | 0.323 | 0.323 | 0.323 | 0.000 | 0.483 | 0.160 | 0.483 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_138 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.407 | 0.000 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_139 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | random_32 | final |  | 0.370 | 0.000 | 0.431 | 0.192 | 0.280 | 0.279 | 0.279 | 0.000 | 0.402 | 0.123 | 0.402 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_140 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.354 | 0.000 | 0.180 | 0.000 | 0.280 | 0.197 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_141 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.438 | 0.000 | 0.289 | 0.000 | 0.389 | 0.389 | 0.389 | 0.000 | 0.389 | 0.000 | 0.389 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_142 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.207 | 0.000 | 0.180 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_143 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.395 | 0.000 | 0.201 | 0.000 | 0.241 | 0.138 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_144 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.365 | 0.000 | 0.260 | 0.000 | 0.360 | 0.360 | 0.360 | 0.000 | 0.360 | 0.000 | 0.360 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
