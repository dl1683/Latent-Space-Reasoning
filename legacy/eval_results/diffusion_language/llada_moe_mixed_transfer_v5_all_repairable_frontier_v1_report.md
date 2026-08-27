# Diffusion Schedule-Selection Benchmark Report

Full model generations: `29`
Arm selections: `41`
Run ID: `diffusion-b3324317dadee840`
Content hash: `b3324317dadee840eb28cd79e4fa2b222894d90000cc38a548d05d710b281194`
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
Trajectory task delta vs fixed: `0.018`
Trajectory task delta vs random: `0.039`
Trajectory wins/ties/losses vs fixed: `3/6/2`
Trajectory wins/ties/losses vs random: `3/6/2`
Oracle generation budget/task: `2.64`
Oracle task score: `0.475`
Oracle headroom vs trajectory: `0.036`
Oracle wins/ties/losses vs trajectory: `5/6/0`
Selector regret vs trajectory: `0.036 over 5/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.070`
Repair task delta vs random: `0.099`
Repair task delta vs trajectory: `0.046`
Repair task delta vs evolved: `0.046`
Repair generation budget delta vs evolved: `0.88`
Repair task delta per extra generation vs evolved: `0.052`
Repair wins/ties/losses vs evolved: `4/4/0`
Oracle headroom vs repair: `0.004`
Oracle wins/ties/losses vs repair: `2/6/0`
Selector regret vs repair: `0.004 over 2/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.329866 | 0.000000 | 0.029571 | - | - |
| random perturbation | repair-covered tasks | 0.300295 | -0.029571 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.399687 | 0.069821 | 0.099393 | 5/2/1 | 5/2/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.422 | 0.504 | 0.442 |
| random | 11 | 1.00 | 0.400 | 0.492 | 0.423 |
| trajectory_selected | 11 | 2.00 | 0.439 | 0.475 | 0.448 |
| repair_selected | 8 | 2.88 | 0.400 | 0.652 | 0.463 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.330 | 0.659 | 0.412 |
| planning | random | 8 | 1.00 | 0.300 | 0.642 | 0.386 |
| planning | trajectory_selected | 8 | 2.00 | 0.354 | 0.618 | 0.420 |
| planning | repair_selected | 8 | 2.88 | 0.400 | 0.652 | 0.463 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_033 | low_confidence_32 | True | denoise_phase_repairable | 0.336 | 0.256 | 385 | True | 6 | 0.600 | True | True | 12.000 | 0.375 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_034 | low_confidence_32 | True | denoise_phase_repairable | 0.434 | 0.311 | 387 | True | 4 | 0.750 | True | True | 18.000 | 0.562 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_035 | low_confidence_32 | True | denoise_phase_repairable | 0.354 | 0.294 | 384 | True | 8 | 0.467 | True | True | 26.000 | 0.812 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_036 | low_confidence_32 | False | outside_repairable_band | 0.323 | 0.223 | 351 | True | 9 | 0.357 | False | False | none | none | none | 0.357 |
| llada-moe-7b-a1b-instruct-hf | plan_037 | low_confidence_32 | True | denoise_phase_repairable | 0.260 | 0.180 | 329 | True | 9 | 0.438 | True | True | 26.000 | 0.812 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_038 | low_confidence_32 | True | denoise_phase_repairable | 0.367 | 0.287 | 348 | True | 6 | 0.600 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_039 | low_confidence_32 | True | denoise_phase_repairable | 0.303 | 0.223 | 354 | True | 8 | 0.429 | True | True | 31.000 | 0.969 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_040 | low_confidence_32 | True | denoise_phase_repairable | 0.261 | 0.201 | 206 | True | 7 | 0.500 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 7 | 4 | low_confidence_32 | final | 28.4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.049 | 0.043 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/1/3 | 0.374 | 0.681 | 0.451 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_033 | False | low_confidence_32 | 1.920 | 0.455 | 1.000 | 0.000 | 0.333 | False | Compare the model of the controller with and without the verifier's intervention, measu... |
| llada-moe-7b-a1b-instruct-hf | plan_034 | True | low_confidence_32 | 2.009 | 0.669 | 1.000 | 0.000 | 0.375 | False | Use the audit results to refine the repair policy to balancing constraint fixes and con... |
| llada-moe-7b-a1b-instruct-hf | plan_035 | True | low_confidence_32 | 2.017 | 0.768 | 1.000 | 0.000 | 0.067 | False | Measure latency, throughput consistency, and error rates across multiple scenarios. |
| llada-moe-7b-a1b-instruct-hf | plan_035 | True | low_confidence_32 | 2.180 | 0.914 | 1.000 | 0.000 | 0.133 | False | Use standardized benchmarking tools to isolate the impact of repair passes and ensure r... |
| llada-moe-7b-a1b-instruct-hf | plan_037 | True | low_confidence_32 | 1.966 | 1.000 | 1.000 | 0.000 | 0.188 | False | If the feature consistently consistent across different slices, it it is be worth to ca... |
| llada-moe-7b-a1b-instruct-hf | plan_037 | True | low_confidence_32 | 2.606 | 0.787 | 1.000 | 0.000 | 0.188 | False | If, on, the other hand, the feature shows inconsistent performance across different sli... |
| llada-moe-7b-a1b-instruct-hf | plan_038 | False | low_confidence_32 | 1.815 | 0.213 | 1.000 | 0.000 | 0.333 | False | Measure final accuracy, intermediate state interpretability, computation time, and inte... |
| llada-moe-7b-a1b-instruct-hf | plan_039 | True | low_confidence_32 | 1.963 | 0.509 | 1.000 | 0.000 | 0.286 | False | Evaluate the selector's performance, consistency, robustness, and decision reliability... |
| llada-moe-7b-a1b-instruct-hf | plan_040 | False | low_confidence_32 | 3.388 | 1.000 | 1.000 | 0.000 | 0.000 | False | and and,, and and, |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_033 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.392 | 0.000 | 0.256 | 0.000 | 0.336 | 0.180 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_034 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.443 | 0.000 | 0.489 | 0.103 | 0.434 | 0.212 | 0.434 | 0.000 | 0.530 | 0.096 | 0.530 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_035 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.360 | 0.000 | 0.337 | 0.092 | 0.354 | 0.354 | 0.324 | 0.000 | 0.343 | 0.019 | 0.354 | 0.011 |
| llada-moe-7b-a1b-instruct-hf | plan_036 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.379 | 0.000 | 0.392 | 0.000 | 0.323 | 0.323 | 0.409 | 0.000 | 0.409 | 0.000 | 0.409 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_037 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.321 | 0.000 | 0.422 | 0.204 | 0.260 | 0.297 | 0.297 | 0.000 | 0.433 | 0.135 | 0.433 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_038 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.399 | 0.000 | 0.287 | 0.000 | 0.367 | 0.367 | 0.367 | 0.000 | 0.367 | 0.000 | 0.387 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | plan_039 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.302 | 0.000 | 0.381 | 0.163 | 0.303 | 0.303 | 0.299 | 0.000 | 0.413 | 0.114 | 0.413 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_040 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.310 | 0.000 | 0.266 | 0.000 | 0.261 | 0.366 | 0.366 | 0.000 | 0.366 | 0.000 | 0.366 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
