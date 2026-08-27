# Diffusion Schedule-Selection Benchmark Report

Full model generations: `29`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-1fc00c32b9d9b51a`
Content hash: `1fc00c32b9d9b51a770a411cd8373b2fb178bbf6f62df9f7b95b573908f47827`
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
Trajectory task delta vs fixed: `0.007`
Trajectory task delta vs random: `0.046`
Trajectory wins/ties/losses vs fixed: `2/9/0`
Trajectory wins/ties/losses vs random: `4/6/1`
Oracle generation budget/task: `2.64`
Oracle task score: `0.423`
Oracle headroom vs trajectory: `0.002`
Oracle wins/ties/losses vs trajectory: `1/10/0`
Selector regret vs trajectory: `0.002 over 1/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.010`
Repair task delta vs random: `0.063`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.88`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.003`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.003 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.319018 | 0.000000 | 0.052786 | - | - |
| random perturbation | repair-covered tasks | 0.266232 | -0.052786 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.329062 | 0.010045 | 0.062830 | 2/6/0 | 4/3/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.414 | 0.504 | 0.436 |
| random | 11 | 1.00 | 0.375 | 0.444 | 0.393 |
| trajectory_selected | 11 | 2.00 | 0.421 | 0.491 | 0.439 |
| repair_selected | 8 | 2.88 | 0.329 | 0.642 | 0.407 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.319 | 0.659 | 0.404 |
| planning | random | 8 | 1.00 | 0.266 | 0.577 | 0.344 |
| planning | trajectory_selected | 8 | 2.00 | 0.329 | 0.642 | 0.407 |
| planning | repair_selected | 8 | 2.88 | 0.329 | 0.642 | 0.407 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_121 | random_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 338 | True | 9 | 0.429 | True | True | 27.000 | 0.844 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_122 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.360 | 0.260 | 370 | True | 4 | 0.778 | True | True | 9.000 | 0.281 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_123 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 341 | True | 5 | 0.750 | True | True | 10.000 | 0.312 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_124 | random_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 235 | True | 8 | 0.467 | True | True | 30.000 | 0.938 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_125 | random_32 | True | denoise_phase_repairable | False |  | 0.218 | 0.138 | 205 | True | 5 | 0.583 | True | True | 19.000 | 0.594 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_126 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 38 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_127 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.295 | 0.235 | 198 | True | 6 | 0.583 | True | True | 6.000 | 0.188 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_128 | random_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 353 | True | 4 | 0.667 | True | True | 16.000 | 0.500 | 0.417 | 0.417 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 7 | 0 | low_confidence_32,random_32 | final | 22.3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.012 | -0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/3/3 | 0.286 | 0.662 | 0.380 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_121 | False | random_32 | 2.069 | 0.736 | 1.000 | 0.000 | 0.286 | False | This ensures the candidate captures the lift, but the margin filters the noise, thus se... |
| llada-moe-7b-a1b-instruct-hf | plan_122 | False | low_confidence_32 | 2.054 | 0.739 | 1.000 | 0.000 | 0.333 | False | This guard ensures that the repair candidate's performance is not just a relative impro... |
| llada-moe-7b-a1b-instruct-hf | plan_123 | False | low_confidence_32 | 2.110 | 0.833 | 1.000 | 0.000 | 0.250 | False | By comparing the observed performance against a fresh-slice baseline that uses a random... |
| llada-moe-7b-a1b-instruct-hf | plan_124 | False | random_32 | 1.832 | 0.238 | 1.000 | 0.000 | 0.333 | False | Treat attention as scarce compute, defining clear metrics, success thresholds, and regu... |
| llada-moe-7b-a1b-instruct-hf | plan_125 | False | random_32 | 1.924 | 0.432 | 1.000 | 0.000 | 0.333 | False | This increases promotion margin without overclaiming, ensuring all relevant candidates... |
| llada-moe-7b-a1b-instruct-hf | plan_127 | False | low_confidence_32 | 2.864 | 0.948 | 1.000 | 0.000 | 0.000 | False | and and,, and and, |
| llada-moe-7b-a1b-instruct-hf | plan_128 | False | random_32 | 2.754 | 0.755 | 1.000 | 0.000 | 0.083 | False | Use clear, reproducible data with names, metrics, units, p.s., visualizations, and cita... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_121 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.328 | 0.000 | 0.273 | 0.000 | 0.373 | 0.344 | 0.373 | 0.000 | 0.373 | 0.000 | 0.373 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_122 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.417 | 0.000 | 0.260 | 0.000 | 0.360 | 0.360 | 0.360 | 0.000 | 0.360 | 0.000 | 0.360 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_123 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.382 | 0.000 | 0.223 | 0.000 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_124 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.310 | 0.000 | 0.223 | 0.000 | 0.301 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_125 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.376 | 0.000 | 0.201 | 0.000 | 0.281 | 0.218 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_126 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.454 | 0.000 | 0.401 | 0.000 | 0.417 | 0.045 | 0.417 | 0.000 | 0.417 | 0.000 | 0.417 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_127 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.393 | 0.000 | 0.294 | 0.000 | 0.295 | 0.295 | 0.354 | 0.000 | 0.354 | 0.000 | 0.354 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_128 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.384 | 0.000 | 0.201 | 0.000 | 0.241 | 0.261 | 0.241 | 0.000 | 0.241 | 0.000 | 0.261 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
