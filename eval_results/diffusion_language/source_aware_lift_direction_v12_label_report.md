# Diffusion Schedule-Selection Benchmark Report

Full model generations: `25`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-bb0e3d91e8840b1a`
Content hash: `bb0e3d91e8840b1a38dbd3aa25396ea80fe652fe90ed46cb5dbf1f2cb3bc4b7c`
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
History mutability: `monotonic 25/25, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `-0.011`
Trajectory task delta vs random: `0.050`
Trajectory wins/ties/losses vs fixed: `1/9/1`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `2.27`
Oracle task score: `0.463`
Oracle headroom vs trajectory: `0.031`
Oracle wins/ties/losses vs trajectory: `2/9/0`
Selector regret vs trajectory: `0.031 over 2/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.002`
Repair task delta vs random: `0.086`
Repair task delta vs trajectory: `0.017`
Repair task delta vs evolved: `0.017`
Repair generation budget delta vs evolved: `0.38`
Repair task delta per extra generation vs evolved: `0.046`
Repair wins/ties/losses vs evolved: `1/7/0`
Oracle headroom vs repair: `0.026`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.026 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.357946 | 0.000000 | 0.083562 | - | - |
| random perturbation | repair-covered tasks | 0.274384 | -0.083562 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.360205 | 0.002259 | 0.085821 | 2/5/1 | 4/4/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.442 | 0.504 | 0.458 |
| random | 11 | 1.00 | 0.381 | 0.408 | 0.388 |
| trajectory_selected | 11 | 2.00 | 0.431 | 0.485 | 0.445 |
| repair_selected | 8 | 2.38 | 0.360 | 0.647 | 0.432 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.358 | 0.659 | 0.433 |
| planning | random | 8 | 1.00 | 0.274 | 0.526 | 0.337 |
| planning | trajectory_selected | 8 | 2.00 | 0.343 | 0.633 | 0.415 |
| planning | repair_selected | 8 | 2.38 | 0.360 | 0.647 | 0.432 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_089 | random_32 | False | outside_repairable_band | False |  | 0.157 | 0.117 | 91 | True | 12 | 0.316 | False | False | none | none | none | 0.316 |
| llada-moe-7b-a1b-instruct-hf | plan_090 | random_32 | False | outside_repairable_band | False |  | 0.240 | 0.180 | 173 | True | 10 | 0.375 | False | False | none | none | none | 0.375 |
| llada-moe-7b-a1b-instruct-hf | plan_091 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.427 | 0.327 | 383 | True | 2 | 0.867 | True | True | 8.000 | 0.250 | 0.467 | 0.467 |
| llada-moe-7b-a1b-instruct-hf | plan_092 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.374 | 0.294 | 382 | True | 7 | 0.533 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_093 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.394 | 0.244 | 333 | True | 6 | 0.455 | True | True | 12.000 | 0.375 | 0.455 | 0.455 |
| llada-moe-7b-a1b-instruct-hf | plan_094 | random_32 | False | outside_repairable_band | False |  | 0.232 | 0.172 | 166 | True | 12 | 0.250 | False | False | none | none | none | 0.250 |
| llada-moe-7b-a1b-instruct-hf | plan_095 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 15 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_096 | random_32 | False | outside_repairable_band | False |  | 0.325 | 0.265 | 370 | True | 12 | 0.353 | False | False | none | none | none | 0.353 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 3 | 1 | low_confidence_32 | final | 31.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/2 | 0.407 | 0.683 | 0.476 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_091 | False | low_confidence_32 | 2.534 | 0.686 | 1.000 | 0.000 | 0.333 | False | This ensures that only oracle-positive and successfully-promoted candidates are include... |
| llada-moe-7b-a1b-instruct-hf | plan_092 | False | low_confidence_32 | 2.523 | 0.818 | 1.000 | 0.000 | 0.067 | False | Use controlled conditions to isolate the effect of gap and alignment alone. . |
| llada-moe-7b-a1b-instruct-hf | plan_092 | False | low_confidence_32 | 1.929 | 0.496 | 1.000 | 0.000 | 0.400 | False | Analyze results to demonstrate that moderate gap and alignment can independently predic... |
| llada-moe-7b-a1b-instruct-hf | plan_093 | True | low_confidence_32 | 2.549 | 1.000 | 1.000 | 0.000 | 0.182 | False | Otherwise, it should be used in a live gate to maximize overall benefit. |
| llada-moe-7b-a1b-instruct-hf | plan_093 | True | low_confidence_32 | 3.271 | 0.968 | 1.000 | 0.000 | 0.273 | False | This decision should be based on a cost-benefit analysis comparing the cost of tomograp... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_089 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.385 | 0.000 | 0.294 | 0.000 | 0.374 | 0.157 | 0.374 | 0.000 | 0.374 | 0.000 | 0.374 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_090 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.269 | 0.000 | 0.201 | 0.000 | 0.261 | 0.240 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_091 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.491 | 0.000 | 0.414 | 0.000 | 0.427 | 0.427 | 0.427 | 0.000 | 0.427 | 0.000 | 0.427 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_092 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.393 | 0.000 | 0.294 | 0.000 | 0.374 | 0.374 | 0.374 | 0.000 | 0.374 | 0.000 | 0.374 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_093 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.332 | 0.000 | 0.437 | 0.193 | 0.394 | 0.394 | 0.394 | 0.000 | 0.533 | 0.139 | 0.533 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_094 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.257 | 0.000 | 0.172 | 0.000 | 0.438 | 0.232 | 0.232 | 0.000 | 0.232 | 0.000 | 0.438 | 0.206 |
| llada-moe-7b-a1b-instruct-hf | plan_095 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.498 | 0.000 | 0.294 | 0.000 | 0.354 | 0.045 | 0.354 | 0.000 | 0.354 | 0.000 | 0.354 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_096 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.327 | 0.000 | 0.265 | 0.000 | 0.240 | 0.325 | 0.325 | 0.000 | 0.325 | 0.000 | 0.325 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
