# Diffusion Schedule-Selection Benchmark Report

Full model generations: `16`
Arm selections: `25`
Run ID: `diffusion-a43504b2dec11ced`
Content hash: `a43504b2dec11cedef8de5c709d345d7c16606f2f0b2c8cb637686fcd642c1f4`
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
History mutability: `monotonic 16/16, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.011`
Trajectory wins/ties/losses vs fixed: `0/7/0`
Trajectory wins/ties/losses vs random: `2/5/0`
Oracle generation budget/task: `2.29`
Oracle task score: `0.486`
Oracle headroom vs trajectory: `0.003`
Oracle wins/ties/losses vs trajectory: `1/6/0`
Selector regret vs trajectory: `0.003 over 1/7 improvable`
Repair arm coverage: `4/7` overall
Repair eligible coverage: `4/5`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.020`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.50`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/4/0`
Oracle headroom vs repair: `0.005`
Oracle wins/ties/losses vs repair: `1/3/0`
Selector regret vs repair: `0.005 over 1/4 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `4/7` overall, `4/5` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.345268 | 0.000000 | 0.019536 | - | - |
| random perturbation | repair-covered tasks | 0.325732 | -0.019536 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.345268 | 0.000000 | 0.019536 | 0/4/0 | 2/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 7 | 1.00 | 0.483 | 0.416 | 0.466 |
| random | 7 | 1.00 | 0.472 | 0.379 | 0.449 |
| trajectory_selected | 7 | 2.00 | 0.483 | 0.416 | 0.466 |
| repair_selected | 4 | 2.50 | 0.345 | 0.659 | 0.424 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 4 | 1.00 | 0.345 | 0.659 | 0.424 |
| planning | random | 4 | 1.00 | 0.326 | 0.595 | 0.393 |
| planning | trajectory_selected | 4 | 2.00 | 0.345 | 0.659 | 0.424 |
| planning | repair_selected | 4 | 2.50 | 0.345 | 0.659 | 0.424 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | False | outside_repairable_band | 0.356 | 0.256 | 352 | True | 10 | 0.471 | False | True | 17.000 | 0.531 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | True | denoise_phase_repairable | 0.393 | 0.333 | 327 | True | 7 | 0.562 | True | True | 15.000 | 0.469 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | False | outside_repairable_band | 0.336 | 0.239 | 329 | True | 12 | 0.294 | False | False | none | none | none | 0.294 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | True | denoise_phase_repairable | 0.295 | 0.235 | 309 | True | 8 | 0.529 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 2 | 0 | low_confidence_32 | final | 38.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/1/0 | 0.354 | 0.688 | 0.438 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_010 | False | low_confidence_32 | 1.441 | 0.963 | 1.000 | 0.000 | 0.250 | False | If the gain is consistent across runs runs with the same seed and order, it is likely r... |
| llada-moe-7b-a1b-instruct-hf | plan_010 | False | low_confidence_32 | 2.205 | 1.000 | 1.000 | 0.000 | 0.188 | False | If not, investigate the the impact of the random seed and test order. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 1.282 | 0.620 | 1.000 | 0.000 | 0.176 | False | Measure the accuracy of multi-step answers for both groups. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 1.413 | 0.887 | 1.000 | 0.000 | 0.176 | False | If group B has significantly worse answers, revert the compression. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 2.000 | 0.595 | 1.000 | 0.000 | 0.294 | False | If group B has significantly better answers, keep the compression for the next release. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.361 | 0.000 | 0.256 | 0.000 | 0.356 | 0.356 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.386 | 0.000 | 0.389 | 0.000 | 0.393 | 0.393 | 0.393 | 0.000 | 0.393 | 0.000 | 0.393 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.261 | 0.000 | 0.239 | 0.000 | 0.336 | 0.296 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.291 | 0.000 | 0.235 | 0.000 | 0.295 | 0.257 | 0.295 | 0.000 | 0.295 | 0.000 | 0.315 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
