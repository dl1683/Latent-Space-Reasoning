# Diffusion Schedule-Selection Benchmark Report

Full model generations: `23`
Arm selections: `41`
Run ID: `diffusion-6c109fce02464b0f`
Content hash: `6c109fce02464b0fd791340fac65d226c526618fac299bbabc60d3603d05ad8e`
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
History mutability: `monotonic 23/23, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span_phase_final_preserve_seeded_gated`
Repair source policy: `fixed`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `decomposed_spend_transfer_rule`
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
Repair promotion margin: `0.000`
Trajectory task delta vs fixed: `0.010`
Trajectory task delta vs random: `0.030`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `4/7/0`
Oracle generation budget/task: `2.09`
Oracle task score: `0.437`
Oracle headroom vs trajectory: `0.002`
Oracle wins/ties/losses vs trajectory: `1/10/0`
Selector regret vs trajectory: `0.002 over 1/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.013`
Repair task delta vs random: `0.041`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.12`
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
| fixed baseline | repair-covered tasks | 0.335268 | 0.000000 | 0.028134 | - | - |
| random perturbation | repair-covered tasks | 0.307134 | -0.028134 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.348438 | 0.013170 | 0.041304 | 1/7/0 | 4/4/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.426 | 0.504 | 0.445 |
| random | 11 | 1.00 | 0.405 | 0.462 | 0.419 |
| trajectory_selected | 11 | 2.00 | 0.435 | 0.504 | 0.453 |
| repair_selected | 8 | 2.12 | 0.348 | 0.659 | 0.426 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.335 | 0.659 | 0.416 |
| planning | random | 8 | 1.00 | 0.307 | 0.601 | 0.381 |
| planning | trajectory_selected | 8 | 2.00 | 0.348 | 0.659 | 0.426 |
| planning | repair_selected | 8 | 2.12 | 0.348 | 0.659 | 0.426 |
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
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | False | value_proxy_source_quality_high | 0.393 | 0.333 | 327 | True | 7 | 0.562 | True | True | 15.000 | 0.469 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | False | outside_repairable_band | 0.336 | 0.239 | 329 | True | 12 | 0.294 | False | False | none | none | none | 0.294 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | True | decomposed_spend_transfer_rule | 0.295 | 0.235 | 309 | True | 8 | 0.529 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_013 | low_confidence_32 | False | outside_repairable_band | 0.304 | 0.244 | 348 | True | 10 | 0.444 | False | True | 32.000 | 1.000 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_014 | low_confidence_32 | False | outside_repairable_band | 0.303 | 0.223 | 329 | True | 10 | 0.412 | False | True | 25.000 | 0.781 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_015 | low_confidence_32 | False | outside_repairable_band | 0.453 | 0.335 | 308 | True | 10 | 0.333 | False | False | none | none | none | 0.333 |
| llada-moe-7b-a1b-instruct-hf | plan_016 | low_confidence_32 | False | outside_repairable_band | 0.241 | 0.201 | 288 | True | 12 | 0.250 | False | False | none | none | none | 0.250 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 1 | 0 | low_confidence_32 | final | 41.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.315 | 0.688 | 0.409 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 1.282 | 0.620 | 1.000 | 0.000 | 0.176 | False | Measure the accuracy of multi-step answers for both groups. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 1.413 | 0.887 | 1.000 | 0.000 | 0.176 | False | If group B has significantly worse answers, revert the compression. |
| llada-moe-7b-a1b-instruct-hf | plan_012 | False | low_confidence_32 | 2.000 | 0.595 | 1.000 | 0.000 | 0.294 | False | If group B has significantly better answers, keep the compression for the next release. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.361 | 0.000 | 0.256 | 0.000 | 0.356 | 0.356 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.386 | 0.000 | 0.389 | 0.000 | 0.393 | 0.393 | 0.393 | 0.000 | 0.393 | 0.000 | 0.393 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.261 | 0.000 | 0.239 | 0.000 | 0.336 | 0.296 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool |  |  |  | 0.291 | 0.000 | 0.235 | 0.000 | 0.295 | 0.257 | 0.295 | 0.000 | 0.295 | 0.000 | 0.315 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | plan_013 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.345 | 0.000 | 0.244 | 0.000 | 0.304 | 0.157 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_014 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.329 | 0.000 | 0.223 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_015 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.457 | 0.000 | 0.505 | 0.000 | 0.453 | 0.453 | 0.558 | 0.000 | 0.558 | 0.000 | 0.558 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_016 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.248 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
