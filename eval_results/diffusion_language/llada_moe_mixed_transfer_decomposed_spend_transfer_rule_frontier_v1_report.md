# Diffusion Schedule-Selection Benchmark Report

Full model generations: `14`
Arm selections: `25`
Run ID: `diffusion-f50e82f88f59111b`
Content hash: `f50e82f88f59111b920573456272f81bd949d711c882224f3bb1cccf8c4cd84c`
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
History mutability: `monotonic 14/14, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Repair transfer source-task min: `0.3075`
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
Oracle generation budget/task: `2.00`
Oracle task score: `0.483`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/7/0`
Selector regret vs trajectory: `0.000 over 0/7 improvable`
Repair arm coverage: `4/7` overall
Repair eligible coverage: `4/5`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.020`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/4/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/4/0`
Selector regret vs repair: `0.000 over 0/4 improvable`

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
| repair_selected | 4 | 2.00 | 0.345 | 0.659 | 0.424 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 4 | 1.00 | 0.345 | 0.659 | 0.424 |
| planning | random | 4 | 1.00 | 0.326 | 0.595 | 0.393 |
| planning | trajectory_selected | 4 | 2.00 | 0.345 | 0.659 | 0.424 |
| planning | repair_selected | 4 | 2.00 | 0.345 | 0.659 | 0.424 |
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
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | False | transfer_source_task_score_low | 0.295 | 0.235 | 309 | True | 8 | 0.529 | True | True | 20.000 | 0.625 | 0.412 | 0.412 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.361 | 0.000 | 0.256 | 0.000 | 0.356 | 0.356 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_010 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.386 | 0.000 | 0.389 | 0.000 | 0.393 | 0.393 | 0.393 | 0.000 | 0.393 | 0.000 | 0.393 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_011 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.261 | 0.000 | 0.239 | 0.000 | 0.336 | 0.296 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_012 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_decomposed_spend_transfer_rule |  |  |  | 0.291 | 0.000 | 0.235 | 0.000 | 0.295 | 0.257 | 0.295 | 0.000 | 0.295 | 0.000 | 0.295 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
