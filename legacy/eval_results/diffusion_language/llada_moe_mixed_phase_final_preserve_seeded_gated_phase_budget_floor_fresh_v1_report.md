# Diffusion Schedule-Selection Benchmark Report

Full model generations: `22`
Arm selections: `41`
Run ID: `diffusion-fae5a3498468b66f`
Content hash: `fae5a3498468b66ffd52c22f51fcd1add4e2ffdc7f729648aeefc915c67410f2`
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
History mutability: `monotonic 22/22, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Repair phase budget: `floor`
Repair denoise skeleton max step: `9.000`
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
Trajectory task delta vs fixed: `0.002`
Trajectory task delta vs random: `0.031`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `2.00`
Oracle task score: `0.574`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/11/0`
Selector regret vs trajectory: `0.000 over 0/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.002`
Repair task delta vs random: `0.042`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/8/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`
Selector regret vs repair: `0.000 over 0/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/8` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.412277 | 0.000000 | 0.040152 | - | - |
| random perturbation | repair-covered tasks | 0.372125 | -0.040152 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.414598 | 0.002321 | 0.042473 | 1/7/0 | 3/5/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.573 | 0.528 | 0.561 |
| random | 11 | 1.00 | 0.543 | 0.483 | 0.528 |
| trajectory_selected | 11 | 2.00 | 0.574 | 0.528 | 0.563 |
| repair_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.228 | 0.807 |
| math | random | 1 | 1.00 | 1.000 | 0.228 | 0.807 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.228 | 0.807 |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| planning | trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| planning | repair_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| science | fixed | 1 | 1.00 | 1.000 | 0.289 | 0.822 |
| science | random | 1 | 1.00 | 1.000 | 0.171 | 0.793 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.289 | 0.822 |
| symbolic | fixed | 1 | 1.00 | 1.000 | 0.016 | 0.754 |
| symbolic | random | 1 | 1.00 | 1.000 | 0.117 | 0.779 |
| symbolic | trajectory_selected | 1 | 2.00 | 1.000 | 0.016 | 0.754 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | False | late_repairable_denoise_skeleton | 0.465 | 0.348 | 331 | True | 9 | 0.467 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | False | source_quality_ok | 0.689 | 0.559 | 263 | False | 12 | 0.278 | False | False | none | none | none | 0.278 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | False | late_repairable_denoise_skeleton | 0.422 | 0.324 | 241 | True | 6 | 0.600 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | False | late_repairable_denoise_skeleton | 0.338 | 0.278 | 373 | True | 2 | 0.882 | True | True | 10.000 | 0.312 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | False | outside_repairable_band | 0.421 | 0.299 | 358 | True | 10 | 0.412 | False | True | 30.000 | 0.938 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | False | late_repairable_denoise_skeleton | 0.391 | 0.301 | 351 | True | 9 | 0.438 | True | True | 20.000 | 0.625 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | False | late_repairable_denoise_skeleton | 0.307 | 0.247 | 322 | True | 8 | 0.417 | True | True | 31.000 | 0.969 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | False | outside_repairable_band | 0.264 | 0.244 | 241 | True | 12 | 0.062 | False | False | none | none | none | 0.062 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.228 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.425 | 0.000 | 0.395 | 0.000 | 0.465 | 0.465 | 0.465 | 0.000 | 0.465 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.448 | 0.000 | 0.586 | 0.000 | 0.689 | 0.580 | 0.689 | 0.000 | 0.689 | 0.000 | 0.689 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.418 | 0.000 | 0.384 | 0.000 | 0.422 | 0.422 | 0.422 | 0.000 | 0.422 | 0.000 | 0.422 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.466 | 0.000 | 0.278 | 0.000 | 0.338 | 0.157 | 0.338 | 0.000 | 0.338 | 0.000 | 0.338 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.334 | 0.000 | 0.299 | 0.000 | 0.421 | 0.421 | 0.421 | 0.000 | 0.421 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.366 | 0.000 | 0.345 | 0.000 | 0.391 | 0.341 | 0.391 | 0.000 | 0.391 | 0.000 | 0.391 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.333 | 0.000 | 0.247 | 0.000 | 0.307 | 0.307 | 0.307 | 0.000 | 0.307 | 0.000 | 0.307 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.274 | 0.000 | 0.223 | 0.000 | 0.264 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.289 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_002 | low_confidence_32 | random_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
