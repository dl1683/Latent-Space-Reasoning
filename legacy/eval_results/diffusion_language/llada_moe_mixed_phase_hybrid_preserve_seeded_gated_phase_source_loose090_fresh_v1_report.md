# Diffusion Schedule-Selection Benchmark Report

Full model generations: `27`
Arm selections: `41`
Run ID: `diffusion-27e1b13d93f3abad`
Content hash: `27e1b13d93f3abad850fb58e62bb89022171cc2d52508442972defbec8e82ff4`
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
Repair pack: `constraint_span_phase_hybrid_preserve_seeded_gated`
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
Repair denoise skeleton max step: `32.000`
Phase-source threshold band: `target>=0.900, text>=0.900, chars>=0.900`
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
Oracle generation budget/task: `2.45`
Oracle task score: `0.654`
Oracle headroom vs trajectory: `0.080`
Oracle wins/ties/losses vs trajectory: `5/6/0`
Selector regret vs trajectory: `0.080 over 5/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.112`
Repair task delta vs random: `0.152`
Repair task delta vs trajectory: `0.110`
Repair task delta vs evolved: `0.110`
Repair generation budget delta vs evolved: `0.62`
Repair task delta per extra generation vs evolved: `0.176`
Repair wins/ties/losses vs evolved: `5/3/0`
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
| selected latent repair | repair-covered tasks | 0.524554 | 0.112277 | 0.152429 | 6/2/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.573 | 0.528 | 0.561 |
| random | 11 | 1.00 | 0.543 | 0.483 | 0.528 |
| trajectory_selected | 11 | 2.00 | 0.574 | 0.528 | 0.563 |
| repair_selected | 8 | 2.62 | 0.525 | 0.665 | 0.560 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.228 | 0.807 |
| math | random | 1 | 1.00 | 1.000 | 0.228 | 0.807 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.228 | 0.807 |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| planning | trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| planning | repair_selected | 8 | 2.62 | 0.525 | 0.665 | 0.560 |
| science | fixed | 1 | 1.00 | 1.000 | 0.289 | 0.822 |
| science | random | 1 | 1.00 | 1.000 | 0.171 | 0.793 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.289 | 0.822 |
| symbolic | fixed | 1 | 1.00 | 1.000 | 0.016 | 0.754 |
| symbolic | random | 1 | 1.00 | 1.000 | 0.117 | 0.779 |
| symbolic | trajectory_selected | 1 | 2.00 | 1.000 | 0.016 | 0.754 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | True | denoise_phase_repairable | 0.465 | 0.348 | 331 | True | 9 | 0.467 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | False | source_quality_ok | 0.689 | 0.559 | 263 | False | 12 | 0.278 | False | False | none | none | none | 0.278 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | True | denoise_phase_repairable | 0.422 | 0.324 | 241 | True | 6 | 0.600 | True | True | 10.000 | 0.312 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | True | denoise_phase_repairable | 0.338 | 0.278 | 373 | True | 2 | 0.882 | True | True | 10.000 | 0.312 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | False | outside_repairable_band | 0.421 | 0.299 | 358 | True | 10 | 0.412 | False | True | 30.000 | 0.938 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | True | denoise_phase_repairable | 0.391 | 0.301 | 351 | True | 9 | 0.438 | True | True | 20.000 | 0.625 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | True | denoise_phase_repairable | 0.307 | 0.247 | 322 | True | 8 | 0.417 | True | True | 31.000 | 0.969 | 0.417 | 0.417 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | False | outside_repairable_band | 0.264 | 0.244 | 241 | True | 12 | 0.062 | False | False | none | none | none | 0.062 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | 5 | 5 | low_confidence_32 | final,history | 36.2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.126 | 0.176 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5/0/0 | 0.561 | 0.668 | 0.587 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_001 | True | low_confidence_32 | 6.587 | 0.599 | 1.000 | 0.000 | 0.333 | False | This way, at least one successful ( the baseline or the intervention) can be published... |
| llada-moe-7b-a1b-instruct-hf | plan_003 | True | low_confidence_32 | 2.781 | 0.000 | 1.000 | 0.000 | 0.533 | False | Decision rule: If accuracy improves by 10% or latency increases by <50%, ship; if accur... |
| llada-moe-7b-a1b-instruct-hf | plan_004 | True | low_confidence_32 | 1.716 | 0.104 | 1.000 | 0.000 | 0.529 | False | This plan should involve comparing the baseline with the research result, analyzing the... |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 2.109 | 0.925 | 1.000 | 0.000 | 0.000 | False | Document the issue and schedule a quick meeting with the relevant team. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 2.893 | 1.000 | 1.000 | 0.000 | 0.062 | False | Ensure the analysis is thorough and includesable to prevent future issues. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 1.423 | 0.936 | 1.000 | 0.000 | 0.167 | False | If the divergence occurs only with the change, the issue is with the optimizer. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 1.388 | 0.850 | 1.000 | 0.000 | 0.167 | False | If it occurs with both, the problem may lie in the model architecture or training loop. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 2.092 | 0.782 | 1.000 | 0.000 | 0.250 | False | This experiment is sufficient to attribute the divergence to the optimizer change. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.228 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | history | 31 | 0.425 | 0.000 | 0.431 | 0.036 | 0.465 | 0.465 | 0.465 | 0.000 | 0.528 | 0.063 | 0.528 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.448 | 0.000 | 0.586 | 0.000 | 0.689 | 0.580 | 0.689 | 0.000 | 0.689 | 0.000 | 0.689 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | history | 30 | 0.418 | 0.000 | 0.455 | 0.071 | 0.422 | 0.422 | 0.422 | 0.000 | 0.486 | 0.064 | 0.486 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.466 | 0.000 | 0.676 | 0.398 | 0.338 | 0.157 | 0.338 | 0.000 | 0.622 | 0.284 | 0.622 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.334 | 0.000 | 0.299 | 0.000 | 0.421 | 0.421 | 0.421 | 0.000 | 0.421 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.366 | 0.000 | 0.487 | 0.142 | 0.391 | 0.341 | 0.391 | 0.000 | 0.584 | 0.193 | 0.584 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_planning_quality_seed_realization_guarded_score_repair_pool | low_confidence_32 | final |  | 0.333 | 0.000 | 0.475 | 0.228 | 0.307 | 0.307 | 0.307 | 0.000 | 0.584 | 0.276 | 0.584 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.274 | 0.000 | 0.223 | 0.000 | 0.264 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.289 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_002 | low_confidence_32 | random_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
