# Diffusion Schedule-Selection Benchmark Report

Full model generations: `6`
Arm selections: `8`
Run ID: `diffusion-4699321baf91294e`
Content hash: `4699321baf91294eb02abe2448bb5b0b9b2b4f9bd71fac6134ca39535ec220e8`
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
History mutability: `monotonic 6/6, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated`
Repair source policy: `fixed`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `always`
Repair source-quality threshold: `0.500`
Repair source min chars: `320`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `999`
Repair source prompt coverage band: `0.000-1.000`
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
Trajectory task delta vs fixed: `0.009`
Trajectory task delta vs random: `0.000`
Trajectory wins/ties/losses vs fixed: `1/1/0`
Trajectory wins/ties/losses vs random: `0/2/0`
Oracle generation budget/task: `3.00`
Oracle task score: `0.352`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/2/0`
Selector regret vs trajectory: `0.000 over 0/2 improvable`
Repair arm coverage: `2/2` overall
Repair eligible coverage: `2/2`
Repair task delta vs fixed: `0.009`
Repair task delta vs random: `0.000`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/2/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/2/0`
Selector regret vs repair: `0.000 over 0/2 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `2/2` overall, `2/2` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.342857 | 0.000000 | -0.009286 | - | - |
| random perturbation | repair-covered tasks | 0.352143 | 0.009286 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.352143 | 0.009286 | 0.000000 | 1/1/0 | 0/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 2 | 1.00 | 0.343 | 0.659 | 0.422 |
| random | 2 | 1.00 | 0.352 | 0.659 | 0.429 |
| trajectory_selected | 2 | 2.00 | 0.352 | 0.659 | 0.429 |
| repair_selected | 2 | 3.00 | 0.352 | 0.659 | 0.429 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 2 | 1.00 | 0.343 | 0.659 | 0.422 |
| planning | random | 2 | 1.00 | 0.352 | 0.659 | 0.429 |
| planning | trajectory_selected | 2 | 2.00 | 0.352 | 0.659 | 0.429 |
| planning | repair_selected | 2 | 3.00 | 0.352 | 0.659 | 0.429 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair | 2 | 0 | low_confidence_32 | final | 33.5 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.013 | -0.014 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/1 | 0.328 | 0.680 | 0.416 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_005 | False | low_confidence_32 | 1.675 | 0.475 | 1.000 | 0.000 | 0.353 | False | This preserves the state of the training run and avoids corrupting the best checkpoint. |
| llada-moe-7b-a1b-instruct-hf | plan_005 | False | low_confidence_32 | 2.715 | 1.000 | 1.000 | 0.000 | 0.118 | False | Document the the state that the checkpoint was restored to for reproducibility. |
| llada-moe-7b-a1b-instruct-hf | plan_008 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Evaluate consistency, and,, and and.. |
| llada-moe-7b-a1b-instruct-hf | plan_008 | False | low_confidence_32 | 2.095 | 0.940 | 1.000 | 0.000 | 0.062 | False | Compare outputs to known facts,, and, and. |
| llada-moe-7b-a1b-instruct-hf | plan_008 | False | low_confidence_32 | 2.861 | 0.940 | 1.000 | 0.000 | 0.000 | False | Assess and, and, and,, and, and, and. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.334 | 0.000 | 0.299 | 0.000 | 0.421 | 0.421 | 0.421 | 0.000 | 0.421 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 |  | random_32 | constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.274 | 0.000 | 0.223 | 0.000 | 0.264 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
