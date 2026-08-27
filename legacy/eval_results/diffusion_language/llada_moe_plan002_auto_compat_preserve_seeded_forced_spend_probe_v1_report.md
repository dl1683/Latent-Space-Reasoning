# Diffusion Schedule-Selection Benchmark Report

Full model generations: `3`
Arm selections: `4`
Run ID: `diffusion-8a8a9e8904e62dbf`
Content hash: `8a8a9e8904e62dbff0263ac46b6985405f3adc67151c302b91333b478dbaad7b`
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
History mutability: `monotonic 3/3, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.109`
Trajectory wins/ties/losses vs fixed: `0/1/0`
Trajectory wins/ties/losses vs random: `1/0/0`
Oracle generation budget/task: `3.00`
Oracle task score: `0.689`
Oracle headroom vs trajectory: `0.000`
Oracle wins/ties/losses vs trajectory: `0/1/0`
Selector regret vs trajectory: `0.000 over 0/1 improvable`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `0.000`
Repair task delta vs random: `0.109`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/1/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `1/1` overall, `1/1` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.688571 | 0.000000 | 0.108929 | - | - |
| random perturbation | repair-covered tasks | 0.579643 | -0.108929 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.688571 | 0.000000 | 0.108929 | 0/1/0 | 1/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.689 | 0.659 | 0.681 |
| random | 1 | 1.00 | 0.580 | 0.572 | 0.578 |
| trajectory_selected | 1 | 2.00 | 0.689 | 0.659 | 0.681 |
| repair_selected | 1 | 3.00 | 0.689 | 0.659 | 0.681 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 1 | 1.00 | 0.689 | 0.659 | 0.681 |
| planning | random | 1 | 1.00 | 0.580 | 0.572 | 0.578 |
| planning | trajectory_selected | 1 | 2.00 | 0.689 | 0.659 | 0.681 |
| planning | repair_selected | 1 | 3.00 | 0.689 | 0.659 | 0.681 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | True | trigger_always | 0.689 | 0.559 | 263 | True | 12 | 0.278 | False | False |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair | 1 | 0 | low_confidence_32 | final | 40.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.075 | -0.105 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/0/1 | 0.583 | 0.688 | 0.609 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_002 | False | low_confidence_32 | 3.718 | 0.963 | 1.000 | 0.000 | 0.000 | False | then the next step, |
| llada-moe-7b-a1b-instruct-hf | plan_002 | False | low_confidence_32 | 2.944 | 0.264 | 1.000 | 0.000 | 0.167 | False | analyze the logs for patterns or failure patterns, such as timeouts, timeouts,, timeout... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.448 | 0.000 | 0.586 | 0.000 | 0.689 | 0.580 | 0.689 | 0.000 | 0.689 | 0.000 | 0.689 | 0.000 |
