# Diffusion Schedule-Selection Benchmark Report

Full model generations: `3`
Arm selections: `4`
Run ID: `diffusion-cdae920acfca7268`
Content hash: `cdae920acfca72683953851168ca6d1738070043a9366052895ed47d77267363`
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
Repair pack: `constraint_span_phase_anchor`
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
Repair denoise skeleton max step: `none`
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
Repair selector: `planning_quality_prompt_coverage_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.000`
Trajectory wins/ties/losses vs fixed: `0/1/0`
Trajectory wins/ties/losses vs random: `0/1/0`
Oracle generation budget/task: `3.00`
Oracle task score: `0.516`
Oracle headroom vs trajectory: `0.209`
Oracle wins/ties/losses vs trajectory: `1/0/0`
Selector regret vs trajectory: `0.209 over 1/1 improvable`
Repair arm coverage: `1/1` overall
Repair eligible coverage: `1/1`
Repair task delta vs fixed: `0.209`
Repair task delta vs random: `0.209`
Repair task delta vs trajectory: `0.209`
Repair task delta vs evolved: `0.209`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.209`
Repair wins/ties/losses vs evolved: `1/0/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/1/0`
Selector regret vs repair: `0.000 over 0/1 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `1/1` overall, `1/1` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.307500 | 0.000000 | 0.000000 | - | - |
| random perturbation | repair-covered tasks | 0.307500 | 0.000000 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.516429 | 0.208929 | 0.208929 | 1/0/0 | 1/0/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1 | 1.00 | 0.307 | 0.659 | 0.395 |
| random | 1 | 1.00 | 0.307 | 0.659 | 0.395 |
| trajectory_selected | 1 | 2.00 | 0.307 | 0.659 | 0.395 |
| repair_selected | 1 | 3.00 | 0.516 | 0.688 | 0.559 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 1 | 1.00 | 0.307 | 0.659 | 0.395 |
| planning | random | 1 | 1.00 | 0.307 | 0.659 | 0.395 |
| planning | trajectory_selected | 1 | 2.00 | 0.307 | 0.659 | 0.395 |
| planning | repair_selected | 1 | 3.00 | 0.516 | 0.688 | 0.559 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | True | denoise_phase_repairable | 0.307 | 0.247 | 322 | True | 8 | 0.417 | True | True | 31.000 | 0.969 | 0.417 | 0.417 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_anchor_repair | 1 | 1 | low_confidence_32 | final | 47.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.159 | 0.209 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/0/0 | 0.516 | 0.688 | 0.559 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 1.423 | 0.936 | 1.000 | 0.000 | 0.167 | False | If the divergence occurs only with the change, the issue is with the optimizer. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 1.388 | 0.850 | 1.000 | 0.000 | 0.167 | False | If it occurs with both, the problem may lie in the model architecture or training loop. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 2.092 | 0.782 | 1.000 | 0.000 | 0.250 | False | This experiment is sufficient to attribute the divergence to the optimizer change. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_anchor_repair | constraint_gap_span_phase_anchor_repair | max_planning_state_score_base_pool |  | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.333 | 0.000 | 0.465 | 0.217 | 0.307 | 0.307 | 0.307 | 0.000 | 0.516 | 0.209 | 0.516 | 0.000 |
