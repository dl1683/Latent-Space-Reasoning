# Diffusion Schedule-Selection Benchmark Report

Full model generations: `25`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-ff98d7df72f8d3ba`
Content hash: `ff98d7df72f8d3bafa897a29547607488b8189cd8d9180562867c2b2b3ce7250`
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
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.118`
Trajectory wins/ties/losses vs fixed: `0/11/0`
Trajectory wins/ties/losses vs random: `5/6/0`
Oracle generation budget/task: `2.27`
Oracle task score: `0.431`
Oracle headroom vs trajectory: `0.016`
Oracle wins/ties/losses vs trajectory: `2/9/0`
Selector regret vs trajectory: `0.016 over 2/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.017`
Repair task delta vs random: `0.180`
Repair task delta vs trajectory: `0.017`
Repair task delta vs evolved: `0.017`
Repair generation budget delta vs evolved: `0.38`
Repair task delta per extra generation vs evolved: `0.046`
Repair wins/ties/losses vs evolved: `1/7/0`
Oracle headroom vs repair: `0.005`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.005 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.320491 | 0.000000 | 0.162562 | - | - |
| random perturbation | repair-covered tasks | 0.157929 | -0.162562 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.337857 | 0.017366 | 0.179929 | 1/7/0 | 6/2/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.415 | 0.504 | 0.437 |
| random | 11 | 1.00 | 0.297 | 0.281 | 0.293 |
| trajectory_selected | 11 | 2.00 | 0.415 | 0.504 | 0.437 |
| repair_selected | 8 | 2.38 | 0.338 | 0.654 | 0.417 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.320 | 0.659 | 0.405 |
| planning | random | 8 | 1.00 | 0.158 | 0.352 | 0.206 |
| planning | trajectory_selected | 8 | 2.00 | 0.320 | 0.659 | 0.405 |
| planning | repair_selected | 8 | 2.38 | 0.338 | 0.654 | 0.417 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_081 | random_32 | False | outside_repairable_band | False |  | 0.137 | 0.117 | 87 | True | 12 | 0.188 | False | False | none | none | none | 0.188 |
| llada-moe-7b-a1b-instruct-hf | plan_082 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 52 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_083 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 2 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_084 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.383 | 0.303 | 351 | True | 3 | 0.786 | True | True | 6.000 | 0.188 | 0.429 | 0.429 |
| llada-moe-7b-a1b-instruct-hf | plan_085 | random_32 | False | outside_repairable_band | False |  | 0.045 | 0.045 | 4 | True | 12 | 0.000 | False | False | none | none | none | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_086 | random_32 | False | outside_repairable_band | False |  | 0.065 | 0.045 | 47 | True | 10 | 0.357 | False | False | none | none | none | 0.357 |
| llada-moe-7b-a1b-instruct-hf | plan_087 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.263 | 0.223 | 324 | True | 5 | 0.722 | True | True | 10.000 | 0.312 | 0.444 | 0.444 |
| llada-moe-7b-a1b-instruct-hf | plan_088 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.260 | 352 | True | 4 | 0.733 | True | True | 8.000 | 0.250 | 0.400 | 0.400 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 3 | 1 | low_confidence_32 | final | 22.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.052 | 0.052 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/0/1 | 0.360 | 0.656 | 0.434 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_084 | False | low_confidence_32 | 1.785 | 0.166 | 1.000 | 0.000 | 0.429 | False | This helps validate the the threshold's robustness and ensures it performs consistently... |
| llada-moe-7b-a1b-instruct-hf | plan_087 | False | low_confidence_32 | 2.049 | 0.690 | 1.000 | 0.000 | 0.222 | False | This table will systematically compare signals under varying conditions to determine wh... |
| llada-moe-7b-a1b-instruct-hf | plan_088 | True | low_confidence_32 | 3.351 | 0.968 | 1.000 | 0.000 | 0.067 | False | This involves re-running tests, refining validation rules, and ensuring alignment with... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_081 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.356 | 0.000 | 0.214 | 0.000 | 0.314 | 0.137 | 0.314 | 0.000 | 0.314 | 0.000 | 0.314 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_082 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.419 | 0.000 | 0.223 | 0.000 | 0.303 | 0.045 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_083 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.210 | 0.000 | 0.201 | 0.000 | 0.301 | 0.045 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_084 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.424 | 0.000 | 0.382 | 0.000 | 0.383 | 0.383 | 0.383 | 0.000 | 0.383 | 0.000 | 0.421 | 0.038 |
| llada-moe-7b-a1b-instruct-hf | plan_085 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.321 | 0.000 | 0.273 | 0.000 | 0.353 | 0.045 | 0.353 | 0.000 | 0.353 | 0.000 | 0.353 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_086 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.497 | 0.000 | 0.399 | 0.000 | 0.366 | 0.065 | 0.366 | 0.000 | 0.366 | 0.000 | 0.366 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_087 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.398 | 0.000 | 0.223 | 0.000 | 0.263 | 0.263 | 0.263 | 0.000 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_088 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.433 | 0.000 | 0.479 | 0.219 | 0.280 | 0.280 | 0.280 | 0.000 | 0.419 | 0.139 | 0.419 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
