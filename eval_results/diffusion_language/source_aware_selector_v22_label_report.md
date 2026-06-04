# Diffusion Schedule-Selection Benchmark Report

Full model generations: `32`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-2034a90bbe1f25f6`
Content hash: `2034a90bbe1f25f618f3b2132df92a375076031bfd99c7e03098a6cacf1b9c2a`
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
History mutability: `monotonic 32/32, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `True`
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
Repair selector: `generated_repair_value_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.005`
Trajectory task delta vs random: `0.010`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `1/9/1`
Oracle generation budget/task: `2.91`
Oracle task score: `0.467`
Oracle headroom vs trajectory: `0.028`
Oracle wins/ties/losses vs trajectory: `4/7/0`
Selector regret vs trajectory: `0.028 over 4/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.027`
Repair task delta vs random: `0.034`
Repair task delta vs trajectory: `0.021`
Repair task delta vs evolved: `0.021`
Repair generation budget delta vs evolved: `1.25`
Repair task delta per extra generation vs evolved: `0.016`
Repair wins/ties/losses vs evolved: `3/4/1`
Oracle headroom vs repair: `0.018`
Oracle wins/ties/losses vs repair: `2/6/0`
Selector regret vs repair: `0.018 over 2/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.345670 | 0.000000 | 0.007027 | - | - |
| random perturbation | repair-covered tasks | 0.338643 | -0.007027 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.373080 | 0.027411 | 0.034438 | 4/3/1 | 4/3/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.433 | 0.504 | 0.451 |
| random | 11 | 1.00 | 0.428 | 0.460 | 0.436 |
| trajectory_selected | 11 | 2.00 | 0.438 | 0.496 | 0.453 |
| repair_selected | 8 | 3.25 | 0.373 | 0.663 | 0.445 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.346 | 0.659 | 0.424 |
| planning | random | 8 | 1.00 | 0.339 | 0.598 | 0.404 |
| planning | trajectory_selected | 8 | 2.00 | 0.353 | 0.648 | 0.427 |
| planning | repair_selected | 8 | 3.25 | 0.373 | 0.663 | 0.445 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_169 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.331 | 0.251 | 408 | True | 4 | 0.789 | True | True | 7.000 | 0.219 | 0.474 | 0.474 |
| llada-moe-7b-a1b-instruct-hf | plan_170 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.295 | 0.235 | 366 | True | 3 | 0.769 | True | True | 7.000 | 0.219 | 0.538 | 0.538 |
| llada-moe-7b-a1b-instruct-hf | plan_171 | low_confidence_32 | False | outside_repairable_band | False |  | 0.365 | 0.223 | 362 | True | 1 | 0.947 | False | True | 7.000 | 0.219 | 0.474 | 0.474 |
| llada-moe-7b-a1b-instruct-hf | plan_172 | low_confidence_32 | False | outside_repairable_band | False |  | 0.468 | 0.408 | 394 | True | 0 | 1.000 | False | True | 6.000 | 0.188 | 0.500 | 0.500 |
| llada-moe-7b-a1b-instruct-hf | plan_173 | random_32 | False | outside_repairable_band | False |  | 0.413 | 0.333 | 274 | True | 10 | 0.438 | False | True | 27.000 | 0.844 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_174 | random_32 | True | denoise_phase_repairable | False |  | 0.295 | 0.235 | 249 | True | 4 | 0.733 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_175 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 404 | True | 5 | 0.750 | True | True | 7.000 | 0.219 | 0.438 | 0.438 |
| llada-moe-7b-a1b-instruct-hf | plan_176 | random_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.160 | 107 | True | 6 | 0.647 | True | True | 26.000 | 0.812 | 0.471 | 0.471 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 5 | 3 | low_confidence_32,random_32 | final | 31.4 | 1.000 | 0.000 | 0.000 | 0.024 | 0.024 | 0.028 | 0.041 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3/2/0 | 0.333 | 0.650 | 0.413 |
| history_prefix_25_repair | 5 | 1 | low_confidence_32,random_32 | history | 48.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/2/2 | 0.304 | 0.688 | 0.400 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_169 | True | low_confidence_32 | 1.416 | 0.925 | 1.000 | 0.000 | 0.158 | False | Useroduce a filtering mechanism to exclude false positives by analyzing historical data... |
| llada-moe-7b-a1b-instruct-hf | plan_169 | True | low_confidence_32 | 1.293 | 0.646 | 1.000 | 0.000 | 0.158 | False | Implement a selection algorithm that prioritizes valid and reliable rows from the repai... |
| llada-moe-7b-a1b-instruct-hf | plan_169 | True | low_confidence_32 | 2.178 | 0.968 | 1.000 | 0.000 | 0.211 | False | Ensure continuous testing and feedback loops to maintain availability while minimizing... |
| llada-moe-7b-a1b-instruct-hf | plan_170 | True | low_confidence_32 | 1.893 | 0.402 | 1.000 | 0.000 | 0.385 | False | It ensures repair is applied only when it, rather than assumptions, improves the accura... |
| llada-moe-7b-a1b-instruct-hf | plan_174 | False | random_32 | 1.998 | 0.527 | 1.000 | 0.000 | 0.133 | False | The documentation boundary is clearly defined to ensure consistency and traceability. |
| llada-moe-7b-a1b-instruct-hf | plan_175 | True | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | Implement on-demand production, efficient inventory management, and improve demand fore... |
| llada-moe-7b-a1b-instruct-hf | plan_175 | True | low_confidence_32 | 1.904 | 0.411 | 1.000 | 0.000 | 0.375 | False | Focus on high-value outputs and adjust production processes to maximize revenue while r... |
| llada-moe-7b-a1b-instruct-hf | plan_176 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Choose between history-prefix and final-preserve candidates based on the replay surface... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_169 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.435 | 0.000 | 0.040 | 0.040 | 0.331 | 0.331 | 0.331 | 0.000 | 0.408 | 0.076 | 0.408 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_170 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.417 | 0.000 | 0.033 | 0.033 | 0.295 | 0.295 | 0.295 | 0.000 | 0.304 | 0.009 | 0.304 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_171 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.472 | 0.000 | 0.000 | 0.000 | 0.365 | 0.365 | 0.365 | 0.000 | 0.365 | 0.000 | 0.365 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_172 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.570 | 0.000 | 0.000 | 0.000 | 0.468 | 0.468 | 0.468 | 0.000 | 0.468 | 0.000 | 0.468 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_173 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.399 | 0.000 | 0.000 | 0.000 | 0.304 | 0.413 | 0.304 | 0.000 | 0.304 | 0.000 | 0.413 | 0.109 |
| llada-moe-7b-a1b-instruct-hf | plan_174 | low_confidence_32 | random_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.407 | 0.000 | 0.000 | 0.000 | 0.240 | 0.295 | 0.295 | 0.000 | 0.295 | 0.000 | 0.295 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_175 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.402 | 0.000 | 0.162 | 0.162 | 0.323 | 0.323 | 0.323 | 0.000 | 0.440 | 0.118 | 0.440 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_176 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.472 | 0.000 | 0.202 | 0.202 | 0.439 | 0.220 | 0.439 | 0.000 | 0.400 | -0.039 | 0.439 | 0.039 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
