# Diffusion Schedule-Selection Benchmark Report

Full model generations: `29`
Counterfactual probe generations: `0`
Arm selections: `41`
Run ID: `diffusion-532134d37927787e`
Content hash: `532134d37927787ed2ff3092f8529276307682efd2dba217b7087ca63cb416c3`
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
History mutability: `monotonic 29/29, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `constraint_span_phase_final_preserve_seeded_gated`
Repair source policy: `evolved`
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
Trajectory task delta vs fixed: `0.003`
Trajectory task delta vs random: `0.045`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/7/1`
Oracle generation budget/task: `2.64`
Oracle task score: `0.476`
Oracle headroom vs trajectory: `0.056`
Oracle wins/ties/losses vs trajectory: `6/5/0`
Selector regret vs trajectory: `0.056 over 6/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/9`
Repair task delta vs fixed: `0.078`
Repair task delta vs random: `0.135`
Repair task delta vs trajectory: `0.074`
Repair task delta vs evolved: `0.074`
Repair generation budget delta vs evolved: `0.88`
Repair task delta per extra generation vs evolved: `0.084`
Repair wins/ties/losses vs evolved: `5/3/0`
Oracle headroom vs repair: `0.003`
Oracle wins/ties/losses vs repair: `2/6/0`
Selector regret vs repair: `0.003 over 2/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/9` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.323929 | 0.000000 | 0.057339 | - | - |
| random perturbation | repair-covered tasks | 0.266589 | -0.057339 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.401964 | 0.078036 | 0.135375 | 6/2/0 | 4/3/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.417 | 0.504 | 0.439 |
| random | 11 | 1.00 | 0.376 | 0.406 | 0.383 |
| trajectory_selected | 11 | 2.00 | 0.421 | 0.504 | 0.442 |
| repair_selected | 8 | 2.88 | 0.402 | 0.666 | 0.468 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | random | 1 | 1.00 | 1.000 | 0.015 | 0.754 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.015 | 0.754 |
| planning | fixed | 8 | 1.00 | 0.324 | 0.659 | 0.408 |
| planning | random | 8 | 1.00 | 0.267 | 0.524 | 0.331 |
| planning | trajectory_selected | 8 | 2.00 | 0.328 | 0.659 | 0.411 |
| planning | repair_selected | 8 | 2.88 | 0.402 | 0.666 | 0.468 |
| science | fixed | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | random | 1 | 1.00 | 1.000 | 0.243 | 0.811 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.243 | 0.811 |
| symbolic | fixed | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | random | 1 | 1.00 | 0.000 | 0.016 | 0.004 |
| symbolic | trajectory_selected | 1 | 2.00 | 0.000 | 0.016 | 0.004 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_073 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.356 | 0.256 | 383 | True | 2 | 0.882 | True | True | 9.000 | 0.281 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_074 | random_32 | True | denoise_phase_repairable | False |  | 0.465 | 0.315 | 342 | True | 3 | 0.769 | True | True | 16.000 | 0.500 | 0.538 | 0.538 |
| llada-moe-7b-a1b-instruct-hf | plan_075 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.336 | 0.256 | 342 | True | 8 | 0.467 | True | True | 16.000 | 0.500 | 0.400 | 0.400 |
| llada-moe-7b-a1b-instruct-hf | plan_076 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.300 | 0.260 | 380 | True | 5 | 0.706 | True | True | 8.000 | 0.250 | 0.412 | 0.412 |
| llada-moe-7b-a1b-instruct-hf | plan_077 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.243 | 0.223 | 387 | True | 5 | 0.667 | True | True | 13.000 | 0.406 | 0.467 | 0.467 |
| llada-moe-7b-a1b-instruct-hf | plan_078 | low_confidence_32 | False | outside_repairable_band | False |  | 0.287 | 0.247 | 395 | True | 1 | 0.929 | False | True | 6.000 | 0.188 | 0.571 | 0.571 |
| llada-moe-7b-a1b-instruct-hf | plan_079 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 321 | True | 4 | 0.636 | True | True | 15.000 | 0.469 | 0.455 | 0.455 |
| llada-moe-7b-a1b-instruct-hf | plan_080 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.235 | 368 | True | 2 | 0.867 | True | True | 5.000 | 0.156 | 0.400 | 0.400 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 7 | 5 | low_confidence_32,random_32 | final | 31.1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.069 | 0.084 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6/0/1 | 0.418 | 0.665 | 0.480 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_073 | True | low_confidence_32 | 3.199 | 0.831 | 1.000 | 0.000 | 0.294 | False | This involves verifying that the signatures are authentic and that the source score is... |
| llada-moe-7b-a1b-instruct-hf | plan_074 | False | random_32 | 2.543 | 1.000 | 1.000 | 0.000 | 0.231 | False | Ensure the cost is included in the overall budget and consider how it affects the cost... |
| llada-moe-7b-a1b-instruct-hf | plan_074 | False | random_32 | 3.230 | 0.869 | 1.000 | 0.000 | 0.154 | False | Monitor the and adjust the budget accordingly to reflect the new budget story. |
| llada-moe-7b-a1b-instruct-hf | plan_075 | False | low_confidence_32 | 2.176 | 0.936 | 1.000 | 0.000 | 0.133 | False | This the run will help ensure that the feature is not only effective on old prompts but... |
| llada-moe-7b-a1b-instruct-hf | plan_076 | True | low_confidence_32 | 1.462 | 1.000 | 1.000 | 0.000 | 0.176 | False | Use this data to to evaluate the controller's ability to make timely and accurate decis... |
| llada-moe-7b-a1b-instruct-hf | plan_076 | True | low_confidence_32 | 2.143 | 0.869 | 1.000 | 0.000 | 0.118 | False | Monitor the controller's performance and make necessary adjustments to the evaluation a... |
| llada-moe-7b-a1b-instruct-hf | plan_077 | True | low_confidence_32 | 3.923 | 0.905 | 1.000 | 0.000 | 0.067 | False | Each outcome should be assessed with associated evidence, confidence,, research implica... |
| llada-moe-7b-a1b-instruct-hf | plan_079 | True | low_confidence_32 | 2.014 | 0.614 | 1.000 | 0.000 | 0.273 | False | Highlight the nature of the evidence,, the scope, and the limitations to avoid overclai... |
| llada-moe-7b-a1b-instruct-hf | plan_080 | True | low_confidence_32 | 1.437 | 1.000 | 1.000 | 0.000 | 0.333 | False | Commit all uncommitted benchmark artifacts to the main repository.2. |
| llada-moe-7b-a1b-instruct-hf | plan_080 | True | low_confidence_32 | 1.472 | 1.000 | 1.000 | 0.000 | 0.200 | False | Complete the pending GPU validation by running the validation validation script.3. |
| llada-moe-7b-a1b-instruct-hf | plan_080 | True | low_confidence_32 | 1.924 | 0.485 | 1.000 | 0.000 | 0.400 | False | Verify the repository of all artifacts and artifacts before running more experiments. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_073 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.437 | 0.000 | 0.404 | 0.147 | 0.356 | 0.045 | 0.356 | 0.000 | 0.415 | 0.059 | 0.415 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_074 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.442 | 0.000 | 0.392 | 0.000 | 0.429 | 0.465 | 0.465 | 0.000 | 0.465 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_075 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.357 | 0.000 | 0.256 | 0.000 | 0.336 | 0.336 | 0.336 | 0.000 | 0.336 | 0.000 | 0.356 | 0.020 |
| llada-moe-7b-a1b-instruct-hf | plan_076 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.395 | 0.000 | 0.493 | 0.233 | 0.300 | 0.180 | 0.300 | 0.000 | 0.517 | 0.216 | 0.517 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_077 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.398 | 0.000 | 0.414 | 0.191 | 0.243 | 0.138 | 0.243 | 0.000 | 0.400 | 0.157 | 0.400 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_078 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.479 | 0.000 | 0.247 | 0.000 | 0.287 | 0.287 | 0.287 | 0.000 | 0.287 | 0.000 | 0.287 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_079 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.386 | 0.000 | 0.407 | 0.163 | 0.324 | 0.324 | 0.324 | 0.000 | 0.442 | 0.118 | 0.442 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_080 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_candidate_aware_promotion_v1_score_repair_pool | low_confidence_32 | final |  | 0.437 | 0.000 | 0.273 | 0.038 | 0.315 | 0.356 | 0.315 | 0.000 | 0.353 | 0.037 | 0.356 | 0.004 |
| llada-moe-7b-a1b-instruct-hf | sci_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | low_confidence_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.243 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  |  | random_32 | fixed_exact_answer_guard |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
