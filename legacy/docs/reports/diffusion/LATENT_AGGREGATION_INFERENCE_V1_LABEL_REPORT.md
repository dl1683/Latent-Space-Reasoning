# Diffusion Schedule-Selection Benchmark Report

Full model generations: `112`
Counterfactual probe generations: `0`
Arm selections: `112`
Run ID: `diffusion-5825f069ef133e5b`
Content hash: `5825f069ef133e5bd9dc82c3dd085353d83006264fffd83aed5c5f7faa087457`
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
History mutability: `monotonic 112/112, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `True`
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
Repair source min chars: `320`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `999`
Repair source prompt coverage band: `0.000-1.000`
Repair value-proxy source-quality max: `0.310`
Repair cost penalty lambda: `0.180`
Repair transfer source-task min: `0.2954`
Repair phase budget: `custom`
Repair denoise skeleton max step: `none`
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
Trajectory task delta vs fixed: `-0.002`
Trajectory task delta vs random: `0.014`
Trajectory wins/ties/losses vs fixed: `4/24/4`
Trajectory wins/ties/losses vs random: `9/20/3`
Oracle generation budget/task: `3.50`
Oracle task score: `0.236`
Oracle headroom vs trajectory: `0.047`
Oracle wins/ties/losses vs trajectory: `21/11/0`
Selector regret vs trajectory: `0.047 over 21/32 improvable`
Repair arm coverage: `16/32` overall
Repair eligible coverage: `16/16`
Repair task delta vs fixed: `0.026`
Repair task delta vs random: `0.059`
Repair task delta vs trajectory: `0.024`
Repair task delta vs evolved: `0.024`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.012`
Repair wins/ties/losses vs evolved: `8/6/2`
Oracle headroom vs repair: `0.011`
Oracle wins/ties/losses vs repair: `8/8/0`
Selector regret vs repair: `0.011 over 8/16 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `16/32` overall, `16/16` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.337031 | 0.000000 | 0.032750 | - | - |
| random perturbation | repair-covered tasks | 0.304281 | -0.032750 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.363438 | 0.026406 | 0.059156 | 7/7/2 | 11/3/2 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 32 | 1.00 | 0.191 | 0.443 | 0.254 |
| random | 32 | 1.00 | 0.175 | 0.411 | 0.234 |
| trajectory_selected | 32 | 2.50 | 0.189 | 0.435 | 0.251 |
| repair_selected | 16 | 4.00 | 0.363 | 0.676 | 0.442 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 32 | 1.00 | 0.191 | 0.443 | 0.254 |
| planning | random | 32 | 1.00 | 0.175 | 0.411 | 0.234 |
| planning | trajectory_selected | 32 | 2.50 | 0.189 | 0.435 | 0.251 |
| planning | repair_selected | 16 | 4.00 | 0.363 | 0.676 | 0.442 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_009 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 274 | True | 11 | 0.412 | True | True | 4.000 | 0.125 | 0.118 | 0.118 |
| llada-8b-instruct-hf | plan_010 | random_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 269 | True | 6 | 0.625 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_011 | random_32 | True | denoise_phase_repairable | False |  | 0.336 | 0.239 | 292 | True | 10 | 0.412 | True | True | 4.000 | 0.125 | 0.059 | 0.059 |
| llada-8b-instruct-hf | plan_012 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.427 | 0.367 | 345 | True | 6 | 0.647 | True | True | 4.000 | 0.125 | 0.118 | 0.118 |
| llada-8b-instruct-hf | plan_013 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.310 | 0.230 | 315 | True | 9 | 0.500 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_014 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 300 | True | 7 | 0.588 | True | True | 3.000 | 0.094 | 0.118 | 0.118 |
| llada-8b-instruct-hf | plan_015 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.451 | 0.276 | 386 | True | 6 | 0.600 | True | True | 2.000 | 0.062 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_016 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.364 | 0.324 | 358 | True | 10 | 0.375 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_017 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.331 | 0.251 | 353 | True | 7 | 0.611 | True | True | 4.000 | 0.125 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_018 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 276 | True | 12 | 0.278 | True | True | 4.000 | 0.125 | 0.056 | 0.056 |
| llada-8b-instruct-hf | plan_019 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.405 | 0.345 | 352 | True | 8 | 0.529 | True | True | 4.000 | 0.125 | 0.059 | 0.059 |
| llada-8b-instruct-hf | plan_020 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.302 | 0.223 | 327 | True | 4 | 0.750 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_021 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 341 | True | 7 | 0.533 | True | True | 4.000 | 0.125 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_022 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 354 | True | 11 | 0.450 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_023 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.338 | 0.235 | 325 | True | 2 | 0.875 | True | True | 4.000 | 0.125 | 0.188 | 0.188 |
| llada-8b-instruct-hf | plan_024 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.433 | 0.353 | 337 | True | 9 | 0.471 | True | True | 4.000 | 0.125 | 0.059 | 0.059 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 16 | 5 | low_confidence_32,random_32 | final | 31.2 | 1.000 | 0.000 | 0.000 | 0.007 | 0.007 | 0.012 | 0.019 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/2/5 | 0.359 | 0.668 | 0.436 |
| history_prefix_25_repair | 16 | 5 | low_confidence_32,random_32 | history | 48.1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6/6/4 | 0.347 | 0.677 | 0.429 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_009 | True | low_confidence_32 | 1.291 | 0.609 | 1.000 | 0.000 | 0.118 | False | Review the conflicting citation and identify the discrepancy. |
| llada-8b-instruct-hf | plan_009 | True | low_confidence_32 | 2.078 | 0.910 | 1.000 | 0.000 | 0.059 | False | Evaluate the impact of the discrepancy on the answer's accuracy and reliability. |
| llada-8b-instruct-hf | plan_009 | True | low_confidence_32 | 2.158 | 0.910 | 1.000 | 0.000 | 0.294 | False | Decide whether to ship, patch, or hold the answer. |
| llada-8b-instruct-hf | plan_010 | False | random_32 | 1.380 | 0.833 | 1.000 | 0.000 | 0.188 | False | Record the original random seed and test order. |
| llada-8b-instruct-hf | plan_010 | False | random_32 | 1.278 | 0.578 | 1.000 | 0.000 | 0.125 | False | Reapply the dependency upgrade. |
| llada-8b-instruct-hf | plan_010 | False | random_32 | 2.040 | 0.639 | 1.000 | 0.000 | 0.125 | False | Compare the results with the original results to confirm the gain is real. |
| llada-8b-instruct-hf | plan_011 | True | random_32 | 2.488 | 0.713 | 1.000 | 0.000 | 0.059 | False | Collect and log data to identify blocked requests. |
| llada-8b-instruct-hf | plan_011 | True | random_32 | 2.549 | 0.820 | 1.000 | 0.000 | 0.059 | False | Implement a process to review blocked requests. |
| llada-8b-instruct-hf | plan_011 | True | random_32 | 3.231 | 0.689 | 1.000 | 0.000 | 0.059 | False | Continuously monitor the filter and its effectiveness. |
| llada-8b-instruct-hf | plan_012 | False | low_confidence_32 | 2.155 | 0.893 | 1.000 | 0.000 | 0.118 | False | Measure the accuracy and accuracy of the answers to determine if the compression outwei... |
| llada-8b-instruct-hf | plan_013 | False | low_confidence_32 | 1.421 | 0.910 | 1.000 | 0.000 | 0.167 | False | Analyzing the prompts written by the contractor. |
| llada-8b-instruct-hf | plan_013 | False | low_confidence_32 | 2.051 | 0.835 | 1.000 | 0.000 | 0.056 | False | Identifying any unique factors that contributed to the improvement. |
| llada-8b-instruct-hf | plan_013 | False | low_confidence_32 | 1.949 | 0.433 | 1.000 | 0.000 | 0.222 | False | Reporting the findings to decide whether to cite the gain publicly. |
| llada-8b-instruct-hf | plan_014 | True | random_32 | 1.311 | 0.654 | 1.000 | 0.000 | 0.118 | False | Verify the final answer correctness. |
| llada-8b-instruct-hf | plan_014 | True | random_32 | 1.368 | 0.795 | 1.000 | 0.000 | 0.176 | False | Review the intermediate assumptions hidden in the trace. |
| llada-8b-instruct-hf | plan_014 | True | random_32 | 2.135 | 0.842 | 1.000 | 0.000 | 0.235 | False | If the assumptions are unsafe, the trace should not be trusted. |
| llada-8b-instruct-hf | plan_015 | False | low_confidence_32 | 1.903 | 0.875 | 1.000 | 0.000 | 0.133 | False | Implement a robust verification process for high-priority tasks and use simplified veri... |
| llada-8b-instruct-hf | plan_015 | False | low_confidence_32 | 1.753 | 0.123 | 1.000 | 0.000 | 0.467 | False | Regularly review and update the verification policy to ensure it effectively reduces la... |
| llada-8b-instruct-hf | plan_016 | False | low_confidence_32 | 1.466 | 1.000 | 1.000 | 0.000 | 0.188 | False | If the number of hallucinated results decreases,, the the experiment is considered succ... |
| llada-8b-instruct-hf | plan_016 | False | low_confidence_32 | 2.147 | 0.863 | 1.000 | 0.000 | 0.125 | False | If the number increases or remains unchanged, proceed to the second experiment with the... |
| llada-8b-instruct-hf | plan_017 | False | low_confidence_32 | 1.358 | 0.835 | 1.000 | 0.000 | 0.278 | False | If the benchmark fails without the hidden examples, it suggests the improvement was due... |
| llada-8b-instruct-hf | plan_017 | False | low_confidence_32 | 2.097 | 0.829 | 1.000 | 0.000 | 0.333 | False | If the benchmark passes without the hidden examples, then it indicates the improvement... |
| llada-8b-instruct-hf | plan_018 | True | low_confidence_32 | 2.481 | 0.668 | 1.000 | 0.000 | 0.000 | False | Compare the results and choose the best approach. |
| llada-8b-instruct-hf | plan_018 | True | low_confidence_32 | 3.281 | 0.775 | 1.000 | 0.000 | 0.000 | False | Implement the chosen and document the new results. |
| llada-8b-instruct-hf | plan_019 | False | low_confidence_32 | 2.127 | 0.868 | 1.000 | 0.000 | 0.176 | False | Additionally, compare the length of the answers generated by the repair system to the l... |
| llada-8b-instruct-hf | plan_020 | False | low_confidence_32 | 2.098 | 0.787 | 1.000 | 0.000 | 0.250 | False | This approach allows for a balanced assessment of the schedule's performance across dif... |
| llada-8b-instruct-hf | plan_021 | True | low_confidence_32 | 1.324 | 0.786 | 1.000 | 0.000 | 0.267 | False | Compare the judge's scores to a predefined threshold and verify the coherence and compl... |
| llada-8b-instruct-hf | plan_021 | True | low_confidence_32 | 2.056 | 0.750 | 1.000 | 0.000 | 0.400 | False | Ensure that high-scoring answers are coherent and that all checklist items are strictly... |
| llada-8b-instruct-hf | plan_022 | False | low_confidence_32 | 2.014 | 0.661 | 1.000 | 0.000 | 0.350 | False | Validation can be done by comparing the logged intermediate states with the model's fin... |
| llada-8b-instruct-hf | plan_023 | False | low_confidence_32 | 2.165 | 1.000 | 1.000 | 0.000 | 0.375 | False | This ensures that the larger model's initial accuracy is not outweighed by the smaller... |
| llada-8b-instruct-hf | plan_024 | False | low_confidence_32 | 1.393 | 0.869 | 1.000 | 0.000 | 0.118 | False | Monitor the model's behavior on these cases to ensure it does not inadvertently reject... |
| llada-8b-instruct-hf | plan_024 | False | low_confidence_32 | 2.070 | 0.729 | 1.000 | 0.000 | 0.176 | False | Document the results and make any necessary adjustments to the model rule to address th... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_009 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_010 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.137 | 0.000 |
| dream-7b-instruct-hf | plan_011 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_012 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.200 | 0.000 |
| dream-7b-instruct-hf | plan_013 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.117 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_014 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_015 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_016 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_017 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_018 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.200 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.200 | 0.000 |
| dream-7b-instruct-hf | plan_019 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_020 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_021 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_022 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.125 | 0.000 | 0.000 | 0.000 | 0.045 | 0.180 | 0.045 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_023 | entropy_32 | entropy_64 | entropy_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_024 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_009 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.304 | 0.000 | 0.102 | 0.102 | 0.260 | 0.260 | 0.260 | 0.000 | 0.356 | 0.096 | 0.356 | 0.000 |
| llada-8b-instruct-hf | plan_010 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.383 | 0.000 | 0.085 | 0.085 | 0.338 | 0.304 | 0.304 | 0.000 | 0.359 | 0.055 | 0.359 | 0.000 |
| llada-8b-instruct-hf | plan_011 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.323 | 0.000 | 0.065 | 0.065 | 0.243 | 0.243 | 0.336 | 0.000 | 0.411 | 0.075 | 0.411 | 0.000 |
| llada-8b-instruct-hf | plan_012 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.451 | 0.000 | 0.000 | 0.000 | 0.427 | 0.427 | 0.427 | 0.000 | 0.427 | 0.000 | 0.447 | 0.020 |
| llada-8b-instruct-hf | plan_013 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.353 | 0.000 | 0.047 | 0.047 | 0.310 | 0.304 | 0.310 | 0.000 | 0.291 | -0.019 | 0.310 | 0.019 |
| llada-8b-instruct-hf | plan_014 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.341 | 0.000 | 0.042 | 0.042 | 0.281 | 0.260 | 0.260 | 0.000 | 0.281 | 0.021 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_015 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.395 | 0.000 | 0.000 | 0.000 | 0.451 | 0.473 | 0.451 | 0.000 | 0.451 | 0.000 | 0.473 | 0.021 |
| llada-8b-instruct-hf | plan_016 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.361 | 0.000 | 0.000 | 0.000 | 0.364 | 0.351 | 0.364 | 0.000 | 0.364 | 0.000 | 0.364 | 0.000 |
| llada-8b-instruct-hf | plan_017 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.387 | 0.000 | 0.000 | 0.000 | 0.331 | 0.331 | 0.331 | 0.000 | 0.331 | 0.000 | 0.374 | 0.043 |
| llada-8b-instruct-hf | plan_018 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.291 | 0.000 | 0.040 | 0.040 | 0.303 | 0.178 | 0.303 | 0.000 | 0.299 | -0.004 | 0.303 | 0.004 |
| llada-8b-instruct-hf | plan_019 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.424 | 0.000 | 0.000 | 0.000 | 0.405 | 0.405 | 0.405 | 0.000 | 0.405 | 0.000 | 0.413 | 0.008 |
| llada-8b-instruct-hf | plan_020 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.395 | 0.000 | 0.000 | 0.000 | 0.302 | 0.157 | 0.302 | 0.000 | 0.302 | 0.000 | 0.302 | 0.000 |
| llada-8b-instruct-hf | plan_021 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.364 | 0.000 | 0.038 | 0.038 | 0.344 | 0.238 | 0.344 | 0.000 | 0.356 | 0.012 | 0.356 | 0.000 |
| llada-8b-instruct-hf | plan_022 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.317 | 0.000 | 0.075 | 0.075 | 0.261 | 0.261 | 0.261 | 0.000 | 0.331 | 0.070 | 0.366 | 0.035 |
| llada-8b-instruct-hf | plan_023 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.453 | 0.000 | 0.047 | 0.047 | 0.338 | 0.338 | 0.338 | 0.000 | 0.339 | 0.001 | 0.359 | 0.020 |
| llada-8b-instruct-hf | plan_024 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.388 | 0.000 | 0.100 | 0.100 | 0.433 | 0.336 | 0.433 | 0.000 | 0.508 | 0.075 | 0.508 | 0.000 |
