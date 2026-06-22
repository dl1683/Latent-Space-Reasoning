# Diffusion Schedule-Selection Benchmark Report

Full model generations: `166`
Counterfactual probe generations: `0`
Arm selections: `168`
Run ID: `diffusion-55950d45936e2c0a`
Content hash: `55950d45936e2c0aeaf68844b135346b00e3ac3cb535102d877bb07c231c5ccd`
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
History mutability: `monotonic 166/166, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.008`
Trajectory task delta vs random: `0.034`
Trajectory wins/ties/losses vs fixed: `9/38/1`
Trajectory wins/ties/losses vs random: `19/27/2`
Oracle generation budget/task: `3.46`
Oracle task score: `0.207`
Oracle headroom vs trajectory: `0.018`
Oracle wins/ties/losses vs trajectory: `17/31/0`
Selector regret vs trajectory: `0.018 over 17/48 improvable`
Repair arm coverage: `24/48` overall
Repair eligible coverage: `24/24`
Repair task delta vs fixed: `0.042`
Repair task delta vs random: `0.086`
Repair task delta vs trajectory: `0.033`
Repair task delta vs evolved: `0.033`
Repair generation budget delta vs evolved: `1.92`
Repair task delta per extra generation vs evolved: `0.017`
Repair wins/ties/losses vs evolved: `14/10/0`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `2/22/0`
Selector regret vs repair: `0.002 over 2/24 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `24/48` overall, `24/24` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.336354 | 0.000000 | 0.043818 | - | - |
| random perturbation | repair-covered tasks | 0.292536 | -0.043818 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.378586 | 0.042232 | 0.086051 | 15/8/1 | 19/4/1 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.180 | 0.399 | 0.235 |
| random | 48 | 1.00 | 0.155 | 0.336 | 0.200 |
| trajectory_selected | 48 | 2.50 | 0.189 | 0.400 | 0.242 |
| repair_selected | 24 | 3.92 | 0.379 | 0.645 | 0.445 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.180 | 0.399 | 0.235 |
| planning | random | 48 | 1.00 | 0.155 | 0.336 | 0.200 |
| planning | trajectory_selected | 48 | 2.50 | 0.189 | 0.400 | 0.242 |
| planning | repair_selected | 24 | 3.92 | 0.379 | 0.645 | 0.445 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_025 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.413 | 0.353 | 344 | True | 9 | 0.571 | True | True | 4.000 | 0.125 | 0.095 | 0.095 |
| llada-8b-instruct-hf | plan_026 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.358 | 0.278 | 353 | True | 9 | 0.471 | True | True | 4.000 | 0.125 | 0.118 | 0.118 |
| llada-8b-instruct-hf | plan_027 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.399 | 0.299 | 356 | True | 8 | 0.588 | True | True | 4.000 | 0.125 | 0.176 | 0.176 |
| llada-8b-instruct-hf | plan_028 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.538 | 0.401 | 386 | True | 8 | 0.500 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_029 | random_32 | True | denoise_phase_repairable | False |  | 0.440 | 0.340 | 336 | True | 10 | 0.474 | True | True | 3.000 | 0.094 | 0.053 | 0.053 |
| llada-8b-instruct-hf | plan_030 | random_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 283 | True | 12 | 0.333 | True | True | 4.000 | 0.125 | 0.056 | 0.056 |
| llada-8b-instruct-hf | plan_031 | random_32 | True | denoise_phase_repairable | False |  | 0.379 | 0.299 | 291 | True | 5 | 0.688 | True | True | 4.000 | 0.125 | 0.062 | 0.062 |
| llada-8b-instruct-hf | plan_032 | random_32 | True | denoise_phase_repairable | False |  | 0.384 | 0.324 | 260 | True | 11 | 0.353 | True | True | 4.000 | 0.125 | 0.059 | 0.059 |
| llada-8b-instruct-hf | plan_033 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.416 | 0.336 | 355 | True | 2 | 0.867 | True | True | 4.000 | 0.125 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_034 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.325 | 0.223 | 281 | True | 8 | 0.500 | True | True | 5.000 | 0.156 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_035 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 363 | True | 5 | 0.667 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_036 | random_32 | True | denoise_phase_repairable | False |  | 0.371 | 0.311 | 289 | True | 8 | 0.429 | True | True | 3.000 | 0.094 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_037 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.337 | 0.257 | 342 | True | 7 | 0.562 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_038 | random_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 311 | True | 6 | 0.600 | True | True | 5.000 | 0.156 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_039 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.338 | 0.278 | 340 | True | 10 | 0.357 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_040 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 370 | True | 4 | 0.714 | True | True | 3.000 | 0.094 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_041 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.320 | 0.260 | 362 | True | 7 | 0.632 | True | True | 4.000 | 0.125 | 0.105 | 0.105 |
| llada-8b-instruct-hf | plan_042 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 352 | True | 7 | 0.500 | True | True | 5.000 | 0.156 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_043 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.336 | 0.256 | 292 | True | 5 | 0.750 | True | True | 3.000 | 0.094 | 0.150 | 0.150 |
| llada-8b-instruct-hf | plan_044 | low_confidence_32 | False | no_repairable_denoise_skeleton | False |  | 0.065 | 0.045 | 10 | True | 12 | 0.000 | True | False | none | none | none | 0.000 |
| llada-8b-instruct-hf | plan_045 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 350 | True | 8 | 0.385 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_046 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.366 | 0.244 | 338 | True | 2 | 0.857 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_047 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.320 | 0.260 | 300 | True | 7 | 0.588 | True | True | 4.000 | 0.125 | 0.176 | 0.176 |
| llada-8b-instruct-hf | plan_048 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.316 | 0.276 | 391 | True | 7 | 0.562 | True | True | 4.000 | 0.125 | 0.188 | 0.188 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 23 | 7 | low_confidence_32,random_32 | final | 31.7 | 1.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.009 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/6/8 | 0.370 | 0.665 | 0.444 |
| history_prefix_25_repair | 23 | 7 | low_confidence_32,random_32 | history | 48.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/7/7 | 0.365 | 0.688 | 0.445 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_025 | False | low_confidence_32 | 1.300 | 0.700 | 1.000 | 0.000 | 0.238 | False | Track the number of misrouted questions and compare the performance of the stronger mod... |
| llada-8b-instruct-hf | plan_025 | False | low_confidence_32 | 1.964 | 0.494 | 1.000 | 0.000 | 0.190 | False | Adjust will be made based on the quick validation results before expanding the rollout. |
| llada-8b-instruct-hf | plan_026 | False | low_confidence_32 | 1.305 | 0.705 | 1.000 | 0.000 | 0.176 | False | Compare the scores from the new checklist with the original checklist to identify any d... |
| llada-8b-instruct-hf | plan_026 | False | low_confidence_32 | 2.040 | 0.691 | 1.000 | 0.000 | 0.294 | False | Additionally, involve multiple reviewers reviewing the checklist to ensure consistency... |
| llada-8b-instruct-hf | plan_027 | False | low_confidence_32 | 2.121 | 0.843 | 1.000 | 0.000 | 0.176 | False | Analyze the results to determine whether the system is reliable or needs further refine... |
| llada-8b-instruct-hf | plan_028 | False | low_confidence_32 | 1.904 | 0.893 | 1.000 | 0.000 | 0.125 | False | Apply the compressor to each trace and compare the compressed output with the original... |
| llada-8b-instruct-hf | plan_028 | False | low_confidence_32 | 2.592 | 0.735 | 1.000 | 0.000 | 0.125 | False | Ensure this percentage is within acceptable limits before enabling the compressor. |
| llada-8b-instruct-hf | plan_029 | True | random_32 | 1.933 | 0.910 | 1.000 | 0.000 | 0.105 | False | Assess the confidence level of each solution path. |
| llada-8b-instruct-hf | plan_029 | True | random_32 | 1.386 | 0.846 | 1.000 | 0.000 | 0.105 | False | Set a confidence threshold to determine if the planner's confidence is within acceptabl... |
| llada-8b-instruct-hf | plan_029 | True | random_32 | 2.052 | 0.764 | 1.000 | 0.000 | 0.368 | False | Adjust the planner to prioritize solution paths below the acceptable confidence thresho... |
| llada-8b-instruct-hf | plan_030 | False | random_32 | 1.980 | 0.667 | 1.000 | 0.000 | 0.056 | False | Create a dataset of known correct answers. |
| llada-8b-instruct-hf | plan_030 | False | random_32 | 1.973 | 0.668 | 1.000 | 0.000 | 0.056 | False | Measure retrieval accuracy rates in both scenarios. |
| llada-8b-instruct-hf | plan_030 | False | random_32 | 1.977 | 0.483 | 1.000 | 0.000 | 0.167 | False | Analyze results to determine whether to gate retrieval. |
| llada-8b-instruct-hf | plan_031 | False | random_32 | 1.279 | 0.619 | 1.000 | 0.000 | 0.188 | False | Establish a baseline average benchmark score. |
| llada-8b-instruct-hf | plan_031 | False | random_32 | 1.420 | 0.910 | 1.000 | 0.000 | 0.188 | False | Test the optimized prompts. |
| llada-8b-instruct-hf | plan_031 | False | random_32 | 2.032 | 0.697 | 1.000 | 0.000 | 0.438 | False | Conclude if the optimized prompts improve the average score without significant latency... |
| llada-8b-instruct-hf | plan_032 | False | random_32 | 1.973 | 0.668 | 1.000 | 0.000 | 0.059 | False | Compare the diff. of the worst examples. |
| llada-8b-instruct-hf | plan_032 | False | random_32 | 2.037 | 0.775 | 1.000 | 0.000 | 0.000 | False | Assess the impact of the changes. |
| llada-8b-instruct-hf | plan_032 | False | random_32 | 1.916 | 0.388 | 1.000 | 0.000 | 0.294 | False | Decide whether to roll back the tokenizer update. |
| llada-8b-instruct-hf | plan_033 | False | low_confidence_32 | 1.648 | 0.000 | 1.000 | 0.000 | 0.733 | False | This ensures that the model improves planning answers only after the verifier sees inte... |
| llada-8b-instruct-hf | plan_034 | True | low_confidence_32 | 1.407 | 0.865 | 1.000 | 0.000 | 0.125 | False | Review the repair policy's implementation details. |
| llada-8b-instruct-hf | plan_034 | True | low_confidence_32 | 1.304 | 0.627 | 1.000 | 0.000 | 0.125 | False | Assess the impact of context deletion on answer quality. |
| llada-8b-instruct-hf | plan_034 | True | low_confidence_32 | 2.146 | 0.865 | 1.000 | 0.000 | 0.125 | False | Prepare a report outlining the pros and cons of the repair policy. |
| llada-8b-instruct-hf | plan_035 | False | low_confidence_32 | 2.017 | 0.768 | 1.000 | 0.000 | 0.067 | False | Measure latency under controlled conditions to account for any variability. |
| llada-8b-instruct-hf | plan_035 | False | low_confidence_32 | 2.846 | 0.936 | 1.000 | 0.000 | 0.067 | False | Record the same performance metrics for both models to ensure comparability. |
| llada-8b-instruct-hf | plan_036 | False | random_32 | 2.068 | 0.865 | 1.000 | 0.000 | 0.071 | False | Evaluate the new scoring system. |
| llada-8b-instruct-hf | plan_036 | False | random_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Assess the impact on quality and accuracy. |
| llada-8b-instruct-hf | plan_036 | False | random_32 | 2.026 | 0.658 | 1.000 | 0.000 | 0.286 | False | Analyze the evaluation results to conclude if the scoring change is harmful. |
| llada-8b-instruct-hf | plan_037 | False | low_confidence_32 | 2.573 | 0.914 | 1.000 | 0.000 | 0.062 | False | Measure the accuracy of the feature's predictions compared to the baseline baseline. |
| llada-8b-instruct-hf | plan_037 | False | low_confidence_32 | 2.702 | 1.000 | 1.000 | 0.000 | 0.188 | False | If the accuracy is significantly improved, calibrate the feature; otherwise, discard th... |
| llada-8b-instruct-hf | plan_038 | False | random_32 | 1.313 | 0.650 | 1.000 | 0.000 | 0.133 | False | Use audit tools to evaluate intermediate state interpretability. |
| llada-8b-instruct-hf | plan_038 | False | random_32 | 1.268 | 0.564 | 1.000 | 0.000 | 0.133 | False | Record intermediate states to assess traceability. |
| llada-8b-instruct-hf | plan_038 | False | random_32 | 1.981 | 0.480 | 1.000 | 0.000 | 0.133 | False | Analyze results to determine the optimal balance between accuracy and auditability. |
| llada-8b-instruct-hf | plan_039 | True | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | This involves evaluating the model on a new, larger dataset to assess its performance. |
| llada-8b-instruct-hf | plan_039 | True | low_confidence_32 | 2.700 | 0.659 | 1.000 | 0.000 | 0.071 | False | By comparing the model's performance on the training data and the test data, you can de... |
| llada-8b-instruct-hf | plan_040 | True | low_confidence_32 | 1.883 | 0.361 | 1.000 | 0.000 | 0.286 | False | This way, the team can demonstrate the product's capabilities across a variety of possi... |
| llada-8b-instruct-hf | plan_041 | False | low_confidence_32 | 2.136 | 0.893 | 1.000 | 0.000 | 0.211 | False | Compare the performance with the selected trajectory to confirm the effectiveness of th... |
| llada-8b-instruct-hf | plan_042 | True | low_confidence_32 | 2.162 | 0.905 | 1.000 | 0.000 | 0.143 | False | If the selector performs well across all slices, it indicates that the selector is robu... |
| llada-8b-instruct-hf | plan_043 | False | low_confidence_32 | 1.366 | 0.820 | 1.000 | 0.000 | 0.250 | False | Use the denoise states as reasoning evidence for the judge. |
| llada-8b-instruct-hf | plan_043 | False | low_confidence_32 | 1.997 | 0.613 | 1.000 | 0.000 | 0.300 | False | Compare the scores to assess the effectiveness of using denoise states as reasoning evi... |
| llada-8b-instruct-hf | plan_045 | False | low_confidence_32 | 2.142 | 1.000 | 1.000 | 0.000 | 0.077 | False | If the benefits outweigh the cost, run the schedule schedule. |
| llada-8b-instruct-hf | plan_045 | False | low_confidence_32 | 2.771 | 0.796 | 1.000 | 0.000 | 0.077 | False | Otherwise, consider alternative the schedule or tasks that can be completed within the... |
| llada-8b-instruct-hf | plan_046 | True | low_confidence_32 | 2.144 | 0.973 | 1.000 | 0.000 | 0.500 | False | The falsification would would be to find a set of deno states that do not contain usefu... |
| llada-8b-instruct-hf | plan_047 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To plan a guardrail evaluation, we a) define a set of guardrails that represent correct... |
| llada-8b-instruct-hf | plan_048 | True | low_confidence_32 | 2.800 | 0.869 | 1.000 | 0.000 | 0.062 | False | Communicate the findings to relevant stakeholders and monitor the release closely to en... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_025 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_026 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_027 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_028 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_029 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_030 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_031 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.065 | 0.000 |
| dream-7b-instruct-hf | plan_032 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_033 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_034 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_035 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_036 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_037 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_038 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_039 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.124 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_040 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_041 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_042 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_043 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_044 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_045 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_046 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_047 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_048 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_025 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.438 | 0.000 | 0.095 | 0.095 | 0.413 | 0.413 | 0.413 | 0.000 | 0.463 | 0.050 | 0.463 | 0.000 |
| llada-8b-instruct-hf | plan_026 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.372 | 0.000 | 0.000 | 0.000 | 0.358 | 0.358 | 0.358 | 0.000 | 0.358 | 0.000 | 0.379 | 0.021 |
| llada-8b-instruct-hf | plan_027 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.408 | 0.000 | 0.000 | 0.000 | 0.399 | 0.126 | 0.399 | 0.000 | 0.399 | 0.000 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_028 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.443 | 0.000 | 0.129 | 0.129 | 0.538 | 0.295 | 0.538 | 0.000 | 0.615 | 0.076 | 0.615 | 0.000 |
| llada-8b-instruct-hf | plan_029 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.402 | 0.000 | 0.113 | 0.113 | 0.399 | 0.440 | 0.440 | 0.000 | 0.507 | 0.066 | 0.507 | 0.000 |
| llada-8b-instruct-hf | plan_030 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.315 | 0.000 | 0.000 | 0.000 | 0.323 | 0.323 | 0.344 | 0.000 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_031 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.431 | 0.000 | 0.059 | 0.059 | 0.324 | 0.324 | 0.379 | 0.000 | 0.401 | 0.021 | 0.401 | 0.000 |
| llada-8b-instruct-hf | plan_032 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.369 | 0.000 | 0.069 | 0.069 | 0.353 | 0.384 | 0.384 | 0.000 | 0.414 | 0.030 | 0.414 | 0.000 |
| llada-8b-instruct-hf | plan_033 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.493 | 0.000 | 0.000 | 0.000 | 0.416 | 0.416 | 0.416 | 0.000 | 0.416 | 0.000 | 0.416 | 0.000 |
| llada-8b-instruct-hf | plan_034 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.333 | 0.000 | 0.155 | 0.155 | 0.325 | 0.325 | 0.325 | 0.000 | 0.419 | 0.094 | 0.419 | 0.000 |
| llada-8b-instruct-hf | plan_035 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.398 | 0.000 | 0.000 | 0.000 | 0.304 | 0.283 | 0.304 | 0.000 | 0.304 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_036 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.371 | 0.000 | 0.000 | 0.000 | 0.391 | 0.391 | 0.371 | 0.000 | 0.371 | 0.000 | 0.391 | 0.020 |
| llada-8b-instruct-hf | plan_037 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.353 | 0.000 | 0.000 | 0.000 | 0.337 | 0.337 | 0.337 | 0.000 | 0.337 | 0.000 | 0.337 | 0.000 |
| llada-8b-instruct-hf | plan_038 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.371 | 0.000 | 0.048 | 0.048 | 0.261 | 0.261 | 0.344 | 0.000 | 0.346 | 0.001 | 0.346 | 0.000 |
| llada-8b-instruct-hf | plan_039 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.335 | 0.000 | 0.078 | 0.078 | 0.338 | 0.338 | 0.338 | 0.000 | 0.395 | 0.057 | 0.395 | 0.000 |
| llada-8b-instruct-hf | plan_040 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.395 | 0.000 | 0.061 | 0.061 | 0.281 | 0.301 | 0.281 | 0.000 | 0.319 | 0.037 | 0.319 | 0.000 |
| llada-8b-instruct-hf | plan_041 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.385 | 0.000 | 0.091 | 0.091 | 0.320 | 0.106 | 0.320 | 0.000 | 0.375 | 0.055 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_042 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.312 | 0.000 | 0.067 | 0.067 | 0.261 | 0.198 | 0.261 | 0.000 | 0.304 | 0.042 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_043 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.412 | 0.000 | 0.000 | 0.000 | 0.336 | 0.217 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-8b-instruct-hf | plan_044 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_denoise_phase_repairability |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 | 0.065 | 0.000 | 0.065 | 0.000 | 0.065 | 0.000 |
| llada-8b-instruct-hf | plan_045 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.303 | 0.000 | 0.071 | 0.071 | 0.324 | 0.324 | 0.324 | 0.000 | 0.366 | 0.042 | 0.366 | 0.000 |
| llada-8b-instruct-hf | plan_046 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.414 | 0.000 | 0.145 | 0.145 | 0.366 | 0.282 | 0.366 | 0.000 | 0.488 | 0.122 | 0.488 | 0.000 |
| llada-8b-instruct-hf | plan_047 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.375 | 0.000 | 0.000 | 0.000 | 0.320 | 0.195 | 0.320 | 0.000 | 0.320 | 0.000 | 0.320 | 0.000 |
| llada-8b-instruct-hf | plan_048 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.397 | 0.000 | 0.129 | 0.129 | 0.316 | 0.316 | 0.316 | 0.000 | 0.422 | 0.105 | 0.422 | 0.000 |
