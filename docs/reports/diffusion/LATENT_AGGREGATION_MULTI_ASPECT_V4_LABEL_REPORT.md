# Diffusion Schedule-Selection Benchmark Report

Full model generations: `168`
Counterfactual probe generations: `0`
Arm selections: `168`
Run ID: `diffusion-e47c4077bdd4a25b`
Content hash: `e47c4077bdd4a25b556d603d3c7677f878ceff400262c0b2b6a458c5c4645d72`
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
History mutability: `monotonic 168/168, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.009`
Trajectory task delta vs random: `0.023`
Trajectory wins/ties/losses vs fixed: `7/39/2`
Trajectory wins/ties/losses vs random: `17/27/4`
Oracle generation budget/task: `3.50`
Oracle task score: `0.187`
Oracle headroom vs trajectory: `0.031`
Oracle wins/ties/losses vs trajectory: `18/30/0`
Selector regret vs trajectory: `0.031 over 18/48 improvable`
Repair arm coverage: `24/48` overall
Repair eligible coverage: `24/24`
Repair task delta vs fixed: `0.056`
Repair task delta vs random: `0.085`
Repair task delta vs trajectory: `0.048`
Repair task delta vs evolved: `0.048`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.024`
Repair wins/ties/losses vs evolved: `15/8/1`
Oracle headroom vs repair: `0.006`
Oracle wins/ties/losses vs repair: `5/19/0`
Selector regret vs repair: `0.006 over 5/24 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `24/48` overall, `24/24` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.275119 | 0.000000 | 0.028690 | - | - |
| random perturbation | repair-covered tasks | 0.246429 | -0.028690 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.331313 | 0.056193 | 0.084884 | 16/6/2 | 15/5/4 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.147 | 0.378 | 0.205 |
| random | 48 | 1.00 | 0.133 | 0.328 | 0.182 |
| trajectory_selected | 48 | 2.50 | 0.156 | 0.388 | 0.214 |
| repair_selected | 24 | 4.00 | 0.331 | 0.648 | 0.410 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.147 | 0.378 | 0.205 |
| planning | random | 48 | 1.00 | 0.133 | 0.328 | 0.182 |
| planning | trajectory_selected | 48 | 2.50 | 0.156 | 0.388 | 0.214 |
| planning | repair_selected | 24 | 4.00 | 0.331 | 0.648 | 0.410 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_225 | random_32 | True | denoise_phase_repairable | False |  | 0.295 | 0.235 | 133 | True | 6 | 0.625 | True | True | 5.000 | 0.156 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_226 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 323 | True | 7 | 0.632 | True | True | 4.000 | 0.125 | 0.158 | 0.158 |
| llada-8b-instruct-hf | plan_227 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 265 | True | 4 | 0.733 | True | True | 4.000 | 0.125 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_228 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 294 | True | 5 | 0.615 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_229 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 404 | True | 6 | 0.538 | True | True | 3.000 | 0.094 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_230 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.316 | 0.276 | 333 | True | 4 | 0.769 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_231 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 259 | True | 5 | 0.583 | True | True | 4.000 | 0.125 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_232 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 335 | True | 5 | 0.615 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_233 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.325 | 0.265 | 341 | True | 2 | 0.833 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_234 | random_32 | True | denoise_phase_repairable | False |  | 0.197 | 0.117 | 114 | True | 4 | 0.692 | True | True | 14.000 | 0.438 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_235 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.458 | 0.358 | 355 | True | 3 | 0.769 | True | True | 4.000 | 0.125 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_236 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 308 | True | 4 | 0.667 | True | True | 5.000 | 0.156 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_237 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 306 | True | 1 | 0.917 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_238 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 350 | True | 2 | 0.867 | True | True | 5.000 | 0.156 | 0.267 | 0.267 |
| llada-8b-instruct-hf | plan_239 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 310 | True | 2 | 0.800 | True | True | 3.000 | 0.094 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_240 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.309 | 0.269 | 409 | True | 1 | 0.917 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_241 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 308 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_242 | random_32 | True | denoise_phase_repairable | False |  | 0.302 | 0.282 | 270 | True | 6 | 0.400 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_243 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.336 | 0.276 | 318 | True | 1 | 0.909 | True | True | 3.000 | 0.094 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_244 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 342 | True | 2 | 0.818 | True | True | 4.000 | 0.125 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_245 | random_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 180 | True | 4 | 0.692 | True | True | 6.000 | 0.188 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_246 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.394 | 0.276 | 364 | True | 6 | 0.455 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_247 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.270 | 0.230 | 302 | True | 1 | 0.900 | True | True | 5.000 | 0.156 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_248 | random_32 | True | denoise_phase_repairable | False |  | 0.295 | 0.235 | 301 | True | 1 | 0.933 | True | True | 5.000 | 0.156 | 0.067 | 0.067 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 24 | 12 | low_confidence_32,random_32 | final | 30.6 | 0.958 | 0.042 | 0.000 | 0.023 | 0.023 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 13/10/1 | 0.328 | 0.660 | 0.411 |
| history_prefix_25_repair | 24 | 4 | low_confidence_32,random_32 | history | 48.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8/11/5 | 0.285 | 0.653 | 0.377 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_225 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | I will plan a fresh v4 freeze for the post-failure diversity replay to test if the sour... |
| llada-8b-instruct-hf | plan_226 | False | low_confidence_32 | 3.000 | 0.469 | 1.000 | 0.000 | 0.526 | False | If the diversity-extension run has unique best-of attempts, it suggests that the new sc... |
| llada-8b-instruct-hf | plan_227 | False | low_confidence_32 | 1.346 | 0.731 | 1.000 | 0.000 | 0.133 | False | A CSV file containing the label rows. |
| llada-8b-instruct-hf | plan_227 | False | low_confidence_32 | 1.374 | 0.785 | 1.000 | 0.000 | 0.133 | False | A CSV file containing the probe rows. |
| llada-8b-instruct-hf | plan_227 | False | low_confidence_32 | 1.374 | 0.785 | 1.000 | 0.000 | 0.133 | False | A CSV file containing the diversity rows. |
| llada-8b-instruct-hf | plan_228 | True | low_confidence_32 | 2.183 | 1.000 | 1.000 | 0.000 | 0.308 | False | Once we have this data, we can calculate the normalized cost and include the cost-norma... |
| llada-8b-instruct-hf | plan_229 | True | low_confidence_32 | 2.091 | 0.831 | 1.000 | 0.000 | 0.308 | False | This analysis should focus on identifying whether the diversity source primarily change... |
| llada-8b-instruct-hf | plan_230 | True | low_confidence_32 | 1.360 | 0.860 | 1.000 | 0.000 | 0.308 | False | The v4 replay should clearly indicate these tasks, and we will use this information to... |
| llada-8b-instruct-hf | plan_230 | True | low_confidence_32 | 2.173 | 1.000 | 1.000 | 0.000 | 0.308 | False | Please provide the list of these tasks to accurately plan the failure table. |
| llada-8b-instruct-hf | plan_231 | True | low_confidence_32 | 1.430 | 1.000 | 1.000 | 0.000 | 0.333 | False | b) Use a source mix that is not related to planning prompts. |
| llada-8b-instruct-hf | plan_231 | True | low_confidence_32 | 2.185 | 1.000 | 1.000 | 0.000 | 0.333 | False | c) Use a source mix that is not related to the planning itself. |
| llada-8b-instruct-hf | plan_232 | False | low_confidence_32 | 2.165 | 0.905 | 1.000 | 0.000 | 0.154 | False | This audit should focus on evaluating the realizer's ability to generate novel and mean... |
| llada-8b-instruct-hf | plan_233 | True | low_confidence_32 | 1.983 | 0.642 | 1.000 | 0.000 | 0.500 | False | This involves defining the limits of the model family that will be used for the slice,... |
| llada-8b-instruct-hf | plan_234 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Use probe material safely by considering it as a useful complement but potentially lowe... |
| llada-8b-instruct-hf | plan_235 | True | low_confidence_32 | 1.414 | 0.963 | 1.000 | 0.000 | 0.231 | False | You can also done by measuring the number and length of unstable spans before and after... |
| llada-8b-instruct-hf | plan_235 | True | low_confidence_32 | 2.040 | 0.725 | 1.000 | 0.000 | 0.385 | False | By comparing these metrics, you can demonstrate that the revision is effectively alteri... |
| llada-8b-instruct-hf | plan_236 | False | low_confidence_32 | 1.412 | 0.955 | 1.000 | 0.000 | 0.250 | False | 2. v4 achieves the required numerical performance but with narrow source diversity. |
| llada-8b-instruct-hf | plan_236 | False | low_confidence_32 | 2.173 | 0.955 | 1.000 | 0.000 | 0.250 | False | 3. v4 meets the numerical criteria but with narrow source diversity. |
| llada-8b-instruct-hf | plan_237 | False | low_confidence_32 | 1.314 | 0.805 | 1.000 | 0.000 | 0.500 | False | Availability: Measure the number of rows created as a result of the candidate-diversity... |
| llada-8b-instruct-hf | plan_237 | False | low_confidence_32 | 1.908 | 0.492 | 1.000 | 0.000 | 0.583 | False | Yield: Measure the number of useful complements created as a result of the candidate-di... |
| llada-8b-instruct-hf | plan_238 | True | low_confidence_32 | 2.074 | 0.758 | 1.000 | 0.000 | 0.200 | False | analyze the strengths and weaknesses of each source, and construct a logical argument t... |
| llada-8b-instruct-hf | plan_239 | True | low_confidence_32 | 1.367 | 0.835 | 1.000 | 0.000 | 0.200 | False | Analyze the performance of the outlier task and assess its impact on overall results. |
| llada-8b-instruct-hf | plan_239 | True | low_confidence_32 | 2.099 | 0.835 | 1.000 | 0.000 | 0.300 | False | Repeat the experiment with different sets of outlier tasks to verify the consistency of... |
| llada-8b-instruct-hf | plan_240 | True | low_confidence_32 | 1.892 | 0.505 | 1.000 | 0.000 | 0.667 | False | These features can help the system determine the level of diversity and uncertainty ass... |
| llada-8b-instruct-hf | plan_241 | True | low_confidence_32 | 1.428 | 1.000 | 1.000 | 0.000 | 0.400 | False | This diagnostic that v4 will perform will report its relationship to v3 evidence. |
| llada-8b-instruct-hf | plan_241 | True | low_confidence_32 | 2.190 | 1.000 | 1.000 | 0.000 | 0.300 | False | This is the recommended way for v4 to report its relationship to v3 evidence. |
| llada-8b-instruct-hf | plan_242 | True | random_32 | 2.020 | 0.738 | 1.000 | 0.000 | 0.000 | False | Prepare the input data. |
| llada-8b-instruct-hf | plan_242 | True | random_32 | 1.981 | 0.662 | 1.000 | 0.000 | 0.000 | False | Collect the output data. |
| llada-8b-instruct-hf | plan_242 | True | random_32 | 2.634 | 0.464 | 1.000 | 0.000 | 0.000 | False | Document logs and performance metrics. |
| llada-8b-instruct-hf | plan_243 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To keep scope honest, I a) separate planning tasks from symbolic and science tasks, b)... |
| llada-8b-instruct-hf | plan_244 | True | low_confidence_32 | 1.426 | 0.912 | 1.000 | 0.000 | 0.182 | False | Can you provide provide more details about the schedules, such as the dates, times, and... |
| llada-8b-instruct-hf | plan_244 | True | low_confidence_32 | 2.092 | 0.839 | 1.000 | 0.000 | 0.455 | False | This will help me to create a comprehensive plan for the contradiction-screening report. |
| llada-8b-instruct-hf | plan_245 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | To plan an ablation for the complement count, we need to keep track of the following: 1... |
| llada-8b-instruct-hf | plan_246 | False | low_confidence_32 | 1.954 | 1.000 | 1.000 | 0.000 | 0.182 | False | This contract will be used to evaluate performance and ensure that in-progress tasks ar... |
| llada-8b-instruct-hf | plan_246 | False | low_confidence_32 | 3.365 | 1.000 | 1.000 | 0.000 | 0.091 | False | The contract will be reviewed and updated regularly to ensure it remains relevant and e... |
| llada-8b-instruct-hf | plan_247 | False | low_confidence_32 | 2.098 | 0.856 | 1.000 | 0.000 | 0.400 | False | Evidence:: The report presents a result that is significantly higher than what is suppo... |
| llada-8b-instruct-hf | plan_248 | False | random_32 | 2.834 | 0.905 | 1.000 | 0.000 | 0.000 | False | This integration will enable a more comprehensive and unified approach to enhancing mod... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_225 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_226 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_227 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_228 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_229 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_230 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_231 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_232 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_233 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_234 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_235 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_236 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_237 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_238 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_239 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_240 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_241 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_242 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_243 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_244 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_245 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_246 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_247 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_248 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.130 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_225 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.381 | 0.000 | 0.000 | 0.000 | 0.045 | 0.295 | 0.295 | 0.000 | 0.295 | 0.000 | 0.295 | 0.000 |
| llada-8b-instruct-hf | plan_226 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.387 | 0.000 | 0.000 | 0.000 | 0.283 | 0.330 | 0.283 | 0.000 | 0.283 | 0.000 | 0.330 | 0.047 |
| llada-8b-instruct-hf | plan_227 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.387 | 0.000 | 0.000 | 0.000 | 0.281 | 0.240 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_228 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.359 | 0.000 | 0.065 | 0.065 | 0.240 | 0.240 | 0.240 | 0.000 | 0.282 | 0.042 | 0.294 | 0.011 |
| llada-8b-instruct-hf | plan_229 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.342 | 0.000 | 0.073 | 0.073 | 0.280 | 0.280 | 0.280 | 0.000 | 0.330 | 0.050 | 0.330 | 0.000 |
| llada-8b-instruct-hf | plan_230 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.453 | 0.000 | 0.120 | 0.120 | 0.316 | 0.253 | 0.316 | 0.000 | 0.391 | 0.075 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_231 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.318 | 0.000 | 0.042 | 0.042 | 0.220 | 0.137 | 0.220 | 0.000 | 0.241 | 0.021 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_232 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.371 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_233 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.459 | 0.000 | 0.164 | 0.164 | 0.325 | 0.105 | 0.325 | 0.000 | 0.443 | 0.118 | 0.443 | 0.000 |
| llada-8b-instruct-hf | plan_234 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.310 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.197 | 0.000 | 0.197 | 0.000 | 0.260 | 0.063 |
| llada-8b-instruct-hf | plan_235 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.463 | 0.000 | 0.148 | 0.148 | 0.458 | 0.458 | 0.458 | 0.000 | 0.550 | 0.093 | 0.550 | 0.000 |
| llada-8b-instruct-hf | plan_236 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.374 | 0.000 | 0.065 | 0.065 | 0.240 | 0.157 | 0.240 | 0.000 | 0.282 | 0.042 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_237 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.444 | 0.000 | 0.000 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_238 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.433 | 0.000 | 0.182 | 0.182 | 0.261 | 0.045 | 0.261 | 0.000 | 0.400 | 0.139 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_239 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.419 | 0.000 | 0.164 | 0.164 | 0.281 | 0.281 | 0.281 | 0.000 | 0.383 | 0.101 | 0.383 | 0.000 |
| llada-8b-instruct-hf | plan_240 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.485 | 0.000 | 0.128 | 0.128 | 0.309 | 0.309 | 0.309 | 0.000 | 0.393 | 0.084 | 0.393 | 0.000 |
| llada-8b-instruct-hf | plan_241 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.422 | 0.000 | 0.204 | 0.204 | 0.200 | 0.200 | 0.200 | 0.000 | 0.360 | 0.160 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_242 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.340 | 0.000 | 0.046 | 0.046 | 0.259 | 0.302 | 0.302 | 0.000 | 0.298 | -0.004 | 0.302 | 0.004 |
| llada-8b-instruct-hf | plan_243 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.440 | 0.000 | 0.078 | 0.078 | 0.336 | 0.240 | 0.336 | 0.000 | 0.374 | 0.038 | 0.374 | 0.000 |
| llada-8b-instruct-hf | plan_244 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.374 | 0.000 | 0.181 | 0.181 | 0.241 | 0.241 | 0.241 | 0.000 | 0.400 | 0.159 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_245 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.357 | 0.000 | 0.042 | 0.042 | 0.282 | 0.282 | 0.240 | 0.000 | 0.261 | 0.021 | 0.282 | 0.021 |
| llada-8b-instruct-hf | plan_246 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.335 | 0.000 | 0.000 | 0.000 | 0.394 | 0.394 | 0.394 | 0.000 | 0.394 | 0.000 | 0.394 | 0.000 |
| llada-8b-instruct-hf | plan_247 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.455 | 0.000 | 0.047 | 0.047 | 0.270 | 0.045 | 0.270 | 0.000 | 0.291 | 0.021 | 0.291 | 0.000 |
| llada-8b-instruct-hf | plan_248 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.436 | 0.000 | 0.000 | 0.000 | 0.295 | 0.295 | 0.295 | 0.000 | 0.295 | 0.000 | 0.295 | 0.000 |
