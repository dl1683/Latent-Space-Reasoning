# Diffusion Schedule-Selection Benchmark Report

Full model generations: `168`
Counterfactual probe generations: `0`
Arm selections: `168`
Run ID: `diffusion-327394506a238648`
Content hash: `327394506a23864833032747d6bad8ccde59bc96e6efbba9d57e0a693efbface`
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
Trajectory task delta vs fixed: `0.008`
Trajectory task delta vs random: `0.018`
Trajectory wins/ties/losses vs fixed: `7/40/1`
Trajectory wins/ties/losses vs random: `14/32/2`
Oracle generation budget/task: `3.50`
Oracle task score: `0.195`
Oracle headroom vs trajectory: `0.021`
Oracle wins/ties/losses vs trajectory: `14/34/0`
Selector regret vs trajectory: `0.021 over 14/48 improvable`
Repair arm coverage: `24/48` overall
Repair eligible coverage: `24/24`
Repair task delta vs fixed: `0.036`
Repair task delta vs random: `0.057`
Repair task delta vs trajectory: `0.036`
Repair task delta vs evolved: `0.036`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.018`
Repair wins/ties/losses vs evolved: `10/13/1`
Oracle headroom vs repair: `0.003`
Oracle wins/ties/losses vs repair: `4/20/0`
Selector regret vs repair: `0.003 over 4/24 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `24/48` overall, `24/24` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.313958 | 0.000000 | 0.020693 | - | - |
| random perturbation | repair-covered tasks | 0.293265 | -0.020693 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.350000 | 0.036042 | 0.056735 | 10/13/1 | 14/8/2 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.166 | 0.384 | 0.221 |
| random | 48 | 1.00 | 0.156 | 0.347 | 0.203 |
| trajectory_selected | 48 | 2.50 | 0.174 | 0.404 | 0.232 |
| repair_selected | 24 | 4.00 | 0.350 | 0.673 | 0.431 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.166 | 0.384 | 0.221 |
| planning | random | 48 | 1.00 | 0.156 | 0.347 | 0.203 |
| planning | trajectory_selected | 48 | 2.50 | 0.174 | 0.404 | 0.232 |
| planning | repair_selected | 24 | 4.00 | 0.350 | 0.673 | 0.431 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_201 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 306 | True | 5 | 0.706 | True | True | 5.000 | 0.156 | 0.176 | 0.176 |
| llada-8b-instruct-hf | plan_202 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.321 | 0.281 | 305 | True | 5 | 0.737 | True | True | 3.000 | 0.094 | 0.105 | 0.105 |
| llada-8b-instruct-hf | plan_203 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 355 | True | 1 | 0.923 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_204 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 372 | True | 7 | 0.684 | True | True | 3.000 | 0.094 | 0.053 | 0.053 |
| llada-8b-instruct-hf | plan_205 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 305 | True | 8 | 0.500 | True | True | 3.000 | 0.094 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_206 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.486 | 0.369 | 372 | True | 2 | 0.857 | True | True | 4.000 | 0.125 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_207 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.378 | 0.278 | 317 | True | 6 | 0.538 | True | True | 3.000 | 0.094 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_208 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.311 | 0.251 | 385 | True | 4 | 0.733 | True | True | 3.000 | 0.094 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_209 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 328 | True | 4 | 0.692 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_210 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.217 | 297 | True | 6 | 0.571 | True | True | 4.000 | 0.125 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_211 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.239 | 364 | True | 7 | 0.533 | True | True | 4.000 | 0.125 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_212 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 339 | True | 4 | 0.714 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_213 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.274 | 0.214 | 309 | True | 4 | 0.636 | True | True | 2.000 | 0.062 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_214 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.299 | 0.239 | 261 | True | 8 | 0.385 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_215 | random_32 | True | denoise_phase_repairable | False |  | 0.335 | 0.235 | 325 | True | 9 | 0.500 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_216 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.391 | 0.311 | 334 | True | 1 | 0.917 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_217 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 292 | True | 2 | 0.818 | True | True | 3.000 | 0.094 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_218 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 291 | True | 7 | 0.533 | True | True | 5.000 | 0.156 | 0.067 | 0.067 |
| llada-8b-instruct-hf | plan_219 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 309 | True | 6 | 0.625 | True | True | 3.000 | 0.094 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_220 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 362 | True | 4 | 0.692 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_221 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.318 | 0.278 | 364 | True | 0 | 1.000 | True | True | 4.000 | 0.125 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_222 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.399 | 0.289 | 329 | True | 7 | 0.562 | True | True | 4.000 | 0.125 | 0.188 | 0.188 |
| llada-8b-instruct-hf | plan_223 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.235 | 355 | True | 2 | 0.846 | True | True | 4.000 | 0.125 | 0.308 | 0.308 |
| llada-8b-instruct-hf | plan_224 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 329 | True | 4 | 0.750 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 24 | 9 | low_confidence_32,random_32 | final | 30.1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 11/10/3 | 0.349 | 0.681 | 0.432 |
| history_prefix_25_repair | 24 | 2 | low_confidence_32,random_32 | history | 48.1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.005 | -0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2/15/7 | 0.310 | 0.688 | 0.405 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_201 | True | low_confidence_32 | 1.616 | 0.288 | 1.000 | 0.000 | 0.294 | False | Identify the tasks with no complement material beyond the anchor. |
| llada-8b-instruct-hf | plan_201 | True | low_confidence_32 | 1.931 | 0.910 | 1.000 | 0.000 | 0.118 | False | Adjust the experiment design to include these tasks. |
| llada-8b-instruct-hf | plan_201 | True | low_confidence_32 | 2.161 | 0.910 | 1.000 | 0.000 | 0.235 | False | Run the experiment to test coverage without lowering the old threshold. |
| llada-8b-instruct-hf | plan_202 | False | low_confidence_32 | 3.246 | 0.726 | 1.000 | 0.000 | 0.053 | False | Ensure the combined is supported and not based on incomplete or irrelevant information. |
| llada-8b-instruct-hf | plan_203 | False | low_confidence_32 | 1.423 | 1.000 | 1.000 | 0.000 | 0.385 | False | This protocol ensures that the selector does not leak information about the other ordin... |
| llada-8b-instruct-hf | plan_203 | False | low_confidence_32 | 2.187 | 1.000 | 1.000 | 0.000 | 0.308 | False | Instead, it focuses on the aspect-deficit of the anchor, allowing for precise selection... |
| llada-8b-instruct-hf | plan_204 | False | random_32 | 2.158 | 0.896 | 1.000 | 0.000 | 0.158 | False | The table will display the relative contributions of each source, highlighting their im... |
| llada-8b-instruct-hf | plan_205 | True | low_confidence_32 | 2.069 | 0.865 | 1.000 | 0.000 | 0.062 | False | Assign weighted scores based on these aspects. |
| llada-8b-instruct-hf | plan_205 | True | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Aggregate the scores to produce a composite score. |
| llada-8b-instruct-hf | plan_205 | True | low_confidence_32 | 2.160 | 0.865 | 1.000 | 0.000 | 0.125 | False | Select the candidate with the highest composite score as best. |
| llada-8b-instruct-hf | plan_206 | True | low_confidence_32 | 2.499 | 0.584 | 1.000 | 0.000 | 0.214 | False | This mechanism will ensure that the concerns identified by the probes are indeed real a... |
| llada-8b-instruct-hf | plan_207 | False | low_confidence_32 | 1.828 | 0.277 | 1.000 | 0.000 | 0.462 | False | If the realized answer onlys the selected latent aspects and is significantly different... |
| llada-8b-instruct-hf | plan_208 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | A falsifiable mechanism claim for a future reasoning system that aggregatesates complem... |
| llada-8b-instruct-hf | plan_209 | False | low_confidence_32 | 2.884 | 1.000 | 1.000 | 0.000 | 0.077 | False | Compare the two between the sections sections to highlight the contrast. |
| llada-8b-instruct-hf | plan_210 | False | low_confidence_32 | 1.942 | 0.955 | 1.000 | 0.000 | 0.143 | False | For example, multiply the rubric coverage score by 0.5. |
| llada-8b-instruct-hf | plan_210 | False | low_confidence_32 | 2.151 | 1.000 | 1.000 | 0.000 | 0.429 | False | This will give more weight to the useful risk warning despite the weak rubric coverage. |
| llada-8b-instruct-hf | plan_211 | True | low_confidence_32 | 2.627 | 1.000 | 1.000 | 0.000 | 0.000 | False | Evaluate the impact of these additions on the model's performance and adjust accordingly. |
| llada-8b-instruct-hf | plan_211 | True | low_confidence_32 | 2.639 | 0.869 | 1.000 | 0.000 | 0.133 | False | Use a validation set to monitor the model's generalization and refine the ontology if n... |
| llada-8b-instruct-hf | plan_212 | False | low_confidence_32 | 1.387 | 0.868 | 1.000 | 0.000 | 0.214 | False | This involves dividing the coverage improvement by the cost of the extra GPU and settin... |
| llada-8b-instruct-hf | plan_212 | False | low_confidence_32 | 2.197 | 1.000 | 1.000 | 0.000 | 0.214 | False | If the result is greater than the threshold, the coverage is worth the cost. |
| llada-8b-instruct-hf | plan_213 | False | low_confidence_32 | 2.510 | 0.758 | 1.000 | 0.000 | 0.091 | False | Compare these aspects to detect inconsistencies. |
| llada-8b-instruct-hf | plan_213 | False | low_confidence_32 | 2.538 | 0.790 | 1.000 | 0.000 | 0.000 | False | Analyze the inconsistencies to understand the discrepancies. |
| llada-8b-instruct-hf | plan_213 | False | low_confidence_32 | 3.288 | 0.790 | 1.000 | 0.000 | 0.000 | False | Document the findings and propose strategies for resolution. |
| llada-8b-instruct-hf | plan_214 | True | low_confidence_32 | 2.524 | 0.770 | 1.000 | 0.000 | 0.077 | False | Predefined pass criteria for achieving the desired outcome. |
| llada-8b-instruct-hf | plan_214 | True | low_confidence_32 | 1.853 | 0.770 | 1.000 | 0.000 | 0.154 | False | Boundary conditions that must be met to classify the experiment as v3.. |
| llada-8b-instruct-hf | plan_214 | True | low_confidence_32 | 2.074 | 0.713 | 1.000 | 0.000 | 0.154 | False | Fail criteria that must be met to classify the experiment as v3. |
| llada-8b-instruct-hf | plan_215 | True | random_32 | 2.839 | 0.910 | 1.000 | 0.000 | 0.062 | False | Aggregate the adjusted scores to produce the final judging score. |
| llada-8b-instruct-hf | plan_216 | True | low_confidence_32 | 1.954 | 0.609 | 1.000 | 0.000 | 0.583 | False | This will allow the model to learn general patterns and reduce overfitting before trans... |
| llada-8b-instruct-hf | plan_217 | False | low_confidence_32 | 1.293 | 0.629 | 1.000 | 0.000 | 0.182 | False | Review the component extractor code. |
| llada-8b-instruct-hf | plan_217 | False | low_confidence_32 | 1.360 | 0.790 | 1.000 | 0.000 | 0.182 | False | Identify potential areas prone to false negatives. |
| llada-8b-instruct-hf | plan_217 | False | low_confidence_32 | 2.138 | 0.865 | 1.000 | 0.000 | 0.273 | False | Provide recommendations for improving the extractor to minimize false negatives. |
| llada-8b-instruct-hf | plan_218 | False | low_confidence_32 | 1.357 | 0.780 | 1.000 | 0.000 | 0.133 | False | For example, if a complement is smaller than a certain percentage (e.g., 10%) of the to... |
| llada-8b-instruct-hf | plan_218 | False | low_confidence_32 | 1.902 | 0.393 | 1.000 | 0.000 | 0.333 | False | This rule will keep aggregation useful by not overwhelming the final answer. |
| llada-8b-instruct-hf | plan_219 | True | low_confidence_32 | 2.129 | 1.000 | 1.000 | 0.000 | 0.062 | False | b)) implement the methods in the same environment; |
| llada-8b-instruct-hf | plan_219 | True | low_confidence_32 | 2.129 | 1.000 | 1.000 | 0.000 | 0.062 | False | c) run the methods on the same dataset; |
| llada-8b-instruct-hf | plan_219 | True | low_confidence_32 | 2.878 | 1.000 | 1.000 | 0.000 | 0.000 | False | d d) analyze the results to compare their performance of the tasks. |
| llada-8b-instruct-hf | plan_220 | False | low_confidence_32 | 2.146 | 0.925 | 1.000 | 0.000 | 0.308 | False | This proof object would allow the reasoner to verify the correctness and validity of th... |
| llada-8b-instruct-hf | plan_221 | False | low_confidence_32 | 1.934 | 0.595 | 0.818 | 0.000 | 0.182 | False | This will help to ensure the validity and reliability of the results before treating th... |
| llada-8b-instruct-hf | plan_222 | True | low_confidence_32 | 1.932 | 1.000 | 1.000 | 0.000 | 0.312 | False | Weigh awareness and completion differently based on their importance in the context of... |
| llada-8b-instruct-hf | plan_222 | True | low_confidence_32 | 2.166 | 1.000 | 1.000 | 0.000 | 0.375 | False | The goal is to find a balance between risk awareness and completion to optimize the mul... |
| llada-8b-instruct-hf | plan_223 | False | low_confidence_32 | 1.906 | 0.421 | 1.000 | 0.000 | 0.385 | False | This rule should be designed to identify which deficits are most significant and warran... |
| llada-8b-instruct-hf | plan_224 | False | low_confidence_32 | 1.404 | 0.910 | 1.000 | 0.000 | 0.250 | False | Examples of aggregation across latent states. |
| llada-8b-instruct-hf | plan_224 | False | low_confidence_32 | 1.419 | 0.910 | 1.000 | 0.000 | 0.188 | False | Examples of selection based on latent states. |
| llada-8b-instruct-hf | plan_224 | False | low_confidence_32 | 2.170 | 0.910 | 1.000 | 0.000 | 0.125 | False | Analysis comparing the richness and depth of information obtained from aggregation and... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_201 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_202 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.136 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_203 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_204 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_205 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_206 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_207 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_208 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.112 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_209 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_210 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_211 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_212 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_213 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_214 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_215 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_216 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_217 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_218 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.138 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_219 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_220 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_221 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_222 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_223 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_224 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_201 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.395 | 0.000 | 0.064 | 0.064 | 0.323 | 0.323 | 0.323 | 0.000 | 0.320 | -0.003 | 0.323 | 0.003 |
| llada-8b-instruct-hf | plan_202 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.423 | 0.000 | 0.000 | 0.000 | 0.321 | 0.321 | 0.321 | 0.000 | 0.321 | 0.000 | 0.361 | 0.040 |
| llada-8b-instruct-hf | plan_203 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.437 | 0.000 | 0.065 | 0.065 | 0.280 | 0.280 | 0.280 | 0.000 | 0.323 | 0.042 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_204 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.359 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_205 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.330 | 0.000 | 0.148 | 0.148 | 0.260 | 0.260 | 0.260 | 0.000 | 0.378 | 0.118 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_206 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.511 | 0.000 | 0.092 | 0.092 | 0.486 | 0.420 | 0.486 | 0.000 | 0.529 | 0.043 | 0.529 | 0.000 |
| llada-8b-instruct-hf | plan_207 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.343 | 0.000 | 0.000 | 0.000 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_208 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.432 | 0.000 | 0.000 | 0.000 | 0.311 | 0.311 | 0.311 | 0.000 | 0.311 | 0.000 | 0.311 | 0.000 |
| llada-8b-instruct-hf | plan_209 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.374 | 0.000 | 0.000 | 0.000 | 0.281 | 0.197 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_210 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.375 | 0.000 | 0.000 | 0.000 | 0.315 | 0.315 | 0.315 | 0.000 | 0.315 | 0.000 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_211 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.368 | 0.000 | 0.139 | 0.139 | 0.339 | 0.339 | 0.339 | 0.000 | 0.473 | 0.134 | 0.473 | 0.000 |
| llada-8b-instruct-hf | plan_212 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.390 | 0.000 | 0.000 | 0.000 | 0.281 | 0.260 | 0.281 | 0.000 | 0.281 | 0.000 | 0.301 | 0.020 |
| llada-8b-instruct-hf | plan_213 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.380 | 0.000 | 0.072 | 0.072 | 0.274 | 0.281 | 0.274 | 0.000 | 0.360 | 0.086 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_214 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.324 | 0.000 | 0.139 | 0.139 | 0.299 | 0.214 | 0.299 | 0.000 | 0.395 | 0.096 | 0.395 | 0.000 |
| llada-8b-instruct-hf | plan_215 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.340 | 0.000 | 0.127 | 0.127 | 0.335 | 0.335 | 0.335 | 0.000 | 0.422 | 0.087 | 0.422 | 0.000 |
| llada-8b-instruct-hf | plan_216 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.502 | 0.000 | 0.087 | 0.087 | 0.391 | 0.324 | 0.391 | 0.000 | 0.434 | 0.042 | 0.434 | 0.000 |
| llada-8b-instruct-hf | plan_217 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.404 | 0.000 | 0.000 | 0.000 | 0.281 | 0.277 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_218 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.328 | 0.000 | 0.000 | 0.000 | 0.281 | 0.282 | 0.281 | 0.000 | 0.281 | 0.000 | 0.282 | 0.001 |
| llada-8b-instruct-hf | plan_219 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.367 | 0.000 | 0.161 | 0.161 | 0.303 | 0.303 | 0.303 | 0.000 | 0.421 | 0.118 | 0.421 | 0.000 |
| llada-8b-instruct-hf | plan_220 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.355 | 0.000 | 0.000 | 0.000 | 0.241 | 0.065 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_221 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.513 | 0.000 | 0.000 | 0.000 | 0.318 | 0.318 | 0.318 | 0.000 | 0.318 | 0.000 | 0.318 | 0.000 |
| llada-8b-instruct-hf | plan_222 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.410 | 0.000 | 0.149 | 0.149 | 0.399 | 0.399 | 0.399 | 0.000 | 0.499 | 0.100 | 0.499 | 0.000 |
| llada-8b-instruct-hf | plan_223 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.440 | 0.000 | 0.000 | 0.000 | 0.315 | 0.315 | 0.315 | 0.000 | 0.315 | 0.000 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_224 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.395 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
