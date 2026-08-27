# Diffusion Schedule-Selection Benchmark Report

Full model generations: `336`
Counterfactual probe generations: `0`
Arm selections: `336`
Run ID: `diffusion-52aa559013574dba`
Content hash: `52aa559013574dba29db7b381df1c866fa29641ab794787a80671e98c413c737`
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
History mutability: `monotonic 336/336, changes 0, remasks 0, rewrites 0, mask increases 0`
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
Trajectory task delta vs fixed: `0.004`
Trajectory task delta vs random: `0.022`
Trajectory wins/ties/losses vs fixed: `13/77/6`
Trajectory wins/ties/losses vs random: `29/62/5`
Oracle generation budget/task: `3.50`
Oracle task score: `0.179`
Oracle headroom vs trajectory: `0.022`
Oracle wins/ties/losses vs trajectory: `33/63/0`
Selector regret vs trajectory: `0.022 over 33/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.040`
Repair task delta vs random: `0.066`
Repair task delta vs trajectory: `0.034`
Repair task delta vs evolved: `0.034`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.017`
Repair wins/ties/losses vs evolved: `25/22/1`
Oracle headroom vs repair: `0.002`
Oracle wins/ties/losses vs repair: `5/43/0`
Selector regret vs repair: `0.002 over 5/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.284007 | 0.000000 | 0.026881 | - | - |
| random perturbation | repair-covered tasks | 0.257126 | -0.026881 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.323549 | 0.039542 | 0.066423 | 28/17/3 | 33/12/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.153 | 0.382 | 0.210 |
| random | 96 | 1.00 | 0.135 | 0.315 | 0.180 |
| trajectory_selected | 96 | 2.50 | 0.157 | 0.386 | 0.214 |
| repair_selected | 48 | 4.00 | 0.324 | 0.659 | 0.408 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.153 | 0.382 | 0.210 |
| planning | random | 96 | 1.00 | 0.135 | 0.315 | 0.180 |
| planning | trajectory_selected | 96 | 2.50 | 0.157 | 0.386 | 0.214 |
| planning | repair_selected | 48 | 4.00 | 0.324 | 0.659 | 0.408 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_249 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 255 | True | 7 | 0.533 | True | True | 4.000 | 0.125 | 0.133 | 0.133 |
| llada-8b-instruct-hf | plan_250 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.281 | 0.201 | 324 | True | 9 | 0.429 | True | True | 3.000 | 0.094 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_251 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.341 | 0.301 | 258 | True | 5 | 0.769 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_252 | random_32 | True | denoise_phase_repairable | False |  | 0.391 | 0.311 | 324 | True | 2 | 0.917 | True | True | 2.000 | 0.062 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_253 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 305 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_254 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 331 | True | 1 | 0.923 | True | True | 4.000 | 0.125 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_255 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.378 | 0.260 | 309 | True | 2 | 0.867 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_256 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 319 | True | 7 | 0.533 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_257 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 398 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_258 | random_32 | True | denoise_phase_repairable | False |  | 0.065 | 0.045 | 64 | True | 4 | 0.636 | True | True | 9.000 | 0.281 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_259 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.303 | 0.223 | 354 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_260 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.381 | 0.301 | 331 | True | 1 | 0.889 | True | True | 4.000 | 0.125 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_261 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 271 | True | 3 | 0.769 | True | True | 5.000 | 0.156 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_262 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.418 | 0.340 | 342 | True | 5 | 0.500 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_263 | random_32 | True | denoise_phase_repairable | False |  | 0.117 | 0.117 | 136 | True | 5 | 0.500 | True | True | 4.000 | 0.125 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_264 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 381 | True | 6 | 0.571 | True | True | 3.000 | 0.094 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_265 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 331 | True | 3 | 0.727 | True | True | 4.000 | 0.125 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_266 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.330 | 0.230 | 336 | True | 3 | 0.727 | True | True | 4.000 | 0.125 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_267 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.301 | 0.201 | 380 | True | 1 | 0.909 | True | True | 4.000 | 0.125 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_268 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.345 | 0.285 | 345 | True | 1 | 0.929 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_269 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 334 | True | 2 | 0.867 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_270 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.319 | 0.299 | 258 | True | 4 | 0.600 | True | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_271 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 287 | True | 4 | 0.667 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_272 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.404 | 0.326 | 293 | True | 1 | 0.917 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_273 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.378 | 0.267 | 267 | True | 4 | 0.636 | True | True | 5.000 | 0.156 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_274 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.309 | 0.269 | 336 | True | 4 | 0.667 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_275 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.391 | 0.269 | 336 | True | 4 | 0.667 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_276 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 314 | True | 1 | 0.923 | True | True | 4.000 | 0.125 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_277 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.334 | 0.294 | 414 | True | 2 | 0.800 | True | True | 3.000 | 0.094 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_278 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 348 | True | 5 | 0.583 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_279 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 348 | True | 5 | 0.545 | True | True | 4.000 | 0.125 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_280 | random_32 | True | denoise_phase_repairable | False |  | 0.273 | 0.193 | 360 | True | 3 | 0.786 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_281 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.296 | 0.256 | 387 | True | 4 | 0.600 | True | True | 4.000 | 0.125 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_282 | random_32 | True | denoise_phase_repairable | False |  | 0.255 | 0.235 | 307 | True | 6 | 0.643 | True | True | 3.000 | 0.094 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_283 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.213 | 0.193 | 399 | True | 6 | 0.500 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_284 | random_32 | True | denoise_phase_repairable | False |  | 0.366 | 0.244 | 311 | True | 3 | 0.727 | True | True | 3.000 | 0.094 | 0.091 | 0.091 |
| llada-8b-instruct-hf | plan_285 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.395 | 0.272 | 339 | True | 2 | 0.778 | True | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_286 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.234 | 0.214 | 276 | True | 6 | 0.500 | True | True | 3.000 | 0.094 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_287 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 369 | True | 1 | 0.909 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_288 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.178 | 0.138 | 126 | True | 4 | 0.500 | True | True | 5.000 | 0.156 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_289 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.324 | 0.244 | 288 | True | 4 | 0.667 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_290 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.233 | 0.193 | 342 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_291 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.180 | 0.180 | 300 | True | 3 | 0.625 | True | True | 5.000 | 0.156 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_292 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.339 | 0.299 | 323 | True | 4 | 0.600 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_293 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 326 | True | 2 | 0.800 | True | True | 3.000 | 0.094 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_294 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.311 | 0.251 | 340 | True | 0 | 1.000 | True | True | 3.000 | 0.094 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_295 | random_32 | True | denoise_phase_repairable | False |  | 0.414 | 0.374 | 328 | True | 4 | 0.667 | True | True | 1.000 | 0.031 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_296 | random_32 | True | denoise_phase_repairable | False |  | 0.329 | 0.269 | 254 | True | 1 | 0.923 | True | True | 3.000 | 0.094 | 0.231 | 0.231 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 48 | 16 | low_confidence_32,random_32 | final | 27.1 | 0.938 | 0.062 | 0.000 | 0.015 | 0.015 | 0.014 | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20/16/12 | 0.305 | 0.667 | 0.395 |
| history_prefix_25_repair | 48 | 10 | low_confidence_32,random_32 | history | 48.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | -0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 12/17/19 | 0.279 | 0.660 | 0.374 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_249 | True | low_confidence_32 | 2.012 | 0.740 | 1.000 | 0.000 | 0.067 | False | Review the v4 aggregation result. |
| llada-8b-instruct-hf | plan_249 | True | low_confidence_32 | 2.068 | 0.865 | 1.000 | 0.000 | 0.067 | False | Use the same selector as v4. |
| llada-8b-instruct-hf | plan_249 | True | low_confidence_32 | 2.157 | 0.865 | 1.000 | 0.000 | 0.133 | False | Execute the replication5 with the updated sample. |
| llada-8b-instruct-hf | plan_250 | True | low_confidence_32 | 1.450 | 1.000 | 1.000 | 0.000 | 0.286 | False | We will calculate the mean lift and the mean excluding the highlift task task to assess... |
| llada-8b-instruct-hf | plan_250 | True | low_confidence_32 | 2.130 | 0.893 | 1.000 | 0.000 | 0.286 | False | Additionally will compare the mean including and excluding the highlift task to determi... |
| llada-8b-instruct-hf | plan_251 | False | low_confidence_32 | 1.405 | 0.860 | 1.000 | 0.000 | 0.154 | False | Gathering data from multiple sources. |
| llada-8b-instruct-hf | plan_251 | False | low_confidence_32 | 2.043 | 0.837 | 1.000 | 0.000 | 0.077 | False | Validate the accuracy of the aggregation. |
| llada-8b-instruct-hf | plan_251 | False | low_confidence_32 | 2.190 | 0.981 | 1.000 | 0.000 | 0.231 | False | Prepare the results for the next fresh slice. |
| llada-8b-instruct-hf | plan_252 | False | random_32 | 2.096 | 0.850 | 1.000 | 0.000 | 0.500 | False | Otherwise, report that removing the source of the lift makes the family view less justi... |
| llada-8b-instruct-hf | plan_253 | True | low_confidence_32 | 2.764 | 0.798 | 1.000 | 0.000 | 0.083 | False | We will normalize the cost by dividing the cost of each run by the total cost of all ru... |
| llada-8b-instruct-hf | plan_254 | True | low_confidence_32 | 1.853 | 0.359 | 1.000 | 0.000 | 0.538 | False | If the anchor is not sufficient or if additional isobar information is needed, use the... |
| llada-8b-instruct-hf | plan_255 | False | low_confidence_32 | 1.408 | 0.910 | 1.000 | 0.000 | 0.200 | False | Identify the useful risk in the diversity row. |
| llada-8b-instruct-hf | plan_255 | False | low_confidence_32 | 1.912 | 0.910 | 1.000 | 0.000 | 0.200 | False | Identify the wrong mitigation in the diversity row. |
| llada-8b-instruct-hf | plan_255 | False | low_confidence_32 | 2.131 | 0.910 | 1.000 | 0.000 | 0.333 | False | Select the complement that correctly mitigs the wrong mitigation while preserving the u... |
| llada-8b-instruct-hf | plan_256 | False | low_confidence_32 | 1.467 | 1.000 | 1.000 | 0.000 | 0.133 | False | Definition of latent aggregation. |
| llada-8b-instruct-hf | plan_256 | False | low_confidence_32 | 1.467 | 1.000 | 1.000 | 0.000 | 0.133 | False | Theoretical foundations of latent aggregation. |
| llada-8b-instruct-hf | plan_256 | False | low_confidence_32 | 2.217 | 1.000 | 1.000 | 0.000 | 0.133 | False | 2 citations supporting the use of latent aggregation. |
| llada-8b-instruct-hf | plan_257 | False | low_confidence_32 | 1.993 | 0.743 | 1.000 | 0.000 | 0.083 | False | I will compare these responses to those generated through synthesizing information and... |
| llada-8b-instruct-hf | plan_257 | False | low_confidence_32 | 2.060 | 0.700 | 1.000 | 0.000 | 0.250 | False | This comparison will help distinguish between the increased length due to synthesis and... |
| llada-8b-instruct-hf | plan_258 | False | random_32 | 1.710 | 0.000 | 1.000 | 0.000 | 0.545 | False | Plan a stricter contradiction audit for future runs. |
| llada-8b-instruct-hf | plan_259 | True | low_confidence_32 | 1.421 | 0.968 | 1.000 | 0.000 | 0.250 | False | Each test should include a check to ensure that all covered tasks are promoted locally. |
| llada-8b-instruct-hf | plan_259 | True | low_confidence_32 | 2.164 | 0.925 | 1.000 | 0.000 | 0.167 | False | By running these tests in sequence, you can verify that the selector behaves correctly... |
| llada-8b-instruct-hf | plan_260 | False | low_confidence_32 | 2.022 | 0.755 | 1.000 | 0.000 | 0.556 | False | This means that the run should be promoted because it shows a positive all-task lift, i... |
| llada-8b-instruct-hf | plan_261 | False | low_confidence_32 | 2.045 | 0.822 | 1.000 | 0.000 | 0.077 | False | Raw Generations 3. |
| llada-8b-instruct-hf | plan_261 | False | low_confidence_32 | 1.289 | 0.619 | 1.000 | 0.000 | 0.154 | False | Replay JSON 5. |
| llada-8b-instruct-hf | plan_261 | False | low_confidence_32 | 1.761 | 0.091 | 1.000 | 0.000 | 0.385 | False | Appendices (if applicable) This spine provides a clear and guides the reader through th... |
| llada-8b-instruct-hf | plan_262 | False | low_confidence_32 | 2.657 | 0.578 | 1.000 | 0.000 | 0.100 | False | Highlight the differences and similarities between the two versions, and discuss the fa... |
| llada-8b-instruct-hf | plan_263 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | The conservative claim wording for a passing v5 replication could be: "Preliminary vali... |
| llada-8b-instruct-hf | plan_264 | False | low_confidence_32 | 2.877 | 1.000 | 1.000 | 0.000 | 0.000 | False | Use the results to evaluate the impact of each component on the overall reasoning perfo... |
| llada-8b-instruct-hf | plan_265 | False | low_confidence_32 | 1.415 | 1.000 | 1.000 | 0.000 | 0.455 | False | If the number of useful complements is low, consider removing the probes from the family. |
| llada-8b-instruct-hf | plan_265 | False | low_confidence_32 | 2.142 | 1.000 | 1.000 | 0.000 | 0.636 | False | However, if a sufficient number of useful complements are found across tasks, decide to... |
| llada-8b-instruct-hf | plan_266 | False | low_confidence_32 | 1.358 | 0.835 | 1.000 | 0.000 | 0.273 | False | Implement a mechanism to track and account for duplicates during the aggregation replay. |
| llada-8b-instruct-hf | plan_266 | False | low_confidence_32 | 2.196 | 1.000 | 1.000 | 0.000 | 0.364 | False | Adjust the replay replay schedule to account for duplicates, ensuring efficient and of... |
| llada-8b-instruct-hf | plan_267 | False | low_confidence_32 | 2.865 | 1.000 | 1.000 | 0.000 | 0.091 | False | Further analysis may be needed to determine the reasons for the difference in bucket pe... |
| llada-8b-instruct-hf | plan_268 | True | low_confidence_32 | 2.042 | 0.748 | 1.000 | 0.000 | 0.357 | False | This can be achieved by implementing a mechanism that masks and anonymizes score detail... |
| llada-8b-instruct-hf | plan_269 | False | low_confidence_32 | 2.184 | 1.000 | 1.000 | 0.000 | 0.267 | False | This will help me provide a more accurate plan for the proof object fields. |
| llada-8b-instruct-hf | plan_270 | False | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Save the current state of the model. |
| llada-8b-instruct-hf | plan_270 | False | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Interrupt the GPU run. |
| llada-8b-instruct-hf | plan_270 | False | low_confidence_32 | 2.040 | 0.793 | 1.000 | 0.000 | 0.000 | False | Load the model state from the saved checkpoint. |
| llada-8b-instruct-hf | plan_271 | False | low_confidence_32 | 1.347 | 0.850 | 1.000 | 0.000 | 0.417 | False | If the larger slice's effect size is lower, v4 remains the current version. |
| llada-8b-instruct-hf | plan_271 | False | low_confidence_32 | 2.122 | 0.905 | 1.000 | 0.000 | 0.417 | False | If the larger slice's effect size is higher, replace v4 with the larger slice and conti... |
| llada-8b-instruct-hf | plan_272 | False | low_confidence_32 | 1.279 | 0.640 | 1.000 | 0.000 | 0.250 | False | Identify the failure score below the anchor. |
| llada-8b-instruct-hf | plan_272 | False | low_confidence_32 | 1.869 | 0.354 | 1.000 | 0.000 | 0.417 | False | Determine why the aggregate still scores below the failure. |
| llada-8b-instruct-hf | plan_273 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Sure, here's a dimension tradeoff table for the report: / Dimension / Tradeoff / /-----... |
| llada-8b-instruct-hf | plan_274 | True | low_confidence_32 | 1.451 | 1.000 | 1.000 | 0.000 | 0.167 | False | Can you provide more information about the possible outcomes and their probabilities? |
| llada-8b-instruct-hf | plan_274 | True | low_confidence_32 | 1.845 | 0.341 | 1.000 | 0.000 | 0.500 | False | Once I have this information, I can help you plan the decision tree to prioritize the n... |
| llada-8b-instruct-hf | plan_275 | True | low_confidence_32 | 2.150 | 0.887 | 1.000 | 0.000 | 0.167 | False | This involves calculating the expenses associated with sampling, evaluating, and select... |
| llada-8b-instruct-hf | plan_276 | False | low_confidence_32 | 2.147 | 0.910 | 1.000 | 0.000 | 0.308 | False | Develop a follow-up plan that addresses these revisions without altering v5 retroactively. |
| llada-8b-instruct-hf | plan_277 | False | low_confidence_32 | 1.369 | 0.944 | 1.000 | 0.000 | 0.500 | False | The strong complement coverage indicates a solid foundation, but the unsupported additi... |
| llada-8b-instruct-hf | plan_277 | False | low_confidence_32 | 2.141 | 0.917 | 1.000 | 0.000 | 0.300 | False | Further analysis and validation should be conducted to assess the reliability and poten... |
| llada-8b-instruct-hf | plan_278 | True | low_confidence_32 | 1.385 | 0.893 | 1.000 | 0.000 | 0.250 | False | Compare the coverage expansion to each rubric to determine the number of weak matches. |
| llada-8b-instruct-hf | plan_278 | True | low_confidence_32 | 1.396 | 0.925 | 1.000 | 0.000 | 0.333 | False | Analyze the weak to strong matches to assess the specificity of the expansion to the ru... |
| llada-8b-instruct-hf | plan_278 | True | low_confidence_32 | 2.202 | 1.000 | 1.000 | 0.000 | 0.250 | False | Make necessary adjustments to improve the specificity of the coverage expansion. |
| llada-8b-instruct-hf | plan_279 | False | low_confidence_32 | 2.082 | 0.925 | 1.000 | 0.000 | 0.091 | False | Identifying key documentation requirements. |
| llada-8b-instruct-hf | plan_279 | False | low_confidence_32 | 2.121 | 1.000 | 1.000 | 0.000 | 0.091 | False | Standardizing documentation formats and templates. |
| llada-8b-instruct-hf | plan_279 | False | low_confidence_32 | 2.871 | 1.000 | 1.000 | 0.000 | 0.091 | False | Regularly reviewing and updating documentation to ensure accuracy and relevance. |
| llada-8b-instruct-hf | plan_280 | False | random_32 | 1.868 | 0.377 | 1.000 | 0.000 | 0.500 | False | This can include incident response, detection, mitigation, and prevention, as well as d... |
| llada-8b-instruct-hf | plan_281 | False | low_confidence_32 | 2.012 | 0.810 | 1.000 | 0.000 | 0.100 | False | If the latent variables do not change significantly within a predefined tolerance thres... |
| llada-8b-instruct-hf | plan_281 | False | low_confidence_32 | 3.377 | 1.000 | 1.000 | 0.000 | 0.000 | False | This ensures efficient use of computational resources and ensures the reliability of th... |
| llada-8b-instruct-hf | plan_282 | False | random_32 | 1.261 | 0.604 | 1.000 | 0.000 | 0.286 | False | This will help in keeping the old work distinct from the new data and experiments. |
| llada-8b-instruct-hf | plan_282 | False | random_32 | 2.140 | 0.868 | 1.000 | 0.000 | 0.214 | False | This way, the old small-n work cannot affect the integrity of the new claim. |
| llada-8b-instruct-hf | plan_283 | False | low_confidence_32 | 2.206 | 1.000 | 1.000 | 0.000 | 0.167 | False | This approach ensures the summary size is manageable without compromising the replayabi... |
| llada-8b-instruct-hf | plan_284 | True | random_32 | 1.361 | 0.790 | 1.000 | 0.000 | 0.182 | False | Identify key stakeholders and understand their constraints. |
| llada-8b-instruct-hf | plan_284 | True | random_32 | 2.060 | 0.833 | 1.000 | 0.000 | 0.000 | False | Conduct research and testing to gather feedback. |
| llada-8b-instruct-hf | plan_284 | True | random_32 | 1.943 | 0.467 | 1.000 | 0.000 | 0.364 | False | Update the candidate anchor to meet the rare constraint. |
| llada-8b-instruct-hf | plan_285 | True | low_confidence_32 | 1.277 | 0.621 | 1.000 | 0.000 | 0.222 | False | Then, the confused answer should be analyzed to understand the confusion.. |
| llada-8b-instruct-hf | plan_285 | True | low_confidence_32 | 1.427 | 0.981 | 1.000 | 0.000 | 0.333 | False | Next, the answer should be corrected to reflect the correct constraint. |
| llada-8b-instruct-hf | plan_285 | True | low_confidence_32 | 2.190 | 0.981 | 1.000 | 0.000 | 0.222 | False | Finally, the corrected answer should be verified to ensure it meets the correct accurat... |
| llada-8b-instruct-hf | plan_286 | True | low_confidence_32 | 2.067 | 0.865 | 1.000 | 0.000 | 0.083 | False | Develop a new taxonomy that incorporates these factors. |
| llada-8b-instruct-hf | plan_286 | True | low_confidence_32 | 2.064 | 0.865 | 1.000 | 0.000 | 0.083 | False | Test the new taxonomy to ensure its accuracy. |
| llada-8b-instruct-hf | plan_286 | True | low_confidence_32 | 2.086 | 0.715 | 1.000 | 0.000 | 0.167 | False | Refine the taxonomy based on testing results. |
| llada-8b-instruct-hf | plan_287 | False | low_confidence_32 | 1.975 | 0.516 | 1.000 | 0.000 | 0.273 | False | This maintains their distinct nature while highlighting how they both contribute to imp... |
| llada-8b-instruct-hf | plan_288 | False | low_confidence_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | GPU. |
| llada-8b-instruct-hf | plan_289 | True | low_confidence_32 | 2.044 | 0.801 | 1.000 | 0.000 | 0.000 | False | Keep only the relevant and concise information. |
| llada-8b-instruct-hf | plan_289 | True | low_confidence_32 | 2.060 | 0.833 | 1.000 | 0.000 | 0.000 | False | Organize the content in a logical order. |
| llada-8b-instruct-hf | plan_289 | True | low_confidence_32 | 2.156 | 0.865 | 1.000 | 0.000 | 0.167 | False | Update the report to reflect the changes and improve readability. |
| llada-8b-instruct-hf | plan_290 | True | low_confidence_32 | 2.049 | 0.850 | 1.000 | 0.000 | 0.083 | False | Evaluate the relevance of each complement based on criteria such as relevance to the ta... |
| llada-8b-instruct-hf | plan_290 | True | low_confidence_32 | 2.103 | 0.831 | 1.000 | 0.000 | 0.333 | False | Rank the complements based on these criteria and then arrange them accordingly in the f... |
| llada-8b-instruct-hf | plan_291 | True | low_confidence_32 | 2.069 | 0.905 | 1.000 | 0.000 | 0.125 | False | This evidence includes the identification and validation of the issues that are still n... |
| llada-8b-instruct-hf | plan_291 | True | low_confidence_32 | 2.182 | 1.000 | 1.000 | 0.000 | 0.375 | False | The more accurate and comprehensive the v4 coverage-gap evidence, the stronger the foun... |
| llada-8b-instruct-hf | plan_292 | False | low_confidence_32 | 2.184 | 1.000 | 1.000 | 0.000 | 0.300 | False | This rule bepans the system to require additional review and validation of the new onto... |
| llada-8b-instruct-hf | plan_293 | False | low_confidence_32 | 2.876 | 1.000 | 1.000 | 0.000 | 0.100 | False | it it would be considered useful. |
| llada-8b-instruct-hf | plan_294 | True | low_confidence_32 | 2.079 | 1.000 | 0.917 | 0.000 | 0.083 | False | This could be a simple navigation task or a basic optimization problem. |
| llada-8b-instruct-hf | plan_294 | True | low_confidence_32 | 1.281 | 0.032 | 0.167 | 0.000 | 0.833 | False | The goal here is to validate both the strong local planning evidence and the lack of sy... |
| llada-8b-instruct-hf | plan_295 | False | random_32 | 2.746 | 0.714 | 1.000 | 0.000 | 0.000 | False | Compare the benefits, costs, and risks to make an informed decision. |
| llada-8b-instruct-hf | plan_296 | False | random_32 | 1.831 | 0.238 | 1.000 | 0.000 | 0.308 | False | Include the following elements: a theoretical framework, potential evidence sources, an... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_249 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_250 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_251 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_252 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_253 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_254 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_255 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_256 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_257 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_258 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_259 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_260 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_261 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_262 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_263 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_264 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_265 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_266 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_267 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_268 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_269 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_270 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_271 | entropy_32 | entropy_64 | entropy_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_272 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_273 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_274 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_275 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_276 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_277 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_278 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_279 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_280 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_281 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_282 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.126 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_283 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_284 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_285 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_286 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_287 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_288 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_289 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_290 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.127 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_291 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_292 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_293 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.180 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_294 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_295 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_296 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_249 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.342 | 0.000 | 0.073 | 0.073 | 0.241 | 0.045 | 0.241 | 0.000 | 0.300 | 0.059 | 0.300 | 0.000 |
| llada-8b-instruct-hf | plan_250 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.294 | 0.000 | 0.052 | 0.052 | 0.281 | 0.301 | 0.281 | 0.000 | 0.310 | 0.029 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_251 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.444 | 0.000 | 0.000 | 0.000 | 0.341 | 0.045 | 0.341 | 0.000 | 0.341 | 0.000 | 0.341 | 0.000 |
| llada-8b-instruct-hf | plan_252 | low_confidence_32 | random_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.475 | 0.000 | 0.113 | 0.113 | 0.273 | 0.391 | 0.391 | 0.000 | 0.459 | 0.068 | 0.459 | 0.000 |
| llada-8b-instruct-hf | plan_253 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.398 | 0.000 | 0.090 | 0.090 | 0.301 | 0.157 | 0.301 | 0.000 | 0.407 | 0.106 | 0.407 | 0.000 |
| llada-8b-instruct-hf | plan_254 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.436 | 0.000 | 0.059 | 0.059 | 0.240 | 0.240 | 0.240 | 0.000 | 0.277 | 0.037 | 0.277 | 0.000 |
| llada-8b-instruct-hf | plan_255 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.462 | 0.000 | 0.000 | 0.000 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_256 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.312 | 0.000 | 0.065 | 0.065 | 0.200 | 0.200 | 0.200 | 0.000 | 0.242 | 0.042 | 0.242 | 0.000 |
| llada-8b-instruct-hf | plan_257 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.419 | 0.000 | 0.000 | 0.000 | 0.324 | 0.324 | 0.324 | 0.000 | 0.324 | 0.000 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_258 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.271 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.065 | 0.000 | 0.065 | 0.000 | 0.065 | 0.000 |
| llada-8b-instruct-hf | plan_259 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.422 | 0.000 | 0.046 | 0.046 | 0.303 | 0.303 | 0.303 | 0.000 | 0.324 | 0.021 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_260 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.489 | 0.000 | 0.000 | 0.000 | 0.381 | 0.381 | 0.381 | 0.000 | 0.381 | 0.000 | 0.381 | 0.000 |
| llada-8b-instruct-hf | plan_261 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.380 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_262 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.402 | 0.000 | 0.048 | 0.048 | 0.418 | 0.418 | 0.418 | 0.000 | 0.425 | 0.008 | 0.425 | 0.000 |
| llada-8b-instruct-hf | plan_263 | low_confidence_32 | low_confidence_32 | random_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | history | 31 | 0.277 | 0.000 | 0.081 | 0.081 | 0.045 | 0.045 | 0.117 | 0.000 | 0.200 | 0.083 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_264 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.342 | 0.000 | 0.044 | 0.044 | 0.241 | 0.241 | 0.241 | 0.000 | 0.283 | 0.041 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_265 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.391 | 0.000 | 0.032 | 0.032 | 0.280 | 0.217 | 0.280 | 0.000 | 0.292 | 0.012 | 0.292 | 0.000 |
| llada-8b-instruct-hf | plan_266 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.394 | 0.000 | 0.000 | 0.000 | 0.330 | 0.217 | 0.330 | 0.000 | 0.330 | 0.000 | 0.330 | 0.000 |
| llada-8b-instruct-hf | plan_267 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.446 | 0.000 | 0.000 | 0.000 | 0.301 | 0.280 | 0.301 | 0.000 | 0.301 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_268 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.485 | 0.000 | 0.099 | 0.099 | 0.345 | 0.345 | 0.345 | 0.000 | 0.400 | 0.055 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_269 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.400 | 0.000 | 0.000 | 0.000 | 0.220 | 0.045 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_270 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.395 | 0.000 | 0.000 | 0.000 | 0.319 | 0.319 | 0.319 | 0.000 | 0.319 | 0.000 | 0.339 | 0.020 |
| llada-8b-instruct-hf | plan_271 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.388 | 0.000 | 0.000 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_272 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.501 | 0.000 | 0.144 | 0.144 | 0.404 | 0.333 | 0.404 | 0.000 | 0.496 | 0.093 | 0.496 | 0.000 |
| llada-8b-instruct-hf | plan_273 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.414 | 0.000 | 0.000 | 0.000 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_274 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.429 | 0.000 | 0.166 | 0.166 | 0.309 | 0.309 | 0.309 | 0.000 | 0.426 | 0.118 | 0.426 | 0.000 |
| llada-8b-instruct-hf | plan_275 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.426 | 0.000 | 0.101 | 0.101 | 0.391 | 0.391 | 0.391 | 0.000 | 0.450 | 0.059 | 0.450 | 0.000 |
| llada-8b-instruct-hf | plan_276 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.420 | 0.000 | 0.000 | 0.000 | 0.260 | 0.200 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_277 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.469 | 0.000 | 0.000 | 0.000 | 0.334 | 0.334 | 0.334 | 0.000 | 0.334 | 0.000 | 0.334 | 0.000 |
| llada-8b-instruct-hf | plan_278 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.358 | 0.000 | 0.126 | 0.126 | 0.241 | 0.200 | 0.241 | 0.000 | 0.338 | 0.096 | 0.338 | 0.000 |
| llada-8b-instruct-hf | plan_279 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.343 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_280 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.391 | 0.000 | 0.000 | 0.000 | 0.303 | 0.303 | 0.273 | 0.000 | 0.273 | 0.000 | 0.303 | 0.030 |
| llada-8b-instruct-hf | plan_281 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.395 | 0.000 | 0.000 | 0.000 | 0.296 | 0.178 | 0.296 | 0.000 | 0.296 | 0.000 | 0.296 | 0.000 |
| llada-8b-instruct-hf | plan_282 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.381 | 0.000 | 0.000 | 0.000 | 0.293 | 0.293 | 0.255 | 0.000 | 0.255 | 0.000 | 0.293 | 0.038 |
| llada-8b-instruct-hf | plan_283 | low_confidence_32 | random_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.329 | 0.000 | 0.053 | 0.053 | 0.213 | 0.241 | 0.213 | 0.000 | 0.243 | 0.030 | 0.243 | 0.000 |
| llada-8b-instruct-hf | plan_284 | low_confidence_32 | random_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.412 | 0.000 | 0.042 | 0.042 | 0.345 | 0.366 | 0.366 | 0.000 | 0.362 | -0.004 | 0.366 | 0.004 |
| llada-8b-instruct-hf | plan_285 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.452 | 0.000 | 0.137 | 0.137 | 0.395 | 0.395 | 0.395 | 0.000 | 0.488 | 0.093 | 0.488 | 0.000 |
| llada-8b-instruct-hf | plan_286 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.345 | 0.000 | 0.063 | 0.063 | 0.234 | 0.197 | 0.234 | 0.000 | 0.331 | 0.097 | 0.331 | 0.000 |
| llada-8b-instruct-hf | plan_287 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.433 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_288 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.315 | 0.000 | 0.083 | 0.083 | 0.178 | 0.178 | 0.178 | 0.000 | 0.241 | 0.063 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_289 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.401 | 0.000 | 0.048 | 0.048 | 0.324 | 0.324 | 0.324 | 0.000 | 0.325 | 0.001 | 0.325 | 0.000 |
| llada-8b-instruct-hf | plan_290 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.393 | 0.000 | 0.212 | 0.212 | 0.233 | 0.233 | 0.233 | 0.000 | 0.444 | 0.212 | 0.444 | 0.000 |
| llada-8b-instruct-hf | plan_291 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.360 | 0.000 | 0.079 | 0.079 | 0.180 | 0.180 | 0.180 | 0.000 | 0.235 | 0.055 | 0.235 | 0.000 |
| llada-8b-instruct-hf | plan_292 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.425 | 0.000 | 0.096 | 0.096 | 0.339 | 0.339 | 0.339 | 0.000 | 0.394 | 0.055 | 0.394 | 0.000 |
| llada-8b-instruct-hf | plan_293 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.393 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_294 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.446 | 0.000 | 0.143 | 0.143 | 0.311 | 0.190 | 0.311 | 0.000 | 0.413 | 0.101 | 0.413 | 0.000 |
| llada-8b-instruct-hf | plan_295 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.451 | 0.000 | 0.000 | 0.000 | 0.291 | 0.291 | 0.414 | 0.000 | 0.414 | 0.000 | 0.414 | 0.000 |
| llada-8b-instruct-hf | plan_296 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.460 | 0.000 | 0.000 | 0.000 | 0.350 | 0.329 | 0.329 | 0.000 | 0.329 | 0.000 | 0.350 | 0.021 |
