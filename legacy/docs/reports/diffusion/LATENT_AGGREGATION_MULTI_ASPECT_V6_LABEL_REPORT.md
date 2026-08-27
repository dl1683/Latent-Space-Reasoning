# Diffusion Schedule-Selection Benchmark Report

Full model generations: `336`
Counterfactual probe generations: `0`
Arm selections: `336`
Run ID: `diffusion-18eedf4ffa0e69ac`
Content hash: `18eedf4ffa0e69aca5606f518ec161780b26c0d019b1d6d396c290697d901804`
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
Trajectory task delta vs fixed: `0.007`
Trajectory task delta vs random: `0.027`
Trajectory wins/ties/losses vs fixed: `14/79/3`
Trajectory wins/ties/losses vs random: `31/60/5`
Oracle generation budget/task: `3.50`
Oracle task score: `0.171`
Oracle headroom vs trajectory: `0.024`
Oracle wins/ties/losses vs trajectory: `30/66/0`
Selector regret vs trajectory: `0.024 over 30/96 improvable`
Repair arm coverage: `48/96` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.043`
Repair task delta vs random: `0.088`
Repair task delta vs trajectory: `0.039`
Repair task delta vs evolved: `0.039`
Repair generation budget delta vs evolved: `2.00`
Repair task delta per extra generation vs evolved: `0.020`
Repair wins/ties/losses vs evolved: `21/26/1`
Oracle headroom vs repair: `0.006`
Oracle wins/ties/losses vs repair: `9/39/0`
Selector regret vs repair: `0.006 over 9/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/96` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.263757 | 0.000000 | 0.044804 | - | - |
| random perturbation | repair-covered tasks | 0.218954 | -0.044804 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.306533 | 0.042775 | 0.087579 | 24/22/2 | 34/12/2 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 96 | 1.00 | 0.140 | 0.371 | 0.197 |
| random | 96 | 1.00 | 0.120 | 0.319 | 0.170 |
| trajectory_selected | 96 | 2.50 | 0.147 | 0.386 | 0.207 |
| repair_selected | 48 | 4.00 | 0.307 | 0.657 | 0.394 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 96 | 1.00 | 0.140 | 0.371 | 0.197 |
| planning | random | 96 | 1.00 | 0.120 | 0.319 | 0.170 |
| planning | trajectory_selected | 96 | 2.50 | 0.147 | 0.386 | 0.207 |
| planning | repair_selected | 48 | 4.00 | 0.307 | 0.657 | 0.394 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_297 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 341 | True | 7 | 0.500 | True | True | 4.000 | 0.125 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.323 | 0.223 | 298 | True | 2 | 0.857 | True | True | 3.000 | 0.094 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.282 | 0.223 | 354 | True | 4 | 0.714 | True | True | 3.000 | 0.094 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_300 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.263 | 0.223 | 375 | True | 7 | 0.500 | True | True | 4.000 | 0.125 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.290 | 0.230 | 276 | True | 2 | 0.818 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_302 | random_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 314 | True | 4 | 0.667 | True | True | 3.000 | 0.094 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_303 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.283 | 0.223 | 334 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_304 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.299 | 0.239 | 328 | True | 7 | 0.417 | True | True | 3.000 | 0.094 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_305 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.360 | 0.260 | 344 | True | 6 | 0.538 | True | True | 4.000 | 0.125 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_306 | random_32 | True | denoise_phase_repairable | False |  | 0.422 | 0.324 | 267 | True | 5 | 0.615 | True | True | 3.000 | 0.094 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_307 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 344 | True | 4 | 0.692 | True | True | 4.000 | 0.125 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_308 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 320 | True | 6 | 0.538 | True | True | 3.000 | 0.094 | 0.308 | 0.308 |
| llada-8b-instruct-hf | plan_309 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.356 | 0.239 | 370 | True | 2 | 0.800 | True | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_310 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 358 | True | 3 | 0.727 | True | True | 4.000 | 0.125 | 0.091 | 0.091 |
| llada-8b-instruct-hf | plan_311 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.284 | 0.244 | 289 | True | 2 | 0.818 | True | True | 4.000 | 0.125 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_312 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.241 | 0.201 | 343 | True | 1 | 0.900 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_313 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.221 | 0.201 | 360 | True | 4 | 0.636 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_314 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 352 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 327 | True | 0 | 1.000 | True | True | 3.000 | 0.094 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_316 | random_32 | True | denoise_phase_repairable | False |  | 0.263 | 0.223 | 288 | True | 4 | 0.692 | True | True | 3.000 | 0.094 | 0.077 | 0.077 |
| llada-8b-instruct-hf | plan_317 | random_32 | True | denoise_phase_repairable | False |  | 0.277 | 0.217 | 285 | True | 4 | 0.667 | True | True | 5.000 | 0.156 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_318 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 312 | True | 11 | 0.286 | True | True | 4.000 | 0.125 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_319 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 328 | True | 2 | 0.818 | True | True | 2.000 | 0.062 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_320 | random_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 269 | True | 1 | 0.900 | True | True | 7.000 | 0.219 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.344 | 0.244 | 357 | True | 0 | 1.000 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_322 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 387 | True | 3 | 0.750 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_323 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 364 | True | 5 | 0.583 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_324 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.220 | 0.180 | 305 | True | 6 | 0.500 | True | True | 3.000 | 0.094 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.296 | 0.256 | 364 | True | 3 | 0.727 | True | True | 4.000 | 0.125 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.284 | 0.244 | 305 | True | 1 | 0.917 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_327 | random_32 | True | denoise_phase_repairable | False |  | 0.297 | 0.217 | 286 | True | 4 | 0.727 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_328 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 314 | True | 5 | 0.583 | True | True | 4.000 | 0.125 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.304 | 0.244 | 371 | True | 2 | 0.818 | True | True | 4.000 | 0.125 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.402 | 0.362 | 359 | True | 7 | 0.417 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_331 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.223 | 0.223 | 213 | True | 8 | 0.000 | True | True | 6.000 | 0.188 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_332 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.240 | 0.180 | 284 | True | 4 | 0.600 | True | True | 4.000 | 0.125 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_333 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.280 | 0.180 | 320 | True | 3 | 0.667 | True | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_334 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.260 | 0.180 | 237 | True | 7 | 0.500 | True | True | 4.000 | 0.125 | 0.083 | 0.083 |
| llada-8b-instruct-hf | plan_335 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.309 | 0.269 | 340 | True | 2 | 0.800 | True | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_336 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.255 | 0.235 | 363 | True | 2 | 0.750 | True | True | 4.000 | 0.125 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.242 | 0.223 | 390 | True | 3 | 0.625 | True | True | 5.000 | 0.156 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_338 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.315 | 0.217 | 313 | True | 2 | 0.750 | True | True | 4.000 | 0.125 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_339 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.268 | 0.247 | 210 | True | 1 | 0.889 | True | True | 6.000 | 0.188 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_340 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.261 | 0.201 | 354 | True | 7 | 0.571 | True | True | 5.000 | 0.156 | 0.071 | 0.071 |
| llada-8b-instruct-hf | plan_341 | random_32 | True | denoise_phase_repairable | False |  | 0.121 | 0.121 | 43 | True | 7 | 0.300 | True | True | 8.000 | 0.250 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_342 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.268 | 0.247 | 381 | True | 3 | 0.786 | True | True | 3.000 | 0.094 | 0.143 | 0.143 |
| llada-8b-instruct-hf | plan_343 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.180 | 0.180 | 337 | True | 4 | 0.600 | True | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_344 | low_confidence_32 | True | denoise_phase_repairable | False |  | 0.200 | 0.180 | 364 | True | 5 | 0.636 | True | True | 3.000 | 0.094 | 0.182 | 0.182 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_phase_final_preserve_seeded_gated_repair | 48 | 20 | low_confidence_32,random_32 | final | 28.5 | 1.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.032 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 22/14/12 | 0.299 | 0.668 | 0.391 |
| history_prefix_25_repair | 48 | 2 | low_confidence_32,random_32 | history | 48.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.005 | -0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/24/15 | 0.261 | 0.662 | 0.361 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_297 | False | low_confidence_32 | 2.166 | 0.905 | 1.000 | 0.000 | 0.143 | False | This could involve incorporating new data,, modifying the model architecture, or experi... |
| llada-8b-instruct-hf | plan_298 | False | low_confidence_32 | 1.365 | 0.790 | 1.000 | 0.000 | 0.143 | False | Identify the strong anchor. |
| llada-8b-instruct-hf | plan_298 | False | low_confidence_32 | 1.400 | 0.865 | 1.000 | 0.000 | 0.143 | False | Look for keywords related to stakeholder constraints. |
| llada-8b-instruct-hf | plan_298 | False | low_confidence_32 | 2.072 | 0.733 | 1.000 | 0.000 | 0.286 | False | Update the task plan to include the stakeholder constraints. |
| llada-8b-instruct-hf | plan_299 | False | low_confidence_32 | 1.914 | 0.445 | 1.000 | 0.000 | 0.429 | False | This gate should filter out additions that do not meet the criteria for being a real co... |
| llada-8b-instruct-hf | plan_300 | False | low_confidence_32 | 1.295 | 0.686 | 1.000 | 0.000 | 0.214 | False | During the analyze phase, evaluate the performance of the generated anchors and compare... |
| llada-8b-instruct-hf | plan_300 | False | low_confidence_32 | 2.169 | 0.968 | 1.000 | 0.000 | 0.286 | False | Use specific metrics such as anchor accuracy and fusion accuracy to distinguish between... |
| llada-8b-instruct-hf | plan_301 | True | low_confidence_32 | 1.962 | 0.544 | 1.000 | 0.000 | 0.364 | False | This way, the new will be based on the full evidence from both v5 and v6, avoiding any... |
| llada-8b-instruct-hf | plan_302 | False | random_32 | 1.955 | 0.522 | 1.000 | 0.000 | 0.444 | False | This penalty would increase the more terms from the task prompt the model's output cont... |
| llada-8b-instruct-hf | plan_303 | False | low_confidence_32 | 2.857 | 0.968 | 1.000 | 0.000 | 0.100 | False | Evaluate data and performance metrics to make an informed decision. |
| llada-8b-instruct-hf | plan_304 | True | low_confidence_32 | 1.989 | 0.715 | 1.000 | 0.000 | 0.083 | False | Analyze historical data to identify effective candidates. |
| llada-8b-instruct-hf | plan_304 | True | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Use machine learning to optimize candidate selection. |
| llada-8b-instruct-hf | plan_304 | True | low_confidence_32 | 2.742 | 0.702 | 1.000 | 0.000 | 0.000 | False | Continuously monitor and adjust strategies based on performance metrics. |
| llada-8b-instruct-hf | plan_305 | False | low_confidence_32 | 1.283 | 0.632 | 1.000 | 0.000 | 0.231 | False | Identify new aspects that capture differentiate between candidates. |
| llada-8b-instruct-hf | plan_305 | False | low_confidence_32 | 1.425 | 0.910 | 1.000 | 0.000 | 0.154 | False | Integrate these aspects into the existing ontology. |
| llada-8b-instruct-hf | plan_305 | False | low_confidence_32 | 2.027 | 0.652 | 1.000 | 0.000 | 0.231 | False | Validate the expanded ontology to ensure it accurately represents the original candidat... |
| llada-8b-instruct-hf | plan_306 | True | random_32 | 2.545 | 0.820 | 1.000 | 0.000 | 0.077 | False | Assign owners to each task. |
| llada-8b-instruct-hf | plan_306 | True | random_32 | 2.545 | 0.820 | 1.000 | 0.000 | 0.077 | False | Define timelines for each task. |
| llada-8b-instruct-hf | plan_306 | True | random_32 | 2.550 | 0.680 | 1.000 | 0.000 | 0.308 | False | Integrate these aspects into the planning answer. |
| llada-8b-instruct-hf | plan_307 | False | low_confidence_32 | 2.165 | 0.905 | 1.000 | 0.000 | 0.154 | False | This approach acknowledges the candidates' effort to provide specific and tangible info... |
| llada-8b-instruct-hf | plan_308 | True | low_confidence_32 | 2.078 | 0.925 | 1.000 | 0.000 | 0.077 | False | This involves determining the order in which data will be collected, processed, and agg... |
| llada-8b-instruct-hf | plan_308 | True | low_confidence_32 | 2.821 | 0.893 | 1.000 | 0.000 | 0.000 | False | By establishing a clear timeline, we can ensure that the data is taken in and analyzed... |
| llada-8b-instruct-hf | plan_309 | False | low_confidence_32 | 2.111 | 0.856 | 1.000 | 0.000 | 0.300 | False | This representation acknowledges the presence of a risk but does not provide a method t... |
| llada-8b-instruct-hf | plan_310 | True | low_confidence_32 | 2.115 | 1.000 | 1.000 | 0.000 | 0.091 | False | This will allow for more detailed and nuanced assessments of performance without relyin... |
| llada-8b-instruct-hf | plan_310 | True | low_confidence_32 | 2.165 | 0.925 | 1.000 | 0.000 | 0.182 | False | This approach can help identify specific strengths and areas for further improvement th... |
| llada-8b-instruct-hf | plan_311 | True | low_confidence_32 | 1.358 | 0.790 | 1.000 | 0.000 | 0.182 | False | Review the ontology expansion to identify potential issues. |
| llada-8b-instruct-hf | plan_311 | True | low_confidence_32 | 1.383 | 0.865 | 1.000 | 0.000 | 0.273 | False | Execute the test case to check for false positives. |
| llada-8b-instruct-hf | plan_311 | True | low_confidence_32 | 2.119 | 0.865 | 1.000 | 0.000 | 0.364 | False | Make necessary adjustments to the ontology expansion to minimize false positives. |
| llada-8b-instruct-hf | plan_312 | False | low_confidence_32 | 1.440 | 1.000 | 1.000 | 0.000 | 0.300 | False | This means that the extractor should not rely on predefined labels for the input data. |
| llada-8b-instruct-hf | plan_312 | False | low_confidence_32 | 2.151 | 0.925 | 1.000 | 0.000 | 0.300 | False | Instead, it should be able to identify and extract relevant aspects from the input base... |
| llada-8b-instruct-hf | plan_313 | False | low_confidence_32 | 2.130 | 0.873 | 1.000 | 0.000 | 0.273 | False | This approach would allow the system to adapt to new data and generate unique perturbat... |
| llada-8b-instruct-hf | plan_314 | False | low_confidence_32 | 2.751 | 0.742 | 1.000 | 0.000 | 0.083 | False | collect the data, conduct the experiment, and analyze the results. |
| llada-8b-instruct-hf | plan_315 | True | low_confidence_32 | 1.311 | 0.925 | 0.778 | 0.000 | 0.222 | False | This audit will involve comparing the paraphrased to the original text to identify any... |
| llada-8b-instruct-hf | plan_315 | True | low_confidence_32 | 2.100 | 1.000 | 0.778 | 0.000 | 0.222 | False | The audit will also involve for any noise in the paraphrased text that changes the mean... |
| llada-8b-instruct-hf | plan_316 | False | random_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | - Decide on the data used to train the model |
| llada-8b-instruct-hf | plan_317 | False | random_32 | 2.067 | 0.865 | 1.000 | 0.000 | 0.083 | False | Run the updated model with the new tasks. |
| llada-8b-instruct-hf | plan_317 | False | random_32 | 2.759 | 0.734 | 1.000 | 0.000 | 0.000 | False | Monitor performance and iterate as needed. |
| llada-8b-instruct-hf | plan_318 | True | low_confidence_32 | 2.129 | 1.000 | 1.000 | 0.000 | 0.071 | False | Calculate the total height of the ladder. |
| llada-8b-instruct-hf | plan_318 | True | low_confidence_32 | 2.143 | 1.000 | 1.000 | 0.000 | 0.071 | False | Design the ladder structure, steps,, steps, and anchor points. |
| llada-8b-instruct-hf | plan_318 | True | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | Consider safety, accessibility, and maintenance requirements. |
| llada-8b-instruct-hf | plan_319 | False | low_confidence_32 | 2.865 | 1.000 | 1.000 | 0.000 | 0.091 | False | This information will be crucial for both you and I to proceed with the audit. |
| llada-8b-instruct-hf | plan_320 | False | random_32 | 1.985 | 0.688 | 1.000 | 0.000 | 0.100 | False | Determine the specific scope of the audit. |
| llada-8b-instruct-hf | plan_320 | False | random_32 | 2.001 | 0.708 | 1.000 | 0.000 | 0.000 | False | Document the findings. |
| llada-8b-instruct-hf | plan_320 | False | random_32 | 2.778 | 0.782 | 1.000 | 0.000 | 0.100 | False | Develop a corrective action plan. |
| llada-8b-instruct-hf | plan_321 | True | low_confidence_32 | 2.036 | 1.000 | 0.667 | 0.000 | 0.333 | False | If scope drift is detected, corrective actions should be taken to maintain the compleme... |
| llada-8b-instruct-hf | plan_322 | False | low_confidence_32 | 1.326 | 0.775 | 1.000 | 0.000 | 0.333 | False | This involves assigning a polarity to each concept in the ontology, which can help iden... |
| llada-8b-instruct-hf | plan_322 | False | low_confidence_32 | 2.069 | 0.850 | 1.000 | 0.000 | 0.583 | False | By incorporating polarity, we can create a stronger semantic contradiction layer and im... |
| llada-8b-instruct-hf | plan_323 | False | low_confidence_32 | 1.456 | 1.000 | 1.000 | 0.000 | 0.167 | False | Determine the impact of each aspect on the overall outcome and prioritize the aspects b... |
| llada-8b-instruct-hf | plan_323 | False | low_confidence_32 | 2.107 | 0.850 | 1.000 | 0.000 | 0.417 | False | Ensure that the answer remains coherent and concise while accounting for the potential... |
| llada-8b-instruct-hf | plan_324 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | If the content is deemed harmful, it should be removed immediately. |
| llada-8b-instruct-hf | plan_324 | False | low_confidence_32 | 2.888 | 1.000 | 1.000 | 0.000 | 0.000 | False | If it is uncertain, it should be flagged for review. |
| llada-8b-instruct-hf | plan_325 | True | low_confidence_32 | 1.370 | 0.839 | 1.000 | 0.000 | 0.273 | False | Evaluate the potential benefits and drawbacks of making it the default source versus al... |
| llada-8b-instruct-hf | plan_325 | True | low_confidence_32 | 1.479 | 1.000 | 1.000 | 0.000 | 0.182 | False | Consider factors such as cost efficiency, performance performance, and resource feasibi... |
| llada-8b-instruct-hf | plan_325 | True | low_confidence_32 | 2.159 | 0.917 | 1.000 | 0.000 | 0.273 | False | Conduct a detailed cost-benefit analysis before making the final decision. |
| llada-8b-instruct-hf | plan_326 | False | low_confidence_32 | 2.160 | 1.000 | 1.000 | 0.000 | 0.417 | False | This ensures that the source family is only used for low-anchor-score tasks. |
| llada-8b-instruct-hf | plan_327 | False | random_32 | 1.973 | 0.659 | 1.000 | 0.000 | 0.091 | False | Implement cost-effective diagnostic strategies. |
| llada-8b-instruct-hf | plan_327 | False | random_32 | 2.739 | 0.689 | 1.000 | 0.000 | 0.000 | False | Monitor results and adjust strategies as needed. |
| llada-8b-instruct-hf | plan_328 | True | low_confidence_32 | 2.038 | 0.790 | 1.000 | 0.000 | 0.000 | False | Identify potential areas for improvement or further investigation. |
| llada-8b-instruct-hf | plan_328 | True | low_confidence_32 | 1.999 | 0.715 | 1.000 | 0.000 | 0.083 | False | Develop a plan for exploring alternatives or alternative approaches. |
| llada-8b-instruct-hf | plan_328 | True | low_confidence_32 | 2.156 | 0.865 | 1.000 | 0.000 | 0.167 | False | Update the roadmap with new insights and priorities. |
| llada-8b-instruct-hf | plan_329 | True | low_confidence_32 | 2.156 | 0.960 | 1.000 | 0.000 | 0.364 | False | This boundary outlines the scope of the evaluation and analysis required to demonstrate... |
| llada-8b-instruct-hf | plan_330 | False | low_confidence_32 | 2.011 | 0.797 | 1.000 | 0.000 | 0.083 | False | Allocate a fixed of variable budget to each task, adjusting based on real-time data and... |
| llada-8b-instruct-hf | plan_330 | False | low_confidence_32 | 2.196 | 1.000 | 1.000 | 0.000 | 0.167 | False | This approach ensures that resources are used efficiently, reducing the likelihood of t... |
| llada-8b-instruct-hf | plan_331 | False | low_confidence_32 | 1.977 | 0.643 | 1.000 | 0.000 | 0.000 | False | Set up environment. |
| llada-8b-instruct-hf | plan_331 | False | low_confidence_32 | 1.960 | 0.611 | 1.000 | 0.000 | 0.000 | False | Record the results. |
| llada-8b-instruct-hf | plan_331 | False | low_confidence_32 | 2.666 | 0.518 | 1.000 | 0.000 | 0.000 | False | Document any discrepancies or issues. |
| llada-8b-instruct-hf | plan_332 | True | low_confidence_32 | 2.065 | 0.865 | 1.000 | 0.000 | 0.100 | False | Update the README with the necessary information. |
| llada-8b-instruct-hf | plan_332 | True | low_confidence_32 | 2.065 | 0.865 | 1.000 | 0.000 | 0.100 | False | Review the updated README to ensure its usability and relevance. |
| llada-8b-instruct-hf | plan_332 | True | low_confidence_32 | 2.153 | 0.865 | 1.000 | 0.000 | 0.200 | False | Commit the changes and the README to the repository. |
| llada-8b-instruct-hf | plan_333 | True | low_confidence_32 | 2.113 | 1.000 | 1.000 | 0.000 | 0.111 | False | Could you please provide more context or details about the aggregation results you're w... |
| llada-8b-instruct-hf | plan_333 | True | low_confidence_32 | 2.094 | 0.808 | 1.000 | 0.000 | 0.333 | False | Once I have a better understanding of the results, I can help you create a status table... |
| llada-8b-instruct-hf | plan_334 | False | low_confidence_32 | 1.990 | 0.708 | 1.000 | 0.000 | 0.083 | False | Verify ` artifact_path` is set. |
| llada-8b-instruct-hf | plan_334 | False | low_confidence_32 | 2.029 | 0.782 | 1.000 | 0.000 | 0.083 | False | Make the commit. |
| llada-8b-instruct-hf | plan_334 | False | low_confidence_32 | 2.737 | 0.688 | 1.000 | 0.000 | 0.083 | False | Push the commit to the remote branch if repository. |
| llada-8b-instruct-hf | plan_335 | False | low_confidence_32 | 2.188 | 0.981 | 1.000 | 0.000 | 0.200 | False | If the title is incorrect, it will be corrected before being included in the final report. |
| llada-8b-instruct-hf | plan_336 | True | low_confidence_32 | 2.127 | 1.000 | 1.000 | 0.000 | 0.000 | False | Use concise language and specific examples to illustrate the benefits of the changes. |
| llada-8b-instruct-hf | plan_336 | True | low_confidence_32 | 1.987 | 0.575 | 1.000 | 0.000 | 0.375 | False | This approach help build user interest and trust while avoiding the pitfalls of making... |
| llada-8b-instruct-hf | plan_337 | True | low_confidence_32 | 2.862 | 1.000 | 1.000 | 0.000 | 0.125 | False | The discussion highlights the local nature of the results and the implications for broa... |
| llada-8b-instruct-hf | plan_338 | True | low_confidence_32 | 1.851 | 0.787 | 1.000 | 0.000 | 0.250 | False | The appendix will be structured to a way that allows for easy reference and analysis of... |
| llada-8b-instruct-hf | plan_338 | True | low_confidence_32 | 3.362 | 1.000 | 1.000 | 0.000 | 0.125 | False | This will include a description of the purpose of each run, the parameters used, and th... |
| llada-8b-instruct-hf | plan_339 | True | low_confidence_32 | 2.034 | 0.770 | 1.000 | 0.000 | 0.000 | False | Discuss any challenges. |
| llada-8b-instruct-hf | plan_339 | True | low_confidence_32 | 1.989 | 0.702 | 1.000 | 0.000 | 0.111 | False | Confirm the next steps. |
| llada-8b-instruct-hf | plan_339 | True | low_confidence_32 | 2.784 | 0.770 | 1.000 | 0.000 | 0.000 | False | Signature:: Date: End. |
| llada-8b-instruct-hf | plan_340 | False | low_confidence_32 | 2.101 | 0.968 | 1.000 | 0.000 | 0.071 | False | This will help determine the required sample size to detect a statistically effect, sta... |
| llada-8b-instruct-hf | plan_341 | False | random_32 | 1.862 | 0.277 | 1.000 | 0.000 | 0.300 | False | Plan a doctrine check after v6. |
| llada-8b-instruct-hf | plan_342 | True | low_confidence_32 | 2.050 | 0.698 | 1.000 | 0.000 | 0.214 | False | This could involve creating a common latent space framework, using techniques such as l... |
| llada-8b-instruct-hf | plan_343 | True | low_confidence_32 | 1.434 | 1.000 | 1.000 | 0.000 | 0.300 | False | Could you provide more information about the proof objects, such as their definitions,... |
| llada-8b-instruct-hf | plan_343 | True | low_confidence_32 | 2.184 | 1.000 | 1.000 | 0.000 | 0.300 | False | This will help me create a concise and that captures the essence of the proof objects. |
| llada-8b-instruct-hf | plan_344 | False | low_confidence_32 | 2.153 | 0.905 | 1.000 | 0.000 | 0.182 | False | This involves creating tasks that require more complex reasoning, deeper language under... |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_297 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_298 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_299 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_300 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_301 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_302 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_303 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_304 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_305 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_306 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.113 | 0.000 | 0.000 | 0.000 | 0.000 | 0.180 | 0.180 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_307 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_308 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_309 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_310 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_311 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_312 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_313 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_314 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_315 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_316 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_317 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_318 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_319 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_320 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_321 | entropy_32 | entropy_32 | entropy_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.117 | 0.117 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_322 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_323 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_324 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_325 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_326 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_327 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_328 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_329 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_330 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_331 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_332 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_333 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_334 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_335 | entropy_32 | entropy_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_336 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_337 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_338 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_339 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_340 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_341 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_342 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_343 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_344 | entropy_32 | entropy_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_297 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.329 | 0.000 | 0.000 | 0.000 | 0.220 | 0.157 | 0.220 | 0.000 | 0.220 | 0.000 | 0.240 | 0.020 |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.436 | 0.000 | 0.000 | 0.000 | 0.323 | 0.323 | 0.323 | 0.000 | 0.323 | 0.000 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.393 | 0.000 | 0.000 | 0.000 | 0.282 | 0.282 | 0.282 | 0.000 | 0.282 | 0.000 | 0.302 | 0.020 |
| llada-8b-instruct-hf | plan_300 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.348 | 0.000 | 0.000 | 0.000 | 0.263 | 0.241 | 0.263 | 0.000 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | low_confidence_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.428 | 0.000 | 0.029 | 0.029 | 0.290 | 0.137 | 0.290 | 0.000 | 0.275 | -0.015 | 0.290 | 0.015 |
| llada-8b-instruct-hf | plan_302 | low_confidence_32 | random_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.346 | 0.000 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_303 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.456 | 0.000 | 0.000 | 0.000 | 0.283 | 0.198 | 0.283 | 0.000 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_304 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.332 | 0.000 | 0.071 | 0.071 | 0.299 | 0.299 | 0.299 | 0.000 | 0.362 | 0.063 | 0.362 | 0.000 |
| llada-8b-instruct-hf | plan_305 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.379 | 0.000 | 0.000 | 0.000 | 0.360 | 0.106 | 0.360 | 0.000 | 0.360 | 0.000 | 0.360 | 0.000 |
| llada-8b-instruct-hf | plan_306 | low_confidence_32 | low_confidence_32 | random_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | random_32 | final |  | 0.439 | 0.000 | 0.090 | 0.090 | 0.402 | 0.402 | 0.422 | 0.000 | 0.486 | 0.064 | 0.486 | 0.000 |
| llada-8b-instruct-hf | plan_307 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.377 | 0.000 | 0.000 | 0.000 | 0.200 | 0.200 | 0.200 | 0.000 | 0.200 | 0.000 | 0.200 | 0.000 |
| llada-8b-instruct-hf | plan_308 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.353 | 0.000 | 0.140 | 0.140 | 0.241 | 0.241 | 0.241 | 0.000 | 0.381 | 0.140 | 0.381 | 0.000 |
| llada-8b-instruct-hf | plan_309 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.448 | 0.000 | 0.000 | 0.000 | 0.356 | 0.336 | 0.356 | 0.000 | 0.356 | 0.000 | 0.356 | 0.000 |
| llada-8b-instruct-hf | plan_310 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.387 | 0.000 | 0.130 | 0.130 | 0.260 | 0.045 | 0.260 | 0.000 | 0.361 | 0.101 | 0.361 | 0.000 |
| llada-8b-instruct-hf | plan_311 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.436 | 0.000 | 0.097 | 0.097 | 0.284 | 0.284 | 0.284 | 0.000 | 0.343 | 0.059 | 0.343 | 0.000 |
| llada-8b-instruct-hf | plan_312 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.448 | 0.000 | 0.000 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_313 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.377 | 0.000 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_314 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.380 | 0.000 | 0.000 | 0.000 | 0.240 | 0.218 | 0.240 | 0.000 | 0.240 | 0.000 | 0.260 | 0.020 |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.463 | 0.000 | 0.204 | 0.204 | 0.240 | 0.261 | 0.240 | 0.000 | 0.400 | 0.160 | 0.400 | 0.000 |
| llada-8b-instruct-hf | plan_316 | low_confidence_32 | random_32 | random_32 |  | random_32 | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.389 | 0.000 | 0.000 | 0.000 | 0.263 | 0.263 | 0.263 | 0.000 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_317 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.377 | 0.000 | 0.000 | 0.000 | 0.220 | 0.220 | 0.277 | 0.000 | 0.277 | 0.000 | 0.277 | 0.000 |
| llada-8b-instruct-hf | plan_318 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.258 | 0.000 | 0.148 | 0.148 | 0.200 | 0.200 | 0.200 | 0.000 | 0.338 | 0.138 | 0.338 | 0.000 |
| llada-8b-instruct-hf | plan_319 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.412 | 0.000 | 0.000 | 0.000 | 0.200 | 0.261 | 0.200 | 0.000 | 0.200 | 0.000 | 0.261 | 0.061 |
| llada-8b-instruct-hf | plan_320 | low_confidence_32 | random_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.382 | 0.000 | 0.000 | 0.000 | 0.294 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.294 | 0.034 |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.493 | 0.000 | 0.072 | 0.072 | 0.344 | 0.241 | 0.344 | 0.000 | 0.347 | 0.003 | 0.347 | 0.000 |
| llada-8b-instruct-hf | plan_322 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.365 | 0.000 | 0.000 | 0.000 | 0.261 | 0.045 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_323 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.349 | 0.000 | 0.000 | 0.000 | 0.280 | 0.085 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_324 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.328 | 0.000 | 0.000 | 0.000 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 | 0.275 | 0.055 |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | random_32 | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.414 | 0.000 | 0.030 | 0.030 | 0.296 | 0.346 | 0.296 | 0.000 | 0.300 | 0.004 | 0.346 | 0.046 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.475 | 0.000 | 0.000 | 0.000 | 0.284 | 0.284 | 0.284 | 0.000 | 0.284 | 0.000 | 0.284 | 0.000 |
| llada-8b-instruct-hf | plan_327 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.393 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.297 | 0.000 | 0.297 | 0.000 | 0.297 | 0.000 |
| llada-8b-instruct-hf | plan_328 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.343 | 0.000 | 0.199 | 0.199 | 0.240 | 0.240 | 0.240 | 0.000 | 0.443 | 0.203 | 0.443 | 0.000 |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.453 | 0.000 | 0.163 | 0.163 | 0.304 | 0.241 | 0.304 | 0.000 | 0.422 | 0.118 | 0.422 | 0.000 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.401 | 0.000 | 0.000 | 0.000 | 0.402 | 0.177 | 0.402 | 0.000 | 0.402 | 0.000 | 0.402 | 0.000 |
| llada-8b-instruct-hf | plan_331 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.217 | 0.000 | 0.081 | 0.081 | 0.223 | 0.223 | 0.223 | 0.000 | 0.276 | 0.054 | 0.276 | 0.000 |
| llada-8b-instruct-hf | plan_332 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.356 | 0.000 | 0.065 | 0.065 | 0.240 | 0.240 | 0.240 | 0.000 | 0.282 | 0.042 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_333 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.376 | 0.000 | 0.203 | 0.203 | 0.280 | 0.217 | 0.280 | 0.000 | 0.420 | 0.140 | 0.420 | 0.000 |
| llada-8b-instruct-hf | plan_334 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.321 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.260 | 0.000 | 0.260 | 0.000 | 0.261 | 0.001 |
| llada-8b-instruct-hf | plan_335 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.450 | 0.000 | 0.000 | 0.000 | 0.309 | 0.309 | 0.309 | 0.000 | 0.309 | 0.000 | 0.309 | 0.000 |
| llada-8b-instruct-hf | plan_336 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.415 | 0.000 | 0.137 | 0.137 | 0.255 | 0.255 | 0.255 | 0.000 | 0.331 | 0.076 | 0.331 | 0.000 |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.376 | 0.000 | 0.069 | 0.069 | 0.242 | 0.065 | 0.242 | 0.000 | 0.285 | 0.043 | 0.285 | 0.000 |
| llada-8b-instruct-hf | plan_338 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.416 | 0.000 | 0.270 | 0.270 | 0.315 | 0.315 | 0.315 | 0.000 | 0.477 | 0.162 | 0.477 | 0.000 |
| llada-8b-instruct-hf | plan_339 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.454 | 0.000 | 0.062 | 0.062 | 0.268 | 0.184 | 0.268 | 0.000 | 0.301 | 0.034 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_340 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.348 | 0.000 | 0.000 | 0.000 | 0.261 | 0.065 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_341 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | history_prefix_25_repair | max_planning_state_score_base_pool |  | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.251 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.121 | 0.000 | 0.121 | 0.000 | 0.121 | 0.000 |
| llada-8b-instruct-hf | plan_342 | low_confidence_32 | random_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.435 | 0.000 | 0.210 | 0.210 | 0.268 | 0.172 | 0.268 | 0.000 | 0.448 | 0.180 | 0.448 | 0.000 |
| llada-8b-instruct-hf | plan_343 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | constraint_gap_span_phase_final_preserve_seeded_gated_repair | constraint_gap_span_phase_final_preserve_seeded_gated_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.358 | 0.000 | 0.106 | 0.106 | 0.180 | 0.180 | 0.180 | 0.000 | 0.260 | 0.080 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_344 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool |  | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | history | 31 | 0.358 | 0.000 | 0.069 | 0.069 | 0.200 | 0.200 | 0.200 | 0.000 | 0.246 | 0.046 | 0.246 | 0.000 |
