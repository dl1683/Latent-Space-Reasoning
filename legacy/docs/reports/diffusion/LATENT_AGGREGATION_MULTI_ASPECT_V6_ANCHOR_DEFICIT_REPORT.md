# Diffusion Schedule-Selection Benchmark Report

Full model generations: `386`
Counterfactual probe generations: `0`
Arm selections: `240`
Run ID: `diffusion-f945e7d4d96ad2c9`
Content hash: `f945e7d4d96ad2c98c3c449e58a54d23474a8890c4dc417b11f9945a40c91273`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `inherit`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `False`
History mutability: `monotonic 290/386, changes 59, remasks 878, rewrites 221, mask increases 576`
History repairs included: `False`
Repair pack: `constraint_gap`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `always`
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
Constraint-gap rescue trigger: `prompt_gap`
Constraint-gap rescue limit: `1`
Constraint-gap rescue min terms: `4`
Constraint-gap rescue source-quality band: `0.300-0.550`
Constraint-gap rescue source controls: `low_confidence_32,random_32,evolved_low_confidence_48,evolved_low_confidence_64`
Repair selector: `planning_quality`
Repair promotion margin: `0.000`
Trajectory task delta vs fixed: `0.003`
Trajectory task delta vs random: `0.048`
Trajectory wins/ties/losses vs fixed: `4/43/1`
Trajectory wins/ties/losses vs random: `22/23/3`
Oracle generation budget/task: `8.04`
Oracle task score: `0.293`
Oracle headroom vs trajectory: `0.026`
Oracle wins/ties/losses vs trajectory: `30/18/0`
Selector regret vs trajectory: `0.026 over 30/48 improvable`
Evolved task delta vs fixed: `0.010`
Evolved task delta vs random: `0.055`
Evolved task delta vs trajectory: `0.007`
Evolved wins/ties/losses vs fixed: `13/31/4`
Evolved wins/ties/losses vs random: `30/14/4`
Evolved wins/ties/losses vs trajectory: `10/33/5`
Oracle headroom vs evolved: `0.019`
Oracle wins/ties/losses vs evolved: `28/20/0`
Selector regret vs evolved: `0.019 over 28/48 improvable`
Repair arm coverage: `48/48` overall
Repair eligible coverage: `48/48`
Repair task delta vs fixed: `0.015`
Repair task delta vs random: `0.060`
Repair task delta vs trajectory: `0.012`
Repair task delta vs evolved: `0.005`
Repair generation budget delta vs evolved: `1.04`
Repair task delta per extra generation vs evolved: `0.004`
Repair wins/ties/losses vs evolved: `6/42/0`
Oracle headroom vs repair: `0.014`
Oracle wins/ties/losses vs repair: `23/25/0`
Selector regret vs repair: `0.014 over 23/48 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `48/48` overall, `48/48` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.263757 | 0.000000 | 0.044804 | - | - |
| random perturbation | repair-covered tasks | 0.218954 | -0.044804 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.278735 | 0.014978 | 0.059781 | 17/27/4 | 31/14/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.264 | 0.684 | 0.369 |
| random | 48 | 1.00 | 0.219 | 0.571 | 0.307 |
| trajectory_selected | 48 | 2.00 | 0.267 | 0.686 | 0.372 |
| evolved | 48 | 7.00 | 0.274 | 0.685 | 0.377 |
| repair_selected | 48 | 8.04 | 0.279 | 0.684 | 0.380 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.264 | 0.684 | 0.369 |
| planning | random | 48 | 1.00 | 0.219 | 0.571 | 0.307 |
| planning | trajectory_selected | 48 | 2.00 | 0.267 | 0.686 | 0.372 |
| planning | evolved | 48 | 7.00 | 0.274 | 0.685 | 0.377 |
| planning | repair_selected | 48 | 8.04 | 0.279 | 0.684 | 0.380 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_297 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.220 | 0.180 | 313 | True | 6 | 0.571 | False | True | 14.000 | 0.219 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | True | trigger_always | False |  | 0.323 | 0.223 | 298 | True | 2 | 0.857 | False | True | 7.000 | 0.219 | 0.429 | 0.429 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | True | trigger_always | False |  | 0.282 | 0.223 | 354 | True | 4 | 0.714 | False | True | 7.000 | 0.219 | 0.357 | 0.357 |
| llada-8b-instruct-hf | plan_300 | low_confidence_32 | True | trigger_always | False |  | 0.263 | 0.223 | 375 | True | 7 | 0.500 | False | True | 7.000 | 0.219 | 0.357 | 0.357 |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | True | trigger_always | False |  | 0.290 | 0.230 | 276 | True | 2 | 0.818 | False | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_302 | evolved_revision_random_32 | True | trigger_always | False |  | 0.277 | 0.217 | 347 | True | 3 | 0.778 | False | True | 11.000 | 0.224 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_303 | low_confidence_32 | True | trigger_always | False |  | 0.283 | 0.223 | 334 | True | 1 | 0.900 | False | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_304 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.389 | 0.289 | 352 | True | 6 | 0.500 | False | True | 10.000 | 0.208 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_305 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.340 | 0.260 | 345 | True | 5 | 0.615 | False | True | 10.000 | 0.208 | 0.385 | 0.385 |
| llada-8b-instruct-hf | plan_306 | random_32 | True | trigger_always | False |  | 0.422 | 0.324 | 267 | True | 5 | 0.615 | False | True | 7.000 | 0.219 | 0.154 | 0.154 |
| llada-8b-instruct-hf | plan_307 | evolved_random_48 | True | trigger_always | False |  | 0.200 | 0.180 | 336 | True | 3 | 0.769 | False | True | 10.000 | 0.208 | 0.462 | 0.462 |
| llada-8b-instruct-hf | plan_308 | low_confidence_32 | True | trigger_always | False |  | 0.241 | 0.201 | 320 | True | 6 | 0.538 | False | True | 7.000 | 0.219 | 0.462 | 0.462 |
| llada-8b-instruct-hf | plan_309 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.336 | 0.239 | 337 | True | 1 | 0.900 | False | True | 14.000 | 0.219 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_310 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.260 | 0.180 | 364 | True | 2 | 0.818 | False | True | 14.000 | 0.219 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_311 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.334 | 0.294 | 310 | True | 2 | 0.818 | False | True | 14.000 | 0.219 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_312 | low_confidence_32 | True | trigger_always | False |  | 0.241 | 0.201 | 343 | True | 1 | 0.900 | False | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_313 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.200 | 0.180 | 366 | True | 2 | 0.818 | False | True | 10.000 | 0.208 | 0.636 | 0.636 |
| llada-8b-instruct-hf | plan_314 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.261 | 0.201 | 278 | True | 1 | 0.917 | False | True | 14.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | True | trigger_always | False |  | 0.240 | 0.180 | 327 | True | 0 | 1.000 | False | True | 7.000 | 0.219 | 0.667 | 0.667 |
| llada-8b-instruct-hf | plan_316 | evolved_random_48 | True | trigger_always | False |  | 0.263 | 0.223 | 389 | True | 3 | 0.769 | False | True | 10.000 | 0.208 | 0.462 | 0.462 |
| llada-8b-instruct-hf | plan_317 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.240 | 0.180 | 326 | True | 1 | 0.917 | False | True | 14.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_318 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.241 | 0.201 | 270 | True | 11 | 0.286 | False | True | 14.000 | 0.219 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_319 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.200 | 0.180 | 327 | True | 2 | 0.818 | False | True | 10.000 | 0.208 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_320 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.281 | 0.201 | 399 | True | 1 | 0.900 | False | True | 14.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | True | trigger_always | False |  | 0.344 | 0.244 | 357 | True | 0 | 1.000 | False | True | 7.000 | 0.219 | 0.667 | 0.667 |
| llada-8b-instruct-hf | plan_322 | low_confidence_32 | True | trigger_always | False |  | 0.261 | 0.201 | 387 | True | 3 | 0.750 | False | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_323 | low_confidence_32 | True | trigger_always | False |  | 0.280 | 0.180 | 364 | True | 5 | 0.583 | False | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_324 | low_confidence_32 | True | trigger_always | False |  | 0.220 | 0.180 | 305 | True | 6 | 0.500 | False | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | True | trigger_always | False |  | 0.296 | 0.256 | 364 | True | 3 | 0.727 | False | True | 7.000 | 0.219 | 0.455 | 0.455 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | True | trigger_always | False |  | 0.284 | 0.244 | 305 | True | 1 | 0.917 | False | True | 7.000 | 0.219 | 0.583 | 0.583 |
| llada-8b-instruct-hf | plan_327 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.281 | 0.201 | 346 | True | 0 | 1.000 | False | True | 14.000 | 0.219 | 0.727 | 0.727 |
| llada-8b-instruct-hf | plan_328 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.281 | 0.201 | 342 | True | 4 | 0.667 | False | True | 14.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | True | trigger_always | False |  | 0.304 | 0.244 | 371 | True | 2 | 0.818 | False | True | 7.000 | 0.219 | 0.545 | 0.545 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | True | trigger_always | False |  | 0.402 | 0.362 | 359 | True | 7 | 0.417 | False | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_331 | evolved_revision_random_32 | True | trigger_always | False |  | 0.281 | 0.201 | 231 | True | 5 | 0.375 | False | True | 11.000 | 0.224 | 0.125 | 0.125 |
| llada-8b-instruct-hf | plan_332 | evolved_low_confidence_64 | True | trigger_always | False |  | 0.295 | 0.235 | 292 | True | 3 | 0.700 | False | True | 14.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_333 | low_confidence_32 | True | trigger_always | False |  | 0.280 | 0.180 | 320 | True | 3 | 0.667 | False | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_334 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.260 | 0.180 | 266 | True | 4 | 0.667 | False | True | 10.000 | 0.208 | 0.167 | 0.167 |
| llada-8b-instruct-hf | plan_335 | low_confidence_32 | True | trigger_always | False |  | 0.309 | 0.269 | 340 | True | 2 | 0.800 | False | True | 7.000 | 0.219 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_336 | low_confidence_32 | True | trigger_always | False |  | 0.255 | 0.235 | 363 | True | 2 | 0.750 | False | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | True | trigger_always | False |  | 0.242 | 0.223 | 390 | True | 3 | 0.625 | False | True | 7.000 | 0.219 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_338 | low_confidence_32 | True | trigger_always | False |  | 0.315 | 0.217 | 313 | True | 2 | 0.750 | False | True | 7.000 | 0.219 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_339 | low_confidence_32 | True | trigger_always | False |  | 0.268 | 0.247 | 210 | True | 1 | 0.889 | False | True | 7.000 | 0.219 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_340 | low_confidence_32 | True | trigger_always | False |  | 0.261 | 0.201 | 354 | True | 7 | 0.571 | False | True | 7.000 | 0.219 | 0.214 | 0.214 |
| llada-8b-instruct-hf | plan_341 | evolved_random_48 | True | trigger_always | False |  | 0.121 | 0.121 | 50 | True | 6 | 0.400 | False | True | 10.000 | 0.208 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_342 | low_confidence_32 | True | trigger_always | False |  | 0.268 | 0.247 | 381 | True | 3 | 0.786 | False | True | 7.000 | 0.219 | 0.357 | 0.357 |
| llada-8b-instruct-hf | plan_343 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.223 | 0.223 | 328 | True | 3 | 0.700 | False | True | 10.000 | 0.208 | 0.600 | 0.600 |
| llada-8b-instruct-hf | plan_344 | low_confidence_32 | True | trigger_always | False |  | 0.200 | 0.180 | 364 | True | 5 | 0.636 | False | True | 7.000 | 0.219 | 0.455 | 0.455 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_revision_repair | 2 | 0 | low_confidence_32,random_32 | final | 64.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/2/0 | 0.412 | 0.698 | 0.483 |
| state_adaptive_history_repair | 48 | 6 | evolved_low_confidence_48,evolved_low_confidence_64,evolved_random_48,evolved_revision_random_32,low_confidence_32,random_32 | history | 45.8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.005 | -0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9/24/15 | 0.269 | 0.671 | 0.369 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_297 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.326 | 0.346 | 0.180 | 0.000 | 0.220 | 0.157 | 0.220 | 0.220 | 0.220 | 0.000 | 0.241 | 0.021 |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.428 | 0.428 | 0.223 | 0.000 | 0.323 | 0.323 | 0.323 | 0.323 | 0.323 | 0.000 | 0.381 | 0.059 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.392 | 0.392 | 0.223 | 0.000 | 0.282 | 0.282 | 0.282 | 0.282 | 0.282 | 0.000 | 0.302 | 0.020 |
| llada-8b-instruct-hf | plan_300 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.346 | 0.346 | 0.223 | 0.000 | 0.263 | 0.241 | 0.263 | 0.263 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.423 | 0.423 | 0.230 | 0.000 | 0.290 | 0.137 | 0.290 | 0.290 | 0.290 | 0.000 | 0.290 | 0.000 |
| llada-8b-instruct-hf | plan_302 | low_confidence_32 | random_32 | random_32 | evolved_revision_random_32 | evolved_revision_random_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.346 | 0.409 | 0.217 | 0.000 | 0.240 | 0.240 | 0.240 | 0.277 | 0.277 | 0.000 | 0.297 | 0.020 |
| llada-8b-instruct-hf | plan_303 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.453 | 0.453 | 0.223 | 0.000 | 0.283 | 0.198 | 0.283 | 0.283 | 0.283 | 0.000 | 0.316 | 0.034 |
| llada-8b-instruct-hf | plan_304 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.325 | 0.375 | 0.289 | 0.000 | 0.299 | 0.299 | 0.299 | 0.389 | 0.389 | 0.000 | 0.389 | 0.000 |
| llada-8b-instruct-hf | plan_305 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.379 | 0.410 | 0.260 | 0.000 | 0.360 | 0.106 | 0.360 | 0.340 | 0.340 | 0.000 | 0.360 | 0.020 |
| llada-8b-instruct-hf | plan_306 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.433 | 0.433 | 0.324 | 0.000 | 0.402 | 0.402 | 0.422 | 0.422 | 0.422 | 0.000 | 0.422 | 0.000 |
| llada-8b-instruct-hf | plan_307 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.378 | 0.396 | 0.180 | 0.000 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.000 | 0.220 | 0.020 |
| llada-8b-instruct-hf | plan_308 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.351 | 0.351 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_309 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.450 | 0.466 | 0.239 | 0.000 | 0.356 | 0.336 | 0.356 | 0.336 | 0.336 | 0.000 | 0.356 | 0.020 |
| llada-8b-instruct-hf | plan_310 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool | evolved_low_confidence_64 | history | 51 | 0.385 | 0.410 | 0.244 | 0.064 | 0.260 | 0.045 | 0.260 | 0.260 | 0.324 | 0.064 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_311 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.429 | 0.470 | 0.294 | 0.000 | 0.284 | 0.284 | 0.284 | 0.334 | 0.334 | 0.000 | 0.334 | 0.000 |
| llada-8b-instruct-hf | plan_312 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.444 | 0.444 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.241 | 0.241 | 0.000 | 0.261 | 0.020 |
| llada-8b-instruct-hf | plan_313 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.375 | 0.405 | 0.180 | 0.000 | 0.221 | 0.221 | 0.221 | 0.200 | 0.200 | 0.000 | 0.221 | 0.021 |
| llada-8b-instruct-hf | plan_314 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.380 | 0.432 | 0.201 | 0.000 | 0.240 | 0.218 | 0.240 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool | low_confidence_32 | history | 26 | 0.458 | 0.458 | 0.201 | 0.021 | 0.240 | 0.261 | 0.240 | 0.240 | 0.261 | 0.021 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_316 | low_confidence_32 | random_32 | random_32 | evolved_random_48 | evolved_random_48 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.387 | 0.420 | 0.223 | 0.000 | 0.263 | 0.263 | 0.263 | 0.263 | 0.263 | 0.000 | 0.284 | 0.021 |
| llada-8b-instruct-hf | plan_317 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_64 | state_adaptive_history_repair | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool | evolved_low_confidence_64 | history | 51 | 0.371 | 0.444 | 0.201 | 0.021 | 0.220 | 0.220 | 0.277 | 0.240 | 0.261 | 0.021 | 0.336 | 0.075 |
| llada-8b-instruct-hf | plan_318 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.255 | 0.288 | 0.201 | 0.000 | 0.200 | 0.200 | 0.200 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_319 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.403 | 0.419 | 0.180 | 0.000 | 0.200 | 0.261 | 0.200 | 0.200 | 0.200 | 0.000 | 0.261 | 0.061 |
| llada-8b-instruct-hf | plan_320 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.384 | 0.452 | 0.201 | 0.000 | 0.294 | 0.260 | 0.260 | 0.281 | 0.281 | 0.000 | 0.315 | 0.034 |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.490 | 0.490 | 0.244 | 0.000 | 0.344 | 0.241 | 0.344 | 0.344 | 0.344 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_322 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.364 | 0.364 | 0.201 | 0.000 | 0.261 | 0.045 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_323 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.353 | 0.353 | 0.180 | 0.000 | 0.280 | 0.085 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_324 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.326 | 0.326 | 0.180 | 0.000 | 0.220 | 0.220 | 0.220 | 0.220 | 0.220 | 0.000 | 0.275 | 0.055 |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.411 | 0.411 | 0.256 | 0.000 | 0.296 | 0.346 | 0.296 | 0.296 | 0.296 | 0.000 | 0.346 | 0.050 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.481 | 0.481 | 0.244 | 0.000 | 0.284 | 0.284 | 0.284 | 0.284 | 0.284 | 0.000 | 0.284 | 0.000 |
| llada-8b-instruct-hf | plan_327 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.390 | 0.480 | 0.201 | 0.000 | 0.260 | 0.260 | 0.297 | 0.281 | 0.281 | 0.000 | 0.297 | 0.016 |
| llada-8b-instruct-hf | plan_328 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.339 | 0.387 | 0.201 | 0.000 | 0.240 | 0.240 | 0.240 | 0.281 | 0.281 | 0.000 | 0.299 | 0.017 |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool | low_confidence_32 | history | 26 | 0.450 | 0.450 | 0.286 | 0.042 | 0.304 | 0.241 | 0.304 | 0.304 | 0.346 | 0.042 | 0.346 | 0.000 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.400 | 0.400 | 0.362 | 0.000 | 0.402 | 0.177 | 0.402 | 0.402 | 0.402 | 0.000 | 0.418 | 0.016 |
| llada-8b-instruct-hf | plan_331 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool | evolved_revision_random_32 | history | 30 | 0.213 | 0.301 | 0.223 | 0.021 | 0.223 | 0.223 | 0.223 | 0.281 | 0.303 | 0.021 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_332 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.358 | 0.408 | 0.235 | 0.000 | 0.240 | 0.240 | 0.240 | 0.295 | 0.295 | 0.000 | 0.295 | 0.000 |
| llada-8b-instruct-hf | plan_333 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.381 | 0.381 | 0.180 | 0.000 | 0.280 | 0.217 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_334 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.313 | 0.360 | 0.180 | 0.000 | 0.260 | 0.260 | 0.260 | 0.260 | 0.260 | 0.000 | 0.261 | 0.001 |
| llada-8b-instruct-hf | plan_335 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.442 | 0.442 | 0.269 | 0.000 | 0.309 | 0.309 | 0.309 | 0.309 | 0.309 | 0.000 | 0.309 | 0.000 |
| llada-8b-instruct-hf | plan_336 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.413 | 0.413 | 0.235 | 0.000 | 0.255 | 0.255 | 0.255 | 0.255 | 0.255 | 0.000 | 0.275 | 0.020 |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.383 | 0.383 | 0.223 | 0.000 | 0.242 | 0.065 | 0.242 | 0.242 | 0.242 | 0.000 | 0.242 | 0.000 |
| llada-8b-instruct-hf | plan_338 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.417 | 0.417 | 0.217 | 0.000 | 0.315 | 0.315 | 0.315 | 0.315 | 0.315 | 0.000 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_339 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.446 | 0.446 | 0.247 | 0.000 | 0.268 | 0.184 | 0.268 | 0.268 | 0.268 | 0.000 | 0.275 | 0.008 |
| llada-8b-instruct-hf | plan_340 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.350 | 0.350 | 0.201 | 0.000 | 0.261 | 0.065 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_341 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.249 | 0.270 | 0.121 | 0.000 | 0.045 | 0.045 | 0.121 | 0.121 | 0.121 | 0.000 | 0.121 | 0.000 |
| llada-8b-instruct-hf | plan_342 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool |  |  |  | 0.432 | 0.432 | 0.247 | 0.000 | 0.268 | 0.172 | 0.268 | 0.268 | 0.268 | 0.000 | 0.318 | 0.050 |
| llada-8b-instruct-hf | plan_343 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | max_planning_quality_score_repair_pool |  |  |  | 0.356 | 0.397 | 0.223 | 0.000 | 0.180 | 0.180 | 0.180 | 0.223 | 0.223 | 0.000 | 0.223 | 0.000 |
| llada-8b-instruct-hf | plan_344 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_score_repair_pool | low_confidence_32 | history | 26 | 0.357 | 0.357 | 0.226 | 0.046 | 0.200 | 0.200 | 0.200 | 0.200 | 0.246 | 0.046 | 0.246 | 0.000 |
