# Diffusion Schedule-Selection Benchmark Report

Full model generations: `336`
Counterfactual probe generations: `0`
Arm selections: `192`
Run ID: `diffusion-fa1f36fb98393ef7`
Content hash: `fa1f36fb98393ef7b68539579a7f004f8f3427dadf0969bc7054c58025e8264e`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `inherit`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `8`
Exact verifier revision: `False`
History mutability: `monotonic 240/336, changes 0, remasks 1283, rewrites 234, mask increases 192`
History repairs included: `False`
Repair pack: `prefix`
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
Constraint-gap rescue trigger: `off`
Constraint-gap rescue limit: `1`
Constraint-gap rescue min terms: `6`
Constraint-gap rescue source-quality band: `0.400-0.500`
Constraint-gap rescue source controls: ``
Repair selector: `planning_quality`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.003`
Trajectory task delta vs random: `0.048`
Trajectory wins/ties/losses vs fixed: `4/43/1`
Trajectory wins/ties/losses vs random: `22/23/3`
Oracle generation budget/task: `7.00`
Oracle task score: `0.291`
Oracle headroom vs trajectory: `0.024`
Oracle wins/ties/losses vs trajectory: `28/20/0`
Selector regret vs trajectory: `0.024 over 28/48 improvable`
Evolved task delta vs fixed: `0.010`
Evolved task delta vs random: `0.055`
Evolved task delta vs trajectory: `0.007`
Evolved wins/ties/losses vs fixed: `14/28/6`
Evolved wins/ties/losses vs random: `31/12/5`
Evolved wins/ties/losses vs trajectory: `11/30/7`
Oracle headroom vs evolved: `0.017`
Oracle wins/ties/losses vs evolved: `27/21/0`
Selector regret vs evolved: `0.017 over 27/48 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.264 | 0.684 | 0.369 |
| random | 48 | 1.00 | 0.219 | 0.571 | 0.307 |
| trajectory_selected | 48 | 2.00 | 0.267 | 0.686 | 0.372 |
| evolved | 48 | 7.00 | 0.274 | 0.680 | 0.375 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.264 | 0.684 | 0.369 |
| planning | random | 48 | 1.00 | 0.219 | 0.571 | 0.307 |
| planning | trajectory_selected | 48 | 2.00 | 0.267 | 0.686 | 0.372 |
| planning | evolved | 48 | 7.00 | 0.274 | 0.680 | 0.375 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_297 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.326 | 0.346 | 0.020 | 0.220 | 0.157 | 0.220 | 0.220 | 0.000 | 0.000 | 0.000 | 0.241 | 0.021 |
| llada-8b-instruct-hf | plan_298 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.428 | 0.428 | 0.000 | 0.323 | 0.323 | 0.323 | 0.323 | 0.000 | 0.000 | 0.000 | 0.381 | 0.059 |
| llada-8b-instruct-hf | plan_299 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.392 | 0.392 | 0.000 | 0.282 | 0.282 | 0.282 | 0.282 | 0.000 | 0.000 | 0.000 | 0.302 | 0.020 |
| llada-8b-instruct-hf | plan_300 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.346 | 0.381 | 0.034 | 0.263 | 0.241 | 0.263 | 0.261 | 0.000 | -0.001 | -0.001 | 0.263 | 0.001 |
| llada-8b-instruct-hf | plan_301 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.423 | 0.423 | 0.000 | 0.290 | 0.137 | 0.290 | 0.290 | 0.000 | 0.000 | 0.000 | 0.290 | 0.000 |
| llada-8b-instruct-hf | plan_302 | low_confidence_32 | random_32 | random_32 | evolved_revision_random_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.346 | 0.397 | 0.051 | 0.240 | 0.240 | 0.240 | 0.277 | 0.000 | 0.037 | 0.037 | 0.297 | 0.020 |
| llada-8b-instruct-hf | plan_303 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.453 | 0.453 | 0.000 | 0.283 | 0.198 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.316 | 0.034 |
| llada-8b-instruct-hf | plan_304 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.325 | 0.375 | 0.050 | 0.299 | 0.299 | 0.299 | 0.389 | 0.000 | 0.090 | 0.090 | 0.389 | 0.000 |
| llada-8b-instruct-hf | plan_305 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.379 | 0.410 | 0.031 | 0.360 | 0.106 | 0.360 | 0.340 | 0.000 | -0.020 | -0.020 | 0.360 | 0.020 |
| llada-8b-instruct-hf | plan_306 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.433 | 0.433 | 0.000 | 0.402 | 0.402 | 0.422 | 0.422 | 0.020 | 0.020 | 0.000 | 0.422 | 0.000 |
| llada-8b-instruct-hf | plan_307 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.378 | 0.396 | 0.018 | 0.200 | 0.200 | 0.200 | 0.200 | 0.000 | 0.000 | 0.000 | 0.220 | 0.020 |
| llada-8b-instruct-hf | plan_308 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.351 | 0.351 | 0.000 | 0.241 | 0.241 | 0.241 | 0.241 | 0.000 | 0.000 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_309 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.450 | 0.466 | 0.016 | 0.356 | 0.336 | 0.356 | 0.336 | 0.000 | -0.020 | -0.020 | 0.356 | 0.020 |
| llada-8b-instruct-hf | plan_310 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.385 | 0.410 | 0.025 | 0.260 | 0.045 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.302 | 0.042 |
| llada-8b-instruct-hf | plan_311 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.429 | 0.470 | 0.042 | 0.284 | 0.284 | 0.284 | 0.334 | 0.000 | 0.050 | 0.050 | 0.334 | 0.000 |
| llada-8b-instruct-hf | plan_312 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.444 | 0.444 | 0.000 | 0.241 | 0.241 | 0.241 | 0.241 | 0.000 | 0.000 | 0.000 | 0.261 | 0.020 |
| llada-8b-instruct-hf | plan_313 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.375 | 0.405 | 0.029 | 0.221 | 0.221 | 0.221 | 0.200 | 0.000 | -0.021 | -0.021 | 0.221 | 0.021 |
| llada-8b-instruct-hf | plan_314 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.380 | 0.432 | 0.052 | 0.240 | 0.218 | 0.240 | 0.261 | 0.000 | 0.021 | 0.021 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_315 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.458 | 0.458 | 0.000 | 0.240 | 0.261 | 0.240 | 0.240 | 0.000 | 0.000 | 0.000 | 0.261 | 0.021 |
| llada-8b-instruct-hf | plan_316 | low_confidence_32 | random_32 | random_32 | evolved_random_48 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.387 | 0.420 | 0.033 | 0.263 | 0.263 | 0.263 | 0.263 | 0.000 | 0.000 | 0.000 | 0.284 | 0.021 |
| llada-8b-instruct-hf | plan_317 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.371 | 0.444 | 0.073 | 0.220 | 0.220 | 0.277 | 0.240 | 0.057 | 0.020 | -0.037 | 0.336 | 0.096 |
| llada-8b-instruct-hf | plan_318 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.255 | 0.288 | 0.034 | 0.200 | 0.200 | 0.200 | 0.241 | 0.000 | 0.041 | 0.041 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_319 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.403 | 0.419 | 0.016 | 0.200 | 0.261 | 0.200 | 0.200 | 0.000 | 0.000 | 0.000 | 0.261 | 0.061 |
| llada-8b-instruct-hf | plan_320 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_64 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.384 | 0.452 | 0.068 | 0.294 | 0.260 | 0.260 | 0.281 | -0.034 | -0.012 | 0.021 | 0.315 | 0.034 |
| llada-8b-instruct-hf | plan_321 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.490 | 0.490 | 0.000 | 0.344 | 0.241 | 0.344 | 0.344 | 0.000 | 0.000 | 0.000 | 0.344 | 0.000 |
| llada-8b-instruct-hf | plan_322 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.364 | 0.364 | 0.000 | 0.261 | 0.045 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_323 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.353 | 0.353 | 0.000 | 0.280 | 0.085 | 0.280 | 0.280 | 0.000 | 0.000 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_324 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.326 | 0.342 | 0.016 | 0.220 | 0.220 | 0.220 | 0.232 | 0.000 | 0.012 | 0.012 | 0.275 | 0.043 |
| llada-8b-instruct-hf | plan_325 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.411 | 0.449 | 0.038 | 0.296 | 0.346 | 0.296 | 0.296 | 0.000 | 0.000 | 0.000 | 0.346 | 0.050 |
| llada-8b-instruct-hf | plan_326 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.481 | 0.481 | 0.000 | 0.284 | 0.284 | 0.284 | 0.284 | 0.000 | 0.000 | 0.000 | 0.284 | 0.000 |
| llada-8b-instruct-hf | plan_327 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_64 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.390 | 0.480 | 0.090 | 0.260 | 0.260 | 0.297 | 0.281 | 0.037 | 0.021 | -0.016 | 0.297 | 0.016 |
| llada-8b-instruct-hf | plan_328 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.339 | 0.387 | 0.048 | 0.240 | 0.240 | 0.240 | 0.281 | 0.000 | 0.041 | 0.041 | 0.299 | 0.017 |
| llada-8b-instruct-hf | plan_329 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.450 | 0.450 | 0.000 | 0.304 | 0.241 | 0.304 | 0.304 | 0.000 | 0.000 | 0.000 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_330 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.400 | 0.400 | 0.000 | 0.402 | 0.177 | 0.402 | 0.402 | 0.000 | 0.000 | 0.000 | 0.418 | 0.016 |
| llada-8b-instruct-hf | plan_331 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.213 | 0.295 | 0.082 | 0.223 | 0.223 | 0.223 | 0.281 | 0.000 | 0.059 | 0.059 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_332 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.358 | 0.408 | 0.050 | 0.240 | 0.240 | 0.240 | 0.295 | 0.000 | 0.055 | 0.055 | 0.295 | 0.000 |
| llada-8b-instruct-hf | plan_333 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.381 | 0.381 | 0.000 | 0.280 | 0.217 | 0.280 | 0.280 | 0.000 | 0.000 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_334 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.313 | 0.360 | 0.047 | 0.260 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.261 | 0.001 |
| llada-8b-instruct-hf | plan_335 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.442 | 0.442 | 0.000 | 0.309 | 0.309 | 0.309 | 0.309 | 0.000 | 0.000 | 0.000 | 0.309 | 0.000 |
| llada-8b-instruct-hf | plan_336 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.413 | 0.413 | 0.000 | 0.255 | 0.255 | 0.255 | 0.255 | 0.000 | 0.000 | 0.000 | 0.255 | 0.000 |
| llada-8b-instruct-hf | plan_337 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.383 | 0.383 | 0.000 | 0.242 | 0.065 | 0.242 | 0.242 | 0.000 | 0.000 | 0.000 | 0.242 | 0.000 |
| llada-8b-instruct-hf | plan_338 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.417 | 0.433 | 0.016 | 0.315 | 0.315 | 0.315 | 0.295 | 0.000 | -0.020 | -0.020 | 0.315 | 0.020 |
| llada-8b-instruct-hf | plan_339 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.446 | 0.446 | 0.000 | 0.268 | 0.184 | 0.268 | 0.268 | 0.000 | 0.000 | 0.000 | 0.275 | 0.008 |
| llada-8b-instruct-hf | plan_340 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.350 | 0.350 | 0.000 | 0.261 | 0.065 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_341 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.249 | 0.270 | 0.021 | 0.045 | 0.045 | 0.121 | 0.121 | 0.076 | 0.076 | 0.000 | 0.121 | 0.000 |
| llada-8b-instruct-hf | plan_342 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.432 | 0.432 | 0.000 | 0.268 | 0.172 | 0.268 | 0.268 | 0.000 | 0.000 | 0.000 | 0.318 | 0.050 |
| llada-8b-instruct-hf | plan_343 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.356 | 0.397 | 0.041 | 0.180 | 0.180 | 0.180 | 0.223 | 0.000 | 0.043 | 0.043 | 0.223 | 0.000 |
| llada-8b-instruct-hf | plan_344 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.357 | 0.357 | 0.000 | 0.200 | 0.200 | 0.200 | 0.200 | 0.000 | 0.000 | 0.000 | 0.243 | 0.043 |
