# Diffusion Schedule-Selection Benchmark Report

Full model generations: `336`
Counterfactual probe generations: `0`
Arm selections: `192`
Run ID: `diffusion-a596aee333e91d15`
Content hash: `a596aee333e91d15cce18f915781fe0c251244a379e27872d091a5ad756e9670`
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
History mutability: `monotonic 240/336, changes 0, remasks 1292, rewrites 215, mask increases 192`
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
Trajectory task delta vs fixed: `0.006`
Trajectory task delta vs random: `0.033`
Trajectory wins/ties/losses vs fixed: `6/40/2`
Trajectory wins/ties/losses vs random: `17/28/3`
Oracle generation budget/task: `7.00`
Oracle task score: `0.322`
Oracle headroom vs trajectory: `0.031`
Oracle wins/ties/losses vs trajectory: `31/17/0`
Selector regret vs trajectory: `0.031 over 31/48 improvable`
Evolved task delta vs fixed: `0.026`
Evolved task delta vs random: `0.053`
Evolved task delta vs trajectory: `0.019`
Evolved wins/ties/losses vs fixed: `24/21/3`
Evolved wins/ties/losses vs random: `30/15/3`
Evolved wins/ties/losses vs trajectory: `19/25/4`
Oracle headroom vs evolved: `0.012`
Oracle wins/ties/losses vs evolved: `17/31/0`
Selector regret vs evolved: `0.012 over 17/48 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.284 | 0.666 | 0.380 |
| random | 48 | 1.00 | 0.257 | 0.586 | 0.339 |
| trajectory_selected | 48 | 2.00 | 0.290 | 0.678 | 0.387 |
| evolved | 48 | 7.00 | 0.310 | 0.670 | 0.400 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.284 | 0.666 | 0.380 |
| planning | random | 48 | 1.00 | 0.257 | 0.586 | 0.339 |
| planning | trajectory_selected | 48 | 2.00 | 0.290 | 0.678 | 0.387 |
| planning | evolved | 48 | 7.00 | 0.310 | 0.670 | 0.400 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_249 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.339 | 0.394 | 0.055 | 0.241 | 0.045 | 0.241 | 0.283 | 0.000 | 0.041 | 0.041 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_250 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_64 | random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.297 | 0.403 | 0.107 | 0.281 | 0.301 | 0.301 | 0.292 | 0.020 | 0.011 | -0.009 | 0.301 | 0.009 |
| llada-8b-instruct-hf | plan_251 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.442 | 0.442 | 0.000 | 0.341 | 0.045 | 0.341 | 0.341 | 0.000 | 0.000 | 0.000 | 0.341 | 0.000 |
| llada-8b-instruct-hf | plan_252 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.471 | 0.471 | 0.000 | 0.273 | 0.391 | 0.391 | 0.391 | 0.119 | 0.119 | 0.000 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_253 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.388 | 0.388 | 0.000 | 0.301 | 0.157 | 0.301 | 0.301 | 0.000 | 0.000 | 0.000 | 0.301 | 0.000 |
| llada-8b-instruct-hf | plan_254 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.430 | 0.454 | 0.024 | 0.240 | 0.240 | 0.240 | 0.316 | 0.000 | 0.076 | 0.076 | 0.316 | 0.000 |
| llada-8b-instruct-hf | plan_255 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.452 | 0.467 | 0.015 | 0.378 | 0.378 | 0.378 | 0.378 | 0.000 | 0.000 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_256 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.308 | 0.343 | 0.035 | 0.200 | 0.200 | 0.200 | 0.220 | 0.000 | 0.020 | 0.020 | 0.242 | 0.022 |
| llada-8b-instruct-hf | plan_257 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.416 | 0.416 | 0.000 | 0.324 | 0.324 | 0.324 | 0.324 | 0.000 | 0.000 | 0.000 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_258 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.272 | 0.272 | 0.000 | 0.045 | 0.045 | 0.065 | 0.065 | 0.020 | 0.020 | 0.000 | 0.065 | 0.000 |
| llada-8b-instruct-hf | plan_259 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.420 | 0.436 | 0.017 | 0.303 | 0.303 | 0.303 | 0.304 | 0.000 | 0.001 | 0.001 | 0.304 | 0.000 |
| llada-8b-instruct-hf | plan_260 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.486 | 0.486 | 0.000 | 0.381 | 0.381 | 0.381 | 0.381 | 0.000 | 0.000 | 0.000 | 0.381 | 0.000 |
| llada-8b-instruct-hf | plan_261 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.380 | 0.406 | 0.026 | 0.240 | 0.240 | 0.240 | 0.261 | 0.000 | 0.021 | 0.021 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_262 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.400 | 0.400 | 0.000 | 0.418 | 0.418 | 0.418 | 0.418 | 0.000 | 0.000 | 0.000 | 0.418 | 0.000 |
| llada-8b-instruct-hf | plan_263 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.274 | 0.274 | 0.000 | 0.045 | 0.045 | 0.117 | 0.117 | 0.072 | 0.072 | 0.000 | 0.117 | 0.000 |
| llada-8b-instruct-hf | plan_264 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.340 | 0.397 | 0.058 | 0.241 | 0.241 | 0.241 | 0.277 | 0.000 | 0.036 | 0.036 | 0.283 | 0.005 |
| llada-8b-instruct-hf | plan_265 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.388 | 0.439 | 0.051 | 0.280 | 0.217 | 0.280 | 0.323 | 0.000 | 0.042 | 0.042 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_266 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.384 | 0.417 | 0.033 | 0.330 | 0.217 | 0.330 | 0.330 | 0.000 | 0.000 | 0.000 | 0.330 | 0.000 |
| llada-8b-instruct-hf | plan_267 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.436 | 0.436 | 0.000 | 0.301 | 0.280 | 0.301 | 0.301 | 0.000 | 0.000 | 0.000 | 0.373 | 0.071 |
| llada-8b-instruct-hf | plan_268 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.475 | 0.475 | 0.000 | 0.345 | 0.345 | 0.345 | 0.345 | 0.000 | 0.000 | 0.000 | 0.345 | 0.000 |
| llada-8b-instruct-hf | plan_269 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.406 | 0.406 | 0.000 | 0.220 | 0.045 | 0.220 | 0.220 | 0.000 | 0.000 | 0.000 | 0.223 | 0.003 |
| llada-8b-instruct-hf | plan_270 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.381 | 0.460 | 0.079 | 0.319 | 0.319 | 0.319 | 0.390 | 0.000 | 0.071 | 0.071 | 0.390 | 0.000 |
| llada-8b-instruct-hf | plan_271 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.389 | 0.408 | 0.019 | 0.241 | 0.241 | 0.241 | 0.263 | 0.000 | 0.021 | 0.021 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_272 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.489 | 0.550 | 0.061 | 0.404 | 0.333 | 0.404 | 0.500 | 0.000 | 0.096 | 0.096 | 0.500 | 0.000 |
| llada-8b-instruct-hf | plan_273 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.411 | 0.411 | 0.000 | 0.378 | 0.378 | 0.378 | 0.378 | 0.000 | 0.000 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_274 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.435 | 0.475 | 0.041 | 0.309 | 0.309 | 0.309 | 0.301 | 0.000 | -0.008 | -0.008 | 0.309 | 0.008 |
| llada-8b-instruct-hf | plan_275 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.428 | 0.428 | 0.000 | 0.391 | 0.391 | 0.391 | 0.391 | 0.000 | 0.000 | 0.000 | 0.429 | 0.038 |
| llada-8b-instruct-hf | plan_276 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.421 | 0.421 | 0.000 | 0.260 | 0.200 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.297 | 0.037 |
| llada-8b-instruct-hf | plan_277 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.475 | 0.475 | 0.000 | 0.334 | 0.334 | 0.334 | 0.334 | 0.000 | 0.000 | 0.000 | 0.334 | 0.000 |
| llada-8b-instruct-hf | plan_278 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.355 | 0.380 | 0.026 | 0.241 | 0.200 | 0.241 | 0.200 | 0.000 | -0.041 | -0.041 | 0.281 | 0.081 |
| llada-8b-instruct-hf | plan_279 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.339 | 0.339 | 0.000 | 0.240 | 0.240 | 0.240 | 0.240 | 0.000 | 0.000 | 0.000 | 0.260 | 0.020 |
| llada-8b-instruct-hf | plan_280 | low_confidence_32 | low_confidence_32 | random_32 | evolved_revision_random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.387 | 0.499 | 0.111 | 0.303 | 0.303 | 0.273 | 0.437 | -0.030 | 0.134 | 0.164 | 0.437 | 0.000 |
| llada-8b-instruct-hf | plan_281 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.392 | 0.392 | 0.000 | 0.296 | 0.178 | 0.296 | 0.296 | 0.000 | 0.000 | 0.000 | 0.296 | 0.000 |
| llada-8b-instruct-hf | plan_282 | low_confidence_32 | low_confidence_32 | random_32 | evolved_revision_random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.377 | 0.429 | 0.051 | 0.293 | 0.293 | 0.255 | 0.351 | -0.038 | 0.059 | 0.096 | 0.351 | 0.000 |
| llada-8b-instruct-hf | plan_283 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.323 | 0.369 | 0.046 | 0.213 | 0.241 | 0.213 | 0.241 | 0.000 | 0.029 | 0.029 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_284 | low_confidence_32 | random_32 | random_32 | random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.410 | 0.410 | 0.000 | 0.345 | 0.366 | 0.366 | 0.366 | 0.021 | 0.021 | 0.000 | 0.384 | 0.017 |
| llada-8b-instruct-hf | plan_285 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.445 | 0.481 | 0.036 | 0.395 | 0.395 | 0.395 | 0.461 | 0.000 | 0.066 | 0.066 | 0.461 | 0.000 |
| llada-8b-instruct-hf | plan_286 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.348 | 0.348 | 0.000 | 0.234 | 0.197 | 0.234 | 0.234 | 0.000 | 0.000 | 0.000 | 0.313 | 0.079 |
| llada-8b-instruct-hf | plan_287 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.435 | 0.472 | 0.038 | 0.260 | 0.260 | 0.260 | 0.310 | 0.000 | 0.050 | 0.050 | 0.330 | 0.020 |
| llada-8b-instruct-hf | plan_288 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.317 | 0.317 | 0.000 | 0.178 | 0.178 | 0.178 | 0.178 | 0.000 | 0.000 | 0.000 | 0.241 | 0.063 |
| llada-8b-instruct-hf | plan_289 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.397 | 0.397 | 0.000 | 0.324 | 0.324 | 0.324 | 0.324 | 0.000 | 0.000 | 0.000 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_290 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.392 | 0.424 | 0.033 | 0.233 | 0.233 | 0.233 | 0.299 | 0.000 | 0.066 | 0.066 | 0.299 | 0.000 |
| llada-8b-instruct-hf | plan_291 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.358 | 0.391 | 0.033 | 0.180 | 0.180 | 0.180 | 0.172 | 0.000 | -0.008 | -0.008 | 0.214 | 0.042 |
| llada-8b-instruct-hf | plan_292 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.413 | 0.456 | 0.043 | 0.339 | 0.339 | 0.339 | 0.389 | 0.000 | 0.050 | 0.050 | 0.389 | 0.000 |
| llada-8b-instruct-hf | plan_293 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.396 | 0.443 | 0.047 | 0.240 | 0.240 | 0.240 | 0.260 | 0.000 | 0.020 | 0.020 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_294 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.442 | 0.442 | 0.000 | 0.311 | 0.190 | 0.311 | 0.311 | 0.000 | 0.000 | 0.000 | 0.354 | 0.042 |
| llada-8b-instruct-hf | plan_295 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.449 | 0.480 | 0.031 | 0.291 | 0.291 | 0.414 | 0.434 | 0.123 | 0.143 | 0.020 | 0.434 | 0.000 |
| llada-8b-instruct-hf | plan_296 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.449 | 0.470 | 0.020 | 0.350 | 0.329 | 0.350 | 0.350 | 0.000 | 0.000 | 0.000 | 0.362 | 0.013 |
