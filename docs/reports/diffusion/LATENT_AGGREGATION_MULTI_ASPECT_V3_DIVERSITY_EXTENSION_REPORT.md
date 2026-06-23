# Diffusion Schedule-Selection Benchmark Report

Full model generations: `168`
Counterfactual probe generations: `0`
Arm selections: `96`
Run ID: `diffusion-ecb76149d3f7714b`
Content hash: `ecb76149d3f7714b2b16957a6f8208c592bbed320314e892f49026a0d46aee23`
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
History mutability: `monotonic 120/168, changes 0, remasks 645, rewrites 92, mask increases 96`
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
Trajectory task delta vs fixed: `0.000`
Trajectory task delta vs random: `0.021`
Trajectory wins/ties/losses vs fixed: `0/24/0`
Trajectory wins/ties/losses vs random: `7/15/2`
Oracle generation budget/task: `7.00`
Oracle task score: `0.337`
Oracle headroom vs trajectory: `0.023`
Oracle wins/ties/losses vs trajectory: `14/10/0`
Selector regret vs trajectory: `0.023 over 14/24 improvable`
Evolved task delta vs fixed: `0.009`
Evolved task delta vs random: `0.030`
Evolved task delta vs trajectory: `0.009`
Evolved wins/ties/losses vs fixed: `5/19/0`
Evolved wins/ties/losses vs random: `11/13/0`
Evolved wins/ties/losses vs trajectory: `5/19/0`
Oracle headroom vs evolved: `0.013`
Oracle wins/ties/losses vs evolved: `9/15/0`
Selector regret vs evolved: `0.013 over 9/24 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 24 | 1.00 | 0.314 | 0.698 | 0.410 |
| random | 24 | 1.00 | 0.293 | 0.640 | 0.380 |
| trajectory_selected | 24 | 2.00 | 0.314 | 0.698 | 0.410 |
| evolved | 24 | 7.00 | 0.323 | 0.696 | 0.416 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 24 | 1.00 | 0.314 | 0.698 | 0.410 |
| planning | random | 24 | 1.00 | 0.293 | 0.640 | 0.380 |
| planning | trajectory_selected | 24 | 2.00 | 0.314 | 0.698 | 0.410 |
| planning | evolved | 24 | 7.00 | 0.323 | 0.696 | 0.416 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_201 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.393 | 0.393 | 0.000 | 0.323 | 0.323 | 0.323 | 0.323 | 0.000 | 0.000 | 0.000 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_202 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.418 | 0.438 | 0.020 | 0.321 | 0.321 | 0.321 | 0.321 | 0.000 | 0.000 | 0.000 | 0.426 | 0.105 |
| llada-8b-instruct-hf | plan_203 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.432 | 0.432 | 0.000 | 0.280 | 0.280 | 0.280 | 0.280 | 0.000 | 0.000 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_204 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.356 | 0.414 | 0.059 | 0.260 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.281 | 0.021 |
| llada-8b-instruct-hf | plan_205 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.333 | 0.333 | 0.000 | 0.260 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_206 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.506 | 0.506 | 0.000 | 0.486 | 0.420 | 0.486 | 0.486 | 0.000 | 0.000 | 0.000 | 0.529 | 0.043 |
| llada-8b-instruct-hf | plan_207 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.339 | 0.339 | 0.000 | 0.378 | 0.378 | 0.378 | 0.378 | 0.000 | 0.000 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_208 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.436 | 0.455 | 0.018 | 0.311 | 0.311 | 0.311 | 0.324 | 0.000 | 0.013 | 0.013 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_209 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.375 | 0.375 | 0.000 | 0.281 | 0.197 | 0.281 | 0.281 | 0.000 | 0.000 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_210 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.380 | 0.411 | 0.031 | 0.315 | 0.315 | 0.315 | 0.315 | 0.000 | 0.000 | 0.000 | 0.336 | 0.021 |
| llada-8b-instruct-hf | plan_211 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.362 | 0.362 | 0.000 | 0.339 | 0.339 | 0.339 | 0.339 | 0.000 | 0.000 | 0.000 | 0.375 | 0.036 |
| llada-8b-instruct-hf | plan_212 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.384 | 0.384 | 0.000 | 0.281 | 0.260 | 0.281 | 0.281 | 0.000 | 0.000 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_213 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.376 | 0.431 | 0.055 | 0.274 | 0.281 | 0.274 | 0.351 | 0.000 | 0.078 | 0.078 | 0.351 | 0.000 |
| llada-8b-instruct-hf | plan_214 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.320 | 0.388 | 0.069 | 0.299 | 0.214 | 0.299 | 0.299 | 0.000 | 0.000 | 0.000 | 0.299 | 0.000 |
| llada-8b-instruct-hf | plan_215 | low_confidence_32 | random_32 | random_32 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.339 | 0.429 | 0.089 | 0.335 | 0.335 | 0.335 | 0.406 | 0.000 | 0.071 | 0.071 | 0.406 | 0.000 |
| llada-8b-instruct-hf | plan_216 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.497 | 0.497 | 0.000 | 0.391 | 0.324 | 0.391 | 0.391 | 0.000 | 0.000 | 0.000 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_217 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.402 | 0.402 | 0.000 | 0.281 | 0.277 | 0.281 | 0.281 | 0.000 | 0.000 | 0.000 | 0.331 | 0.050 |
| llada-8b-instruct-hf | plan_218 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.325 | 0.402 | 0.077 | 0.281 | 0.282 | 0.281 | 0.324 | 0.000 | 0.042 | 0.042 | 0.324 | 0.000 |
| llada-8b-instruct-hf | plan_219 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.355 | 0.395 | 0.040 | 0.303 | 0.303 | 0.303 | 0.303 | 0.000 | 0.000 | 0.000 | 0.323 | 0.020 |
| llada-8b-instruct-hf | plan_220 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.350 | 0.374 | 0.024 | 0.241 | 0.065 | 0.241 | 0.261 | 0.000 | 0.020 | 0.020 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_221 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.508 | 0.508 | 0.000 | 0.318 | 0.318 | 0.318 | 0.318 | 0.000 | 0.000 | 0.000 | 0.338 | 0.020 |
| llada-8b-instruct-hf | plan_222 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.405 | 0.405 | 0.000 | 0.399 | 0.399 | 0.399 | 0.399 | 0.000 | 0.000 | 0.000 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_223 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.434 | 0.471 | 0.037 | 0.315 | 0.315 | 0.315 | 0.315 | 0.000 | 0.000 | 0.000 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_224 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.392 | 0.392 | 0.000 | 0.260 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.263 | 0.003 |
