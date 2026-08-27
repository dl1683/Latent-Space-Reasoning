# Diffusion Schedule-Selection Benchmark Report

Full model generations: `168`
Counterfactual probe generations: `0`
Arm selections: `96`
Run ID: `diffusion-fbb3ce2c7d50eb86`
Content hash: `fbb3ce2c7d50eb866e1c8d29aa980011afbee0fbfa4a1fe4d2c979e5ab104288`
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
History mutability: `monotonic 120/168, changes 0, remasks 637, rewrites 117, mask increases 96`
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
Trajectory task delta vs fixed: `0.008`
Trajectory task delta vs random: `0.036`
Trajectory wins/ties/losses vs fixed: `2/20/2`
Trajectory wins/ties/losses vs random: `8/13/3`
Oracle generation budget/task: `7.00`
Oracle task score: `0.317`
Oracle headroom vs trajectory: `0.034`
Oracle wins/ties/losses vs trajectory: `15/9/0`
Selector regret vs trajectory: `0.034 over 15/24 improvable`
Evolved task delta vs fixed: `0.027`
Evolved task delta vs random: `0.055`
Evolved task delta vs trajectory: `0.019`
Evolved wins/ties/losses vs fixed: `11/11/2`
Evolved wins/ties/losses vs random: `13/9/2`
Evolved wins/ties/losses vs trajectory: `9/14/1`
Oracle headroom vs evolved: `0.015`
Oracle wins/ties/losses vs evolved: `8/16/0`
Selector regret vs evolved: `0.015 over 8/24 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 24 | 1.00 | 0.275 | 0.671 | 0.374 |
| random | 24 | 1.00 | 0.246 | 0.586 | 0.331 |
| trajectory_selected | 24 | 2.00 | 0.283 | 0.670 | 0.380 |
| evolved | 24 | 7.00 | 0.302 | 0.668 | 0.393 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 24 | 1.00 | 0.275 | 0.671 | 0.374 |
| planning | random | 24 | 1.00 | 0.246 | 0.586 | 0.331 |
| planning | trajectory_selected | 24 | 2.00 | 0.283 | 0.670 | 0.380 |
| planning | evolved | 24 | 7.00 | 0.302 | 0.668 | 0.393 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_225 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.380 | 0.380 | 0.000 | 0.045 | 0.295 | 0.295 | 0.295 | 0.250 | 0.250 | 0.000 | 0.295 | 0.000 |
| llada-8b-instruct-hf | plan_226 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.381 | 0.381 | 0.000 | 0.283 | 0.330 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.330 | 0.047 |
| llada-8b-instruct-hf | plan_227 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.379 | 0.396 | 0.018 | 0.281 | 0.240 | 0.281 | 0.260 | 0.000 | -0.021 | -0.021 | 0.301 | 0.041 |
| llada-8b-instruct-hf | plan_228 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.359 | 0.393 | 0.034 | 0.240 | 0.240 | 0.240 | 0.281 | 0.000 | 0.041 | 0.041 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_229 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.340 | 0.399 | 0.060 | 0.280 | 0.280 | 0.280 | 0.390 | 0.000 | 0.110 | 0.110 | 0.390 | 0.000 |
| llada-8b-instruct-hf | plan_230 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.448 | 0.472 | 0.024 | 0.316 | 0.253 | 0.316 | 0.329 | 0.000 | 0.013 | 0.013 | 0.329 | 0.000 |
| llada-8b-instruct-hf | plan_231 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.313 | 0.398 | 0.085 | 0.220 | 0.137 | 0.220 | 0.258 | 0.000 | 0.038 | 0.038 | 0.263 | 0.005 |
| llada-8b-instruct-hf | plan_232 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.364 | 0.364 | 0.000 | 0.221 | 0.221 | 0.221 | 0.221 | 0.000 | 0.000 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_233 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.466 | 0.466 | 0.000 | 0.325 | 0.105 | 0.325 | 0.325 | 0.000 | 0.000 | 0.000 | 0.325 | 0.000 |
| llada-8b-instruct-hf | plan_234 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.309 | 0.309 | 0.000 | 0.260 | 0.260 | 0.197 | 0.197 | -0.063 | -0.063 | 0.000 | 0.323 | 0.126 |
| llada-8b-instruct-hf | plan_235 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.451 | 0.451 | 0.000 | 0.458 | 0.458 | 0.458 | 0.458 | 0.000 | 0.000 | 0.000 | 0.458 | 0.000 |
| llada-8b-instruct-hf | plan_236 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.379 | 0.425 | 0.047 | 0.240 | 0.157 | 0.240 | 0.282 | 0.000 | 0.042 | 0.042 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_237 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.449 | 0.449 | 0.000 | 0.303 | 0.303 | 0.303 | 0.303 | 0.000 | 0.000 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_238 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_64 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.431 | 0.455 | 0.024 | 0.261 | 0.045 | 0.261 | 0.261 | 0.000 | 0.000 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_239 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.416 | 0.416 | 0.000 | 0.281 | 0.281 | 0.281 | 0.281 | 0.000 | 0.000 | 0.000 | 0.283 | 0.001 |
| llada-8b-instruct-hf | plan_240 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.487 | 0.487 | 0.000 | 0.309 | 0.309 | 0.309 | 0.309 | 0.000 | 0.000 | 0.000 | 0.346 | 0.037 |
| llada-8b-instruct-hf | plan_241 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.420 | 0.420 | 0.000 | 0.200 | 0.200 | 0.200 | 0.200 | 0.000 | 0.000 | 0.000 | 0.284 | 0.084 |
| llada-8b-instruct-hf | plan_242 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.341 | 0.341 | 0.000 | 0.259 | 0.302 | 0.302 | 0.302 | 0.043 | 0.043 | 0.000 | 0.302 | 0.000 |
| llada-8b-instruct-hf | plan_243 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.433 | 0.498 | 0.065 | 0.336 | 0.240 | 0.336 | 0.358 | 0.000 | 0.021 | 0.021 | 0.379 | 0.021 |
| llada-8b-instruct-hf | plan_244 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.373 | 0.393 | 0.019 | 0.241 | 0.241 | 0.241 | 0.290 | 0.000 | 0.049 | 0.049 | 0.290 | 0.000 |
| llada-8b-instruct-hf | plan_245 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_64 | evolved_low_confidence_64 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.360 | 0.432 | 0.073 | 0.282 | 0.282 | 0.240 | 0.379 | -0.042 | 0.096 | 0.139 | 0.379 | 0.000 |
| llada-8b-instruct-hf | plan_246 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.329 | 0.398 | 0.069 | 0.394 | 0.394 | 0.394 | 0.415 | 0.000 | 0.021 | 0.021 | 0.415 | 0.000 |
| llada-8b-instruct-hf | plan_247 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.456 | 0.456 | 0.000 | 0.270 | 0.045 | 0.270 | 0.270 | 0.000 | 0.000 | 0.000 | 0.270 | 0.000 |
| llada-8b-instruct-hf | plan_248 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.431 | 0.431 | 0.000 | 0.295 | 0.295 | 0.295 | 0.295 | 0.000 | 0.000 | 0.000 | 0.295 | 0.000 |
