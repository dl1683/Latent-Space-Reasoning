# Diffusion Schedule-Selection Benchmark Report

Full model generations: `99`
Arm selections: `88`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved promotion margin: `0.015`
Trajectory task delta vs fixed: `0.029`
Trajectory task delta vs random: `0.042`
Trajectory wins/ties/losses vs fixed: `6/15/1`
Trajectory wins/ties/losses vs random: `9/10/3`
Evolved task delta vs fixed: `0.039`
Evolved task delta vs random: `0.052`
Evolved task delta vs trajectory: `0.010`
Evolved wins/ties/losses vs fixed: `9/13/0`
Evolved wins/ties/losses vs random: `12/9/1`
Evolved wins/ties/losses vs trajectory: `3/19/0`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 22 | 1.00 | 0.436 | 0.497 | 0.451 |
| random | 22 | 1.00 | 0.423 | 0.493 | 0.440 |
| trajectory_selected | 22 | 2.50 | 0.465 | 0.526 | 0.480 |
| evolved | 22 | 4.50 | 0.475 | 0.520 | 0.486 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | math_001 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.046 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_001 | entropy_32 | entropy_32 | origin_64 | origin_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.259 | 0.259 | 0.000 | 0.128 | 0.128 | 0.243 | 0.243 | 0.115 | 0.115 | 0.000 |
| dream-7b-instruct-hf | plan_002 | entropy_32 | entropy_32 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.398 | 0.398 | 0.000 | 0.542 | 0.542 | 0.593 | 0.593 | 0.051 | 0.051 | 0.000 |
| dream-7b-instruct-hf | plan_003 | entropy_32 | entropy_32 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.427 | 0.427 | 0.000 | 0.106 | 0.106 | 0.359 | 0.359 | 0.252 | 0.252 | 0.000 |
| dream-7b-instruct-hf | plan_004 | entropy_32 | origin_64 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.307 | 0.307 | 0.000 | 0.283 | 0.178 | 0.303 | 0.303 | 0.020 | 0.020 | 0.000 |
| dream-7b-instruct-hf | plan_005 | entropy_32 | entropy_64 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.299 | 0.299 | 0.000 | 0.319 | 0.319 | 0.319 | 0.319 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_006 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.391 | 0.391 | 0.000 | 0.434 | 0.434 | 0.434 | 0.434 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_007 | entropy_32 | entropy_32 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.384 | 0.384 | 0.000 | 0.340 | 0.340 | 0.433 | 0.433 | 0.093 | 0.093 | 0.000 |
| dream-7b-instruct-hf | plan_008 | entropy_32 | origin_64 | origin_64 | origin_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.273 | 0.273 | 0.000 | 0.138 | 0.243 | 0.243 | 0.243 | 0.104 | 0.104 | 0.000 |
| dream-7b-instruct-hf | sci_001 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.314 | 0.314 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | sym_002 | entropy_32 | entropy_64 | entropy_32 | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.242 | 0.242 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.040 | 0.040 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.332 | 0.479 | 0.147 | 0.399 | 0.473 | 0.399 | 0.529 | 0.000 | 0.130 | 0.130 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.424 | 0.447 | 0.023 | 0.604 | 0.604 | 0.602 | 0.637 | -0.002 | 0.032 | 0.035 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.508 | 0.508 | 0.000 | 0.443 | 0.284 | 0.443 | 0.443 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.289 | 0.366 | 0.078 | 0.283 | 0.283 | 0.283 | 0.347 | 0.000 | 0.064 | 0.064 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.338 | 0.376 | 0.038 | 0.378 | 0.349 | 0.378 | 0.378 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.363 | 0.363 | 0.000 | 0.298 | 0.341 | 0.298 | 0.298 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_state_score_evolved_pool | 0.509 | 0.532 | 0.023 | 0.610 | 0.411 | 0.610 | 0.610 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.410 | 0.410 | 0.000 | 0.283 | 0.264 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.109 | 0.109 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.040 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
