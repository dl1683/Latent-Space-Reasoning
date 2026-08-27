# Diffusion Schedule-Selection Benchmark Report

Full model generations: `115`
Arm selections: `96`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Repair selector: `planning_quality`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.029`
Trajectory task delta vs random: `0.042`
Trajectory wins/ties/losses vs fixed: `6/15/1`
Trajectory wins/ties/losses vs random: `9/10/3`
Oracle generation budget/task: `5.23`
Oracle task score: `0.491`
Oracle headroom vs trajectory: `0.026`
Oracle wins/ties/losses vs trajectory: `8/14/0`
Evolved task delta vs fixed: `0.045`
Evolved task delta vs random: `0.058`
Evolved task delta vs trajectory: `0.016`
Evolved wins/ties/losses vs fixed: `11/11/0`
Evolved wins/ties/losses vs random: `14/8/0`
Evolved wins/ties/losses vs trajectory: `5/17/0`
Oracle headroom vs evolved: `0.010`
Oracle wins/ties/losses vs evolved: `6/16/0`
Repair arm coverage: `8/22` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.067`
Repair task delta vs random: `0.103`
Repair task delta vs trajectory: `0.067`
Repair task delta vs evolved: `0.028`
Repair wins/ties/losses vs evolved: `5/3/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 22 | 1.00 | 0.436 | 0.497 | 0.451 |
| random | 22 | 1.00 | 0.423 | 0.493 | 0.440 |
| trajectory_selected | 22 | 2.50 | 0.465 | 0.526 | 0.480 |
| evolved | 22 | 4.50 | 0.480 | 0.520 | 0.490 |
| repair_selected | 8 | 6.00 | 0.479 | 0.689 | 0.532 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | math_001 | entropy_32 | entropy_32 | entropy_32 | entropy_32 |  | evolved_entropy_96 | fixed_exact_answer_guard | fixed_exact_answer_guard |  | 0.046 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_001 | entropy_32 | entropy_32 | origin_64 | origin_64 |  | origin_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.259 | 0.259 | 0.000 | 0.000 | 0.128 | 0.128 | 0.243 | 0.243 | 0.000 | 0.000 | 0.243 | 0.000 |
| dream-7b-instruct-hf | plan_002 | entropy_32 | entropy_32 | entropy_64 | entropy_64 |  | origin_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.398 | 0.398 | 0.000 | 0.000 | 0.542 | 0.542 | 0.593 | 0.593 | 0.000 | 0.000 | 0.597 | 0.000 |
| dream-7b-instruct-hf | plan_003 | entropy_32 | entropy_32 | entropy_64 | entropy_64 |  | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.427 | 0.427 | 0.000 | 0.000 | 0.106 | 0.106 | 0.359 | 0.359 | 0.000 | 0.000 | 0.359 | 0.000 |
| dream-7b-instruct-hf | plan_004 | entropy_32 | origin_64 | entropy_64 | entropy_64 |  | evolved_entropy_96 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.307 | 0.307 | 0.000 | 0.000 | 0.283 | 0.178 | 0.303 | 0.303 | 0.000 | 0.000 | 0.303 | 0.000 |
| dream-7b-instruct-hf | plan_005 | entropy_32 | entropy_64 | entropy_64 | evolved_entropy_48 |  | evolved_entropy_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool |  | 0.299 | 0.310 | 0.000 | 0.000 | 0.319 | 0.319 | 0.319 | 0.356 | 0.000 | 0.000 | 0.356 | 0.000 |
| dream-7b-instruct-hf | plan_006 | entropy_32 | entropy_32 | entropy_32 | entropy_32 |  | evolved_entropy_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.391 | 0.391 | 0.000 | 0.000 | 0.434 | 0.434 | 0.434 | 0.434 | 0.000 | 0.000 | 0.434 | 0.000 |
| dream-7b-instruct-hf | plan_007 | entropy_32 | entropy_32 | entropy_64 | entropy_64 |  | entropy_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.384 | 0.384 | 0.000 | 0.000 | 0.340 | 0.340 | 0.433 | 0.433 | 0.000 | 0.000 | 0.433 | 0.000 |
| dream-7b-instruct-hf | plan_008 | entropy_32 | origin_64 | origin_64 | origin_64 |  | origin_64 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 |  | 0.273 | 0.273 | 0.000 | 0.000 | 0.138 | 0.243 | 0.243 | 0.243 | 0.000 | 0.000 | 0.243 | 0.000 |
| dream-7b-instruct-hf | sci_001 | entropy_32 | entropy_32 | entropy_32 | entropy_32 |  | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  | 0.314 | 0.314 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| dream-7b-instruct-hf | sym_002 | entropy_32 | entropy_64 | entropy_32 | entropy_32 |  | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  | 0.242 | 0.242 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  | 0.040 | 0.040 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_score_repair_pool | 0.332 | 0.479 | 0.492 | 0.043 | 0.399 | 0.473 | 0.399 | 0.529 | 0.592 | 0.063 | 0.592 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_score_repair_pool | 0.424 | 0.443 | 0.547 | 0.023 | 0.604 | 0.604 | 0.602 | 0.654 | 0.695 | 0.040 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_score_repair_pool | 0.508 | 0.508 | 0.367 | 0.021 | 0.443 | 0.284 | 0.443 | 0.443 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 | 0.289 | 0.366 | 0.287 | 0.000 | 0.283 | 0.283 | 0.283 | 0.347 | 0.347 | 0.000 | 0.347 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 | 0.338 | 0.376 | 0.298 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_score_repair_pool | 0.363 | 0.365 | 0.314 | 0.021 | 0.298 | 0.341 | 0.298 | 0.363 | 0.424 | 0.061 | 0.424 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 | 0.509 | 0.532 | 0.500 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_score_repair_pool | 0.410 | 0.410 | 0.303 | 0.080 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard |  | 0.109 | 0.109 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard |  | 0.040 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
