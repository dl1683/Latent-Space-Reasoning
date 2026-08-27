# Diffusion Schedule-Selection Benchmark Report

Full model generations: `99`
Arm selections: `88`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state_v2`
Evolved promotion margin: `0.015`
Trajectory task delta vs fixed: `0.018`
Trajectory task delta vs random: `0.031`
Trajectory wins/ties/losses vs fixed: `6/13/3`
Trajectory wins/ties/losses vs random: `6/14/2`
Oracle generation budget/task: `4.50`
Oracle task score: `0.481`
Oracle headroom vs trajectory: `0.027`
Oracle wins/ties/losses vs trajectory: `8/14/0`
Evolved task delta vs fixed: `0.029`
Evolved task delta vs random: `0.043`
Evolved task delta vs trajectory: `0.011`
Evolved wins/ties/losses vs fixed: `9/11/2`
Evolved wins/ties/losses vs random: `10/12/0`
Evolved wins/ties/losses vs trajectory: `4/18/0`
Oracle headroom vs evolved: `0.015`
Oracle wins/ties/losses vs evolved: `5/17/0`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 22 | 1.00 | 0.436 | 0.497 | 0.451 |
| random | 22 | 1.00 | 0.423 | 0.493 | 0.440 |
| trajectory_selected | 22 | 2.50 | 0.454 | 0.527 | 0.472 |
| evolved | 22 | 4.50 | 0.465 | 0.521 | 0.479 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | math_001 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | evolved_entropy_96 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.046 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_001 | entropy_32 | entropy_32 | origin_64 | origin_64 | origin_64 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.299 | 0.299 | 0.000 | 0.128 | 0.128 | 0.243 | 0.243 | 0.115 | 0.115 | 0.000 | 0.243 | 0.000 |
| dream-7b-instruct-hf | plan_002 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | origin_64 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.437 | 0.437 | 0.000 | 0.542 | 0.542 | 0.542 | 0.542 | 0.000 | 0.000 | 0.000 | 0.597 | 0.055 |
| dream-7b-instruct-hf | plan_003 | entropy_32 | entropy_32 | entropy_64 | entropy_64 | entropy_64 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.454 | 0.454 | 0.000 | 0.106 | 0.106 | 0.359 | 0.359 | 0.252 | 0.252 | 0.000 | 0.359 | 0.000 |
| dream-7b-instruct-hf | plan_004 | entropy_32 | origin_64 | entropy_64 | entropy_64 | evolved_entropy_96 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.349 | 0.349 | 0.000 | 0.283 | 0.178 | 0.303 | 0.303 | 0.020 | 0.020 | 0.000 | 0.303 | 0.000 |
| dream-7b-instruct-hf | plan_005 | entropy_32 | entropy_64 | entropy_64 | entropy_64 | evolved_entropy_48 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.377 | 0.377 | 0.000 | 0.319 | 0.319 | 0.319 | 0.319 | 0.000 | 0.000 | 0.000 | 0.356 | 0.037 |
| dream-7b-instruct-hf | plan_006 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | evolved_entropy_48 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.470 | 0.470 | 0.000 | 0.434 | 0.434 | 0.434 | 0.434 | 0.000 | 0.000 | 0.000 | 0.434 | 0.000 |
| dream-7b-instruct-hf | plan_007 | entropy_32 | entropy_32 | entropy_64 | entropy_64 | entropy_64 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.416 | 0.416 | 0.000 | 0.340 | 0.340 | 0.433 | 0.433 | 0.093 | 0.093 | 0.000 | 0.433 | 0.000 |
| dream-7b-instruct-hf | plan_008 | entropy_32 | origin_64 | origin_64 | origin_64 | origin_64 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.342 | 0.342 | 0.000 | 0.138 | 0.243 | 0.243 | 0.243 | 0.104 | 0.104 | 0.000 | 0.243 | 0.000 |
| dream-7b-instruct-hf | sci_001 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.314 | 0.314 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| dream-7b-instruct-hf | sym_002 | entropy_32 | entropy_64 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.242 | 0.242 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.040 | 0.040 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | 0.292 | 0.453 | 0.161 | 0.399 | 0.473 | 0.399 | 0.529 | 0.000 | 0.130 | 0.130 | 0.529 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | evolved_low_confidence_48 | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | 0.447 | 0.469 | 0.022 | 0.604 | 0.604 | 0.602 | 0.637 | -0.002 | 0.032 | 0.035 | 0.654 | 0.018 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.533 | 0.533 | 0.000 | 0.443 | 0.284 | 0.443 | 0.443 | 0.000 | 0.000 | 0.000 | 0.443 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | 0.363 | 0.448 | 0.085 | 0.283 | 0.283 | 0.283 | 0.347 | 0.000 | 0.064 | 0.064 | 0.347 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | random_32 | random_32 | low_confidence_32 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.382 | 0.382 | 0.000 | 0.378 | 0.349 | 0.349 | 0.349 | -0.029 | -0.029 | 0.000 | 0.378 | 0.029 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | random_32 | evolved_random_48 | evolved_random_48 | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | 0.414 | 0.457 | 0.043 | 0.298 | 0.341 | 0.341 | 0.363 | 0.044 | 0.065 | 0.021 | 0.363 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | random_32 | random_32 | low_confidence_32 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.491 | 0.491 | 0.000 | 0.610 | 0.411 | 0.411 | 0.411 | -0.199 | -0.199 | 0.000 | 0.610 | 0.199 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | 0.484 | 0.484 | 0.000 | 0.283 | 0.264 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | random_32 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.109 | 0.109 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | fixed_exact_answer_guard | fixed_exact_answer_guard | 0.040 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
