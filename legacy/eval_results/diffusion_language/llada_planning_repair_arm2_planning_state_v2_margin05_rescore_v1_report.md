# Diffusion Schedule-Selection Benchmark Report

Full model generations: `48`
Arm selections: `40`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state_v2`
Evolved promotion margin: `0.015`
Repair promotion margin: `0.050`
Trajectory task delta vs fixed: `-0.023`
Trajectory task delta vs random: `0.013`
Trajectory wins/ties/losses vs fixed: `1/4/3`
Trajectory wins/ties/losses vs random: `2/4/2`
Oracle generation budget/task: `6.00`
Oracle task score: `0.467`
Oracle headroom vs trajectory: `0.078`
Oracle wins/ties/losses vs trajectory: `8/0/0`
Evolved task delta vs fixed: `0.008`
Evolved task delta vs random: `0.044`
Evolved task delta vs trajectory: `0.031`
Evolved wins/ties/losses vs fixed: `4/2/2`
Evolved wins/ties/losses vs random: `6/2/0`
Evolved wins/ties/losses vs trajectory: `4/4/0`
Oracle headroom vs evolved: `0.047`
Oracle wins/ties/losses vs evolved: `6/2/0`
Repair task delta vs fixed: `0.016`
Repair task delta vs random: `0.052`
Repair task delta vs trajectory: `0.039`
Repair task delta vs evolved: `0.008`
Repair wins/ties/losses vs evolved: `2/6/0`
Oracle headroom vs repair: `0.039`
Oracle wins/ties/losses vs repair: `4/4/0`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| random | 8 | 1.00 | 0.376 | 0.701 | 0.457 |
| trajectory_selected | 8 | 2.00 | 0.389 | 0.699 | 0.467 |
| evolved | 8 | 4.00 | 0.420 | 0.685 | 0.486 |
| repair_selected | 8 | 6.00 | 0.428 | 0.683 | 0.492 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | prefix_25_repair | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | repair_margin_guard_kept_evolved_0.050 | 0.292 | 0.453 | 0.453 | 0.000 | 0.399 | 0.473 | 0.399 | 0.529 | 0.529 | 0.000 | 0.592 | 0.063 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | prefix_50_repair | prefix_50_repair | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | max_planning_state_v2_score_repair_pool | 0.447 | 0.469 | 0.543 | 0.075 | 0.604 | 0.604 | 0.602 | 0.637 | 0.662 | 0.025 | 0.662 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | prefix_25_repair | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | repair_margin_guard_kept_evolved_0.050 | 0.533 | 0.533 | 0.533 | 0.000 | 0.443 | 0.284 | 0.443 | 0.443 | 0.443 | 0.000 | 0.464 | 0.021 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | repair_margin_guard_kept_evolved_0.050 | 0.363 | 0.448 | 0.448 | 0.000 | 0.283 | 0.283 | 0.283 | 0.347 | 0.347 | 0.000 | 0.347 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | low_confidence_32 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | repair_margin_guard_kept_evolved_0.050 | 0.382 | 0.382 | 0.382 | 0.000 | 0.378 | 0.349 | 0.349 | 0.349 | 0.349 | 0.000 | 0.378 | 0.029 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | random_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_v2_score_base_pool | max_planning_state_v2_score_evolved_pool | repair_margin_guard_kept_evolved_0.050 | 0.414 | 0.457 | 0.457 | 0.000 | 0.298 | 0.341 | 0.341 | 0.363 | 0.363 | 0.000 | 0.363 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | low_confidence_32 | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | repair_margin_guard_kept_evolved_0.050 | 0.491 | 0.491 | 0.491 | 0.000 | 0.610 | 0.411 | 0.411 | 0.411 | 0.411 | 0.000 | 0.610 | 0.199 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | prefix_25_repair | prefix_25_repair | max_planning_state_v2_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_state_v2_score_repair_pool | 0.484 | 0.484 | 0.588 | 0.104 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
