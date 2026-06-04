# Diffusion Schedule-Selection Benchmark Report

Full model generations: `56`
Arm selections: `40`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
History repairs included: `True`
History repair fractions: `0.25`
History visible repair included: `False`
History rescue fractions: `0.50`
History rescue visible: `True`
History rescue trigger: `baseline_or_selector_disagreement`
History rescue source controls: ``
Repair selector: `planning_quality_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `-0.000`
Trajectory task delta vs random: `0.036`
Trajectory wins/ties/losses vs fixed: `0/7/1`
Trajectory wins/ties/losses vs random: `4/1/3`
Oracle generation budget/task: `7.00`
Oracle task score: `0.490`
Oracle headroom vs trajectory: `0.078`
Oracle wins/ties/losses vs trajectory: `6/2/0`
Evolved task delta vs fixed: `0.039`
Evolved task delta vs random: `0.075`
Evolved task delta vs trajectory: `0.039`
Evolved wins/ties/losses vs fixed: `4/4/0`
Evolved wins/ties/losses vs random: `8/0/0`
Evolved wins/ties/losses vs trajectory: `4/4/0`
Oracle headroom vs evolved: `0.039`
Oracle wins/ties/losses vs evolved: `6/2/0`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.077`
Repair task delta vs random: `0.113`
Repair task delta vs trajectory: `0.078`
Repair task delta vs evolved: `0.039`
Repair generation budget delta vs evolved: `3.00`
Repair task delta per extra generation vs evolved: `0.013`
Repair wins/ties/losses vs evolved: `6/2/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| random | 8 | 1.00 | 0.376 | 0.701 | 0.457 |
| trajectory_selected | 8 | 2.00 | 0.412 | 0.698 | 0.483 |
| evolved | 8 | 4.00 | 0.451 | 0.682 | 0.509 |
| repair_selected | 8 | 7.00 | 0.490 | 0.689 | 0.539 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | Delta vs Source | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| history_prefix_25_repair | 8 | 3 | history | 49.1 | 0.000 | -0.026 | 3/1/4 | 0.425 | 0.653 | 0.482 |
| history_prefix_50_repair | 4 | 1 | history | 35.2 | 0.000 | -0.047 | 1/1/2 | 0.450 | 0.663 | 0.504 |
| history_visible_repair | 4 | 0 | history | 10.8 | 0.058 | 0.000 | 0/4/0 | 0.497 | 0.738 | 0.557 |
| prefix_25_repair | 8 | 2 | final | 48.0 | 0.000 | 0.006 | 5/1/2 | 0.456 | 0.682 | 0.513 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_guarded_score_repair_pool | final |  | 0.332 | 0.479 | 0.492 | 0.043 | 0.399 | 0.473 | 0.399 | 0.529 | 0.592 | 0.063 | 0.592 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | prefix_25_repair | prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_guarded_score_repair_pool | final |  | 0.424 | 0.443 | 0.547 | 0.023 | 0.604 | 0.604 | 0.602 | 0.654 | 0.695 | 0.040 | 0.695 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_guarded_score_repair_pool | history | 26 | 0.508 | 0.508 | 0.367 | 0.021 | 0.443 | 0.284 | 0.443 | 0.443 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | history_prefix_50_repair | history_prefix_50_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_guarded_score_repair_pool | history | 39 | 0.289 | 0.366 | 0.315 | 0.029 | 0.283 | 0.283 | 0.283 | 0.347 | 0.375 | 0.029 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | history_visible_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.338 | 0.376 | 0.298 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_guarded_score_repair_pool | history | 39 | 0.363 | 0.365 | 0.369 | 0.076 | 0.298 | 0.341 | 0.298 | 0.363 | 0.479 | 0.116 | 0.479 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | history_visible_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.509 | 0.532 | 0.500 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | history_prefix_25_repair | history_prefix_25_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_guarded_score_repair_pool | history | 20 | 0.410 | 0.410 | 0.303 | 0.080 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
