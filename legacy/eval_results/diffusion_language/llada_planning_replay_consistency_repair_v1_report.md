# Diffusion Schedule-Selection Benchmark Report

Full model generations: `46`
Arm selections: `40`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
History repairs included: `False`
Repair pack: `replay_consistency`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `source_quality_or_short`
Repair source-quality threshold: `0.500`
Repair source min chars: `320`
Repair source controls: ``
History rescue fractions: ``
History rescue visible: `False`
History rescue trigger: `baseline`
History rescue source controls: ``
Prompt-guided rescue trigger: `off`
Prompt-guided rescue limit: `1`
Prompt-guided rescue source-quality threshold: `0.450`
Prompt-guided rescue source controls: ``
Repair selector: `planning_quality_delta_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `-0.000`
Trajectory task delta vs random: `0.036`
Trajectory wins/ties/losses vs fixed: `0/7/1`
Trajectory wins/ties/losses vs random: `4/1/3`
Oracle generation budget/task: `5.75`
Oracle task score: `0.477`
Oracle headroom vs trajectory: `0.065`
Oracle wins/ties/losses vs trajectory: `6/2/0`
Evolved task delta vs fixed: `0.039`
Evolved task delta vs random: `0.075`
Evolved task delta vs trajectory: `0.039`
Evolved wins/ties/losses vs fixed: `4/4/0`
Evolved wins/ties/losses vs random: `8/0/0`
Evolved wins/ties/losses vs trajectory: `4/4/0`
Oracle headroom vs evolved: `0.026`
Oracle wins/ties/losses vs evolved: `4/4/0`
Repair arm coverage: `8/8` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.064`
Repair task delta vs random: `0.100`
Repair task delta vs trajectory: `0.065`
Repair task delta vs evolved: `0.026`
Repair generation budget delta vs evolved: `1.75`
Repair task delta per extra generation vs evolved: `0.015`
Repair wins/ties/losses vs evolved: `4/4/0`
Oracle headroom vs repair: `0.000`
Oracle wins/ties/losses vs repair: `0/8/0`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.698 | 0.484 |
| random | 8 | 1.00 | 0.376 | 0.701 | 0.457 |
| trajectory_selected | 8 | 2.00 | 0.412 | 0.698 | 0.483 |
| evolved | 8 | 4.00 | 0.451 | 0.682 | 0.509 |
| repair_selected | 8 | 5.75 | 0.477 | 0.678 | 0.527 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source | Masked/Run | Guard Penalty | PQ Delta | Task Delta | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| replay_unstable_25_repair | 7 | 0 | final | 16.0 | 0.000 | 0.000 | 0.000 | 0/7/0 | 0.428 | 0.693 | 0.494 |
| state_adaptive_history_repair | 7 | 4 | history | 47.1 | 0.000 | -0.003 | 0.003 | 4/1/2 | 0.431 | 0.655 | 0.487 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | replay_unstable_25_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.332 | 0.479 | 0.000 | 0.000 | 0.399 | 0.473 | 0.399 | 0.529 | 0.529 | 0.000 | 0.529 | 0.000 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | evolved_low_confidence_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.424 | 0.443 | 0.000 | 0.000 | 0.604 | 0.604 | 0.602 | 0.654 | 0.654 | 0.000 | 0.654 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_delta_guarded_score_repair_pool | history | 26 | 0.508 | 0.508 | 0.021 | 0.021 | 0.443 | 0.284 | 0.443 | 0.443 | 0.464 | 0.021 | 0.464 | 0.000 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_guarded_score_repair_pool | history | 39 | 0.289 | 0.366 | 0.029 | 0.029 | 0.283 | 0.283 | 0.283 | 0.347 | 0.375 | 0.029 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  | 0.338 | 0.376 | 0.000 | 0.000 | 0.378 | 0.349 | 0.378 | 0.378 | 0.378 | 0.000 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_delta_guarded_score_repair_pool | history | 39 | 0.363 | 0.365 | 0.076 | 0.076 | 0.298 | 0.341 | 0.298 | 0.363 | 0.479 | 0.116 | 0.479 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_spend_gate_kept_evolved_source_quality_or_short |  |  | 0.509 | 0.532 | 0.000 | 0.000 | 0.610 | 0.411 | 0.610 | 0.610 | 0.610 | 0.000 | 0.610 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | state_adaptive_history_repair | state_adaptive_history_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015 | max_planning_quality_delta_guarded_score_repair_pool | history | 20 | 0.410 | 0.410 | 0.080 | 0.080 | 0.283 | 0.264 | 0.283 | 0.283 | 0.323 | 0.040 | 0.323 | 0.000 |
