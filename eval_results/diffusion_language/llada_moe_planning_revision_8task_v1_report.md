# Diffusion Schedule-Selection Benchmark Report

Full model generations: `32`
Arm selections: `32`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `False`
History mutability: `monotonic 16/32, changes 0, remasks 256, rewrites 68, mask increases 256`
History repairs included: `False`
Repair pack: `prefix`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `always`
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
Constraint-gap rescue trigger: `off`
Constraint-gap rescue limit: `1`
Constraint-gap rescue min terms: `6`
Constraint-gap rescue source-quality band: `0.400-0.500`
Constraint-gap rescue source controls: ``
Repair selector: `planning_quality`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.002`
Trajectory task delta vs random: `0.042`
Trajectory wins/ties/losses vs fixed: `1/7/0`
Trajectory wins/ties/losses vs random: `3/5/0`
Oracle generation budget/task: `4.00`
Oracle task score: `0.436`
Oracle headroom vs trajectory: `0.022`
Oracle wins/ties/losses vs trajectory: `1/7/0`
Selector regret vs trajectory: `0.022 over 1/8 improvable`
Evolved task delta vs fixed: `0.024`
Evolved task delta vs random: `0.064`
Evolved task delta vs trajectory: `0.022`
Evolved wins/ties/losses vs fixed: `2/6/0`
Evolved wins/ties/losses vs random: `4/4/0`
Evolved wins/ties/losses vs trajectory: `1/7/0`
Oracle headroom vs evolved: `0.000`
Oracle wins/ties/losses vs evolved: `0/8/0`
Selector regret vs evolved: `0.000 over 0/8 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| evolved | 8 | 4.00 | 0.436 | 0.649 | 0.489 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| planning | trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| planning | evolved | 8 | 4.00 | 0.436 | 0.649 | 0.489 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.425 | 0.425 | 0.000 | 0.465 | 0.465 | 0.465 | 0.465 | 0.000 | 0.000 | 0.000 | 0.465 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.448 | 0.448 | 0.000 | 0.689 | 0.580 | 0.689 | 0.689 | 0.000 | 0.000 | 0.000 | 0.689 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.418 | 0.418 | 0.000 | 0.422 | 0.422 | 0.422 | 0.422 | 0.000 | 0.000 | 0.000 | 0.422 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.466 | 0.466 | 0.000 | 0.338 | 0.157 | 0.338 | 0.338 | 0.000 | 0.000 | 0.000 | 0.338 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.334 | 0.334 | 0.000 | 0.421 | 0.421 | 0.421 | 0.421 | 0.000 | 0.000 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.366 | 0.366 | 0.000 | 0.391 | 0.341 | 0.391 | 0.391 | 0.000 | 0.000 | 0.000 | 0.391 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | 0.333 | 0.404 | 0.071 | 0.307 | 0.307 | 0.307 | 0.481 | 0.000 | 0.174 | 0.174 | 0.481 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.274 | 0.274 | 0.000 | 0.264 | 0.283 | 0.283 | 0.283 | 0.019 | 0.019 | 0.000 | 0.283 | 0.000 |
