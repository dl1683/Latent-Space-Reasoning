# Diffusion Schedule-Selection Benchmark Report

Full model generations: `12`
Arm selections: `12`
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
History mutability: `monotonic 6/12, changes 0, remasks 96, rewrites 28, mask increases 96`
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
Trajectory task delta vs fixed: `0.006`
Trajectory task delta vs random: `0.000`
Trajectory wins/ties/losses vs fixed: `1/2/0`
Trajectory wins/ties/losses vs random: `0/3/0`
Oracle generation budget/task: `4.00`
Oracle task score: `0.395`
Oracle headroom vs trajectory: `0.058`
Oracle wins/ties/losses vs trajectory: `1/2/0`
Selector regret vs trajectory: `0.058 over 1/3 improvable`
Evolved task delta vs fixed: `0.064`
Evolved task delta vs random: `0.058`
Evolved task delta vs trajectory: `0.058`
Evolved wins/ties/losses vs fixed: `2/1/0`
Evolved wins/ties/losses vs random: `1/2/0`
Evolved wins/ties/losses vs trajectory: `1/2/0`
Oracle headroom vs evolved: `0.000`
Oracle wins/ties/losses vs evolved: `0/3/0`
Selector regret vs evolved: `0.000 over 0/3 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 3 | 1.00 | 0.331 | 0.659 | 0.413 |
| random | 3 | 1.00 | 0.337 | 0.659 | 0.418 |
| trajectory_selected | 3 | 2.00 | 0.337 | 0.659 | 0.418 |
| evolved | 3 | 4.00 | 0.395 | 0.631 | 0.454 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 3 | 1.00 | 0.331 | 0.659 | 0.413 |
| planning | random | 3 | 1.00 | 0.337 | 0.659 | 0.418 |
| planning | trajectory_selected | 3 | 2.00 | 0.337 | 0.659 | 0.418 |
| planning | evolved | 3 | 4.00 | 0.395 | 0.631 | 0.454 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.418 | 0.418 | 0.000 | 0.422 | 0.422 | 0.422 | 0.422 | 0.000 | 0.000 | 0.000 | 0.422 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | 0.333 | 0.404 | 0.071 | 0.307 | 0.307 | 0.307 | 0.481 | 0.000 | 0.174 | 0.174 | 0.481 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 | random_32 | random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | 0.274 | 0.274 | 0.000 | 0.264 | 0.283 | 0.283 | 0.283 | 0.019 | 0.019 | 0.000 | 0.283 | 0.000 |
