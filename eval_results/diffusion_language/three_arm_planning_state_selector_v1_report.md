# Diffusion Three-Arm Benchmark Report

Full model generations: `55`
Arm selections: `66`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Trajectory task delta vs fixed: `0.029`
Trajectory task delta vs random: `0.042`
Trajectory wins/ties/losses vs fixed: `6/15/1`
Trajectory wins/ties/losses vs random: `9/10/3`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 22 | 1.00 | 0.436 | 0.497 | 0.451 |
| random | 22 | 1.00 | 0.423 | 0.493 | 0.440 |
| trajectory_selected | 22 | 2.50 | 0.465 | 0.526 | 0.480 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Reason | Selector | Fixed Task | Random Task | Trajectory Task | Delta vs Fixed | Delta vs Random |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | math_001 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_001 | entropy_32 | entropy_32 | origin_64 | max_planning_state_score | 0.259 | 0.128 | 0.128 | 0.243 | 0.115 | 0.115 |
| dream-7b-instruct-hf | plan_002 | entropy_32 | entropy_32 | entropy_64 | max_planning_state_score | 0.398 | 0.542 | 0.542 | 0.593 | 0.051 | 0.051 |
| dream-7b-instruct-hf | plan_003 | entropy_32 | entropy_32 | entropy_64 | max_planning_state_score | 0.427 | 0.106 | 0.106 | 0.359 | 0.252 | 0.252 |
| dream-7b-instruct-hf | plan_004 | entropy_32 | origin_64 | entropy_64 | max_planning_state_score | 0.307 | 0.283 | 0.178 | 0.303 | 0.020 | 0.124 |
| dream-7b-instruct-hf | plan_005 | entropy_32 | entropy_64 | entropy_64 | max_planning_state_score | 0.299 | 0.319 | 0.319 | 0.319 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_006 | entropy_32 | entropy_32 | entropy_32 | max_planning_state_score | 0.391 | 0.434 | 0.434 | 0.434 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_007 | entropy_32 | entropy_32 | entropy_64 | max_planning_state_score | 0.384 | 0.340 | 0.340 | 0.433 | 0.093 | 0.093 |
| dream-7b-instruct-hf | plan_008 | entropy_32 | origin_64 | origin_64 | max_planning_state_score | 0.273 | 0.138 | 0.243 | 0.243 | 0.104 | 0.000 |
| dream-7b-instruct-hf | sci_001 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | 0.314 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | sym_002 | entropy_32 | entropy_64 | entropy_32 | fixed_exact_answer_guard | 0.242 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | 0.040 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score | 0.332 | 0.399 | 0.473 | 0.399 | 0.000 | -0.074 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | random_32 | max_planning_state_score | 0.424 | 0.604 | 0.604 | 0.602 | -0.002 | -0.002 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score | 0.508 | 0.443 | 0.284 | 0.443 | 0.000 | 0.159 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score | 0.289 | 0.283 | 0.283 | 0.283 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score | 0.338 | 0.378 | 0.349 | 0.378 | 0.000 | 0.029 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score | 0.363 | 0.298 | 0.341 | 0.298 | 0.000 | -0.044 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score | 0.509 | 0.610 | 0.411 | 0.610 | 0.000 | 0.199 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score | 0.410 | 0.283 | 0.264 | 0.283 | 0.000 | 0.019 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | fixed_exact_answer_guard | 0.109 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
