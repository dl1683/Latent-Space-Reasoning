# Diffusion Three-Arm Benchmark Report

Full model generations: `55`
Arm selections: `66`
Exact-task trajectory policy: `fixed`
Trajectory task delta vs fixed: `0.020`
Trajectory task delta vs random: `0.071`
Trajectory wins/ties/losses vs fixed: `4/18/0`
Trajectory wins/ties/losses vs random: `8/13/1`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 22 | 1.00 | 0.436 | 0.497 | 0.451 |
| random | 22 | 1.00 | 0.385 | 0.428 | 0.396 |
| trajectory_selected | 22 | 2.50 | 0.456 | 0.535 | 0.475 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Reason | Fixed Task | Random Task | Trajectory Task | Delta vs Fixed | Delta vs Random |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | math_001 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_001 | entropy_32 | entropy_32 | entropy_64 | max_trajectory_score | 0.128 | 0.128 | 0.168 | 0.040 | 0.040 |
| dream-7b-instruct-hf | plan_002 | entropy_32 | entropy_32 | entropy_32 | max_trajectory_score | 0.542 | 0.542 | 0.542 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_003 | entropy_32 | entropy_32 | entropy_64 | max_trajectory_score | 0.106 | 0.106 | 0.359 | 0.252 | 0.252 |
| dream-7b-instruct-hf | plan_004 | entropy_32 | origin_64 | entropy_64 | max_trajectory_score | 0.283 | 0.106 | 0.303 | 0.020 | 0.196 |
| dream-7b-instruct-hf | plan_005 | entropy_32 | entropy_64 | entropy_32 | max_trajectory_score | 0.319 | 0.319 | 0.319 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_006 | entropy_32 | entropy_32 | entropy_64 | max_trajectory_score | 0.434 | 0.434 | 0.434 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_007 | entropy_32 | entropy_32 | entropy_32 | max_trajectory_score | 0.340 | 0.340 | 0.340 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_008 | entropy_32 | origin_64 | origin_64 | max_trajectory_score | 0.138 | 0.263 | 0.263 | 0.124 | 0.000 |
| dream-7b-instruct-hf | sci_001 | entropy_32 | entropy_32 | entropy_32 | fixed_exact_answer_guard | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | sym_002 | entropy_32 | entropy_64 | entropy_32 | fixed_exact_answer_guard | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | max_trajectory_score | 0.399 | 0.220 | 0.399 | 0.000 | 0.179 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_trajectory_score | 0.604 | 0.604 | 0.604 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | max_trajectory_score | 0.443 | 0.200 | 0.443 | 0.000 | 0.243 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_trajectory_score | 0.283 | 0.283 | 0.283 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | max_trajectory_score | 0.378 | 0.261 | 0.378 | 0.000 | 0.116 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | max_trajectory_score | 0.298 | 0.391 | 0.298 | 0.000 | -0.094 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | low_confidence_32 | max_trajectory_score | 0.610 | 0.106 | 0.610 | 0.000 | 0.504 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | max_trajectory_score | 0.283 | 0.158 | 0.283 | 0.000 | 0.124 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | fixed_exact_answer_guard | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | fixed_exact_answer_guard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
