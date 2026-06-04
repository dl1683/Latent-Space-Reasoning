# Diffusion Three-Arm Benchmark Report

Full model generations: `55`
Arm selections: `66`
Trajectory task delta vs fixed: `-0.034`
Trajectory task delta vs random: `-0.016`
Trajectory wins/ties/losses vs fixed: `5/15/2`
Trajectory wins/ties/losses vs random: `6/14/2`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 22 | 1.00 | 0.435 | 0.497 | 0.450 |
| random | 22 | 1.00 | 0.417 | 0.494 | 0.436 |
| trajectory_selected | 22 | 2.50 | 0.401 | 0.560 | 0.441 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Fixed Task | Random Task | Trajectory Task | Delta vs Fixed | Delta vs Random |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | math_001 | entropy_32 | entropy_32 | entropy_64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_001 | entropy_32 | entropy_32 | entropy_64 | 0.128 | 0.128 | 0.168 | 0.040 | 0.040 |
| dream-7b-instruct-hf | plan_002 | entropy_32 | entropy_32 | entropy_32 | 0.520 | 0.520 | 0.520 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_003 | entropy_32 | entropy_32 | entropy_64 | 0.106 | 0.106 | 0.359 | 0.252 | 0.252 |
| dream-7b-instruct-hf | plan_004 | entropy_32 | origin_64 | entropy_64 | 0.283 | 0.232 | 0.303 | 0.020 | 0.071 |
| dream-7b-instruct-hf | plan_005 | entropy_32 | entropy_64 | entropy_32 | 0.319 | 0.319 | 0.319 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_006 | entropy_32 | entropy_32 | entropy_64 | 0.434 | 0.434 | 0.434 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_007 | entropy_32 | entropy_32 | entropy_32 | 0.340 | 0.340 | 0.340 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_008 | entropy_32 | origin_64 | entropy_64 | 0.138 | 0.178 | 0.178 | 0.040 | 0.000 |
| dream-7b-instruct-hf | sci_001 | entropy_32 | entropy_32 | entropy_32 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | sym_002 | entropy_32 | entropy_64 | entropy_32 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | random_32 | 1.000 | 1.000 | 0.000 | -1.000 | -1.000 |
| llada-8b-instruct-hf | plan_001 | low_confidence_32 | random_32 | low_confidence_32 | 0.399 | 0.478 | 0.399 | 0.000 | -0.079 |
| llada-8b-instruct-hf | plan_002 | low_confidence_32 | low_confidence_32 | low_confidence_32 | 0.604 | 0.604 | 0.604 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_003 | low_confidence_32 | random_32 | low_confidence_32 | 0.443 | 0.242 | 0.443 | 0.000 | 0.201 |
| llada-8b-instruct-hf | plan_004 | low_confidence_32 | low_confidence_32 | low_confidence_32 | 0.283 | 0.283 | 0.283 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_005 | low_confidence_32 | random_32 | low_confidence_32 | 0.378 | 0.261 | 0.378 | 0.000 | 0.116 |
| llada-8b-instruct-hf | plan_006 | low_confidence_32 | random_32 | random_32 | 0.298 | 0.350 | 0.350 | 0.052 | 0.000 |
| llada-8b-instruct-hf | plan_007 | low_confidence_32 | random_32 | random_32 | 0.610 | 0.453 | 0.453 | -0.157 | 0.000 |
| llada-8b-instruct-hf | plan_008 | low_confidence_32 | random_32 | low_confidence_32 | 0.283 | 0.243 | 0.283 | 0.000 | 0.040 |
| llada-8b-instruct-hf | sci_001 | low_confidence_32 | random_32 | random_32 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | sym_002 | low_confidence_32 | low_confidence_32 | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
