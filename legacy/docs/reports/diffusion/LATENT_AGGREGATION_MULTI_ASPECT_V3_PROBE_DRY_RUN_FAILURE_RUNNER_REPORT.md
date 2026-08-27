# Diffusion Schedule-Selection Benchmark Report

Full model generations: `120`
Counterfactual probe generations: `0`
Arm selections: `144`
Run ID: `diffusion-a36648832682bbe3`
Content hash: `a36648832682bbe3fba85fdc040d35a63b7642ca5332589813597e42007c71db`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `inherit`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `False`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `False`
History mutability: `monotonic 120/120, changes 0, remasks 0, rewrites 0, mask increases 0`
History repairs included: `False`
Repair pack: `prefix`
Repair source policy: `evolved`
Adaptive source gate mode: `custom`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `none`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `counterfactual_micro_probe_v1`
Counterfactual probe mode: `all`
Counterfactual probe policy: `span_tomography_probe_v4`
Repair source-quality threshold: `0.500`
Repair source min chars: `320`
Repair source prompt-gap min: `0`
Repair source prompt-gap max: `999`
Repair source prompt coverage band: `0.000-1.000`
Repair value-proxy source-quality max: `0.310`
Repair cost penalty lambda: `0.180`
Repair transfer source-task min: `0.2954`
Repair phase budget: `custom`
Repair denoise skeleton max step: `none`
Phase-source threshold band: `target>=0.960, text>=0.960, chars>=0.950`
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
Trajectory task delta vs fixed: `0.008`
Trajectory task delta vs random: `0.018`
Trajectory wins/ties/losses vs fixed: `7/40/1`
Trajectory wins/ties/losses vs random: `14/32/2`
Oracle generation budget/task: `2.50`
Oracle task score: `0.176`
Oracle headroom vs trajectory: `0.002`
Oracle wins/ties/losses vs trajectory: `4/44/0`
Selector regret vs trajectory: `0.002 over 4/48 improvable`

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.166 | 0.415 | 0.228 |
| random | 48 | 1.00 | 0.156 | 0.379 | 0.211 |
| trajectory_selected | 48 | 2.50 | 0.174 | 0.439 | 0.240 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.166 | 0.415 | 0.228 |
| planning | random | 48 | 1.00 | 0.156 | 0.379 | 0.211 |
| planning | trajectory_selected | 48 | 2.50 | 0.174 | 0.439 | 0.240 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Reason | Selector | Fixed Task | Random Task | Trajectory Task | Delta vs Fixed | Delta vs Random | Oracle | Oracle Task | Oracle Delta vs Trajectory |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| dream-7b-instruct-hf | plan_201 | entropy_32 | entropy_32 | entropy_64 | max_planning_state_score_base_pool | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | entropy_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_202 | entropy_32 | origin_64 | origin_64 | max_planning_state_score_base_pool | 0.138 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | origin_64 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_203 | entropy_32 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | entropy_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_204 | entropy_32 | entropy_64 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | entropy_32 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_205 | entropy_32 | entropy_32 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | entropy_32 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_206 | entropy_32 | entropy_32 | origin_64 | max_planning_state_score_base_pool | 0.128 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | origin_64 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_207 | entropy_32 | origin_64 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | entropy_64 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_208 | entropy_32 | origin_64 | origin_64 | max_planning_state_score_base_pool | 0.111 | 0.000 | 0.117 | 0.117 | 0.117 | 0.000 | origin_64 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_209 | entropy_32 | origin_64 | origin_64 | max_planning_state_score_base_pool | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | origin_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_210 | entropy_32 | entropy_64 | entropy_32 | max_planning_state_score_base_pool | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | entropy_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_211 | entropy_32 | entropy_32 | origin_64 | max_planning_state_score_base_pool | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | origin_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_212 | entropy_32 | entropy_32 | origin_64 | max_planning_state_score_base_pool | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | entropy_32 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_213 | entropy_32 | entropy_32 | origin_64 | max_planning_state_score_base_pool | 0.128 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | origin_64 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_214 | entropy_32 | origin_64 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.000 | 0.045 | 0.000 | 0.045 | entropy_32 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_215 | entropy_32 | entropy_64 | entropy_32 | max_planning_state_score_base_pool | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | origin_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_216 | entropy_32 | origin_64 | origin_64 | max_planning_state_score_base_pool | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | origin_64 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_217 | entropy_32 | entropy_32 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | entropy_32 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_218 | entropy_32 | origin_64 | origin_64 | max_planning_state_score_base_pool | 0.140 | 0.117 | 0.045 | 0.045 | -0.072 | 0.000 | entropy_32 | 0.117 | 0.072 |
| dream-7b-instruct-hf | plan_219 | entropy_32 | entropy_64 | entropy_64 | max_planning_state_score_base_pool | 0.128 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | entropy_64 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_220 | entropy_32 | entropy_64 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | entropy_32 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_221 | entropy_32 | entropy_64 | entropy_32 | max_planning_state_score_base_pool | 0.128 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | entropy_32 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_222 | entropy_32 | entropy_64 | entropy_32 | max_planning_state_score_base_pool | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | entropy_32 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_223 | entropy_32 | entropy_32 | origin_64 | max_planning_state_score_base_pool | 0.111 | 0.000 | 0.000 | 0.117 | 0.117 | 0.117 | origin_64 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_224 | entropy_32 | entropy_64 | origin_64 | max_planning_state_score_base_pool | 0.128 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | origin_64 | 0.045 | 0.000 |
| llada-8b-instruct-hf | plan_201 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.393 | 0.323 | 0.323 | 0.323 | 0.000 | 0.000 | low_confidence_32 | 0.323 | 0.000 |
| llada-8b-instruct-hf | plan_202 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.418 | 0.321 | 0.321 | 0.321 | 0.000 | 0.000 | random_32 | 0.341 | 0.020 |
| llada-8b-instruct-hf | plan_203 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.432 | 0.280 | 0.280 | 0.280 | 0.000 | 0.000 | low_confidence_32 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_204 | low_confidence_32 | random_32 | random_32 | max_planning_state_score_base_pool | 0.356 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | random_32 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_205 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.333 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | low_confidence_32 | 0.260 | 0.000 |
| llada-8b-instruct-hf | plan_206 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.506 | 0.486 | 0.420 | 0.486 | 0.000 | 0.066 | low_confidence_32 | 0.486 | 0.000 |
| llada-8b-instruct-hf | plan_207 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.339 | 0.378 | 0.378 | 0.378 | 0.000 | 0.000 | low_confidence_32 | 0.378 | 0.000 |
| llada-8b-instruct-hf | plan_208 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.436 | 0.311 | 0.311 | 0.311 | 0.000 | 0.000 | low_confidence_32 | 0.311 | 0.000 |
| llada-8b-instruct-hf | plan_209 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.375 | 0.281 | 0.197 | 0.281 | 0.000 | 0.084 | low_confidence_32 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_210 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.380 | 0.315 | 0.315 | 0.315 | 0.000 | 0.000 | low_confidence_32 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_211 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.362 | 0.339 | 0.339 | 0.339 | 0.000 | 0.000 | low_confidence_32 | 0.339 | 0.000 |
| llada-8b-instruct-hf | plan_212 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.384 | 0.281 | 0.260 | 0.281 | 0.000 | 0.021 | low_confidence_32 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_213 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.376 | 0.274 | 0.281 | 0.274 | 0.000 | -0.008 | random_32 | 0.281 | 0.008 |
| llada-8b-instruct-hf | plan_214 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.320 | 0.299 | 0.214 | 0.299 | 0.000 | 0.084 | low_confidence_32 | 0.299 | 0.000 |
| llada-8b-instruct-hf | plan_215 | low_confidence_32 | random_32 | random_32 | max_planning_state_score_base_pool | 0.339 | 0.335 | 0.335 | 0.335 | 0.000 | 0.000 | low_confidence_32 | 0.335 | 0.000 |
| llada-8b-instruct-hf | plan_216 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.497 | 0.391 | 0.324 | 0.391 | 0.000 | 0.068 | low_confidence_32 | 0.391 | 0.000 |
| llada-8b-instruct-hf | plan_217 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.402 | 0.281 | 0.277 | 0.281 | 0.000 | 0.004 | low_confidence_32 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_218 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.325 | 0.281 | 0.282 | 0.281 | 0.000 | -0.001 | random_32 | 0.282 | 0.001 |
| llada-8b-instruct-hf | plan_219 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.355 | 0.303 | 0.303 | 0.303 | 0.000 | 0.000 | low_confidence_32 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_220 | low_confidence_32 | random_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.350 | 0.241 | 0.065 | 0.241 | 0.000 | 0.176 | low_confidence_32 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_221 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.508 | 0.318 | 0.318 | 0.318 | 0.000 | 0.000 | low_confidence_32 | 0.318 | 0.000 |
| llada-8b-instruct-hf | plan_222 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.405 | 0.399 | 0.399 | 0.399 | 0.000 | 0.000 | low_confidence_32 | 0.399 | 0.000 |
| llada-8b-instruct-hf | plan_223 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.434 | 0.315 | 0.315 | 0.315 | 0.000 | 0.000 | low_confidence_32 | 0.315 | 0.000 |
| llada-8b-instruct-hf | plan_224 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | 0.392 | 0.260 | 0.260 | 0.260 | 0.000 | 0.000 | low_confidence_32 | 0.260 | 0.000 |
