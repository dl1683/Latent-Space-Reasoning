# Diffusion Schedule-Selection Benchmark Report

Full model generations: `120`
Counterfactual probe generations: `24`
Arm selections: `168`
Run ID: `diffusion-8f074298c349ede9`
Content hash: `8f074298c349ede914e28638e79b2096123929311c931df988f21962854aa578`
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
Trajectory task delta vs fixed: `0.007`
Trajectory task delta vs random: `0.025`
Trajectory wins/ties/losses vs fixed: `5/40/3`
Trajectory wins/ties/losses vs random: `13/32/3`
Oracle generation budget/task: `2.50`
Oracle task score: `0.161`
Oracle headroom vs trajectory: `0.007`
Oracle wins/ties/losses vs trajectory: `6/42/0`
Selector regret vs trajectory: `0.007 over 6/48 improvable`
Repair arm coverage: `24/48` overall
Repair eligible coverage: `24/24`
Repair task delta vs fixed: `0.008`
Repair task delta vs random: `0.036`
Repair task delta vs trajectory: `0.000`
Repair task delta vs evolved: `0.000`
Repair generation budget delta vs evolved: `0.00`
Repair task delta per extra generation vs evolved: `0.000`
Repair wins/ties/losses vs evolved: `0/24/0`
Oracle headroom vs repair: `0.011`
Oracle wins/ties/losses vs repair: `5/19/0`
Selector regret vs repair: `0.011 over 5/24 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `24/48` overall, `24/24` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.275119 | 0.000000 | 0.028690 | - | - |
| random perturbation | repair-covered tasks | 0.246429 | -0.028690 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.282920 | 0.007801 | 0.036491 | 2/20/2 | 8/13/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 48 | 1.00 | 0.147 | 0.392 | 0.208 |
| random | 48 | 1.00 | 0.128 | 0.338 | 0.181 |
| trajectory_selected | 48 | 2.50 | 0.154 | 0.403 | 0.216 |
| repair_selected | 24 | 2.00 | 0.283 | 0.670 | 0.380 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 48 | 1.00 | 0.147 | 0.392 | 0.208 |
| planning | random | 48 | 1.00 | 0.128 | 0.338 | 0.181 |
| planning | trajectory_selected | 48 | 2.50 | 0.154 | 0.403 | 0.216 |
| planning | repair_selected | 24 | 2.00 | 0.283 | 0.670 | 0.380 |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_225 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.295 | 0.235 | 133 | True | 6 | 0.625 | True | True | 7.000 | 0.219 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_226 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.283 | 0.223 | 323 | True | 7 | 0.632 | True | True | 7.000 | 0.219 | 0.316 | 0.316 |
| llada-8b-instruct-hf | plan_227 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.281 | 0.201 | 265 | True | 4 | 0.733 | True | True | 7.000 | 0.219 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_228 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 294 | True | 5 | 0.615 | True | True | 7.000 | 0.219 | 0.385 | 0.385 |
| llada-8b-instruct-hf | plan_229 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.280 | 0.180 | 404 | True | 6 | 0.538 | True | True | 7.000 | 0.219 | 0.385 | 0.385 |
| llada-8b-instruct-hf | plan_230 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.316 | 0.276 | 333 | True | 4 | 0.769 | True | True | 7.000 | 0.219 | 0.538 | 0.538 |
| llada-8b-instruct-hf | plan_231 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.220 | 0.180 | 259 | True | 5 | 0.583 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_232 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.221 | 0.201 | 335 | True | 5 | 0.615 | True | True | 7.000 | 0.219 | 0.308 | 0.308 |
| llada-8b-instruct-hf | plan_233 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.325 | 0.265 | 341 | True | 2 | 0.833 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_234 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.197 | 0.117 | 114 | True | 4 | 0.692 | True | True | 20.000 | 0.625 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_235 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.458 | 0.358 | 355 | True | 3 | 0.769 | True | True | 7.000 | 0.219 | 0.462 | 0.462 |
| llada-8b-instruct-hf | plan_236 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 308 | True | 4 | 0.667 | True | True | 7.000 | 0.219 | 0.417 | 0.417 |
| llada-8b-instruct-hf | plan_237 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.303 | 0.223 | 306 | True | 1 | 0.917 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_238 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.261 | 0.201 | 350 | True | 2 | 0.867 | True | True | 7.000 | 0.219 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_239 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.281 | 0.201 | 310 | True | 2 | 0.800 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_240 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.309 | 0.269 | 409 | True | 1 | 0.917 | True | True | 7.000 | 0.219 | 0.500 | 0.500 |
| llada-8b-instruct-hf | plan_241 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.200 | 0.180 | 308 | True | 1 | 0.900 | True | True | 7.000 | 0.219 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_242 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.302 | 0.282 | 270 | True | 6 | 0.400 | True | True | 7.000 | 0.219 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_243 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.336 | 0.276 | 318 | True | 1 | 0.909 | True | True | 7.000 | 0.219 | 0.545 | 0.545 |
| llada-8b-instruct-hf | plan_244 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.241 | 0.201 | 342 | True | 2 | 0.818 | True | True | 7.000 | 0.219 | 0.364 | 0.364 |
| llada-8b-instruct-hf | plan_245 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.240 | 0.180 | 180 | True | 4 | 0.692 | True | True | 7.000 | 0.219 | 0.231 | 0.231 |
| llada-8b-instruct-hf | plan_246 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.394 | 0.276 | 364 | True | 6 | 0.455 | True | True | 7.000 | 0.219 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_247 | low_confidence_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.270 | 0.230 | 302 | True | 1 | 0.900 | True | True | 7.000 | 0.219 | 0.400 | 0.400 |
| llada-8b-instruct-hf | plan_248 | random_32 | False | counterfactual_probe_recorded_no_repair | True | measured_generation | 0.295 | 0.235 | 301 | True | 1 | 0.933 | True | True | 7.000 | 0.219 | 0.067 | 0.067 |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dream-7b-instruct-hf | plan_225 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.134 | 0.000 | 0.000 | 0.000 | 0.066 | 0.045 | 0.066 | 0.000 | 0.000 | 0.000 | 0.066 | 0.000 |
| dream-7b-instruct-hf | plan_226 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_227 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_228 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_229 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_230 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_231 | entropy_32 | origin_64 | origin_64 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.112 | 0.000 | 0.000 | 0.000 | 0.117 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_232 | entropy_32 | origin_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_233 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_234 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.042 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_235 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_236 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_237 | entropy_32 | origin_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.110 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.117 | 0.000 | 0.000 | 0.000 | 0.117 | 0.000 |
| dream-7b-instruct-hf | plan_238 | entropy_32 | entropy_64 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_239 | entropy_32 | entropy_64 | entropy_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_240 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_241 | entropy_32 | origin_64 | entropy_32 |  |  | entropy_32 | max_planning_state_score_base_pool |  |  |  |  |  | 0.153 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 | 0.180 | 0.000 | 0.000 | 0.000 | 0.180 | 0.000 |
| dream-7b-instruct-hf | plan_242 | entropy_32 | entropy_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_243 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_244 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_245 | entropy_32 | entropy_32 | entropy_32 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.128 | 0.000 | 0.000 | 0.000 | 0.045 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| dream-7b-instruct-hf | plan_246 | entropy_32 | origin_64 | entropy_32 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_247 | entropy_32 | entropy_32 | origin_64 |  |  | origin_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dream-7b-instruct-hf | plan_248 | entropy_32 | entropy_32 | origin_64 |  |  | entropy_64 | max_planning_state_score_base_pool |  |  |  |  |  | 0.114 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_225 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.380 | 0.000 | 0.235 | 0.000 | 0.045 | 0.295 | 0.295 | 0.000 | 0.295 | 0.000 | 0.295 | 0.000 |
| llada-8b-instruct-hf | plan_226 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.381 | 0.000 | 0.223 | 0.000 | 0.283 | 0.330 | 0.283 | 0.000 | 0.283 | 0.000 | 0.330 | 0.047 |
| llada-8b-instruct-hf | plan_227 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.379 | 0.000 | 0.201 | 0.000 | 0.281 | 0.240 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_228 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.359 | 0.000 | 0.180 | 0.000 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_229 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.340 | 0.000 | 0.180 | 0.000 | 0.280 | 0.280 | 0.280 | 0.000 | 0.280 | 0.000 | 0.280 | 0.000 |
| llada-8b-instruct-hf | plan_230 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.448 | 0.000 | 0.276 | 0.000 | 0.316 | 0.253 | 0.316 | 0.000 | 0.316 | 0.000 | 0.316 | 0.000 |
| llada-8b-instruct-hf | plan_231 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.313 | 0.000 | 0.180 | 0.000 | 0.220 | 0.137 | 0.220 | 0.000 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_232 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.364 | 0.000 | 0.201 | 0.000 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_233 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.466 | 0.000 | 0.265 | 0.000 | 0.325 | 0.105 | 0.325 | 0.000 | 0.325 | 0.000 | 0.325 | 0.000 |
| llada-8b-instruct-hf | plan_234 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.309 | 0.000 | 0.117 | 0.000 | 0.260 | 0.260 | 0.197 | 0.000 | 0.197 | 0.000 | 0.260 | 0.063 |
| llada-8b-instruct-hf | plan_235 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.451 | 0.000 | 0.358 | 0.000 | 0.458 | 0.458 | 0.458 | 0.000 | 0.458 | 0.000 | 0.458 | 0.000 |
| llada-8b-instruct-hf | plan_236 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.379 | 0.000 | 0.180 | 0.000 | 0.240 | 0.157 | 0.240 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_237 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.449 | 0.000 | 0.223 | 0.000 | 0.303 | 0.303 | 0.303 | 0.000 | 0.303 | 0.000 | 0.303 | 0.000 |
| llada-8b-instruct-hf | plan_238 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.431 | 0.000 | 0.201 | 0.000 | 0.261 | 0.045 | 0.261 | 0.000 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_239 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.416 | 0.000 | 0.201 | 0.000 | 0.281 | 0.281 | 0.281 | 0.000 | 0.281 | 0.000 | 0.281 | 0.000 |
| llada-8b-instruct-hf | plan_240 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.487 | 0.000 | 0.269 | 0.000 | 0.309 | 0.309 | 0.309 | 0.000 | 0.309 | 0.000 | 0.346 | 0.037 |
| llada-8b-instruct-hf | plan_241 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.420 | 0.000 | 0.180 | 0.000 | 0.200 | 0.200 | 0.200 | 0.000 | 0.200 | 0.000 | 0.284 | 0.084 |
| llada-8b-instruct-hf | plan_242 | low_confidence_32 | random_32 | random_32 |  | random_32 | random_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.341 | 0.000 | 0.282 | 0.000 | 0.259 | 0.302 | 0.302 | 0.000 | 0.302 | 0.000 | 0.302 | 0.000 |
| llada-8b-instruct-hf | plan_243 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.433 | 0.000 | 0.276 | 0.000 | 0.336 | 0.240 | 0.336 | 0.000 | 0.336 | 0.000 | 0.336 | 0.000 |
| llada-8b-instruct-hf | plan_244 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.373 | 0.000 | 0.201 | 0.000 | 0.241 | 0.241 | 0.241 | 0.000 | 0.241 | 0.000 | 0.241 | 0.000 |
| llada-8b-instruct-hf | plan_245 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.360 | 0.000 | 0.180 | 0.000 | 0.282 | 0.282 | 0.240 | 0.000 | 0.240 | 0.000 | 0.282 | 0.042 |
| llada-8b-instruct-hf | plan_246 | low_confidence_32 | low_confidence_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.329 | 0.000 | 0.276 | 0.000 | 0.394 | 0.394 | 0.394 | 0.000 | 0.394 | 0.000 | 0.394 | 0.000 |
| llada-8b-instruct-hf | plan_247 | low_confidence_32 | random_32 | low_confidence_32 |  | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.456 | 0.000 | 0.230 | 0.000 | 0.270 | 0.045 | 0.270 | 0.000 | 0.270 | 0.000 | 0.270 | 0.000 |
| llada-8b-instruct-hf | plan_248 | low_confidence_32 | low_confidence_32 | random_32 |  | random_32 | low_confidence_32 | max_planning_state_score_base_pool |  | repair_spend_gate_kept_evolved_counterfactual_micro_probe_v1 |  |  |  | 0.431 | 0.000 | 0.235 | 0.000 | 0.295 | 0.295 | 0.295 | 0.000 | 0.295 | 0.000 | 0.295 | 0.000 |
