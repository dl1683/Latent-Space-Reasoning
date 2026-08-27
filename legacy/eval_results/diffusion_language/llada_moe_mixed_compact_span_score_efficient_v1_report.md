# Diffusion Schedule-Selection Benchmark Report

Full model generations: `75`
Arm selections: `52`
Run ID: `diffusion-cc5266def1e987d4`
Content hash: `cc5266def1e987d4c20aa7811dacaa426c244a244d9fa52ccad1afa54bf0da0c`
Exact-task trajectory policy: `proposal_history`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `16`
Exact verifier revision: `True`
History mutability: `monotonic 53/75, changes 0, remasks 304, rewrites 68, mask increases 304`
History repairs included: `False`
Repair pack: `constraint_span`
Repair source policy: `non_revision_plus_gap_trajectory`
Adaptive source gate mode: `score_efficient`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `0.500`
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
Repair selector: `planning_quality_prompt_coverage_guarded`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.002`
Trajectory task delta vs random: `0.031`
Trajectory wins/ties/losses vs fixed: `1/10/0`
Trajectory wins/ties/losses vs random: `3/8/0`
Oracle generation budget/task: `6.82`
Oracle task score: `0.631`
Oracle headroom vs trajectory: `0.057`
Oracle wins/ties/losses vs trajectory: `6/5/0`
Selector regret vs trajectory: `0.057 over 6/11 improvable`
Exact proposal-history sources: `evolved:fallback=1, evolved:final=2, trajectory_selected:fallback=1, trajectory_selected:final=2`
Evolved task delta vs fixed: `0.023`
Evolved task delta vs random: `0.052`
Evolved task delta vs trajectory: `0.021`
Evolved wins/ties/losses vs fixed: `4/6/1`
Evolved wins/ties/losses vs random: `5/6/0`
Evolved wins/ties/losses vs trajectory: `4/6/1`
Oracle headroom vs evolved: `0.036`
Oracle wins/ties/losses vs evolved: `7/4/0`
Selector regret vs evolved: `0.036 over 7/11 improvable`
Repair arm coverage: `8/11` overall
Repair eligible coverage: `8/8`
Repair task delta vs fixed: `0.080`
Repair task delta vs random: `0.120`
Repair task delta vs trajectory: `0.078`
Repair task delta vs evolved: `0.049`
Repair generation budget delta vs evolved: `1.12`
Repair task delta per extra generation vs evolved: `0.043`
Repair wins/ties/losses vs evolved: `6/2/0`
Oracle headroom vs repair: `0.001`
Oracle wins/ties/losses vs repair: `1/7/0`
Selector regret vs repair: `0.001 over 1/8 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `8/11` overall, `8/8` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.412277 | 0.000000 | 0.040152 | - | - |
| random perturbation | repair-covered tasks | 0.372125 | -0.040152 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.492321 | 0.080045 | 0.120196 | 6/1/1 | 7/1/0 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 11 | 1.00 | 0.573 | 0.528 | 0.561 |
| random | 11 | 1.00 | 0.543 | 0.483 | 0.528 |
| trajectory_selected | 11 | 2.00 | 0.574 | 0.537 | 0.565 |
| evolved | 11 | 6.00 | 0.595 | 0.525 | 0.578 |
| repair_selected | 8 | 7.12 | 0.492 | 0.668 | 0.536 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | fixed | 1 | 1.00 | 1.000 | 0.228 | 0.807 |
| math | random | 1 | 1.00 | 1.000 | 0.228 | 0.807 |
| math | trajectory_selected | 1 | 2.00 | 1.000 | 0.228 | 0.807 |
| math | evolved | 1 | 6.00 | 1.000 | 0.254 | 0.813 |
| planning | fixed | 8 | 1.00 | 0.412 | 0.659 | 0.474 |
| planning | random | 8 | 1.00 | 0.372 | 0.600 | 0.429 |
| planning | trajectory_selected | 8 | 2.00 | 0.415 | 0.659 | 0.476 |
| planning | evolved | 8 | 6.00 | 0.444 | 0.635 | 0.491 |
| planning | repair_selected | 8 | 7.12 | 0.492 | 0.668 | 0.536 |
| science | fixed | 1 | 1.00 | 1.000 | 0.289 | 0.822 |
| science | random | 1 | 1.00 | 1.000 | 0.171 | 0.793 |
| science | trajectory_selected | 1 | 2.00 | 1.000 | 0.289 | 0.822 |
| science | evolved | 1 | 6.00 | 1.000 | 0.289 | 0.822 |
| symbolic | fixed | 1 | 1.00 | 1.000 | 0.016 | 0.754 |
| symbolic | random | 1 | 1.00 | 1.000 | 0.117 | 0.779 |
| symbolic | trajectory_selected | 1 | 2.00 | 1.000 | 0.117 | 0.779 |
| symbolic | evolved | 1 | 6.00 | 1.000 | 0.153 | 0.788 |

## Adaptive Source Gate

| Candidate | Task | Add Source | Reason | Primary | Trajectory | Gap Terms | Traj PQ | Quality Ceiling | Generated | Selected | Gap Term Sample |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| llada-moe-7b-a1b-instruct-hf | plan_001 | False | same_as_primary | low_confidence_32 | low_confidence_32 | 9 | 0.348 | 0.500 | 0 | 0 | gpu,jobs,overnight,gives,reliable,other,tests,reasoning |
| llada-moe-7b-a1b-instruct-hf | plan_002 | False | planning_quality_above_ceiling | evolved_low_confidence_48 | low_confidence_32 | 12 | 0.559 | 0.500 | 0 | 0 | pipeline,fails,once,every,thousand,noisy,hours,customer |
| llada-moe-7b-a1b-instruct-hf | plan_003 | False | same_as_primary | low_confidence_32 | low_confidence_32 | 6 | 0.324 | 0.500 | 0 | 0 | model,offline,triples,production,either,release |
| llada-moe-7b-a1b-instruct-hf | plan_004 | False | prompt_gap_below_floor | evolved_low_confidence_48 | low_confidence_32 | 2 | 0.278 | 0.500 | 0 | 0 | looks,used |
| llada-moe-7b-a1b-instruct-hf | plan_005 | False | same_as_primary | low_confidence_32 | low_confidence_32 | 10 | 0.299 | 0.500 | 0 | 0 | halfway,complete,disk,usage,spikes,writes,start,failing |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | add | evolved_low_confidence_48 | low_confidence_32 | 9 | 0.301 | 0.500 | 1 | 1 | customer,shows,wrong,needs,today,deeper,later,plan |
| llada-moe-7b-a1b-instruct-hf | plan_007 | False | same_as_primary,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 8 | 0.247 | 0.500 | 0 | 0 | gpu,diverges,free,debugging,cheapest,sequence,isolate,cause |
| llada-moe-7b-a1b-instruct-hf | plan_008 | False | not_low_confidence,planning_quality_below_floor | evolved_low_confidence_48 | random_32 | 12 | 0.223 | 0.500 | 0 | 0 | benchmark,improves,outputs,look,generic,evasive,whether,system |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_repair | 9 | 6 | evolved_low_confidence_48,low_confidence_32 | final | 33.6 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.037 | 0.041 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7/0/2 | 0.460 | 0.662 | 0.510 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-moe-7b-a1b-instruct-hf | plan_001 | True | low_confidence_32 | 6.583 | 0.599 | 1.000 | 0.000 | 0.333 | False | This way, at least one successful result (either the baseline or the intervention) can... |
| llada-moe-7b-a1b-instruct-hf | plan_002 | False | evolved_low_confidence_48 | 3.420 | 0.232 | 1.000 | 0.000 | 0.167 | False | Check the data source, transformation logic, and output validation to isolate the root... |
| llada-moe-7b-a1b-instruct-hf | plan_003 | True | low_confidence_32 | 2.775 | 0.000 | 1.000 | 0.000 | 0.533 | False | Decision rule: If accuracy improves by 10% or latency increases by <50%, ship; if accur... |
| llada-moe-7b-a1b-instruct-hf | plan_004 | True | evolved_low_confidence_48 | 1.699 | 0.029 | 1.000 | 0.000 | 0.412 | False | This plan should involve analyzing the baseline's methodology, comparing the results wi... |
| llada-moe-7b-a1b-instruct-hf | plan_005 | False | low_confidence_32 | 1.675 | 0.475 | 1.000 | 0.000 | 0.353 | False | This preserves the state of the training run and avoids corrupting the best checkpoint. |
| llada-moe-7b-a1b-instruct-hf | plan_005 | False | low_confidence_32 | 2.715 | 1.000 | 1.000 | 0.000 | 0.118 | False | Document the the state that the checkpoint was restored to for reproducibility. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | False | evolved_low_confidence_48 | 2.109 | 0.925 | 1.000 | 0.000 | 0.000 | False | Document the issue and schedule a quick meeting with the relevant team. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | False | evolved_low_confidence_48 | 2.821 | 0.841 | 1.000 | 0.000 | 0.062 | False | Prioritize the immediate fix to minimize customer impact. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 2.109 | 0.925 | 1.000 | 0.000 | 0.000 | False | Document the issue and schedule a quick meeting with the relevant team. |
| llada-moe-7b-a1b-instruct-hf | plan_006 | True | low_confidence_32 | 2.893 | 1.000 | 1.000 | 0.000 | 0.062 | False | Ensure the analysis is thorough and includesable to prevent future issues. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 1.423 | 0.936 | 1.000 | 0.000 | 0.167 | False | If the divergence occurs only with the change, the issue is with the optimizer. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 1.388 | 0.850 | 1.000 | 0.000 | 0.167 | False | If it occurs with both, the problem may lie in the model architecture or training loop. |
| llada-moe-7b-a1b-instruct-hf | plan_007 | True | low_confidence_32 | 2.092 | 0.782 | 1.000 | 0.000 | 0.250 | False | This experiment is sufficient to attribute the divergence to the optimizer change. |
| llada-moe-7b-a1b-instruct-hf | plan_008 | True | evolved_low_confidence_48 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Use diverse prompts to assess understanding, accuracy,, and depth. |
| llada-moe-7b-a1b-instruct-hf | plan_008 | True | evolved_low_confidence_48 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Evaluate creativity, original,, and relevance of. |
| llada-moe-7b-a1b-instruct-hf | plan_008 | True | evolved_low_confidence_48 | 2.871 | 0.968 | 1.000 | 0.000 | 0.000 | False | Check for consistency in reasoning, accuracy, and depth of. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-moe-7b-a1b-instruct-hf | math_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_low_confidence_32 |  | evolved_revision_low_confidence_32 | exact_answer_proposal_final_match | exact_answer_proposal_final_match |  |  |  |  | 0.228 | 0.254 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_001 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.425 | 0.425 | 0.431 | 0.036 | 0.465 | 0.465 | 0.465 | 0.465 | 0.528 | 0.063 | 0.528 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_002 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.448 | 0.479 | 0.561 | 0.000 | 0.689 | 0.580 | 0.689 | 0.684 | 0.684 | 0.000 | 0.689 | 0.005 |
| llada-moe-7b-a1b-instruct-hf | plan_003 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.418 | 0.418 | 0.487 | 0.103 | 0.422 | 0.422 | 0.422 | 0.422 | 0.538 | 0.116 | 0.538 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_004 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_prompt_coverage_guarded_score_repair_pool | evolved_low_confidence_48 | final |  | 0.466 | 0.491 | 0.299 | 0.021 | 0.338 | 0.157 | 0.338 | 0.358 | 0.359 | 0.001 | 0.359 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_005 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.334 | 0.334 | 0.299 | 0.000 | 0.421 | 0.421 | 0.421 | 0.421 | 0.421 | 0.000 | 0.421 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_006 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.366 | 0.410 | 0.487 | 0.114 | 0.391 | 0.341 | 0.391 | 0.433 | 0.584 | 0.151 | 0.584 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_007 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_revision_random_32 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_prompt_coverage_guarded_score_repair_pool | low_confidence_32 | final |  | 0.333 | 0.404 | 0.465 | 0.072 | 0.307 | 0.307 | 0.307 | 0.481 | 0.516 | 0.035 | 0.516 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | plan_008 | low_confidence_32 | random_32 | random_32 | evolved_low_confidence_48 | constraint_gap_span_repair | constraint_gap_span_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_planning_quality_prompt_coverage_guarded_score_repair_pool | evolved_low_confidence_48 | final |  | 0.274 | 0.279 | 0.287 | 0.021 | 0.264 | 0.283 | 0.283 | 0.286 | 0.307 | 0.021 | 0.307 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sci_001 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 |  | evolved_revision_low_confidence_32 | exact_answer_proposal_history_no_match_kept_fixed | exact_answer_proposal_history_no_match_kept_fixed |  |  |  |  | 0.289 | 0.289 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| llada-moe-7b-a1b-instruct-hf | sym_002 | low_confidence_32 | random_32 | random_32 | evolved_revision_random_32 |  | evolved_revision_random_32 | exact_answer_proposal_final_match | exact_answer_proposal_final_match |  |  |  |  | 0.117 | 0.153 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
