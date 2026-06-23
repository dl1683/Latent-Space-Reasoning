# Diffusion Schedule-Selection Benchmark Report

Full model generations: `168`
Counterfactual probe generations: `0`
Arm selections: `120`
Run ID: `diffusion-dbc014ca2a4827dd`
Content hash: `dbc014ca2a4827dd098147277a38e758229ff8a362869fe1b55557c5767d127b`
Exact-task trajectory policy: `fixed`
Trajectory selector: `planning_state`
Evolved selector: `planning_quality_fallback`
Evolved quality margin: `0.010`
Evolved selector tolerance: `0.015`
Evolved promotion margin: `0.015`
Revision promotion margin: `0.050`
Revision schedules included: `True`
Revision remask fraction: `0.250`
Revision steps: `8`
Exact verifier revision: `False`
History mutability: `monotonic 120/168, changes 0, remasks 768, rewrites 188, mask increases 768`
History repairs included: `False`
Repair pack: `constraint_span_history_contrast`
Repair source policy: `non_revision_plus_gap_trajectory`
Adaptive source gate mode: `score_efficient`
Adaptive source gap min terms: `6`
Adaptive source quality floor: `0.250`
Adaptive source quality ceiling: `0.500`
History repair fractions: `0.25`
History visible repair included: `False`
Repair spend trigger: `always`
Counterfactual probe mode: `triage`
Counterfactual probe policy: `deterministic_missing_constraint_probe_v1`
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
Repair selector: `generated_repair_value_v1`
Repair promotion margin: `0.020`
Trajectory task delta vs fixed: `0.007`
Trajectory task delta vs random: `0.052`
Trajectory wins/ties/losses vs fixed: `3/21/0`
Trajectory wins/ties/losses vs random: `10/13/1`
Oracle generation budget/task: `7.00`
Oracle task score: `0.303`
Oracle headroom vs trajectory: `0.021`
Oracle wins/ties/losses vs trajectory: `8/16/0`
Selector regret vs trajectory: `0.021 over 8/24 improvable`
Evolved task delta vs fixed: `0.006`
Evolved task delta vs random: `0.051`
Evolved task delta vs trajectory: `-0.001`
Evolved wins/ties/losses vs fixed: `5/17/2`
Evolved wins/ties/losses vs random: `10/11/3`
Evolved wins/ties/losses vs trajectory: `2/19/3`
Oracle headroom vs evolved: `0.022`
Oracle wins/ties/losses vs evolved: `9/15/0`
Selector regret vs evolved: `0.022 over 9/24 improvable`
Repair arm coverage: `24/24` overall
Repair eligible coverage: `24/24`
Repair task delta vs fixed: `0.017`
Repair task delta vs random: `0.062`
Repair task delta vs trajectory: `0.010`
Repair task delta vs evolved: `0.011`
Repair generation budget delta vs evolved: `1.00`
Repair task delta per extra generation vs evolved: `0.011`
Repair wins/ties/losses vs evolved: `3/21/0`
Oracle headroom vs repair: `0.011`
Oracle wins/ties/losses vs repair: `7/17/0`
Selector regret vs repair: `0.011 over 7/24 improvable`

## Lean Three-Arm Headline

This is the public-facing comparison: fixed baseline, random perturbation, and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics.

Repair coverage: `24/24` overall, `24/24` eligible.

| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixed baseline | repair-covered tasks | 0.275045 | 0.000000 | 0.044765 | - | - |
| random perturbation | repair-covered tasks | 0.230280 | -0.044765 | 0.000000 | - | - |
| selected latent repair | repair-covered tasks | 0.292167 | 0.017122 | 0.061887 | 7/15/2 | 10/11/3 |

## Arm Summary

| Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 24 | 1.00 | 0.275 | 0.642 | 0.367 |
| random | 24 | 1.00 | 0.230 | 0.533 | 0.306 |
| trajectory_selected | 24 | 2.00 | 0.282 | 0.642 | 0.372 |
| evolved | 24 | 6.00 | 0.281 | 0.638 | 0.371 |
| repair_selected | 24 | 7.00 | 0.292 | 0.641 | 0.379 |

## Family Arm Summary

| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planning | fixed | 24 | 1.00 | 0.275 | 0.642 | 0.367 |
| planning | random | 24 | 1.00 | 0.230 | 0.533 | 0.306 |
| planning | trajectory_selected | 24 | 2.00 | 0.282 | 0.642 | 0.372 |
| planning | evolved | 24 | 6.00 | 0.281 | 0.638 | 0.371 |
| planning | repair_selected | 24 | 7.00 | 0.292 | 0.641 | 0.379 |

## Adaptive Source Gate

| Candidate | Task | Add Source | Reason | Primary | Trajectory | Gap Terms | Traj PQ | Quality Ceiling | Generated | Selected | Gap Term Sample |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| llada-8b-instruct-hf | plan_346 | False | not_low_confidence,prompt_gap_below_floor,planning_quality_below_floor | evolved_random_48 | random_32 | 4 | 0.244 | 0.500 | 0 | 0 | failure,shows,zero,many |
| llada-8b-instruct-hf | plan_347 | False | same_as_primary,not_low_confidence,prompt_gap_below_floor,planning_quality_below_floor | random_32 | random_32 | 4 | 0.223 | 0.500 | 0 | 0 | answer,improves,assigning,become |
| llada-8b-instruct-hf | plan_348 | False | same_as_primary | low_confidence_32 | low_confidence_32 | 6 | 0.265 | 0.500 | 0 | 0 | candidate,answer,adds,generic,step,ordering |
| llada-8b-instruct-hf | plan_349 | False | prompt_gap_below_floor | evolved_random_48 | low_confidence_32 | 2 | 0.281 | 0.500 | 0 | 0 | lacks,criteria |
| llada-8b-instruct-hf | plan_350 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 5 | 0.244 | 0.500 | 0 | 0 | evidence-or-measurement,names,evidence,measurement,aspect |
| llada-8b-instruct-hf | plan_351 | False | same_as_primary,prompt_gap_below_floor | low_confidence_32 | low_confidence_32 | 3 | 0.272 | 0.500 | 0 | 0 | strong,anchor,overgeneralizes |
| llada-8b-instruct-hf | plan_353 | False | prompt_gap_below_floor,planning_quality_below_floor | evolved_low_confidence_48 | low_confidence_32 | 1 | 0.223 | 0.500 | 0 | 0 | reward |
| llada-8b-instruct-hf | plan_354 | False | same_as_primary,prompt_gap_below_floor | low_confidence_32 | low_confidence_32 | 2 | 0.260 | 0.500 | 0 | 0 | contributes,many |
| llada-8b-instruct-hf | plan_355 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 5 | 0.230 | 0.500 | 0 | 0 | small-n,revive,small,work,honestly |
| llada-8b-instruct-hf | plan_359 | False | same_as_primary,prompt_gap_below_floor | low_confidence_32 | low_confidence_32 | 2 | 0.324 | 0.500 | 0 | 0 | old-ontology,plan |
| llada-8b-instruct-hf | plan_360 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 4 | 0.180 | 0.500 | 0 | 0 | adds,plausible,present,prompt |
| llada-8b-instruct-hf | plan_361 | False | same_as_primary,prompt_gap_below_floor | low_confidence_32 | low_confidence_32 | 3 | 0.290 | 0.500 | 0 | 0 | conflicts,plan,resolution |
| llada-8b-instruct-hf | plan_366 | False | prompt_gap_below_floor,planning_quality_below_floor | evolved_low_confidence_48 | low_confidence_32 | 4 | 0.180 | 0.500 | 0 | 0 | must,making,bloated,plan |
| llada-8b-instruct-hf | plan_373 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 3 | 0.201 | 0.500 | 0 | 0 | supplies,contradicts,plan |
| llada-8b-instruct-hf | plan_375 | False | prompt_gap_below_floor | evolved_low_confidence_48 | low_confidence_32 | 0 | 0.294 | 0.500 | 0 | 0 |  |
| llada-8b-instruct-hf | plan_376 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 1 | 0.180 | 0.500 | 0 | 0 | cited |
| llada-8b-instruct-hf | plan_377 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 3 | 0.235 | 0.500 | 0 | 0 | might,accidentally,labels |
| llada-8b-instruct-hf | plan_378 | False | prompt_gap_below_floor | evolved_random_48 | low_confidence_32 | 4 | 0.282 | 0.500 | 0 | 0 | reproducible,plan,command,checklist |
| llada-8b-instruct-hf | plan_380 | False | prompt_gap_below_floor,planning_quality_below_floor | evolved_low_confidence_48 | low_confidence_32 | 0 | 0.201 | 0.500 | 0 | 0 |  |
| llada-8b-instruct-hf | plan_381 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 4 | 0.201 | 0.500 | 0 | 0 | high,unique,coverage,introduces |
| llada-8b-instruct-hf | plan_384 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 2 | 0.223 | 0.500 | 0 | 0 | make,plan |
| llada-8b-instruct-hf | plan_385 | False | same_as_primary,prompt_gap_below_floor,planning_quality_below_floor | low_confidence_32 | low_confidence_32 | 3 | 0.180 | 0.500 | 0 | 0 | uses,plan,format |
| llada-8b-instruct-hf | plan_386 | False | same_as_primary,not_low_confidence,prompt_gap_below_floor,planning_quality_below_floor | random_32 | random_32 | 3 | 0.117 | 0.500 | 0 | 0 | task,many,plan |
| llada-8b-instruct-hf | plan_389 | False | same_as_primary | low_confidence_32 | low_confidence_32 | 8 | 0.326 | 0.500 | 0 | 0 | evidence-backed,future,paper,section,needs,plan,backed,narrative |

## Repair Spend Gate Diagnostics

| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_346 | evolved_random_48 | True | trigger_always | False |  | 0.241 | 0.201 | 289 | True | 3 | 0.727 | False | True | 4.000 | 0.083 | 0.091 | 0.091 |
| llada-8b-instruct-hf | plan_347 | random_32 | True | trigger_always | False |  | 0.263 | 0.223 | 278 | True | 4 | 0.667 | False | True | 4.000 | 0.125 | 0.000 | 0.000 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | True | trigger_always | False |  | 0.285 | 0.265 | 333 | True | 6 | 0.400 | False | True | 3.000 | 0.094 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_349 | evolved_random_48 | True | trigger_always | False |  | 0.458 | 0.340 | 315 | True | 1 | 0.900 | False | True | 6.000 | 0.125 | 0.100 | 0.100 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | True | trigger_always | False |  | 0.304 | 0.244 | 342 | True | 5 | 0.556 | False | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | True | trigger_always | False |  | 0.375 | 0.272 | 331 | True | 3 | 0.667 | False | True | 3.000 | 0.094 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_353 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.261 | 0.201 | 287 | True | 0 | 1.000 | False | True | 4.000 | 0.083 | 0.273 | 0.273 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | True | trigger_always | False |  | 0.340 | 0.260 | 322 | True | 2 | 0.778 | False | True | 3.000 | 0.094 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | True | trigger_always | False |  | 0.250 | 0.230 | 296 | True | 5 | 0.600 | False | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | True | trigger_always | False |  | 0.364 | 0.324 | 275 | True | 2 | 0.889 | False | True | 3.000 | 0.094 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | True | trigger_always | False |  | 0.200 | 0.180 | 332 | True | 4 | 0.600 | False | True | 4.000 | 0.125 | 0.300 | 0.300 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | True | trigger_always | False |  | 0.310 | 0.290 | 341 | True | 3 | 0.667 | False | True | 3.000 | 0.094 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_366 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.220 | 0.180 | 359 | True | 4 | 0.636 | False | True | 3.000 | 0.062 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | True | trigger_always | False |  | 0.221 | 0.201 | 332 | True | 3 | 0.727 | False | True | 3.000 | 0.094 | 0.182 | 0.182 |
| llada-8b-instruct-hf | plan_375 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.334 | 0.294 | 374 | True | 0 | 1.000 | False | True | 3.000 | 0.062 | 0.444 | 0.444 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | True | trigger_always | False |  | 0.240 | 0.180 | 294 | True | 1 | 0.857 | False | True | 5.000 | 0.156 | 0.286 | 0.286 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | True | trigger_always | False |  | 0.275 | 0.235 | 296 | True | 3 | 0.625 | False | True | 5.000 | 0.156 | 0.375 | 0.375 |
| llada-8b-instruct-hf | plan_378 | evolved_random_48 | True | trigger_always | False |  | 0.241 | 0.201 | 200 | True | 2 | 0.500 | False | True | 3.000 | 0.062 | 0.250 | 0.250 |
| llada-8b-instruct-hf | plan_380 | evolved_low_confidence_48 | True | trigger_always | False |  | 0.261 | 0.201 | 332 | True | 0 | 1.000 | False | True | 4.000 | 0.083 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | True | trigger_always | False |  | 0.261 | 0.201 | 371 | True | 4 | 0.556 | False | True | 4.000 | 0.125 | 0.222 | 0.222 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | True | trigger_always | False |  | 0.283 | 0.223 | 303 | True | 2 | 0.800 | False | True | 3.000 | 0.094 | 0.200 | 0.200 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | True | trigger_always | False |  | 0.220 | 0.180 | 321 | True | 3 | 0.667 | False | True | 4.000 | 0.125 | 0.333 | 0.333 |
| llada-8b-instruct-hf | plan_386 | random_32 | True | trigger_always | False |  | 0.137 | 0.117 | 98 | True | 3 | 0.667 | False | True | 5.000 | 0.156 | 0.111 | 0.111 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | True | trigger_always | False |  | 0.404 | 0.326 | 238 | True | 8 | 0.300 | False | True | 5.000 | 0.156 | 0.200 | 0.200 |

## Repair Candidate Diagnostics

| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| constraint_gap_span_history_contrast_repair | 24 | 3 | evolved_low_confidence_48,evolved_random_48,low_confidence_32,random_32 | final | 29.6 | 0.958 | 0.042 | 0.000 | 0.088 | 0.088 | 0.006 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4/17/3 | 0.291 | 0.661 | 0.383 |

## Planning Span Target Diagnostics

| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llada-8b-instruct-hf | plan_346 | True | evolved_random_48 | 1.707 | 0.554 | 1.000 | 0.000 | 0.455 | False | Identify tasks with zero positive ontology deltas. |
| llada-8b-instruct-hf | plan_346 | True | evolved_random_48 | 1.902 | 0.865 | 1.000 | 0.000 | 0.182 | False | Develop new ontology concepts for these tasks. |
| llada-8b-instruct-hf | plan_346 | True | evolved_random_48 | 2.591 | 0.790 | 1.000 | 0.000 | 0.273 | False | Analyze results and make adjustments to increase positive ontology deltas. |
| llada-8b-instruct-hf | plan_347 | False | random_32 | 2.046 | 0.673 | 1.000 | 0.000 | 0.250 | False | This allows for better tracking, collaboration, and clear accountability, ultimately im... |
| llada-8b-instruct-hf | plan_348 | False | low_confidence_32 | 2.138 | 1.000 | 1.000 | 0.000 | 0.000 | False | Assign each phase to a team member or task. |
| llada-8b-instruct-hf | plan_348 | False | low_confidence_32 | 2.100 | 0.928 | 1.000 | 0.000 | 0.000 | False | Set milestones and checkpoints to track progress and ensure completion. |
| llada-8b-instruct-hf | plan_348 | False | low_confidence_32 | 2.829 | 0.936 | 1.000 | 0.000 | 0.100 | False | Regularly review the adjust the timeline as needed to ensure the project on track and w... |
| llada-8b-instruct-hf | plan_349 | False | evolved_random_48 | 1.663 | 0.357 | 1.000 | 0.000 | 0.200 | False | Analyze the existing final plan. |
| llada-8b-instruct-hf | plan_349 | False | evolved_random_48 | 1.342 | 0.790 | 1.000 | 0.000 | 0.200 | False | Identify gaps in rollback criteria. |
| llada-8b-instruct-hf | plan_349 | False | evolved_random_48 | 1.903 | 0.865 | 1.000 | 0.000 | 0.200 | False | Implement a safe search algorithm to find potential complements. |
| llada-8b-instruct-hf | plan_350 | False | low_confidence_32 | 3.189 | 0.928 | 1.000 | 0.000 | 0.000 | False | Use tools like time tracking, feedback forms, and performance dashboards to gather data. |
| llada-8b-instruct-hf | plan_350 | False | low_confidence_32 | 2.184 | 1.000 | 1.000 | 0.000 | 0.333 | False | Regularly review these metrics to assess the candidate's improvement and adjust adjustm... |
| llada-8b-instruct-hf | plan_351 | False | low_confidence_32 | 2.115 | 0.981 | 1.000 | 0.000 | 0.111 | False | Then, the scope should be expanded to include the necessary elements. |
| llada-8b-instruct-hf | plan_351 | False | low_confidence_32 | 2.070 | 0.917 | 1.000 | 0.000 | 0.111 | False | Next, the boundaries should be set to ensure the scope does not exceed acceptable limits. |
| llada-8b-instruct-hf | plan_351 | False | low_confidence_32 | 2.177 | 0.981 | 1.000 | 0.000 | 0.333 | False | Finally, the policy should be developed to complement the defined scope and boundaries. |
| llada-8b-instruct-hf | plan_353 | False | evolved_low_confidence_48 | 1.343 | 0.910 | 0.818 | 0.000 | 0.182 | False | Normalize the length of the answers to a standard format. |
| llada-8b-instruct-hf | plan_353 | False | evolved_low_confidence_48 | 1.936 | 0.910 | 0.545 | 0.000 | 0.455 | False | Review the normalized answers to detect and audit false-positive results. |
| llada-8b-instruct-hf | plan_354 | False | low_confidence_32 | 1.422 | 0.925 | 1.000 | 0.000 | 0.222 | False | analyze the overlap between aspects, classify the overlap, |
| llada-8b-instruct-hf | plan_354 | False | low_confidence_32 | 2.011 | 0.621 | 1.000 | 0.000 | 0.222 | False | prioritize the overlap, resolve the overlap, test the audit, review and validate the au... |
| llada-8b-instruct-hf | plan_355 | False | low_confidence_32 | 2.065 | 0.865 | 1.000 | 0.000 | 0.100 | False | Implement the perturbation in the model architecture and training. |
| llada-8b-instruct-hf | plan_355 | False | low_confidence_32 | 2.077 | 0.865 | 1.000 | 0.000 | 0.000 | False | Evaluate the impact on performance and robustness. |
| llada-8b-instruct-hf | plan_355 | False | low_confidence_32 | 2.788 | 0.790 | 1.000 | 0.000 | 0.000 | False | Document the process and results for reproducibility. |
| llada-8b-instruct-hf | plan_359 | False | low_confidence_32 | 2.601 | 0.955 | 1.000 | 0.000 | 0.111 | False | Task ID 2. |
| llada-8b-instruct-hf | plan_359 | False | low_confidence_32 | 1.786 | 0.637 | 1.000 | 0.000 | 0.222 | False | Old Ontology complement (if applicable) 3. |
| llada-8b-instruct-hf | plan_359 | False | low_confidence_32 | 1.835 | 0.731 | 1.000 | 0.000 | 0.222 | False | Owner complement (if applicable) 4. |
| llada-8b-instruct-hf | plan_360 | True | low_confidence_32 | 1.452 | 1.000 | 1.000 | 0.000 | 0.200 | False | Please provide the list of concerns, including their potential impact, relevant stakeho... |
| llada-8b-instruct-hf | plan_360 | True | low_confidence_32 | 2.202 | 1.000 | 1.000 | 0.000 | 0.200 | False | Once I have this information, I can create a detailed plan for the audit. |
| llada-8b-instruct-hf | plan_361 | False | low_confidence_32 | 1.333 | 0.873 | 1.000 | 0.000 | 0.556 | False | If the temporal-order complement is before the anchor mitigation order, prioritize the... |
| llada-8b-instruct-hf | plan_361 | False | low_confidence_32 | 2.083 | 0.873 | 1.000 | 0.000 | 0.556 | False | If the anchor mitigation order is after the temporal-order complement, prioritize the t... |
| llada-8b-instruct-hf | plan_366 | True | evolved_low_confidence_48 | 2.147 | 0.905 | 1.000 | 0.000 | 0.273 | False | This rule should prioritize the inclusion of new aspects that will significantly enhanc... |
| llada-8b-instruct-hf | plan_373 | False | low_confidence_32 | 1.419 | 1.000 | 1.000 | 0.000 | 0.455 | False | use the candidate's concrete owner as the escalation path, |
| llada-8b-instruct-hf | plan_373 | False | low_confidence_32 | 1.435 | 1.000 | 1.000 | 0.000 | 0.364 | False | use the candidate's concrete owner as escalation, |
| llada-8b-instruct-hf | plan_373 | False | low_confidence_32 | 2.120 | 0.892 | 1.000 | 0.000 | 0.455 | False | use the candidate's concrete owner as the escalation path. |
| llada-8b-instruct-hf | plan_375 | False | evolved_low_confidence_48 | 2.561 | 0.925 | 0.778 | 0.000 | 0.222 | False | This can be done by using various tools and techniques to analyze and integrate the inf... |
| llada-8b-instruct-hf | plan_376 | False | low_confidence_32 | 2.118 | 0.856 | 1.000 | 0.000 | 0.286 | False | This involves referencing the relevant data, findings, or results from the v6 version t... |
| llada-8b-instruct-hf | plan_377 | False | low_confidence_32 | 1.413 | 0.981 | 1.000 | 0.000 | 0.375 | False | Train the aspect extractor on the labeled data and then test it on the unlabeled data. |
| llada-8b-instruct-hf | plan_377 | False | low_confidence_32 | 2.048 | 0.786 | 1.000 | 0.000 | 0.500 | False | Measure the performance on the unlabeled data to determine if there is any label leakag... |
| llada-8b-instruct-hf | plan_378 | False | evolved_random_48 | 1.975 | 0.638 | 1.000 | 0.000 | 0.000 | False | Install necessary dependencies. |
| llada-8b-instruct-hf | plan_378 | False | evolved_random_48 | 1.975 | 0.638 | 1.000 | 0.000 | 0.000 | False | Set up environment. |
| llada-8b-instruct-hf | plan_378 | False | evolved_random_48 | 1.975 | 0.638 | 1.000 | 0.000 | 0.000 | False | Configure CUDA environment. |
| llada-8b-instruct-hf | plan_380 | False | evolved_low_confidence_48 | 1.889 | 1.000 | 0.444 | 0.000 | 0.556 | False | This would allow you to focus solely on the measurement value of the aspect without con... |
| llada-8b-instruct-hf | plan_381 | False | low_confidence_32 | 1.890 | 0.378 | 1.000 | 0.000 | 0.333 | False | If the contradictions are significant, consider excluding the source family from the an... |
| llada-8b-instruct-hf | plan_384 | False | low_confidence_32 | 1.459 | 1.000 | 1.000 | 0.000 | 0.200 | False | This this provides a measure of the number of tasks that have been newly covered. |
| llada-8b-instruct-hf | plan_384 | False | low_confidence_32 | 2.170 | 1.000 | 1.000 | 0.000 | 0.400 | False | Answer: Count the number of old no-complement tasks that are now covered by the new ont... |
| llada-8b-instruct-hf | plan_385 | False | low_confidence_32 | 2.154 | 0.925 | 1.000 | 0.000 | 0.222 | False | Additionally, you would need to define the proof required to verify the correctness of... |
| llada-8b-instruct-hf | plan_386 | False | random_32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | Aggregation should abstain when the number of weaker candidates exceeds that of the str... |
| llada-8b-instruct-hf | plan_389 | False | low_confidence_32 | 2.538 | 0.790 | 1.000 | 0.000 | 0.000 | False | Identify the key features of v6. |
| llada-8b-instruct-hf | plan_389 | False | low_confidence_32 | 2.538 | 0.790 | 1.000 | 0.000 | 0.000 | False | Analyze the performance of v6. |
| llada-8b-instruct-hf | plan_389 | False | low_confidence_32 | 2.774 | 0.770 | 1.000 | 0.000 | 0.000 | False | Discuss the factors that contributed to the failure of v6. |

## Task Comparisons

| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | Traj Selector | Evolved Selector | Repair Selector | Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct-hf | plan_346 | low_confidence_32 | low_confidence_32 | random_32 | evolved_random_48 | constraint_gap_span_history_contrast_repair | random_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_generated_repair_value_v1_score_repair_pool | evolved_random_48 | final |  | 0.379 | 0.402 | 0.049 | 0.049 | 0.221 | 0.221 | 0.304 | 0.241 | 0.259 | 0.018 | 0.304 | 0.045 |
| llada-8b-instruct-hf | plan_347 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.383 | 0.383 | 0.000 | 0.000 | 0.241 | 0.241 | 0.263 | 0.263 | 0.263 | 0.000 | 0.263 | 0.000 |
| llada-8b-instruct-hf | plan_348 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.321 | 0.321 | 0.000 | 0.000 | 0.285 | 0.045 | 0.285 | 0.285 | 0.285 | 0.000 | 0.285 | 0.000 |
| llada-8b-instruct-hf | plan_349 | low_confidence_32 | random_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_random_48 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.471 | 0.508 | 0.000 | 0.000 | 0.379 | 0.375 | 0.379 | 0.458 | 0.458 | 0.000 | 0.458 | 0.000 |
| llada-8b-instruct-hf | plan_350 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.385 | 0.385 | 0.000 | 0.000 | 0.304 | 0.178 | 0.304 | 0.304 | 0.304 | 0.000 | 0.344 | 0.040 |
| llada-8b-instruct-hf | plan_351 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.423 | 0.423 | 0.000 | 0.000 | 0.375 | 0.375 | 0.375 | 0.375 | 0.375 | 0.000 | 0.375 | 0.000 |
| llada-8b-instruct-hf | plan_353 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.449 | 0.464 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.261 | 0.261 | 0.000 | 0.283 | 0.021 |
| llada-8b-instruct-hf | plan_354 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.437 | 0.437 | 0.000 | 0.000 | 0.340 | 0.340 | 0.340 | 0.340 | 0.340 | 0.000 | 0.340 | 0.000 |
| llada-8b-instruct-hf | plan_355 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.387 | 0.387 | 0.000 | 0.000 | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.000 | 0.279 | 0.029 |
| llada-8b-instruct-hf | plan_359 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.480 | 0.480 | 0.000 | 0.000 | 0.364 | 0.364 | 0.364 | 0.364 | 0.364 | 0.000 | 0.364 | 0.000 |
| llada-8b-instruct-hf | plan_360 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | max_generated_repair_value_v1_score_repair_pool | low_confidence_32 | final |  | 0.358 | 0.358 | 0.203 | 0.203 | 0.200 | 0.045 | 0.200 | 0.200 | 0.380 | 0.180 | 0.380 | 0.000 |
| llada-8b-instruct-hf | plan_361 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.423 | 0.423 | 0.000 | 0.000 | 0.310 | 0.310 | 0.310 | 0.310 | 0.310 | 0.000 | 0.310 | 0.000 |
| llada-8b-instruct-hf | plan_366 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | constraint_gap_span_history_contrast_repair | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | max_generated_repair_value_v1_score_repair_pool | evolved_low_confidence_48 | final |  | 0.338 | 0.375 | 0.065 | 0.065 | 0.220 | 0.045 | 0.220 | 0.220 | 0.282 | 0.062 | 0.282 | 0.000 |
| llada-8b-instruct-hf | plan_373 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.400 | 0.400 | 0.000 | 0.000 | 0.221 | 0.221 | 0.221 | 0.221 | 0.221 | 0.000 | 0.221 | 0.000 |
| llada-8b-instruct-hf | plan_375 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | random_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.520 | 0.539 | 0.000 | 0.000 | 0.314 | 0.362 | 0.314 | 0.334 | 0.334 | 0.000 | 0.362 | 0.028 |
| llada-8b-instruct-hf | plan_376 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.426 | 0.426 | 0.000 | 0.000 | 0.240 | 0.240 | 0.240 | 0.240 | 0.240 | 0.000 | 0.240 | 0.000 |
| llada-8b-instruct-hf | plan_377 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.391 | 0.391 | 0.000 | 0.000 | 0.275 | 0.275 | 0.275 | 0.275 | 0.275 | 0.000 | 0.275 | 0.000 |
| llada-8b-instruct-hf | plan_378 | low_confidence_32 | low_confidence_32 | low_confidence_32 | evolved_random_48 | evolved_random_48 | evolved_revision_low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.253 | 0.340 | 0.000 | 0.000 | 0.282 | 0.282 | 0.282 | 0.241 | 0.241 | 0.000 | 0.319 | 0.078 |
| llada-8b-instruct-hf | plan_380 | low_confidence_32 | random_32 | low_confidence_32 | evolved_low_confidence_48 | evolved_low_confidence_48 | low_confidence_32 | max_planning_state_score_base_pool | max_planning_quality_fallback_evolved_pool | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.452 | 0.484 | 0.000 | 0.000 | 0.261 | 0.198 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_381 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.355 | 0.355 | 0.000 | 0.000 | 0.261 | 0.261 | 0.261 | 0.261 | 0.261 | 0.000 | 0.261 | 0.000 |
| llada-8b-instruct-hf | plan_384 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.404 | 0.404 | 0.000 | 0.000 | 0.283 | 0.283 | 0.283 | 0.283 | 0.283 | 0.000 | 0.283 | 0.000 |
| llada-8b-instruct-hf | plan_385 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | constraint_gap_span_history_contrast_repair | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.376 | 0.376 | 0.000 | 0.000 | 0.220 | 0.220 | 0.220 | 0.220 | 0.220 | 0.000 | 0.220 | 0.000 |
| llada-8b-instruct-hf | plan_386 | low_confidence_32 | low_confidence_32 | random_32 | random_32 | random_32 | evolved_revision_random_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.314 | 0.314 | 0.000 | 0.000 | 0.065 | 0.065 | 0.137 | 0.137 | 0.137 | 0.000 | 0.158 | 0.021 |
| llada-8b-instruct-hf | plan_389 | low_confidence_32 | random_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | low_confidence_32 | max_planning_state_score_base_pool | evolved_margin_guard_kept_base_pool_0.015_revision_0.050 | repair_margin_guard_kept_evolved_0.020 |  |  |  | 0.361 | 0.361 | 0.000 | 0.000 | 0.404 | 0.045 | 0.404 | 0.404 | 0.404 | 0.000 | 0.404 | 0.000 |
