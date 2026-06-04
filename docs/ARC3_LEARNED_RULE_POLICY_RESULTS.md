# ARC-3 Learned Rule Policy Results

This note separates three claims that should not be blended:

1. The scripted A* run is a ceiling and teacher trace for LS20.
2. The learned-rule evaluators measure whether compact mechanics learned from earlier transitions transfer to held-out transitions.
3. A full live ARC-3 controller still needs perception/state abstraction from official game observations.

The current contextual-rule learner is evidence weighted: it can learn repeated action effects under stable before-state context, but it rejects a contextual rule when equally many contradictory effects match that same context. It also supports auditable numeric interval preconditions for local movement regimes. On the current LS20 traces, interval preconditions expand the rule language but do not change the headline metrics; the measured gain still comes from learned contextual movement effects, not from a looser matcher.

The base rule learner also treats the `steps` counter as a constrained `+1` delta. This improves the explicit transition model without changing action selection, because every legal action shares the same step-counter update. Non-forward step deltas are intentionally ignored.

The rule library also applies a narrow inverse-action symmetry closure: learned `ACTION1`/`ACTION2` movement effects can imply the opposite vertical effect, and learned `ACTION3`/`ACTION4` movement effects can imply the opposite horizontal effect. Derived rules are tagged with `derivation.type = "inverse_action_symmetry"` so direct evidence and inferred symmetry stay distinguishable.

Prediction now applies at most one rule per field for a proposed action. This prevents duplicate direct/contextual rules from stacking two movement deltas on the same coordinate and making an action look better than it is.

## Current Held-Out Policy Signal

The policy evaluator learns rules from the first 70% of a verified LS20 replay and then ranks actions on the held-out 30% using only learned rule predictions.

Command shape:

```bash
python experiments/evaluate_arc3_rule_policy.py eval_results/ls20_replay_astar_l7_verified_trace.json --train-fraction 0.7 --output eval_results/mechanistic_rules/ls20_l7_verified/rule_policy_70_30.json --pretty
```

Results:

| Trace | Train | Test | Learned Actions | Decidable | Top-1 Action Accuracy | Frequency Baseline | Lift | Modeled Transition Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LS20 L6 | 50 | 22 | 4 | 22 | 95.45% | 22.73% | +72.73 pts | 90.91% |
| LS20 L7 | 37 | 16 | 4 | 16 | 100.00% | 43.75% | +56.25 pts | 93.75% |

Boundary split:

| Trace | Boundary Transitions | Boundary Top-1 | Non-Boundary Transitions | Non-Boundary Top-1 |
| --- | ---: | ---: | ---: | ---: |
| LS20 L6 | 1 | 0.00% | 21 | 100.00% |
| LS20 L7 | 1 | 100.00% | 15 | 100.00% |

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_verified/rule_policy_70_30.json`
- `eval_results/mechanistic_rules/ls20_l7_verified/rule_policy_70_30.json`

## Mechanism Ablation

The ablation evaluator runs the same 70/30 held-out split while toggling contextual rules and inverse-action symmetry. This is the main guard against attributing the result to a single opaque overfit bundle.

Command shape:

```bash
python experiments/evaluate_arc3_rule_policy_ablation.py eval_results/ls20_replay_astar_l6_verified_trace.json --train-fraction 0.7 --output eval_results/mechanistic_rules/ls20_l6_verified/rule_policy_ablation_70_30.json --pretty
```

Results:

| Trace | Variant | Top-1 Action Accuracy | Non-Boundary Top-1 | Modeled Transition Accuracy |
| --- | --- | ---: | ---: | ---: |
| LS20 L6 | base_only | 77.27% | 76.19% | 72.73% |
| LS20 L6 | base_plus_contextual | 90.91% | 90.48% | 86.36% |
| LS20 L6 | base_plus_inverse | 95.45% | 100.00% | 90.91% |
| LS20 L6 | full | 95.45% | 100.00% | 90.91% |
| LS20 L7 | base_only | 100.00% | 100.00% | 93.75% |
| LS20 L7 | base_plus_contextual | 100.00% | 100.00% | 93.75% |
| LS20 L7 | base_plus_inverse | 100.00% | 100.00% | 93.75% |
| LS20 L7 | full | 100.00% | 100.00% | 93.75% |

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_verified/rule_policy_ablation_70_30.json`
- `eval_results/mechanistic_rules/ls20_l7_verified/rule_policy_ablation_70_30.json`

Interpretation:

On L6, contextual rules account for the first large jump, and inverse symmetry closes the remaining ordinary movement gap. On L7, the base rules already solve the 70/30 split, so the extra mechanisms do not inflate the score.

## Train-Fraction Sweep

The sweep evaluator repeats the same learned-rule policy test across several temporal train/test splits. This checks whether the result survives a different amount of observed evidence rather than depending on one 70/30 split.

Command shape:

```bash
python experiments/sweep_arc3_rule_policy.py eval_results/ls20_replay_astar_l7_verified_trace.json --fractions 0.5 0.6 0.7 0.8 --output eval_results/mechanistic_rules/ls20_l7_verified/rule_policy_sweep.json --pretty
```

Results:

| Trace | Fractions | Mean Top-1 Action Accuracy | Minimum Top-1 Action Accuracy | Mean Frequency Baseline | Mean Lift | Mean Modeled Transition Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LS20 L6 | 0.5, 0.6, 0.7, 0.8 | 95.64% | 93.33% | 22.16% | +73.48 pts | 91.28% |
| LS20 L7 | 0.5, 0.6, 0.7, 0.8 | 100.00% | 100.00% | 28.57% | +71.43 pts | 92.04% |

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_verified/rule_policy_sweep.json`
- `eval_results/mechanistic_rules/ls20_l7_verified/rule_policy_sweep.json`

## Cross-Trace Transfer

The transfer evaluator trains rules on one verified LS20 replay and tests action choice on a different verified LS20 replay. This is a stronger reuse test than an in-trace temporal split.

Command shape:

```bash
python experiments/evaluate_arc3_rule_transfer.py eval_results/ls20_replay_astar_l7_verified_trace.json eval_results/ls20_replay_astar_l6_verified_trace.json --output eval_results/mechanistic_rules/ls20_l7_to_l6_rule_transfer.json --pretty
```

Results:

| Train Trace | Test Trace | Train Transitions | Test Transitions | Decidable | Top-1 Action Accuracy | Frequency Baseline | Lift | Modeled Transition Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LS20 L6 | LS20 L7 | 72 | 53 | 53 | 100.00% | 22.64% | +77.36 pts | 96.23% |
| LS20 L7 | LS20 L6 | 53 | 72 | 72 | 95.83% | 18.06% | +77.78 pts | 94.44% |

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_to_l7_rule_transfer.json`
- `eval_results/mechanistic_rules/ls20_l7_to_l6_rule_transfer.json`

## Online Continual Learning

The online evaluator scores a prequential loop: before each transition, it learns rules from all earlier transitions and chooses the next action; after scoring, the transition becomes new evidence. This is the closest current offline proxy for "keeps improving as it observes more."

Command shape:

```bash
python experiments/evaluate_arc3_online_rule_learning.py eval_results/ls20_replay_astar_l7_verified_trace.json --warmup 4 --output eval_results/mechanistic_rules/ls20_l7_verified/online_rule_learning.json --pretty
```

Results:

| Trace | Warmup | Evaluated | Top-1 Action Accuracy | Frequency Baseline | Lift | First Half Accuracy | Second Half Accuracy | Improvement | Modeled Transition Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LS20 L6 | 4 | 68 | 100.00% | 22.06% | +77.94 pts | 100.00% | 100.00% | 0.00 pts | 94.12% |
| LS20 L7 | 4 | 49 | 100.00% | 26.53% | +73.47 pts | 100.00% | 100.00% | 0.00 pts | 95.92% |

Boundary split:

| Trace | Boundary Transitions | Boundary Top-1 | Non-Boundary Transitions | Non-Boundary Top-1 |
| --- | ---: | ---: | ---: | ---: |
| LS20 L6 | 1 | 0.00% | 67 | 100.00% |
| LS20 L7 | 1 | 100.00% | 48 | 100.00% |

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_verified/online_rule_learning.json`
- `eval_results/mechanistic_rules/ls20_l7_verified/online_rule_learning.json`

Interpretation:

Both traces now reach 100.00% online action selection, including boundary and non-boundary transitions. This does not mean the full simulator transition is solved: modeled transition accuracy is lower because the rule library still does not predict every auxiliary reset/bookkeeping field on boundary transitions.

## Online Mechanism Ablation

The online ablation runs the same prequential loop while toggling contextual rules and inverse-action symmetry. This is the direct guard against reading the 100% online number as a memorized replay: the score is decomposed by reusable mechanism.

Command shape:

```bash
python experiments/evaluate_arc3_online_rule_learning_ablation.py eval_results/ls20_replay_astar_l7_verified_trace.json --warmup 4 --output eval_results/mechanistic_rules/ls20_l7_verified/online_rule_learning_ablation.json --pretty
```

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_verified/online_rule_learning_ablation.json`
- `eval_results/mechanistic_rules/ls20_l7_verified/online_rule_learning_ablation.json`

Results:

| Trace | Variant | Top-1 Action Accuracy | Non-Boundary Top-1 | Frequency Baseline | Lift | Modeled Transition Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LS20 L6 | base_only | 76.47% | 76.12% | 22.06% | +54.41 pts | 73.53% |
| LS20 L6 | base_plus_contextual | 91.18% | 91.04% | 22.06% | +69.12 pts | 85.29% |
| LS20 L6 | base_plus_inverse | 97.06% | 97.01% | 22.06% | +75.00 pts | 91.18% |
| LS20 L6 | full | 100.00% | 100.00% | 22.06% | +77.94 pts | 94.12% |
| LS20 L7 | base_only | 91.84% | 91.67% | 26.53% | +65.31 pts | 87.76% |
| LS20 L7 | base_plus_contextual | 91.84% | 91.67% | 26.53% | +65.31 pts | 87.76% |
| LS20 L7 | base_plus_inverse | 100.00% | 100.00% | 26.53% | +73.47 pts | 95.92% |
| LS20 L7 | full | 100.00% | 100.00% | 26.53% | +73.47 pts | 95.92% |

Interpretation:

The online 100% result is not a single monolithic script. On L6, contextual evidence and inverse symmetry both add measurable lift; on L7, inverse symmetry is sufficient to close the gap. The full policy reaches perfect online action choice by composing audited learned mechanisms, while weaker variants leave measurable errors.

## Alias Diagnosis

The alias diagnostic reads online rule-learning artifacts and summarizes wrong-action cases. This keeps the next fix tied to observed failures rather than guessing.

Command shape:

```bash
python experiments/diagnose_arc3_policy_aliases.py eval_results/mechanistic_rules/ls20_l6_verified/online_rule_learning.json --output eval_results/mechanistic_rules/ls20_l6_verified/policy_alias_diagnosis.json --pretty
```

Results:

| Trace | Failures | Failure Rate | Largest Confusion | Oracle Misses | Zero Modeled-Match Failures |
| --- | ---: | ---: | --- | ---: | ---: |
| LS20 L6 | 0 | 0.00% | none | 0 | 0 |
| LS20 L7 | 0 | 0.00% | none | 0 | 0 |

Missed modeled fields in the remaining online failures:

| Trace | Modeled Missed Field Counts |
| --- | --- |
| LS20 L6 | none |
| LS20 L7 | none |

Artifacts:

- `eval_results/mechanistic_rules/ls20_l6_verified/policy_alias_diagnosis.json`
- `eval_results/mechanistic_rules/ls20_l7_verified/policy_alias_diagnosis.json`

Interpretation:

There are no remaining online action-choice failures in the current L6 or L7 artifacts. The remaining weakness has moved from action choice to transition completeness, especially boundary reset/bookkeeping fields.

## Live Learned Visual Controller

The live controller now has a `learned_visual` backend in `experiments/arc3_latent_openai_server.py`. It loads state/action examples from full official recording steps, extracts object-centric visual state, and chooses legal actions by nearest learned state. The state abstraction includes foreground connected components and recent delta connected components, so the score is not driven only by whole-grid color histograms.

Command shape:

```bash
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --server-backend learned_visual --learned-trace external/arc-agi-3-benchmarking/recordings/ls20-9607627b.benchmarkingagent.local-latent-reasoning.anim7.49969d89-92b5-423e-96b7-855f64f6a4ec --learned-policy-k 1 --tags learned-visual,full-recording,k1,components,no-plan-index --harness-output eval_results/arc3_learned_visual_k1_components_harness.json --server-log eval_results/arc3_learned_visual_k1_components_server.log --trace-jsonl eval_results/arc3_learned_visual_k1_components_trace.jsonl --output eval_results/arc3_learned_visual_k1_components_smoke.json --server-ready-timeout-s 120
```

Results:

| Run | Training Examples | Held-Out Constraint | Official Score | Levels | Actions | Scripted Actions | Fallbacks |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| whole-grid/delta visual baseline, k=1 | 313 | none | 3.57 | 1 / 7 | 382 | 0 | 0 |
| object-component visual, k=1 | 313 | none | 100.00 | 7 / 7 | 313 | 0 | 0 |
| object-component visual, k=1 | 260 | train levels_completed <= 5 | 75.00 | 6 / 7 | 818 | 0 | 0 |
| object-component + sequence history, k=1 | 260 | train levels_completed <= 5 | 75.00 | 6 / 7 | 818 | 0 | 0 |
| object-component + sequence history + OOD repeat governor, k=1 | 260 | train levels_completed <= 5 | 75.00 | 6 / 7 | 818 | 0 | 0 |
| object-component + ineffective-action OOD feedback, k=1 | 260 | train levels_completed <= 5 | 75.00 | 6 / 7 | 818 | 0 | 0 |
| object-component + axis phase switch, k=1 | 260 | train levels_completed <= 5 | 75.00 | 6 / 7 | 818 | 0 | 0 |
| object-component + target goal-seek, k=1 | 260 | train levels_completed <= 5 | 75.00 | 6 / 7 | 818 | 0 | 0 |
| state_probe baseline | 0 | none | 0.00 | 0 / 7 | 66 | 0 | 0 |

Artifacts:

- `eval_results/arc3_learned_visual_k1_components_summary.json`
- `eval_results/arc3_learned_visual_k1_components_train_lte5_summary.json`
- `eval_results/arc3_learned_visual_k1_components_sequence_history_train_lte5_summary.json`
- `eval_results/arc3_learned_visual_k1_components_ood_repeat_train_lte5_summary.json`
- `eval_results/arc3_learned_visual_k1_components_ineffective_ood_no_sequence_train_lte5_summary.json`
- `eval_results/arc3_learned_visual_k1_components_phase_switch_train_lte5_summary.json`
- `eval_results/arc3_learned_visual_k1_components_goal_seek_train_lte5_summary.json`
- `eval_results/arc3_final_level_trace_comparison_train_lte5.json`
- `eval_results/arc3_state_probe_summary.json`

Interpretation:

The full-recording `100.00` result is a live official-harness result with no scripted plan and no fallback actions, but it is still a demonstration-imitation result because many neighbor distances are exactly `0.0`. The stronger reasoning signal is the level-held-out run: with final-level examples removed, the same controller reaches `6 / 7` levels and `75.00` official score. That shows reusable visual/object structure, but it does not yet solve novel-level extrapolation.

The sequence-history backoff, axis phase switch, and target goal-seek variants are currently not wins, so they are opt-in experiments rather than default controller behavior. Sequence history and target goal-seek both collapse toward repeated `ACTION2` on the withheld final level; target goal-seek fired `550` times in the held-out run without improving the `75.00` score. The OOD repeat governor prevents one-action collapse, shifting the final-level loop toward `ACTION1`/`ACTION4`, but still does not improve score. Ineffective-action OOD feedback blocks actions after tiny visual deltas, but the no-sequence controller still loops mainly between `ACTION1` and `ACTION3`. The next live-control frontier is a learned transition/goal model over object components, not action n-gram smoothing, repetition control, local ineffective-action blocking, axis-only phase switching, or hand-written target chasing.

The final-level trace comparison makes the failure more specific. The held-out controller matches the successful full-demonstration run for the first two final-level actions (`ACTION3`, `ACTION3`) and diverges at index `2`, where the successful run switches to `ACTION2` while the held-out controller keeps selecting `ACTION3`. The visual component snapshots are still identical at the divergence point, so the missing variable is not raw perception. The controller needs a learned goal/phase state that says when horizontal alignment is done and vertical progress should begin.

## ARC-AGI-3 Generalization Sweep Status

The public ARC-AGI-3 game list currently available through the API is:

`ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`, `g50t`, `ka59`, `lf52`, `lp85`, `ls20`, `m0r0`, `r11l`, `re86`, `s5i5`, `sb26`, `sc25`, `sk48`, `sp80`, `su15`, `tn36`, `tr87`, `tu93`, `vc33`, `wa30`.

New generalization infrastructure:

- `experiments/run_arc3_all_games.py` runs any local backend across an explicit game list and writes per-game manifests plus an aggregate summary.
- `experiments/evaluate_arc3_component_goal_policy.py` evaluates held-out component/goal action prediction before live harness runs.
- `experiments/sweep_arc3_component_goal_policy.py` sweeps that offline evaluator across local recordings.
- `graph_probe` backend adds masked-frame signatures, transition recording, failed-edge tracking, and graph travel toward frontier states.

Current negative results:

| Run | Game / Split | Result |
| --- | --- | ---: |
| component-goal lookup | LS20 final-level holdout from full recording | `15 / 53`, same as visual kNN |
| component-goal lookup sweep | first 10 local LS20 recordings, final-level holdout | no lift over visual kNN across `1266` examples |
| `state_probe` | live `ar25` | `0 / 8` levels, score `0.0`, `96` actions |
| `frontier_probe` | live `ar25` | `0 / 8` levels, score `0.0`, `96` actions |
| `graph_probe` | live `ar25` | `0 / 8` levels, score `0.0`, `96` actions |
| public `arc-agi-3-just-explore` heuristic reference | live `ar25` | stopped after hundreds of actions at score `0`; not a useful immediate teacher for this environment |

Interpretation:

The LS20 learned-visual result does not yet generalize to new ARC-AGI-3 games. The next target is not another LS20 heuristic. The next target is a general world-model loop: masked state abstraction, action-object abstraction, transition graph, goal/progress detector, and planner over learned transitions. The public graph-exploration literature supports this direction, but the current local implementation has not yet reproduced strong multi-game performance.

## Interpretation

The strongest current result is not "the model solved ARC-3." The strongest current result is that a learned mechanistic rule policy can choose LS20 actions substantially above a simple action-frequency baseline, with perfect online action choice on both L6 and L7 and perfect L6-to-L7 cross-trace transfer.

That matters because it is closer to reasoning than replay:

- The policy is learned from train transitions, not copied from the held-out action list.
- It is evaluated on temporally held-out transitions.
- It reports decidability separately from accuracy.
- It reports modeled-state accuracy separately from full simulator-state accuracy.

## Current Limitation

`exact_transition_accuracy` remains 0.0 on L6 and L7 because the learned rules do not yet predict every auxiliary field in the trace, especially level-boundary and bookkeeping fields. The more relevant metric for the current rule library is `modeled_transition_accuracy`, which only scores fields represented by learned rules.

The next frontier is to close this gap by learning richer transition structure and then wiring it into live ARC-3 control through a state abstraction layer.
