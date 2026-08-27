# ARC-3 mechanistic validation checklist

This checklist is the recovery path after the host shell/process layer is
healthy. It separates cheap offline validation from real LS20/ARC-3 runs.

## 1. Self-contained smoke

Command:

```powershell
python experiments/run_arc3_mechanistic_smoke.py --output-dir eval_results/mechanistic_rules/smoke
```

Expected evidence:

- `eval_results/mechanistic_rules/smoke/smoke_result.json` exists.
- `passed` is `true`.
- `score.status` is `reusable`.
- `planner_evaluation.solved` is `1`.

## 2. Focused unit tests

Command:

```powershell
python -m pytest tests/test_arc3_transition_extractor.py tests/test_arc3_object_inference.py tests/test_arc3_rule_inference.py tests/test_arc3_rule_checker.py tests/test_arc3_repair_planner.py tests/test_arc3_rule_grading.py tests/test_arc3_validated_rule_export.py tests/test_arc3_validated_rule_application.py tests/test_arc3_validated_rule_planner.py tests/test_arc3_rule_planner_evaluation.py tests/test_arc3_mechanistic_pipeline.py tests/test_arc3_mechanistic_manifest_audit.py tests/test_arc3_mechanistic_run_score.py tests/test_arc3_mechanistic_smoke.py
```

Expected evidence:

- All listed tests pass.
- No ARC API call is made.
- No game server is started.
- No GPU is required.

## 3. Known-good LS20 replay gate

Command:

```powershell
python experiments/replay_ls20_plan.py --plans eval_results/ls20_static_astar_plans_through_l7.json --through-level 7 --require-solved-through 7 --output eval_results/ls20_replay_astar_l7_verified.json
```

Expected evidence:

- Replay verifier exits successfully.
- Levels 1 through 7 are solved locally.
- Level 7 ends in `GameState.WIN`.
- This checks the current full local LS20 floor before touching external ARC-3 submission paths.

## 4. Offline LS20 mechanistic pipeline

Command:

```powershell
python experiments/run_arc3_mechanistic_pipeline.py eval_results/ls20_replay_astar_l7_verified_trace.json --output-dir eval_results/mechanistic_rules/ls20_l7_verified --pretty
```

Expected evidence:

- `eval_results/mechanistic_rules/ls20_l7_verified/manifest.json` exists.
- Pipeline emits `transitions.jsonl`, `objects.json`, `rules.json`,
  `rule_checks.json`, `graded_rules.json`, `validated_rules.json`,
  `contextual_rules.json`, `contextual_rule_checks.json`,
  `contextual_graded_rules.json`, `contextual_validated_rules.json`, and
  `repairs.json`.

## 5. Manifest audit

Command:

```powershell
python experiments/audit_arc3_mechanistic_manifest.py eval_results/mechanistic_rules/ls20_l7_verified/manifest.json --pretty
```

Expected evidence:

- Audit exits successfully.
- Output files parse.
- Manifest counts match artifact contents.
- Rule checks link to candidate rules.
- Repairs link to contradicted rules.
- Validated-rule library matches graded validated rules.
- Contextual-rule count matches `contextual_rules.json`.
- Contextual-rule-check count matches `contextual_rule_checks.json`.
- Contextual graded-rule count matches `contextual_graded_rules.json`.
- Contextual validated-rule library matches graded contextual validated rules.
- Contextual checks link to candidate contextual rules.

## 6. Mechanistic run score

Command:

```powershell
python experiments/score_arc3_mechanistic_run.py eval_results/mechanistic_rules/ls20_l7_verified/manifest.json --pretty
```

Expected evidence:

- `audit_passed` is `true`.
- `status` is one of `observed_only`, `needs_repair`, or `reusable`.
- `invalid` means the artifacts cannot be used as evidence.
- For contextual runs, `contextual_contradiction_rate` should be `0.0` before
  treating contextual rules as planner inputs.
- For contextual planner runs, use `contextual_validated_rules.json`, not raw
  `contextual_rules.json`.

## 7. Abstract planner evaluation

Generate compact scenarios from validated rules:

```powershell
python experiments/generate_arc3_rule_planner_scenarios.py eval_results/mechanistic_rules/ls20_l7_verified/validated_rules.json --output eval_results/mechanistic_rules/ls20_l7_verified/planner_scenarios.json --pretty
```

Command:

```powershell
python experiments/evaluate_arc3_rule_planner.py eval_results/mechanistic_rules/ls20_l7_verified/validated_rules.json eval_results/mechanistic_rules/ls20_l7_verified/planner_scenarios.json --output eval_results/mechanistic_rules/ls20_l7_verified/planner_eval.json --max-depth 6 --pretty
```

Expected evidence:

- Scenario results are written.
- `solved`, `expected_solved_matches`, and `action_matches` give a cheap
  rule-library quality metric.

Contextual level-6 command:

```powershell
python experiments/generate_arc3_rule_planner_scenarios.py eval_results/mechanistic_rules/ls20_l6_verified/validated_rules.json --contextual-rules eval_results/mechanistic_rules/ls20_l6_verified/contextual_validated_rules.json --output eval_results/mechanistic_rules/ls20_l6_verified/planner_scenarios_with_context.json --pretty
python experiments/evaluate_arc3_rule_planner.py eval_results/mechanistic_rules/ls20_l6_verified/validated_rules.json eval_results/mechanistic_rules/ls20_l6_verified/planner_scenarios_with_context.json --contextual-rules eval_results/mechanistic_rules/ls20_l6_verified/contextual_validated_rules.json --output eval_results/mechanistic_rules/ls20_l6_verified/planner_eval_with_context.json --max-depth 6 --pretty
```

Expected evidence:

- `scenarios` is `6`.
- `solved` is `6`.
- `expected_solved_matches` is `6`.
- `action_matches` is `6`.

## 8. Held-out rule generalization

Command:

```powershell
python experiments/evaluate_arc3_rule_generalization.py eval_results/ls20_replay_astar_l6_verified_trace.json --train-fraction 0.7 --output eval_results/mechanistic_rules/ls20_l6_verified/rule_generalization_70_30.json --pretty
python experiments/evaluate_arc3_rule_generalization.py eval_results/ls20_replay_astar_l7_verified_trace.json --train-fraction 0.7 --output eval_results/mechanistic_rules/ls20_l7_verified/rule_generalization_70_30.json --pretty
```

Expected evidence:

- `status` is `predictive`.
- `contradicted` is `0`.
- `applicable_precision` is `1.0`.
- Report `transition_coverage`; low coverage means the rules are precise but
  incomplete, not that the task is solved online.

## 9. Reproduce ordered A* solving

Level 6 and level 7 are solved by ordered-subgoal static A*:

```powershell
python experiments/solve_ls20_static_astar.py --level 6 --target-order 1,0 --max-depth 220 --max-states 800000 --max-seconds 90 --output eval_results/ls20_static_astar_l6_order_1_0.json
python experiments/solve_ls20_static_astar.py --level 7 --target-order 0 --max-depth 260 --max-states 800000 --max-seconds 90 --output eval_results/ls20_static_astar_l7_order_0.json
```

Expected evidence:

- Level 6 result has `solved: true`, `action_count: 72`.
- Level 7 result has `solved: true`, `action_count: 53`.
- The stitched plan in `eval_results/ls20_static_astar_plans_through_l7.json`
  passes the replay gate in section 3.

## 10. Legacy bounded search checks

These commands are now regression probes rather than the primary solution path:

Bounded static planner command:

```powershell
python experiments/solve_ls20_static.py --level 6 --max-depth 260 --max-states 250000 --max-seconds 60 --output eval_results/ls20_static_plan_l6_bounded.json
```

Bounded runtime-search command:

```powershell
python experiments/search_ls20_runtime.py --level 6 --output eval_results/ls20_runtime_search_l6_bounded.json
```

Expected evidence:

- Search commands respect time/state/frontier bounds.
- Any candidate level 6 plan is replayed locally before being added to the
  official scripted plan file.
