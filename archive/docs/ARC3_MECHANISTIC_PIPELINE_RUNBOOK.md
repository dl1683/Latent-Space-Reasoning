# ARC-3 mechanistic pipeline runbook

This is the offline path from existing ARC-3/LS20 replay artifacts to reusable
mechanistic reasoning data. It does not call ARC-3, start the local game server,
use GPU, or run a new solver.

## Preconditions

The shell/process layer must be healthy. If process commands are hanging, first
kill the stale runtime search from a fresh terminal:

```powershell
powershell -ExecutionPolicy Bypass -File experiments/kill_arc3_runtime_search.ps1
```

Then verify the known-good local full LS20 replay:

```powershell
python experiments/replay_ls20_plan.py --plans eval_results/ls20_static_astar_plans_through_l7.json --through-level 7 --require-solved-through 7 --output eval_results/ls20_replay_astar_l7_verified.json
```

Before using real LS20 artifacts, run the self-contained offline smoke test:

```powershell
python experiments/run_arc3_mechanistic_smoke.py --output-dir eval_results/mechanistic_rules/smoke
```

## Extract transitions

The one-command offline path is:

```powershell
python experiments/run_arc3_mechanistic_pipeline.py eval_results/ls20_replay_l5_timed_rotation_fix.json --output-dir eval_results/mechanistic_rules/ls20_l5 --pretty
```

It writes `transitions.jsonl`, `objects.json`, `rules.json`,
`rule_checks.json`, `graded_rules.json`, `validated_rules.json`,
`contextual_rules.json`, `contextual_rule_checks.json`,
`contextual_graded_rules.json`, `contextual_validated_rules.json`,
`repairs.json`, and `manifest.json`.

Audit the manifest before treating the artifacts as evidence:

```powershell
python experiments/audit_arc3_mechanistic_manifest.py eval_results/mechanistic_rules/ls20_l5/manifest.json --pretty
```

Score the run as an internal mechanistic-progress signal:

```powershell
python experiments/score_arc3_mechanistic_run.py eval_results/mechanistic_rules/ls20_l5/manifest.json --pretty
```

For stage-by-stage debugging, run the commands below.

```powershell
python experiments/extract_arc3_transitions.py eval_results/ls20_replay_l5_timed_rotation_fix.json --output eval_results/mechanistic_rules/ls20_l5_transitions.jsonl
```

Output: one JSON object per normalized `TransitionTrace`.

## Infer objects

```powershell
python experiments/infer_arc3_objects.py eval_results/mechanistic_rules/ls20_l5_transitions.jsonl --output eval_results/mechanistic_rules/ls20_l5_objects.json --pretty
```

Output: a JSON array of conservative `ObjectHypothesis` records.

## Infer candidate rules

```powershell
python experiments/infer_arc3_rules.py eval_results/mechanistic_rules/ls20_l5_objects.json --output eval_results/mechanistic_rules/ls20_l5_rules.json --pretty
```

Output: a JSON array of conservative `RuleHypothesis` records. Repeated
identical effects without counterexamples become `candidate`; one-off or
conflicting effects remain `unknown`. Numeric movement-delta rules are promoted
only when the observed delta for that action/field pair is unique. This keeps
contextual transitions such as pushers, blocked moves, deliveries, and level
boundaries from becoming false unconditional movement rules.

## Infer contextual rules

Contextual rules preserve repeated special cases that should not become
unconditional action rules. For example, level 6 contains a repeated pusher
exception where `ACTION1` at `x=49` moves `y` by `+15`.

```powershell
python experiments/infer_arc3_contextual_rules.py eval_results/mechanistic_rules/ls20_l6_verified/transitions.jsonl --output eval_results/mechanistic_rules/ls20_l6_verified/contextual_rules.json --pretty
```

The full pipeline now emits this file automatically, checks contextual rules
against matching trace contexts, grades them, and exports only validated
contextual rules:

```powershell
python experiments/check_arc3_contextual_rules.py eval_results/mechanistic_rules/ls20_l6_verified/transitions.jsonl eval_results/mechanistic_rules/ls20_l6_verified/contextual_rules.json --output eval_results/mechanistic_rules/ls20_l6_verified/contextual_rule_checks.json --pretty
```

The current level-6 verified run has one contextual rule, two contextual checks,
one validated contextual rule, and zero contextual contradictions. Level 7 has
zero contextual rules.

## Static A* LS20 planning

The current verified local LS20 plan uses the ordered-subgoal A* planner:

```powershell
python experiments/solve_ls20_static_astar.py --level 6 --target-order 1,0 --max-depth 220 --max-states 800000 --max-seconds 90 --output eval_results/ls20_static_astar_l6_order_1_0.json
python experiments/solve_ls20_static_astar.py --level 7 --target-order 0 --max-depth 260 --max-states 800000 --max-seconds 90 --output eval_results/ls20_static_astar_l7_order_0.json
```

The verified stitched plan is:

```text
eval_results/ls20_static_astar_plans_through_l7.json
```

Replay it against the real downloaded LS20 runtime before treating it as a
benchmark result:

```powershell
python experiments/replay_ls20_plan.py --plans eval_results/ls20_static_astar_plans_through_l7.json --through-level 7 --require-solved-through 7 --output eval_results/ls20_replay_astar_l7_verified.json
```

Current evidence:

- Level 6 solves in 72 actions with target order `1,0`.
- Level 7 solves in 53 actions.
- Replay through level 7 reaches `GameState.WIN`.
- `eval_results/mechanistic_rules/ls20_l6_verified/manifest.json` audits and scores as reusable with contradiction rate `0.0`.
- `eval_results/mechanistic_rules/ls20_l7_verified/manifest.json` audits and scores as reusable with contradiction rate `0.0`.

## Check candidate rules

```powershell
python experiments/check_arc3_rules.py eval_results/mechanistic_rules/ls20_l5_transitions.jsonl eval_results/mechanistic_rules/ls20_l5_rules.json --output eval_results/mechanistic_rules/ls20_l5_rule_checks.json --pretty
```

Output: a JSON array of `PredictionCheck` records. Checks can be `supported`,
`contradicted`, or `not_applicable`. A supported check is local evidence, not a
global proof.

## Grade rules

```powershell
python experiments/grade_arc3_rules.py eval_results/mechanistic_rules/ls20_l5_rules.json eval_results/mechanistic_rules/ls20_l5_rule_checks.json --output eval_results/mechanistic_rules/ls20_l5_graded_rules.json --pretty
```

Output: a JSON array of graded rules with `validated`, `rejected`,
`tentative`, or `untested` status.

## Export validated rules

```powershell
python experiments/export_arc3_validated_rules.py eval_results/mechanistic_rules/ls20_l5_graded_rules.json --output eval_results/mechanistic_rules/ls20_l5_validated_rules.json --pretty
```

Output: a compact validated-rule library for downstream planners. Tentative,
rejected, and untested rules are excluded and counted.

## Export validated contextual rules

```powershell
python experiments/export_arc3_validated_contextual_rules.py eval_results/mechanistic_rules/ls20_l6_verified/contextual_graded_rules.json --output eval_results/mechanistic_rules/ls20_l6_verified/contextual_validated_rules.json --pretty
```

Output: a compact contextual-rule library with a top-level
`contextual_rules` list. It excludes tentative, rejected, and untested
contextual candidates.

## Apply validated and contextual rules

```powershell
python experiments/apply_arc3_validated_rules.py eval_results/mechanistic_rules/ls20_l5_validated_rules.json eval_results/mechanistic_rules/example_state.json --action enter_shape_pad --output eval_results/mechanistic_rules/example_prediction.json --pretty
```

Output: a predicted next-state object plus a record of which validated rules
applied or were skipped because their before-value precondition did not match.
The same applicator also accepts `contextual_rules` in the rule library object.
Contextual rules apply only when every explicit precondition matches the input
state.

Level-6 pusher smoke:

```powershell
python -c "import json; from experiments.apply_arc3_validated_rules import predict_state; lib={'contextual_rules': json.load(open('eval_results/mechanistic_rules/ls20_l6_verified/contextual_rules.json'))}; r=predict_state(lib, {'level_index':5,'levels_completed':5,'x':49,'y':10}, 'ACTION1'); print(r.predicted_state)"
```

Expected evidence: `y` becomes `25`. The rule's learned preconditions are
`level_index=5`, `levels_completed=5`, `x=49`, and `y=10`.

## Plan with validated rules

```powershell
python experiments/plan_arc3_with_validated_rules.py eval_results/mechanistic_rules/ls20_l5_validated_rules.json eval_results/mechanistic_rules/example_initial_state.json eval_results/mechanistic_rules/example_goal_state.json --output eval_results/mechanistic_rules/example_abstract_plan.json --max-depth 6 --pretty
```

Output: a short abstract action sequence if the exported rule library can
compose validated transitions to satisfy the requested goal-state fields.

## Evaluate planner scenarios

Generate scenarios from the validated library:

```powershell
python experiments/generate_arc3_rule_planner_scenarios.py eval_results/mechanistic_rules/ls20_l5_validated_rules.json --output eval_results/mechanistic_rules/planner_scenarios.json --pretty
```

Then evaluate them:

```powershell
python experiments/evaluate_arc3_rule_planner.py eval_results/mechanistic_rules/ls20_l5_validated_rules.json eval_results/mechanistic_rules/planner_scenarios.json --output eval_results/mechanistic_rules/planner_eval.json --max-depth 6 --pretty
```

Output: solved count, expected-solved matches, action-sequence matches, and one
result record per compact planning scenario.

For contextual-rule evaluation, merge the validated contextual-rule library at
scenario-generation and evaluation time:

```powershell
python experiments/generate_arc3_rule_planner_scenarios.py eval_results/mechanistic_rules/ls20_l6_verified/validated_rules.json --contextual-rules eval_results/mechanistic_rules/ls20_l6_verified/contextual_validated_rules.json --output eval_results/mechanistic_rules/ls20_l6_verified/planner_scenarios_with_context.json --pretty
python experiments/evaluate_arc3_rule_planner.py eval_results/mechanistic_rules/ls20_l6_verified/validated_rules.json eval_results/mechanistic_rules/ls20_l6_verified/planner_scenarios_with_context.json --contextual-rules eval_results/mechanistic_rules/ls20_l6_verified/contextual_validated_rules.json --output eval_results/mechanistic_rules/ls20_l6_verified/planner_eval_with_context.json --max-depth 6 --pretty
```

Current level-6 evidence: `6/6` scenarios solved, `6/6`
expected-solved matches, and `6/6` action matches. Pair scenarios are skipped
when one rule changes another rule's contextual precondition.

## Plan repairs

```powershell
python experiments/plan_arc3_repairs.py eval_results/mechanistic_rules/ls20_l5_rule_checks.json --output eval_results/mechanistic_rules/ls20_l5_repairs.json --pretty
```

Output: a JSON array of `RepairRecord` items. Each repair asks for the smallest
next trace shape needed to separate hidden preconditions from false effects.

## Next layer

The next implementation target is richer contextual rule factoring. The current
pipeline learns a simple repeated pusher exception; the static solver already
models pads, moving pads, target blocking, pickups, and delivery state. Those
should become explicit preconditioned rules with held-out replay checks.
