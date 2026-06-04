# Mechanistic Reasoning Schema

This document defines the first reusable schema for moving from ARC-AGI-3
recordings to verified causal rules.

The goal is to make reasoning improvements come from learned mechanics rather
than from longer prompts or larger models.

## Core Records

### TransitionTrace

A single environment-backed transition.

```json
{
  "task_id": "ls20-9607627b",
  "level": 5,
  "step": 32,
  "action": "ACTION1",
  "before_frame_ref": "recordings/.../step_031.json",
  "after_frame_ref": "recordings/.../step_032.json",
  "before_objects": [],
  "after_objects": [],
  "changed_regions": [],
  "metadata": {
    "levels_completed_before": 4,
    "levels_completed_after": 4
  }
}
```

### ObjectHypothesis

A proposed persistent object or latent state carrier.

```json
{
  "object_id": "obj.rotation_pad.0",
  "kind": "modifier_pad",
  "observed_properties": {
    "position": [14, 35],
    "color_signature": [1, 0, 0],
    "shape_signature": "arrow-like"
  },
  "latent_properties": {
    "effect": "rotation_increment",
    "track_membership": "track.0"
  },
  "confidence": 0.82,
  "evidence": ["trace:ls20:5:32", "trace:ls20:5:36"]
}
```

### RuleHypothesis

A candidate causal rule with explicit preconditions and postconditions.

```json
{
  "rule_id": "rule.rotation_pad.increment",
  "scope": "ls20",
  "preconditions": [
    {"type": "overlap", "a": "player", "b": "modifier_pad.rotation"}
  ],
  "postconditions": [
    {"type": "attribute_delta", "object": "player", "attribute": "rotation", "delta": 1, "mod": 4}
  ],
  "negative_conditions": [
    {"type": "blocked_move"}
  ],
  "support": {
    "predicted": 7,
    "correct": 7,
    "incorrect": 0
  },
  "status": "verified"
}
```

Current implemented rule hypotheses are intentionally more conservative than
the full schema above. Unconditional action-field effects are only promoted when
the observed effect is stable. For numeric movement deltas, an action-field pair
must have a unique observed delta; conflicting deltas remain `unknown` until a
contextual precondition can separate them.

### ContextualRule

A repeated special-case transition with explicit state preconditions.

```json
{
  "rule_id": "6:ACTION1:y:delta:15:when:level_index=5,levels_completed=5,x=49",
  "level_id": "6",
  "action": "ACTION1",
  "field": "y",
  "effect": {"delta": 15},
  "preconditions": {
    "level_index": 5,
    "levels_completed": 5,
    "x": 49
  },
  "support": 2,
  "status": "candidate",
  "evidence_steps": [20, 66]
}
```

Contextual rules are executable by the same state predictor as validated rules,
but only when every precondition matches. The pipeline checks, grades, and
exports validated contextual rules to `contextual_validated_rules.json` before
using them as planner inputs. Planner scenarios include them as one-step cases
and skip invalid pairs where another rule changes a contextual precondition.

### ContextualPredictionCheck

A deterministic check of a contextual rule against only traces whose before-state
matches its preconditions.

```json
{
  "rule_id": "6:ACTION1:y:delta:15:when:level_index=5,levels_completed=5,x=49,y=10",
  "level_id": "6",
  "step_index": 20,
  "action": "ACTION1",
  "field": "y",
  "preconditions": {
    "level_index": 5,
    "levels_completed": 5,
    "x": 49,
    "y": 10
  },
  "expected": {"delta": 15},
  "observed": {"before": 10, "after": 25},
  "status": "supported"
}
```

### PredictionCheck

A deterministic check of one rule or rule set against a held-out transition.

```json
{
  "trace_id": "trace:ls20:5:36",
  "rules_used": ["rule.rotation_pad.increment", "rule.track.advance"],
  "predicted_after": {
    "player": {"x": 14, "y": 35, "rotation": 2}
  },
  "actual_after": {
    "player": {"x": 14, "y": 35, "rotation": 2}
  },
  "result": "pass"
}
```

### RepairRecord

When a rule set fails, this localizes what went wrong.

```json
{
  "trace_id": "trace:ls20:6:85",
  "failed_prediction": "player y should become 35",
  "actual": "player stayed at y=30",
  "candidate_causes": [
    "unmodeled blocking target",
    "wrong moving-pad phase",
    "missing delivered-target state"
  ],
  "next_probe": "test target blocking with matching vs nonmatching latent attributes"
}
```

## Validation Gates

A rule cannot become `verified` unless it passes all of these gates:

- It predicts at least one transition not used to propose the rule.
- It has an explicit precondition set.
- It has a failure case or negative condition when applicable.
- It improves planner success or reduces prediction error.
- It is replay-checked against the real environment or official recording.

## Near-Term Files To Build

The first implementation added:

- `experiments/extract_arc3_transitions.py`
- `experiments/infer_arc3_objects.py`
- `experiments/infer_arc3_rules.py`
- `experiments/infer_arc3_contextual_rules.py`
- `experiments/check_arc3_rules.py`
- `experiments/check_arc3_contextual_rules.py`
- `experiments/export_arc3_validated_contextual_rules.py`
- `eval_results/mechanistic_rules/`

The current verified planner path also includes
`experiments/solve_ls20_static_astar.py`, which solves the downloaded LS20 game
locally through level 7 / `GameState.WIN`. The next schema step is to lift the
static solver's contextual mechanics into learned preconditioned rules.

## Why This Matters

This schema is the bridge from the current hand-built solver to the intended
general reasoning system. The model can propose hypotheses, but the repository
should decide which hypotheses survive by prediction and replay.
