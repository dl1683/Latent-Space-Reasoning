# ARC-3 Reasoning Evidence Gates

This repo should make strong claims when the evidence is strong. The goal is to
show a reasoning system that improves solution quality by combining model
proposals, learned mechanics, verification, and repair. The LS20 `100.0`
official score is valuable because it proves the environment bridge, mechanics,
planner, and verifier can produce a perfect policy for one ARC-3 game. The next
bar is to turn that policy into reusable reasoning evidence.

## Result Classes

### Class 0: Hardcoded Or Scripted Ceiling

Definition: an action list, source-derived solver, or game-specific policy is
executed through the official harness. This is still useful: it establishes a
solved target behavior, a replay/verifier loop, and a benchmark ceiling for the
learned system to match.

Allowed claim:

- "The bridge and verified policy can complete this game."
- "This is the current upper-bound target for learned policies."

Not sufficient by itself:

- "The model reasons through ARC-3."
- "The method generalizes."
- "This is a benchmark breakthrough."

Current examples:

- `eval_results/arc3_scripted_astar_l7_summary.json`: `100.0`, `7 / 7`.
- `eval_results/arc3_latent_one_call_scripted_fallback_summary.json`: `100.0`, `7 / 7`, but recovered by scripted fallback.
- `eval_results/arc3_latent_three_call_scripted_guard_summary.json`: `100.0`, `7 / 7`, but all model calls failed to emit legal actions and verified mechanics supplied actions.

### Class 1: Mechanistic Reconstruction And Quality Improvement

Definition: the system infers transition rules from traces and uses validated
rules plus search to predict, repair, or improve transitions.

Allowed claim:

- "The system learned reusable mechanics for observed transitions."
- "The rule library predicts held-out transitions under explicit checks."
- "Verified mechanics improved action quality over the raw model policy."

Required evidence:

- Train/test split over transition traces.
- `manifest.json` audit passes.
- `validated_rules.json` and `contextual_validated_rules.json` are produced by
  checks, not manually edited.
- Held-out transition prediction accuracy is reported.
- Planner success is reported separately from official harness score.
- A baseline model or probe policy is compared against the verified
  mechanic-guided policy.

Current held-out LS20 evidence:

- `eval_results/mechanistic_rules/ls20_l6_verified/rule_generalization_70_30.json`
  reports `16 / 16` supported applicable held-out checks, `0`
  contradictions, and `0.7273` transition coverage.
- `eval_results/mechanistic_rules/ls20_l7_verified/rule_generalization_70_30.json`
  reports `15 / 15` supported applicable held-out checks, `0`
  contradictions, and `0.9375` transition coverage.

These are real learned-rule prediction checks on held-out transitions. They are
not yet live online control, because uncovered transitions still require richer
state abstraction and planning.

Current status:

- L6 and L7 mechanistic manifests audit clean.
- L6 has one validated contextual rule and zero contextual contradictions.
- This is promising, but it is still mostly within a single solved game.

### Class 2: Online Mechanistic Reasoning

Definition: the agent observes the live environment, proposes or selects actions
from a learned mechanistic state model, and repairs rules when predictions fail.

Allowed claim:

- "The agent is reasoning through a live environment with verified mechanics."

Required evidence:

- No precomputed per-step action script.
- Action choice comes from current observed state plus learned rules.
- Every action has an audit trace: state abstraction, applicable rules, planned
  outcome, selected action, and post-action verifier result.
- Wrong model proposals are either rejected by a deterministic guard or become
  repair examples.

### Class 3: Generalization

Definition: the same induction and planning loop transfers beyond LS20.

Allowed claim:

- "The reasoning method generalizes across ARC-3 games."

Required evidence:

- At least one additional public ARC-3 game is evaluated.
- No game-specific source tags or hand-authored per-game scripts.
- A fixed induction/planning algorithm is used across games.
- Per-game rule libraries are learned from allowed observations and validated
  with held-out transitions.

## Evidence Rules

- Count `scripted_plan` fallback actions as mechanistic scaffolding, not raw
  model reasoning.
- Count source-derived solvers as solved targets and verifier baselines, not
  learned generalization.
- Count official harness completion together with trace-level attribution.
- Use score only with a trace-level attribution table.
- Report when guards or fallbacks improve the model's raw action quality.
- Treat a model response with no legal action as a useful repair signal when a
  guard recovers the run.

## Required Scorecard Fields

Every ARC-3 result summary should report:

- `official_score`
- `levels_completed`
- `total_actions`
- `model_actions`
- `model_legal_actions`
- `model_aligned_with_mechanics`
- `mechanistic_overrides`
- `fallback_actions`
- `scripted_actions`
- `no_legal_action_outputs`

The headline score becomes meaningful when these fields show how much quality
came from the model, the learned or verified mechanics, and the guard/repair
loop.

## Next Real Milestone

The next meaningful milestone is a quality-improvement run:

1. Generate a transition-trace split for LS20.
2. Learn rules from the training split.
3. Evaluate held-out transition prediction.
4. Use learned rules to select live actions without a per-step script.
5. Repeat the same loop on another ARC-3 game.

That would show the actual paradigm: small or imperfect models produce
candidates, verified mechanics improve them, and the system keeps turning
failures into better future behavior.
