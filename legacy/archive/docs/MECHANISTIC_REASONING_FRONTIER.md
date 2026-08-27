# Mechanistic Reasoning Frontier

This note captures the current research direction from the ARC-AGI-3 LS20 work.

The important result is not that a hand-built solver can solve LS20 levels. The
important result is that the same small set of latent mechanics explains large
behavioral jumps:

- action-effect discovery
- object identity
- stateful modifiers
- moving modifiers
- transport/pusher dynamics
- budget/reset constraints
- goal predicate matching

The practical frontier is to make those mechanics discoverable and reusable
without hand-writing a game-specific solver.

## Working Thesis

General reasoning can be improved without pure model scaling by adding a
mechanistic layer between perception and action:

1. Observe state transitions.
2. Infer compact latent objects and rules.
3. Search over the inferred transition system.
4. Verify actions against real environment feedback.
5. Promote verified rules into reusable priors for future tasks.

This turns "reasoning" from a longer token trace into an explicit cycle of
world-model induction, planning, and correction.

## Why LS20 Was Useful

LS20 exposed the failure mode clearly. A model or naive policy can observe
frames but does not naturally infer the hidden causal substrate:

- color/shape/rotation pads are stateful transformations
- pusher arrows are transport operators
- pickups reset a countdown budget
- hidden tracks move modifier pads before each player action
- target entry is blocked unless latent player attributes match the goal

Each missing mechanic caused an exact score plateau. Adding the mechanic caused
an official score jump:

- baseline exploration: `0 / 7`
- scripted levels 1-2: `2 / 7`
- pusher model: `3 / 7`, then `4 / 7`
- timed moving-pad correction: `5 / 7`

The score improved when the system learned or encoded causal structure, not
when it spent more tokens.

## Next Architecture

The next version should not be another LS20 script. It should be a reusable
mechanistic reasoning loop:

- `TransitionTrace`: compact before/action/after records.
- `ObjectHypothesis`: stable entities, positions, colors, shapes, rotations,
  tags, and hidden state variables.
- `RuleHypothesis`: candidate transition rules with preconditions,
  postconditions, and confidence.
- `PlannerState`: symbolic state projection produced by applying verified
  rules.
- `Verifier`: checks predicted transitions against actual environment frames.
- `RepairLoop`: localizes prediction errors to missing rules or wrong
  preconditions.

The key technical bar is that a new rule must earn its way in by predicting
future transitions, not by looking plausible in language.

## Near-Term Implementation Path

1. Convert ARC-3 recordings into transition examples.
2. Cluster persistent visual components into object hypotheses.
3. Infer action-effect operators from repeated deltas.
4. Add rule candidates for attribute modifiers and transport operators.
5. Run short model-assisted proposal only for candidate rule generation.
6. Keep verification deterministic and environment-backed.
7. Use symbolic search over verified rules before calling a model again.

This keeps the approach cheap: small models can propose hypotheses, but exact
verification and planning do most of the work.

## Success Criteria

For ARC-AGI-3, this line of work should aim for:

- solve all LS20 levels from transition traces, not hard-coded source tags
- transfer the same induction loop to at least one other public ARC-3 game
- reduce model calls per solved level after rules are learned
- produce auditable rule traces explaining each solution
- avoid GPU dependence for the main planning loop

The evidence gates are defined in `docs/ARC3_REASONING_VALIDATION_GATES.md`.
They are meant to show when the system improves action quality through learned
or verified mechanics, and to separate solved-target ceilings from reusable
reasoning evidence.

The paradigm shift is not "latent space reasoning" as a name. It is turning
reasoning into a self-improving, verified causal-model builder that any model
can use as scaffolding.
