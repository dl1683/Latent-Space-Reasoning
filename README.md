# Latent Space Reasoning

**Program (opened 2026-08-27): the native mathematics of latent spaces.**
Embedding and latent spaces are not number systems. This program builds their
mathematics from axioms upward — what a latent space is, which relations and
operations are meaningful — and derives definitions and propositions from
there, Euclid-style, rather than porting ℝⁿ mathematics onto embedding vectors.
Classical goals carry over (measure closeness, update on evidence, compose,
infer, prove); the native constructs may look nothing like their classical
counterparts.

- Current state: [STATE.md](STATE.md) · running log: [NOTEBOOK.md](NOTEBOOK.md)

Status 2026-08-29 (end of day): Across today's Qwen3-0.6B/1.7B interventions, we did not demonstrate native coordinates, interchangeable state, persistent state, or a transferable control-cost law; we observed only late lexical steering, pair-specific supervised response control, and null or failing fixed block-12 anchor/span constructions. Seven constructions (coordinate_v1/v2/v3, interchange_v1/v2, state_bus_v1r1, control_cost_v1) are closed by their own pre-declared rules, each with a fresh audit (#27–#31); licensed sentences and never-say lists are in STATE.md. The frozen-pretrained residual-stream line is stopped as an allocation decision, not as evidence that pretrained models lack latent mathematics. Next (Codex direction round 10): `necessity_navigator_v1` — a small model trained from scratch to navigate a hidden world with a known noncommutative action algebra, to test whether that mathematics can be read from a learned state that behaviour makes necessary; pretrained models are compared to it afterwards.

## Prior program and correction (closed 2026-08-27)

The previous program — LLM embedding perturbation, diffusion latent repair,
multi-latent aggregation — is archived unmodified under [`legacy/`](legacy/).
Its nested-arithmetic perturbation claims ("perturbation unlocks capabilities
that scaling cannot", beats temperature, better cost-per-capability) were
**withdrawn** after independent controls by Igor Rivin (PRs #4, #5) showed the
benchmark measured termination under a token cap, not arithmetic. Record:
[legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md](legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).
Standing results from that program are summarized in
[legacy/README.md](legacy/README.md). One finding carries forward: greedy
decoding determinism is hardware-dependent, so any diversity claim must report
the stack's numerical noise floor.

## Rules for claims

A claim is exploratory until it has a stated axiom base or predeclared task
slice, raw artifacts, a falsifier, and a control run for the cheapest
alternative explanation. Empirical checks on real embedding spaces report the
stack's determinism and noise floor.
