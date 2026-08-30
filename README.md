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

Status 2026-08-29: no native latent mathematics has been demonstrated. After `coordinate_v3`, `interchange_v1/v2`, `state_bus_v1r1`, and `control_cost_v1`, this is not working as a native-mathematics discovery program under the current frozen-residual intervention substrate. In `control_cost_v1`, the fixed block-12 uniform prefix-span Jacobian field attained its registered A endpoint in 1/8 held-out recipients, B was not a native-valid readout, and censoring prevents claims about cost ranking, cross-versus-within effort, transfer, or directional cost asymmetry. This closes that construction family and triggers a substrate-reconsideration dialogue; it is an allocation pivot, not evidence that pretrained residual streams lack usable structure. Next (Codex direction round 10, design gate in progress; audit #31 records a competing candidate — a real-model co-designed causally addressable state — and the required dialogue precedes any build): `necessity_navigator_v1`, a small model trained from scratch to navigate a hidden world with a known noncommutative action algebra, to test whether that mathematics can be read from a learned state that behaviour makes necessary. Evening: the one-write real-model state artifact (`onewrite_state_v1`) was killed pre-lock — the base model could not apply a stated rule to visibly given tags (visible 0.34 = no-tags 0.34), so no state hypothesis was tested; a from-scratch navigation substrate (`necessity_navigator_v1`) is built as an optional calibration control and unrun.

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
