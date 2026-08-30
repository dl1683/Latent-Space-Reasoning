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

Status 2026-08-29: no native latent mathematics has been demonstrated. The frozen-residual intervention line on Qwen3-0.6B/1.7B (`coordinate_v1/v2/v3`, `interchange_v1/v2`, `state_bus_v1r1`, `control_cost_v1`) is stopped as an allocation pivot, not as evidence that pretrained residual streams lack usable structure: it produced only bounded causal facts — late interventions steer lexical output, a repeatedly injected bus acts as a supervised controller, and the registered anchor/span constructions fail their stated laws (in `control_cost_v1` the block-12 prefix-span field attained its A endpoint in 1/8 recipients, B was not a native-valid readout, and censoring blocks every cost-law, ranking, transfer, or asymmetry reading). The most important discovered constraint is that the tested Qwen3 bases through 4B cannot reliably apply a two-variable table even when the facts are visible, so rule-dependent behavioral readouts are instrument-invalid at this scale; that killed the one-write state artifact (`onewrite_state_v1`) before any state hypothesis was tested. The one pending artifact is `onewrite_recall_v1` — write one private tag into the frozen model once, change every word around it, and test whether a single hidden write lets it recall that tag — built and validated pre-lock (visible-copy 1.0 vs cue 0.0), with a protocol-criterion ruling pending before any lock or run. A from-scratch navigation substrate (`necessity_navigator_v1`) is built as an optional one-round calibration control and unrun. Full record and licensed wording: [STATE.md](STATE.md); experiment log: [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).

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
