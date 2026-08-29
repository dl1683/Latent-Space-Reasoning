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

Status 2026-08-29: the toy quotient-world program (Rounds 36–37) has ended and NLM-007 is closed; the program now works on real-model artifacts (frozen Qwen3-1.7B-Base, CPU). The frozen-residual-stream line — coordinate_v1/v2 (killed at their capability baselines), coordinate_v3 (a reproducible 32/32 vs 0/32 causal next-token effect that audit #27 classifies as a narrow late lexical-control effect, not a latent grammatical coordinate), interchange_v1 (closed by its locked raw-sign baseline; no swap arm ran) — is **stopped as an allocation pivot, not as evidence that frozen pretrained residual streams lack usable native structure** (audit #28). **`state_bus_v1r1` is closed as a fixed-construction FAIL: held-out same-state code swaps preserved all categorical choices but not the preregistered confidence-signature tolerance, while cross-state taxonomy choice changes were confined to two fixed-cycle transitions and the registered heldout-taxonomy and third/first movement gates failed. The result supports only a repeatedly injected supervised response controller with pair-specific lexical/semantic steering—not autonomous persistence, abstraction, general interchangeability, or native latent mathematics.** `interchange_v2` (bias-controlled frozen anchor replacement; `experiments/run_interchange.py` + `experiments/config/interchange_v2.json`) is a pre-declared FAIL — cross-state replacement moved the probe signature 0.009 of the class separation — with its audit pending. No native latent mathematics has been demonstrated. Next artifact: reachability / control cost (Codex direction round 8). Licensed sentences and never-say lists: [STATE.md](STATE.md) "Current statement" and "Real-model line"; log: [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).

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
