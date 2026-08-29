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

Status 2026-08-29: NLM-007 (LM residual-stream dynamics) is closed under the terminal allocation rule; the toy quotient-world program (Rounds 36–37) ENDED under the 2026-08-29 governance amendment (exact certificates diagnostic-only; one audit per result; real models only; measurement-to-artifact ratio tripwire) — no learned artifact passed the complete exact reducer and Round 37 showed no architectural win; verdicts are retained under `experiments/results/operational_quotient_*/verdict.json` and `experiments/results/presentation_quotient_v1_*/verdict.json` (runners and configs live in git history). The current artifact line is the two-bit causal coordinate in a real residual stream (`experiments/run_coordinate.py`): `coordinate_v1` (Qwen3-0.6B, tense × polarity) is `UNINTERPRETABLE — INVALID POLARITY BASELINE` (demo stopped at calibration; `experiments/results/coordinate_v1/`), and `coordinate_v2` (Qwen3-1.7B, tense × grammatical number) was killed at its pre-declared baseline gate (`01` 12/16 against `>=14/16`; `experiments/results/coordinate_v2/`). The next step is a Codex decision. Licensed wording for every closed round is in [STATE.md](STATE.md).

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
