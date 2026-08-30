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

## Status (restarted 2026-08-30 — theory-first)

The program restarted on 2026-08-30 in theory-first mode (math first, intuition second, compute third; no heavy LLM work). `theory/AXIOMS.md` now carries an adopted foundation — future-response geometry (mathematics audit #42): identity as zero future-response distance over the world's own response laws, nonexpansive descent of legal moves, a linear observability seminorm, and the local-to-global map problem as the open question. Nothing proved so far is new mathematics; the distinctive content is the boundary between a denizen's native responses and an analyst's instruments, under which every earlier empirical construction reads as instrument-level. The dialogue is verbatim in `theory/dialogue/004.md`.

### Empirical record (closed lines, 2026-08-27 → 2026-08-30)

Across two days of audited CPU experiments, no tested construction established a transferable causal state or native latent mathematics in a real model; the durable result is a localized set of instrument-validity, source-extraction, actuator, and prompt/output-geometry constraints that any future substrate must satisfy.

`register_bridge_preflight_v1` is a noncausal feasibility PASS: in frozen Qwen3-1.7B-Base, a predeclared cross-fitted rank-≤8 linear decoder read the state explicitly assigned to a record tag by an in-prompt legend under held-out entities, two held-out templates, and a disjoint balanced permutation bank at 0.815 accuracy, versus 0.125 input-embedding, 0.135 categorical, and 0.016 paired reassigned-legend original-state controls; the same intact decoders followed the state newly denoted by the paired reassigned legend at 0.852. This establishes a prompt-family-bounded, presentation-transferring explicit-legend signal at the record span—not a code-level or causal bridge, persistent register, synthetic-consumer usability, or native latent mathematics.

The empirical program paused after `register_bridge_preflight_v1` and audit #39 and was then restarted theory-first. Orientation: the [handoff](docs/HANDOFF_2026_08_30.md) (record of everything tried and closed), then the canonical [current state](STATE.md). The audited two-day closeout is [the structured negative](docs/STRUCTURED_NEGATIVE_2026_08_29.md), and append-only run/audit provenance is in the [experiment ledger](experiments/ledger.jsonl).

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
