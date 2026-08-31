# Latent Space Reasoning

**Program (2026-08-27 → 2026-08-31, CLOSED): the native mathematics of latent spaces.**
Embedding and latent spaces are not number systems. This program builds their
mathematics from axioms upward — what a latent space is, which relations and
operations are meaningful — and derives definitions and propositions from
there, Euclid-style, rather than porting ℝⁿ mathematics onto embedding vectors.
Classical goals carry over (measure closeness, update on evidence, compose,
infer, prove); the native constructs may look nothing like their classical
counterparts.

- Current state: [STATE.md](STATE.md) · running log: [NOTEBOOK.md](NOTEBOOK.md)

## Status: PROGRAM CLOSED (2026-08-31)

The discovery program is **closed**. Across 50 audited experiments and 4 rounds
of Codex direction dialogue, no tested construction established native latent
mathematics — every proposed object (separator languages, executable
differences, cell complexes, internal port sufficiency) mapped to existing
established mathematics (coalgebra, PSR, automata, causal abstraction). The
terminal experiment (PSQ-3α: task-trained Qwen3-1.7B one-action intervention)
returned NO_INTERFACE (69.14% accuracy, gate ≥95%).

### Deposits

**Theory.** `theory/AXIOMS.md` carries an adopted foundation — future-response
geometry (D1–D9, Theorems 1/4/7/8): identity as zero future-response distance,
nonexpansive legal moves, surgeon-vs-denizen refinement, and the Robust
Port-Compression Conjecture (unaudited closing deposit: no fixed layer's
newest-token residual realizes the full append-action process once prefix
complexity exceeds d).

**Five transferable insights:**
1. A model has many operational latent spaces — indexed by (actions, observations, horizon).
2. Information ≠ state: state has three gates (present → addressable → composable).
3. The right null is the system's cheapest native mechanism, not random features.
4. A quotient must be earned by a transport — held-out presentation ≠ quotient-level generalization.
5. Absence requires a collision witness (same carrier, different future), not a failed probe.

**Empirical record.** Across two days of audited CPU experiments (2026-08-27 →
2026-08-30), no tested construction established a transferable causal state.
The durable empirical result is a localized set of instrument-validity,
source-extraction, actuator, and prompt/output-geometry constraints that any
future substrate must satisfy. Orientation: the
[handoff](docs/HANDOFF_2026_08_30.md), the canonical [current state](STATE.md),
and the [structured negative](docs/STRUCTURED_NEGATIVE_2026_08_29.md).
Append-only provenance is in the [experiment ledger](experiments/ledger.jsonl).

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
