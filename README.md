# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project is building the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands. Like Euclid building geometry by asking what axioms a flat plane requires, we ask: what axioms does a latent space require? What are its native notions of *place*, *move*, *distance*, and *composition*?

## The headline

**Cosine similarity — the standard tool for comparing neural network representations — is blind to the model's actual behavioral structure.** Not imprecise. *Blind.* It inverts the ranking: states cosine calls most similar are the most behaviorally different. What does see the structure? The model's own output distributions, measured through logit lens + Jensen-Shannon divergence.

Using this instrument across 18 audited experiments, we've found that a small language model maintains a **structured behavioral algebra** entirely invisible to ℝⁿ metrics:

| Finding | Number | What it means |
|---------|--------|---------------|
| **Peak selective amplification** | 62× | At layers 21-25, the model amplifies the queried fact's signature 62× while suppressing irrelevants to near-zero |
| **Cosine at the same point** | 0.98 | Standard metric sees "nearly identical" where the model sees "completely different" |
| **Greedy congruence** | 97% | Same-place histories produce the same next-token prediction — the argmax algebra is real |
| **Distributional congruence** | 0% | But the full output distributions *always* differ — every greedy fiber contains distinguishable states |
| **Commitment bottleneck** | 0.05 bits | Entropy drops to near-zero at L25 (the model fully commits), then re-broadens to 5.5-7.7 bits |
| **Synchronization idempotence** | 100% | Both S^W and S^G literal signature renderers are greedy-idempotent on the tested carrier (textual echo not ruled out; terminal anti-echo test inconclusive due to weak counterfactual interface) |
| **Sequence/path dependence** | 70.8-89.6% | Two append sequences asserting the same corrected world produce different response laws — order matters |

## What we've found

### The resolution layer

Between layers 21 and 25 (of 28), the model selectively amplifies the behavioral signature of the queried fact while suppressing all irrelevant facts to near-zero. Cosine similarity stays above 0.91 throughout — it cannot see any of this.

- **62× peak selectivity** (PLIM/KROT, L25)
- **Not attention routing** — attention weights show no selectivity (r < 0.25); attention to the queried entity is high at ALL layers
- **Whole-sequence operation** — the resolution signal is distributed across all input positions, not concentrated at the queried entity
- **Multi-fact generalization** — in 3-fact worlds, all irrelevant facts are suppressed equally and simultaneously

### The commitment bottleneck

Tracking Shannon entropy through all 28 layers reveals a dramatic structural phenomenon: the model funnels through a near-deterministic bottleneck at L24-25 (entropy ≈ 0.05 bits, top-1 mass ≈ 0.999), then re-broadens the distribution to 5.5-7.7 bits in the final output.

This explains two otherwise contradictory findings:
- **Greedy congruence (97%)** holds because the commitment determines the argmax
- **Distributional congruence (0%)** fails because the re-broadened distribution encodes history-dependent structure

The re-broadened distribution is not noise — the tokens with the largest probability differences between same-place histories are overwhelmingly history-related entity values. The model leaks its entire fact-world into every output distribution.

### The behavioral algebra

The model supports a **coarse partial action algebra** of greedy commitments:

- **Places** are greedy answer profiles (which entity gets which answer)
- **Moves** are typed continuations: empty, neutral, correction, restatement
- **Place preservation** is near-total for identity-like operations (100% empty, 95% neutral/restatement) and genuinely state-changing for corrections (35%)
- **Synchronization** via literal signature restatement is approximately idempotent: JSD(S, S²) ≈ 0.07, 100% greedy idempotence (textual echo not ruled out — a terminal 2304-pass anti-echo factorial found the counterfactual-record interface too weak to answer the question)
- **Two renderers:** S^W (from experimenter-known world) and S^G (from the model's own observable greedy answers). Both are literal append operators; neither has been demonstrated as a semantic canonicalizer. The terminal Phase 4d factorial found a telling asymmetry: the model follows confirming records (~98%) but largely ignores contradicting ones (~17-44%).

The two append sequences yield different response laws despite ending with the same per-entity declared values (JSD distance ~0.20, greedy commutativity 70.8-89.6%), establishing sequence/path dependence under the tested operations. This does not rule out ordinary textual order or multiplicity effects — the two paths differ in assertion order, multiplicity, and token distance.

A new finding: **correction itself does not reliably descend to the quotient** (58-80%). The same correction is ignored by some presentation orders and accepted by others — the fiber's distributional residual (invisible to argmax) affects how the model responds to further operations.

All results generalize to held-out entities the model has never seen in the training prompts.

### The nine breakpoints (Phase 1)

Across 50+ earlier experiments, we catalogued nine places where ℝⁿ mathematics fails in latent space. Each is a constraint on what native math must look like.

| # | Breakpoint | What it means |
|---|-----------|---------------|
| 1 | **Presence ≠ causation** | A concept can be perfectly decodable yet have zero causal effect. Linear probes find ghosts. |
| 2 | **Single-site ≠ distributed** | Facts are distributed properties of entire layer transformations. Patching one site does nothing; patching the whole state changes everything. |
| 3 | **Vector distance ≠ semantic distance** | Points close in cosine can be functionally opposite. |
| 4 | **Fixed dimensions ≠ fixed structure** | Effective dimensionality changes with context and task. |
| 5 | **Vector composition ≠ computational composition** | The model composes through its forward pass, not through vector arithmetic. |
| 6 | **Observation ≠ state** | The act of choosing what to probe constrains what you can find. |
| 7 | **Snapshot ≠ computation** | A representation at layer *l* can't be understood without the trajectory through all layers. |
| 8 | **ℝⁿ tools find ℝⁿ structure** | PCA finds linear structure because PCA *is* linear structure. The measurement imposes itself on the answer. |
| 9 | **Metric blindness to composition** | Four fact-worlds with cosine ≈ 1.000 produce dramatically different behavioral outcomes under intervention. |

Full details: [`theory/BREAKPOINT_REGISTRY.md`](theory/BREAKPOINT_REGISTRY.md)

## Theoretical framework

We're building axioms for latent space the way a denizen of that world would: not importing geometry from outside, but asking what mathematical structures are needed to *navigate*.

**The five navigation requirements:**

1. **Identity** — when have I returned to the same place? (Not: when are two vectors close)
2. **Moves** — what interventions does this world permit? (Not: what vectors can I add)
3. **Cost** — what effort does a move require? (Not: what's the Euclidean distance)
4. **Map** — can I predict consequences of moves I haven't made? (Not: can I interpolate)
5. **Laws** — what regularities hold across regions? (Not: what's the basis)

The formal development is in [`theory/AXIOMS.md`](theory/AXIOMS.md).

## Repository structure

```
theory/               Axioms, breakpoint registry, formal constructions
experiments/           All experiment code (one file per experiment)
  ledger.jsonl         Machine-readable experiment log
  results/             Raw outputs, JSON artifacts
  EXPERIMENTS.md       Human-readable experiment summaries
docs/                  Handoff documents, structured negatives
legacy/                Prior program (archived, unmodified)
STATE.md               Canonical current state of all claims
NOTEBOOK.md            Reverse-chronological running log
```

## Current status

**Phase 2** (active). Phase 1 (50 experiments, 2026-08-27 → 2026-08-31) established the nine breakpoints and the ℝⁿ trap. Phase 2 (18 experiments, 2026-08-31) builds genuinely non-ℝⁿ instruments and discovers the behavioral algebra. The terminal Phase 4d anti-echo factorial (2304 forward passes) closed the renderer-tuning program: record-append confirms but cannot contradict, so textual echo remains unresolved. S^G is locked as a literal append operator.

The central empirical claim (Codex-audited): *In a bounded three-fact prompt world in one small language model, greedy answer signatures form an approximate behavioral quotient with nontrivial predictive fibers. Two literal signature renderers — one from the hidden world (S^W) and one from the model's own observable greedy answers (S^G) — are both greedy-idempotent (textual echo not ruled out; a terminal 2304-pass anti-echo factorial found the counterfactual-record interface too weak to answer the question). The two append sequences yield different response laws despite ending with the same per-entity declared values, establishing sequence/path dependence under the tested operations; this does not rule out ordinary textual order or multiplicity effects.*

Current state: [`STATE.md`](STATE.md) · Running log: [`NOTEBOOK.md`](NOTEBOOK.md) · Phase 1 handoff: [`docs/HANDOFF_2026_08_30.md`](docs/HANDOFF_2026_08_30.md)

## Methodology

Every claim follows a strict evidence protocol:

- **Codex-audited.** An independent AI reviewer adversarially checks every result for overclaims, instrument artifacts, and alternative explanations. Claims are adopted only in auditor-licensed language.
- **Negative results are first-class.** Failed experiments are logged permanently and shape future directions. We've withdrawn prior claims when controls revealed artifacts.
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the git commit, command, config, and metrics.

## Prior work and corrections

The previous program (LLM embedding perturbation, diffusion latent repair) is archived under [`legacy/`](legacy/). Its nested-arithmetic claims were **withdrawn** after independent controls showed the benchmark measured termination under a token cap, not arithmetic capability. Full record: [`legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`](legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).

## Contributing

This is early-stage mathematical research. We're looking for people excited about:

- **Mechanistic interpretability** — especially if you've hit the limits of linear probes and want something deeper
- **Abstract algebra / category theory** — we need mathematical structures that aren't vector spaces
- **Causal inference** — our instruments are causal interventions on neural network internals
- **Philosophy of mathematics** — seriously: what *kind* of mathematical object is a latent space?

Start by reading the [breakpoint registry](theory/BREAKPOINT_REGISTRY.md) — each breakpoint is an open problem. If one excites you, open an issue.

## License

MIT
