# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project is building the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands. Like Euclid building geometry by asking what axioms a flat plane requires, we ask: what axioms does a latent space require? What are its native notions of *place*, *move*, *distance*, and *composition*?

## Why this matters

Mechanistic interpretability treats latent space as ℝⁿ — a bag of numbers you can probe with linear algebra. This works surprisingly well for finding features. But it misses something fundamental: **the model doesn't think in vectors. It thinks in transformations.** The vector is a snapshot; the computation is the object.

We have 50+ audited experiments showing exactly where ℝⁿ mathematics breaks down inside real models — nine documented breakpoints where standard tools give confident wrong answers, or miss structure the model actively uses. These aren't edge cases. They're systematic blind spots in how we understand AI internals.

**The ambition:** build mathematical tools that see what cosine similarity, PCA, and linear probes cannot. If we succeed, we change how the field understands, controls, and aligns AI systems.

## What we've found

### The nine breakpoints

Across 50+ experiments on Qwen3 models, we've catalogued nine places where ℝⁿ mathematics fails in latent space. Each one is a clue about what native math must look like.

| # | Breakpoint | What it means |
|---|-----------|---------------|
| 1 | **Presence ≠ causation** | A concept can be perfectly decodable from activations yet have zero causal effect. Linear probes find ghosts. |
| 2 | **Single-site ≠ distributed** | Facts aren't stored at locations — they're distributed properties of entire layer transformations. Patching one site does nothing; patching the whole state changes everything. |
| 3 | **Vector distance ≠ semantic distance** | Points close in cosine can be functionally opposite. The metric structure of ℝⁿ doesn't reflect meaning. |
| 4 | **Fixed dimensions ≠ fixed structure** | The effective dimensionality of a representation changes with context and task. There is no fixed *d*. |
| 5 | **Vector composition ≠ computational composition** | The model composes through its forward pass, not through vector arithmetic. Addition and concatenation are human projections. |
| 6 | **Observation ≠ state** | The act of choosing what to probe constrains what you can find. Instruments aren't neutral. |
| 7 | **Snapshot ≠ computation** | A model's representation at layer *l* can't be understood without the trajectory through all layers. The object is a path, not a point. |
| 8 | **ℝⁿ tools find ℝⁿ structure** | PCA finds linear structure because PCA *is* linear structure. The measurement imposes itself on the answer. |
| 9 | **Metric blindness to composition** | Four fact-worlds with cosine similarity ~1.000 produce dramatically different behavioral outcomes under intervention. ℝⁿ distance sees "same" where the computation sees "different." |

Full details with experimental evidence: [`theory/BREAKPOINT_REGISTRY.md`](theory/BREAKPOINT_REGISTRY.md)

### Fusion-fission: facts become entangled

Our most striking ongoing investigation. Store two independent facts in a language model ("HESK is red" and "VORN is blue"). Ask about one. At which layers are the facts independently controllable, and at which do they become computationally entangled?

**Method:** Causal transfer matrices. Run a donor model with altered facts, capture its internal state at each layer, transplant only the fact-storage positions into a host model, and measure how both answers shift. This gives a 2×2 matrix **K** at every layer:

```
K_l = [[dA/dA, dA/dB],
       [dB/dA, dB/dB]]
```

where `dX/dY` = how much answer-X shifts when fact-Y is changed via transplant.

**What the K matrix reveals:**
- **Strong diagonal, weak off-diagonal** → facts are independently addressable (SEPARATE)
- **Strong off-diagonal** → changing one fact inevitably changes the other (FUSED)
- **Both weak** → the transplant has no effect at this layer (NO_CONTROL)

This is a continuous, causal, layer-by-layer map of how factual representations interact during computation — something no static metric can see.

*Status: instrument validated (zero self-patch noise after BPE boundary fix), config-dependent patterns observed, cross-config replication under investigation.*

### The three-gate model

One of our most useful theoretical insights. Information in a latent space passes through three gates:

1. **Present** — the information exists somewhere in the activation (a linear probe can find it)
2. **Addressable** — the model can read and use it (causal intervention affects output)
3. **Composable** — the model can combine it with other information to produce novel outputs

Most interpretability work tests gate 1. Gates 2 and 3 are where the real action is — and where ℝⁿ tools break down.

## Theoretical framework

We're building axioms for latent space the way a denizen of that world would: not importing geometry from outside, but asking what mathematical structures are needed to *navigate*.

**The five navigation requirements** (what a latent-space denizen must define):

1. **Identity** — when have I returned to the same place? (Not: when are two vectors close)
2. **Moves** — what interventions does this world permit? (Not: what vectors can I add)
3. **Cost** — what effort does a move require? (Not: what's the Euclidean distance)
4. **Map** — can I predict consequences of moves I haven't made? (Not: can I interpolate)
5. **Laws** — what regularities hold across regions? (Not: what's the basis)

The formal development is in [`theory/AXIOMS.md`](theory/AXIOMS.md). Current directions include typed latent actions (partial-action categories over behavioral response classes), predictive-state algebra, and gauge-transported operator algebras.

## Repository structure

```
theory/               Axioms, breakpoint registry, Codex dialogue transcripts
experiments/           All experiment code (one file per experiment)
  ledger.jsonl         Machine-readable experiment log (394 entries)
  results/             Raw outputs, JSON artifacts
  EXPERIMENTS.md       Human-readable experiment summaries
docs/                  Handoff documents, structured negatives
legacy/                Prior program (archived, unmodified)
STATE.md               Canonical current state of all claims
NOTEBOOK.md            Reverse-chronological running log
```

## Current status

**Phase 2** (active). Phase 1 (50 experiments, 2026-08-27 → 2026-08-31) established the nine breakpoints and the ℝⁿ trap. Phase 2 builds genuinely non-ℝⁿ instruments and theory.

Active threads:
- **Fusion-fission instrument** — causal transfer matrices across layers, configs, and models
- **Typed latent actions** — partial-action categories derived from causal response laws
- **Breakpoint exploitation** — each breakpoint is a constraint on what native math must look like

Current state: [`STATE.md`](STATE.md) · Running log: [`NOTEBOOK.md`](NOTEBOOK.md) · Phase 1 handoff: [`docs/HANDOFF_2026_08_30.md`](docs/HANDOFF_2026_08_30.md)

## Methodology

Every claim follows a strict evidence protocol:

- **Codex-audited.** An independent AI reviewer (OpenAI Codex CLI) adversarially checks every result for overclaims, instrument artifacts, and alternative explanations. Claims are adopted only in Codex-licensed language.
- **Negative results are first-class.** Failed experiments are logged permanently and shape future directions. We've withdrawn prior claims when controls revealed artifacts.
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls. The v1→v2 fusion-fission arc is a case study — five instrument defects were found and fixed before any interpretive claims.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the git commit, command, config, and metrics.

## Prior work and corrections

The previous program (LLM embedding perturbation, diffusion latent repair) is archived under [`legacy/`](legacy/). Its nested-arithmetic claims were **withdrawn** after independent controls showed the benchmark measured termination under a token cap, not arithmetic capability. Full record: [`legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`](legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md). One finding carries forward: greedy decoding determinism is hardware-dependent, so any diversity claim must report the numerical noise floor.

## Contributing

This is early-stage mathematical research. We're looking for people excited about:

- **Mechanistic interpretability** — especially if you've hit the limits of linear probes and want something deeper
- **Abstract algebra / category theory** — we need mathematical structures that aren't vector spaces
- **Causal inference** — our instruments are causal interventions on neural network internals
- **Philosophy of mathematics** — seriously: what *kind* of mathematical object is a latent space?

Start by reading the [breakpoint registry](theory/BREAKPOINT_REGISTRY.md) — each breakpoint is an open problem. If one excites you, open an issue.

## License

MIT
