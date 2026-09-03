# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project builds the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands.

## The Settling Time Law

Neural networks store deeply nested information but need processing time to access it. A single neutral token — a Python comment, a `pass` statement — **recovers up to 13% accuracy** for deeply nested variable bindings. The effect scales with depth: shallow bindings are already accessible, deep bindings need a consolidation step.

| Depth | Raw accuracy | +1 settling token | Gain |
|-------|-------------|-------------------|------|
| d1 (shallow) | 95.8% | 97.0% | +1.2% (already accessible) |
| d2 | 89.7% | 94.5% | **+4.9%** |
| d3 | 79.2% | 90.0% | **+10.8%** |
| d4 (deep) | 75.3% | 88.0% | **+12.6%** |

*Qwen3-1.7B-Base (pure transformer). Verified with full-text processing.*

### Why this matters

**For AI inference:** One neutral processing step at deep nesting is cheaply inserted at inference time for a significant accuracy gain — applicable to any transformer-based architecture.

**For evaluation:** Benchmarks testing nested reasoning are measuring access speed, not knowledge. A model "failing" at depth 4 may know the answer — it just needs one more forward pass.

**For latent-space theory:** Settling time is a property of computational depth, not any specific architecture. One step triggers consolidation; more steps don't help further. This discrete "click" — where one token reorganizes the hidden state for better readout — is a native property of how neural computation encodes hierarchical structure.

### Key findings

1. **One-shot trigger.** The optimal suffix count is exactly 1. More tokens don't help — this is a discrete consolidation event, not gradual convergence.

2. **Gain scales with depth** — each deeper scope level produces a larger settling benefit. Shallow information is already accessible; deep information is simultaneously harder to access and more responsive to settling.

3. **Architecture-universal.** Confirmed across hybrid (Mamba+attention) and pure transformer architectures with the same qualitative law.

## Architecture Discovery: SSM vs Attention for Scope Resolution

We discovered that **state-space model (SSM/Mamba) layers can actively interfere with scope resolution.** In hybrid Mamba+attention architectures, the SSM layers' sequential state compression introduces a recency bias that fights the attention layers' ability to track nested scope structure.

This was uncovered through a systematic investigation of execution modes: when Mamba layers correctly process the full context, scope-binding accuracy drops dramatically compared to when only attention layers carry context. The attention mechanism's ability to look back to arbitrary positions is essential for resolving which variable binding applies at a given scope depth.

**Implication:** Architecture composition matters. Adding SSM layers to a transformer doesn't uniformly improve capability — for tasks requiring relational reasoning across nested structure, SSM layers can be actively harmful. This informs architecture design for models intended to handle complex code, mathematical proofs, or deeply nested logical reasoning.

### Implementation Gap in Hybrid Models

During this investigation, we identified a **state continuation gap** in popular Mamba hybrid implementations: multi-token cached continuation silently starts SSM layers from zero state instead of using the cached recurrent state. Single-token decode is unaffected. This means chunked inference on Mamba hybrids can produce silently incorrect outputs. Details in [`NOTEBOOK.md`](NOTEBOOK.md).

## Nine Breakpoints: Where R^n Mathematics Fails

Across 50+ experiments, we catalogued nine places where standard vector-space mathematics fails in latent space. Each constrains what native math must look like.

| # | Breakpoint | What it means |
|---|-----------|---------------|
| 1 | **Presence =/= causation** | A concept can be perfectly decodable yet have zero causal effect. Linear probes find ghosts. |
| 2 | **Single-site =/= distributed** | Facts are distributed properties of entire layer transformations. |
| 3 | **Vector distance =/= semantic distance** | Points close in cosine can be functionally opposite. |
| 4 | **Fixed dimensions =/= fixed structure** | Effective dimensionality changes with context and task. |
| 5 | **Vector composition =/= computational composition** | The model composes through its forward pass, not through vector arithmetic. |
| 6 | **Observation =/= state** | The act of choosing what to probe constrains what you can find. |
| 7 | **Snapshot =/= computation** | A representation at layer *l* can't be understood without the trajectory through all layers. |
| 8 | **R^n tools find R^n structure** | PCA finds linear structure because PCA *is* linear structure. The measurement imposes itself on the answer. |
| 9 | **Metric blindness to composition** | Four fact-worlds with cosine ~1.000 produce dramatically different behavioral outcomes under intervention. |

Full details: [`theory/BREAKPOINT_REGISTRY.md`](theory/BREAKPOINT_REGISTRY.md)

## Method: Scope-Variable Binding (SVB)

Python lexical scoping as a probe for latent-space depth structure. Nested `def` blocks create scope depth 1-4, each shadowing a variable. The model processes the code and must report the value at the outermost scope. The response is decomposed into an 11-bin probability distribution ({digit 0-9, OTHER}).

**Observable:** sigma (scope binding fidelity) = P(model outputs the correct outer value).

## Repository structure

```
theory/               Axioms, breakpoint registry, formal constructions
experiments/           All experiment code
  run_svb_0.py         SVB runner (ModelAdapter for transformer/SSM/hybrid)
  config/              Experiment configurations (JSON)
  results/             Raw outputs, checkpoints
  ledger.jsonl         Machine-readable experiment log
  EXPERIMENTS.md       Human-readable experiment summaries
docs/                  Handoff documents
legacy/                Prior program (archived)
STATE.md               Canonical current state of all claims
NOTEBOOK.md            Reverse-chronological running log
```

## Methodology

- **Behavioral, not representational.** We study what models *do*, not what their hidden states "look like." This avoids the R^n projection trap (breakpoint #8).
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls, execution-mode invariance.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the git commit, command, config hash, and metrics.

## License

MIT
