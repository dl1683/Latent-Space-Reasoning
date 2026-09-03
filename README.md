# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project builds the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands.

## The Settling Time Law

Neural networks store deeply nested information but need processing time to access it. A single neutral token — a Python comment, a `pass` statement — **nearly doubles accuracy** for deeply nested variable bindings. This effect is universal across architectures and scales linearly with depth.

| Depth | Raw accuracy | With 1 settling token | Gain |
|-------|-------------|----------------------|------|
| d1 (shallow) | 0.681 | 0.669 | -2% (already accessible) |
| d2 | 0.497 | 0.645 | **+30%** |
| d3 | 0.280 | 0.430 | **+54%** |
| d4 (deep) | 0.229 | 0.431 | **+88%** |

*Falcon-H1-1.5B-Instruct (hybrid Mamba+attention). Same law confirmed on Qwen3-1.7B-Base (pure transformer) with ~5x smaller magnitude but identical depth-scaling structure.*

### Why this matters

**For AI inference:** Every architecture benefits from settling tokens at deep nesting. One neutral processing step is cheaply inserted at inference time for a significant accuracy gain — applicable to GPT-style transformers, Llama, Qwen, and anything built on attention.

**For evaluation:** Benchmarks testing nested reasoning are measuring access speed, not knowledge. A model "failing" at depth 4 may know the answer — it just needs one more forward pass. This reframes how we evaluate nested reasoning.

**For latent-space theory:** Settling time is architecture-invariant. It's not a property of attention or recurrence — it's a property of depth itself, native to how neural computation encodes hierarchical structure. This is the first confirmed candidate for a universal law of latent-space geometry.

### Key findings

1. **One-shot trigger.** The optimal suffix count is always exactly 1 (confirmed at 7-point resolution [0,1,2,3,4,6,8]). More tokens don't help — this is a discrete consolidation event, not gradual processing.

2. **Gain scales linearly with depth** at ~30% per scope level. Each depth level's settling recovers performance approximately one level shallower.

3. **Python-specific trigger.** Python structural tokens (docstrings > comments > `pass` > `assert True`) are effective; C++ comments and bare newlines are not. The model responds to Python markers learned during pretraining.

4. **Bandwidth scales with depth.** Deep bindings tolerate more settling tokens (d3 benefits from s1-s8; d1 is damaged by any suffix). Deep information is simultaneously harder to access and more robust to perturbation.

5. **Architecture-universal.** Confirmed across hybrid (Falcon-H1, Mamba+attention) and pure transformer (Qwen3). Same qualitative law, different magnitudes.

6. **Anti-settling.** Suffix tokens that introduce competing values (e.g., `# x = 0`) actively damage accuracy by 26-34%, scaling with depth. The attention pattern matters, not just the computation.

### Mechanism: idempotent consolidation

The settling effect behaves like an approximate projection operator: applying it once reorganizes the hidden state for better readout; applying it twice doesn't improve further (C(C(h)) ≈ C(h)). Different neutral token types trigger approximately the same consolidation. Non-neutral tokens trigger different projections that can interfere with the correct binding.

Full framework with testable predictions in [`NOTEBOOK.md`](NOTEBOOK.md).

## Method: Scope-Variable Binding (SVB)

Python lexical scoping as a probe for information depth. Nested `def` blocks create scope depth 1-4, each shadowing a variable. The model processes the code via DynamicCache state injection and must report the value at the outermost (deepest) scope. The response is decomposed into an 11-bin probability law ({digit 0-9, OTHER}).

**Observable:** sigma (scope binding fidelity) = P(model outputs the correct outer value).

**Experiments completed:**
- SVB-0: Baseline depth curve on Falcon-H1 (162 calls, 3 min)
- SVB-1: Extended depth 3-4 with suffix profiles (432 calls, 60 min)
- SVB-2: Fine-grained 7-point suffix resolution (945 calls, 43 min)
- SVB-Qwen3-Formal: Cross-architecture universality (621 calls, 3.3 min)
- 7 suffix mechanism probes: content, structure, format, optimality

## Prior work

### Nine breakpoints (Phase 1)

Across 50+ experiments, we catalogued nine places where R^n mathematics fails in latent space. Each is a constraint on what native math must look like.

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

### Corrections

The previous program (LLM embedding perturbation, diffusion latent repair) is archived under [`legacy/`](legacy/). Its nested-arithmetic claims were **withdrawn** after controls showed the benchmark measured termination under a token cap, not arithmetic capability. Full record: [`legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`](legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).

## Repository structure

```
theory/               Axioms, breakpoint registry, formal constructions
experiments/           All experiment code
  run_svb_0.py         SVB runner (ModelAdapter for transformer/SSM/hybrid)
  config/              Experiment configurations (JSON)
  results/             Raw outputs, checkpoints
  ledger.jsonl         Machine-readable experiment log
  EXPERIMENTS.md       Human-readable experiment summaries
docs/                  Handoff documents, structured negatives
legacy/                Prior program (archived, unmodified)
STATE.md               Canonical current state of all claims
NOTEBOOK.md            Reverse-chronological running log
```

## Methodology

- **Negative results are first-class.** Failed experiments are logged permanently and shape future directions. We've withdrawn prior claims when controls revealed artifacts.
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the git commit, command, config hash, and metrics.

## License

MIT
