# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project builds the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands.

## The Settling Time Effect

Neural networks store deeply nested information but don't always surface it on the first readout. A single appended comment line — `# No changes.` — **raises mean correct-value probability by up to 12.6 percentage points** for deeply nested variable bindings. The effect scales with nesting depth: shallow bindings are already accessible, deep bindings benefit most.

| Depth | σ (no suffix) | σ (+1 suffix) | Gain (pp) |
|-------|--------------|---------------|-----------|
| d1 (shallow) | 0.958 | 0.970 | +1.2 (already accessible) |
| d2 | 0.897 | 0.945 | **+4.9** |
| d3 | 0.792 | 0.900 | **+10.8** |
| d4 (deep) | 0.753 | 0.880 | **+12.6** |

*Qwen3-1.7B-Base (pure transformer). σ = mean P(correct value) over n=27 cells per condition (3 variable names × 9 values). Suffix: `# No changes.\n` appended once.*

### Why this matters

**For AI inference:** One appended neutral line at deep nesting is a cheap intervention for a meaningful accuracy gain. This is directly applicable to transformer-based systems processing nested code or structured reasoning.

**For evaluation:** If depth-dependent information requires a settling step to surface, benchmarks testing nested reasoning may be measuring readout difficulty rather than whether the model encoded the answer. A model scoring low at depth 4 may improve substantially with a single additional processing step.

**For latent-space theory:** The settling effect is depth-dependent and suffix-sensitive — a property of how the model's computation interacts with nesting structure. Ongoing research is testing whether this supports a native transition law and whether it transfers across templates, tasks, and architectures.

### Key findings

1. **One-shot peak.** Among tested suffix counts {0, 1, 2, 4}, suffix count 1 yields the highest σ at all depths d2–d4. Additional suffixes do not improve further.

2. **Gain scales with depth** — each deeper scope level produces a larger settling benefit. Shallow information is already accessible; deep information benefits most.

3. **Ongoing architecture investigation.** Research is testing whether the Qwen3 pattern generalizes across clean architectures, tasks, and prompt families.

## Nine Breakpoints: Design Constraints for Native Mathematics

Across 50 audited Phase 1 experiments, we catalogued nine places where standard vector-space mathematics fails in latent space. Each constrains what native math must look like.

| # | Breakpoint | What it means |
|---|-----------|---------------|
| 1 | **Presence =/= causation** | A concept can be strongly decodable yet have zero causal effect. Linear probes find ghosts. |
| 2 | **Single-site =/= distributed** | Specific interventions show whole-state dependence for factual recall. |
| 3 | **Vector distance =/= semantic distance** | Points close in cosine can be behaviorally different. |
| 4 | **Fixed dimensions =/= fixed structure** | Effective dimensionality may be context- and task-dependent. |
| 5 | **Vector composition =/= computational composition** | The model composes through its forward pass, not through vector arithmetic. |
| 6 | **Observation =/= state** | The act of choosing what to probe constrains what you can find. |
| 7 | **Snapshot =/= computation** | A representation at layer *l* can't be understood without the trajectory through all layers. |
| 8 | **R^n tools find R^n structure** | PCA finds linear structure because PCA *is* linear structure. The measurement imposes itself on the answer. |
| 9 | **Metric blindness to composition** | Four fact-worlds with cosine ~1.000 produce dramatically different behavioral outcomes under intervention. |

Full details: [`theory/BREAKPOINT_REGISTRY.md`](theory/BREAKPOINT_REGISTRY.md)

## Method: Scope-Variable Binding (SVB)

Python lexical scoping as a probe for latent-space depth structure. Nested `def` blocks create scope depth 1–4, each shadowing a variable. The model processes the code and must report the value at the outermost scope. The response is decomposed into an 11-bin probability distribution ({digit 0–9, OTHER}).

**Observable:** σ (scope binding fidelity) = P(model outputs the correct outer value).

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

- **Behavioral consequences are claim-bearing; representational measurements are diagnostic scaffolding.** We study what models *do*, not what their hidden states "look like." This avoids the R^n projection trap (breakpoint #8).
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls, execution-mode checks.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the command, config, and metrics.

## License

MIT
