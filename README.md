# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project builds the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands.

## A Depth-Dependent Readout Effect

In one Qwen3-1.7B prompt family, deeply nested bindings were less likely to be read out correctly, and one fixed appended comment — `# No changes.` — **raised mean correct-value probability by up to 12.6 percentage points**. The observed change grew with nesting depth on this panel.

| Depth | σ (no suffix) | σ (+1 suffix) | Gain (pp) |
|-------|--------------|---------------|-----------|
| d1 (shallow) | 0.958 | 0.970 | +1.2 (already accessible) |
| d2 | 0.897 | 0.945 | **+4.9** |
| d3 | 0.792 | 0.900 | **+10.8** |
| d4 (deep) | 0.753 | 0.880 | **+12.6** |

*Qwen3-1.7B-Base (pure transformer). σ = mean P(correct value) over n=27 cells per condition (3 variable names × 9 values). Suffix: `# No changes.\n` appended once.*

### Why this matters

**For AI inference:** One appended fixed comment line produced a meaningful response gain at deep nesting in this model and prompt family. That makes suffix-conditioned readout a concrete inference control to test across transformer systems; transfer, latency, and end-to-end cost remain open measurements.

**For evaluation:** The result shows that a nested-reasoning score can depend materially on the text immediately before readout. It does not establish that the answer was already encoded or that extra computation alone caused the gain; it makes those mechanisms experimentally separable.

**For latent-space theory:** The measured response is depth-dependent and suffix-sensitive. Ongoing research is separating lexical cueing, position, added computation, and attention-state change before testing whether a native transition law survives across templates, tasks, and architectures.

### Key findings

1. **Content-driven, not surface-driven.** A 2×2 crossover experiment separating semantic content from surface features shows content drives 6.3× more effect than variable-mention confounds. The model distinguishes what a comment *means*, not just its syntactic form.

2. **Well-defined operators on a probability simplex.** Lumpability analysis (R² = 0.79–0.98 across 7 role types) demonstrates that suffix operations act as approximately well-defined transformations on the (correct-digit, shadow-digit, residual) probability simplex. This is the coordinate system native to the model's computation.

3. **Single-count maximum with depth scaling.** Among tested suffix counts {0, 1, 2, 4}, suffix count 1 yields the highest σ at all depths d2–d4. The absolute gain rises from 1.2 pp at d1 to 12.6 pp at d4.

4. **Operator algebra is finer than semantic role.** Different surface texts within the same semantic class produce detectably different operators (within-class coefficient of variation 0.33–0.51). The model distinguishes paraphrases that humans consider equivalent.

5. **Robust order dependence under investigation.** Applying suffix A then suffix M produces a measurably different output distribution than M then A (median TV = 0.068). Filler-based controls confirm the direction of the order effect but a nonlinear position-gain rival remains open. Multi-filler robustness and held-out prefix tests are in progress.

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
