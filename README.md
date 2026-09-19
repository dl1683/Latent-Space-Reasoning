# Latent Space Reasoning

> *How does math change at high dimensions? The phenomena that emerge — concentration of measure, almost-orthogonal packing, interference, distance uniformity — are not curiosities but the native physics of these spaces.*

This project studies the **mathematics of high-dimensional spaces**: the geometry, algebra, and analysis that become qualitatively different when d >> 1. The scope spans neural network representations, traditional machine learning, and pure high-dimensional mathematics — any domain where high-dimensional structure determines behavior.

LLM latent spaces (d=768–4096+), kernel feature maps, embedding spaces, and high-dimensional statistical models all operate in regimes where low-dimensional intuition fails. Understanding *how* it fails — and what replaces it — is the research question.

## Empirical foundation: perturbation sensitivity in language models

86 experiments across 10 language models established a consistent hierarchy in how code perturbations affect model output:

**Position >> operator family >> sign >> answer identity >> meaning.**

1. **WHERE you modify code is the dominant factor.** After-expression edits (comments, whitespace, dead code) produce consistent, reliable effects. Expression-adjacent modifications (parenthesization, type casts, identity operations) produce heterogeneous, cell-specific effects. The reliability gap between position types is large (AA ≈ 0.7–0.9 vs DD ≈ 0.1–0.5) and universal across tested models.

2. **Sensitivity has multiplicative structure.** Behavioral distance decomposes as d[cell, pert] ≈ fragility[cell] × magnitude[pert], with R² = 0.82–0.88 in log space. This is a first-order Taylor approximation (small-angle regime), with a 12–25% structured residual from cell × perturbation group interaction.

3. **Models agree on what kinds of perturbations matter, but not on which expressions are vulnerable.** Within-model perturbation-type ordering correlates at +0.84 within architecture families. Cross-model cell vulnerability correlations are near zero. The perturbation-level structure transfers; the cell-level structure is model-specific.

4. **Expression adjacency drives heterogeneity.** The same semantic operation (e.g., wrapping in parentheses) produces a reliable cell-ranking when placed on its own line (O-O ≈ 0.66) but an unreliable one when adjacent to the expression (D-D ≈ 0.39). Token-level interaction between modification and expression tokens is consistent with cell-specific, position-dependent noise (interaction vs syntactic novelty not isolated).

5. **Meaning went undetected on the tested arithmetic panels.** Within the same operator family, meaning-preserving vs meaning-changing perturbations produce indistinguishable 11-bin output distributions (effect sizes +0.002/+0.008/−0.090 across 3 architectures, all consistent with zero). The answer-identity confound makes meaning inseparable from answer change in this domain.

**Models tested:** Qwen3 (0.6B/1.7B, base and instruct), SmolLM2 (135M/360M/1.7B, base and instruct), Phi-1.5, StableLM-2-1.6B, CodeGen-350M. All experiments are CPU-only, deterministic, and fully logged.

See [current state](STATE.md) and [experiment history](experiments/EXPERIMENTS.md).

## Research history

1. **General mathematics of high-dimensional spaces** (current, ACTIVE per user ruling 2026-09-18): Analytical/theoretical exploration of how math changes when d >> 1. Scope: pure math, statistical physics, information theory, random matrix theory, kernel methods. Results so far: interference hierarchy scaffold (t-body terms suppressed by d^{-(t-1)/2}; rigorous general-degree proof pending), the sparse-exact-optimum theorem (the specified diagonal quadratic superposition autoencoder has a sparse-exact, flat-spectrum global optimum), estimator-validity constraints on measured spectral exponents, and the unigram-softmax rank-2 collapse result (corrected 2026-09-19: earlier "representation-computation gap" and "α* = s/2" claims were overclaims — no universal capacity law or eigenspectrum interpretation is established). The behavioral transfer data is exhausted (intrinsically rank-3), so open work is analytical/proof-first on genuinely high-dimensional objects with numeric predictions pre-registered before data is touched. LLM latent spaces remain a later instance of this program, revisited once the general foundation is built.
2. **Perturbation sensitivity** (empirical foundation): 86 experiments across 10 models characterizing the sensitivity hierarchy. Balanced factorial designs, 69 cells × 24 perturbation types, full-logit and 11-bin readouts.
3. **Spectral/frame analysis** (closed 2026-09-18): Tensor composition line killed by row-permutation argument. Power-law-like fitted slopes observed after Zipfian training — origin unresolved (estimator bias and equilibrium structure both implicated; see HP cycle). Information-theoretic eigenspace priority observed. Spectral steepening descriptive but not predictive.
4. **Omega modulus** (closed 2026-09-17): ~28,000 readouts across 5 architectures. Cross-architecture amplification confirmed; mechanism unresolved; cross-question transfer negative.
5. **Cumulative capability acquisition** (S1–S65, closed): explored cross-model grafting, reasoning over encoders. No artifact established.
6. **Prior program** ([legacy/](legacy/README.md)): latent perturbation, diffusion repair, aggregation. Produced the nine breakpoints below.

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

**For latent-space theory:** The measured response is depth-dependent and suffix-sensitive. The historical unresolved question is how to separate lexical cueing, position, added computation, and attention-state change before testing a transition law across templates, tasks, and architectures. This panel is deferred; current work follows the [strategy boundary](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md#next-work-boundary).

### Key findings

1. **The tested content contrast exceeds the variable-mention contrast.** In a 2×2 crossover, the assertion-versus-misleading contrast was 6.3× the variable-mention effect. This rules out variable mention alone as the explanation on that panel; it does not isolate meaning from every surface feature. Responses also differed between paraphrases, as described below.

2. **No stable operator algebra on the response simplex.** Lumpability of the (correct-digit, shadow-digit, residual) projection was high (R² = 0.79–0.98 across 7 role types), but the fitted stochastic-matrix operator did not survive composition: behavioral equivalence between surfaces was not preserved under a second suffix (amplification 2.6–13×), and a scalar logit-bias model fit the single-suffix data as well. The simplex operators are a diagnostic, not a native coordinate system.

3. **Single-count maximum with depth scaling.** Among tested suffix counts {0, 1, 2, 4}, suffix count 1 yields the highest σ at all depths d2–d4. The absolute gain rises from 1.2 pp at d1 to 12.6 pp at d4.

4. **Responses are surface-specific, not role-specific.** Different surface texts within the same semantic class produce detectably different response shifts (within-class coefficient of variation 0.33–0.51). The model distinguishes paraphrases that humans consider equivalent; no role-level operator was established.

5. **Order effects are positional, not compositional.** Applying suffix A then suffix M produces a measurably different output distribution than M then A (median TV = 0.068). Controlled experiments with three independent token-matched filler pairs confirm the order effect is real and consistent (cross-pair correlation 0.69–0.92), but a scalar gain fit absorbs it: the effect direction is positional and the magnitude is context-dependent. A later reading of the same filler data showed that equal-length, position-matched blocks do not produce equal effects either (differences above 0.002 in at least 23 of 24 roots for every filler), so "position" here means the relative offsets to earlier content together with the content of the appended block, not token count alone.

6. **Presentation differences survive a neutral continuation and are exposed by a reassignment announcement.** On a 24-cell panel with three same-meaning surface pairs, the neutral comment `# No changes.` kept all 32 initially close pairs within 0.01 total variation on the 11-bin readout, while the comment `# Reassigning {var} now.` separated 29 of them. This is a reduced-readout, unmatched-position observation about which continuations reveal which distinctions; it establishes nothing about erasure or about the full next-token law.

## Supporting foundation: which distinctions can a model safely forget?

The supporting theory studies interchangeability of histories relative to a family of future questions: two histories are interchangeable for a question family if every question in the family answers the same, and the organizing quantity is how much a later action can re-separate histories that were interchangeable before it. Results so far are mathematical and conditional: exact selective overwrite has a lens structure (preserved context times store); transition-closed realization sets give history-length-independent error bounds; a finite behavioral state budget of N places turns local update errors of size ε into a lifetime error of at most (N−1)ε, sharp; contraction is a sufficient but not necessary certificate. A trained 17K-parameter overwrite cell realized perfect fact replacement on its training panel without being a contraction on its state cube, which closed the coordinate-certificate route. No real-model artifact has been established. That prior line leaves open whether genuinely close real-model histories exist on a matched-position, full-law interface and which matched continuations separate them; it is not the current run queue.

## Nine Breakpoints: Design Constraints for Native Mathematics

The first nine breakpoint themes summarize constraints from the Phase 1 experiments. They are clues about tested interfaces and constructions, not universal failures of vector mathematics. The registry also contains later themes; the underlying audited records control their scope.

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
theory/               Axioms, breakpoint registry, continuation-refinement calculus (CR-1 to CR-16)
  closed/             Theory of closed lines (archived, headers mark status)
experiments/           Core experiment runners and bounded strategy probes
  config/              Experiment configurations (JSON)
  results/             Result files (small; every result.json kept; bulk dumps listed in MANIFEST.md)
  ledger.jsonl         Machine-readable experiment log
  EXPERIMENTS.md       Human-readable experiment summaries, newest first
docs/                  Archived state, notebook, and handoffs
legacy/                Prior program (archived)
STATE.md               Canonical current state of all claims
NOTEBOOK.md            Reverse-chronological running log (recent); older entries in docs/
```

## Methodology

- **Behavioral consequences are claim-bearing; representational measurements are diagnostic scaffolding.** We study what models *do*, not what their hidden states "look like." This avoids the R^n projection trap (breakpoint #8).
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls, execution-mode checks.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the command, config, and metrics.

## License

MIT
