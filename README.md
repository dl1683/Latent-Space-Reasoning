# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project explores the **native mathematics of latent spaces**: take the space as a world and ask what a denizen would need to invent to navigate it. What constitutes identity, a move, effort, a map or a reusable law? Native means coordinate-independent and effective for the medium; existing mathematics is a foundation to use. This philosophy is broader than any one research direction.

## Current direction: cumulative capability acquisition

**Can a persistent AI system acquire capabilities from successive pretrained models, combine them, and preserve what it already knows?**

This is the current application of the philosophy, not a replacement for it. We are exploring:

- **Capability mapping and grafting:** identify useful computations and their interfaces, then investigate combining language, domain and multimodal capabilities across models. Capabilities need not occupy isolated regions. Reducing unnecessary verbosity is another application when quality is preserved and total cost falls.
- **Reasoning over encoders:** use pretrained representations through reusable updates and composition, separating the encoder's contribution from what newly attached reasoning machinery learns.
- **Growth across model generations:** integrate additional encoders and representations from decoder-only models, with diffusion and multimodal donors as prospective extensions. The persistent representation may itself expand rather than force every contribution into one fixed vector space.

**Connected growth** retains donor models at runtime. **Assimilated growth** retains an acquired capability after donor removal. Neither alignment nor adding models establishes useful growth by itself. A token embedding table is not a model's contextual knowledge and computation; exporting a backbone does not automatically make it small or cheap.

Candidate evidence starts with A in a base system, B in a donor, acquisition of useful B behavior with A retained, and held-out tasks requiring both. Further acquisitions test cumulative growth; donor removal is a separate assimilation branch. Relevant controls include simple routing, direct evidence forwarding, structured readouts, retrieval and matched learning budgets. Acquisition and recurring inference costs are counted separately. Current work is autonomous strategy/discovery; definitions, sources and bounded probe results live in the canonical direction's strategy notebook.

**Current research allocation:** investigate useful successive acquisition, with
ordinary teaching, reusable records and connected growth as serious alternatives.
S50/S51's exact local episode is closed; S53's reaction-request execution is
deferred after its bounded access check. S52 retains the later-use distinction,
and S54/S55's source lookups are closed with their evidence limits preserved.
S58 adds a bounded local positive: correct specialist pseudo-labels improve a
species head over frozen recipient features after donor removal. Soft targets
do not beat hard labels under the locked recipe. These findings do not establish
new backbone knowledge, cumulative acquisition or a cost advantage.
No model run or dependency-repair campaign is queued. Follow the
[current strategy boundary](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md#next-work-boundary).

**Status: exploratory direction.** No cross-model grafting, cumulative acquisition or encoder-reasoning artifact has been established. Start with the [canonical direction](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md) and [current state](STATE.md). Earlier findings below remain supporting evidence with their original limits.

## Research history and continuity

The [prior program](legacy/README.md) explored latent perturbation, diffusion repair and aggregation, including the permanent arithmetic-claim correction. The native-mathematics program then investigated frozen-encoder geometry, residual dynamics, causal interventions, learned carriers, behavioral quotients and continuation laws. Its conditional mathematics, bounded positive findings and structured negatives remain part of the foundation; closed lines are not silently reopened. [Experiment history](experiments/EXPERIMENTS.md) · [Mathematical foundation](theory/AXIOMS.md) · [Structured negative](docs/STRUCTURED_NEGATIVE_2026_08_29.md)

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
