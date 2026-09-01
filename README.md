# Latent Space Reasoning

> *Every neural network has a vast mathematical world inside it. We treat it as ordinary vector space and apply linear algebra. But what if it has its own mathematics — structure that exists, that the model uses, and that our standard tools literally cannot see?*

This project attempted to build the **native mathematics of latent spaces** — not porting existing math onto embeddings, but discovering what math the space itself demands.

## Status: empirical program closed (2026-09-01)

After 75+ experiments across 5 phases, two eligibility screens, one terminal composition result, and multiple independent Codex adversarial audits, the empirical program is closed. **No positive scientific result survives audit.** The theoretical framework (axioms D1–D9, Theorems 1/4/7/8) is sound standard mathematics. One bounded theory attempt (composition-identifiability theorem) remains; if it fails its kill gate, the program archives.

The central hypothesis — "latent spaces have native mathematics" — is **not refuted** but is **unsupported** by this work. The experiments failed at eligibility (models couldn't provide a behavioral interface) or at composition (engineered systems learned shortcuts, not algebra), never reaching the geometric measurements that could test the bet.

## What we observed

### Bounded observations (Codex-audited, in one small LM prompt world)

In a three-fact prompt micro-world on frozen Qwen3-0.6B-Base (Phases 1–2, 68 experiments):

- **Cosine similarity misses response-law distinctions** in bounded settings. States with cosine ≈ 0.98 produce behaviorally different outputs under logit-lens JSD. This establishes that cosine is an *insufficient* instrument for behavioral structure, not that it is categorically blind.
- **Greedy answer signatures form an approximate behavioral quotient** with nontrivial predictive fibers. Path dependence is confirmed, but ordinary textual order/multiplicity effects are not ruled out.
- **Observational selectivity is verbalizer-sufficient** (OSQ-1: V=1.01). The 62× late-layer amplification is real but fully explained by 3-bin answer-token routing — ordinary language-model decoding, not native behavioral algebra.
- **Composition fails** (QPC-1). Transplanted query-state at L21 does not compose with the recipient world. The Qwen prompt micro-world is closed.

### Terminal composition result (Phase 3)

**LAC-0: Learned Action Carrier (739K params, 3 seeds).** A typed neural machine achieves 100% primitive execution and F=1.0 cross-world portability — capabilities the matched untyped transformer cannot achieve (12.5%, chance). But composition fails: held-out endpoint 14–34% (gate ≥85%), sequential agreement 12–34% (gate ≥90%). **Circuit selection** (Codex design gate): default initialization learns endpoint shortcuts (96%) but 0% sequential; Xavier learns sequential execution (95%) but not composed carriers (34%). These are different optimization basins. Neither passes. Terminal per §14.7(b).

**EAC-1** passed all 7 causal gates but was ruled architecturally tautological — the carrier IS the next-state embedding.

### Eligibility failures (Phases 4–5)

All tested model sizes fail the two-dial world (Z8×Z8, 64 states, Python-completion) capability gate (≥95%):
- Qwen3-0.6B-Base: 48% (permutation task)
- Qwen3-1.7B-Base: 50–54%
- Qwen3-4B-Base: 56% (smoke)
- Qwen3-8B-Base: 55.5%
- Qwen3-8B-Instruct: 50–64%

Small-to-medium base models cannot track multi-step state evolution through prompts.

### Nine breakpoints (Phase 1)

Across 50+ experiments, we catalogued nine places where ℝⁿ mathematics fails in latent space. Each is a constraint on what native math must look like — not evidence that native math was found.

| # | Breakpoint | What it means |
|---|-----------|---------------|
| 1 | **Presence ≠ causation** | A concept can be perfectly decodable yet have zero causal effect. Linear probes find ghosts. |
| 2 | **Single-site ≠ distributed** | Facts are distributed properties of entire layer transformations. |
| 3 | **Vector distance ≠ semantic distance** | Points close in cosine can be functionally opposite. |
| 4 | **Fixed dimensions ≠ fixed structure** | Effective dimensionality changes with context and task. |
| 5 | **Vector composition ≠ computational composition** | The model composes through its forward pass, not through vector arithmetic. |
| 6 | **Observation ≠ state** | The act of choosing what to probe constrains what you can find. |
| 7 | **Snapshot ≠ computation** | A representation at layer *l* can't be understood without the trajectory through all layers. |
| 8 | **ℝⁿ tools find ℝⁿ structure** | PCA finds linear structure because PCA *is* linear structure. The measurement imposes itself on the answer. |
| 9 | **Metric blindness to composition** | Four fact-worlds with cosine ≈ 1.000 produce dramatically different behavioral outcomes under intervention. |

Full details: [`theory/BREAKPOINT_REGISTRY.md`](theory/BREAKPOINT_REGISTRY.md)

## Theoretical framework

The axiomatic framework defines behavioral place, move, cost, and composition for deterministic transition-output systems. The formal development (D1–D9, Theorems 1/4/7/8, Open Problem 7, Conjectures 5/7) is in [`theory/AXIOMS.md`](theory/AXIOMS.md).

The adopted theory is sound standard mathematics — Moore-behavioral pseudometrics, observability seminorms, finite-memory append worlds, finite-access asymmetry, and surgeon/denizen world separation. The distinctive material is the registration-relative interface (D2), coherent presentation transport (D6), executable-germ restrictions (D9), and the native bridge definition. Nothing currently proved is genuinely new mathematics; the framework's value is in governing claims and preventing hidden decoders.

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

## Methodology

Every claim follows a strict evidence protocol:

- **Codex-audited.** An independent AI reviewer adversarially checks every result for overclaims, instrument artifacts, and alternative explanations. Claims are adopted only in auditor-licensed language.
- **Negative results are first-class.** Failed experiments are logged permanently and shape future directions. We've withdrawn prior claims when controls revealed artifacts.
- **Instrument-first.** Before interpreting results, validate the instrument: baseline retrieval, self-patch controls, sham-patch controls.
- **Reproducible.** CPU-only experiments, deterministic seeds, full configs logged. Every experiment in the ledger includes the git commit, command, config, and metrics.

## Prior work and corrections

The previous program (LLM embedding perturbation, diffusion latent repair) is archived under [`legacy/`](legacy/). Its nested-arithmetic claims were **withdrawn** after independent controls showed the benchmark measured termination under a token cap, not arithmetic capability. Full record: [`legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`](legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).

## Remaining work

One bounded theory attempt: a composition-identifiability theorem for action carriers in behavioral quotients, with a hard kill gate (five conditions including prior-art delta, nontrivial separation, LAC retrodiction without fitting, and a finite CPU falsifier). If it fails, the program archives as a negative-results and methodological contribution. Details in [`STATE.md`](STATE.md).

## License

MIT
