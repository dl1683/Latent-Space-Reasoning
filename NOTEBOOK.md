# NOTEBOOK

Reverse-chronological running log. Newest first. Each entry: what was done, what
was learned, what's next. Canonical state lives in STATE.md.

---

## 2026-08-27 — Round 1 revised after Claude attack

- Appended `## Codex — revision` to `theory/dialogue/001.md`, answering A1–A7
  point by point. Withdrew existential non-collapse as an axiom; adopted finite
  context rank with anchor-dependent radii.
- Proved the finite representation theorem: context rank is 1 exactly when
  every anchor's context neighborhoods form an inclusion chain. Derived the
  incompatibility-graph coloring characterization.
- Created `theory/AXIOMS.md` as the living formal surface. Local refinement is
  now conditional on a completed probe family; cross-realization agreement is a
  measurement of transportability, not an identity axiom.
- Created `theory/EXPERIMENTS.md` with the post-smoke, pre-measurement NLM-001
  preregistration: primary directed asymmetry, graded context rank, held-out
  transfer against contextual-cosine and learned-metric baselines, paraphrase
  nulls, cross-system checks, exact predictions, and kill conditions.
- No experiment was run in this revision; the confirmatory measurement remains
  unrun. A concurrent Claude turn added the CPU measurement runner and raw
  hidden-state capture. Next: Claude audits the revision and adds the
  preregistered analysis without changing the frozen slice or thresholds.

## 2026-08-27 — Repository restarted

- Entire prior program moved unmodified to `legacy/` (its README, docs,
  experiments, results, and correction record intact and internally linked).
  Root now holds only the new program: README, STATE, NOTEBOOK, `theory/`,
  fresh `experiments/` ledger. Local-only ignore patterns mirrored for `legacy/`.
- Four watchdogs live (20-min liveness, hourly ops, 2-hour Codex audit +
  anti-tunnel, 2-hour entropy sweep). Codex round 1 launched: axiom candidates,
  first target construct, falsifier on a real embedding space, prior art.

## 2026-08-27 — Native latent mathematics, dialogue round 1

- Wrote `theory/dialogue/001.md`: two candidate foundations, with a committed
  start from contextual substitutability neighborhoods rather than coordinates.
- Derived a presentation-invariant T0 topology from the first four relational
  axioms; explicitly left metric, origin, addition, and ambient dimension
  unearned.
- Pre-registered a CPU-only Qwen3-0.6B probe: held-out next-token
  substitutability versus raw/repaired coordinate metrics, including a direct
  falsifier for contextual non-collapse and controls for norm/tied-unembedding
  confounds.
- Next: Claude attacks probe circularity, the status of the non-collapse axiom,
  and whether the topology result has any content beyond a renamed basis theorem.

## 2026-08-27 — Direction set: native mathematics of latent spaces

- Prior LLM-perturbation program closed; arithmetic claims withdrawn after Igor
  Rivin's PRs #4/#5 (merged) and reanalysis of stored data. Correction record in
  `docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`; README rewritten to a findings
  table; three doc indexes consolidated to `docs/NAVIGATION.md`; orphaned docs
  archived.
- Residue: decoding determinism is hardware-dependent (GH200 non-deterministic,
  RTX 5090 deterministic); perturbation is a causal diversity source only on
  deterministic stacks. Process gates added (termination, direct control first,
  null model, clustered stats, propagation).
- New direction opened (see STATE.md). Starting the Codex dialogue on axiom
  candidates and the first target construct.
