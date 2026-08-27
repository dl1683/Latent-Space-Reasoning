# NOTEBOOK

Reverse-chronological running log. Newest first. Each entry: what was done, what
was learned, what's next. Canonical state lives in STATE.md.

---

## 2026-08-27 — Round 2b instrument calibration amendment

- Read the disclosed eight-item full-pipeline calibration. Within-block KL
  scale varied up to 16-fold, making the preregistered instance-specific MAD
  threshold invalid and yielding zero passes.
- Locked per-paraphrase scale normalization, four-of-four sign agreement, a
  block-pooled magnitude scale, and the explicit 12.5% random-sign null.
- Demoted directed asymmetry to exploratory after 0–18% sign agreement at
  chance. Revised post-calibration predictions to \(Q=2.5\), \(R=0.20\), and
  \(\Delta_{\rm rev}=+0.07\).
- The eight inspected words are excluded from primary confirmation; the
  remaining 72 are primary and all 80 are sensitivity. No confirmatory run was
  performed.

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

## 2026-08-27 — NLM-003: R beats F, both dominated by the chart metric; blackboard live

- NLM-003 (locked `e2a1fb2`, true fine-label endpoint, same artifact): R
  (substitution-profile agreement) beats F (Fisher pullback) — Δ_F−R = −0.104
  [−0.148, −0.058], gate met (R 0.734 vs F 0.630). Decisive context: plain cosine
  in the DINOv2 chart scores 0.946, Euclidean 0.935 — both native constructs
  lose by 20–30 points to the imported metric on the informative endpoint.
  Support thin: 130/400 anchors had a same-class candidate among 40 draws.
- Reading before Codex round 8: DINOv2 is trained so that one chart metric is
  meaningful; in such a world the denizen's best one-step map *is* that metric.
  NLM-001 found the same in the LM world (contextual cosine at L14 = 1.000). So
  far every world tested has a chart metric that already is the map for
  one-step consequences. Where native structure could still differ from the
  chart: (i) two-step moves — does substitution-then-transport equal
  transport-then-substitution (composition / laws), which a one-step metric
  cannot express; (ii) worlds whose chart was not trained to be metric (raw
  residual states of a non-contrastive model, or a randomly-initialized
  encoder as a null world); (iii) cross-class world-paths (M1: 38% detours).
- Blackboard: `@iqidis/blackboard-mcp` installed globally, registered for
  Claude Code (user scope) and Codex; mandated in global CLAUDE.md and the
  setup skill; Codex verified `bb_list`/`bb_create`/`bb_add_entries` and seeded
  the project board (`.blackboard/5df235ea`, git-ignored). This session cannot
  call `bb_*` until restart; Codex rounds now use it.

## 2026-08-27 — NLM-002 non-LM branch run: endpoint killed, chart-path structure found

- Artifact frozen (CIFAR-100 → DINOv2-small, 6000/2000, sha256 8de4f0b0…);
  locked with two recorded implementation decisions; run in 133 s on CPU.
- M2: the raw-pixel k=32 kNN fine-label endpoint is nearly uninformative
  (0.115 accuracy; 0.12 agreement with embedding kNN) → preregistered endpoint
  kill condition met. M3 (F vs R) is therefore a tie on noise (Δ = −0.004
  [−0.034, +0.026]); no primitive verdict. Lesson: independence is necessary,
  informativeness is not optional — the true fine label (no head trained on it)
  is both and should have been the endpoint.
- M1 (chart-path closure), the informative result: along chart-straight lines
  between same-class embeddings the coarse-semantic readout is monotone in 98%
  of paths (flicker 2% [0.3, 3.7]); between classes, fine-label kNN flickers on
  38% [32, 44] of lines and any-readout on 78% [73, 83] — straight lines between
  classes pass through third classes. Within-class chart lines are near
  world-paths for semantics; cross-class lines are not. Pixel-statistic heads
  are weak (52–59% test accuracy) and their flicker is partly head noise.
- Process: Codex sessions are now always fresh with terse file-pointing
  prompts (Devansh); `_meta/INDEX.md` row for this project updated for sister
  agents; no blackboard MCP is configured on this machine.

## 2026-08-27 — NLM-001 closed negative; NLM-002 designed as a primitive competition

- NLM-001 verdict (Codex round 3, `1584514`): instrument-void for confirmation
  (runtime metadata reconstructed post hoc), bounded negative falsifier of the
  lexical-KL instrument. Native calibration-KL lost to a symmetric metric on
  contextual hidden states (Qwen Δ = −0.058 [−0.22, +0.03]; unlearned centered
  cosine at layer 14 reached 1.000 on held-out orderings vs native 0.954);
  context reversals exceeded the paraphrase null in 2/3 systems (Qwen Q = 2.12
  [1.70, 2.56]); directedness absent. T2/T3 demoted to bookkeeping; T1's
  conjunction-closure premise fixed. Fresh Tier-3 audit adopted.
- NLM-002 (`theory/dialogue/002.md`, skeleton, not locked): mutual-kill
  competition between F (one fixed Fisher response-law geometry pulled back
  through frozen decoders/heads) and R (probe-indexed substitutability tested
  outside LMs, on DINOv2 image embeddings), each with an independent behavioral
  endpoint and a common-support Q estimator.
- Artifact prep for arm R: no image data existed locally. Building
  `experiments/results/vision_cifar100_dinov2s/` — CIFAR-100 (fine + coarse
  labels) → DINOv2-small CLS embeddings on CPU (35 ms/image), plus label-free
  pixel statistics (mean RGB, luminance, edge density) so probe blocks can ask
  questions not derived from the class taxonomy. Manifest carries dataset and
  encoder revisions, split indices, seed, sha256.
- F-arm implementation note for the lock: the LM Fisher pullback can be
  estimated as G = mean over sampled tokens t~K_c(z) of the outer product of
  ∇_z log K_c(z)(t) (VJPs through the frozen decoder; ~64 samples per state and
  calibration probe, D = 1024, CPU-feasible in ~30 min); the DINOv2 arm's Fisher
  is exact for a linear head (G = mean_z W^T F(p) W).

## 2026-08-27 — Tier-3 re-contextualization (Claude, before the auditor's answer)

**Live question.** Is closeness in a latent space context-indexed and directed
in a way no symmetric contextual representation reproduces — and can a native
invariant (context rank, Q = B/W) say how many orderings are actually required?

**Tunnel check — honest answer: partially tunneled.** In one day the program
narrowed from "native mathematics of latent spaces" to "next-token-KL
substitution probes on 80 lexical tokens of three tiny causal LMs". That is a
fine first instrument; it is not the object. Risks: (i) everything measured is a
property of one decoder family; (ii) 'latent space' has so far meant 'input
embedding row', the least latent thing in the model; (iii) the primitive
(substitutability under probes) was chosen in round 1 and never competed.

**Alternatives now live, each with its decisive result:**
1. Non-LM latent spaces with downstream-head probes (DINOv2, ESM-2, CLIP,
   wav2vec are cached). If the same axioms/invariants organize a vision or
   protein space, this is about latent spaces; if not, it is about LMs.
2. The probabilistic/denotational axiomatization (dialogue 001 §1B) on
   diffusion latents — the legacy program left instrumented diffusion stacks
   that already denote laws. Decisive: evidence-update is definable there and
   predicts something the relational axioms cannot.
3. Probe family as the object: states × probes as a formal context (Galois
   connection / formal concept analysis) — existing mathematics for exactly
   the (X, C, N) structure. Decisive: the concept lattice of a real system has
   nontrivial structure that context rank flattens.
4. Intermediate residual states as latent states (not embedding rows), probes
   = continuation from that depth. Decisive: context rank changes with depth.
5. The moot-maker: contextual cosine at the right layer matches native transfer
   (H3 kill 3/6). Then "native" = "the model's contextual representation" and
   the program must change primitive or object.

**What reframes earlier work.** The legacy perturbation program was, in these
terms, a substitution probe with the whole prompt as state and free generation
as the readout; its withdrawn results were the readout's insensitivity, not the
state's. Its one surviving residue — measure the stack's numerical noise floor
before interpreting any latent-space difference — is now a standing gate here
(η) and belongs in every project that measures representations.

**Direction still makes sense?** Yes as a first falsifiable instrument, with the
explicit condition that NLM-001's verdict (either way) must be followed by at
least one of alternatives 1–4, not by NLM-002 on more words.

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
