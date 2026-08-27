# Experiments Log

Reverse chronological. Only gate-passed conclusions are stated as confirmed.
Program opened 2026-08-27; prior program's log is at `legacy/experiments/EXPERIMENTS.md`.

---

## NLM-003 — R beats F on the true fine-label endpoint; cosine dominates both (2026-08-27)

- **Design.** Locked at `e2a1fb2` (`theory/EXPERIMENTS.md`, NLM-003). Same
  frozen CIFAR-100/DINOv2-small artifact and runner as NLM-002, endpoint
  switched to the true fine label (no head is trained on it):
  `python experiments/run_nlm002_vision.py --cache experiments/results/vision_cifar100_dinov2s --out nlm003_v1 --endpoint fine_label`.
  Ledger `nlm003_v1_true_fine_endpoint`; artifact
  `experiments/results/nlm003_v1/analysis.json`.
- **Directional gate met.** Profile-continuity `R` 0.734 vs Fisher pullback `F`
  0.630, Δ_{F−R} = −0.104 [−0.148, −0.058] over 6,199 scored pairs; support
  thin (130/400 anchors had a same-fine-class candidate among 40 draws).
- **Chart metrics dominate.** Cosine 0.946 and Euclidean 0.935 on the same
  anchors beat both native constructs by 20–30pp.
- **Tier-3 audit #2 reclassification (adopted).** NLM-003 is a **narrow
  instrument comparison** — "these implementations lose to cosine on this
  endpoint" — not evidence that native geometry is generally dominated (one
  encoder, one endpoint, one-step random substitutions, one seed, 130 supported
  anchors). `R` takes five values with 0.5 tie credit and includes the coarse
  head (fine nested in coarse), so tie accounting and an `R`-without-coarse
  rerun are required. Next gate: random-init null (NLM-004,
  `nlm004_prereg_null_world`), cheap-baseline ladder, kNN k-sensitivity,
  nonlinear re-charting, composed / out-of-distribution moves.

## NLM-002 — non-LM branch (CIFAR-100/DINOv2): endpoint killed, chart-path structure found (2026-08-27)

- **Design.** CIFAR-100 → DINOv2-small CLS, 6000 train / 2000 test, raw pixels
  stored (`experiments/results/vision_cifar100_dinov2s/`, built by
  `experiments/build_vision_cache.py`). Runner `experiments/run_nlm002_vision.py`
  (default endpoint `rawpixel_knn`). Ledger `nlm002_v1_nonlm_branch`; artifact
  `experiments/results/nlm002_v1/analysis.json`. CPU, 133 s.
- **M2 kill condition met.** Raw-pixel k=32 kNN fine label is nearly
  uninformative (0.115 accuracy; 0.12 agreement with embedding kNN, which
  scores 0.761), so the locked endpoint is invalid and M3 (`F` 0.601 vs `R`
  0.605, Δ = −0.004 [−0.034, +0.026], 16,660 pairs) is a tie on noise, not a
  primitive verdict. Lesson: an endpoint must be independent of both candidates
  *and* informative — the true fine label is both (→ NLM-003).
- **M1 chart-path structure (informative, audit-qualified).** Along straight
  lines between same-class embeddings the coarse-semantic readout flickers on
  only 2% of paths; between classes the fine-label kNN flickers on 38% and
  any-readout on 78%. Audit #2: the 2% figure is weak evidence (affine argmax
  is near-monotone by construction) and kNN flicker is at k=32 only — a
  k-sensitivity analysis is required before any world-path claim. Pixel-stat
  heads are weak (52–59% test acc), so their 21–24% flicker is partly head noise.
- **Implementation decisions flagged at lock:** pixel statistics of
  interpolated points are approximated (no pixels exist off the data), and the
  fine-label head is never trained.

## NLM-001 — verdict: negative on predictive novelty (2026-08-27)

- **Design.** Analysis-preregistered at `fea3a8f` over sequestered raw
  matrices (`experiments/results/nlm001_v1/manifest.json`); three CPU systems
  (Qwen3-0.6B, gemma-3-270m, SmolLM2-360M); primary = 72 calibration-unseen
  words, all 80 as sensitivity; `--rule pooled --scale-normalize`. Command and
  metrics: ledger `nlm001_v1_primary_72`; artifacts
  `experiments/results/nlm001_v1/analysis_primary_72.json`,
  `analysis_sensitivity_80.json`.
- **Central bet fails.** Native calibration-KL closeness does not beat a learned
  diagonal Mahalanobis metric on the model's own contextual hidden states for
  held-out orderings: Qwen Δ = −0.058 [−0.222, +0.034]; gemma Δ = +0.017
  [−0.02, +0.06]. Every predictor scores 0.95–1.00 — the robust held-out labels
  are large-gap easy pairs (instrument limitation). Post-verdict reading
  (ledger `nlm001_v1_postverdict_note`): unlearned centered contextual cosine at
  layer 14 reaches 1.000 vs native 0.954, and the preregistered selection rule
  chose an overfit metric (calib 1.000, held-out 0.947), so the reported Δ
  understates the native loss.
- **Context reversals exceed the paraphrase null** in Qwen (Q = 2.12
  [1.70, 2.56], R = 0.18) and SmolLM (Q = 17.1 but W ≈ 0.005, so Q is not
  interpretable there); not in gemma (Q = 1.40 [0.90, 2.55]).
- **Directedness absent.** Robust (≥2-of-4) asymmetric pairs: 1.5% Qwen, 9.2%
  SmolLM. Cross-system transfer τ_b: 0.14 (qwen|gemma), 0.47 (qwen|smollm),
  0.14 (gemma|smollm).
- **Kill conditions 3 (predictive novelty), 6 (coordinate confound), 8
  (instrument metadata recorded post hoc) apply.** Tier-3 fresh audit adopted:
  T2 is geometrically vacuous, κ is an invariant of the probe table not the
  space, B>W may recover the hand-authored block taxonomy, and no NLM-001
  outcome could have earned "cosine is the wrong object".
- **What we learned.** The substitutability/KL primitive on lexical embedding
  rows adds nothing over a symmetric learned metric on contextual states. Do not
  run NLM-002 on more words; next is a competition among primitives (see
  `STATE.md`). Runners must record tokenizer revision, library versions, thread
  and batch settings at run time.

## NLM-001 — instrument calibration, pre-verdict (2026-08-27)

- **NLM-001 — contextual substitutability, context rank, and transfer.** Frozen
  theory contract: `theory/EXPERIMENTS.md`. One CPU entrypoint,
  `experiments/run_lexical_closeness.py` (using the existing substitution-probe
  helper); frozen slice:
  `experiments/config/lexical_probe_v1.json`. The 12-word smoke and disclosed
  eight-item full-pipeline validation are calibration only. The latter
  invalidated the MAD robustness rule and put asymmetry signs at chance; H1 is
  exploratory. Primary analysis uses the 72 calibration-unseen words, with all
  80 reported only as sensitivity. Three-system raw matrices were acquired
  concurrently before the Round-2b amendment and stayed sequestered until the
  amended contract was committed at `fea3a8f`; the verdict entry above is the
  first outcome analysis. Ledger: `substitution_probe_smoke_qwen3_0p6b`,
  `nlm001_pipeline_smoke_8`; artifacts `experiments/results/pipeline_smoke_8/`.
