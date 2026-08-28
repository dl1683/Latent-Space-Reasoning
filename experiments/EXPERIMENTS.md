# Experiments Log

Reverse chronological. Only gate-passed conclusions are stated as confirmed.
Program opened 2026-08-27; prior program's log is at `legacy/experiments/EXPERIMENTS.md`.

---

## NLM-007 — LM residual-stream dynamics; middle-depth ridge lead withdrawn under the identity baseline; displacement ladder adjudicated (audit #8 wording); forward-time move adjudicated NOT MET = nonpass, not a kill (Round 20, audit #9); within-style null = diagnostic only; style B running (2026-08-28)

- **Lock.** Round 13, documentation-only (ledger `nlm007_round13_lock`;
  design `theory/dialogue/003.md`, `theory/EXPERIMENTS.md`); Round 14
  amendment `097e2df`; Round 16 correction (completed law read at the
  substituted slot; final pair uses `head(Yhat)` on the post-norm state,
  ledger `nlm007_round16_corrected_rerun_predeclared`). Qwen3-0.6B (28
  layers), 80 one-token words × 16 carriers, four carrier-block folds; six
  layer pairs; law ladder word-mean / kNN / ridge / low-rank affine / kernel
  ridge; per-carrier oracle; within-word carrier permutations; two-way
  cluster bootstrap. Decision: ≥0.05 lead over the best static chart with
  lower bound >0 on successor cosine and both completed-law readouts in ≥2
  layer pairs. CPU only.
- **Capture.** `experiments/run_lm_dynamics.py` →
  `experiments/results/lm_dyn_v1/manifest.json` (model revision c1899de2…,
  batch 16, batched-vs-single nulls ≤ 6.1e-5, 79 s). `states.npz` is
  git-ignored; sha256 `6ec9520845811bbd…` recorded in the manifest.
- **Artifacts (`experiments/results/lm_dyn_v1/`; all kept).**
  - `analysis.json` — fallback run, pairs L0→1 / L8→9 / L27→28, 20 shuffles,
    500 boot (ledger `nlm007_fallback_declared`, `nlm007_v1_fallback`; 1427 s,
    19% over the 20-min cap). Successor-endpoint numbers valid; completed-law
    numbers read at the last token — **secondary only, invalid for the lock**
    (Tier-3 audit #5).
  - `analysis_ext.json` — extension, pairs L4→5 / L12→13 / L20→21 (ledger
    `nlm007_ext_predeclared`, `nlm007_ext_v1`; 1100 s). Same validity split:
    successor valid, completed-law secondary/invalid for the lock.
  - `analysis_slot.json` — **canonical slot-endpoint result**: corrected rerun
    over all six pairs, 20 shuffles, 500 boot, seed 13007 (ledger
    `nlm007_slot_v1`; 2145 s of a 3300 s budget; reload check unchanged).
    Exploratory at the reduced 20/500 budget; its L8/L12 qualification is
    withdrawn below.
  - `analysis_basesmoke.json` — moot-maker smoke at L8→L9 only, 2 shuffles /
    20 boot, point estimates (ledger `nlm007_baselines_smoke_L8`; 796 s).
    Pipeline validation; superseded by `analysis_base.json`.
  - `analysis_base.json` — predeclared six-pair moot-maker run
    (identity-plus-residual and per-carrier affine; ledger
    `nlm007_baselines_v1`). Took 4540.8 s against the predeclared 3300 s
    budget: **budget-incomplete exploratory artifact** — measured values
    retained, no planned full-budget gate earned; the null-making withdrawal
    still applies (Round 18, audit #7).
  - `identity_check.json` — stored-true-successor identity test of the slot
    completion at every pair and carrier (ledger `nlm007_identity_check_v1`;
    audit #6 action 3). **Valid**: routing validated to measured precision
    (per-pair max KL 1.9e-6 to 6.2e-6 over 16 × 80 cells); no per-carrier
    error profile or fresh-float32 comparison was stored.
  - `analysis_deltasmoke.json` — `--target delta` pipeline smoke at L8→L9
    (1 shuffle / 10 boot; ledger `nlm007_delta_smoke_L8`). **Not a result.**
  - `analysis_delta.json` — **valid, adjudicated (Round 19, audit #8)**:
    five-pair displacement ladder (ledger `nlm007_delta_predeclared`,
    `nlm007_delta_v1`; 1750.3 s of the 5700 s wall; support 1.0). Reading
    below.
  - `forward_manifest_A.json` / `forward_manifest_B.json` — forward-time
    captures, sentinel A = '.' and B = ',' (ledger
    `nlm007_forward_predeclared`, `nlm007_forward_locality_control`);
    `forward_states_A/B.npz` git-ignored. Locality control passes under the
    Round 20 corrected clause (ledger `nlm007_forward_locality_ruling`);
    A/B unappended q-states and laws identical bit-exactly (ledger
    `nlm007_forward_AB_equality`).
  - `analysis_fwdsmoke.json` — `--source forward` pipeline smoke at F8, A
    (1 shuffle / 10 boot; ledger `nlm007_forward_smoke_F8A`). **Not a
    result.**
  - `analysis_fwdA.json` — forward-time move, sentinel A = '.', layers
    0/4/8/12/20, 20 shuffles / 500 boot (ledger `nlm007_forward_fwdA`;
    2220 s). **Valid; adjudicated Round 20 + audit #9**: the primary arm
    did not meet the preregistered two-layer same-sentinel criterion (only
    `F20` qualifies) — a nonpass under the historical contract, not a kill.
  - `analysis_fwdB.json` — sentinel B = ',' control/replication arm, same
    settings (ledger `nlm007_forward_fwdB`; 1823 s). **Valid; adjudicated
    Round 20**: `F12` and `F20` qualify (ridge); cannot rescue the period
    arm. Reading below.
  - `analysis_stylesmoke.json` — `--style-null` + KL-rank pipeline smoke at
    F8, A (2 shuffles / 10 boot; ledger `nlm007_stylenull_smoke_F8A`).
    **Not a result.**
  - `analysis_styleA.json` — within-style-family target null, sentinel A,
    layers 0/4/8/12/20, 20 shuffles / 500 boot (ledger
    `nlm007_stylenull_predeclared`, `nlm007_stylenull_styleA`; 2213 s;
    support 1.0). **Diagnostic only (audit #9)**: the null is an
    alignment-destruction diagnostic, not a clean style null; its KL-rank
    endpoint ranked K = 7 candidates instead of the preregistered 10 —
    labelled, **not contract-valid on that endpoint**. No claim.
  - `analysis_styleB.json` — sentinel B arm of the same control,
    **running**; carries the same K = 7 label. No status until scored.
- **Successor endpoint (valid in all runs).** L0→L1: word-mean = ridge =
  kernel = 0.949, shuffled null 0.95 — lexical persistence, no law beyond
  word identity. From L4 on, full-dimensional ridge beats word-mean and the
  best static chart at every depth (ridge/chart/word-mean: L4 0.927/0.884/
  0.886; L8 0.941/0.860/0.861; L12 0.977/0.898/0.888; L20 0.965/0.901/0.897;
  L27 0.976/0.883/0.864, the last on normed vectors). Shuffle penalty grows
  with depth.
- **Slot-endpoint gate reading (Round 17, superseded at L8/L12 by Round 18).**
  On `analysis_slot.json` the pairs L8→L9, L12→L13, L27→L28 cleared every
  locked gate mechanically (support 1.0); L4→L5 and L20→L21 cleared both slot
  readouts and the word-mean gate but missed the all-fold +0.05
  successor-cosine lead (a stricter convention than the original lock, audit
  #6); L0→L1 fails every lead gate. Word-mean slot skill decays with depth
  (0.95, 0.84, 0.78, 0.70, 0.43, 0.40) while ridge holds 0.92–0.98 and the
  chart collapses late (0.50, 0.51). Round 16 scorecard: five of six
  predictions held; the L27→L28 attenuation prediction failed.
- **Withdrawal at L8→L9 and L12→L13 (Round 18 + audit #7; ledger
  `nlm007_baselines_v1`).** Pooled ridge − identres on successor cosine /
  slot skill / slot ordering: L8 −0.008/−0.021/−0.020; L12 −0.007/−0.009/
  −0.013 (only slot skill and ordering are completed-law slot metrics). On
  shared words and held-out carrier blocks, identity-plus-shared-displacement
  is at least as good as full ridge within a post-hoc one-sided 0.02 pooled
  margin on the three recorded comparison metrics at L8→L9 and L12→L13; the
  finite-ladder ridge wording is withdrawn as a conservative policy. The
  intervals support "no demonstrated positive ridge advantage under this
  margin", not "no lead" or equivalence. The measured relation is consistent
  with identity plus a calibration-mean displacement under this design; the
  experiment does not determine whether the displacement is carrier-, state-,
  or word-dependent. The Round 17 two-pair criterion does not survive as a
  claim. Identity-plus-shared-displacement does not meet the chosen margin at
  L0 (+0.46), L4 (+0.033/+0.019/+0.022), L20 (+0.018/+0.034/+0.032), or L27;
  L4 and L20 remain non-qualifying but live, while L27 is not a valid
  raw-residual persistence comparison. Per-carrier affine is far below the
  cross-carrier field everywhere (within-carrier diagnostic only).
- **Displacement ladder (Round 19 + audit #8; `analysis_delta.json`).**
  Only `L20->L21` passes the predeclared three-endpoint gate (kernel;
  positive clustered lower bounds on displacement cosine, slot skill, slot
  ordering) — retained as one bounded qualifying pair under the registered
  displacement-and-slot-law gate. `L0` is lexical persistence. `L4` has a
  small live remainder but fails the gate. `L8/L12` separate strongly from
  the word-conditioned displacement mean on displacement coordinates, with
  kernel minimal among the tested ladder, but slot-ordering leads are only
  0.003–0.022 and slot-skill lower bounds are mixed — the gate fails. Adopted
  wording: held-out-carrier evidence for predictable displacement variation
  beyond a word-conditioned mean, with a kernel as the minimal tested
  predictor; carrier/template versus state dependence remains unresolved. The
  carrier shuffle is a carrier-alignment diagnostic, not a state-independence
  null (shuffled field reported for ridge/low-rank only). "The slot law
  barely registers it" is a readout fact, not a world fact.
- **Forward-time move (Round 20 + audit #9; `analysis_fwdA.json`,
  `analysis_fwdB.json`).** Sentinel '.': `F0` token-identity dominated
  (shared mean = word-conditioned mean = 0.67 ≈ field 0.69). `F4/F8/F12`:
  displacement cosine ridge/kernel 0.71–0.78 vs word-conditioned mean
  0.48–0.53; law skill at the sentinel position 0.39–0.57 vs 0.01–0.02;
  carrier-shuffled field 0.12–0.32 vs 0.67–0.81; but ordering leads
  0.00–0.08 with lower bounds ≤ 0 in half the folds — three-endpoint gate
  fails. `F20` qualifies (ridge: +0.16–0.23 / +0.50–0.61 / +0.020–0.058,
  all LBs > 0). Sentinel ',': same shape; `F12` and `F20` qualify (ridge),
  `F8` misses one ordering LB by −0.002, `F4` one skill LB. Token-identity
  control: the '.'-fitted predictor on the ',' target scores 0.43–0.54 vs
  0.26–0.30 for the shared mean. Adopted wording: the period sentinel did
  not meet the preregistered two-layer, three-endpoint qualification
  criterion — a nonpass under the historical contract, not a kill of forward
  transport; in the shared-word, held-out-carrier design, sentinel
  displacement is predictably improved over the word-conditioned mean from
  F4 onward and the response law registers that variation in cosine and
  skill; the ordering endpoint was later diagnosed as insensitive/saturated,
  so the qualification failure is not a substantive null result. The comma
  arm falsifies "token identity or position prevents any qualifying layer".
  Carrier/template presentation versus state dependence remains unresolved
  (audit #8). Ordering is replaced prospectively by KL-to-truth candidate
  rank (K = 10); no existing run is reclassified.
- **Within-style-family null, sentinel '.' (`analysis_styleA.json`;
  diagnostic only).** Mechanically `F4/F8/F20` beat both the word-conditioned
  mean and the null on cosine, skill, KL-rank (ridge KL-rank 0.82–0.90 vs
  word-mean 0.31–0.41); `F12` misses one fold's KL-rank LB; `F0` fails. The
  null collapses below the shared mean (0.16–0.50 vs 0.47–0.62). Audit #9:
  a field refit on a broken carrier pairing predicts the wrong carrier's
  displacement, so "beats the within-style null" is not informative evidence
  for a state-linked component; "style-robust" is withdrawn as a claim. The
  KL-rank endpoint here ranked K = 7 (kNN-1/5/20 omitted; fixed in
  `269e46c`) and is not contract-valid.
- **What we learned.** Identity is the null for residual-stream transport.
  The present data support persistence plus a calibration-average
  displacement as a competitive finite-design description at L8 and L12,
  retain small unresolved remainders at L4 and L20, and do not yet establish
  a native or generally reusable affine law. The forward step is a bounded
  held-out-carrier displacement-forecasting result that does not yet
  distinguish a state-space regularity from a carrier/template-conditioned
  nuisance law. A permutation null that a flexible model trivially beats is
  not a control. Bounded to one model and shared words. Next in audit #9
  order: within-family leave-one-carrier-out control vs per-word/per-block
  mean displacement (`--loco`, `3a8b859`; to be predeclared) → cross-fitted
  style residualization → unseen-word split → second family.

## Round 12 closure — frozen-encoder program closed; pivot to worlds with dynamics (2026-08-27)

- Ledger `nlm006b_round12_adjudication`; dialogue `theory/dialogue/003.md`;
  commit `3294718`. NLM-006b corrected to non-diagnostic under its own
  label-preservation gate (below); frozen-encoder closeness/map work closes
  as scope management.
- **Residue (narrow, this encoder/dataset).** Training supplies a
  task-effective chart metric, affine-path smoothness, and graceful chart
  degradation under identity-destroying moves; no native construct tested
  (substitutability profiles, Fisher pullback, their transported variants)
  competes with it. Not a general claim about native constructs.
- Next program: causal-LM residual streams, where the forward pass is the
  world's own transport (NLM-007).

## NLM-006b — calibrated transport audit; chart survives, NON-DIAGNOSTIC under lock (2026-08-28)

- **Design.** Locked Round 11 (`nlm006b_prereg_transport_audit`): independent
  candidate strata (20 same-/20 cross-fine-label per anchor), transported-pair
  predictors F_T / R_T vs cosine_T / euclid_T on (T_e x, T_e y), true
  fine-label endpoint, label-preservation gate p_e ≥ 0.80, calibrated
  displacement gate. Ledger `nlm006b_v1`; artifact
  `experiments/results/nlm006b_v1/analysis.json`; transports
  `experiments/results/vision_cifar100_dinov2s_edits_v2/` (edits.npz
  git-ignored, sha256 9cc0e7c0…; displacement.json committed). 471 s, CPU.
- **Chart survives every displaced transport.** Support 400/400; displacement
  gate passes for crop50/invert/mix50/occlude50 (0.98–1.0 above control q95).
  TT chart lead over best native: crop50 +0.208, invert +0.227, occlude50
  +0.222, mix50 +0.090 (paired CIs exclude 0).
- **Non-diagnostic (Round 12).** Label preservation 0.19–0.46 for all four
  displaced families vs the 0.80 gate (controls hflip 0.77, shift 0.76): every
  family is OOD under the identity gate, so chart survival is descriptive
  only and no native/chart verdict is issued.
- **Order effect.** ST−TS cosine ≈ 0.035 for all displaced families (CIs
  exclude 0); ≈ 0 for hflip. Real, small, outside the invariance class only.

## NLM-006 v1 — transports outside the invariance class; EXPLORATORY (cosine-selected negatives) (2026-08-28)

- **Design.** Six transport families re-encoded by the frozen encoder
  (`experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`, keyed
  `test_emb_<family>`: hflip, shift1px, crop50, invert, mix50, occlude50;
  `displacement.json` alongside), stratified candidates, true fine-label
  endpoint. Relabeled **exploratory** by Tier-3 audit #3 before results were
  read: hard negatives were cosine-selected, so the pool is adversarial to any
  chart-like ranking. Ledger `nlm006_v1_exploratory`; artifact
  `experiments/results/nlm006_v1/analysis.json`.
- **Uninterpretable for the primitive contest.** Every predictor scores below
  0.5 (cosine 0.411, Euclid 0.402, F 0.486, R_no_coarse 0.477) — cosine's
  "collapse" is manufactured by selecting negatives with the tested metric.
- **Exploratory signal.** Support 400/400 (stratification fixes NLM-005's
  support failure). Order sensitivity appears only outside the invariance
  class: ST−TS cosine 0.05–0.10 with CIs excluding 0 for crop50/invert/mix50/
  occlude50; 0.00 for hflip/shift1px. Displacement mean cos: hflip 0.96,
  shift 0.98 vs crop 0.63, invert 0.49, mix 0.43, occlude 0.66.
- **Next.** NLM-006b (ledger `nlm006b_prereg_transport_audit`): independent
  candidate strata, transported-pair predictors, label-preservation and
  calibrated displacement gates. Lesson: candidate pools must never be
  selected by the metric under test.

## Round 10 closure — frozen-encoder closeness/map line closed (2026-08-27, narrowed by audit #3)

- Ledger `round10_frozen_chart_closure`. The NLM-003 R-over-F claim is
  withdrawn (coarse-taxonomy leak, see diagnostics below); NLM-005 is void on
  support; no native construct built so far (substitutability profiles, Fisher
  pullback) competes with the trained chart metric on this artifact.
- **Residue as narrowed by Tier-3 audit #3:** training creates a
  task-effective chart and affine-path smoothness *in this encoder/dataset*
  (cosine 0.946 trained vs 0.575 random-init; same-class chart-line flicker
  12.7% vs 95%). Not a general claim that native constructs are dominated, and
  not proof of intrinsic geometry or of "straight routes inherited from
  training" beyond this encoder and dataset.
- Replacement line: NLM-006/006b — stratified transports outside the trained
  invariance class.

## NLM-005 — composed transport/substitution; VOID on support (2026-08-27)

- **Design.** Locked `a12aad4` (artifact lock `aab0f69`). hflip and 1-px-shift
  transports re-encoded by the frozen encoder, composed with random
  substitutions in both orders (ST, TS), true fine-label endpoint. Ledger
  `nlm005_v1_composition`; artifact `experiments/results/nlm005_v1/analysis.json`.
  Transport families now live in
  `experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`
  (`test_emb_hflip`, `test_emb_shift1px`; byte-identical to the original
  NLM-005 file, which was removed as superseded).
- **Void by kill condition 3:** support 129/400 (32%) < 80%. Order gaps
  non-diagnostic: ST−TS cosine ≤ 0.006 (hflip 0.006 [−0.003, 0.017], shift
  0.004 [−0.003, 0.013]); shift1px R_no_coarse 0.027 [−0.003, 0.057] on a
  sensitivity row. Cosine leads native candidates by ≈0.32 on every order.
- **Lessons.** hflip/1-px shift are augmentations DINOv2 was trained to be
  invariant to, so they are near-identity moves in its world — transports must
  lie outside the trained invariance class. 40 random candidates over 100
  classes cannot reach 80% support — candidate sampling must be stratified.

## NLM-003 v2 diagnostics — R's win was a coarse-head leak (2026-08-27)

- **Design.** Same lock, artifact, endpoint as NLM-003; new anchor sample; audit
  #2 diagnostics (tie accounting, R without coarse head, cheap-baseline ladder,
  kNN k-sensitivity). Ledger `nlm003_v2_diagnostics` (Round 9: sensitivity
  accounting, not new evidence); artifact
  `experiments/results/nlm003_v2_diagnostics/analysis.json`.
- **Leak.** `R_no_coarse` 0.586 < `F` 0.667 (R with coarse 0.762; fine labels
  nest inside coarse classes). The NLM-003 R-over-F directional claim is
  withdrawn. Δ_{F−R} on this resample −0.095 [−0.142, −0.049]. R ties on
  22–33% of comparisons.
- **Ladder.** cosine 0.934, PCA-32 cosine 0.941, Euclid 0.933; pixel-stat
  Euclid 0.624, raw-pixel cosine 0.622. kNN same-class flicker 0.18/0.13/0.10
  vs cross-class 0.41/0.38/0.37 at k = 8/32/128 — world-path contrast robust to k.

## NLM-004 — random-init null world; SUPPORTED (2026-08-27)

- **Design.** Preregistered in ledger (`nlm004_prereg_null_world`) before
  scoring: random-init DINOv2-small chart
  (`experiments/results/vision_cifar100_randinit/`), true fine-label endpoint.
  Ledger `nlm004_v1_null_world`; adjudication `nlm004_round9_adjudication`
  (supported, exploratory — bootstrap CIs not in artifact); artifact
  `experiments/results/nlm004_v1/analysis.json`. CPU, 230 s.
- **Supported.** Cosine 0.575 in the null chart vs 0.946 trained (gap 0.371;
  gates ≤ 0.70 and ≥ 0.20). Embedding-kNN fine accuracy 0.069 vs 0.761.
  Same-class chart-line kNN flicker 95% (null) vs 12.7% (trained). Semantic
  heads collapse (coarse 0.21) while pixel-statistic heads stay strong (rgb
  0.83, luma 0.82) — cheap-baseline confound noted.
- **Reading.** The chart's task-effective metric and affine-path smoothness are
  created by training in this encoder/dataset; the null chart has neither.

## NLM-003 — R beats F on the true fine-label endpoint; cosine dominates both (2026-08-27) — R-over-F WITHDRAWN (see v2 diagnostics)

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
