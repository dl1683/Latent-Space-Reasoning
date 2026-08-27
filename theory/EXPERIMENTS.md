# Native latent mathematics — experiment preregistrations

## NLM-001 — contextual substitutability, context rank, and transfer

**Status:** closed 2026-08-27 as a bounded negative falsifier of this instrument;
instrument-void for confirmatory claims under kill condition 8. The historical
registration below was locked at `fea3a8f` before anyone read the outcome. Raw
80-word matrices had been acquired concurrently before the Round 2b amendment,
so this was an analysis preregistration over sequestered data, not a
pre-acquisition registration. The audited verdict is at the end.

The 12-word smoke and `nlm001_pipeline_smoke_8` are calibration, not evidence.
The latter ran the full pipeline on eight disclosed words, two per lexical
class, drawn from the frozen 80. Those words selected the amended instrument:
the primary confirmatory item analysis is therefore the remaining 72 words.
The full 80 is a labeled sensitivity analysis only. The authoritative exclusion
list is the `items` array in
`experiments/results/pipeline_smoke_8/Qwen__Qwen3-0.6B.npz`.

The eight-item calibration found 16-fold within-block carrier-scale variation,
zero passes under the Round-2 MAD rule, 39–45% four-of-four ordering-sign
agreement (random-sign probability 12.5%), and only 0–18% four-of-four
asymmetry-sign agreement. After scale normalization, 8–21% of order gaps passed
the amended sign-and-magnitude rule. These facts were seen before the
confirmatory run and are disclosed here.

### Hypothesis

Lexical substitutability requires multiple context orderings and contains a
stable component that transfers to held-out probes better than the strongest
contextual or learned metric. Some ordering structure transports across
independently trained systems. Directedness is not a primary NLM-001 claim.

### Instrument

- One CPU entrypoint: `experiments/run_lexical_closeness.py`; it may import the
  existing measurement helper `experiments/substitution_probe.py`.
- Frozen items/probes: `experiments/config/lexical_probe_v1.json` (80 words;
  four lexical classes; four blocks × four paraphrases). Analyze the 72
  calibration-unseen words primarily and all 80 only as sensitivity.
- Primary system: `Qwen/Qwen3-0.6B`.
- Cross-systems: `google/gemma-3-270m` and
  `HuggingFaceTB/SmolLM2-360M`.
- Load one system at a time in float32 `eval()` mode; no generation and no GPU.
- Record resolved model/tokenizer revisions, library versions, embedding tying,
  thread count, and batch size. Any frozen item failing the one-token check is a
  configuration failure: abort rather than edit the slice.

Calibration blocks are `gloss` and `continuation`; held-out blocks are
`association` and `grammar`. Results are clustered by anchor and reported both
pooled and by lexical class.

### Operationalization

For probe paraphrase \(j\), insert state \(x\)'s input-embedding row at the
frozen slot and read the full-vocabulary final next-token law \(K_{B,j}(x)\).
Compute the 80×80 directed-KL matrix by matrix multiplication, discard the laws,
and retain only the matrix and registered hidden states. This is exact, requires
no sampling or generation, and bounds memory to one probe at a time. No outcome
coarsening is permitted. Retained mass is therefore report-only, never a void
condition.

For each paraphrase define its carrier scale and normalized loss

\[
r_{B,j}(x\to y)=D_{KL}(K_{B,j}(x)\Vert K_{B,j}(y)),\quad
m_{B,j}=\operatorname{median}_{x\ne y}r_{B,j}(x\to y),\quad
\widetilde r_{B,j}=r_{B,j}/m_{B,j}.
\]

All medians defining \(m_{B,j}\) use the 72 primary words. Set

\[
D_B(x\to y)=\operatorname{median}_j\widetilde r_{B,j}(x\to y).
\]

For transfer, set

\[
D_C=\operatorname{median}(D_{\rm gloss},D_{\rm continuation}),
\qquad
D_H=\operatorname{median}(D_{\rm association},D_{\rm grammar}).
\]

For a statistic family \(S\) (ordering gaps or asymmetries), derive every
\(\widetilde s_{i,j}\) from \(\widetilde r_{B,j}\) and define one block-pooled
scale

\[
\nu_0^{S,B}=\operatorname{median}_i\left[
1.4826\operatorname{median}_j
|\widetilde s_{i,j}-\operatorname{median}_\ell\widetilde s_{i,\ell}|
\right].
\]

The per-instance four-sample MAD is used only to estimate this pooled scale; it
is never an instance-specific veto. On the first probe and first eight states,
define the numerical null in KL units by

\[
\eta=\max_{x,y}\left|r_{\rm batch}(x\to y)-r_{\rm single}(x\to y)\right|.
\]

Let \(\widetilde\eta=\eta/\min_{B,j}m_{B,j}\). A four-paraphrase statistic from
family \(S\) is **robust in block \(B\)** iff all four signs are the same and
nonzero and

\[
|\operatorname{median}_j\widetilde s_{i,j}|>
\theta_{S,B}:=\max(2\nu_0^{S,B},10\widetilde\eta).
\]

For a fixed two-paraphrase half, **half-admissible** means both signs agree and
the half median exceeds the same \(\theta_{S,B}\). Under independent fair random
signs, four-of-four agreement has probability \(2/2^4=1/8=12.5\%\); a fixed
two-versus-two reversal also has probability 12.5%. Report the corresponding
binomial null explicitly, but use anchor-clustered inference because candidate
pairs are not independent. Every anchor-bootstrap replicate recomputes
\(m_{B,j}\), \(\nu_0^{S,B}\), robust masks, labels, and active-anchor status from
its sampled anchors; learned baseline fits remain frozen as specified below.

### Measurements and exact predictions

Compute E1 and H2–H4 separately for every system. The primary thresholds apply
to the primary system; cross-system claims use only word-string-aligned states
and never align coordinates.

#### E1 (formerly H1). Directed asymmetry — exploratory

\[
\widetilde a_{B,j}(x,y)=\widetilde r_{B,j}(x\to y)
-\widetilde r_{B,j}(y\to x).
\]

The eight-item calibration found four-of-four sign agreement of 0%, 18%, 4%,
and 18% by block, versus the 12.5% random-sign probability; asymmetry magnitude
was about 10% of KL scale. Positive scale normalization cannot change these
signs. The former 0.20 primary prediction is withdrawn, not softened. Report
normalized blockwise agreement, magnitude, and cross-realization sign agreement
as exploratory diagnostics only. No NLM-001 result can earn a directedness
claim; that requires a new preregistration on unseen items and probes.

#### H2. Context rank

Fix the preregistered split of each block's four paraphrases into halves
\(B^1=(1,2)\) and \(B^2=(3,4)\). For two halves \(U,V\) and anchor \(x\), let
\(q_x(U,V)\) be the fraction of candidate pairs half-admissible in both halves
whose ordering signs oppose; it is undefined when no pair is half-admissible in
both. Define

\[
W=\operatorname{median}_{x,B}q_x(B^1,B^2),\qquad
B=\operatorname{median}_{x,A<B,h\in\{1,2\}}q_x(A^h,B^h),\qquad Q=B/W,
\]

omitting undefined cells. If \(W=0<B\), set \(Q=+\infty\); if \(W=B=0\), H2 is
undefined and unsupported. Bootstrap anchors and recompute \(W,B,Q\) in every
replicate.

- Post-calibration point prediction: \(Q=2.5\) in the primary system.
- Primary support gate: \(Q\ge2\) and the anchor-bootstrap 95% lower bound is
  above 1.5. If \(B\le W\), contexts disagree no more than paraphrases do and
  the measured system is treated as effectively rank one.
- Still report \(G_{0.10}\), its reversal witnesses, and
  \(\widehat\kappa_{0.10}=\chi(G_{0.10})\), where an edge requires at least 10%
  of anchors to have a robust reversal. This statistic localizes which blocks
  can share an ordering, but it is not evidence for H2 because four noisy
  vertices can saturate at \(K_4\).
- Report \(Q\), \(W\), \(B\), and \(\widehat\kappa_{0.10}\) for every system;
  there is no cross-system H2 support gate in NLM-001.

#### H3. Structured pluralism and held-out transfer

Within the two calibration blocks, an anchor is reversal-active when at least
10% of candidate-pair orderings are robust in each full block and have opposing
signs.

- Post-calibration point prediction 1: reversal-active anchor fraction
  \(R=0.20\).
- Post-calibration point prediction 2: on those anchors, native calibration KL
  exceeds the strongest symmetric baseline's held-out pairwise accuracy by
  \(\Delta_{\rm rev}=+0.07\).
- Support gates: \(R\ge0.15\), \(\Delta_{\rm rev}\ge0.05\), and anchor-bootstrap
  95% lower bounds above 0.05 and 0 respectively. The prevalence forecast moved
  because four-of-four robustness reduces eligibility; the transfer-effect
  forecast did not move because calibration produced no baseline comparison.

#### H4. Cross-realization transportability

For every system pair, block, and anchor, compute Kendall \(\tau_b\) between
candidate rankings by \(D_B\). Aggregate by the median over anchors and blocks.

- Point prediction: median cross-system \(\tau_b=0.35\).
- Support gate: lexical-state-bootstrap 95% lower bound above 0.20.
- Also report \(R\) and \(\Delta_{\rm rev}\) separately for each system; do not
  average away a sign disagreement.

### Baselines

1. Raw cosine, Euclidean distance, norm difference, centered cosine,
   coordinate-standardized cosine, and all-but-top cosine on input embeddings.
2. One-step unembedding-law KL.
3. Raw, centered, and all-but-top contextual cosine at the substituted position
   for embedding, one-quarter, one-half, and final layers.
4. Nonnegative diagonal Mahalanobis and rank-16 PSD Mahalanobis metrics fitted
   on calibration-block pairs.

Layer, component count, metric rank, and regularization are selected only by
leave-one-paraphrase-out calibration. Contextual baselines may compute held-out
hidden states but never see held-out KL labels; centering and principal
components are fitted on calibration hidden states only. Metric fits are done
once on all calibration-block pairs among the 72 primary words. Retain
per-anchor calibration-label accuracy for every baseline. In each
anchor-bootstrap replicate, recompute each
baseline's mean calibration accuracy on the sampled anchors, select the
strongest (fixed baseline order breaks ties), and compare native KL with that
baseline on the sampled held-out reversal-active anchors. Thus the bootstrap
reselects the competitor without refitting it; its interval is conditional on
the frozen fits.

Held-out pairwise labels exist only when the ordering is robust separately in
both held-out blocks and the two signs agree. No pooled or one-block label may
replace this intersection.

### Verdict and kill conditions

1. **Directedness withdrawn:** calibration put asymmetry signs at chance. E1 is
   exploratory and cannot support a directional claim in NLM-001.
2. **First non-collapse bet fails:** \(B\le W\) in the primary system, regardless
   of \(\widehat\kappa_{0.10}\). It is killed strongly if the 95% upper bound on
   \(Q\) is at most 1. Graph saturation cannot rescue it.
3. **Predictive novelty killed:** the strongest contextual or learned metric
   matches or exceeds native calibration KL on held-out accuracy
   (\(\Delta_{\rm rev}\le0\)).
4. **Transfer killed:** the 95% upper bound on \(\Delta_{\rm rev}\) is at most
   zero, even if context non-collapse survives.
5. **Cross-realization semantic-stability claim killed:** asymmetry-sign
   agreement is exploratory; the primary claim is killed if rank correlation
   does not beat zero under the preregistered word-label permutation. Recompute
   eligible rankings after each permutation. Within-system structure may still
   be real.
6. **Coordinate-confound verdict:** repaired contextual cosine or a learned
   Mahalanobis metric recovers the result within 0.02 accuracy. Report the
   native novelty claim as null even if raw cosine loses.
7. **Composition-confound verdict:** a pooled effect that is absent in every
   lexical class, or survives in only one class, cannot be called general latent
   closeness.
8. **Instrument void:** revision, tokenization, full-law normalization, or null
   checks fail. Do not repair the protocol after looking at held-out results.

No generation accuracy is reported, so the termination gate is not applicable.
No claim leaves the repository until the held-out, clustered, baseline, and
cross-realization gates above are actually run.

### Round 2 lock changes

Full-vocabulary laws replace coarsening because the exact computation already
fits the CPU budget. H2 support now compares between-context reversals with its
within-context paraphrase null; chromatic number is descriptive. The numerical
null, two-paraphrase robustness, two-block held-out labels, and bootstrap
baseline reselection are specified at implementation resolution. The current
analyzer must add the H2 anchor bootstrap, use both between-block halves, and
retain per-anchor calibration accuracies for genuine within-replicate baseline
selection before confirmatory execution.

### Round 2b instrument amendment

The disclosed eight-item pipeline calibration invalidated instance-specific
paraphrase MAD as a noise threshold: it measured carrier-scale heterogeneity and
made every statistic fail. The locked replacement is per-paraphrase KL-scale
normalization, four-of-four sign agreement, and a magnitude threshold against
\(2\nu_0\) pooled at block and statistic-family level. H1 is exploratory because
its signs were already inspected at chance. H2/H3 predictions are now
post-calibration and pre-confirmatory. The analyzer must implement these rules,
the 72-item primary exclusion, both-half H2 bootstrap, and genuine bootstrap
baseline reselection before execution. Raw matrices were acquired concurrently
before this amendment commit. They remain eligible only if no 72-word outcome
was inspected before the lock; otherwise all of NLM-001 is exploratory.

### Verdict (2026-08-27, primary 72-word analysis; Round 3 audited)

Analysis: `experiments/results/nlm001_v1/analysis_primary_72.json` (rule: 4/4
sign, |median| > max(2ν₀, 10η̃), scale-normalized; 8 calibration words
excluded). Sensitivity: `analysis_sensitivity_80.json`.

| | Qwen3-0.6B (primary) | gemma-3-270m | SmolLM2-360M |
| --- | --- | --- | --- |
| H2 Q = B/W (gate Q≥2, LB>1.5) | **2.12 [1.70, 2.56] — diagnostic gate crossed** | 1.40 [0.90, 2.55] — fail | 17.1 [10.7, 36.3] — denominator pathology: W = 0.005 |
| H3 R (gate ≥0.15, LB>0.05) | 0.18 [0.10, 0.28] — diagnostic gate crossed | 0.14 [0.07, 0.21] — fail | 0.014 — fail |
| H3 Δ_rev vs selected baseline (gate ≥0.05, LB>0) | **−0.058 [−0.22, +0.03] — fail** | +0.017 [−0.02, +0.06] — fail | undefined |
| selected baseline | diagonal Mahalanobis, layer-7 hidden states | diagonal Mahalanobis, layer 4 | diagonal Mahalanobis, layer 32 |
| H1 (exploratory) robust asymmetric pairs, ≥2 blocks | 0.015 | 0.002 | 0.092 |
| H4 median τ_b (descriptive only) | Qwen–SmolLM 0.46; Qwen–gemma 0.14; gemma–SmolLM 0.13 | | |

**Classification: bounded negative falsifier of this instrument.** Formally,
kill condition 8 makes NLM-001 instrument-void for confirmatory claims: the
runner did not record tokenizer revision, library versions, thread count, or
batch size at run time; those fields were reconstructed post hoc. It is not
reported as a positive or negative confirmation of the relational hypothesis.
It is nevertheless a bounded negative falsifier for the decision that matters:
this lexical next-token-KL instrument did not earn further investment. Under
the locked analysis, the primary native predictor lost to the preregistered
diagonal Mahalanobis competitor on reversal-active anchors (0.889 versus 0.947;
\(\Delta_{\rm rev}=-0.058\), 95% interval \([-0.22,+0.03]\)). Thus predictive
novelty fails by kill condition 3 and the coordinate-confound verdict in kill
condition 6 applies. The interval does not strongly kill transfer under kill
condition 4; it kills the claimed advantage.

The H2 number does not rescue the instrument. Qwen crossed its numerical gate,
but the 80-word sensitivity value was 1.97, just below the point gate. SmolLM's
\(Q=17.1\) is dominated by \(W\simeq0.005\), and gemma supported no primary
gate. More fundamentally, \(B>W\) may recover the hand-authored block taxonomy
and differently selected robust-pair subsets; it is not evidence that context
rank is an invariant of the latent space. Near-ceiling performance by the
leading contextual and learned predictors shows that the retained held-out
labels were mostly easy, large-gap pairs.

Directedness was not observed by the registered exploratory endpoint. That
does not establish symmetry or kill context-conditioned directedness; NLM-001
was already incapable of earning a directedness claim. H4 remains descriptive
because the required support interval is not present in the result artifact.

The strongest permissible empirical sentence is: **for these lexical items
and carrier probes, decoder-induced KL orderings differed across authored
context blocks, while the registered point estimate favored a symmetric
learned metric on the model's contextual representation over the native
calibration-KL construct on robust held-out orderings.** Nothing here bears on
"cosine similarity is the wrong object for latent spaces." No possible NLM-001
outcome could have earned that sentence because the design lacked a competitive
decoder-aware geometry, decoder-independent outcomes, presentation changes,
non-lexical replication, and a confirmatory directedness endpoint.

Post-verdict: any endpoint change is NLM-002. NLM-002 is not more lexical words;
it is a competition between primitives, specified in `theory/dialogue/002.md`.

## NLM-002 — map primitive competition on CIFAR-100/DINOv2 (DRAFT)

**Status:** Lock remains valid; Round 6 non-LM run is scored and marked `B3`-inconclusive pending endpoint redesign.
### Locked scope

- One non-LM branch is lockable in this draft: CIFAR-100/DINOv2 cached embeddings.
- LM branch remains unlockable until a dedicated held-out LM continuation
  endpoint is named and frozen.
- CPU-only path only.
- Artifact: 6000 train / 2000 test cached-state split, fixed encoder revision.

### Data and probes

- `X_train`: 6000 images with raw pixels, DINOv2 embeddings, fine/coarse labels,
  and pixel-stat blocks.
- `X_test`: 2000 images, disjoint and held-out for all evaluation endpoints.
- Probe blocks for head training and substitution tables:
  - `PB_fine` (frozen at test time, no head training)
  - `PB_coarse`
  - `PB_rgb_mean`
  - `PB_luma`
  - `PB_edge`
- Fine labels and raw-pixel kNN labels are endpoint-only and are not used for head
  fitting.

### Shared candidate endpoints (endpoint-independence requirement)

Both \(F\) and \(R\) are scored on the same held-out behavioral consequence:
raw-pixel \(k=32\) nearest-neighbor fine-label prediction on \(X_{\text{train}}\),
with cosine/Euclidean distance in raw image space and frozen tie-breaking.
Prediction is whether the kNN label matches the original test fine label after a
substitution move endpoint.

### Design locks

- F uses exact head Fisher on the three/four trained probe blocks:
  `PB_coarse`, `PB_rgb_mean`, `PB_luma`, `PB_edge`. No fine-label head is trained.
- R uses substitution profiles from the same cached pair table and probe blocks.
- Shared pair support: a pair is admissible only if every block in the comparison
  has defined directional consequence for both anchors and both directions.
- Common support estimator:
\[
\mathcal S_U=\{(i,j):\forall c\in U,\;E_c(i,j)=1\},\quad
Q=\frac{B}{W}\text{ on }\mathcal S_{\cdot}
\]
with \(\mathcal S_{\cdot}\) reported as a percentage of all candidate pairs.

### Measurement order (CPU prereg)

#### 1) Chart-path closure in non-LM separatrix geometry

- **Hypothesis:** chart-straight interpolation between same-class and cross-class
  states is not native world transport on this artifact.
- **Exact prediction:** ≥60% of tested interpolation families show flicker/non-
  monotone consequence transitions under the legacy 9-point geometry.
- **Kill condition:** flicker ≤5% with bootstrap CI excluding moderate flicker
  for both same-class and cross-class families.
- **CPU bound:** cached-embedding probe sweeps; linear in move count × 9 points ×
  probe count.

#### 2) Endpoint independence

- **Hypothesis:** raw-pixel fine-label kNN endpoint is not inflated by
  fine-label training.
- **Exact prediction:** replacing any head-based endpoint proxy with the same
  raw-pixel kNN endpoint changes held-out substitution scores by ≤0.02 (paired).
- **Kill condition:** paired endpoint swing >0.02, or any fine-label influence in
  the head training block list.
- **CPU bound:** one index build + one query sweep on CPU.

#### 3) Competitive prediction of fine-label substitution consequence

- **Hypothesis:** one primitive (exact \(F\) or substitution-profile \(R\)) will
  dominate held-out fine-label substitution consequence on common-support pairs.
- **Exact prediction:** paired held-out prediction margin
\(\Delta_{F-R}\ge0.05\) (or \(\Delta_{R-F}\ge0.05\)), with bootstrap CI lower
bound above 0.
- **Kill conditions:** both \(|\Delta|<0.05\), or both are within 0.02 and no
  stable residual \(Q\) contrast is recovered on common support.
- **CPU bound:** matrix operations on cached embeddings and profiles; no model
  retraining loop.

### Manifest lock placeholder

- `cache_sha256`: 8de4f0b0c47e1272de8c948c770517a66edd7beff933ccc29d995886c8145791
  (`experiments/results/vision_cifar100_dinov2s/cache.npz`; encoder revision
  ed25f3a31f01632728cabb09d1542f84ab7b0056; 6000/2000; seed 0; raw pixels in
  `pixels.npz` for the endpoint).
- **Locked 2026-08-27 (Claude), before any scoring**, with two implementation
  decisions the draft left open, recorded here so Codex can object in round 6
  (if it does, results are relabeled exploratory):
  1. Along embedding-space chart paths (measurement 1) the consequence readouts
     are embedding-space: the four trained heads (coarse, rgb, luma, edge) and
     embedding-space k=32 kNN fine label; interpolated embeddings have no pixels,
     so the raw-pixel endpoint cannot be read along a path. Flicker = more than
     one label transition on the 9-point grid; reported per readout and as
     any-readout. 300 same-class and 300 cross-class test pairs, seed 0.
  2. Measurement 3 operationalization: a move is substitution of anchor x by
     candidate y (both test states); its consequence is whether the raw-pixel kNN
     fine label of y equals that of x. Predictors rank candidates by closeness to
     x — F: Fisher pullback distance with G = mean of the four heads' exact pooled
     Fisher on 1000 train states; R: number of trained-head predictions preserved
     under the substitution (profile agreement); baselines: cosine, Euclidean.
     Score = pairwise accuracy over (preserved, not-preserved) candidate pairs per
     anchor, 400 anchors × 40 candidates, anchor bootstrap for Δ_{F−R}.
- Implementation: `experiments/run_nlm002_vision.py`.
### Round 6 Verdict (2026-08-27)

**Status:** Round 6 scoring complete; non-LM run adjudication is partial.

- **Decision 1 (locked implementation):** accepted. Chart-path flicker is measured on embedding-space readouts as locked, because interpolated points have no pixel support.
- **Decision 2 (locked implementation):** operationally coherent, but the fixed independent endpoint is too weak for a primitive claim.

#### M1 result vs prediction

- `PB_coarse` same-class flicker is 0.020 (95% CI [0.003, 0.037]); cross-class flicker is 0.183 (95% CI [0.14, 0.227]).
- `any_readout` flicker is 0.51 same-class and 0.78 cross-class.
- Preregistered requirement of `>=60%` same-class/cross-class non-monotone is not met for same-class chart-straight paths, but cross-class lines are highly non-monotone.
- Interpretation: same-class interpolation is operationally stable and can be treated as a *provisional world-path* regime; cross-class interpolation violates world-path behavior by repeatedly crossing third-class structure.

#### M2 endpoint-control

- Raw-pixel fine-label kNN endpoint: 0.115 accuracy vs 0.761 embedding-kNN fine accuracy, with only 0.1205 agreement.
- This is a >0.02 endpoint swing and exceeds the registered independence-kill threshold.
- The endpoint remains conceptually clean (true fine label, no fine head), but it is not informative enough to resolve the primitive competition.

#### M3 competition

- With endpoint invalidated, `F` vs `R` is a tie: `F=0.6014`, `R=0.6050`, `\Delta_{F-R}=-0.0036` (95% CI [-0.0343, 0.0261]), `n=16660`.

#### Verdict and next-round design under the guiding question

- Mark `NLM-002` non-LM branch as **B3-inconclusive / exploratory** until endpoint redesign is locked.
- Keep the true fine-label endpoint idea (no fine-label head trained) as the independence requirement.
- Redesign the independent consequence target so it is both independent and informative before finalizing any `F` vs `R` decision.
- Carry forward the path predicate: compare primitives primarily on same-class chart paths that survive the M1 world-path criterion; treat cross-class paths as non-world in this run.
## NLM-003 — true-fine consequence competition (LOCK)

**Status:** LOCKED (`cache_sha256=8de4f0b0c47e1272de8c948c770517a66edd7beff933ccc29d995886c8145791`; unchanged artifact; no GPU).

- **Endpoint:** `--endpoint fine_label` in `run_nlm002_vision.py` (true test fine label), no fine-label head is trained.
- **Hypothesis:** substitution-transport is better captured by profile continuity than Fisher geometry; paired effect target is `Delta_{F-R} <= -0.05` (R wins by at least 5pp).
- **Construction (unchanged):** exact-Fisher `F` from `PB_coarse`, `PB_rgb_mean`, `PB_luma`, `PB_edge`; `R` from profile agreement on the same probes.
- **Common-support rule:** score only on
  \(S_U = \{(i,j): \forall c\in U, E_c(i,j)=1\}\) (both anchors and both directions defined); abort directional claim if support is empty.
- **Competitors:** `cosine` and `euclid` remain fixed baselines.
- **Gate:** require paired `Delta_{F-R}` and bootstrap CI excluding zero with one-sided lower bound `< -0.05`; kill if `|\Delta|<0.02`, if paired endpoint sensitivity `>0.02` in M2, or if support collapses.
- **Inference:** anchor bootstrap (1000 resamples) over mean pairwise score differences.
- **Decision output:** report `(Delta_F_minus_R, CI95, n_pairs, Q)` first, then pass/fail.

## Round 8 — NLM-003 adjudication: chart wins; R correction

The original lock appeared to pass because `R` reached 0.7343 mean anchor
accuracy against `F` at 0.6303, with `Delta_{F-R} = -0.1040` and 95% CI
`[-0.1478, -0.0584]`. The required sensitivity rerun changes the primitive
interpretation: with `PB_coarse` removed, `R` reaches only 0.5860 while `F`
reaches 0.6668, with `Delta_{F-R} = -0.0950` and 95% CI
`[-0.1422, -0.0490]`. The apparent `R` win was a taxonomy leak: fine labels
nest inside the coarse classes carried by the original profile.

Therefore the original `R`-over-`F` directional claim is **withdrawn**. Round
8 retains only the narrower chart-vs-native comparison: cosine (0.9341 in the
diagnostic artifact; 0.9464 in the original table) and Euclidean distance
outperform both leak-free native candidates on the same 129 supported anchors.
The result is still conditional on thin support and does not establish native
geometry. The correct classification is: **NLM-003 is a corrected narrow
instrument comparison; `R` without the coarse head does not beat `F`, and no
tested native candidate earns replacement status.**

Under the guiding question, this means that in this measured world a denizen can
currently navigate fine-label consequences most accurately by reading ordinary
distances in DINOv2's chart. Is that already the native geometry because the
encoder was trained to make it so? **No, not in the strong sense.** The weak
operational reading is that DINOv2's
pretraining has made chart proximity carry visual regularities that happen to
align with fine-label preservation; it is not evidence that the chart is an
intrinsic native geometry, and DINOv2 was not trained on these CIFAR fine labels.
A world in which this fails would preserve the same latent states while a
declared nonlinear re-chart, a domain shift, or a composed/unseen move changes
cosine and Euclidean rankings and destroys their consequence prediction, while
a transport- or profile-based map continues to predict the held-out outcome.
That is the distinction between a chart that is useful in this world and a map
that belongs to the world.

### Next measurement

This next measurement was superseded by the Round 10 closure below. The frozen
encoder line must not spend another round on arbitrary chart reparameterizations
or transports already included in the encoder's trained invariance class.

### Tier-3 audit #2 corrections (2026-08-27, fresh unprimed auditor; adopted verbatim)

- NLM-003 is a **narrow instrument comparison**: "these implementations lose to
  cosine on this endpoint." It is not evidence that native geometry is
  generally dominated (one encoder, one endpoint, one-step random
  substitutions, one seed, 130 supported anchors).
- R has only five possible values (0–4 agreeing heads) and ties receive 0.5
  credit; tie counts and a tie/outcome cross-tab are not reported and must be.
  R includes the coarse head, and fine labels are nested inside coarse classes,
  so R carries taxonomy-derived signal; rerun R without the coarse head.
- NLM-002 M1: the 2% coarse-head flicker is weak evidence (affine argmax is
  near-monotone by construction); kNN flicker is reported at k = 32 only — a k
  sensitivity analysis is required before any world-path claim.
- Next gate must combine: R without the coarse head, full tie accounting, kNN
  sensitivity, the random-init null (NLM-004, preregistered in the ledger),
  a cheap-baseline ladder under the same endpoint (random-init features,
  PCA/color/edge features, nearest-centroid), nonlinear re-charting, and
  composed / out-of-distribution moves. The program continues only if it finds
  a held-out consequence that simple chart metrics cannot explain.

## NLM-004 — random-init null-world adjudication (Round 9)

**Status:** **SUPPORTED AS EXPLORATORY NULL-WORLD EVIDENCE** against the
registered point-estimate thresholds; no claim of native geometry. The ledger
preregistration was written before scoring and is not downgraded merely because
Claude wrote it. However, `analysis.json` does not contain the preregistered
anchor-bootstrap CIs for the trained-versus-null cosine and embedding-kNN
comparisons, so the stronger confirmatory CI clause is not auditable here.

The lock's registered predictions are met on the reported estimates:

- cosine fine-label consequence accuracy is `0.575` in the random-init chart
  versus `0.946` in the trained chart, a trained-minus-null gap of `0.371`
  (registered thresholds: null `<=0.70`, gap `>=0.20`);
- embedding-kNN fine-label accuracy is `0.069` in the null versus `0.761` in
  the trained chart (registered null threshold: `<=0.25`);
- the null does not make the native candidates competitive: `F=0.581`,
  `R=0.568`, and `R` without the coarse head `=0.546`, with `R` ties at
  `0.329` (the tie accounting is now retained as a required diagnostic, not a
  new native result);
- the null chart's pixel-statistic heads remain predictive (`PB_rgb_mean`
  `0.8335`, `PB_luma` `0.8205`, `PB_edge` `0.531`), while its coarse head is
  only `0.2075`. This is the expected cheap-baseline confound: a random net can
  carry simple image statistics without carrying the trained semantic map.

The M1 null is the sharper separation. Same-class fine-kNN chart-path flicker
is `0.953` in the null (`k=32`; `0.977` at `k=8`, `0.867` at `k=128`) versus
`0.127` in the trained artifact. Null cross-class fine-kNN flicker is `0.987`,
so the null has no useful same-class world-path separation; the pixel-statistic
heads' accuracy is not evidence that it has one.

Under the guiding question, training has done two related but distinct things:
it made chart nearness predict fine-label consequences, and it made affine
segments in that chart look smooth under the trained readouts. A denizen of the
trained world can therefore use the chart both as a local map and as a usable
first approximation to a path. The null says this is not a generic property of
random coordinates or pixel-statistic readouts. It still does **not** show that
chart straightness is an intrinsic law: affine interpolation
`(1-t)x + ty` is an imported chart path, and the readout is itself tied to the
trained representation. Composition and transport are required to distinguish
a learned, useful chart from a chart-independent native geometry.

The adjudication therefore supports: **training creates a task-effective chart
and affine-path smoothness under a trained representation for this
encoder/dataset.** It does not support:
"cosine is native," "cosine is merely arbitrary," or any general claim beyond
this encoder, endpoint, and one random-init seed. Artifact:
`experiments/results/nlm004_v1/analysis.json`; preregistration:
ledger id `nlm004_prereg_null_world`.

## NLM-005 — composed transport/substitution gate (LOCK)

**Status:** LOCKED as the next gate. CPU-only; no experiment is run by this
lock. The `nlm003_v2_diagnostics` rerun is a **sensitivity rerun of NLM-003,
not new evidence** and cannot change the Round 8 verdict by itself.

### Hypothesis

Training can make one-step chart nearness and chart-straight paths coherent
without making vector-chart composition a native law. On a held-out composed
move, substitution and transport will expose order dependence that a simple
chart metric cannot explain, while `R` without the coarse head or `F` will be
more stable. If the chart survives both orders with its current margin, the
native-map hypothesis is killed for this world and the result is an
operationally native chart only.

### World, moves, and endpoint

- Keep the NLM-003 DINOv2 cache, true fine-label endpoint, held-out test pool,
  400-anchor/40-candidate layout, and common-support accounting. No
  fine-label head is trained.
- Substitution is the existing move `S_y(x)=y`: replace anchor image `x` by
  candidate image `y`. Its consequence is the inherited true fine-label
  relation `fine(y)=fine(x)`, scored by paired candidate ranking.
- Transport is concrete image-world transport, not a latent proxy. For each
  test image `I`, apply two fixed, label-preserving edits selected before
  scoring: horizontal reflection and a one-pixel right translation with
  declared padding. Re-encode each edited image with the same frozen DINOv2
  encoder on CPU, `T_e(x)=E(e(I_x))`. The original image and edit parameters
  are retained so the operation is reproducible and its inherited label is
  explicit.
- For each `(x,y,e)`, evaluate both order-conditioned composed pairs:
  `ST: x -> S_y(x) -> T_e(y)` and `TS: x -> T_e(x) -> S_y`, where the final
  `S_y` deliberately uses the unedited candidate `y`. Thus the two paths are
  not assumed to commute: the scored pairs are `(x,T_e(y))` and `(T_e(x),y)`
  against the same fine-label-preservation outcome. Direct unedited
  substitution and direct transport are controls.
- Score cosine, Euclidean, `F`, and `R` without `PB_coarse`; retain raw-pixel,
  PCA/color/edge, and random-init chart baselines where available. Report
  anchor-bootstrap CIs, support, pair counts, and all ties.

### Exact predictions and decision rule

- **Primary composition prediction:** on at least one of the two edit families,
  the trained chart's ST-versus-TS ranking changes by at least `0.05` in
  paired mean accuracy, while the direct unedited control does not; and the
  best native candidate (`F` or `R` without coarse) leads the best simple chart
  metric by at least `0.05` with a paired bootstrap lower bound above zero on
  that order-sensitive family.
- **Chart-survival alternative:** if cosine/Euclidean lead the best native
  candidate by at least `0.05` on both ST and TS, with at least `80%` anchor
  support, NLM-005 kills the native-rescue hypothesis and records the chart as
  operationally native for this measured world.
- **Non-diagnostic outcome:** an ST-versus-TS gap below `0.02`, support below
  `80%`, or a missing/unstable endpoint makes the composition result void; it
  cannot be narrated as either native success or chart success.

### Required sensitivity accounting

Rerun the original NLM-003 table with `R` formed from only
`PB_rgb_mean`, `PB_luma`, and `PB_edge`; report exact tie counts, total
comparisons, tie fraction, and a tie-by-outcome cross-tab. Report nonlinear
kNN-fine M1 flicker at `k in {8,32,128}` for same- and cross-class paths.
These are diagnostics of NLM-003 stability, not fresh evidence. If the
headline direction or any relevant score moves by more than `0.02` across the
registered k values, or if ties are not fully accounted for, no native claim
may be promoted.

### Kill conditions

1. Cosine/Euclidean retain a `>=0.05` lead over both native candidates on both
   composed orders with `>=80%` support: kill the native-rescue hypothesis.
2. ST and TS are within `0.02` for every metric: kill the order-sensitive
   composition claim as non-diagnostic.
3. Support falls below `80%`, the endpoint is not independent, or either edit
   is not applied and re-encoded exactly as locked: void the gate.
4. kNN sensitivity changes the NLM-003 directional interpretation by `>0.02`
   or tie accounting is incomplete: retain the Round 8 narrow verdict but
   block any stronger continuation claim.

### CPU cost

Re-encoding the 2,000 held-out test images for each of two edits at the
observed CPU rate of approximately `35 ms/image` is about `140 seconds` of
encoder time, plus at most ten minutes for feature, ranking, bootstrap, and
diagnostic scoring. The full lock is budgeted at `15 CPU-minutes`, one process,
with no GPU. The cached NLM-003 sensitivity rerun is separately budgeted at
`5 CPU-minutes` and does not count as new evidence.

### NLM-005 artifact lock (Claude, 2026-08-27, before scoring)

- Transports: `experiments/results/vision_cifar100_dinov2s/edits.npz`, sha256
  `c6d7cd251d716124ada1c1bc2950c84977db61ff9dbdc1272aa930e42a513b13` — test
  split (2000) under `hflip` (horizontal reflection) and `shift1px` (one-pixel
  right translation, edge-replicate padding), re-encoded by the frozen
  DINOv2-small encoder (revision ed25f3a3…) with the same preprocessing.
- Runner: `experiments/run_nlm002_vision.py --edits <edits.npz> --endpoint
  fine_label` (measurement 4). Predictors: cosine, Euclidean, F (Fisher
  pullback, 4 heads), R without `PB_coarse`. 400 anchors × 40 candidates;
  anchor bootstrap (1000).

## Round 10 — close the frozen-chart line; move outside invariance

### NLM-005 adjudication

**Status: VOID AND NON-DIAGNOSTIC.** NLM-005 scored only 129/400 anchors
(`32.25%`), below the locked `80%` common-support requirement. This alone
triggers kill condition 3 and prevents either a native-rescue or chart-success
claim from the gate. On `hflip`, every ST−TS gap is at most `0.0064`; on
`shift1px`, the chart metrics' gaps are at most `0.0036`, while
`R_no_coarse` is `0.0271`. Thus the shorthand claim that every metric has a
gap no larger than `0.006` is not literally true for that one sensitivity row,
but the support failure makes the result void regardless. There is no robust
order-sensitive composition signal.

Cosine remains ahead of the best native candidate on every scored ST/TS order:
the native-minus-chart differences range from about `-0.314` to `-0.326` for
`hflip` and `-0.323` to `-0.325` for `shift1px`. This is descriptive evidence
on a thin, improperly supported sample, not a confirmation of chart survival
under general transport. The edits were hflip and one-pixel translation, both
augmentations the encoder was trained to make approximately invariant to; they
were near-identity moves in this learned world. The 40-random-candidate design
over 100 classes also made an 80% support target structurally implausible.

### Program closure and residue

Close the **frozen-encoder closeness/map competition** as a program. This does
not promote cosine to intrinsic geometry and does not say that every transport
preserves the chart. It records the five-measurement residue:

1. in a trained world, chart nearness is the best measured one-step map for the
   tested consequence;
2. chart-straight paths are coherent under the trained readouts;
3. the corresponding map and paths collapse in a random-init chart;
4. the tested native candidates do not compete after the coarse-taxonomy leak
   is removed; and
5. the only tested transports were trained-invariant near-identity edits, so
   NLM-005 cannot extend the conclusion beyond that class.

The honest guiding-question answer is therefore: a denizen inherits a useful
chart and some chart-path regularities from training. Those are navigation
equipment supplied by the world's history, not yet laws native to the latent
world. Further frozen-encoder score-chasing on the same class of moves is
closed; the transferable residue is the requirement to identify the world's
admissible moves before calling a chart metric native.

### NLM-006 design: stratified transports outside the invariance class

The replacement line is a transport audit using four predeclared edit families:
large crops, color inversion, image mixing, and occlusion. Each edit must be
specified before scoring and must be verified to produce a non-near-identity
embedding displacement on calibration images. Use held-out image identities,
the true fine-label outcome, and a stratified candidate pool: 20 same-fine-class
candidates plus 20 cross-class hard negatives per anchor, with the strata and
candidate identities frozen before metric scoring. Report each stratum
separately; stratification makes support achievable but is not itself evidence
for a semantic claim.

**Hypothesis.** If chart nearness is inherited navigation equipment rather than
a law of the world, at least one transport family outside the trained
invariance class will break its held-out consequence ranking or expose a
transport-aware native predictor that remains stable when cosine/Euclidean do
not. If all genuinely displaced families preserve the chart lead, the useful
chart may be the world's operational map for this tested move envelope.

**Decisive result.** Support the replacement hypothesis if at least two of four
edit families, at `>=80%` anchor support, show either (a) a transport-aware
native predictor leading the best chart metric by `>=0.05` with a paired
anchor-bootstrap lower bound above zero, or (b) the chart lead collapsing to
`<=0.02` on a predeclared composition/order test while the direct control stays
stable. A chart lead of `>=0.05` on all four families, with measured
non-near-identity displacement and the same support, closes this replacement
too and leaves the chart as the operational map for that measured envelope.

**Kill conditions.** Kill the outside-invariance hypothesis if the chart retains
the preregistered lead on every valid family. Void the run if any edit remains
inside the encoder's measured invariance class, if candidate selection changes
after scoring, if support falls below `80%` despite stratification, or if the
endpoint is not held out from metric construction. A single successful edit is
not enough to overturn the five-measurement residue.

**CPU cost.** Four edits over the 2,000-image held-out split are approximately
`280 seconds` of re-encoding at the observed `35 ms/image`, plus at most ten
minutes for stratified ranking, composition, and anchor-bootstrap scoring: a
roughly 15-minute CPU run, one process, no GPU.

### NLM-006 artifact lock (Claude, 2026-08-27, before scoring)

- Transports: `experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`,
  sha256 `9cc0e7c082dab6c0dfb804198388154eb7c62aaa830ead3e698985e0756d0d0b`;
  test split (2000) re-encoded by the frozen DINOv2-small encoder under
  `crop50` (central 50% bicubic-upscaled), `invert` (255−x), `mix50`
  (0.5·x + 0.5·partner, partner permutation seed 6, stored), `occlude50`
  (central 50% zeroed); `hflip` and `shift1px` retained as near-identity
  controls.
- Displacement check (200 held-out states, `displacement.json`), mean
  cosine(x, T_e x): hflip 0.959, shift1px 0.976, crop50 0.604, invert 0.467,
  mix50 0.422, occlude50 0.664. **Predeclared threshold: a family is outside
  the invariance class iff mean ≤ 0.80.** The four new families qualify; the
  two controls do not.
- Candidates: stratified — up to 20 same-fine-class + 20 cross-class hard
  negatives (nearest by cosine among other classes), frozen by seed; 400
  anchors; endpoint = true fine label; support gate ≥ 80%.
- Predictors (unchanged constructions): chart = cosine, Euclidean; native = F
  (Fisher pullback, 4 heads), R without `PB_coarse`. "Transport-aware" means
  the predictor is scored on the transported pair (ST: (x, T_e y); TS:
  (T_e x, y)), i.e. it reads the world's response to the move; the direct
  pair (x, y) is the control.
- Note on the hard-negative pool: negatives are selected by cosine, which
  biases the contest *against* the chart metric; this is conservative for the
  hypothesis "some move breaks the chart" and is reported as such.
- Runner: `experiments/run_nlm002_vision.py --edits <edits_v2> --stratified
  --endpoint fine_label`.

### Tier-3 audit #3 corrections (2026-08-27, fresh unprimed auditor; adopted verbatim)

- **Residue narrowed:** "training creates a task-effective chart and
  affine-path smoothness in this encoder/dataset" — not "native mathematics
  has been found" and not "the frozen-encoder line is scientifically closed".
  Round 10's closure is scope management.
- **NLM-004:** supports training-dependence of the measured behavior (trained
  cosine 0.934 vs null 0.575; path flicker 0.127 vs 0.953). "Straight routes
  inherited from training" is too strong: the route is affine interpolation,
  an imported chart path; the result shows chart smoothness under a trained
  representation. The random-init network is a narrow null (architecture and
  preprocessing without learned weights), not a native-geometry null; one
  seed; anchor-bootstrap CIs for the trained/null comparison are missing.
- **NLM-006 as locked (v1) cannot support its positive branch:** the
  transport-aware native predictor was undefined; cross-class hard negatives
  are selected by cosine and cosine is then scored on them (selection on the
  tested metric); inversion/mixing/crop/occlusion are not verified
  label-preserving, so a chart failure could be identity destruction (OOD),
  not transport law; the invariance threshold was a raw cosine number, not a
  rule relative to an inside-invariance control. **NLM-006 v1 is relabeled
  exploratory.** Repaired design (NLM-006b): independently selected candidates
  (random same-class + random cross-class); explicit transport-aware
  predictors R_T and F_T (profiles / Fisher distance read on the transported
  pair (T_e x, T_e y)) with chart-on-transported as the matched control;
  label-preservation rate per family (embedding-kNN fine label of T_e x equals
  fine(x); families below a predeclared rate are OOD families, reported
  separately); displacement gate: the family's q10 displacement
  d = 1 − cos(E x, E T x) must exceed the near-identity control's q95 on ≥ 80%
  of calibration images.
- **Program drift flagged:** the measured line is one-step consequence
  ranking, affine interpolation, and edits through one frozen encoder — closer
  to an encoder-invariance study than to native mathematics. NLM-006b restores
  the question only if legal transport, identity preservation, cost, and the
  native predictor are defined before scoring.

## NLM-006b — calibrated transport audit (LOCK, Round 11)

**Status: LOCKED BEFORE SCORING.** NLM-006 v1 is exploratory only. NLM-006b
tests whether a transport-aware predictor survives independently selected
candidates and edits that are both displaced in the encoder and plausibly
label-preserving. The narrow question is whether the frozen encoder's
task-effective chart remains the best measured map for this transport envelope;
the result cannot by itself establish intrinsic native mathematics.

### World, artifact, and candidate lock

- Keep the v2 transport artifact and its stored partner permutation fixed:
  `experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`, sha256
  `9cc0e7c082dab6c0dfb804198388154eb7c62aaa830ead3e698985e0756d0d0b`.
  The four test-split families are `crop50`, `invert`, `mix50`, and
  `occlude50`; `hflip` is the inside-invariance calibration control and
  `shift1px` is a secondary near-identity control. The artifact's descriptive
  mean cosine values are respectively `0.604`, `0.467`, `0.422`, and `0.664`
  for the four new families, versus `0.959` for hflip and `0.976` for shift1px.
  These means are not the displacement gate.
- Use 400 held-out anchors and, for each anchor, independently sample without
  replacement up to 20 random candidates with the same true fine label and 20
  random candidates with a different true fine label. Candidate identities and
  the family artifact are frozen before any metric is evaluated. The fine
  labels define the two sampling strata only; no tested metric may select,
  reorder, or replace candidates. The endpoint is the true held-out fine label;
  no fine-label head is trained.
- The primary run is the runner's `--independent --endpoint fine_label` path
  with the locked artifact and seed. Direct unedited pairs, `ST=(x,T_e y)`,
  and `TS=(T_e x,y)` remain controls; the primary transport-aware comparison is
  `TT=(T_e x,T_e y)`.

### Transport-aware predictors and matched chart controls

Let `E` be the frozen encoder, `\tilde{x}=T_e x`, and `\tilde{y}=T_e y`.
Higher scores mean "closer" for every predictor. Let `G` be the fixed average
Fisher pullback computed from the four label-free probe heads on the training
embeddings, with no test edits or fine labels entering its construction. Define:

- `F_T(x,y;e) = -(\tilde{y}-\tilde{x})^T G (\tilde{y}-\tilde{x})`.
- `R_T(x,y;e) = sum_{h in U} 1[h(\tilde{x}) = h(\tilde{y})]`, where
  `U={PB_rgb_mean, PB_luma, PB_edge}` and each `h` is the fixed training-fit
  head. The coarse head is excluded, so this is the locked `R_no_coarse`
  construction evaluated after transport.
- The matched chart controls on the same transported pair are
  `cosine_T(x,y;e) = cos(E(\tilde{x}),E(\tilde{y}))` and
  `euclid_T(x,y;e) = -||E(\tilde{y})-E(\tilde{x})||_2`.

For each anchor, pairwise accuracy asks whether the predictor ranks a
same-fine-label candidate above a different-fine-label candidate. Report
`F_T`, `R_T`, `cosine_T`, and `euclid_T` on the identical pairs, along with the
direct, ST, and TS controls. The best native is the larger of `F_T` and `R_T`;
the best chart is the larger of `cosine_T` and `euclid_T`. Selection of the best
member is made once from the preregistered family table, not per bootstrap
replicate.

### Label-preservation and displacement gates

For every family `e`, predeclare

`p_e = mean_i 1[kNN_32(E(T_e x_i), X_train) = fine(x_i)]`

over the full 2,000-image held-out split, using the fixed training embedding
index and its fine labels. A family counts as **label-preserving** only when
`p_e >= 0.80`. A family with `p_e < 0.80` is an **OOD family**: report its
displacement and scores separately, but do not use it as evidence for either a
native transport law or chart failure. This is a diagnostic gate, not a claim
that embedding-kNN is the world's true label oracle.

On the fixed 200-image calibration subset, define the image-wise displacement
`d_e(i)=1-cos(E(x_i),E(T_e x_i))`. Let `q95_hflip` be the 95th percentile of
the hflip `d` values on that same calibration subset. A family is outside the
measured invariance class only if its **q10 displacement criterion** is met in
the calibrated sense

`g_e = mean_i 1[d_e(i) > q95_hflip] >= 0.80`.

Report `q10(d_e)`, `q95_hflip`, and `g_e` for every family. The strict
image-wise comparison is the gate; the old raw-mean `cosine <= 0.80` rule is
retired. A family failing this gate is an inside-invariance family and is
non-diagnostic for the outside-transport question.

### Exact preregistered predictions

The following are predictions, not observed results. They are registered per
family so that an outcome cannot be retrofitted into the aggregate "two of
four" statement:

| family | predicted eligibility | predicted transport result |
|---|---|---|
| `crop50` | `p_e >= 0.80`, `g_e >= 0.80`, support `>= 320/400` | best-native `TT` lead over best-chart `TT` `>= +0.05`, paired anchor-bootstrap 95% lower bound `> 0` |
| `invert` | `p_e >= 0.80`, `g_e >= 0.80`, support `>= 320/400` | best-native `TT` lead over best-chart `TT` `>= +0.05`, paired anchor-bootstrap 95% lower bound `> 0` |
| `mix50` | `p_e < 0.80` (OOD); displacement is expected to pass but is not sufficient for validity | no native/chart verdict; report as OOD and do not count it toward either branch |
| `occlude50` | `p_e >= 0.80`, `g_e >= 0.80`, support `>= 320/400` | best-native `TT` lead over best-chart `TT` `>= +0.05`, paired anchor-bootstrap 95% lower bound `> 0` |

The family-level native-rescue prediction is therefore at least two valid
families meeting the `+0.05` criterion. A family that instead has a stable
direct control but a best-chart-versus-best-native `TT` lead of `<= +0.02`
counts as chart-breakdown evidence only if it is label-preserving, outside the
calibrated invariance class, and meets the same support gate.

### Support, decision, and kill rules

- Support is the fraction of the 400 anchors with both a same-label and a
  different-label candidate and a non-degenerate pairwise comparison. Require
  `>= 0.80` (`>=320/400`) for each family used in a verdict; report same-label
  and cross-label stratum counts separately. The stratified design does not
  itself count as evidence.
- Reopen the native-transport line only if at least two families simultaneously
  pass label preservation, calibrated displacement, and support, and each has
  a best-native `TT` lead `>= +0.05` over the matched chart control with a
  paired anchor-bootstrap 95% lower bound `> 0`. The direct/ST/TS controls must
  be reported so a generic endpoint collapse cannot be mistaken for transport
  structure.
- Finally close the frozen-encoder transport line for this measured envelope if
  every valid family retains a best-chart `TT` lead `>= +0.05` over the best
  native predictor with paired lower bound `> 0`, at `>=320/400` support, and
  no valid family meets the native-rescue criterion. This closes the narrowed
  empirical line, not native mathematics in general.
- A family below `0.80` label preservation or below the displacement gate is
  OOD/inside-invariance and is reported separately, not counted as a kill or a
  rescue. If fewer than two families are valid, the round is non-diagnostic,
  not a closure.
- Void the run if candidate identities are changed after scoring, the endpoint
  enters metric construction, the artifact or calibration subset changes, the
  paired bootstrap is absent, or the support accounting is not auditable.

No NLM-006b scoring was part of Round 11. Round 12 adjudicates the resulting
artifact below; the four family predictions remain historical preregistration
claims, not retrofitted interpretations.

## Round 12 — NLM-006b adjudication and frozen-encoder closure

**Status: NON-DIAGNOSTIC UNDER THE LOCK; PROGRAM CLOSED BY SCOPE, NOT BY A
NATIVE-LAW VERDICT.** The locked artifact was scored with independent
candidates, the true fine-label endpoint, paired bootstrap intervals, and
400/400 supported anchors for every family. The result must be adjudicated
against the predeclared identity gate before reading the chart comparison.

| family | fine-label preservation (p_e) | label gate | displacement gate | support | TT chart lead over best native |
| --- | ---: | --- | --- | ---: | ---: |
| `crop50` | 0.458 | fail; OOD | pass (1.000) | 400/400 | +0.208 |
| `invert` | 0.317 | fail; OOD | pass (1.000) | 400/400 | +0.227 |
| `mix50` | 0.185 | fail; OOD | pass (1.000) | 400/400 | +0.090 |
| `occlude50` | 0.416 | fail; OOD | pass (0.985) | 400/400 | +0.222 |

The predeclared label-preservation threshold was `p_e >= 0.80`. Therefore
**zero of four families is valid** for the native-transport or gated
chart-failure branches. The near-identity controls are themselves only
0.772 (`hflip`) and 0.761 (`shift1px`), which makes this kNN readout a weak
calibration proxy; it does not permit lowering the locked threshold after
seeing the outcomes. The displacement result is still clear: all four new
families pass the calibrated outside-invariance gate, and support is complete.

Descriptively, the chart-survival pattern is strong on the transported pair:
`cosine_T` leads `F_T` by 0.208, 0.227, 0.090, and 0.222 for the four families,
with paired 95% intervals for native-minus-chart entirely below zero. But
these are identity-destroying OOD edits under the lock, not gated evidence
that the chart survives legal semantic transport. The prior ledger wording
that this result "closes the replacement line" is corrected: NLM-006b is
**non-diagnostic for the locked native/chart decision**, and it neither
reopens native transport nor earns a lock-valid chart-survival closure.

There is one real composition signal outside the measured invariance class:
the chart's `ST-TS` order effect is about +0.034 to +0.036 for `crop50`,
`invert`, `mix50`, and `occlude50`, with paired intervals excluding zero.
The hflip control is about +0.001 with an interval spanning zero. This is a
small non-commutation signal, not a native-predictor win; because the displaced
families fail identity preservation, it cannot be promoted to a law of
semantic transport.

The frozen-encoder program nevertheless closes as a **scope decision**. The
measured residue is a trained representation's task-effective chart metric,
affine-path smoothness, and relative chart robustness as edits progressively
destroy the endpoint identity. No tested native construct competes. No further
score-chasing on this frozen image encoder is planned. The next program must
make dynamics—not analyst-imposed image edits—the legal moves.

## NLM-007 — residual-stream dynamics law-complexity audit (LOCK, amended Round 14)

**Status: LOCKED BEFORE SCORING.** This is a documentation-only lock. No NLM-
007 scoring or generation is part of Round 13. The narrow question is whether
the forward-pass transport of a causal LM admits a reusable law across unseen
carrier contexts. A win by a regressor over a coordinate-nearness control is
not, by itself, evidence of native mathematics; it must transfer across
carriers and cash out in the world's completed response law.

### C1--C5 adjudication

- **C1 — concede and amend.** A ridge field and kNN on residual coordinates are
  both chart constructions. Ridge beating 1-NN establishes, at most, that the
  measured block action is more smoothly represented by an affine field than
  by that local chart rule. The old claim that this alone finds a native map is
  withdrawn.
- **C2 — concede and adopt.** The primary comparison is a law-complexity
  ladder: global mean successor; kNN regression with `k={1,5,20}`; affine
  ridge; low-rank affine; and RBF kernel ridge. The minimal class is reported
  retrospectively as the first class within `0.02` of the best held-out score,
  separately for successor and completed-law endpoints; it is not used to
  choose a result after seeing the data.
- **C3 — concede and adopt with a ceiling boundary.** The primary split holds
  out one complete carrier block at a time. A per-carrier, word-cross-fitted
  oracle is a ceiling for how much predictable structure exists within a
  carrier, not evidence of a reusable cross-carrier law and not a competing
  primary method.
- **C4 — concede and amend for implementation.** A successor cosine is only
  coordinate forecasting. For each predicted slot successor, the analysis
  must retain or deterministically reconstruct the actual non-slot hidden
  sequence at that successor depth, replace only the slot, run the remaining
  transformer blocks plus final norm and language-model head, and compare that
  completed law with the true final law by KL and ordering preservation.
- **C5 — adopt with an instrument boundary.** The existing capture stage
  records all hidden-state indices and final laws, the batched-vs-single
  numerical null, and runtime metadata. The lexical configuration supplies 80
  one-token words and 16 carriers in four blocks of four paraphrases. It must
  be run as exactly Qwen3-0.6B with 28 hidden layers; the analysis stage is
  built only after this lock. The current slot-only `states.npz` is sufficient
  for successor scoring but not, alone, for the world-completed endpoint.

### Locked world, cells, and layer pairs

Use the unchanged `experiments/config/lexical_probe_v1.json`: 80 one-token
words crossed with the 16 fixed carrier probes
`{gloss_1..4, cont_1..4, assoc_1..4, gram_1..4}`. A cell is `(carrier, word)`.
The carrier blocks are `gloss`, `continuation`, `association`, and `grammar`;
the primary split is therefore four outer folds, each holding out all four
paraphrase carriers in one block and training on the other 12 carriers. Words
are intentionally shared across carriers: this is a carrier-transfer law
test, not a vocabulary-generalization test.

The model is `Qwen/Qwen3-0.6B`, CPU float32 evaluation, with no text
generation. Require `num_hidden_layers == 28`; hidden-state index 0 is the
input-embedding state and indices 1--28 are the transformer outputs. Analyze
these six predeclared adjacent pairs:

| depth region | pair | interpretation |
|---|---|---|
| early | `L0→L1`, `L4→L5` | embedding/early block transport |
| middle | `L8→L9`, `L12→L13` | middle residual transport |
| late | `L20→L21`, `L27→L28` | late and final-block transport |

For every pair, let `X=z_l,c(w)` and `Y=z_(l+1),c(w)`. Model and tokenizer
revision, exact config hash/name, batch size, CPU thread count, Python,
PyTorch, Transformers, dtype, device, layer count, and the capture artifact
hash must be present before held-out scores are opened. The existing
batched-vs-single null is mandatory and is reported for states, log laws, and
KL; missing or non-finite null values void confirmatory interpretation.

### Law-complexity ladder and controls

All coordinate preprocessing is fit on the outer-fold calibration cells only:
per-coordinate centering/scaling for distance and regression, with zero-scale
coordinates omitted. Predictions are transformed back to the original
residual coordinates before endpoint scoring. Hyperparameters are selected by
leave-one-carrier-block-out validation within the three calibration blocks;
the held-out block is never used for selection.

1. **Mean:** the calibration mean of `Y`, independent of `X` and carrier.
2. **Chart-local:** kNN regression for `k=1,5,20`, using standardized
   Euclidean distance in `X`, with calibration targets only.
3. **Affine:** centered ridge `Y=b+XW`, with
   `lambda={1e-4,1e-3,1e-2,1e-1,1,10,100}`; and reduced-rank affine fields
   with `rank={8,32,128}` and the same ridge grid. The selected ridge and
   low-rank members are reported separately.
4. **Nonlinear:** RBF kernel ridge, with regularization from the same grid and
   `gamma={0.1,1,10}/median(||X_i-X_j||^2)` computed inside calibration.

The static chart controls are 1-NN successor lookup by cosine similarity and
by negative Euclidean distance in the unmodified residual chart. Their member
is selected once by the same inner blocked validation and then frozen. They
are controls, not native candidates. The carrier-shuffled null independently
permutes the 12 calibration targets across carriers within each word, using
the fixed RNG seed `13007`; this preserves each word's target marginal while
breaking carrier pairing. Use 100 such permutations and report their null
distribution. A shuffled field that matches the unshuffled field indicates a
marginal-state or presentation artifact, not a transport law.

The per-carrier oracle is fit separately within each carrier using a fixed
five-way, class-stratified word split over the 80 words; each oracle is tested
on its held-out words and never supplies predictions to the cross-carrier
field. Report oracle performance as the within-carrier ceiling. For bounded
higher-is-better scores, transfer recovery is
`(field-mean)/(oracle-mean)`; for errors it is
`(mean-field)/(mean-oracle)`. A non-positive or degenerate denominator is
reported as undefined rather than repaired.

### Endpoints and exact law predictions

The primary successor endpoint is cosine between predicted and actual `Y`.
Also report normalized successor error
`||Yhat-Y|| / ||Y-Ybar_cal||`, where `Ybar_cal` is the outer-fold calibration
mean. The primary completed-law endpoint is obtained by inserting `Yhat` into
the actual full hidden sequence at layer `l+1`, retaining all non-slot states,
and executing layers `l+2..28` followed by the final norm and LM head. For
`L27→L28`, only the final norm and head remain. Let `q` be the true final
next-token law and `qhat` the completed predicted law. Report raw
`KL(q || qhat)` (lower is better), normalized KL skill
`1 - KL(q || qhat)/KL(q || qmean)`, and ordering preservation.

Ordering preservation is the paired Kendall-style agreement within each
carrier: for every anchor word `a`, order the other words `b` by
`KL(q_a || q_b)` and by `KL(qhat_a || qhat_b)`, count a concordant pair as 1,
discordant as 0, and an exact tie as 0.5. The same ordering is computed from
the mean and chart controls. The raw law, KL skill, and ordering score are
all retained; no generated token or generation accuracy is used.

For each outer-fold test cell, retain paired differences against the frozen
best static chart. Higher-is-better differences are field minus chart;
error differences are chart minus field. The dynamics-map support criterion
requires a `>=0.05` lead with a paired 95% lower bound above zero on successor
cosine and on both completed-law readouts (normalized KL skill and ordering)
in at least two of the six layer pairs. Normalized successor error is a
required diagnostic and must not reverse the successor conclusion.

The exact pre-score minimal-class predictions are:

| depth region | predicted minimal transferable class | prediction |
|---|---|---|
| early (`L0→L1`, `L4→L5`) | low-rank affine | low-rank affine is within `0.02` of the best successor and completed-law score; the carrier-shuffled null loses it |
| middle (`L8→L9`, `L12→L13`) | low-rank affine | low-rank affine remains the smallest class within `0.02`, with at least one pair meeting the gated lead |
| late (`L20→L21`, `L27→L28`) | kernel ridge | nonlinear curvature is needed to reach the best score, while transfer weakens at the final block |

These are registered predictions, not observed results. A successful early or
late score does not license a claim about all depths or all language models.

### Bootstrap, support, decision, and kill rules

For each outer-fold result, use a paired two-way cluster bootstrap: resample
the 80 words and the four held-out carriers independently with replacement,
take their Cartesian cell product, and use 2,000 replicates with seed `13007`.
Aggregate the four fold estimates with equal fold weight and report foldwise
and pooled intervals. No cell, layer pair, or bootstrap replicate is treated
as an independent word-level observation.

Support requires finite predictions, finite completed laws, a non-degenerate
actual successor, and a defined law-ordering comparison. Require at least
`95%` of the 320 cells in every held-out block, and report support separately
for every carrier and layer pair. Failure is non-diagnostic; it is not silently
repaired by dropping difficult words or carriers.

Kill the reusable dynamics-map candidate if it fails the `+0.05` gate on
successor or completed-law endpoints in two layer pairs, if the advantage
disappears on held-out carrier blocks, if the carrier-shuffled null matches it
within `0.02`, or if the per-carrier oracle shows that the apparent lead is
only a carrier-specific effect. A successor improvement without completed-law
improvement is coordinate forecasting only and kills the navigation claim.
Void the run for endpoint leakage, post-score candidate or hyperparameter
selection, changed carrier/word membership, absent full-sequence completion
context, missing numerical or revision metadata, absent word/carrier-clustered
bootstrap, or incomplete support. These rules do not claim that LM dynamics
lack lawful transport; they bound this instrument.

### CPU budget

One CPU-only process, one model, fixed recorded thread count, and a hard
20-minute wall-clock cap cover capture, numerical null, ladder fitting,
world-completed passes, and clustered bootstrap. No GPU and no generation are
permitted. If the cap cannot cover all six pairs, the analysis is incomplete
and earns no gated verdict; pair reduction must be decided before scoring,
never after seeing outcomes.

### Tier-3 audit #4 corrections (2026-08-28, fresh unprimed auditor; adopted verbatim)

- **NLM-006b / Round 12:** the frozen-encoder program was not scientifically
  killed; it is **paused / deprioritized**. NLM-006b is non-diagnostic (every
  displaced family failed the 0.80 label-preservation gate; the gate's own
  near-identity controls reached only 0.77, so the kNN proxy is poorly
  calibrated). "No tested native construct competes" is an OOD ranking
  observation, not evidence against native transport. "Chart robustness" is
  withdrawn as a residue claim.
- **NLM-007 may not be scored until the analyzer implements:** a per-word
  carrier-average successor baseline (lexical-persistence moot-maker); kNN
  regression (k = 5, 20) among the static chart controls, member selected by
  inner validation; model/tokenizer revision pinned against the capture
  manifest before scoring; support rules that mark degenerate cells undefined
  (zero denominators, non-finite completed laws) and exclude them rather than
  repairing with 1e-12; completed-law ordering differences with clustered CIs;
  minimal-class reporting separately for successor and completed-law
  endpoints, among ladder members only; a float16-reload check on stored laws
  against fresh float32 laws; and the full clustered gate.
- **Scope:** NLM-007's result, either way, is a single-model interpretability
  result until (i) a class-stratified unseen-word split and (ii) a second
  model family are run. Both are registered as required follow-ups, not
  optional.

### Round 14 amendment — re-lock before scoring (2026-08-27)

**Status: LOCKED AGAIN BEFORE SCORING.** Round 14 is documentation-only. The
repaired analyzer may be inspected, but no NLM-007 score or generation is part
of this round. The following amendments are binding on the first scored run.

#### Required controls and validity checks

- Add the **word-conditioned mean successor** as a separate lexical-persistence
  moot-maker. For each word, average the 12 calibration-carrier successors
  and apply that vector to the held-out carriers. Report it for successor and
  completed-law endpoints, but do not count it as a ladder member for minimal-
  class reporting. A candidate field is not transport evidence if it is within
  `0.02` of this word-mean baseline on successor cosine and both completed-law
  readouts in the pairs used for the claim. To clear the lexical-persistence
  gate, the candidate must instead beat the word-mean baseline by at least
  `0.02`, with a paired clustered 95% lower bound above zero, on all three
  endpoints in at least two pairs. If neither condition is met, the result is
  unresolved against lexical persistence and earns no transport claim.
- Expand the static chart control family to include kNN regression with
  `k={5,20}` alongside the raw 1-NN cosine and Euclidean lookups. Select the
  static-control member by the same inner blocked validation and freeze it
  before held-out scoring. These remain chart controls, not native candidates.
- Verify both model and tokenizer revisions against the capture manifest,
  along with the exact config identity, before opening held-out scores. A
  mismatch or missing pin voids confirmatory interpretation.
- Treat zero denominators, non-finite successors, non-finite completed laws,
  and undefined law-ordering cells as undefined. Exclude them from the
  corresponding endpoint and support denominator, report the undefined-cell
  counts, and never repair them with `1e-12` or another finite placeholder.
- Report paired completed-law ordering differences with word/carrier-clustered
  95% intervals, not only point estimates. Report minimal class separately for
  successor and completed-law endpoints, and select it only among the ladder
  members `mean`, `knn1`, `knn5`, `knn20`, `ridge`, `lowrank`, and `kernel`.
- Reload the stored float16 laws and compare them with the fresh float32 laws
  before scoring: record maximum law/log-law/KL discrepancies and ordering
  agreement. Failure of the declared precision check voids the completed-law
  interpretation.

The class-stratified unseen-word split and a second model family are required
follow-ups. Until both are run, any NLM-007 result is a single-model
interpretability result, not a claim about language-model dynamics generally.

#### Carrier-shuffled null: interpretation by depth

The carrier-shuffled null permutes calibration targets across carriers within a
word, preserving the word's target marginal. Its meaning is conditional on
whether the block action is carrier-dependent at the depth being scored:

- At a **carrier-independent depth**, targets for a word are exchangeable (up
  to the recorded numerical/reload noise). Shuffling cannot break the field;
  `shuffled ~= unshuffled` is therefore a finding about the world —
  context-free transport at that depth — not a kill. In the 16-word smoke,
  `shuffled = 0.955` and the field is `0.955`, so this interpretation is
  explicitly permitted.
- At a **carrier-dependent depth**, shuffling breaks the carrier pairing. A
  shuffled field that matches the unshuffled field within `0.02` is then the
  registered marginal-state/presentation failure and kills the reusable
  carrier-conditioned transport claim for that depth. Report the within-word
  carrier spread and the numerical/reload tolerance used to classify the
  depth; do not infer carrier dependence from the null alone.

“Transfer across carriers” consequently means different things at the two
  depths. When transport is carrier-independent, it means that one
  carrier-invariant law learned from some carriers predicts the same word's
  successor on held-out carriers; it does not mean that the law predicts
  carrier-specific variation. The word-mean gate still has to be cleared, so
  carrier independence can be a real world finding without earning a
  reusable, state-dependent transport claim.

#### CPU-cap contingency

The full plan remains one CPU process, six pairs, 100 shuffles, 2,000 paired
word/carrier bootstrap replicates, and a hard 20-minute wall-clock cap. Before
held-out scores are opened, record the planned budget. If a six-pair run is
projected to exceed the cap, apply a predeclared reduction in this order:

1. reduce layer pairs to one representative per depth region
   (`L0->L1`, `L8->L9`, `L27->L28`);
2. if still necessary, reduce the shuffle count from 100 to 20 and label the
   null diagnostic rather than lock-valid;
3. if still necessary, reduce the bootstrap from 2,000 to 500 and label all
   intervals exploratory rather than lock-valid.

Never reduce the held-out carrier split, word-mean/static controls, completion
endpoint, undefined-cell accounting, or revision/precision checks. Any such
fallback is an incomplete or exploratory run and cannot earn the six-pair
gated verdict; the choice must be fixed before scoring and cannot be made
after seeing outcomes.

#### Exact predictions after the amendment

The pre-score class predictions are unchanged, but every prediction is now
read alongside the word-mean gate and the depth-specific null interpretation:

| depth region | exact minimal-class prediction among ladder members | required reading of the word-mean/null controls |
| --- | --- | --- |
| early (`L0->L1`, `L4->L5`) | low-rank affine is within `0.02` of the best successor and completed-law score | a transport reading requires the `+0.02` word-mean separation gate; a shuffled match is evidence of carrier-independent/context-free transport, not by itself a kill |
| middle (`L8->L9`, `L12->L13`) | low-rank affine remains the smallest class within `0.02`, with at least one pair meeting the existing `+0.05` chart gate | the same word-mean separation gate applies; a carrier-dependent shuffled match within `0.02` kills the reusable-field reading for that pair |
| late (`L20->L21`, `L27->L28`) | kernel ridge is needed to reach the best score, while transfer weakens at the final block | the same word-mean separation gate applies, with the predicted weakening allowed to appear as failure at `L27->L28`, not as a post-score reinterpretation |

These remain falsifiable depth-region predictions. No score from the smoke
artifact is a result: it validates only the 16-word, `L0->L1` pipeline.
