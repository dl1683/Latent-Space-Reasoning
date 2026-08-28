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
and executing layers `l+2..28` followed by the final norm and LM head at the
slot position. For `L27→L28`, this means reading the next-token law at the
substituted slot position itself through the final norm and LM head; it does
not mean reading the law at the sequence's last token when that is a different
position. Let `q` be the true next-token law at that same slot position and
`qhat` the completed predicted law. Report raw
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

## Round 15 — NLM-007 fallback adjudication and corrected late endpoint (2026-08-28)

**Codex, documentation-only adjudication; no experiment was run in Round 15.**

### Fallback verdict

The fallback was declared before scoring as one representative pair per depth
region, with 20 shuffles and 500 bootstrap replicates. It therefore remains an
incomplete run for the full six-pair lock, even though its observed middle pair
is informative.

- **`L0->L1`: lexical persistence.** The word-conditioned mean equals the field
  (pooled successor cosine `0.949` for both), with support `1.0` and shuffled
  null `0.95`. This is a carrier-independent/context-free block action, not a
  state-dependent transport law; it fails the word-mean separation gate.
- **`L8->L9`: successor-only single-pair evidence.** Ridge reaches about
  `0.941` successor cosine versus `0.86` for the best static chart and
  `0.861` for the word mean. The prior completed-law skill and ordering
  numbers are void because they were read at the last token rather than the
  locked slot. The successor lead and shuffled drop remain exploratory; this
  pair cannot be called a completed-law qualifying pair until the corrected
  slot endpoint is rerun. One pair would not meet the lock's two-pair verdict
  in any case.
- **`L27->L28`: void completed endpoint.** The successor lead is not usable for
  the navigation claim because the completed law was read at the last token.
  After layer 27 no remaining layer connects the substituted slot to that
  location when they differ, producing KL `0`, undefined skill, ordering
  `1.0` for every predictor, and support only about `0.42–0.56`. This is a
  design flaw in the endpoint, not evidence about late transport.

The result also corrects two complexity readings. At middle depth the law is
affine but not low-rank at the tested bound: full ridge beats rank-`<=128`
prediction by about `0.05`. The within-carrier oracle is below the cross-carrier
field (64 training words versus 960 cross-carrier cells), so the lead is not
explained by a carrier-specific oracle. The run exceeded the 20-minute cap by
`19%`; this is recorded as a budget failure of the fallback, not hidden.

### Corrected final-block endpoint

Before any late-block score is interpreted, the analyzer must insert the
predicted `L28` slot state into the actual full sequence and read the next-token
law at the substituted slot position itself through the final norm and language
model head. The true comparison law is the true law at that same slot position.
The last-token readout is invalid unless the slot is the last token. The old
`L27->L28` completed endpoint is not repaired retrospectively; it is void.

### Decision and pre-run predictions

**Decision: run the remaining three pairs** `L4->L5`, `L12->L13`, and
`L20->L21` in the next measurement, with the corrected final-block endpoint.
This is the shortest path to the required second qualifying pair; the held-out
carrier split, word-mean baseline, static controls, support rules, revision
pins, reload check, completion endpoint, and clustered statistics remain
unchanged. The estimated cost is about 24 CPU minutes, so this extension is
predeclared with a 30-minute wall-clock budget and one CPU process. It is not a
license to reinterpret a run that exceeds that new budget.

The exact predictions are fixed as follows:

| pair | minimal class prediction | endpoint/gate prediction |
|---|---|---|
| `L4->L5` | low-rank affine remains within `0.02` of the best ladder score | early transport is lexical-persistence dominated: the word mean remains within `0.02` on successor and both completed-law readouts, so this pair fails the transport gate; a shuffled match is expected and is interpreted as carrier independence |
| `L12->L13` | full ridge is minimal; rank-`<=128` low-rank affine misses by more than `0.02` | this is the predicted second qualifying pair: ridge beats the chart by `>=0.05` and the word mean by `>=0.02` on successor and both corrected completed-law readouts, with positive clustered lower bounds and a shuffled field below the unshuffled field by more than `0.02` |
| `L20->L21` | kernel ridge is minimal | successor transfer may beat the chart, but late attenuation is predicted to prevent a complete three-endpoint `+0.05` gate; the corrected endpoint is expected to remain defined and supported, unlike the old `L27->L28` readout |

These predictions are deliberately stronger than the regional class labels and
must be logged before held-out scores are opened. If `L12->L13` does not supply
the second qualifying pair, NLM-007 remains single-pair support and earns no
two-pair dynamics-map verdict.

### What this law means, and what would make it native

Under the guiding question, a denizen of this latent world may be able to learn
a rule for how a state moves at middle depth from some carrier contexts and
reuse that rule on carriers it has not seen. The current evidence is only
successor forecasting: it is a reusable, carrier-transferring full-dimensional
regression field on shared words, not yet a completed-world transport law or a
new geometry. The world supplies a move; the denizen still has to show that its
prediction cashes out at the correct slot. The early result shows why reuse
alone is insufficient: a word's successor can persist without depending on the
incoming state.

This is not yet a native law of latent space. It is one model's law on shared
words and a carrier split. The required follow-ups are now concrete and
registered: (1) a class-stratified unseen-word split, with calibration and
held-out word sets disjoint within each carrier-block fold and the same three
endpoint gates; and (2) the same amended protocol on a second model family,
SmolLM2 or Gemma, with model and tokenizer revisions pinned and the same
carrier-transfer, word-mean, shuffled-null, completion, and support gates.
Only persistence across unseen words and a second family can move this from a
single-model interpretability result toward a native-law claim.

## Tier-3 audit #5 — NLM-007 fallback claim (2026-08-28, fresh Codex auditor)

**Adopted corrections (verbatim where quoted).**

1. **The completed-law endpoint was contract-invalid at every pair.** The lock
   (Round 14, above) reads the completed law *at the slot position*; the
   analyzer read `logits[:, -1, :]` — the last token — for both the true and
   the completed law. At `L8→L9` the number is therefore "a different
   suffix-mediated intervention", not the locked endpoint; at `L27→L28` it is
   void by construction. Repaired in `analyze_lm_dynamics.py`: the primary
   completed-law endpoint is now the next-token law at the substituted slot
   position (true law from the unmodified forward at the same position); the
   last-token readout is kept as a labelled secondary "downstream" endpoint.
   No completed-law number from the fallback run or the Round 15 extension run
   counts toward the lock.
2. **Status wording, adopted verbatim:** Record the result as: "NLM-007 fallback: L8→L9 provides exploratory evidence that a full-dimensional ridge field predicts stored successor states across held-out carrier templates on shared words; the lock-valid completed-law endpoint was not implemented, the fallback lacks the required second pair, and the result is bounded to one model."

Do not call it a native law, a language-model-general law, or a completed-world navigation result. Do not treat L0 as proof of exact context independence. Do not treat L27 as evidence for or against late transport. The next decisive measurement is the corrected slot-position endpoint plus the predeclared second qualifying pair; only after that should the unseen-word and second-family tests decide whether "affine law" is more than a local regression name.
3. **`L0→L1` wording:** "The tested successor endpoint is dominated by
   word-conditioned lexical persistence, with no detected carrier-conditioned
   gain." Not "context-free block action"; not "no law beyond word identity".
4. **Minimal-class ladder:** `word_mean` was counted as a ladder member,
   against Round 14. Repaired: it is a moot-maker only.
5. **Naming:** the defensible name is "a full-dimensional, regularized affine
   predictor wins within this finite ladder on this artifact" — a
   coordinate-dependent regression field, not a discovered affine law of latent
   space. The within-carrier oracle is not a ceiling argument (64 vs 960
   training cells; different model restrictions).
6. **Identity gaps:** `tokenizer_revision` in the manifest is the model
   revision; analysis now asserts model name, config name, and probe count in
   addition to the model revision. The float16 reload check covers one law
   matrix, not the scored successors; a full-precision comparison with a
   predeclared threshold is still owed.

**Strongest alternative explanation (verbatim):** The strongest alternative is not "the shuffle is wrong" or "the ridge leaked the held-out target." It is that the analyzer has found a smooth, implementation-specific conditional regression in one deterministic transformer: the L8 residual state encodes word identity plus carrier/template style, and a high-dimensional regularized field denoises or interpolates that code better than 1-NN and the word mean. The resulting gain can be real, held-out across these four carrier blocks, and still say nothing about a denizen-invented or native affine law. The wrong completed readout makes this alternative especially important, because the observed law gain is a suffix-mediated last-position intervention rather than the preregistered same-slot response.

**Alternative explorations and cheaper baselines (verbatim instructions):**

- Re-run the completed endpoint at the substituted slot position for every scored layer pair, and compare the true and predicted laws at that same position.
- Complete the predeclared `L4→L5`, `L12→L13`, and `L20→L21` pairs before making any two-pair dynamics-map claim.
- Run the class-stratified unseen-word split with word identities disjoint between calibration and held-out cells.
- Repeat the amended protocol on a second model family with independently pinned model and tokenizer revisions.
- Measure append-token and next-position state transport so the move is the model's forward-time transition rather than only a same-slot layer transition.
- Report within-word carrier spread and a predeclared numerical tolerance before interpreting the carrier-shuffled null as evidence of carrier dependence.
- Compare freshly recomputed float32 hidden successors and completed laws against stored float16 artifacts for all probes and all scored pairs, using a threshold fixed before scoring.
- Fit an identity-plus-residual baseline and a block-local residual baseline before calling the winning map an affine law.
- Fit a per-carrier affine correction with the same effective training budget and compare it against the cross-carrier field.
- Fit a fixed PCA-whitened linear map and a low-dimensional ridge field to test whether the apparent full-dimensional advantage is coordinate-scale or rank dependent.
- Equalize the oracle and cross-carrier field training sample sizes and hyperparameter freedoms before using the oracle as a ceiling argument.
- Remove carrier-template lexical/style cues or balance them across folds, then test whether the L8 lead survives.

**Continue verdict:** the next decisive measurement is the corrected
slot-position endpoint plus the predeclared second qualifying pair; only after
that do the unseen-word and second-family tests decide whether "affine law" is
more than a local regression name.

### Addendum to audit #5 — final-pair completion and the post-norm state (Claude, 2026-08-28)

Direct identity test of the repaired completer (probe 0, eight words; replace
the slot with the *stored true* successor and compare the slot law with the
unmodified forward): `L8→L9` KL `4e-7` (exact); `L27→L28` KL `1.32`. Cause:
in this stack the last entry of `output_hidden_states` (hidden index 28) is
the **post-final-norm** state — `head(Z28)` reproduces the true slot law to
`6e-7`, `head(norm(Z28))` does not. So the captured `L27→L28` successor is the
normed state, and the lock-valid completion for the final pair is the LM head
applied to `Yhat` directly at the slot; its last-token readout is undefined by
construction (no layer follows). Implemented; identity now `6e-7` at both
pairs. Consequence for interpretation: the `L27→L28` successor endpoint
predicts a normed vector (scale removed), which makes its cosine not directly
comparable with the other pairs' cosines on raw residual states.

## Round 16 — corrected slot-endpoint rerun and alternative order

**Codex, 2026-08-27. Documentation-only; no experiment was run.** Round 16
adjudicates the two endpoint defects exposed by Tier-3 audit #5 and its
addendum, and predeclares the next measurement. The old completed-law numbers
are not repaired retrospectively: the fallback and extension both read the
law at the sequence's last token, whereas the lock names the substituted slot.
That is a contract failure at every pair. The last-token values remain labelled
secondary downstream diagnostics only. The successor endpoint does not use
that readout, so the extension's successor scores stand as scored, subject to
the extension's reduced exploratory budgets and its incomplete pair coverage.

The addendum resolves the final-pair implementation separately. Stored hidden
index 28 is post-final-norm, so for `L27->L28` the completed law is
`head(Yhat)` at the substituted slot; there is no remaining transformer layer,
and a last-token completion is undefined unless that slot is the last token.
The repaired completer's stored-successor identity tests pass at `L8->L9`
(`KL=4e-7`) and `L27->L28` (`KL=1.32` before the post-norm correction and
`6e-7` after it). The final-pair successor is a normed-vector prediction;
its cosine is therefore not directly comparable to the raw-residual pair
cosines without that qualification.

### Corrected full rerun: preregistration

Run all six fixed pairs — `L0->L1`, `L4->L5`, `L8->L9`, `L12->L13`,
`L20->L21`, and `L27->L28` — with the same 80 words, 16 carriers, four
held-out carrier-block folds, ladder and static controls, revision pins,
support accounting, and float32-versus-stored-float16 checks. For every pair,
the primary completed endpoint is now the next-token law at the substituted
slot. For `L27->L28`, use the post-norm `head(Yhat)` path and report the
last-token endpoint as undefined, not as a secondary score. Do not generate
text.

Use 20 carrier shuffles and 500 word/carrier-clustered bootstrap replicates,
seed `13007`, one CPU process, and a **55-minute hard wall-clock budget**.
The budget is predeclared from the observed approximately 24 minutes per
three-pair extension, or approximately 48 minutes for six pairs, plus a fixed
margin. This is an exploratory corrected rerun relative to the original
100-shuffle/2,000-bootstrap full lock; it can establish corrected endpoint
evidence, but it cannot retroactively earn the original full-budget label.
No pair reduction, control removal, or budget change is permitted after scores
are opened. If the 55-minute budget is exceeded, record the run as incomplete
and make no two-pair dynamics-map claim.

The primary forecast is fixed below. “Slot skill” means normalized KL skill at
the substituted slot, and “ordering” means the paired within-carrier law
ordering at that same slot. The word mean is a separate moot-maker, not a
ladder member.

| pair | slot-endpoint skill prediction | slot ordering prediction | predeclared reading |
| --- | --- | --- | --- |
| `L0->L1` | word mean approximately matches ridge/kernel, within `0.02`; no candidate clears the word-mean separation gate | word mean approximately matches the field; no `+0.05` transport lead | lexical persistence; shuffled field approximately matches unshuffled field |
| `L4->L5` | ridge/kernel modestly exceed the word mean, but at least one corrected skill comparison fails the `+0.05` chart or `+0.02` word-mean gate | ordering lead is below the full gate or has a non-positive clustered lower bound | a successor advantage may be real, but this pair does not qualify as a completed-law transport pair |
| `L8->L9` | full ridge remains clearly above chart and word mean; low-rank remains more than `0.02` behind the best field | ridge clears `+0.05` over chart with positive lower bound and beats word mean by `+0.02` | one qualifying pair, conditional on the corrected slot endpoint |
| `L12->L13` | full ridge is minimal; it clears `+0.05` over chart and `+0.02` over word mean with positive lower bounds | the same three-endpoint gate clears; this is the predicted second qualifying pair | corrected two-pair dynamics evidence if all support and null checks pass |
| `L20->L21` | ridge remains within `0.02` of kernel; corrected skill may beat chart and word mean, but the complete gate fails | ordering lead is below `+0.05` or uncertain across folds | late attenuation; defined endpoint, no full pair qualification predicted |
| `L27->L28` | finite `head(Yhat)` skill is expected, but final-block attenuation prevents a complete gate; ridge is within `0.02` of kernel on the ladder | ordering is finite but not predicted to clear `+0.05`; do not compare it with the invalid last-token ordering | corrected final-pair result is interpretable, but not evidence for or against late transport if the gate fails |

At the slot, the word-conditioned mean may still be strong because the true
slot law is a function of the prefix plus the word. A strong word mean says
that the carrier-averaged lexical/prefix marginal already predicts much of
the response; if it is within `0.02` of a field on all three endpoints, the
result is lexical persistence or a marginal effect, not state-conditioned
transport. Only a field that separates from that word mean, beats the chart,
and survives the shuffled-null and clustered gates supports a reusable
context-conditioned law.

### Alternative explorations: fixed order

1. **Cheap moot-makers on the existing artifact.** First fit an
   identity-plus-residual baseline, `Yhat = X + mean_cal(Y-X)`, on each outer
   fold, and a per-carrier affine diagnostic. For the latter, use each of the
   12 calibration carriers separately, five-way class-stratified word
   cross-fitting with the same seed and ridge regularization selected only
   inside the calibration words; train on 64 words and test on 16, so the
   aggregate uses 12×64 training cells, matching the field's total training
   cell budget. Score both baselines on successor cosine/error and the
   corrected slot skill/ordering. The per-carrier result is a within-carrier
   diagnostic, not a competitor on the held-out carrier block. If either
   baseline closes the ridge lead, the apparent law is a cheap residual or
   carrier-local fit and the native-law wording is withdrawn.

2. **Forward-time transport.** Measure the model's actual next-position move,
   not only same-slot layer transport. For each existing carrier/word cell,
   append one fixed declared one-token sentinel to the carrier-plus-word
   sequence. At each selected layer index `l in {0,4,8,12,20,27}`, let `X` be
   the hidden state at the final word position before the append and `Y` the
   hidden state at the sentinel's next position after the append. Fit the
   same carrier-block-held-out ladder, with the sentinel fixed and included
   in the manifest; compare successor prediction and the next-position law
   after inserting `Yhat` at that position. This is the first follow-up that
   tests the denizen's forward-time move rather than an imposed layer move.

3. **Class-stratified unseen-word split.** Keep the amended slot protocol,
   but make calibration and held-out word identities disjoint within every
   carrier-block fold, class-stratified with the split seed fixed before
   scoring. Retain the word mean only where its calibration words exist and
   report the changed estimand explicitly. This is required before claiming
   lexical generalization.

4. **Second model family.** Only after the unseen-word result is recorded,
   repeat the amended protocol on one independently pinned second family
   (SmolLM2 first if available, otherwise Gemma), including tokenizer/model
   revisions, all six pairs where the architecture permits them, the slot
   endpoint, word mean, shuffled null, support, and clustered gates. A second
   family is required before any claim about language-model dynamics rather
   than one decoder's implementation.

PCA-whitened/low-dimensional fields, oracle equalization, and carrier-style
balancing remain later diagnostics. They do not precede the four steps
above. Until the unseen-word split and second-family replication both pass
their declared controls, the result remains a bounded, one-model
interpretability result.

### Guiding question: what this world manufactures

At `L0`, the word mean is already the field: the world's first block carries a
word-conditioned destination that is largely insensitive to which carrier
presented it. By `L4` and onward, a full-dimensional affine field beats both
that word mean and the static chart, while the carrier-shuffled penalty grows
with depth. The most economical reading is not that context was always hidden
in a fixed state, but that the early blocks manufacture a context-dependent
state: they turn the same lexical identity into different destinations as the
carrier is processed, and later blocks make those distinctions increasingly
necessary for prediction. The pattern is evidence about how this particular
world builds contextual dependence, not yet a native law of latent space.

A denizen trying to navigate would therefore have to invent more than a list
of places or a static chart. It would need an identity test that separates
word persistence from genuine state dependence, a coordinate- or
representation-robust rule for transporting a state conditioned on its
context, and a completion map that predicts what the world says at the moved
state. It would also need to discover when the move is context-free, when it
is affine, and when its regularity breaks — and verify all of that on unseen
words, unseen carriers, forward-time moves, and other realizations of the
world. That is the beginning of a native navigation calculus: not naming a
regressor, but learning which distinctions the world itself makes costly.

## Round 17 — NLM-007 corrected six-pair adjudication and cheap moot-makers

**Codex, documentation-only adjudication; no experiment was run in Round 17.**
The scored artifact is `experiments/results/lm_dyn_v1/analysis_slot.json`,
the corrected slot-position rerun predeclared in Round 16. I checked its raw
pooled endpoint values, per-fold clustered endpoint records, support values,
and metadata directly; the ledger's mechanical labels are not the source of
this verdict. The run took `2145.1 s` of the predeclared `3300 s` budget, used
20 shuffles and 500 bootstrap replicates, and has support `1.0` for every
pair. The law reload check is present (`max_abs_logp_diff=0.0078125`,
`max_abs_pairwise_kl_diff=0.0020809`, ordering agreement `0.9998174`), and the
model/config/revision pins match the capture artifact.

### Gate adjudication

The chart columns below require the `>=0.05` lead on all three endpoints,
with positive clustered lower bounds. The word-mean column requires a
`>=0.02` lead on all three endpoints, also with positive clustered lower
bounds. `Y` means the JSON's raw per-fold records clear the stated threshold;
`N` means at least one required endpoint/fold does not. The shuffle gate is
the selected ridge field's successor-cosine drop; support is the minimum
held-out-block support.

| pair | chart: cosine / slot skill / ordering | word mean: all three | shuffle >0.02 | support >=0.95 | qualifies |
| --- | --- | --- | --- | --- | --- |
| `L0->L1` | N / N / N | N | N | Y | No |
| `L4->L5` | N / Y / Y | Y | Y | Y | No |
| `L8->L9` | Y / Y / Y | Y | Y | Y | Yes |
| `L12->L13` | Y / Y / Y | Y | Y | Y | Yes |
| `L20->L21` | N / Y / Y | Y | Y | Y | No |
| `L27->L28` | Y / Y / Y | Y | Y | Y | Yes |

The `L4->L5` successor-cosine lead is below `0.05` in three outer folds.
`L20->L21` clears that threshold in three folds but has an association fold
at `0.044`, so it does not clear the all-fold successor gate. This is why
their strong corrected slot skills do not qualify them. At `L27->L28`, the
successor cosine is computed on the post-final-norm state; it is a valid
endpoint under the declared `head(Yhat)` completion, but it is not directly
comparable in scale to raw-residual successor cosines.

### Minimal class, word mean excluded from the ladder

The first ladder member within `0.02` of the best ladder score is reported
separately for each endpoint. The ladder is `mean`, `knn1`, `knn5`, `knn20`,
`ridge`, `lowrank`, `kernel`; `word_mean` is a moot-maker and `chart` is a
control, neither a minimal-class member.

| pair | successor endpoint | corrected slot-law endpoint |
| --- | --- | --- |
| `L0->L1` | `knn5` | `lowrank` |
| `L4->L5` | `ridge` | `ridge` |
| `L8->L9` | `ridge` | `ridge` |
| `L12->L13` | `ridge` | `ridge` |
| `L20->L21` | `ridge` | `ridge` |
| `L27->L28` | `ridge` | `ridge` |

These labels do not say that ridge is the world's native law. They say only
which member of this finite, coordinate-dependent ladder is first within the
declared tolerance of the best score. For example, kernel is numerically best
or near-best at several depths, but full ridge is already within `0.02` and
therefore is the minimal class under the registered rule.

### Prediction scorecard and decision

The Round 16 per-pair predictions are held in five cases and falsified in one:

- `L0->L1`: held — word mean matches the field, no separation, and the
  shuffled field matches; this is lexical persistence under the tested
  successor endpoint.
- `L4->L5`: held — the field separates from the word mean and clears the slot
  readouts, but fails the complete chart gate through successor cosine.
- `L8->L9`: held — full ridge is the minimal qualifying field and clears all
  three endpoint gates; low-rank remains more than `0.02` behind the best
  field.
- `L12->L13`: held — full ridge is minimal, low-rank misses by more than
  `0.02`, and all gates clear.
- `L20->L21`: held — ridge is within `0.02` of kernel, the slot readouts are
  strong, but the complete gate fails at successor cosine.
- `L27->L28`: **falsified** — the predicted final-block attenuation did not
  prevent a complete gate; this pair clears every gate at the corrected slot.

Thus NLM-007 meets the numerical two-pair dynamics-map criterion on the
corrected endpoint: in fact, `L8->L9`, `L12->L13`, and `L27->L28` qualify.
This is a corrected, exploratory result at the reduced `20/500` budget, not a
retroactive full-budget confirmatory result under the original `100/2,000`
lock. The exact permitted object-level wording is: **a full-dimensional,
regularized affine predictor wins within this finite ladder** on these three
pairs, against the selected static chart and the word-mean moot-maker, on
shared words and held-out carrier blocks. The corrected slot evidence changes
the result from successor-only evidence with a void completed endpoint to
corrected completed-law evidence, but it does not change the object into a
native affine law, intrinsic geometry, or a language-model-general law.

The final pair is evidence for the stated finite-ladder result despite the
normed-vector qualification. It is not evidence that final-block attenuation
never occurs: the endpoint and parameterization differ from raw-residual
pairs, and the result still awaits the registered unseen-word and second-
family tests.

### What depth manufactures

The word-mean slot skill falls with depth as
`0.95, 0.84, 0.78, 0.70, 0.43, 0.40`, while ridge remains about
`0.92–0.98`. The static chart's slot skill is already only about `0.50` and
`0.51` at `L20->L21` and `L27->L28`, while ridge is about `0.93` and `0.95`.
The economical reading is that the first block preserves a word-conditioned
destination, then early blocks manufacture carrier-dependent distinctions;
later blocks make those distinctions increasingly necessary and a static
chart increasingly uninformative. This is a depth profile of one decoder on
shared words, not a causal proof that context is created from nothing.

Under the guiding question, the denizen therefore needs an identity test that
separates lexical persistence from state dependence, a context-conditioned
transport rule, and a completion map that cashes a predicted move out in the
world's own response law. It must also learn where the rule is context-free,
where an affine field is sufficient, and whether the distinction survives new
words, forward-time moves, and another realization.

### Next measurement: cheap moot-makers (pre-registered)

The next measurement in the Round 16 order is the existing-artifact baseline
run behind `--baselines`; the current smoke test is pipeline validation only
and is not a result. Run all six fixed pairs, with the same 80 words, 16
carriers, four held-out carrier-block folds, corrected slot completion,
support accounting, revision pins, and one CPU process. Use the predeclared
reduced `20` shuffles and `500` clustered bootstrap replicates, with a hard
`55-minute` wall-clock budget inherited from the corrected six-pair run; no
GPU and no generation. The identity-plus-residual predictor is

`Yhat = X + mean_cal(Y - X)`.

The per-carrier affine diagnostic fits a separate ridge field on each of the
12 calibration carriers, with five-way class-stratified word cross-fitting
(`64` training words and `16` test words per carrier), the same seed and
within-calibration regularization selection. It is scored on successor cosine
and error plus corrected slot skill and ordering. It is a within-carrier
diagnostic, not a competitor on the held-out carrier block; report its
carrier-wise values and aggregate summary separately from the cross-carrier
field.

Pre-run predictions for the full six-pair run: identity-plus-residual will be
competitive where the residual is stable, may be competitive at `L0` and
`L4`, and is not expected to close the ridge lead at `L12`, `L20`, or `L27`.
The per-carrier affine diagnostic will be strong within carrier and may
approach ridge, especially at later depths, but will not explain away the
cross-carrier result unless its within-carrier advantage is matched by the
cross-carrier baseline under the same endpoint. These predictions are fixed
for the not-yet-scored full run; the existing one-pair `L8->L9` smoke artifact
is pipeline validation and is not silently folded into the six-pair result.
That smoke nevertheless already points the other way at `L8`: its
identity-plus-residual point scores exceed ridge on successor cosine and slot
skill, and its foldwise endpoint differences close the ridge lead. This is a
withdrawal flag for `L8`, pending the full baseline run's declared endpoint
and budget accounting, not a new corrected-rerun gate result.

The null-making outcome is predeclared: if either baseline closes the corrected
ridge lead on the relevant three endpoints in the full baseline run, the
wording “a full-dimensional, regularized affine predictor wins within this
finite ladder” is withdrawn for the affected pair(s), and no native-law
interpretation survives there. A failure to close the lead does not prove
nativeness; it only leaves the stronger alternative less economical and
advances the next registered tests.

## Tier-3 audit #6 — corrected slot rerun and extension (2026-08-28, fresh Codex auditor)

**Adopted corrections.** (1) `L4→L5` and `L20→L21` are *non-qualifying but
live*; "killed" is rejected. (2) The all-fold `+0.05` rule used in Round 17 is
a **stricter adjudication convention** than the Round 13 aggregate gate and is
labelled as such wherever it is applied. (3) `L27→L28` is reported in its own
post-norm / direct-head endpoint family and never as an ordinary point on the
depth curve. (4) "Ridge wins" means *first ladder member within 0.02 of the
best*; kernel is numerically best on some endpoints. (5) "Context-free" at
`L0` is replaced by "word-conditioned lexical persistence with no detected
carrier-conditioned gain". (6) "The shuffle penalty grows with depth" is a
descriptive statement about this score; no manufactured-context causal
language without within-word carrier spread, a predeclared tolerance, and
template/style controls. (7) The identity test (probe 0, eight words) is
narrow; it must be extended across probes and pairs and stored in the artifact.

**Claim-by-claim verdict (verbatim):**

| claim | verdict | required wording |
|---|---|---|
| Extension L12 ridge .977 vs chart .898 and word mean .888 | CONFIRMED, successor only | Four fold point leads exceed .05; all recorded lower bounds are positive; one lower bound is .0435. This is held-out-carrier coordinate forecasting. |
| Extension L12 completed-law advantage | VOID for the lock | The extension read the last token; no completed-law number from `analysis_ext.json` counts. |
| Corrected slot implementation | CONFIRMED as a code repair | Same-slot true/predicted law is now computed; identity tests are strong but narrow. |
| L8 and L12 corrected qualifying pairs | SUPPORTED, exploratory | Three-endpoint finite-ladder evidence at reduced 20/500 budget; not native-law evidence. |
| L27 corrected qualification | SUPPORTED only within its endpoint family | Direct post-norm head completion is valid, but not comparable to raw-residual pairs and not evidence of remaining transformer transport. |
| Three qualifying pairs prove a reusable dynamics map | TOO STRONG | They support a bounded finite-ladder result, not a general dynamics map. |
| L0 is context-free | OVERSTATED if exact | Use “word-conditioned lexical persistence with no detected carrier-conditioned gain.” |
| Shuffle penalty grows with depth | DESCRIPTIVE/EXPLORATORY | Report the score pattern; do not identify it with manufactured semantic context without spread/style controls. |
| L4 is killed | REJECT | It fails a stringent qualification threshold but has positive corrected slot evidence. Keep it non-qualifying, not killed. |
| L20 is killed | REJECT | Same. Its one .044 successor fold blocks the stricter all-fold convention; this is not a scientific null. |
| Full ridge wins | QUALIFY THE WORD “wins” | Ridge is minimal within .02; kernel is numerically best on some endpoints. |
| Native affine law / intrinsic geometry / LM-general law | NOT SUPPORTED | Block until cheap moot-makers, unseen-word split, forward-time move, and second family are completed. |

**Tunnel-vision audit and strongest alternative (verbatim):**

The current program has corrected its most obvious tunnel—mistaking a
successor forecast and a last-token suffix intervention for a same-slot world
law—but it still concentrates on one family of instruments: adjacent layer
pairs, coordinate regression, carrier-block holdout, and cosine/KL readouts.
The strongest alternative, already named in Tier-3 audit #5, is:

> The residual state encodes word identity plus carrier/template style, and a
> high-dimensional regularized field denoises or interpolates that code better
> than 1-NN and the word mean. The gain can be real and transferable across
> these four carrier blocks while saying nothing about a denizen-invented or
> native affine law.

This alternative is more economical than a native-law explanation until the
cheap tests are run. It also explains why the word mean is strong at L0 and
why its performance declines as the representation becomes more carrier-
specific. It does not require leakage or a broken shuffle.

Other live alternatives that must not be collapsed into one claim are:

1. Identity plus a shared displacement, `Yhat = X + mean_cal(Y-X)`, may close
   the L8 or other ridge lead. The current corrected artifact contains no
   `identres` output.
2. A per-carrier affine field may be strong within carrier but is not a
   held-out-carrier competitor. Its value is diagnostic unless it closes the
   same outer-fold endpoint under an equalized design. The current artifact
   contains no `per_carrier_affine` output.
3. Carrier-template lexical/style cues may drive the apparent dependence.
   Balancing/removing those cues is required before semantic language is used.
4. The actual move may be forward-time append-token transport, not same-slot
   layer transport. This is untested.
5. The result may disappear when word identities are disjoint between
   calibration and held-out sets. This is untested.
6. The result may be specific to one decoder family. A second independently
   pinned family is untested.
7. A PCA-whitened or lower-dimensional field may reproduce the apparent
   advantage, showing coordinate scaling or effective-rank dependence rather
   than a full-dimensional law.
8. The per-carrier oracle comparison cannot decide this: it has a different
   sample size and restriction from the cross-carrier field, so it is not a
   valid ceiling argument.

**Required next actions (verbatim):**

1. Run the already predeclared `--baselines` path on all six existing pairs,
   with no new generation: identity-plus-residual and the per-carrier affine
   diagnostic, same slot endpoint, same support and clustered reporting. The
   analyzer gates these outputs behind `--baselines`
   (`experiments/analyze_lm_dynamics.py:200-208`, `347`, and `459`); their
   absence from `analysis_slot.json` is confirmed. If identity-plus-residual
   closes ridge on a pair's three endpoints, withdraw the finite-ladder ridge
   wording for that pair.
2. Report within-word carrier spread at every depth and a predeclared
   numerical/reload tolerance before interpreting shuffled drops as evidence
   of carrier dependence. Keep 20 shuffles explicitly diagnostic, or expand
   under a new preregistration.
3. Extend the identity test across all probes and all six pairs, and record
   it in the result artifact. Separately compare freshly recomputed float32
   states and laws—not only one law matrix—with stored float16 artifacts under
   a threshold fixed before inspection.
4. Preserve L4 and L20 as non-qualifying but live. Do not call their failure a
   kill unless a new preregistered test specifically tests those hypotheses.
   Clarify whether the all-fold .05 convention is a new robustness rule or
   the intended original aggregate gate.
5. Run forward-time append-token/next-position transport. This is the move a
   denizen of the causal world actually makes and separates a model law from
   an analyst-imposed layer transition.
6. Run the class-stratified unseen-word split with disjoint calibration and
   held-out word identities. Until then, no lexical-generalization claim is
   permitted.
7. Repeat the amended slot protocol on one independently pinned second model
   family. Until then, no claim about language-model dynamics generally is
   permitted.
8. Keep L27 in a separately labelled post-norm/direct-head endpoint family in
   every chart and narrative. Do not use its raw cosine as an ordinary point
   on the same depth effect-size curve.

**Final audit disposition (verbatim):** The corrected run should remain in the ledger as a valuable exploratory
artifact. Its most defensible residue is: lexical persistence at L0; a
carrier-transfer regression advantage over the selected chart and word-mean
controls at several later pairs; and a repaired same-slot completion test that
supports that finite comparison at L8, L12, and the special final pair.

The report must not say that the project discovered a native affine law, proved
that early blocks manufacture semantic context, or established a generally
reusable law of language-model dynamics. The next highest-leverage action is
the existing cheap moot-maker run, followed by spread/style controls and the
forward-time and unseen-word tests. The scientific line is alive, but the
native interpretation is not yet earned.

## Round 18 — moot-maker adjudication and displacement preregistration

**Codex, 2026-08-27. Documentation-only; no experiment was run.** This round
adjudicates the predeclared cheap-moot-maker run from the live JSON and ledger,
then replaces the withdrawn affine interpretation with a displacement test.

### Budget and withdrawal ruling

The six-pair run took `4540.8 s` against the predeclared `3300 s` (`55 min`)
hard wall. The unanticipated per-carrier diagnostic cost was about six minutes
per pair. Under the Round 16 rule, the run is **budget-incomplete**: it earns
no new gate claim, no restored full-budget label, and no new qualification. The
withdrawal rule is deliberately null-making rather than claim-making, so it
remains applicable.

The JSON's pooled ridge-minus-identity-plus-residual differences are:

| pair | successor cosine | slot skill | slot ordering | ruling |
| --- | ---: | ---: | ---: | --- |
| `L0->L1` | `+0.463` | `+0.958` | `+0.376` | identity baseline does not explain lexical field |
| `L4->L5` | `+0.033` | `+0.019` | `+0.022` | small remainder; non-qualifying, live |
| `L8->L9` | `-0.008` | `-0.021` | `-0.020` | identity-plus-residual closes ridge |
| `L12->L13` | `-0.007` | `-0.009` | `-0.013` | identity-plus-residual closes ridge |
| `L20->L21` | `+0.018` | `+0.034` | `+0.032` | small remainder; non-qualifying, live |
| `L27->L28` | `+0.195` | `+5.173` | `+0.172` | invalid raw-state/post-norm comparison |

The per-carrier affine diagnostic is far below the cross-carrier field across
the reported endpoints, so it does not supply the explanation and is retained
only as a within-carrier diagnostic. The predeclared condition is met at
`L8->L9` and `L12->L13`, exactly the two pairs that carried the Round 17
two-pair criterion. The wording **“a full-dimensional, regularized affine
predictor wins within this finite ladder”** is withdrawn for those pairs, as is
any native-law interpretation attached to them.

The two-pair criterion therefore **does not survive as a claim**. `L4->L5` and
`L20->L21` retain point-estimate remainders of roughly `0.02–0.03` across the
three endpoints, but they fail the complete successor/chart convention and
cannot replace the two withdrawn pairs; they remain non-qualifying, live
observations rather than kills. `L27->L28` remains a separately labelled
post-norm/direct-head family, but its identity result is not interpretable:
`identres` compares a raw `X` with a normed target `Y`, so it is not a meaningful
persistence baseline there. NLM-007 now establishes a bounded null result:
the middle-depth ridge advantage at `L8` and `L12` is explained by persistence
plus a shared displacement under this finite, shared-word design; it does not
establish a state-dependent affine law, a native law, or a reusable dynamics
map.

### Guiding question: the move and the place

For a denizen of this world, the middle-depth move is best written as

`T(X) = X + Δbar_cal + ε(X,w,c)`,

where `Δbar_cal` is the shared calibration displacement and `ε` is the part
that still depends on the incoming state, word, or carrier. The result says
that, at `L8->L9` and `L12->L13`, the measured move is mostly persistence: the
state carries its own identity forward and the block adds a common shift. The
transport content is therefore not the absolute destination `Y`, but the
displacement `Δ = Y-X`; only a predictor of `Δ` beyond its shared or lexical
mean can be evidence for state-dependent motion.

The early blocks and the final block do not obey the same simple reading. The
early blocks leave a small but nonzero residual dependence after the shared
shift is removed, while the final block changes coordinate family through
post-normalization and a direct head. A denizen's law of motion must therefore
be typed by endpoint and representation: persistence, displacement, and
completion are separate operations, and a law learned in raw residual
coordinates cannot silently be applied to a normed state.

“Same place” cannot mean equal stored coordinates. It means observational
identity: two states are the same place when the declared probes and downstream
response law cannot distinguish them within a fixed, declared tolerance. When a
move mostly persists, that identity relation is stable under the move's shared
displacement; the displacement is still a move if it changes the world's
downstream response. The denizen must discover both the invariant carried along
the trajectory and the residual that changes consequences, rather than treating
small Euclidean motion or coordinate equality as primitive.

### Next measurement: displacement ladder (pre-registered)

**Choice and justification.** Choose the displacement ladder before the
forward-time move because the identity null has now shown that the central
middle-depth question is whether any state-dependent motion remains after
persistence and a shared shift are removed. Forward-time append-token transport
is still the world's more literal move, but measuring it first would confound a
genuine temporal law with the unresolved decomposition of the already observed
layer transition.

Use the five raw-residual pairs `L0->L1`, `L4->L5`, `L8->L9`, `L12->L13`, and
`L20->L21`. Keep `L27->L28` in the experiment register but exclude it from the
primary displacement ladder: its `Y` is post-norm while `X` is raw, so no
raw-residual `Y-X` comparison is valid. A compatible pre-norm capture would
need a separate preregistration; the direct-head endpoint is not silently
pooled with the other five.

For each eligible pair, predict `Δ=Y-X` from `X` on the existing four
held-out-carrier folds, then reconstruct `Yhat=X+Δhat` and score the corrected
law at the substituted slot. The ladder is the constant mean displacement
`Δbar_cal` (zero-order baseline), kNN `k={1,5,20}`, full ridge, rank-`<=128`
low-rank affine, and kernel ridge. The word-conditioned mean displacement
`E_cal[Δ|w]` is a separate lexical moot-maker, not a ladder member; the static
chart is reported only as a reconstructed-Y reference, not as a displacement
law.

A candidate is a state-dependent displacement result only if it beats the
word-conditioned displacement mean by at least `0.02`, with a positive paired
word/carrier-clustered 95% lower bound, on displacement cosine and on both
corrected slot-law endpoints (skill and ordering), with finite cells and
support `>=0.95`. Report the carrier-shuffled field and within-word carrier
spread at every depth with a fixed numerical/reload tolerance; a shuffle drop
over `0.02` is diagnostic, not sufficient by itself. A candidate that beats
the constant mean but not the word-conditioned mean is lexical or marginal
persistence, not state-dependent transport; no native-law wording follows
from a pass.

Use the same 80 words, 16 carriers, four folds, revision/config pins, slot
completion, seed `13007`, 20 carrier shuffles, 500 clustered bootstrap
replicates, one CPU process, no generation, and no GPU. Set a **95-minute hard
wall**: the observed baseline run was `75.7 min` for six pairs, including the
previously unpriced per-carrier diagnostic at about `36 min` total; the new
budget adds approximately twenty percent margin. If the wall is exceeded,
record the run as incomplete and draw no displacement gate claim.

Predictions are fixed as follows: `L0` remains lexical persistence, with the
word-conditioned displacement mean near the field; `L4` and `L20` retain small
`0.02–0.03`-scale residuals on the three endpoints but do not form a complete
two-pair transport result; `L8` and `L12` show no qualifying state-dependent
displacement beyond the word-conditioned mean; and `L27` remains outside the
raw-residual family. The per-carrier affine arm remains a diagnostic and is
not expected to explain a held-out-carrier result.

### Think-before-you-run residue

Standing rule text for `AGENTS.md` (Claude to place):

> Before fitting or interpreting a transition law on a residual stream, write
> down and run the identity-plus-shared-displacement null on the same held-out
> splits, endpoints, support accounting, and clustered gates. If the target is
> a difference, promote `Δ=Y-X` and its mean displacement to the primary
> decomposition before adding model capacity; no state-dependent transport
> claim is admissible until it beats that null and the lexical moot-maker.

This was the obvious missing control in Rounds 13–17. Its absence allowed a
predictor that carried `X` through unchanged to be narrated as a learned affine
law; the standing rule blocks that category error at design time.

## Tier-3 audit #7 — the identity-baseline withdrawal (2026-08-28, fresh Codex auditor)

**Adopted corrections.** (1) The closure rule "pooled ridge − identres ≤ 0.02 on
all three comparison metrics" was chosen by Claude after seeing the scores; it
is a **conservative post-hoc one-sided null-making policy**, not a preregistered
equivalence test, and is labelled so wherever the withdrawal is cited. (2) The
clustered intervals support "no demonstrated positive ridge advantage under
this margin", not "no lead" or "equivalence". (3) "Persistence plus a shared
displacement" is replaced by "consistent with identity plus a calibration-mean
displacement under this shared-word, held-out-carrier design; whether the
displacement is carrier-, state-, or word-dependent is unresolved". (4) The
three comparisons are successor cosine, slot skill, slot ordering — only the
latter two are completed-law slot metrics. (5) "Exact" completion → "routing
validated to measured precision (per-pair max KL 1.9e-6 to 6.2e-6 over 16 × 80
cells)"; no per-carrier error profile or fresh-float32 comparison was stored.
(6) `L4→L5` and `L20→L21` retain small live point-estimate remainders; not
killed, not promoted. (7) `L27→L28` identres is not a persistence test.

**Claim-by-claim verdict (verbatim):**

| Claim | Verdict | Required wording |
|---|---|---|
| (a) Identity-plus-residual closes ridge at `L8→L9` and `L12→L13` | Supported as a conservative point-estimate withdrawal; overclaimed as “no lead” or equivalence | “On shared words and held-out carrier blocks, identity-plus-shared-displacement is at least as good as full ridge within a post-hoc one-sided 0.02 pooled margin on the three recorded comparison metrics at `L8→L9` and `L12→L13`; the finite-ladder ridge wording is withdrawn as a conservative policy.” |
| (a) “All three slot endpoints” | Slightly imprecise | The three comparisons are successor cosine, slot skill, and slot ordering. Only slot skill and slot ordering are completed-law slot metrics; successor cosine is a separate endpoint. |
| (a) “Persistence plus a shared displacement” | Descriptive null, not established law | “The measured relation is consistent with identity plus a calibration-mean displacement under this shared-word, held-out-carrier design. The experiment does not determine whether displacement is carrier-, state-, or word-dependent.” |
| (b) Does not close at `L0`, `L4`, `L20`, `L27` | Numerically true under the post-hoc rule, but interpretation differs by pair | “Identity-plus-shared-displacement does not meet the chosen pooled one-sided margin at `L0`, `L4`, `L20`, or `L27`; `L4` and `L20` remain non-qualifying but live, while `L27` is not a valid raw-residual persistence comparison.” |
| (c) Slot completion is exact | Routing is strongly confirmed; “exact” is too strong | “Stored-true-successor substitution reproduces the unmodified slot law with per-pair maximum KL approximately `1.9e-6` to `6.2e-6` over the checked carriers and words, validating completion routing to measured precision.” |
| (c) “At every carrier” | The maximum covers every checked cell, but per-carrier distributions are not stored | Report the per-pair maximum over `16 × 80` cells. Do not imply that a carrier-wise error distribution or fresh float32 successor comparison was recorded. |
| (d) Run overran budget and is budget-incomplete | Confirmed | “The baseline run took `4540.8 s` against the predeclared `3300 s` budget and remains a budget-incomplete exploratory artifact. Its measured values are retained, but it does not earn the planned full-budget gate.” |

**Ordering judgment (verbatim):** Use this order:

1. displacement ladder first, to adjudicate whether anything beyond identity plus average displacement remains;
2. forward-time transport second, to test whether same-slot layer motion is the right notion of motion at all;
3. unseen-word and style controls before semantic or lexical-generalization language;
4. second-family replication before any general language-model claim.

The durable residue is:

> Identity is the null for residual-stream transport. The present data support persistence plus a calibration-average displacement as a competitive finite-design description at `L8` and `L12`, retain small unresolved remainders at `L4` and `L20`, and do not yet establish a native or generally reusable affine law.

## Round 19 — displacement adjudication and forward-time transport contract

**Codex, documentation-only; no experiment run in this round.** Round 18 is
adjudicated directly from `experiments/results/lm_dyn_v1/analysis_delta.json`;
the ledger entries `nlm007_delta_predeclared` and `nlm007_delta_v1` are checked
against that JSON.

### Round 18 prediction scorecard

The run finished in `1750.3 s` of the recorded `5700 s` wall, with support
`1.0` for every eligible pair and fold. The ledger's mechanical reading is
correct: only `L20->L21` passes the registered three-endpoint gate. In delta
mode those endpoints are displacement cosine, slot skill, and slot ordering;
slot skill is relative to the mean-displacement completion and is not on the
successor-mode skill scale.

| prediction | adjudication | reading |
| --- | --- | --- |
| `L0->L1` remains lexical persistence | held | Word-conditioned displacement equals the field (`0.948`); shuffle changes nothing. |
| `L4->L5` retains a small residual without a complete result | held | Kernel has a small cosine lead and tiny ordering change; it is live and non-qualifying, not killed. |
| `L8/L12` show no qualifying displacement beyond the word mean | mixed | The strict three-endpoint gate holds, but displacement cosine beats the word mean by about `0.07–0.22`, with clustered lower bounds above `0.05` in every fold; shuffled fields are `0.35–0.52` versus fields `0.60–0.85`. Kernel is minimal. Ordering leads are only `0.003–0.022` and slot-skill lower bounds are mixed. |
| `L20->L21` retains a small residual without a complete result | falsified | Kernel clears all three: about `+0.025–0.051` cosine, `+0.13–0.32` slot skill, and `+0.023–0.038` ordering, all with positive clustered lower bounds. |
| `L27->L28` remains outside the raw-residual family | held | Its target is post-norm while `X` is raw. |

`L20->L21` is one bounded qualifying pair. It establishes, in this one-model,
shared-word, held-out-carrier design, a kernel-class predictor of `Delta=Y-X`
that beats the word-conditioned displacement mean on the measured coordinates
and downstream slot-law readout. “Kernel minimal” is a finite-ladder label,
not proof that the underlying world law is nonlinear. The preferred name is
**state-dependent displacement beyond the word-conditioned mean, with a kernel
as the minimal tested predictor**. Do not promote “state-dependent nonlinear
displacement” as an established law; “nonlinear” is only a model-class
description here. No general dynamics-map, native-law, unseen-word, or
second-family claim follows.

### Motion versus consequence under the guiding question

The middle-depth result is not only a readout artifact: displacement cosine is
a direct target and the state-conditioned field separates from the
word-conditioned mean on held-out carriers. But “large state-dependent
displacement” is not yet “large consequence.” The same-slot next-token law is
nearly saturated by the identity component, so a changed displacement can
leave that law nearly unchanged. Slot skill is already the more sensitive
residual readout because it compares against the mean-displacement completion,
but its mixed middle-depth intervals prevent treating it as a universal
consequence certificate.

A denizen needs the distinction between motion and consequential motion, but
consequential motion should not be added as a sixth ontological primitive. It
is a typed derived predicate on an admissible move, relative to a declared
downstream response law and tolerance: the move is consequential when that law
distinguishes its endpoint from its start. “Same place” therefore means
observational equivalence under the denizen's probes and response law, not
equal stored coordinates. The sentence “motion everywhere from L4,
consequential motion only late” remains conditional: it may describe the
world, or it may expose a same-slot readout insensitive to displacement
direction. Forward-time transport is the discriminator.

### Next measurement: forward-time append-token transport

This is the next measurement in the fixed order. It tests the move a causal
world actually makes, rather than another same-slot layer transition. It is
CPU-only and documentation-only here; no capture or scoring is authorized in
Round 19.

**Inputs and sentinel.** Reuse the frozen `lexical_probe_v1` 80 one-token
items and 16 carrier templates. For each complete token sequence
`S=(t_0,...,t_{m-1})`, append the one-token ASCII period `.` as primary
sentinel `s_A`; freeze and record its tokenizer ID in the capture manifest.
Run ASCII comma `,` as `s_B` at the same appended position as the token
identity control, also recording its ID. Both must be exactly one non-special
token under the pinned tokenizer. If not, invalidate the run rather than
substituting a token after inspection.

**States and endpoint.** Let `q=m-1` and `r=m`. At each selected layer
`l in {0,4,8,12,20}`, define `X=h_l(S)[:,q,:]` and
`Y_s=h_l(S||s)[:,r,:]`. Thus `X` is the final original-position state
before append, while `Y` is the sentinel's next-position state. Predict
`Delta_s=Y_s-X` on held-out carrier blocks. Store `h_l(S||s)[:,q,:]` and its
law as the causal locality check: appending after `q` must not alter `q`
beyond tolerance.

Fit and score both sentinels at identical positions. Apply the period-trained
predictor to the comma target without refitting as the negative token-identity
control. The position control is the original terminal position `q` in the
same appended run: compare appended versus unappended state and law there,
and complete there only as a negative endpoint. The primary endpoint is the
next-token law at `r`, where the predicted sentinel state is inserted and the
completed law is read.

**Residual-rule ladder.** Use the same four carrier-block folds, 80 words,
support accounting, and calibration-only inner selection. For each sentinel
and layer fit: identity `Yhat=X`; shared mean displacement
`Yhat=X+mean_cal(Delta)`; word-conditioned mean displacement
`Yhat=X+mean_cal(Delta|w)`; kNN `k={1,5,20}`; ridge; rank-`<=128` low-rank
affine; and kernel ridge. Identity and shared mean are the required nulls;
the word-conditioned mean is a separate lexical/marginal null. Shuffle `Y`
within each word across calibration carriers as the diagnostic carrier null.

For each `Yhat`, insert it at `r` in the appended sequence at layer `l`, run
the remaining layers, and read the law at `r`. Score displacement cosine,
law skill relative to the shared-mean-displacement completion, law ordering,
and raw KL. Store fresh-float32/reload checks, finite cells, per-carrier
support, within-word carrier spread, and both controls.

**Gates, budget, and predictions.** A candidate passes a layer only if it
beats the word-conditioned displacement mean by `0.02`, with a positive
paired word/carrier-clustered 95% lower bound on all three primary endpoints,
finite cells, support `>=0.95`, and passing reload/locality controls. A
forward-time result requires this at two of the five selected layers for the
same sentinel; the comma arm is a control/replication, not a post-hoc token
selection. The position-control difference must be at most predeclared
`1e-4` in absolute float32 units (or the measured batched-vs-single numerical
floor if larger), or the primary endpoint is void. A pass earns no native or
cross-model claim.

Capture both sentinels in about two CPU minutes. Analysis is about 30 CPU
minutes per five layer points at 20 shuffles and 500 clustered bootstrap
replicates, one process, no generation, and no GPU. Predict early movement to
be token-identity/shared-mean dominated, with middle/late sentinel-position
laws more sensitive to the L8/L12 direction than the original-slot law and a
kernel field the leading middle-depth candidate. The cautious alternative is
that token identity or position dominates and no layer passes, favoring a
readout explanation. Do not fold in the unseen-word split: it is the next
separate generalization/style follow-up with disjoint calibration and held-out
word identities and its own artifact and gates.

### Round 20 ruling: forward locality tolerance

**Ruling, before any forward score is opened:** the absolute `1e-4` state
tolerance was mis-scaled for residual coordinates whose magnitude at `q` can
reach about `378`. The primary endpoint is not void on this control. Exact
causal masking remains the mathematical invariant; the observed state
difference is float32 kernel-path variation with sequence length.

For this run, the corrected position control is, for every captured layer and
sentinel, both:

`max_i |h_l(S||s)[q,i] - h_l(S)[q,i]| <= max(1e-6 * M_q, epsilon_state_floor)`,
where `M_q = max_i |h_l(S)[q,i]|`, and
`max_j |log p_j(S||s,q) - log p_j(S,q)| <= max(1e-4, epsilon_loglaw_floor)`.

The floors are the measured batched-vs-single numerical floors for the same
appended sequence at `q`. This relative-state plus absolute-log-law clause is
the corrected predeclared intent and applies to the current run because it was
settled before any forward score was opened; future preregistrations must state
it explicitly. Here `3.624e-4 <= max(3.78e-4, 1.22e-4)` and
`6.58e-5 <= max(1e-4, 2.96e-5)`, so both locality controls pass. The forward
endpoint remains eligible subject to its other gates. Interpret the control as
evidence against causal nonlocality beyond numerical variation, not as exact
float32 equality or as evidence for any broader dynamics claim.

## Tier-3 audit #8 — displacement claims and forward-time implementation (2026-08-28, fresh Codex auditor)

**Adopted corrections.** (1) Displacement wording: "a kernel predictor captures
held-out-carrier displacement variation beyond the word-conditioned
displacement mean on the measured residual coordinates; whether that variation
is state-, carrier-, template-, or word-dependent remains unresolved." "Kernel
minimal" = minimal among the tested finite ladder under the registered metric
and tolerance, not intrinsic nonlinearity. (2) The carrier shuffle (permute Y
across calibration carriers within word) is a **carrier-alignment diagnostic,
not a state-independence null**; it also destroys a carrier/template-style
explanation. The shuffled field is reported for ridge and low-rank only —
"kernel beats its shuffled null" is not established. (3) "The slot law barely
registers it" is a fact about the declared same-slot readout, not about the
world. (4) `L20→L21` = "one bounded qualifying pair under the registered
displacement-and-slot-law gate" — no "learned nonlinear law". (5)
"Consequential motion" is a derived predicate, relative to a declared response
law and tolerance. (6) Forward-time implementation verified: X = h_l(S)[q]
(unappended), Y = h_l(S‖s)[r]; insertion at hidden index l at r (hook on layer
l−1; embedding row for l = 0); law read at r; identity/shared-mean nulls;
no-refit token-identity control; calibration-only standardization and
selection. Missing check (now run, see ledger `nlm007_forward_AB_equality`):
A/B `H_q_unappended` equality. (7) Round 20 locality pass is narrow (margin
~1.8e-5 against a global-max-magnitude bound); wording: "no detectable causal
nonlocality beyond measured numerical/kernel-path variation under this run's
corrected tolerance." (8) Float16 storage bounds small displacement/ordering
effects; not precision-independent.

**Strongest alternative explanation (verbatim):** The strongest alternative is a carrier/template-conditioned nuisance law encoded in the residual state.

Under this explanation:

- the word-conditioned mean captures lexical identity;
- `X` carries presentation/style context;
- kernel learns a nonlinear carrier- or template-conditioned correction;
- the carrier shuffle collapses because it destroys the carrier pairing;
- the same-slot law ignores much of that correction at middle depth;
- L20 passes because the late stack’s readout is more sensitive to the same nuisance direction.

This explanation fits all current displacement observations without requiring a reusable state-space law.

**Alternative explorations and cheaper baselines (verbatim):**

The highest-value controls are:

- A fixed-input style-balance control: match or residualize carrier/template features before fitting displacement.
- A within-style or within-template null that preserves carrier style while removing state pairing.
- A style-held-out split, not only a carrier-held-out split.
- A per-word, per-style mean displacement baseline.
- A low-dimensional block/template-only predictor to test whether style variables explain the kernel lead.
- A direct \(Y-X\) decomposition into word mean, carrier mean, shared mean, and residual.
- A target permutation preserving carrier-level marginal structure rather than destroying all carrier alignment.
- Explicit A/B equality checks for `H_q_unappended`.
- Per-layer/per-sentinel locality and precision reports using fresh float32 values.
- Unseen-word evaluation before any lexical or semantic generalization language.
- Replication on a second model family before any general language-model claim.

**Final status (verbatim):** The displacement result should be retained as:

> Held-out-carrier evidence for predictable displacement variation beyond a word-conditioned mean, with a kernel as the minimal tested predictor; carrier/template versus state dependence remains unresolved.

The L20 result should be retained as:

> One bounded qualifying pair under the registered displacement-and-slot-law gate.

The forward result, before its scores are inspected, is procedurally eligible provided all other gates pass. Its most important interpretive test is whether sentinel-position law sensitivity survives the token-identity control and remains after style and precision concerns are addressed.

## Round 20 — forward-time adjudication, endpoint ruling, and next control (2026-08-28)

**Codex, documentation-only; no experiment run.** The forward JSON artifacts
and the four requested ledger entries were checked directly. The Round 20
locality ruling is applied before interpreting the scores: the corrected
scale-aware state and log-law tolerances pass, so the endpoint is eligible.
This means only that there is no detectable causal nonlocality beyond measured
numerical/kernel-path variation under that tolerance; it is not exact float32
locality. The A/B unappended-state audit is stronger: `H_q_unappended`, its
law, and `H_slot` are bit-identical across captures (all recorded maxima are
`0.0`).

### Round 19 preregistration ruling

The locked rule requires at least two of the five layers for the **same
sentinel**. The comma arm is a control/replication, not a post-hoc token
selection. The primary period arm therefore **does not meet** the
forward-time result: `analysis_fwdA.json` has only `F20` as a mechanical
qualifying layer, with ridge. `F0` is token-identity/lexical dominated;
`F4`, `F8`, and `F12` have large displacement-cosine and law-skill leads but
fail the complete gate because ordering lower bounds are not positive in the
required folds. Support is `1.0` throughout.

The comma arm has a secondary replication result: `analysis_fwdB.json` has
two qualifying layers, `F12` and `F20`, both with ridge. `F4` passes its three
endpoints in the point estimate but misses a skill lower bound in one fold;
`F8` misses one ordering lower bound by `-0.002`; `F0` remains
token-identity/lexical dominated. This does not rescue the period arm and does
not change the same-sentinel rule. The overall preregistered forward-time
claim is therefore **not met**, not indeterminate.

| Round 19 prediction | Adjudication | Exact reading |
| --- | --- | --- |
| Early layers are token-identity/shared-mean dominated | Held | `F0` is the lexical/token-identity regime in both sentinels. |
| Sentinel-position laws are more sensitive to the middle/late direction than the original-slot law | Partly held | Cosine and law-skill leads are large from `F4` onward, but the full three-endpoint gate is not met at every such layer. |
| Kernel is the leading middle-depth candidate | Partly held / mixed | Kernel is the minimal tested class at several middle-depth endpoints, but ridge is minimal at other endpoints and the tendency is not a native-class result. |
| No layer passes because token identity or position dominates | Falsified in the strong form | `F20` passes for `.`, and `F12/F20` pass for `,`; locality passes under the corrected tolerance. Token identity remains a real component, not the whole explanation. |

The allowed claim is narrow. In this one-model, shared-word,
held-out-carrier design, the forward sentinel displacement is predictable from
the preceding state beyond the word-conditioned mean from `F4` onward, and the
sentinel-position response law registers that variation in the cosine and
skill readouts. The no-refit token-identity control transfers partially across
sentinels (about `0.43–0.54` versus about `0.26–0.30` for the shared mean), so
the signal is not explained by sentinel identity alone. The period arm earns
one bounded qualifying layer; the comma arm earns two-layer replication under
the same mechanical gate. No state-independent-of-presentation claim,
unseen-word claim, second-family claim, native-law claim, or general dynamics
claim follows. Per Audit #8, “state-dependent” must remain qualified because
`X` carries carrier/template presentation context; the strongest live
alternative is a carrier/template-conditioned nuisance law encoded in the
residual state.

### The ordering endpoint

The per-anchor concordance of within-carrier KL orderings across words is now
ruled **insensitive/saturated for this question**, as a measurement diagnosis,
not as a retroactive gate change. The same pattern recurs in the layer
displacement ladder and both forward sentinels: cosine and law skill can be
large while the ordering lead stays near zero or within roughly `±0.02`.
Word identity and inherited law order dominate the across-word ranking, and
every predictor preserves much of that order. The historical ordering gate
therefore remains binding for the claims it governed; no failed layer is
passed after the fact.

For future runs, replace that endpoint with a fixed **KL-to-truth rank among
candidate predictors**. At each held-out response-law cell, rank the fixed
candidate set `{identity, shared mean, word mean, kNN-1, kNN-5, kNN-20,
ridge, low-rank, kernel, chart}` by finite `KL(q_truth || q_hat)`, lower is
better, with midranks for ties. Convert rank `r` among `K` candidates to
`R = 1 - (r-1)/(K-1)`. Compare the preselected candidate's `R` against the
word-conditioned mean on the same cells; the future endpoint requires a point
lead of at least `0.02` and a positive word/carrier-clustered 95% lower bound,
with support, finite-cell, reload, and locality gates unchanged. Candidate
selection remains calibration-only. This is a future endpoint contract and
does not reclassify any existing result.

### Guiding-question ruling: motion, consequence, and nativeness

The forward-time experiment identifies the most natural measured move so far:
the world advances from the final original-position state to the appended
sentinel's next-position state. Its response registers the move at the
sentinel position, not merely at an analyst-chosen layer transition. In that
limited sense, this is the first serious candidate for a native law of motion.
It is not yet a native law. A denizen would need the regularity to survive the
Audit #8 style controls, a class-stratified unseen-word split, and a second
model family, with the same null ladder and response-law endpoint. It must
also remain predictive after carrier/template presentation is balanced or
held out. The exciting but bounded “so what” is: the latent world appears to
have a forward step whose consequences can be forecast, but we have not yet
shown that the forecast belongs to the world rather than to its presentation.

### Next measurement, fixed in order

The next measurement is the cheapest Audit #8 style control: a
**within-style-family target null** on the existing raw forward captures
`forward_states_A.npz` and `forward_states_B.npz`, with their existing
`analysis_fwdA.json` and `analysis_fwdB.json` as the comparison artifacts.
Because each individual template occurs once, the declared style family is
the four-probe config block (`gloss`, `continuation`, `association`, or
`grammar`), not an invented within-template replicate.

For each outer held-out-block fold, sentinel, and selected layer, preserve the
80 words, all original folds, calibration-only standardization and model
selection, and the existing 20 shuffles/500 clustered bootstrap structure.
On calibration carriers only, permute `Y` (equivalently `Delta=Y-X`) across
carriers **within each style-family block and word**. This preserves the
style-family marginal while removing the exact state/carrier pairing that the
current all-carrier shuffle destroys. Refit the same selected ridge and
kernel fields to each null and evaluate on the unchanged held-out carriers.
Retain displacement cosine and law skill as primary diagnostics; use the new
KL-rank endpoint in place of across-word ordering for the future consequence
gate. A style-robust result must still beat the word-conditioned mean by
`0.02` with positive clustered lower bounds on cosine, skill, and KL-rank, and
must additionally beat the within-style-family null by `0.02` with a positive
clustered lower bound on those three endpoints. Support remains `>=0.95`, all
cells finite, and reload/locality controls must pass.

Predictions are fixed as follows. If carrier/template style explains the
lead, the within-style-family null will retain much of the original lead and
the style-balanced state-conditioned advantage will collapse toward the
word-conditioned mean; no two-layer native-law reading is earned. If a
state-linked component remains beyond style, the style-preserving null will
collapse toward the lexical mean while the original field retains positive
cosine, skill, and KL-rank leads, possibly attenuated. Either outcome is a
control result, not a retroactive pass of the period arm.

After this control, run the predeclared class-stratified unseen-word split:
calibration and held-out word identities must be disjoint, with the same
sentinel pair, folds, null ladder, clustered gates, and response-law endpoint.
Only after that split is adjudicated is a second model-family replication
next. No experiment is run in Round 20 itself.

## Tier-3 audit #9 — Round 20 adjudication, KL-rank endpoint, within-style null (2026-08-28, fresh Codex auditor)

**Adopted corrections.** (1) Round 20's "not met" is a **nonpass under the
historical contract, not a kill of forward transport**; required wording:
"The period sentinel did not meet the preregistered two-layer, three-endpoint
qualification criterion: only F20 qualified. … In the shared-word,
held-out-carrier design, sentinel displacement is predictably improved over
the word-conditioned mean from F4 onward, and the response law registers that
variation in cosine and skill. The ordering endpoint was later diagnosed as
insensitive/saturated, so the qualification failure is not a substantive null
result." The comma arm falsifies "token identity or position prevents any
qualifying layer". (2) **KL-rank implementation defect:** the ranked candidate
set omitted kNN-1/5/20 (K = 7, not the preregistered 10). Repaired in the
analyzer (kNN candidates now completed and ranked); the style-A run and the
in-flight style-B run used K = 7 and are labelled so; not contract-valid for
the KL-rank endpoint. Future reports carry raw KL, skill, pairwise win rate,
KL-rank under the exact candidate set, and candidate-set sensitivity.
(3) **The within-style-family null is an alignment-destruction diagnostic,
not a clean style null**; "beats the within-style null by 0.02" is not
informative evidence for a state-linked component in the win direction. The
style-A run's "style-robust" reading is withdrawn as a claim; it stands only
as the diagnostic it is. (4) The cheapest fair control on the existing
captures: within each style block hold out one carrier, fit on the other
three, compare the state-conditioned predictor against a leave-one-carrier-out
per-word/per-block mean displacement baseline with clustered inference —
diagnostic of within-family state information, not cross-family transfer; the
whole-block outer hold-out remains the cross-family test. (5) Priority order
(verbatim below).

**Round 20 ruling audit (verbatim):**

“Not met” is mechanically correct, but it must not be phrased as a kill.

The registered rule requires at least two qualifying layers for the same sentinel, with positive clustered lower bounds on cosine, skill, and ordering.

- Period (`.`): only F20 qualifies.
- Comma (`,`): F12 and F20 qualify.
- Both have complete support.

Thus the period arm did not meet the preregistered criterion. The comma arm cannot retroactively rescue it because it was explicitly a control/replication arm.

However, the ordering endpoint was simultaneously diagnosed as insensitive/saturated. F4–F12 show large cosine and skill leads, while ordering lower bounds hover around zero or become negative. That makes the failure endpoint-limited, not evidence that forward transport is absent.

The strongest defensible wording is:

> The period sentinel did not meet the preregistered two-layer, three-endpoint qualification criterion: only F20 qualified. This is a nonpass under the historical contract, not a kill of forward transport. In the shared-word, held-out-carrier design, sentinel displacement is predictably improved over the word-conditioned mean from F4 onward, and the response law registers that variation in cosine and skill. The ordering endpoint was later diagnosed as insensitive/saturated, so the qualification failure is not a substantive null result.

The comma arm additionally shows that the strong alternative “token identity or position prevents any qualifying layer” is false: F12 and F20 qualify there. It still does not establish a native law, state independence from presentation, unseen-word generalization, or second-family replication.

Sources: [Round 20 contract and adjudication](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/EXPERIMENTS.md:2330), [period artifact](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/lm_dyn_v1/analysis_fwdA.json), [comma artifact](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/lm_dyn_v1/analysis_fwdB.json).

**Within-style null audit (verbatim):**

The concern is valid. The current null is an alignment-destruction diagnostic, not a clean style null.

The analyzer permutes `Y` or `Δ` across the four carriers within each block and word, then refits ridge/kernel. With four carriers there are only `4! = 24` arrangements per word-block. The scored null predictor uses one seeded permutation; the 20-shuffle result is reported separately for cosine.

This breaks the exact `X ↔ Y` carrier pairing. A flexible predictor then sees one carrier’s state paired with another carrier’s displacement. It is entirely expected that it predicts the wrong carrier and falls below even the shared mean.

That is exactly what the F8 smoke shows:

- shared mean: approximately 0.47;
- ridge: approximately 0.72;
- kernel: approximately 0.74;
- ridge style-null: approximately 0.30;
- kernel style-null: approximately 0.21.

The null’s low skill and low KL-rank are consequences of the broken target alignment. They do not show that style is absent. Thus:

> “Beats the within-style null by 0.02” is not informative evidence for a state-linked component in the WIN direction unless the null has first been shown to preserve the relevant nuisance structure and to pass for a known truth.

The existing outer fold already holds out an entire four-carrier block, so direct memorization of that block label is restricted. But this does not eliminate:

- style features shared across blocks;
- template or carrier-specific nuisance;
- residual style encoded in `X`;
- carrier-conditioned displacement that the permutation deliberately destroys.

The cheapest useful control on the existing captures is:

1. Within each style block, hold out one carrier.
2. Fit on the other three carriers.
3. Compare the state-conditioned predictor against a leave-one-carrier-out per-word/per-block mean displacement baseline.
4. Use the same clustered inference.

This tests whether state information predicts a carrier beyond the style-family mean. It is diagnostic, not proof of cross-family generalization.

The current whole-block outer holdout should remain the cross-family transfer test. A valid per-block mean cannot be computed for the already-held-out block from zero target examples, so it should not be smuggled into that outer-fold comparison.

A cleaner but more expensive control is cross-fitted residualization of predeclared template/block/style features from `X`, with corresponding nuisance adjustment of `Δ`. A within-block-only fit is cheap but can still exploit carrier identity, so it cannot stand alone.

The smoke is explicitly marked “not a result”; it should not support a style-robust claim.

Source: [style-null code](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/analyze_lm_dynamics.py:408), [style-null preregistration](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/EXPERIMENTS.md:2422), [F8 smoke](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/lm_dyn_v1/analysis_stylesmoke.json).

**Priority ordering (verbatim):**

My priority ordering is:

1. Fix the KL-rank candidate-set mismatch.
2. Replace the current style-null gate with a within-family leave-one-carrier-out analysis against a per-word/per-block mean.
3. Run cross-fitted style residualization or a genuinely style-preserving conditional permutation.
4. Run the predeclared unseen-word split.
5. Only then consider second-family replication.

Bottom line: the data support a bounded held-out-carrier forward-displacement forecasting result. They do not yet distinguish a state-space regularity from a carrier/template-conditioned nuisance law.

## Round 21 — LOCO within-family control pre-registration (2026-08-28)

**Codex, documentation-only; no experiment is run in this round.** This round
adjudicates the two Round 20 within-style-family runs and locks the next
measurement requested by Tier-3 audit #9. The live JSONs and ledger readings
are confirmed: style A (`.`) lists `F4`, `F8`, and `F20` as mechanical passes;
style B (`,`) lists `F8`, `F12`, and `F20`. Both have complete support. In both
arms the within-style-family target null falls below the shared/word mean from
`F4` on while ridge and kernel remain high.
Across the two JSONs, the pooled null cosine is approximately `0.16–0.54`,
versus `0.45–0.66` for the shared/word means and `0.68–0.82` for ridge/kernel
over `F4–F20`; support is `1.0` throughout.

### Adjudication of the Round 20 style runs

The observed pattern is mechanically the **state-linked branch** of the Round
20 prediction: the null collapses and the original field retains positive
cosine, skill, and KL-rank separation. The style-A mechanical `style_robust`
label and the analogous style-B label are nevertheless withdrawn as claims.
Audit #9 establishes why: the null permutes `Y` (or `Delta`) across carriers
within block and word, so it pairs one carrier's state with another carrier's
displacement. A flexible field is then expected to predict the wrong carrier.
Its collapse below the shared mean is an alignment-destruction diagnostic,
not evidence that style has been removed. The registered prediction is
therefore scored as “state-linked branch occurred mechanically, but the
branch is uninformative for the causal interpretation.” The runs establish
only that the original held-out-carrier displacement signal survives comparison
with this broken-pairing diagnostic; they do not establish style robustness or
a state-linked component.

The KL-rank endpoint has a separate contract defect. The style-A and style-B
JSONs rank `K=7` candidates because `knn-1`, `knn-5`, and `knn-20` were omitted;
the analyzer repair in commit `269e46c` restores the preregistered `K=10`
candidate universe prospectively. These existing runs are therefore labelled
`K=7` and are not valid for a Round 20 KL-rank gate. This does not erase their
mechanical descriptive values, but it blocks any contract-valid style claim.
The historical period result remains a nonpass, not a kill: only `F20`
qualified for `.`, while `F12` and `F20` qualified for `,`; the comma arm does
not retroactively rescue the same-sentinel period rule.

### LOCO design ruling

The implemented `--loco` control is the fair cheapest test Audit #9 asked for,
with a deliberately narrower question. Within each of the four style blocks,
it holds out one carrier, fits on the other three (240 cells), selects ridge
lambda by inner leave-one-carrier-out over those three, and evaluates the
held-out carrier's 80 words. It compares identity, the shared mean of the
three training carriers, the per-word/per-block mean of those carriers
(`blockword_mean`), and ridge. It reports displacement cosine, response-law
skill relative to the shared mean, and KL-rank among exactly four candidates,
then computes ridge minus `blockword_mean` with word-clustered bootstrap per
held-out carrier and a pooled carrier-by-word bootstrap over the 16 held-out
carriers. This is a valid diagnostic of within-family state information
conditional on the observed words and family; it is not a test of cross-family
transfer. The existing whole-block outer holdout remains the cross-family
test.

The control is fair for that conditional question, but not a clean causal
style-separation experiment. The block-word baseline is allowed to look up
the held-out carrier's word identities; that is intentional protection
against calling lexical persistence “state,” but means the result cannot
generalize to unseen words or a denizen that does not possess the word key.
The ridge sees only 240 training cells and can still exploit carrier/template
identity or style encoded in `X`; LOCO tests prediction to another carrier in
the same family, not removal of those nuisance coordinates. Standardization
is correctly fit only on the three training carriers (and separately on the
inner two during lambda selection), so it does not leak held-out states, but
the three-carrier inner selection is noisy. The pooled bootstrap treats the
16 carrier rows as exchangeable; because they are nested in only four style
blocks, its interval is secondary to the per-carrier word-clustered records
and cannot by itself prove family-independent state information. These are
scope limitations, not reasons to reject the control.

### Locked LOCO run

Run both sentinels (`.` and `,`) at `F0`, `F4`, `F8`, `F12`, and `F20`, using
the existing raw forward captures and unchanged 80-word held-out sets. For
each layer there are 16 outer held-out-carrier fits, each trained on 240
cells, with inner three-carrier leave-one-out lambda selection; each held-out
carrier receives four 80-word predictor completions. Use `500` word-clustered
bootstrap replicates for every per-carrier contrast and the pooled carrier-
by-word bootstrap. The expected runtime is about 60 minutes by scaling the
roughly 37-minute, five-layer style runs from 40 to 64 completion batches per
layer and accounting for the extra ridge fits. Set a hard CPU wall of
`75 minutes`; an overrun is budget-incomplete and earns no gate claim. No GPU,
generation, new words, or second model family is included.

A held-out carrier passes an endpoint if ridge minus `blockword_mean` has a
point estimate at least `0.02` and a positive word-clustered 95% lower bound.
A layer is a LOCO pass only if the pooled contrast meets that rule on all
three endpoints—displacement cosine, law skill, and four-candidate KL-rank—
and at least `8/16` held-out carriers pass all three endpoint checks. The
run-level within-family diagnostic is positive only if at least `2/5` layers
pass for each sentinel. This is a breadth rule for this diagnostic, not a new
native-law claim. Preserve support `>=0.95`, finite cell accounting,
calibration-only selection, float reload, and forward locality checks; any
failed validity gate blocks the corresponding layer. Report raw KL, skill,
cosine, carrier-level contrasts, the `K=4` candidate universe, and both
bootstrap forms. Do not compare this four-candidate rank numerically to the
invalid `K=7` style ranks or the prospective `K=10` endpoint.

Predictions are locked before scoring, for both sentinels:

| Layer | State-linked explanation | Carrier/template-nuisance explanation |
| --- | --- | --- |
| `F0` | No LOCO pass; ridge and block-word mean are near the lexical/token-identity regime. | Same: no conditional state advantage is expected. |
| `F4` | Positive ridge-minus-block-word-mean contrasts on all three endpoints; expected to pass the breadth rule. | The block-word mean closes the apparent advantage; no layer pass. |
| `F8` | Positive contrasts and a pass expected. | Ridge advantage collapses toward zero or below `0.02`; no pass. |
| `F12` | Positive contrasts and a pass expected. | Block-word mean remains competitive; no pass. |
| `F20` | Positive contrasts and a pass expected, possibly with the largest law consequence. | Any remaining ridge lead is carrier/template nuisance encoded in `X`; no pass. |

These predictions concern within-family conditional information only. A LOCO
pass would say that the state contains predictive variation beyond the family
mean for already-seen words; it would not say that the variation transfers to
a held-out style family, survives style residualization, generalizes to new
words, or is native to the latent world. A LOCO failure would leave both
explanations viable if the predictor is underpowered, while a pass would
narrow—but not eliminate—the carrier/template alternative. The next controls
remain cross-fitted style residualization or a genuinely style-preserving
conditional permutation, then the disjoint class-stratified unseen-word
split, then a second model family.

### Guiding-question interpretation

If the forward step is predictable within a style family beyond that family's
mean but not across families, a denizen would have to treat style as part of
the operational state of its world: the same lexical content presented in
different styles would not be the same navigational place if it changes the
lawful successor. But this would be a local, stratified state coordinate, not
evidence for a universal state variable. The denizen would need a family-
conditioned map and an identity test that asks “same content and same style
sheet?” before applying a move law.

That outcome would not by itself reveal a defect in the world. It could expose
a defect in our quotient—our notion of “same place” may have erased a
presentation coordinate that the world's dynamics preserve—or it could show
that the world genuinely has separate sheets whose laws do not transfer. The
mathematical task is therefore to choose observational equivalence from
predictive consequences: merge two states only when the declared moves and
response laws remain interchangeable. LOCO tests that question inside one
sheet; residualization, unseen words, and the whole-block holdout decide
whether the sheets are intrinsic structure or an artifact of our chart.

## Tier-3 audit #10 — LOCO A, unseen-word branch, second lens (2026-08-28, fresh Codex auditor)

**Adopted corrections.** (1) LOCO A wording: "On already-seen words, within a
style family, X predicts a held-out carrier's displacement and response-law
consequence better than the three-carrier per-word family mean at F4–F20" —
not a presentation-independent state or a native law. (2) The LOCO baseline
is variance-disadvantaged (three-carrier mean vs a 240-cell regularized
ridge); before interpretation, compare ridge against the strongest X-free
lexical baseline: a word-only ridge (word one-hot, inner-selected) and a
shrunk word mean (shrinkage selected inside calibration carriers), keeping
the unshrunk block-word mean as the historical baseline. (3) LOCO does not
distinguish latent state from a smooth carrier/style code; "presentation may
itself be an operational state coordinate if changing presentation changes
the lawful successor" — the required control is cross-fitted residualization
of predeclared presentation coordinates from both X and Δ. (4) The pooled
LOCO bootstrap (16 carriers nested in 4 blocks treated as exchangeable) is
secondary; block-first resampling required for any cross-family statement.
(5) F0: "no detected conditional gain at F0", not "no F0 state dependence".
(6) Hazards: `per_carrier_affine` uses Z directly (wrong in forward mode —
now guarded); in delta mode `successor_cos`/`oracle_ceiling_succ_cos` are
displacement cosine (naming); in forward mode the "last" readout equals the
slot readout (not independent). (7) Unseen-word branch: the lexical null
disappears with the word-mean — add a class-mean displacement null and a
word-only (frozen input-embedding) predictor as the primary X-free lexical
baselines; fix the KL-rank universe for unseen mode; add fail-fast asserts
(disjoint ids, every class in every fold, nonzero counts); block-first,
class-preserving pooled bootstrap; raw KL and skill primary.

**Recommended unseen-word gate and predictions (verbatim):**

### Recommended unseen-word gate

For each sentinel and layer:

- Compare ridge against the strongest X-free lexical baseline: class mean or word-only embedding predictor.
- Require point lead `>= 0.02` and a positive word/block-clustered 95% lower bound on displacement cosine, response-law skill, and fixed-universe KL-rank.
- Require the pooled contrast to survive a block-first, class-preserving word bootstrap.
- Require all eight fold keys to be valid; preferably require at least 6/8 keys to have positive endpoint contrasts, with no systematic collapse in one held-out block.
- Preserve the existing support threshold and forward locality/reload gates.
- Report raw KL and skill regardless of rank outcome.
- Do not call the result lexical generalization unless it beats the lexical baseline, not merely the global mean.

### Predictions to predeclare

State-linked explanation:

- F0 remains near the lexical/token-identity regime.
- F4–F20 retain positive but possibly attenuated ridge leads over class and word-only lexical baselines.
- Response-law skill survives better than raw ordering.
- The effect may weaken substantially because unseen words remove direct word-conditioned lookup.

Lexical/interpolation explanation:

- Ridge approaches the class/word-only baseline on unseen words.
- X-based kNN may remain competitive because it interpolates lexical identity through the residual representation.
- Any apparent gain over the global mean but not over the word-only lexical field is not a state-law result.

Carrier/style-nuisance explanation:

- Performance may survive unseen words if the field mainly extrapolates a smooth carrier/style code.
- Therefore unseen words alone do not resolve state versus presentation; residualization remains necessary.

A failure should be interpreted as failed extrapolation by this field, not proof that no structured law exists.

**Second lens — hostile structural properties (verbatim):**

The Second Lens in [AGENTS.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/AGENTS.md:15) asks which properties make structured reasoning difficult and what the next latent space must change.

| Candidate hole | Status | Skeptical reading |
|---|---|---|
| Motion invisible to the response law at middle depth | Partly proven, but readout-specific | Middle-depth displacement is directly predictable, while the original ordering endpoint barely changes. The sentinel-position law registers motion in cosine and skill. What is proven is that one response law—especially ordering—is insensitive, not that the world cannot register the motion. |
| Identity-dominated transitions | Proven locally | F0 and L0 are dominated by lexical/token identity and the word-conditioned mean. This is a real property of the measured input transition, not a theorem about all latent transitions. |
| Presentation entangled with state | Strong unresolved concern, not proven | `X` contains carrier/template information, and LOCO preserves that information. Whole-block holdout limits trivial block memorization but does not remove smooth style coordinates or carrier-conditioned dynamics. |
| Laws holding only within template families | Not proven | Whole-block holdout shows some transfer across blocks on shared words. LOCO shows within-family prediction. Neither establishes transfer to unseen words, unseen families, or another model family. |
| Ordering-saturated readouts | Proven for this endpoint | The same ordering failure recurs across layer displacement and both sentinel arms while cosine and skill lead. This diagnoses an inherited, word-dominated endpoint—not a general saturation theorem about the latent world. |

The most serious current hole is therefore not “there is no motion.” It is that the representation and response law do not yet provide a stable quotient separating:

- lexical content;
- presentation/style;
- operational state;
- consequential motion.

A next-generation latent space should:

- define “same place” by interchangeability of declared moves and response laws;
- expose or test presentation coordinates rather than silently mixing them into state;
- provide consequence-sensitive readouts based on raw predictive divergence, not only inherited orderings;
- support multi-step closure, not only one-step prediction;
- generalize across unseen lexical identities, style families, and model families;
- make precision and support part of the representation’s contract.

If changing style changes the lawful successor, style may legitimately be part of operational state. The defect may instead be in our quotient: we may have incorrectly declared differently presented states to be the same place.

**Tunnel-vision assessment (verbatim):**

The current program is at risk of tunnel vision in several ways:

- One decoder family, one model revision, one 80-word probe set, and four highly structured blocks dominate the evidence.
- Repeated measurements reuse the same residual coordinates and closely related response-law endpoints.
- Regression success is being treated as evidence about latent-world structure, although it may only show implementation-specific compressibility.
- “State versus presentation” may be a false dichotomy; the experiment has not yet intervened on presentation while holding operational state fixed.
- The ordering endpoint was diagnosed only after repeatedly failing to support the desired interpretation, creating a risk of metric migration without an independent consequence calibration.
- No result yet demonstrates useful navigation, multi-step composition, semantic generalization, or a denizen-level primitive.

The strongest antidote is not another regressor on the same cells. It is orthogonalization: unseen words, style residualization, a fixed lexical baseline, a hierarchical inference rule, multi-step response consequences, and a second model family.

## Round 22 — LOCO adjudication, unseen-word lock, and second lens (2026-08-28)

**Codex, documentation-only; no experiment is run.** The requested JSONs,
ledger entries, analyzer path, and Round 21 contract were checked directly.
The forward-mode oracle values in the historical artifacts remain excluded:
`nlm007_oracle_defect_forward` records that those diagnostics predicted `X`
from `X` and are not evidence or a gate.

### LOCO A/B adjudication

The Round 21 mechanical rule is satisfied by both sentinel runs, but with an
important asymmetry:

| Sentinel | `F0` | `F4` | `F8` | `F12` | `F20` | run-level prediction |
| --- | --- | --- | --- | --- | --- | --- |
| `.` (A) | fail | pass, 12/16 | pass, 11/16 | pass, 15/16 | pass, 15/16 | positive; 4/5 layers |
| `,` (B) | fail | fail | fail | pass, 13/16 | pass, 12/16 | positive; 2/5 layers |

All folds have support `1.0`; A took `2902.1 s` and B `3090.8 s`, both below
the `4500 s` wall. The pooled A contrasts over `F4–F20` are approximately
`+0.09–+0.13` cosine, `+0.23–+0.31` law skill, and `+0.29–+0.40` KL-rank,
with 11–15 of 16 carriers passing all three. B has positive cosine at `F4`
and `F8`, but their skill or KL-rank lower bounds miss; its qualifying layers
have approximately `+0.07–+0.10`, `+0.15–+0.20`, and `+0.20–+0.26` on the
three pooled contrasts, with 12–13 of 16 carriers passing all three.

Audit #10's established wording is therefore the maximum current claim:

> On already-seen words, within a style family, `X` predicts a held-out
> carrier's displacement and response-law consequence better than the
> three-carrier per-word family mean at `F4–F20`.

This is a bounded within-family diagnostic. It is not evidence for a
presentation-independent state variable, a native law, unseen-word
generalization, or general dynamics. The result is also not yet interpretable
as state information because the three-carrier block-word mean is
variance-disadvantaged relative to a 240-cell regularized ridge, and `X` may
encode a smooth carrier/style coordinate.

### Round 21 prediction score

The state-linked prediction is **partially held mechanically**: its `F0`
failure is held in both arms; A passes all four predicted later layers, while B
passes only `F12/F20`. Both runs are positive under the two-of-five diagnostic
rule, but the sentinel asymmetry weakens any claim of a universal layer profile.
The carrier/template-nuisance prediction of “no layer pass” is **not held
mechanically**: the block-word mean did not close ridge in either run. That is
not evidence against the nuisance explanation, because the historical baseline
is the unfair comparator and LOCO retains carrier/style information in `X`.
The correct score is therefore “mechanically contradicted, scientifically
unresolved.”

The equalized X-free lexical baselines demanded by Audit #10 **must be run as
a LOCO addendum before interpreting the LOCO gap as conditional state
information**. The addendum will preserve the same 16 held-out-carrier folds,
five layers, both sentinels, delta target, three endpoints, support and reload
checks, and 500 word-clustered bootstrap replicates. It will add:

1. **word-only ridge:** one-hot word features only, with its regularization
   selected by inner leave-one-carrier-out calibration within the three
   training carriers; and
2. **shrunk word mean:** the per-word calibration mean shrunk toward the
   calibration shared mean, with shrinkage selected inside calibration only.

Report both against the historical unshrunk block-word mean, and gate the
state-conditioned field against the stronger of the two equalized X-free
baselines for each endpoint while retaining raw KL, skill, cosine, carrier
contrasts, and block-first diagnostics. Budget: one CPU process per sentinel,
`500` bootstrap replicates, with a `75-minute` hard wall per sentinel (`150`
minutes total); an overrun is budget-incomplete and earns no interpretation
gate. The unseen-word split does **not** supersede this addendum: it removes
word-conditioned lookup for its own claim, but cannot repair the seen-word
LOCO comparison or adjudicate its unfair baseline.

### Unseen-word run: locked predeclaration

The next run is predeclared with the already implemented analyzer path:

- `--unseen-words 2`, both sentinels, and `F0/F4/F8/F12/F20`;
- eight block-by-word-fold keys (`gloss`, `continuation`, `association`,
  `grammar` crossed with `w0/w1`), with calibration and held-out word IDs
  disjoint and every lexical class present in both sides;
- `20` shuffles, `500` bootstrap replicates, seed `13007`, support/finite,
  reload, and forward-locality gates unchanged;
- primary X-free lexical nulls `class_mean` and `wordonly_knn`, where the
  latter is frozen input-embedding cosine kNN with `k=5` over calibration
  words, averaged over their calibration targets;
- fixed `K=11` KL-rank universe: `{identity, mean, class_mean, wordonly_knn,
  knn1, knn5, knn20, ridge, lowrank, kernel, chart}`; and
- fail-fast disjoint-ID, class-coverage, and nonzero-count assertions plus
  block-first, class-preserving pooled bootstrap (blocks, then carriers,
  then words).

The smoke is pipeline validation only: it took `634.8 s` for one sentinel,
`F8`, one shuffle, and ten bootstrap replicates; it is not evidence. The
scale budget is approximately `6350 s` (`106 minutes`) for ten
layer-sentinel slices at the smoke's base rate, so reserve a `150-minute`
hard CPU wall for the full two-sentinel run. No post-hoc budget reduction
earns a gate claim. The gate is adopted from Audit #10 with one operational
clarification: report both nulls, and compare ridge with the stronger
X-free baseline per fold/endpoint. Require a point lead of at least `0.02`
and a positive word/block-clustered 95% lower bound on displacement cosine,
law skill, and fixed-universe KL-rank; the block-first pooled contrast must
also be positive, at least `6/8` fold keys must have positive endpoint
contrasts, no held-out block may systematically collapse, and all validity
gates must pass. A failure is failed extrapolation by this field, not proof
that no structured law exists.

Predictions are fixed by layer and explanation:

| Layer | State-linked | Lexical interpolation | Carrier/style nuisance |
| --- | --- | --- | --- |
| `F0` | no gate pass; lexical/token-identity regime | ridge closes to class/word-only nulls; no pass | no conditional gain expected |
| `F4` | positive but attenuated lead; pass remains expected | closes to the strongest X-free null; no pass | smooth style code may preserve a positive lead |
| `F8` | positive lead and law skill survive; pass expected | X-based field closes to null, though residual kNN may compete | lead may survive across unseen words; unresolved without residualization |
| `F12` | positive lead; pass expected, possibly stronger than F8 | no lead over the strongest X-free null | persistent style-coded lead remains possible |
| `F20` | positive lead; pass expected and potentially most stable | closes to null; no pass | persistent late style-coded lead remains possible |

These are predictions, not results. A pass would show that the field is not
merely a word-conditioned lookup, but would still not separate state from
presentation. The single measurement that would most sharpen the second lens
is a cross-fitted presentation intervention/residualization that holds lexical
identity and operational task fixed while changing or removing predeclared
style coordinates from both `X` and `Delta`, then tests the same held-out
response-law consequences. The unseen-word run is the immediate lexical gate;
this style-controlled measurement is the sharper discriminator of the central
state-versus-presentation hole.

### Second lens: current answer

The latent world is hostile to structured reasoning when a denizen cannot tell
which differences matter for lawful navigation. The current evidence supports
two local holes: the input transition is identity-dominated at `F0` (lexical
content overwhelms a reusable move), and the across-word ordering readout is
saturated for this endpoint (it fails to register middle-depth motion that
cosine and law skill do register). The latter is a hole in the response
instrument, not proof that the world itself cannot register motion.

Two stronger concerns remain unproven. Presentation is entangled with
operational state: whole-block transfer rules out only trivial held-block
memorization, while LOCO shows seen-word carrier prediction that can still be
a smooth style code. Laws restricted to template families are also not proven:
whole-block transfer is evidence against the strongest family-only story, but
unseen lexical identities and a second model family remain untested. Thus the
serious current hole is a missing predictive quotient separating lexical
content, presentation, operational state, and consequential motion. We may
have declared differently presented states to be the same place even when
their lawful successors differ.

The next-generation latent space must make “same place” observational and
predictive: two states are equivalent only when declared moves and downstream
response laws remain interchangeable. It must expose or factor presentation
coordinates, use consequence-sensitive divergence rather than inherited
orderings alone, support multi-step closure, and generalize across unseen
lexical identities, style families, and model families. If style changes the
lawful successor, style may belong to operational state; the defect would then
be our quotient, not necessarily the world. Precision, support, and the
validity of the response law must be part of the representation's contract.

No new axiom is warranted in Round 22. This is a sharper empirical boundary
and a locked measurement plan, not a demonstrated invariant of latent space.

## Tier-3 audit #11 — LOCO B and the equalized addendum (2026-08-29, fresh Codex auditor)

**Adopted corrections.** (1) LOCO B: F12/F20 pass; F4 misses skill and
KL-rank; **F8 misses skill only** (its KL-rank LB is +0.021); F0 fails; B is
weaker in breadth (2/5 vs 4/5) — a sentinel-specific instrument result, not
evidence that B carries less state information. (2) **Implementation defect
in the equalized addendum:** the inner leave-one-carrier-out selection
centred the word-only ridge and the shrunk word mean on the outer
three-carrier shared mean, which includes the validation carrier's targets —
a direct pressure toward maximal shrinkage. Outer margins are not leaked, but
"the data selected maximal shrinkage" is invalid as implemented; the
addendum must be rerun with the inner two-carrier centre. (3) The
`strongest_equalized` comparator was chosen on held-out outcomes
(conservative for ridge but without nominal coverage); the corrected version
selects the baseline inside calibration, freezes it, and evaluates once.
(4) KL-rank universes differ (B: K = 4; equalized A: K = 6) — breadth
comparable, KL-rank effect sizes not. (5) Wording: "the word-conditioned
component captured by these tested estimators is negligible for the measured
forward displacement in this design" — not "no per-word lexical signal"; the
positive object is **X-conditioned residual predictability**, not
state-conditioned structure; "the forward law is about context rather than
content", "the state-conditioned component is large", and "audit #10's
variance objection is answered" are withdrawn as over-claims. (6) Fair
competitors once word means carry nothing: fixed shared/class mean; a properly
nested word-only estimator; frozen-input-embedding lexical interpolator; a
predictor using predeclared style/template coordinates only; a hierarchical
carrier/style random-effect model.

**Equalized-A audit (verbatim):**

The artifact reports:

| Layer | Cosine margin | Skill margin | KL-rank margin | Carriers passing all 3 |
|---|---:|---:|---:|---:|
| F0 | −0.039 [−0.141, +0.040] | −0.709 [−2.502, +0.228] | −0.012 [−0.220, +0.179] | 3/16 |
| F4 | +0.127 [+0.111, +0.144] | +0.301 [+0.167, +0.433] | +0.276 [+0.140, +0.389] | 13/16 |
| F8 | +0.120 [+0.104, +0.135] | +0.228 [+0.129, +0.314] | +0.264 [+0.163, +0.353] | 11/16 |
| F12 | +0.098 [+0.086, +0.112] | +0.267 [+0.192, +0.352] | +0.328 [+0.252, +0.397] | 13/16 |
| F20 | +0.087 [+0.076, +0.099] | +0.242 [+0.164, +0.316] | +0.341 [+0.261, +0.413] | 14/16 |

So the mechanical claim is accurate. The A run is stronger than B under the stated rule.

But all 80 folds select:

- `lam_wordonly = 100.0`, the largest value in the grid;
- `alpha_shrunk = 1.0`, the exact shared-mean endpoint.

The critical implementation problem is in [`loco_control`](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/analyze_lm_dynamics.py:349):

```python
Y3 = Yc_.reshape(len(tr), n, D)
shared = Yc_.mean(0)
```

Inside the inner leave-one-carrier-out loop, `Yi3` contains only two carriers, but both equalized helpers continue using the outer three-carrier `shared`:

```python
wordonly_ridge(lam, Yi3)
shrunk_wordmean(al, Yi3)
```

The functions therefore center inner-validation predictions on a mean that includes the validation carrier’s targets. At maximal shrinkage, the inner predictor moves toward:

\[
m_3 = \frac{y_{\mathrm{validation}}+y_1+y_2}{3},
\]

which is explicitly pulled toward the validation target. This creates a direct pressure toward maximal shrinkage.

This does not constitute outer-held-out-carrier target leakage: the final outer predictor still uses only the three outer-training carriers. Therefore the recorded outer margins are not automatically meaningless. But it invalidates the claimed inner calibration procedure and the conclusion that the data independently selected maximal shrinkage. The persistence of maximal shrinkage after using the proper two-carrier center is unknown.

The proper inner calculation would use the two-carrier calibration mean separately for each inner fold. No corrected run was performed here.

**Under- and over-claims (verbatim):**

Under-claimed:

- If the equalized result survives corrected calibration, it would support a meaningful statement that residual `X` prediction exceeds tested X-free lexical prediction.
- The consistent collapse of word-conditioned means across the seen-word runs is a real design-level pattern.
- The B/A contrast is useful as a sentinel-sensitivity diagnostic.

Over-claimed:

- “No per-word lexical signal.”
- “The variance objection is answered.”
- “The state-conditioned component is large.”
- “The forward law is about context rather than content.”
- Any presentation-independent or native-law claim.
- Any claim that F0 has no state dependence; the correct wording remains “no detected conditional gain at F0.”

The current positive object should be called `X-conditioned residual predictability`, not state-conditioned structure.

**Strongest alternative explanation (verbatim):**

The strongest alternative is:

> `X` contains a smooth presentation/template coordinate that varies systematically across carriers within each family, and the forward displacement and response-law consequence also vary along that coordinate. Ridge learns this carrier/style geometry. The equalized lexical baselines collapse because word identity is not the source of the variation, but that does not make the variation operational state.

This explanation accounts for:

- LOCO A passing within families;
- B being weaker but still positive late;
- the shared/per-word means collapsing together;
- ridge retaining a large advantage;
- whole-block transfer not being sufficient to rule out style, because a smooth style coordinate can transfer across blocks.

The unresolved causal distinction is therefore not “word versus state.” It is:

\[
\text{lexical content} \quad\text{vs}\quad
\text{presentation} \quad\text{vs}\quad
\text{contextual operational state}.
\]

**Second lens (verbatim):**

The result does say something important, but narrower than the current narrative:

> In this design, lexical content is not a sufficient predictor of the later forward step; context-bearing `X` contains predictable variation that word-conditioned means do not capture.

That is potentially exciting for the latent-world program: a denizen may need to navigate by operational context, not lexical content. But it is not yet a proven state law.

The current local holes are:

- F0 identity/token dominance: lexical identity overwhelms reusable movement at the first forward transition.
- Ordering saturation: the ordering readout repeatedly fails to register motion that cosine and response-law skill can detect. This is a hole in that readout, not proof that the latent world cannot register motion.
- Missing quotient: the representation does not yet provide a stable predictive separation between content, presentation, state, and consequential motion.

Presentation entanglement and family-only laws remain unresolved, not proven.

The constructive requirement for the next latent space is therefore:

> Define “same place” by interchangeability of declared moves and downstream response laws, rather than by lexical identity or superficial representational similarity.

If changing presentation changes the lawful successor, presentation may legitimately be part of operational state. The defect may be in our quotient, not necessarily in the world.

No experiments were run and no repository files were modified in this audit. The findings were recorded on the blackboard, and its convergence and synthesis checks completed successfully.

## Round 23 — unseen-word adjudication and presentation residualization (2026-08-28)

**Codex, documentation-only; no experiment is run.** The Round 22 unseen-word
artifacts, analyzer path, four requested ledger entries, and Round 22 contract
were checked directly. The running corrected equalized LOCO artifacts
(`analysis_locoeq2A/B.json`) were deliberately not opened.

### Adjudication of the Round 22 lock

The predeclared gate is met by both sentinel arms at four of five layers:

| Sentinel | F0 | F4 | F8 | F12 | F20 | full-gate keys at F4/F8/F12/F20 |
|---|---|---|---|---|---|---|
| `.` (A) | fail; continuation block collapses | pass | pass | pass | pass | 7/8, 7/8, 8/8, 8/8 |
| `,` (B) | fail | pass | pass | pass | pass | 5/8, 6/8, 8/8, 8/8 |

All eight block-by-word-fold keys have support `1.0`. At the passing layers,
7–8/8 keys are positive on all three endpoints; the block-first pooled
contrasts are positive and clear the point and lower-bound requirements. The
pooled ridge-minus-strongest-X-free-null margins are approximately:

- A: cosine `+0.14–+0.19`, skill `+0.33–+0.47`, and K=11 KL-rank
  `+0.35–+0.57`, with block-first lower bounds above `0.09`;
- B: cosine `+0.11–+0.17`, skill `+0.31–+0.41`, and K=11 KL-rank
  `+0.31–+0.52`, with block-first lower bounds above `0.09`.

At F0, A's pooled cosine lead is only about `0.019` and its continuation
block collapses; B's lead is about `0.018`. Both therefore fail the locked
layer gate. The class-mean and frozen-input-embedding `wordonly_knn` nulls
are effectively the shared-mean predictor at every layer in both artifacts.
The model revision, tokenizer revision, capture hash, CPU setting, reload
check, and locality metadata are present in each result manifest. This is a
valid Round 22 result, not a smoke inference.

The three locked predictions adjudicate as follows:

1. **State-linked:** held mechanically, but only in the permitted bounded
   form. The forward displacement `Delta = Y-X` remains conditionally
   predictable from `X`, and its downstream response-law consequence remains
   better predicted than either tested X-free lexical null at F4/F8/F12/F20.
   The same pattern transfers to word identities absent from calibration.
2. **Lexical interpolation:** the predeclared version did not hold. A
   class-conditioned mean and a frozen-input-embedding kNN field do not close
   the ridge gap on unseen words. This retires “the result is merely a
   word-conditioned lookup” for these tested lexical fields, not every
   conceivable lexical predictor; an embedding-to-displacement ridge or
   another lexical representation has not been run.
3. **Carrier/style nuisance:** remains live. Its prediction was that a smooth
   presentation coordinate could survive unseen-word transfer, and the
   observed survival is compatible with exactly that account. Unseen words
   remove word identity, not block/template coordinates. This prediction is
   therefore non-discriminating, not confirmed as causal.

The maximum earned statement is:

> In this decoder and probe design, `X` carries residual predictability of the
> forward displacement and its response-law consequence that generalizes
> across unseen word identities, beyond the tested class-mean and
> frozen-input-embedding lexical nulls.

In Audit #11's terminology, the object is **X-conditioned residual
predictability**, extended by **generalizing across unseen word identities**.
It is not yet presentation-independent state structure. Presentation
independence, model generality, a native law, useful multi-step navigation,
and a law that is intrinsic rather than decoder-specific remain unearned.

### Ruling on the equalized LOCO addendum

The unseen-word result changes the role of the addendum but does not erase it.
It is no longer a prerequisite for the separate unseen-word claim: that claim
already removes word-conditioned lookup and passes against the two locked
X-free lexical nulls. The corrected equalized LOCO rerun remains required to
interpret the **seen-word LOCO gap**, because the recorded addendum was
defect-affected: inner selection used a centre containing the validation
carrier, and the historical comparator was selected on held-out outcomes.

Thus `locoeq2A/B` is now a repair and a diagnostic of estimator variance and
the seen-word presentation/state alternative, not a gate that can veto the
unseen-word result. Its status remains “must complete before making a fair
seen-word equalized-LOCO interpretation.” The historical outer margins remain
descriptive only, exactly as Audit #11 ruled.

### Next measurement: cross-fitted presentation residualization

This is the next measurement in the fixed order, after the corrected
equalized LOCO reruns and before a second model family. It reuses the existing
forward captures, sentinel arms, layer checkpoints, `Delta` target, eight
block-by-word-fold keys, and three consequence endpoints.

#### Predeclared coordinates

The primary presentation design `P_static` contains only lexical-free
coordinates available in the captures:

- centered one-hot indicators for the four template blocks;
- tokenized prefix length, suffix length, total template length, and the slot
  index;
- sentinel/readout position and its relative position in the tokenized
  template.

The fixed augmented sensitivity design `P_aug` adds two carrier-level
coordinates without using any target `Y` or `Delta` from the test cell:

- the carrier mean of `X` over the other available lexical items, with the
  current test word left out; and
- rank-4 scores of that carrier-mean state in a carrier subspace whose basis
  is fitted on calibration carriers only. If a split has fewer than four
  estimable directions, the rank is truncated to the predeclared available
  rank; it is never selected from held-out outcomes.

`P_static` is the primary test of explicit template/presentation removal;
`P_aug` is a mandatory sensitivity, not a post-hoc choice between favorable
results. The carrier summary is an X-only covariate, not an X-free lexical
null; the lexical nulls remain separately defined and are not silently
relabelled as presentation-free.

#### Cross-fitted residualization procedure

For every outer block-by-word fold and each sentinel/layer, standardize the
chosen `P` using calibration cells only. Fit two multivariate nuisance maps,

\[
X = f_X(P)+X_\perp, \qquad
Delta = f_\Delta(P)+Delta_\perp,
\]

with the nuisance regularization selected inside the calibration portion and
then frozen. Estimate the carrier subspace and all map parameters afresh in
each outer fold. Apply the frozen maps to held-out cells; no held-out
`Y`, `Delta`, response law, bootstrap outcome, or comparator score may enter
coordinate construction, rank selection, or nuisance fitting.

Fit the existing state-conditioned field on `X_perp` to predict `Delta_perp`.
For displacement, score in residual space. For response-law endpoints,
restore only the calibration-fitted presentation displacement component when
constructing the candidate successor for the original decoder law, so that
the consequence test stays on the decoder's state manifold:

\[
\widehat Y = X + f_\Delta(P) + \widehat{Delta}_\perp.
\]

The strongest X-free lexical nulls predict `Delta_perp` without `X_perp`, are
reassembled with the same frozen presentation component, and are evaluated
on the same held-out law. A presentation-only `P -> Delta` arm is reported as
a diagnostic; it is not allowed to become the state claim by renaming.

#### Gate and interpretation

Use the same three primary endpoints: displacement cosine, response-law
skill against the shared-mean completion, and fixed-universe K=11 KL-rank.
Use the same support, finite/reload, locality, and eight-key validity checks.
For both `P_static` and `P_aug`, the residual field must beat the stronger of
the residualized `class_mean` and `wordonly_knn` nulls by at least `0.02` with
a positive word/block-clustered 95% lower bound on all three endpoints;
the block-first pooled contrast must be positive, at least 6/8 keys must be
positive on all three, and no held-out block may collapse. These are the
primary pass requirements.

Report the paired block-first contrast against the un-residualized Round 22
field on all three endpoints using the same bootstrap. Treat retention of at
least half of the un-residualized point margin on each endpoint as the
predeclared state-linked retention marker; failure of that marker is not a
new kill condition, but is evidence that presentation removal consumed the
signal and must be interpreted alongside the residual-vs-null gate. A
residualized pass with retention is the state-linked prediction. Collapse to
the X-free null, especially under `P_aug`, is the style-nuisance prediction.
If `P_static` passes but `P_aug` collapses, the result says the static
coordinates were incomplete; it does not establish state. If both pass, the
presentation concern is narrowed but not eliminated because unmeasured
presentation coordinates remain possible.

Budget: analysis of the existing captures, one CPU process per sentinel,
`20` shuffles and `500` bootstrap replicates, with a `60-minute` hard wall per
sentinel (`120` minutes total). No new capture is permitted in this budget;
an overrun or missing coordinate for a fold is budget-incomplete, not a pass.

#### Predictions

Under the state-linked account, F0 remains in the identity/token regime, while
F4–F20 retain positive residual-field margins over the X-free nulls and retain
at least half of the un-residualized margins, perhaps with attenuation. Under
the style-nuisance account, residualization removes most of the F4–F20 lead:
the residual field approaches the X-free null and loses the retention marker,
with the presentation-only arm explaining much of the original margin. A
mixed outcome is possible and will be reported as such, not forced into
either explanation.

#### Second model family

After residualization is adjudicated, repeat the full locked protocol in
`SmolLM2-360M` as the second family, subject to the same CPU-only hardware
constraint. Before capture, pin the exact model and tokenizer revisions, all
weights, framework versions, dtype, device, thread count, capture hash,
template/config file, literal item list and one-token checks, sentinel token
IDs, probe text and tokenization, and the normalized layer map. Use the five
checkpoints corresponding to normalized depths `0, 4/28, 8/28, 12/28,
20/28`, rounded once and frozen before looking at outcomes. Use the same two
sentinels, `Delta = Y-X`, block-by-word folds, `P_static` and `P_aug`, nulls,
three endpoints, K=11 universe, bootstrap, and residual-vs-unresidualized
comparison. If either sentinel is not a single token under the pinned
tokenizer, record a protocol failure and preregister any replacement before
scoring; do not silently substitute it.

Budget: one CPU capture/analysis process, `20` shuffles and `500` bootstrap
replicates, with a `180-minute` hard wall for the second-family run. A
second-family claim requires the pinned manifests and validity gates plus the
same two-of-five layer rule for both sentinel arms; otherwise the result is a
bounded family-specific diagnostic. State-linked prediction is transfer of
the residualized F4–F20 pattern. Style-nuisance prediction is collapse after
residualization. Neither prediction licenses a native-law claim without
cross-family consistency and the remaining consequence/composition checks.

### Second lens: holes after unseen words

The unseen-word result sharpens the answer to the guiding question: a denizen
can sometimes predict the world's next move from context-bearing `X` even
when the inserted word identity was never seen during calibration. The
exciting consequence is that navigation may require an operational context,
not a dictionary key—but only if that predictive context survives removal of
presentation coordinates.

The proven local holes are:

- **F0 identity/token dominance:** the first transition remains dominated by
  lexical/token identity; both sentinels fail the conditional-gain gate there.
- **Ordering-saturated readout:** the inherited across-word ordering endpoint
  fails to register motion that cosine and response-law skill detect. This is
  proven for this endpoint and design, not for the latent world as a whole.

Several apparent holes are now classified more carefully. “The surviving
signal is word lookup” is not supported by the unseen-word result for the
tested class-mean and embedding-kNN nulls. The earlier equalized-LOCO reading
that maximal shrinkage was data-selected is an implementation artifact and
remains withdrawn. “Motion is invisible” is a readout artifact, not a proof
that motion cannot be consequential. Presentation entanglement and
template-family-only laws remain unresolved, not proven and not dismissed;
the unseen split crosses word identities and block folds but remains one
decoder and one probe design.

The remaining structural hole is a missing predictive quotient separating
lexical content, presentation, operational state, and consequential motion.
The next-generation latent space must define “same place” by interchangeability
of declared moves and downstream response laws; expose or factor presentation
coordinates; use consequence-sensitive divergence rather than inherited
ordering alone; support multi-step composition; and generalize across unseen
words, styles, and model families. Precision, support, and response-law
validity must be part of the representation's contract.

The single measurement that most sharpens this now is the predeclared
cross-fitted presentation residualization above. Multi-step composition
(`F4 -> F8 -> F12` along the token clock) is the next navigation test after
the state/presentation ambiguity is narrowed; a non-punctuation sentinel is a
useful later style-family stress test. No new axiom is warranted in Round 23.

## Tier-3 audit #12 — unseen-word result, sentinel '.' (2026-08-29, fresh Codex auditor)

**Adopted corrections.** (1) Mechanical pass confirmed (F4–F20; F0 fails);
status: **"mechanical pass under the recorded reduction; formal gate pending a
contract-correct bootstrap"** — the predeclared class-preserving word
bootstrap was not implemented (words resampled without class strata and
nested within blocks although they are crossed with blocks); only four
block clusters exist, so intervals are sensitivity summaries. Repaired
prospectively in the analyzer (class-stratified, crossed word draws).
(2) The lexical null family is weak (four class means; k = 5 frozen-embedding
kNN over 40 words); required before "not lexical": nested frozen-embedding→Δ
ridge, nested embedding-conditioned kernel, a predeclared k ladder.
(3) Wording: "not exact held-out-word lookup and not the tested lexical
interpolator" — never "not word lookup" unqualified; "the tested lexical nulls
fail", not "lexical content is absent"; the ~0.06 seen→unseen drop is a point
comparison at F8 only. (4) F0: "non-qualifying, with the continuation
held-out block providing the strongest local failure pattern"; no formal
collapse statistic exists. (5) The positive object remains X-conditioned
residual predictability, now "transferring across the held-out word fold and
held-out block".

**Overclaims to reject (verbatim):**

These formulations are not supported:

- “Not word lookup” without saying “not exact lookup and not the tested lexical interpolator.”
- “Not class lookup” as a statement about all class-conditioned lexical models.
- “The forward law is about context rather than content.”
- “The state-conditioned component is large.”
- “A native law generalizes across unseen words.”
- “F0 has no state dependence.”
- “Presentation has been ruled out.”
- “The latent space supports content-based structured reasoning.”

The positive object should remain:

> `X`-conditioned residual predictability.

That is stronger and more honest than a generic regression result, but weaker than state-conditioned structure.

**Strongest alternative explanation (verbatim):**

The strongest single explanation is a combined smooth-coordinate account:

> `X` contains smooth lexical and presentation/template coordinates. The later displacement and response-law consequence vary systematically along those coordinates. Ridge and kernel learn this implementation-specific geometry. The class mean and five-neighbor embedding nulls collapse because they are too coarse, not because the variation is necessarily operational state.

This explains:

- F4–F20 success;
- F0’s weak result;
- continuation’s F0 failure;
- the near-shared-mean lexical nulls;
- the success of a generic kernel;
- transfer across the four blocks;
- survival to unseen words through interpolation in contextualized embedding space.

This alternative does not require leakage or a broken split. It only requires that the current residual representation entangles lexical content, presentation, and operational context.

**Alternative explorations and cheaper baselines (verbatim):**

The repository’s verbatim alternative instructions remain appropriate:

- A fixed-input style-balance control: match or residualize carrier/template features before fitting displacement.
- A within-style or within-template null that preserves carrier style while removing state pairing.
- A style-held-out split, not only a carrier-held-out split.
- A per-word, per-style mean displacement baseline.
- A low-dimensional block/template-only predictor.
- A direct `Δ = Y − X` decomposition into word mean, carrier mean, shared mean, and residual.
- A target permutation preserving carrier-level marginal structure rather than destroying all carrier alignment.
- Per-layer and per-sentinel locality and precision reports using fresh float32 values.
- Replication on a second model family before any general language-model claim.

For this specific audit, the highest-value additions are:

1. Nested frozen-embedding-to-`Δ` ridge.
2. Nested embedding-conditioned kernel.
3. Predeclared embedding-k sensitivity.
4. Cross-fitted residualization of frozen lexical embedding and presentation coordinates from both `X` and `Δ`.
5. A corrected crossed, class-preserving bootstrap.
6. Multi-step forward composition, not only one-step prediction.

**Second lens (verbatim):**

The result says something important but narrow:

> Lexical content alone is not a sufficient predictor of the later forward step; context-bearing `X` contains predictable variation that the tested word-only and class-only predictors do not capture.

For a denizen, this means lexical identity is not yet a reliable definition of “same place.” The denizen may need to navigate by operational context and by the response laws attached to that context.

But the result does not show that the context is a clean state variable. It may be:

- operational state;
- presentation/style;
- smooth lexical geometry;
- or an inseparable mixture.

So the current space gives evidence for context-conditioned structure, but not yet for content-based structured reasoning in the strong sense. It lacks a stable predictive quotient separating:

- lexical content;
- presentation;
- operational state;
- consequential motion.

The constructive requirement for the next latent space is therefore:

> Define “same place” by interchangeability of declared moves and downstream response laws, rather than by lexical identity or superficial representational similarity.

That space should expose presentation coordinates, use consequence-sensitive divergence, support multi-step closure, and generalize across lexical identities, style families, and model families.

Final disposition: retain the unseen-word result as a valuable mechanical and descriptive finding; block the stronger “not lexical,” “state-conditioned,” and “native forward law” readings until the stronger lexical controls and contract-correct hierarchical bootstrap are run. No experiments were run and no repository files were modified.

## Round 24 — audit #12 repair ruling and residualization contract (2026-08-28)

**Codex, documentation and prospective analyzer amendment; no experiment was
run.** The running `analysis_locoeq2A/B.json` artifacts were not opened. The
Round 23 smoke was read as a pipeline preview only.

### 1. Audit #12 repairs

#### Crossed bootstrap

The class-preserving repair in the pre-amendment analyzer was not literally
crossed. It stratified word positions by lexical class, but the pooled
block-first loop drew a new word vector independently for each block/fold
matrix. That is a valid class-preserving sensitivity bootstrap, not the
predeclared crossed word-by-block bootstrap, because a crossed factor must use
the same word draw across the blocks in a replicate.

The prospective repair is now contract-correct: for each bootstrap replicate,
draw one class-stratified word resample for each held-out word-fold key and
reuse it for every sampled block carrying that key; resample carriers within
the sampled block. The word draw is shared across the block factor, while the
carrier draw remains nested in block. The per-key bootstrap remains
class-preserving. The K=13 word-fold layout is checked rather than inferred
from width alone. Until this amended analyzer is run, the old A/B intervals
remain sensitivity summaries.

#### Stronger lexical nulls

The four nulls are fair and nested for the claim they test:

- `class_mean` is the calibration-carrier/calibration-word class mean;
- `wordonly_knn` is an X-free frozen-input-embedding interpolator;
- `wordonly_ridge_emb` maps frozen input embeddings to calibration-word mean
  displacement; and
- `wordonly_kernel_emb` is the corresponding embedding-conditioned kernel.

The embedding standardizer is fit on calibration words only. The k ladder
(`1, 3, 5, 10, 20`), ridge lambda grid, and kernel gamma/lambda grid are
selected by a two-fold class-stratified split of calibration words, then
refit on all calibration words. No held-out target enters selection or fit.
These are strong X-free lexical baselines, but still test the chosen frozen
embedding representation rather than every possible lexical account.

#### Amended unseen-word gate

Round 22 is amended: at every headline unseen-word layer, the state-conditioned
field must beat the **strongest of all four** X-free nulls
(`class_mean`, `wordonly_knn`, `wordonly_ridge_emb`,
`wordonly_kernel_emb`) by the locked margin, with positive clustered lower
bounds on cosine, law skill, and the fixed **K=13** KL-rank universe. The
block-first pooled contrast, at least 6/8 positive all-three keys, support,
and no-block-collapse requirements remain unchanged.

The existing A/B artifacts (`K=11`, two nulls, and the old nested-within-block
word bootstrap) do not need to be deleted or rerun as historical artifacts.
They remain mechanical-only/descriptive under the old reduction and cannot
support the amended formal gate. The amended gate is folded into the four
residualization runs, which must emit the stronger nulls, K=13 universe, and
contract-correct bootstrap. The raw un-residualized shadow field and its raw
four-null margin must also be retained in those same folds so that the amended
unseen gate and the retention denominator are not reconstructed from old
fold summaries. The current prospective patch adds the raw ridge arm and
paired contrasts; the raw four-null shadow remains a launch prerequisite and
is not yet claimed as implemented.

### 2. Residualization implementation ruling

The implementation is directionally faithful to Round 23, but the following
prospective repairs are binding before scoring:

- `P_static` has the declared centered four-block indicators, tokenized prefix,
  suffix, and total lengths, slot index, sentinel position, and relative
  position. This coordinate set is correct.
- `P_aug` is an X-only sensitivity with a leave-current-word-out carrier mean
  and a rank-4 carrier-mean subspace. To make it genuinely cross-fitted, the
  carrier mean now uses the outer calibration-word pool, and the PCA basis is
  fit on calibration carriers and calibration words only; the rank is truncated
  only when fewer directions are estimable. This is stricter than using all
  words merely because their X values are observable.
- Nuisance maps are `P -> X` and `P -> Delta`, with separate lambda selection
  by inner leave-one-calibration-block-out scoring. The augmented basis is
  refit on each inner training-carrier set. The main field ladder is selected
  in residual space, not by the raw `X -> Delta` inner folds.
- Displacement scoring uses `X_perp` and `Delta_perp`. The lexical nulls are
  fit to `Delta_perp` without `X_perp`, so they remain X-free within the
  residualized problem.
- Law completion uses the required reassembly
  `Yhat = X + f_Delta(P) + Delta_perp_hat`. The presentation-only `P ->
  Delta` cosine is retained as a diagnostic and cannot be relabelled as a
  state predictor.
- A same-fold, same-invocation un-residualized ridge arm is required. Comparing
  only against Round 22 artifact fold values is not adequate: it cannot give
  a paired block-first contrast or guarantee identical fold-level selection.
  The prospective analyzer patch adds this raw arm and paired contrasts. The
  raw four-null shadow margin must still be emitted alongside it for the
  retention marker, not inferred from the residualized nulls; until then the
  four runs are not ready to launch.

The retention marker is therefore: for each endpoint and the same outer
folds, the residual ridge-minus-strongest-residual-null point margin retains at
least half of the un-residualized ridge-minus-strongest-raw-null point margin;
the paired block-first contrast against the same-run raw ridge is reported
with its clustered interval. Failure of retention is not a kill condition,
but it is evidence that presentation removal consumed the signal. A paired
contrast or denominator reconstructed from the old A/B artifacts is not a
pass.

### 3. Smoke preview and predictions

The F8, sentinel-`.` static smoke is consistent with the state-linked
prediction: presentation-only `P -> Delta` cosine is `0.42`; residual-space
X-free nulls are about `0.06–0.07`, while ridge is `0.60` (the corresponding
un-residualized preview is `0.66`). Skill is `0.36` versus about `0.015`, and
ridge leads the strongest null by about `+0.56` cosine, `+0.34` skill, and
`+0.44–+0.50` KL-rank.

This preview shows that the static residualization path is wired plausibly
and that a large X-conditioned signal survives this one test. It does not
establish a formal gate: it is one sentinel, one layer, one design, and a
tiny smoke bootstrap. It does not establish causal presentation removal,
state structure, model generality, or a native law. It confirms the Round 23
prediction as a preview, not as evidence sufficient to adjudicate it.

### 4. Locked run order

No run is launched in Round 24. The four CPU-only runs are, in order:

1. sentinel A, `P_static`;
2. sentinel A, `P_aug`;
3. sentinel B, `P_static`; and
4. sentinel B, `P_aug`.

Each uses the existing captures, five layers `F0/F4/F8/F12/F20`,
`--unseen-words 2`, 20 shuffles, 500 bootstrap replicates, and a 60-minute
hard wall per sentinel/design. One process runs at a time. The readout order
is manifest/reload/locality/support validity; then F0; then F4, F8, F12, F20;
within each layer, residual displacement, strongest-four-null margins and
clustered gates, reassembled law endpoints, presentation-only diagnostic,
and paired raw-field retention. Adjudicate static before augmented, and read
both sentinel arms without choosing the favorable arm. An overrun or missing
fold coordinate is budget-incomplete, not a pass.

### 5. Second lens

The statement “presentation explains `0.42` of the raw forward displacement
while `X_perp` explains the residual far beyond content nulls” changes the
diagnostic resolution, not the answer. It makes presentation a measured,
substantial coordinate and makes survival after its static removal worth
testing. It does not prove that the remainder is operational state: smooth
lexical geometry, unmeasured presentation, and their mixture remain live.

The hole is now more explicit: this latent space lacks a stable predictive
quotient separating lexical content, presentation, operational state, and
consequential motion. A denizen still cannot define “same place” by lexical
identity or by raw similarity alone. The constructive target remains a space
where sameness means interchangeability of declared moves and downstream
response laws, with exposed presentation coordinates, consequence-sensitive
readouts, multi-step closure, and transfer across words, styles, and model
families. No new native axiom is warranted.

## Round 25 — raw-shadow launch ruling, budget amendment, and equalized-A adjudication (2026-08-29)

**Codex, documentation-only; no experiment was run in this round.** The live
analyzer, the F8/A residualization smoke, the raw-shadow ledger entry, the
corrected equalized-A ledger entry, and the Round 24 contract were checked.

### Raw four-null shadow prerequisite

The Round 24 launch prerequisite is **met as a pipeline prerequisite**. The
same invocation and same outer folds produce the residual field, the raw
un-residualized ridge shadow, and the raw four-null shadow. In the unseen-word
delta design, “raw target” means the un-residualized held-out `Delta = Y-X`
target (`Yt_raw`); the raw completion reconstructs the successor from the
held-out `X` and that predicted displacement.

The fixed raw arms are:

- `unres_ridge`, with its ridge lambda selected inside calibration folds;
- `unres_class_mean`;
- `unres_wordonly_knn`;
- `unres_wordonly_ridge_emb`; and
- `unres_wordonly_kernel_emb`.

`unres_mean` is the raw shared-mean completion used as the skill reference,
not a fifth X-free null in the four-null margin. All four X-free nulls are
calibration-only fits: the class mean is fit on calibration carriers and
words, while the embedding kNN/ridge/kernel choices are selected on the
calibration-word inner split and then refit on calibration words. The raw
arms score against `Yt_raw`; raw skill is `1 - KL(raw candidate, truth) /
KL(unres_mean completion, truth)`.

For KL-rank, each `unres_*` arm is substituted into the `ridge` slot of the
fixed K=13 candidate universe. It is therefore compared on the same rank
scale without adding raw arms to the formal candidate universe. The raw
shadow is a comparator/retention arm, not a new ladder member.

The F8/A smoke records 160 defined cells. `unres_ridge` exceeds each of the
four raw nulls by approximately `+0.190–+0.192` cosine, `+0.291–+0.296`
skill, and `+0.289–+0.305` KL-rank at the point level. Its
`pooled_gates_block_first` entries cover four blocks and eight word-fold keys;
the twelve raw-ridge-versus-null pooled means are approximately
`+0.187–+0.189` cosine, `+0.335–+0.337` skill, and `+0.302–+0.313` KL-rank,
with positive block-first 95% intervals. This verifies the arm, support, raw
target/reference wiring, K=13 substitution, per-null margins, and pooled
contrast. The smoke remains pipeline validation, not a formal result.

### Retention marker: frozen definition

Let `N4` be the ordered set
`{class_mean, wordonly_knn, wordonly_ridge_emb, wordonly_kernel_emb}`. For
each endpoint `e` in `{cosine, skill, KL-rank}`, use the same outer fold/cell
support and define the point margins

\[
m^{raw}_e = \min_{n\in N4} \operatorname{mean}\bigl(raw\_ridge_e-raw\_null_{n,e}\bigr),
\qquad
m^{res}_e = \min_{n\in N4} \operatorname{mean}\bigl(ridge_e-res\_null_{n,e}\bigr).
\]

The “strongest null” is thus the fixed null with the **smallest point
margin** (equivalently, the largest held-out score) for that endpoint. The
residual and raw sides use exactly the same rule; ties use the declared order
above. No null is dropped or made the sole gate comparator after seeing a
favorable endpoint, and no calibration winner replaces this conservative
minimum: every null is a predeclared arm with its own calibration-only tuning,
and every per-null margin remains reportable. The strongest-null name is only
the mechanically derived minimum-margin label.

The retention marker is valid only when `m^{raw}_e > 0`, and passes endpoint
`e` when `m^{res}_e >= 0.5 * m^{raw}_e`. The run reports all four raw and all
four residual margins, the strongest-null name for each endpoint, and the
same-fold paired block-first contrast `ridge - unres_ridge` with its
clustered interval. The paired contrast is a separate diagnostic; it cannot
replace or reconstruct the retention denominator from the old Round 22 A/B
files. Failure of retention is not a kill condition, but means presentation
removal consumed more than half of that endpoint's raw margin and must be
read with the residual-vs-null gate.

### Budget amendment before scoring

The measured one-layer full-invocation times are `756 s` before the raw ridge
arm, `998 s` with the raw ridge arm, and `1294 s` with all five raw-shadow
arms under CPU contention. A five-layer run therefore projects to roughly
63–108 minutes before ordinary variance. The Round 24 `60-minute` wall is
not credible for the locked invocation.

Before any formal score is opened, amend the wall to **120 minutes per
sentinel/design run and 8 hours for the four serial runs**. Retain the fixed
design, including `knn1`, `knn5`, and `knn20`; those candidates are part of
the locked K=13 universe and are not dropped to fit the old wall. One process
runs at a time. A run exceeding 120 minutes, or missing a fold coordinate,
is budget-incomplete and cannot earn a gate claim. This is a prospective
budget amendment, not an outcome-dependent relaxation.

### Conditional launch and locked order

Once the corrected equalized LOCO rerun for sentinel B has finished, launch
the four CPU-only residualization runs in this order, with no interleaving:

1. sentinel A, `P_static`;
2. sentinel A, `P_aug`;
3. sentinel B, `P_static`; and
4. sentinel B, `P_aug`.

Each run retains `F0/F4/F8/F12/F20`, two unseen-word folds, 20 shuffles,
500 bootstraps, the K=13 universe, the raw-shadow arms, and the validity
checks. This round confirms the launch authorization conditionally; no run
is launched or scored in Round 25 itself. Read validity first, then F0,
then F4/F8/F12/F20; within each layer read residual margins and gates,
reassembled law endpoints, the presentation-only diagnostic, and paired
retention.

### Corrected equalized LOCO addendum A

Under the Round 22/Audit #11 wording, corrected `locoeq2A` is a **valid
mechanical positive for the sentinel-A seen-word within-family diagnostic**:
the properly calibrated word-only ridge and shrunk-word-mean baselines sit
`0.003–0.01` above the shared mean rather than collapsing exactly onto it,
and the ridge field passes the equalized gate at `F4/F8/F12/F20`; `F0` fails.
The correction resolves Audit #11's inner-centre defect and removes the
claim that maximal shrinkage was forced by the implementation. It does not
license “no per-word lexical signal,” “context rather than content,” “the
state-conditioned component is large,” or a presentation-independent/native
law claim. The earned wording remains the bounded Audit #11 object:

> On already-seen words, within sentinel A's style-family design, the
> context-bearing `X` field predicts the held-out carrier's forward
> displacement and response-law consequence beyond these properly nested,
> equalized X-free lexical baselines at `F4–F20`.

The B corrected rerun is required before any combined A/B equalized reading.
The unseen-word result remains separately bounded as X-conditioned residual
predictability transferring across held-out word identities; neither result
establishes clean operational state, a native law, or model generality.

### Second lens

The denizen can now ask whether a move survives removal of measured
presentation coordinates without silently changing the comparator. That is
the exciting next step: a latent world becomes navigable only when its
“same place” relation and move cost survive fixed nulls, raw-vs-residual
accounting, and held-out response-law tests. The current hole remains a
missing predictive quotient between lexical content, presentation,
operational state, and consequential motion. No new axiom is warranted.

## Tier-3 audit #13 — equalized A, residualization instrument, public demo (2026-08-29, fresh Codex auditor)

**Adopted corrections.** (1) Equalized A: "Audit #11's inner-centre *defect*
concern is resolved by the corrected sentinel-A data" — not "audit #11 is
resolved"; the comparator is the **calibration-selected equalized comparator**
(chosen by inner cosine, frozen for all endpoints), not "the stronger" one; the
baselines sit roughly 0.002–0.009 above the shared mean; the pooled equalized
interval is secondary (carriers, not blocks first). Maximum wording: "On
already-seen words, within sentinel A's style-family design, the
context-bearing X field predicts the held-out carrier's forward displacement
and response-law consequence beyond the properly nested, calibration-selected
X-free lexical comparator at F4–F20." (2) **Retention marker defect:** raw and
residual margins are not commensurate (cosine on different targets; skill
against different references; KL-rank by ridge-slot substitution). Until a
common-scale repair (reassemble residual predictions to full Δ and score
against raw Δ with a common skill reference; recompute the strongest-null
minimum inside every bootstrap replicate; a coherent raw universe or a
continuous KL margin), the runs may say only "the predeclared robustness
marker is mechanically met", never "half of the signal survives". The raw
four-null shadow is valid for the amended raw unseen-word comparison. The
residual-vs-null gate, law reassembly and the presentation-only arm are
coherent; `P_static→Δ` = 0.427 is a held-out cosine, never "explains 42%".
(3) **Public demo:** materially violated audits #10–#12 ("context state",
"context takes over", "manufactures", "presentation explains 0.42"; nearest-
state predictor coloured as content); every replacement adopted verbatim and
republished; named-word rows labelled as selected illustrations. (4) Reverse
tunnel: it would now be unfair to dismiss the X-conditioned advantage as
lookup, noisy-mean artifact, class mean, embedding interpolation or pipeline
accident; and presentation may itself be part of operational state — the
target is an operational equivalence relation defined by moves and
consequences, not presentation invariance at all costs.

**Retention-marker defect (verbatim):**

The present numerator and denominator do not compare like with like.

#### Displacement cosine

Raw:

\[
\cos(\widehat\Delta,\Delta)
\]

Residual:

\[
\cos(\widehat\Delta_\perp,\Delta_\perp)
\]

Residualization changes the target geometry and scale. A margin can become larger after removing an easy common component even when less raw predictive information remains. The smoke demonstrates exactly this: the nominal cosine “retention” is about \(0.523/0.187\approx2.8\), which cannot mean 280% of a signal survived.

#### Law skill

The residual arms normalize skill against the reassembled residual mean completion. Raw arms normalize against `unres_mean`. Because the denominators differ, subtracting or ratioing the resulting skills is not an apples-to-apples retention measurement.

#### KL-rank

Each raw arm is separately substituted into the residual K=13 universe’s `ridge` slot. This gives raw arms a common ordinal reference, but it is not the same estimand as the jointly ranked residual ridge-versus-residual-null margin:

- the raw ridge and raw null are never jointly present;
- both are compared against residual competitors;
- a raw null can be ranked against its residual counterpart;
- the residual null occupies its own candidate slot, whereas the raw null replaces ridge.

This is adequate as a coarse diagnostic, not as a formal KL-rank retention denominator.

### Required repair

Keep residual-space cosine as the residual predictability gate. For retention, add a common-scale comparison:

1. Reassemble every residual prediction as
   \[
   \widehat\Delta_{\rm full}=f_\Delta(P)+\widehat\Delta_\perp.
   \]
2. Reassemble every residual lexical null with the same \(f_\Delta(P)\).
3. Score raw and reassembled-residual arms against the same raw \(\Delta\), same true law, and same skill reference.
4. For rank, either build a coherent full raw K=13 shadow universe or use a common continuous KL margin rather than separate ridge-slot substitutions.
5. Bootstrap the paired common-scale margins and, if using “strongest null,” recompute the minimum inside each replicate.

Until then, use:

> The predeclared robustness marker is mechanically met.

Do not use:

> At least half of the predictive signal survives residualization.

**Strongest alternative explanation (verbatim):**

> \(X\) contains smooth lexical and presentation/template coordinates. The later displacement and response-law consequence vary systematically along those coordinates. Ridge and kernel learn this implementation-specific geometry. The class mean and frozen-embedding lexical nulls collapse because they are too coarse, not because the variation is necessarily operational state.

This explains:

- F4–F20 success;
- F0 weakness;
- cross-block transfer;
- unseen-word interpolation;
- near-mean lexical nulls;
- generic ridge/kernel success;
- the sizeable \(P_{\rm static}\to\Delta\) cosine;
- survival after incomplete static residualization.

It does not require leakage or broken splits.

The strongest competing explanation in the positive direction is:

> The model has a real context-conditioned transition regularity; presentation affects the lawful successor because presentation is part of operational state, and the analyst’s attempt to quotient it away removes legitimate state information.

The current observational residualization cannot decide between those accounts.

---

**Second lens (verbatim):**

### Proven locally

- **F0 identity/token dominance:** no qualifying conditional gain at the first forward transition under this instrument.
- **Ordering-endpoint saturation:** the inherited ordering readout repeatedly fails to register motion captured by displacement cosine and response-law skill.

### Strong concerns, not proven holes

- presentation entangled with operational state;
- smooth lexical geometry surviving unseen-word transfer;
- one model-specific conditional field;
- reliance on externally supplied template metadata;
- only one-step, decoder-specific consequence.

### Not proven

- family-only laws;
- presentation explains the whole X-conditioned advantage;
- motion is invisible to the world;
- the latent space cannot support structured reasoning;
- the residual field is a native law.

The deeper structural problem is that the denizen does not receive a native quotient. `P_static` uses analyst-known block identities, prefix lengths, suffix lengths, and positions. If “same place” can only be defined using metadata outside the latent world, the representation itself has not exposed the coordinates needed for navigation.

A next-generation latent space should therefore:

- expose lexical, presentation, and operational coordinates rather than requiring external reconstruction;
- define sameness through interchangeability of declared moves and downstream response laws;
- treat presentation as state when it causally changes those laws, and quotient it only when it does not;
- include consequence-sensitive divergences;
- support multi-step composition and closure;
- transfer across unseen words, styles, and model families;
- make precision, support, and controllability part of the representation contract.

No new native axiom is earned. The next constructive target is an operationally testable quotient, not another renaming of regression success.

No repository files were modified, no experiments were run, and none of the excluded result files were opened. Findings were recorded on the blackboard; convergence and synthesis completed successfully.

## Round 26 — A-static residualization adjudication and remaining readout contract (2026-08-29)

**Codex, documentation-only; no experiment was run.** The Round 23
predeclaration and predictions, the Round 24/25 amendments, Audit #13, the live
`--residualize` implementation, `analysis_resSA.json`, and the five named
ledger entries were checked directly. The running/queued `analysis_resAA.json`,
`analysis_resSB.json`, and `analysis_resAB.json` artifacts were not opened.
Findings were recorded on the blackboard before this adjudication; convergence
and synthesis completed with no open signal or dispute.

### Artifact validity and mechanical score

`resSA` is sentinel A (`.`), `P_static`, five forward-layer checkpoints, two
unseen-word folds, 20 shuffles, and 500 bootstrap replicates. It completed in
`4405.7 s` of the amended `7200 s` wall. The model/tokenizer pins, CPU/dtype
metadata, fixed K=13 universe, reload record, eight fold keys, and support
accounting are present; support is `1.0` in every key.

The primary residual-vs-null gate adjudicates as follows. Each endpoint margin
is the conservative minimum over the fixed four residualized X-free nulls; the
reported lower bound belongs to that minimum-margin comparator:

| Layer | `X_perp` ridge cosine | strongest residual null cosine | cosine margin [LB] | skill margin [LB] | KL-rank margin [LB] | full / positive keys | block collapse | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| F0 | 0.268 | -0.005 | +0.273 [+0.068] | -3.570 [-12.692] | +0.042 [-0.384] | 2/8 / 6/8 | association | fail |
| F4 | 0.619 | 0.061 | +0.558 [+0.517] | +0.388 [+0.175] | +0.395 [+0.222] | 7/8 / 7/8 | none | pass |
| F8 | 0.596 | 0.073 | +0.523 [+0.485] | +0.305 [+0.198] | +0.403 [+0.315] | 7/8 / 8/8 | none | pass |
| F12 | 0.556 | 0.059 | +0.497 [+0.458] | +0.339 [+0.207] | +0.412 [+0.197] | 6/8 / 8/8 | none | pass |
| F20 | 0.624 | 0.070 | +0.554 [+0.512] | +0.477 [+0.411] | +0.612 [+0.558] | 8/8 / 8/8 | none | pass |

Thus F4/F8/F12/F20 pass the amended residual gate; F0 fails. The same-fold
raw shadow is present and reproduces the un-residualized scale. At F4/F8/F12/
F20 its ridge cosine is `0.712/0.661/0.654/0.764` versus strongest raw-null
cosine `0.525/0.474/0.504/0.622`. This validates the raw shadow as a comparator;
it does not repair the retention estimand described below.

### Round 23 prediction scorecard and branch

The `P_static` result takes the **non-collapse, state-linked-side branch of the
primary gate**: F0 remains in the registered identity/token regime, while
F4–F20 retain positive residual-field margins beyond the strongest
residualized X-free lexical null. The registered `P_static` style-nuisance
prediction of collapse is missed.

That is not a state verdict. Round 23 explicitly reserved the mixed branch
`P_static` pass plus `P_aug` collapse as “the static coordinates were
incomplete, not state.” Because `P_aug` is unread, A-static alone does not
distinguish the state-linked account from that mixed branch. The maximum
earned statement is:

> In this decoder and probe design, after removal of the **registered static
> presentation coordinates**, `X_perp` retains held-out predictability of
> `Delta_perp` and its reassembled response-law consequence beyond the
> strongest residualized X-free lexical nulls at F4–F20, across held-out word
> folds and held-out blocks.

This is residual predictability after removal of the registered coordinates.
It is not presentation independence, clean operational state, a native law,
model generality, or multi-step navigation.

### Retention-marker ruling

Audit #13 controls the wording for this artifact. `resSA` was produced by the
pre-patch analyzer and contains neither
`gates["ridge"]["retention_common_scale"]` nor
`pairs[*]["retention_common_scale_block_first"]`. Its older raw-versus-residual
marker is mechanically true at F4–F20 on all three endpoints, but the compared
targets, skill references, and KL-rank estimands are not commensurate.
Therefore the only admissible retention statement for `resSA` is:

> The predeclared robustness marker is mechanically met.

Do not say that half of the signal survives, that signal was amplified, or
that a common-scale retention threshold passed.

A patched A-static rerun is required before any **A-static common-scale
retention claim** or any symmetric “all four sentinel-by-design cells retain”
claim. `resAA` and the two B runs cannot substitute for the missing A-static
cell: sentinel and presentation design are not exchangeable factors. This
rerun is not required for the valid primary residual-vs-null result above and
does not precede the already locked chain. Finish A-augmented, B-static, and
B-augmented first; then run the queued patched A-static analysis before any
cross-run retention synthesis or second-family launch. This is a future
CPU-analysis requirement, not an experiment run in Round 26.

### What the presentation-only arm changes

The held-out `P_static -> Delta` cosine averages by layer are
`0.625/0.491/0.427/0.456/0.584` at F0/F4/F8/F12/F20. These are cosines, not
variance explained and not percentages. They establish that the registered
block, length, and position coordinates predict a large component of the raw
forward displacement.

That changes the scientific reading of the earlier unseen-word result. Its
recorded raw advantage is reproduced by the same-fold shadow with stronger
nulls, but lexical novelty never removed presentation. A substantial, and
possibly much, part of the earlier “X-conditioned” lead may therefore have
been presentation-mediated. The result did not warrant “context rather than
presentation” before, and the measured size of this arm makes that caveat
central rather than peripheral.

*Audit #14 (Tier-3, `theory/EXPERIMENTS.md`, ledger `nlm007_audit14_adopted`) withdrew the sentence "much of the raw lead may have been presentation-mediated" as an over-read and replaced the ruling; read this paragraph as corrected there.*

`resSA` adds a separate positive fact: the registered `P_static` coordinates
do not account for the whole X-linked advantage. After their calibration-only
removal from both `X` and `Delta`, `X_perp` still predicts `Delta_perp` far
beyond the residualized lexical nulls and restores better response-law
consequences. It does not identify that remainder as state; unmeasured
presentation, smooth lexical geometry, operational context, and mixtures
remain live.

### Guiding question and second lens

For the denizen, the registered template variables are part of the measured
geography: they change where the forward move tends to go. The local proven
facts are now:

- registered static presentation coordinates predict a substantial component
  of held-out forward displacement;
- those registered coordinates do not exhaust the X-linked residual
  predictability at F4–F20;
- F0 identity/token dominance and the inherited ordering readout's saturation
  remain the two proven local holes.

The structural hole “presentation is entangled with operational state” is
**not proven**. The measurement establishes presentation sensitivity, not
inseparability. It has neither supplied an operational definition of state nor
shown by intervention whether presentation should be quotiented away. A
presentation variable that changes downstream response laws may be legitimate
operational state. Conversely, survival after a finite analyst-supplied
residualization does not show that the remainder is presentation-free. The
current space still lacks a **demonstrated native quotient** separating lexical
content, presentation, operational state, and consequential motion; that is a
measurement limit and constructive target, not a proof that no such quotient
can exist.

The next latent space must expose those candidate coordinates to its denizen
rather than require block identities and token-length metadata supplied by an
analyst. It must define sameness by interchangeability under declared moves
and downstream response laws; treat presentation as state when controlled
changes alter those laws and quotient it when they do not; use
consequence-sensitive divergences; support multi-step composition and closure;
transfer across unseen words, styles, and model families; and make precision,
support, and controllability part of the representation contract. No new
native axiom is earned.

### Remaining three-run readout and `P_aug` branch

Read the remaining artifacts without selecting a favorable arm:

1. A-augmented (`resAA`);
2. B-static (`resSB`);
3. B-augmented (`resAB`).

For each run, read the `7200 s` budget, manifest/revision/capture identity,
reload/locality/support checks, and presence and finiteness of both common-
scale retention fields first. Then read F0, followed by F4/F8/F12/F20. Within
each layer read, in order: residual displacement and strongest-four-null
gates; reassembled skill and KL-rank; key counts and block collapse;
presentation-only diagnostic; same-fold raw shadow; and the per-fold plus
block-first common-scale retention ratios. Static is adjudicated before
augmented for B, and no sentinel or layer may be chosen after seeing outcomes.

For `P_aug`, the style-nuisance branch is broad approach of the residual field
to the strongest residualized X-free null across F4–F20: failure of the
`+0.02` and positive clustered-LB requirements on the three endpoints,
insufficient positive keys and/or block collapse, with loss of the repaired
common-scale marker as corroborating evidence. Because A-static has now
passed, an A-augmented collapse is specifically the predeclared **static
coordinates incomplete, not state** branch. A repaired retention failure
alone is not a primary-gate collapse and remains interpretive rather than a
kill condition.

## Tier-3 audit #14 — A-static residualization, Round 26, demo (2026-08-29, fresh Codex auditor)

**Adopted corrections.** (1) Mechanical verdict upheld (F4–F20 pass; F0
fails). (2) Not a cosine-geometry mirage: residualization lowers the ridge
cosine at every passing layer (raw 0.65–0.76 → residual 0.56–0.62) while the
lexical nulls fall to ~0.06, so the margin grows because the selected lexical
contrast became easier; shuffled q95 ≤ 0.13 and residual normalized error
0.78–0.83 (vs 1.0) rule out target-shrinkage artifacts. (3) The gate is
"too easy" for a state claim, not for the registered narrow claim: the fair
residual comparator is a fully refitted conditional-randomization (Freedman–
Lane) null preserving nuisance geometry, and a flexible calibration-only
P_aug/lexical interaction field without cell-level X⊥ — both to be
preregistered. (4) **Round 26's "much of the raw lead may have been
presentation-mediated" is an over-read**; a presentation-only cosine gives no
variance share, fraction, mediation or overlap. Replacement for Round 26's
ruling: "P_static took the non-collapse branch. Locally, the result
establishes registered-presentation sensitivity and survival of X-linked
residual predictability after cross-fitted removal of those registered
coordinates. It identifies neither the surviving field as operational state
nor the result as presentation-independent." (5) Full-gate vs positive keys
must be distinguished (F4 7/8, F8 7/8, F12 6/8, F20 8/8 full-gate; misses
family-localized in gloss/association); the four checkpoints are correlated
measurements, not replications. (6) K = 13 KL-rank is coherent for the
residual gate; skill and KL-rank are two reductions of one KL measurement;
in residual mode the `identity` candidate is effectively the presentation-
only completion. (7) F0: "no qualifying conditional gain at F0 under this
instrument" — a genuine negative control. (8) Demo replacements adopted
verbatim and republished.

**Joint license (verbatim):**

The current sentence

> “Much of the raw lead may have been presentation-mediated”

is an over-read.

A `P_static→Δ` cosine of `0.43–0.63` shows that block, length and position metadata predict the **direction** of raw displacement on held-out cells. It does not give:

- variance explained;
- a fraction of the raw ridge advantage;
- mediation;
- an additive decomposition;
- the overlap between what `P_static` and raw `X` predict;
- a causal effect of presentation;
- proof that the coordinates are pure presentation rather than operational context.

The residual arm independently shows that the fitted linear `P_static` nuisance maps do not remove all `X–Δ` association: `X⊥` still predicts `Δ⊥`, and the prediction improves the reassembled decoder law.

The precise joint license is:

> Registered static template coordinates predict held-out raw displacement. After cross-fitted removal of those registered coordinates from both `X` and `Δ`, `X⊥` still predicts `Δ⊥` and its reassembled response-law consequence beyond the registered residual X-free lexical nulls at F4–F20. These facts establish presentation sensitivity and residual X-linked predictability; they do not identify how much of the raw ridge advantage is attributable to presentation or whether the remainder is operational state.

This means “P_static took the non-collapse branch; proves neither state nor presentation-independence” is logically correct but under-states the local positive result. Replace it with:

> `P_static` took the non-collapse branch. Locally, the result establishes registered-presentation sensitivity and survival of X-linked residual predictability after cross-fitted removal of those registered coordinates. It identifies neither the surviving field as operational state nor the result as presentation-independent.

Corresponding “much of the raw lead” and “large component” wording in [Round 26](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/EXPERIMENTS.md:4028>) and the two relevant [NOTEBOOK entries](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/NOTEBOOK.md:41>) should be treated as failing this audit.

The strongest upward correction is equally important: it is now unfair to dismiss the result as exact lookup, class mean, frozen-embedding interpolation, noisy mean estimation or a broken completion pipeline. Those explanations have been directly narrowed. What remains live is a smoother and harder alternative.

**Strongest alternative explanation (verbatim):**

> `P_static` removes only an analyst-chosen linear projection of block identity, token lengths and positions. The residual `X⊥` still contains high-dimensional, nonlinear template/carrier geometry and smooth lexical coordinates shared across held-out blocks and words. `Δ⊥` varies along the same implementation-specific geometry. Ridge and kernel recover that relation, and decoder completion registers it because the same decoder is sensitive to those directions. The result therefore survives every registered lexical null and static residualization without requiring a clean operational state variable.

This account explains:

- residual ridge/kernel success;
- near-zero X-free residual nulls;
- the substantial presentation-only cosine;
- transfer across held-out words and blocks;
- the localized gloss/association failures;
- F0’s different regime;
- one-model specificity;
- reassembled response-law improvement.

It requires neither leakage nor a broken split.

The strongest competing positive explanation is:

> The decoder contains a genuine context-conditioned transition regularity, and presentation is partly legitimate operational state because it changes lawful successors. Quotienting it away indiscriminately removes physics rather than nuisance.

A-static observational residualization cannot choose between these explanations.

**Alternative explorations (verbatim instructions):**

> Add a preregistered residual-geometry null. In every outer block-by-word fold, residualize X and Δ exactly as in the scored arm; permute Δ⊥ across calibration carriers within template family and word; rerun the complete inner selection, refit, and held-out scoring for ridge and kernel. Preserve the crossed class-stratified block-first bootstrap. Gate observed cosine, normalized error, law skill, and continuous KL margin against this fully refitted null distribution.

> Add a fair residual-space X-free comparator. Predict Δ⊥ from a calibration-only nuisance family containing P_static, the predeclared P_aug carrier summaries, frozen lexical embedding, and predeclared low-rank interactions, but no held-out cell X⊥. Match the state field’s tuning discipline and effective-capacity control. Do not use a P-only zero residual as the fair comparator.

> Complete the common-scale decomposition before attributing the raw lead. Score P-only, raw X-ridge, reassembled residual ridge, and every corresponding null against the same raw Δ, the same true law, and the same skill reference. Recompute the strongest-null minimum inside every bootstrap replicate. Report incremental raw-scale squared error or R² and continuous KL improvement; do not infer overlap from two cosines.

> Read A-augmented, B-static, B-augmented, and patched A-static in the locked order without selecting a sentinel, layer, or design after seeing outcomes. Treat A-augmented collapse as “the static coordinates were incomplete, not state.” Treat A-augmented survival as narrowing, not eliminating, the presentation account.

> Run a causal presentation-equivalence test. Construct matched inputs with fixed lexical item and operational task while changing only a predeclared presentation intervention. If the downstream response law and declared moves remain interchangeable, quotient the presentations; if they change reproducibly, treat presentation as operational state. Do not define state by analyst labels alone.

> Freeze a fresh template population and second model family before inspection, repeat the full hierarchy, and then test multi-step composition from F4 through F8 and F12. A one-step field that does not compose is not yet a navigation law.

**Second lens after A-static (verbatim):**

A-static changes the map but does not close the central hole.

Proven locally:

- F0 is an identity/token-dominated transition under this instrument.
- The inherited ordering endpoint is saturated relative to cosine and consequence-sensitive KL measures.
- Registered block/length/position coordinates predict raw forward displacement on held-out cells.
- Those registered coordinates do not exhaust X-linked residual predictability at F4–F20 under the fitted linear nuisance model.

Not proven:

- presentation and operational state are inseparable;
- the residual is presentation-free;
- family-only laws;
- presentation explains a fraction of the raw ridge advantage;
- the field is native to the latent world;
- the field composes;
- another decoder exposes the same structure;
- structured reasoning cannot live here.

The deeper demonstrated deficiency is epistemic and architectural: the denizen is not given a native quotient. The current analysis needs external block identities, token lengths and positions to propose which places should count as equivalent. That is evidence that the representation has not yet **demonstrated** its own navigation coordinates—not proof that such coordinates cannot be recovered.

The constructive requirement is therefore:

> Build a latent space in which lexical, presentation and operational coordinates are exposed or controllably factorable; define sameness by interchangeability under declared moves and downstream response laws; treat presentation as state exactly when controlled presentation changes alter those laws; require consequence-sensitive metrics, multi-step closure, fresh-style and unseen-word transfer, cross-family replication, and explicit precision/support contracts.

No new axiom is earned. The next object is an operationally testable quotient, not a stronger name for regression success.

No tracked repository file was modified and no experiment was run. Eight typed findings were recorded on the blackboard; convergence and synthesis completed with no open signal or dispute.

## Round 27 — A-augmented adjudication and comparator preregistration (2026-08-29)

**Codex, documentation-only; no experiment was run.** The Round 23
three-branch predictions, Rounds 24–26, Tier-3 audit #14, the live augmented
residualization and common-scale code, the permitted `analysis_resAA.json` and
`analysis_resSA.json` artifacts, and the four named ledger entries were checked
directly. The running/queued `analysis_resSB.json`, `analysis_resAB.json`, and
`analysis_resSA2.json` artifacts were not opened. Mechanical reductions below
were recomputed from the JSON rather than copied from the ledger.

### Artifact validity and mechanical score

`resAA` is sentinel A (`.`), `P_aug`, five forward-layer checkpoints, two
unseen-word folds, 20 shuffles, and 500 bootstrap replicates. It completed in
`4737.8 s` of the amended `7200 s` wall. The model/tokenizer pins, fixed K=13
universe, reload record, eight fold keys, and support accounting are present;
support is `1.0` in every key.

The augmented coordinates are cross-fitted as implemented. `carrier_basis`
forms a rank-at-most-4 SVD basis from carrier-mean `X` states on calibration
carriers and calibration words. For each probe-word cell, `design` computes a
carrier mean over the outer calibration-word pool with the current word left
out, then appends that mean's scores in the frozen basis. The basis is rebuilt
on the inner training carriers during nuisance-lambda selection. Thus the
added design columns are rank-4 scores of a leave-current-word-out carrier
summary; the full carrier-mean vector is an intermediate, not an appended
1024-dimensional feature.

The primary residual-vs-null gate adjudicates as follows. Each endpoint margin
is the conservative minimum over the four fixed residualized X-free lexical
nulls, and each lower bound is the crossed class-stratified block-first lower
bound for that minimum-margin comparator.

| Layer | `X_perp` ridge cosine | strongest residual-null cosine | cosine margin [LB] | skill margin [LB] | K=13 KL-rank margin [LB] | full / positive keys | `P_aug -> Delta` cosine | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F0 | 0.335 | -0.006 | +0.341 [+0.246] | +0.156 [+0.022] | +0.303 [+0.122] | 2/8 / 7/8 | 0.639 | pass |
| F4 | 0.617 | 0.062 | +0.555 [+0.513] | +0.458 [+0.310] | +0.485 [+0.352] | 8/8 / 8/8 | 0.498 | pass |
| F8 | 0.595 | 0.074 | +0.521 [+0.480] | +0.346 [+0.239] | +0.432 [+0.312] | 7/8 / 8/8 | 0.446 | pass |
| F12 | 0.555 | 0.060 | +0.495 [+0.455] | +0.369 [+0.254] | +0.425 [+0.238] | 6/8 / 8/8 | 0.475 | pass |
| F20 | 0.612 | 0.071 | +0.541 [+0.494] | +0.457 [+0.377] | +0.557 [+0.471] | 8/8 / 8/8 | 0.608 | pass |

No held-out block collapses at any layer. Thus all five layers pass the
registered residual gate. As in audit #14, full-gate and merely positive keys
are distinct; the five checkpoints are correlated measurements, not
replications. The presentation-only numbers are held-out directional cosines,
not variance shares, fractions, mediation, overlap, or causal effects.

### Round 23 three-branch adjudication

The static-plus-augmented pair takes the predeclared **both-design
non-collapse branch**.

1. **State-linked prediction:** the registered mechanical prediction holds,
   including the mandatory `P_aug` sensitivity: at F4–F20, `X_perp` predicts
   `Delta_perp` and the reassembled response-law consequence beyond the four
   registered residual X-free lexical nulls after either registered
   presentation design is removed. In audit #14's terms the positive object
   is surviving **X-linked residual predictability**, not identified state.
2. **Lexical-interpolation prediction:** the amended tested version misses.
   Class mean, frozen-embedding kNN, frozen-embedding ridge, and
   frozen-embedding kernel do not close the gap. This narrows those fixed
   X-free lexical accounts; it does not eliminate every smooth lexical field
   or presentation-by-lexical interaction.
3. **Carrier/style-nuisance prediction:** its registered collapse prediction
   misses for both `P_static` and `P_aug`. The presentation concern is thereby
   narrowed, not eliminated. A finite rank-4 carrier-summary design does not
   remove high-dimensional nonlinear template/carrier geometry, smooth
   lexical coordinates, or their interactions from `X_perp`.

The audit #14 joint license therefore extends to the augmented coordinates:

> Registered static template coordinates and registered augmented
> carrier-summary coordinates predict held-out raw displacement. After
> cross-fitted removal of either registered design from both `X` and `Delta`,
> `X_perp` still predicts `Delta_perp` and its reassembled response-law
> consequence beyond the registered residual X-free lexical nulls at
> F4–F20. These facts establish presentation sensitivity and residual X-linked
> predictability; they do not identify how much of the raw ridge advantage is
> attributable to presentation, whether the remainder is operational state,
> or whether the result is presentation-independent.

This is stronger than A-static alone because the mandatory augmented
sensitivity also survives. It is still observational evidence in one decoder
and one authored template population, not a native-law or state verdict.

### F0 ruling

F0's augmented pass is a **real conditional gain for the predeclared
residualized estimand**, not a held-out-target or carrier-mean leakage artifact.
The augmented carrier summary is X-only and cross-fitted; the residual field
and every X-free null receive the same fitted presentation displacement on
reassembly. Its residual normalized error is `0.923` versus approximately
`1.0` for the residual null, and the maximum ridge shuffled q95 is `0.122`.

It is not a rescue of the raw F0 field or a refutation of identity/token
dominance. On the same folds, un-residualized ridge cosine is `0.687` versus a
strongest raw-null cosine near `0.669`, while `P_aug -> Delta` is `0.639`.
The carrier-mean/subspace scores absorb a carrier-level, lexically averaged
component of `X` that is entangled with the identity-dominated first move;
subtracting its fitted image changes the target and reference geometry and
exposes a smaller conditional residual problem. Only `2/8` keys clear the
full per-key gate although `7/8` have the correct sign. The admissible
statement is therefore:

> Under the registered A-augmented residualization, F0 has a positive pooled
> conditional residual gain and no block collapse, but the strict key-level
> evidence is sparse. The raw first transition remains identity/token
> dominated; the result does not identify the exposed residual as operational
> state.

### Repaired common-scale retention

The same-run common-scale block is present and finite. It reassembles each
residual field and residual null to full `Delta`, scores raw and residual arms
against the same raw `Delta`, true decoder law, and `unres_mean` skill
reference, uses continuous KL improvement, and recomputes the strongest-null
minimum inside each bootstrap replicate. Direct readings of
`pairs[*]["retention_common_scale_block_first"]` are:

| Layer | cosine ratio median [95% CI] | skill ratio median [95% CI] | continuous-KL-margin ratio median [95% CI] |
|---|---:|---:|---:|
| F0 | 2.513 [1.337, 5.098] | 1.460 [0.871, 4.559] | 1.255 [0.612, 5.393] |
| F4 | 1.105 [0.968, 1.323] | 1.062 [0.811, 1.758] | 0.778 [0.495, 1.195] |
| F8 | 1.166 [1.044, 1.326] | 1.137 [0.792, 1.618] | 0.942 [0.652, 1.249] |
| F12 | 1.231 [1.102, 1.428] | 1.160 [0.957, 1.353] | 1.127 [0.996, 1.243] |
| F20 | 1.105 [1.020, 1.248] | 0.951 [0.787, 1.073] | 0.974 [0.784, 1.098] |

Every ratio median exceeds the predeclared `0.5` threshold. Fourteen of the
fifteen interval lower bounds also exceed `0.5`; F4's continuous-KL lower
bound is `0.495`. The licensed retention statement is:

> Under the amended common-scale marker, A-augmented retains at least half of
> the same-run raw ridge-versus-strongest-null margin at the bootstrap median
> on cosine, law skill, and continuous KL improvement at every measured
> layer. A uniform 95%-interval claim across all layer-endpoint cells is not
> earned because the F4 continuous-KL interval narrowly crosses `0.5`.

This is retention of a declared **predictive margin on a common scale**, not
a fraction of latent “signal,” variance, state, or mediation. It does not
substitute for the still-missing A-static common-scale cell in `resSA2`.

### Comparator 1 lock: fully refitted Freedman–Lane residual-geometry null

This is preregistered now for the existing captures and must not be tuned from
the remaining chain's outcomes.

- Scope: all four sentinel-by-design cells (`A/B x P_static/P_aug`), five
  layers, the same two unseen-word folds, eight outer block-by-word keys,
  support rules, and K=13 capture/law contract. F0 is reported separately and
  cannot supply the F4–F20 state-reading criterion.
- In every outer key, fit the nuisance coordinates and maps exactly as in the
  scored arm. Hold `X_perp`, the held-out `Delta_perp`, and the held-out law
  fixed. For each of 20 deterministic-seed null refits, permute calibration
  `Delta_perp` across carriers **within template family and word**. Rerun the
  complete calibration-only inner selection, refit ridge and kernel on the
  permuted residual targets, and score the unchanged held-out cells. No fit or
  prediction may be reused across permutations except the already frozen
  nuisance residualization and true-law cache.
  *Clarification (Tier-1 review of the implementation, 2026-08-29, applied by
  Claude at Codex's recommendation before any run): deterministic X-only
  preprocessing and factorizations — the calibration standardizers of X⊥ and
  the eigendecomposition of the centred standardized X⊥ Gram — may be reused
  across refits because they do not depend on the permuted targets; every
  target-dependent quantity (target means, cross-products, weights, kernel
  coefficients, predictions, inner-selection scores) is recomputed per refit.
  The exact test is layer-level: the 20 refit statistics are pooled
  block-balanced by aligned refit index across the eight keys, and the
  one-sided exact p is (1 + #refits not beaten)/21; per-key p-values are
  diagnostics. Observed and null statistics share one common cell mask per
  key (support reported as `fl_null_support`; a key below 0.95 is incomplete).*
- Preserve the class-stratified crossed bootstrap: one word draw per word-fold
  key shared across sampled blocks, carriers resampled within block. Report
  every per-key statistic and the block-first pool.
- Four primary statistics, all predeclared: displacement cosine (higher),
  normalized residual error (lower), response-law skill on the reassembled
  successor (higher), and continuous KL improvement over the same fixed
  residual X-free reference (higher). Ridge is primary; kernel is a mandatory
  sensitivity reported under the same rule.
- A layer passes the null only if the observed ridge is more extreme than all
  20 fully refitted nulls on all four statistics (one-sided exact
  `p <= 1/21`), the crossed block-first 95% interval for observed minus the
  permutation median is positive on cosine, skill, and continuous KL and
  negative on normalized error, cosine/skill/error improvements over the
  permutation median are each at least `0.02`, at least `6/8` keys have the
  correct sign on all four statistics, no block collapses, and support is at
  least `0.95`. Continuous KL requires only a positive margin and lower bound,
  not an arbitrary `0.02`-nat threshold. A cell-level positive requires at
  least two of F4/F8/F12/F20; F0 is a diagnostic.

With one residualized run costing `4400–4740 s`, 20 complete refits per
fold-key conservatively project to `88,000–94,800 s` (`24.4–26.3 h`) per
sentinel/design cell and `97.8–105.3 CPU h` for all four. The hard budget is
`30 h` per cell and `120 CPU h` total, one CPU process at a time, no new
capture and no GPU. An overrun or fewer than 20 complete refits in any key is
budget-incomplete, not a pass.

Predictions are discriminating. A target-shrinkage or residual-geometry
artifact predicts that the refitted null approaches the observed residual
field and blocks at least one of the four statistics. A real aligned
`X_perp–Delta_perp` relation predicts rejection of this null. That rejection
still does not identify state: a nonlinear presentation/carrier relation can
be aligned and therefore also reject permutation.

### Comparator 2 lock: registered residual-space X-free field

This is the cheaper and more interpretation-specific moot-maker. In every
outer fold, form a calibration-only feature family with no held-out cell
`X_perp`:

- the ten registered `P_static` columns;
- the same rank-at-most-4 leave-current-word-out carrier-summary scores used
  by `P_aug`;
- the first 16 principal scores of the frozen input embedding, with the basis
  fit on calibration words only; and
- the fixed 4-by-16 carrier-summary/lexical-score outer products (64
  interaction columns).

The rank `16` and the interaction form are fixed now. Standardizers, lexical
bases, carrier bases, and all interaction columns are rebuilt in each outer
fold and each inner training fold. Fit a multivariate ridge map to
`Delta_perp`; choose lambda from the state field's existing grid by the same
inner leave-one-calibration-block-out criterion, with the class-stratified
calibration-word split retained. Fit the cell-level `X_perp` ridge under the
same inner procedure. Report each fit's effective ridge degrees of freedom
`df = tr[Z (Z^T Z + lambda I)^-1 Z^T]`. For the mandatory matched-capacity
sensitivity, take the comparator's calibration-selected `df` as the target
and choose the state ridge lambda from the same frozen grid whose
calibration-design `df` is closest, breaking ties toward smaller `df`; this
matching uses no held-out target. No held-out outcome chooses features, rank,
lambda, capacity match, or which fit is reported.

A layer passes this comparator only if the cell-level `X_perp` ridge beats the
X-free field by at least `0.02` with a positive crossed block-first lower bound
on displacement cosine, normalized-error improvement, and response-law skill,
has a positive continuous-KL improvement with positive lower bound, has at
least `6/8` all-four positive keys, no block collapse, and support at least
`0.95`. The same result must hold in the degrees-of-freedom-matched
sensitivity. A sentinel/design cell requires at least two qualifying layers
among F4/F8/F12/F20; F0 remains separate.

A comparator-only reanalysis is estimated at approximately one present
residualized-run cost per cell: `4400–4740 s`, or `4.9–5.3 CPU h` for four
cells. Allow a `7200 s` hard wall per cell and `8 CPU h` total for interaction
construction, law completion, and bootstrap overhead. Run one CPU process at
a time; no new capture and no GPU.

The strongest nuisance prediction is that this field closes F0 and much of
F4–F12, showing that smooth carrier/presentation-by-lexical structure does not
require held-out cell `X_perp`. The context-conditioned prediction is that the
cell-level field retains the registered margins, perhaps attenuated. A mixed
sentinel/layer result is reported as specificity and earns no general state
reading.

### Locked order and decision boundary

Do not interrupt or inspect the remaining chain. Finish and adjudicate in the
existing order `resSB` (B-static) -> `resAB` (B-augmented) -> `resSA2`
(patched A-static common-scale cell). Then run the registered X-free field on all
four cells as the cheapest direct moot-maker, followed by the fully refitted
Freedman–Lane null on all four cells. Both comparator families precede any
second-model capture or state wording; no sentinel, layer, or presentation
design may be selected after outcomes. Only after the two comparators and the
four-cell common-scale synthesis are adjudicated may the pinned second-family
protocol begin. A comparator collapse closes the state-reading branch but
remains a transferable nuisance-law result.

### Second lens after A-augmented

Proven locally now:

- registered static and augmented presentation/carrier-summary coordinates
  predict held-out raw displacement direction;
- after cross-fitted removal of either registered design, X-linked residual
  predictability survives at F4–F20 beyond the registered X-free lexical
  nulls and improves the reassembled response law;
- the raw F0 transition remains identity/token dominated, although `P_aug`
  exposes a pooled conditional residual gain with weak full-key coverage; and
- the inherited ordering endpoint remains saturated relative to cosine and
  consequence-sensitive KL measurements.

Still unproven: presentation and operational state are inseparable; the
residual is presentation-free; the field is intrinsic rather than a smooth
implementation-specific carrier/lexical relation; the field composes;
another decoder exposes it; or structured reasoning cannot live here.
Presentation sensitivity is proven; **presentation entangled with state is
not**.

The next latent space must expose or controllably factor lexical,
presentation, and operational coordinates to its denizen; define sameness by
interchangeability under declared moves and downstream response laws; treat
presentation as state exactly when controlled changes alter those laws;
provide consequence-sensitive divergences and multi-step closure; transfer
across unseen words, fresh styles, and model families; and include precision,
support, and controllability in the representation contract.

The single most sharpening immediate measurement is the **registered residual-space
X-free presentation/lexical interaction field**. It directly tests the
strongest surviving explanation on the same held-out task without cell-level
`X_perp`, and it is the cheapest comparator that can moot the state reading.
No new axiom is earned, so `theory/AXIOMS.md` is unchanged.

## Round 28 — B-static adjudication and two-sentinel static ruling (2026-08-29)

**Codex, documentation-only; no experiment was run.** The Round 23
predictions, Rounds 24–27, Tier-3 audit #14, `analysis_resSB.json`, the
permitted `analysis_resSA.json` and `analysis_resAA.json` comparison artifacts,
and ledger entry `nlm007_resid_resSB` were checked directly. Every mechanical
number below was reduced from the JSON. The running/queued
`analysis_resAB.json` and `analysis_resSA2.json` artifacts were not opened, and
the uncommitted analyzer change under separate review was not modified.

### Artifact validity and mechanical score

`resSB` is sentinel B (`,`), `P_static`, five forward-layer checkpoints, two
unseen-word folds, 20 shuffles, and 500 bootstrap replicates. It completed in
`4598.4 s` of the amended `7200 s` wall. The model/tokenizer pins, fixed K=13
universe, reload record, eight block-by-word keys, and support accounting are
present; support is `1.0` in every key. The locality maximum is `0.125`, and
the reload record gives maximum log-probability difference `0.0078125`, KL
ordering agreement `0.999651`, and maximum pairwise-KL difference `0.001323`.

Each endpoint margin below is the conservative minimum over the four fixed
residualized X-free lexical nulls. Each lower bound belongs to that endpoint's
minimum-margin comparator.

| Layer | `X_perp` ridge cosine | strongest residual-null cosine | cosine margin [LB] | skill margin [LB] | K=13 KL-rank margin [LB] | full / positive keys | `P_static -> Delta` cosine | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F0 | 0.268 | 0.001 | +0.267 [+0.050] | -7.465 [-26.697] | +0.113 [-0.286] | 4/8 / 4/8 | 0.628 | fail |
| F4 | 0.558 | 0.061 | +0.497 [+0.438] | +0.349 [+0.224] | +0.396 [+0.251] | 4/8 / 8/8 | 0.508 | pass |
| F8 | 0.564 | 0.075 | +0.489 [+0.442] | +0.361 [+0.295] | +0.419 [+0.339] | 7/8 / 8/8 | 0.413 | pass |
| F12 | 0.517 | 0.064 | +0.453 [+0.409] | +0.405 [+0.318] | +0.492 [+0.434] | 8/8 / 8/8 | 0.444 | pass |
| F20 | 0.578 | 0.089 | +0.489 [+0.440] | +0.415 [+0.335] | +0.577 [+0.498] | 8/8 / 8/8 | 0.622 | pass |

No held-out block collapses at F4–F20. Thus F4/F8/F12/F20 pass the amended
residual-vs-null gate and F0 fails. At each passing layer all eight keys have
the correct point sign; strict full-gate coverage is nevertheless only `4/8`
at F4. Full-gate and point-positive counts remain distinct, and the four
passing checkpoints are correlated measurements rather than replications.
The presentation-only values are held-out directional cosines, not variance
shares, fractions, mediation, overlap, or causal effects.

### Round 23 prediction scorecard and the static-pair license

For the second sentinel, the registered state-linked-side mechanical
prediction holds in its bounded form: F0 remains non-qualifying, while
F4–F20 retain positive residual-field margins and reassembled response-law
improvements beyond the strongest of all four residual X-free lexical nulls.
The tested lexical-interpolation account again misses: class mean,
frozen-embedding kNN, frozen-embedding ridge, and frozen-embedding kernel do
not close the B-static gap. The registered `P_static` style-nuisance collapse
prediction also misses. This does not adjudicate B-augmented or eliminate a
nonlinear presentation-by-lexical field.

A-static and B-static therefore take the same `P_static` non-collapse branch.
Audit #14's joint license can now be stated across the registered sentinel
pair:

> Across both registered punctuation sentinels in this decoder and template
> population, static block, length, and position coordinates predict the
> direction of held-out raw displacement. After cross-fitted removal of those
> registered coordinates from both `X` and `Delta`, `X_perp` still predicts
> `Delta_perp` and its reassembled response-law consequence beyond the four
> registered residual X-free lexical nulls at F4–F20. These facts establish
> two-sentinel registered-presentation sensitivity and residual X-linked
> predictability; they do not identify how much of the raw ridge advantage is
> attributable to presentation, whether the remainder is operational state,
> or whether the result is presentation-independent.

The sentinel pair is a robustness check within one decoder, move, authored
template population, and shared set of folds. It is not two independent
replications and earns no intrinsic/native-law, composition, fresh-style,
cross-family, or general structured-reasoning claim.

### F0: a locally degenerate skill ratio, not a uniform `-7.5` effect

The B-static F0 skill deserves a narrower reading than its pooled value. The
eight fold-key ridge skills are:

- gloss: `-0.042/-0.048`;
- continuation: `+0.298/+0.307`;
- association: `-30.093/-30.304`; and
- grammar: `+0.083/+0.079`.

Their arithmetic average is the reported `-7.465`. The fold-mean residual
mean-law KL values themselves remain finite (`0.607–1.155`), so the artifact
does not show a globally zero pooled reference. Instead, the combination of
moderate pooled KL, extreme association-fold skill, and the very wide
block-first interval is the signature of **local cellwise denominator
ill-conditioning** in the normalized skill: some association cells make the
mean-law reference too close to truth for the ratio to be a stable magnitude.
The JSON stores fold reductions rather than every cell denominator, so the
claim is local degeneracy of the normalized statistic, not a count of exactly
which cells have a near-zero reference.

Consequently, `-7.5` must not be compared as an effect size across sentinels or
presentation designs, and it does not mean every B-static F0 key is
catastrophically wrong. But it also does not rescue F0: only `4/8` keys are
full-gate or point-positive, both gloss and association collapse at the block
level, and the KL-rank lower bound is negative. The correct layer ruling is
still **no qualifying conditional gain at B-static F0 under this instrument**.
Across cells, F0 remains cell-specific: A-static fails, A-augmented has the
already-adjudicated sparse conditional pass, B-static fails, and B-augmented
is unread. Do not pool those distinct residual estimands into one F0 verdict;
read the joint gate and absolute/continuous KL evidence beside normalized
skill.

### Repaired common-scale retention for B-static

The same-run `retention_common_scale_block_first` field is present and finite.
Direct residual/raw strongest-null predictive-margin ratios are:

| Layer | cosine ratio median [95% CI] | skill ratio median [95% CI] | continuous-KL-margin ratio median [95% CI] |
|---|---:|---:|---:|
| F0 | -1.800 [-14.927, 3.578] | -23.428 [-94.863, 6.386] | -3.682 [-17.529, 14.834] |
| F4 | 1.237 [1.051, 1.628] | 1.234 [0.806, 2.445] | 0.780 [0.426, 1.125] |
| F8 | 1.221 [1.092, 1.453] | 1.158 [0.864, 1.464] | 1.111 [0.741, 1.436] |
| F12 | 1.251 [1.106, 1.535] | 1.139 [0.884, 1.349] | 1.102 [0.835, 1.332] |
| F20 | 1.261 [1.118, 1.488] | 1.053 [0.954, 1.224] | 1.033 [0.959, 1.163] |

Every F4–F20 ratio median exceeds the predeclared `0.5` threshold. Eleven of
the twelve interval lower bounds also exceed `0.5`; F4 continuous KL is the
exception at `0.426`. The licensed statement is therefore:

> Under the amended common-scale marker, B-static retains at least half of the
> same-run raw ridge-versus-strongest-null predictive margin at the bootstrap
> median on cosine, law skill, and continuous KL improvement at F4–F20. A
> uniform 95%-interval statement across all passing layer-endpoint cells is
> not earned because the F4 continuous-KL interval crosses `0.5`.

This is a predictive-margin statement on a common raw-`Delta` scale, not a
fraction of latent signal, variance, state, or mediation. It does not license
a joint A-static/B-static retention claim: the original `resSA` artifact lacks
this repaired field, and only `resSA2` can fill that static cell.

### Sentinel asymmetries

The strict full/point-positive key counts for A-static versus B-static are:

| Layer | A-static | B-static |
|---|---:|---:|
| F0 | 2/8 / 6/8 | 4/8 / 4/8 |
| F4 | 7/8 / 7/8 | 4/8 / 8/8 |
| F8 | 7/8 / 8/8 | 7/8 / 8/8 |
| F12 | 6/8 / 8/8 | 8/8 / 8/8 |
| F20 | 8/8 / 8/8 | 8/8 / 8/8 |

B-static is consistently lower in residual ridge cosine than A-static at
F4–F20, by about `0.03–0.06`, and its F4 strict coverage is materially weaker:
`4/8` versus `7/8`. The missing B F4 intervals are spread across gloss,
association, and grammar rather than forming a point-sign or block collapse.
Conversely, B has `8/8` full keys at F12 where A has `6/8`. The presentation-
only directions are similar across the static sentinels. These are useful
specificity facts, but no layer verdict differs and no favorable sentinel or
layer is selected.

### Second lens: what the sentinel pair adds

The second sentinel makes an A-specific punctuation accident less plausible.
Proven locally now under `P_static`: registered presentation coordinates
predict raw displacement direction for both sentinels; those registered
coordinates do not exhaust X-linked residual predictability at F4–F20 for
either sentinel; and the raw F0 transition remains identity/token dominated,
with B-static also failing its consequence gate. The previously proven
ordering-readout saturation is unchanged.

No new hostile structural hole is proved by this replication. In particular,
presentation and operational state remain unproven as inseparable; the
residual is not shown to be presentation-free; family-only laws, failure of
composition, and inability of structured reasoning to live in the space are
not established. The stronger two-sentinel fact is still an architectural and
epistemic deficiency: the denizen has not been given a demonstrated native
quotient, while the analyst must supply block, length, and position
coordinates to propose equivalence. That is a constructive requirement for
the next latent space, not proof that no recoverable quotient exists.

The next space should expose or controllably factor lexical, presentation,
and operational coordinates; define sameness by interchangeability under
declared moves and downstream response laws; treat presentation as state only
when controlled changes alter those laws; and support consequence-sensitive
multi-step closure with fresh-style, unseen-word, and cross-family transfer.
No new axiom is earned.

### Locked order after B-static

The Round 27 order is confirmed without amendment. Finish and adjudicate
`resAB` (B-augmented), then `resSA2` (patched A-static common-scale cell).
Next run the registered residual-space X-free field on all four cells as the cheapest
direct moot-maker, followed by the fully refitted Freedman–Lane null on all
four cells. Both comparator families precede the pinned second-model-family
protocol. No sentinel, layer, or presentation design is selected after
outcomes.

## Tier-3 audit #15 — A-augmented validity, tunnel vision, and the second lens (fresh Codex auditor)

**Adopted verdict.** The A-augmented numerical association is real under the
implemented analysis, but several interpretations are withdrawn or narrowed.
The aggregate residual-vs-null gate is mechanically met at all five correlated
checkpoints, and no direct held-out `Y` or `Delta` leakage was found. The
maximum defensible statement is:

> In sentinel A and this fixed authored template population, the registered
> static nuisance model and the implemented rank-4 carrier-summary nuisance
> model do not absorb the X–Delta association at F4–F20. On the same
> correlated folds, `X_perp` predicts `Delta_perp` and improves the reassembled
> decoder-law prediction beyond four registered X-free lexical predictors.
> This is an outcome-clean but transductive, same-data sensitivity result; it
> establishes neither operational state, presentation independence,
> fresh-style transfer, composition, nor a native law.

The current queue is strongly tunnel-visioned: one decoder, one authored
template population, one punctuation-append move, one sentinel pair, and one
decoder self-readout. No representation-level hole hostile to structured
reasoning is established by the residualizations. The specific inherited
ordering statistic is a proven local measurement hole.

### Direct numerical verification

The audit independently reduced `analysis_resAA.json` and confirmed the Round
27 table:

| Layer | Ridge / strongest null cosine | Cosine margin [LB] | Skill margin [LB] | KL-rank margin [LB] | Full / positive keys |
|---|---:|---:|---:|---:|---:|
| F0 | 0.335 / -0.006 | +0.341 [0.246] | +0.156 [0.022] | +0.303 [0.122] | 2/8 / 7/8 |
| F4 | 0.617 / 0.062 | +0.555 [0.513] | +0.458 [0.310] | +0.485 [0.352] | 8/8 / 8/8 |
| F8 | 0.595 / 0.074 | +0.521 [0.480] | +0.346 [0.239] | +0.432 [0.312] | 7/8 / 8/8 |
| F12 | 0.555 / 0.060 | +0.495 [0.455] | +0.369 [0.254] | +0.425 [0.238] | 6/8 / 8/8 |
| F20 | 0.612 / 0.071 | +0.541 [0.494] | +0.457 [0.377] | +0.557 [0.471] | 8/8 / 8/8 |

Runtime was `4737.8 / 7200 s`; support was `1.0` in every key; the run used
eight block-by-word keys but only four top-level template-family clusters, 20
shuffles, and 500 bootstrap replicates. Reload ordering agreement was
`0.9996166`, with maximum pairwise-KL difference `0.001494`. The stored
float16 locality maximum was `0.125`; the authoritative float32 manifest value
was `0.0003624`, narrowly within the historical `0.000378` tolerance.

The common-scale ratios were also copied correctly:

| Layer | Cosine | Skill | Continuous KL |
|---|---|---|---|
| F0 | 2.513 [1.337, 5.098] | 1.460 [0.871, 4.559] | 1.255 [0.612, 5.393] |
| F4 | 1.105 [0.968, 1.323] | 1.062 [0.811, 1.758] | 0.778 [0.495, 1.195] |
| F8 | 1.166 [1.044, 1.326] | 1.137 [0.792, 1.618] | 0.942 [0.652, 1.249] |
| F12 | 1.231 [1.102, 1.428] | 1.160 [0.957, 1.353] | 1.127 [0.996, 1.243] |
| F20 | 1.105 [1.020, 1.248] | 0.951 [0.787, 1.073] | 0.974 [0.784, 1.098] |

Fourteen of fifteen interval lower bounds exceed `0.5`; F4 continuous KL is
`0.495`.

### A-aug design integrity and the Round 23 lock

The implementation is outcome-clean: the basis, nuisance maps, and
regularization use calibration data; the current calibration word is excluded
where applicable; inner nuisance selection rebuilds the basis on inner
training carriers; and held-out `Y`, `Delta`, law outcomes, and bootstrap
results do not construct the features. Direct held-out-target leakage is not
supported.

It is nevertheless transductive. Every held-out word on a held-out carrier
receives carrier-summary scores made from that same carrier's `X` states over
the calibration-word pool. The target outcome is held out, but the carrier is
not an unseen input distribution. “Held-out block” therefore means held-out
outcomes, not a wholly unseen carrier or presentation.

The pre-result Round 23 text is not ultimately neutral between the two
readings. It says that `P_aug` “adds two carrier-level coordinates” and then
lists (1) the carrier mean of `X` and (2) rank-4 scores of that mean. Its
literal meaning was **full carrier mean plus rank-4 scores**. The code appended
only `CM @ V`, at most four score columns; the 1024-dimensional carrier mean
was an intermediate. Therefore:

- name the observed run `P_aug-score4`;
- mark A-aug **outcome-clean, transductive, contract-validity qualified**;
- treat it as internally valid for the implemented score-only sensitivity,
  not as the literal registered `P_aug` result; and
- record the literal full-mean-plus-score `P_aug-full` sensitivity as unrun.

### Claim adjudications

- **Both-design non-collapse:** mechanically correct only for the implemented
  nested designs. `P_static` and `P_aug-score4` reuse the same sentinel-A cells
  and folds. This is one experiment surviving two nested nuisance fits, not
  two demonstrations.
- **Joint license:** asymmetric. `P_static` is analyst-metadata-only;
  `P_aug-score4` includes X-derived carrier information and cannot
  independently establish presentation sensitivity. Registered static
  metadata predict raw displacement. Adding four transductive, X-derived
  carrier-summary scores still does not absorb the residual X–Delta
  association. Operational state, unmeasured presentation, a carrier/prefix
  fingerprint, and mixtures all remain live.
- **F0:** no direct target leakage was found, but this is only a positive
  pooled association for the registered residualized estimand. Raw ridge
  exceeds the raw null by about `0.019` cosine, only `2/8` keys clear the full
  gate, and the skill lower bound is `0.022`. Raw F0 remains identity/token
  dominated.
- **Common-scale retention:** the bootstrap-median ratio exceeds `0.5` in
  every cell. “Retains at least half” is withdrawn because numerator and
  denominator come from different fitted estimator systems and strongest-null
  competitions. Ratios above one do not show strengthening, recovery, or more
  latent information. The field is a robustness ratio for predictive margins.
- **Layers and endpoints:** the five checkpoints share decoder, captures,
  folds, move, sentinel, nuisance fits, and null family. They are a correlated
  depth profile, not five replications. Cosine/error and skill/KL reductions
  are likewise related views of the same vector prediction and decoder-law
  discrepancy. Four template families make the block-first intervals internal
  sensitivity summaries, not style-population confidence intervals.

### Strongest missed alternative

The strongest alternative missed by both queued comparators is a
high-dimensional prefix/carrier fingerprint. `X_perp` and `Delta_perp` are
deterministic descendants of the same authored input, and the decoder readout
is sensitive to the same implementation-specific directions. This account is
aligned enough to beat a Freedman–Lane null, cell-level enough to beat an
X-free field, compatible with unseen-word transfer on a smooth authored
manifold, and compatible with decoder-law improvement. It is not a causally
sufficient reusable operational state, and it is not a navigation law unless
it survives intervention and composition.

### CPU-only alternatives

| Exploration | Estimated cost | What it decides |
|---|---:|---|
| Full available carrier-summary rank ladder `{1,2,4,8,full}` plus nonlinear carrier kernel | Cosine screen in minutes; full law gate about 1–1.5 h/cell | Whether rank-4 truncation created the apparent survival |
| Contextual X-free baseline from full tokenized prefix features or an earlier frozen prefix representation | Under 30 min screen; about 1–1.5 h with completion/bootstrap | Whether the four word-only lexical nulls are simply too coarse |
| Fresh frozen template population, 16 templates × 80 words | About 5 min capture/sentinel; roughly 1–3 h focused analysis | Whether the relation transfers beyond the authored template manifold |
| Different move: content-bearing append, negation/operator insertion, or binding update | About 5 min capture/move plus 1–2 h analysis | Whether this is punctuation-position mechanics |
| Matched presentation-interchangeability test | Roughly 10 min capture plus ≤1 h targeted scoring | Whether presentation variants are genuinely interchangeable under moves and laws |
| Two-step writeback/composition | Roughly 10–20 min new capture plus about 1 h scoring | Whether the one-step field composes or is merely a local fingerprint |
| Second pinned decoder | Existing lock allows about 3 CPU h | Replication across one more decoder—not genericity or mechanism |

The fresh-template, different-move, and full-rank carrier baseline outrank the
approximately 100-CPU-hour four-cell Freedman–Lane expansion. The refitted
null remains useful for residual geometry, but it is not the next dollar of
CPU.

### Over-claimed kills and second-lens ruling

- **Ordering:** only the across-word, within-carrier pairwise-KL statistic is
  insensitive. Ordering-sensitive navigation, move order, commutativity, and
  multi-step ordinal structure remain open.
- **Lexical interpolation:** only the four registered word-only estimators
  fail. Contextual lexical structure, full-prefix representations, subword
  interactions, and high-dimensional lexical/presentation fields remain open.
- **Frozen encoder:** stopping the prior assay was defensible scope
  management. It closes the tested candidate/edit/readout envelope, not frozen
  encoders, native maps, or operational quotients generally.

A representation-level hole hostile to structured reasoning is not yet
proven. Proven locally: the inherited ordering statistic is insensitive; raw
F0 is identity/token dominated under this move and instrument; static metadata
predict raw displacement; and two fitted nuisance families do not exhaust the
X-linked association. Not proven: presentation and operational state are
structurally inseparable, the space lacks every recoverable quotient, the
residual is presentation-free, or structured reasoning cannot live here.

The cheapest representation-level hole test is controlled interchangeability:
freeze semantically and operationally equivalent presentation variants, match
their displacement norms without held-out outcomes, swap or write back their
states or moves, and test whether downstream moves and laws remain
interchangeable while genuinely different operational states remain
separable. Failure across a held-out presentation family and a second move,
with operational controls intact, would be direct evidence that presentation
obstructs a stable identity relation. Observational residualization cannot
establish that.

## Round 29 — contract repair and external-axis steering

**Codex, documentation-only; no experiment was run.** The excluded
`analysis_resAB.json` and `analysis_resSA2.json` artifacts were not opened, and
the uncommitted analyzer diff under separate Tier-1 review was not touched.
Audit #15 changes the order before any Freedman–Lane run.

### Steering adjudication

The rank ladder is **not redundant** with the registered X-free field. The
rank ladder changes the nuisance family used to residualize both `X` and
`Delta` and asks whether rank-4 underfitting manufactured the remainder; it
also repairs the literal Round 23 full-mean contract. The X-free field leaves
the scored residual geometry fixed and asks whether a calibration-only field
without cell-level `X_perp` can predict `Delta_perp`. They can close different
failure modes.

The fresh template population **precedes** the four armed X-free cells. It is
cheaper (`~3 h` focused analysis versus `4.9–5.3 h`) and is the first test on
an external axis. Existing arming is an operational convenience, not a reason
to postpone the more informative measurement.

The interchangeability test is designed **now**, before its templates or
second-move outcomes exist. The 16 fresh templates will be authored as eight
blind matched presentation pairs so the fresh-population capture also supplies
the intervention test. This avoids outcome-conditioned definitions of
“equivalent presentation.”

The single measurement capable of making the state/navigation line moot
fastest is the matched interchangeability test across a frozen fresh
presentation family and a second move. If equivalent variants systematically
fail swap/writeback while genuinely different operational controls remain
separable, the current observational regression ladder becomes
characterization of a presentation-bound fingerprint, not a route to a stable
operational quotient. Its incremental targeted score is at most about one CPU
hour once the two captures exist.

### Amended fixed order and budgets

| Order | Work | Expected CPU | Hard wall / launch rule |
|---:|---|---:|---|
| 0 | Finish `resAB -> resSA2` without early inspection | existing budgets | Do not alter or open the excluded artifacts early |
| 1 | Carrier-summary rank/contract probe on existing captures | cosine screen in minutes; one full-law A cell `~1.5 h` | `2 h`; fixed full-law cell, no screen-selected promotion |
| 2 | Freeze and capture 16 fresh templates × 80 words for both sentinels; capture the fixed second move | about `10 min` fresh capture plus `5–10 min` second-move capture | Hash texts/items/config before capture; one CPU process |
| 3 | Matched presentation-interchangeability score across both move classes | `<=1 h` targeted scoring | `90 min`; no full analysis outcome may redefine pairs or controls |
| 4 | Full fresh-population analysis under the same contract | about `3 h` | `4 h`; both sentinels, no favorable-sentinel selection |
| 5 | Full different-move analysis | `2–3 h` | `4 h`; one fixed move, no replacement after scoring |
| 6 | Registered X-free field on all four existing cells | `4.9–5.3 h` | existing `8 h` total wall and gate; retained even if its role becomes nuisance characterization |
| 7 | Freedman–Lane on `A-static` only, conditionally | `24.4–26.3 h` | existing `30 h` cell wall; launch only if every prior state-reading gate remains live |
| 8 | Second pinned decoder | about `3 h` | after the one-cell refitted null; replication only |

No four-cell Freedman–Lane expansion is authorized by Round 29. The single
`A-static` cell launches only if: the fixed full-mean contract cell retains at
least two F4–F20 passes; the fresh population and different move each retain
their gates; interchangeability does not meet the hostile-hole criterion; and
the A-static registered X-free comparison does not close the cell-level field.
If any condition fails, the refitted null is not the highest-leverage next
measurement.

### Probe 1 — carrier-summary rank ladder and literal contract repair

On existing captures, fit `P_aug-score-r` for fixed ranks `{1,2,4,8,full}`,
where `full` means every estimable calibration-carrier direction, plus one
nonlinear kernel on the calibration-word carrier mean. Rebuild every basis,
standardizer, and nuisance fit inside the existing outer and inner folds. No
held-out `Y`, `Delta`, law, screen score, or bootstrap result may construct a
feature or choose regularization.

The rank/kernel pass is an exploratory held-out displacement-cosine screen
only. It reports residual ridge cosine, strongest residual X-free-null cosine,
normalized error, support, and the full rank-response curve for every layer
and sentinel/design cell; it cannot earn a law or state claim. Independently
of that screen, the one preselected full-law cell is sentinel A with the
literal Round 23 `P_aug-full` design: append the full leave-calibration-word-
pool carrier mean **and** the rank-4 scores. It uses the existing K=13,
crossed block-first, support, reload/locality, completion, and common-scale
contract at F0/F4/F8/F12/F20.

**Gate.** Rank-4 truncation remains an adequate explanation if `P_aug-full`
leaves fewer than two qualifying layers among F4–F20 under the existing
three-endpoint residual-vs-null gate, or if any apparent pass is closed by the
strongest residual X-free null with margin `<0.02`. Two or more full-law
passes with at least `6/8` all-endpoint-positive keys, no block collapse,
support `>=0.95`, and the existing positive crossed lower bounds rule out only
this fixed truncation account. F0 remains separate.

**Predictions.** The nuisance-underfit account predicts monotone absorption as
rank rises and a `P_aug-full` collapse at F4–F20. The context/fingerprint
accounts predict survival at at least two layers, possibly with attenuation;
survival does not distinguish them.

### Probe 2 — fresh frozen template population

Before any capture or new score, author 16 new templates blind to all result
files, organized as four semantic/operational families and eight matched
presentation pairs. Freeze the literal texts, pair map, family labels, the
same 80-word list, both sentinels, tokenization checks, and hashes. No template
may be edited or dropped after capture except a predeclared protocol failure,
which voids the population rather than triggering substitution.

Run the same unseen-word folds, K=13 candidate universe, `P_static` residual
contract, four X-free lexical nulls, crossed class-preserving bootstrap,
support/reload/locality checks, and three endpoints. This is a fresh-population
replication of the relation, not a zero-shot coefficient-transfer claim; a
frozen old-to-fresh coefficient score may be reported only as a secondary
diagnostic.

**Gate.** A sentinel qualifies only if at least two of F4/F8/F12/F20 clear the
existing residual-vs-null point and crossed-lower-bound gates, at least `12/16`
fresh templates have the correct sign on cosine, skill, and continuous KL, no
one four-template family collapses, and support is `>=0.95`. The external-axis
result is positive only if both sentinels qualify. F0 is reported separately.

**Predictions.** A reusable operational relation predicts two-sentinel
qualification with breadth across all four fresh families. A smooth
prefix/carrier fingerprint predicts collapse, family concentration, or marked
attenuation. A pass narrows template-population specificity but does not prove
genericity or state; a fail caps all current positive wording to the original
authored manifold.

### Probe 3 — fixed different move

The second move is fixed now as insertion of tokenizer-single-token `not`
immediately before the registered word slot. `X` is the original word-slot
state and `Y` is the aligned word-token state after insertion; `Delta=Y-X`.
If the pinned tokenizer does not encode the operator as exactly one token in
every frozen carrier, the probe is void and no replacement is chosen after
outcomes. Capture uses the same 80 words and frozen fresh matched population.

Analyze F0/F4/F8/F12/F20 with the same unseen-word split, identity-plus-shared-
displacement and word-conditioned moot-makers, four X-free lexical nulls,
K=13 response-law endpoints, crossed block-first bootstrap, support,
reload/locality, and no post-score layer selection.

**Gate.** The move qualifies only if at least two F4–F20 layers beat the
strongest X-free lexical null by `>=0.02` with positive crossed lower bounds on
displacement cosine, response-law skill, and continuous KL improvement, at
least `6/8` fold keys are jointly positive, no family collapses, and support
is `>=0.95`. F0 is separate.

**Predictions.** A punctuation-position fingerprint predicts no qualifying
operator-insertion layer or family-local collapse. A more reusable
context-conditioned relation predicts at least two qualifying layers, perhaps
at different depths. Either outcome remains one move in one decoder and does
not establish composition.

### Probe 4 — matched presentation interchangeability

Use the eight fresh matched pairs, both punctuation sentinels, and the fixed
operator insertion. Match displacement norms using calibration words only.
For each held-out word and direction of each pair, write the calibrated,
norm-matched move from presentation variant A into variant B's state and vice
versa, then complete the actual decoder law. Fixed-input repeat completions
measure the numerical noise floor. Negative controls swap a matched-norm move
from a genuinely different operational condition; pair and control mappings
are frozen before capture.

Primary quantities are normalized successor error and continuous-KL
degradation relative to the same-presentation held-out move. Intervals resample
matched pair, word, and family; the two move classes are reported separately
and jointly. No latent-scorer quality claim is made.

**Stable-interchangeability gate.** At at least two F4–F20 layers in each move
class, the upper 95% bound on equivalent-presentation swap degradation must be
`<= max(0.02, 2 × fixed-input q99 noise)` on both primary quantities, while
the lower 95% bound for different-operational-state degradation must exceed
that equivalence bound by `>=0.02`; at least `6/8` matched pairs must agree and
no family may reverse the separation.

**Hostile-hole gate.** At at least two F4–F20 layers in **both** move classes,
equivalent-presentation swaps must degrade both primary quantities by
`>=0.02` with positive lower bounds in at least `6/8` pairs and every family,
while different operational controls remain separable above the numerical
floor. This directly establishes a presentation-obstructed identity relation
for the frozen population and moves. Anything between the stable and hostile
gates is inconclusive.

**Predictions.** A recoverable operational quotient predicts stable
interchangeability for equivalent presentations and separation for different
operational states. The prefix/carrier-fingerprint account predicts systematic
equivalent-pair swap failure, especially across the fresh families and second
move. Passing the hostile-hole gate makes further observational residualizer
refinement non-decisive for state and pivots the constructive program toward a
latent space with explicit quotient coordinates.

### Second-lens conclusion

Round 29 does not add an axiom or claim a representation-level hole. It makes
the denizen's required identity test operational: two presentations count as
the same place only if declared moves and downstream laws are interchangeable,
while genuinely different operational states remain distinguishable. The
queue now asks that question before spending roughly 100 CPU hours refining a
single observational axis. `theory/AXIOMS.md` remains unchanged.

## Round 30 — probe-1 Tier-1 review and probes 2–4 design gate

**Codex, documentation-only; no experiment was run.** The excluded running or
queued result artifacts were not opened. The uncommitted
`experiments/analyze_lm_dynamics.py` diff was reviewed but not edited. This
round fixes the capture, completion, fold, null, and interchangeability
contracts before implementation or scoring.

### Part 1 — Tier-1 review of the Round 29 probe-1 diff

**Verdict: NOT-READY.** The calibration-only construction is outcome-clean,
and the literal `P_aug-full` ridge is mathematically regularized, but two
execution/correctness defects block the rank screen and one contract ambiguity
blocks the nonlinear arm. Two additional numerical/provenance repairs are
required before the literal full-mean law cell.

1. **Fatal zero-shuffle screen path.** `--screen` sets `n_shuffle=0`, but fold
   serialization unconditionally evaluates `np.percentile(v, 95)` for each
   empty shuffled-null list. The first completed fold therefore raises before
   an artifact can be written. The retention diagnostic also enters a
   zero-bootstrap calculation and catches its own empty-percentile error,
   leaving an avoidable error payload. Minimal fix: when `n_shuffle==0`, emit
   `null`, `n=0`, and no quantile; when `n_boot==0`, emit point estimates only
   and do not enter any retention, LOCO, style-null, or pooled-bootstrap path.
   Add a tiny no-completion fixture that reaches JSON serialization with
   `n_boot=n_shuffle=0`.

2. **Rank 8 is not fold-locally estimable in the inner loop.** An outer
   leave-one-family-out fold has 12 calibration carriers, hence at most 11
   centered carrier-mean directions. Each inner leave-one-calibration-family-
   out fold has 8 carriers, hence at most 7. The `full` branch estimates this
   numerical rank, but a fixed `--aug-rank 8` slices eight inner SVD rows and
   includes the centered null/non-identifiable direction. Minimal fix: compute
   `r_est` in every outer and inner basis and use
   `r_used=min(r_requested,r_est)`; `full` means exactly `r_est`. Record
   requested rank, fold-local realized rank, tolerance, singular values (or
   their extrema), and retained standardized column count. The rank-8 outer
   fit may remain rank 8 while its inner tuning folds are rank 7; that fact is
   part of the estimand, not a reason to reuse an outer basis inside inner
   validation.

3. **The nonlinear flag does not by itself select the registered carrier-mean
   kernel.** `--aug-kernel` switches the nuisance regressor to RBF kernel ridge
   on whatever augmented design the other flags happen to create. By default
   that is `P_static + score4`, with no 1024-dimensional carrier mean, whereas
   Round 29 registers one nonlinear arm on the calibration-word carrier mean.
   Minimal fix: either make `--aug-kernel` imply `--aug-full-mean` and fixed
   rank 4, or fail unless they are supplied together. Freeze the nonlinear arm
   as RBF kernel ridge on literal `P_aug-full = P_static + score4 + full mean`,
   with `(gamma, lambda)` selected only inside the calibration folds.

4. **The full-mean ridge is well-posed in mathematics but under-instrumented
   numerically.** Appending 1024 mean coordinates creates a wide, collinear
   nuisance design. `Standardizer` removes zero-variance columns and every
   registered ridge `lambda` is strictly positive, so neither linear ridge nor
   kernel ridge is singular or non-unique in exact arithmetic. The present
   primal `X^T X` eigensolve is float32, however, and does not report effective
   rank, retained columns, negative roundoff eigenvalues, or the smallest
   regularized denominator. Minimal fix: use a float64 primal solve or a
   sample-space dual/SVD solve for the wide design, clamp only roundoff-negative
   eigenvalues under a declared tolerance, assert finite predictions for every
   grid point, and store the diagnostics. The positive lambda grid handles
   rank deficiency; it does not by itself audit float32 spectral error.

5. **The flag-off numerical path is unchanged, but exact artifact identity is
   not.** With all four new flags off, rank 4, linear `RidgeFamily`, basis
   placement, inner selection, predictions, shuffles, completion, and
   bootstraps take the prior numerical path. A process already running is also
   unaffected by later source edits after import. A queued flag-off process
   will, however, add `n_design_cols` and `carrier_rank` to every residualized
   fold, so the output schema is not byte-identical. Minimal fix: emit new
   probe-1 metadata only when a probe-1 flag is active, or version the schema
   explicitly. Run a frozen-fixture equality test over selected parameters,
   predictions, gates, and support for the all-flags-off path.

6. **`--screen` does not lock its own registered run shape or supply a single
   screen summary.** It currently relies on later assertions and can be mixed
   with unrelated modes; the intended ranks, layers, source, target, word
   split, and point-only outputs are not fail-fast enforced. Minimal fix:
   require `--source forward --target delta --unseen-words 2 --residualize aug
   --pairs 0 1 2 3 4`, reject `--loco`, `--style-null`, `--baselines`,
   `--identity-*`, `--xfree-field`, and `--fl-null`, and write one explicit
   `screen_summary` per F0/F4/F8/F12/F20 with residual ridge cosine, each of the
   four residual X-free-null cosines and their maximum, margin, normalized
   error, support, requested/realized rank, and fold counts. CIs and law fields
   are absent, not NaN evidence.

**Leakage ruling.** No held-out `Y`, `Delta`, law, screen score, bootstrap, or
completion output enters a basis, standardizer, hyperparameter choice, or
nuisance fit. Outer bases and nuisance targets use calibration carriers and
calibration words; inner bases and fits are rebuilt on the inner training
families. A held-out carrier's design does use that carrier's `X` states on the
outer calibration-word pool, with the current calibration word left out when
applicable. This is the registered outcome-clean, within-carrier transductive
summary, not target leakage and not inductive transfer to a wholly unseen
carrier. The artifact and every claim must retain that qualifier.

After fixes 1–6, the fixed screen consists of linear ranks
`{1,2,4,8,full}` and the one literal-full-mean kernel arm. The preselected law
cell remains sentinel A, linear `--aug-rank 4 --aug-full-mean`, completion on,
with the existing F0/F4/F8/F12/F20 K=13 and crossed-gate contract. No screen
outcome may select that cell or change its regularization grid.

### Part 2 — probes 2 and 3 capture and analysis design lock

#### Frozen population and provenance

Use the committed derived config
`experiments/config/lexical_probe_fresh_v1.json`; do not create a new runner
or alter any template, item, pair, control, family, sentinel, or move after
capture. The current self-declared `frozen_sha256` is not the raw file SHA-256
(the live raw file hash at review time is
`12c724015218bedf58644d0fcbbf5eef68f4db3bd1f16a9977f42007aec2fd06`).
Before capture, provenance must therefore define rather than guess the hash
domain. Every capture manifest stores:

- `config_sha256_raw`, the bytes actually read; `config_git_blob` and the
  containing commit; and the config's `frozen_sha256` separately as
  `config_declared_sha256`;
- canonical hashes over ordered item strings, ordered `(name, block,
  template, pair)` rows, the presentation-pair map, and the operational-
  control map;
- model and tokenizer revision, layer count, embedding/vocabulary dimensions,
  library versions, device, compute/storage dtype, thread count, batch size,
  command, elapsed time, array filename/hash, array shapes, and the exact
  token IDs and per-probe positions described below.

Add `--expected-config-sha256` to `capture_forward` and fail before model work
if the raw bytes differ. The literal expected value is recorded in the launch
ledger after the capture implementation commit; it is not taken from the
self-referential field.

#### One capture path, three prospective artifacts

Extend `experiments/run_lm_dynamics.py::capture_forward`; add no capture
script. Preserve the current sentinel path when the new flag is absent. The
new flags are:

- `--insert-before-slot TOKEN`: mutually exclusive with `--sentinel`; require
  exactly one non-special tokenizer token and insert its embedding row
  immediately before the independently inserted word row;
- `--repeat-null`: repeat the identical moved-sequence batches once and store
  calibration-cell state/law numerical-noise arrays rather than only maxima;
- `--expected-config-sha256 HASH`: the pre-model raw-config guard above.

`--tag` remains the artifact identifier. The three fixed calls are sentinel A,
sentinel B, and `--insert-before-slot " not" --tag NOT`, all on the same fresh
config, one CPU process at a time. The operator must encode as the already
frozen single token ID 537; any mismatch, special-token status, word-slot
split, array misalignment, or config/hash mismatch voids the whole prospective
population. There is no replacement operator.

For the two sentinel captures, retain the existing arrays and meanings:
`H_q_unappended` is `X` at the original last suffix position,
`H_sent` is `Y` at the appended sentinel position, and `law_sent` is the true
moved-sequence law at that sentinel readout. Retain `H_slot`, `H_last`,
`law_last`, and `law_q_unappended` as diagnostics. Add per-probe source/readout
positions and `repeat_target_nerr` / `repeat_readout_kl` arrays when
`--repeat-null` is active.

For insertion capture, write `insert_states_NOT.npz` with these exact arrays:

- `H_word_original[P,L+1,N,D]`: original-sequence word-slot state `X`;
- `H_word_moved[P,L+1,N,D]`: the same word token after the inserted token,
  aligned by token identity rather than absolute index, and used as `Y`;
- `law_word_original[P,N,V]` and `law_word_moved[P,N,V]`: laws read at the
  original and moved word positions; the latter is the true response law;
- `law_last_moved[P,N,V]`, secondary only; `slot_original[P]`,
  `slot_moved[P]`, `sequence_len_original[P]`, and
  `sequence_len_moved[P]`;
- `items`, `pos`, `probes`, and `blocks` in frozen order; and, under
  `--repeat-null`, `repeat_target_nerr[P,L+1,N]` plus
  `repeat_readout_kl[P,N]`.

The matching `insert_manifest_NOT.json` additionally stores `move_kind =
insert_before_slot`, operator text/ID, `source_alignment = word_token`, every
original/moved slot, the prefix/suffix token IDs, and two controls: all states
strictly before the insertion are identical in float32, and layer-0 original
versus moved word embeddings are identical. Because this architecture has no
absolute position added to the embedding row, insertion `F0` has `Delta=0` by
construction; displacement cosine and normalized-error denominators there are
undefined. Report F0 as a structural alignment/null check, never as a passing
or failing move layer.

#### Analyzer source and completion semantics

Add `forward_insert` to `--source` and add `--move-tag` (required for that
source; fixed to `NOT` here). The regular two punctuation analyses continue to
use `--source forward --sentinel-tag A|B`. All three use `--target delta
--unseen-words 2 --residualize static --pairs 0 1 2 3 4`; insertion F0 is
retained only as the structural null above.

For insertion at layer index `l`, set

`X = H_word_original[:,l]`, `Y = H_word_moved[:,l]`, and `Delta = Y - X`.

Extend `WorldCompleter.laws` with an `insert_before_slot_emb` argument. It must
rebuild the **moved** sequence `prefix + inserted-token + word + suffix`, set
the replacement/readout position to `original_slot + 1`, and install
`Yhat = X + Delta_hat` there. Hidden index `l` is written by a forward hook on
decoder block `l-1`; for `l=0`, replace the moved sequence's embedding row
directly. Read logits at the moved word position. With `Yhat=None`, the
unmodified moved sequence supplies `q_true`; the stored
`law_word_moved` is only the reload target. Never complete the original
sequence, write at the old absolute slot, or compare against an original-
sequence law. Under `P_static` residualization, reassemble
`Delta_hat = f_Delta(P_static) + Delta_perp_hat` before writeback.

The primary response-law quantities are `KL(q_true || qhat)`, skill relative
to the same-fold shared-displacement completion, and continuous improvement
`KL(null)-KL(candidate)`. The probe-3 three-endpoint gate uses displacement
cosine, response-law skill margin, and continuous-KL margin. Last-suffix laws
are diagnostic and cannot satisfy a gate.

#### Folds, `P_static`, nulls, K=13, and inference

The primary fold contract remains the one implied by Round 29's frozen gates:
four leave-one-family-out outer carrier folds (`question`, `instruction`,
`comparison`, `enumeration`) crossed with two class-stratified unseen-word
folds. Each key has 12 calibration carriers × 40 calibration words and four
held-family carriers × 40 held-out words; nuisance bases, standardizers, and
all hyperparameters are rebuilt in the inner leave-one-of-three-calibration-
families-out loop. This yields the registered eight fold keys. The config's
directional `calibration_blocks = {question,instruction}` and
`heldout_blocks = {comparison,enumeration}` is also reported once as a
secondary frozen external-split diagnostic; it does not replace the fourfold
primary, enter model selection, or count toward the `6/8` or `12/16` gates.

For the sentinel move, fresh `P_static` has ten coordinates in this fixed
order: four family indicators, centered over all 16 carriers, then
`[prefix_token_count, suffix_token_count, moved_sequence_length,
word_slot_position, sentinel_readout_position,
sentinel_readout_position/moved_sequence_length]`, all zero-based positions.
For insertion use the same schema and family centering with
`[prefix_token_count, suffix_token_count, moved_sequence_length,
original_word_slot_position, moved_word_slot_position,
moved_word_slot_position/moved_sequence_length]`; the manifest records both
slots. No text embedding, hidden state, result, pair label, or outcome is a
`P_static` coordinate.

The cheapest forward-move nulls carry over without reinterpretation:

1. identity, `Delta_hat=0`;
2. identity plus the calibration shared displacement,
   `Yhat=X+mean(Delta_cal)`;
3. POS-class mean residual displacement;
4. frozen-input-embedding word-only cosine kNN;
5. frozen-input-embedding word-only ridge; and
6. frozen-input-embedding word-only RBF kernel ridge.

Items 3–6 are the four registered X-free lexical nulls; all their tuning is
nested inside calibration words/families. The fixed unseen-word response-law
rank universe is exactly K=13:
`identity`, `mean`, `class_mean`, `wordonly_knn`,
`wordonly_ridge_emb`, `wordonly_kernel_emb`, `knn1`, `knn5`, `knn20`,
`ridge`, `lowrank`, `kernel`, and `chart`. The word-conditioned and shared-
displacement arms are moot-makers, not evidence that an X-conditioned field
exists.

Support requires finite displacement cosine/error and finite primary law
values on one common K=13 cell mask. Reload compares fresh float32 moved laws
with stored float16 laws; sentinel locality keeps the existing causal check,
and insertion locality checks every pre-insertion position. The block-first
bootstrap resamples families first, templates within the sampled held family,
and POS-stratified held-out words with one word draw crossed across every
sampled family carrying that word-fold key. Gates remain exactly Round 29:
two qualifying F4–F20 layers; positive point and crossed lower-bound margins
against the strongest of the four X-free lexical nulls on all three endpoints;
`6/8` all-endpoint-positive keys; no family collapse; support `>=0.95`.
Probe 2 additionally requires both sentinels and `12/16` template breadth.

### Part 3 — probe-4 matched-interchangeability design lock

Implement this as an early-return `--interchangeability` mode in
`experiments/analyze_lm_dynamics.py`, not as another runner. New flags are
`--interchangeability`, `--append-tags A B`, `--insert-tag NOT`, and
`--repeat-completions 2`. In this mode, reject ladder/residualization/null
flags other than the existing `--n-boot` and `--tag`; require the frozen fresh
config, all three capture/manifest hashes, F4/F8/F12/F20, and at least two
identical completion passes. The fixed output is
`analysis_interchangeability_fresh_v1.json`.

For move source `m`, layer `l`, word `w`, and carrier/presentation `p`, define
`d[p,w] = Y[p,w]-X[p,w]` from the appropriate aligned capture. For every
ordered direction donor `a` to recipient `b` in a frozen matched pair, compute
one positive scalar using **calibration words only**:

`alpha[a->b] = sqrt(sum_w ||d[b,w]||^2 / sum_w ||d[a,w]||^2)`.

The sums use the current outer word-fold's calibration words; scales are
separate by move source, sentinel, layer, pair, and direction, with no clipping
or held-out norm. A zero/non-finite denominator voids that layer/source. The
swapped target is `Yswap[b,w] = X[b,w] + alpha[a->b] d[a,w]`. Repeat in the
reverse direction. Apply the identical rule in both directions of every
frozen `operational_controls.control_pairs` pair; those are the only negative
controls and may not be rematched after capture.

Write `Yswap[b,w]` into carrier `b`'s **own moved sequence** at its own target
readout position. Punctuation uses the appended sequence and sentinel position;
insertion uses the inserted-token sequence and aligned moved word position.
The hidden-index rule is the same as probe 3: hook block `l-1`, or replace the
embedding row at `l=0` (which is not gated). The true target is `Y[b,w]`; the
true law is the unmodified recipient moved sequence's law at that readout.
The same-presentation reference writes stored `Y[b,w]` through the identical
hook and sequence, so hook/reload error is shared rather than silently treated
as interchangeability failure.

For each held-out cell, the two primary degradations are:

- `D_state = ||Yswap-Y|| / ||Y-X|| - ||Ysame-Y|| / ||Y-X||`;
- `D_KL = KL(q_true || q_swap) - KL(q_true || q_same)` in nats.

Zero move norms are unsupported. No cosine, pairwise ordering, automated
quality score, or last-token readout can replace either primary quantity.
Run the identical moved-sequence capture twice on calibration cells to obtain
state-repeat normalized errors, and run identical `Ysame` completions
`--repeat-completions` times at every gated hook for law-repeat KL differences.
For each source/layer/endpoint, set
`tau = max(0.02, 2 * calibration fixed-input q99 noise)`. Store the complete
noise distribution summaries, q99, `tau`, and supported counts; never estimate
the threshold from held-out swaps.

Aggregate the two directions of a matched pair as one pair cluster. The
bootstrap is family-first: resample four families, resample the two frozen
matched pairs within each sampled family, and use one POS-stratified held-out-
word draw crossed across all sampled pairs for a word-fold key. Directions
remain paired. Operational-control intervals resample the four frozen control
pairs with the same crossed word draws; report each target family separately
so a control-rich family cannot hide a reversal. Punctuation A and B are
reported separately; a punctuation layer qualifies only when both sentinels
meet the clause, so no sentinel averaging or selection is allowed. Operator
insertion is the second move class. Joint summaries never replace either
class.

**Stable-interchangeability gate.** A layer passes for one source only if, on
both `D_state` and `D_KL`, the equivalent-swap upper 95% bound is `<=tau`, the
different-operational-control lower 95% bound is at least the equivalent upper
bound plus `0.02`, at least `6/8` matched pairs satisfy the point criterion in
both directions, and every family has equivalent degradation `<=tau` with
positive control separation. The punctuation clause must pass for A and B at
the same layer. Stable interchangeability requires at least two common
F4–F20 punctuation layers and at least two F4–F20 insertion layers.

**Hostile-hole gate.** A layer passes for one source only if the equivalent-
swap point degradation is `>=0.02` and its lower 95% bound is positive on both
primary quantities; at least `6/8` matched pairs meet both conditions in both
directions; every family's point degradation is `>=0.02` with a positive
family lower bound; and every frozen different-operational control remains
above its endpoint's `tau` with a positive lower bound. Again A and B must
pass at the same punctuation layer. The hostile-hole verdict requires at least
two F4–F20 layers in both move classes. Anything else is inconclusive.

The insertion is called the fixed **`not`-token insertion**, not semantic
negation: across the frozen noun, verb, adjective, and function-word cells it
is not guaranteed to implement one coherent linguistic operation. The two
gates concern interchangeability of these measured moves and response laws in
this decoder/population only. Stable passage would support a local operational
quotient; hostile passage would establish a presentation-obstructed identity
relation in scope and pivot construction toward exposed quotient coordinates.
Neither licenses a generic reasoning-quality or model-family claim.

### Round 30 disposition

Probe 1 remains ahead of capture but is **NOT-READY** until the six minimal
repairs above receive a new Tier-1 review. Probes 2–4 now have a design lock,
not an implementation or result. After a repaired probe-1 diff, the order from
Round 29 is unchanged. No axiom or empirical claim is added.

## Tier-3 audit #16 — B-static wording, fresh-population void, and bridge requirement (fresh Codex auditor)

### Verdict

Round 28's B-static numerical result survives direct verification, but its interpretation needs tightening:

> Across the two correlated punctuation sentinels in one decoder and one authored population, registered block/length/position metadata predict raw displacement direction, and `X_perp` retains predictive association with `Delta_perp` after those metadata are cross-fitted out at F4–F20. This is sensitivity to registered presentation-derived metadata plus surviving X-linked residual predictability under `P_static`; it is not independent replication, a presentation decomposition, operational state, or a native law.

The fresh population does not survive design audit. Zero of its eight pairs establishes presentation-only equivalence across all four declared word classes. It is voided before capture for confirmatory probes 2–4 and retained unchanged only as a design negative or exploratory mixed-frame stress set.

The B-aug crashes are a first-class numerical-instrument finding. They are not a kill of B-aug, but neither are they harmless execution noise.

No representation-level hole hostile to structured reasoning is established. The Round 30 probe-4 design could at most test raw interchangeability up to scalar rescaling; it misses recoverable presentation-conditioned chart transitions.

The audit did not open `analysis_resAB.json` or the working analyzer.

### B-static wording corrections

- F4–F20 pass the registered residual-versus-null gate.
- All passing layers have `8/8` point-positive keys and no family collapse.
- B-static weakens an A-sentinel-only accident, but it remains a correlated same-population robustness check.
- The two sentinels share decoder, word inventory, authored templates, folds, analysis, and move class. Calling B-static a replication or saying it "reproduces" A-static is too strong.

The direct `P_static → Delta` cosines are `0.628/0.508/0.413/0.444/0.622` at F0/F4/F8/F12/F20. These are not a "presentation-only arm." `P_static` contains registered block, length, and position metadata. The result shows predictability from those presentation-derived static coordinates—not pure presentation, variance explained, mediation, causal attribution, or a presentation fraction.

The F0 sentence is:

> B-static F0 is non-qualifying; its extreme pooled skill is driven by two association keys and is consistent with unstable cellwise normalization, so the magnitude is not a comparable effect size.

That normalization diagnosis remains an inference because the artifact does not expose every cell denominator.

The retention sentence is:

> All twelve F4–F20 ratio medians exceed 0.5, but no uniform interval claim is earned; these are ratios between fitted predictive-margin systems, not fractions of retained signal, variance, state, or mediation.

### B-aug crashes: numerical-instrument non-robustness

> The registered B-aug low-rank path is numerically non-robust in the F8 grammar block: torch SVD of the fitted coefficient matrix failed to converge twice. The failure is not a B-aug result and does not prove ill-conditioned `X_perp`; localization requires finite-input checks, effective rank, singular-spectrum or condition telemetry, and backend-sensitivity validation.

The first crash row's "cause unknown" wording is superseded. A fallback result cannot silently erase the finding. It must retain:

- the original two failures;
- the failed fold and matrix;
- finite-input and spectral diagnostics;
- agreement where both backends converge; and
- explicit amended-implementation status for the failed cell.

### Fresh-population linguistic audit

| Pair | Noun | Verb | Adjective | Function word | All-class verdict |
|---|---|---|---|---|---|
| `q_pair1` | Modality and definiteness change: "does X" → "could the X" | Bare mention/coercion → definite nominal | Nominalization/coercion changes | Mostly ill-formed or nominalized | Reject |
| `q_pair2` | Slot coerces noun toward a verb | Closest clean case; "on earth" changes pragmatic intensity only | Coerced/ill-formed in a verb slot | Ill-formed in a verb slot | Reject all-class; verb-only exploratory |
| `i_pair1` | Bare object → definite object; politeness also added | Nominalized rather than verb role | Nominalized/ill-formed | Ill-formed | Reject |
| `i_pair2` | Coerced into verb position | `should` → `really ought to` changes deontic force | Coerced/ill-formed | Ill-formed | Reject |
| `c_pair1` | Bare/mentioned noun → definite noun; comparison frame also changes | Nominalized/coerced | Nominalized/coerced | Ill-formed | Reject |
| `c_pair2` | Coerced degree reading | Ill-formed/coerced | `far` changes comparison magnitude | Ill-formed | Reject |
| `e_pair1` | Definiteness changes | Imperative/event reading → definite nominal | Ill-formed/coerced | Ill-formed | Reject |
| `e_pair2` | `some` and `would` change quantification/modality | Nominalized/coerced | Nominalized/coerced | Ill-formed | Reject |

A linguist would reject all eight as all-four-class matched-presentation pairs. `q_pair2` is the sole structurally plausible edit, but only for verb cells; rescuing that subset after outcomes would be exploratory.

The four controls are genuinely different frames, but not clean "same register, different operation" controls across all classes. They change speech act, syntax, length, category licensing, syntactic role, frame shape, or surface distance. They can serve only as coarse far-difference controls and cannot attribute separation specifically to operational state.

"Authored blind" is withdrawn. The licensed provenance statement is:

> The population was prospectively authored and committed before any new capture or score.

No direct new-template outcome conditioning is demonstrated, but the author knew prior results and responded to audit #15. The config's declared digest also differs from its raw SHA-256.

### Void ruling and 12-point replacement rule

The current v1 population is void before capture for confirmatory probes 2–4. The frozen file remains unchanged. It may remain a permanent negative design artifact or an explicitly exploratory mixed-grammaticality stress population whose results cannot earn presentation or hostile-hole claims.

The replacement rule is registered before replacement texts are authored:

1. Create an entirely new version; never mutate or relabel v1.
2. Require pre-capture approval of all `8 pairs × 4 POS classes = 32` cells.
3. Within every cell, preserve syntactic dependency, category licensing, polarity, modality, definiteness, quantification, degree, tense, speech act, and continuation demand.
4. Both members must be independently judged grammatical and operationally equivalent; "equally malformed" does not count.
5. Presentation edits must be limited to predeclared non-truth-conditional discourse, register, or orthographic variation outside the slot's dependency frame.
6. Controls must match register, grammaticality, approximate token length, and surface-edit magnitude while changing one declared operation.
7. Use an outcome-blind author and a separate linguistic adversary who have no access to new-template model behavior.
8. Freeze texts, pair maps, cell-level linguistic judgments, controls, tokenization, raw file hash, Git blob, gates, and commands before capture.
9. Any failed linguistic or tokenization cell voids the entire candidate version before model work; generate a new version from scratch rather than substituting a template.
10. Predeclare a scalar bridge and a calibration-only diagonal/low-rank/orthogonal bridge ladder. A hostile-hole verdict is unavailable unless the simple bridge ladder also fails.
11. Require the hostile equivalent-swap lower bound to exceed the endpoint-specific numerical threshold `tau`, not merely zero.
12. Predeclare a near-zero move-norm floor before computing normalized degradation.

### Mandatory caveats if v1 probe 4 is run anyway

> The frozen v1 "presentation pairs" were not linguistically validated as operationally equivalent across noun, verb, adjective, and function-word cells; several change syntactic licensing, modality, definiteness, degree, or quantification.

> Accordingly, this probe measures interchangeability under a mixed presentation-and-operation perturbation, not presentation-only interchangeability.

> A swap failure cannot be attributed to presentation and cannot establish a presentation-obstructed identity relation or hostile quotient hole.

> The operational controls are coarse different-frame controls confounded by grammaticality, syntactic role, length, and surface distance; their separation does not isolate operational difference.

> The scalar RMS bridge tests equality up to rescaling only; failure remains compatible with a recoverable presentation-conditioned chart transition.

> Even a stable pass would establish only robustness across these mixed frames under two local moves and one-position readouts, not a presentation quotient.

> All pair-by-POS cells are reported without post-outcome rescue; any noun-only, verb-only, pair-only, family-only, sentinel-only, or layer-only interpretation is exploratory.

> The reported intervals are internal sensitivity summaries over four authored families, not confidence intervals for a population of presentation families.

### Round 29–30 design validity

What was good:

- Moving fresh populations, a second move, and interchangeability ahead of a roughly 100-CPU-hour four-cell permutation expansion was correct.
- Round 30 correctly ruled probe 1 NOT-READY.
- The insertion alignment and F0 structural-null interpretation are explicit.
- Round 30 correctly stopped calling the `not` insertion semantic negation.
- It caught the raw-hash mismatch and wide-design numerical risks.

Blocking defects:

1. **The linguistic design gate happened after freezing.** The population was frozen before its claimed equivalence relation was adversarially checked.
2. **Probe 4 assumes too much coordinate identity.** It maps donor moves with one positive scalar. A latent world may use presentation-conditioned gauges connected by a simple chart-transition map. Raw swaps can fail while a stable operational quotient remains recoverable.
3. **The hostile gate is not guaranteed to beat measured noise.** Stable passage uses `tau=max(0.02,2×q99 noise)`, but hostile passage requires only a point degradation `≥0.02` and a lower bound above zero. If `tau>0.02`, a "hostile" result can remain inside the declared noise region.
4. **Normalized successor error has a near-zero-denominator risk.** Zero moves are excluded, but arbitrarily small moves can still inflate the ratio.
5. **The controls do not isolate operation.** They are gross frame changes with POS-dependent grammaticality.
6. **Four families do not support population-style inference.** Family-first bootstrap bounds remain internal authored-family stability summaries.
7. **Probe 3 is not one coherent operation.** Inserting `not` before nouns, verbs, adjectives, and function words creates different or malformed constructions. Round 30's wording caveat is correct, but the result cannot serve as a clean second operational move.
8. **Fresh-population "replication" is too strong.** It would refit within one decoder on the same 80 words and related synthetic slot-filling paradigm. Call it a fresh authored-population stress test.

Correct order now:

- Finish the protected running chain without opening excluded artifacts.
- Record and audit the B-aug numerical amendment.
- Repair and re-review probe 1.
- Void fresh v1 before capture.
- Register and freeze a linguistically valid v2 plus the bridge/noise repairs.
- Run the contextual-prefix baseline and bridge screen before full interchangeability.
- Only then retain the remainder of the Round 29 external-axis order.

### Alternatives and tunnel-vision ruling

The program remains one decoder, one 80-word inventory, synthetic slot filling, one-position self-readout, one-step interventions, and analyst-authored equivalence. The strongest mechanism-level alternative is a recoverable presentation-conditioned gauge; the strongest immediate confound is full-prefix syntactic licensing.

| Exploration | CPU-only estimate | What it decides |
|---|---:|---|
| Tokenized full-prefix/syntactic-licensing X-free screen | 10–30 min; about 1 h with completion | Whether surface grammar and slot licensing explain the relation |
| Calibration-only scalar → diagonal → low-rank/orthogonal bridge ladder | 15–30 min state screen; ≤1 h with writeback | Whether raw swap failure is repaired by a chart-transition map |
| Well-formed operational micro-world with binding, polarity, or quantifier updates | 15–30 min capture; 2–4 h analysis | Whether a coherent operation supports transfer across presentations |
| Multi-position teacher-forced consequence law, next `k={4,8}` positions | 20–40 min capture; 1–2 h scoring | Whether one-position KL is hiding or manufacturing consequence |
| Two-step move composition/writeback | 10–20 min capture; about 1 h scoring | Whether the field composes rather than acting as a local fingerprint |
| Second pinned decoder | About 3 h | Decoder specificity only—not mechanism or genericity |

### Second-lens ruling

A correctly designed interchangeability test is the right kind of denizen question: two presentations count as the same place only if declared moves and response laws survive transport between them while genuinely different operational states remain distinct.

It can prove, locally:

- that one declared identity relation is stable or obstructed;
- that raw presentation variants do or do not share measured moves and local laws; and
- that a candidate next latent space needs explicit quotient or bridge coordinates.

It cannot prove:

- that no nonlinear or gauge-transformed quotient exists;
- that structured reasoning cannot live in the representation;
- that donor writebacks are on-manifold;
- that the relation composes across multiple moves;
- that a one-position decoder law is the correct consequence currency; or
- model-family, task-family, or generic reasoning claims.

With the v1 pairs, it cannot prove even the local presentation-obstruction claim. The pair premise fails before the latent measurement begins.

## Round 31 — audit #16 adoption, valid-population authoring, and corrected external-axis order

**Codex, documentation/configuration only; no experiment was run.** `analysis_resAB.json` was not opened, and the uncommitted analyzer diff under separate Tier-1 review was not read or edited. Audit #16 is adopted above. Fresh v1 is void for confirmatory probes 2–4 and remains unchanged as an exploratory mixed-frame stress set. Its provenance is "prospectively authored and committed before any new capture or score," not "authored blind."

### Think-before-any-future-run

The expected first outcome is that v2 survives an independent all-cell linguistic adversary and tokenization audit; failure voids the entire version, while approval only makes it eligible for capture. If the contextual-prefix X-free field closes the X-linked association, full-prefix grammar/licensing is the cheapest explanation. If the association survives, that confound is narrowed but operational state remains unidentified. If a calibration-only bridge repairs raw swaps, the space has recoverable presentation-conditioned charts; only bridge failure beyond the numerical threshold with equivalent pairs and intact controls can support a local hostile identity result. The single simplest confound capable of explaining every row is that the tokenized full prefix and its POS-licensing relation—not operational state—predict the move and response.

### V2 population contract and outcome-blind authoring

`experiments/config/lexical_probe_fresh_v2.json` instantiates the 12-point rule without a frozen hash. It contains the same 80 items, four declared-operation families (`repeat`, `omit`, `capitalize`, `reverse`), four templates per family, eight matched presentation pairs, four operation controls, 32 explicit pair-by-POS author judgments, and 16 explicit control-by-POS author judgments. The author is this fresh Codex session and has not seen any per-template behavior, tokenization-derived selection, or score for these texts. A separate fresh Codex linguistic adversary with no access to new-template model behavior must review every cell next; tokenization is checked only after that review; the raw hash and Git blob are recorded only after both approve. One failed cell voids v2 and requires v3 rather than an edit.

The frame is deliberately metalinguistic: every inventory string occupies the literal dependency frame `the word <X>`. Thus `dog`, `run`, `red`, and `because` are all mentioned words rather than being forced into noun, verb, adjective, or function-word roles. Pair 1 changes only initial register (`Please`/`Kindly`); pair 2 changes only initial discourse framing (`For reference`/`For clarity`). The core `plan to OPERATION the word <X>` remains literal within every pair. Each control retains that wrapper and core and substitutes one operation verb only. This is the author's linguistic judgment, not independent approval.

The fixed `not`-token insertion is withdrawn as the clean second operation. The replacement coherent move is an **operation-verb update in the metalinguistic micro-world**: under matched wrappers, `repeat → omit` and `capitalize → reverse`, with the same mentioned word and aligned word-slot readout. This changes one declared operation for all four POS inventories. Its capture/completion contract must be implemented and Tier-1 reviewed before use; no v1 insertion result can substitute for it.

### Contextual-prefix X-free analyzer mode

Implement one flag-gated analyzer mode, `--contextual-prefix-xfree`, with `--prefix-feature-set token_ids_v1`, the existing `--source`, move/sentinel tag, layer list, unseen-word folds, bootstrap count, and output tag; reject interchangeability, ladder, residualizer-selection, and permutation-null flags in the same call. Inputs are the frozen config plus capture manifests' exact prefix/suffix token IDs and positions, POS label, move alignment, calibration/test folds, and target `Delta` (or the already fixed `Delta_perp` when explicitly paired with a frozen residual design); the mode must never consume item strings/IDs, item embeddings, cell-level `X`, held-out outcomes for feature construction, or a model representation. `token_ids_v1` is fixed as position-specific one-hot token IDs for the last eight prefix and first four suffix positions, full-prefix unigram and adjacent-bigram counts, prefix/suffix lengths and slot/readout positions, POS one-hot, and POS-by-boundary-token interactions; unseen columns map to zero, standardization and ridge/kernel tuning are rebuilt only inside calibration families/words. Output, at every F0/F4/F8/F12/F20 cell, the contextual X-free prediction's displacement cosine, normalized error, response-law skill, continuous-KL improvement, support, fold/key/family values, coefficient/effective-df diagnostics, and paired margins against the cell-level `X` field. A state-reading gate remains live only when the `X` field beats this baseline by `>=0.02` with positive crossed lower bounds on cosine, normalized error, skill, and continuous KL, at least `6/8` jointly positive keys, no family collapse, support `>=0.95`, and at least two common F4–F20 layers for both punctuation sentinels; otherwise the full-prefix licensing account closes or materially narrows the line. Run the fixed all-layer state screen first (`10–30 min`) and the predeclared all-layer completion score second (about `1 h`); no screen-selected layer promotion.

### Calibration-only bridge-screen analyzer mode

Implement a separate early-return `--bridge-screen` mode with `--bridge-ladder scalar diagonal lowrank orthogonal`, fixed `--bridge-ranks 1 2 4 8 16`, both punctuation tags, the operation-update move tag, `--repeat-completions`, `--n-boot`, and output tag; reject residualization, X-free, ladder, and permutation-null flags. Inputs are only frozen v2 matched pairs/controls, approved tokenization and capture hashes, aligned donor/recipient `X,Y,Delta`, calibration-word folds, repeat-noise arrays, and recipient moved-sequence completion context. Every map is zero-preserving and fitted per move/source/layer/pair direction on calibration words only: positive scalar RMS; diagonal ridge shrunk toward that scalar; `alpha I + UV^T` with calibration-selected fixed rank; and scaled orthogonal Procrustes. Inner calibration folds select regularization/rank/branch, never held-out swaps. Before normalized degradation, freeze `rho_move=max(q99 calibration absolute repeat-state difference, 1e-6 × median calibration move norm)` per source/layer/recipient and mark `||Delta||<=rho_move` unsupported. For each endpoint freeze `tau=max(0.02,2×calibration fixed-input q99 noise)`. Output every bridge's held-out normalized successor degradation and continuous-KL degradation relative to same-presentation writeback, repeat/noise summaries, `rho_move`, `tau`, support, pair/direction/family/control rows, selected bridge complexity, and family-first crossed intervals. Scalar failure alone is never hostile. Stable repair by any calibration-selected bridge blocks a hole verdict. A hostile layer requires the best simple bridge's equivalent-swap lower 95% bound to exceed endpoint-specific `tau` on both endpoints—not merely zero—at `>=6/8` pairs in both directions and every family, while all controls remain separated above `tau`; the final hostile verdict still requires two common F4–F20 layers in both punctuation and coherent operation-update move classes. Budget: `15–30 min` state screen and `<=1 h` fixed all-layer writeback.

### Corrected fixed order and budgets

| Order | Work | Expected CPU | Hard wall / launch rule |
|---:|---|---:|---|
| 0 | Finish protected `resAB → resSA2` | existing per-cell budget | Do not open excluded artifacts early; one CPU process |
| 1 | Preserve and audit the B-aug numerical amendment | inside the protected rerun plus review | Retain both failures, failed fold/matrix, finite/spectrum/backend telemetry, and amended-cell status |
| 2 | Repair probe 1's six Round 30 defects and obtain fresh Tier-1 RUN-READY review | review only before execution | Then fixed rank screen in minutes; preselected `P_aug-full` law cell about `1.5 h`, `2 h` wall |
| 3 | Independent v2 linguistic adversary, tokenization audit, then hash/Git/config/commands freeze | no model work | Any failed pair/control/tokenization cell voids v2 and creates a new version |
| 4 | Contextual-prefix X-free screen and completion on the existing relation | `10–30 min`; about `1 h` with completion | Fixed all layers; no promotion from screen outcomes |
| 5 | Capture approved v2 for both punctuation sentinels and the coherent operation-verb update | `15–30 min` | One CPU process; approved raw hash required before model load |
| 6 | Calibration-only bridge ladder | `15–30 min` state; `<=1 h` writeback | Must precede full interchangeability; scalar failure cannot earn hostility |
| 7 | Full v2 interchangeability | about `1 h` | `90 min` wall; hostile LB must exceed `tau`, move floor enforced, both move classes required |
| 8 | Full fresh-population stress analysis, then coherent operation-update analysis | about `3 h`, then `2–4 h` | `4 h` each; both sentinels, no pair/family/POS rescue |
| 9 | Registered X-free field on four existing cells | `4.9–5.3 h` | existing `8 h` wall |
| 10 | Freedman–Lane `A-static` only, conditionally | `24.4–26.3 h` | `30 h`; only if every earlier state-reading gate remains live |
| 11 | Second pinned decoder | about `3 h` | Decoder-specific replication only |

The four-cell Freedman–Lane expansion remains unauthorized. Multi-position consequence (`k={4,8}`), two-step composition/writeback, and a second decoder remain live alternatives if the one-position/one-step program stays ambiguous.

### Second lens after audit #16

**Proven locally:** raw F0 remains identity/token dominated under the tested move; the inherited across-word within-carrier ordering statistic is insensitive for this probe; registered block/length/position metadata predict raw displacement; the tested static and score-4 nuisance fits do not exhaust X-linked residual predictability at F4–F20; B-aug's fitted low-rank path is numerically non-robust at one repeated cell; and fresh v1 fails its linguistic premise before capture.

**Unproven:** operational state; presentation independence or structural presentation/state inseparability; a presentation-free residual; failure of every recoverable quotient or gauge; composition; on-manifold donor writeback; the adequacy of one-position KL as consequence; family/model/task generality; or hostility of this representation to structured reasoning.

The next latent space must provide either explicit quotient coordinates that identify operationally equivalent places across presentation or explicit bridge coordinates/maps connecting presentation-conditioned charts. Those coordinates must be available to the denizen rather than supplied post hoc by the analyst, must preserve declared moves and response laws across equivalent presentations while separating genuinely different operations, must expose a calibrated move norm and consequence currency, and must support multi-step composition. A valid v2 hostile result would motivate that construction locally; it would not prove that no richer quotient exists.

No new axiom is earned. No experiment was run in Round 31.
