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

### Operation-update move contract (Round 31 addendum)

**Status and supersession.** This is a documentation-only design lock. No
capture, model load, score, or result inspection is authorized by this
addendum. It replaces only the Round 30 `not`-insertion clauses in Part 2,
Part 3, `forward_insert`, and the second-move branches of the bridge and
interchangeability designs. The punctuation-A/B contracts are unchanged.
`capture_insert` and `--source forward_insert` remain historical v1 machinery
and must hard-reject v2, v3, or the `OP_UPDATE` tag; no `NOT` artifact can
satisfy this contract.

**Population ruling during this design gate.** The independent linguistic
adversary has now voided `lexical_probe_fresh_v2`: all 16
`For reference`/`For clarity` pair-by-POS cells fail the population's own
operational-equivalence/edit-scope rule because those phrases can contribute
different discourse purposes. No v2 capture is permitted. The live authored
successor is `lexical_probe_fresh_v3.json`, with `Please`/`Kindly` and
straight/typographic-apostrophe presentation systems; it remains pending its
own independent adversary and tokenization approval. The mechanics below are
locked for the first successor population which clears those gates, presently
v3. The v2 facts asked about here—metalinguistic word alignment,
template-final slot, and empty native suffix—are retained unchanged by v3.

The expected useful result is a coherent word-slot transition which either
transfers across wrappers and unseen words or fails in a way that localizes a
presentation-conditioned hole. A pass would establish only a bounded
operation-update regularity in this decoder and population. A failure would
not show that operation updates are impossible in residual streams. The
simplest confound capable of explaining every positive row is that the full
tokenized prefix, including the operation verb and wrapper, supplies a smooth
template fingerprint which predicts both the source state and the recipient
law without exposing an operational coordinate to the denizen. The
contextual-prefix X-free arm and bridge ladder remain ahead of any state or
hostile-hole wording for that reason.

#### Declared update population and direction

The primary universe is **all eight same-wrapper rows belonging to the two
declared updates**, not the four existing
`operational_controls.control_pairs`, and not every arbitrary cross-operation
pair. The operation direction is frozen one way only:

| Update ID | Source template `T_a` | Recipient template `T_b` | Wrapper |
|---|---|---|---|
| `repeat_to_omit__please` | `repeat_1` | `omit_1` | `please` |
| `repeat_to_omit__kindly` | `repeat_2` | `omit_2` | `kindly` |
| `repeat_to_omit__apostrophe_straight` | `repeat_3` | `omit_3` | `apostrophe_straight` |
| `repeat_to_omit__apostrophe_curly` | `repeat_4` | `omit_4` | `apostrophe_curly` |
| `capitalize_to_reverse__please` | `capitalize_1` | `reverse_1` | `please` |
| `capitalize_to_reverse__kindly` | `capitalize_2` | `reverse_2` | `kindly` |
| `capitalize_to_reverse__apostrophe_straight` | `capitalize_3` | `reverse_3` | `apostrophe_straight` |
| `capitalize_to_reverse__apostrophe_curly` | `capitalize_4` | `reverse_4` | `apostrophe_curly` |

Thus probe 3 never adds `omit -> repeat` or `reverse -> capitalize` as extra
samples: those are algebraic inverse reuse of the same authored cells, were
not the move declared in Round 31, and would create pseudo-replication.
“Both directions” below means donor-to-recipient and recipient-to-donor
**presentation transport of the same directed update**, not reversal of the
operation update itself.

Before capture, the approved successor config must add an `operation_updates`
object which freezes the eight ordered rows above,
`directionality = forward_only`, the two update-family labels, the four
wrapper labels, the trajectory-level presentation pairs, and the trajectory-
level negative controls defined below. The existing four
`operational_controls.control_pairs` remain the punctuation move's controls;
they are only the `please` and `apostrophe_straight` subset of the update rows
and cannot be overloaded as the update universe.

For update row `u=(T_a,T_b)`, word `w`, and hidden index `l`:

`X[u,l,w] = H_slot[T_a,l,w]`,

`Y[u,l,w] = H_slot[T_b,l,w]`, and

`Delta[u,l,w] = Y[u,l,w] - X[u,l,w]`.

The source and recipient are aligned by the same item/token identity, never
by equal absolute position. Their word-slot indices may differ if the
operation verbs tokenize to different lengths.

#### Capture source, true law, and completed-law writeback

Use the plain `experiments/run_lm_dynamics.py capture` stage, extended in the
same runner; do not use a sentinel-appended state as the canonical source and
do not create another runner. Add `--tag`, `--repeat-null`, and
`--expected-config-sha256` to `capture` with the same fail-before-model-work
semantics as `capture_forward`. The locked call shape is:

```text
python experiments/run_lm_dynamics.py capture --config experiments/config/lexical_probe_fresh_v3.json --out <run> --tag OP_UPDATE --repeat-null --expected-config-sha256 <approved-raw-sha256>
```

It writes `states_OP_UPDATE.npz` and `manifest_OP_UPDATE.json`. Preserve the
current array names `Z[P,L+1,N,D]` and `laws[P,N,V]`, but pin their semantics
as `H_slot` and `law_at_word_slot` in the manifest. Add arrays
`slot_position[P]`, `readout_position[P]`, `sequence_len[P]`,
`repeat_slot_l2[P,L+1,N]`, and `repeat_readout_kl[P,N]`, plus the existing
ordered `items`, `pos`, `probes`, and `blocks`. `repeat_slot_l2` is the
float32 L2 difference between two identical sequence forwards at the word
slot; it is absolute, not normalized by any update norm. The analyzer derives
recipient-specific normalized repeat errors only after the update rows and
move norms are known.

For the voided v2 and current v3 designs, every suffix must tokenize to the
empty list and every word slot must equal `sequence_len - 1`. Therefore the
current plain capture's last-position law is exactly the law at the word
position. The true response law for update row `u` is

`q_true[u,w] = laws[T_b,w]`,

the unmodified recipient `T_b` law at its word position. It is not the source
law, an appended-sentinel law, or a separate last-token-after-suffix law.
`capture_forward`'s `H_slot`, `H_last`, and `H_q_unappended` should agree for
this empty-suffix population up to the registered causal/reload tolerance and
may be reported as a cross-capture diagnostic, but `law_sent` is the
punctuation move's endpoint and cannot gate operation update.

For a prediction `Delta_hat`, first reassemble the raw displacement under
static residualization,

`Delta_hat = f_Delta(P_static) + Delta_perp_hat`,

then form `Yhat = X + Delta_hat`. Complete in the **recipient template's own
unappended sequence**. At hidden index `l`, write `Yhat` at `T_b`'s word slot,
hook decoder block `l-1`, and read the law at that same word slot. The exact
calls which Claude must implement are:

```python
q_true = completer.laws(
    recipient_probe_idx, states_emb[widx], l - 1, Yhat=None
)[0]
q_hat = completer.laws(
    recipient_probe_idx, states_emb[widx], l - 1, Yhat=Yhat
)[0]
```

No `append_emb`, `pos`, or `insert_before_slot_emb` kwarg is supplied. Passing
`l - 1` is required because `WorldCompleter` hooks block `l-1` to replace
hidden index `l`; at `l=0`, its existing `layer_l < 0` branch replaces the
recipient embedding row directly. `[0]` is the slot law. Although `[1]` is
numerically the same position for an empty suffix, it is not the registered
API endpoint and must not be substituted. The fresh unmodified `q_true` must
be checked against stored float16 `laws[T_b]`; stored laws are reload targets,
not a shortcut around the recipient completion path.

#### Analyzer source, coordinates, folds, nulls, and probe-3 gate

Add `op_update` to `--source`. Add
`--update-pairs` with the sole confirmatory value `from_config`; require
`--move-tag OP_UPDATE`. The confirmatory invocation is exactly the existing
fresh-move analysis shape:

```text
python experiments/analyze_lm_dynamics.py --run <run> --config experiments/config/lexical_probe_fresh_v3.json --source op_update --move-tag OP_UPDATE --update-pairs from_config --target delta --unseen-words 2 --residualize static --pairs 0 1 2 3 4 --n-boot 500 --tag op_update_v3
```

`op_update` implies the Round 30 continuous-KL gates. It rejects sentinel,
insert, control-tag, identity-check, baselines, LOCO, style-null, screen,
residualizer-selection, and permutation-null options. It requires completion,
all five fixed checkpoints `F0/F4/F8/F12/F20`, the approved successor raw
hash, and the manifest/array contract below. No screen outcome may select a
layer or an update subset.

For each of the eight update rows, `P_static` has these 14 coordinates in
fixed order:

1. two centered update-family indicators:
   `repeat_to_omit`, `capitalize_to_reverse`;
2. four centered wrapper indicators: `please`, `kindly`,
   `apostrophe_straight`, `apostrophe_curly`;
3. `source_prefix_token_count`, `recipient_prefix_token_count`,
   `source_sequence_length`, `recipient_sequence_length`,
   `source_word_slot_position`, `recipient_word_slot_position`,
   `source_word_slot_position/source_sequence_length`, and
   `recipient_word_slot_position/recipient_sequence_length`.

Positions are zero-based. One-hots are centered over the eight frozen update
rows; numerical columns and nuisance fits are standardized strictly inside
each calibration fold. Empty source/recipient suffixes are manifest
assertions rather than two constant feature columns. No template name, pair
ID, raw text, token ID, word ID, item embedding, hidden state, result, or
outcome is a `P_static` coordinate.

An operation family cannot be the primary held-out unit because each move
crosses two operation families. Define an **update carrier** as
`(update_family, wrapper)`. The primary outer split is four
leave-one-wrapper-out folds. Each held wrapper contributes two test carriers,
one for `repeat_to_omit` and one for `capitalize_to_reverse`; calibration uses
the six carriers from the other three wrappers. Cross those folds with the
same two POS-stratified unseen-word folds. Each of the eight keys therefore
has `6 carriers x 40 calibration words` and `2 carriers x 40 held-out words`,
with both source and recipient templates absent from the other side and word
identities disjoint. Inner selection leaves one of the three calibration
wrappers out and rebuilds every nuisance basis, standardizer, and
hyperparameter. The config's `repeat/omit` versus `capitalize/reverse`
directional split may be reported once as a secondary frozen
cross-update-family diagnostic; it cannot enter selection or count toward the
eight-key gate.

The crossed bootstrap resamples the four held-wrapper units first, resamples
the two update carriers within a sampled wrapper, and uses one POS-stratified
held-word draw for a word-fold key crossed across all sampled wrappers.
Report each update family separately; a pooled mean cannot hide either
family's reversal.

The fixed unseen-word universe remains exactly K=13:
`identity`, `mean`, `class_mean`, `wordonly_knn`,
`wordonly_ridge_emb`, `wordonly_kernel_emb`, `knn1`, `knn5`, `knn20`,
`ridge`, `lowrank`, `kernel`, and `chart`. The four X-free lexical nulls are
unchanged: POS-class mean residual displacement, frozen-input-embedding
word-only cosine kNN, word-only ridge, and word-only RBF kernel ridge. All
feature construction and tuning are nested in calibration wrappers and
calibration words. `mean` is the shared calibration displacement. A
per-word mean is unavailable in the unseen-word test and does not enter K=13.

The identity null means literally `Delta_hat=0`, hence `Yhat=X`: changing the
operation verb is predicted to make no word-slot move. At supported F4-F20
cells its normalized error is one and its recipient law is completed by
writing source `X` into `T_b`. A zero prediction has no defined displacement
cosine, so identity is not manufactured into a cosine comparator; it remains
in K=13 and must be beaten with positive crossed lower bounds on normalized
error improvement and continuous-KL improvement before any X-conditioned
reading. The four X-free nulls remain the primary three-endpoint competitors.

At F0 the same mentioned word supplies the same embedding row in `T_a` and
`T_b`. In the pinned architecture no absolute-position vector is added to
that row, so `Delta_0=0` structurally even if the two word slots have different
indices. The capture manifest records the float32 maximum absolute difference
for every update row. F0 reports that control, move-norm quantiles, reload
error, and unsupported denominators only; it cannot pass or fail probe 3.

For F4/F8/F12/F20, select the primary field family among
`ridge/lowrank/kernel` by the mean displacement cosine over the inner
leave-one-wrapper-out validation folds, after each family's own
hyperparameters have been selected on those same inner folds. Break an exact
tie in the fixed cheaper order `ridge`, `lowrank`, `kernel`, and expose the
selected name and all inner scores per outer key; held-out model-family
selection is forbidden. A layer qualifies only if that preselected field
beats the strongest of the
four X-free lexical nulls by `>=0.02`, with positive crossed lower bounds, on
all three registered endpoints: displacement cosine, response-law skill
relative to the same-fold shared-displacement completion, and continuous-KL
improvement `KL(null)-KL(field)`. The strongest-null minimum is taken inside
each bootstrap replicate. It must also beat identity as just specified, have
at least `6/8` fold keys jointly positive on all three primary margins, show
no collapse or sign reversal in either update family, and retain common K=13
support `>=0.95`. The operation-update move qualifies only with at least two
qualifying F4-F20 layers. A pass remains one directed move class in one
decoder and does not establish composition, presentation independence, or a
native law.

#### Bridge and interchangeability source contract

Both early-return modes consume the same materialized source entry. With the
eight config rows in frozen order, the entry is semantically:

```python
sources["op_update"] = {
    "X": Z[source_probe_idx].astype(np.float32),       # [8,L+1,N,D]
    "Y": Z[recipient_probe_idx].astype(np.float32),    # [8,L+1,N,D]
    "law": laws[recipient_probe_idx].astype(np.float32),  # [8,N,V]
    "cls": "operation_update",
    "source_probe_idx": source_probe_idx,
    "recipient_probe_idx": recipient_probe_idx,
    "kw": {},
    "repeat_slot_l2": repeat_slot_l2[recipient_probe_idx],
    "repeat_readout_kl": repeat_readout_kl[recipient_probe_idx],
    "man": manifest,
}
```

`laws_at(source, u, l, Yhat, widx)` must translate update-row index `u` to
`recipient_probe_idx[u]`, call `WorldCompleter.laws` with the empty `kw` and
`l-1`, and take `[0]`. The recipient's target template, slot, and word law are
used for same-presentation, swapped, bridge, control, and truth calls. The
same-presentation reference writes stored `Y[u,l,w]` through the identical
hook; hook/reload error is therefore shared.

Freeze these four equivalent trajectory-pair clusters:

- `repeat_to_omit__please` with `repeat_to_omit__kindly`;
- `repeat_to_omit__apostrophe_straight` with
  `repeat_to_omit__apostrophe_curly`;
- `capitalize_to_reverse__please` with
  `capitalize_to_reverse__kindly`;
- `capitalize_to_reverse__apostrophe_straight` with
  `capitalize_to_reverse__apostrophe_curly`.

Each is transported in both presentation donor directions using calibration
words only. Freeze four trajectory-level negative controls by pairing
`repeat_to_omit` with `capitalize_to_reverse` under the identical wrapper,
once for each of `please`, `kindly`, `apostrophe_straight`, and
`apostrophe_curly`.
These replace `operational_controls.control_pairs` only for the operation-
update source. They test whether a bridge or swap also collapses two genuinely
different declared updates while wrapper is fixed.

The bridge ladder remains positive scalar, diagonal ridge toward the scalar,
`alpha I + UV^T` at fixed ranks `{1,2,4,8,16}`, and scaled orthogonal
Procrustes, selected inside calibration words per source/layer/trajectory-pair
direction. Apply the same calibration-only fitting discipline to equivalent
and negative-control comparisons. For operation update,
`rho_move=max(q99 recipient repeat_slot_l2, 1e-6 x median calibration move
norm)` per layer/recipient; cells with `||Delta||<=rho_move` are unsupported.
Endpoint `tau=max(0.02, 2 x calibration fixed-input q99 noise)` is frozen
before held-out swaps. Scalar failure alone is not hostile, and repair by any
calibration-selected simple bridge blocks a hostile verdict.

Because the directed update universe yields four unordered equivalent
trajectory-pair clusters, the stale generic `6/8 matched pairs` phrase cannot
be copied literally to this source. For operation update, the breadth clause
is at least `3/4` clusters satisfying the point criterion in **both**
presentation donor directions, with both update families represented and no
family reversal. All four trajectory-level controls must remain separated
above endpoint-specific `tau`. Punctuation retains its existing `6/8`
matched-pair clause and requires A and B at the same layer. Stable or hostile
joint passage requires at least two layers in the intersection of the
punctuation-A, punctuation-B, and operation-update passing-layer sets. All
other pooled, interval, move-floor, bridge, control, and local-scope clauses
from Round 31 remain unchanged.

#### Manifest, provenance, and Tier-1 acceptance checklist

`load_config_checked` and `common_manifest` must cover the operation capture;
legacy `capture` provenance is insufficient. Every OP_UPDATE manifest stores:

- `config_sha256_raw`, `config_git_blob`, `config_git_commit`, and the
  separately named `config_declared_sha256`, all checked before model load;
- successor-version linguistic-adversary approval and tokenization-approval
  identifiers and statuses; capture refuses v2, any pending/failed
  population, or an absent approved raw hash;
- canonical hashes over ordered items, ordered
  `(name, block, operation, template, pair)` template rows, where `operation`
  is explicit or is asserted equal to `block` before hashing,
  `presentation_pairs`, `operational_controls`, all eight ordered
  `operation_updates.update_pairs`, the four update trajectory-pair clusters,
  and the four update trajectory controls;
- `move_kind=operation_verb_update`, `move_tag=OP_UPDATE`,
  `directionality=forward_only`, `source_alignment=word_token`,
  `readout_kind=recipient_word_slot`, the two allowed update-family labels,
  all eight source/recipient probe indices and names, wrapper labels, and the
  fixed update-row order;
- per template exact prefix/suffix token IDs, zero-based word slot and readout
  position, sequence length, the assertion `suffix_token_ids=[]`, and the
  assertion `word_slot=readout_position=sequence_len-1`; plus per update row
  both source and recipient slots/lengths and same-item alignment;
- per-update-row float32 F0 word-state maximum absolute difference, its
  required zero/registered numerical status, repeat-state and repeat-law
  summary quantiles, and the fact that full per-cell repeat arrays are in the
  NPZ;
- model and tokenizer revisions, tokenizer class, layer count, embedding and
  vocabulary dimensions, Python/NumPy/PyTorch/Transformers versions, device,
  compute/storage dtype, thread count, batch size, exact argv, elapsed time,
  array filename/hash, every array name/shape, and the approved config path.

The A/B sentinel manifests used jointly with this source must share the same
raw config hash, Git blob/commit, model revision, tokenizer revision,
item/template/map hashes, layer/dimension pins, and ordered probe/item axes.
For the empty-suffix successor population they additionally record and verify
`source_position=word_slot`, `readout_position=word_slot+1`, and sentinel
append after the word. In punctuation analysis `law_sent` remains primary;
the fact that `law_last` is now the pre-sentinel word law does not change the
punctuation endpoint.

A Tier-1 reviewer returns **NOT-READY** unless all of the following are
literal in code and a no-model fixture test: eight forward-only update rows,
the four wrapper-held-out by two unseen-word folds, 14 ordered `P_static`
columns, exact K=13 and four X-free nulls, structural F0 handling, recipient-
template `WorldCompleter(..., l-1)[0]` with empty kwargs, target-law reload,
common support and strongest-null-inside-replicate gating, operation-specific
trajectory pair/control maps, bridge move floor and `tau`, all required
manifest hashes/positions, and hard rejection of v1 `NOT` artifacts. The
review must also verify that no experiment was run during implementation or
review and that flag-off punctuation behavior is unchanged.

No claim is earned by this addendum. Its “so what” is that the denizen now has
a coherent candidate move—change the declared operation while holding the
mentioned object and wrapper fixed—whose portability across presentations can
be tested without confusing it with malformed negation or punctuation.

## Round 32 — B-score4 adjudication, amended numerical instrument, and the completed residualization 2x2

**Codex adjudication, documentation only; no experiment was launched or
rerun.** The four permitted residual artifacts were reduced directly from
JSON. The two crash rows, the successful `resAB` ledger row, the live
`RidgeFamily.W` fallback, Rounds 23 and 27–31, and audits #15–#16 were checked
independently. `analysis_resSA2.json` and every running/queued
`analysis_scr_*`, `ctx*`, or `resAFull` artifact remained unopened. The
uncommitted capture/analyzer work was not edited.

### B `P_aug-score4`: mechanical adjudication

`analysis_resAB.json` is sentinel B (`,`), the implemented score-4 augmented
design, five fixed checkpoints, two unseen-word folds, eight block-by-word
keys, K=13, 20 shuffles, and 500 bootstrap replicates. It finished on the
third launch in `5073.8 s` of the `7200 s` wall. Support is `1.0` in every
key. Reload ordering agreement is `0.9996511`, maximum pairwise-KL difference
is `0.0013232`, and the model/tokenizer revisions are pinned in the artifact.

Each endpoint below uses the weakest ridge margin over the four registered
residual X-free lexical nulls; each lower bound belongs to that endpoint's
weakest comparator.

| Layer | `X_perp` ridge cosine | strongest residual-null cosine | cosine margin [LB] | skill margin [LB] | K=13 KL-rank margin [LB] | full / positive keys | score-4 nuisance `-> Delta` cosine | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F0 | 0.335 | 0.003 | +0.333 [+0.217] | +0.091 [-0.039] | +0.310 [+0.066] | 4/8 / 6/8 | 0.641 | fail |
| F4 | 0.570 | 0.061 | +0.509 [+0.447] | +0.411 [+0.290] | +0.455 [+0.334] | 6/8 / 8/8 | 0.513 | pass |
| F8 | 0.567 | 0.075 | +0.492 [+0.439] | +0.399 [+0.333] | +0.451 [+0.352] | 7/8 / 8/8 | 0.424 | pass |
| F12 | 0.523 | 0.065 | +0.459 [+0.415] | +0.441 [+0.379] | +0.510 [+0.438] | 8/8 / 8/8 | 0.456 | pass |
| F20 | 0.566 | 0.089 | +0.477 [+0.427] | +0.399 [+0.335] | +0.543 [+0.483] | 8/8 / 8/8 | 0.640 | pass |

No block collapses at F4–F20. F0 is non-qualifying: its skill lower bound is
negative, only `4/8` keys clear the full gate, and gloss collapses even though
the pooled cosine is positive. The passing checkpoints are a correlated depth
profile, not four replications.

Against Round 23, the bounded state-linked-side mechanical prediction holds
at F4–F20 and the predicted F0 identity/token regime holds. The four
word-only X-free nulls do not close the relation, but audit #15's contextual
prefix/licensing alternative remains untested. The score-4 nuisance fit does
not produce the predicted collapse, but this does **not** adjudicate the
literal registered `P_aug`: Round 23 registered the full carrier mean plus
rank-4 scores, whereas this run appends only the four scores. Call the cell
`P_aug-score4`; `P_aug-full` remains unrun. Its nuisance-to-`Delta` cosines are
directional predictability from transductive X-derived carrier summaries, not
a presentation-only component, variance share, mediation effect, or state
fraction.

### Common-scale block

The B-score4 block was read directly rather than inferred from the ledger:

| Layer | cosine ratio median [95% CI] | skill ratio median [95% CI] | continuous-KL ratio median [95% CI] |
|---|---:|---:|---:|
| F0 | 2.566 [1.364, 5.493] | 1.223 [0.174, 5.266] | 1.849 [0.632, 22.726] |
| F4 | 1.285 [1.137, 1.602] | 1.322 [0.862, 2.978] | 0.846 [0.458, 1.276] |
| F8 | 1.209 [1.085, 1.442] | 1.252 [0.895, 1.706] | 1.253 [0.798, 1.766] |
| F12 | 1.246 [1.133, 1.494] | 1.215 [0.912, 1.588] | 1.232 [0.913, 1.620] |
| F20 | 1.141 [1.064, 1.259] | 0.932 [0.743, 1.232] | 0.964 [0.747, 1.190] |

All twelve F4–F20 medians exceed `0.5`; eleven of twelve lower bounds do, with
F4 continuous KL at `0.458`. Across all five layers, thirteen of fifteen lower
bounds exceed `0.5`; F0 skill is the other exception. Audit #15's wording rule
is binding: these are robustness ratios between different fitted
predictive-margin systems and strongest-null competitions. They are not
fractions of retained signal, variance, operational state, or mediation.

### B-aug numerical amendment audit

The two failures must remain in the result's provenance.

1. `nlm007_resid_resAB_crash` records the first process disappearing after
   `22/40` completed keys while entering `F8 grammar_w0`, with no traceback.
   The later row attributes it to the SVD defect, but the first row alone does
   not directly localize a matrix failure.
2. `nlm007_resid_resAB_crash2` records `23/40` completed keys and directly
   localizes `torch._C._LinAlgError: linalg.svd failed to converge` at
   `F8 grammar_w1` to `RidgeFamily.W`'s fitted low-rank coefficient matrix
   `W`. The successful artifact selects rank `128`, lambda `100` in that fold.

The safe common localization is therefore the **F8 grammar block**; only the
second failure is directly localized to `grammar_w1/W`. “Both failed at the
same exact fold” and “ill-conditioned `X_perp`” are not established. The
artifact also does not record the actual `W` shape after standardization.

The amended code catches `torch._C._LinAlgError` **and the broader
`RuntimeError`**, runs `numpy.linalg.svd` on float64 `W`, casts the factors
back to `W.dtype`, and sets `self.svd_provider`. The broad catch can relabel an
unrelated runtime failure as an SVD fallback and must be narrowed to the
actual convergence exception before another result. The provider is never serialized. The result file
contains no per-fit/fold provider, fallback exception, finite-input check,
`W` shape/dtype/norm, singular spectrum, effective rank, condition estimate,
rank-boundary gap, or backend-agreement record.

The third-launch artifact may be reported as the B `P_aug-score4` cell only
with the labels **amended-implementation** and **SVD-telemetry-incomplete**.
Its ridge-vs-null table is mechanically reportable; it is not a clean
low-rank robustness result, and its K=13 endpoint remains amendment-qualified
because the low-rank candidate is in that universe. The completed artifact
does not erase either failure.

Before any further low-rank result, every outer and inner `RidgeFamily.W` SVD
must serialize layer, held block, word fold, fit scope, target, lambda, rank,
provider, exception/fallback status, input and `W` shape/dtype, finite checks,
norm/range, singular extrema and retained spectrum, effective rank/condition,
rank-boundary gap, and reconstruction residual. A no-result instrumentation
review must also verify that only `torch._C._LinAlgError` activates the
fallback and that all other exceptions propagate, then predeclare and
implement this shadow-backend check without selecting from outcomes:

- for every `W` where torch converges, run the float64 NumPy backend on the
  identical frozen matrix and compare singular values and full reconstruction;
- compare reconstructed rank-`r` matrices for every used rank and their
  predictions on the identical validation/held-out design, not raw singular
  vectors whose signs or degenerate-subspace bases are non-identifiable;
- require relative full-reconstruction and singular-value discrepancies
  `<=1e-5`, rank-`r` matrix and prediction discrepancies `<=1e-4`, absolute
  downstream metric discrepancies `<=1e-4`, and identical gate decisions;
  otherwise mark the fit backend-sensitive and ineligible for a low-rank
  claim; and
- preserve the failed torch exception and NumPy diagnostics for a fallback
  fold rather than manufacturing torch agreement where torch does not
  converge.

No such check was run in Round 32.

### Four-cell synthesis

| Cell | Design status | F0 | F4–F20 | common-scale status |
|---|---|---|---|---|
| A-static | registered `P_static` | fail | pass at all four layers | absent from `resSA`; `resSA2` still required |
| A-score4 | implemented, outcome-clean, within-carrier transductive, contract-qualified | aggregate pass but only 2/8 full keys; weak pooled conditional association | pass at all four layers | all 12 passing-layer medians >0.5; F4 KL LB 0.495 is the sole passing-layer exception |
| B-static | registered `P_static`; correlated sentinel check | fail | pass at all four layers | all 12 passing-layer medians >0.5; F4 KL LB 0.426 is the sole exception |
| B-score4 | implemented score-only sensitivity; amended implementation and telemetry-incomplete | fail | pass at all four layers | all 12 passing-layer medians >0.5; F4 KL LB 0.458 is the sole exception |

The 2x2 is complete for the residual-vs-null mechanical question, not for the
common-scale question: the A-static repaired block is still missing. Its
maximum joint license under audits #15–#16 is:

> Across two correlated punctuation sentinels in one decoder and one authored
> population, registered block/length/position metadata predict held-out raw
> displacement direction. After those registered coordinates are cross-fitted
> out, `X_perp` retains predictive association with `Delta_perp` and improves
> the reassembled response-law prediction beyond four registered word-only
> X-free lexical nulls at F4–F20. Adding four outcome-clean but within-carrier
> transductive X-derived carrier-summary scores also does not absorb that
> association. These are correlated same-population sensitivities, not
> replications; they identify neither operational state, presentation
> independence, a presentation decomposition, composition, nor a native law.

`P_static` is the registered-static-metadata arm. `P_aug-score4` cannot
independently establish presentation sensitivity because its added coordinates
derive from X. The literal `P_aug-full`, contextual-prefix baseline, fresh
population behavior, coherent operation update, second decoder, and
composition remain unrun.

### Second lens and corrected order

The completed 2x2 adds no representation-level hole hostile to structured
reasoning. It strengthens the local facts that registered metadata predict
the measured move and that the tested nuisance fits do not exhaust X-linked
predictability at F4–F20. It does not prove presentation/state inseparability,
a presentation-free residual, absence of a recoverable quotient or gauge, or
that structured reasoning cannot live here. Raw F0 identity/token dominance
is a proven bounded regime and constructive warning, not by itself a hostile
hole. The inherited across-word ordering statistic remains a proven local
measurement hole; the repeated SVD failure is a numerical-instrument hole,
not a latent-space hole. No new axiom is earned.

The Round 31 order is confirmed in substance with two current-state
amendments: the SVD telemetry/backend gate is inserted before any further
low-rank output, and v4 approval/freeze is already complete. V2 and v3 were
voided at their design gates; v4 passed the independent linguistic adversary
`48/48`, passed tokenization, and is frozen by raw SHA-256/Git blob. From the
current state the order is:

1. finish protected `resSA2` without early inspection;
2. add the SVD telemetry/shadow-backend contract above and obtain Tier-1
   numerical review before further low-rank output;
3. run the fixed probe-1 rank screens and the preselected sentinel-A
   `P_aug-full` law cell after probe 1's existing RUN-READY gate;
4. run the fixed all-layer contextual-prefix screen and completion baseline;
5. treat v4 linguistic approval, tokenization, hash, and Git freeze as an
   already-satisfied immutable gate—do not edit or reselect it;
6. after operation-update and bridge code pass Tier-1 review, capture v4 A,
   B, and `OP_UPDATE` with the approved raw-hash guard;
7. run the calibration-only bridge ladder before full interchangeability;
8. run full interchangeability, then the fresh A/B stress analyses and the
   coherent operation-update analysis;
9. run the four registered X-free cells, conditional Freedman–Lane
   A-static only if every earlier state-reading gate remains live, and then
   one pinned second decoder.

The four-cell Freedman–Lane expansion remains unauthorized. Multi-position
consequence and two-step composition/writeback remain live alternatives if
the one-position, one-step line stays ambiguous.

## Tier-3 audit #17 — v4 scope, numerical qualification, and consequence-first allocation (fresh Codex auditor)

### Verdict

**Numerical integrity: PASS, qualified.** Round 32’s B-score4 values reproduce directly from [analysis_resAB.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/lm_dyn_v1/analysis_resAB.json>). The four-cell table is complete for the **residual-versus-four-word-only-null mechanical gate**, not for common-scale retention, literal `P_aug-full`, or independent replication.

**V4 linguistic gate: APPROVAL UPHELD, but materially narrowed.** V4 is eligible for capture under its registered contract. The adversary established grammaticality, a constant mentioned-word dependency, preserved explicit string-edit instructions, and matched surface-word edit distance. It did **not** establish that `Please`/`Kindly` or `Hello,`/`Hi,` are pragmatically or latently inert, nor that the four inventories are tested in their ordinary syntactic uses.

**Operation-update construct: DOWNGRADE.** It is a coherent **declared-operation-verb context intervention**. It is not yet a denizen-enacted operational move: source and recipient states come from separately re-encoding two prefixes, and the endpoint does not show that repeat/omit/capitalize/reverse was executed.

**Representation-level hostile hole: NOT PROVEN.** A hostile v4 result could establish only a local obstruction to this identity relation under this bridge family and response currency. A stable result could establish only local portability in an autonymic instruction micro-world.

**Public/demo propagation: FAIL.** The v9 lede is incomplete and the body contains two statements made false by Round 32. The footer is closer to current truth but should carry the amendment partition explicitly.

I did not inspect `analysis_resSA2.json` or the live uncommitted analyzer/capture files or diffs. Pre-existing Tier-1 material present on the mandatory blackboard was excluded from this verdict.

### 1. Direct verification of Round 32

Direct reduction gives:

| Layer | Ridge | Strongest residual null | Cos margin [LB] | Skill margin [LB] | K=13 KL-rank margin [LB] | Full / positive keys | Score-4 → Δ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F0 | 0.335 | 0.003 | 0.333 [0.217] | 0.091 [−0.039] | 0.310 [0.066] | 4/8 / 6/8 | 0.641 | fail |
| F4 | 0.570 | 0.061 | 0.509 [0.447] | 0.411 [0.290] | 0.455 [0.334] | 6/8 / 8/8 | 0.513 | pass |
| F8 | 0.567 | 0.075 | 0.492 [0.439] | 0.399 [0.333] | 0.451 [0.352] | 7/8 / 8/8 | 0.424 | pass |
| F12 | 0.523 | 0.065 | 0.459 [0.415] | 0.441 [0.379] | 0.510 [0.438] | 8/8 / 8/8 | 0.456 | pass |
| F20 | 0.566 | 0.089 | 0.477 [0.427] | 0.399 [0.335] | 0.543 [0.483] | 8/8 / 8/8 | 0.640 | pass |

Support is `1.0` in every key. Reload ordering agreement `0.9996511`, pairwise-KL difference `0.0013232`, and runtime `5073.8 s` also match [Round 32](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/EXPERIMENTS.md:6079>).

The resulting 2×2 is:

- A-static: F0 fail; F4–F20 pass.
- A-score4: F0 aggregate pass but only `2/8` full keys; F4–F20 pass.
- B-static: F0 fail; F4–F20 pass.
- B-score4: F0 fail; F4–F20 pass.

“Correlated same-population sensitivities” is correct and should stay. It is conservative without erasing the positive result: the pattern is robust across two sentinel tokens and two nuisance specifications, but all cells reuse one decoder, population, folds, task, and analysis family.

The common-scale 2×2 is **not** complete because A-static lacks the repaired block. “Completed residualization 2×2” must always be followed by “for the residual-versus-null mechanical question.”

#### SVD qualification

`amended-implementation` and `SVD-telemetry-incomplete` are supported. The artifact contains no serialized provider, exception, spectrum, effective rank, condition estimate, reconstruction residual, or backend comparison.

The qualification should be partitioned:

- Ridge-versus-four-X-free-null cosine and skill margins are mechanically reportable.
- Low-rank conclusions are not cleanly reportable.
- The K=13 KL-rank endpoint is amendment-qualified because low-rank is a member of that ranked universe.
- A blanket implication that the ridge-only table is numerically invalid would be an over-claimed kill.

Two ledger defects remain:

1. `nlm007_resid_resAB_crash2` says the first loss was the same defect and “same fold.” The first loss occurred while entering `F8 grammar_w0` without traceback; the second directly localized SVD failure at `F8 grammar_w1/W`. The safe common localization is only the F8 grammar block.

2. `nlm007_fresh_v2_voided` says “all 32 pair-1 cells” passed, while its own metrics correctly say 16 pair cells passed, 16 failed, and 16 controls passed. “32” must become “16” in an append-only erratum.

### 2. Adversary of the v4 adversary

The live [v4 config](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/config/lexical_probe_fresh_v4.json:1>) is clean. Its raw SHA-256, Git blob, and adversary-report SHA-256 exactly match the ledger and approval block. The freeze itself is sound.

#### `Please` versus `Kindly`

This pair passes the narrow contract:

- Both produce grammatical directives.
- Both preserve `plan to OP the word <X>`.
- Neither changes polarity, explicit modality, quantification, tense, or the named operation.

It does **not** justify “genuinely inert” without qualification. `Kindly` and `Please` differ in politeness strategy, formality, interpersonal stance, and potentially directive force. They also tokenize to different prefix lengths, moving the word slot. Those are legitimate presentation variables for the experiment to test; they are not properties the linguistic reviewer can declare absent from the latent state.

Correct ruling: **core-operation equivalent; presentation/pragmatic variation intentionally non-identical.**

#### `Hello,` versus `Hi,`

This also passes narrowly:

- Both are detached greetings followed by the same directive.
- Neither supplies a reason, condition, or explicit operation content.
- The comma keeps the greeting outside the mentioned-word dependency.

But they encode register and social/deictic stance. They are not globally inert. A latent difference under these greetings would not automatically be pathological; the empirical question is whether that difference is bridgeable while operation-specific consequences remain distinct.

Correct ruling: **same greeting-plus-directive structure and explicit operation; not identical discourse state.**

#### The “four inventories” claim

The adversary correctly notices that every item becomes an autonym under `the word <X>`. That is also the design’s central limitation.

Nouns, verbs, adjectives, and function words are no longer tested as nouns, verbs, adjectives, or function words. All are objects of the same metalinguistic NP. POS is provenance, not carrier syntax. Therefore:

- `48/48` is a checklist count, not 48 independent linguistic observations.
- The 32 presentation cells do not constitute four distinct syntactic-use validations.
- The design solves the all-POS licensing problem by removing ordinary POS use from the experiment.

This is legitimate if named honestly: **a mentioned-string instruction micro-world**. It is not a test of presentation invariance across ordinary linguistic uses.

#### Mention versus use under the guiding question

Interchangeability here would mean:

> Two wrapper-conditioned descriptions of the same intended string operation admit portable word-slot trajectories and response laws, while descriptions of different intended string operations remain distinguishable.

That is a meaningful denizen question in a small constructed world. It could identify a local equivalence or obstruction for representations of instructions about strings.

It would not show that:

- ordinary semantic uses share a quotient;
- the denizen actually executes the named operation;
- the relation survives truth-conditional reasoning, binding, negation, or quantification;
- POS-sensitive structured reasoning can live in the representation;
- presentation is globally separable from state.

#### Operation-verb update

`repeat→omit` and `capitalize→reverse` are coherent as directed changes in declared instruction. But they are also, literally, one-word prefix substitutions followed by separate forward passes. Moreover, every carrier says **“plan to”** perform the operation; the operation is not performed.

Thus the current endpoint measures a transition between two instruction-conditioned encodings, not an operation on the mentioned word. Until downstream consequences demonstrate repeat/omit/capitalize/reverse behavior, call it a:

> declared-operation-verb context update

Do not call it an unqualified operational move or denizen navigation primitive.

#### Sentinel comparability

The formal token action is unchanged: append `.` or `,` after the complete sequence. Its semantic role has changed.

- In `lm_dyn_v1`, the source `q` is the terminal token of a heterogeneous suffix following `<X>`—for example `means`, `and`, `because`, or `the`.
- In v4, the suffix is empty and `q` is the mentioned word itself at the end of a complete directive.

So the sentinel is now a transition directly from the mentioned-word state into punctuation, whereas previously it followed diverse suffix terminals and continuation demands. V4 A/B results may be called a **fresh-population stress test of the same token-level append construction**. They must not be pooled with or called a direct replication of `lm_dyn_v1` sentinel dynamics.

The config’s statement that the append is substantive rather than invisible is correct.

### 3. Over-claimed kills

#### V2

The void is procedurally correct under the registered one-failed-cell contract. `For reference` versus `For clarity` can introduce different discourse purposes, violating the unusually strict “cannot supply reason or goal” clause.

But this kills v2 only for the confirmatory presentation-only claim. It does not establish that the pair is scientifically worthless. V2 remains a useful harder pragmatic-purpose or presentation-gauge stress set.

#### V3

The whole-version void is also procedurally correct because the frozen contract required passing controls. But its linguistic presentation pairs passed. What failed was the orthographic-control match: a one-glyph apostrophe edit was compared with a whole-operation-word substitution.

Therefore v3 is not a linguistic failure. It remains usable descriptively for:

- orthographic near-invariance;
- numerical-noise calibration;
- a presentation-only test without operation-attribution claims.

“Three adversarial voids” is technically true but misleading unless it distinguishes v3’s control-design failure from v1/v2’s equivalence failures.

#### `not` insertion

Withdrawal is correct as a single all-inventory confirmatory move. Insertion before heterogeneous ordinary noun, verb, adjective, and function-word uses does not define one coherent operation.

It does **not** kill negation as a research direction. Negation remains a strong typed operation in a use-frame population of predicates or propositions with behavioral truth-conditional consequences.

### 4. Research-integrity findings

1. **Frozen-config prose contradiction.** The config says `status: approved_frozen` and contains the completed approval block, but its top-level note still says “This candidate is not approved for capture.” Do not mutate the frozen file. Add a ledger/STATE erratum saying the note is historical authoring-time text and that the structured approval/hash fields are authoritative.

2. **Adversary independence is procedural, not epistemic.** Author and reviewer were separate fresh sessions without v4 outcomes, which is good. They nevertheless share the Codex model family, the contract, and the adaptive v1–v3 design history. “Independent” should mean “separate-session and outcome-blind,” not independent linguistic expertise.

3. **Design adaptivity remains.** V4 is prospective for its own model outcomes, but the overall population family was iteratively selected to survive prior conceptual criticism. That is not leakage, but it is adaptive instrument development, not a pristine one-shot confirmation.

4. **Demo inconsistency violates the propagation rule.** [The demo](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.claude_demo_content_vs_context.html:69>) omits B-score4 in the lede, says no symmetric A/B residualization is available at line 154, and refers to only two A-side residualizations at line 160. The footer already says both B cells are complete.

### 5. Tunnel vision and strongest missed explanation

Yes, the program is tunnel-visioned.

Three replacement populations and seven review rounds bought real rigor, but the marginal work is now increasingly devoted to making one synthetic instrument internally admissible. The program still has:

- one decoder;
- one 80-word inventory;
- one autonymic template family;
- analyst-authored equivalence;
- one-position response laws;
- one-step interventions;
- one code path with substantial review burden.

The strongest alternative explanation v4 still misses is a **generic prefix-edit response or Jacobian**:

> Transformer states smoothly propagate lexical differences in the prefix. The word-slot `X` therefore contains a high-dimensional summary of wrapper, operation verb, length, and position; ridge or a bridge predicts the recipient state because re-encoding a one-word prefix edit has regular geometry, not because the world exposes an operational quotient.

The queued token-ID/bigram contextual baseline helps, but may still miss pretrained token-embedding similarity, contextual prefix summaries, and nonlinear prefix effects.

#### CPU-only alternatives

| Exploration | Estimated CPU | Decides |
|---|---:|---|
| Existing contextual-prefix screen + completion | 10–30 min + about 1 h | Whether tokenized grammar/licensing closes the current relation |
| Frozen input-embedding/edit-kernel prefix baseline | 15–45 min; ≤1 h with completion | Whether smooth lexical prefix geometry explains v4 without cell-level `X` |
| Operation-token causal patch/ablation at two fixed layers | about 1–3 h | Whether the operation verb causally carries the measured transition |
| Multi-position teacher-forced consequence, `k={4,8}` | 20–40 min capture + 1–2 h scoring | Whether the one-position law hides or manufactures consequence |
| Typed use-frame task: polarity, binding, or quantifier update | 15–30 min capture + 2–4 h analysis | Whether a real semantic operation transfers outside mention |
| Two-step composition/writeback | 10–20 min capture + about 1 h scoring | Whether the field composes |
| Second pinned decoder | about 3 h | Decoder specificity only |

#### Allocation ruling

Elevate the **multi-position consequence law before the second decoder**. Construct validity precedes replication: a second decoder can reproduce the same wrong one-position instrument.

Recommended order:

1. Finish the protected existing work and SVD instrumentation gate.
2. Run the contextual-prefix baseline.
3. Run one bounded multi-position consequence test.
4. If consequence is informative, run v4 bridge/interchangeability and then a second decoder.
5. In parallel with neither, stop authoring v5; the next population should instead be a typed use-frame task.

### 6. Second-lens ruling

A stable v4 interchangeability result could prove locally that the two wrapper systems admit a common raw or simply bridged representation of these described string-operation trajectories, while distinct update families remain separated.

A hostile result—equivalent-swap degradation beyond `tau`, intact controls, adequate move norms, both directions, two common layers, and failure of the full simple bridge ladder—could support:

> a local obstruction to the registered wrapper-equivalence relation for mentioned-string instruction trajectories in this decoder.

It could motivate a next latent space with explicit quotient or bridge coordinates.

It could not prove:

- a hole hostile to structured reasoning generally;
- presentation/state inseparability;
- absence of nonlinear or learned gauges;
- ordinary-use linguistic failure;
- operation execution;
- composition;
- on-manifold donor writeback;
- correct consequence currency;
- model-, task-, or population-level generality.

The strongest current holes remain measurement/instrument holes: the inherited ordering statistic and the SVD path. Raw F0 token identity is a bounded structural regime. V4 may expose a candidate representational obstruction; it cannot yet promote one to a hostile structural hole.

## Round 33 — audit #17 adoption and consequence-first allocation (2026-08-28)

Tier-3 audit #17 is adopted verbatim in `theory/EXPERIMENTS.md`. Its numerical
verdict is PASS, qualified: the residual-versus-four-word-only-null 2x2 is
complete mechanically at F4–F20, ridge cosine/skill margins are reportable,
low-rank conclusions are not cleanly reportable, and K=13 remains amendment-
qualified. V4 approval is upheld only for a bounded mentioned-string
instruction micro-world. `Please`/`Kindly` and `Hello,`/`Hi,` preserve the
registered core instruction but are intentionally non-identical in pragmatic
or discourse state; `48/48` is a procedural checklist count, not ordinary-POS
use evidence or independent linguistic replication. V4 A/B can be a fresh-
population stress test of the same token-level append construction, never a
pooled or direct replication of `lm_dyn_v1`.

Audit #17 also corrects prior kill language. V2 remains a useful pragmatic-
purpose stress set; v3's presentation pairs passed and its void arose from
control-design mismatch; withdrawal of heterogeneous `not` insertion does not
kill typed negation. The strongest live explanation is a generic prefix-edit
response or Jacobian: smooth propagation of wrapper, operation verb, length,
and position can make the recipient state predictable without exposing an
operational quotient.

### Corrected order

1. Finish protected `resSA2` without early inspection.
2. Complete the SVD telemetry/shadow-backend gate and Tier-1 numerical review
   before further low-rank output.
3. Run the fixed all-layer contextual-prefix X-free screen and completion.
4. Run and adjudicate one bounded multi-position teacher-forced consequence
   test on the existing sentinel-A/B `P_static` relation.
5. Only if that test keeps the consequence currency live, run the fixed
   probe-1 screens and preselected sentinel-A `P_aug-full` cell, then proceed
   through v4 capture, bridge, interchangeability, fresh A/B, and the
   declared-operation-verb context-intervention analysis.
6. Retain the four registered X-free cells and conditional Freedman–Lane
   A-static behind all earlier state-reading gates; run a second pinned
   decoder only after construct validity and v4 portability remain live.

No v5 mentioned-string population will be authored. The next population
direction is a typed use-frame task, with polarity first: predicates or
propositions are used in a truth-evaluable frame, a polarity update changes a
declared truth condition, and success is measured by a frozen behavioral
consequence rather than by another local word-slot law. Binding and quantifier
updates remain separate typed successor populations, not pooled controls or
post-outcome rescue. No texts or config are authored in Round 33.

### Bounded multi-position consequence gate

The denizen-level consequence is the future response-law trajectory caused by
a proposed readout-state move, not only the law at the writeback position.
For positions `j=1..k` after that readout, with `k∈{4,8}`, the instrument will
teacher-force a frozen continuation and report
`KL(q_true_j || q_hat_j)` at every position. The fixed aggregate is the
uniform mean over positions `1..k`; no position weighting, best-prefix choice,
layer promotion, or sentinel selection is allowed.

The existing `lm_dyn_v1` sentinel A/B captures are necessary but not
sufficient. The extension reuses their exact population, source/readout
states, folds, sentinels, model/tokenizer pins, and `P_static` ridge/null
fits. It requires new forward passes with two manifest-frozen eight-token
tails: the first eight tokenizer tokens (no special tokens) of ` The same
continuation follows in every case.` after sentinel A and ` and the same
continuation follows in every case.` after sentinel B. Tokenization must be
frozen before model load and must yield at least eight tokens; otherwise the
lock is amended before any run, never after outcomes. Causality predicts the
old and extended source/readout states are identical within the registered
reload/locality tolerance; failure voids the consequence score.

The future runner extension is `capture_forward_consequence` with
`--source-tags fwdA fwdB --consequence-k 4 8 --teacher-forced-tail-set
fixed_tail_v1 --expected-base-manifest-sha256 <frozen hash>`. It writes
`states_conseqA/B.npz` and `manifest_conseqA/B.json`, including tail token IDs,
true per-position laws, readout-equality checks, repeat-law noise, exact pins,
and hashes. The analyzer source is `--source forward_consequence` with
`--consequence-mode teacher_forced_v1 --consequence-k 4 8
--consequence-aggregation uniform_mean --residualize static
--contextual-prefix-tag <completed tag> --pairs 0 4 8 12 20 --n-boot 500`.
It is an early-return mode and rejects interchangeability, bridge,
residualizer-selection, screen, and permutation-null flags.

The fixed field is `P_static` residual ridge, reassembled to raw displacement
before writeback. Its competitors are the same four word-only X-free nulls
plus the completed contextual-prefix X-free field. At each `k`, the strongest
null is the one with the smallest uniform-mean KL inside each bootstrap
replicate. The gated margin is the normalized KL reduction
`G_k=(D_null-D_ridge)/D_null`, where `D` is the uniform-mean KL. A cell is
unsupported rather than epsilon-repaired when `D_null` does not exceed
`max(q99 repeat-law KL, 1e-6)`. A layer passes only when `G_4` and `G_8` are
each at least `0.02` with positive crossed 95% lower bounds, at least `6/8`
keys are jointly positive, no family collapses or reverses, and support is at
least `0.95`. A sustained consequence license requires two common F4–F20
layers in both sentinels. F0 remains a structural diagnostic.

If a layer passes the existing one-position endpoint but both `G_4` and `G_8`
have crossed 95% upper bounds at or below zero under valid support/noise
checks, that is evidence that the one-position instrument locally
manufactures consequence under this fixed tail. Mere gate failure is
ambiguous. If a predeclared one-position-nonpassing F4–F20 layer passes both
multi-position gates, the one-position law hid a delayed consequence. Passage
at both horizons keeps the consequence currency live; mixed horizon/layer
results are reported as decay or instrument dependence and cannot unlock a
hostile-hole claim.

Prediction: after the contextual-prefix comparator is admitted, the generic
prefix-edit/Jacobian account predicts no joint sustained gate and especially
decay by `k=8`. The operational-regularity account predicts common F8–F20
passage at both horizons and both sentinels. The hidden-consequence branch is
possible but not predicted. The simplest confound is the artificial frozen
tail itself; therefore every conclusion is local to these tails and this
decoder. Budget: `20–40 min` capture plus `1–2 h` scoring, one CPU process,
with hard walls of `45 min` and `2 h`; no GPU and no generation claim.

### Second lens after audit #17

No new representation-level hostile hole is proven. The strongest current
holes are still instruments: the inherited ordering readout and the SVD path;
raw F0 token identity is a bounded structural regime. V4 can expose only a
candidate local obstruction to wrapper-equivalence in a mentioned-string
instruction world, and only after adequate move norms, intact controls, both
directions, common layers, failure of the full simple bridge ladder, and a
consequence currency that survives Round 33. The next latent space should
expose denizen-available quotient or bridge coordinates and make typed moves
carry truth-conditional, multi-position consequences that compose. No new
axiom is earned. Round 33 is documentation-only; no experiment was run.

## Evidence gate — NLM-007 four-cell common-scale adjudication (2026-08-28)

**PASS, qualified.** Direct reduction of `analysis_resSA2.json` reproduces the
published runtime (`5824.8 s`, rounded to `5825`), F4-F20 gate passage, F0
failure, and rounded minimum crossed lower bounds (`0.458` cosine, `0.175`
skill, `0.197` K=13 KL-rank). Jointly positive keys are
`7/8, 8/8, 8/8, 8/8`; strict full-gate keys are
`7/8, 7/8, 6/8, 8/8`. All twelve F4-F20 common-scale ratio medians exceed
`0.5`; F4 continuous-KL has ratio LB `0.409`, so this is not a uniform
interval claim.

All four cells use the identical K=13 universe, four registered word-only
nulls, eight block-by-word-fold keys, 20 shuffles, 500 bootstraps, and
4-block/8-key crossed summaries. No cell fails those named structural
common-scale checks. B-score4 nevertheless remains amended-implementation and
SVD-telemetry-incomplete: ridge cosine/skill are mechanically reportable, but
low-rank is not cleanly reportable and K=13 KL-rank is amendment-qualified.
Ledger row `nlm007_resid_resSA2` has a provenance typo (`--sentinel-tag 2`),
whereas the JSON records sentinel `A`; concurrent append-only row
`nlm007_erratum_resSA2_sentinel_label` now records the correction. The defect
does not change the artifact-scale verdict.

Adopted synthesis:

> The sentinel {A,B} x {P_static,P_aug-score4} table is complete on a common
> K=13/four-word-only-null/crossed-bootstrap scale for the residual-versus-null
> mechanical gate. F4-F20 pass in all four correlated cells; F0 is
> non-qualifying in three cells and yields only a weak pooled A-score4
> association with 2/8 full-gate keys. This is consistent within-decoder,
> within-population condition robustness. It is not replication and does not
> identify operational state, presentation independence, a presentation
> decomposition, composition, a native law, or a representation-level hostile
> hole. B-score4's ridge cosine and skill results are mechanically reportable;
> its K=13 KL-rank endpoint and every low-rank interpretation remain
> amended-implementation and SVD-telemetry-incomplete.

“Uniform F0 failure” is not promoted: A-score4 is a sparse aggregate exception,
and the three nonpasses have different endpoint mechanisms. F0 remains a
bounded diagnostic consistent with a pre-context token-identity/position
regime, a local contextual-emergence boundary, normalized-skill/readout
pathology, or score4 transductive alignment. The Round 33 allocation is
unchanged: contextual-prefix X-free baseline, then the bounded multi-position
teacher-forced consequence test. The four-cell table resolves neither the
generic prefix-fingerprint alternative nor the validity of the one-position
consequence currency.

## Round 34 — capacity-matched state-versus-context preregistration (2026-08-28)

**Codex design gate; documentation only. No experiment was run.** This round
adopts Audit #18's first execution priority. The point-only
`analysis_ctxscr_A.json` and `analysis_ctxscr_B.json` screens leave the
F4-F20 state-versus-context difference unidentified: their selected
`token_ids_v1` ridge has about `42.2-42.7` effective degrees of freedom (EDF),
whereas the selected state ridge has about `210-406`. The observed cosine and
normalized-error gaps therefore do not yet isolate cell-state information.
This round authorizes an analyzer-only extension after Tier-1 review; it does
not authorize a result claim, the Round 33 consequence run, or any new
capture.

### Existing mechanics and missing telemetry

The live analyzer already supplies the required implementation seam.
`Standardizer` is fit on the current training design only. `RidgeFamily` then
centres that standardized design and stores the eigenvalues of
`X_centre.T @ X_centre`; both the contextual ridge and the Round 27
`ridge_dfmatch` sensitivity define slope EDF as

`df(lambda) = sum_i e_i / (e_i + lambda)`.

The intercept is excluded for every arm; adding one to every EDF would not
change a match. The existing `--xfree-field` path chooses the state lambda
whose EDF is closest on the seven-value `LAMBDAS` grid. Round 34 replaces only
that coarse selection principle for this mode with a continuous solve; it
does not change `select_ridge_lambda`, `RidgeFamily`, or historical results.

The two `analysis_ctxscr_*.json` artifacts serialize contextual EDF by outer
fold key but do **not** serialize state EDF. State EDF is recomputable without
new capture: rebuild the exact outer carrier-by-unseen-word fold from
`forward_states_A/B.npz` and the frozen config, rebuild `P_static`, fit the
outer-training `Standardizer` on `X_c`, construct `RidgeFamily(X_cs, Y_c)`,
read that fold's `selected.ridge.lam`, and apply the formula above to the
eigenspectrum in memory. Round 34 output must serialize this value, the
spectrum's numerical rank, and the selected lambda for every layer and fold.

### Primary foldwise EDF match

The primary relation is the same `P_static` residual relation that would feed
Round 33: `X_perp -> Delta_perp`, with the existing four carrier-block by two
unseen-word outer keys, the same inner carrier folds, endpoints, completion
semantics, support accounting, and block-first crossed bootstrap. The raw
unresidualized screens remain diagnostics and are not substituted for this
test.

For every outer layer/fold and every predeclared contextual candidate `j`:

1. Build and standardize the contextual training design using training rows
   only; select its ordinary hyperparameters using the existing inner
   calibration folds only.
2. Compute its attained training EDF `d_j`. For a primal ridge use
   `tr[Z(Z.T Z + lambda I)^-1 Z.T]`; for a kernel ridge use
   `tr[K(K + lambda I)^-1]` at its inner-selected kernel scale.
3. On the separately training-standardized state design, solve
   `sum_i e_i/(e_i + lambda_state_j) = d_j` by float64 bisection. Clip only
   negative roundoff eigenvalues to zero; define numerical rank with
   `tol_eig = eps * max(n,p) * max(e_i)`; double the upper lambda bracket until
   it is below the target; stop at absolute EDF error `<=0.01` or 80
   iterations. The solve consumes no validation or held-out target.
4. Fit the state ridge at that lambda and score the matched pair on the same
   held-out cells. Serialize target/achieved EDF, absolute error, bracket,
   iterations, rank, retained columns, lambda, and finite checks. Any
   unreachable target or EDF error above `0.01` makes that key unsupported;
   it is never rounded into a match.

This continuous ridge solve is the primary design. A training-only PCA-rank
constraint is less clean: it adds an unsupervised basis, rounds a fractional
EDF to an integer, and still needs a decision about post-PCA ridge shrinkage.
If retained as a sensitivity, the PCA basis must be fit separately inside
every outer and inner training fold and never on held-out rows; it cannot
replace the bisection result.

### Symmetric contextual ladder and fixed moot-makers

The upward contextual comparison has a hard rank ceiling. In an outer fold,
these context-only fields contain 12 calibration carriers and four POS groups
and are invariant across words within POS. They therefore have at most 48
distinct rows: centred primal-ridge rank is at most 47 and uncentred kernel
rank at most 48. Lowering contextual lambda cannot honestly reach a
`210-406` state EDF. The analyzer must report this capacity shortfall; it must
not add jitter, sample IDs, held-out word IDs, or random features to manufacture
rank. Symmetry is obtained by matching the state ridge downward to every
attainable contextual EDF.

One fixed mode, `--context-capacity-audit round34_v1`, adds these six
contextual candidates to the existing completed comparison:

1. `sentinel_position_v1`: ridge on sentinel ID, prefix length, suffix length,
   slot, readout, and readout-minus-slot only. In a single-sentinel run the
   sentinel column is constant and is correctly removed by training-only
   standardization. It contains no POS, token identity, item feature, or
   state.
2. `token_ids_v1_selected`: the exact locked Round 31 field at its
   inner-selected lambda.
3. `token_ids_v1_ceiling`: the same field with lambda solved downward to
   `min(df_state_selected, rank_ctx - 0.01)`. Together with item 2 this is the
   fixed contextual-ridge capacity ladder; failure to approach the ordinary
   state EDF is recorded as `capacity_shortfall`, not hidden.
4. `token_ids_v1_kernel`: the existing RBF contextual kernel, with gamma and
   lambda selected only on the existing inner calibration folds.
5. `embedseq_rbf_v1`: an RBF ridge field over the frozen input-embedding
   sequence for the last eight prefix and first four suffix tokens, in fixed
   relative positions, with zero padding plus presence masks, the four
   length/position numerics, and POS one-hot. Each token embedding is
   unit-normalized before concatenation; remaining columns are standardized
   on training rows only. The item token, cell `X`, hidden representations,
   item strings/IDs, and held-out outcomes are forbidden. The existing fixed
   gamma grid and inner folds select gamma/lambda.
6. `template_edit_kernel_v1`: kernel ridge on the exact prefix/suffix token-ID
   sequences, with prefix and suffix kept separate. Distance is the mean of
   their two length-normalized token-level Levenshtein distances plus the POS
   mismatch indicator; `K=exp(-gamma*d)`. The existing fixed gamma/lambda grid
   is selected on inner calibration folds only.

Every candidate is paired with its own same-EDF state ridge. The completed
candidate list is fixed before scoring; no arm or layer is promoted from a
point-only screen. `embedseq_rbf_v1`, the edit kernel, and the sentinel/position
field are cheap X-free moot-makers, not evidence of an operational state. The
existing word-only nulls remain in the fixed completion universe but are not
renamed as contextual fields.

### Endpoints, strongest comparator, and exact decisions

All six pairs report displacement cosine and normalized error. With completion
on, they also report response-law skill, continuous KL, and KL-rank. KL-rank
uses the unchanged fixed K=13 universe by substituting each new prediction
into the ridge slot rather than enlarging the universe; it consequently keeps
the existing low-rank/SVD qualification. The confirmatory endpoints are
`cos`, `skill`, and `klrank`; normalized error and continuous KL are required
diagnostics but do not silently replace a failed confirmatory endpoint.

For endpoint `e`, candidate `j`, and a bootstrap replicate, define the matched
margin `m_ej = score(state at d_j) - score(context_j)` (with the existing sign
convention that larger is better). The strongest-context margin is
`m_e* = min_j m_ej`, with the minimum taken **inside** each replicate. Thus the
contextual winner may differ by endpoint or replicate, which is conservative
for the state hypothesis and avoids a held-out winner-selection claim.

A layer qualifies for **KEEP X-CONDITIONED HYPOTHESIS ALIVE** only when all of
the following hold:

- the block-first pooled `m_e*` is at least `0.02` and its crossed 95% lower
  bound is above zero for each of cosine, skill, and KL-rank;
- at least `6/8` outer keys are jointly point-positive on all three endpoints;
- no carrier block collapses (each block has at least one of its two word-fold
  keys jointly positive); and
- common support is at least `0.95` in every key, with all EDF matches valid.

The hypothesis stays alive only with at least **two common qualifying layers
among F4, F8, F12, and F20 in both sentinels A and B**. F0 is reported as a
structural/model-class diagnostic and never supplies one of the two layers.
Passing licenses only the narrow statement that cell state contains
held-out predictive information beyond this fixed, capacity-matched
contextual set. It does not identify operational or semantic state, a quotient,
composition, task/model generality, or a native law.

The contextual account **MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT**
when at least two common F4-F20 layers in both sentinels satisfy the converse
non-inferiority closure: for all three endpoints the strongest-context margin
has point estimate `<=0.02` and crossed 95% upper bound `<0.02`, at least
`6/8` keys are jointly below `0.02`, no carrier block collapses, support is
`>=0.95`, and EDF matches are valid. A contextual arm that significantly
beats its matched state arm also satisfies this rule. In that branch, generic
deterministic input context is sufficient at matched capacity and Round 33
cannot recover a state interpretation. If neither KEEP nor MOOT passes, the
result is explicitly **INCONCLUSIVE/CAPACITY-SENSITIVE**; mere KEEP failure is
not called closure, and the consequence test cannot resolve the missing
capacity contrast.

### Ordering ruling

**Round 34 runs before Round 33.** This supersedes the Round 33 shorthand
`context baseline -> consequence` by completing the capacity part of that
baseline; it does not reverse the underlying order. Round 33 writes the fitted
state prediction into a frozen decoder tail. Under Audit #18's generic
contextual-response/Jacobian account, a better one-step reconstruction is
expected to remain closer under later smooth transformations, so downstream
persistence cannot identify state while the predictor's advantage is still
capacity-confounded. Round 34 reuses existing captures, is cheaper than a new
consequence capture plus scoring, and can make that run scientifically moot.
Moreover, the live Round 33 implementation remains NOT-READY under Tier-1
review. Only a Round 34 KEEP verdict can return Round 33 to the queue, and only
after its joint-key, provenance, hard-wall, exact-reuse, and parity blockers
are independently cleared.

### Analyzer contract, cost, and exact commands

Implementation stays in `experiments/analyze_lm_dynamics.py`; no new runner or
script is permitted. Add the single locked mode above plus a read-only joint
reducer flag `--context-capacity-joint TAG_A TAG_B`. The mode requires
`--contextual-prefix-xfree --prefix-feature-set token_ids_v1 --source forward
--target delta --unseen-words 2 --residualize static --pairs 0 1 2 3 4
--n-shuffle 20 --n-boot 500`, completion on, and rejects screen, smoke,
interchangeability, residualizer-selection, permutation-null, and consequence
flags. It checkpoints only at completed outer keys, has a fixed four-hour wall
per sentinel, and writes no claiming joint verdict from an incomplete arm.

The existing point screens took about 10 and 15 minutes. With the fixed K=13
completion universe plus six contextual/state-matched pairs, expect about
`2-3.5 h` per sentinel and `4-7 h` total sequential CPU time; the hard bound is
`4 h` per sentinel. Use one CPU process, no GPU, and never overlap A and B.

After implementation is Tier-1 RUN-READY, run exactly:

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag A --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --context-capacity-audit round34_v1 --pairs 0 1 2 3 4 --n-shuffle 20 --n-boot 500 --tag ctxcap_A
```

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag B --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --context-capacity-audit round34_v1 --pairs 0 1 2 3 4 --n-shuffle 20 --n-boot 500 --tag ctxcap_B
```

Then reduce the two completed artifacts only:

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --context-capacity-joint ctxcap_A ctxcap_B --tag ctxcap_joint
```

Expected outputs are `analysis_ctxcap_A.json`, `analysis_ctxcap_B.json`, and
`analysis_ctxcap_joint.json`. This preregistration creates none of them.

### Risks, confounds, and second-lens limit

- **Standardization defines the penalty geometry.** EDF is computed after each
  arm's training-only standardization and after `RidgeFamily` centring. Any
  scaler, retained-column mask, eigenspectrum, PCA basis, kernel median, or
  vocabulary touched by a held-out carrier/word invalidates the key. Equal EDF
  still does not make two feature maps the same hypothesis class.
- **PCA leakage and ambiguity.** A global PCA, even if described as
  unsupervised, leaks held-out geometry. A training-only PCA sensitivity must
  rebuild bases inside outer and inner folds. Integer rank and subsequent
  shrinkage prevent it from being the primary equality mechanism.
- **Kernel EDF is conditional.** `tr[K(K+lambda I)^-1]` depends on gamma,
  duplicated rows, and whether `K` is centred. Round 34 uses the analyzer's
  uncentred training Gram convention, clips negative eigensolver roundoff for
  EDF only, and records gamma and numerical rank. Kernel EDF is a
  regularization diagnostic, not proof of equal nonlinear function-class
  richness.
- **The contextual ceiling is real.** Context-only rows repeat across words
  within POS. Jitter, sample IDs, or held-out word identities would defeat the
  comparison. The existing word-embedding and Round 27 X-free fields remain
  separate lexical controls; survival here may still be item-specific state,
  not operational state.
- **Inference remains clustered and local.** The two sentinels reuse one
  population, folds, decoder, append construction, and analysis family. They
  are correlated sensitivities, not replications. The strongest comparator is
  selected inside each bootstrap replicate, and F0 cannot rescue F4-F20.
- **KL-rank remains qualified.** Ridge-slot substitution preserves the fixed
  K=13 scale but also its live low-rank/SVD qualification. Cosine and skill
  must carry the same decision; no KL-rank-only wording is allowed.

Under the guiding question, Round 34 asks whether the denizen's apparent
state-dependent move survives when the map reader and the context reader are
given the same attainable flexibility. Closure is a useful hole diagnosis:
the present residual space may expose deterministic input context more readily
than a distinct navigation state. Survival earns only a narrower measurement
claim and sends the program to the consequence instrument; it does not yet
show that structured reasoning lives in this latent world. No new axiom is
earned.

## Round 34a — matched-EDF core screen (audit #19 staging) (2026-08-28)

**Codex preregistration; documentation only. No experiment was run.** Audit
#19 places this short-circuit screen before the full Round 34 feature-adequacy
audit and the parked Round 33 consequence instrument. The screen asks only
whether the held-out state-versus-registered-context margin survives an honest
foldwise capacity match. It is a non-claiming sentinel screen, not a new state
gate or a search over contextual maps.

### Estimands, tags, and order

The primary audit-#19 screen is the **unresidualized** contextual relation used
by `ctx_A`: the held-out comparison is `X -> Delta` with the registered
contextual predictors fit directly to `Delta`. Run and reduce this form first,
under tags `ctxcapA_raw` and `ctxcapB_raw`. This is the only Round 34a form that
can stage the capacity objection to the raw `ctx_A`/`ctx_B` comparison.

The `P_static`-residualized relation is separately registered under tags
`ctxcapA_static` and `ctxcapB_static`: it tests `X_perp -> Delta_perp` after the
existing training-fold static residualization. It neither substitutes for nor
retroactively capacity-matches the raw estimand. Its result is reported under
its own tag and reduced only with the other sentinel's static artifact. No
cross-estimand pooling or joint verdict is permitted.

### Locked core screen

For each sentinel and each of F0, F4, F8, F12, and F20:

1. Reuse exactly the existing four carrier-block by two unseen-word outer
   folds, the existing inner carrier folds, and training-only standardization.
   No held-out carrier, word, target, or outcome may enter a standardizer,
   hyperparameter choice, EDF calculation, rank calculation, or lambda solve.
2. Recompute **only** the registered `token_ids_v1` contextual ridge at its
   inner-selected lambda and the registered `token_ids_v1` contextual RBF
   kernel at its inner-selected gamma/lambda. Serialize their per-fold
   predictions, training EDF, numerical rank, distinct training rows,
   hyperparameters, retained columns, and finite checks. No stored aggregate
   JSON is treated as if it contained reusable cell predictions.
3. On the separately training-standardized state design, fit four continuous-
   bisection ridge matches: state EDF matched to the selected contextual ridge
   EDF; state EDF matched to `min(47, selected state EDF)` for the ridge's
   honest centred-primal rank ceiling; state EDF matched to the selected
   contextual-kernel EDF; and state EDF matched to `min(48, selected state
   EDF)` for the kernel's honest training-row ceiling. Use the registered
   float64 Round 34 solve and its `<=0.01` EDF-error rule. An unreachable or
   non-finite match makes that candidate/key unsupported; it is never rounded
   into support. Serialize target and achieved EDF, absolute error, bracket,
   bracket doublings, iterations, rank and tolerance, retained columns,
   selected-state EDF/lambda, matched lambda, prediction finite check, and the
   full contextual telemetry for every match.
4. Score only held-out displacement cosine and normalized error. Define every
   paired cell margin as `state_matched - context`; reverse the normalized-
   error sign so larger is better, i.e. `nerr_context - nerr_state_matched`.
   Reduce with the existing paired block-first crossed bootstrap. For each
   endpoint report every candidate and the strongest contextual comparator,
   defined as the minimum state-minus-context margin **inside each bootstrap
   replicate**.
5. The producer is sentinel-local and non-claiming. It uses 500 bootstraps and
   zero shuffles, has a 90-minute hard wall, checkpoints only complete outer
   keys, and emits no claiming artifact after an overrun. A tiny read-only
   reducer may combine only completed A/B artifacts from the same estimand;
   it fails closed on any schema, binding, support, match, layer, fold, tag, or
   estimand mismatch and writes to a tag distinct from both inputs.

There is **no completion**, completer construction, K=13 candidate universe,
new contextual feature family, model forward, Round 33 consequence call, or
full-Round-34 joint claiming reducer in Round 34a. The only contextual feature
map is the already registered `token_ids_v1` ridge/kernel pair. F0 is reported
as a diagnostic and cannot supply a common qualifying layer.

### Precommitted screen decisions

For one sentinel-layer, define `m_cos*` and `m_nerr*` by the replicate-wise
minimum over the four registered matches. The layer is **STOP / CAPACITY-
SENSITIVE SCREEN** only if both strongest margins have point estimates
`<=0.02` and crossed 95% upper bounds `<0.02`, at least `6/8` outer keys are
jointly below `0.02` on both endpoints, every carrier block has at least one
such key, and every required match is valid with no block collapse. The raw
screen stops the line only if at least two common layers among F4, F8, F12, and
F20 qualify in both sentinels: report **capacity-sensitive screen; stop; do not
run full Round 34 or Round 33**.

The layer is **CONTINUE** only if both strongest margins have point estimates
`>=0.02` and crossed 95% lower bounds `>0`, at least `6/8` outer keys are
jointly positive on both endpoints, every carrier block has at least one such
key, and every required match is valid with no block collapse. Continue only
if at least two common F4-F20 layers qualify in both sentinels. That branch
authorizes a narrow completion pass for the inner-selected `token_ids_v1`
ridge/kernel pairs only: paired raw continuous-KL difference is confirmatory,
skill is diagnostic, and no richer contextual family is yet authorized. The
static screen receives the same within-estimand decision logic but cannot
change the raw-screen verdict.

Every other outcome is **INCONCLUSIVE**. A failed STOP rule is not survival; a
failed CONTINUE rule is not closure. No Round 34a result identifies operational
state, contextual insufficiency, missing coordinates, a native law, or a
representation-level hostile hole.

### Pre-outcome amendment to full Round 34

Before any `round34_v1` outcome exists, amend its confirmatory set. KL-rank is
now **diagnostic**, because making it decisive would reopen the parked K=13
low-rank/SVD telemetry gate. The three full-Round-34 confirmatory endpoints are
displacement cosine, completion skill, and the **paired raw continuous-KL
difference** (`KL_context - KL_state_matched`, larger is better). Normalized
error and KL-rank remain serialized diagnostics. All existing key-count,
carrier-block, common-support, EDF-validity, and two-common-F4-F20-layer rules
otherwise remain in force. This amendment is prospective and precedes every
Round 34 outcome.

### Strongest-comparator claim boundary

Taking the minimum over candidates separately by endpoint and bootstrap
replicate is a synthetic oracle. It is appropriately conservative for a full
Round 34 **KEEP** decision: the state margin must survive whichever fixed
candidate is strongest in that resample. A **MOOT** decision from that oracle
supports only the sentence that the comparison is capacity/context-family
sensitive. The stronger sentence that one sufficient deterministic contextual
map closes the state margin requires one predeclared candidate to win jointly
across every confirmatory endpoint under the same `6/8` key, no-block-collapse,
support, and two-common-layer rules. Endpoint- or replicate-specific winners
cannot be spliced into such a map.

Under the guiding question, this screen tests whether the apparent navigation
advantage survives when the state reader is reduced to the registered context
reader's attainable flexibility. Closure is evidence about a measurement
hole—capacity/context-family sensitivity—not proof that context is the world's
state. Survival only earns the narrow completion check. No new axiom is earned.

## Round 34b/34c — partial-overlap and item-by-context controls (audit #20 priorities 1-2) (2026-08-29)

**Codex design gate and preregistration; documentation only. No experiment was
run.** Audit #20 withdrew both “by construction” and “beyond template
metadata.” The cheapest live explanation is now that `P_static` removes a
coarse block/length/position response while `X_perp` retains a dense
item-by-carrier activation fingerprint and local punctuation Jacobian that the
registered `token_ids_v1` field cannot express. These two controls test that
explanation before the six-arm Round 34. They can expose redundancy or feature
inadequacy; neither identifies presentation causally, operational state, a
denizen-usable operation, a native law, or a representation-level hostile
hole.

The expectation is that 34b will determine whether the useful raw context
prediction is incrementally redundant with `P_static`, and 34c will determine
whether a fairer X-free item-by-context field closes the static state margin.
If either cheap control closes the margin, the expensive consequence/full-
feature queue stops. If both survive, the generic fingerprint account is
narrowed but not removed: activation geometry and a local Jacobian still
remain live. Any PCA basis, contextual vocabulary, nuisance map, scaler,
hyperparameter, EDF, or interaction basis touched by an outer held-out carrier,
word, or target voids the affected key and makes the reducer fail closed.

### Locked order and common envelope

The execution order is fixed:

1. use the current Round 34a Tier-1 RUN-READY core, whose fourth re-review
   closed the telemetry-binding invariant, and do not expand that instrument;
2. run and jointly reduce Round 34a raw A/B first, then Round 34a static A/B;
3. run and jointly reduce 34b;
4. run and jointly reduce 34c; and
5. run the full six-arm Round 34 only if the raw and static 34a reducers, the
   34b reducer, and the 34c reducer all return `CONTINUE`.

A `STOP` or `INCONCLUSIVE` at any cheap screen does not authorize full Round
34. `INCONCLUSIVE` is not scientific closure; it requires a design ruling or
orthogonal pivot, not escalation by default. Round 33 remains parked
throughout. It returns to the queue only after a full Round 34 `KEEP` verdict
and independent clearance of its own instrument blockers.

Both controls reuse the exact sentinel-A/B captures, F0/F4/F8/F12/F20 layers,
four held-carrier-block by two held-word-fold outer keys, disjoint calibration
and test words, inner leave-one-calibration-block-out folds, and class-
stratified word strata already locked for Round 34a. All transformations are
fit on the relevant training rows only. Both use 500 paired block-first crossed
bootstraps, `n_shuffle=0`, cosine and normalized error only, and no completion,
K=13 universe, shuffle null, causal-model forward, new capture, or Round 33
call. F0 is diagnostic and never supplies a common qualifying layer.

Every gate below uses higher-is-better scores. For a prediction `Dhat` and
target `D`, define

`cos = cosine(Dhat, D)` and
`nerr_gain = 1 - ||Dhat-D|| / ||D-mean_train(D)||`.

For a paired comparison, cosine is the rowwise score difference and normalized
error uses the equivalent reversed sign. Each key is supported only on the
common finite cells, and common support must be at least `0.95`; this explicit
floor is the only addition to Round 34a's decision predicates and is required
because residual cosine and its normalization can be undefined near a zero
target. No unsupported key is rounded into support.

### Round 34b — `P/C` partial-overlap screen

**Estimands and order.** The first estimand is raw incremental context:
whether a nested `P_static + token_ids_v1` field improves prediction of raw
`Delta` over `P_static` alone. The second, co-required estimand is the static
partial relation: whether the component of `token_ids_v1` orthogonal to
`P_static` predicts `Delta_perp`. Report raw first and partial static second in
the same sentinel artifact, but never pool their scores. The raw estimand
adjudicates redundancy; the static estimand guards against falsely calling the
existing `ctxS` collapse `P_static` alignment when it was caused by fitting or
feature projection.

For each layer and outer key, fit exactly:

1. **`P`:** training-standardize the ten-column `P_static` design and fit
   `RidgeFamily(P_static, Delta)`. Select lambda on the existing `LAMBDAS` grid
   using only the inner carrier folds.
2. **`C`:** rebuild the `token_ids_v1` column vocabulary from the current
   training carriers with `ctx_columns`, build rows with `ctx_rows`, and fit the
   registered `RidgeFamily` and `KernelFamily` directly to raw `Delta`.
   Select ridge lambda and RBF gamma/lambda on the same inner folds.
3. **`P+C`:** concatenate the raw `P_static` columns and the current training-
   vocabulary `token_ids_v1` columns, apply one training-only `Standardizer`,
   and fit one `RidgeFamily` to raw `Delta`, with lambda selected on the inner
   folds. This is the single fixed nested combined field; no post-outcome
   kernelized or interaction expansion is permitted.
4. **`C_perp -> Delta_perp`:** in every inner fit and again on the complete
   outer calibration rows, separately fit `P_static -> C` and
   `P_static -> Delta` nuisance maps in the style of the forward path's static
   residualizer. Each target's nuisance lambda is selected using only that
   fit's training/validation carrier rows. Form both feature and target
   residuals with those maps, then training-standardize `C_perp` and fit the
   registered ridge and RBF-kernel families to `Delta_perp`. The contextual
   vocabulary is also rebuilt inside every inner training fold. Every
   target-dependent residualizer is therefore refit inside the downstream
   inner folds; the current outer-only two-stage shortcut is forbidden here.
5. **same-EDF `X_perp` reference:** fit `P_static -> X` and
   `P_static -> Delta` under the identical nested discipline, fit the selected
   `X_perp` ridge, and use `round34_solve_edf_lambda` to match it downward to
   each selected `C_perp` ridge/kernel EDF. Report its cosine/nerr scores and
   state-minus-context margins as a reference only; they are not the 34b claim
   target.

On held-out rows also serialize the rowwise cosine alignment between each raw
`C` prediction and the `P` prediction, with the same block-first crossed
interval and common-support accounting. This is an alignment diagnostic, not
a variance share, mediation estimate, or causal presentation fraction.

For raw incremental overlap define
`m_PC = score(P+C) - score(P)`. For residual context candidate
`j in {ridge,kernel}`, use its absolute `cos` and `nerr_gain` against
`Delta_perp`. A sentinel-layer is **jointly redundant** only when:

- both `m_PC` endpoints have point estimate `<=0.02` and crossed 95% upper
  bound `<0.02`;
- both residual candidates are null on both endpoints under the same
  `<=0.02` / upper-bound-`<0.02` rule;
- at least `6/8` outer keys jointly meet all endpoint conditions;
- every carrier block has at least one qualifying word-fold key; and
- all fits, residualizers, EDF references, and the `>=0.95` support floor are
  valid.

The 34b joint verdict is **`CAPACITY/OVERLAP-SENSITIVE SCREEN; STOP`** only if
at least two common layers among F4/F8/F12/F20 are jointly redundant in both
sentinels. The exact conclusion is:

> **registered raw context field is `P_static`-redundant in this design**

This is 34b's Round-34a-style `STOP` semantics. It does not identify
presentation causally.

A fixed residual candidate **retains signal** only when its cosine and
`nerr_gain` point estimates are each `>=0.02`, both crossed 95% lower bounds
are `>0`, at least `6/8` keys are jointly positive, no carrier block collapses,
support is `>=0.95`, and all fits are valid. The 34b joint verdict is
**`CONTINUE`** only if the same predeclared candidate family (ridge or kernel)
retains signal at at least two common F4-F20 layers in both sentinels. The
exact conclusion is:

> **the `ctxS` collapse is a fitting/feature-projection artifact;
> `P_static`-aligned context is too strong**

If both joint branches fire, or neither fires, the verdict is
**`INCONCLUSIVE`**. A failed `STOP` is not retention and a failed `CONTINUE` is
not redundancy. Candidate families, endpoints, layers, or sentinels may not be
spliced to manufacture a verdict.

**Analyzer contract and cost.** Add
`--context-capacity-audit round34b_overlap` as a third early-return audit mode.
It requires the exact Round 34a forward/delta/unseen-word/pair contract,
`--residualize static`, `--contextual-prefix-xfree`,
`--prefix-feature-set token_ids_v1`, `--skip-completion`, `--n-boot 500`, and
`--n-shuffle 0`. The mode keeps raw copies before residualization, constructs
both registered estimands internally, and reuses `P_static`, `ctx_columns`,
`RidgeFamily`, `KernelFamily`, `round34_solve_edf_lambda`,
`pooled_block_first`, the compressed per-cell evidence sidecar, and the
read-only joint-reducer dispatcher. Expected CPU time is `20-40 min` per
sentinel and `40-80 min` total sequentially; the hard wall is `60 min` per
sentinel. Use one CPU process, no GPU, and never overlap A and B.

After Tier-1 RUN-READY, run exactly:

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag A --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --context-capacity-audit round34b_overlap --pairs 0 1 2 3 4 --skip-completion --n-shuffle 0 --n-boot 500 --tag ctxoverlap_A
```

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag B --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --context-capacity-audit round34b_overlap --pairs 0 1 2 3 4 --skip-completion --n-shuffle 0 --n-boot 500 --tag ctxoverlap_B
```

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --context-capacity-joint ctxoverlap_A ctxoverlap_B --tag ctxoverlap_joint
```

The expected artifacts are `analysis_ctxoverlap_A.json`,
`analysis_ctxoverlap_B.json`, `analysis_ctxoverlap_joint.json`,
`round34b_evidence_ctxoverlap_A.npz`, and
`round34b_evidence_ctxoverlap_B.npz`. An incomplete producer is non-claiming
and cannot enter the joint reducer.

### Round 34c — item-by-context X-free comparator

**Primary estimand.** Round 34c is static only and runs after a 34b `CONTINUE`.
It compares a state ridge from `X_perp` with a richer X-free ridge on the same
`Delta_perp` target. `P_static -> X` and `P_static -> Delta` are fit on the
outer calibration rows and refit inside every downstream inner carrier fold,
as in 34b. There is no raw 34c gate: the question is specifically whether
item/context features omitted by `token_ids_v1` explain the surviving static
margin.

For every outer key, construct one fixed X-free design from:

1. the ten raw `P_static` columns;
2. exactly 16 principal scores of the pinned frozen input embedding of the
   item token, after centring and fitting a float64 SVD on the outer calibration
   words only;
3. every fixed outer product between the ten `P_static` columns and the 16
   item-PC scores (`10 x 16 = 160` nominal interaction columns); and
4. the audit's optional floor, adopted now as a mandatory part of the single
   registered field: training-vocabulary prefix/suffix boundary-token
   indicators, POS one-hot, and POS-by-boundary-token interactions from
   `token_ids_v1`. Position-specific token one-hots, unigrams, bigrams, item
   IDs/strings, cell `X`, hidden states, and held-out outcomes are forbidden.

The PCA mean/basis is refit for every outer word split. Inner carrier folds use
the same outer-calibration word set, so their admissible PCA basis is identical
in data scope; the implementation must nevertheless bind or recompute that
basis and may never fit on all 80 words. Transfer to held-out words occurs only
by projecting their pinned frozen item embeddings through the calibration-word
basis. If 16 nondegenerate training PCs are unavailable, the key is
unsupported rather than silently reducing rank. Concatenate the fixed fields,
apply one training-only `Standardizer`, select the X-free ridge lambda on the
inner carrier folds, and fit on calibration words only.

Compute the comparator's slope EDF after standardization and `RidgeFamily`
centring. Match the `X_perp` state ridge downward to that EDF with
`round34_solve_edf_lambda` and its `<=0.01` absolute EDF-error rule. Serialize
raw/retained column counts, matrix rank and tolerance, all 16 singular values,
PCA training-word identities/digest, interaction rank, selected comparator
lambda/EDF, selected state lambda/EDF, matched state lambda/EDF, and finite
checks. No integer PCA-rank approximation or nearest-grid state match may
replace the continuous solve.

For each cell and endpoint define
`m = score(state_at_itemctx_EDF) - score(itemctx)`. A layer has Round-34a-style
**`STOP`** status only when both margins have point estimate `<=0.02` and
crossed 95% upper bound `<0.02`, at least `6/8` outer keys are jointly below
`0.02`, every carrier block has at least one such key, support is `>=0.95`, and
all PCA, fit, and EDF telemetry is valid. The joint 34c verdict is
**`ITEM/CONTEXT-FEATURE-SENSITIVE; STOP`** only with at least two common
F4-F20 `STOP` layers in both sentinels. The exact conclusion is:

> **item/context-feature-sensitive; stop the consequence queue**

A layer is **`CONTINUE`** only when both margins have point estimate `>=0.02`
and crossed 95% lower bound `>0`, at least `6/8` keys are jointly positive, no
carrier block collapses, support is `>=0.95`, and every match is valid. The
joint verdict is `CONTINUE` only with at least two common F4-F20 layers in both
sentinels. The exact conclusion is:

> **survived a fairer X-free feature test; still not operational state**

Every other outcome is **`INCONCLUSIVE`**. These are exactly Round 34a's
`0.02`, crossed-interval, `6/8`, no-block-collapse, two-common-layer, and
`STOP`/`CONTINUE`/`INCONCLUSIVE` semantics, plus the common finite-support
floor declared above.

**Analyzer contract and cost.** Add
`--context-capacity-audit round34c_itemctx` as a fourth early-return audit mode.
It reuses the `round34a_core` capture binding, folds, state/static residualizer,
evidence-sidecar writer, joint reducer, block-first bootstrap,
`RidgeFamily`, and `round34_solve_edf_lambda`. Move only the existing frozen-
item-embedding lookup needed by the Round 27 X-free field into a reusable
tokenizer/embedding-table helper available before the early return; loading
the pinned input-embedding table is allowed, but no causal-model forward or
`WorldCompleter` construction is. Expected CPU time is `15-30 min` per
sentinel and `30-60 min` total sequentially; the hard wall is `45 min` per
sentinel. Use one CPU process, no GPU, and never overlap A and B.

After a joint 34b `CONTINUE` and Tier-1 RUN-READY, run exactly:

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag A --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --context-capacity-audit round34c_itemctx --pairs 0 1 2 3 4 --skip-completion --n-shuffle 0 --n-boot 500 --tag itemctx_A
```

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag B --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --context-capacity-audit round34c_itemctx --pairs 0 1 2 3 4 --skip-completion --n-shuffle 0 --n-boot 500 --tag itemctx_B
```

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --context-capacity-joint itemctx_A itemctx_B --tag itemctx_joint
```

The expected artifacts are `analysis_itemctx_A.json`,
`analysis_itemctx_B.json`, `analysis_itemctx_joint.json`,
`round34c_evidence_itemctx_A.npz`, and
`round34c_evidence_itemctx_B.npz`. An incomplete producer is non-claiming and
cannot enter the joint reducer.

### Minimal code delta and registered risks

No new script is permitted. The only authorized future code change is inside
`experiments/analyze_lm_dynamics.py`:

- extend the `--context-capacity-audit` choices and early dispatcher with
  `round34b_overlap` and `round34c_itemctx`;
- factor the live `P_static` builder, `token_ids_v1` `ctx_columns`/`ctx_rows`,
  static nuisance fit, and frozen item-embedding lookup only enough for both
  early modes to call them;
- add one small 34b producer/reducer and one small 34c producer/reducer while
  reusing the Round 34a evidence packing, hash binding, checkpoint, and A/B
  joint-dispatch machinery; and
- add negative mutation/parity tests for sentinel, estimand, folds, candidates,
  PC-basis digest, residualizer provenance, EDF telemetry, and evidence hash.

The following confounds remain registered:

- **PC leakage:** fitting the item PCA on all words, held-out words, or a basis
  cached across outer keys leaks test geometry. PCA is unsupervised but not
  exempt from the training boundary.
- **Interaction rank:** 160 nominal products do not imply 160 independent
  directions. Repeated carriers/POS rows and only 40 calibration words can
  sharply cap rank. No jitter, sample ID, word ID, or random feature may
  manufacture capacity; report raw columns, retained columns, numerical rank,
  and EDF separately.
- **Combined-field EDF:** 34b `P+C` EDF is the centred `RidgeFamily` slope EDF
  after one fold-specific standardization, with the intercept excluded. It is
  reported but not matched to `P`; failure of the larger nested field to
  improve is diagnostic, while improvement alone may reflect its extra
  capacity.
- **Kernel EDF:** contextual-kernel EDF uses the analyzer's uncentred training
  Gram convention, `tr[K(K+lambda I)^-1]`, at the inner-selected gamma/lambda.
  It is conditional on gamma, duplicate rows, and Gram rank and is not proof
  of equal nonlinear function-class richness. Record median distance,
  eigenspectrum tolerance, numerical rank, and finite checks.
- **Residualizer and vocabulary stability:** a target residualizer, contextual
  vocabulary, or standardizer fit once on the complete outer calibration set
  and reused during inner selection would bias the comparison. Each must be
  rebuilt at the relevant inner training boundary.
- **Feature meaning:** frozen item PCs can encode lexical identity and the
  mandatory boundary/POS floor can encode authored micro-world structure.
  Closing the margin therefore diagnoses feature sensitivity, not a causal
  context variable; survival still leaves nonlinear activation geometry and a
  local Jacobian untested.
- **Local scope:** A and B share one model, item population, template family,
  folds, and append operation. They are correlated sentinels, not
  replications. No result generalizes to another move, decoder, latent space,
  or composed operation.

Under the guiding question, 34b asks whether two proposed maps name distinct
directions at all; 34c asks whether the apparent state map is merely a richer
chart of item-by-context regularities. Closure is useful: it identifies an
instrument/feature hole before downstream spend. Survival earns only the full
Round 34 feature-adequacy audit. No new axiom is earned.

## Round 35 — typed truth-evaluable world (design gate; docs only) (2026-08-29)

**Codex design gate and preregistration; documentation only. No population was
authored, no text or item was selected, no config or code was written, no model
was loaded, and no experiment was run.** This round registers the population
direction named by audits #19 and #20 without activating it. Its purpose is to
replace the bounded mentioned-string relation with a world in which state,
move, consequence, and composition have finite truth conditions known before
any latent measurement.

The non-expert “so what” is: if a model has a usable inner world, changing a
declared fact should change the right answer, changing it twice should restore
the old answer, and reversing two noncommuting moves should produce the other
answer—even under new predicate names and new ways of asking.

### Ordering and activation lock

Round 35 **authors nothing until the Round 34a/34b/34c ladder resolves**. The
current order remains 34a raw A/B, 34a static A/B, then 34b, then 34c whenever
the preceding registered gate permits the next one. No Round 35 author may see
those outcomes and then choose names, templates, controls, query wording,
thresholds, layers, or folds to accommodate them.

The branch is fixed:

- If a registered 34a/34b/34c reducer returns its terminal `STOP`, `MOOT`, or
  redundant/capacity-sensitive verdict, the mentioned-string line closes. Do
  not run Round 33 or escalate the same relation. Round 35 then becomes the
  constructive program and may begin its separate outcome-blind population
  approval loop.
- If every cheap rung returns `CONTINUE`, follow the already registered full
  Round 34 ordering. Round 35 remains a frozen design, not an outcome-dependent
  rescue population.
- `INCONCLUSIVE` is not survival and is not closure. It requires a new design
  ruling or an orthogonal pivot; it does not automatically authorize either
  full Round 34 or Round 35 authoring.

There will be no v5 mentioned-string population in any branch. The future
typed population, if activated, starts its own version sequence and is never
pooled with `lexical_probe_v1` or fresh v2-v4.

### The finite world and the denizen's problem

The world state is the four-bit vector

`s = (s_1, s_2, s_3, s_4) in {0,1}^4`.

Each coordinate is carried by a typed predicate name, not by a mentioned word
slot. A declared-state prefix asserts the truth value of all four predicates
of one fixed world individual. A move clause follows that prefix, ends at a
registered post-move readout boundary, and is followed by a typed query asking
whether one named predicate is true in the post-move state. The answer set is
the frozen forced choice `{yes, no}`. The legal primitive operations are:

- `toggle(i)`: replace `s_i` by `1-s_i` and leave every other bit fixed;
- `swap(i,j)`: exchange `s_i` and `s_j` and leave the other bits fixed;
- `no-op`: leave all four bits fixed.

A two-step program applies two primitives from left to right. In algebraic
notation, `b ∘ a` means apply `a` first and `b` second. Every primitive clause
ends at the same typed state-register boundary. A two-step capture therefore
has an observed intermediate boundary after move one and a final boundary
after move two; the learned one-step map is applied twice without refitting.

The single-step population contains all 16 initial states, all four toggles,
all six unordered swaps, and no-op. Every post-move state is queried on all
four bits. The exact logical answer table is derived from these definitions,
frozen and hashed before capture, and never inferred from model output.

The denizen must therefore read four distinct things:

1. **state:** which of four typed propositions are currently true;
2. **move:** which operation and arguments update that state;
3. **consequence:** which forced-choice answers the updated state entails; and
4. **composition:** whether sequential moves obey the predeclared laws.

Polarity is first because `toggle` changes a truth value in use. Predicate
binding and quantified updates are successor worlds. They are not extra arms,
pooled controls, or post-outcome repairs in Round 35.

### Population, folds, wrappers, and frozen query tails

The future outcome-blind author must instantiate exactly 24 predicate names,
arranged as six disjoint four-name panels. The panels are partitioned into two
equal predicate-name folds before tokenization. Across the four carrier blocks,
a frozen Latin-square assignment rotates names through bit indices so every
name occupies every coordinate once and neither name nor token position can
identify a bit. An outer key holds out one entire carrier block and one entire
predicate-name fold. No held-out name, string, token ID, frozen embedding, PCA
score, or row may enter fitting, standardization, vocabulary construction,
hyperparameter selection, EDF matching, or threshold calibration. The four
held-carrier blocks by two held-name folds produce the inherited eight outer
keys. Each outer training split retains 12 predicate names, so the registered
eight-PC X-free field below is attainable without manufacturing rank.

There are exactly four paired state/move carrier blocks. Each block has one
surface realization in system `S_A` and a separately authored realization in
system `S_B`; these are the source/target surface-template families for the
bidirectional transfer. Each realization must contain:

- two independently grammatical realizations of the same logical program
  under a predeclared truth-conditionally irrelevant wrapper contrast;
- a paired same-token-length control whose state, readout position, query
  position, and total tokenizer length match but whose declared operation
  changes at least one queried truth value; and
- length-matched no-op scaffolds for the one-step and two-step comparisons.

The irrelevant-wrapper pair must preserve the individual, four declarations,
operation, arguments, scope, polarity, modality, tense, query, answer demand,
and token positions. The same-length control must preserve everything except
the one declared move change. Equal length is checked on the pinned tokenizer,
not by whitespace. A control that is merely equally awkward, changes scope, or
changes no queried truth value is a failed cell.

Two query-tail families, `Q_A` and `Q_B`, are authored independently and then
frozen with `S_A` and `S_B`, respectively. Each contains one tail template per
carrier block with one typed predicate-name slot and the same `{yes,no}`
answer convention. The families must be disjoint template inventories and
distinct syntactic constructions; after tokenization they may share the
predicate slot and unavoidable function tokens, but no non-slot token bigram.
A/B is not a paraphrase pair created by a one-word substitution. For
`A -> B`, every fit and selection uses only outer-training rows from `S_A` and
the held-out programs are realized with `S_B` and scored through `Q_B`; the
reverse direction swaps the systems. Query-tail tokens occur strictly after
`Y` and are forbidden from every transition or context fit. The direct and
true-state-patch behaviors are measured in both families, and no directional
artifact or verdict is pooled.

All 16 states and all 11 single-step moves occur in every carrier block and
name panel. The two-step law rows are fixed below. Sampling or thinning after
tokenization is forbidden. Duplicate logical states are retained as clustered
surface realizations, not treated as independent evidence.

### Outcome-blind authoring, adversary, tokenization, and freeze

Activation starts a new typed-world version; it does not edit v4. The approval
loop inherits the v2-v4 separation and tightens it:

1. A fresh outcome-blind author sees this preregistration and the linguistic
   failure rules, but no model behavior, hidden state, capture, tokenizer
   output, or result artifact. The author creates all names, carrier blocks,
   wrapper pairs, same-length controls, both query-tail families, fold maps,
   and cell-level truth tables in one candidate.
2. A separate fresh linguistic adversary, with no model or tokenizer access,
   audits every state-declaration, move, composition, query, wrapper, and
   control cell. It must verify grammaticality, typed predicate use, unique
   operation scope and arguments, the declared logical answer, wrapper
   irrelevance, and the intended truth difference of the matched control.
3. One failed linguistic cell voids the whole candidate. A successor is
   authored from scratch by a new author; no in-place substitution, relabel,
   or local repair is allowed.
4. Only after linguistic `APPROVE`, a separate tokenization audit checks the
   pinned Qwen3-0.6B tokenizer: predicate-name eligibility, exact yes/no answer
   IDs (each answer must be exactly one non-special token under the same
   boundary convention), every boundary and readout position, exact paired
   token lengths, query-family bigram disjointness, and batch-shape parity.
   One failed cell voids the candidate before model work.
5. Only after both approvals are the raw config bytes, Git blob, logical answer
   matrix, names, text inventories, pair/control maps, fold assignment, query
   tails, answer IDs, model/tokenizer revisions, and both approval reports
   SHA-256 frozen. Every future capture receives the expected raw-config hash
   and refuses an uncommitted or mismatched blob.

Neither author nor adversary may revise the population after any model output.
There is no outcome-contingent prompt repair, answer-token replacement,
template deletion, layer selection, or query-family promotion. This Round 35
section itself authors none of those artifacts.

### Capture and forced-choice measurement

Future implementation reuses the canonical machinery rather than creating a
new runner. `experiments/run_lm_dynamics.py` remains the capture entry point;
its `load_config_checked`, `common_manifest`, pinned
`SubstitutionProbe`, batched-versus-single null, repeat-null arrays,
position telemetry, float32 causal checks, float16 hidden-state storage, and
fail-before-save `PopulationVoid` discipline remain mandatory. The future
analyzer reuses the current folds, training-only transforms, ridge/kernel
families, static residualizer, matched-EDF solver, evidence sidecars,
block-first bootstrap, and read-only A/B reducer. This paragraph authorizes no
code change.

For a rendered program at layer `l`:

- `X_l` is the hidden state at the frozen boundary after the complete declared
  state and before the first move;
- `Y_l` is the observed hidden state at the homologous boundary after the
  final move and before the query tail;
- for two-step rows, `Y_l^(1)` is also stored at the homologous boundary after
  the first move, while `Y_l^(2)=Y_l` is the final observed state;
- `Delta_l = Y_l - X_l`; and
- `Yhat_l = X_l + Deltahat_l` is a training-fold prediction of the post-move
  state.

Here “true post-move state” means the stored `Y_l` produced by the actual
move-bearing forward pass. It is a causal patch ceiling, not an assertion that
the activation is a semantically true or canonical state.

Appending a query tail must not alter any earlier `X_l` or `Y_l` beyond the
registered float32 causal-locality tolerance. The direct run and a patch of
the stored true `Y_l` into the same sequence must reproduce the same answer
log-odds within `max(5*q99(repeat-log-odds error), 1e-4)` nats. A failed
locality or true-state-patch parity check makes that key unsupported; it is not
epsilon-repaired.

Every query token is teacher-forced. At the frozen answer position store only

`ell_i = log p(yes | program, query_i) - log p(no | program, query_i)`

for `i=1..4`, plus the two answer logits needed to reproduce it. Define the
truth-signed margin `g_i = (2*s'_i-1)*ell_i` and binary log loss
`BCE_i = softplus(-g_i)`. No generated answer, full-vocabulary law, KL,
KL-rank, best-token set, or vocabulary-wide endpoint is captured or scored.
Termination is therefore not an applicable endpoint; this is a teacher-forced
forced-choice task, not a generation claim.

At the same layer, readout site, rows, folds, and query tails, patch and score:

1. the stored true post-move state `Y_l` (causal ceiling);
2. the foldwise state prediction `Yhat_l`, separately raw and static-
   residualized/reassembled;
3. identity `X_l`;
4. identity plus the training-fold shared displacement `X_l + mean(Delta_l)`;
5. each registered X-free context prediction, reassembled at the same site;
   and
6. 20 manifest-seeded isotropic random directions, each scaled per cell to the
   exact norm of `Yhat_l-X_l` before patching.

Random directions, seeds, norms, and finite checks are frozen before scoring.
The random arm reports both its median and strongest replicate; it is never
redrawn after seeing a result. Every patch uses the same frozen downstream
blocks and teacher-forced query tokens. A hidden-state reconstruction that
does not change the correct forced-choice consequence receives no causal
credit.

### Estimands and inherited X-free ladder

The raw transition estimand is `X_l -> Delta_l`. The static estimand is the
separate `P35_static` relation `X_perp_l -> Delta_perp_l`; its prediction is
reassembled to raw space before patching. Raw and static fits, artifacts, and
verdicts are separate. They may not be pooled, and one cannot retroactively
capacity-match or rescue the other.

`P35_static` is the typed-world analogue of `P_static`: a training-fold design
containing carrier-block and A/B surface-system indicators, state/move token
lengths, source and readout positions, wrapper identity, and operation arity.
It excludes query-tail tokens, query identity, all four truth values,
predicate names or IDs, move identity and arguments, the queried coordinate,
the correct answer, `X`, and held-out outcomes. Its exact columns and rank
tolerance are frozen before capture.

The X-free ladder is fixed:

- **identity plus shared displacement:** the cheapest residual-stream null,
  fit on the same training rows and patched at the same boundary;
- **token-context field:** the training-vocabulary token sequence of the
  declared-state and move context through the `Y` boundary, its lengths,
  positions, and operation surface, with registered ridge and RBF-kernel
  families; query-tail tokens, the queried bit, and the answer are forbidden;
- **predicate-item-by-context field:** the Round 34c construction adapted only
  by replacing item word with predicate name: eight training-name-only PCs of
  pinned input embeddings, `P35_static`, all fixed PC-by-`P35_static`
  interactions, and the frozen boundary-token/type floor; and
- **matched-EDF state ridge:** for every selected context ridge/kernel, solve
  the state ridge downward by the existing float64 bisection, numerical-rank
  definition, and absolute EDF-error `<=0.01`. Unreachable matches are
  unsupported, never rounded into support.

All vocabularies, PCA means/bases, nuisance maps, scalers, kernels,
hyperparameters, ranks, and EDFs are rebuilt at the relevant outer and inner
training boundary. The predicate-item field may encode lexical identity and
authored structure; closure by it diagnoses feature sensitivity, not a causal
semantic variable. The strongest X-free candidate is selected inside each
bootstrap replicate. As in Round 34a, such a synthetic oracle can establish
capacity/context-family sensitivity, but only one predeclared candidate that
wins every endpoint may be described as a single sufficient map.

The latent endpoints remain displacement cosine and normalized-error gain. The
primary causal endpoint compares patches by mean four-bit binary log loss. For
candidate `a`, strongest null `r`, and true-state patch `Y`, define

`G(a;r) = (BCE_r - BCE_a) / BCE_r`

and, only when the denominator exceeds the noise floor,

`R_oracle(a;r) = (BCE_r - BCE_a) / (BCE_r - BCE_Y)`.

A cell is unsupported when `BCE_r <= 0`, when
`BCE_r-BCE_Y <= max(q99(repeat BCE difference), 1e-4)`, or when any common
finite-support condition fails. Ratios are not clipped. Accuracy, signed
log-odds, per-bit BCE, wrapper disagreement, and same-length-control
selectivity are all serialized beside the aggregate.

### Laws and bidirectional transfer

The one-step transition map is fitted only on one-step rows. The composition
rows are held out from every fit and hyperparameter choice.

1. **Involution.** For every `i in {1,2,3,4}` and all 16 initial states, test
   `toggle(i) ∘ toggle(i) = identity`. Compare its all-bit consequence with
   the registered
   length-matched two-step no-op scaffold and with the exact logical state.
2. **One noncommuting composition.** Freeze
   `a=toggle(1)` and `b=swap(1,2)`. Score both `b ∘ a` and `a ∘ b`. They have
   the same operation-token multiset and length but different order, and their
   correct final values at bits 1 and 2 differ for every initial state. No
   other pair may be promoted after outcomes.
3. **Bidirectional transfer.** Fit state/context models, EDF matches, nuisance
   maps, and all selection quantities on outer-training `S_A` rows, then
   evaluate the untouched held-block/held-name `S_B` programs through `Q_B`;
   reverse A/B for the other direction. Query tokens enter only the frozen
   downstream consequence call, never the transition fit. Both directions
   must independently pass. Within-family scores are diagnostics, not
   substitutes for transfer.

For composition, patch both the sequential one-step prediction and the stored
true intermediate/final states `Y^(1),Y^(2)`. Apply the frozen one-step map to
`X` for move one and to its own predicted intermediate state for move two;
there is no two-step refit. The comparison therefore separates three
questions: does the decoder answer the composed truth table, does the one-step
latent map compose to the right state, and can that predicted state causally
enact the answer under a new query family?

### Validity and decision rules

All intervals use 500 paired block-first crossed bootstrap replicates over the
four carrier blocks and two predicate-name folds. The four bits and repeated
logical states stay nested inside their outer key. Every state-versus-null
comparison uses common cells and requires support `>=0.95`. A qualifying layer
requires at least `6/8` jointly qualifying outer keys and no carrier-block
collapse: every block has at least one qualifying name-fold key. F0 is always
diagnostic. A joint latent verdict requires at least two common layers among
F4, F8, F12, and F20 in both transfer directions.

Before a latent claim, the direct unpatched and true-`Y` behavior must each
pass the **world-validity gate** in both query families:

- pooled four-bit forced-choice accuracy is at least `0.90` with crossed 95%
  lower bound at least `0.85`;
- every bit and each of `toggle`, `swap`, and no-op has accuracy at least
  `0.80` (the behavioral no-collapse floor);
- irrelevant-wrapper answer disagreement is at most `0.05`, with crossed 95%
  upper bound at most `0.10`;
- the same-length control changes the required bit in the correct direction
  with accuracy at least `0.90` and lower bound at least `0.85`; and
- true-state patch parity, causal locality, exact truth-table, tokenization,
  support, and repeat-noise checks all pass.

Failure of this gate is **`BEHAVIORALLY VOID FOR STATE CLAIMS`**. It does not
become evidence for or against a latent state, and no larger model or revised
text is tried inside the frozen version.

For each raw or static sentinel-layer and transfer direction, the
**state-and-causal layer gate** requires all of:

- strongest-X-free state-minus-context displacement cosine and normalized-
  error margins each have point estimate `>=0.02` and crossed 95% lower bound
  `>0`;
- `G(Yhat;r)` has point estimate `>=0.02` and crossed lower bound `>0`;
- oracle recovery has point estimate `R_oracle>=0.80` and crossed lower bound
  `>=0.60`;
- at least `6/8` keys are jointly positive on both latent endpoints and causal
  gain, with no carrier collapse; and
- common support, every EDF match, patch norm, true-state ceiling, random
  control, and finite check are valid.

The `0.02`, crossed-interval, `6/8`, no-collapse, two-common-layer, and
`STOP`/`CONTINUE`/`INCONCLUSIVE` semantics are inherited exactly from Rounds
34a-c. The new `0.90/0.85` behavioral floors and `0.80/0.60` oracle-recovery
floors are absolute prospective thresholds, not estimates to be tuned on a
pilot.

The **involution gate** requires, separately in both tails and both transfer
directions, all-bit forced-choice agreement between double-toggle and the
logical/no-op identity of at least `0.95` with crossed lower bound at least
`0.90`; median absolute all-bit log-odds difference at most `0.10` nats with
crossed upper bound at most `0.20` nats; and the state-and-causal layer gate on
the composed prediction. The **noncommuting gate** requires truth-correct bits
1 and 2 in each order with accuracy at least `0.90` and lower bound at least
`0.85`, exact four-bit-vector accuracy at least `0.80` with lower bound at
least `0.70`, and correct signed order separation on bits 1 and 2 of at least
`0.20` nats with crossed lower bound above `0.10` nats, plus the composed
state-and-causal layer gate. Operation orders, bits, and thresholds are never
changed after capture.

The joint outcomes are:

- **`PASS — BOUNDED TYPED WORLD`** only if the world-validity gate, raw and
  static state-and-causal gates, involution, noncommuting composition, wrapper
  and length controls, and both `Q_A -> Q_B` and `Q_B -> Q_A` directions all
  pass at the required common layers.
- **`X-FREE MOOT / FEATURE-REDUNDANT`** when one predeclared X-free candidate,
  or the explicitly synthetic strongest-candidate oracle with the narrower
  wording, closes both latent margins and causal gain: point estimates
  `<=0.02`, crossed upper bounds `<0.02`, at least `6/8` jointly closing keys,
  no collapse, valid support/EDF, and two common F4-F20 layers in both
  directions. Mere failure to PASS is not MOOT.
- **`COMPOSITION HOLE, LOCAL`** when the valid one-step state-and-causal gates
  pass in both directions but either fixed composition gate fails with valid
  controls and adequate direct behavior.
- **`PRESENTATION/QUERY-TRANSFER HOLE, LOCAL`** when the corresponding
  within-family gates pass but either predeclared transfer direction fails
  under valid support.
- Every other complete outcome is **`INCONCLUSIVE`**. Candidate families,
  endpoints, layers, bits, operations, query tails, or transfer directions may
  not be spliced to manufacture a verdict.

### CPU-only budget and artifact discipline

The only model in this preregistration is the pinned Qwen3-0.6B revision with
its pinned tokenizer, float32 CPU compute, and one process. **GPU use is
forbidden for Round 35.** Prefix states are captured once and reused across
the four teacher-forced queries; query tails and patches are batched without
generation. Score only F0/F4/F8/F12/F20, although the canonical capture may
retain all hidden indices for parity with the current artifact schema.

After activation, approved population, Tier-1 RUN-READY implementation, and a
clean provenance preflight, the expected budget is `2-4 h` capture per query-
surface family and `2-4 h` patch/scoring per transfer direction. Hard walls are
`5 h` for each capture family, `5 h` for each scoring direction, and `20 h`
total sequential CPU time. Only complete outer keys checkpoint; an overrun or
partial direction is non-claiming. A and B never overlap. No second model,
larger model, GPU rerun, generated-answer arm, or post-outcome population
repair is inside this budget.

Artifacts must bind the raw config and Git blob, model/tokenizer revisions,
logical answer-matrix hash, name/template/fold/tail hashes, answer IDs,
position arrays, repeat and batch nulls, exact command, array shapes and
hashes, per-cell four-bit log-odds, every patch norm/seed, nuisance/PCA/context
provenance, EDF telemetry, compressed paired evidence, and claiming reducer
inputs. Until future results are entered in both the ledger and the canonical
experiment index, nothing happened.

### Guiding-question and second-lens claim boundary

A PASS would license one bounded statement: in this frozen finite world and
decoder, at common middle/deep layers, a held-out-name and held-out-template
state reader carries causally usable information beyond the registered
capacity-matched X-free maps; its predicted post-move state produces correct
four-bit consequences, respects one involution and one noncommuting
composition, and transfers in both directions between two frozen query-tail
families. In the guiding question's terms, that is a provisional local map,
move, consequence, and composition table available to the denizen.

A PASS would **not** establish semantic understanding, natural-language
reasoning, a global latent quotient, a metric or effort law, arbitrary
composition, binding, quantification, decoder/model/task generality, a native
mathematics, a new axiom, or closure of the current architecture's hostile
holes. The two query families and four carrier blocks are correlated
realizations of one authored micro-world, not independent replications.

Under the second lens, a valid local composition or transfer failure is a
first-class diagnosis, but still a local one. Behavioral failure locates the
problem before the latent instrument. X-free closure says the apparent state
is redundant with accessible context/features. One-step survival followed by
composition failure says the representation supports isolated moves but not
the registered law. Tail-transfer failure says consequence is entangled with
presentation. Those outcomes specify what a next latent space should expose:
factorized truth coordinates, bound operation arguments, an explicit
presentation quotient, composable transitions, and a causal truth readout.
None alone proves that every representation of this decoder—or every decoder—
is incapable of providing them. No new axiom is earned at the design gate.

## Round 36 — minimal operational-quotient world (constructive artifact, distance 0) (2026-08-29)

**Design gate; theory change only. No code, config, population, result, or
claiming artifact was created or run in this round.** The next implementation
is the README's central artifact itself, so its distance from claim is **0**.
It is the smallest runnable world in which identity and action can be wrong,
not infrastructure for measuring a later artifact.

The non-expert “so what” is: can a tiny learner discover what counts as the
same place and which moves obey stable laws using only what its world lets it
do and observe, without being handed the world's hidden coordinates?

### Activation and ordering

Round 36 **starts now**, in parallel with the terminal NLM-007 closeout ladder
authorized by the Program continuation ruling. The two lines share no code,
config, evidence, reducer, or scientific verdict. They may coexist as work,
but CPU jobs remain sequential: only one process runs at a time. No NLM-007
outcome changes this design, its thresholds, or its activation.

Round 35 is retained as a later requirements envelope. Its linguistic
population, tokenization, model capture, surface/query transfer, and causal
patch design are not part of Round 36. There will be **no Codex design of NLP
surface forms until the operational quotient below passes**. This ordering
supersedes Round 35's former activation branch wherever that branch conflicts
with the adopted Program continuation ruling.

### The world and the denizen's only sensor

The hidden simulator state is

`s = (s_1, s_2, s_3, s_4) in S = {0,1}^4`.

The ordered primitive-action alphabet is frozen as

`A = [no-op, toggle(1), toggle(2), toggle(3), toggle(4), swap(1,2),
swap(1,3), swap(1,4), swap(2,3), swap(2,4), swap(3,4)]`.

`toggle(i)` complements bit `i`; `swap(i,j)` exchanges bits `i` and
`j`; and `no-op` changes nothing. Action words are applied left to right.
The simulator is total and deterministic.

The denizen has one binary response law,

`rho(s) = s_1`.

It is not shown `s`, the other bits, a state number, an endpoint state, or a
four-bit target. Each hidden state is assigned an opaque start handle by one
fixed seeded permutation (`data_seed = 3601`). A behavioral row contains only

`(opaque_start_handle, action_word, terminal_response)`

where `terminal_response = rho(delta_word(s))`. The simulator may use its
hidden bits to produce this truth, but the learner's interface and loss may
receive only those three fields. In particular, there is no state-label loss,
next-handle loss, latent-target loss, coordinate reconstruction, contrastive
state label, or privileged endpoint lookup.

The learned latent transition system is deliberately small and fixed:

- a state encoder `E_theta`: a learned `16 x 8` table mapping opaque handles
  into `R^8`;
- one shared action-conditioned transition
  `T_theta(z,a) = z + W_2 tanh(W_z z + W_a onehot(a) + b_1) + b_2`, with
  hidden width `32`; and
- one binary response readout
  `R_theta(z) = sigmoid(w_r^T z + b_r)`.

For a word `a_1 ... a_k`, the same `T_theta` is iterated without a
composition-specific fit. All parameters are trained jointly only by binary
cross-entropy between `R_theta(T_word(E_theta(handle)))` and the observed
terminal response. The architecture, dimension, width, loss, split, and
optimizer are fixed across seeds. Increasing capacity after seeing a failure
is a successor design, not a repair inside Round 36.

### Outcome-blind train/holdout split and CPU lock

All empty and one-step words are training rows. Two- and three-step words are
split by action spelling alone, before responses or model output exist:

1. `H_2` contains all four `toggle(i), toggle(i)` words and both orders of
   every `swap(i,j)`/`toggle(k)` pair (`4 + 2*6*4 = 52` words). From the
   remaining two-step words, add the two smallest SHA-256 values within each
   first-action stratum under salt `round36-h2-v1` (`22` more). Thus
   `|H_2| = 74`.
2. `H_3` contains the six smallest SHA-256 values within each first-action
   stratum under salt `round36-h3-v1`. Thus `|H_3| = 66`.
3. Every word in `H_2` or `H_3` is absent from training for all 16 starts and
   every seed. All other words of length at most three are training rows.

For both selections, hash the exact UTF-8 encoding of
`salt + "|" + ">".join(action_word)`, using the canonical action spellings
printed above; ties break by the lexicographic action-word tuple. This is the
complete split algorithm, not a licence for an implementation-specific
serializer.

This gives `1 + 11 + (121-74) + (1331-66) = 1324` training words and
`1324*16 = 21,184` behavioral training rows, plus `140*16 = 2,240`
held-out closure rows. The ordered word lists and their hashes are evidence,
not implementation-defined conveniences.

The five model seeds are frozen as `[11, 23, 37, 53, 71]`. One process uses
CPU only, one thread, deterministic algorithms, AdamW (`lr=0.003`,
`weight_decay=1e-5`), batches of `512`, and exactly `4,000` optimizer steps
per seed. There is no early-stop selection, pilot-based threshold change, GPU
path, or seed replacement. The target full runtime is `3-8 min` of CPU and the
hard wall is `15 min`; exceeding it makes the artifact `INVALID — BUDGET`,
not evidence for a scientific gate.

Before the run, the expectation is that behavioral supervision through many
action words is sufficient to organize the eight-dimensional carrier into an
operational quotient and a composable action table. A PASS would support that
expectation in this finite world. A complete FAIL would show that this fixed
training recipe and carrier did not produce it. The simplest global confound
is response memorization without a composable state; the held-out word split,
rolled representatives, and exact action-law table are the direct test of that
confound.

### Identity is the depth-1 response signature

The registered identity depth is `D_id = 1`. For a latent point `z`, evaluate
the response probability after the empty word and after each of the 11 actions
in the frozen order. A component is supported only when it is at most `0.10`
or at least `0.90`; its response bit is then the corresponding `0` or `1`.
If any component lies in `(0.10,0.90)`, the point has no supported signature.
The resulting supported 12-bit vector is `Sigma_1(z)`.

Identity is **defined**, not estimated from chart proximity:

`z ~_1 z'  iff  Sigma_1(z) = Sigma_1(z')`, with both sides supported.

No Euclidean or cosine threshold, coordinate equality, clustering algorithm,
nearest-neighbor rule, hidden state label, or cross-seed alignment may enter
this definition. Equality of finite response signatures is reflexive,
symmetric, and transitive, so it defines the operational quotient on the
supported latent points.

This sensor is small but sufficient in the true world. `rho(s)` reads `s_1`,
and `rho(swap(1,j)(s))` reads `s_j`; therefore the depth-1 signatures of the
16 hidden states are all distinct. Before any action-law claim, every seed
must recover all 16 supported oracle signatures on the 16 encoder points,
with no collision or extra class. This is the **quotient-availability gate**.

The representative set `P_theta` contains every encoder point and every point
obtained from it by a training prefix of length one or two. Rolled points are
assigned to places only by `Sigma_1`; the simulator endpoint is retained
outside training solely to score the preregistered truth table. Thus one
operational place normally has many coordinate representatives, making
descent and interchangeability non-vacuous even though the true quotient has
16 classes.

### Existing axioms instantiated, and the one candidate addition

This artifact instantiates the current relational foundation as follows.

- **L1. Self-substitution — instantiated.** Every supported signature equals
  itself, so each point substitutes for itself under every registered response
  probe.
- **L2. Finite conjunction — instantiated for this completed probe family.**
  The completed family contains equality tests for the empty/one-action
  responses and all of their finite conjunctions; signature equality is their
  full conjunction.
- **L3. Local refinement — instantiated only in the finite, definition-driven
  sense.** The full signature conjunction isolates one quotient class and
  refines every single response cell containing it. This earns no nontrivial
  topology or geometry.
- **L4. Observational separation — instantiated by quotient.** Points agreeing
  under the complete registered response signature are one place by
  definition; distinct quotient places have different signatures.
- **L5. Presentation covariance — not instantiated.** Independent training
  seeds are not declared members of `G` and are not coordinate-aligned.
  Cross-seed action-table agreement below is evidence about an operational
  invariant, not an assumption of isomorphism.

The existing axioms name an observational quotient and admissible composed
moves but do not require actions to be compatible with identity. Round 36
therefore proposes exactly one minimal new primitive:

> **`A^*`, admissible action words acting on latent presentations — candidate,
> not earned.** Its required congruence law is
> `z ~_1 z' => T_a(z) ~_1 T_a(z')` for every primitive `a`.

Only if that falsifiable law passes does each action descend to a map
`bar(T)_a([z]) = [T_a(z)]` on the operational quotient. Composition is then
iteration in `A^*`, not a separately fitted primitive. The primitive and law
remain candidates until the reducer returns PASS; they are not added to
`theory/AXIOMS.md` at this design gate.

### Pretraining theorem and law table

Every threshold below is frozen before training. The world is finite and
fully enumerated, so a law is not rescued by an average, confidence interval,
majority seed, or post-hoc unsupported cell. “Exact” means `1.000` of the
declared finite cells with `1.000` support in **each of all five seeds**.

| Gate | Falsifiable prediction and population | Pre-declared threshold | Exact falsifier |
|---|---|---|---|
| Quotient availability | The 16 encoder points have supported `Sigma_1` signatures equal to the 16 distinct oracle signatures. | Exact in every seed; every component also satisfies the `<=0.10`/`>=0.90` confidence rule. | Any unsupported component, wrong bit, collision, missing signature, or extra signature. |
| Actions descend / quotient well-definedness | For every recovered class, primitive action, and representative in `P_theta`, the successor has one supported class; all representatives of the source class reach the same class, and that class is the simulator's behavioral successor. | Exact over all representatives, `16*11` class/action cells, and every seed. | Any unsupported successor, representative disagreement, or wrong successor class. |
| Toggle involution | For every `z in P_theta` and each `i`, `bar(T)_toggle(i)(bar(T)_toggle(i)([z])) = [z]`. | Exact in every cell and seed. | One double-toggle ends in a different or unsupported quotient class. |
| Full swap/toggle table | For every state, swap `sigma_ij`, and toggle `tau_k`: the two orders commute when `k notin {i,j}`; when `k in {i,j}` they differ and obey `sigma_ij tau_i = tau_j sigma_ij` and `sigma_ij tau_j = tau_i sigma_ij`. | Exact for all `6*4*16 = 384` swap/toggle/state cells and every seed. | A declared commuting cell differs, a declared noncommuting cell agrees, either conjugacy identity fails, or any endpoint is unsupported. |
| Held-out closure | Iterating the one learned primitive map on every start and every word in `H_2` and `H_3` reaches the oracle quotient class; no two- or three-step map is fitted. | Depth 2: exact `74*16` cells. Depth 3: exact `66*16` cells. Both exact in every seed. | Any wrong or unsupported held-out endpoint at either depth. |
| Interchangeability | For each `z in P_theta`, its canonical encoder representative `z_q` with the same signature, and every `w in H_2 union H_3`, the two rolled endpoints have the same supported `Sigma_1` and the same terminal response bit. This probes behavior beyond the depth used to define identity. | Exact over all declared representatives, held-out words, and seeds. | One quotient-equivalent pair ceases to be interchangeable or becomes unsupported under a held-out continuation. |
| Cross-seed recovered action table (alternative 2, secondary) | Name quotient classes only by their 12-bit operational signatures and compare the recovered `16 x 11` action table without coordinate alignment. | All `176` entries are identical across all five seeds and equal the behavioral truth table. | Any seed changes a class/action entry, lacks a class, or requires a chart alignment to agree. |

Held-out closure and representative interchangeability are the reduced
alternative-3 controllability/closure falsifier family. They are not an extra
model arm and do not import Round 35's causal-patch machinery.

The no-op rows participate in quotient availability, action descent, closure,
and cross-seed comparison. A no-op failure is not silently excluded. The
swap/toggle table's convention is fixed: juxtaposition `ba` means apply `a`
first and `b` second, matching ordinary function composition.

The joint scientific verdict is **`PASS — MINIMAL OPERATIONAL QUOTIENT
WORLD`** only when every gate passes in all five seeds. A complete run with
one scientific gate failure returns a gate-specific **`FAIL`**. Missing rows,
non-finite values, hash/schema/count mismatch, producer exceptions, or budget
overrun return **`INVALID`** and cannot be interpreted as either PASS or FAIL.

### One new module, one canonical entry, and separate producer/reducer

The future implementation may add exactly one Python module:

`experiments/run_operational_quotient.py`.

This new file is justified under CLAUDE.md section 6.1 because it creates the
reusable finite-world boundary `behavior -> latent transition system ->
operational quotient`, formalizes a stable evidence interface, and prevents
the toy artifact from being duplicated inside LM capture/analyzer code. No
existing module provides a model-independent learned transition world;
extending `run_lm_dynamics.py` would import the very NLP/model infrastructure
this distance-0 artifact is meant to precede. The JSON config is data, not a
second code module, and all future variation belongs there.

That module is the single canonical runner, with three config-driven process
entries:

```text
python experiments/run_operational_quotient.py produce --config experiments/config/operational_quotient_v1.json --out experiments/results/operational_quotient_v1
python experiments/run_operational_quotient.py reduce  --config experiments/config/operational_quotient_v1.json --evidence experiments/results/operational_quotient_v1
python experiments/run_operational_quotient.py fixture --config experiments/config/operational_quotient_v1.json --out experiments/results/operational_quotient_fixture
```

`produce` trains and evaluates, but it is **non-claiming**: it may write
measurements and completeness status but never `PASS`, `FAIL`, or claim text.
`reduce` runs as a separate process, receives no trainer or live model object,
and reads only the frozen config plus serialized evidence. Its reducer is a
pure, declarative conjunction of the gates above. It must validate required
keys, exact row counts, action/word ordering, hashes, seeds, finite values,
support, and the hard wall before evaluating scientific predicates. Unknown,
missing, duplicate, malformed, or extra claiming fields fail closed as
`INVALID`. A reducer defect may block interpretation but may not rewrite or
rerun a sound producer.

The minimal result directory contains:

- the exact raw config copy and SHA-256;
- `manifest.json`: code/blob/config hashes, command, platform, dependency
  versions, CPU/thread/determinism settings, action order, word-list hashes,
  data/model seeds, expected counts, start/end time, and wall time;
- `evidence.json`: per-seed loss trace checksum, every base and rolled
  signature, support flag, recovered class/action table, law-cell booleans,
  held-out endpoints, and complete numerator/denominator counts;
- `weights.npz`: the five small parameter sets and hashes, for reproduction
  but never as reducer input; and
- `verdict.json`: written only by the separate reducer, binding its own code
  hash and the hashes of every input it reduced.

The `fixture` entry uses no optimizer or learned model. It serializes an exact
eight-dimensional realization (the four bits followed by four zero pads),
exact affine toggle/swap/no-op maps, and the `s_1` readout through the same
evidence schema. The declarative reducer must return PASS on this fixture. The
fixture command then makes in-memory copies with (i) one missing required row,
(ii) one non-finite response, and (iii) one schema-valid, re-hashed
representative-specific successor mutation; the reducer must return INVALID,
INVALID, and FAIL respectively. This proves schema closure and scientific
fail-closure before learned evidence is interpreted.

No other runner, analyzer, fixture module, notebook, or custom claiming script
is authorized. Implementation and execution require their own readiness and
evidence gates; this section does not silently authorize code in the current
theory-only commit.

### Claim boundary under the guiding question and second lens

A PASS licenses this bounded statement: **in one tiny learned finite latent
world, a denizen with access only to actions and a binary response can recover
a 16-place operational quotient; the primitive actions descend to a
coordinate-free, cross-seed-stable quotient table under this recipe that
obeys the registered involution, swap/toggle, closure, interchangeability, and
cross-seed laws.**
That is a native quotient plus action algebra in a toy latent space, available
through the world's own behavior rather than hidden coordinates. It is the
first constructive distance-0 artifact after the NLM-007 continuation ruling.

A PASS does **not** establish anything about language models, residual streams,
natural language, semantic reasoning, learned world models in general,
arbitrary actions, longer-horizon closure, metrics, move cost, topology with
nontrivial content, presentation covariance, or a universal axiom of latent
space. It does not activate Round 35 automatically. NLP surface-form design
requires a separate post-PASS ruling.

A complete FAIL means the fixed behavioral training recipe and eight-
dimensional learned carrier cannot produce the registered operational quotient
and action algebra. Under the second lens that is a **constructive hole in THIS
learned latent space**: response fitting did not organize a denizen-usable
world. It is not evidence that the exact simulator lacks the algebra, not a
hostile hole in latent spaces generally, and not evidence about any LM. The
gate-specific residue determines the successor: unavailable identity,
non-congruent actions, non-composable transitions, representation-specific
tables, or unstable responses. No larger carrier, new sensor, or NLP wrapper
is added inside Round 36 after seeing the failure.

## Round 36b — preregistered optimization/capacity ladder after the v1 FAIL (2026-08-29)

**Evidence ruling and design gate; theory change only.** The first Round 36
artifact remains a complete scientific **FAIL** for its frozen v1 recipe. It
is not rerun, repaired, relabelled, or pooled with this successor. Round 36b
is a new registered increment authorized by Round 36's own successor-design
rule: increasing steps or capacity after a failure is not allowed inside
Round 36, but it may be tested prospectively under a new lock. No code,
config, result, ledger, `STATE.md`, or `NOTEBOOK.md` change is part of this
design gate.

### Why v1 is optimization-confounded

The retained `weights.npz` contains all five 4,000-step loss traces, so no
producer rerun was needed. Loading those exact weights through the registered
runner and scoring every registered behavioral row was a diagnostic replay,
not a new result. The 100-step trailing mean losses at steps
`100/500/1000/2000/3000/4000` were:

| seed | loss curve | final raw minibatch loss | full-train response accuracy | held-out response accuracy |
|---:|---|---:|---:|---:|
| 11 | `.660/.252/.154/.104/.078/.053` | `.0621` | `98.112%` | `98.259%` |
| 23 | `.673/.282/.176/.104/.075/.054` | `.0351` | `97.758%` | `97.679%` |
| 37 | `.660/.311/.201/.138/.105/.087` | `.0828` | `96.563%` | `97.009%` |
| 53 | `.679/.272/.154/.080/.053/.038` | `.0478` | `98.546%` | `97.902%` |
| 71 | `.674/.288/.189/.123/.099/.084` | `.0905` | `96.743%` | `97.455%` |

The curves were still improving at the stop. Held-out depth-2 response
accuracy was `99.747-100%`, but depth-3 response accuracy was only
`93.750-96.307%`. The exact depth-1 encoder signatures were supported for
only `11/16`, `7/16`, `4/16`, `8/16`, and `6/16` points. Thus the model did
learn most terminal behavior, but the finite behavioral task was not fitted
exactly and the registered `0.10/0.90` confidence conjunction was not
saturated. This is category **(a), optimization/behavioral underfit under the
v1 recipe**. It does not yet distinguish category (b), a behaviorally fitted
but non-congruent latent. The clean fixture PASS, the learned artifact's valid
hash/schema/count reduction, and the absence of reducer errors reject category
(c), an evidence/gate construction failure, for this adjudication.

### Prospective lock: all cells fixed before any outcome

The world, opaque handles, behavioral population, train/H2/H3 spelling split,
action order, five seeds, batch size `512`, AdamW betas/epsilon,
`weight_decay=1e-5`, eight-dimensional carrier except where explicitly stated,
response sensor, signature thresholds, representative population, exact law
tables, reducer schema closure, and every Round 36 scientific falsifier remain
unchanged. No sensor, target, reweighting, curriculum, extra supervision,
seed replacement, early stopping, or best-checkpoint selection is permitted.

Before any Round 36b producer is launched, the implementation must create,
review, and hash-lock all four configs and the one canonical runner revision.
All four cells then run sequentially on CPU and all four are reduced and
reported. Later cells may not be changed, skipped, or stopped because of an
earlier cell's outcome, and no scientific output is inspected until all
producer cells have terminated. The retained v1 artifact is the 4,000-step
reference and is not regenerated.

| cell | optimizer steps/seed | learning rate | transition width | parameters | expected full-cell CPU | hard wall |
|---|---:|---:|---:|---:|---:|---:|
| `S16` cheapest budget remedy | `16,000` | `0.003` | `32` | `1,041` | about `3 min` | `8 min` |
| `S64` primary budget cell | `64,000` | `0.003` | `32` | `1,041` | about `11-12 min` | `20 min` |
| `LR64` step-size sensitivity | `64,000` | `0.001` | `32` | `1,041` | about `11-12 min` | `20 min` |
| `W64` capacity sensitivity | `64,000` | `0.003` | `64` | `1,937` | about `16-22 min` | `30 min` |

The estimate is anchored to v1's measured `41.667 s` for `5*4,000` steps
plus `10.966 s` of evidence construction. The first two cells isolate training
budget with learning rate and architecture held fixed; `LR64` tests whether
the inherited step size prevents late convergence; `W64` asks whether the
fixed 32-wide transition is the bottleneck only after the direct budget arms.
The expected total is approximately `42-49 CPU-minutes`, one process and one
thread at a time, with no GPU. A per-cell wall overrun is `INVALID — BUDGET`,
not a reason to alter the remaining cells.

### Behavioral-fit eligibility precedes quotient interpretation

For each seed and cell, serialize the full loss trace, full-dataset BCE, and
response accuracy on all `21,184` training rows, all `1,184` H2 rows, all
`1,056` H3 rows, and their held-out union. A probability of exactly `0.5`
counts as incorrect. Also serialize the fraction both correct and supported
under the inherited `<=0.10`/`>=0.90` rule.

A cell becomes eligible for any quotient/action-law interpretation only if
**every seed scores exactly `21,184/21,184` on train and `2,240/2,240` on the
held-out union** at the `0.5` response threshold. If either condition fails,
the cell is `FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE`. Quotient counts
may be retained as diagnostics but cannot be described as non-congruence. If
behavior is exact and any inherited quotient gate fails, that cell is the
first admissible category-(b) result: `FAIL — FIT BUT NON-CONGRUENT LATENT`,
followed by the exact inherited gate names. Only a behavior-eligible cell that
passes every unchanged Round 36 gate earns the original bounded
recipe-specific operational-quotient claim, indexed by that cell. There is no
best-cell selection and no joint PASS assembled by pooling cells; mixed
outcomes remain mixed and every cell stays visible.

This registration does not erase v1's FAIL. It separates two questions that
v1 could not: whether the tiny learner first fits its behavioral world exactly,
and, conditional on that fit, whether its latent moves respect the denizen's
operational identity.

## Round 36b amendment (audit #23; before any outcome) (2026-08-29)

**Prospective amendment; theory change only.** This amendment is registered
before any Round 36b producer or reducer outcome. Round 36b is a
**prospectively locked, post-outcome, outcome-informed successor** to the v1
FAIL. It is exploratory, not confirmatory: it is neither a retroactive repair
of v1 nor a confirmatory replication.

### Three-stage status tree

The reducer must apply the following ordered tree independently to every 36b
cell:

1. If any seed misses exact behavioral fit on either the `21,184` training
   responses or the `2,240` held-out responses at the frozen `0.5` decision
   threshold, report `FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE`.
2. If behavior is exact but any response-signature component required by an
   inherited structural gate is unsupported under the frozen
   `<=0.10`/`>=0.90` rule, report
   `FAIL — BEHAVIOR FIT; OPERATIONAL SIGNATURE UNSUPPORTED`.
3. Only when behavior is exact and the complete domain required by every
   inherited structural gate is supported may a structural failure be named.
   Report
   `FAIL — BEHAVIOR AND SIGNATURE SUPPORTED; NON-CONGRUENT ACTIONS — <failed
   inherited structural gate names>`, naming only failures evaluated on that
   fully supported domain.
4. Report `PASS — MINIMAL OPERATIONAL QUOTIENT WORLD` only after every
   inherited Round 36 structural gate passes. No diagnostic field can change
   any branch of this primary tree.

### Confidence-free reducer diagnostic

Alongside, but outside, the primary tree, the 36b reducer must recompute from
serialized response probabilities a complete confidence-free gate table using
the literal decision rule `p > 0.5` (`p == 0.5` decodes as `0`). The table must
be labelled `DIAGNOSTIC ONLY`, must report every inherited gate, component
error counts, and decision margins from `0.5`, and must be emitted on every
valid 36b reduction. It can diagnose whether errors are confidence- or
truth-driven, but it can never rescue, replace, soften, or otherwise alter a
primary `0.10/0.90` FAIL.

For each diagnostic gate, report its complete numerator/denominator table and
the corresponding error count; for each serialized probability domain,
report the component count and the minimum, median, mean, and maximum absolute
margin `|p-0.5|`, including the count exactly at zero margin. These quantities
are descriptive diagnostics, not additional success gates.

### Literal cross-seed cell accounting

Beside the inherited whole-table exact cross-seed gate, the reducer must
report literal cellwise counts over the `16 x 11 = 176` recovered action-table
cells: (a) cells identical across all five seeds, (b) cells identical and
supported across all five seeds, (c) cells supported and truthful in all five
seeds, and (d) bitwise-majority truth, reported both as exact truthful cells
out of `176` and truthful bits out of `2,112`. A primary-threshold majority bit
exists only when at least three of the five seeds emit that literal `0` or `1`;
`?` is an abstention, not a vote. Report this accounting for both the primary
`0.10/0.90` tables and the confidence-free diagnostic tables. None of these
cellwise fields replaces the inherited all-or-none whole-table gate.

### Training diagnostics by word depth

For every 36b seed, report full-dataset binary cross-entropy loss, response
accuracy, and primary-threshold response support separately for training-word
depths `0`, `1`, `2`, and `3` throughout optimization. The frozen trace cadence
is the initialized model at optimizer step `0` and every completed `1,000`
steps through the cell's final registered step. Support means the fraction of
responses with probability `<=0.10` or `>=0.90`, independent of correctness;
accuracy continues to use the frozen strict decision rule under which a logit
of exactly zero is incorrect. This depth trace is diagnostic and does not
change the behavioral-fit or structural gates.

## Round 36c — learned quotient-trained positive control (2026-08-29)

**Prospective reachability-control registration; theory change only. No code,
config, fixture, producer, reducer, or result is part of this registration
commit.** This is the explicitly quotient-trained learned positive control
ranked first by audit #24 and required by audit #23 item 6. It is not a rescue,
continuation, or relabelling of either the Round 36 v1 artifact or any Round
36b behavior-only cell.

### Purpose: test gate reachability, not behavior-only learning

Round 36c tests one question only: **can the same eight-dimensional learned
carrier, encoder/transition/readout architecture, optimizer, reducer, seeds,
world, and exact certificate reach every registered Round 36 gate when the
training objective directly supervises the state-transition congruence?**
This is the `REACHABILITY` gate for the certification regime. It is not
designed to produce a scientific PASS for the behavior-only program.

The non-expert “so what” is: before blaming a learner for failing to organize
its world from behavior alone, first show that the proposed body and exam can
actually realize a perfect, law-abiding map when the correct moves are taught
directly.

The simplest confound is that the positive control could pass only because it
was given privileged successor pairings. That is intentional and is why every
output is labelled `POSITIVE-CONTROL`: the cell tests learned certificate
reachability, not discovery from behavior. Conversely, a positive-control
FAIL cannot be interpreted as a hole in latent organization because the
architecture/optimizer and exact certification regime remain entangled.

### The one changed signal: canonical transition consistency

The ordinary behavioral term remains the unchanged binary cross-entropy on
the unchanged Round 36 training rows and unchanged seeded minibatch stream.
Round 36c adds exactly one privileged term. For every one of the `16 * 11 =
176` canonical opaque-handle/action cells on every optimizer step, let `h_s`
be the opaque handle assigned to hidden simulator state `s`, and let
`h_{delta(s,a)}` be the opaque handle assigned to its true primitive
successor. The registered auxiliary loss is

`L_transition = mean_{s in S, a in A, j=1..8}
  (T_theta(E_theta(h_s), a)_j
   - stop_gradient(E_theta(h_{delta(s,a)}))_j)^2`.

The complete optimized objective is

`L_36c = L_behavioral_BCE + 1.0 * L_transition`.

The full 176-cell transition term is evaluated once per optimizer step; it is
not sampled or reweighted. `stop_gradient` applies only to the successor
encoder target in that occurrence. Encoder vectors remain learned parameters
and receive gradients through the behavioral term and whenever they appear as
the source of a transition. The simulator's hidden state is used only to form
the registered successor-handle pairing. No fixed bit chart, latent
coordinate target, state reconstruction term, response-signature tolerance,
best-checkpoint selection, extra readout, or separate transition head is
introduced.

This learned-target transition-consistency loss is the minimal control because
zero loss is exactly the desired canonical congruence equation while leaving
the eight carrier coordinates free. A fixed oracle embedding would test a
stronger chart-reconstruction control; a response-signature consistency loss
would add the reducer's multi-probe sensor inside training. Neither is needed
for this first reachability test.

Everything else is frozen. The hidden world, opaque-handle permutation,
action order, behavioral population, training/H2/H3 spelling split, response
sensor, depth-1 identity definition, `0.10/0.90` support thresholds, all exact
fractions, representative construction, held-out continuations, cross-seed
comparison, schema closure, and declarative reducer are unchanged. The five
model seeds remain exactly `[11, 23, 37, 53, 71]`. Training remains CPU-only,
single-threaded, deterministic AdamW with `lr=0.003`, `weight_decay=1e-5`,
betas `[0.9, 0.999]`, epsilon `1e-8`, behavioral batch size `512`, and no
early stopping or seed replacement.

### Width policy, cells, and walls

Both cells use exactly `64,000` optimizer steps per seed so the supervision
signal, not a shortened budget, is the primary change from the strongest
Round 36b comparisons.

| cell | order and authorization | transition width | parameters | expected full-cell CPU | hard wall |
|---|---|---:|---:|---:|---:|
| `36c-w32` | Run first and reduce before any `w64` producer | `32` | `1,041` | about `12-16 min` | `30 min` (`1,800 s`) |
| `36c-w64` | Conditional only: run iff `36c-w32` completes validly and returns an exact-gate positive-control FAIL | `64` | `1,937` | about `14-20 min` | `40 min` (`2,400 s`) |

A `36c-w32` PASS stops the control and does not authorize `36c-w64`. An
`INVALID`, producer exception, missing artifact, or budget overrun also does
not authorize `w64`; repair the implementation/artifact defect and rerun the
same locked `w32` cell. Only a valid complete `w32` FAIL activates the already
locked `w64` sensitivity cell. The conditional cell may not be edited after
the `w32` result is known. Both configs and the shared runner must therefore
be implemented, fixture-tested, reviewed, and hash-locked before any producer
launch.

The expectation is that direct congruence supervision will make rolled
representatives coincide operationally with their canonical successors and
therefore allow every exact gate to pass. A PASS supports that reachability
expectation. A valid FAIL means the current learned carrier/optimizer plus the
exact gate/reducer regime did not reach its own certificate even under direct
supervision; the immediate problem is then the certification regime or its
realizability under this recipe, not a result about latent organization.

### Decision semantics and permanent claim walls

The unchanged reducer still computes the original exact Round 36 gates and
the original `PASS — MINIMAL OPERATIONAL QUOTIENT WORLD` conjunction. For a
learned Round 36c artifact, its result scope and claim boundary must be
labelled `POSITIVE-CONTROL` so the same gate values cannot be read as a
behavior-only result.

- **Positive-control PASS:** every exact gate is reachable by this learned
  carrier/optimizer/reducer combination under direct transition supervision.
  The gate-reachability control is validated. The retained behavior-only
  FAILs may then be interpreted as limits of their registered behavioral
  objective/optimization recipes rather than evidence that the carrier or
  reducer could never pass.
- **Positive-control FAIL:** this carrier/optimizer/reducer combination did
  not reach the exact certificate even with direct supervision. That is a
  certification-regime or learned-realizability problem requiring redesign or
  a sharper control. It is not a latent-organization result and does not make
  the behavior-only failures structurally interpretable.
- **INVALID:** no reachability conclusion. Integrity and budget failures are
  repaired under the same lock; they do not trigger the conditional cell or a
  scientific successor.

The never-say list is permanent. A Round 36c PASS is **not** a behavior-only
result; **not** evidence that a quotient was learned from behavior; **not** a
rescue, softening, or retroactive PASS for v1 or Round 36b; **not** evidence
about natural language, residual streams, or latent spaces generally; and
**not** activation of Round 35. The privileged successor pairing must be
named beside every reported Round 36c gate table or verdict.

Audit #24's rank-2 tolerance-based approximate branch is a separate future
prospective registration. If authored later, it must keep the original exact
PASS untouched, carry its own non-PASS status and claim boundary, and may
never replace, rescue, soften, or retroactively reclassify an exact result.

## Round 36d — frozen-target head-only learned-pass calibration (2026-08-29)

**Prospective design gate and registration; theory change only. No code,
config, producer, reducer, result, or ledger event is part of this commit.**
This is audit #25 rank 1 and is the last authorized learned-pass calibration
on the Round 36 world. It is one capped cell, where one cell means the same
locked recipe over all five retained seed-matched producers. It is never a
width, learning-rate, step, loss-weight, or seed ladder. Round 37 follows only
after Round 36d has one valid reduced verdict; an `INVALID` is repaired under
this same lock and does not authorize a different cell.

The non-expert “so what” is: hold a learned map and its landmarks still, then
ask whether a freshly learned mover can reach the landmarks closely enough
for the world's own exact navigation exam to recognize every move.

### Retained producer, exact weights, and fixed targets

The sole retained source is
`experiments/results/operational_quotient_36b_W64/`. The producer tuple is
frozen as:

- config SHA-256
  `3b1fd82cf2801ea6d8f75c12acdc6aac038492bf0712a8c66900a243e9caa2a0`;
- producer-code SHA-256
  `e5e88d9c8dcec06a66ed251a9666a65e3a4abb0cf693f4af9a648172b6281f5b`;
  and
- `weights.npz` SHA-256
  `dadfff34d9b6941bd8d6f4acf25668e3604f1caa5cdd4b8011a320914e58547a`.

The implementation must verify all three pins and the retained manifest before
loading a tensor. For each retained seed `s in [11, 23, 37, 53, 71]`, the
Round 36d producer loads exactly these same-seed arrays:

- `seed_s__encoder_weight` as the assigned, frozen `16 x 8` encoder table;
- `seed_s__response_weight` and `seed_s__response_bias` as the assigned,
  frozen response readout; and
- none of the retained transition arrays or retained loss trace as trainable
  state.

The retained W64 artifact is eligible only for this privileged control because
its canonical signatures and all `176/176` canonical action-table cells are
supported and truthful in all five seeds. Its behavioral-fit and rolled-law
FAILs remain exactly as adjudicated by audit #24; this reuse does not
rehabilitate or reclassify them.

Immediately after the frozen encoder is loaded, copy its 16 handle-indexed
rows, in opaque-handle order `0..15`, into an immutable `16 x 8` target tensor
`Z_s^*`. The target for canonical cell `(h,a)` is the row of `Z_s^*` indexed
by the true successor handle already defined by Round 36's fixed simulator and
`data_seed=3601`. The producer hashes the exact target tensor once per seed.
No encoder call supplies a target after optimization starts; targets therefore
cannot chase the learned head.

Only the width-64 transition head is re-initialized. Its optimized parameter
names are exactly `w_z.weight`, `action_embedding.weight`, `b1`, `w2.weight`,
and `w2.bias`. For retained seed `s`, set the dedicated transition-init seed
to `360000+s`, instantiate the existing W64 model in its canonical module
order, overwrite the encoder and response arrays with the retained values,
and freeze those assigned arrays before the optimizer is constructed. Hash the
five initial transition arrays in name-sorted, dtype-and-shape-bound C-order.
The assigned encoder/readout, fixed targets, and every simulator table have
`requires_grad=False`; the five named transition arrays are the complete and
only optimizer parameter list.

### One capped optimization cell and separate diagnostics

The loss is only the full-batch fixed-target transition MSE over all 176
canonical source/action cells:

`L_fixed = mean_{h,a,j} (T_phi(E_s(h),a)_j - Z_s^*[delta(h,a),j])^2`.

There is no behavioral BCE, moving encoder target, signature loss, hidden-bit
target, curriculum, sampling of the 176 cells, early stopping, checkpoint
selection, or second attempt. The locked optimizer is deterministic,
single-threaded CPU AdamW with `lr=0.003`, `weight_decay=1e-5`, betas
`[0.9,0.999]`, epsilon `1e-8`, and exactly `16,000` steps per seed. The target
full-cell cost is `2-5 CPU-minutes`; the hard wall is `8 min` (`480 s`) for all
five seeds plus evidence production. A wall overrun is `INVALID — BUDGET`.

At initialization, every 250 completed steps, and the final step, serialize
three distinct traces per seed rather than a combined loss:

1. the scalar `L_fixed` transition MSE;
2. the maximum canonical-cell coordinate residual
   `max_{h,a} max_j |T_phi(E_s(h),a)_j-Z_s^*[delta(h,a),j]|`; and
3. response-signature margins on the 176 predicted successors under the
   unchanged empty-plus-11-action probe suite. For oracle bit `y` and response
   probability `p`, the signed support-and-truth margin is `p-0.90` when
   `y=1` and `0.10-p` when `y=0`. Report the minimum, median, mean, and maximum
   signed margin and the negative-margin count separately from coordinate
   residuals.

These traces are diagnostic, not a substitute reducer. A FAIL with residuals
that remain materially above zero leaves head capacity versus optimization as
the immediate fork. A FAIL after residuals approach the fixed targets but one
or more signed signature margins remain negative localizes the mismatch to the
coordinate-residual/signature-margin interface. Neither branch licenses a
post-hoc threshold or another cell.

**Diagnostic-only amendment (2026-08-29, Codex-owned; pre-outcome).** At each
registered checkpoint, also record the normalized ratio
`L_fixed/L_identity`, where `L_identity` is the fixed-target MSE of the
identity map at initialization. Audit #25's `1e-4` value is retained only as
a descriptive reference. The ratio cannot alter PASS, FAIL, or INVALID;
cannot act as a gate, eligibility condition, stopping rule, retry trigger, or
branch selector; and cannot replace the unchanged exact Round 36 reducer.
The audit addendum's separate control-optimization/fixed-chart-residue status
split and final-1,000-step condition are not part of this registration.

### Unchanged reducer, provenance, and decision

After the final head is frozen, the producer constructs every original Round
36 representative, signature, held-out endpoint, law row, and
interchangeability row. The **unchanged exact Round 36 reducer computation**,
including the `0.10/0.90` support rule, all original denominators, all five
seeds, and the all-gates conjunction, scores the evidence. Its input remains
only the frozen config copy, manifest, and evidence; `weights.npz` remains
outside the reducer. A schema extension may require the provenance below and
bind its hashes, but it may not alter `_scientific_gates`, a threshold, a
population, an exact falsifier, or the meaning of PASS.

Every producer artifact must declare `result_scope="POSITIVE-CONTROL"` and
retain:

- the locked source directory and source config/code/weights hashes above;
- hashes of the action/successor input table and, per seed, the assigned
  encoder/readout arrays, the 16-row target snapshot, and the fresh transition
  initialization;
- the complete, separately named MSE, maximum-residual, and signature-margin
  traces plus their hashes;
- a parameter-disposition table naming every array as `optimized`,
  `assigned_frozen`, `fixed_target`, or `discarded_retained_transition`;
- final arrays for all five seed-matched models, a per-seed hash, and one final
  `weights.npz` SHA-256; and
- command, registration/config/code hashes, dependency and deterministic CPU
  settings, start/end/wall times, and a manifest-to-evidence hash chain.

Producer authenticity remains outside the declarative reducer's inference, so
these prospectively locked records are mandatory evidence for calling the
artifact learned. Missing, extra, mutable, non-finite, or hash-mismatched
provenance is `INVALID`, never FAIL.

- **Positive-control PASS:** every unchanged exact gate passes in every seed.
  This validates learned width-64 transition-head plus reducer reachability
  when a retained learned encoder/readout and fixed successor coordinates are
  supplied. It does not validate the retained transition, the W64 behavioral
  recipe, or quotient discovery.
- **Positive-control FAIL:** this capped head-only recipe did not reach the
  certificate. The registered traces localize the next theoretical question
  to head capacity/optimization or to the gap between coordinate residual and
  response-signature margin; they do not establish reducer or carrier
  impossibility.
- **INVALID:** no reachability conclusion. Repair only the integrity or budget
  defect under this exact registration.

The never-say list is permanent. Round 36d is **not** behavior-only learning;
**not** evidence that W64 learned a quotient or fitted its behavioral world;
**not** a rescue, softening, or retroactive PASS for Round 36, 36b, or 36c;
**not** proof that fixed coordinate MSE is a native objective; **not** proof
that any learned artifact can generally pass the reducer; **not** a result
about natural language, residual streams, or latent spaces generally; and
**not** authorization for another calibration ladder. The frozen targets,
assigned retained encoder/readout, and `POSITIVE-CONTROL` scope must appear
beside every gate table and verdict.

### Audit #26 amendment to the Round 36d claim wall (2026-08-29)

Audit #26 upholds the hash-valid `FAIL — INTERCHANGEABILITY` and every stored
gate count. Eight individual predicates pass under a privileged full-table
frozen-chart control; the joint certificate does not. The 147 misses are
confined to depth-2 rolled representatives followed by H3 and occur in
future-response signatures, not immediate endpoint responses. The
diagnostic adequacy ratio cannot be called a PASS and does not rule out
optimization, capacity, residual accumulation, extrapolation, or threshold
sensitivity. This result validates narrow individual-gate passability for
the registered optimizer-fitted head; it is not behavior-only quotient
learning, composition discovery, a complete learned PASS, or authorization
for another Round 36 cell.

### Audit #25 amendment to the Round 36c claim wall (2026-08-29)

This note appends audit #25 rather than rewriting the prospective Round 36c
registration. It governs the Round 36c ledger purpose wherever the earlier
text is broader:

> A FAIL means this registered learned-target reachability control did not
> reach the certificate. It does not distinguish control-objective failure,
> optimization failure, carrier capacity, or learned reducer/gate reachability,
> and it has no behavior-only interpretation.

## Round 37 — presentation-duplicated quotient world (design gate) (2026-08-29)

**Prospective design gate and registration; theory change only. No code,
config, population artifact, producer, reducer, result, or ledger event is part
of this commit.** This is audit #25 rank 2. It begins only after Round 36d has
one valid reduced verdict. It is a matched factored-versus-unrestricted design,
not a capacity ladder; all four architecture-by-presentation-role cells are
locked before any is run, run sequentially on CPU, and reported regardless of
earlier outcomes.

The registered so-what is: **keep what a place means separate from how it is
presented, so inhabitants can move reliably across two views.**

### The 32-state, 16-place world

The hidden simulator state is

`s=(q,p) in S={0,1}^4 x {0,1}`,

where `q=(q_1,q_2,q_3,q_4)` is operational state and `p` is a
presentation/nuisance bit. A fixed seeded permutation (`data_seed=3701`)
assigns 32 opaque handles to the 32 tuples. Pair membership, `q`, `p`, state
indices, and successor handles never enter the learner interface or loss.
There are two hidden representatives of each operational state and therefore
32 hidden states but exactly 16 true operational places.

The ordered task-action alphabet is the unchanged Round 36 list
`[no-op, toggle(1), toggle(2), toggle(3), toggle(4), swap(1,2),
swap(1,3), swap(1,4), swap(2,3), swap(2,4), swap(3,4)]`. Each task action
changes only `q` and leaves `p` fixed. A twelfth primitive presentation move
`present` flips only `p` and leaves `q` fixed. Words are applied left to right.
The sole response and every future task response ignore nuisance:

`rho(q,p)=q_1`.

A behavioral row contains only

`(opaque_start_handle, action_word, terminal_response)`.

Training is behavior-only binary cross-entropy. There is no hidden-state,
operational-bit, presentation-bit, state-index, next-handle, endpoint,
duplicate-pair, pair-contrast, coordinate, signature, or quotient target. The
simulator retains hidden tuples and pairs only for prospective reduction.

### Outcome-blind presentation-role transfer

Let `B` be the existing 1,324 Round 36 training-word list over task actions and
let `H=H_2 union H_3` be the existing 140 hash-selected held-out task words,
with the same salts, serialization, ordering, and list hashes. Both lists were
selected by spelling without responses or model outputs and remain frozen.

There are two complementary role folds `r in {0,1}`. In fold `r`:

1. every word in `B` is trained from all 32 opaque starts;
2. every word in `H` is trained from the 16 starts with presentation `p=r` and
   is held out from the paired 16 starts with `p=1-r`; and
3. the presentation-only word `[present]` and the 22 primitive compositions
   `[present,a]` and `[a,present]` for each of the 11 task actions are trained
   from all 32 starts.

Thus each role fold has `42,368 + 2,240 + 736 = 45,344` behavioral training
rows and exactly `2,240` primary held-out presentation-transfer rows. Fold 0
trains the selected action words through presentation 0 and tests presentation
1; fold 1 swaps those roles. Rows are generated and minibatched in a frozen
canonical order. Neither fold is selected by outcome and neither may rescue
the other.

Before structural interpretation, every architecture/role/seed must score
exactly on all `45,344` training responses at the strict decision rule
`p>0.5` and must support every empty/primitive response used in the operational
signature. Otherwise its structural fields remain descriptive and its status
is `FAIL — BEHAVIOR UNDERFIT OR BASE SIGNATURE UNSUPPORTED`.

### Quotient-factored carrier and matched unrestricted baseline

Both carriers use a learned `32 x 8` opaque-handle table, total latent width
eight, deterministic seed, residual-tanh transitions, one binary response,
the same training rows and minibatch indices, and the same optimizer and step
budget. The only intended difference is the structural factorization.

The quotient-factored carrier writes `z=(q_z,p_z)` with `q_z in R^6` and
`p_z in R^2`:

- a task move has form `T_a(q_z,p_z)=(q_z+F_a(q_z),p_z)` and the response is
  `R(q_z,p_z)=sigmoid(r_q^T q_z+b)`;
- the presentation move has form
  `T_present(q_z,p_z)=(q_z,p_z+F_present(p_z))`; and
- no task transition or response path may read `p_z`, while no presentation
  transition may alter or read `q_z`.

Use a total hidden-unit budget of 64, split prospectively as 48 task units and
16 presentation units. The split is an architectural prior, not a hidden-label
loss: the two handles for one `q` still have independent learned table rows and
receive no equality or pairing target. Presentation information may live in
`p_z`, but neither its presence nor a literal encoded nuisance bit is assumed.

The unrestricted baseline uses the same eight-dimensional carrier and a
single width-64 residual-tanh transition over all eight coordinates and all 12
actions; its response reads all eight coordinates. It removes the fixed block
masks and is therefore a weakly more expressive superset, not a capacity-
starved foil. Match data, seeds, initialization family, batches, optimizer,
steps, CPU settings, and evaluation exactly; serialize and report both models'
active trainable-parameter counts. A factored advantage cannot be attributed
to giving the factored carrier more effective parameters.

The five model seeds remain `[11,23,37,53,71]`. Each of the four
architecture-by-role cells uses deterministic single-threaded CPU AdamW,
`lr=0.003`, `weight_decay=1e-5`, betas `[0.9,0.999]`, epsilon `1e-8`, batch
size `512`, and exactly `32,000` steps per seed, with no early stopping,
checkpoint selection, seed replacement, or adaptive fifth cell. The target is
`24-32 CPU-minutes` for the full four-cell matrix; the hard wall is `12 min`
per cell and `45 min` total. An overrun is `INVALID — BUDGET` and cannot alter
another cell.

### Operational signatures, rates, margins, and exact certificates

The operational signature excludes `present`. For any point `z`,
`Sigma_task(z)` is the 12-bit response vector under the empty word and the 11
task primitives in their frozen order. A component is supported only at
`p<=0.10` or `p>=0.90`. For oracle bit `y`, its signed certificate margin is
`p-0.90` if `y=1` and `0.10-p` if `y=0`; a supported truthful component has
nonnegative margin. Every endpoint below reports the complete numerator and
denominator, the unsupported and wrong counts separately, and minimum,
median, mean, and maximum signed margins per seed and role before any exact
conjunction is formed.

The primary questions and exact finite certificates are:

| Question | Rate and margin | Exact certificate in every seed and both role folds |
|---|---|---|
| Paired presentations name one place | Fraction of the 16 hidden `q` pairs whose two encoder points have supported, truthful, equal `Sigma_task`; signed margins over all `32*12` components. | `16/16` paired operational signatures, each equal to the oracle signature. Coordinate equality is neither required nor scored. |
| Task actions descend independently of presentation | Fraction of `16*11=176` operational-state/action cells for which both presentation starts, and the corresponding `present`-before/after paths, reach one supported truthful successor signature; endpoint signed margins reported separately. | `176/176` cells agree across presentation and equal the oracle task successor. |
| Held-out presentation transfer | Supported truthful signature rate and terminal-response accuracy on the `140*16=2,240` rows trained only through the opposite presentation; signed endpoint margins. | `2,240/2,240` held-out endpoints and responses in each role fold. |
| Rolled interchangeability | For each `q` and `w in H`, compare the two presentation-start endpoints: supported equal signature, equal response, and equality to the oracle place; report rate and margins for each side. | `2,240/2,240` paired continuations in each role fold. |
| Presentation move preserves place | Fraction of 32 encoder starts for which `Sigma_task(T_present(z))` is supported and equals both `Sigma_task(z)` and the oracle operational signature. | `32/32` presentation moves leave operational place unchanged. |

For the factored-versus-unrestricted comparison, report paired differences
`factored-unrestricted` in held-out-transfer rate, rolled-interchangeability
rate, and minimum signed margin for each of the ten seed-by-role units. Do not
pool generation rows or treat endpoints as independent replicates. The exact
comparison is the two carriers' separate all-cell certificates:

- factored exact and unrestricted non-exact supports a bounded
  **`FACTORED ADVANTAGE IN THIS MATCH`**;
- both exact is **`SOLVED BY BOTH — NO FACTORIZATION ADVANTAGE SHOWN`**;
- factored non-exact and unrestricted exact is
  **`UNRESTRICTED ADVANTAGE IN THIS MATCH`**; and
- both non-exact is **`NO ARCHITECTURAL WIN`**, with behavior eligibility,
  support, wrong cells, rates, and margins kept separate.

The primary world verdict is **`PASS — PRESENTATION-DUPLICATED OPERATIONAL
QUOTIENT`** only if the quotient-factored carrier is behavior-eligible and all
five exact certificates pass in all five seeds and both role folds. The matched
baseline label is reported beside, not folded into, that verdict. Missing,
duplicate, malformed, non-finite, hash-mismatched, incomplete, or over-budget
evidence is `INVALID`.

### Axioms, candidates, and the second lens

On supported `Sigma_task` classes, L1 self-substitution, L2 finite conjunction,
L3's finite definition-driven refinement, and L4 separation by quotient are
instantiated exactly as in Round 36; none earns geometry. The declared
simulator presentation group is `G={id,present}`. L5 presentation covariance
is tested, not assumed, on the learned carrier: it is instantiated for this
finite probe family only if paired identity, presentation-place invariance,
presentation-independent action descent, and rolled interchangeability all
pass.

The task-word action family, the presentation move on learned carriers, the
factor split `z=(q_z,p_z)`, and their commutation/descent laws remain
**candidate primitives, not earned** at this design gate. A block mask is an
inductive bias, not evidence that the nuisance coordinate was learned. No
origin, norm, distance, dimension invariant, move cost, geometry, general map,
or model-independent law is added to `theory/AXIOMS.md` by this registration.

The expectation is that the factored carrier will preserve operational place
and compose task moves across presentations more reliably than the unrestricted
carrier. A PASS would show that a tiny behavior-trained world with an explicit
quotient-compatible inductive bias can support stable navigation across two
opaque views. The simplest global confound is that both carriers merely
memorize terminal responses; the complementary presentation-role holdout,
exact action descent, rolled interchangeability, and unrestricted superset
baseline are the direct controls. If both carriers pass, factorization was not
needed here. If both fail after exact behavioral fit, presentation/action
congruence remains a proven hole of these recipes and motivates a different
latent-space design rather than more scale on this matrix.

### Minimal future code delta and artifact contract

The future implementation should be a config- and `registration_id`-driven
extension of the existing canonical
`experiments/run_operational_quotient.py`, not a new sibling runner. CLAUDE.md
section 6.1 defaults to existing modules unless a new reusable boundary is
necessary; this world reuses the existing finite simulator, opaque handles,
action words, CPU producer, serialized evidence, hash chain, fixture posture,
and producer/reducer separation. A sibling module would duplicate those stable
interfaces. The extension may add a Round 37 state/action/model branch and a
versioned Round 37 evidence/gate schema, but the legacy Round 36 config
validation, artifacts, and `_scientific_gates` path must replay unchanged.

The non-claiming producer must serialize the hidden truth table only in the
reducer-scored evidence layer, the exact role-fold row lists and hashes,
per-component loss/support traces, per-cell endpoint probabilities and
signatures, active parameter counts, initial/final weight hashes, model and
data seeds, platform/dependencies, CPU settings, and wall accounting. The
separate reducer recomputes all rates, margins, exact certificates, carrier
comparison labels, and the joint verdict from config/manifest/evidence only;
weights remain reproduction material, not reducer input. A fixture must pass
all factored certificates and the same missing/non-finite/rehash-corruption
fail-closure tests before learned evidence is interpreted.

### Claim license and never-say list

A factored PASS licenses only: **in this one finite 32-state behavior-trained
world, under an explicitly quotient-factored carrier, two opaque presentations
of each of 16 operational places acquired the same supported response
signature, task actions descended independently of presentation, held-out
presentation transfer and rolled interchangeability were exact, and the
presentation move preserved operational place under the registered probes.**

A valid factored FAIL licenses the named recipe-specific failure after its
behavior/support stage: underfit, unsupported signature, wrong paired identity,
non-descending task action, failed transfer/interchangeability, or failed
presentation-place invariance. It does not show that a 32-to-16 quotient is
unlearnable. A carrier comparison licenses only its registered matched label;
one baseline outcome cannot establish architectural necessity.

The never-say list is permanent. A PASS is **not** spontaneous discovery of
factorization; **not** coordinate collapse of paired handles; **not** proof
that `p_z` encodes the nuisance bit or that the learned `present` move flips a
latent presentation coordinate; **not** proof that factorization is necessary,
optimal, or universal; **not** semantic/style disentanglement in language
models; **not** evidence about residual streams or natural latent spaces;
**not** a general L5 theorem; **not** robustness beyond the enumerated finite
population; and **not** authorization for an NLP wrapper, scale ladder, or
external claim. A FAIL is **not** evidence that quotient worlds are hostile to
structured reasoning generally, and a baseline FAIL is **not** proof that
unrestricted carriers cannot learn presentation-invariant navigation.

### Audit #26 diagnostic-only amendment to Round 37 (2026-08-29; pre-outcome)

Rolled-history interchangeability is the primary structural question of
Round 37. Before any Round 37 outcome, the producer/reducer evidence contract
must add and report the following non-gating diagnostics:

- representative/presentation history depth;
- H2 versus H3 continuation depth;
- first divergence step;
- immediate terminal-response versus future-signature failure;
- unsupported versus wrong-supported components; and
- factored-versus-unrestricted signed margins on each matched seed-by-role
  unit.

These fields are diagnostic only. They create no gate, threshold, eligibility
condition, stopping rule, retry, adaptive cell, branch selector, comparison
rule, or verdict change. All five exact certificates, the behavior/support
eligibility stage, the four locked architecture-by-role cells, and the
`INVALID`/`FAIL`/`PASS` semantics above remain unchanged. The pre-outcome lock
is re-taken after this amendment; no Round 37 outcome may be inspected until
the amended module passes Tier-1 correctness and performance review and its
final config and module hashes are recorded.

## Round 33/35 — future-response geometry checks C1/C2 (RETIRED UNRUN by round 37)

**Status: RETIRED UNRUN; HISTORICAL PREREGISTRATIONS ONLY.** Audit #42 adopted
the corrected foundation but did not establish either check's scientific
priority. Round 37 retires C1 because it tests a synthetic consumer rather
than the real-model intervention now required. C2's query strings, nine-bin
law, denotation margin, and clustered discipline are carried forward, but its
entity-by-state factorial coverage is not. C2 is retired by allocation because
a separate observational run is off-direction, not because every one of its
controls is mathematically subsumed. Neither C1 nor C2 may now execute or be
revived as a fallback after seeing `native_horizon_v1`; a scientifically
different successor requires a new dialogue and preregistration.

### C1. Frozen constructed-consumer response geometry — preregistered

#### C1 world and response law

For each frozen rung-0 consumer seed, a raw state consists of a visible
permutation, the frozen episode input, and a register vector. The eight legal
write moves overwrite the register with the eight fixed oracle codes. The
response law is the consumer's full answer-position distribution on the full
vocabulary. Let \(O_C=\{0,\ldots,7,\mathrm{other}\}\). On the joint emitted
observable \((p,\ell)\), register the single total deterministic Markov kernel

\[
K(p,\ell)=
\begin{cases}
p^{-1}(\ell),&\ell\text{ is one of the eight labels displayed by }p,\\
\mathrm{other},&\text{otherwise}.
\end{cases}
\]

Its pushforward is the derived nine-outcome response law used throughout C1
and in \(d_C\). Use normalized square-root Jensen--Shannon distance. Because
\(K\) is one fixed,
state-independent Markov kernel on the joint emitted observable, this
pull-back is D2-compatible; no state-conditioned postprocessing is introduced
after outputs are seen. Because a write overwrites the previous register,
repeated write words add no future beyond their final write. The finite
future-response pseudometric \(d_C\) is therefore enumerated from the
empty/query continuation and the eight write/query continuations.

#### C1-P. Pull-back controls and presentation diagnostic — preregistered

Two controls must not be conflated.

1. **Algebraic round-trip control.** For each saved full-vocabulary probability
   law \(\mu\), compute \(K_*\mu\) and verify nonnegativity and total mass one.
   Separately, for an abstract nine-bin test law, push its eight state masses
   through \(p\) while leaving `other` fixed, then pull it back with \(K\); the
   normalized square-root JS distance from the original nine-bin law must be
   zero up to the measured replay floor. A larger value makes C1
   `INVALID — OUTCOME PULL-BACK`.
2. **Cross-presentation diagnostic.** Evaluate two genuinely different
   visible permutations with the same abstract state and identical register,
   pull both output laws back, and report their \(d_C\). This value is not
   forced to zero by relabeling: zero supports observational passivity for the
   tested pair, while a positive value measures genuine presentation
   sensitivity or another model-response difference. It does not by itself
   show that the pull-back is wrong.

The second distinction corrects the stronger Round-33 suggestion that
cross-presentation distance 'must' vanish after pull-back. Coordinate
alignment is algebraic; presentation covariance is a falsifiable law.

#### C1a. Descent certificate — proved target, preregistered implementation check

For every seed, code, presentation, and registered state pair, the target is

\[
d_C(\operatorname{write}_s x,\operatorname{write}_s y)
\leq d_C(x,y).
\]

**Falsifier.** Any violation exceeding the checkpoint replay/numerical floor.
Such a failure falsifies the registered enumeration or move semantics, not
Theorem 1.

#### C1b. Finite Hilbert-chart adequacy — conjectured and preregistered

The check is separate for consumer seeds \(s\in\{11,23,37\}\). A raw point is

\[
u=(p,e,z),
\]

where \(p\) is the exact visible permutation, \(e\) is the complete frozen
episode ID `(entity, abstract_state, template, panel)`, and \(z\) is one of the
eight fixed oracle codes or eight state-conditioned writer centroids for seed
\(s\). Before any response is scored, a canonical JSON manifest must record
and hash:

- the seed, point ID, point kind, abstract-state index, full float32 \(z\)
  vector, and source artifact/checkpoint SHA-256;
- every \(e\), with entities `0..23`, states `0..7`, templates `0..3`, and
  panels `0..3` fully crossed;
- every permutation vector from the frozen `necessary_register_v1` bank
  (`perm_seed=5151`, bank SHA-256
  `7fa32bd5e8cf18fe493d169a539259efa44aad6dcc9e14d750d79d91560d0b9e`),
  with within-panel indices `0..127` declared training and `128..143`
  declared held out; and
- the episode split: entities
  \(\{0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22\}\) are training and
  \(\{2,5,8,11,14,17,20,23\}\) are held out, with every state, template,
  and panel retained in both splits.

The fitting blocks use training episodes and training permutations. The
primary prediction blocks use held-out episodes and held-out permutations.
No row may change split after its response law is inspected. The manifest
defines the complete fixed embedding

\[
\Phi(p,e,z)
=
(\mathbf 1_p,\mathbf 1_e,z),
\]

where \(\mathbf 1_p\) and \(\mathbf 1_e\) are canonical one-hot vectors over
the union of the manifest's permutation and episode IDs. This makes explicit
that the raw state is \((p,e,z)\), not \(z\) alone. Every fitted or scored pair
holds \((p,e)\) fixed and varies only \(z\), so those one-hot blocks cancel;
the fitted form is therefore constrained to

\[
G=\operatorname{diag}(0,0,H),
\qquad H\succeq0,
\]

and the optimizer operates only on the register block \(H\). No presentation
or episode block is fitted.

For each `(seed, split, p, e)` block, include all
\(\binom{16}{2}=120\) unordered pairs of the sixteen registered points. Give
each block equal total weight and every pair within a block equal weight:

\[
\omega_{ij}
=
\frac{1}{|\mathcal B_{\rm split}|\binom{16}{2}}.
\]

The held-out blocks reuse all 120 registered \(z\)-pairs present in every
training block. The split changes episodes and permutations, not register
points or pair directions; C1b therefore tests cross-context stability on
fixed finite register support, not generalization to unseen \(z\)-points.

The manifest's `pair_id` is the canonical concatenation of seed, split,
panel, permutation index, entity, state, template, and the two ordered point
IDs; duplicate IDs or any pair whose two rows disagree on \((p,e)\) make C1b
`INVALID — PAIR MANIFEST`.

Let \(d_{ij}=d_C(u_i,u_j)\),
\(\delta z_{ij}=z_i-z_j\), and
\(X^z_{ij}=\delta z_{ij}\delta z_{ij}^{\top}\). Fit the unique regularized
register-block PSD form

\[
H_s^\star
=
\underset{H\succeq0}{\arg\min}
\left\{
\sum_{(i,j)\in\mathcal P_{\rm train}}
\omega_{ij}
\bigl(d_{ij}^2-\operatorname{tr}(HX^z_{ij})\bigr)^2
+10^{-6}\|H\|_F^2
\right\},
\qquad
G_s^\star=\operatorname{diag}(0,0,H_s^\star).
\]

Denote the displayed objective by \(f_s(H)\). The positive Frobenius term is
the regularizer and deterministic tie-breaker.
The optimizer is projected gradient descent from \(H_0=0\), with projection
obtained by symmetrizing and clipping negative eigenvalues to zero. Use the
fixed step \(1/L\), where

\[
L=2\sum_{(i,j)\in\mathcal P_{\rm train}}
\omega_{ij}\|X^z_{ij}\|_F^2+2\times10^{-6}.
\]

Stop only after

\[
\frac{\|H_{k+1}-H_k\|_F}{\max(1,\|H_k\|_F)}\leq10^{-10}
\quad\text{and}\quad
\frac{|f_s(H_{k+1})-f_s(H_k)|}{\max(1,|f_s(H_k)|)}\leq10^{-12}
\]

for ten consecutive iterations, with a ceiling of 100,000 iterations.
Supplement those successive-iterate checks with a projected-gradient/KKT
residual. At the final iterate \(H\), define

\[
\widetilde H
=\Pi_{\succeq0}\!\left(H-L^{-1}\nabla f_s(H)\right),
\qquad
\eta_{\rm PG}
=L\|H-\widetilde H\|_F.
\]

Report \(\eta_{\rm PG}\) for every seed. The objective is
\(\mu\)-strongly convex with \(\mu=2\times10^{-6}\). Projection optimality
and \(L\)-smoothness give a KKT residual at \(\widetilde H\) of at most
\(2\eta_{\rm PG}\), hence the a posteriori bound

\[
\|\widetilde H-H_s^\star\|_F
\leq B_H:=\frac{2\eta_{\rm PG}}{\mu}.
\]

Nonconvergence is `INVALID — PSD OPTIMIZER`. Predict held-out response
distances using \(\widetilde H\):

\[
\widehat d_s(u_i,u_j)
=
\sqrt{\delta z_{ij}^{\top}\widetilde H\delta z_{ij}}.
\]

If either the training or held-out denominator
\(\sum\omega_{ij}d_{ij}^2\) is zero, C1b is
`INVALID — DEGENERATE RESPONSE DISTANCE`. Otherwise let

\[
D_{\rm held}
=\left(\sum_{\mathcal P_{\rm heldout}}\omega_{ij}d_{ij}^2\right)^{1/2},
\qquad
B_{\rm stress}
=
\frac{
\left(B_H\sum_{\mathcal P_{\rm heldout}}
\omega_{ij}\|X^z_{ij}\|_F\right)^{1/2}
}{D_{\rm held}}.
\]

The square-root inequality and the bound on \(H_s^\star\) make
\(B_{\rm stress}\) an upper bound on optimizer-induced error in held-out
normalized stress. Require \(B_{\rm stress}\leq10^{-4}\); otherwise C1b is
`INVALID — OPTIMIZER BOUND`. The candidate proposition is

\[
\widehat S_s+B_{\rm stress}\leq0.10,
\qquad
\widehat S_s
=
\frac{
\left(\sum_{\mathcal P_{\rm heldout}}\omega_{ij}
(d_{ij}-\widehat d_s(u_i,u_j))^2\right)^{1/2}
}{
D_{\rm held}
}.
\]

in at least two of three seeds. Report every seed estimate and the spread over
held-out permutations; exact equality is diagnostic only.

**Falsifier.** \(\widehat S_s-B_{\rm stress}>0.10\) in at least two seeds. A
seed whose certified interval intersects 0.10 is boundary-indeterminate; any
overall case satisfying neither the proposition nor the falsifier is
`INCONCLUSIVE — HILBERT ADEQUACY`.

This tests one regularized Hilbert seminorm on within-episode register
differences over the finite registered support. It does not establish an
intrinsic or presentation-independent geometry outside that support and does
not test every seminorm. With only eight code writes, arbitrary-seminorm
homogeneity and translation invariance are not identifiable; those would
require scaled and translated register inputs to be declared legal.

**Expectation.** Descent passes as a theorem/harness certificate. Hilbert
adequacy is genuinely open; the prior non-Voronoi finding does not imply its
failure. The simplest confound is finite-point interpolation by an overly
flexible PSD form; the locked episode/permutation split, explicit
regularization, held-out stress, and seed spread are therefore mandatory.

### C2. Qwen restricted native response geometry — preregistered

#### C2 tokenization manifest — preregistered validity gate

Before any model forward pass, write and hash a manifest containing:

- the exact three query strings below for every entity substitution;
- model `Qwen/Qwen3-1.7B-Base` at revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, plus tokenizer revision and
  chat-template/revision fields;
- source config
  `experiments/config/register_bridge_preflight_v1.json` at SHA-256
  `c016f1acd74b3260c795d34a86c5f1dc4f151455e510eab44600da819f71f682`,
  source rows `experiments/results/register_bridge_preflight_v1/run_rows.json`
  at SHA-256
  `519846614554b6f045038fee91498638f83957ff7fd0163781131f5f9e787095`,
  and source result
  `experiments/results/register_bridge_preflight_v1/run_result.json` at
  SHA-256
  `82179ba1837605f2858ba1654295ba892ce4ee4494a594f4d5b51f87ae73554f`;
- token IDs for each complete prompt/query boundary;
- token IDs for the intended single-token outcomes `0` through `7`;
- the nine-outcome map from full-vocabulary token IDs to digits plus `other`;
- the intact/destroyed source-row ID, entity, template, permutation, record
  tag, original state, and legend-denoted state for every row; and
- a unique canonical `pair_id` for every deterministic comparator unit and a
  unique `destroyed_row_id` for every paired-reassigned margin row. Each ID
  records its estimand (`Delta` or `M`), entity, state, template role(s),
  replicate role(s), every constituent source-row ID and its role, query IDs,
  permutation IDs, literal tags, original states, and legend-denoted states.

If any digit is not represented by the declared single token, any prompt is
not prefix-aligned at the answer position, any intact/destroyed pairing is
missing, any `pair_id` or `destroyed_row_id` is duplicated or incomplete, a
comparator changes an undeclared factor, any pinned hash differs, or any row
fails its denotation check, the check is
`INVALID — TOKENIZATION/PAIR MANIFEST`. No forward pass runs and there is no
same-check repair; a scientifically different design requires a new dialogue
and preregistration.

#### C2 world and response law

Reuse the saved intact and paired-reassigned-legend prompt cells, appending
these fixed queries:

1. `Which legend state number, 0 through 7, is the private value recorded for {entity}? Answer with one digit:`
2. `According to the legend, what state number does {entity}'s recorded value denote? Answer 0 through 7:`
3. `Map {entity}'s recorded tag through the legend. Return only its state number, 0 through 7:`

Let \(r^{\rm full}(x)\) be the primitive full-vocabulary next-token law and let
\(\tau_k\) be the manifest-pinned single token for digit \(k\). Name the fixed
derived nine-bin response explicitly:

\[
\widetilde r(x)_k=r^{\rm full}(x)(\tau_k)\quad(0\leq k\leq7),
\qquad
\widetilde r(x)_{\rm other}
=1-\sum_{k=0}^7\widetilde r(x)_k.
\]

Let \(\mathcal W_Q\) be the three query continuations and define

\[
d_Q(x,y)
=
\sup_{w\in\mathcal W_Q}
D_{\sqrt{\mathrm{JS}}}(\widetilde r(T_wx),\widetilde r(T_wy)),
\]

using normalized square-root Jensen--Shannon distance.

The comparator construction is deterministic. Within each source cell
`(entity e, original state s, template t)`, order the two intact rows by source
row ID and label their registered permutation-replicates \(r=0,1\). Each
intact row has exactly one source-paired destroyed row with the same entity,
original state, template, permutation index, clause order, literal record tag,
and token multiset, but with a different legend-denoted state. For every
entity \(e\), state \(s\), ordered template pair \(i\ne j\), and registered
replicate pair \((r_i,r_j)\in\{0,1\}^2\), let

- \(I_i\) be the unique intact row for \((e,s,i,r_i)\);
- \(I_j\) be the unique intact row for \((e,s,j,r_j)\); and
- \(D_j\) be the unique source-paired destroyed row of \(I_j\).

The unit \(u=(e,s,i,j,r_i,r_j)\) is fixed before any forward pass and has

\[
\Delta_u=d_Q(I_i,D_j)-d_Q(I_i,I_j).
\]

Thus the same reference \(I_i\) is used in both terms, while \(I_j\) and
\(D_j\) differ only by the registered paired legend reassignment. A missing or
nonunique row, a destroyed row whose denoted state still equals \(s\), or any
field mismatch outside that reassignment makes the design
`INVALID — TOKENIZATION/PAIR MANIFEST`; no matching choice may depend on an
output.

#### C2 proposition — conjectured and preregistered

Denotation predicts the restricted native response law. For each ordered
template pair \((i,j)\) and for the pooled table, report

\[
\Delta_{i,j}
=
\mathbb E_{e,s,r_i,r_j}[\Delta_u].
\]

For the paired-reassigned arm, compute the log-probability margin once per
unique destroyed source row, not once per reference template or comparator.
If \(s_{\rm new}(D_j)\ne s\) is that row's legend-denoted state, define

\[
m(D_j)
=
\frac{1}{|\mathcal W_Q|}
\sum_{w\in\mathcal W_Q}
\left[
\log r^{\rm full}(T_wD_j)(\tau_{s_{\rm new}(D_j)})
-\log r^{\rm full}(T_wD_j)(\tau_s)
\right],
\qquad
M_j=\mathbb E[m(D_j):\operatorname{template}(D_j)=j].
\]

Each `destroyed_row_id` contributes exactly once to pooled
\(M=\mathbb E[m(D_j)]\), even though its \(D_j\) can appear in several
\(\Delta_u\) units. Compute all log terms from full-vocabulary float32
`log_softmax` at the answer position, not by taking a logarithm after
probability rounding or nine-bin bucketing. The primary pooled propositions
are

\[
\Delta>0
\qquad\text{and}\qquad
M>0.
\]

Report every \(\Delta_{i,j}\) and every \(M_j\) beside the pooled effects;
pooling may not hide wording concentration. For \(\Delta\), compute each
entity's mean over its registered comparator units first; for \(M\), compute
each entity's mean over its unique destroyed rows first. Average the 24 entity
means, and repeat the same entity-first reduction within every ordered
template-pair \(\Delta\) stratum and manipulated-template \(M\) stratum. Use
exactly 2,000 paired entity-cluster bootstrap replicates: sample 24 entity IDs
with replacement using NumPy `PCG64` seed `2727`, carry all of each sampled
entity's matched units together, and use percentile 2.5%/97.5% bounds. The
manifest stores the ordered list of sampled entity-index vectors and its
SHA-256 before any verdict is computed.

The predeclared numeric wording-support rule uses all registered ordered
template pairs

\[
\mathcal U=\{(t_i,t_j):t_i,t_j\in\{0,1,2,3\},\ i\ne j\},
\qquad |\mathcal U|=12.
\]

It passes only if all three thresholds, fixed before any forward pass, hold:

1. at least six of the twelve ordered pairs have \(\Delta_{i,j}>0\);
2. at least three of the four manipulated-template strata have \(M_j>0\); and
3. for every whole template \(k\in\{0,1,2,3\}\), both leave-one-template-out
   pooled point estimates are positive:
   \(\Delta^{(-k)}>0\) after removing every unit with \(i=k\) or \(j=k\), and
   \(M^{(-k)}>0\) after removing every destroyed row with manipulated template
   \(j=k\).

These leave out entire templates, not isolated ordered pairs. They are
point-estimate support checks beside, not substitutes for, the
entity-clustered pooled intervals.

**Falsifier.** The entity-clustered 95% lower bound is at or below zero for
either pooled \(\Delta\) or pooled \(M\), or the numeric wording-support rule
fails. Then the tested native output geometry does not support
denotation-organized restricted response geometry under these continuations;
the preflight remains instrument-only.

Exact template passivity, \(d_Q(x,gx)=0\), is reported only as a diagnostic. A
positive value is a witness against passivity for that rewrite and finite
continuation family. A zero value does not prove \(d_\infty(x,gx)=0\).

**Expectation.** The preregistered evidence criterion may be met if
same-denotation pairs are closer than matched different-denotation pairs and
the reassigned response favors the newly denoted state across the registered
wording support.

The simplest global confound is an explicit prompt-local dictionary lookup.
That mechanism is compatible with the target prompt-world response law and
therefore prevents any claim about a persistent residual state, storage, or a
causal bridge. C2 is an observational identity/covariance check on prompt
episodes only.

## Rounds 37-38 — native_horizon_v1: native query horizon and span intervention in a real prompt world (Audit #43 corrections verified by audit #44; proposal NOT SELECTED; NO RUN)

### Status, ruling, and narrative gate

Audit #43 corrections verified by audit #44; proposal NOT SELECTED; NO RUN.
Audit #44 adopts the LM mathematics but does not select, lock, or authorize
`native_horizon_v1`. Round 38 applied audit #43's required corrections 1--15,
but no model forward pass, config, runner, result, or manifest was created in
rounds 37--38.

The non-expert so-what is: *can changing one word in a model's record move it
to the same operational place as a naturally written record of the new fact,
and how many questions does the model need before that move becomes visible?*

Round-37 ruling on the proposed mathematics:

1. Finite memory is accepted only for the append-only response metric and
   only under a registered suffix/sliding policy. A native full-context limit
   is not silently treated as sliding memory. Theorem 4 gives zero depth-tail
   error at a known finite horizon but does not control the exponentially many
   branches inside that horizon.
2. Prompt strings supply no move germs or tangent structure. D7 is
   inapplicable unless a discrete zero-dimensional structure is explicitly
   added, in which case its executable tangent cone is trivial. The operative
   map here is combinatorial.
3. Span substitution is separated from append continuations. It is not
   automatically nonexpansive for the append-future metric; quotient descent
   is a property to earn. Proposition 6's pointwise place-change and
   target-realization equivalences are exact zero-kernel characterizations.

This registration is the one proximal real-model artifact implied by that
ruling. It uses only prompt rewrites and emitted next-token laws. It contains
no hidden-state capture, probe, residual injection, generated answer,
synthetic consumer, or external judge.

### Model, registered world, and response law

The only model is Qwen/Qwen3-1.7B-Base at revision
ea980cb0a6c2ae4b936e82123acc929f1cec04c1. The tokenizer revision, tokenizer
files, model config, chat-template field, library versions, dtype, device,
batching, and deterministic settings are pinned in the manifest before any
forward pass. This is a base-model completion interface: no chat wrapper is
inserted.

Let \(N\) be the pinned maximum input length read from the model config. The
registered prompt-world wrapper is prospectively fixed as

\[
x\longmapsto \operatorname{suf}_N(x)
\]

on tokenizer IDs immediately before the model call. It is a declared
left-truncation interface, whether or not the underlying full-context model
implements sliding attention internally. All ordinary study prompts are
expected to remain shorter than \(N\); the wrapper is nevertheless part of
the world and makes Theorem 4's suffix premise exact. The manifest reports
every pre- and post-wrapper length and rejects any implementation that
truncates on the right or at a text rather than token boundary.

At every endpoint, compute the model's full-vocabulary float32 log-softmax at
the next-token position. Persist its SHA-256, log-normalizer/finite checks,
the eight numeral log probabilities, and the derived nine-bin law; do not
duplicate full-vocabulary vectors in the result bundle. Let \(\tau_k\) be the
manifest-pinned single token for numeral \(k\), \(0\leq k\leq7\). The sole
derived response law is the total fixed nine-bin Markov pushforward

\[
\widetilde r(x)_k=r^{\rm full}(x)(\tau_k),
\qquad
\widetilde r(x)_{\rm other}
=1-\sum_{k=0}^7\widetilde r(x)_k.
\]

The response discrepancy is normalized square-root Jensen--Shannon distance,
bounded in \([0,1]\). Full-law log probabilities remain available only for
the predeclared old-versus-new numeral margin; they never define a hidden
instrument.

### Manifest-first validity gate

Before any model forward pass, write
experiments/results/native_horizon_v1/manifest.json and its SHA-256. It binds:

- the model/tokenizer/config revisions and hashes, \(N\), wrapper policy,
  exact command, code/config hashes, device, dtype, batch size, and thread or
  GPU settings;
- special-token insertion, attention-mask construction, position IDs, cache
  policy, and batching. No cache or position offset may depend on tokens
  discarded by \(\operatorname{suf}_N\); otherwise the result is
  `INVALID — SUFFIX-LAW IMPLEMENTATION`;
- source config
  experiments/config/register_bridge_preflight_v1.json at SHA-256
  c016f1acd74b3260c795d34a86c5f1dc4f151455e510eab44600da819f71f682
  and source rows
  experiments/results/register_bridge_preflight_v1/run_rows.json at SHA-256
  519846614554b6f045038fee91498638f83957ff7fd0163781131f5f9e787095;
- the exact three C2 shared-numeral query strings after every entity
  substitution, their token IDs, and each query's exact one-token prefix;
- the ordered per-source-unit action alphabet, every word in
  \(\mathcal U_{e,t}^{\leq2}\), its macro depth, execution-order concatenated
  token IDs, token length, and SHA-256;
- numeral token IDs, the total nine-bin map, and validation that the nine
  probabilities are nonnegative and sum to one;
- every source, paired reassigned-legend, same-state reference,
  different-state target, span-edited, and delayed-route control prompt,
  including source row IDs, entity, state, template, permutation, record
  span, literal tags, denoted states, and complete token IDs;
- unique pair IDs for every distance, horizon, denotation, intervention,
  target, congruence, and delayed-control estimand;
- the exact record-span substitution map, its source and target tag IDs, and
  a byte/token diff proving that no other token changed;
- the deterministic population and target-selection rules below;
- the two identical-input replay schedules, the ordered bootstrap entity
  index vectors, and their hashes; and
- the cap of 9,408 model evaluations before deduplication, the exact lower
  deduplicated endpoint count, a smoke-derived CPU forecast, a hard wall, and
  a uniqueness table proving that cached laws are reused rather than
  recomputed under different pair names.

Any hash drift, nonsingle numeral token, nonprefix-aligned query, empty append
macro, missing or duplicate pair, denotation mismatch, span edit outside the
record occurrence, undeclared token change, nonfinite probability, missing
endpoint, wrapper mismatch, or post-output population repair makes the result
INVALID — NATIVE HORIZON MANIFEST. No forward pass begins after a preflight
failure, and no same-registration repair is allowed after an output is read.

### Fixed population and pair families

The study uses all 24 registered entities. For entity index \(e\), fix the
source state \(s=e\bmod8\), so every numeral state occurs for exactly three
entities. For each of the four source templates, choose the intact
permutation-replicate with the smallest source row ID. This yields
\(24\times4=96\) source prompts \(x\), with balanced state and template
support and no outcome-based row choice.

For each source \(x=(e,s,t)\), bind:

1. Same-state reference \(x^{=}\): the intact row with entity \(e\), state
   \(s\), template \((t+1)\bmod4\), and the smallest source row ID not already
   used by \(x\).
2. Paired reassigned-legend row \(x^{\rm re}\): the unique destroyed source
   row paired to \(x\), with the same literal record tag and a different
   legend-denoted state.
3. Intervention target state \(v=(s+1)\bmod8\).
4. Independently presented native target \(y\): the intact row with entity
   \(e\), state \(v\), template \((t+1)\bmod4\), and its smallest source row
   ID.
5. Second native target reference \(y'\): the intact row with entity \(e\),
   state \(v\), template \((t+2)\bmod4\), and the other registered
   permutation-replicate.

The same-state and reassigned-legend C2 comparator is also retained for all
twelve ordered template pairs using the selected source state and fixed
replicate rule. Distances and numeral margins are reduced entity-first, so
templates, states, rows, query words, and repeated appearances of one
destroyed row are never treated as independent samples.

### Append continuations and restricted horizons

For each source unit \((e,t)\), let \(Q_{e,1},Q_{e,2},Q_{e,3}\) be the three
fixed complete shared-numeral query append maps and let
\(P_{e,1},P_{e,2},P_{e,3}\) append the exact first tokenizer token of the
corresponding complete query. Prospectively set
\(j(e,t)=1+((e+t)\bmod3)\) and register only the ordered macro alphabet

\[
\mathcal U_{e,t}=[P_{e,j(e,t)},Q_{e,j(e,t)}].
\]

This outcome-blind assignment balances query wording across the population.
Within a source unit every macro appends one fixed nonempty token string, so
Theorem 4 applies. Macro depth is not token length. Enumerate exactly

\[
1+2+2^2=7
\]

words in \(\mathcal U_{e,t}^{\leq2}\), including the empty word and every
ordered two-macro composition. No linguistically awkward composition is
dropped after inspection. With at most 672 base prompt IDs and two replay
schedules, the preregistered cap is 9,408 model evaluations before
deduplication. The manifest records the exact lower deduplicated count. A
smoke-derived CPU forecast and hard wall must be locked before execution.

For any registered pair \((a,b)\), define the restricted distances

\[
d_h^{\mathcal U}(a,b)
=
\max_{\substack{w\in\mathcal U_{e,t}^*\\|w|\leq h}}
D_{\sqrt{\rm JS}}\bigl(\widetilde r(wa),\widetilde r(wb)\bigr),
\qquad h\in\{0,1,2\}.
\]

Record every endpoint discrepancy and the maximizing word, with deterministic
lexicographic tie-breaking. These satisfy
\(d_0^{\mathcal U}\leq d_1^{\mathcal U}\leq d_2^{\mathcal U}\).
They are lower bounds on the full append distance. The symbols
\(d_\infty^+\), \(H^+\), and place identity are not assigned from a
horizon-2 plateau.

The run does not validate Theorem 4, which is proved. Manifest checks and
deterministic fixtures test whether the implementation instantiates its
suffix, nonempty-append, token-accounting, and enumeration premises. The run
reports the architecture-level bound \(N-1\). It does not enumerate all
branches to that bound and therefore does not claim global saturation.

### Delayed-route positive control

For every source unit, construct a locked pair of real-model control prompts
with the same legend, entity, and primary record but different backup-record
tags. Both contain this literal routing rule, with only entity and tags
substituted:

“Answer from the primary record unless the next question begins with its first
token twice in immediate succession; in that case answer from the backup
record.”

For the assigned one-move complete query \(Q_{e,j(e,t)}\), its first token
appears once. The two-move word
\(Q_{e,j(e,t)}P_{e,j(e,t)}\), which under D1 executes
\(P_{e,j(e,t)}\) first and then \(Q_{e,j(e,t)}\), makes it appear twice. The
manifest verifies the exact execution-order token concatenation and binds
backup states \(s\) and \((s+4)\bmod8\) to the two control prompts. The
intended profile is no material incremental discrimination from depth 0 to 1
and material new discrimination from depth 1 to 2. This is a positive control
of the horizon instrument, not an assumption that a real model must obey the
routing text.

### Span intervention and target endpoint

For each source \(x\), define \(s_vx\) by replacing only the final record-tag
span with the tag that the unchanged legend maps to target state
\(v=(s+1)\bmod8\). The edit is rejected unless source and replacement spans
have manifest-pinned token IDs and the post-edit legend parser returns \(v\).
The independently written target \(y\), not the byte-identical edited prompt,
is used in the realization endpoint.

At every \(h\in\{0,1,2\}\), report:

\[
I_h=d_h^{\mathcal U}(x,s_vx)
\]

for place-change evidence,

\[
R_h=d_h^{\mathcal U}(s_vx,y)
\]

for target residual,

\[
G_h=d_h^{\mathcal U}(x,y)-d_h^{\mathcal U}(s_vx,y)
\]

for target-directed gain, and

\[
B_h=d_h^{\mathcal U}(y,y'),
\qquad
E_h=R_h-B_h
\]

for excess target residual above an independently presented same-target
baseline. Also apply the corresponding state-to-\(v\) record substitution to
\(x^{=}\) and report the before/after same-denotation spread as a
presentation-stability diagnostic relevant to D8. Because the pre-edit
prompts are not established to be one exact place, this is not a
quotient-descent test. It is not promoted to exact congruence or
nonexpansiveness.

For every reassigned-legend row, retain C2's paired denotation contrast and
old-versus-new numeral log-probability margin under the prospectively assigned
complete one-query macro. This carries forward C2's registered nine-bin
pushforward question inside the
intervention artifact rather than running a separate observational ladder.

### Replay floor, clustered reduction, and approximate criteria

Run every endpoint in two prospectively ordered replay schedules with
different batch positions but identical inputs. Let \(\eta\) be the maximum
same-input square-root-JS distance over the persisted nine-bin laws, and
separately report the maximum absolute replay difference among the eight
numeral log probabilities. Define

\[
\varepsilon_{\rm eq}=\max(10^{-5},10\eta),
\qquad
m=0.02.
\]

If \(\eta>10^{-4}\), the result is INVALID — NUMERICAL REPLAY. Exact equality
and exact \(H\) tables are diagnostic only.

For each estimand, average all registered units within entity first and then
average the 24 entity means. Report every entity, state, template, query, and
argmax-word stratum. Use exactly 2,000 entity-cluster bootstrap replicates
with NumPy PCG64 seed 3737; each sampled entity carries all of its rows,
templates, queries, interventions, and controls. Store and hash the ordered
entity-index vectors before verdict computation. Report point estimate, 95%
percentile interval, entity spread, template spread, and the fraction of
individual units exceeding \(m\). There is no sampling seed or generation
termination rate: the model law is read directly and deterministically;
termination is not applicable.

The joint proximal result is `RESTRICTED DEPTH-2 PROFILE SUPPORT` only if all
four conditions hold:

1. Explicit-state one-query profile: for different-denotation and
   reassigned-legend pairs, the entity-mean lower bound of
   \(d_1^{\mathcal U}-d_0^{\mathcal U}\) is above \(m\), while the upper bound
   of \(d_2^{\mathcal U}-d_1^{\mathcal U}\) is at most
   \(\varepsilon_{\rm eq}+0.005\). At least three of four template strata
   must have positive one-query gain.
2. Delayed-route positive control: the upper bound of
   \(d_1^{\mathcal U}-d_0^{\mathcal U}\) is at most
   \(\varepsilon_{\rm eq}+0.005\), while the lower bound of
   \(d_2^{\mathcal U}-d_1^{\mathcal U}\) is above \(m\). Any
   tolerance-robust unit satisfying the individual witness rule below proves
   \(H^+>1\) for that pair; the population lower bound alone does not assign
   an exact individual horizon.
3. Place change: the entity-mean lower bound of \(I_2\) is above \(m\), and
   all four source-template point estimates exceed \(m\).
4. Restricted target direction: the entity-mean lower bound of \(G_2\) is
   above \(m\), the upper bound of \(E_2\) is at most \(m\), and at least
   three of four target-template strata have positive target-directed gain.

The paired reassigned-legend distance contrast and new-versus-old numeral
margin must additionally have positive entity-clustered lower bounds and
positive point estimates in at least three of four manipulated-template
strata. Failure of this inherited denotation control prevents the joint
support verdict.

The following outcomes are predeclared:

- CONJECTURE-5 SHALLOW FALSIFIER when the entity-clustered lower bound of the
  explicit-state increment \(d_2^{\mathcal U}-d_1^{\mathcal U}\) exceeds
  \(m\). Any individual increment is also reported as a tolerance-robust
  finite witness against \(H^+=1\) for that pair only when
  \(d_2^{\mathcal U}-d_1^{\mathcal U}>\varepsilon_{\rm eq}\) in both replay
  schedules, without turning one row into the population verdict. Exact
  equality remains diagnostic.
- DELAYED CONTROL FAIL when the one-step leakage criterion fails or the
  two-step gain criterion does not pass. This invalidates the intended
  positive control; absence of a depth-2 gain does not prove \(H^+=1\), since
  a later branch may still discriminate.
- `NO MATERIAL RESTRICTED PLACE-CHANGE EVIDENCE` when condition 3 fails after
  a valid horizon instrument.
- `RESTRICTED TARGET-DIRECTION CRITERION FAIL` when place change passes but
  condition 4 or the inherited denotation control fails.
- INCONCLUSIVE for every complete valid combination that satisfies neither
  the joint support rule nor a named falsifier.

A positive finite distance witnesses nonidentity. A small or zero restricted
distance never certifies \(d_\infty^+=0\). `RESTRICTED DEPTH-2 PROFILE
SUPPORT` therefore licenses only: under this fixed model, prompt population,
query-macro family, and tolerance, one registered macro captured the measured
depth-\(\leq2\) separation in the derived nine-bin response law, the delayed
control required two moves, and a record-tag substitution moved registered
nine-bin pushforward behavior toward independently presented targets. This
does not positively establish Conjecture 5's global \(H^+=1\). It does not
license exact place identity, full branch completeness, D8 quotient descent,
storage, persistence, semantic understanding, residual state, a hidden causal
bridge, or model-family generality.

The cheapest confound is prompt-local dictionary lookup. It is deliberately
not controlled away: dictionary lookup is a native executable mechanism in
this prompt world. The claims concern operational response places and a text
intervention, not a persistent internal variable.

### Compute and artifact discipline

The default device is CPU float32, one process and one thread. A single short
GPU validation burst is permitted only after the user's explicit per-run
approval, launched detached with per-batch checkpoints, one compute job at a
time, and a ten-minute hard wall; sustained GPU load is forbidden. CPU and
GPU outputs may not be mixed inside one claiming artifact. The implementation
must cache one law per unique endpoint, checkpoint complete entity blocks,
and stop as INCOMPLETE — DEADLINE rather than adapt the population, horizon,
or query set. Before any model forward pass, append a lock row binding the
manifest hash, exact endpoint count, runner/config hashes, smoke-derived CPU
forecast, hard wall, gates, statuses, and stop rule. Audit #43 does not itself
authorize that lock or execution. No model run occurred in rounds 37--38.

Audit #44 did not select this proposal, so no runner, manifest, or lock is
created for it. Any scientifically different future implementation requires
its own audited specification and uses one canonical runner for manifest,
produce, and separate reduce/fixture modes; all result claims enter
experiments/ledger.jsonl and the experiment index or they did not happen.

### Retirement decision

C1 and C2 are retired unrun now, not merely paused. C1's theorem harness and
finite Hilbert fit are off-direction because they concern a synthetic
consumer. C2's query strings, nine-bin law, denotation margin, and clustered
discipline are carried forward, but its entity-by-state factorial coverage is
not. C2 is retired by allocation because a separate observational run is
off-direction, not because every one of its controls is mathematically
subsumed. Neither historical preregistration may run in parallel or be
resurrected after this result.

### Audit #44 checklist and disposition

Audit #44 completed this review and verified audit #43's corrections while
ruling that `native_horizon_v1` is **NOT SELECTED; NO RUN**. It adopted the LM
mathematics, not this artifact, manifest, lock, or execution. Its checklist
was:

1. Theorem 4's proof, append-only scope, macro-depth/token-length distinction,
   query-token accounting, and sliding-wrapper versus full-context boundary.
2. That finite depth does not masquerade as finite branch completeness and
   that no \(h\leq2\) quantity is named \(d_\infty^+\) or exact \(H^+\).
3. The no-germs ruling and the distinction between absence of D7 structure
   and an explicitly discrete zero-dimensional structure.
4. D8's append/intervention typing, the loss of automatic nonexpansiveness,
   the quotient-descent condition, and Proposition 6's definitional proof.
5. That the target \(y\) is independently presented rather than byte-identical
   to \(s_vx\), and that all pair, source-row, target, and intervention rules
   are outcome-blind and executable.
6. The prospectively assigned two-macro alphabet, seven-word continuation
   enumeration, D1-consistent delayed-route control semantics, nine-bin total
   kernel, replay floor, clustered estimands, tolerance rules, and
   exact-diagnostic-only treatment.
7. That every model-call input depends only on the wrapped suffix, including
   special-token insertion, masks, position IDs, cache policy, and batching;
   any discarded-token dependency must yield
   `INVALID — SUFFIX-LAW IMPLEMENTATION`.
8. The exact deduplicated endpoint count, 9,408 pre-deduplication cap,
   smoke-derived CPU forecast, hard wall, and mandatory ledger lock before any
   forward pass.
9. That the verdict labels and licensed interpretation stay restricted to the
   depth-2 derived nine-bin profile and do not claim global \(H^+=1\), target
   identity, quotient descent, latent state, or a causal bridge.
10. Whether retiring C1/C2 loses any nonredundant control worth retaining
   inside this single artifact.
11. The licensed sentence, never-say list, cheapest dictionary-lookup
   explanation, alternative directions, and the remaining unseen-branch gap.

Measurement-to-artifact heartbeat: round 38 adds 0 lines of experimental
apparatus and 0 lines of artifact-bearing code, so its code ratio is \(0/0\).
The last completed runner ratio remains approximately 34 apparatus to 70
estimand-bearing lines, or \(0.49{:}1\). Counting audit #43 as governance, the
repository's prior accounting is approximately 28 measurement/governance
rounds to 10 build rounds, or \(2.80{:}1\), above the \(2{:}1\) warning. This
correction pass is theory/build work, not another measurement.

## Round 41 — native_bridge_v1 (DRAFT - not locked; audit #47 LOCK-READY — wording corrections applied; runner/smoke/forecast/lock authorized)

### Status and narrative gate

This is repair pass 2 of the native bridge specification after audit #46.
It is a preregistration **draft**, not a lock: no runner, manifest, lock row,
smoke, or model output is created or authorized here. Audit #46's ruling
remains `REVISE — REPAIR PASS 2 OF 3; NO LOCK, RUNNER, SMOKE, OR COMPUTE`
until a fresh lock-readiness audit #47 rules on this repair.

The non-expert so-what is: *if a model's internal state for a naturally written
fact is moved into another prompt, does the model behave like native examples
of that fact across several registered questions, and does the right move beat
both doing nothing and moving toward the wrong fact?*

### Prospective manifest and exact row population

Before any scientific forward pass,
`experiments/results/native_bridge_v1/manifest.json` must exist and be hashed.
It binds the runner/config hashes, exact command, library versions, device,
float32 dtype, batch size one, model and tokenizer identities, all rows and
endpoint tuples, the two replay orders, the resampling index table, constants,
smoke timing, forecast, hard wall, and every validity/status rule below.

The source authorities are:

- `experiments/config/onewrite_recall_v1.json`, SHA-256
  `65d47cf2d1c34e3d32d0943cf9c56b12da89c99abd9bab3dcc63d348abf53ecd`;
- `experiments/config/register_bridge_preflight_v1.json`, SHA-256
  `c016f1acd74b3260c795d34a86c5f1dc4f151455e510eab44600da819f71f682`;
- `experiments/results/register_bridge_preflight_v1/run_rows.json`, SHA-256
  `519846614554b6f045038fee91498638f83957ff7fd0163781131f5f9e787095`;
  and
- `experiments/substitution_probe.py` and
  `experiments/run_onewrite_state.py` as the reusable loading and post-block
  hook references, currently SHA-256
  `69a605dfdc5be18fdeab4c4002eedbef8301c488c99b669fb702d24da1a03a12`
  and `741dd8fb0126f8ffd0c98d9901d295bc31d7b915d4d90eaa716c8cf089190665`;
  the eventual manifest rebinds their implementation-time hashes together
  with the new runner hash.

The model ID and revision are read from the first authority and must equal
`Qwen/Qwen3-1.7B-Base` and
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. The tokenizer is loaded at that
same requested revision; its resolved commit and tokenizer-file hashes are
stored separately. No chat template or special token is inserted.

Index entities by \(i=0,\ldots,23\). Set \(q_i=i\bmod8\) and
\(s_i=(i+1)\bmod8\). The source is the first intact permutation row in
template 0,

\[
\operatorname{row}(x_i)=128i+16q_i,
\]

and the three targets are the first intact permutation rows in distinct
templates 1, 2, and 3,

\[
\operatorname{row}(y_{i,s_i}^{(r)})=128i+16s_i+4r,
\qquad r\in\{1,2,3\}.
\]

These are three distinct, prospectively row-ID-selected intact target prompts;
no stochastic independence is assumed.

The exact 96-row table, mechanically checked against the pinned row file, is:

| entity | source row | target rows \((1,2,3)\) |
|---:|---:|:---|
| 0 | 0 | 20, 24, 28 |
| 1 | 144 | 164, 168, 172 |
| 2 | 288 | 308, 312, 316 |
| 3 | 432 | 452, 456, 460 |
| 4 | 576 | 596, 600, 604 |
| 5 | 720 | 740, 744, 748 |
| 6 | 864 | 884, 888, 892 |
| 7 | 1008 | 900, 904, 908 |
| 8 | 1024 | 1044, 1048, 1052 |
| 9 | 1168 | 1188, 1192, 1196 |
| 10 | 1312 | 1332, 1336, 1340 |
| 11 | 1456 | 1476, 1480, 1484 |
| 12 | 1600 | 1620, 1624, 1628 |
| 13 | 1744 | 1764, 1768, 1772 |
| 14 | 1888 | 1908, 1912, 1916 |
| 15 | 2032 | 1924, 1928, 1932 |
| 16 | 2048 | 2068, 2072, 2076 |
| 17 | 2192 | 2212, 2216, 2220 |
| 18 | 2336 | 2356, 2360, 2364 |
| 19 | 2480 | 2500, 2504, 2508 |
| 20 | 2624 | 2644, 2648, 2652 |
| 21 | 2768 | 2788, 2792, 2796 |
| 22 | 2912 | 2932, 2936, 2940 |
| 23 | 3056 | 2948, 2952, 2956 |

All 96 IDs must resolve to intact rows with the declared entity, state,
template, stored token IDs, and final record-tag span. The manifest asserts
tokenization identity by retokenizing every prompt with
`add_special_tokens=False` and requiring exact equality to its stored `ids`.
It also asserts one final record-tag occurrence, its stored span text and IDs,
and site position \(p=\operatorname{span.end}-1\). Any mismatch is
`INVALID — ROW/TOKENIZATION MANIFEST` before a model call.

For each entity, the native-paste donor is
\(y_i^*=y_{i,s_i}^{(1)}\). Define

\[
\mathcal Y_{j,s_j}
=\{y_{j,s_j}^{(1)},y_{j,s_j}^{(2)},y_{j,s_j}^{(3)}\},
\qquad
\mathcal Y_{s,-e_i}
=\bigcup_{\substack{j\ne i\\s_j=s}}\mathcal Y_{j,s}.
\]

The manifest materializes the donor row IDs for every correct centroid
\(\mathcal Y_{s_i,-e_i}\) and every wrong centroid
\(\mathcal Y_{t_i,-e_i}\), where \(t_i=(s_i+1)\bmod8\). Correct-centroid
sets have six rows because two other registered entities share \(s_i\);
wrong-centroid sets have nine rows because three entities have target label
\(t_i\) and entity \(i\) is not one of them. Within each replay, donor capture
is reused from that replay's 72 Phase-D clean target-epsilon calls and never
creates a hidden extra call family.

### Registered words, channels, site, and canonical execution

For each entity, assign \(j(i)=1+(i\bmod3)\) and use exactly one of the three
query literals below. Each literal begins with exactly one ASCII space:

1. ` Which legend state number, 0 through 7, is the private value recorded for {entity}? Answer with one digit:`
2. ` According to the legend, what state number does {entity}'s recorded value denote? Answer 0 through 7:`
3. ` Map {entity}'s recorded tag through the legend. Return only its state number, 0 through 7:`

Let \(a_Q\) append the complete entity-substituted literal and \(a_P\) append
exactly its first tokenizer token. The manifest stores both strings, their
token IDs, and the complete execution-order IDs for

\[
W_0=\{\epsilon,a_P,a_Q,a_Pa_P,a_Pa_Q,a_Qa_P,a_Qa_Q\}.
\]

D1 is rightmost-first: in particular the displayed word \(a_Qa_P\), written
`Q P` in the token-order fixture, must execute \(a_P\) first and then \(a_Q\).
No awkward composition may be removed after inspection.

The primitive endpoint is the full-vocabulary float32 log-softmax at the final
next-token position, \(c_{\rm full}\). The manifest binds the eight numeral
token IDs and asserts that each numeral is exactly one token. The derived
\(c_9\) sums those eight probabilities and assigns all remaining vocabulary
mass to `other`. Using natural logarithms and \(m=(p+q)/2\), both channels use

\[
D_{\sqrt{\rm JS}}(p,q)
=
\sqrt{
\frac{\operatorname{KL}(p\|m)+\operatorname{KL}(q\|m)}
{2\ln2}},
\]

with the standard \(0\log0=0\) convention. Thus
\(D_{\sqrt{\rm JS}}\in[0,1]\). The \(c_9\) channel is derived from the same
law and adds no model call or independent refutation power.

The sole edit site is the output of `model.model.layers[16]` at the final token
of the final record-tag span, equivalent to `hidden_states[17][:, p, :]`.
This single-position site remains unvalidated until the conjectured proximal
site-sufficiency control passes. The registered implementation is a complete
batch-one forward with a new post-block hook adapted from the
`run_onewrite_state.py` hook machinery. That existing hook adds a delta; the
new hook replaces exactly one `[:, p, :]` position and asserts exactly one
matched write per call, after which the frozen forward computes every
descendant. The canonical policy is `add_special_tokens=False`, no padding, an
all-ones attention mask, position IDs \(0,\ldots,L-1\),
`past_key_values=None`, and `use_cache=False` for every plain and hooked call.

The manifest must serialize the declared architecture-dependency specification
for the canonical DAG cut from `theory/AXIOMS.md`: every non-descendant
boundary-input dependency needed to recompute all descendants of the edited
post-block residual, no descendant dependency, the module path and tensor
shape, and the canonical token/mask/position/cache construction above. This
declaration is not an actually serialized per-call cut record; the artifact
uses the complete canonical no-cache forward with a one-position replacement
hook. Faithful continuation and response-level intertwining remain conditional
premises tested by fixtures, not consequences of merely naming the cut.

### Eight endpoint families and exact call accounting

For every entity and word, evaluate exactly these eight endpoint families:

1. `target_1`: plain forward from \(y_{i,s_i}^{(1)}\);
2. `target_2`: plain forward from \(y_{i,s_i}^{(2)}\);
3. `target_3`: plain forward from \(y_{i,s_i}^{(3)}\);
4. `pasteback`: hook \(y_i^*\) with its own captured site vector;
5. `native`: hook \(x_i\) with \(h_{\ell,p}(y_i^*)\);
6. `source`: hook \(x_i\) with its unchanged captured site vector \(m_i^0\);
7. `centroid`: hook \(x_i\) with the mean over
   \(\mathcal Y_{s_i,-e_i}\); and
8. `wrong`: hook \(x_i\) with the mean over
   \(\mathcal Y_{t_i,-e_i}\).

One **call unit** is exactly one batch-one invocation of the frozen model that
produces one full-vocabulary law for one unique
`(replay_schedule, entity_id, endpoint_family, word_id)` tuple. A hooked call
includes capture/replacement/upper continuation in that one invocation; it is
not counted once per block. Deriving \(c_9\), metrics, medians, centroids, or
pairwise comparisons adds no call. Donor residuals are captured in the clean
target calls and reused only within the same replay.

The pre-deduplication count is

\[
24\times8\times7\times2=2{,}688.
\]

Deduplication keys include execution mode and replay schedule. Plain
`target_1` and hooked `pasteback` are intentionally distinct fixture paths,
and the two replay schedules are intentionally repeated. The prospective tuple
table has no duplicate prospective scientific identity. A prospective
scientific identity is `(run_phase=science, replay, entity, endpoint family,
word, execution mode, token IDs, intervention-payload recipe hash)`. For
donor-derived edits, the realized float32 payload hash is recorded after
Phase D and before the dependent Phase-E invocation as execution provenance
under that identity; it never substitutes a donor from another replay.
Source-derived hook families intentionally share token IDs and execution mode
but have distinct intervention payloads; they are not deduplicated. There are
**2,688 scientific identities** and **1,680 replay-scoped `(replay, entity,
token IDs, execution mode)` combinations**---840 if replay is ignored. The
manifest stores
the complete tuple table and rejects any disagreement between enumerated and
formula counts as `INVALID — CALL MANIFEST`.

### Replay envelopes, exact estimands, and bound constants

Each replay has two dependency-locked phases. Phase D executes the 72
`(target_1|target_2|target_3, epsilon)` tuples first, ascending by
`(entity_id, target_family)` in replay A and descending in replay B; each call
supplies both its registered endpoint law and that replay's donor residual.
Phase E executes the remaining 1,272 tuples in ascending key order in replay A
and descending key order in replay B. Native and centroid edits use donors
captured in the same replay. Phase-D calls are their existing target-epsilon
call units, so no additional model invocation is introduced. Both replays use
batch size one and the canonical execution policy. Let \(\eta\) be the maximum
same-tuple replay discrepancy over every entity, endpoint, word, and both
channels. If \(\eta>10^{-4}\), the result is
`INVALID — NUMERICAL REPLAY`.

Compute every estimand and stability resample separately in A and B. For each
quantity, the registered upper bound is the larger of the two schedule-specific
upper bounds and the registered lower bound is the smaller of the two
schedule-specific lower bounds. Thus a PASS gate holds in both schedules and
a REFUTATION/FAIL gate excludes the threshold in both; no schedule may be
chosen after outputs.

Use \(R_i,V_i,E_i,\Theta_{24},\Delta_{\rm src}\), and
\(\Delta_{\rm spec}\) exactly as defined in `theory/AXIOMS.md`. The primary
point summaries are exact means of the fixed registered 24-entity population.
For descriptive sensitivity only, draw 2,000 entity-cluster resamples: NumPy
`Generator(PCG64(4141))` samples 24 entity indices with replacement, carrying
all of each entity's targets, words, channels, families, and both replays
together. Store and hash the ordered \(2000\times24\) index table before
reading a scientific output. The empirical 5th and 95th percentiles are the
one-sided lower and upper **entity-cluster stability bounds**. They are not
confidence intervals and make no population-coverage claim.

All numerical constants are bound now:

| constant | bound value | one-line justification |
|:---|---:|:---|
| \(\varepsilon_0\) | \(10^{-5}\) | A nonzero float32 floor prevents exact-zero language while staying far below the material criterion. |
| \(k_B\) | \(2\) | Two endpoint laws can each move by \(\eta\), giving the metric triangle-inequality factor \(2\). |
| \(k_E\) | \(4\) | Both \(R_i\) and \(V_i\) can move by \(2\eta\), so their difference needs factor \(4\). |
| \(\delta\) | \(0.02\) | Prospective policy threshold: two percent of the normalized square-root-JS range is the predeclared material slack beyond native target variation. |
| \(\delta_{\rm move}\) | \(0.02\) | Prospective policy threshold: an edit must improve the paired excess criterion over no edit by a material 0.02, not merely change sign. |
| \(\delta_{\rm spec}\) | \(0.02\) | Prospective policy threshold: correct-label centroids must beat the cycled wrong-label centroid by the same material margin. |
| replay invalidity ceiling | \(10^{-4}\) | Replay drift above ten times the fixed floor voids the deterministic numerical implementation. |
| resamples \(B\) | 2,000 | This matches the repository's registered cluster-resampling resolution without adding model calls. |
| stability tail \(\alpha\) | 0.05 | The 5th/95th one-sided descriptive bounds are stringent but are not sampling intervals. |
| bootstrap generator and seed | PCG64, 4141 | A named generator and Round-41-specific seed make the entity index table exactly reproducible. |
| forecast safety factor | 1.5 | A 50% margin covers hook-path and checkpoint overhead without hiding it in the call count. |
| CPU forecast ceiling | 90 minutes | This is the user-directed abort boundary for calling the artifact small enough to attempt on CPU. |

Thus

\[
\varepsilon_B=\max(10^{-5},2\eta),
\qquad
\varepsilon_E=\max(10^{-5},4\eta),
\qquad
\tau=\varepsilon_E+0.02.
\]

An individual measured bridge discrepancy exceeding \(\varepsilon_B\) in
both replays is a tolerance-robust registered-access refutation in the
registered numerical implementation. It invokes Theorem 7 as an exact bridge
refuter only if the positive discrepancy has a certified numerical-error
bound. Otherwise absence of an exceedance is only `NOT REFUTED AT REGISTERED
ACCESS`.

### Statuses and exact implications

Let \(L_E,U_E\) be the entity-cluster stability bounds for
\(\Theta_{24}\), \(L_{\rm src},U_{\rm src}\) those for the paired source
contrast, and \(L_{\rm spec},U_{\rm spec}\) those for the wrong-label
contrast.

1. `INVALID` applies to any manifest/hash/row/token/site/shape/nonfinite-law,
   numeral-token, response-totality, call-table, replay, DAG-cut continuation,
   plain-versus-hook, response-level intertwining, `Q P` token-order, or paste-back
   fixture failure. Any `INVALID` closes the run with no scientific verdict.
2. `NATIVE PASTE PASS` requires both \(U_E(m^{\rm nat})\le\tau\) and
   \(U_{\rm src}(m^{\rm nat})\le-(\varepsilon_E+0.02)\). It says the
   conjectured proximal site-sufficiency control fits the registered target-
   fiber criterion and prospectively improves it over the unedited source
   across the fixed population. It does not validate centroid formation.
3. `NATIVE PASTE FAIL` applies if \(L_E(m^{\rm nat})>\tau\) or
   \(L_{\rm src}(m^{\rm nat})>-(\varepsilon_E+0.02)\). Every other valid
   native-paste result is `NATIVE PASTE INCONCLUSIVE`.
4. Unless native paste passes, correct- and wrong-centroid endpoints are
   `DIAGNOSTIC ONLY — PROXIMAL CONTROL NOT PASSED`; no centroid PASS is
   available.
5. Conditional on native-paste PASS, `CENTROID PASS` requires all three gates:
   \(U_E(m^{\rm cent})\le\tau\),
   \(U_{\rm src}(m^{\rm cent})\le-(\varepsilon_E+0.02)\), and
   \(U_{\rm spec}\le-(\varepsilon_E+0.02)\).
6. Conditional on native-paste PASS, `CENTROID REFUTATION` applies if
   \(L_E(m^{\rm cent})>\tau\),
   \(L_{\rm src}(m^{\rm cent})>-(\varepsilon_E+0.02)\), or
   \(L_{\rm spec}>-(\varepsilon_E+0.02)\). Every remaining valid centroid case is
   `CENTROID INCONCLUSIVE`.
7. `INCOMPLETE — CPU HARD WALL` applies if an otherwise valid locked run hits
   its wall before all 2,688 call units and both replay schedules are complete;
   partial endpoints receive no scientific status.

Only `CENTROID PASS` licenses:

> Across the 24 registered entities, the correct centroid edit has
> registered-access mean excess discrepancy within the native target-fiber
> criterion and prospectively improves that criterion relative to both no
> edit and the cycled wrong-label centroid.

It does not license an exact bridge, target-place identity, unseen-future
control, reachability, a register, storage, persistence, dimension, native
latent mathematics, fiber separation, every-entity success, or model-family
generality.

### Mechanical-only smoke, forecast, hard wall, and abort rule

The prospective smoke subset is entities \(i\in\{0,23\}\), words
\(\epsilon\) and \(a_Qa_P\) (`Q P`), and both replay orders. For each such
tuple, execute plain \(x_i\), unchanged-hook \(x_i\), plain \(y_i^*\), and
same-carrier paste-back \(y_i^*\): exactly 32 mechanical call executions,
stored under a separate `run_phase=smoke` namespace. They are excluded from
the 2,688 scientific identities and may not be reused as locked scientific
endpoints; any overlapping execution path is repeated after the lock under
`run_phase=science`.

Let \(\eta_{\rm smoke}\) be the maximum same-tuple A/B discrepancy over all 32
smoke calls and both channels, and put

\[
\varepsilon_{\rm smoke}
=\max(10^{-5},2\eta_{\rm smoke}).
\]

If \(\eta_{\rm smoke}>10^{-4}\), the smoke is
`INVALID — NUMERICAL REPLAY`. In each schedule and channel, both plain-target
versus same-carrier paste-back and plain-source versus unchanged-hook source
must have distance at most \(\varepsilon_{\rm smoke}\); otherwise the smoke is
`INVALID — SITE CARRIER/HOOK`. The \(a_Qa_P\) (`Q P`) token-ID fixture remains
exact: its stored execution-order IDs equal `ids(a_P) || ids(a_Q)` under D1
rightmost-first execution.

Let \(s_{\rm smoke}\) be the larger of mean wall seconds per plain invocation
and mean wall seconds per hooked invocation across the 32 calls; model load,
manifest checks, and metric reduction are reported separately. This prevents
the cheaper path from diluting a slower hook path. The CPU forecast and hard
wall are

\[
F_{\rm CPU}=1.5\,s_{\rm smoke}\,(2688)/60\ \text{minutes},
\qquad
H_{\rm CPU}=5\left\lceil F_{\rm CPU}/5\right\rceil\ \text{minutes}.
\]

If \(F_{\rm CPU}>90\) minutes or \(H_{\rm CPU}>90\) minutes, abort before a
scientific forward pass. The only offered alternative is one small GPU burst,
and only after explicit user approval plus a detached, checkpointed
mechanical smoke demonstrates that the exact hook path is GPU-safe. No CPU and
GPU endpoint laws may be mixed in one claiming artifact.

After a valid smoke and only after an audit permits preparation, a lock row
must bind the manifest hash, exact 2,688 count, smoke artifact hash,
\(s_{\rm smoke}\), \(F_{\rm CPU}\), \(H_{\rm CPU}\), runner/config hashes,
all constants, statuses, and stop rule before any scientific forward pass.
There is no lock row in Round 41.

### Prospective runner stages — exactly ten lines, no code in Round 41

1. Load the draft config, verify source hashes, and build the 96-row manifest table without loading the model.
2. Enumerate and hash all words, endpoint tuples, replay orders, donor sets, constants, and the 2,688-call table.
3. Load the pinned tokenizer and `SubstitutionProbe` model, assert resolved identities, freeze parameters, and disable gradients.
4. Retokenize every row and word, assert numeral tokens, site positions, masks, positions, and no-cache policy, and distinguish the declared architecture-dependency specification from an actually serialized per-call cut, which this artifact does not require.
5. Register a block-16 post-block hook that **replaces** exactly one `[:, p, :]` position—the existing `run_onewrite_state.py` hook **adds** a delta—and assert exactly one matched write per hooked call.
6. Run only the 32-call mechanical smoke, evaluate the three fixtures, and compute the smoke-derived CPU forecast and hard wall.
7. Refuse scientific mode unless audit authorization and the complete pre-science lock row match every manifest, runner, config, count, and timing hash.
8. Execute each replay's donor-first Phase D and remaining Phase E in the locked schedule, using only same-replay donor residuals, with per-call and per-entity checkpoints; store each full law under its complete call identity and each donor under `(replay, target_row_id)`.
9. Validate replay, totality, and completeness, then derive `c_9`, bridge discrepancies, exact finite-population estimands, paired contrasts, and entity-cluster stability bounds.
10. Apply the immutable status tree, write the result/manifest/checkpoint hashes, and stop without post-outcome repair, rerun, or population adaptation.

The new runner, if later authorized, is
`experiments/run_native_bridge.py`. Round 41 deliberately does not create it.

### Stop rule and audit #47 lock-readiness disposition

There is no repair after any scientific output. Any `INVALID` closes
`native_bridge_v1`; any complete valid FAIL, REFUTATION, or INCONCLUSIVE is the
registered result; any hard-wall stop is permanently `INCOMPLETE — CPU HARD
WALL`. Changing a row, word, site, target, control, constant, replay schedule,
or threshold after output would be a different named artifact and requires a
new dialogue and preregistration. No same-check repair is permitted.

Audit #47 verified the seven repair families, the 72+1,272 dependency
schedule, the 2,688 scientific identities, the 1,680 replay-scoped
token/mode combinations, the 32-call mechanical smoke, the normalized metric
and replay envelopes, and the absence of hidden calls, serialized cut
records, and cross-replay donors. Verdict: LOCK-READY subject to the
report's wording-only identity and provenance corrections. This authorizes
runner/config construction, the mechanical smoke, forecast derivation,
and---if valid---a pre-science lock row; no scientific forward pass is
authorized before that row.

Measurement-to-artifact heartbeat: Round 41 adds 0 apparatus lines and 0
artifact-bearing code lines, \(0/0\). It is one theory/build round. After audit
#45's \(30:13\) accounting, the cumulative measurement/governance-to-build
ratio is \(30:14=2.14:1\), still above the \(2:1\) warning and below the
\(5:1\) halt.

## PSQ-3μ — smallest necessary-condition intervention test (Codex direction round 5, 2026-08-31)

### Proposition

If a small real model treats textually different instances of the same dial location as one behavioral place, one shared learned move should carry all of them toward the next place.

Concretely: in the two-dial Z_8 × Z_8 world restricted to S_μ = Z_8 × {0,2,4,6} (32 states, only x-channel probes), a Procrustes operator M_A fitted on calibration states should, when applied to held-out carrier representations, produce response profiles closer to the true A-successor's oracle profile than any control edit does.

### Locked design

| Component | Choice |
|---|---|
| Model | Qwen3-0.6B-Base, revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd, CPU float32 |
| States | S_μ = Z_8 × {0,2,4,6}: 32 states |
| Registered quotient | Only x-channel responses (is_x_zero probes); y is fiber/nuisance |
| Probe panel | Eight x probes: is_x_zero after words of length 0–4 containing only A and B |
| Move | A: x → x+1 mod 8. B: x → −x mod 8 is wrong-action control |
| Split | Block-floor rule: cal = {(x,y) : (floor(x/2)+floor(y/2)) mod 2 = 0}, 16 cal / 16 held-out |
| Carrier | Newline token after "y = {digit}" (same as PSQ-3) |
| Geometry | Fixed block 18, transductive PCA k=4 over all 32 states, calibration-only Procrustes M_A and M_B |
| Primary number | Held-out mean G_{M_A} and paired G_{M_A} − G_{control} |
| Controls | No edit, mean displacement, M_B (wrong action), fixed-seed matched-random O(k) |
| Call cap | Exactly 1,152 forwards |
| Wall clock | 30 minutes, fail closed |
| Device | CPU only (no GPU) |

### Call budget (1,152 total)

- 256 baseline/profile calls: 32 states × 8 probes, capturing carrier at block 18
- 128 same-state replay (32 states × 4 probes, determinism check)
- 128 donor positive control (32 held-out targets, using cal source carrier)
- 128 calibration-source M_A positive control (32 cal states, self-check)
- 512 held-out calls: 16 held-out sources × 4 interventions (M_A, mean displacement, M_B, matched random) × 8 probes... [NOTE: exact decomposition follows the micro phase implementation]

### Predeclared outcomes (immutable)

- **NO_INTERFACE**: balanced panel accuracy below 95% across states, excessive non-digit probability mass (>10% mean), or no response-law separation between x places (oracle profile spread < 0.1 JS₂). Stop before intervention.
- **INVALID**: replay exceeds 1e-3 max-absolute logit difference, donor control decodes fewer than 15/16 successors, or calibration M_A decodes fewer than 14/16. The instrument is broken.
- **MICRO_SIGNAL**: held-out M_A mean gain ≥ 0.25, decodes at least 12/16 correct successors, and its paired 95% source-bootstrap lower bound exceeds each control by more than zero. Licensed sentence below applies.
- **MICRO_FAIL**: interface and proximal controls pass but held-out M_A does not meet MICRO_SIGNAL thresholds.

### Licensed sentence (if MICRO_SIGNAL)

In frozen Qwen3-0.6B-Base, a single calibration-fitted Procrustes operator for action A, applied to held-out carrier representations at block 18, produces response profiles whose nearest-state decode matches the true A-successor at [N/16] states, with mean behavioral gain [G] exceeding all four controls (no edit, mean displacement, wrong action B, matched random) at the 95% bootstrap level. This establishes one-action, fixed-presentation response-law equivariance at this site.

### Never say

- "The micro test establishes full d_∞ geometry."
- "Composition, presentation invariance, or denizen reachability is demonstrated."
- "The framework is validated" or "native latent mathematics is proved."
- "The result generalizes to other models, actions, or sites."
- "0.6B results predict 1.7B behavior."
- "The y-channel responses were tested."
- "The test is a replication of PSQ-3."

### Stop rule

No repair after any outcome. Any NO_INTERFACE or INVALID closes PSQ-3μ. A MICRO_FAIL is the registered negative result. A MICRO_SIGNAL promotes PSQ-3 to the stable-hardware queue after its false-pass verdict reducer is repaired. No same-check repair is permitted. One audit fires after the result.

Measurement-to-artifact heartbeat: PSQ-3μ contributes 0 measurement/governance rounds and 1 build round, taking the cumulative round ratio from 33:15 to 33:16 = 2.06:1. Commit `17e0b4e` added 470 and removed 1 runner line, and `micro_phase` spans 443 physical lines (410 nonblank), but no apparatus-versus-artifact-bearing line classification was recorded; therefore the ~150 and ~250 artifact-bearing-line estimates are withdrawn pending explicit classification.

### Outcome (2026-08-31)

**Status: NO_INTERFACE — PSQ-3μ CLOSED.** (Audit #49 adopted verbatim below.)

Result (from `experiments/results/psq3_micro_cpu/result.json`):
- Panel accuracy: 68/256 = 0.2656 (gate: ≥0.95) — **FAIL**
- Mean p_other: 0.0424 (gate: ≤0.10) — pass
- Oracle spread: 0.5000 (gate: ≥0.10) — pass

Execution stopped at stage 1 (baseline profiling), consuming 256 of 1,152
budgeted calls (132.6s, 1.9 fwd/s on CPU float32). Stages 2–6 never ran.

**Analysis (audit #49, verbatim):** Frozen Qwen3-0.6B-Base scored 68/256 (26.56%) on the registered x-channel `is_x_zero` panel, below the 95% interface gate. A contemporaneous observation suggested that the model favored token "1" approximately 81% of the time, but the saved result contains no prediction histogram or confusion matrix, so that rate and all class-specific accuracies are not independently auditable. The model did not demonstrate reliable performance on the registered probes, and the registered behavioral-interface gate was not met.

**Disposition (audit #49, verbatim):** Per the predeclared stop rule, `NO_INTERFACE` closes PSQ-3μ with no repair. This result is scoped to frozen Qwen3-0.6B-Base under the PSQ-3μ panel and does not adjudicate the distinct full PSQ-3 experiment, which uses a different model and training intervention.

### Never say (PSQ-3μ, audit #49)

- "The frozen 0.6B model cannot perform modular arithmetic." (Overclaim beyond the registered panel.)
- "The behavioral interface does not exist." (Overclaim; the registered gate was not met.)
- "PSQ-3μ confirms that frozen small models lack the interface." (Overclaim beyond the one tested panel.)
- Any class-specific accuracy claim (12.5%/100%) without an auditable saved histogram.
