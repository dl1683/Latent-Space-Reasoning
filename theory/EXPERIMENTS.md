# Native latent mathematics — experiment preregistrations

## NLM-001 — directed substitutability, context rank, and transfer

**Status:** design locked 2026-08-27; confirmatory measurement unrun. The
analyzer must be aligned with the H2 bootstrap and baseline-selection rules
below before the run.

The existing 12-word smoke run is calibration, not evidence. It fixed the fact
that exact repeats are identical while batched-versus-single evaluation is not,
and showed that directed KL is numerically measurable. All predictions below
were written after that smoke but before the 80-item, held-out, or cross-system
measurement.

### Hypothesis

Lexical substitutability is directed and requires multiple context orderings,
yet contains a stable component that transfers to held-out probes better than
the strongest symmetric contextual or learned metric. Some directional and
ordering structure also transports across independently trained systems.

### Instrument

- One CPU entrypoint: `experiments/run_lexical_closeness.py`; it may import the
  existing measurement helper `experiments/substitution_probe.py`.
- Frozen items/probes: `experiments/config/lexical_probe_v1.json` (80 words;
  four lexical classes; four blocks × four paraphrases).
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

Define

\[
r_{B,j}(x\to y)=D_{KL}(K_{B,j}(x)\Vert K_{B,j}(y)),
\qquad
D_B(x\to y)=\operatorname{median}_j r_{B,j}(x\to y).
\]

For transfer, set

\[
D_C=\operatorname{median}(D_{\rm gloss},D_{\rm continuation}),
\qquad
D_H=\operatorname{median}(D_{\rm association},D_{\rm grammar}).
\]

For any four paraphrase statistics \(s_j\), define

\[
\nu=1.4826\operatorname{median}_j|s_j-\operatorname{median}(s)|.
\]

Let \(\nu_0\) be the block median of \(\nu\) over eligible pairs. On the first
probe and first eight states, define the numerical null in KL units by

\[
\eta=\max_{x,y}\left|r_{\rm batch}(x\to y)-r_{\rm single}(x\to y)\right|.
\]

The exact-repeat max \(|\Delta\log p|=0\) and batched-versus-single max
\(|\Delta\log p|=2.3\times10^{-5}\) are reported diagnostics; neither replaces
the KL-scale \(\eta\). A four-paraphrase gap is **robust** iff its sign agrees in
at least three paraphrases and its absolute median exceeds
\(\max(3\nu,3\nu_0,10\eta)\). For a fixed two-paraphrase half, robustness means
both signs agree and the same magnitude threshold is passed.

### Measurements and exact predictions

Compute H1–H3 separately for every system. The primary thresholds apply to the
primary system; cross-system claims use only word-string-aligned states and
never align coordinates.

#### H1. Directed asymmetry

\[
a_{B,j}(x,y)=r_{B,j}(x\to y)-r_{B,j}(y\to x).
\]

- Point prediction: 0.20 of unordered pairs are robustly asymmetric in at least
  two blocks in the primary system.
- Support gate: lexical-state-bootstrap 95% lower bound above 0.10; pairs are
  not treated as independent.
- Cross-realization point prediction: asymmetry-sign agreement is 0.65 among
  pairs robust in both systems; support requires a lexical-state-bootstrap lower
  bound above 0.50 against word-label permutation.

#### H2. Context rank

Fix the preregistered split of each block's four paraphrases into halves
\(B^1=(1,2)\) and \(B^2=(3,4)\). For two halves \(U,V\) and anchor \(x\), let
\(q_x(U,V)\) be the fraction of candidate pairs robust in both halves whose
ordering signs oppose; it is undefined when no pair is robust in both. Define

\[
W=\operatorname{median}_{x,B}q_x(B^1,B^2),\qquad
B=\operatorname{median}_{x,A<B,h\in\{1,2\}}q_x(A^h,B^h),\qquad Q=B/W,
\]

omitting undefined cells. If \(W=0<B\), set \(Q=+\infty\); if \(W=B=0\), H2 is
undefined and unsupported. Bootstrap anchors and recompute \(W,B,Q\) in every
replicate.

- Point prediction: \(Q=3.0\) in the primary system.
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
10% of candidate-pair orderings reverse robustly.

- Point prediction 1: reversal-active anchor fraction \(R=0.30\).
- Point prediction 2: on those anchors, native calibration KL exceeds the
  strongest symmetric baseline's held-out pairwise accuracy by
  \(\Delta_{\rm rev}=+0.07\).
- Support gates: \(R\ge0.20\), \(\Delta_{\rm rev}\ge0.05\), and anchor-bootstrap
  95% lower bounds above 0.10 and 0 respectively.

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
once on the full calibration data. Retain per-anchor calibration-label accuracy
for every baseline. In each anchor-bootstrap replicate, recompute each
baseline's mean calibration accuracy on the sampled anchors, select the
strongest (fixed baseline order breaks ties), and compare native KL with that
baseline on the sampled held-out reversal-active anchors. Thus the bootstrap
reselects the competitor without refitting it; its interval is conditional on
the frozen fits.

Held-out pairwise labels exist only when the ordering is robust separately in
both held-out blocks and the two signs agree. No pooled or one-block label may
replace this intersection.

### Kill conditions

1. **Directedness killed:** the 95% upper bound on robust asymmetric-pair
   fraction is below 0.05.
2. **First non-collapse bet fails:** \(B\le W\) in the primary system, regardless
   of \(\widehat\kappa_{0.10}\). It is killed strongly if the 95% upper bound on
   \(Q\) is at most 1. Graph saturation cannot rescue it.
3. **Predictive novelty killed:** the strongest contextual or learned metric
   matches or exceeds native calibration KL on held-out accuracy
   (\(\Delta_{\rm rev}\le0\)).
4. **Transfer killed:** the 95% upper bound on \(\Delta_{\rm rev}\) is at most
   zero, even if asymmetry or \(\kappa>1\) survives.
5. **Cross-realization semantic-stability claim killed:** robust sign agreement
   does not beat 0.50 under the preregistered permutation test, or rank
   correlation does not beat zero. Recompute the eligible robust-pair
   intersection after each word-label permutation. Within-system structure may
   still be real.
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
