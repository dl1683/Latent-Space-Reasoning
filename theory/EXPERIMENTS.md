# Native latent mathematics — experiment preregistrations

## NLM-001 — contextual substitutability, context rank, and transfer

**Status:** amended and locked after instrument calibration, 2026-08-27. Raw
80-word matrices were acquired concurrently before this amendment; no NLM-001
analysis of them has been read here. This is therefore an analysis
preregistration over sequestered data, not a pre-acquisition registration. The
analyzer must be aligned with every Round 2/2b rule below before analysis.

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
