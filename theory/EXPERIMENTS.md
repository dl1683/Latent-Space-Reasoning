# Native latent mathematics — experiment preregistrations

## NLM-001 — directed substitutability, context rank, and transfer

**Status:** preregistered 2026-08-27; confirmatory measurement unrun.

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
no sampling or generation, and bounds memory to one probe at a time.

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

Let \(\nu_0\) be the block median of \(\nu\) over eligible pairs and \(\eta\)
the 95th-percentile change in the same statistic under exact-repeat and
batched-versus-single evaluation. A gap is **robust** iff its sign agrees in at
least three paraphrases and its absolute median exceeds
\(\max(3\nu,3\nu_0,10\eta)\).

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

For each block, order candidates by median \(D_B\); frozen config order resolves
serialization ties but cannot create evidence. Join two blocks when at least 10%
of anchors contain an opposite candidate ordering that passes the robustness
rule in both blocks. Let \(G_{0.10}\) be this four-vertex graph and define the
robust estimate \(\widehat\kappa_{0.10}=\chi(G_{0.10})\). This is a noise- and
prevalence-thresholded lower bound on exact graded context rank, not the exact
invariant when incompatibilities are sparse.

- Point prediction: \(\widehat\kappa_{0.10}=4\) in the primary system: every
  pair of semantic blocks has a robust incompatibility edge.
- Cross-realization prediction: \(\widehat\kappa_{0.10}\ge3\) in at least two
  of three systems.
- Report the graph and witnesses; the scalar alone is insufficient.

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
components are fitted on calibration hidden states only. In each bootstrap
replicate, compare native KL with the best reselected baseline; do not freeze a
weak competitor after seeing results.

### Kill conditions

1. **Directedness killed:** the 95% upper bound on robust asymmetric-pair
   fraction is below 0.05.
2. **First non-collapse bet killed:** the robust incompatibility graph is empty
   (\(\widehat\kappa_{0.10}=1\)) in all three systems.
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
