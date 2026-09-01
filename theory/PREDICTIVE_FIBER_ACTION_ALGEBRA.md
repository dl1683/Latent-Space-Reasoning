# Predictive-Fiber Action Algebra

The canonical algebraic artifact for the behavioral algebra discovered in
Qwen3-0.6B across 18 audited experiments (Phase 2, 2026-08-31).

## 1. Core object

$$
\mathfrak{A} = (X, W, Q, G, \tau, \gamma, \pi, \mathcal{A}, S^W, S^G)
$$

| Symbol | Name | Definition |
|--------|------|------------|
| $X$ | Prompt histories | Raw textual prompt sequences |
| $W$ | Semantic worlds | Experimenter-known ground-truth fact assignments (e.g., {ZOG=big, MIP=hot, PLIM=red}) |
| $Q = X / {\sim_\infty}$ | Future-response quotient | $x \sim_\infty y$ iff $r_c(wx) = r_c(wy)$ for every legal continuation $w$ and response channel $c$ |
| $G = X / {\sim_\text{greedy}}$ | Greedy quotient | Equality of greedy answer vector across all registered queries |
| $\tau: X \to W$ | World map | Assigns each prompt history its generating semantic world |
| $\gamma: X \to G$ | Greedy map | Assigns each prompt history its greedy answer signature |
| $\pi: Q \to G$ | Projection | Maps fine future-response state to coarse greedy commitment |
| $\mathcal{A}$ | Continuation monoid | Typed continuation operations under concatenation |
| $F_g = \pi^{-1}(g)$ | Fibers | Pre-images of greedy places; encode history-dependent structure invisible to argmax |
| $S^W_w$ | World-conditioned restatement | Canonical restatement constructed from the experimenter-known world $w \in W$. Approximately idempotent; non-natural with correction under tested append action. **Not** representative-independent: requires the hidden world, not the observable greedy signature $g$ |
| $S^G_g$ | Signature-indexed restatement | Canonical restatement constructed from the observable greedy signature $g \in G$ alone. Approximately idempotent; perfect observed descent on the empirical carrier (fixes $S^W$'s one failure). Representative-independent: uses only observables. Defined via renderer $R(g)$: $S^G_g(x) = x \cdot R(g)$ |

## 2. Greedy places (carrier table)

A greedy place $g \in G$ is defined by the full greedy answer signature across
all registered queries. Example for 3 entities (ZOG, MIP, PLIM):

| Place | ZOG answer | MIP answer | PLIM answer | # fiber members | Example histories |
|-------|-----------|-----------|-------------|-----------------|-------------------|
| $g_0$ | big | hot | red | 2 | w000_std, w000_rev |
| $g_1$ | big | hot | blue | 2 | w001_std, w001_rev |
| ... | ... | ... | ... | ... | ... |

**Registered set:** 12 distinct greedy places from 16 histories (8 worlds x 2 orders).
**Held-out set:** 11 distinct greedy places from 16 histories.

Non-singleton fibers exist: multiple histories map to the same greedy signature
but differ distributionally (0% distributional congruence, JSD 0.07-0.45).

## 3. Continuation generators (Cayley-style action table)

| Generator | Type | Notation | Place preservation | Status |
|-----------|------|----------|-------------------|--------|
| Empty | Identity | $\varepsilon$ | 100% | **Established** |
| Neutral | Near-identity | $N$ | 95.8% | **Approximate** |
| Correction | State-changing partial | $C_{e \leftarrow v}$ | 35-42% | **Established** |
| Restatement | Idempotent retraction | $S^W_w$ | 89.6-93.8% | **Established** |

**Place preservation** = fraction of greedy signatures unchanged after the operation.

### Descent properties

An operation *descends* to $G$ when all representatives of a fiber map to the
same target place under the operation.

| Operation | Empty descent | Restatement descent |
|-----------|--------------|---------------------|
| Registered | 12/12 (100%) | 11/12 (91.7%) |
| Held-out | 15/15 (100%) | 15/15 (100%) |

**Empty descent is perfect.** Restatement descent is near-perfect (one failure:
registered fiber `MIP=blue|PLIM=blue|ZOG=small` has members from different
worlds w101 and w111 with different MIP ground truths — so $S^W_w$
maps them to different MIP answers. This confirms the $S^W$ vs $S^G$ typing
issue: restatement uses the hidden world, not the observable greedy signature).

## 4. Law sheet

### Established laws

**L1: Identity.** $\varepsilon \circ a = a \circ \varepsilon = a$ for all $a \in \mathcal{A}$.
*Status:* **Established.** Empty continuation preserves all greedy places (100%).

**L2: Idempotence of $S^W$.** $(S^W_w)^2 \approx S^W_w$.
*Status:* **Established.** 100% greedy idempotence (96/96 across both entity sets).
JSD distance (√JSD) mean = 0.070 (range 0.025-0.140).

**L2a: Idempotence of $S^G$.** $(S^G_g)^2 \approx S^G_g$.
*Status:* **Established.** 100% greedy idempotence (96/96 across both entity sets).
JSD($S^G$, $(S^G)^2$) mean = 0.077 registered, 0.071 held-out (range 0.025-0.191).
Comparable to $S^W$. Place preservation: 32/32 (100%).
Fixed cyclic shuffle 0/32 (pairing-sensitive; textual echo unresolved). On the 32 tested histories, one fixed cyclic reassignment of the same value words among entity labels preserved no full greedy signature (0/32 vs 32/32 for S^G). This shows that the effect is sensitive to entity-value pairing rather than the renderer template or value-word multiset alone; it does not distinguish semantic use from last-mention copying and does not rule out textual echo.

Anti-echo alias control (Phase 4c, Codex evidence gate REVISE): no anti-echo evidence. The faithful alias arm did not exceed either comparator by the predeclared 30-point margin. A subsequent direct R(g) recovered 47.9%, ruling out a deterministic latest-explicit-assignment rule. Implementation defects: shuffled alias uses out-of-type values (disjoint entity domains); alias grammar differs from direct format; aliases not counterbalanced. This particular alias renderer did not override the direct decoy; this does not prove the model cannot resolve aliases. The literal signature renderer preserved 32/32 full greedy signatures — do not call this semantic or latent-space invariance. The v2 non-commuting square establishes sequence/path dependence but does not rule out ordinary textual order or multiplicity effects.

#### Phase 4d terminal anti-echo factorial (pre-registered; RUN — NO_INTERFACE_OR_INVALID__TERMINAL_DEMOTION)

**So what:** this is the last prompt-only test of whether signature restatement
does more than let the most recent literal record dominate the answer; any
non-pass ends renderer tuning and moves the program to intervention-defined
continuation laws.

Let each entity set have an explicitly declared shared value domain $V$:

\[
V_{\rm reg}=\{\text{big, small, hot, cold, red, blue}\},\qquad
V_{\rm held}=\{\text{fast, slow, tall, short, loud, quiet}\}.
\]

The base prompt says that every named entity may take any one value in the
relevant $V$.  This makes cross-pair values type-valid rather than silently
assuming the old disjoint entity-specific types.  Fix a public, pre-run,
fixed-point-free involution $\nu:V\to V$ pairing the two words from each old
binary domain (big/small, hot/cold, and so on).  For an observed in-domain
signature $g$, define the counterfactual signature
$\bar g=\nu\circ g$.  Thus $\bar g_e\ne g_e$ at every coordinate, even
when $g$ contains duplicate or formerly cross-type values.  No row is
silently repaired or filtered: an observed value outside $V$ is saved and
counts against the domain-interface gate.

Write $R(h)$ for a direct `Record` block rendering signature $h$.  Let
$K_\pi$ be an alias key and $A_\pi(h)$ the same `Record` grammar with entity
names replaced by aliases.  The aliases are the arbitrary identifiers `Q7`,
`V4`, and `J2`.  The three cyclic Latin-square maps $\pi$ put every entity
under every alias and in every assignment position exactly once.  Alias
records are emitted in fixed alias order, not entity order, so alias identity,
entity identity, and record position are counterbalanced.

For a base history $B_x$ and entity query $q_e$, the **primary direct
factorial** is

\[
\begin{array}{ll}
D0:&B_xq_e,\\
D1:&B_xR(g)q_e,\\
D2:&B_xR(\bar g)q_e,\\
D3:&B_xR(\bar g)R(g)q_e,\\
D4:&B_xR(g)R(\bar g)q_e.
\end{array}
\]

The runner must prove before inference that every D3/D4 query pair has the
same complete token multiset.  Their only manipulated factor is block order.
Let

\[
L_e(p)=\operatorname{logit}_p(g_e)-
       \operatorname{logit}_p(\bar g_e).
\]

The primary continuous estimand is

\[
\Delta_{\rm order}=\mathbb E_{\rm world}
  [L_e(D3)-L_e(D4)],
\]

with world-cluster weighting; the paired greedy endpoint is the mean rate of
following the final block, $\tfrac12(1[D3=g_e]+1[D4=\bar g_e])$.

For each of the three alias maps, the **secondary 2 x 2 alias factorial** is

\[
\begin{array}{ll}
A0:&B_xK_\pi A_\pi(g)q_e,\\
A1:&B_xK_\pi A_\pi(\bar g)q_e,\\
A2:&B_xR(\bar g)K_\pi A_\pi(g)q_e,\\
A3:&B_xR(\bar g)K_\pi A_\pi(\bar g)q_e,\\
A4:&B_xR(g)K_\pi A_\pi(g)q_e,\\
A5:&B_xR(g)K_\pi A_\pi(\bar g)q_e.
\end{array}
\]

A1 is the alias-necessity arm: unlike Phase 4c's alias-only arm, it can pass
only by moving the answer away from the base signature toward $\bar g$.
A2 versus A3 is the requested faithful-versus-type-valid-counterfactual
comparison after a direct decoy; A4 versus A5 supplies the symmetric direct
context.  The alias-content estimand averages the paired A0/A1, A2/A3, and
A4/A5 changes rather than treating nested entity or alias-map rows as
independent evidence.

All intervals are deterministic 10,000-resample percentile intervals over
semantic-world clusters.  Entity queries, the two base presentation orders,
and alias maps remain nested inside a world.  Registered and held-out entity
sets are reported separately; a gate that says "each set" requires both.
Exact counts are diagnostics, not all-or-none verdicts.

**Locked gates.** They are evaluated in order.

1. **Integrity/interface.** All D3/D4 token-multiset checks, fixed-point-free
   counterfactual checks, alias-balance checks, and one-token value-verbalizer
   checks must pass.  At least 95% of base coordinates in each set must emit a
   value in the declared shared domain.  D1 must follow $g$ on at least 90%
   of coordinates in each set.  D2 must follow $\bar g$ on at least 60%,
   with world-cluster 95% lower bound at least 40%.  Failure is
   `NO_INTERFACE_OR_INVALID`; later mechanism gates are not interpreted.
2. **Direct recency.** `RECENCY_EXPLAINS` requires, in each set: final-block
   following at least 70% with cluster lower bound at least 50%; D3-minus-D4
   target-following at least 30 percentage points with lower bound above zero;
   and $\Delta_{\rm order}\ge1.0$ logit with lower bound above zero.  If this
   passes, the literal $S^G$ renderer is demoted regardless of alias results.
3. **Alias necessity.** A1 must follow $\bar g$ at least 60% in each set
   (cluster lower bound at least 40%).  Relative to A0, it must reduce target
   following by at least 30 points and shift the target-minus-counterfactual
   logit contrast by at least 1.0 logit, with both paired cluster bounds above
   zero in the counterfactual direction.  The logit shift must be point-positive
   for all three alias maps in both sets.  Failure kills alias interpretation;
   A2--A5 become descriptive only.
4. **Alias anti-echo.** Conditional on Gate 3, average target-following for
   alias-target minus alias-counterfactual across the two direct contexts
   (A2/A3 and A4/A5) must be at least 30 points in each set with lower bound
   above zero.  The corresponding logit-contrast shift must be at least 1.0
   with lower bound above zero.  The two discordant cells A2 and A5 must follow
   their final alias payload at least 60% with lower bound at least 40%, and
   every alias map's point shift must be positive in both sets.

**Terminal adjudication.** `RECENCY_EXPLAINS` scientifically demotes $S^G$
to a sequence-sensitive syntactic append operator.  Gate-3 failure says only
that aliases are not an interpretable instrument, but because this is the
terminal renderer control it also ends any semantic upgrade and demotes
$S^G$ by allocation.  A Gate-3 pass followed by Gate-4 failure is a clean
anti-echo non-pass and demotes $S^G$.  Integrity/interface failure is
scientifically inconclusive but allocation-terminal: no prompt or alias repair
follows.  Only Gate-3 and Gate-4 passes together, with Direct Recency not
passing, license the narrow sentence: "A counterbalanced keyed re-encoding of
the observed signature moved answers against a conflicting direct record, so
verbatim entity-value copying and the registered final-block recency rule are
insufficient for these prompts."  Even that outcome does not establish a
latent-space invariant, a semantic retraction, or native mathematics.

**Result (2026-08-31).** 2304 forward passes, Qwen3-0.6B CPU.  Gate 0
(integrity) passed: 100% domain validity, all token multisets verified, alias
counterbalancing exact.  **Gate 1 (interface) FAILED** in both sets: confirming
records followed at 97.9%/91.7% (D1) but contradicting records followed at only
43.8%/16.7% (D2, threshold 60%).  Gate 2 (direct recency) failed: logit order
effect −0.043/−0.086 with CI crossing zero.  Gate 3 (alias necessity) failed:
A1 counterfactual follow 4.2%/6.9%.  Gate 4 (alias anti-echo) failed: greedy
rates too low despite strong logit effects (3.6/2.4 nats).  **Verdict:
NO_INTERFACE_OR_INVALID__TERMINAL_DEMOTION.**  $S^G$ is locked as a literal
append operator; no further renderer tuning.  Raw data:
`experiments/results/signature_restatement_v1/phase4d_results.json`.

**L3: Correction changes place.** $C_{e \leftarrow v}$ maps greedy place $g$ to a
different place $g'$ where entity $e$'s answer is $v$, in 58-65% of cases.
*Status:* **Established** (as a partial action — not defined everywhere).

### Non-naturality (established)

**L4: $S^W$ is non-natural with correction under the tested append action.**
$S^W_{w'} \circ C \neq C \circ S^W_w$ in general.
*Status:* **Established** (v2 experiment, correctly typed square).

| Metric | Registered | Held-out |
|--------|-----------|----------|
| JSD distance mean | 0.208 | 0.208 |
| Greedy commutativity | 89.6% (43/48) | 70.8% (34/48) |
| Task kernel diff mean | 0.155 | 0.155 |

The typed square with both paths ending at the corrected world $w'$ does not
commute. This is not a recency/contradictory-text artifact: in path
$C \to S^W_{w'}$, both the correction and the corrected-world restatement assert
the same values, yet the result still differs from $S^W_w \to C$.

**Consequence:** Prediction remains presentation-path dependent after both paths
have reached the same declarative world. One failed naturality square for one
content-bearing canonicalizer rules out that particular clean separation; it does
**not** prove that no alternative canonicalizer or product decomposition exists.

**L4b: $S^G$ is non-natural with correction (pointwise).**
The adaptive canonicalizer $K(x) = x \cdot R(\gamma(x))$ does not commute with
correction: $K(Cx) \neq C(Kx)$ in general.
*Status:* **Established** (signature_restatement_v1 experiment).

| Metric | Registered | Held-out |
|--------|-----------|----------|
| JSD distance mean ($S^G$) | 0.193 | 0.188 |
| JSD distance mean ($S^W$) | 0.208 | 0.208 |
| Greedy commutativity ($S^G$) | 85.4% (41/48) | 79.2% (38/48) |

This is a **pointwise** comparison, not quotient-level naturality, because
correction itself does not descend to $G$ (see O2 below). The observable
canonicalizer $S^G$ eliminates the objection that $S^W$'s non-naturality
depends on hidden information.

### Established (was Open)

**L4a: $S^G$ descent to $G$ (empirical carrier).**
*Status:* **Established on the empirical carrier.** 12/12 registered coordinate
checks (4/4 non-singleton fibers), 15/15 held-out (5/5 non-singleton fibers).
Perfect observed descent on the tested histories — fixes $S^W$'s one failure
(fiber spanning worlds w101/w111). Does not prove descent over all histories
in the globally defined $X$.

For comparison, $S^W$ descent: 11/12 registered (91.7%), 15/15 held-out (100%).

### Open questions

**O1: ~~Existence of representative-independent $S^G_g$.~~**
*Status:* **RESOLVED — YES.** $S^G_g$ exists, is approximately idempotent (L2a),
has perfect observed descent on the empirical carrier (L4a), and preserves
greedy places (100%). Constructed from the observable greedy signature via
renderer $R(g)$. Established in experiment `signature_restatement_v1`.
"Held-out" means new entity names and value words under the same template,
not held-out prompt structures or tasks.

**O2: Correction descent to $G$.**
Does correction descend to the greedy quotient? Direct measurement: correction
descent is 7/12 (58.3%) registered, 12/15 (80.0%) held-out. Fiber members
given the same correction $C_{e \leftarrow v}$ produce different post-correction
signatures. This means the typed non-naturality square is pointwise, not
quotient-level.
*Status:* **Partially answered — NO for many fibers.** Correction does not
reliably descend.

### Not established

**N1: Global predictive × presentation nonfactorization.**
The non-naturality of $S^W$ with correction rules out one specific factorization
(world-conditioned restatement as a product component). It does **not** establish
that no alternative product decomposition of the fiber into predictive and
presentation components exists.
*Status:* **Not established.**

### Candidate laws (not yet tested)

**L5: Associativity of continuation concatenation.** $(a \circ b) \circ c = a \circ (b \circ c)$.
*Status:* **Candidate.** Inherited from string concatenation; not empirically verified at
the distributional level.

**L6: Neutral near-commutativity.** $N \circ a \approx a \circ N$ for non-correction $a$.
*Status:* **Candidate.** Neutral has 95.8% place preservation but commutativity not tested.

## 5. Fiber structure

Each greedy place $g$ has a fiber $F_g = \pi^{-1}(g)$ containing all prompt
histories that produce the same greedy answer signature.

### Within-fiber distances (from predictive_fiber_v1)

| Pair type | Mean JSD | Interpretation |
|-----------|---------|----------------|
| Benign presentation | 0.254 | Order/repetition differences within same world |
| History pair | 0.292 | Different fact-worlds, same greedy answer |
| Cross-world | 0.157 | Different irrelevant facts |

**History > Benign** at baseline and after corrections (3/3 each): the
distributional residual IS task-relevant predictive state.

**History < Benign** after restatement (0/3): restatement partially collapses
the within-fiber distance. $S^W_w$ is a genuine retraction.

**Cross-world smallest**: presentation (how you told the model) matters more
than what else you told it.

### The commitment bottleneck

At layers 24-25, the logit-lens entropy drops to 0.05-0.30 bits (near-deterministic
commitment). The final output re-broadens to 5.5-7.7 bits. This explains:
- Greedy congruence (97%): the bottleneck fixes the argmax
- Distributional incongruence (0%): the re-broadened distribution encodes history

This is an **implementation annotation** — it shows where the algebraic structure
becomes visible in the computation, not a separate algebraic law.

## 6. Honest claims and bounds

### What this algebra IS

1. A finite partial action algebra of greedy commitments over a small prompt family.
2. An operational, non-$\mathbb{R}^n$ account of behavioral places, actions, and
   hidden fibers.
3. A world-conditioned approximately idempotent restatement ($S^W_w$) with near-perfect
   observed descent.
4. A representative-independent approximately idempotent restatement ($S^G_g$) with
   perfect observed descent on the empirical carrier, constructed from the observable
   greedy signature alone.
5. A demonstrated pointwise non-naturality: both $S^W$ and $S^G$ do not commute with
   correction under the tested append action — prediction is presentation-path dependent.

### What this algebra is NOT

1. A universal transformer law (tested on one model, one prompt family).
2. An exact algebra (3-7% defect rates on operations).
3. A mechanism-level explanation (behavioral evidence only, no causal circuits).
4. A proof of global nonfactorization (one canonicalizer fails; others untested).

### The one-sentence claim

> In a bounded three-fact prompt world in one small language model, greedy answer
> signatures form an approximate behavioral quotient with nontrivial predictive
> fibers, while a world-conditioned canonical restatement is approximately
> idempotent but non-natural with correction: two update paths denoting the same
> corrected world produce different response laws and, on held-out names,
> different greedy answers in 14 of 48 cases.

## 7. Evidence index

| Experiment | What it establishes | Commit |
|-----------|-------------------|--------|
| logit_lens_resolution_v1 | Resolution layer L21-25, 62x selectivity | 7ec05d6 |
| attention_control_v1 | Not attention routing (r < 0.25) | c57efc1 |
| mlp_decomposition_v1 | Value-space operation | 0f7e7e4 |
| three_fact_resolution_v1 | Multi-fact generalization | 6e7b5ca |
| position_contribution_v1 | Whole-sequence distributed | 85cb6a2 |
| continuation_congruence_v1 | 97% greedy, 0% distributional | a1ed285 |
| distributional_congruence_v1 | 0/96 distributional congruence | 048bae1 |
| entropy_structure_v1 | Commitment bottleneck 0.05 bits | cc83d06 |
| rebroadening_test_v1 | Re-broadened distribution meaningful | 915988c |
| predictive_fiber_v1 | Two-component residual (predictive + presentation) | 4978a85 |
| predictive_fiber_action_v2 | $S^W_w$ idempotent, non-natural with correction, path dependence confirmed | e9e54ef |
| signature_restatement_v1 | $S^G_g$ exists: 100% idempotent, 100% descent, non-natural with correction, correction non-descent (58-80%), fixed cyclic shuffle 0/32 (pairing-sensitive), anti-echo alias Gate 3 FAILS (faithful ≈ shuffled) | pending |

## 8. CPA-1 pre-registration: commitment-port absorption after failed textual correction

**Status:** DESIGN-GATE PRE-REGISTRATION, UNRUN, DEFERRED. Deferred in favor
of Commitment Hysteresis v1 (Section 9), which tests a related but more
fundamental question with fewer forward passes (456 vs 2,077). CPA-1 is
authorized only after the Hysteresis experiment completes and is audited.
No model forward pass is authorized by this section. This is a new
intervention-defined line, not a repair of the terminal prompt-renderer line.
The Phase 4d selection, carriers, sites, estimands, thresholds, and call count
below must be bound to a manifest before scientific compute.

**So what:** *A failed textual correction can be made to stick by replacing one
causal state at the model's own commitment boundary -- the first evidence that
a model decision is a transplantable, directed state rather than a point in
vector space.*

### 8.1 Fixed population and exact prompts

Use frozen `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, CPU float32, batch one, cache
disabled. The source population is fixed from the completed Phase 4d artifact
`experiments/results/signature_restatement_v1/phase4d_results.json`. A
coordinate belongs to the population exactly when its stored
`D2_counterfactual_only` row has `output_class != "counterfactual"`. No row is
selected from CPA-1 outcomes. This gives 67 directed coordinates: 27 registered
and 40 held-out, spanning all eight semantic-world clusters in each set.

For selected coordinate $i$, let $B_i$ be its stored shared-domain base
history, $e_i$ its queried entity, $g_i$ its stored observed base signature,
and $\bar g_i=\nu\circ g_i$ the already-locked Phase 4d fixed-point-free
counterfactual. Let $R(h)$ be the already-locked direct `Record` rendering of
signature $h$. The presentation order of $B_i$ is retained. Define

\[
\begin{aligned}
x_i^- &= B_i\,R(\bar g_i)\,q_{e_i},
&&\text{the resistant Phase 4d contradiction},\\
x_i^+ &= B_i\,R(g_i)\,q_{e_i},
&&\text{the matched incumbent-confirming carrier},\\
y_i &= B_{\bar g_i}\,R(\bar g_i)\,q_{e_i},
&&\text{the congruent counterfactual donor},\\
z_i &= B_{h_i}\,R(h_i)\,q_{e_i},
&&\text{the queried-value specificity donor},
\end{aligned}
\]

where $B_h$ uses the same shared-domain declaration, entity order, grammar,
and query as $B_i$ but declares signature $h$, and

\[
(h_i)_{e_i}=(g_i)_{e_i},\qquad
(h_i)_j=(\bar g_i)_j\quad(j\ne e_i).
\]

Thus $z_i$ matches the counterfactual donor on both nonqueried coordinates but
retains the incumbent value at the queried coordinate. Every value is one
token under the Phase 4d preflight. The runner must verify equal token length,
equal non-value token IDs, identical query-port index, fixed-point freedom,
and the stored 67-row selection before installing hooks. Any failure is
`INVALID_CARRIER`; no scientific result survives.

For example, the first fixed registered coordinate is
`w000_std / MIP`, with stored $g=(\text{big},\text{hot},\text{blue})$ and
$\bar g=(\text{small},\text{cold},\text{red})$. Its exact resistant recipient
is:

```text
Every named item has exactly one value from this shared value vocabulary: big, small, hot, cold, red, blue. ZOG: big. MIP: hot. PLIM: red.
Record:
ZOG: small
MIP: cold
PLIM: red
End record.
MIP:
```

Its exact counterfactual donor $y$ changes the base declarations to
`ZOG: small. MIP: cold. PLIM: red.` and keeps the displayed record and query
unchanged. Its exact specificity donor $z$ uses
`ZOG: small. MIP: hot. PLIM: red.` in both the base declarations and record.
The other 66 prompt quadruples follow this same deterministic construction
from their stored row IDs; the manifest saves every complete prompt and hash.

### 8.2 Commitment-port action -- definition

Let $H_\ell$ be the post-block-$\ell$ whole-sequence carrier and let $p_i$ be
the final query position of coordinate $i$. For a donor $y$ and recipient $x$
with the same registered carrier layout, define the surgeon action

\[
P_{\ell,p_i}^{y\leftarrow x}:H_\ell\to H_\ell
\]

by replacing exactly the single row $h_{\ell,p_i}(x)$ with
$h_{\ell,p_i}(y)$ and leaving every other row unchanged, then executing the
model's unchanged blocks $\ell+1,\ldots,27$, final norm, and unembedding. This
is an exact same-carrier substitution. It does not add, average, rotate,
project, normalize, or compare residual vectors.

Use intervention sites

\[
\mathcal L=\{18,21,22,23,24,25,26\},
\]

with zero-based post-block indexing, matching the existing hook convention.
Layer 21, the entry to the measured resolution transition, is the prospectively
primary site; layer 18 is the prospectively primary pre-transition comparator.
Layers 22--26 report the locked transition curve and may not replace either
primary site after inspection. Layer 25 is a near-output positive-control
ceiling: success there cannot rescue a layer-21 nonpass.

For every downstream post-block checkpoint $k>\ell$, let $\rho_k(u)$ be the
final-norm-plus-unembedding law of the final query row. Let $\rho_{\rm out}(u)$
be the actual final next-token law. All response discrepancies use the existing
natural-log square-root Jensen--Shannon distance $D$ on the full vocabulary,
exactly as implemented by `js_dist` in the proven instrument. Argmax is a
secondary behavioral endpoint only. (The capture ratio below is invariant to
any common positive normalization of $D$.)

For recipient $x$, target donor $y$, edited run $u=P_{\ell,p}^{y\leftarrow x}x$,
and checkpoint $k$, define **target capture**

\[
\chi_{\ell,k}(x\leftarrow y)=
\frac{D(\rho_k(u),\rho_k(x))-D(\rho_k(u),\rho_k(y))}
     {D(\rho_k(x),\rho_k(y))}.
\]

By the reverse triangle inequality, $-1\le\chi\le1$: an unchanged source has
$\chi=-1$, an exact target has $\chi=1$, and an equidistant response has
$\chi=0$. If the denominator is below the locked material-separation floor
$0.05$, set $\chi=-1$ for the primary aggregation rather than dropping the
row. For the primary layer define the **transition-absorption score**

\[
a_{21}(x\leftarrow y)=
\min\{\chi_{21,22}(x\leftarrow y),
       \chi_{21,25}(x\leftarrow y),
       \chi_{21,{\rm out}}(x\leftarrow y)\}.
\]

This asks whether a target state installed at the entry to the resolution
transition remains targetward across that transition and the remaining native
suffix, not merely whether the immediate logit lens reads the copied donor row.

### 8.3 CPA-1 proposition and falsifier

**Proposition CPA-1 (commitment-port absorption, empirical).** On Phase 4d
coordinates where a fixed-point-free contradictory record failed to control
the answer, the counterfactual query-port action
$P_{21,p_i}^{y_i\leftarrow x_i^-}$ is an approximately target-absorbing,
queried-value-specific continuation action: it moves the final full-vocabulary
response law from the resistant source toward the congruent counterfactual
donor, the move survives the remaining native suffix, the matched donor that
changes only nonqueried commitments does not reproduce it, and the effect is
materially stronger than the same target-port substitution at layer 18.

The strongest confound is trivial dominance by any donor query row, independent
of the queried value. The $z_i$ arm is the direct moot-maker: it has the same
grammar, positions, and two counterfactual nonquery values as $y_i$ but retains
the incumbent queried value. A positive target action without a positive
target-minus-$z$ specificity margin does not test the proposition.

The direct falsifier is a valid carrier on which the layer-21 target action
remains sourceward, does not change argmax at a material rate, or is no more
targetward than the $z_i$ action. A layer-21 effect that is equally present at
layer 18, or a layer-25 effect without a layer-21 effect, establishes at most a
late query port and falsifies the claimed commitment-transition action.

### 8.4 Locked estimands, clustering, and verdict bars

Rows are nested inside the already-fixed semantic-world cluster. Registered
and held-out sets are adjudicated separately. Each point estimate is the mean
of equal-weight world-cluster means. Bounds are deterministic 10,000-resample
percentile intervals over the eight whole-world clusters per set, seed 51203;
entity, presentation order, carrier, layer, and downstream checkpoint remain
nested.

Before the proposition is interpreted, all of these interface checks must pass:

1. every carrier/token/query-position check and every same-row hook check is
   valid;
2. unedited deterministic replay and layer-21 self-patch have maximum
   full-law distance at most $10^{-5}$;
3. $y_i$ follows $(\bar g_i)_{e_i}$ greedily on at least 90% of coordinates in
   each set, with world-cluster lower bound at least 75%; and
4. $D(\rho_{\rm out}(x_i^-),\rho_{\rm out}(y_i))\ge0.05$ on at least 90% of
   coordinates in each set. Sub-floor rows remain in the primary aggregation
   as $\chi=-1$.

Failure is `NO_INTERFACE_OR_INVALID`; mechanism bars are not interpreted and
there is no prompt, layer, threshold, or donor repair.

Conditional on interface validity, `CPA1_CONFIRMED` requires **all** of the
following approximate/effect-size bars in each entity set:

1. layer-21 final target capture has estimate at least $0.50$ and lower bound
   at least $0.25$;
2. layer-21 greedy counterfactual following is at least 70%, with lower bound
   at least 50% (the selected Phase 4d source rate is exactly zero by
   construction);
3. layer-21 final specificity,
   $\chi(x^-\leftarrow y)-\chi(x^-\leftarrow z)$, is at least $0.50$ with
   lower bound above zero;
4. the transition-absorption score $a_{21}$ is at least $0.25$ with lower bound
   above zero; and
5. the localization contrast
   $\chi_{21,{\rm out}}(x^-\leftarrow y)-
    \chi_{18,{\rm out}}(x^-\leftarrow y)$ is at least $0.25$ with lower bound
   above zero.

Exact row counts are diagnostics only. A valid result is `CPA1_REFUTED` if, in
either set, the upper bound for layer-21 target capture is below $0.20$, the
upper bound for greedy counterfactual following is below 40%, or the upper
bound for target-minus-$z$ specificity is at most zero. A valid result that
meets neither approximate confirmation nor refutation bars is
`CPA1_INCONCLUSIVE`. If the target action passes its capture, following,
specificity, and absorption bars but misses localization, report
`PORT_ACTION_NOT_COMMITMENT_LOCALIZED`; do not call it CPA-1 confirmation.
If layer 21 fails but layer 25 passes, report `LATE_PORT_ONLY`; this is an
activation-proximity diagnostic, not confirmation.

The $y_i\to x_i^+$ arm is an adversarial incumbent-confirming-context control.
Its effect and its paired difference from $y_i\to x_i^-$ are descriptive: the
Phase 4d resistance selection makes them unsuitable for an unqualified
hysteresis claim. No threshold may be added after inspection.

### 8.5 Implementation and exact CPU budget

Create one runner, `experiments/run_commitment_port_action_v1.py`; do not alter
or rerun `run_signature_restatement_v1.py`. Reuse:

- Phase 4d carriers, fixed involution, record grammar, tokenizer preflight,
  model revision, and SHA helpers from `run_signature_restatement_v1.py`;
- whole-stack hidden capture and final-norm-plus-unembedding from
  `run_logit_lens_resolution_v1.py`;
- the hook shape from `run_causal_resolution_v1.py`, but replace only
  `output[0][:,-1,:]`, never the complete residual sequence; and
- the Phase 4d whole-world clustered estimator, extended to the locked CPA-1
  estimands.

Every edited call computes the full downstream logit-lens trajectory and final
full-law JSD before reduction, and saves every component distance, capture
score, greedy token, target/incumbent logits, top tokens, prompt and carrier
hashes, hook count, model/tokenizer revisions, runner/config/manifest hashes,
and source row ID. Retain the float32 full-vocabulary laws needed to recompute
every primary layer-21 capture, specificity, absorption, interface, replay, and
self-patch endpoint; nonprimary curve laws may be reduced to the registered
scalar components plus hashes. The runner must refuse overwrite and checkpoint
each completed call identity.

The exact scientific call table has 2,077 batch-one CPU forward passes:

| Calls | Purpose |
|---:|---|
| $67\times4\times2=536$ | capture and deterministic replay of $x^-$, $x^+$, $y$, and $z$ |
| $67\times7=469$ | $y\to x^-$ at all seven sites |
| $67\times7=469$ | $y\to x^+$ at all seven sites |
| $67\times7=469$ | $z\to x^-$ specificity arm at all seven sites |
| $67\times2=134$ | layer-21 self-patches of $x^-$ and $x^+$ |

No generation, sampling, training, GPU work, PCA, cosine, Euclidean distance,
linear probe, learned direction, vector addition, or interpolation is allowed.
A tokenizer-only preflight and manifest construction add zero model forwards.

Projected new-runner classification: approximately 100--130 lines implement
the carrier/action/response artifact and approximately 150--190 lines implement
validation, checkpointing, provenance, reduction, and reporting, for a projected
apparatus-to-artifact ratio of 1.2--1.9:1. This theory/design round is one
artifact-definition round and zero measurement rounds; the first scientific run
would be one measurement round.

### 8.6 R^n-trap and claim wall

The experiment uses a residual tensor as a carrier but no structure inherited
from $\mathbb R^n$. Its action is exact same-phase substitution at a named
causal port, and its equality/direction/cost-free claims are defined only by
future response laws. Under any bijective recoding of the port carrier, the
action and every $\chi$ value are unchanged after conjugating the substitution.
The mathematical candidate is therefore a directed partial action on
response-law places with an absorption property, not a vector displacement.

This is still a **surgeon-world** action and has close analogues in causal
abstraction, automata, and hysteresis theory. A pass would establish a new
native operational primitive for this model -- a target-absorbing commitment
port -- not novel mathematics in the theorem-discovery sense. Never say that
the port is a denizen move, that a residual vector is a semantic object, that
the action proves a register, that one position stores the whole fact world,
that full future-response identity is certified, or that CPA-1 alone is native
mathematics. A nonpass kills this commitment-port proposition and this exact
carrier/action construction, not internal intervention laws generally.

## 9. Commitment Hysteresis v1 pre-registration (PRIMARY)

**Status:** DESIGN-GATE PRE-REGISTRATION, UNRUN. Primary next experiment
(Codex design gate recommendation). No model forward pass is authorized
by this section until the runner, config, and manifest are bound.

**So what:** *A fact copied into the model before commitment may leave a causal
trace that survives putting the original fact-state back — giving latent
motion history and direction, not merely a destination.*

### 9.1 Population and prompts

Use frozen `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, CPU float32, batch one, cache
disabled. Three binary families from the standard Phase 2 prompt surface:

| Family | Entities | Values |
|--------|----------|--------|
| 1 | ZOG, MIP | big, small |
| 2 | PLIM, KROT | hot, cold |
| 3 | HESK, VORN | red, blue |

Prompt template:
```
{A}: {va}. {B}: {vb}.
{Q}:
```

For every family, run all four worlds and both direct queries. A donor world
$y = x^e$ differs from host $x$ in exactly one entity's value. No reversed
declaration order, fresh names, unseen wording, or longer continuations —
those are separate staircase rungs, authorized only after this rung passes
and is audited.

### 9.2 Intervention span

Let $P$ be every token wholly before the queried entity on the second line —
the complete declaration prefix, including punctuation/newline tokens that lie
wholly before the boundary.

**Preflight must reject a prompt if:**
- a token straddles the declaration/query boundary;
- host and donor lengths differ;
- their query-suffix token IDs differ;
- prefix masks do not align exactly.

### 9.3 Actions — definition

Let $J_\ell^{x \leftarrow y}$ copy the clean donor's post-block-$\ell$ hidden
rows on $P$ into the host execution. Query-suffix rows remain untouched.

Define:
- $F_{21} = J_{21}^{x \leftarrow y}$: early donor-prefix action (pre-commitment)
- $F_{25} = J_{25}^{x \leftarrow y}$: late-action comparison (post-commitment)
- $R_m = J_m^{x \leftarrow x}$, $m \in \{21, \ldots, 25\}$: restore the natural
  host prefix after the forward action
- $C_{25}$: restore every position — not merely $P$ — from the clean host at
  L25 (instrumentation closure control)

**Arms per directed host/donor/query row:**

| Arm | Action | Purpose |
|-----|--------|---------|
| 1 | self-patch at L21 | Control: identity check |
| 2 | $F_{21}$ | Primary: early donor-prefix transplant |
| 3 | $F_{25}$ | Comparison: late donor-prefix transplant |
| 4 | $R_{21} F_{21}$ | Immediate restoration |
| 5 | $R_{22} F_{21}$ | Progressive restoration curve |
| 6 | $R_{23} F_{21}$ | Progressive restoration curve |
| 7 | $R_{24} F_{21}$ | Progressive restoration curve |
| 8 | $R_{25} F_{21}$ | Key: does donor influence survive commitment? |
| 9 | $C_{25} F_{21}$ | Full-state restoration control |

Population: 12 undirected square edges, 48 directed intervention/query rows,
24 clean prompt-query rows. **Total: 456 CPU forward passes.**

### 9.4 Prefix-relative hysteresis — definition

Let $x, y$ be histories with aligned declaration-prefix span $P$, and let
$s_\ell(x; q)$ denote the model's complete execution state after block $\ell$
under registered query $q$.

The natural prefix substitution $J_\ell^{x \leftarrow y}$ replaces
$s_\ell(x; q)|_P$ by $s_\ell(y; q)|_P$, leaving all positions outside $P$
unchanged, after which the model's ordinary transition resumes.

For $F = J_{\ell_0}^{x \leftarrow y}$ and $R_m = J_m^{x \leftarrow x}$, the
execution has **prefix-relative hysteresis at $m$** when

$$
[Fx]_Q \neq [x]_Q, \qquad [R_m F x]_Q \neq [x]_Q,
$$

even though the designated prefix has been restored to its clean host state
at layer $m$.

### 9.5 Proposition — finite-kernel hysteresis witness

Let $F = J_{21}^{x \leftarrow y}$, let $R_m = J_m^{x \leftarrow x}$, and let
$C_{25}$ restore the complete clean host state at L25. If

$$
[Fx]_Q \neq [x]_Q,
$$
$$
[R_{21} F x]_Q = [x]_Q,
$$
$$
[R_{25} F x]_Q \neq [x]_Q,
$$

and

$$
[C_{25} F x]_Q = [x]_Q,
$$

then $R_{25}$ is not a left inverse of $F$ on the registered response quotient.
Consequently, sequential execution has transferred response-relevant residue
outside the restored prefix, and the system exhibits prefix-relative hysteresis.

If $F_{21}$ is materially more donor-directed than $F_{25}$, the accessibility
of the move is additionally depth-dependent across the L21–L25 commitment
interval.

The proposition is exact mathematics. Empirical adjudication uses the
approximate criteria below; exact equalities are diagnostics only.

### 9.6 Response-law distance

For each query $q$, retain the complete next-token probability distribution
over the model vocabulary — never argmax, selected tokens, or top-$k$
truncation.

Use the normalized response-law distance:

$$
d_{Q_f}(Sx, Ty) = \max_{q \in Q_f} \sqrt{\frac{D_{\mathrm{JS}}(p_\theta(\cdot \mid S(x,q)),\; p_\theta(\cdot \mid T(y,q)))}{\log 2}}.
$$

This is a finite registered quotient, not a certificate of complete
future-law equality. A positive value is nevertheless a valid witness that
the full future laws differ.

### 9.7 Estimands and gates

For directed edge $i: x \to y = x^e$, let $b_i = d_{q_e}(x, y)$ be the
natural response-law separation on the changed entity's query.

An edge is carrier-eligible when $b_i \geq \max(0.02, 8\eta)$, where $\eta$
is the maximum clean-replay/control discrepancy.

Define:
- $M_i = d_Q(F_{21} x, x) / b_i$ — move size
- $T_i = (b_i - d_{q_e}(F_{21} x, y)) / b_i$ — donor-directed progress
- $H_i = d_Q(R_{25} F_{21} x, x) / b_i$ — restored-prefix residue
- $U_i = (b_i - d_{q_e}(R_{25} F_{21} x, y)) / b_i$ — donor-directed residue after restoration
- $L_i = T_i^{(21)} - T_i^{(25)}$ — commitment localization

Statistical unit: undirected binary-world edge (12 clusters). Average the two
directions and both query executions inside each edge. Deterministic
10,000-resample cluster bootstrap, seed 51203.

### 9.8 Locked verdict bars

**Interface checks (must all pass):**
1. Every carrier/token/query-position/prefix-mask check is valid
2. Normalized self-patch, $R_{21} F_{21}$, and $C_{25} F_{21}$ controls: 95%
   upper bound $\leq 0.02$
3. At least 9/12 edge clusters carrier-eligible

**`COMMITMENT_HYSTERESIS_REGISTERED` requires ALL of:**
1. Lower bound of mean $M_i \geq 0.25$
2. Lower bound of mean $T_i \geq 0.20$
3. Lower bound of mean $H_i \geq 0.15$
4. Lower bound of mean $U_i \geq 0.10$
5. Lower bound of mean $L_i \geq 0.10$

**Stop conditions:**

| Status | Rule | Consequence |
|--------|------|-------------|
| `INVALID_CARRIER` | <9/12 eligible edges or control UB > 0.02 | Fix integrity only; no scientific verdict |
| `NO_ACTION_INTERFACE` | 95% UB of $M$ or $T$ < 0.10 | Kill whole-prefix substitution as action interface |
| `REVERSIBLE_ACTION` | Action passes but UB of $H$ and $U$ both < 0.05 | Kill hysteresis law; retain reversibility |
| `HYSTERESIS_NOT_COMMITMENT_LOCALIZED` | $H, U$ pass but UB of $L$ < 0.05 | Keep generic hysteresis; kill localization claim |
| `INCONCLUSIVE_ALLOCATION_STOP` | Valid but intervals between pass/kill | Stop; no tuning |
| `COMMITMENT_HYSTERESIS_REGISTERED` | All success gates pass | Audit, then design held-out-name rung |

No generation, prompt repair, threshold tuning, or parameter sweep after
inspection. One terminal attempt.

### 9.9 R^n trap and claim wall

The experiment uses residual tensors as carriers but no structure inherited
from $\mathbb{R}^n$. Its actions are exact naturally occurring causal
substitutions. The mathematical object is sequential action on response-law
equivalence classes with a hysteresis property. Under any bijective recoding
of the carrier, the actions and all estimands are unchanged after conjugating
the substitution.

**Claim wall:** A pass establishes prefix-relative hysteresis at the
commitment transition — a donor-directed response-law move that survives
prefix restoration. It does NOT establish intrinsic denizen action, full
future-response identity, held-out-name generality, unseen-wording
generality, or native mathematics alone. A nonpass kills this exact
carrier/action/span construction, not internal intervention laws generally.

### 9.10 Result: INCONCLUSIVE_ALLOCATION_STOP (2026-09-01, Codex evidence gate: REVISE, corrections adopted)

456 forward passes, 48 directed edges, 12 clusters, all eligible.
Controls perfect ($\eta = 0$, self-patch and C25 both 0.000).

Corrected metrics (Codex independent reduction per Section 9.7):

| Estimand | Mean | 95% CI | Gate | Status |
|----------|------|--------|------|--------|
| $M$ | 0.438 | [0.383, 0.504] | $\geq 0.25$ | **PASS** |
| $T$ | 0.200 | [0.144, 0.247] | $\geq 0.20$ | FAIL |
| $H$ | 0.403 | [0.322, 0.512] | $\geq 0.15$ | **PASS** |
| $U$ | 0.092 | [$-$0.005, 0.169] | $\geq 0.10$ | FAIL |
| $L$ | 0.162 | [0.091, 0.222] | $\geq 0.10$ | FAIL |

Under this exact whole-prefix intervention, an L21 transplant produced a
response-law shift that remained after the clean host prefix was restored
at L25. The corrected preregistered reduction cleared the descriptive $M$
and $H$ thresholds but did not clear the donor-directed $T$/$U$ or
localization $L$ lower-bound gates; the locked result is
INCONCLUSIVE_ALLOCATION_STOP. It does not establish absence of
donor-directed transfer, generic disruption, or commitment-specific
hysteresis.

Codex audit note: original reducer averaged all queries instead of using
changed-entity query for $T$/$U$/$L$ per Section 9.7. High $H$ remains
compatible with ordinary causal propagation from replacing an entire
upstream prefix. This experiment is terminal.

#### Post-hoc layer-resolved observation (descriptive; not pre-registered)

The restore profile across layers $\ell \in \{21,...,25\}$ shows a pattern
at the commitment boundary:

| Restore $\ell$ | $H_\ell$ | $U_\ell$ | Efficiency $U/H$ | Pos. $U$ edges |
|----------------|----------|----------|-------------------|----------------|
| 21 | 0.000 | 0.000 | --- | 0/48 |
| 22 | 0.297 | $-$0.051 | $-$0.17 | 20/48 |
| 23 | 0.480 | $-$0.102 | $-$0.21 | 17/48 |
| 24 | 0.507 | $-$0.092 | $-$0.18 | 16/48 |
| 25 | 0.423 | +0.108 | +0.25 | 38/48 |

At L24$\to$L25: 25/48 edges flip from anti-directional to pro-directional;
3/48 flip the other way (ratio 25:3). Codex notes this is compatible with
ordinary causal propagation from replacing an entire upstream prefix --- not
confirmed as a commitment-specific mechanism.

---

## Section 10: Endogenous Response-Quotient Selector (ERQ-1)

### 10.1 Motivation

Section 9's commitment hysteresis used a *surgeon-defined* action: the
experimenter chose which prefix positions to replace, which layers to
transplant at, and which layers to restore. The resulting signal is
compatible with ordinary causal propagation from replacing an entire
upstream prefix. ERQ-1 asks whether the model's *own* block-25 computation
performs a query-selective response-law transformation that an identity
carry of the same state cannot produce.

### 10.2 Mathematical object

The object is a phase-indexed **response-kernel category**, not a group:

$$
\mathcal{Q}_{\mathcal{R}}
= (Q_{24}, Q_{25}, Q_{\mathrm{out}};\;
   \bar{B}_{25}, \bar{I}_{25}, \bar{F}).
$$

Carrier and maps:

- $H_{24}$: whole-sequence hidden state entering `model.model.layers[25]`,
  i.e. the post-block-24 carrier.
- $B_{25}: H_{24} \to H_{25}$: the model's actual block-25
  computation --- the **endogenous action** under test.
- $I_{25}: H_{24} \to H_{25}$: identity carry $I(h) = h$, the
  **ordinary-propagation control**.
- $F: H_{25} \to \Delta(V)$: unchanged blocks 26--27, final layernorm,
  and unembedding. The **full suffix** observer.
- $O: H_{25} \to \Delta(V)$: immediate final-layernorm-plus-unembedding
  (logit lens). The **immediate** observer.

### 10.3 Phase equivalence

Define equivalence by registered future response laws:

$$
h \sim_\ell h'
\iff
\rho_c(h) = \rho_c(h')
\quad \forall\, c \in \mathcal{R}_\ell.
$$

The registered ports include $O$, $F$, and their compositions with
$B_{25}$ and $I_{25}$. Because the continuation family is prefix-closed,
the native block descends to $\bar{B}_{25}: Q_{24} \to Q_{25}$.
Composition is sequential execution; there is no assumed inverse or
vector operation.

### 10.4 Distance

All distances exclusively use the normalized response-law metric:

$$
d(p, p') = \sqrt{\frac{D_{\mathrm{JS}}(p, p')}{\log 2}},
\qquad
D_{\mathrm{JS}}(p, p') = \tfrac{1}{2} D_{\mathrm{KL}}(p \| m)
                        + \tfrac{1}{2} D_{\mathrm{KL}}(p' \| m),
\quad m = \tfrac{1}{2}(p + p').
$$

Full vocabulary, no top-$k$ truncation. Range $[0, 1]$.

### 10.5 Selective-identification morphism

The candidate law is a **query-indexed selective identification
morphism** --- not a categorical coequalizer with a claimed universal
property.

For a world edge changing fact $e$, block 25 should:

1. **Separate** that edge more when $e$ is queried (amplification).
2. **Identify** it more when the other fact is queried (compression).
3. Do both **beyond** the identity-carry suffix.
4. **Preserve** the pattern under declaration-order reversal.

### 10.6 Estimands

For endpoint $z \in \{O, F\}$, define:

$$
A_z = \mathbb{E}\bigl[d^B_{z,\text{relevant}} - d^I_{z,\text{relevant}}\bigr],
$$

$$
C_z = \mathbb{E}\bigl[d^I_{z,\text{irrelevant}} - d^B_{z,\text{irrelevant}}\bigr],
$$

$$
\Sigma_z = A_z + C_z.
$$

Here "relevant" means the edge changes the queried fact; "irrelevant"
means it changes the non-queried fact.

Ordinary propagation is literally the $I_{25}$ arm: the same naturally
occurring upstream state enters the same remaining suffix, with no
donor replacement. Its null is $\Sigma_F \approx 0$.

ERQ-1 predicts $A > 0$, $C > 0$, and therefore $\Sigma > 0$. The same
semantic edge must reverse its role when the query changes. A generic
perturbation or indiscriminate block effect cannot satisfy that paired
difference-in-differences.

The immediate $O$ endpoint localizes the operation to block 25. The true
$F$ endpoint establishes that it remains effective through the actual
suffix. An $O$-only result is merely a logit-lens phenomenon.

### 10.7 Population and budget

Reuse the three fixed families from Section 9:

- ZOG / MIP --- big / small
- PLIM / KROT --- hot / cold
- HESK / VORN --- red / blue

Use four worlds, both queries, and standard / reversed declaration order:

$$
3 \times 4 \times 2 \times 2 = 48 \text{ prompt cells}.
$$

Run exactly three arms per cell:

1. Native clean execution.
2. No-op output-hook replay (instrument control).
3. Block-25 identity bypass.

**Total: 144 CPU forward passes.** No generation, donor state, position
mask, layer sweep, training, PCA, cosine, interpolation, or vector
arithmetic.

### 10.8 Statistical design

Statistical unit: the 12 undirected semantic-world edges. Query roles
and the two presentation orders are paired and nested inside each edge.

Bootstrap: deterministic 10,000-resample percentile bootstrap over whole
edge clusters, seed `51203`. Family means are robustness diagnostics.
Exact row counts remain diagnostics, not primary verdict gates.

### 10.9 Locked preregistration

| Gate | Locked bar |
|------|------------|
| Instrument integrity | Model/revision/layer/token checks pass; exactly one hook invocation per arm; every law finite and normalized; max clean-vs-noop $d_F \le 10^{-5}$ |
| Material support | $\ge 9/12$ edge clusters have native relevant $d_F \ge 0.05$ |
| Bypass viability | On $\ge 44/48$ cells, bypass probability mass on the family's two registered value tokens $\ge 50\%$ of native mass |
| Immediate native action | $A_O$ lower bound $\ge 0.10$; $C_O$ lower bound $\ge 0.05$; $\Sigma_O$ lower bound $\ge 0.20$ |
| Suffix survival | $A_F, C_F$ estimates $\ge 0.05$ with lower bounds $> 0$; $\Sigma_F$ estimate $\ge 0.10$ and lower bound $\ge 0.05$ |
| Presentation stability | Upper bound of native-minus-identity presentation excess $\le 0.05$ |
| Family robustness | Every family's $\Sigma_F$ estimate $\ge 0.02$ |

### 10.10 Verdicts

In priority order:

| Verdict | Condition | Consequence |
|---------|-----------|-------------|
| `INVALID_OR_NO_PROPAGATION_CONTROL` | Instrument, material, or bypass gate fails | Fix integrity only; no scientific verdict |
| `ENDOGENOUS_RESPONSE_QUOTIENT_REGISTERED` | All seven gates pass | Block 25 is an effective query-selective sieve for this model and carrier |
| `LOCAL_OBSERVER_ONLY` | Immediate action passes, suffix survival fails | Block-25 selection is a logit-lens artifact that does not persist |
| `ORDINARY_PROPAGATION_SUFFICIENT` | Both immediate and suffix fail | Block 25 adds nothing query-selective beyond identity carry |
| `QUERY_SELECTIVE_BUT_PRESENTATION_UNSTABLE` | Action passes but presentation stability fails | Signal is declaration-order-dependent |
| `ONE_SIDED_BLOCK_ACTION` | Only $A$ or only $C$ passes, not both | Block amplifies or compresses but does not selectively identify |
| `INCONCLUSIVE_ALLOCATION_STOP` | Valid but intervals between pass/kill | Stop; no tuning |

Every nonpass is terminal. No alternate layer, carrier subset, threshold,
or prompt repair.

### 10.11 $\mathbb{R}^n$ trap and claim wall

The experiment uses residual tensors as carriers but no structure inherited
from $\mathbb{R}^n$. The endogenous action $B_{25}$ is the model's own
block computation, not an experimenter-designed substitution. The identity
carry $I_{25}$ is the simplest possible control: pass the state unchanged.
All estimands are defined in response-law distance $d$, which is invariant
under any bijective recoding of the carrier.

**Claim wall:** A pass establishes an effective, registration-relative
response-quotient morphism for this model and this carrier. It does NOT
establish a universal algebra, an autonomous denizen move, a mechanism
beyond ordinary late decoding, or native mathematics alone. The strongest
surviving null is: block 25 is simply a generic late query-conditioned
decoder that sharpens the requested verbalizer and damps unrelated token
mass; deleting it is an off-manifold lesion. A pass would show that one
native transformer block acts as a query-controlled sieve, keeping apart
worlds that differ in the fact being asked about while identifying worlds
that differ only elsewhere --- effective selective identification beyond
identity carry.

A nonpass kills this exact block/carrier/span construction, not internal
endogenous computation laws generally.

### 10.12 Result: INVALID_OR_NO_PROPAGATION_CONTROL (2026-09-01)

144 forward passes. Instrument integrity PASS (noop $d = 0.0$). Material
support PASS (12/12 edge clusters eligible). **Bypass viability FAIL (7/48
cells retain $\ge 50\%$ value token mass).**

| Gate | Value | Bar | Status |
|------|-------|-----|--------|
| Instrument | $d_{\text{noop}} = 0.0$ | $\le 10^{-5}$ | **PASS** |
| Material | 12/12 | $\ge 9/12$ | **PASS** |
| Bypass viability | 7/48 | $\ge 44/48$ | FAIL |
| $A_O$ | 0.254 [0.198, 0.313] | lb $\ge 0.10$ | **PASS** (descriptive) |
| $C_O$ | 0.174 [0.086, 0.256] | lb $\ge 0.05$ | **PASS** (descriptive) |
| $\Sigma_O$ | 0.428 [0.304, 0.539] | lb $\ge 0.20$ | **PASS** (descriptive) |
| $A_F$ | 0.159 [0.111, 0.208] | est $\ge 0.05$, lb $> 0$ | **PASS** (descriptive) |
| $C_F$ | 0.013 [$-$0.017, 0.043] | est $\ge 0.05$, lb $> 0$ | FAIL |
| $\Sigma_F$ | 0.172 [0.126, 0.214] | est $\ge 0.10$, lb $\ge 0.05$ | **PASS** (descriptive) |
| Stability | ub = 0.038 | $\le 0.05$ | **PASS** (descriptive) |
| Family | all $> 0.02$ | each $\ge 0.02$ | **PASS** (descriptive) |

The identity bypass is too destructive --- block 25 is a critical
computation step, and skipping it pushes blocks 26--27 into off-manifold
states. This is exactly the pre-declared strongest null: deleting the block
is an off-manifold lesion. No scientific verdict. The O-endpoint data is
descriptively consistent with query-selective action but the control is
invalid. Terminal.

---

## Section 11: Observational Selectivity Quotient (OSQ-1)

### 11.1 Motivation

ERQ-1 demonstrated that the identity bypass at block 25 is too destructive
to serve as an ordinary-propagation control (7/48 bypass viability). But
its O-endpoint data --- purely observational logit-lens measurements at
block output, requiring no bypass --- showed strong query-selective action
($\Sigma_O = 0.428$ $[0.304]$). The earlier logit-lens resolution v1
independently measured selectivity ratios of 3.78--62.43$\times$ at
L21--25, with ratio $\approx 1$ at early layers.

OSQ-1 formalizes these observational measurements with the statistical
apparatus developed for ERQ-1: edge-cluster bootstrap, declaration-order
reversal, and family robustness gates. It is **purely observational** ---
no intervention, no bypass, no transplant. The control is the early-layer
null: at layers before the resolution span, the model has not yet decided
which fact to amplify, so selectivity should be $\approx 0$.

OSQ-1 shares ERQ-1's 48-cell population (same families, templates, worlds).
It is therefore a locked consolidation of the layer-resolved selectivity
trajectory, not an independent confirmation on new stimuli.

### 11.2 Mathematical object

The object is a **layer-resolved selectivity contrast** with emergence
and quotient quantities:

$$
S(\ell) = \mathbb{E}_{e,o}[\delta_{e,o}(\ell)], \quad
\delta_{e,o}(\ell) = d^{\mathrm{rel}}_{e,o}(\ell) - d^{\mathrm{irr}}_{e,o}(\ell)
$$

$$
B = \tfrac{1}{5}\sum_{\ell=0}^{4} S(\ell), \quad
G(\ell) = S(\ell) - B
$$

$$
R(\ell) = \mathbb{E}[d^{\mathrm{rel}}], \quad
I(\ell) = \mathbb{E}[d^{\mathrm{irr}}], \quad
\mathrm{OSQ}(\ell) = \frac{R(\ell) - I(\ell)}{R(\ell) + I(\ell)}
$$

where:

- $\ell \in \{0, \ldots, L{-}1\}$ indexes post-block layers.
- $e$ ranges over 12 undirected edges (3 families $\times$ 4 each).
- $o \in \{\mathrm{std}, \mathrm{rev}\}$ is declaration order.
- $d^{\mathrm{rel}}_{e,o}(\ell)$ is JSD$_{\mathrm{norm}}$ at layer $\ell$
  between the two edge endpoints when Q queries the changed fact.
- $d^{\mathrm{irr}}_{e,o}(\ell)$ is the same when Q queries the unchanged fact.
- $B$ is the early-window baseline (layers 0--4).
- $G(\ell)$ is the emergence gain above baseline.
- OSQ$(\ell)$ is a scale-free selectivity quotient.

The JSD$_{\mathrm{norm}}$ uses the logit lens: apply the model's own final
layernorm + unembedding to the last-token hidden state at layer $\ell$ to
get a pseudo-distribution, then compute $\sqrt{\mathrm{JSD}(p, q) / \ln 2}$.

### 11.3 Why purely observational

No intervention means no off-manifold confound. The logit lens uses the
model's own unembedding matrix --- it is the model's native readout at
intermediate layers, not an external R$^n$ projection. The selectivity
gain $S(l)$ compares what the model would say at layer $l$ about two
different inputs (changing one fact), conditioned on which fact the query
asks for. The comparison is between the model's own computations, not
between external metrics.

### 11.4 Estimands

| Estimand | Definition | Interpretation |
|----------|-----------|----------------|
| $G(\ell)$ | $S(\ell) - B$ | Emergence gain above early baseline |
| $\mathrm{OSQ}(\ell)$ | $(R - I)/(R + I)$ | Scale-free selectivity |
| $\ell_{\mathrm{peak}}$ | $\arg\max_\ell G(\ell)$ | Peak emergence layer |
| $\ell_{\mathrm{onset}}$ | First $\ell$ with $G(\ell), G(\ell{+}1) \ge 0.10$, both lb $> 0$ | Onset layer |
| $S_{\mathrm{std}}$, $S_{\mathrm{rev}}$ | Selectivity per declaration order at anchor | Presentation stability |
| $V$ | $S^\pi(a) / S(a)$ | Verbalizer sufficiency ratio |

### 11.5 Population and budget

48 prompts: 3 families $\times$ 4 worlds $\times$ 2 queries $\times$
2 declaration orders. Each prompt: 1 forward pass capturing hidden states
at all $L$ layers. **Budget: 48 forward passes.** Edge measurements:
12 edges $\times$ 2 queries $\times$ 2 orders $\times$ $L$ layers
$= 48 \cdot L$ JSD values.

### 11.6 Statistical design

**Bootstrap.** 10,000 resamples over 12 undirected edge clusters, seed
51203, percentile CI. Computed independently at each layer $l$.

**Declaration-order reversal.** Templates ``{A}: {va}. {B}: {vb}.\\n{Q}:``
(standard) and ``{B}: {vb}. {A}: {va}.\\n{Q}:`` (reversed). Presentation
stability measured at $l_{\mathrm{peak}}$: $|S_{\mathrm{std}}(l_{\mathrm{peak}})
- S_{\mathrm{rev}}(l_{\mathrm{peak}})|$.

**Family robustness.** Per-family $S_f(l_{\mathrm{peak}})$ must exceed
minimum threshold for all three families.

### 11.7 Locked gates

| Gate | Quantity | Bar |
|------|----------|-----|
| Integrity | Logit lens at $L{-}1$ vs final logits | $d \le 10^{-5}$ all cells |
| Material | Edges with $d_{\mathrm{rel}} \ge 0.05$ at any layer | $\ge 9/12$ |
| Early-role null | Simultaneous $S(0{:}4)$ bands | within $[-0.10, 0.10]$ |
| Late-window emergence | $\bar{G}_{21{:}25}$ lower bound | $\ge 0.10$ |
| Anchor strength | $G(25)$ lower bound | $\ge 0.30$ |
| Material separation | $R(25)$ lower bound | $\ge 0.40$ |
| Scale-free selectivity | OSQ$(25)$ lower bound | $\ge 0.30$ |
| Timing | $\ell_{\mathrm{peak}} \in [21,25]$, $\ell_{\mathrm{onset}} \in [19,25]$ | |
| Persistence | $S(27)$ lower bound | $\ge 0.05$ |
| Presentation | Per-order $S_{\mathrm{std}}(25)$, $S_{\mathrm{rev}}(25)$ lb | each $\ge 0.20$ |
| Family | Per-family $S_f(25)$ | each $\ge 0.20$ |
| Verbalizer null | $V = S^\pi(25)/S(25)$ | $V < 0.80$ and residual lb $\ge 0.05$ |

### 11.8 Verdict table (priority order)

| Verdict | Condition |
|---------|-----------|
| INVALID\_MEASUREMENT | Integrity fails |
| **OBSERVATIONAL\_SELECTIVITY\_BROAD** | All core gates pass, verbalizer null rejected |
| OBSERVATIONAL\_SELECTIVITY\_VERBALIZER\_SUFFICIENT | All core gates pass, 3-bin readout accounts for signal |
| PRESENTATION\_SENSITIVE\_SELECTIVITY | Signal pass, presentation fails |
| FAMILY\_SPECIFIC\_SELECTIVITY | Signal pass, one or more families fail |
| SELECTIVITY\_NOT\_LATE\_EMERGENT | Selectivity exists but early null or timing fails |
| TRANSIENT\_LOGIT\_LENS\_SELECTIVITY | L25 pass, L27 persistence fails |
| NO\_REGISTERED\_LATE\_SELECTIVITY | Valid run, late-window and anchor below bars |
| INCONCLUSIVE\_ALLOCATION\_STOP | Any remaining pattern |

Every verdict is terminal for this population.

### 11.9 Strongest null

**Verbalizer sufficiency.** Coarse-grain each vocabulary distribution to
$\pi_f(p) = (p(v_0), p(v_1), 1{-}p(v_0){-}p(v_1))$ and recompute
$S^\pi(25)$. If $V = S^\pi(25)/S(25)$ has upper bound $\ge 0.80$, the
selectivity is accounted for by the two registered answer tokens ---
ordinary late query-conditioned answer decoding, not a discovered latent
algebra. The ``VERBALIZER\_SUFFICIENT'' verdict attaches; the basic
observational result may still pass.

**Generic late-decoder null.** The final unembedding head amplifies the
two registered answer tokens and ignores the nonqueried fact. This is
useful model behavior but not a native-math structure.

### 11.10 R$^n$ trap check and claim wall

**R$^n$ trap.** Escapes the direct residual-coordinate trap: no cosine,
PCA, Euclidean distance, or fitted coordinate map enters the estimand.
The observer is model-owned; the result is invariant to bijective recoding
of the hidden carrier when the observer is conjugated accordingly.

**Does NOT establish native latent-space mathematics:**
- Final norm plus unembedding is applied *counterfactually* at
  intermediate layers.
- JSD is an externally selected information metric.
- Observation alone defines no endogenous action or denizen-executable
  move.

The licensed object is a **query-indexed response-law resolution profile**,
not a block morphism or causal quotient.

**Claim wall.** A passing verdict means: at specific layers, the model's
intermediate computation (as read by its own unembedding) is selectively
sensitive to changes in the queried fact and insensitive to changes in
the irrelevant fact, robustly across declaration orders and entity
families. It does NOT establish causal mechanism, endogenous vs.
inherited selectivity, logit-lens faithfulness, hidden-state geometry,
or independent replication beyond these three families.

A nonpass kills this claim for this model and measurement apparatus.

### 11.11 Result

**Verdict: OBSERVATIONAL_SELECTIVITY_VERBALIZER_SUFFICIENT.**

All 11 core gates PASS. The verbalizer-sufficiency null is NOT rejected
($V = 1.0095$; $S^{\pi}(25) = 0.647$, $S(25) = 0.641$). The three-bin
answer-token distribution accounts for the entire selectivity signal.

**Layer profile.** Early baseline $B = 0.008$ (layers 0--4 near zero).
Emergence onset at $\ell = 24$; peak at $\ell = 25$. Resolution-window
mean emergence gain $\bar{G}_{21{:}25} = 0.239$ [0.137]. At the anchor
layer ($\ell = 25$): $G = 0.633$ [0.554], $R = 0.785$ [0.712],
$I = 0.144$, $\mathrm{OSQ} = 0.706$ [0.612]. Persistence at $\ell = 27$:
$S = 0.240$ [0.184].

**Robustness.** Presentation-stable: standard-order $S(25) = 0.644$
[0.506], reversed-order $S(25) = 0.638$ [0.512]. Family-stable:
ZOG\_MIP 0.664, PLIM\_KROT 0.696, HESK\_VORN 0.563.

**Interpretation.** The selectivity at layer 25 is real and robust: the
model's intermediate computation, as read by its own unembedding, is
strongly and stably selective for the queried fact. However, this
selectivity is entirely accounted for by the two registered answer tokens.
The model's late-layer computation is amplifying the correct answer token
and suppressing the incorrect one --- ordinary answer decoding, not a
deeper distributional structure beyond what the final readout already
needs. This is Codex's pre-declared strongest null: ordinary late
query-conditioned answer decoding.

**Consequence for the program.** Purely observational logit-lens
measurements have reached their ceiling. The selectivity they detect is
the minimum necessary for correct answers. Any future experiment claiming
deeper structure must go beyond answer-token routing --- either by using
a measurement that is not explained by the verbalizer, or by
demonstrating causal intervention effects.

## Section 12 — Query-Port Composition (QPC-1)

### 12.1 Preregistration

**Proposition.** If the query-position latent state at layer 21 carries a
separable query-selector action (not merely a partial answer code), then
transplanting a donor's query-position state into a host prompt should
compose with the host's world context: the edited output should favor the
host's answer to the *transplanted* query ($t$), not the donor's clean
answer ($d$) or the host's original answer ($s$).

**Three-way clash population.** Extend the two-value families to three
values each (big/small/medium, hot/cold/warm, red/blue/green), using
entity pairs ZOG/MIP, PLIM/KROT, DREN/VORN. For each family, query
direction $q \in \{A, B\}$, cyclic rotation $r \in \mathbb{Z}_3$, and
declaration order (standard/reversed):

$$x(q) = v_r,\quad x(\bar q) = v_{r+1},\quad x \text{ asks } q$$
$$y(q) = v_r,\quad y(\bar q) = v_{r+2},\quad y \text{ asks } \bar q$$

Three deliberately distinct answers: $s = v_r$ (host clean), $t = v_{r+1}$
(host's answer to transplanted query), $d = v_{r+2}$ (donor clean).

**Intervention.** Exact row-copy at the query position (final token):
$h_{\ell, p}(x) \leftarrow h_{\ell, p}(y)$ at $\ell \in \{21, 25\}$.
No prefix replacement, no block bypass, no interpolation.

**Budget.** 36 cells $\times$ 4 arms (clean, self-patch, L21 donor, L25
donor) = 144 CPU forwards.

**Estimands.** Restrict to three-token triplet $\{s, t, d\}$:
$\rho_u(z) = p_u(z) / (p_u(s) + p_u(t) + p_u(d))$.

- $F_\ell$: fraction where $t = \arg\max \rho_\ell$
- $C_\ell = \mathbb{E}[\rho_\ell(t) - \rho_\ell(d)]$ (target over donor)
- $W_\ell = \mathbb{E}[\rho_\ell(t) - \rho_\ell(s)]$ (target over host)
- $L = C_{21} - C_{25}$ (localization)

Bootstrap: 10,000 resamples over 9 family$\times$rotation clusters,
seed 51203.

**Locked gates (10).** Carrier integrity, self-patch ($\le 10^{-5}$),
clean interface ($\ge 0.85$, cluster LB $\ge 0.70$, family $\ge 0.75$),
L21 viability (triplet mass $\ge 50\%$ on $\ge 0.85$, LB $\ge 0.70$),
target following ($F_{21} \ge 0.70$, LB $\ge 0.50$), beats donor
($C_{21} \ge 0.20$, LB $> 0$), beats host ($W_{21} \ge 0.20$, LB $> 0$),
presentation (both orders $F_{21} \ge 0.60$, $C_{21} > 0$), family
robustness (every family $F_{21} \ge 0.50$, $C_{21} \ge 0.05$),
transition localization ($L \ge 0.15$, LB $> 0$).

**Verdict table (8).** INVALID\_INSTRUMENT, NO\_VIABLE\_QUERY\_PORT,
QUERY\_PORT\_COMPOSITION\_LOCALIZED, QUERY\_PORT\_COMPOSITION\_REGISTERED,
DONOR\_VERBALIZER\_COPY, HOST\_INERTIA, LATE\_PORT\_ONLY,
INCONCLUSIVE\_ALLOCATION\_STOP.

### 12.2 Result

**Verdict: INCONCLUSIVE\_ALLOCATION\_STOP.**

**Instrument.** All instrument gates PASS. Carrier: 36 cells, 144 forwards.
Self-patch: max discrepancy $= 0.0$ (exact). Clean interface: $94.4\%$
correct, cluster LB $= 0.86$, all families $\ge 0.75$ (DREN\_VORN
$91.7\%$, PLIM\_KROT $100\%$, ZOG\_MIP $91.7\%$). L21 viability: $94.4\%$,
cluster LB $= 0.83$.

**Composition gates (all FAIL).**

| Gate | Criterion | Observed | CI |
|------|-----------|----------|----|
| Target following | $F_{21} \ge 0.70$ | $0.306$ | $[0.139, 0.472]$ |
| Beats donor | $C_{21} \ge 0.20$ | $-0.062$ | $[-0.201, 0.077]$ |
| Beats host | $W_{21} \ge 0.20$ | $0.125$ | $[-0.024, 0.273]$ |
| Presentation | $F_{21} \ge 0.60$ both | std $0.278$, rev $0.333$ | --- |
| Family | $F_{21} \ge 0.50$ all | $0.17$--$0.42$ | --- |

**Localization PASS:** $L = 0.639$ $[0.548, 0.726]$.

**Donor and source at L21:** Donor wins $58.3\%$ $[0.361, 0.778]$; source
wins $11.1\%$ $[0.0, 0.278]$. The intervention strongly disrupts the host
(source drops from $94.4\%$ clean to $11.1\%$) but does not reliably
direct toward the target --- the output scatters across the triplet with
a partial donor lean.

**At L25:** $F_{25} = 0.028$, $C_{25} = -0.701$ $[-0.793, -0.607]$. Pure
donor-verbalizer copying: the L25 state IS the answer code, and
transplanting it produces the donor's answer. This confirms localization
--- L21 is fundamentally different from L25 --- but what L21 carries is
not a composable query-selector action.

**Interpretation.** The query-position state at L21 carries partial
answer-routing information but not a separable query selector. Transplanting
it disrupts the host and partially transfers the donor's answer tendency,
but does not compose with the recipient's world to produce the recipient's
answer to the transplanted query. The three-way clash decisively separates
composition from verbalizer copying, and composition fails.

**Consequence for the program.** This is the terminal experiment for the
frozen Qwen3-0.6B two-fact readout surface, per the Codex allocation
ruling. The surface has been exhausted: R$^n$ tools project their own
structure (Phase 1), block bypass is too destructive (ERQ-1), whole-prefix
transplant is too blunt (hysteresis v1), observational selectivity is
entirely answer-token routing (OSQ-1), and query-port composition fails
(QPC-1). The next direction requires a fundamental pivot.

---

## Section 13 — Micro-World Closure: Qwen3-0.6B Two-Fact Recall

### 13.1 Summary of the exhausted surface

Approximately 70 experiments across two phases on frozen Qwen3-0.6B,
all using the two-entity one-fact-per-entity recall task
("{A}: {va}. {B}: {vb}.\n{Q}:"), with three nonsense-word entity
families and binary/ternary value assignments.

### 13.2 What was established

1. **Query-selective computation exists at L21--25** (62$\times$
   selectivity, OSQ-1 Section 11). The model performs dramatically
   different internal computation depending on which entity is queried.

2. **This selectivity is entirely answer-token routing** ($V = 1.01$,
   OSQ-1). Collapsing the vocabulary to $\{v_0, v_1, \text{rest}\}$
   preserves 100\% of the selectivity signal. The model amplifies the
   correct answer token and suppresses the incorrect one, nothing more.

3. **The query-position state carries partial answer code, not a
   compositional query action** (QPC-1, Section 12). Transplanting the
   query-position hidden state from a donor produces partial donor-answer
   leakage ($D_{21} = 58\%$), not target-following composition
   ($F_{21} = 31\%$, at three-way chance).

4. **Block bypass is too destructive** (ERQ-1, Section 10). Skipping
   block 25 pushes the model off-manifold; 7/48 cells retain viable
   output. No scientific verdict possible.

5. **Whole-prefix transplant is too blunt** (hysteresis v1, Section 9).
   Transplanting the entire prefix at L21 leaves a trace that survives
   L25 restoration, but the trace is not donor-directed --- it is
   nonspecific propagation.

6. **R$^n$ tools project their own structure** (Phase 1, 50 experiments).
   PCA, cosine similarity, Procrustes, and linear probes find structure
   that they themselves impose. PSQ-3$\alpha$ returned NO\_INTERFACE
   (69.14\%, gate $\ge 95\%$) under PCA/Procrustes intervention.

### 13.3 What was NOT established

The micro-world closure does **not** establish that native latent-space
mathematics does not exist. It establishes that this specific
observer $\times$ task $\times$ model combination cannot reveal it.
Possible reasons:

- Qwen3-0.6B (0.6B parameters) may be too small for compositional
  internal structure beyond answer-token routing.
- Two-fact recall may be too simple to require compositional operations.
- The intervention toolkit (block bypass, prefix transplant, row copy)
  may be too coarse to detect structure that exists.

### 13.4 Transferable deposits

1. **Five insights** (Phase 1): operational latent spaces are indexed by
   (actions, observations, horizon); information $\ne$ state (three
   gates: present $\to$ addressable $\to$ composable); the right null is
   the system's cheapest native mechanism; quotients must be earned by
   transport; absence requires a collision witness.

2. **Three-way clash design** (QPC-1): a decisive experimental design
   that separates compositional query actions from answer-code transplant
   by making source, target, and donor answers all distinct.

3. **Verbalizer null** (OSQ-1): any observational selectivity claim must
   pass the verbalizer null --- coarse-graining to answer tokens and
   checking whether selectivity survives.

4. **Instrument viability as a hard gate**: bypass viability, self-patch,
   and carrier integrity checks prevent off-manifold artifacts from
   masquerading as scientific results.

### 13.5 Pivot direction

Awaiting Codex postmortem (scratchpad/codex\_qpc1\_pivot.txt).
