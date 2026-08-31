# Predictive-Fiber Action Algebra

The canonical algebraic artifact for the behavioral algebra discovered in
Qwen3-0.6B across 17 audited experiments (Phase 2, 2026-08-31).

## 1. Core object

$$
\mathfrak{A} = (X, W, Q, G, \tau, \gamma, \pi, \mathcal{A}, S^W)
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
JSD($S$, $S^2$) mean = 0.070 (range 0.025-0.140). The retraction is genuine.

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

### Partial

**L4a: World-restatement descent.** $S^W_w$ descent to $G$.
*Status:* **Partial.** 11/12 registered (91.7%), 15/15 held-out (100%). The one
failure is precisely where a greedy fiber spans different worlds — $S^W_w$ maps
members to different targets because it uses the hidden world, not the shared
greedy signature.

### Open questions

**O1: Existence of representative-independent $S^G_g$.**
Does a restatement defined from the observable greedy signature $g$ alone (without
the hidden world $w$) exist and retain approximate idempotence and descent?
*Status:* **Open.** The current $S^W_w$ is world-conditioned. A true $S^G_g$ from
greedy signatures has not been constructed or tested.

**O2: Correction descent to $G$.**
Does correction descend to the greedy quotient? Correction reaches its asserted
target token in only 29/48 registered and 28/48 held-out cases.
*Status:* **Open.**

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
3. A world-conditioned idempotent restatement ($S^W_w$) with near-perfect descent.
4. A demonstrated non-naturality: $S^W$ does not commute with correction under
   the tested append action — prediction is presentation-path dependent.

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
