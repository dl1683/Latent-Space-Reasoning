# FBA Bridge: Product Factorization in Recurrent Belief States

Math-first definitions for the FBA-0 experiment (distance 1 control).

## Definition 1 (Response-law equivalence)

Let $\mathcal{W}$ be a POMDP with state space $S$, action space $A$,
observation function $O$, and transition function $T$. An **action sequence**
is $a = (a_1, \ldots, a_T) \in A^T$.

Two action sequences $a, a'$ are **response-law equivalent**, written
$a \sim_R a'$, iff for every initial state $s_0 \in S$ and every observation
history $(o_1, \ldots, o_{T+1})$:

$$P(o_{T+1} \mid s_0, a) = P(o_{T+1} \mid s_0, a')$$

The equivalence classes $[a]_R$ partition $A^T$ into **response classes**.

## Definition 2 (Recurrent belief state)

A **recurrent belief encoder** is a function $f: (O \times A)^* \times O \to \mathbb{R}^d$
that processes observation-action sequences recurrently to produce a belief
vector $z = f(o_1, a_1, \ldots, a_T, o_{T+1}) \in \mathbb{R}^d$.

A **decoder** is $h: \mathbb{R}^d \to \Delta(S)$ mapping belief to a distribution
over terminal states.

## Definition 3 (Product factorization)

A recurrent belief encoder $f$ with output $z \in \mathbb{R}^d$ exhibits
**product factorization** with respect to a state decomposition $S = L \times F$
(where $L$ is a "place" factor and $F$ is a "fiber" factor) if there exist
projections $\pi_L: \mathbb{R}^d \to \mathbb{R}^{d_L}$ and
$\pi_F: \mathbb{R}^d \to \mathbb{R}^{d_F}$ with $d_L + d_F = d$ such that:

(i) **Slot separation**: $z = [\pi_L(z), \pi_F(z)]$ (concatenation).

(ii) **Factor sufficiency**: there exist decoders $h_L: \mathbb{R}^{d_L} \to \Delta(L)$
and $h_F: \mathbb{R}^{d_F} \to \Delta(F)$ such that the joint decoder
$h(z) \approx h_L(\pi_L(z)) \otimes h_F(\pi_F(z))$.

## Definition 4 (Branch interchange test)

Given a product-factorized encoder with episodes $\{(z_i, \ell_i, f_i)\}_{i=1}^N$
where $\ell_i \in L$ and $f_i \in F$ are the true factor values, the
**branch interchange test** constructs cross-episode hybrids:

For episodes $i, j$ with $\ell_i \neq \ell_j$ and $f_i \neq f_j$:

$$z_{ij} = [\pi_L(z_i), \pi_F(z_j)]$$

**Interchange success** requires:

$$h(z_{ij}) \approx (\ell_i, f_j)$$

The cross-accuracy is the fraction of hybrids correctly decoded. The test
passes if cross-accuracy exceeds the historyless null (the accuracy achievable
from a single observation without recurrence).

## Proposition 1 (Interchange implies decoder-compatible branch separation)

If a recurrent encoder consistently passes the branch interchange test
(cross-accuracy significantly above historyless null, with Bonferroni-corrected
CI), then the encoder's branches carry **decoder-compatible** factor information:
cross-episode hybrids decode to the expected (location, state) combination
at rates exceeding the registered historyless null (the accuracy achievable
from a single observation without recurrence).

This is an empirical decoder-compatibility statement, not a claim of exclusive
independent factor encoding. Branches may carry redundant or entangled
information; the test shows the decoder can extract the correct factors from
cross-episode hybrids, not that each branch exclusively encodes one factor.

## Same-factor preservation controls (wrong-channel tests)

The interchange test includes **same-factor preservation controls** to
check decoder-compatibility in each branch direction:

- **Same-location pairs** ($\ell_i = \ell_j$): when both episodes share
  the same location, swapping the place branch should preserve location
  decoding. This is a **same-factor preservation** check: it tests that
  the designated branch carries decoder-compatible location information,
  not that it exclusively determines location (redundant encoding across
  branches could also pass).

- **Same-state pairs** ($f_i = f_j$): analogous same-factor preservation
  check for the fiber branch.

## Connection to FBA-0

FBA-0 instantiates this framework with $L = \mathbb{Z}/8$ (locations),
$F = \mathbb{Z}/4$ (states), a 16/16 independently-updated recurrent
architecture as the encoder, and the six-way comparison (flat, matched, 
asymmetric split, modular, flat-bottleneck) as controls. The kill gates
(K4, K6, K7a, K7b, paired effects) operationalize the branch interchange
test (Definition 4) and its same-factor preservation controls.

FBA-0 is a **control experiment** (distance 1): it tests whether an
engineered product architecture can support decoder-compatible branch
interchange in a synthetic POMDP. It does NOT constitute native
latent-space mathematics — the central artifact of this project is the
discovery of intervenable structure in real model latent spaces, for
which FBA-0 provides empirical grounding.
