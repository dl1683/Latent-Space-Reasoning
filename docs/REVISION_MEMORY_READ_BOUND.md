# A one-bit readout can require corpus-sized revision memory

28 November 2026. Classification: mathematical foundation for revision and edge economics. Base `c6f71dc0`. This is a scoped information bound and a small arithmetic verification, not a learned result or a proposed replacement for the frozen reader gate. No model, held packet, grader or execution authority touched.

## Decision

Do not budget memory by the size of today's answer, or by how many distinct output labels the current operation has. Even one fixed binary aggregate can require almost all source information to support arbitrary later withdrawals. This extends the exact fixed-linear-state [residual reopening result](DUAL_READOUT_RESIDUAL_COST.md) to arbitrary finite nonlinear states and bounded-error reads in one stylized setting, and complements the exact cap/ID-deletion example in [revision-conditioned predictive state](REVISION_CONDITIONED_PREDICTIVE_STATE.md).

This is a communication-complexity-style application of standard Fourier analysis on the Boolean cube and the degree bound for decision trees (see [O'Donnell, Analysis of Boolean Functions](https://arxiv.org/abs/2105.10386)). The ingredients are classical. No priority or general lower bound on natural-language reasoning is claimed.

## Setup

Let the independently addressable source facts X be n independent uniform bits. Encode the opening corpus into any persistent state Z=f(X) with at most K distinct states (K <= 2^B for B retained bits). Before the withdrawal is known, arbitrary computation is allowed. Count every source-dependent retained bit accessible to the reader, including cached outputs, learned parameters, metadata, historical logs, and worker messages; fixed source-independent program code is free.

After publication, independently sample a public mask Q in {0,1}^n from a declared distribution pi. Its ones identify the sources that survive an ID-only withdrawal. The fixed task is to report the parity of the surviving facts, chi_Q(X)=(-1)^(Q dot X), dot product modulo two. The opening aggregate is just one bit. The withdrawal payload does not contain old values, and there is no other X-dependent side channel. The reader may open at most r individual bits of the old source archive, adaptively, after seeing Z,Q. All such opens, including ones performed during withdrawal maintenance before the readout, count. This is a worst-case cap on every trace, not average reads. The readout must predict a binary label; an abstention may be randomized to a guess for this bound but cannot count as a correct grounded answer without further information.

Let p_r = max over t of Pr_pi[d_H(Q,t)<=r], the largest probability mass of a Hamming ball of radius r in the mask distribution. Then

    Pr[correct] <= min(1, 1/2 + (1/2)*sqrt(K*p_r)).

For uniform masks, p_r=V(n,r)/2^n where V(n,r)=sum_{j=0}^r binomial(n,j). In particular, to achieve advantage delta>0 above chance, one needs

    B >= n - log2 V(n,r) + log2(4*delta^2).

No restriction to linear encoders, finite-depth latent computation, or nonadaptive source probes is imposed. Fixed finite precision and the stated archive interface are essential.

## Proof

For a nonempty state cell C=f^{-1}(z), X conditional on Z=z is uniform on C. Define b_C(t)=E_{X in C}[chi_t(X)]. Orthogonality of the Boolean characters yields

    sum_t b_C(t)^2 = 2^n / |C|.

For fixed z and q the reader is a depth-at-most-r decision tree on source bits. Extend it arbitrarily to all source inputs outside C; branches that are infeasible in C can terminate with an arbitrary label without increasing the depth. Its sign-valued output g_{z,q}(x) has Fourier expansion sum_s a_s chi_s(x) supported on |s|<=r: each root-to-leaf indicator is a product of at most r single-bit indicators. Parseval gives sum_s a_s^2=1. Consequently,

    E[chi_q(X)*g_{z,q}(X) | Z=z]
      = sum_{|s|<=r} a_s b_C(q XOR s)
      <= sqrt(sum_{|s|<=r} b_C(q XOR s)^2).

The right hand side is independent of which adaptive tree was selected. Averaging over q and applying Jensen/Cauchy-Schwarz, the conditional average correlation is at most

    sqrt(sum_t b_C(t)^2 * Pr_pi[d_H(Q,t)<=r])
      <= sqrt(p_r * 2^n / |C|).

Average over state cells with weights |C|/2^n and apply Jensen again. The mean of the squared cell bound is p_r times the number of nonempty cells, at most K*p_r. A binary correct prediction has correlation +1, an incorrect prediction -1, so success is (1+correlation)/2. This proves the claim. Independent random coins can be fixed and the result averaged, provided every realized encoder obeys the same storage cap and every realized reader obeys the probe cap.

For example, n=1,024, r=32, and 90% accuracy on uniform withdrawals require at least 822 retained bits. With B=256 the upper bound is 1/2 plus approximately 2^(-284.16). This illustrates a hard independent-fact/mask family, not a prediction of storage for 1,024 real documents.

A known fixed surviving mask is the opposite limit: one precomputed bit suffices without reopening. The concentration p_r term distinguishes predictable edit families from diverse ones. If an edit event gives the deleted values as well as IDs, one can instead maintain the aggregate by XORing those values; the bound does not apply to that richer event interface. Count that payload or the storage from which it is generated. Likewise, if the archive answers arbitrary aggregate queries rather than returning individual bits, the read model changes. One call is not one bit.

## Exact small check (standard-library Python)

The following exhausts all 256 maps of eight possible 3-bit source worlds into two states. An independent dynamic program maximizes correct predictions over every adaptive binary probe tree of depth <= r for each cell and each mask, rather than assuming a linear map or choosing a particular decoder. It checks the squared inequality in integer arithmetic, including uniform, fixed nonzero and skewed mask priors.

```python
from functools import lru_cache
from math import comb, log2
n = 3
N = 1 << n
ones = [sum(1 << x for x in range(N) if (x & q).bit_count() % 2)
        for q in range(N)]
bit = [sum(1 << x for x in range(N) if x >> i & 1) for i in range(n)]

@lru_cache(None)
def opt(C, q, r):
    if not C:
        return 0
    p = (C & ones[q]).bit_count()
    best = max(p, C.bit_count() - p)
    if r:
        best = max(best, max(opt(C & b, q, r-1) + opt(C & ~b, q, r-1)
                             for b in bit))
    return best

weights = [[1]*N, [0,1,0,0,0,0,0,0], [8,4,2,1,1,1,1,1]]
checks = 0
maxima = []
for w in weights:
    W = sum(w)
    m = []
    for r in range(n+1):
        M = max(sum(w[q] for q in range(N) if (q ^ t).bit_count() <= r)
                for t in range(N))
        best = 0
        for C in range(1 << N):
            cells = [s for s in [C, ((1 << N)-1) ^ C] if s]
            K = len(cells)
            S = sum(w[q]*sum(opt(s,q,r) for s in cells) for q in range(N))
            assert (2*S-N*W)**2 <= N*N*W*K*M
            best = max(best, S)
            checks += 1
        m.append(f'{best}/{N*W}')
    maxima.append(m)
print(checks, maxima)
print(1024-log2(sum(comb(1024,i) for i in range(33)))+log2(.64))
```

Executed: 3,072 exact bound checks, no violations. Maximum success across encoders by r=0,1,2,3: uniform 45/64,64/64,64/64,64/64; fixed 8/8 throughout; skewed 127/152,152/152,152/152,152/152. The large arithmetic example gives 821.6782847608256 before rounding up. The small check is deliberately not a tightness proof or empirical estimate; some bounds are vacuous.

## Consequence for the full-world design

An edge system should learn which future source/rule interventions share predictive structure, not assume every compression can survive arbitrary changes. Preserve revision-addressable residuals or reopen the minimum necessary source evidence, and keep their provenance. In a future separately registered comparison, report old-value availability, source-dependent backing bytes, and bytes reopened **during maintenance as well as at query time**. Stratify results by independently specified edit families, including unexpected ones, rather than averaging successful predictable updates with costly novel repairs. Compare ordinary incremental computation using the same delta payloads; it may be optimal. A learned worker earns its keep only if it identifies useful structured distinctions and keeps total cost below that control at matched quality. This theorem is a stress boundary, not a reason to build parity workers, estimate semantic edit entropy from an unvalidated embedding, or run another codec/cache experiment.

Next actual experiment remains reader qualification followed by learned, query-blind semantic operation acquisition and causal use. The [registered sequence](LATENT_WORLD_SYSTEM_DESIGN.md#7-research-sequence-and-decision-gates) stays unchanged. No private material was accessed; no new gate or authorization implied.
