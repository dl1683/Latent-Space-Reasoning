# GIRS representation obstruction V1

Date: 2026-09-07
Status: PROVED for the exact all-graphs obstruction; independent proof gate GO.
Review: [fresh adversarial report](REVIEW_GIRS_REPRESENTATION_OBSTRUCTION_V1.md).
No novelty, runtime, or mission-completion claim.

## Purpose and registered test

The mission needs conditions that distinguish matrices with economical native
representations from arbitrary dense matrices. A condition that every dense matrix
satisfies cannot by itself certify compression in a rational representation model.
This unit tests that failure mode for graph-induced cut-rank conditions.

H0: the all-graphs implication in Conjecture 2 of Chandrasekaran, Epperly, and
Govindarajan (arXiv:1911.05858v3) is valid: GIRS-c implies a DV representation
with each state dimension at most 2c.

H1: on a fixed graph all scalar-block matrices are GIRS-1, while the rational
parameter space of state-width-two DV representations has dimension less than n^2.
The literal all-graphs implication then fails.

Decision rule: require independent review of the exact source definitions,
quantifiers, cut inequality, parameter count, and rational-image argument.
Kill this proposed refutation if any fails. No numerical fit failure is evidence.
Holdout: rederive the count for heterogeneous widths and check the n=128 hypercube,
without treating a fit experiment as a proof. Do not extrapolate to 2D meshes.

## Source boundary and nearest work

1. Chandrasekaran, Epperly, Govindarajan,
   [GIRS, v3](https://arxiv.org/pdf/1911.05858v3), Definitions 1-2, equation (2.17),
   Conjecture 2. GIRS-c bounds each cut-block rank by c times the edge boundary.
   DV uses neighbor-coupled linear states with unique solutions. Conjecture 2
   asserts state widths <=2c for every GIRS-c pair. This is the exact target;
   the separate 2D-mesh Conjecture 4 is not targeted.
2. Govindarajan, Chandrasekaran, Dewilde,
   [Tree Quasi-separable Matrices](https://arxiv.org/html/2402.13381), section 4:
   minimal tree representations are controlled by tree-cut ranks. That result
   does not assert the all-graphs DV implication tested here. Tree theory is a
   positive baseline, not a new result of this project.
3. Chandrasekaran, Epperly, Govindarajan,
   [Minimal Rank Completions for Overlapping Blocks](https://doi.org/10.1016/j.laa.2021.06.011),
   treats the completion problem arising in cycle representations. It does not
   establish uniform state width for arbitrary graphs. Publisher text inspected;
   no assertion that its full proof was independently reproduced.

Searches for GIRS with hypercube, expander, counterexample, and Conjecture 2 found
no prior counterexample in the returned results. This is not evidence of novelty.
The algebraic dimension argument and hypercube spectral calculation below are
elementary controls. The potentially useful contribution is their exact application
to a representation criterion, if that application survives review.

## 1. A cut condition that admits every matrix

Let G be a simple undirected graph on n vertices, with one scalar coordinate per
vertex. Write delta(S) for the set of edges crossing S and its complement.
Suppose

    |delta(S)| >= h min(|S|, n-|S|)     for every S, h>0.

Every real or complex n-by-n matrix then satisfies

    rank A[S,S^c] <= min(|S|,n-|S|) <= |delta(S)|/h.

Thus every such matrix is GIRS-(1/h). The graph need not be the sparsity graph
of A: the source defines an arbitrary graph-partitioned pair (A,G). Conflating
those objects would change the conjecture.

Take G=Q_k, with vertices {0,1}^k and edges between bitstrings differing in one
coordinate. It has n=2^k vertices and degree k. Its Laplacian L has eigenvectors
chi_T(x)=(-1)^(sum_{j in T} x_j), eigenvalues 2|T|, for T a coordinate subset.
These characters are an orthogonal basis: distinct characters have zero inner
product by pairing bitstrings differing in a coordinate of their symmetric
difference. Hence the least nonzero eigenvalue is 2.

For f=1_S-(|S|/n)1, orthogonality to constants gives

    |delta(S)| = f^T L f >= 2 ||f||^2 = 2 |S|(n-|S|)/n
               >= min(|S|,n-|S|).

The first identity counts each crossing edge once, using the unnormalized graph
Laplacian. Empty/full cuts are zero. Therefore every matrix on Q_k is GIRS-1.
For a generic matrix, the half-cube cut has rank n/2 and exactly n/2 crossing
edges, so the minimal GIRS constant is exactly one, not merely bounded by one.

## 2. DV parameter count, including nonuniform state sizes

For state g_i in F^(r_i), input x_i in F, and output b_i in F, Definition 2 is

    g_i = V_i^T x_i + sum_{j adjacent i} W_ij g_j,
    b_i = D_i x_i + sum_{j adjacent i} U_ij g_j.

Here D_i is scalar, V_i has r_i entries, U_ij has r_j entries, and W_ij has
r_i r_j entries. Count each edge in both directions. The raw number of scalar
parameters is

    P(r_1,...,r_n) = n + sum_i r_i
                     + sum_(i,j directed edge) r_j
                     + sum_(i,j directed edge) r_i r_j.

Gauge equivalences can only lower image dimension, so ignoring them is safe for
this lower bound. If max_i r_i <= r and maximum degree is Delta,

    P <= n [1 + (Delta+1)r + Delta r^2].

More precisely at uniform width r, P=n(1+r)+2|E|(r+r^2).
All smaller widths embed into this same chart by padding states with zero
coordinates, padding generator blocks with zero, and leaving the unused state
equations g=0. Padding preserves unique solvability. Discrete choices of widths
therefore cannot evade the bound.

In stacked notation the represented matrix is

    A = D + U (I-W)^(-1) V^T.

The allowed parameter domain has det(I-W) != 0. Each matrix entry is a rational
function of the P parameters. If P<n^2, the n^2 coordinate functions are
algebraically dependent over Q: the rational function field in P indeterminates
has transcendence degree P. There is a nonzero polynomial F in n^2 matrix-entry
variables, with rational coefficients, vanishing on every represented matrix.
This proves the image lies in a proper algebraic subset. It also proves measure
zero over R and absence of an open set in matrix space. No numerical rank or
optimization assumption enters the argument.

In particular there exist real and rational matrices outside the image. Rational
existence follows because a nonzero rational polynomial cannot vanish on all of
Q^(n^2), by induction on its number of variables. Replacing F by F det(A) gives
invertible rational examples too. This is an existence proof, not a supplied
explicit numeric counterexample or efficient procedure for finding one.

## 3. Finite counterexample to the literal all-graphs conjecture

For Q_6, n=64, Delta=6, and the conjectured width r=2,

    P = 64 [1 + 7*2 + 6*4] = 2496 < 4096 = n^2.

Every 64-by-64 matrix is GIRS-1 on Q_6, yet all width-at-most-two DV matrices
lie in a proper algebraic subset. There exist invertible rational GIRS-1
matrices on this fixed graph without the asserted DV representation. This refutes
Conjecture 2 as written in arXiv:1911.05858v3; the independent gate accepted the
exact source interpretation and mathematical argument.

The general necessary condition for generic matrices on Q_k is

    1 + (k+1)r + k r^2 >= 2^k.

Consequently the required maximum DV state width is Omega(sqrt(n/log n)), even
though the GIRS constant remains one. This is a descriptor obstruction, not an
arithmetic-circuit lower bound. It does not deny DV representations with larger
states, or efficient representations of sparse matrices and their inverses.

Holdout arithmetic: Q_7 gives n=128, degree 7, and width-two P=128*45=5760,
versus n^2=16384. Nonuniform widths <=2 obey the same upper bound by section 2.

## 4. Cost accounting and research consequence

- Mathematical object: exact DV representation with a rational parameter-to-matrix
  map on the specified graph, scalar input/output block at each node, and arbitrary
  real or complex generator parameters (not just rational generator entries).
- Representation: at most P raw scalars for the proposed width; these cannot
  describe all n^2 independent matrix entries. The graph is fixed and uncharged
  here, which only strengthens the candidate representation's advantage.
- Discovery and encoding: unlimited work cannot overcome the image obstruction.
  Approximation and arbitrary digit-encoding decoders are outside this model.
- Output and application: no new matvec, matmul, inversion, or decoding speedup
  is claimed. Dense output still contains n^2 entries. Solving the implicit state
  system has its own cost, not established by counting representation scalars.
- Precision: an exact algebraic result; no quantitative finite-precision
  approximation lower bound, conditioning guarantee, or robust numerical witness.
- Fair baseline: generic dense storage for arbitrary matrices; existing TQS,
  SSS, CSS, and sparse-inverse routines for their actual restricted families.

So what: cut ranks normalized by a generous graph boundary can certify every
dense matrix. HEURISTIC research direction: investigate compatibility/realizability
conditions on local generators. Restricting the graph or matrix population could
also avoid the obstruction; this theorem does not prescribe a unique remedy.

The positive task remains to identify such conditions for useful matrix families,
derive composition/representation-growth laws, and build a fair implementation.
The hypercube obstruction supplies a falsifier for proposed generalizations; it
does not settle the mesh conjecture or complete any mission success tier.
