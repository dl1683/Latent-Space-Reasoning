# Native latent mathematics — living axioms

Status: Round 1 revised, 2026-08-27. This file contains only the active
relational foundation. Dialogue and rejected formulations stay in `dialogue/`.

## Primitives

- \(X\): latent states, after observational quotient.
- \(C\): admissible probes/contexts.
- \(N_c(x)\subseteq X\): states accepted as substitutes for \(x\) under \(c\).
- \(c\otimes d\): conjunction of probes when admitted.
- \(G\): declared exact presentation changes, acting on states and probes.

A raw vector array is a presentation, not a latent space. No origin, addition,
scalar multiplication, inner product, metric, or dimension is primitive.

## Axioms and status

### L1. Self-substitution — adopted

\[
x\in N_c(x)\qquad(x\in X,c\in C).
\]

### L2. Finite conjunction — adopted for the completed probe family

Whenever \(c\otimes d\) is admitted,

\[
N_{c\otimes d}(x)=N_c(x)\cap N_d(x).
\]

A finite measured probe table need not already contain its conjunctions.

### L3. Local refinement — conditional

A **topological latent system** has a completed probe family \(C^*\) satisfying

\[
y\in N_c(x)\Longrightarrow
\exists d\in C^*\;N_d(y)\subseteq N_c(x).
\]

This is not assumed of the first finite measurement. Its empirical analogue is
refinement-defect rate.

### L4. Observational separation — adopted by quotient

\[
\bigl[\forall c:\ y\in N_c(x)\ \text{and}\ x\in N_c(y)\bigr]
\Longrightarrow x=y.
\]

### L5. Presentation covariance — adopted for declared exact maps

For \(g\in G\),

\[
g[N_c(x)]=N_{gc}(gx),
\qquad g(c\otimes d)=gc\otimes gd.
\]

Which architecture changes belong to \(G\) is a separate identification
problem. Independently trained models are not presumed isomorphic.

### Contextual non-collapse — retired as an axiom

Existence of one reversal is noise-sensitive. It is replaced by context rank.

## Definitions

### Closeness profile

\[
\Phi_x(y)=\{c:y\in N_c(x)\},
\qquad
y\succeq_x z\iff\Phi_x(z)\subseteq\Phi_x(y).
\]

This relation may be directed, partial, and context-sensitive.

### Finite context rank

A quasi-metric is \(d:X^2\to[0,\infty)\) with \(d(x,x)=0\), positive
off-diagonal entries, and the triangle inequality; symmetry is not required.
For finite \(X,C\),

\[
\kappa(X,C,N)=\min k
\]

such that quasi-metrics \(d_1,\ldots,d_k\), a map \(i:C\to[k]\), and
anchor-dependent radii \(\rho_c:X\to(0,\infty]\) satisfy

\[
N_c(x)=\{y:d_{i(c)}(x,y)<\rho_c(x)\}.
\]

Context rank measures incompatible ordinal cuts, not calibrated magnitude or
geodesic structure.

For a graded probe \(r_c(x,y)\), enlarge the neighborhood family by every strict
sublevel cut and require all cuts from the same \(c\) to share \(i(c)\). The
result is **graded context rank**; it introduces no arbitrary single tolerance.

### Cross-realization transportability

For independently trained systems, align states by an external identifier and
measure agreement of directed signs, context-incompatibility edges, and neighbor
rankings. This grades probe transportability; it is not an identity axiom.

## Theorems and status

### T1. Induced topology — proved, conditional on L1–L4

The sets \(N_c(x)\) form a basis for a \(T_0\) topology. Declared presentation
maps are homeomorphisms under L5.

Proof: local refinement supplies a basic neighborhood at each point of a basic
intersection; conjunction refines the two supplied neighborhoods; separation
gives \(T_0\). Full proof: `dialogue/001.md`.

### T2. Single-quasi-metric representation — proved for finite systems

Assume L1 and finite \(X,C\). Then \(\kappa=1\) iff, for every \(x\),

\[
\forall c,d:\quad
N_c(x)\subseteq N_d(x)\ \text{or}\ N_d(x)\subseteq N_c(x).
\]

Necessity follows because balls with one center are radius-ordered. For
sufficiency, order states by their neighborhood-membership profiles at each
anchor, assign distinct profile classes values in \([1,2)\), set the diagonal to
zero, and cut at suitable radii. All nontrivial two-edge paths have length at
least 2, so the triangle inequality holds. Full proof: `dialogue/001.md`.

### T3. Context-rank coloring — proved for finite systems

Join contexts \(c,d\) when their neighborhoods are incomparable at some anchor.
Then

\[
\kappa(X,C,N)=\chi(G_C).
\]

Each color class satisfies T2; contexts represented by one quasi-metric cannot
share an incompatibility edge.

For graded probes, join two probes when any pair of their sublevel cuts is
incomparable, and require all cuts from one probe to share a color. The same
proof gives graded context rank as the chromatic number of this probe graph.

## Open problems

1. Find a non-degenerate invariant that constrains magnitude or paths, not only
   ordinal cuts; T2 makes the triangle inequality nearly free on finite data.
2. Characterize the global-radius version \(\rho_c(x)=\rho_c\), where context
   scales are comparable across anchors.
3. Give infinite-system quasi-metrization conditions without assuming first
   countability by fiat.
4. Decide when a probe measures semantics rather than decoder behavior;
   cross-realization agreement is evidence, not a solution.
5. Measure whether L3 holds approximately before treating the empirical probe
   system as topological.
6. Add evidence-update and typed composition only after closeness survives its
   first falsifier.
