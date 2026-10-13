# Deep anatomy V1: the mechanisms underneath the walls

Date: 2026-09-18. Owner-directed excavation: not why the project failed
(verdicts), but what the failures reveal about the NATURE of arithmetic,
matrix multiplication, and structure. Six mechanism-laws, each stated,
mechanized (why it is true), mapped to the walls it explains, and labeled
(PROVED-in-repo / IMPORTED-classical / DERIVED-here-unreviewed /
CONJECTURE). Sources named for imports. This document is analysis, not a
claim of novelty; where a law is a known theorem wearing new clothes, it
says so.

---

## Law 1 — The Rigidity Law: there is no matrix logarithm, at three levels

**Statement.** Scalar arithmetic admits transport to cheaper operations
(logarithms: multiplication becomes addition) because the scalar
multiplicative structure is abelian and enormously symmetric. The matrix
algebra M_n admits no such transport, at three successive structural
levels, and each failure mode is a classical theorem.

**Mechanism.**
- *Group level.* (R+, x) is abelian and divisible: the log is one of
  continuum-many isomorphisms onto (R, +). For matrices, the invertible
  group GL_n has projective quotient PGL_n which is SIMPLE as an abstract
  group. A simple group has no nontrivial cheap quotient to transport
  into; abelianization is trivial. The logarithm's target must be abelian
  (addition is); PGL_n cannot reach any abelian structure without
  collapsing entirely. IMPORTED (classical group theory).
- *Algebra level.* Every algebra automorphism of M_n is inner
  (Skolem-Noether): a "change of coordinates" on the algebra is
  conjugation, and conjugation transports multiplication to itself. No
  coordinate change on M_n even perturbs the multiplication problem.
  IMPORTED.
- *Tensor level (the deepest).* A bilinear presentation of multiplication
  is a choice of tensor in the isomorphism class of the matmul tensor
  ⟨n,n,n⟩; re-presentation = tensor equivalence = ISOTOPY of the
  corresponding algebra. Albert's theorem: for associative unital
  algebras, isotopy coincides with isomorphism ([Garibaldi 2023](https://www.cambridge.org):
  "for associative algebras, isotopy is the same as isomorphism";
  Pchelintsev 2020 for the alternativity extension). So EVERY bilinear
  presentation of matrix multiplication is the structure tensor of an
  algebra isomorphic to M_n. The escape door labeled "find a better
  presentation" opens onto the same room. IMPORTED.

**What it explains.** Why every "re-coordinatize the matrices" scheme in
the record (alternative bases, encoding tricks, the DPS48 basis) at best
reduces constants: the tensor's orbit under all presentations is pinned
to M_n up to isomorphism, and the complexity of M_n's structure tensor
is what it is. The logarithm revolution has no matrix analogue because
the three doors it could come through (group quotient, algebra
automorphism, tensor isotopy) are each welded shut by a classical
theorem.

**Label.** IMPORTED theorems; DERIVED synthesis.

---

## Law 2 — The Burnside Envelope Law: descriptor growth is irreducibility

**Statement.** The observed growth of descriptors under composition is
not a defect of particular formats. Any class of cheaply-described
matrices that acts irreducibly on C^n has ALL of M_n as its
multiplicative envelope; some bounded number of compositions of cheap
elements must reach arbitrarily expensive (generic) matrices.

**Mechanism.** Burnside's theorem on matrix algebras: a subalgebra of
M_n(C) acting irreducibly is M_n itself (equivalently, Jacobson density:
the enveloping algebra of an irreducible set is everything). The linear
span of all products of elements of a class C is the algebra C generates.
If C contains the identity and anything with no common invariant
subspace, the generated algebra is M_n. Concretely: M_n is spanned by
products of at most a few elements of any such C; a generic matrix is a
linear combination of words in C's elements; representing it exactly in
any descriptor system that prices words by length forces length (or
coefficient count) to grow. The repo's growth theorems (tree widths,
CRC merged state, GIRS exact width) are format-specific faces of this:
each family acts irreducibly, so its multiplicative envelope is
everything, so exact compactness cannot survive composition.

**What it explains.** Why "growth under composition" appeared in EVERY
exact representation format studied, without exception. It also explains
the one non-failure: rank-one matrices act reducibly (common invariant
flag), which is exactly why the rank-one class composes freely (Unit B's
exact closure) — and why it carries no structure: reducible classes are
meagre and representable already as scalars times fixed operators.

**Label.** IMPORTED theorem (Burnside/Jacobson density); DERIVED
application (unreviewed as a formal descriptor-cost statement; the
mechanism is sound).

---

## Law 3 — The Closure Trichotomy: what exact closure costs

**Statement.** Linear representation classes exactly closed under
multiplication are subalgebras; over C, subalgebras decompose
(semisimple part ⊕ radical) into direct sums of matrix algebras and a
nilpotent part; therefore exact closure is available only in the
following three flavors:
  (a) subalgebra closure — which contains a rescaled copy of the original
      matrix-multiplication problem (the repo PROVED semisimple
      subalgebras inherit ω; the radical route dies in exact bilinear
      rank);
  (b) cheap-unit classes (banded, hierarchical, displacement, butterfly)
      — never closed; they degrade ADDITIVELY or per-level because their
      defining seminorm measures interaction-kernel mass that convolves
      under products;
  (c) rank-type classes — exactly closed (rank(AB) ≤ min(rank A, rank B))
      but rating the identity at maximal cost: structure-free.
No class achieves cheap unit + exact closure + structural richness
simultaneously; (a)+(Law 2) forbid it.

**Mechanism.** Wedderburn-Artin for the semisimple part; the repo's own
frontier for the direct-sum tightness; for (b), the kernel-convolution
picture: each structured class is a locality statement (band = index
locality, low-rank = correlation locality, hierarchy = scale locality),
products convolve localities, and a bound on kernel values is not a
bound on convolved kernel values (bandwidths add; H-matrix block ranks
double; quasiseparable orders add; displacement ranks add).

**The sweet spot.** Displacement rank is the unique classical point that
nearly escapes the trichotomy: it rates the identity at cost ≤ 2, it
degrades only additively under products (∇(XY) = ∇(X)Y + X∇(Y)-type
identities), and its generator calculus IS a working matrix-native
arithmetic (Toeplitz/Hankel/Cauchy superfast algorithms, 40 years of
them). This is not an accident the project kept colliding with: the
displacement formalism is the near-optimal point of the trichotomy
space, which is why every attempt to beat it (B389/B392, the corner
unit's bridges) required contract pinning or collapsed to it.

**What it explains.** The entire landscape of the repo's structured
families in one picture; also why the strongest specialist baselines
were always displacement-type objects.

**Label.** IMPORTED (Wedderburn-Artin), PROVED-in-repo (inheritance of
ω, radical refutation), DERIVED synthesis (trichotomy framing,
displacement sweet-spot reading — unreviewed).

---

## Law 4 — The Carry-Resolvent Identification: composition corrections are walks

**Statement.** The "correction" that makes chunked arithmetic work — the
carry in positional notation, the truncation error in hierarchical
arithmetic, the rounding error in floating point, the composition error
in any descriptor algebra — is one mathematical object: a propagated
walk (a resolvent). Scalar arithmetic survives because its walk is
BOUNDED; matrix chains escape when their walk is unbounded; the
difference is saturation versus conditioning.

**Mechanism.** A carry chain is the finite-state resolvent of the
digit-recurrence (c_{i+1} = floor((a_i + b_i + c_i)/β)): because digits
are bounded mod β, the walk saturates after one step in expectation and
carries have O(1) amortized length. Floating-point scalar arithmetic is
the certification of a bounded walk (relative error per op ε, chains
grow as ε√k under conditioning bounds — the entire content of
backward-error analysis, Wilkinson/Higham). The matrix side: the
truncation dichotomy PROVED this week — Theorem A (additive certified
budgets under uniform locality + non-expansivity) is precisely "the walk
is bounded"; Theorem B (resolvent powers force width Ω(k)) is precisely
"the walk escapes" — and the escape rate is the resolvent's spectral
behavior (1-a)^{-k}, i.e., the CONDITION NUMBER of the chain. The
information-locality length of C^k is ka/(1-a): composition spreads
dependence like a random walk with drift, and no truncation at fixed
width can track it.

**What it explains.** Why "floating point for matrices" (certified
approximate closure) is not a representation problem but a DYNAMICS
problem: the certificate exists exactly when the chain's conditioning is
controlled (Theorem A), and conditioning is a property of the operators
being composed, not of the descriptor format. It also gives the honest
statement of what scalar arithmetic has that matrix arithmetic lacks:
not exactness, not compactness — BOUNDEDNESS OF THE MIXING WALK.
Positional notation is a clever compression, but the carry is why it
COMPOSES: the compression's error channel is a saturated walk. Any
matrix-native arithmetic must purchase the same property, and Theorem B
says the purchase price for spreading families is linear width.

**Label.** DERIVED synthesis (unreviewed); components PROVED-in-repo
(Theorems A/B) and IMPORTED (backward error analysis).

---

## Law 5 — The Noncommutativity Premium: ω is the price of noncommutativity

**Statement.** The rank of an algebra's multiplication tensor, per
dimension, is a quantitative measure of its noncommutativity. Commutative
semisimple (entrywise C^n): rank n on dimension n — linear. Convolution
algebras C[x]/(x^n): rank 2n-1 — linear. Matrix algebras M_n: rank n^ω on
dimension n² — the premium exponent ω/2 ≈ 1.186 per dimension. The
mission "make matrix arithmetic as cheap as scalar arithmetic" is
exactly the demand to remove the noncommutativity premium, and every
bilinear wall in the record is the premium refusing to vanish.

**Mechanism.** Tensor rank of the structure tensor is the bilinear
complexity of the algebra's multiplication. Commutative structures
diagonalize (entrywise = n independent scalar products); any
noncommutativity forces off-diagonal mixing whose tensor does not split
into independent triangles. Upper direction: fast algorithms exist (rank
n^ω < n³) but only through global cancellation exploiting associativity
laser-style. Lower direction: the conjecture that no presentation
reaches dimension-linear rank for M_n is open in general (this is
essentially ω > 2, open) — hence the label.

**What it explains.** Why the project's positive components (subset
convolution, box products, rank-one dominance) were all commutative or
abelian phenomena: every place the premium was dodged, the structure had
quietly become commutative. The premium is also the sharpest way to say
what the separator obstructions were probing: slicing barriers are
statements that the premium's mass cannot be concentrated into
exploitable pieces.

**Label.** DERIVED framing; the premium's nonvanishing at the bilinear
level is CONJECTURE (equivalent in strength to open lower-bound
questions); the per-dimension calculations are elementary.

---

## Law 6 — The Fixed-Point Property: matrix multiplication is the closure of arithmetic on matrices

**Statement.** Take any class of matrix descriptors closed under a
native product, linear or not; quotient by the descriptor equivalence;
the result is an algebra acting on C^n; if it is rich enough to be
interesting it contains M_k for growing k (Laws 2-3); and the arithmetic
of THAT algebra is matrix multiplication again. The mission's target —
arithmetic whose atoms are matrices — is a fixed point: the closure of
any such arithmetic, stripped of its notation, is the very operation we
started from. Scalar arithmetic is NOT a fixed point in this sense: the
closure of digit-chunk arithmetic is the integers, whose multiplication
is a genuinely different (commutative, premium-free) object than
chunk-composition.

**Mechanism.** Laws 1+2+3 composed: rigidity kills re-presentation,
Burnside forces the envelope to be everything, Wedderburn names the
everything. The one escape found in the whole record — nonlinear closure
(the symmetric-power family, NONLINEAR_COMPOSITION_SCOPE_V1) — is an
escape into commutative substructure (symmetric powers abelianize),
i.e., out of the fixed point through the only door Law 5 leaves open.

**What it explains.** The peculiar self-similarity of every failure:
each wall, examined closely, was M_n again. This is also the honest
statement of why the mission felt always-almost-there: a fixed point
greets every approach from the inside with the face of the original
problem.

**Label.** DERIVED synthesis (unreviewed; a framing, not a theorem —
its formalization would be the Floor-tier statement).

---

## What the anatomy illuminates going forward

1. The six laws are six faces of one object: M_n is simple, rigid,
   noncommutative, self-similar, and mixing — and arithmetic design is
   the attempt to buy one of those properties away. Each purchase has a
   price list: conjugation-invariance (constants only), reducibility
   (structure loss), truncation (walks), commutativity (the premium).
2. The genuinely unvisited cell the laws leave open is the BACKWARD
   direction: every law guards composition (forward). Decomposition —
   factoring matrices, simplifying composed descriptors, inverting the
   walk — is where the envelope argument runs in reverse and no wall in
   this record stands. Factoring is the inverse-resolvent problem; the
   carry identification (Law 4) suggests its arithmetic exists exactly
   when the forward walk is bounded, i.e., the same stability boundary —
   but nothing here proves nonexistence of decomposable-native
   arithmetic.
3. The displacement sweet spot (Law 3) is the strongest existing
   "matrix-native arithmetic" and its optimality-in-trichotomy reading
   suggests the sharpest well-posed theory question available: prove or
   refute that displacement-type formalisms are optimal points of the
   trichotomy space. A no-go would be the Floor theorem completing the
   map; a construction beyond it would be genuinely new arithmetic.

This document is the drill the owner ordered: the failure map read as
anatomy. No verdicts; mechanisms. Review before any PROVED label;
syntheses labeled DERIVED until then.
