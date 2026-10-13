# Deep anatomy V2: the exploration

Date: 2026-09-18. Owner-directed maximum-depth continuation of
[DEEP_ANATOMY_V1.md](DEEP_ANATOMY_V1.md): parallel areas, the reach of
the walk law, a counting-theorem candidate, and the constructive space
that survives everything. Labels: IMPORTED (known, named), DERIVED
(unreviewed synthesis), CONJECTURE (open, sharpened). No verdicts.

## 1. The walk law is one law wearing five costumes

Law 4 (composition corrections are propagated walks) is not specific to
descriptor arithmetic. The same law, already proved or practiced under
other names:

- PHYSICS: Lieb-Robinson bounds — local dynamics spread correlations
  inside a linear light cone; operators remain quasi-local outside it.
  Tensor-network simulation cost tracks entanglement growth: area-law /
  logarithmic (bounded walk) is cheap; volume-law (spreading) forces bond
  dimension linear in time — the exact Omega(k) of the truncation
  dichotomy's Theorem B. IMPORTED (Lieb-Robinson; entanglement-growth
  literature). The dichotomy's resolvent locality length ka/(1-a) is a
  light-cone velocity.
- NUMERICAL LINEAR ALGEBRA: the direct-versus-iterative split is the
  operator-side versus state-side resolution of the same walk. Direct
  methods compose operator descriptors (LU factors) — viable when walks
  are bounded (elliptic; H-LU exists). Krylov methods never materialize
  A^k; they walk the vector and compress the trajectory (Lanczos
  three-term recurrences) — the prescribed escape when walks spread.
  IMPORTED-practice, DERIVED-framing.
- AUTOMATIC DIFFERENTIATION: forward mode composes Jacobians
  (operator-side) and loses on spreading problems; reverse mode walks
  states (vector-Jacobian products, never composing operators) — the
  anatomy prescribes the shape reverse-mode AD already has. Backprop as
  the state-side escape is a retrodiction of Law 4. DERIVED-framing of
  IMPORTED practice.
- H-MATRIX PRACTICE: elliptic operators (resolvents with instantaneous
  tails, zero effective velocity) compose under compression; hyperbolic /
  transport (finite light-cone velocity) degrade — the known empirical
  split, predicted by the walk axis a priori. IMPORTED-practice,
  DERIVED-prediction.
- ELEMENTARY ARITHMETIC: the positional carry is the walk, saturated by
  the modulus; floating point is its certification (Law 4, V1).

One law, five fields. The meter's walk axis is the first instrument that
measures it uniformly.

## 2. The counting-theorem candidate (quantitative Burnside)

CONJECTURE (proof strategy sketched): the exact descriptor growth of a
matrix family under composition equals, up to constants, the WORD GROWTH
of the algebra the family generates — the number of distinct products of
length <= k.

- Lower bound: an exact descriptor system of per-node size B must
  distinguish the distinct reachable products; over finite fields a
  counting argument gives B >= min(k log_q m, n^2) for generic
  m-generator families. Over arbitrary fields: distinct words remain
  independent modulo the polynomial identities of M_n (n >= 2) — the
  essential-monomial theory (Drensky) gives exponential word growth for
  every noncommutative family. IMPORTED ingredients (PI theory),
  DERIVED application.
- Upper bound: polynomial word growth admits compact descriptors.
  Commutative families: distinct products = multisets, binomial(k+m-1, m-1).
  Finite groups: saturation (the group is finite; table arithmetic is a
  bounded exact descriptor system). Class-c nilpotent: degree-c growth.
  The Gromov-flavored boundary: polynomial word growth <-> near-abelian
  <-> cheap exact arithmetic. IMPORTED (growth of algebras/groups),
  DERIVED bridge.
- Smoke (n=8, 600 sampled words): distinct-product counts at k=12 —
  commutative family 57 (polynomial), permutation family 230 heading to
  saturation at |group| (bounded), generic family 599/600 (essentially
  every word distinct — exponential). Recorded in the ledger row.

If proved, every format-specific growth theorem in this repository
(trees, CRC, GIRS) becomes a corollary, and "search for slow-mixing
families" becomes a principled design rule with a computable invariant.

## 3. Parallel civilizations (each already living by the laws)

1. MODEL REDUCTION: balanced truncation carries an additive certified
   error bound (sum of neglected Hankel singular values) — a Theorem-A
   certificate in engineering practice for 50 years; Hankel singular
   values ARE the walk spectrum (past-future coupling norms). IMPORTED;
   the anatomy identifies them as the walk axis measurement.
2. FILTER BANKS / WAVELETS: self-similar bases (dilation equations) exist
   and telescope — in the COMMUTATIVE sector (convolution). The DPS48
   race failure is the noncommutative shadow of the reason wavelets
   exist: self-similarity is constructible where the premium vanishes.
   DERIVED-framing.
3. TENSOR NETWORKS: bond-dimension management under real-time evolution
   = the truncation dichotomy in physics clothing. IMPORTED.
4. QUERY OPTIMIZATION: intermediate-relation growth under joins is
   descriptor growth; acyclic (bounded-treewidth) queries are the
   exactly-closed class (Yannakakis); the repo's GIRS/tree widths are the
   same quantity. IMPORTED-parallel, DERIVED identification.
5. DEEP LEARNING: learned weight matrices are generic (premium-full —
   the meter's verdict), and training walks STATES (backprop) rather
   than composing operators; LoRA chains exploit rank closure. The
   field's architecture is a retrodiction of the laws. DERIVED-framing
   of IMPORTED practice.

## 4. The constructive space that survives everything (four prongs)

(i) STATE-SIDE ARITHMETIC: number systems for vectors under dynamics —
    quantics/tensor-train compressed states with certified rank
    management; the walk certificate decides feasibility. (ii)
    ABELIANIZATION / WEDDERBURN ARITHMETIC: decompose the workload's
    generated algebra into simple components; arithmetic cost is
    max_i k_i^omega — small components mean cheap arithmetic. The
    repository's commutant engine (structured_multiply.py) is exactly
    this instrument, historically aimed at physics fragmentation
    instead of arithmetic design. (iii) WALK-CERTIFIED TRUNCATION
    CHAINS: displacement/H-matrix/balanced-truncation arithmetic with
    a priori chain certificates (Theorems A/B, Hankel bounds).
    (iv) BACKWARD ARITHMETIC: factorization-native computation
    (Schur, Levinson-Durbin, GKO, H-LU) — the direction with no walls;
    extension to walk-bounded families beyond displacement structure is
    open constructive territory.

## 5. The honest omega statement

The premium framing restates, does not lower-bound: how much of the
noncommutativity premium is payable in the border (approximate
presentation) sector is essentially the field's open question. The
anatomy's contribution is the map, not a new barrier. CONJECTURE-level
throughout.

## Registered next units

1. premium_meter_retrodiction_v1 (running): complete the quantum
   chain-level and PDE walk-axis retrodictions.
2. counting_theorem_v1 (to register): quantitative Burnside — word
   growth = descriptor growth; Drensky lower bound, Gromov-type
   boundary, format-independence.
3. wedderburn_arithmetic_probe (to register): repurpose the commutant
   engine as the abelianization instrument; test the small-blocks
   criterion on a real operator family.
