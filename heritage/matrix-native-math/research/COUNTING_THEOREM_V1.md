# The counting theorem V1: descriptor growth is word growth

Date: 2026-09-18
Status: PROVED / SCOPED — two independent gates passed: adversarial
review SCOPED_PROOF_GO with four major repairs applied (reviewer
agent_8bd5ec23), then a fresh delta review DELTA_GO with exhaustive
independent numerics (reviewer agent_ed008dbd; 64/64 and 81/81 words
exact; three cosmetic polish items applied post-gate without touching
mathematical content). Scope: exact chains, bit-counted descriptors,
per-k finite-field reading; the cliff corollary for invertible families
(char 0) only; singular/semigroup intermediate growth open. No speedup,
no mission tier; the contribution is the bridge (descriptor floors =
word counts), the numeral separation construction, and the tight
Theta(k log m) LIST characterization. Lineage: [deep anatomy V1](DEEP_ANATOMY_V1.md) Law 2 (Burnside
envelope), [V2 exploration](DEEP_ANATOMY_V2_EXPLORATION.md) section 2.

## 0. Statement

An EXACT DESCRIPTOR CHAIN for a family F = {A_1, ..., A_m} in M_n(K) is a
rule assigning to each operator a finite descriptor such that (i) each
family member has a descriptor, (ii) there is a composition rule
producing a descriptor of the composed product from the two operand
descriptors, and (iii) exact: the descriptor determines the operator.
D(k) = the worst-case descriptor size after k compositions. W(k) = the
number of DISTINCT products of length exactly k over the m generators.

MAIN CLAIM (bit model; per-k reading). Over infinite or exact fields
(and over finite fields per k, see Theorem 2):

    log_2 W(k) - O(1)  <=  D(k)        (lower: labels must distinguish)
    D(k) = k log_2 m + O(1) is ACHIEVED by the LIST format for generic
    families (store the factor sequence): the bound is TIGHT and LIST is
    optimal — a Theta(k log m) characterization, not a one-sided bound.
    On the cheap side, D(k) = O(m log k) (count vectors) for
    simultaneously diagonalizable generic families, O(log |G|) (group
    registry) inside finite groups, and bounded for any family read in
    fixed coordinates of its generated algebra, at the price of
    Theta(k) bit growth per coordinate over exact Z when entries grow
    (e.g. powers of 2).

The classification of W(k) (invertible families, char 0):

- (U1) Simultaneously diagonalizable with generic spectra: W(k) =
  binom(k + m - 1, m - 1) up to degeneracies (an upper bound in general;
  equality needs genericity). Descriptors: count vectors, O(m log k).
- (U2) Families inside a finite group G < GL_n: W(k) <= |G|; registry
  arithmetic is exact closure forever.
- (U3) Any family: fixed-basis coordinates in the generated algebra A
  compose via structure constants; descriptor COUNT is bounded by
  dim A (Burnside's envelope, quantitative), while bit size grows with
  entry growth over exact rings.
- (L1) SEPARATION LEMMA (below; repaired per review): explicit
  invertible families with all m^k words distinct. For these,
  D(k) >= k log_2 m, saturated by LIST.
- (T) TITS CLIFF (imported; INVERTIBLE families only): finitely
  generated subgroups of GL_n(char 0) have word growth either
  polynomial (iff virtually nilpotent, by Gromov + Bass-Guivarc'h) or
  exponential (Tits alternative + Milnor-Wolf). No intermediate regime
  (Grigorchuk's group is non-linear). For SINGULAR families
  (semigroups), no dichotomy is claimed here — the intermediate-growth
  question for finitely generated linear semigroups is open in this
  draft. Within its scope, the premium is a cliff, not a dial.

RELATION TO PRIOR GROWTH THEOREMS (corrected per review): the counting
bound is a representation-independent FLOOR beneath every exact format.
The repository's TREE/CRC/GIRS growth theorems measure specific formats'
heights ABOVE this floor — and in TREE's sharp commuting case
(annihilating generators, word growth saturated), the format's TQS-width
growth exceeds the counting floor outright, i.e. that theorem measures
format suboptimality, not arithmetic necessity. No retroactive
"unification as instances" is claimed.

## 1. The separation lemma (repaired)

LEMMA 1. For any m >= 2, k >= 1, over any field K with |K| >
(k - 1) * binom(m^k, 2) + 1 (a NONZERO avoiding value; q = 0
always avoids the roots but makes D singular; k = 1 needs m distinct
lambdas available) — and over any infinite field — there exist
m INVERTIBLE 2x2 matrices A_1, ..., A_m such that all m^k products of
length k are pairwise distinct. Over Z, the integer evaluation
(q = m + 1; lambda_i = i) separates every pair of words by at least 1.

Proof. Let lambda_1, ..., lambda_m be distinct nonzero scalars, q a
scalar, and in M_2:
    D = diag(1, q),   S = E_{0,1} (the single up-shift entry),   A_i = D + lambda_i S
(each A_i is upper triangular with nonzero diagonal — invertible
whenever 1, q are nonzero). Consider a word w = A_{i_1} ... A_{i_k}.
Expanding each factor as D + lambda_{i_t} S, a term of the expansion
applies S at a subset of positions; since S^2 = 0 in M_2, only terms
with at most one S survive. The (0,0) entry collects the all-D term 1
plus zero-S contributions = 1. The (0,1) entry: a single S at position
t moves the running row index from 0 to 1; D-factors before t act at
row 0 (contribute 1), D-factors after t act at row 1 (contribute q
each; there are k - t of them); the S at t contributes
lambda_{i_t}. Summing over t:
    (A_{i_1}...A_{i_k})_{0,1}  =  P_w(q)  =  sum_{t=1}^{k} lambda_{i_t} q^{k - t}.
The exponents k - t are distinct across t, so P_w is a polynomial in q
whose coefficient vector IS the word (coefficient of q^{k-t} is
lambda_{i_t}). Distinct words give distinct coefficient vectors (the
lambda_i are distinct), hence nonzero difference polynomials of degree
<= k - 1. Over an infinite field, avoid the finitely many roots of all
C(m^k, 2) differences at one q, and all words are pairwise distinct
matrices. Over a finite field, the same avoidance needs
|K| > (k-1) binom(m^k, 2). Over Z, take q = m + 1 and lambda_i = i:
P_w is then the base-(m+1) numeral with digits i_t in {1, ..., m}, so
distinct words give integers differing by >= 1: exact separation with
no conditioning caveat. Genericity: each coincidence locus
{w = w'} is Zariski-closed and the exhibited point lies outside it, so
the separating tuples form a nonempty Zariski-open set. QED.

## 2. The counting lower bound (repaired)

THEOREM 2. (i) FINITE FIELD. Over F_Q with Q > (k-1) binom(m^k, 2),
the Lemma-1 family has m^k distinct length-k products; any exact
descriptor chain with B-bit descriptors satisfies 2^B >= m^k, i.e.
D(k) >= k log_2 m. (ii) EXACT INTEGER/BIT MODEL. Over Z with the
numeral evaluation, the m^k products are integers-of-words separated
by >= 1 at the (0,1) coordinate (and distinct matrices); any exact
bounded-precision descriptor chain must injectively label them, giving
the same bound with NO conditioning qualifier.

Proof. Exactness makes the descriptor map injective on the reachable
set; pigeonhole. QED.

## 3. The upper bounds (repaired)

THEOREM 3. (U1) Simultaneously diagonalizable family with generic
spectrum: the complete chain invariant is the count vector
(c_1, ..., c_m) (how many times each generator was applied), O(m log k)
bits; composition = componentwise increment; the operator is recovered
from prod_i A_i^{c_i} per coordinate. (U2) Finite group G: descriptors
are elements of a fixed registry of G, log_2 |G| bits; composition is
the group table: exact bounded closure. (U3) Generated algebra A with
dim A = d: fixed basis, d coordinate scalars, structure-constant
composition; descriptor COUNT bounded by d for the whole chain — the
quantitative Burnside envelope; bit sizes track entry growth over
exact rings (Theta(k) for generic families, matching and tightening
the lower bound: dense storage IS the terminal bounded format). QED.

## 4. The cliff (imported, correctly scoped)

THEOREM 4. For finitely generated SUBGROUPS of GL_n over char-0 fields:
word growth is polynomial iff virtually nilpotent (Gromov 1981; converse
Bass-Guivarc'h), otherwise exponential (Tits 1972 alternative + 
Milnor-Wolf for the solvable case). No intermediate growth occurs in
linear groups (the classical reason Grigorchuk's group is non-linear).

COROLLARY (cliff, invertible families, char 0). For invertible matrix
families, exact-chain descriptor cost floors sit in one of two regimes:
near-abelian (virtually nilpotent envelope: polynomial word growth,
logarithmic-to-bounded descriptor counts) or exponential (Theta(k log m)
floors, saturated by LIST). Singular/semigroup families: OPEN here.

## 5. Scope and honesty (updated)

- Bit model, per-k finite-field reading; over a FIXED F_Q the Lemma-1
  family exists only for k <= dim(A) * log Q / log m (saturation), as the
  model requires.
- LIST saturates rather than evades: the theorem is a tight two-sided
  characterization for generic families.
- Approximate chains are the walk law's domain, not this theorem's.
- The TREE sharp-growth family has saturated word growth: its TQS-width
  growth is height above the counting floor (format suboptimality), so
  no corollary relationship is claimed. CRC/GIRS are single-capacity
  statements, similarly above the floor.
- No speedup, no mission tier, no novelty claim for imported
  ingredients (Gromov, Tits, Milnor-Wolf, Bass-Guivarc'h). The claimed
  contribution: the bridge (descriptor floors = word counts), the
  numeral separation construction, and the tight LIST characterization.

## 6. Review receipt

Adversarial review agent_8bd5ec23 (fresh context, independent
re-derivation and numerics): SCOPED_PROOF_GO conditional on repairs
1-4 (finite-field thresholds; exponent formula f(t) = k - t and n = 2;
cliff restricted to invertible families with honest cheap-side bit
accounting; unification restated as floor-not-instances) and minors
5-9 (count-vector descriptors; integer-numeral form replacing the
conditioning caveat; per-k model reading; LIST saturation; preamble
cleanup + Zariski genericity). All applied in this revision; the
original draft's defects are preserved in the ledger row. A fresh delta
review of THIS revision is required before any PROVED label.
