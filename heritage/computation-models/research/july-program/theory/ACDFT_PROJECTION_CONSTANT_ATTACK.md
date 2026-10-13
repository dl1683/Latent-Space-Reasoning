# ACDFT Route-B Projection-Constant Attack

**Date:** 2026-07-14  
**Role:** Work-loop Batch 9, Unit 3, iterations 9-10  
**Status:** **HEURISTIC / `HONEST-STOP` AT THE UNCHANGED ENDPOINT.** Routes (a)-(c)
were attacked in the frozen order; neither `CONFIRM` nor all-routes `KILL` fired.

## 0. Frozen target and precommit

The inherited finite-grid derivation in
`research/ACDFT_DISCRETE_WIDENING_RESULT.md` closes by its random-coding route if
one proves, uniformly for every relevant rank-`d` projection,

`Pr(||P_E(xi-a)||_2 <= 30 sqrt(d)) <= (C * 61/127)^d`

with

`C <= C_star := 2.0293920940...`,

at every even `d >= 354`. The grid, radius, phase-net size 2048, expurgation rule,
and target rate `[d/32-11]^+` are unchanged.

**Route order - FROZEN.**

1. reconstruct the actual proof of Rudelson-Vershynin arXiv:1402.4492 and expose
   every constant that reaches Corollary 1.4;
2. check adjacent explicit-constant projection/product-measure, rearrangement,
   Esseen/Rogozin, Livshyts-Paouris-Pivovarov, and related small-ball results; then
3. use the exact DFT cross-block stable-subspace family rather than an adversarial
   arbitrary subspace.

**`CONFIRM` - FROZEN.** A complete written derivation at the frozen endpoint with
an explicit constant at most `C_star`, labelled **PROVED-PENDING-REVIEW**.

**`KILL` - FROZEN.** A proved lower bound showing that the achievable constant is
strictly above `C_star` for all three ordered routes. A loose upper bound, a large
constant extracted from one proof, or a failed source search cannot issue KILL.

**`HONEST-STOP` - FROZEN.** No complete at-threshold upper derivation and no
all-routes lower obstruction. This preserves the exact missing constant and does
not move the target.

**`VOID` - FROZEN.** Any change to six-bit support, radius 30, arbitrary shift,
phase budget, projection rank, stable-space quantifier, first nontrivial dimension,
or random-coding/expurgation rate.

**Claim discipline.** New derivations are at most **PROVED-PENDING-REVIEW**.
Primary-source statements are **PROVED (literature at source scope)** only at their
stated type. Source comparisons and route verdicts are **HEURISTIC**. Numerical
calculations are **EMPIRICAL** unless converted into exact inequalities on paper.

## 1. CPU-only coordinate-stable stress registration

**Purpose - HEURISTIC.** Before treating Fourier structure as automatically
helpful, test a coordinate-aligned stable subspace that can arise from a parity-row
DFT restriction. The check estimates the Cramer-Chernoff rate for

`S_d = U_1^2 + ... + U_d^2`, where `U_i` is uniform on `{-63,...,63}`,

at `S_d <= 900 d`, and compares its exponential base with
`alpha_star = (61/127) C_star = 0.9747473837...`.

**Precommitted CPU returns - FROZEN.**

- `COORD-RATE-BELOW`: the optimized asymptotic lower-tail base is strictly below
  `alpha_star`; the coordinate case does not obstruct the threshold.
- `COORD-RATE-ABOVE`: the optimized asymptotic lower-tail base is strictly above
  `alpha_star`; this triggers an exact-proof attempt before any obstruction claim.
- `COORD-ARITH-VOID`: support, threshold, or comparison value changes, or the code
  performs a parameter search outside the single Chernoff optimization.

**Execution scope - EMPIRICAL.** One CPU process evaluates the finite scalar
log-moment-generating function and its one-dimensional convex optimization. It
does not touch the GPU, write an experiment file, or claim an exact finite-`d`
probability.

## 2. Route (a): the actual Rudelson-Vershynin proof

### 2.1 Source-scope theorem

Rudelson and Vershynin, [Small ball probabilities for linear images of
high-dimensional distributions](https://arxiv.org/abs/1402.4492), Corollary 1.4,
prove that independent coordinates satisfying
`L(X_i,t) <= p` obey

`L(P_E X, t sqrt(d)) <= (C p)^d`

for an arbitrary rank-`d` orthogonal projection. This is **PROVED (literature at
source scope)**, but `C` is only called an absolute constant. It is exactly the
right type and exactly the constant missing from the frozen widening proof.

### 2.2 Constant dependency graph

**PROVED-PENDING-REVIEW.** A literal reconstruction does not expose a constant at
or below `C_star`. The proof of Theorem 1.1 chooses its terminal constant `C_1`
large enough to dominate all of the following:

1. the one-dimensional Fourier/rearrangement constant in Theorem 4.1;
2. the ball-volume conversion in Proposition 2.2;
3. the small-coordinate Brascamp-Lieb branch in Proposition 5.1; and
4. the removal constant `C_0` in Proposition 6.4.

The fourth item itself is reciprocal to the tensorization constant: Corollary 6.3
uses an unnamed beta-integral constant `C_beta`, sets
`c=1/(3 C_beta)`, and Proposition 6.4 requires `C_0 >= 1/c`. The terminal
induction then uses `M=C_1/2`. None of `C_beta`, the Proposition 2.2 volume
constant, or the Proposition 5.1 constant is numerically instantiated in the
source.

There is also a decisive bookkeeping warning inside the one-dimensional branch.
Theorem 4.1 bounds one Fourier tail by `8 pi a_j^2`, assumes `a_j <= 1/2`, and
absorbs it into a displayed `2 C a_j`. If that displayed `C` is instantiated
literally, this absorption alone requires `C >= 4 pi`, so that local displayed
density bound is at least `2C >= 8 pi`.

**HEURISTIC LIMIT.** This does not prove that the optimal projection constant is
at least `8 pi`, or even above `C_star`. It proves only that mechanically
assigning constants to this proof as written cannot return `2.0293920940`;
sharpening or replacing its lossy steps remains logically possible. Route (a)
therefore fails `CONFIRM` and cannot fire `KILL`.

## 3. Route (b): adjacent explicit-constant literature

### 3.1 Product-density projection theorem

Livshyts, Paouris, and Pivovarov, [On sharp bounds for marginal densities of
product measures](https://arxiv.org/abs/1507.07949), prove the sharp bound
`||pi_E f||_infinity <= 2^(d/2)` when the one-dimensional product densities are
bounded by one. After scaling a common coordinate-density bound `K`, this becomes
`(sqrt(2) K)^d`. This is **PROVED (literature at source scope)** and has an
explicit sharp constant, but the six-bit coordinates here are discrete atoms, not
bounded densities.

Juskevicius and Lee, [Small ball probabilities, maximum density and
rearrangements](https://arxiv.org/abs/1503.09190), give sharp rearrangement
comparisons for sums of independent random vectors with bounded densities. This
is also **PROVED (literature at source scope)**. The summands
`(P_E e_i) U_i` in the frozen problem are discrete and supported on lines, so a
direct application has the wrong input type.

Paouris and Valettas, [A Gaussian small deviation inequality for convex
functions](https://arxiv.org/abs/1611.01723), concern convex functions of Gaussian
vectors. This is **PROVED (literature at source scope)** but is not a discrete
product-coordinate projection inequality and supplies no transfer preserving the
frozen radius and scalar concentration `61/127`.

### 3.2 Explicit audit of the natural smoothing transfer

Consider the most direct bridge to the sharp product-density theorem: add
independent `Y_i ~ Uniform[-30,30]`. The convolution of the uniform six-bit atom
with `Y_i` has density at most

`K = 61/(127*60) = p/60`, where `p=61/127`,

because an interval of radius 30 contains at most 61 integer atoms.

Let `V_d = Vol(B_2^d(sqrt(d)))^(1/d)`. If
`||P_E(xi-a)|| <= 30 sqrt(d)` and
`||P_E Y|| <= 30 kappa sqrt(d)`, then the convolved projection lies in the ball
of radius `30(1+kappa)sqrt(d)`. Moreover

`E ||P_E Y||^2 = 300d`,

so the elementary Markov transfer gives nonzero dimension-free retained mass only
for `kappa > 1/sqrt(3)`. Applying the sharp density bound to the enlarged ball
and dividing out `p^d` yields the per-dimension constant

`C_smooth(d,kappa) = (sqrt(2)/2) V_d (1+kappa) q^(-1/d)`,

where `q` is the retained noise probability. Therefore every instance of this
specific density-plus-Markov pipeline satisfies

`C_smooth(d,kappa) > (sqrt(2)/2) V_d (1+1/sqrt(3))`.

At the first frozen dimension `d=354`, direct evaluation gives
`V_354 = 4.0919858320...` and hence the lower bound
`C_smooth > 4.5640171508... > C_star`.

The algebra above is **PROVED-PENDING-REVIEW**; the decimal evaluation is
**EMPIRICAL**. It kills this natural smoothing pipeline, not the sharp marginal
theorem and not every conceivable discrete transference. Esseen/Rogozin routes
encounter the same missing step in the checked record: an explicit
multidimensional integration or tensorization constant small enough to survive
the Euclidean-ball factor. No checked source supplied that step. Route (b) fails
`CONFIRM`, while the all-routes quantifier of `KILL` remains unmet.

## 4. Route (c): the actual DFT stable spaces

### 4.1 Exact family

For an output index set `R` and remote-input index set `C`, the directional
block is `A=F_{R,C}` and

`A^* A = P_C F^* P_R F P_C`.

The relevant complex stable space is the spectral subspace of this time-frequency
limiting operator with eigenvalues at least `1/4`; realification doubles its
dimension. This identity is **PROVED-PENDING-REVIEW** by direct matrix algebra.
It is genuine structure, but arbitrary balanced `R` and arbitrary input
placement leave a large family of non-contiguous row and column sets.

### 4.2 A structured member is coordinate aligned

**PROVED-PENDING-REVIEW.** Let `N` be even, take `R` to be the even output
frequencies, and let `C` contain at most one representative from each pair
`{j,j+N/2}`. Orthogonality of roots of unity gives

`(F_{R,C}^* F_{R,C})_{j,j'} = 1/2` when `j=j'`, and `0` otherwise.

Thus every singular value is `1/sqrt(2)` and the stable space is the full
coordinate span on `C`. The allowed family therefore includes easy-looking
coordinate spaces; Fourier origin alone cannot be treated as uniform
delocalization.

The preregistered CPU stress test for the centered coordinate case optimized the
exact scalar Chernoff expression for
`U ~ Uniform{-63,...,63}` and `sum U_i^2 <= 900d`. It returned

| quantity | value |
|---|---:|
| optimizing lambda | `-0.000363843487852414` |
| Cramer rate | `0.0756727639335534` |
| exponential upper-bound base | `0.927119543877884` |
| frozen allowed base `(61/127) C_star` | `0.974747383736549` |

This is **EMPIRICAL** and returns `COORD-RATE-BELOW`. It is only a centered
asymptotic Chernoff upper bound, not the arbitrary-shift finite-`d` theorem. It
shows that the coordinate member is not an obvious obstruction to the target.

### 4.3 Why the structure did not close the uniform theorem

**HEURISTIC.** The spectral identity does not by itself give the required
arbitrary-shift anti-concentration. In particular, a universal proof must cover
all balanced row sets and all remote-input placements, while the known convenient
Fourier-submatrix results require contiguity or randomness. The basic trace and
operator inequalities used in the flagship singular-dimension argument constrain
how many eigenvalues exceed `1/4`; they do not bound the arithmetic alignment of
the resulting spectral projector strongly enough to replace the arbitrary-
projection theorem.

No direct argument was found that converts the DFT spectral form into

`sup_a Pr(||P_E(xi-a)|| <= 30 sqrt(d)) <= 0.9747473837^d`

uniformly over the frozen family. Conversely, the coordinate calculation is
compatible with that inequality and supplies no family-wide lower obstruction.
Route (c) therefore fails both `CONFIRM` and `KILL`.

## 5. Frozen return

**HEURISTIC VERDICT: `HONEST-STOP`.** No route produced a complete explicit
constant at or below `2.0293920940`, and no proof put every route above that
threshold. The target, radius, grid, phase net, rank, and rate have not moved.

What is now explicit:

1. **PROVED-PENDING-REVIEW:** the Rudelson-Vershynin proof as printed contains
   unnamed dependencies and a local constant absorption already far above the
   required number; this is a proof-path failure, not a theorem lower bound.
2. **PROVED-PENDING-REVIEW:** sharp product-density constants do not directly type
   against atoms, and the most natural uniform-smoothing plus Markov transfer has
   constant greater than `4.564` at `d=354`.
3. **PROVED-PENDING-REVIEW / EMPIRICAL:** the DFT family includes coordinate
   stable spaces, but that stress case is compatible with the desired exponent;
   no uniform arithmetic regularity theorem was obtained for the whole family.

The remaining live obligation is exactly the inherited one: an explicit discrete
projection bound for this family at `C <= C_star`, or a genuine lower obstruction
covering every allowed proof route. This unit does not relabel an unspecified
constant as progress.

## 6. Execution integrity

**EMPIRICAL.** All numerical work used CPU only. No GPU process, subagent, or
experiment search was launched. The only optimization was the preregistered
one-dimensional Chernoff calculation; the volume constant was a deterministic
evaluation of the displayed formula. No source theorem was silently extended
from continuous densities or Gaussian inputs to the discrete projection problem.

## 7. Unexecuted Gate-0 stub: direct discrete Esseen on DFT stable spaces

**REGISTERED ONLY - CONJECTURE / NO EXECUTION.** The single highest-leverage
remaining route is a direct characteristic-function computation specialized
simultaneously to the centered six-bit law
'phi_U(t)=sin(127t/2)/(127 sin(t/2))' and to each concrete stable projector
'P_E=1_[1/4,1](P_C F^* P_R F P_C)' from the frozen DFT family, rather than another
arbitrary-projection constant extraction: with one fully normalized finite-
dimensional Esseen smoothing inequality, write the projected characteristic
function exactly as 'prod_j phi_U(<P_E e_j,t>)', split its torus integral into
explicit major/minor arcs using the DFT projector's arithmetic, and close the
constant chain all the way through the Euclidean-ball/smoothing factor to
'sup_a Pr(||P_E(xi-a)||<=30 sqrt(d)) <= alpha_star^d', where
'alpha_star=(61/127)C_star=0.9747473837...' for every even 'd>=354'.
Precommitted return 'DIRECT-ESSEEN-CONFIRM' requires every normalization, arc
volume, lattice multiplicity, spectral-projector bound, and finite-'d' remainder
to be explicit and their product at most 'alpha_star'; 'DIRECT-ESSEEN-ROUTE-KILL'
requires a proved obstruction for this exact specialized integral route, not a
large intermediate constant; 'DIRECT-ESSEEN-HONEST-STOP' preserves the unchanged
endpoint if neither closes; and 'VOID' fires on any change to support, radius,
shift quantifier, phase budget, stable-space family, dimension floor, or target
rate. This paragraph registers the route and constant ledger only; no source
sweep, Fourier integration, numerical search, CPU run, or proof attempt is
performed in Batch 10.

## NARRATIVE

**Honest one-sentence story - HEURISTIC.** *The projection route is still open at
exactly `2.0293920940`: the published proof is too lossy to certify it, the
cleanest explicit-density bridge is more than twice too large, and DFT structure
has not yet paid for the missing constant.*

**Obviousness and triviality attack - HEURISTIC.** Merely observing that an
absolute constant is unspecified would be trivial; exposing the dependency chain,
quantifying the smoothing failure, and testing an exact coordinate-aligned member
make the stop useful. It is still neither a new model nor a theorem pair, and it
cannot be presented as a mission result.
