# DIGIT_GEMM_COMPLEXITY_V1 - how many low-precision GEMMs does one exact higher-precision product need?

Status: IN PROGRESS (started 2026-09-29). Distance from claim: 1 (Floor-tier theorems and exact searches about the
arithmetic of matrix products on low-precision hardware; the constructions found, if any, are the artifact itself).
Mission tie-in: AGENTS.md section 1 (matrix-first digit/residue arithmetic = founding question Q5,
archive/root-history/PROGRAM_HISTORY.md:2479-2481), section 3 (every count carries its value set, accumulator bound and
post-processing model), section 11 (each exclusion is paired with the constructive question of what escapes it).
Code: `code/digit_gemm.py` (single module). Data: `results/digit_gemm_complexity_v1/`.

## 0. PRE-REGISTRATION (written after the literature read, before any computation; hashed by `code/prereg.py`)

**Disclosure.** Before writing this section I read the papers listed below (full text of AWE, arXiv 2609.24519; the
HTML of 2603.10634 and 2504.08009; abstracts of the others) and, while reading, sketched IN MY HEAD (no code, no
numbers computed) four arguments that shape the predictions: (a) any exact scheme with post-processing linear over Q
restricts to subspaces U_A, U_B (spans of the plane vectors) on which the reconstruction form has rank 1, so Sylvester's
rank inequality gives r >= rho_A + rho_B - 1; (b) rho_A >= ceil(log_|V| N_A) because a rho-dimensional subspace meets V^r in
at most |V|^rho points (N_A = number of input values), with an arithmetic-progression refinement (a 1-dimensional
subspace can serve N_A = |V| inputs only if V is an arithmetic progression); (c) if that bound is attained, both
plane codes must be MDS and the relation spaces must be orthogonal under the diagonal form 1/lambda; (d) a dependent
plane f.u whose coefficient ratios lie outside (V-V)/(V-V) keeps at most |V|^(rho-1) points. None of (a)-(d) has been
written down, checked or attacked; they are part of what this unit must prove or refute. Nothing below about the
outcome of any exhaustive search, SAT/ILP run, or modular table has been computed.

**Question.** For a hardware value set V (what one GEMM operand entry can hold, up to a per-plane scale) and an
accumulator that is exact under a stated bound, what is the minimum number r of GEMMs whose outputs determine the exact
product of two integer matrices of widths (M_A, M_B) (or of the truncated significands used by an FP64 emulation)?

**Models (formalised in section 1; each result states its model).** A scheme applies entrywise encodings
P_k: X_A -> V, Q_k: X_B -> V (k = 1..r; arbitrary functions, representations may be adaptive/redundant), computes
G_k = P_k(A) Q_k(B) exactly, and post-processes.
- M1: C = sum_k lambda_k G_k, lambda in Q^r, valid for every inner dimension n (AWE's model is the special case with
  planes linear in limbs, integer weights, dyadic lambda).
- M2: C == sum_k lambda_k G_k (mod M) followed by symmetric reduction, valid for n <= N (M2p: M prime;
  M2crt: M = prod m_i pairwise coprime, each product serving one modulus - Ozaki-II; M2: general).
- M3: arbitrary post-processing F(G_1..G_r), valid for every n. M4: arbitrary F, valid for n <= N.
- Mpoly (classical control): limbs in a digit set D, planes linear in limbs with values in V for every limb tuple,
  target = all 2s-1 coefficients of the digit convolution (the polynomial-product tensor).
Value sets: V_FP4 = 2 x E2M1 = {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12} (15 values, FP32 accumulation, K* = floor(2^24/(c_A c_B)));
V_FP8 = E4M3 integers in [-16,16] (33 values, as used by 2603.10634; wider E4M3 integer sets reported separately with
their K*); V_INT8 = [-128,127] (INT32 accumulation, n*128^2 < 2^31); V_INT4 = [-8,7]. Block scales are per-plane
constants only (absorbed into lambda). Inputs of width M: the symmetric range [-(2^(M-1)-1), 2^(M-1)-1] (AWE convention).

**Named prior art, closest first, and what each did NOT do.**
- M1 / FP4: (1) Hayashi, Mukunoki, Hoshino, Katagiri, AWE, arXiv 2609.24519 (21 Sep 2026): search-found counts
  (INT8xINT8 6, INT4xINT8 4, INT4xINT4 3, table of (M_A, M_B)); did NOT prove any lower bound; its r = 5 search covered
  only planes with components |.| <= 2 (independent sides) or <= 3 with l1 <= 5 (same planes); open problem stated.
  (2) Hayashi et al., arXiv 2608.06812: base-13 limbs, 9 products for INT8xINT8; no optimality claim. (3) Classical
  bilinear complexity: Toom, Karatsuba, Winograd (Math. Systems Theory 10, 1976; minimal polynomial-multiplication
  algorithms are interpolation/CRT based), Fiduccia-Zalcstein (JACM 1977), Brockett-Dobkin (1978, bilinear algorithms
  and linear codes), Buergisser-Clausen-Shokrollahi (1997): rank 2s-1 of polynomial multiplication and code
  connections, all over a field with FREE coefficients; did NOT treat a value set with holes on the planes.
- M1 / INT8 digits (Ozaki-I): Ozaki-Ogita-Oishi-Rump (Numer. Algorithms 2012); Ootomo-Ozaki-Yokota (IJHPCA 2024,
  arXiv 2306.11975); Mukunoki (arXiv 2508.00441, FP8 Ozaki-I). Did NOT give lower bounds on slice-product counts or
  study Karatsuba-type reductions under the INT8 value set.
- M2 / CRT (Ozaki-II): Ozaki-Uchino-Imamura (arXiv 2504.08009; INT8, 1 GEMM per modulus, 14-16 GEMMs for FP64-level
  accuracy); Uchino-Ozaki-Imamura (arXiv 2603.10634; FP8 E4M3 with |v| <= 16, Karatsuba 3 GEMMs per modulus, 3N or
  3N+1 GEMMs, N >= 12); AWE section V-C (FP4 mod-m: 1-product moduli {2..9,11,13}, 82 three-product moduli, integer
  program -> 59 products for the FP64 significand at K = 16384). None proved per-modulus lower bounds or optimality of
  the modulus selection. Engineering context (no counting results): Matsuoka, Ozaki 2.5 (arXiv 2609.09095); Schwarz et
  al., ESC/ADP accuracy guard (arXiv 2511.13778); EmuGEMM fused kernels (arXiv 2606.25453). arXiv 2602.19090 is an
  eigenvector refinement paper that uses the Ozaki scheme (no counting content).

**H0 (boring).** For every (value set, width pair) studied, the unit produces only a certified map: standard
rank/counting lower bounds, published counts, and gaps. Either the standard bound already equals the published count, or
the gap stays open (no exhaustive decision closes it) and no scheme with fewer products than the best published one
is found.
**H1.** At least one of: (H1a) a (value set, width/precision) pair admits a scheme with strictly fewer GEMMs than the
best published one, verified exactly (identity checked over all admissible input pairs, value set and accumulator bound
checked); or (H1b) a certified lower bound STRICTLY above the standard Sylvester/counting bound (a) + (b) is proved or
decided exhaustively and equals a published count, closing a gap (a Floor theorem: the published scheme is optimal in
the stated model). Proving that the standard bound (a)+(b) alone meets a published count is recorded as H1b-weak (new
relative to the literature, which states no lower bound, but not a strengthening of the bound).

**Registered predictions (from the in-head sketches only).**
- P1 (M1, V_FP4, INT8xINT8): lower bound 5 from (a)+(b); the exhaustive decision of r = 5 finds NO scheme
  (probability 0.75), i.e. AWE's 6 is optimal in M1 -> H1b. If a 5-product scheme is found -> H1a.
- P2 (M1, V_FP4 table): (a)+(b) alone certify AWE optimal (H1b-weak) at INT4xINT4 (3), INT4xINT5 (3), INT5xINT5 (3),
  INT4xINT8 (4). For the cells where AWE = bound + 1 and the tight case is searchable (both widths <= 11): the tight
  case is infeasible at every such cell (probability 0.7, correlated), so AWE is optimal there too.
- P3 (M2, per modulus): for prime moduli p the M1 argument runs over F_p, so p > |V| forces >= 3 products; hence the
  FP8 3-GEMMs-per-prime-modulus of 2603.10634 and AWE's 3-product prime moduli are optimal per modulus. "2 GEMMs per
  modulus" is impossible for every prime p > |V| and predicted impossible for every prime power p^e > |V|
  (probability 0.7; small p, especially p = 2, is the risky case).
- P4 (M2crt, FP8 V = [-16,16], FP64-level modulus product 2^110 at K <= 2^16): an integer program over pairwise coprime
  moduli with certified per-modulus counts (1-GEMM moduli where V covers Z/m, 3-GEMM moduli where a construction is
  verified) needs FEWER than the published 36 FP8 GEMMs (probability 0.6; mechanism: mixing 1-GEMM small moduli, which
  2603.10634 did not use) -> H1a, stated only for that family and K bound.
- P5 (FP64 via INT8, exact 54-bit signed significands): M1 and M2p need >= 13 (= 2*7 - 1); the tight case r = 13 is
  infeasible (probability 0.6) giving >= 14; arbitrary post-processing (M4) only gives the injectivity bound
  ceil(54/8) = 7 and the output-information bound 5 at n = 4096 (the brief's claimed 7 is a valid bound in M4, not a
  tight one); M3 reduces to M1. No scheme below the CRT count for exact products is found (H0 for FP64).
- P6 (Mpoly, s = 2, 3 limbs, small V): with digits D = contiguous part of V, the minimum r equals s^2 for V_FP4
  (no Karatsuba sum fits) and 2s - 1 only when V's contiguous part is at least s times the digit range; SAT/ILP gives
  the exact minima with UNSAT at r - 1.

**Decision rule.** H1a needs an exact verified scheme beating the best published count for the same value set,
accumulator bound and post-processing model. H1b needs a proof (or an exhaustive decision resting on proved lemmas,
reproduced by a second independent implementation or solver) that r >= published count where (a)+(b) gives less.
Otherwise H0.

**Kill rule.** If after the SAT/ILP/enumeration campaign (budget 6 wall-hours, <= 4 threads, single-thread solvers)
every searched cell sits at the published count and no lower bound above (a)+(b) closes a gap, deliver the map of
(model, bound, best-known scheme, gap) and change method: study the models that open slack (nonlinear carries, exponent
alignment, per-row scaling, non-bilinear or bounded-n post-processing, mixed formats) - do not stop.

**Rigor labels.** Theorems: PROVED-pending-gate until an independent referee passes them (AGENTS.md section 6).
Exhaustive decisions: EXACT-COMPUTATION (certified by the proved reduction + two independent implementations/solvers).
Counts quoted from papers: LITERATURE. No wall-clock claims (nothing is timed in this unit).

## 1. Models (formal)

V is a finite set of rationals (what one operand entry of the low-precision GEMM can hold; a per-plane scale factor,
e.g. a constant block scale, is absorbed into the reconstruction coefficient, so each plane takes values in V itself).
Inputs are integer matrices with entries in X_A (left) and X_B (right); width M means X = [-(2^(M-1)-1), 2^(M-1)-1].
A scheme with r products consists of entrywise encodings P_k: X_A -> V, Q_k: X_B -> V (arbitrary functions; adaptive
or redundant representations allowed), the r GEMMs G_k = P_k(A) Q_k(B) computed exactly (accumulator bound stated
per result), and a post-processing.
- **M1 (linear over Q).** C = sum_k lambda_k G_k, lambda in Q^r, for every inner dimension n. By linearity in the
  inner index this is equivalent to the scalar identity sum_k lambda_k P_k(x) Q_k(y) = x y for all x, y. AWE's
  Definition IV.1 is the special case "planes are integer combinations of limbs in V, integer weights, dyadic lambda".
- **M1aff.** As M1 plus O(n^2) corrections (row sums of A-planes, column sums of B-planes, n x constant), i.e. the
  zero-point trick. Centering P'_k = P_k - P_k(0) turns it into an M1 scheme whose plane k takes values in the
  translate V - P_k(0) (same size |V|, same difference set V - V).
- **M2 (linear mod M).** C == sum_k lambda_k G_k (mod M), then symmetric reduction; valid while 2 max|C| < M.
  M2p: M prime. M2crt: M = prod m_i pairwise coprime with every product serving one modulus (Ozaki-II).
- **M3.** Arbitrary post-processing F(G_1..G_r), valid for every n. **M4.** Arbitrary F, valid for n <= N only.
- **Mpoly** (classical control): limbs in a digit set, planes linear in limbs, target = all coefficients of the
  digit convolution (the polynomial-product tensor).

## 2. Lower-bound theory (rigor: PROVED-pending-gate unless marked otherwise)

Notation: for a scheme, U_A = span{P(x)} in Q^r, rho_A = dim U_A (likewise U_B, rho_B); beta(u, v) = sum lambda_k u_k v_k;
ell(V, X) = least j such that X is contained in {theta . v : v in V^j} for some theta in Q^j. D = (V - V) \ {0},
R = D/D (a finite set of rationals; |R| = 382 for V_FP4).

**Lemma 1 (linear structure).** In any M1 scheme (all lambda_k != 0 WLOG) there are linear functionals c on U_A and
d on U_B with c(P(x)) = x, d(Q(y)) = y and beta(u, v) = c(u) d(v) on U_A x U_B.
Proof. Put c(u) = beta(u, Q(y0))/y0 and d(v) = beta(P(x0), v)/x0 for fixed nonzero x0, y0. Then c(P(x)) = x and
d(Q(y)) = y, and the two bilinear forms beta and c(x)d agree on the spanning sets {P(x)} x {Q(y)}. QED.

**Theorem 2 (value-set Sylvester bound).** r >= rho_A + rho_B - 1, rho_A >= ell(V, X_A), rho_B >= ell(V, X_B), hence
r >= ell(V, X_A) + ell(V, X_B) - 1; moreover ell(V, X) >= ceil(log_|V| |X|), and ell(V, X) = 1 iff X lies in a scaled
copy theta V. The same holds over any field in which the values live, in particular over F_p for M2p (with X and V
replaced by their residues).
Proof. The restriction of beta to U_A x U_B has matrix G_A^T Lambda G_B (G_A, G_B bases as columns, Lambda = diag(lambda)
invertible); it equals c(x)d, which has rank 1, and Sylvester's inequality gives 1 >= rho_A + rho_B - r. Choose rho_A
coordinates T whose coordinate functionals form a basis of the dual of U_A; then c = sum_{k in T} theta_k e_k and
x = c(P(x)) = sum_{k in T} theta_k P_k(x), so X_A lies in theta . V^T and ell <= rho_A. Counting |theta . V^j| <= |V|^j
gives the log bound. QED.

**Theorem 3 (the tight case is MDS and self-dual up to a diagonal twist).** If r = ell_A + ell_B - 1 then
(i) c is not a combination of fewer than ell_A coordinate functionals of U_A (likewise d); (ii) U_A and U_B are MDS
codes; (iii) the extended codes E_A = {(u, c(u))} and E_B = {(v, d(v))} are MDS [r+1, ell_A] and [r+1, ell_B] codes and
E_B = (Lambda+ E_A)^perp with Lambda+ = diag(lambda_1..lambda_r, -1).
Proof. Tightness forces rho = ell on both sides. (i) is the definition of ell. (ii) Let T be ell_A coordinates whose
functionals on U_A were dependent; they span W_T with dim < ell_A, so by (i) c is not in W_T and some u* in U_A has
u*_k = 0 on T and c(u*) != 0. Then z = Lambda u* is supported on at most r - ell_A = ell_B - 1 coordinates and
sum_k z_k v_k = beta(u*, v) = c(u*) d(v) for all v in U_B, so d is a combination of <= ell_B - 1 coordinate functionals,
contradicting (i) for B. (iii) Any ell_A coordinates of E_A are independent by (ii) (all planes) or by (i) (planes plus
the value coordinate); sum_k lambda_k u_k v_k - c(u) d(v) = 0 gives the orthogonality and ell_A + ell_B = r + 1 the
equality. QED.
Consequence: when the ranks are ell_A = ell_B = s the tight count r = 2s - 1 is attained only by "generalized Toom"
schemes: if the r + 1 columns of E_A lie on a conic (for s = 3) the scheme is Toom-Cook with evaluation points t_k and
base point s (then both sides decouple); otherwise the two sides are Gale-dual point configurations. Classical
Karatsuba/Toom is the conic case.

**Lemma 4 (ratio lemmas; hold in a tight scheme when |X_A| > |V|^(ell_A - 1), e.g. INT8 over V_FP4: 255 > 225).**
(R1, dual) Every relation among ell_A + 1 planes of U_A (a minimum-weight codeword of the dual code) has all ratios of
its coefficients in R. (R2, primal) Every minimum-weight codeword of U_A (vanishing on ell_A - 1 coordinates) has all
ratios of its entries in R; and if V is a set of integers and d is its primitive integer direction, then
|c(d)| <= 2 max|X_A| in units of the value step (collision bound). Both hold for U_B too.
Proof. (R1) If two coefficients have ratio outside R, the pair of plane values is determined by the relation and the
other ell_A - 1 plane values (distinct pairs of V-values give distinct weighted sums exactly when the ratio is not in
D/D), so the relation has at most |V|^(ell_A - 1) solutions in V^(ell_A + 1); the covering points project injectively
to these coordinates (any ell_A coordinates are free by Theorem 3), so |X_A| <= |V|^(ell_A - 1). (R2) The covering
points take at most |V|^(ell_A - 1) < |X_A| values on the ell_A - 1 coordinates, so two of them agree there; their
difference is a nonzero multiple of the codeword with entries in D on its support, and its c-value is a nonzero
difference of two inputs. QED.

**Lemma 5 (M3 reduces to M1aff).** If a post-processing F computes C exactly for every inner dimension n, then there is
phi in Q^r with x y = sum_k phi_k (P_k(x) - P_k(0)) (Q_k(y) - Q_k(0)) for all x, y: an M1aff scheme with the same r.
Proof. Let g(x,y) = (P_k(x)Q_k(y))_k and g^ = g - g(0,0). Two multisets of input pairs with equal sum of g^ can be
padded with (0,0) pairs to equal size n without changing either side, so F forces equal sums of x*y: the kernel of
the group homomorphism Z^(X_A x X_B) -> Z^r (e_(x,y) -> g^) lies in the kernel of e_(x,y) -> x*y, which therefore factors
through a homomorphism on the image lattice, extendable to phi in Q^r. Setting x = 0, then y = 0, removes the cross
terms. QED. So nonlinear post-processing buys nothing at GEMM scale; only bounded-n tricks (M4) or modular reduction
(M2) can escape the M1 bounds.

**Theorem 6 (per-modulus bounds; answers "2 GEMMs per modulus").** (a) Prime modulus p: Theorem 2 over F_p gives
r >= 2 ell_p - 1, so r = 1 (iff V covers Z/p) or r >= 3: two GEMMs per prime modulus are impossible for every value
set. FP8 (V = [-16,16]): every prime p > 33 needs >= 3; with (c)-(d) below, EVERY modulus in the list of arXiv
2603.10634 (1089, 1024, 961, 841, 625, 529, 511 = 7*73, 509, 503, 499, 491, 487) needs >= 3, so its per-modulus count
is optimal and any saving must come from the modulus SELECTION (section 3.3);
FP4 (V_FP4): every prime p >= 17 needs >= 3 (AWE's 3-product prime moduli are optimal). (b) A 2-product scheme mod m
reduces to one mod every prime factor, so every prime factor of m must be a 1-product prime. (c) Prime powers p^e,
e >= 2, 2 products (proof: write x = x0 + p x1, expand to second p-adic digit; the rank of the resulting digit form
over F_p forces a plane to take p values in each residue class, or two planes n1(x0) n2(x0) >= p values): necessary
that |V| >= p^2, or p^(3/2) <= |V|, or p^2 <= |V| |V cap pZ|. FP8: only p in {2,3,5,7} survive; FP4: only p in {2,3,5}.
(d) Generation lemma: in a 2-product scheme mod m = p^e write a(x), b(y) in V^2 for the plane pairs, B(a,b) =
lambda1 a1 b1 + lambda2 a2 b2, a* = a(1), b* = b(1). The bilinear form N(a,b) = B(a,b) - B(a,b*) B(a*,b) vanishes on all
used pairs; if both representation sets generate (Z/m)^2 then N == 0, and evaluating N on unit vectors forces the
non-unit lambda to vanish (a 1-product scheme, impossible when V does not cover Z/m). Each side needs >= m distinct
representatives, so m <= max over the p + 1 maximal submodules {g.a == 0 mod p} of their number of points of V^2
(`two_product_moduli.json`). FP8: excludes 1024, 729, 625, 343 and every power of p >= 11; OPEN only for
{64, 128, 256, 512, 81, 243, 125, 49}. FP4: OPEN only for {16, 32, 64, 128, 27, 81, 25}. Deciding these would
settle whether mixed CRT counts can drop further (at most ~1-2 FP8 GEMMs, ~2-3 FP4 products).

## 3. Results (EXACT-COMPUTATION unless marked; code/digit_gemm.py, data results/digit_gemm_complexity_v1/)

**3.1 AWE reproduced** (`awe_reproduction.json`). INT8xINT8 with 6 FP4 GEMMs (w = (1,29,37); planes a_i, a_i + a_j;
lambda = (-65,-261,259,29,37,1073)): matrix identity exact, identity exact over all 875 x 875 admissible pairs,
coverage 148 >= 127, K* = 116,508. Alternative w = (1,11,39): coverage 134. INT4xINT8 with 4: coverage 10 / 154. AWE's
model (Definition IV.1) = M1 restricted to integer limb combinations with limbs in V, integer weights, dyadic lambda.

**3.2 Exact ell and the M1 lower-bound table vs AWE** (`ell_and_bound_table.json`). ell(V_FP4, INT_M) = 2 for
M = 4, 5, 6 (INT4 needs the AP refinement: V_FP4 contains no 15-term progression) and ell(V_FP4, INT7) = 3 EXACTLY:
covering 127 values with two digits forces the denominator q into {1,2} (p-adic class counts: 15 x maxclass >= 127 needs
maxclass >= 9, only S mod 2 has it) and |n_k| <= 126 q (collision bound); the complete search finds no witness. For
M >= 8 the log bound is used (ell(8..10) = 3 attained by base-13 limbs). Lower bound ell_A + ell_B - 1 EQUALS AWE's
published count at 7 cells, so AWE is optimal in M1 there: INT4xINT4 = 3, INT4xINT5 = 3, INT5xINT5 = 3, INT4xINT8 = 4
(Theorem 2 with the log/AP bound: H1b-weak) and INT4xINT7 = 4, INT5xINT7 = 4, INT7xINT7 = 5 (need exact ell(7) = 3,
strictly above the log bound: H1b). Also {0,+-1}xINT8 = 3 (AWE Table II) is optimal (ell = 1 + 3 - 1). Gap-1 cells:
(4,6), (4,9), (5,6), (5,8), (5,9), (6,6), (6,7), (7,8), (8,8).
Three gap-1 cells closed by an exhaustive tight-case decision (`int6_controls.json`): in any 3-product scheme for
INT_M x INT6 with M in {4,5,6} the INT6 side is, by Theorem 3 and Lemma 4 (63 > 15), an R-good MDS [3,2] plane code
with |X| >= 63 carrying a functional that covers [-31,31]. There are 29 such codes (single dependent row f with
f1, f2, f1/f2 in R) and none carries a covering functional (complete 2-D search over every allowed denominator q with
the collision bound |n.d| <= 62q). Controls: the same search finds 72 covering functionals for INT5 on the Karatsuba
code f = (1,1) (AWE's INT5xINT5 = 3 is such a tight scheme), and at the reduced target [-10,10] it returns exactly the
brute-force sets on all 29 INT6 codes. Hence AWE's 4 is optimal in M1 at (4,6), (5,6), (6,6) (EXACT-COMPUTATION
resting on Theorems 2-3 and Lemma 4): 10 of AWE's published cells are now certified optimal.
Monotonicity (PROVED, one line): an R-good MDS [r, ell] plane code with r > ell + 1 contains the R-good MDS
[ell + 1, ell] sub-code spanned by its free coordinates and one dependent row, whose point set contains the covering
set; the collision and denominator bounds are derived from the covering set itself, so a covering functional of the
big code is found by the sub-code search. Hence INT6 has NO tight partner of any kind: every INT6 x INT_M scheme needs
>= ell(M) + 2 products. This certifies (6,7) = 5 (AWE) as well: 11 cells. For INT8 the covered-code list is NOT
empty (section 3.6b): (8,8) = 6 is certified by the pairing test, (7,8) stays open.

**3.3 FP8 Ozaki-II: 32 GEMMs instead of 36 (H1a, same model and target)** (`modular.json`, `crt_verified.json`). Value
set E4M3 integers |v| <= 16, FP32 accumulation exact for k <= 65,536 (products <= 256), modulus product P > 2^111
(the P/2 > 2^110 of arXiv 2603.10634). Candidate family: 1-GEMM moduli m <= 33 (V covers Z/m), 3-GEMM moduli certified
by the 2-limb Karatsuba recipe x == a0 + s a1 with a0, a1, a0 + a1 in V (711 moduli up to 817). A MILP (scipy HiGHS)
over pairwise-coprime choices gives 32 = 8 x 3 + 8 x 1: {769, 767, 751, 749, 733, 719, 709, 701} (3 each) and
{33, 32, 31, 29, 25, 23, 19, 17} (1 each), log2 P = 113.56 (so the truncation/accuracy is at least as good). Every
per-modulus identity was checked over all residue pairs, and an end-to-end run (`fp8_e2e.json`: random integer
matrices 8x256x8 at 50 bits, 16x1024x16 at 48 bits, 8x65536x8 at 40 bits; FP8-exact planes, float32-accumulated
GEMMs, per-modulus recombination, CRT) reproduces the exact integer product in every entry; restricting to 3-GEMM
moduli reproduces the published 36
(12 moduli). Mechanism: 2603.10634 dismissed 1-GEMM moduli because alone they reach only ~2^47; mixed with 3-GEMM moduli
they are the cheapest bits (4.1-5.0 bits per GEMM vs ~3.2). Cost honesty: 16 CRT terms instead of 12 (more
reconstruction multiply-adds per output), 32 instead of 36 FP32 outputs read; no wall-clock measured. Optimality is
claimed only within the declared candidate family (square-trick moduli such as 1089 = 33^2 and any 2-product prime
powers are not in it). The same MILP for FP4 gives 62 > AWE's 59 (their 3-product list is richer): no FP4 improvement.
Scope and novelty (checked against Matsuoka, Ozaki 2.5, arXiv 2609.09095): its "all-byte" (15 moduli <= 256) and
"hybrid" (squares <= 33^2 plus a byte tail) sets store byte residues as INT8 operands ("the centred residue of a byte
modulus fits an int8 operand directly"), i.e. they are INT8 or mixed-format schemes; its square moduli use 3 products
(two stored planes, epilogue {1, s, s}), consistent with Theorem 6. The 32 < 36 improvement is therefore a statement
about PURE-FP8 hardware (no INT8 unit, e.g. B300 per AWE); where INT8 exists, INT8 CRT needs 15-16 GEMMs. The
mechanism (MILP over mixed per-modulus costs) is AWE section V-C's, transferred to FP8.

**3.4 INT8 (engine table)** (`config_table.json`). For b-bit-magnitude integer inputs and inner dimension k, the INT8
CRT with the MILP-optimal set of pairwise-coprime moduli <= 256 (1 GEMM each) needs e.g. 4 (b = 8), 6 (b = 16),
8 (b = 24), 10 (b = 32), 14-15 (b = 48), 16 (b = 53) GEMMs at k in {4096, 16384}, against Theorem 2's M1/M2p floor
2 ceil((b+1)/8) - 1 = 3, 5, 7, 9, 13, 13 and the M4 injectivity floor ceil((b+1)/8). Signed 8-bit digit schoolbook
needs ceil((b+1)/8)^2. Exact-FP64-significand products (b = 53) therefore sit 3 GEMMs above the bilinear floor.

**3.4b Engine interface (`config_table.json`, requested for the coupled `ozaki_fair_engine_v1` gate).** Rows are keyed by
(value set, bits b of the scaled-and-truncated integer inputs, inner dimension k). Each row carries: the certified
scheme (INT8: MILP-optimal coprime moduli <= 256 with one GEMM each, and the signed-8-bit-digit schoolbook
alternative), GEMM count (all GEMMs have inner dimension k, i.e. inner-dimension multiple 1), reconstruction reads
(4 bytes per GEMM output per C entry) and multiply-adds (one per CRT term), exactness conditions (INT32: k * 128^2 <
2^31; CRT: prod(m) > 2k(2^b - 1)^2; beyond the accumulator bound split k into exact segments and add the segment
outputs in INT64), the lower bounds that apply (M1/M2p: 2 ceil((b+1)/8) - 1; M4: ceil((b+1)/8)) and rigor labels
(exactness PROVED; count optimal within the declared family by MILP; lower bounds PROVED-pending-gate). The FP8 and FP4
FP64-level rows (section 3.3) carry per-modulus recipes (representation tables, weight s, reconstruction formula).
Error model for choosing b (PROVED, elementary): with power-of-two row/column scaling and round-to-nearest truncation
to magnitudes <= 2^b - 1 (scale exponent chosen so the row maximum maps below 2^(b-1)... 2^b - 1), each entry error is
<= 2^(1-b) times its row (column) maximum, hence |C^ - C|_ij <= 2^(1-b) (max_l |A_il| sum_l |B_lj| +
(1 + 2^(1-b)) sum_l |A_il| max_l |B_lj|); the target-error-to-b mapping is the engine unit's contract.

**3.6 Exhaustive decision of the tight INT8xINT8 case r = 5 (model M1, V_FP4) - method.** Any 5-product scheme has
rho_A = rho_B = 3 = ell(INT8) (Theorem 2), so it is tight; by Theorem 3 both plane codes are MDS [5,3] codes; by
Lemma 4 (255 > 225) each is R-good and has |X| >= 255. Enumeration (`codes53_N255.json`): 4,534 canonical dependent
rows (entries and pairwise ratios in R, |G| >= 255), 142,549,036 (f, g) pairs passing the column-ratio test, 283,000
with |X| >= 255, 170,592 R-good codes (f canonical, g any sign/order variant; max |X| = 779). Orbit dedupe under
coordinate permutations and sign flips: keep a listing iff the free triple {1,2,3} minimises the Pluecker key (sorted
|p_(T-i+j)|/|p_T|; ties kept, so every orbit keeps >= 1 listing): 15,976 representatives (`reps53.json`), then exact
orbit canonicalisation (`canonical53`: lexicographic minimum over the 120 orderings of the sign-normalised systematic
matrix) to the orbit list `orbits53.json`. For each orbit representative ALL rational covering functionals are
enumerated in one pass (`covering_functionals_rat`): c lies on the rational line {c.u1 = 1, c.u2 = k2}, c = c0 + t w
with w = u1 x u2 (u1 = the value-1 representative, one of each +- pair since X = -X; u2 = the representative of the
smallest value k2 <= 13 not represented by the multiple k2 u1, which exists because 13 u1 is not in S^3); every
further value k forces t = (k - c0.u)/(w.u) for some u in X (or is attained on the whole line by a point of
span(u1,u2), which then imposes nothing), so the exact rational key sets of four such values are intersected, t is
restricted by the collision bound |c.d_jl| <= 254 (pruning only; any genuine covering functional satisfies it), and
the survivors are checked exactly for all 255 values. No denominator lemma is needed. Completeness follows from
Theorem 3 and Lemma 4 (plus: if every value were attained inside span(u1,u2), the 255 covering points would lie in a
plane, which holds at most 225 points of S^5). An intermediate integer-denominator implementation (q enumerated from a
p-adic count) was abandoned because that count barely restricts q on these structured point sets. Pre-run gate (`cover_controls.json`, `lines_controls.json`): on AWE's 6-plane code the search finds 360
covering functionals (180 up to sign) including (1,29,37) and (1,11,39); on 6 real codes with the target shrunk to
[-12,12] the plane generator, the line generator and the compiled line kernel all return exactly the brute-force
sets (1716, 1719, 1554, 1719, 1404, 1061 functionals); a sign bug in the compiled integer ceiling was caught by these
controls before the production run. Production implementation: `covering_functionals_rat` (numba, exact integer
keys of t, only the first non-auto k2 per u1). Its pre-run gate (`rational_kernel_controls.json`): on AWE's 6-plane code
at the production target K = 127 its Q = 1 part equals the stored integer set exactly (180 up to sign, including
(1,29,37) and (1,11,39)), and it also returns 2,838 half-integral covering functionals of that code; in five synthetic
symmetric point sets with a planted functional N/Q (Q = 1, 2, 2, 3, 5) and no collision pruning it returns the planted
functional every time; every returned functional re-checks exactly. ~6 s per code; 10 parts of 568 orbit
representatives, 3 workers, per-chunk incremental saving.

**3.6b Result: INT8 x INT8 needs 6 FP4 products in M1 - AWE's 6 is optimal (EXACT-COMPUTATION resting on Theorems 2-3
and Lemma 4; proof gate pending; STATUS: final pairing on this unit's complete covered list pending - see 3.6c).** The pre-run claim
"no scheme iff no orbit is covered" was too strong: 28 of the 5,678 orbits DO carry covering functionals (457 up to
sign, denominators 1, 2, 3, 4, 6, 8, 12; e.g. f = g-rows (1/3,1/3,1/3), (1/3,-2/3,4/3) with c = (1,-6,35)), found by
an independent reimplementation (separate subagent, own enumeration, own orbit list of the same size 5,678, own
covering search over the denominators allowed by a p-adic filter; every functional re-verified exactly here), and
cross-checked by this unit's production kernel (section 3.6c). A 5-product scheme needs BOTH sides covered AND
coupled by Theorem 3(iii): E_B = (Lambda+ E_A)^perp, i.e. W(A, c) = {(z, t) in Q^6 : sum_k z_k gam_k = t c} must be
diagonally equivalent (plane coordinates permuted, the functional coordinate kept last) to the extended code
{(u, f.u, g.u, d.u)} of a covered (B, d) - lambda is then read off from the scaling. Test (`tight53_pairing_verifier.json`):
the diagonal-equivalence invariant (systematic 3x3 block normalised to ones in its first row and column; remaining 2x2
block) of W for each of the 457 covered (A, c), against all 120 coordinate orderings of all 457 covered extended codes
(16,440 distinct keys). Positive control: 5 covered extended codes, randomly permuted and diagonally rescaled, each
match themselves. Result: 0 matches. The ratio-lemma route (`pin_partner`, `tight53_partner_check.json`) also returns
no partner for any of the 457. Hence no 5-product INT8 x INT8 scheme exists in M1 over V_FP4, and with the
monotonicity remark of section 3.2 (the INT8 side of any tight (7,8) or (8,8) scheme is an R-good [5,3] code; for
(7,8) the INT7 side would be a [5,3] code covering 127 values, whose E-code must be diagonally equivalent to W(A,c)
of a covered INT8 code - NOT yet tested, so (7,8) stays open) only (8,8) = 6 is added: 12 of AWE's cells certified.

**3.6c Cross-check (in progress).** On the 7 verifier-covered orbits re-run so far
(`tight53_covered_crosscheck_p*.json`) this unit's kernel returns exactly the verifier's functionals on 6 and ONE MORE on
the 7th (verifier rep 972: c = (85,16,-19)/6, exact re-check passes): the verifier's p-adic denominator filter or search
is incomplete, so its covered list is a lower bound only and the pairing test of 3.6b is re-run on this unit's complete
list (`tight53_summary.json` -> `tight53_pairing_production.json`). The full production search (all 5,678 orbits,
resumable chunks) is running; the verdict on (8,8) is conditional on it.

**3.7 FP64 via INT8: the map (model, bound, best scheme, gap).** Exact product of 54-bit signed significands
(|x| <= 2^53 - 1), k = 4096: M1 and M2p floor 13 (Theorem 2: ell = ceil(log_256(2^54 - 1)) = 7); M4 (any
post-processing, bounded n) floor 7 by injectivity (the brief's "ceil(t/b) = 7" is a valid bound only in this weakest
model) and 5 by output information ((2*53 + 12)/(16 + 12)); M3 = M1aff (Lemma 5). Best scheme: INT8 CRT with 16
MILP-optimal coprime moduli <= 256 (exact); Ozaki-II's 14-16 for FP64-LEVEL (not exact) accuracy. Gap: 13 vs 16 in
M1/M2p; the tight [13,7] case is not searched (MDS [13,7] codes over INT8 are far beyond enumeration; a density or
anti-concentration argument is the promising route).

**3.7b The 13-vs-16 opening, made precise.** (i) Within M2crt with one INT8 GEMM per modulus, 16 is OPTIMAL for exact
54-bit significand products at k = 4096 (EXACT-COMPUTATION: the MILP ranges over every pairwise-coprime subset of
2..256, and V = INT8 covers Z/m only for m <= 256, so no other 1-GEMM modulus exists; 15 moduli give at most
118.00 < 119.0 bits, MILP max with 15 moduli, status optimal). (ii) Any count below 16 therefore needs either a 2-GEMM modulus (Theorem 6(b,c): only prime powers
p^e with p^2 <= 256, e.g. 2^e, 3^e, 5^e, 7^e, 11^e, 13^e, can qualify; NOT decided) or an M1 scheme with 13-15 products.
(iii) A 13-product M1 scheme is a tight generalized-Toom scheme: both plane codes MDS [13,7] over V = INT8 with >= 2^54
points each, every dependent row with coefficient ratios in R(INT8) (Lemma 4 applies: 2^54 > 256^6 = 2^48), and the
two extended codes twisted-dual (Theorem 3). HEURISTIC obstruction (not a proof): the 6 dependent planes must each stay
in [-128,127] on >= 1/4 of the free box V^7, while an MDS row has 7 nonzero coefficients; anticoncentration of
sum f_i u_i makes one such constraint hold on a fraction ~1.4/|f|_2 and six of them jointly on far less than 1/4.
Turning this into a proof (an inverse Littlewood-Offord bound over the value box) would close 13; a counting proof of
14 needs the r = 14 analysis (rho = 7 and 8). Status: OPEN, with a stated route.

**3.10 Chains of products (persistent residue arithmetic): range and base-extension cost model** (`chain_cost_model.json`,
`python code/digit_gemm.py chain`; counts EXACT-COMPUTATION by MILP, model PROVED elementary). For an exact chain
A_1 ... A_q with |entries| <= H = 2^b - 1 and inner dimension k, the range after s factors is R_s = H^s k^(s-1) (in
general R_s = prod H_t * prod k_t), and each product in residue form costs one INT8 GEMM per modulus. Fixed base:
(q-1) N(R_q) GEMMs; growing base (add moduli by base extension before each product, elementwise work
O(N_s * dN_s) per entry, no GEMM): sum_s N(R_s) - about 60% of the fixed base. Hard cap: the product of all pairwise
coprime m <= 256 is lcm(1..256) = 2^362.8, so exact INT8-residue chains end at log2(2 R_q) <= 362.8: q <= 19 for 7-bit
data, 13 for 16-bit, 5 for 53-bit at k = 4096. Against rounding every intermediate back to b bits ((q-1) N(R_2) CRT
GEMMs; 1 GEMM per product when b <= 7), persistent exactness costs a factor ~(q+1)/2 in GEMMs (b = 16, q = 8: 119 vs 42;
b = 53, q = 4: 74 vs 48). So what: persistent residues never save GEMMs; they buy exactness and remove the per-step
reconstruction/re-encode (bytes moved), which is the only currency in which they can win - the engine must price
reconstruction bytes, not GEMM count, to justify them.

**3.8 Remaining open cells and escapes (constructive obligations, AGENTS.md section 11).** (5,8) is the one gap-1
cell with ell <= 3 not decided (both sides have covered codes: INT5 has 781 covered R-good [4,2] codes, INT8 has at
least AWE's [4,3] code; the tight case needs the twisted-duality pairing, implemented as `pin_partner` but not run).
(4,9), (5,9): decided iff no R-good [4,3] code covers INT9 (243 candidate rows, `codes43`); NOT run to completion - the
[4,3] covering search has only 6 collision directions and point sets up to 1,735, costing minutes per row, beyond this
unit's compute budget on the shared machine. Models that escape the M1 exclusions and are NOT decided:
M1aff (zero-point corrections: per-plane translated value sets S - s_k; same lower bounds, affine codes), M2 with
bounded n (the injectivity floor is far below the bilinear one), block-adaptive NVFP4/MXFP4 scales (per-element value
set becomes a union of scaled copies of S, outside per-element encodings), mixed formats (AWE's 2 FP8 + 1 FP4
INT8xINT8 scheme), and 2-product prime-power moduli for CRT (Theorem 6(d) open list).

**3.9 Mpoly control (the brief's "polynomial-product tensor" target).** When the post-processing must recover every
digit-convolution coefficient (fixed positional weights), the target has bilinear rank 2s - 1 and, by the
Winograd/Fiduccia-Zalcstein characterisation (minimal algorithms for polynomial multiplication over Q are evaluation at
2s - 1 distinct points of P^1(Q)), r = 2s - 1 holds iff 2s - 1 distinct points (p:q) admit plane vectors
c (q^(s-1), q^(s-2) p, ..., p^(s-1)) with c (.) a in V for every digit tuple a. With V_FP4 and the full contiguous digit
range D = [-4,4], every admissible plane is a multiple of a single limb (any two nonzero coefficients reach a hole of
V_FP4), so the products are monomials and r = s^2 (4 for s = 2, 9 for s = 3). With D = [-2,2]: s = 2 has the points
0, infinity, +-1 (a0 +- a1 in [-4,4]), so r = 3 (Karatsuba); s = 3 has only 0 and infinity (a0 + a1 + a2 reaches 5),
so r = 5 is impossible, and the 3-term Karatsuba (a_i b_i and (a_i + a_j)(b_i + b_j), sums in [-4,4]) gives r = 6.
This is the classical side of AWE's remark that Karatsuba needs m >= 3 and Toom-3 m >= 5 significand bits; it is a
CONTROL, not the right model for integer emulation: the digit weights are free there (AWE), the target collapses to the
rank-1 form (w.a)(v.b), and the relevant bounds are Theorems 2-3.

## 4. Verdict (2026-10-01)

**H1 holds, on both branches, with stated scope.**
- **H1a (EMPIRICAL-exact, verified scheme):** FP8 Ozaki-II for FP64-level products (V = E4M3 integers |v| <= 16,
  k <= 2^16, P > 2^111) needs 32 FP8 GEMMs, not the published 36 (section 3.3): per-modulus identities checked over
  all residue pairs, end-to-end exact on three shapes. Scope: pure-FP8 hardware; mechanism = AWE's mixed-cost MILP
  transferred to FP8; cost: 16 CRT terms instead of 12; no wall-clock measured.
- **H1b (EXACT-COMPUTATION on PROVED-pending-gate lemmas):** AWE's FP4 counts are optimal in M1 at 12 cells. 4 by the
  standard bound alone (H1b-weak: (4,4), (4,5), (5,5), (4,8)) and 8 strictly above it: (4,7), (5,7), (7,7) by exact
  ell(V_FP4, INT7) = 3; (4,6), (5,6), (6,6) by the INT6 tight-case exhaustion; (6,7) by monotonicity; and the flagship
  (8,8) INT8 x INT8 = 6 (section 3.6b: >= 28 covered orbits, none with a Theorem-3 partner) - CONDITIONAL until the
  complete production list (3.6c) is paired; until then 11 cells are certified.
- **P1-P6:** P1 confirmed (6 optimal), but its mechanism was wrong (covered codes exist; the coupling kills them).
  P2 confirmed at the searched cells; (5,8), (7,8), (4,9), (5,9) remain open. P3 confirmed for primes; prime powers
  partly open (Theorem 6(d) list). P4 confirmed (32 < 36). P5: floors 13 (M1/M2p), 7/5 (M4) proved; the r = 13 tight
  case was not searched; best exact scheme 16 (INT8 CRT). P6: confirmed for V_FP4 (r = s^2 with full digit range).
- **Engine coupling (`config_table.json`):** INT8 CRT rows (MILP-optimal coprime moduli <= 256, 1 GEMM each,
  inner-dimension multiple 1, 4 bytes read per GEMM output) sit 2-3 GEMMs above the M1/M2p floor 2 ceil((b+1)/8) - 1
  for b >= 24; rigor: exactness PROVED, counts optimal within the declared family (MILP), floors PROVED-pending-gate.

**So what.** Low-precision emulation counts are now partly a closed theory: the bilinear (M1) floor, its tight case
(generalized Toom, Theorem 3) and the value-set ratio lemmas decide small cells exactly, and confirm that AWE's
search-found counts are optimal where decided. The one constructive gain (FP8 32 < 36) comes from modulus SELECTION,
not from cleverer per-modulus algebra (Theorem 6: per-modulus counts are already optimal).

**Open / next.** (i) Independent proof gate on Theorems 2-3, Lemma 4, Theorem 6, and on the pairing test. (ii) Cells
(7,8), (5,8): the easy-side partner must be searched over lambda (continuous family), not matched against a list.
(iii) Escapes from M1 that the engine can use: M2 with bounded n, block-scaled formats, mixed FP8/FP4 planes,
2-product prime-power moduli. (iv) FP64 via INT8: the r = 13 tight case vs the CRT's 16.


## FINAL RESULT: the complete INT8 x INT8 search and pairing test (2026-10-02; EMPIRICAL exhaustive computation, proof gate PENDING)

The production search over ALL 5,678 canonical plane-code representatives finished (resumable chunks, `tight53_summary.json`: searched 5678, complete true): 28 codes admit a one-sided 5-product functional set (793 functionals up to sign; the independent verifier's list, 457 functionals on the same 28 orbits, is a subset, as expected). The pairing test (Theorem 3: both sides of a 5-product scheme must fit together) over the COMPLETE list found **0 matches** (16,920 keys; positive control: 5 of 5 random self-matches found; negative control: the old route also returns nothing). Result: within the model of the unit (INT8 x INT8 -> product of two 8-bit limbs via FP4 plane codes, bilinear schemes with linear post-processing over Q), **no 5-product scheme exists, so AWE's 6-product scheme is optimal: the 12th cell is certified** subject to the pending independent proof-gate pass on Theorems 1-3 and the search kernel. Scripts: `code/digit_gemm_tools/{compare53,pair53,partner53}.py` (copied from the working directory).

## Addendum 2026-10-03 (unit prime_power_two_product_v1): Theorem 6(d) open list
The two-product identity xy = (1-lambda) u_x u_y + lambda v_x v_y (mod p^(a+t), a >= 2t, planes ctr_M(Z/M) inside V; proof gate pending) constructs and exhaustively verifies (all m^2 pairs, matrix controls) the previously OPEN FP8 two-product prime powers {64, 128, 81, 125} and INT8 two-product moduli up to 4096; FP8 {256, 512, 243, 49} remain not constructed. The CRT frontier is UNCHANGED at every conclusive tested target (FP8 31 / 34 products at 2^111 / 2^119; INT8 15 / 16), so mixed CRT counts do not drop. See results/prime_power_two_product_v1/REPORT.md.
