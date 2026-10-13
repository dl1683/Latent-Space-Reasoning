# FLOOR_LDR_V1: composition and descriptor growth of structured-matrix classes (LDR, butterfly, Monarch, hierarchical)

Status: **the unit's deliverable is the controlled MAP (section 3); H0 STANDS and no new theorem is claimed.** The one
H1 candidate that survived earlier rounds, M2(c), is an explicit statement plus an independent tangent-space proof of a
direct consequence of Dao et al. 2022 App. D.3 (section 6). Conjectures M5 (equality for k >= 3; saturation at k = m) and
B1-sufficiency remain open. Created 2026-09-29; repair round 3 (section 9) 2026-09-30.
Labels: all text that existed before repair round 3 (including the [R2] text) was attacked by a cold no-memory referee on
2026-09-30 and survived; its proofs read "PROOF (referee-attacked, survived)". Text added in repair round 3 is tagged [R3],
reads "PROOF [R3, gate pending]" and is script-checked but ungated. "PROVED" is reserved for the project's proof-review
gate and is not used in this document (section 1).
Gate history: (1) cold no-memory Opus referee (2026-09-29): accepted M1, M2(a)-(c) scoped to {L Pi R} after the
containment step, B1 upper bound, B2, Prop K bound, Theorem R identities, Prop U identity; rejected M4/M2 framing for
Dao's class, HSS-as-Sylvester and the Bresar uniqueness claim, LDR Kronecker non-closure beyond Kronecker pairs; repairs
applied. (2) gate 5 (Stateful Codex referee, 2026-09-29): algebraic core survives, all four result JSON files reproduce,
not passed unchanged; repair round 2 (section 8) applied the six requested repairs. (3) cold no-memory Opus referee
(2026-09-30): NO FATAL findings; every PROOF statement survived independent attack and every finite claim reproduced with
independent code (exact Fraction/sympy ranks, modular ranks mod 2^31-1 and 1e9+7, gauge kernel upper bounds, the hash
of the registered text); the document nevertheless "must not pass unchanged": 16 findings (2 major: novelty of M2(c), a
reinstated withdrawn unification in section 7; 14 minor), all applied in repair round 3 (section 9), which has not been
re-gated. Distance from claim: 1 (Floor-tier statement about existing representations; no measurement infrastructure).
Machine budget: single thread, small exact linear algebra only (a wall-clock benchmark shares the host).

## 0. PRE-REGISTRATION (written before any derivation output or computation)

**Gap.** AGENTS.md section 2 Floor tier: a unifying theory of butterfly/Monarch, low displacement rank, hierarchical
matrices, with theorems on which classes admit compressed, composition-friendly encodings and which do not.
Existing repo material (not re-derived here): DISPLACEMENT_GENERALIZATION_V1 (Sylvester chain rule, Bresar
uniqueness, class table with one-line butterfly/HSS rows), SUBALGEBRA_LANDSCAPE_V1 row "butterfly/Monarch: NO",
TREE_COMPOSITION_FRONTIER_V1 (tree/TQS additive ranks), FLOOR_THEORY_V1 (three law-packages, butterfly cascade form).

**H0.** The literature (Pan; Kailath-Sayed-Chun; Olshevsky-Shokrollahi; Hackbusch; Bebendorf; Chandrasekaran-Gu;
Eidelman-Gohberg; Dao et al. Monarch/butterfly) already contains every closure/growth/expressivity statement the
project could state; the unit yields only a controlled map (classes x operations, citations, exact costs, strongest
baseline) and no new theorem.
**H1.** At least one statement in the map is (a) absent from the literature as far as a focused search finds and
(b) proved here with a self-attack. Candidate H1 statements registered: (i) exact dimension of the Monarch variety
and of its k-fold product classes (descriptor growth is exactly additive, saturation point); (ii) a slice-rank
growth lower bound showing Monarch is not product-closed with bounded growth (n^(1/2) vs additive); (iii) exact
generic dimension of butterfly and butterfly-product varieties; (iv) exact Kronecker-product displacement rank in
the Sylvester setting; (v) a unified "rank profile of linear maps" statement covering LDR and hierarchical classes
(off-diagonal rank as Sylvester displacement rank for projector pairs).

**Registered numeric predictions (before the checker is run).** m = sqrt(n), n = m^2.
P1. dim Mon(m) = 2m^3 - m^2 (m = 2,3,4,5). P2. dim Mon^2(m) = 3m^3 - 2m^2, and Mon.Mon = Mon^2 as sets.
Mon^2 is dense iff m = 2. P3. dim Mon^k(m) = min(m^4, (k+1)m^3 - k m^2) for k = 3 (m = 3,4), k = 4 (m = 4,5);
in particular dense exactly when k >= m. P4. Butterfly variety (L = log2 n factors, each with 2n nonzeros):
dim = n(L+1) for n = 4, 8, 16; product of k butterflies: min(n^2, n(kL+1)). P5. Kronecker product of two generic
Toeplitz matrices under the pair (Z_e (x) Z_e', Z_f (x) Z_f'): displacement rank equals 2n1 + 2n2 - c for a small
constant c (to be read off); it is NOT bounded independently of n1, n2. P6 (LDR identities, exact over Q):
rank nabla(MN) <= r1 + r2 generically = r1 + r2; rank nabla_{B,A}(M^-1) = rank nabla_{A,B}(M) exactly;
projector identity rank([P,M]) = rank(M_12) + rank(M_21); quasiseparable order of a product of two order-1
matrices is 2 generically.
**Decision rule.** H1 accepted iff one of (i)-(v) is proved in a stated scope, survives the attack list, matches
the checker, and is not present in the sources named below. Predictions that fail are reported as failures and the
statement is corrected, not the prediction. **Kill rule.** If P1/P2 fail at any tested (m,k), the Monarch statements
are withdrawn. If every H1 item is found in the literature, H0 stands and section 3's map is the deliverable and the
next step is the most promising gap in it.

**Three closest sources per class (and what each did NOT do).**
- LDR: Kailath-Sayed (eds.), *Fast Reliable Algorithms for Matrices with Structure*, SIAM 1999, and Kailath-Sayed
  SIAM Rev. 37 (1995) "Displacement structure: theory and applications" - do not treat classes outside Sylvester
  pairs, nor descriptor growth for Kronecker products in different operator pairs, nor a common HSS/LDR statement;
  Pan, *Structured Matrices and Polynomials* (2001) - operator-pair conversions and cost tables, no exact
  non-closure/dimension statements; Olshevsky-Shokrollahi, "A superfast algorithm for confluent rational tangential
  interpolation" / Olshevsky-Pan 1998 - superfast Cauchy-like arithmetic, stability caveats, no growth theorem for
  repeated products.
- Hierarchical/semiseparable: Eidelman-Gohberg (Oper. Theory 1999-2005) and Eidelman-Gohberg-Haimovici, *Separable
  Type Representations of Matrices and Fast Algorithms* (2014) - product/inverse orders exactly; Chandrasekaran-Gu et
  al. (HSS, SIMAX 2006-7) - fast HSS arithmetic, rank additivity; Hackbusch, *Hierarchical Matrices: Algorithms and
  Analysis* (2015) and Bebendorf (2008) - H-matrix arithmetic with truncation, which is approximate closure; none
  presents the linear-map rank-profile view that ties HSS to displacement.
- Butterfly/Monarch: Dao et al. ICML 2019 (butterfly), ICML 2022 (Monarch, Thm on M^(b,n) parameter lower bound
  n^(3/2)) and Dao et al. ICLR 2020 (Kaleidoscope) - expressivity and "products are not closed", no exact variety
  dimension, no product-class dimension count; Le Magoarou-Gribonval, "Flexible multilayer sparse approximations of
  matrices" (2016) - sparse-product factorization; Zheng-Riccietti-Gribonval, "Efficient identification of butterfly
  sparse matrix factorizations" (SIMODS 2023) - identifiability of the butterfly factorization, not product depth.

## 1. Setting and rigor labels

KNOWN (classical, cited) / PROOF (defined below) / EMPIRICAL (exact finite check) / CONJECTURE / SPECULATION. "Generic rank of
the Jacobian" over Q: the dimension of the image variety equals the rank of the parameter map's differential at a
generic point; a Jacobian rank computed modulo a prime at one integer point is a rigorous LOWER bound for that
generic rank (rank can only drop mod p or at a special point). Upper bounds come from explicit gauge symmetries.
Combined, they give the exact dimension at the tested size.

Proof-status convention (revised in repair round 3). The gate rule of AGENTS.md section 6 reserves "PROVED" for the
project's proof-review gate, so this document does not use it. **PROOF (referee-attacked, survived)** = proof written and
self-attacked, then attacked by a cold no-memory referee on 2026-09-30 (the scoped cores were also accepted by the
2026-09-29 cold referee and by gate 5) with no fatal finding; it covers all text that existed before repair round 3,
including the [R2] text. **PROOF [R3, gate pending]** = statement added in repair round 3, checked by the scripts named in
section 9 but attacked by nobody but its author. **KNOWN** = cited; section/page locators are as reported by the cold
referee and were not re-read by the repair agent, and the numerical-analysis citations (Grasedyck-Hackbusch, Borm,
Bebendorf-Hackbusch, Faustmann et al., Chandrasekaran et al., Candes et al., Li et al., Gautschi/Beckermann,
Strang-Nguyen) are marked "cited, not re-verified this round" in the map. **EMPIRICAL** = exact finite check, rigorous at
the tested sizes and silent about other sizes.

## 2. New statements (H1 candidates) with proofs

Notation: n = m^2, Pi = the transposition permutation of the m x m index grid (an involution, Pi = Pi^T = Pi^-1).
**Class convention (repaired).** Below "Mon" means **Mon_LPiR = {L Pi R : L, R block-diagonal, m blocks of size m x m}**
(this is not Dao's MM* class, which is Mon^2 conjugated by Pi, see below). Dao et al. 2022 (Def. 3.1) define Monarch as M = P L P^T R = Pi L Pi R, i.e. the set Pi.Mon_LPiR:
the same set up to a FIXED row permutation Pi, which does NOT commute with products. Single-matrix statements (M1
slice form, dimension, slice rank of one matrix) transfer after applying Pi to the rows; **product, inverse and
composition statements do not transfer** and hold below only for Mon_LPiR. Dao's own class under products: a product
of two Dao matrices is Pi L1 Pi R1 Pi L2 Pi R2 (R1, L2 do not merge). Exact Jacobian rank mod p (code
floor_ldr_checks.py dao, results/floor_ldr_v1/dao.json, EMPIRICAL): 16, 81, 208 at m = 2, 3, 4 (Dao single: 12, 45,
112 = 2m^3-m^2), i.e. dense at m = 2, 3 and 4m^3-3m^2 = 208 < 256 at m = 4; so Dao-class products grow much faster
than Mon_LPiR^2 (dimensions 16, 63, 160 by M2). **[R3, MINOR-6] These ranks are exact, not merely lower bounds:** the
product Pi L1 Pi R1 Pi L2 Pi R2 is Pi.Mon^3, whose gauge upper bound is min(m^4, 4m^3 - 3m^2) = 16, 81, 208, and the
mod-p Jacobian ranks meet it (section-1 method). Dao's MM* class (Dao et al. 2022 Sec 3.4, (P L1 P^T) R (P L2 P^T) =
Pi (L1 Pi R Pi L2) Pi) is the doc's Mon^2 conjugated by Pi, of the same dimension 3m^3 - 2m^2; this is why M2(c) is a
direct consequence of Dao App. D.3 (section 6). Dao-class inverse and transpose closure FAIL as a set [R2: exact certificate]:
results/floor_ldr_v1/dao_exact.json stores, for m = 3, exact integer blocks L_a, R_b, D = Pi L Pi R, D^-1 (Fractions;
D D^-1 = I checked entrywise) and the offending minors. A matrix X is in Dao's class iff every slice of Pi X has rank
<= 1 (M1). All slices of Pi D have rank 1 (all 2x2 minors 0), but slice (0,0) of Pi D^-1 has the 2x2 minor
det[[-4/27, 1/18],[1/12, 0]] = -1/216 != 0 (rows 0,1; cols 0,1; the whole slice has det 1/60, rank 3), and slice (0,0)
of Pi D^T has the minor det[[0,-3],[-1,0]] = -3 != 0 (slice det 54, rank 3). So D is in the class while D^-1 and D^T
are not. The certificate is re-verified from the JSON alone by an independent code path (closed form D[(i,j),(b,c)] =
L_j[i][b] R_b[j][c], entrywise sums, direct minors; verify_dao_exact in code/floor_ldr_checks.py). The earlier float
check (dao.json, dao_closure) is supplementary. Indices (a,e),(b,c) in [m]^2 (row pair, column pair).
**[R3, MINOR-7] The failure is a fixed-permutation artifact. PROOF [R3, gate pending].** For D = Pi L Pi R,
D^-1 Pi = R^-1 Pi L^-1 and D^T Pi = R^T Pi L^T, and both lie in Mon_LPiR (block-diagonal inverses and transposes; Pi Pi = I).
So Dao's class is closed under inverse and transpose up to moving the fixed Pi to the other side: (Pi.Mon)^-1 =
(Pi.Mon)^T = Mon.Pi on the invertible members. Checked on the stored certificate (results/floor_ldr_v1/r3.json
dao_piside): both identities hold entrywise and every slice of D^-1 Pi and of D^T Pi has rank <= 1, while Pi D^-1 and
Pi D^T (the offenders above) are not in Mon_LPiR. The certificate is correct as stated; its significance is only that
the SET Pi.Mon_LPiR is not inverse- or transpose-closed with the leading Pi held fixed.

**Prop M1 (slice form). PROOF (referee-attacked, survived).** With blocks L_a[e,b], R_b[a,c]: M[(a,e),(b,c)] = L_a[e,b] R_b[a,c].
Hence Mon = {M : each of the m^2 slices M[a,.,b,.] (an m x m matrix in (e,c)) has rank <= 1}, and the m^2 slices are
independent rank-one matrices u_ab v_ab^T with u_ab = L_a[:,b], v_ab = R_b[a,:]. So Mon is a product of m^2 rank-one
cones and **dim Mon = m^2(2m-1) = 2m^3 - m^2** (not the 2 n^(3/2) parameter count of the factorisation).

**Theorem M2 (product class, exact dimension; class Mon_LPiR only). PROOF (referee-attacked, survived). Not new [R3]: (c)
is a direct consequence of Dao et al. 2022 App. D.3 (see the novelty note after the containment paragraph); the proof below
is an independent tangent-space proof.** (a) Mon.Mon = Mon^2 := {L Pi S Pi R}, because
(L1 Pi R1)(L2 Pi R2) = L1 Pi (R1 L2) Pi R2 and R1 L2 is an arbitrary block-diagonal S (take L2 = I). (b) In slice
form, M[(a,e),(b,c)] = sum_{a'} S[a,a',b] l_{a,a'}[e] r_{b,a'}[c] with l_{a,a'} = L_a[:,a'], r_{b,a'} = R_b[a',:],
S[a,a',b] = S_{a'}[a,b]. Equivalently slice_{ab}(M) = L_a diag(S[a,.,b]) R_b, with L_a, R_b the a-th and b-th diagonal
blocks of L and R (PROOF [R3, gate pending]; exact check for m = 2, 3, 4 in r3.json slice_formula): the m^2 slices share the
factors L_a along a and R_b along b and differ only through the diagonal middle factor. This replaces the earlier vacuous
statement "every slice has rank <= m" (the slices are m x m). (c) **Over a field F of characteristic 0 (Q, R, C), dim Mon^2 =
3m^3 - 2m^2 for every m >= 2** (dimension of the Zariski closure of the image = generic rank of the differential), so the
codimension in the m^4-dimensional space is m^2 (m-1)(m-2): Mon^2 is dense iff m = 2, proper for m >= 3. No statement
is made over finite fields; the mod-32003 Jacobian ranks below are supplementary only.
Proof of (c). Upper bound: 3m^3 parameters minus the 2m^2-dimensional gauge (l_{aa'} -> lam_{aa'} l_{aa'},
r_{ba'} -> rho_{ba'} r_{ba'}, S_{aa'b} -> S_{aa'b}/(lam_{aa'} rho_{ba'})); the torus acts with trivial stabiliser
wherever all l_{aa'}, r_{ba'} are nonzero, so every such fibre contains a 2m^2-dimensional orbit. Lower bound: **[R2] explicit witness W**
(indices a, a', b in {0,...,m-1}): l_{a,a'} = r_{b,a'} = e_{a'} (the standard basis of F^m, i.e. L = R = I) and
S_{a a' b} = S_{a'}[a,b] = 2^(a a' b), so that M = Pi S Pi has M[(a,e),(b,c)] = delta_{ec} 2^(a e b). W satisfies the
three conditions used below: (i) for each a, {l_{a,a'}}_{a'} is a basis of F^m, and for each b, {r_{b,a'}}_{a'} is a basis;
(ii) all S_{aa'b} != 0; (iii) for every ordered pair c != d the m x m matrix Q_{ab} = S_{acb}/S_{adb} = 2^((c-d) a b)
is not of the form (x_a/y_b): if it were, Q_{00} Q_{11} = Q_{01} Q_{10} (since x_0 x_1/(y_0 y_1) is symmetric), i.e.
2^(c-d) = 1, false in characteristic 0 because c != d (m >= 2 supplies the indices 0 and 1). At W take a tangent vector
(dl, dr, dS) with dM = 0. Expand dl_{a,d} = sum_c X^a_{dc} l_{a,c}, dr_{b,c} = sum_d Y^b_{cd} r_{b,d} (unique by (i)).
In the basis l_{a,c} (x) r_{b,d} the coefficient of dM_{ab} at (c,d), c != d, is
S_{adb} X^a_{dc} + S_{acb} Y^b_{cd} = 0 for all a,b, i.e. X^a_{dc} = -Q_{ab} Y^b_{cd}. If some X^a_{dc} != 0 then (by (ii))
Y^b_{cd} != 0 for all b and X^{a'}_{dc} != 0 for all a', giving Q_{ab} = -X^a_{dc}/Y^b_{cd}, contradicting (iii); if all X
vanish then all Y vanish. So the off-diagonal parts of X, Y are 0; the diagonal coefficient (c,c) gives dS_{acb} =
-S_{acb}(X^a_{cc} + Y^b_{cc}), which is exactly the 2m^2-dimensional gauge (dl_{a,c} = X^a_{cc} l_{a,c} != 0 makes the
map (X_diag, Y_diag) -> (dl, dr, dS) injective). Kernel dimension = 2m^2, rank of the differential at W = 3m^3 - 2m^2;
the generic rank is >= this (lower semicontinuity) and <= the gauge bound, so both equal 3m^3 - 2m^2. QED.
("Dense" means Zariski-dense only: m = 2 gives the full m^4.) Supplementary checks of the witness
(results/floor_ldr_v1/m2_witness.json): the Jacobian rank at W is 3m^3 - 2m^2 = 16, 63, 160, 325 for m = 2..5 modulo
32003 (kernel = 2m^2 = 8, 18, 32, 50), and **exactly over Q** (Fraction elimination) for m = 2, 3, with the slice form
of M at W confirmed.
**Containment Mon <= Mon^2 (Latin-square step, needed for "strict").** I is not in Mon (its slices have rank m), so
Mon = Mon.I does not give Mon <= Mon.Mon; it is shown directly: given a Monarch M with slices u_ab v_ab^T, set
(indices mod m) l_{a,a+b} = u_ab, r_{b,a+b} = v_ab (other l, r arbitrary) and S_x[a,b] = delta_{x,a+b}. Then
sum_{a'} S_{a'}[a,b] l_{a,a'} r_{b,a'}^T = u_ab v_ab^T, so M is in Mon^2. Checked numerically, m = 3, 4 (max abs error 0;
results/floor_ldr_v1/dao.json, latin_square). Hence Mon <= Mon^2 and, since dimensions differ by
(3m^3 - 2m^2) - (2m^3 - m^2) = m^3 - m^2 > 0 for m >= 2, the containment is strict. Non-closure itself follows from
the dimension count alone (Mon^2 = Mon.Mon has larger dimension than Mon): Mon_LPiR is **not** product-closed.
Growth per multiplication is m^3 - m^2 = n^(3/2) - n new degrees of freedom.

**Novelty status of M2(c) [R3, MAJOR-1; H0 STANDS].** Dao et al. 2022 (arXiv:2204.00595) Sec 3.4 writes the MM* class as
(P L1 P^T) R (P L2 P^T), which is the doc's Mon^2 conjugated by Pi. Their App. D.3 (pp. 29-31, as read by the cold
referee, not re-read by the repair agent) writes each block as M~_ij = A_i D_ij C_j, gives the symmetry group explicitly
(A_i -> A_i P S_i^-1, C_j -> P^T S'_j^-1 C_j, D_ij -> S_i P^T D_ij P S'_j: 2m^2 continuous parameters plus a permutation)
and shows that for any factorisation F(i,j) = M~_i1^-1 M~_ij M~_1j^-1 M~_11 = C_1^-1 (diagonal) C_1, with the diagonaliser
unique up to permutation and scaling when the eigenvalues are distinct. Hence the generic fibre of the parametrisation is
a finite union of 2m^2-dimensional orbits, and dim = 3m^3 - 2m^2 follows in a few lines. Dao never prints the number, so
"not found" was literally true, but the registered decision rule (section 0) reads "not present in the sources named" and
Dao 2022 is a named source. M2(c) is therefore relabelled "explicit statement plus an independent tangent-space proof of
a direct consequence of Dao App. D.3", and the (k+1)m^3 - km^2 upper bound of M5, a parameter-minus-gauge count, is not
offered as novel either.

**Prop M3 (slice-rank growth is not bounded by additivity; an example, not a corollary of M2). PROOF (referee-attacked,
survived). The slice-rank characterisation of Monarch matrices is KNOWN (Dao et al. 2022, Section 3.3, Eq. 2 and
Algorithm 1; App. D.2), not new; slice rank depends on the convention (J (x) I is a Dao Monarch matrix).** Sylvester-type classes
satisfy rank(AB) <= r_A + r_B; the Monarch slice rank does not. Take A = Pi R with R block-diagonal, all blocks the
all-ones matrix J, and B = Pi; both are Monarch (slice rank 1). AB = Pi R Pi has slices diag(S_{a'}[a,b])_{a'} = I_m,
rank m = sqrt(n) (slice rank in the Mon_LPiR convention). So no function f(r_A, r_B) independent of n bounds the slice rank of a product.

**Prop M4 (inverse; class Mon_LPiR only). PROOF (referee-attacked, survived).** Mon_LPiR is closed under inversion:
(L Pi R)^-1 = R^-1 Pi L^-1 (Pi = Pi^-1) with block-diagonal inverses. **As a set, Dao's class Pi.Mon is not inverse- or
transpose-closed** (m = 3: exact certificate results/floor_ldr_v1/dao_exact.json with the offending minors -1/216
(inverse) and -3 (transpose), see Notation [R2]); **[R3, MINOR-7] but this is an artifact of the fixed leading Pi:**
D^-1 Pi = R^-1 Pi L^-1 and D^T Pi = R^T Pi L^T lie in Mon_LPiR (Notation, PROOF [R3, gate pending]), so Dao's class is
closed under inverse and transpose up to moving the fixed Pi to the other side. Sums have slice rank <= 2 (not in
class in general). Mon_LPiR is closed under inverse but not product. For comparison: LDR is closed under product and
inverse along chained or swapped operator pairs (a groupoid, not a group) with subadditive rank growth, while
quasiseparable/HSS classes are closed under product and inverse by direct block computation and the nullity theorem (no
operator pair is involved).

**Conjecture M5 (k-fold; class Mon_LPiR).** dim Mon^k = min(m^4, (k+1)m^3 - k m^2); dense exactly at k = m = sqrt(n). Upper bound
holds for every k (Mon^k is the image of B_0 Pi B_1 Pi ... B_k, (k+1)m^3 parameters minus k diagonal junction
gauges of m^2 each). EMPIRICAL equality at (m,k) in {2,3,4,5}x{1,2}, (3,3), (4,3), (4,4), (5,4)
(results/floor_ldr_v1/monarch.json: 12/12 match to the registered prediction P1-P3).
**Status split [R2], relabelled [R3].** PROOF (referee-attacked, survived): the upper bound for every k; equality for
k = 1 (M1) and k = 2 (M2); the NECESSITY "dense requires k >= m", because (k+1)m^3 - k m^2 >= m^4 iff k >= m.
EMPIRICAL (exact finite checks: the mod-p Jacobian rank is a rigorous lower bound and min(m^4, (k+1)m^3 - km^2) a
rigorous upper bound, so a match is the exact dimension at that (m,k)): the twelve listed (m,k) and, added in repair
round 3, the eight pairs (2,3), (3,4), (5,3), (5,5), (6,1), (6,2), (6,5), (6,6) with dimensions 16, 81, 425, 625, 396, 576,
1116, 1296 (verified by the cold referee with independent code on 2026-09-30, and re-run here: results/floor_ldr_v1/
monarch_extra.json, 8/8 match). The tested (m,k) are exactly these 20 pairs, and saturation at k = m is now seen for
m = 2..6. **CONJECTURE (open):** equality for all (m,k), and in particular the SUFFICIENCY "Mon^k is dense at k = m"
(Monarch products saturate dense at depth sqrt(n)), checked at (m,k) = (2,2), (3,3), (4,4), (5,5), (6,6) only.
**Monotonicity [R3, MINOR-10]. PROOF [R3, gate pending]:** Mon^k is contained in Mon^(k+1) for every k >= 1. Reason: k = 1
is the Latin-square containment of M2; for k >= 2, Mon^k = Mon^(k-1).Mon and Mon is contained in Mon^2 by the
Latin-square step, so Mon^k is contained in Mon^(k-1).Mon^2 = Mon^(k+1); explicitly, the last factor pair (B_(k-1), B_k) = (L, R) of B_0 Pi B_1 ... Pi B_k is replaced
by (L', S, R') with L'_a[:, a+b] = L_a[:, b], R'_b[a+b, :] = R_b[a, :], S_x[a,b] = [x = a+b mod m] (indices mod m).
Checked exactly at m = 2, 3 and k = 1, 2, 3, with the shifted Latin square as negative control (r3.json monotone).
Consequences: dim Mon^k is non-decreasing in k, and if Mon^m is Zariski-dense then Mon^k is dense for every k >= m, so
the sufficiency conjecture concerns only k = m; the earlier weaker inclusion Mon^k in Mon^(k+2) (Pi Pi = I) is
subsumed. **Dao's class is not product-closed for any m >= 2 [R3, MINOR-6]. PROOF [R3, gate pending]:** for D = Pi.Mon,
D.D = {Pi L1 Pi R1 Pi L2 Pi R2} = Pi.Mon^3 (R1 and L2 do not merge), so dim(D.D) = dim Mon^3 >= dim Mon^2 = 3m^3 - 2m^2
(monotonicity and M2(c)), which exceeds dim D = 2m^3 - m^2 by m^3 - m^2 > 0; hence D.D is not contained in D. The exact
Dao ranks 16, 81, 208 (m = 2, 3, 4) are consistent with this.

**Prop M6 (Kronecker product after a fixed reordering; classes Mon_LPiR and Dao's Pi.Mon). PROOF [R3, gate pending].**
[R3, MINOR-8] This closes the earlier map cell "Kronecker: OPEN (not analysed)". Let M_s be in Mon(m_s) with slices
M_s[a,.,b,.] = u^s_{ab} (v^s_{ab})^T (s = 1, 2; M1), and let Q be the fixed permutation that reorders the lexicographic
Kronecker index ((a1,e1),(a2,e2)) to ((a1,a2),(e1,e2)) on rows and columns alike (Q depends only on (m1, m2)). Then
(Q (M1 (x) M2) Q^T)[((a1,a2),(e1,e2)), ((b1,b2),(c1,c2))] = M1[(a1,e1),(b1,c1)] M2[(a2,e2),(b2,c2)] =
(u^1_{a1 b1} (x) u^2_{a2 b2})[(e1,e2)] (v^1_{a1 b1} (x) v^2_{a2 b2})[(c1,c2)], so its slice at the block pair
((a1,a2),(b1,b2)) is (u^1 (x) u^2)(v^1 (x) v^2)^T, of rank <= 1, and by M1 the matrix Q (M1 (x) M2) Q^T lies in
Mon(m1 m2) (size (m1 m2)^2 = n1 n2). For Dao's class, Q (Pi_{m1} (x) Pi_{m2}) Q^T = Pi_{m1 m2}, hence
Q (Pi_{m1} M1 (x) Pi_{m2} M2) Q^T = Pi_{m1 m2} . Q (M1 (x) M2) Q^T lies in Pi.Mon(m1 m2). The plain lexicographic order
fails (the slices of M1 (x) M2 in the lexicographic grid have rank > 1 for generic members), so the reordering is
essential. Referee-verified exactly at (m1,m2) = (2,2), (2,3), (3,2) (cold referee, 2026-09-30); re-run here at those and
(3,3), with the lexicographic order and a random dense matrix as negative controls (results/floor_ldr_v1/r3.json
kron_reorder, 4/4 match). So, up to the fixed reordering Q, both Monarch conventions are Kronecker-closed; together with
M4 the class Mon_LPiR is inverse-friendly AND Kronecker-friendly (up to fixed permutations) and lacks only product closure.

**Prop B1 (butterfly dimension and depth). Upper bound PROOF (referee-attacked, survived); the k = 1 dimension n(L+1) is KNOWN
(identifiability analysis of Zheng-Riccietti-Gribonval, SIMODS 2023), not new; equality for k >= 2 EMPIRICAL.** Butterfly B =
F_L ... F_1 with F_j supported on (i,i),(i,i xor 2^(j-1)), L = log2 n, 2nL parameters. A diagonal D between adjacent
factors preserves both supports (F_{j+1} F_j = (F_{j+1} D^-1)(D F_j): D scales the rows of F_j and D^-1 the columns of
F_{j+1}); a k-fold product of butterflies has kL - 1 junctions, so
**dim <= n(kL+1)**. [R3, MINOR-16] The repo code (butterfly_suite) multiplies the factors in the opposite stride order,
F_1 F_2 ... F_L, i.e. it computes the transpose variety of B (the supports of the F_j are symmetric); dimensions are
unaffected. Hence a dense (Zariski-dense) product needs k >= (n-1)/L butterflies: **[R2] this is a NECESSARY
count only** (PROOF (referee-attacked, survived) from the upper bound); that k = ceil((n-1)/L) butterflies SUFFICE for general n is not proved
(EMPIRICAL at n = 4, 8, 16, where the dense dimension is first reached at k = 2, 3, 4 = ceil((n-1)/L), and, added in
repair round 3, at n = 32, k = 7 = ceil(31/5): cold-referee-verified and re-run in results/floor_ldr_v1/
butterfly_extra.json). **Prior art on sufficiency [R3, MINOR-14]:** Kaleidoscope (Dao et al. 2020), Lemma E.3 (locator as
reported by the cold referee), shows that every n x n matrix lies in (B B*)^(2n-2), an O(n) sufficiency bound (4n - 4
alternating-orientation butterflies, a different variety from the fixed-orientation B^k analysed here); the exact minimal
depth for B^k is open, between the necessary (n-1)/L and O(n). The Jacobian
rank equals n(kL+1) (or n^2 once that is reached) for n = 4, 8, 16 and k up to saturation (results/.../butterfly.json,
9/9 match P4). Since n(2L+1) > n(L+1), the product of two butterflies is not a butterfly: EMPIRICAL (an exact finite check; the
inequality dim B^2 > dim B is rigorous at the tested n by the section-1 method) for n = 4, 8, 16 and, [R3] in
butterfly_extra.json and by the cold referee's independent computation, n = 32 (ranks 192, 352, 992 at k = 1, 2, 6 and
dense 1024 at k = 7). For general n non-closure needs only one explicit product outside the class and the general equality
is CONJECTURE; closure or non-closure under sums is not claimed. Retained: the upper
bound n(kL+1) and the (n-1)/L necessary depth count. **Prop B2 (Kronecker; classical, Van Loan's Kronecker factorisation of the
FFT, not new).** B_{n1} (x) B_{n2} = (B_{n1} (x) I)(I (x) B_{n2}) has factor
supports inside the butterfly supports of size n1 n2 (hypercube ordering), so butterflies are closed under Kronecker
product with L = L1 + L2. PROOF (referee-attacked, survived; one line).

**Prop U (projector displacement identity). Elementary; for singular operators cf. the generalized displacement of
Kailath-Sayed. PROOF (referee-attacked, survived).** For an idempotent P, PM - MP = PM(I-P) - (I-P)MP, so
rank[P,M] = rank(M_12) + rank(M_21) (blocks in disjoint rows and columns). **WITHDRAWN (cold referee):** the earlier
claim that HSS/quasiseparable classes are special cases of Sylvester displacement rank with the pair (P,P).
Quasiseparable order bounds the lower block M21 and the upper block M12 SEPARATELY, and HSS bounds block rows
M[I,I^c]; neither equals rank[P_I,M] (they agree only up to a factor 2 in r). Counterexample **for a nontrivial
idempotent, 0 < rank P < n [R2: hypothesis stated]** that M -> rank((I-P) M P) is not rank(AM - MB) for any pair (A,B)
(both diagonal blocks, of sizes m1 = rank P >= 1 and m2 = n - m1 >= 1, are nonempty; for P = 0 or P = I the map is
identically 0 and the question is void): M = I gives rank(A - B) = rank((I-P)P) = 0, so A = B; M = P gives
[A,P] = 0, so A = A1 + A2 is block diagonal with respect to im P + ker P; an M with only the upper-right block N
nonzero has (I-P) M P = 0, so A1 N = N A2 for every m1 x m2 matrix N, which (N = E_ij) forces A1 = cI, A2 = cI with
the same c, i.e. A = cI and [A,M] = 0 for all M; but (I-P) M P != 0 for an M with nonzero lower-left block
(available since m1, m2 >= 1), contradiction. The identity rank[P,M] = rank(M_12) + rank(M_21) itself holds for every
idempotent. What survives is only that identity. The operator is singular (kernel = the
block-diagonal algebra = commutant of P), which is why diagonal blocks are free descriptors; the Toeplitz operators
(Z_e, Z_f), e != f, are bijective on M_n (disjoint spectra). Checked: rank [P,M] = 3 = 2 + 1 (n=8 exact).

**Theorem R (rank-profile calculus for Sylvester families). KNOWN.** [R3, MINOR-5] Pan (2013),
arXiv:1303.0353, Thm 17 and Cor 18 (citing Pan 2000 and Pan 2001, Sec 1.5; locators as reported by the cold referee), state
exactly this general-operator calculus: sum d_{A,B}(aM+bN) <= d(M)+d(N); the transpose rule; product
d_{A,C}(MN) <= d_{A,B}(M)+d_{B,C}(N); inverse d_{A,B}(M^-1) = d_{B,A}(M). The earlier claim that the joint statement was
"not found stated" was WRONG (cold referee, 2026-09-30); nothing in this paragraph is new, the identities below are the
derivation. For chain-compatible pairs (A,B)(B,C), nabla_{A,C}(MN) = nabla_{A,B}(M) N + M nabla_{B,C}(N). **[R2] "Ranks
add" means only the SUBADDITIVE bound** rank nabla_{A,C}(MN) <= rank nabla_{A,B}(M) + rank nabla_{B,C}(N), and likewise
rank nabla_{A,B}(M+N) <= rank nabla_{A,B}(M) + rank nabla_{A,B}(N) for sums (same pair); **equality is not universal**:
for N = M^-1 with the swapped pair, nabla_{A,A}(MM^-1) = nabla_{A,A}(I) = 0 although both summands have the same rank
r = rank nabla_{A,B}(M) (which may be positive), and N = -M kills the sum. Equality holds for generic members in the
checked case (Toeplitz, n = 6, ranks 2 + 2 = 4). Inverses swap the pair with EQUAL rank, exactly
(nabla_{B,A}(M^-1) = -M^-1 nabla_{A,B}(M) M^-1, M invertible).
Instances: Toeplitz/Hankel/Cauchy/Vandermonde-like [(Z_e,Z_f), (D_x,D_y), (D_x,Z_f); Kailath-Sayed 1995/1999, Kailath-Chun 1994, Pan],
block-diagonal and circulant algebras [rank 0]. **WITHDRAWN (cold
referee):** the claim that Sylvester is "the only additive-product mechanism of linear-map rank-profile type". The map
M -> (I-P) M P also has additive rank under products, (I-P)MNP = (I-P)MP . PNP + (I-P)M(I-P) . (I-P)NP, and is not of
Leibniz form. What Bresar-type theorems give is narrower: linear maps on all of M_n satisfying an exact rule
d_AC(MN) = d_AB(M) N + M d_BC(N) are of Sylvester form (Bresar 1991; Hvala 1998). Monarch slice maps violate
additivity (M3), which is a separate fact.
NOT covered: (0) semiseparable/quasiseparable/HSS (additivity there is the direct block computation of section 4, not Sylvester displacement); (1) strongly admissible H-matrices (product blocks combine C_sp block ranks: Grasedyck-Hackbusch 2003,
Hackbusch 2015; cited, not re-read); (2) the sparse part of sparse+low-rank; (3) Monarch/butterfly; (4) Kronecker
products (Prop K); (5) pair-mismatched products (a groupoid, not a group); (6) descriptor size: membership is
captured, the O(nr) storage needs the generator theorems of Eidelman-Gohberg and Chandrasekaran-Gu.

**Prop K (Kronecker under the Kronecker operator pair).** (i) *Upper bound.* PROOF (referee-attacked, survived). If
nabla_{A1,B1}(M1) = X (column space U, dim r1) and nabla_{A2,B2}(M2) = Y (column space W, dim r2), then
nabla_{A1(x)A2, B1(x)B2}(M1(x)M2) = X (x) A2M2 + M1B1 (x) Y has column space inside U(x)F^{n2} + F^{n1}(x)W, so its rank is
**<= r1 n2 + r2 n1 - r1 r2**.
(ii) *[R2] Exact unbounded family; the bound (i) is attained. PROOF (referee-attacked, survived).* Let (Z_e x)_i = x_{i-1}, (Z_e x)_0 = e x_{n-1}, and
take e != f, e' != f', ee' != ff' in any field (e.g. 2, 3, 5, 7). For all n1, n2 >= 2,
**rank nabla_{Z_e(x)Z_e', Z_f(x)Z_f'}(I_{n1} (x) I_{n2}) = n1 + n2 - 1**, while rank nabla_{Z_e,Z_f}(I_{n1}) =
rank nabla_{Z_e',Z_f'}(I_{n2}) = 1. Proof: write Z_e = S + e E_1, Z_f = S + f E_1 (S = Z_0 the nilpotent shift,
E_1 = e_0 e_{n1-1}^T) and Z_e' = S' + e' E_2, Z_f' = S' + f' E_2 (E_2 = e_0 e_{n2-1}^T). The S(x)S' terms cancel and
nabla(I(x)I) = Z_e(x)Z_e' - Z_f(x)Z_f' = (e'-f') S(x)E_2 + (e-f) E_1(x)S' + (ee'-ff') E_1(x)E_2. In the lexicographic
basis e_i (x) e_p: S(x)E_2 has entries at (row (i,0), column (i-1,n2-1)), i = 1..n1-1; E_1(x)S' at (row (0,p), column
(n1-1,p-1)), p = 1..n2-1; E_1(x)E_2 at (row (0,0), column (n1-1,n2-1)). The three row sets are pairwise disjoint and so
are the three column sets, so the sum is a weighted partial permutation matrix and its rank is the number of nonzero
weights: [e'!=f'](n1-1) + [e!=f](n2-1) + [ee'!=ff'] = n1 + n2 - 1. QED. With r1 = r2 = 1 this equals
r1 n2 + r2 n1 - r1 r2, so (i) is tight, and the displacement rank of a Kronecker product of two rank-1 members under
the Kronecker pair is **not bounded by any function of (r1, r2) independent of (n1, n2)**. Exact check (Fractions):
the formula holds for n1, n2 in 2..6 (e,f,e',f' = 2,3,5,7) and for three degenerate parameter sets that switch terms off
(results/floor_ldr_v1/ldr_kron_family.json, 52/52).
*Scope, stated precisely.* Non-closure has a proof (PROOF (referee-attacked, survived)) only for the Kronecker operator pair
(A1(x)A2, B1(x)B2) built from the Toeplitz-type factor pairs (Z_e,Z_f), (Z_e',Z_f'). It is not a statement about other
operator pairs on the product space: I_{n1 n2} has displacement rank 1 under a single-level pair (Z_g, Z_h) of size n1 n2.
**[R3, MINOR-11] The open question must be posed for reconstructible operators.** Without such a requirement it is answered
trivially: for A = B = I every matrix has displacement 0 (rank 0), and such a pair carries no information (the
displacement does not determine the matrix). The meaningful question restricts to pairs (A,B) chosen from the factor
operators (not the Kronecker pair) whose operator M -> AM - MB is injective, equivalently bijective on N x N matrices
(Sylvester: spec A and spec B disjoint): does some such pair keep the displacement rank of M1 (x) M2 bounded by a function
of r1, r2 alone? This remains OPEN; nothing here excludes it. The K(ii) family itself is legitimate in this sense.
**PROOF [R3, gate pending]:** for (e,f,e',f') = (2,3,5,7) the Kronecker pair has disjoint spectra for all n1, n2, because
an eigenvalue lam.mu of Z_e (x) Z_e' satisfies (lam.mu)^(n1 n2) = e^(n2) e'^(n1) (the eigenvalues of Z_e are the n1-th
roots of e), an eigenvalue of Z_f (x) Z_f' satisfies (lam.mu)^(n1 n2) = f^(n2) f'^(n1), and 2^(n2) 5^(n1) != 3^(n2) 7^(n1) by
unique factorisation; so the displacement operator is bijective. Exact check of rank N^2 at (n1,n2) in {(2,2), (2,3),
(3,2), (3,3)}; negative control (e,f,e',f') = (2,3,3,2) at (2,2) has common spectrum and rank 8 < 16; trivial control
A = B = I gives rank 0 (r3.json reconstructible). Other parameter sets may have common spectra; K(ii) does not need
injectivity, and this note only shows that the exhibited family is not an artifact of a degenerate pair.
(iii) *Generic Toeplitz factors, EMPIRICAL only.* For random Toeplitz factors (r = 2, pair (Z_e,Z_f)) the exact rank is
2n1+2n2-4 at (n1,n2) in {3x3, 3x4, 4x4, 4x5, 5x5} (8, 10, 12, 14, 16; prediction P5 with c = 4). These finite checks
do not prove any asymptotic statement; the unboundedness proof is (ii), and the growth rate for generic factors is not
proved.
(iv) *[R2] Quasiseparable Kronecker, exact unbounded family. PROOF (referee-attacked, survived).* Order of M (N x N): ord(M) = max over cuts t of
max(rank M[t:, :t], rank M[:t, t:]) (lower and upper off-diagonal blocks). Let J_k be the k x k nilpotent shift,
J_k[i+1,i] = 1 and all other entries 0, B_k = I_k + J_k. Then ord(J_k) = ord(B_k) = 1 (k >= 2) while, in the lexicographic
Kronecker order (index (i,p) -> ik + p), **ord(J_k (x) B_k) = k**. Proof: J_k and B_k are lower bidiagonal, and their
block M[t:, :t] has the single possible nonzero entry (t, t-1), so its rank is 1 and the upper block is 0; hence order 1.
K = J_k (x) B_k is block sub-diagonal: its only nonzero k x k blocks are K[(s+1),(s)] = B_k, s = 0..k-2. At the cut
t = ks + u (0 <= u < k) the lower-left block K[t:, :t] contains the piece B_k[u:, :] (rows u..k-1 of block (s,s-1)) and
the piece B_k[:, :u] (columns 0..u-1 of block (s+1,s)), in disjoint rows and columns, so its rank is
rank B_k[u:, :] + rank B_k[:, :u] = (k-u) + u = k where both pieces exist (B_k is unit lower triangular, hence
invertible), and the upper block is 0. At t = k (s = 1, u = 0, needs k >= 2) the block K[k:, :k] contains the whole
block B_k, of rank k. Hence ord(K) = k exactly. QED. So the quasiseparable order of a Kronecker product of two order-1
matrices is not bounded by any function of the factor orders (even J_k (x) I_k, orders 1 and 0, has order k). Exact
check k = 2..6 (results/floor_ldr_v1/qs_kron_exact.json, 5/5). The earlier random order-1 factors gave k+1
(EMPIRICAL, k = 3..6; ldr.json), consistent with, and weaker than, this family.
Butterfly (B2) is Kronecker-closed; Monarch, in both conventions, is Kronecker-closed after the fixed reordering Q (M6,
[R3]); block-diagonal algebras (index bookkeeping) and circulants of coprime sizes (CRT reindexing) are Kronecker-closed
only after a Kronecker-compatible reordering (checked in r3.json closed_algebras; used in the section-7 target-cell note);
LDR (for the Kronecker pair, (ii)) and quasiseparable (lexicographic order, (iv)) are not, by the explicit families.

## 3. The controlled map (classes x operations)

Legend: rk = rank parameter, n = size, m = sqrt(n). Cells: statement [label; source]. Native-descriptor operation
counts; dense->class cells charge the conversion.

| Class | matvec | product of two members | inverse | Kronecker | sum | dense -> class | precision | strongest specialist baseline |
|---|---|---|---|---|---|---|---|---|
| LDR Toeplitz/Hankel-like, pair (Z_e,Z_f) | O(rk n log n) [KNOWN; Pan 2001] | in class along matching pairs, rank <= r1+r2 (subadditive bound; equality not universal, holds in the checked generic case n=6: 4) [KNOWN: Pan 2013 Thm 17 / Cor 18; checked] | same rank, swapped pair [KNOWN: Pan 2013; checked] | Kronecker pair: rank <= r1n2+r2n1-r1r2, attained at r1=r2=1 (I(x)I: n1+n2-1) [Prop K (i),(ii)]; generic Toeplitz 2n1+2n2-4 EMPIRICAL only | rank <= r1+r2 (same pair) [KNOWN: Pan 2013] | nabla costs O(n^2); rank via range finder O(n^2 rk); generator->dense O(rk n^2) | problem- and algorithm-dependent: GKO-type O(n^2) solvers with pivoting are usually reported stable [Gohberg-Kailath-Olshevsky 1995], but generator growth can occur (Sweet-Brent 1995; Gu 1998) and no general weak-stability guarantee is claimed for superfast solvers [cited, not re-verified this round] | FFT convolution (Toeplitz); dense GEMM |
| LDR Cauchy-like (D_x,D_y) and Vandermonde-like (D_x,Z_f) | Vandermonde-like: multipoint evaluation/interpolation O(rk n log^2 n), numerically unstable for real nodes; Cauchy-like: FMM-type approximate evaluation (accuracy set by a truncation tolerance) or exact O(rk n^2) [cited, not re-verified this round] | as above, nodes must chain | as above | as Prop K | as above | as above | real-node Vandermonde and Cauchy matrices are exponentially ill-conditioned (Gautschi; Beckermann), roots of unity (the DFT) being a well-conditioned special case rather than a general rule [cited, not re-verified this round] | FMM (Cauchy kernel); FFT (the DFT is Vandermonde-like at roots of unity, not Cauchy-like) |
| Butterfly (L = log n sparse factors) | 2n log n [Dao 2019] | not in class for n = 4, 8, 16, 32 (exact finite check, rigorous at those n; general n CONJECTURE); k-fold dim <= n(kL+1) (upper bound PROOF (referee-attacked, survived)), equality EMPIRICAL for n <= 16 and n = 32 (k = 1, 2, 6, 7), depth (n-1)/L NECESSARY only, O(n) sufficiency known [Kaleidoscope Lemma E.3: every matrix in (B B*)^(2n-2)] [B1] | inverse lands in B*: product of inverse factors in reversed stride order, a different equal-dimension variety [KNOWN: Kaleidoscope App. H, p.31, as reported by the cold referee] | closed, L = L1+L2 [B2] | not claimed (neither closure nor non-closure is proved) | exact recovery if promise holds: Zheng-Riccietti-Gribonval 2023; approximate: Li et al. 2015 [cited, not re-verified this round] | orthogonal factors stable | FFT for DFT; dense GEMM |
| Monarch, class Mon_LPiR = {L Pi R} (Dao's own class Pi.Mon differs under products, see Notation) | 2 n^(3/2) [Dao 2022] | not in class for every m >= 2: dim Mon^2 = 3m^3-2m^2 [M2, a direct consequence of Dao App. D.3]; slice rank m [M3, characterisation KNOWN]; dim Mon^k <= (k+1)m^3-km^2 PROOF (referee-attacked, survived), equality PROOF (referee-attacked, survived) for k <= 2, EMPIRICAL at the 20 tested (m,k) listed in M5, CONJECTURE in general [M5]; Dao's class also not product-closed for any m >= 2 [M5, PROOF [R3, gate pending]], Dao-class products 16, 81, 208 at m = 2, 3, 4 [EMPIRICAL, exact] | closed for Mon_LPiR [M4]; Dao's class closed up to moving the fixed Pi (D^-1 Pi, D^T Pi in Mon_LPiR) [M4, PROOF [R3, gate pending]]; the exact certificate dao_exact.json shows non-closure only as a set | closed after a fixed reordering Q, in both conventions: Q(M1(x)M2)Q^T in Mon(m1 m2); plain lexicographic order fails [M6, PROOF [R3, gate pending]; referee-verified exact at (2,2), (2,3), (3,2)] | slice rank <= 2, not in class | projection = m^2 rank-one SVDs, O(n^(5/2)); exact if in class [Dao 2022] | block SVDs stable | FFT (two-stage Cooley-Tukey), dense GEMM |
| HSS / quasiseparable / semiseparable (off-diagonal block ranks; no operator pair, not a Sylvester class) | O(rk n) [Chandrasekaran-Gu; Eidelman-Gohberg] | in class, order <= r1+r2, sharp (order 1 x 1 -> 2, n=8 check), cost O(n rk^2) [KNOWN; direct block calculation] | in class, same off-diagonal ranks (nullity theorem; Strang-Nguyen 2004; Eidelman-Gohberg) [KNOWN; checked order 1] | not closed (lexicographic order): J_k(x)(I_k+J_k) has order k, factors order 1 [K(iv), PROOF (referee-attacked, survived)]; generic k+1 EMPIRICAL | order <= r1+r2 [checked 2] | O(n^2 rk) randomized/SVD sweep | ULV/orthonormal generators stable [Chandrasekaran et al. 2006; cited, not re-verified this round] | banded solvers, 1-D FMM |
| H / H^2 (strong admissibility) | O(rk n log n) / O(rk n) [Hackbusch 2015] | block rank grows by C_sp; truncated product approximate [Grasedyck-Hackbusch 2003; Borm 2010; cited, not re-verified this round] | approximate H-inverse, exponential convergence [Bebendorf-Hackbusch 2003; Faustmann-Melenk-Praetorius 2015; cited, not re-verified this round]; not exactly rank-preserving | CONJECTURE (unproved here): not closed with bounded block rank | needs truncation | ACA O(rk^2 n log n) from kernel entries; O(n^2 rk) from dense | error control by truncation | FMM, multigrid, sparse LU |
| sparse + low-rank | nnz + 2 rk n | low-rank part <= r1+r2 exact; sparse part S1S2 (bandwidth b1+b2, or degree d1d2) | Woodbury; S^-1 dense unless S banded (then semiseparable) | cross term S1(x)U2V2^T has rank rank(S1) r2 and nnz(S1) n2^2 [elementary]; whether the Kronecker product admits a bounded sparse+low-rank form is unproved [CONJECTURE: not closed] | adds | no exact recovery in general (Candes et al. 2011 needs incoherence; cited, not re-verified this round) | Woodbury cancellation | sparse LU + Woodbury |

Sparse+low-rank product (PROOF (referee-attacked, survived), direct expansion): (S1 + U1V1^T)(S2 + U2V2^T) = S1S2 + [S1U2 + U1(V1^TU2), U1][V2, S2^TV1]^T.

## 4. Growth laws for repeated multiplication (composition-friendliness)

Descriptor size after q products of members:
- LDR (compatible pair chain): rank <= sum r_i (PROOF (referee-attacked, survived) by induction from the product identity; upper
  bound, linear in q). **[R2] The generic-growth statement "equality, 2q for Toeplitz" is CHECKED at q = 2 only** (n = 6: 4); it was not
  tested for q >= 3 and is not proved. Size O(rk n).
- HSS/quasiseparable: (MN)_12 = M_11 N_12 + M_12 N_22 gives r12(MN) <= r12(M) + r12(N) (upper bound, linear in q);
  attained in the checked example (order 1 x 1 -> 2, n = 8), not universal.
- Monarch (class Mon_LPiR; Dao's class grows faster, see Notation): dimension <= (k+1)m^3 - k m^2 (PROOF (referee-attacked,
  survived), every k); equality PROOF (referee-attacked, survived) for k <= 2, EMPIRICAL at the 20 (m,k) pairs listed in M5;
  slope n^(3/2) - n per product, but the shape changes (B Pi B Pi B ...) rather than per-factor ranks adding.
  **[R2] "Dense requires k >= m = sqrt(n)" is PROOF (referee-attacked, survived) (necessary); "Monarch saturates dense at
  depth sqrt(n)" is the CONJECTURED sufficiency of M5** (checked at m = 2..6 with k = m; [R3] density at k = m then
  persists for every k >= m by monotonicity).
- Butterfly: dimension <= n(kL+1) (PROOF (referee-attacked, survived)), equality EMPIRICAL for n <= 16 and n = 32: slope
  <= nL per butterfly. **[R2] Dense requires k >= (n-1)/L butterflies: a NECESSARY count only; sufficiency ("depth
  n/log n") is not proved; an O(n) bound is known (Kaleidoscope Lemma E.3).**
- H-matrices: C_sp-multiplicative per product unless truncated (KNOWN); no exact bounded growth.
Amortisation (cost arithmetic and heuristic, not a theorem): a chain of q LDR/HSS factors costs (sum r_i) times n-ish per
matvec against n^2 for one dense matvec, winning while sum r_i is below about n / log n [heuristic]; a Monarch chain of
length q costs q * 2 n^(3/2) and beats one dense matvec only while q < sqrt(n)/2 (a statement about factor-by-factor
application, independent of whether the product class saturates dense).

## 5. Attack list (self-attack of section 2)

1. M2 at m = 2: numerics give dimension 16 = n^2, consistent with codimension m^2(m-1)(m-2) = 0.
2. Is R1 L2 an arbitrary S? Yes (take L2 = I).
3. Is the Jacobian rank mod p = 32003 a lower bound over Q? Yes (rank at an integer point mod p <= rank there over Q <=
generic rank); together with the rigorous gauge upper bound it pins the dimension at each tested size.
4. Convention (CORRECTED by referee): Dao's P L P^T R = Pi.Mon_LPiR differs from Mon_LPiR by a fixed row permutation
that does not commute with products; only single-matrix statements transfer (see Notation, dao.json).
5. Field: the differential rank is the same over R and C (char 0).
6. Butterfly: only the stride order 1,2,..,2^(L-1) repeated was tested; other interleavings are other varieties.
7. [R3] Monotonicity: Mon^k is contained in Mon^(k+1) for every k (Latin-square step on the last factor pair, M5); the
   weaker Mon^k in Mon^(k+2) (I = Pi Pi) that was asserted before is subsumed.
8. M2 genericity (REPAIRED, gate 5): the earlier "nonempty Zariski-open condition" is replaced by the explicit witness W
   (L = R = I, S_{a'}[a,b] = 2^(a a' b)) with a proof of the three conditions in characteristic 0; no appeal to "a random
   point exists"; finite-field ranks are supplementary.
9. [R2] Prop K (ii): the proof is an exact partial-permutation count valid over any field when the three weights are
   nonzero; the attack "I_N has rank 1 under another pair" is accepted and written into the scope.
10. [R2] Prop K (iv): order is defined as max of lower/upper block ranks (as in the checker); the result is for the
   lexicographic Kronecker order only; other orders are not claimed.
11. [R2] Dao certificate: the class test "slices of Pi X have rank <= 1" is M1; the certificate is re-derived by a
   closed-form path that shares no code with the generator (verify_dao_exact).
12. [R2] Theorem R equality is not universal (N = M^-1, N = -M); nothing is claimed beyond the subadditive bound and the
   checked generic case.
13. [R3] Prop M6 (Kronecker after reordering): "is Q independent of the matrices?" - yes, it depends only on (m1, m2);
   "does Q preserve the slice structure?" - it maps the slice at ((a1,a2),(b1,b2)) to (u1 (x) u2)(v1 (x) v2)^T by direct
   index bookkeeping; the plain lexicographic order and a random dense matrix are the negative controls (r3.json).
14. [R3] Monotonicity: "the Latin-square step needs block-diagonal L, R" - it acts on the last two block-diagonal factors
   only and leaves the head B_0 Pi ... Pi B_(k-2) untouched; checked at m = 2, 3, k = 1, 2, 3 with the shifted square as
   negative control.
15. [R3] Dao's class not product-closed for every m >= 2: rests on M2(c) (any m) and monotonicity (any m); the finite exact
   ranks 16, 81, 208 are consistent with it.
16. [R3] Pi-side closure: a two-line identity; checked on the stored certificate with Pi D^-1 and Pi D^T (the offenders) as
   negative controls.
17. [R3] Reconstructibility of the K(ii) pair: unique factorisation of e^(n2) e'^(n1) versus f^(n2) f'^(n1) for (2,3,5,7);
   other parameter sets can have a common spectrum (the (2,3,3,2) negative control does), which K(ii) does not need.
18. [R3] Every citation locator added in round 3 (Dao Sec 3.3, App. D.2, D.3, C, p.22; Pan 2013 Thm 17 and Cor 18;
   Kaleidoscope Lemma E.3 and App. H p.31) is as reported by the cold referee and was not re-read by the repair agent.

## 6. Novelty assessment (focused search, web only; repaired in round 3)

**H0 STANDS.** The registered decision rule (section 0) accepts H1 only if a candidate is "not present in the sources
named"; none survives it. M2(c) is a direct consequence of Dao et al. 2022 App. D.3 (the novelty note after M2), and every
other candidate is KNOWN or elementary (below). No new theorem is claimed; the unit's deliverable is the controlled map
(section 3), with the exact statements of section 2 as its evidence.

KNOWN (an earlier draft listed several as new): the Monarch slice-rank characterisation (Dao et al. 2022, Sec 3.3, Eq. 2,
Algorithm 1, App. D.2 [locators as reported by the cold referee]); the butterfly k = 1 dimension n(L+1)
(Zheng-Riccietti-Gribonval 2023); butterfly Kronecker closure (Van Loan FFT factorisation); the projector commutator
identity (Kailath-Sayed generalized displacement); the Kronecker displacement bound r1n2+r2n1-r1r2 (elementary); the n^(3/2)
parameter bound for Monarch products (Dao et al. 2022 App. C, proof of Thm 4); the generic dimension 3m^3 - 2m^2 of the
MM* class (implicit in Dao App. D.3, see M2); Theorem R, the rank-profile calculus for Sylvester families (Pan 2013 Thm 17
and Cor 18, citing Pan 2000 and Pan 2001 Sec 1.5); LDR/HSS additive product laws and inverse-rank preservation;
multilevel-Toeplitz Kronecker growth (folk). **Attribution correction [R3, MINOR-4]:** Dao et al. 2022 and Kaleidoscope do
NOT state that Monarch is "not product-closed" (Dao 2022 states closure under (conjugate) transposition, p.22); the
non-closure here follows from the dimension count of M2 and is attributed to M2, not to Dao.
Not found in a limited web search (MathSciNet not reached), hence only "not found", not "absent": the k >= 2 butterfly
dimensions n(kL+1) (equality, EMPIRICAL only). The (k+1)m^3 - km^2 upper bound is a parameter-minus-gauge count and is
not offered as novel. [R2] The exact families added in repair round 2 (Prop K (ii) I(x)I, (iv) J_k(x)(I_k+J_k), the Dao
certificate) and the [R3] additions (M6, monotonicity, the Pi-side closure, the reconstructibility note) are elementary
computations and are not claimed as novel.

## 7. Verdict

**H0 STANDS (repair round 3).** The pre-registered rule ("H1 accepted iff ... not present in the sources named") is not
met: M2(c) is a direct consequence of Dao et al. 2022 App. D.3 (section 6), and every other candidate is KNOWN or
elementary. Status: the deliverable is the controlled MAP (section 3), real work since it fixes what is known and where
each statement is exact, conditional or open; no new theorem is claimed. Conjectures M5 (equality for k >= 3; saturation
at k = m) and B1-sufficiency remain open. Beyond bare citation the map rests on M1, M2(c) (an explicit statement plus an
independent tangent-space proof of a direct consequence of Dao App. D.3), the containment, M3, M4 (Mon_LPiR
inverse-closed), the B1 upper bound, B2, the U identity for every idempotent, the subadditive Sylvester calculus R (KNOWN,
Pan 2013), K (i), (ii), (iv), and the [R3] additions (M6; monotonicity Mon^k in Mon^(k+1); Dao's class not product-closed
for any m >= 2 and closed under inverse and transpose up to moving the fixed Pi; reconstructibility of the K(ii) pair).
Text existing before round 3 survived the cold referee of 2026-09-30; the [R3] additions are ungated.
So what: (i) LDR (Sylvester displacement rank; operator pairs) has subadditive rank growth under product along chained
pairs (a groupoid, not a group), exact rank preservation under inverse with the swapped pair, and non-closure under
Kronecker for the Kronecker operator pair (K (ii)). (ii) Quasiseparable/HSS (off-diagonal block ranks; there is NO operator
pair - Prop U is WITHDRAWN and Theorem R covers only Sylvester families) has additive order under product and
order-preserving inverse, by direct block computation and the nullity theorem, and non-closure under Kronecker in
lexicographic order (K (iv)). Classes (i) and (ii) share these properties by DIFFERENT mechanisms; no unification is
claimed. (iii) Sparse-factor product classes (butterfly, Monarch) are Kronecker-closed up to a fixed reordering (butterfly
B2; both Monarch conventions M6); Mon_LPiR is inverse-closed, and Dao's class is inverse-closed up to moving the fixed Pi
(M4; the dao_exact certificate shows only that the fixed-Pi SET is not inverse-closed, so it is no loss of
inverse-friendliness); what both lack is product closure, with product dimension growing by at most n^(3/2) - n per
factor (Monarch, exact for k <= 2, equality EMPIRICAL at the 20 tested (m,k)) or nL per factor (butterfly, upper bound).
So Mon_LPiR is inverse-friendly AND Kronecker-friendly, up to fixed permutations. **Downgraded [R2]:** dense saturation at
depth sqrt(n) for Monarch is M5's CONJECTURED sufficiency (only the necessity k >= m is proved; density at k = m persists
for k >= m by monotonicity [R3]); depth n/log n for butterflies is only the NECESSARY count (n-1)/L (sufficiency unproved,
checked n <= 16 and n = 32; an O(n) bound is known, Kaleidoscope Lemma E.3); generic Toeplitz displacement-rank growth 2q
is checked at q = 2 only. Established non-closures are AT LEAST: Mon_LPiR under product (M2, every m >= 2); Dao's class
under product (every m >= 2, via monotonicity and M2) and, as a set with the leading Pi fixed, under inverse and
transpose (certificate); butterflies under product at n = 4, 8, 16, 32 (exact finite check, rigorous at those n); LDR
under Kronecker for the Kronecker operator pair (K (ii)); quasiseparable under Kronecker in lexicographic order
(K (iv)); the rest is unproved.
**The target cell, with an expressivity qualifier [R3, MINOR-9].** Closure alone is cheap: block-diagonal algebras and
circulants are closed under product and inverse with zero descriptor growth and, after a fixed reordering (index
bookkeeping for block-diagonal algebras; the CRT reindexing for circulants of coprime sizes; checked in r3.json
closed_algebras), under Kronecker product as well. The interesting cell is "expressive AND closed": a class closed under
product, inverse and Kronecker with bounded growth that is also expressive (informally: not confined to a small
commutative or semisimple subalgebra such as circulants or block-diagonal matrices; not formalised here). SPECULATION:
plausibly no class in the map fills it. That combination is the target for a new representation (AGENTS.md section 11,
question 2), and nothing here exhibits one.
Cost honesty: statements concern descriptor dimension and rank, not runtime; dense->class conversion is O(n^2) or worse
and dominates single uses (Monarch dense projection O(n^(5/2)) exceeds a dense matvec); no wall-clock claim.
Next step (registered rule applied: H0 stands, the map is the deliverable, and the next step is the most promising gap in
it): attack the empty cell "expressive AND closed under product, inverse and Kronecker with bounded growth": (a) prove M5
for all k by a tangent induction on the alternating chain B C B C ... (the k = 2 case is the argument of Dao App. D.3;
k >= 3 is open); (b) test whether a lattice of Sylvester operators (per-axis operators, multilevel Toeplitz) gives bounded
growth under Kronecker with a tensor-structured descriptor; (c) constructive obligation: a class strictly between
Monarch and dense with per-product growth O(n) rather than n^(3/2), closed under inverse, costed against Mon^k.

## 8. Repair round 2 (after gate 5, Stateful Codex referee, 2026-09-29)

[R3 note: statuses in this section are as of repair round 2. The cold no-memory referee of 2026-09-30 has since attacked
all of this text and found no fatal error (section 9), so "not re-gated" below is historical.]

Reviewer provenance: Stateful Codex (gate 5): algebraic core survives, all four earlier result JSON files reproduce, not
passed unchanged; the six requested repairs are applied below. The cold no-memory referee accepted the scoped cores
earlier. Predictions R2-P1..R2-P4 (docstring of code/floor_ldr_checks.py) were derived from the proofs and written
before the four new suites were run; all four matched (each suite runs in under 5 s, single thread). Nothing in
section 0 (the hashed registration) was edited.

| # | Requested repair | Applied | Evidence |
|---|---|---|---|
| 1 | M2(c): witness and field | explicit W (L = R = I, S_{a'}[a,b] = 2^(a a' b)); conditions (i)-(iii) proved; statement restricted to characteristic 0; mod-p only supplementary | m2_witness.json: rank 16, 63, 160, 325 (mod 32003, m = 2..5); exact over Q for m = 2, 3 |
| 2 | Dao inverse/transpose certificate | exact integer blocks, Fractions, offending minors -1/216 (inverse) and -3 (transpose); independent verifier | dao_exact.json, verify_dao_exact |
| 3 | Downgrade M5 / B1 / summary | Monarch depth sqrt(n) = CONJECTURED sufficiency (necessity proved); butterfly depth (n-1)/L = NECESSARY only; Toeplitz 2q checked at q = 2 only | sections 2 (M5, B1), 3, 4, 7 |
| 4 | Prop K / quasiseparable | (a) ord(J_k (x) (I_k + J_k)) = k, factors order 1, proved; (b) rank nabla(I (x) I) = n1 + n2 - 1 under the Kronecker pair with rank-1 factors, proved, bound (i) attained; non-Kronecker-pair statement left OPEN | qs_kron_exact.json (k = 2..6), ldr_kron_family.json (52/52) |
| 5 | Prop U / Theorem R | 0 < rank P < n stated in the counterexample; "ranks add" = subadditive bound, equality not universal | section 2 |
| 6 | Status, gate history, ledger | header rewritten; ledger row floor_ldr_v1-gate5-repair | experiments/ledger.jsonl |

Not applied: none of the six is declined. Left open by design: whether some non-Kronecker operator pair keeps the
displacement rank of a Kronecker product bounded (K scope note).

**What survived attack (labels per section 1; scopes as stated, characteristic 0 for dimension statements; [R3]
"established" replaced because the [R2] text was ungated when this section was written).** M1 (dim Mon_LPiR = 2m^3 - m^2);
M2 (dim Mon^2 = 3m^3 - 2m^2, dense iff m = 2, so Mon_LPiR is not product-closed, growth m^3 - m^2 per product);
containment Mon <= Mon^2; M3 (slice rank of products unbounded; characterisation KNOWN); M4 (Mon_LPiR inverse-closed,
Dao's class not, exact certificate); the upper bound (k+1)m^3 - km^2 for every k and the necessity k >= m; B1 upper bound
n(kL+1) and the necessity k >= (n-1)/L; B2 (butterfly Kronecker closure, KNOWN); Prop U identity (every idempotent) and
the non-Sylvester counterexample for 0 < rank P < n; Theorem R identities with the subadditive bound; Prop K (i) bound,
(ii) tight unbounded family for the Kronecker pair, (iv) unbounded quasiseparable family. **Open:** M5 equality for
k >= 3 and Monarch saturation at k = m (checked m <= 4), butterfly dimension equality and sufficiency of depth
(n-1)/L for general n (checked n <= 16), generic Toeplitz Kronecker growth and 2q product growth beyond the checked sizes,
non-Kronecker-pair LDR Kronecker unboundedness (posed for reconstructible pairs, see the K scope note). **Novelty
(superseded by section 6, repair round 3):** H0 stands; nothing is claimed as new.

## 9. Repair round 3 (cold no-memory referee, 2026-09-30)

Reviewer provenance: cold no-memory Opus referee (2026-09-30), fresh context, attacked the whole document. Verdict: NO FATAL
findings; every PROOF statement survived independent attack; every finite claim reproduced with independent code (exact
Fraction/sympy ranks, modular ranks mod 2^31-1 and 1e9+7, gauge kernel upper bounds; Monarch dimensions for all 12 listed
(m,k) plus (5,3), (5,5), (3,4), (2,3), (6,1), (6,2), (6,5) = 1116, (6,6) = 1296; M2 dimensions 16, 63, 160, 325 for
m = 2..5; the Dao certificate, det D = -69120, minors -1/216 and -3; butterflies n = 4, 8, 16, 32; Prop U 15 exact cases;
Prop K(ii) 224 cases; the hash of the registered text, 5121 chars, 544ae8f4...). The referee nevertheless ruled that the
document must not pass unchanged. Statuses assigned: SUPPORTED - M1, M2(a)(b), the M2(c) proof, containment/strictness/
non-closure, M3, M4 (Mon_LPiR inverse-closed), the Dao certificate itself (its significance was overstated), the M5 bound,
necessity and EMPIRICAL rows, the B1 upper bound, necessity and EMPIRICAL rows, B2, the Prop U identity and counterexample,
the Theorem R identities, Bresar/Hvala, Prop K (i), (ii), (iv) and (iii) EMPIRICAL, the Dao ranks 16/81/208 (exact), the
section-4 growth laws. OVERSTATED - the novelty of M2(c), several section-3 cells, the section-6 novelty and section 7.
WRONG - "Theorem R joint statement not found stated". NOT ATTEMPTED by the referee - the numerical-analysis citations
(Grasedyck-Hackbusch, Borm, Bebendorf-Hackbusch, Faustmann et al., Chandrasekaran et al., Candes et al., Li et al.,
Gautschi/Beckermann, Strang-Nguyen), which stay in the map marked "cited, not re-verified this round". The repair agent
(Claude) applied all 16 findings and declined none; where a repair needed a claim it could not verify, the text was
weakened instead. Section/page locators below are as reported by the referee and were not re-read.

Scripts (code/floor_ldr_checks.py, single thread; predictions R3-P1..R3-P8 in its docstring were written before the runs and
all matched): `r3` (exact Fractions, 1.5 s, results/floor_ldr_v1/r3.json, all_match true: helper controls, kron_reorder 4/4,
monotone 6/6, dao_piside, slice_formula 3/3, reconstructible 4/4 plus negative and trivial controls, closed_algebras 6+2);
`monarch_extra` (mod-p Jacobian ranks, 89 s, 8/8 match); `butterfly_extra` (n = 32, 54 s, 4/4 match). Every claim checked by
`r3` has a positive and a negative control (keys named NEG_* in the JSON).

| # | Finding | What was done | Evidence |
|---|---|---|---|
| MAJOR-1 | novelty of M2(c): Dao 2022 Sec 3.4 MM* = Pi.Mon^2.Pi and App. D.3 gives the 2m^2 symmetry group and the fibre structure | H0 STANDS; M2(c) relabelled "explicit statement plus an independent tangent-space proof of a direct consequence of Dao App. D.3"; the (k+1)m^3 - km^2 bound not offered as novel; header and sections 2 (M2 note), 6, 7, 8 updated | section 2 M2 note; section 6 |
| MAJOR-2 | section 7 reinstated the withdrawn LDR/HSS unification | LDR (Sylvester) and quasiseparable/HSS (off-diagonal ranks) listed separately, shared properties attributed to different mechanisms, "matching-pair" dropped for HSS | section 7 |
| MINOR-3 | slice-rank characterisation is Dao Sec 3.3 (Eq. 2, Alg. 1; App. D.2), not 3.2 | corrected in M3 and section 6 | - |
| MINOR-4 | Dao and Kaleidoscope never state "not product-closed" (Dao: closed under (conjugate) transpose, p.22); the n^(3/2) argument is Dao App. C | non-closure attributed to M2; the registered wording is corrected in the paragraph below, not in section 0 | section 6 |
| MINOR-5 | Theorem R is KNOWN: Pan 2013 arXiv:1303.0353 Thm 17 and Cor 18 | header relabelled KNOWN with the citation; "not found stated" withdrawn | Theorem R |
| MINOR-6 | "established non-closures are exactly ..." is incomplete | "at least"; added Dao's class not product-closed (every m >= 2), butterfly non-closure at n = 4, 8, 16, 32, Dao ranks 16/81/208 exact | section 7; Notation; M5 |
| MINOR-7 | the Dao inverse/transpose certificate is a fixed-permutation artifact | D^-1 Pi and D^T Pi in Mon_LPiR, PROOF [R3, gate pending], checked on the stored certificate; significance restated in Notation, M4, section 7 | r3.json dao_piside |
| MINOR-8 | Monarch Kronecker cell is closable; Mon_LPiR is both Kronecker- and inverse-friendly; butterfly inverses land in B* | Prop M6 with a two-line proof, checked at (2,2), (2,3), (3,2), (3,3); the dichotomy removed; Kaleidoscope App. H cited for the butterfly inverse cell | r3.json kron_reorder |
| MINOR-9 | the empty-cell target needs an expressivity qualifier | section 7: closure alone is cheap (block-diagonal algebras; circulants, CRT for coprime sizes); target = "expressive AND closed" | r3.json closed_algebras |
| MINOR-10 | monotonicity understated | Mon^k in Mon^(k+1) for every k (Latin-square step on the last pair); density at k = m persists for k >= m; attack item 7 updated | r3.json monotone |
| MINOR-11 | Prop K scope note ill-posed (A = B = I answers it trivially) | reconstructibility (injective displacement operator) required; the K(ii) pair (2,3,5,7) proved bijective | r3.json reconstructible |
| MINOR-12 | label consistency | labels rebuilt per section 1 (PROOF (referee-attacked, survived) / PROOF [R3, gate pending]); K(ii), K(iv), the map, section 4 and section 8 relabelled; "PROVED" no longer used | whole document |
| MINOR-13 | several section-3 cells say more than the evidence | butterfly product/sum cells; the 20 tested Monarch (m,k) listed exactly; LDR stability weakened; Cauchy-like/Vandermonde-like row corrected (the DFT is Vandermonde-like); H-matrix Kronecker and sparse+low-rank Kronecker cells marked CONJECTURE/unproved | section 3; monarch_extra.json; butterfly_extra.json |
| MINOR-14 | missing prior art on depth | Kaleidoscope Lemma E.3 (every matrix in (B B*)^(2n-2)) added beside the (n-1)/L necessity | B1; map |
| MINOR-15 | "Kailath-Sayed-Chun" is not one work | Kailath-Sayed (1995/1999) and Kailath-Chun (SIMAX 15, 1994) in Theorem R; the registered occurrence is corrected below | Theorem R |
| MINOR-16 | cosmetic: B1 gauge convention and the repo stride order; M2(b) vacuous; "Corollary M3"; "MM*-type" | B1 gauge written for B = F_L ... F_1 with the repo's transpose-variety order stated; M2(b) replaced by slice_ab = L_a diag(S) R_b; "Corollary M3" renamed Prop M3; "MM*-type" removed | r3.json slice_formula |

**New claims added in this round ([R3]; all PROOF [R3, gate pending] unless marked, each script-checked; none is claimed as
new mathematics):** (1) M6: Q(M1 (x) M2)Q^T in Mon(m1 m2), and the Dao-class version with Pi_{m1 m2}; (2) Mon^k in
Mon^(k+1) for every k, and density at k = m implies density for k >= m; (3) Dao's class is not product-closed for any
m >= 2; (4) D^-1 Pi and D^T Pi in Mon_LPiR (Dao's class closed under inverse and transpose up to moving the fixed Pi);
(5) slice_ab(L Pi S Pi R) = L_a diag(S[a,.,b]) R_b; (6) the K(ii) Kronecker pair at (2,3,5,7) has an injective (bijective)
displacement operator, and A = B = I answers the unrestricted open question trivially; (7) block-diagonal algebras and
circulants of coprime sizes are Kronecker-closed after a fixed reordering (classical, cited by name; checked); (8)
EMPIRICAL: eight further Monarch (m,k) pairs and butterfly n = 32, matching the referee's independent numbers.

**Corrections to the registered text (section 0 is byte-identical to the hashed registration; corrections live here).**
(i) "Kailath-Sayed-Chun" in the H0 source list is not one work: Kailath-Sayed (SIAM Rev. 1995; eds., 1999) and Kailath-Chun
(SIMAX 15, 1994) are separate. (ii) The sentence on Dao et al. ("expressivity and 'products are not closed', no exact variety
dimension, no product-class dimension count") is inaccurate in two ways: Dao 2022 and Kaleidoscope do not state that Monarch
is not product-closed (Dao 2022 states closure under (conjugate) transposition), and Dao 2022 App. D.3 contains the
ingredients of the MM* class dimension count (the 2m^2 symmetry group and the fibre structure); the n^(3/2) bound is Dao
App. C (proof of Thm 4). (iii) Registered candidate H1 (i), the exact Monarch product dimension, is therefore present in a
named source, and the registered decision rule gives H0. The hash of the registered text (5121 chars from
"PRE-REGISTRATION") was recomputed with the procedure of code/prereg.py before and after the edits of this round and equals
results/floor_ldr_v1/prereg.sha256. Note for editors: the procedure hashes from the FIRST case-insensitive occurrence of
the registration keyword, so that word must not occur anywhere above the section-0 heading.

**Open after this round:** M5 equality for k >= 3 and saturation at k = m for general m; butterfly dimension equality and
depth sufficiency for general n; LDR Kronecker unboundedness for a reconstructible non-Kronecker pair; generic Toeplitz
2q product growth beyond q = 2; the H-matrix and sparse+low-rank Kronecker cells and butterfly sums (CONJECTURE/unproved);
the "expressive AND closed" target cell. The PROVED label is reserved for the project's proof-review gate; this document
does not assign it.
