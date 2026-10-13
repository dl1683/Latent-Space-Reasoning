# RECT_FLOOR_SCORE_V1 (unit rect_floor_score_v1)

Opened 2026-09-28. Distance from claim: 1 (score of existing artifacts under the unit-A floor model; the practical
target is a scheme that beats tuned Karstadt-Schwartz Strassen in total arithmetic at n <= ~16k). Rules: AGENTS.md
sections 3, 6, 8, 12. Builds on `FRAMEWORK_FLOOR_V1.md` (unit A, PROVED, scoped) and
`../archive/2026-10-01/research/RANK48_SHARED_FACTOR_SEARCH_V1.md` (unit B, weak negative: no rank-48 4x4 scheme with proportional rows found).

## 0. PRE-REGISTRATION (written before any scheme was fetched or scored)

**H0.** No publicly obtainable exact scheme with exponent < log2 7 = 2.8074 has an effective total operation count
below tuned Karstadt-Schwartz (KS) Strassen (same op/cutoff conventions as `code/tuned_race_v2.py`
`Costs.ks_cost`) for ANY n = 2^p, 6 <= p <= 16, under the unit-A model with all lower-order terms.
**H1.** Some obtainable exact scheme with exponent < 2.8074 and duplicate rows (or collinear triples) does, at
some n <= 2^16 (as a symmetrized/cyclic scheme reaching square sizes; padding to the scheme's natural size is charged).

**Decision rule.** Scoring set = every scheme I can OBTAIN as machine-readable data and VERIFY exactly (all input
basis pairs, Fractions/integers). Unverified schemes are not scored. H1 needs a verified scheme whose exact
chain-count total (model F: additions, leaf multiplications; model C addition of non-+-1 scalings reported as
a sensitivity) is < tuned KS at some scored n, with padding charged (n not a multiple of the scheme size is
padded up to the next multiple; the baseline is charged at the SAME n, i.e. the padded scheme competes against KS at the requested n).
Otherwise H0 (closed NEGATIVE, scope = obtained schemes; not a proof that none exists), with the blocking invariant
recorded per scheme (exponent, D, triples, seed size, base size mkn vs reachable n).

**Kill rule.** If H0 holds after scoring every obtainable scheme, close NEGATIVE and record the mechanism.
Control (identifiability): the scorer must reproduce (i) KS Strassen = the baseline itself (Strassen d-level chain
c* -> 5) and (ii) Ballard-Benson <3,3,3;23> c* = 2 with the model; it must be capable of returning H1 on a
synthetic scheme with exponent < log2 7 (e.g. a hypothetical <3,3,6;40> with duplicate rows is NOT synthesised as a
result; instead the crossover formula is checked on Strassen-with-cutoff vs Winograd to reproduce known ordering).

**Registered predictions (before computing).** (a) Symmetrized c* is 2 + 3 rho(L(x)R(x)Q), i.e. 2 if ANY of L, R, Q has
duplicate rows, 5 if none does and ANY of L, R, Q has a collinear triple. (b) Because the symmetrized base is n' = mkn
(54 for <3,3,6>) and r' = r^3 (64000), the number of usable recursion levels at n <= 16384 is 2 (n' = 2916; 3 levels = 157464),
and the lower-order terms (transforms with base D'/r' near 1) will dominate; I predict NO scheme with exponent < 2.8074 beats tuned
KS at n <= 2^16 (H0). (c) Ballard-Benson 23 (exponent 2.854) never beats KS at any n <= 2^16 despite c* = 2 (exponent gap:
crossover would need n^{0.047} > ~2.5 i.e. n > 1e11).

**Prior art (closest first; what each did NOT do).** Beniamini-Schwartz SPAA'19 (sparse decompositions; scores <3,3,3;23> only; no
rectangular symmetrization accounting). Karstadt-Schwartz JACM'20 (alt basis, Strassen/Winograd; coefficient 5). Perminov arXiv 2606.02480 (meta flip graph;
reports exponents/ranks, no addition/coefficient accounting). Kauers-Moosbauer flip graphs (rank searches over Z2, then lifts; no cost
accounting). Smirnov 2013 <3,3,6;40> (rank; the symmetrized addition cost never computed). Sedoglavic fmm catalogue (ranks/links;
no row-class invariants). Gap: nobody has computed the row-class (unit-A) coefficient or an exact total-arithmetic curve for sub-Strassen-exponent schemes.

**Disclosure (added when the results were in).** Before writing the pre-registration above I had already opened
`results/tuned_race_v2/model_arith.json` and seen that the unit-C model gives the alternative-basis rank-48 scheme
2-3.6% FEWER counted operations than tuned KS at n = 4096..16384. So H0 as worded (no exponent < log2 7 scheme beats tuned KS below
2^16) was already in doubt when registered; it is nevertheless what I registered. The informative content of this unit is (i) which OTHER schemes
do, (ii) the rectangular coefficient theory, (iii) the mechanism that blocks the rest, and (iv) that a win survives when the alternative-basis transform-once trick is NOT used
but a KS inner algorithm IS (hybrid architecture below). Nothing else was computed before registration.

## 1. Rectangular generalization of the unit-A coefficient (task 2)

Notation. A scheme for <m,k,n; r> is (L, R, Q): L in F^{r x mk}, R in F^{r x kn}, Q = P^T in F^{r x mn}, with
sum_t L_t (x) R_t (x) Q_t = the matrix-multiplication tensor (trilinear form sum x_ij y_jl z_li). Assume r > max(mk, kn, mn) and no zero rows (true for every non-trivial scheme; rank >= max(mk,kn,mn)). The side condition r > max(mk,kn,mn) excludes degenerate cases m, k or n = 1.

**1.1 Direct rectangular recursion (PROVED (gates: section 7), by re-reading unit A).** j levels give a scheme for <m^j,k^j,n^j; r^j> whose three maps are L^{(x)j}, R^{(x)j}, Q^{(x)j}.
Lemma T, Lemma U, Lemma L and Theorems 1-3 of FRAMEWORK_FLOOR_V1 use only: the maps are r x d_X matrices with d_X < r, and the algorithm is an arbitrary circuit for each Kronecker power. Squareness (d_A = d_B = d_C) is never used.
Hence c* = 2 + rho(L) + rho(R) + rho(Q) is the coefficient of r^j (products r^j), with d_A = mk, d_B = kn, d_C = mn, and decoder count exactly a(P,j) = a(Q,j) + r^j - (mn)^j (Lemma T with m -> r^j, n -> (mn)^j).
This is a statement about the RECTANGULAR problem <m^j,k^j,n^j>; it does not by itself give a square-matrix algorithm.

**1.2 Square matrices: cyclic symmetrization.** <m,k,n> (x) <k,n,m> (x) <n,m,k> = <mkn,mkn,mkn> (Kronecker product of matrix-multiplication tensors; each factor is a cyclic rotation of the same tensor, because the trilinear form is invariant under (x,y,z) -> (y,z,x),
which turns the roles (A, B, C^T) into (B, C^T, A)). The scheme S (x) S^rot (x) S^rot2 has rank r^3, base size B = mkn, and its three encoder maps are
A' = L (x) R (x) Q, B' = R (x) Q (x) L, C'^T = Q (x) L (x) R (rows indexed by the product triples (t1,t2,t3) in the respective order, columns by the input coordinates). B' matches L (x) R (x) Q up to a ROW permutation (reorder the triple index) as well as a column permutation, and likewise C'^T; the three maps therefore have the same row-class multiset, collinear triples, and circuit cost (D, triples and cost are unaffected by the permutations). Exponent: omega = log_B r^3 = 3 ln r / ln(mkn)
(<3,3,6;40>: 3 ln 40 / ln 54 = 2.7743). A square base needs no symmetrization (exponent ln r / ln m); a square scheme may still be symmetrized.
Using the rectangular scheme directly in a square recursion without rotating is impossible: after j levels A is m^j x k^j.

**Lemma K (Kronecker rows; PROVED (gates: section 7)).** Let X, Y, Z be r x d_i matrices without zero rows and M = X (x) Y (x) Z (rows x_a (x) y_b (x) z_c). Then
(Remark: the duplicate-free hypothesis of part (b) is not used by the proof of Lemma K(b) as written; this is harmless.)
(a) x_a(x)y_b(x)z_c and x_a'(x)y_b'(x)z_c' are proportional iff each factor pair is, so D(M) = D(X)D(Y)D(Z) and u(M) = u(X)u(Y)u(Z);
(b) if D(M) = r^3 (all factors duplicate-free) then M has a collinear triple iff one of X, Y, Z has one.
Proof. (a) A Kronecker product of nonzero vectors determines its factors up to scalars. (b) (<=) if x_1,x_2,x_3 are dependent (distinct classes) then x_i (x) w for a fixed row w of Y(x)Z is a dependent triple of distinct classes.
(=>) Three distinct-class rows x_i (x) n_i of X (x) N (N = Y (x) Z) that are dependent: c_1 x_1 n_1^T + c_2 x_2 n_2^T + c_3 x_3 n_3^T = 0, all c_i != 0. If x_1, x_2 are proportional then
x_1 (c_1 n_1 + c_2' n_2)^T = -c_3 x_3 n_3^T forces x_3 proportional to x_1 (otherwise n_3 = 0), hence the n_i are dependent and pairwise non-proportional (else two rows share a class): a triple in N.
If the x_i are pairwise non-proportional and dependent, x_3 = alpha x_1 + beta x_2 and the identity forces n_1, n_2, n_3 mutually proportional: a triple in X. If the x_i are pairwise non-proportional and independent, the matrices x_i n_i^T are independent. Induct on N. QED.

**Theorem R (coefficient of the symmetrized scheme; PROVED (gates: section 7) given unit A). SCOPE (gate 3): the exact values c* = 2 and c* = 5 below are established for model F (free scalings). In model C (scalings counted) the duplicate case additionally needs sign-clean duplicates and the triple case needs both synthesis coefficients to be +-1 (unit A Theorems 2-3); otherwise only intervals are proved (rho in [0,1] resp. [1,3]); four scored rational records contain duplicated maps that are not sign-clean. The all-triple-free lower bound c* >= 5 holds in both models. The finite score curves charge scalings in model C and are upper bounds independent of this label.** All three maps of S^sym equal M = L (x) R (x) Q up to row and column permutations, so c*(S^sym) = 2 + 3 rho(M) (coefficient of r^{3j} at base B^j) and
- rho(M) = 0, c* = 2, iff at least ONE of D_L, D_R, D_Q is < r (one duplicated map suffices; the decay rate of the second-order term is (D_L D_R D_Q / r^3)^j);
- if D_L = D_R = D_Q = r and at least one of L, R, Q has a collinear triple: rho(M) = 1 (unit-A Theorem 3 applied to M with the closure seed s_L r^2), c* = 5 exactly;
- if all three are duplicate- and triple-free: c* >= 5 (Theorem 1); the refined staged bound of unit-A Theorem 4 is not re-derived for M (CONJECTURE: > 5).
Contrast with the direct/square model, where c* is small only if EACH map has a duplicate/triple: symmetrization makes one duplicated (or triple-carrying) map enough. Cost of the trade: the second-order base is D_L D_R D_Q (or s_L r^2), so the leading coefficient is approached only as fast as (D/r)^j.

**Model checks (EMPIRICAL; `code/framework_floor_checks.py rect-selftest` plus controls).** Strassen (2x2x2;7): 5.0041 at j=16 and 5.0000 at j=32 (unit A: same); forced symmetrization 5.0001 at j=8.
Laderman <3,3,3;23>: 5.0000 (square and forced sym). Ballard-Benson grey333-23-152 (D = 20/20/20): 2.03 at j=32 (unit A naive chains 2.08), forced sym 2.10 at j=8 (limit 2 = BS Table 2).
grey333-23-221 (D = 23/22/22, two duplicated maps): square 3.48 at j=32 (limit 3), forced sym 3.76 at j=6 decaying like (23*22*22/23^3)^j toward 2 (Theorem R; the rate is the practical problem).
The interleaved-lift DP (`lift_adds`) equals the minimum over all stage orders of an explicit numpy stage-lift with per-gate op counting (4x4x4;48 and 3x3x6: every order consistent with the closed form, DP min = explicit min, greedy-CSE circuits re-evaluated exactly).

## 2. Exact cost model used for scoring (fixed before scoring; changes recorded in section 6)

Architecture ("hybrid"): j levels of the scheme S (symmetrized when not square) applied to the padded n_pad x n_pad matrices, n_pad = B^j s >= n (padding charged: the algorithm computes the n_pad product), leaves = s x s products computed by the best tuned inner algorithm
f(s) = min(classical, tuned Winograd, tuned KS; tuned_race_v2 op counts, any cutoff, any integer s = b0 2^d). Then

total = r'^j f(s) + s^2 [ E_A + E_B + E_Q + (r'^j - d'^j) + lambda_j ],   r' = r or r^3, d' = m^2 or B^2 = (mkn)^2,

each scalar circuit operation acting on an s x s block. E_X = additions of the stage-lifted circuit for the Kronecker power of the map (exact minimum over ALL interleavings of stage lifts, `lift_adds`; symmetrized case 3E since the three maps coincide),
the decoder count r'^j - d'^j is Lemma T (equality for the transposed circuit), and lambda_j = number of products whose per-term scale (product over the three maps of the smallest |coefficient|, rows normalised to minimum magnitude 1) is not +-1
(model C; model F drops lambda and all scalings; all-+-1 schemes have C = F). Row circuits: greedy pair-CSE (Paar), then line-closure chains from verified collinear triples (1 add per generated row; ratio-bounded search |a|,|b| <= 4, so triple counts are lower bounds),
then free copies for duplicates. All row circuits are UPPER bounds, so the totals are UPPER bounds (UB) on the best hybrid. Universal lower bound (LB) for the same architecture (Theorem 1, gated with unit A): adds >= sum_X (D_X^j - u_X^j) + (r'^j - d'^j), reported as min over (j, s).
Baseline: min(tuned KS, tuned Winograd) at n = 2^p, unconstrained cutoff (`code/tuned_race_v2.py` Costs; KS alone is also in the JSON). Ratio < 1 means fewer counted arithmetic operations than the tuned baseline.
NOT modelled: data movement, memory, stability, parallelism. NOT covered: alternative-basis (transform-once) versions of the outer scheme (unit C's alt48 does that; quoted for comparison), different schemes per level, KMW-style merged rectangular calls.

## 3. Schemes obtained and verified (task 3)

Sources (fetched into a scratch dir with `--data`; not committed; licences of the source repositories apply):
- Perminov `dronperminov/FastMatrixMultiplication` (`schemes/status.json`, 680 formats up to 16x16x16; branch master). Every (format, field) whose best rank has exponent 3 ln r / ln(n1 n2 n3) < log2 7:
  93 distinct source files (79 field-best entries plus alternative fields/variants of the same rank), including the 23 new sub-Strassen schemes of arXiv 2606.02480 and those inherited from the Sedoglavic fmm catalogue (Maple TriadSet `.mpl`: <3,3,6;40> Smirnov 2013, <4,4,4;48>, <8,8,8;336>, <12,12,12;1040>, ...),
  the meta-flip-graph / Kauers-Moosbauer `.m` files (<3,4,6;54> and 24 variants), LiTa/others (`<16,16,16;2237>`, `<15,16,16;2223>`) and the Perminov results JSON (ZT/Z/Q). Three parsers (Maple, Mathematica lists, JSON with u/v/w) written for this unit.
- Kauers-Moosbauer-Wood 2602.11041 `github.com/mkauers/matrix-multiplication/structured/666.exp` and `666r153.exp` (two <6,6,6;153> decompositions, parser `parse_exp_uvw`). Their exponent 2.8019 is NOT the rank exponent (log_6 153 = 2.8075): it comes from
  18 pairs of products that share a factor being recomputed as single larger rectangular products (<2,1,1>-type merges; quoted from the paper: "the best we found contains 18 such copies", leading coefficient 7.67 vs 7 for Strassen in the paper's normalisation). That is exactly the
  shared-(proportional)-factor mechanism of unit B, realised inside one 6x6 scheme. A merged pair is worth at most 2 - 2^{omega/3} ~ 0.087 of a product, so at most 18*0.087/153 ~ 1.0% of the product cost per level; I scored the plain rank-153 base (exponent 2.8075 > log2 7) and use 1%/level only as an optimistic credit.
- Verification: every scheme is checked EXACTLY-MOD-p (two primes 999983 and 1000003; all d_A d_B d_C tensor coefficients, i.e. every basis triple, must equal a matrix-multiplication tensor <m,k,n> under one of the 8 index layouts). Schemes failing are not scored (none failed; the rank, dimension and layout are recorded in the JSON).
  Small ones were also checked in Fractions in units A and B (4x4x4;48 identical to the PLinOpt `rational` scheme up to the catalogue's own convention: NOT compared entry by entry).
- Not obtainable / not scored: Sedoglavic .bz2 Maple formats for schemes absent from the Perminov mirror (not needed: the mirror carries all 79 sub-Strassen entries), Smirnov's original files (the mirror's 3x3x6 is the fmm catalogue tensor, identical source), Moran-Schwartz-Yuan Appendix A (unit B), AlphaEvolve complex schemes (complex; Kaporin 48 has no real form).
  Only scheme classes with exponent < log2 7 were requested; near-threshold non-sub-Strassen bases (<6,6,6;153>, <3,3,3;23> variants, Laderman) enter as controls.

## 4. Scores (task 4)

Reproduce: `python code/framework_floor_checks.py rect --data <scratch> --out <json> [--min-rank a --max-rank b --per-size 2 --kmw --controls --cse-rows 1200 --inner-alt48]`,
`python code/framework_floor_checks.py rect-table --only <json...>`, `python code/framework_floor_checks.py rect-selftest --data <scratch>`.
Data: `results/rect_floor_score_v1/scores.json` (per scheme: verified layout, per-map D / unit classes / triples / seeds, exact chains, per-n hybrid totals for model F and C at n = 2^6..2^16 with (j, s, n_pad), the universal LB, tuned KS and best-of-KS/Winograd op counts),
`table.md` (all rows, 2^8..2^16), `scores_inner_alt48.json` (same four best schemes with unit C's alt48 also allowed as the inner algorithm and as baseline).
Scored: 78 records = 71 Perminov/fmm scheme files (up to 2 variants per format/field/rank; 93 sub-Strassen source files exist) + 2 KMW files + 5 controls; 16x16x16;2237 not scored (see section 6). Every one verified exactly-mod-p (0 failures). Baseline op counts (tuned KS = min-cutoff KS; best of KS/Winograd, identical at these n): 2^10 1.008e9, 2^12 4.923e10, 2^14 2.408e12, 2^16 1.179e14.
Cell = UB total / best(tuned KS, tuned Winograd) at n = 2^p, model C (scalings counted), padding charged, `j` outer levels; < 1 is a win. c* is the model coefficient (Theorem R for rectangular, unit A for square; '>=5' when a side is triple-free, unit-A Theorem 4 bound not applied).

| scheme | field | r | exponent | base B | +-1 only | D_L/D_R/D_Q | triples L/R/Q | seed L/R/Q | c* | prod. saving/level | adds/product (1 level) | 2^10 | 2^12 | 2^14 | 2^16 | first win | first p with LB<1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4x4x4 | Q | 48 | 2.7925 | 4 | n | 48/48/48 | 16/0/0 | 32/48/48 | >=5 | 0.020 | 7.42 | 1.011 | 0.992 | 0.976 | 0.957 | 2^11 | 2^6 |
| 3x3x6 | Q | 40 | 2.7743 | 54 | n | 40/40/40 | 48/0/0 | 13/40/40 | 5 | 0.124 | 8.73 | 1.097 | 1.010 | 0.976 | 0.958 | 2^13 | 2^10 |
| 8x8x8 | Q | 336 | 2.7974 | 8 | n | 336/336/336 | 256/144/144 | 164/202/201 | 5 | 0.020 | 7.79 | 1.033 | 1.001 | 0.988 | 0.974 | 2^13 | 2^6 |
| 6x6x12 | Q | 280 | 2.7856 | 432 | n | 280/280/280 | 456/120/120 | 96/167/169 | 5 | 0.124 | 5.83 | 2.354 | 1.160 | 0.994 | 0.977 | 2^14 | 2^13 |
| 9x9x12 | Q | 600 | 2.7896 | 972 | n | 600/600/600 | 840/360/360 | 232/331/324 | 5 | 0.115 | 5.87 | 8.594 | 1.758 | 1.038 | 1.003 | none | 2^14 |
| 5x5x12 | ZT | 204 | 2.7972 | 300 | y | 186/204/204 | 252/48/48 | 67/158/158 | 2 | 0.057 | 4.46 | 1.544 | 1.046 | 1.061 | 1.037 | none | 2^11 |
| 3x4x6 | ZT | 54 | 2.7982 | 72 | y | 54/54/54 | 72/12/18 | 19/42/37 | 5 | 0.038 | 5.17 | 1.201 | 1.169 | 1.170 | 1.041 | none | never (blocked) |
| 4x4x10 | ZT | 115 | 2.8048 | 160 | y | 100/115/115 | 130/16/15 | 36/99/100 | 2 | 0.013 | 4.57 | 1.299 | 1.087 | 1.095 | 1.081 | none | never (blocked) |
| 3x4x7 | Q | 63 | 2.8052 | 84 | n | 63/63/63 | 63/3/27 | 22/60/40 | 5 | 0.009 | 5.13 | 1.361 | 1.231 | 1.212 | 1.196 | none | never (blocked) |
| 4x4x9 | Q | 104 | 2.8036 | 144 | n | 104/104/104 | 100/32/16 | 46/74/89 | 5 | 0.019 | 6.10 | 1.721 | 1.270 | 1.219 | 1.198 | none | never (blocked) |
| 12x12x12 | Q | 1040 | 2.7957 | 12 | n | 1040/1040/1040 | 720/1448/720 | 556/394/553 | 5 | 0.029 | 13.79 | 1.200 | 1.125 | 1.095 | 1.069 | none | 2^15 |
| 9x12x16 | Q | 1035 | 2.7937 | 1728 | n | 1005/1005/1005 | 1149/148/105 | 439/883/907 | 2 | 0.097 | 22.96 | - | 5.974 | 2.002 | 1.263 | none | 2^15 |
| 6x6x6 | Q | 153 | 2.8075 | 6 | y | 145/145/145 | 50/50/50 | 105/103/103 | 2 | -0.000 | 5.05 | 1.135 | 1.102 | 1.105 | 1.099 | none | never (blocked) |
| 2x2x2 | control | 7 | 2.8074 | 2 | y | 7/7/7 | 3/3/3 | 4/4/4 | 5 | 0.000 | 3.00 | 1.006 | 1.003 | 1.001 | 1.000 | none | 2^6 |
| 3x3x3 | control | 23 | 2.8540 | 3 | y | 23/23/23 | 16/16/16 | 11/11/11 | 5 | -0.053 | 3.61 | 1.180 | 1.173 | 1.170 | 1.169 | none | never (blocked) |

Selected rows (the four winners, the near miss, every Perminov-new or KMW scheme class, controls); all 78 records are in `results/rect_floor_score_v1/table.md`. Column "first p with LB<1": the smallest n at which the Theorem-1 bound of the architecture drops below the baseline ("never (blocked)" = provably blocked FOR THE STATED HYBRID ARCHITECTURE ONLY: the inner cost f(s) is a fixed tuned operation count, not a lower bound on inner algorithms).

Controls: Strassen/Winograd 1.000-1.012 (never below the tuned baseline, as required: identifiability control passes, the scorer does not manufacture wins); Laderman and Ballard-Benson <3,3,3;23> variants 1.17 at every n (c* = 5 / 2 / 3 but exponent 2.854: their exponent excess exceeds any coefficient gain, prediction (c) confirmed).
Unit-C alt-basis rank-48 (transform once) for comparison, from `results/tuned_race_v2/model_arith.json`: 0.986 (2^12), 0.964 (2^14), 0.944 (2^16). With unit C's alt48 allowed as the inner algorithm AND as the baseline (`scores_inner_alt48.json`):
3x3x6;40 outer + alt48 inner is 0.995 of alt48 at 2^16 (1.012 at 2^14, 1.005 at 2^15: it only pays for n = 2^16), 4x4x4;48 staged outer gives 1.003 (no gain), 8x8x8;336 1.016, 6x6x12;280 1.035.

Checks of the model itself. (i) selftest: explicit numpy stage lifts with per-gate op counting in every valid stage order equal the batch-width closed form and the DP minimum equals the minimum over orders (4x4x4;48: 620,544 adds for naive chains, 3x3x6: 294,768), greedy-CSE circuits re-evaluated exactly.
(ii) Strassen and Laderman reach 5.0000, Ballard-Benson 2.0343 (k=32) in the same chain model. (iii) Unit-C cross-check: with Winograd as inner algorithm my staged 4x4x4;48 model gives 5.043e10 / 3.504e11 / 2.423e12 at n = 4096 / 8192 / 16384, versus unit C's exact-SLP staged 4.971e10 / 3.419e11 / 2.383e12 (s2..s3):
my greedy-CSE upper bounds are 1.0-3.3% weaker than the PLinOpt SLPs (4096: 5.043e10 vs 4.993/4.971/4.881e10 for s1/s2/s3; 16384: 2.423e12 vs 2.401/2.366/2.383e12), as expected for an UB. The wins above therefore come from allowing a KS inner algorithm below the staged levels, which unit C's staged model did not do.
Cross-scheme sanity: 8x8x8;336 = 7 x 48 (Strassen (x) rank-48), exponent 2.7974; its curve is the 4x4x4 curve shifted by one Strassen level, as expected.

## 5. Verdict on H0 / H1 and mechanisms

**H0 is REJECTED (H1 accepted with a small margin).** Four exact, verified sub-Strassen-exponent schemes have an UPPER-bound counted-operation total below tuned KS/Winograd for some n <= 2^16, with padding and scalings charged:
<4,4,4;48> from n = 2^11 (0.992 at 2^12, 0.976 at 2^14, 0.957 at 2^16; the known unit-B/C scheme, here without the alternative-basis trick), <3,3,6;40> from 2^13 (0.976 / 0.958 at 2^14 / 2^16; symmetrized <54,54,54;64000>, ONE level, s = 304 / 1216, coefficient 5 with only L carrying triples),
<8,8,8;336> from 2^13 (0.988 / 0.974), <6,6,6x2;280> = <6,6,12;280> from 2^14 (0.994 / 0.977, one symmetrized level, base 432). No scheme other than these four wins at n <= 2^16 (and none wins below 2^11); the best margin at the practical n = 2^14 is 2.4%, at 2^16 4.3%, all op counts only.
Nothing here beats unit C's alternative-basis 4x4 result (3.6% at 2^14, 5.6% at 2^16), and combining the best exponent found (3x3x6;40, 2.7743) with unit C's alt48 inner gains only 0.5% at n = 2^16 (nothing below). RIGOR: EMPIRICAL op-count model (UB circuits, exact-mod-p verified schemes); no memory traffic, precision or wall-time claim.
**So what:** the exponent lever is real but small at n <= 2^16: the best public sub-Strassen exponent (2.7743) buys 12% fewer products at one symmetrized level, of which the adds (8.7 per product at that level), the 1/8 scalings (1-3%) and the padding to multiples of 54 eat most; net 2-4%.

**Mechanism that blocks the other 69 schemes (what invariant stops each).** Among the 73 scored schemes: 60 are provably blocked in this hybrid architecture only (inner cost f(s) is a fixed tuned count, not a lower bound; the Theorem-1 lower bound of the same hybrid is >= 1 at every n <= 2^16, Theorem 1 passed the unit-A gate),
the rest (5x5x12;204 ZT, 9x9x12;600, 9x11x12;738, 9x12x16;1035, 12x12x12;1040, 8x16x16;1230, 14x14x14;1594, 15x15x15;1977) have UB >= 1 but LB < 1 (UNRESOLVED: their row circuits above 400 rows are naive, so UB is loose; 9x9x12;600 is the nearest, UB 1.038 / 1.003 at 2^14 / 2^16, model F 0.993 at 2^16).
The blocking invariants, in decreasing order of explanatory power:
1. Exponent excess and base size. One outer level saves PS = 1 - r'/B^{log2 7} of the products (col. "prod. saving/level"): winners have PS = 0.12 (3x3x6, 6x6x12), 0.02 (4x4x4, 8x8x8 but they reach 3 or 2 levels); every LB-blocked scheme has PS <= 0.091 and most <= 0.05, while one level costs >= 1 add per product (decoder r' - d') plus 4.6-8.7 adds/product of encoder circuits at the winners (5.8-8.7). At n <= 2^16 a base B >= 300 allows one level only (B^2 > n), so the saving is one PS.
2. Padding to multiples of B and the second-order cost: rectangular bases (B = mkn) with B ~ 10^3 force n_pad = B s with the leaf s = 2..200 too small for the inner KS to be efficient (ratios 2..25 at n = 2^10..2^12).
3. The coefficient lever does NOT decide it: 45 of the 73 schemes have a duplicated map (Theorem R: c* = 2) and none wins; the duplicates only help after (D_L D_R D_Q / r^3)^j << 1 (D/r on the 91 duplicated maps: min 0.80, median 0.97, max 0.999, so (D/r)^j is still 0.86 at the median for j = 5), so for j <= 5 they are irrelevant, exactly the "second-order base" caveat of unit A. c* = 5 with a closure seed (3x3x6: s_L/r = 13/40 on L only, R and Q triple-free with seed r) wins because its exponent is 2.774.
4. Circuit quality on triple-free maps: R and Q of 3x3x6;40 (naive-like 158 and 146 adds for 40 rows over 18 inputs) carry most of the 8.7 adds/product; this is where alternative-basis / sparsification can still gain (LB 0.90 at 2^16 vs UB 0.958).
5. Scalings: 3x3x6;40 has entries +-1/8 (1.6% of the total at 2^12, 0.5% at 2^14, 2.5% at 2^16 (model C minus model F; F picks two levels at 2^16)). Perminov's ZT variants of the same rank do not exist for 40.
6. KMW <6,6,6;153>: plain rank exponent 2.8075 > log2 7, UB 1.099; their merged calls (shared factors, i.e. proportional rows) are worth <= 1% of products per level, far below the 10% gap. The shared-factor mechanism is real but second order.
**KILL RULE: not triggered** (H0 fails); the negative half of the result stands for the other scored schemes and, for the LB-blocked ones, is a lower bound (scope: uniform j-level recursions of ONE scheme over a tuned KS/Winograd/classical inner algorithm, arbitrary circuits for the outer maps incl. transform-once ones; NOT mixed schemes per level, NOT other inner algorithms such as alt48, NOT all algorithms).

**Registered predictions, outcome.** (a) Theorem R coefficients: CONFIRMED (Strassen/Laderman 5, BB 2, forced-sym numerics). (b) "no scheme with exponent < log2 7 beats tuned KS at n <= 2^16": REFUTED (four schemes, above; I had seen unit C's alt48 curve before registering, see the disclosure). (c) Ballard-Benson 23 never beats KS: CONFIRMED (1.17).
Control paragraph of the pre-registration, restated cleanly: the scorer returns ratios >= 1.000 for Strassen and Winograd against the tuned baseline (it does not manufacture wins) and returns < 1 only for schemes whose product saving exceeds their add cost; both hold.

## 6. Next unit, corrections, status

**Recommended next unit: `altbasis_rect_v1` (distance 0-1).** Sparsify the three maps of the symmetrized <3,3,6;40> (and second, <6,6,12;280>) by change of basis so that the transform is applied once (Karstadt-Schwartz / DPS-4x4 pattern) and only a sparse core is lifted per level.
H0: no basis triple (phi_A in GL_9, phi_B, phi_C in GL_18 over Z[1/2], entries in {0,+-1,+-1/2}) brings the UB of <3,3,6;40> below unit C's alt48 curve (0.964 at 2^14, 0.944 at 2^16). H1: a basis lowers E (currently 3E = 3 x ~150 adds per product-normalised level) enough for UB <= 0.94 at 2^16 and <= 0.96 at 2^14.
Kill rule: exact search over the sparsest bases (as in unit C's KS step) plus a scalar-free representative fails to reach UB < 0.96 at 2^16 => close NEGATIVE and stop the exponent-lever line at n <= 2^16. Identifiability: the same search on <4,4,4;48> must reproduce unit C's alt48 counts.
Not recommended: more rectangular catalogue scoring (this unit exhausted the public sub-Strassen exponents: only bases with B <= 8 or the two ~0.12-saving schemes matter) and shared-factor 4x4 rank-48 search (unit B).

**Open / not done.** (1) Independent proof gate on Lemma K, Theorem R and the LB claims (Theorem 1 of unit A). (2) 16x16x16;2237 (LiTa) NOT SCORED: verification and circuit analysis exceeded the time budget (I stopped the run). Class counts without verification: D = 2158/2174/2153 of r = 2237 (all maps duplicated, c* = 2, D/r = 0.965), and the Theorem-1 lower bound of the hybrid is 0.80-0.86 at n = 2^11..2^16, so it is NOT blocked (exponent 2.7818, up to 4 levels at 2^16); the file is UNVERIFIED and its UB needs good row circuits. It is the largest open candidate of this catalogue. (3) Ratio-bounded triple search (|a|,|b| <= 4) so triple counts are lower bounds; D and unit counts are exact. (4) Row circuits above 400 rows are naive (UB loose there, flagged in the unresolved list). (5) Theorem R's third case (all three factors triple-free) lacks the staged >5 bound.
(6) I did not verify the tuned_race_v2 KS/Winograd counts independently beyond re-deriving their cost formulas by reading `Costs`; a KS-inner count error would move every ratio by the same factor. (7) Perminov's repository contains more schemes at ranks above the field-best rank (not scored: exponent above log2 7 or above the field-best).
Files: this document, `code/framework_floor_checks.py` (new `rect`, `rect-table`, `rect-selftest` commands; +~600 lines), `results/rect_floor_score_v1/{scores.json,table.md,scores_inner_alt48.json}`. No commit made (supervisor commits). Suggested commit message: "Add rectangular symmetrized floor theory and exact scoring of sub-Strassen schemes against tuned baselines".
Status: **PROVED (Lemma K, Theorem R, LB) for the stated scope, given unit A; scores EMPIRICAL op-count upper bounds; overclaims corrected (cold referee repairs applied 2026-09-29).**

## 7. Gate history

- **Gate 3 (2026-09-29), Stateful Codex, joint gate with unit A and repr_automaton_v1.** Theorem R scope repaired (model F vs model C labels, sign-clean duplicates, +-1 synthesis coefficients).
- **Cold no-memory referee (Claude Opus subagent, 2026-09-29).** ACCEPT the cores (Lemma K, Theorem R, LB claims within the architecture). Repairs applied: cyclic-symmetrization wording (row as well as column permutation); 'provably blocked' restricted to the stated hybrid architecture (inner cost f(s) fixed tuned count); Lemma K(b) unused duplicate-free hypothesis noted and side condition r > max(mk,kn,mn) (excludes m, k or n = 1) stated.
