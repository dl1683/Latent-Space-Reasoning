# FRAMEWORK_FLOOR_V1 (unit A): the leading-coefficient floor of tensor-power bilinear schemes

Status: **PROVED (scoped to the stated models) — passed independent gate 3 (Stateful Codex referee, 2026-09-29; every lemma/theorem holds in its stated model; gate history below)**; see 'Gate history' at the end. Created 2026-09-28. Distance from claim: 1 (Floor-tier theorem about an
existing artifact family; no measurement infrastructure).

## 0. PRE-REGISTRATION (written before the registered computations; see disclosure)

**Disclosure.** Before writing this section I ran one exploratory script (rank, projective-class and
collinear-triple counts) on the PLinOpt 4x4 rank-48 variants and one 23-term scheme, to see whether the
question was well posed. What I saw: for `canonical`, `sparse`, `204`, `accurate` the rows of L and R have
0 collinear triples and P^T has 16; for `rational` L has 16, R and P^T 0; `3x3x3_23_55` has 20 classes among 23
rows of L and of R. Nothing below about the predicted numbers for Strassen, Winograd, Laderman, Strassen^2,
the other 23-term schemes and the class of the 23-term Ballard-Benson-type scheme had been computed.
Those are the registered predictions.

**Class covered.** A scheme is (L, R, P) with L, R in F^{r x d}, P in F^{d x r}, d = n0^2, computing
C = P((La) (.) (Rb)) for n0 x n0 matrices. Its k-th tensor power is the rank-r^k scheme
(L^{(x)k}, R^{(x)k}, P^{(x)k}) for n = n0^k over the scalar leaf. The ALGORITHM class is: any
fan-in-2 linear circuit for each of the three maps (no assumption that it is staged, tensor-lifted, or
recursive), r^k scalar products in between. The **exact op model** (two versions, both reported):
(F) additions/subtractions only, all scalar multiplications free (scalars sit on circuit edges);
(C) additions plus every multiplication by a scalar not in {+1,-1} counted. Every lower bound proved
in model (F) holds in (C). Products count r^k. The leading coefficient is the coefficient of r^k = n^{log_{n0} r}
in the total operation count; "ops at scalar leaf" is assumed (cutoffs and lower-order terms are not part of the claim).

**Claim under test (agent's).** If the rows of L, R and columns of P are pairwise non-proportional, the
coefficient is >= 5 for any staging; with duplicate/proportional rows it weakens.
**My job's claim (H1).** A precise general floor in terms of collinearity structure of the rows of L,
R and of P^T, with a matching construction where one exists.

**H0.** The statement only recovers Beniamini-Schwartz (SPAA'19 Claims 5.1, 5.6) / ACDA'25 Thm 6.1:
floor 5 for duplicate-free schemes, floor 2 at best otherwise, nothing more. **H1.** A strictly more
general and correct floor exists, stating a necessary condition for a rank-r scheme to beat 5 (duplicate-free ones cannot) and by how
much the coefficient can be lowered from the known 17/3 for rank 48.

**Decision rule.** H1 is accepted only if (a) a theorem is proved in the stated class, (b) it survives my
own attack list, (c) the checker reproduces the kill-rule values, and (d) it contains a statement that is
not a restatement of BS Claims 5.1/5.6 (a sharper per-map value than "0 or >= 1", or a class larger than BS's
staged decompositions). Otherwise H0.

**Kill rule.** If the formula fails to reproduce (i) Strassen = 5 and (ii) BS's published value 2 for the
fully decomposed <3,3,3;23> algorithm (BS SPAA'19 Table 2), the statement is wrong: fix or retract.

**Registered numeric predictions (from the theory as first sketched, BEFORE the checker is run).**
Notation: for each M in {L, R, Q = P^T} (all r x d), D_M = number of projective row classes,
rho(M) = 0 if D_M < r; = 1 if D_M = r and some three rows are linearly dependent (collinear triple);
> 1 otherwise. Leading coefficient c* = 2 + rho(L) + rho(R) + rho(Q).
- Strassen, Winograd: D = 7 = r on all three, collinear triples present on all three: c* = 5.
- Laderman <3,3,3;23>: D = 23 on all three (BS: no duplicate rows); c* >= 5, equals 5 iff triples on all three sides.
- Strassen^{(x)2} (r = 49): triples present on all three sides: c* = 5.
- A Ballard-Benson-type <3,3,3;23> with duplicate rows on all three of L, R, Q: c* = 2 (BS Table 2 value 2).
- PLinOpt rank-48 variants: c* >= 3 + rho of the two sides without triples > 3 + 2 = 5 strictly, with 17/3 the best known.

**Named prior art (closest first).** Beniamini-Schwartz, "Faster matrix multiplication via sparse
decomposition", SPAA 2019 (Claims 5.1, 5.6, Cor. 3.14, Table 2); Karstadt-Schwartz, "Matrix
multiplication, a little faster", JACM 2020; the ACDA'25 paper (Theorem 6.1 lower bound for single-stage
decompositions; not read in full by me, cited via the 2026-09-28 verification note); the transposition
(Tellegen / Bordewijk) principle; Bostan-Lecerf-Schost, "Tellegen's principle into practice" (ISSAC 2003);
Probert / Bshouty (15 additions for Strassen-like); Dumas-Pernet-Sedoglavic (PLinOpt data) for the schemes.

## 1. Setting and notation

Scheme S = (L, R, P): L, R in F^{r x d}, P in F^{d x r}, d = n0^2, C = P((La) (.) (Rb)). Put Q := P^T (r x d),
so all three maps L, R, Q are "encoder-type" r x d matrices. Assume no zero rows in L, R and no zero column in P
(a zero factor deletes a product). rank L = rank R = rank Q = d (otherwise some input is invisible), hence r > d.
For M in {L, R, Q}: **D(M)** = number of projective row classes (rows equal up to a nonzero scalar), **u(M)** = number of
classes proportional to a unit vector e_i, and a **collinear triple** = three rows of distinct classes that are linearly
dependent (rank <= 2; all three coefficients are then nonzero). "Duplicate/proportional rows" means D(M) < r.

k-th power algorithm A_k(S; circuits): alpha = L^{(x)k} vec(A) and beta = R^{(x)k} vec(B) by arbitrary fan-in-2 linear
circuits, gamma_t = alpha_t beta_t (r^k scalar products), vec(C) = P^{(x)k} gamma by an arbitrary circuit. The depth-k recursive
block algorithm and every staged / alternative-basis / "transform once" variant is a special case, and so is any non-tensor
circuit for the three Kronecker-power maps. Let a(M,k) = minimum additions of a circuit for M^{(x)k}. Total
N*(k) = r^k + a(L,k) + a(R,k) + a(P,k); leading coefficient c*(S) := liminf_k N*(k)/r^k (coefficient of n^{log_{n0} r}; liminf because the limit need not exist for optimal circuits of arbitrary shape; every lower bound below bounds the liminf, hence every subsequence and the limit whenever it exists). Likewise rho(M) := liminf_k a(M,k)/r^k. Upper bounds (constructions) are proved for explicit staged circuits, where the limit exists, so they bound the limsup as well.
Model (F): scalings free (scalars on edges). Model (C): additionally every non-{+1,-1} edge weight costs 1.

## 2. Results

Labels: PROVED = proof below, attacked by me (section 5), independent proof gates: Stateful Codex gates 1-3 (gate 3 PASS, 2026-09-29) and a COLD no-memory Opus referee (2026-09-29, ACCEPT as PROVED for the stated scopes; AGENTS.md sec. 6/12). Read PROVED as PROVED within the stated circuit model and scope.

**Lemma T (transposition count). PROVED.** For any circuit G computing an m x n matrix M (every input used, no output
identically zero, and NO DEAD GATES: every gate feeds some output - cold-referee correction; a dead gate x0+x1 with outputs x0, x1 gives adds(G)=1 but adds(G^T)=2) the transposed circuit G^T computes M^T and adds(G^T) = adds(G) + m - n; scaling counts are equal.
Proof. Give each output its own sink node (indegree 1, cost 0). adds(G) = E - (V - n), E = edges, V = nodes (a node of indegree g
costs g-1). G^T reverses edges and keeps weights; its sources are the m sinks and its sinks the n inputs, so adds(G^T) = E - (V - m).
Subtract. Correctness is the Tellegen/Bordewijk principle (the reversed weighted DAG computes the transpose). Since (G^T)^T = G the
same bijection gives, for the minima, a(P,k) = a(Q,k) + r^k - d^k **exactly** for every k. Checked numerically (Strassen k=2: 57 -> 90 = 57+49-16;
rank-48 k=2: 6560 -> 8608 = 6560+2304-256). QED.
Consequence: since a(P,k) = a(Q,k) + r^k - d^k exactly and (d/r)^k -> 0, rho(P) = 1 + rho(Q), and **c* >= 2 + rho(L) + rho(R) + rho(Q)** in general (the "1" of the products and the "1" of the r^k - d^k fan-in of the decoder), with equality whenever the three sequences a(M,k)/r^k converge; 'exact' below means the lower bound and the explicit staged construction coincide, so the limit exists and equals the value.

**Lemma U (universal counting bound). PROVED.** Any fan-in-2 linear circuit computing M^{(x)k} uses >= D(M)^k - u(M)^k additions (model F, hence C).
Proof. The rows of M^{(x)k} are the Kronecker products y_1 (x) ... (x) y_k of rows of M, and two such are proportional iff factorwise proportional
(a Kronecker product of nonzero vectors determines its factors up to scalars), so there are D^k output classes, u^k of them unit classes. A node's value is
proportional to an input or to an addition gate (follow single-input scaling nodes back). An output whose class is not a unit class needs an addition gate in
that class, and distinct classes need distinct gates. QED.

**Theorem 1 (universal floor). PROVED.** For every scheme, k and circuits (finite-k inequality, no limit needed):
N*(k) >= r^k + [D_L^k - u_L^k] + [D_R^k - u_R^k] + [r^k - d^k + D_Q^k - u_Q^k]. Hence, taking liminf of the ratio to r^k (the right side over r^k converges),
**c*(S) = liminf N*(k)/r^k >= 2 + #{M in {L,R,Q} : D(M) = r}**. In particular c* >= 5 when the rows of L, R and the columns of P are pairwise non-proportional,
for ANY number of stages and ANY intermediate widths (indeed for any circuits), and c* >= 2 always (BS Claim 5.1).
Proof. Lemma U on L and R; Lemma T + Lemma U on P gives the finite-k inequality; divide by r^k; the right-hand side converges with no assumption on the circuits' own limit, since (d/r)^k -> 0, (u/r)^k -> 0 (u < r since r > d), (D/r)^k -> 1 if D = r else 0. QED.
The agent's argument is thus correct and stronger than stated (no staging needed); "duplicates weaken it" becomes exact: a side with D < r drops from >= 1 to 0,
while the decoder keeps >= 1 (not 2) if D(Q) < r.

**Theorem 2 (proportional rows are free at leading order). PROVED; the statement depends on the model.** Suppose D(M) < r. Along the explicit staged circuit:
model F: rho(M) = 0. Model C: rho(M) = 0 if the duplicates are SIGN-CLEAN (proportional rows agree up to sign, i.e. each copy is the representative times +-1, so the copies are free); for other duplicates (a copy is the representative times a scalar not in {+1,-1}) the construction proves only rho(M) in [0,1] (at most one scaling per copy in the final stage). Unqualified rho = 0 is NOT claimed in model C.
Proof. Chain: stage 1 = any circuit (cost a1) for the D class representatives (d -> D; D >= d since rank M = d); stage 2 = copies (free in model F; in model C free for sign-clean duplicates, otherwise at most one scaling per copy, contributing at most 1 to the r^k coefficient).
Lemma L gives the D-stage cost a1 (D^k - d^k)/(D - d) (a1 k d^{k-1} if D = d) = o(r^k) because D < r. rho >= 0 is trivial. QED. (BS Table 2: "2 n^{log 23}" is r^k x (1 + 0 + 0 + 1); its next monomial has base D. All duplicates in the tested data are sign-clean, so the tested values are exact in model C too.)

**Lemma L (lower-order stages; item (d)). PROVED.** Stage-by-stage lift of a chain d = h_0 <= ... <= h_c with stage j of cost q_j per unit batch costs exactly
sum_j q_j (h_j^k - h_{j-1}^k)/(h_j - h_{j-1}) (with q_j k h^{k-1} when h_j = h_{j-1}). If h_c = r and h_{c-1} < r (STRICT hypothesis, needed for the division), the r^k-coefficient of the final stage is q_c/(r - h_{c-1}). All earlier stages together are o(r^k): the unequal-width ones are O(h_{c-1}^k), and an equal-width stage of width q < r contributes q_j k q^{k-1}, which is o(r^k) but NOT O(q^k). Downstream arguments only use o(r^k).
Proof: when axis f is lifted, earlier axes are at width h_j and later ones at h_{j-1}; geometric sum. Verified by exact equality with explicit circuit counts (section 4).
So any state of width q < r is reachable at cost o(r^k), and only the final stage into width r matters. QED.

**Theorem 3 (value on a triple side). PROVED.** If D(M) = r and M has a collinear triple, rho(M) = 1 in model F (scalings free). In model C: rho(M) = 1 if some triple has both SYNTHESIS coefficients in {+1,-1}, i.e. some row equals +-m_i +-m_j for two other rows m_i, m_j (from alpha m_i + beta m_j + gamma m_l = 0 the coefficients are -alpha/gamma and -beta/gamma; the condition is that both are +-1; over a general field |.| is undefined, and over C the modulus condition |alpha| = |beta| = |gamma| is NOT sufficient, since e.g. coefficients 1 and i are not +-1). Otherwise rho(M) is only shown to lie in [1,3] in model C (one addition plus at most two scalings per synthesised row); a broader free-scaling model in which such edge scalings are free is model F, where the value is 1 exactly.
Proof. Lower bound: Theorem 1. Upper bound (a construction): let m_l = alpha m_i + beta m_j. State Z = the other r-1 rows (any circuit, cost o(r^k) by Lemma L); final stage = copies plus one gate for m_l; lifted cost r^k - (r-1)^k + o(r^k), so a(M,k)/r^k <= 1 + o(1). When triples chain (a line-closure: a seed S such that the remaining rows can be ordered so each is a
2-term combination of earlier rows) the same construction with seed size s <= r-1 gives the UPPER bound a(M,k) <= (r^k - s^k) + a1 (s^k - d^k)/(s - d), second-order base s. This is a construction, not an equality with the minimum a(M,k); the lower bound is Lemma U / Theorem 1. The checker outputs such seeds with exact certificates:
Strassen 4 (of 7), Winograd 4, Laderman 11 (of 23), Strassen^{(x)2} 18-20 (of 49), each rank-48 triple side 32 (of 48). QED.

**Theorem 4 (triple-free side, staged model; a lower bound, not tight). PROVED within the model.** Model S: state Z of q < r arbitrary forms (any lower-order chain), then an arbitrary
circuit from Z to the r rows; coefficient a/(r-q) (Lemma L). If D(M) = r and M has no collinear triple, let mu >= 1 be the maximum number of row-pair lines (lines through exactly two rows)
concurrent at a point that is not a row. Then rho_S(M) >= 1 + min_{h>=1} h / min(r, h(mu + h - 1)).
Proof. Let t = |Z n rows|, p = number of non-row elements of Z, T = number of non-row gates (temporaries), h = p + T, m = r - t rows outside Z. Each such row needs its own gate, so a >= m + T,
and q = t + p, so a/(r-q) >= (m+T)/(m-p) = 1 + h/(m-p) >= 1 + h/m. A ROW-OUTPUT gate (a gate whose output is a row of M) cannot have two row inputs, since its output would be a linear combination of two other rows, i.e. a collinear triple (rows of distinct classes, or a duplicate of a class, both excluded by D = r and no triple). Arbitrary gates may have two row inputs and produce a NON-row helper; that case is not excluded and is handled by counting the helper as one of the T temporaries (it costs an addition in a >= m + T and enlarges h). So each row gate has a helper input w (a non-row state element or temporary). The counting proof is otherwise unchanged.
If the other input is a row x, then y lies on the line through x and w, i.e. w lies on the row-pair line (x,y); w generates at most one row per such line (after one of the pair is generated both are present), hence at most mu rows.
If both inputs are helpers, y lies on the line through two helpers, which carries <= 2 rows (three would be a triple): <= 2 C(h,2) = h(h-1) rows. So m <= h mu + h(h-1) and m <= r. QED.
Numbers: mu = 4 on every triple-free side of the five rank-48 variants (pencil histogram {2:120, 3:16, 4:16}), giving rho_S >= 1.125 (h = 5 or 6) in the staged Model S only. Hence: **for arbitrary fan-in-2 circuits c* in [5, 17/3]; within the staged Model S only, c* in [5.25, 17/3]** for all five variants. The 5.25 is a staged final-stage-model bound and is never a bound for arbitrary circuits.

**Scope statements.** (b) Theorems 1-3 and Lemma T are about arbitrary circuits, but only for bilinear schemes that are the k-th tensor power of ONE base scheme with r^k scalar products; they say nothing
about Pan trilinear / approximate / border-rank algorithms, Winograd's commutative inner-product trick, input-dependent branching, or schemes that change base per level (the analogue would need D of Kronecker
products of different bases; not proved). Theorem 4 is proved only for staged circuits (Model S). Interleaved schedules (advancing different tensor factors to different stages): the adjacent-swap identity
cost(a then b) - cost(b then a) = X (q_b Delta_a - q_a Delta_b), independent of the other factors, is the chain-scheduling exchange rule whose optimum coarsens chains (= the best stage-by-stage over merged stages);
I did NOT write this out (CONJECTURE for arbitrary chains; EMPIRICAL: an exact lattice DP found 0 of 600 random toy chains below the best coarsening). For triple-free sides with arbitrary non-staged circuits I can prove only
rho >= 1; whether helper sets of size o(r^k) can serve r^k rows there is OPEN.
(c) Characteristic: Lemmas T, U and Theorems 1, 2 are field-independent (pure circuit counting); "proportional" and "collinear" are over F, so a duplicate or triple may exist over one field and not another; scalings
matter only in model C (all tested duplicates are sign-clean; in the tested data, all over Q, every triple side has at least one triple with synthesis coefficients +-1, which is what the checker's `clean_triples` counts, via |alpha| = |beta| = |gamma| over Q; that test is a Q-only shortcut for the +-1 condition and is not used over other fields).
(d) is Lemma L. Practical caveat: "leading coefficient" is asymptotic. A single triple gives state width r-1 and (47/48)^k decays only after k of order 100; closure seeds with s/r = 0.48 (Laderman) or 0.67 (rank-48 Q)
give real effect within about 10-20 levels.

## 3. What is new relative to the prior art (H0 vs H1)

Recovered (H0 part): the floor 5 for duplicate-free schemes and 2 for the all-duplicate case (BS SPAA'19 Claims 5.1, 5.6; Table 2) for multi-stage decompositions. Not found in the sources I read (I did NOT search exhaustively;
ACDA'25 Theorem 6.1 was not read): (1) the floor holds for arbitrary circuits of the tensor-power maps, not only (multi-)stage decompositions (Lemma U + T); (2) the per-side values rho = 0 (duplicates; model F, or sign-clean duplicates in model C) / 1 (a collinear triple; model F, or synthesis coefficients +-1 in model C) - triple-free sides are a THIRD case (rho >= 1, staged bound 1.125)
with certificates, so c* = 2 + #{D = r sides} exactly whenever every side is duplicate-or-triple, making 5 tight (model F) for Strassen, Winograd, Strassen^{(x)2} and the memory-typed Laderman-like <3,3,3;23> scheme (provenance unestablished; BS state ">= 5" for Laderman); (3) the refined staged
bound 1 + eps(mu) on triple-free sides; (4) a NECESSARY condition for beating 5 (some side has proportional rows, D < r); it is not sufficient, since the remaining sides can keep the total >= 5. Decision rule (d) is met by (2)-(4). Verdict: **H0 rejected in part, H1 accepted in part**: the headline floor is BS/KS-known; the
exactness/necessary-condition content: novelty relative to the literature is unassessed (the proofs themselves passed the gates).

## 4. Numerical cross-check (code/framework_floor_checks.py; output results/framework_floor_v1/summary.json)

Each scheme was first verified as an exact matmul identity under an auto-detected index convention (8 layouts, both P orientations). Laderman was typed from memory: it passes the exact tensor identity, but its historical provenance is unestablished (no source file was vendored). Sources: PLinOpt data (jgdumas/plinopt),
Ballard-Benson `grey333-23-*` and Smirnov 23 (arbenson/fast-matmul codegen/algorithms). c* (liminf) is EXACT in model F when every side is duplicate-or-triple (in model C the same value holds for a scheme exactly when its duplicates are sign-clean and each triple side has a triple with both synthesis coefficients +-1, true for every tabulated scheme by the checker's Q-shortcut test; otherwise only [value, value + 2] per triple side is shown), else the Theorem 4 lower bound (staged Model S only) is shown with the known 17/3 as upper bound; the arbitrary-circuit lower bound is the Theorem 1 value 5.

| scheme | r | D_L / D_R / D_Q | collinear triples L/R/Q | c* (model F) | closure seed (L,R,Q) |
|---|---|---|---|---|---|
| Strassen, Winograd (own + PLinOpt) | 7 | 7/7/7 | 3/3/3 | **5 exact** | 4,4,4 |
| Laderman 23 | 23 | 23/23/23 | 16/16/16 | **5 exact** (BS: >= 5) | 11,11,11 |
| Strassen^{(x)2} | 49 | 49/49/49 | 42/42/42 | **5 exact** | 20,19,18 |
| BB grey333-23-152 | 23 | 20/20/20 | 8/10/9 | **2 exact** (= BS Table 2) | not needed |
| BB grey333-23-142; 55-add; JS | 23 | 22/22/22; 20/20/20; 20/20/20 | present | **2 exact** | not needed |
| BB grey333-23-221, -143, -144; Smirnov 139; 58-add | 23 | L = 23, R and Q < 23 | present | **3 exact** | 11-15 on L |
| rank-48 canonical, sparse, 204, accurate | 48 | 48/48/48 | 0/0/16 | **in [5, 17/3] (arbitrary circuits); [5.25, 17/3] (staged Model S only)** | Q: 32 |
| rank-48 rational | 48 | 48/48/48 | 16/0/0 | **in [5, 17/3] (arbitrary circuits); [5.25, 17/3] (staged Model S only)** | L: 32 |

Kill rule: (i) Strassen c* = 5, and the certified chains (seed 4, naive state circuits) give total(k)/7^k = 5.0041 at k = 16 and 5.0000 at k = 32, matching Karstadt-Schwartz. (ii) BS Table 2 leading coefficient 2 for the
fully decomposed <3,3,3;23>: the formula gives 2 for every all-duplicate <3,3,3;23> in the data (grey333-23-142, -152, 55-add, JS), and Table 2's second monomial n^{log_3 20} has base equal to D = 20 of grey333-23-152 on all three sides
(corroboration; I cannot prove BS used exactly that file). Both pass; the statement is not killed. Registered predictions all matched (Strassen/Winograd/Strassen^2 = 5; Laderman = 5 with triples on all sides;
all-duplicate 23-term = 2; rank-48 strictly above 5).
Other checks: explicit stage-lifted circuits for all three sides of Strassen, Winograd, Laderman, grey-221 and rank-48 sparse (k <= 3) equal the Kronecker power, their measured addition counts equal the Lemma L closed form exactly and are
>= D^k - u^k (all True); Tellegen counts as in Lemma T; interleaving DP 0/600. Finite-depth effective coefficient total(k)/r^k of the certified chains (all lower-order terms and naive state circuits included):
Laderman 5.61 (k=4), 5.05 (k=8), 5.0002 (k=16); Strassen^{(x)2} 5.34 (k=4), 5.01 (k=8); BB-152 4.37 (k=8), 2.78 (k=16), 2.08 (k=32).

## 5. Attacks tried on my own proofs

1. Zero rows/columns, r = d, unit rows: handled (r > d; u^k counted; u < r).
2. Scalings hiding new classes: a scaling gate is proportional to its input, so it creates no class (Lemma U); model C only adds cost.
3. Shared subexpressions/fan-out cheating the gate count: the count is over gates, one class per gate; sharing is allowed and irrelevant.
4. Fan-in > 2 after transposition: counted as indegree - 1 additions; verified on explicit circuits (two schemes, k = 2).
5. Output side with duplicate columns: Lemma T gives r^k - d^k regardless, so rho(P) >= 1 always and rho(P) = 1 + rho(Q); BB decoders reach exactly 1 (2 = 1 + 0 + 0 + 1).
6. "Leading coefficient" gaming: Theorem 3 with s = r-1 is honest only asymptotically; I report closure seeds and the finite-k curve so 5 is not read as a small-n claim.
7. Theorem 4 loopholes: temporaries that are rows (counted as rows), helpers in the state (penalised through p), two-helper gates (h(h-1) term), a helper serving both rows of a line (impossible: the second row would already be present),
rows already in Z, row-output gates with two row inputs (excluded by no-triple); non-row gates with two row inputs (allowed, counted as temporaries in T). Not attacked successfully: cleverly built helper sets are bounded only through the crude h(h-1) term, which is why the bound is loose.
8. Non-staged circuits on triple-free sides: no proof; a counting contradiction with helper sets of size ~ sqrt(r^k) was searched for and NOT found, so it is OPEN and not claimed.
9. Interleaved schedules: exchange rule only sketched; DP evidence only on toy chains.
10. Data identity: every scheme re-verified exactly (tensor identity); Laderman from memory passed; the pack's "48 pairwise-distinct rows" reproduced for all five variants.
11. Class scope: an algorithm that is not the tensor power of one scheme is outside all statements (scope (b)).

## 6. Consequence for the mission and target for unit B

**So what.** In the tensor-power framework the leading coefficient of a scheme whose L rows, R rows and P columns are pairwise non-proportional is at least 5 for any circuits whatsoever, exactly 5 when each of L, R, P^T has a collinear
triple (Strassen, Winograd, Laderman, Strassen^{(x)2}), and for the published rank-48 scheme it lies in [5, 17/3] over arbitrary circuits ([5.25, 17/3] only if circuits are restricted to the staged Model S): nothing below 5 is possible for it or for any
duplicate-free scheme, and the 17/3 (cited as the published-scheme staged coefficient of the pack, verified independently by exact executable counts but not constructed in this document) can improve by at most 2/3 (0.417 in Model S). Proportional rows on some side (a shared factor: two products with the same left factor, right factor or decoding column) are NECESSARY to go below 5 (not sufficient: the other sides may keep the total >= 5); (model F; in model C only for sign-clean duplicates, Theorem 2) each such map removes its whole >= 1 asymptotically, at rate (D/r)^k.
This is a Floor-tier statement; it makes no speedup claim.

**Concrete target for unit B (rank48_shared_factor_search_v1)**, decreasing leverage, scored by `scheme_report` / `effective_curve` in the checker (importable):
1. A <4,4,4;48> scheme with a proportional pair in one of L, R, Q (D < 48). A duplicated side removes its whole >= 1 (rho = 0). CONDITIONAL upper bound: assuming the remaining triple-free side of a newly found scheme still admits the 4/3 construction known for the catalogued triple-free sides (NOT established for an arbitrary new scheme), one duplicated input side plus the 16-triple side on Q gives c* <= 2 + 4/3 + 1 = 13/3 < 5.
Every catalogued variant has D = 48 on all three maps. Practical value needs D/48 well below 1 (cost of that map ~ (D/48)^k relative to the leading term), roughly D <= 44 for gains at k <= 6.
2. Otherwise collinear triples on two or three of L, R, Q (each catalogued variant has them on exactly one map: 16 triples, closure seed 32; four in Q, 'rational' in L): two triple maps give c* >= 5.125 with upper bound 16/3 (CONDITIONAL on the same 4/3 assumption for the triple-free side), three give exactly 5.
3. HEURISTIC hint: a flip move (a(x)b(x)c + a(x)b'(x)c') needs two terms sharing a factor, so every flip-graph state with an available flip has D < r on that map; scoring every visited state by (D_L, D_R, D_Q, triple counts) is free and targets item 1.
4. Kill for B: if no 48-term scheme with a proportional pair (or with triples on two maps) exists in the searched orbit, the framework value is stuck in [5, 17/3] there ([5.25, 17/3] in the staged model); pivot to other exponents (e.g. <3,3,6;40>) instead of more 4x4 polishing.

Open items: the true value for triple-free maps (gap [1.125, 4/3]); non-staged circuits on triple-free maps; the interleaving exchange lemma; novelty search (ACDA'25 Thm 6.1, KS follow-ups).

Status: **PROVED (scoped) after gate 3 (2026-09-29; two prior gates FAILED/repaired).** Files: this document, `code/framework_floor_checks.py`, `results/framework_floor_v1/summary.json`, `results/framework_floor_v1/data/` (vendored scheme files plus `MANIFEST.json` with source URLs and SHA-256; PLinOpt data is CeCILL-B; the checker prefers these local files and needs no network).

## 7. Gate history

- **Gate 1 (2026-09-28), Stateful Codex referee, independent proof gate.** Verdict: FAILS AS WRITTEN; no counterexample to the theorem cores. Required repairs, all applied in repair round 1 (this revision):
  1. Rank-48 interval scope: arbitrary circuits c* in [5, 17/3]; staged Model S only [5.25, 17/3]. Fixed in the Theorem 4 numbers, table, section 6 and target item 4.
  2. Theorem 1 / definitions: c* and rho defined via liminf; Lemma T consequence restated as an inequality with equality when sequences converge; Theorem 1 stated as a finite-k inequality first.
  3. Theorem 2: model F rho = 0; model C rho = 0 only for sign-clean duplicates, otherwise rho in [0,1].
  4. Lemma L: strict hypothesis h_{c-1} < r added; equal-width term is q_j k q^{k-1} = o(r^k), not O(q^k); downstream uses only o(r^k).
  5. Theorem 3: line-closure formula is an UPPER bound on a(M,k), not equality; lower bound is Lemma U.
  6. Bounds 13/3 and 16/3 restated as CONDITIONAL on a 4/3 construction for the remaining side of a new scheme.
  7. 'Exact criterion for beating 5' replaced by 'necessary (not sufficient) condition' (sections 0-6; other docs checked by grep: no repetition).
  8. Checker: float 1e-9 Gram-determinant prefilter replaced by a sound modular prefilter (rank mod p of a random projection cannot exceed rational rank; no false negatives) followed by exact rational nullspace confirmation, for both triples and coplanar quadruples.
  9. Offline reproducibility: scheme files vendored under results/framework_floor_v1/data/ with MANIFEST.json (URLs, SHA-256); the loader uses local files first. Laderman remains typed from memory (provenance unestablished).
  10. Status set to PROVED-pending-delta-gate; ledger row appended.
- **Gate 2 (2026-09-28), Stateful Codex referee, delta gate on repair round 1.** Reviewer provenance: Stateful Codex referee (2026-09-28, delta). Verdict: repair round 1 accepted except two required repairs, applied in repair round 2 (this revision):
  1. Theorem 3, model C: the 'clean' condition |alpha| = |beta| = |gamma| is undefined over a general field and insufficient over C. Restated as: rho = 1 when both synthesis coefficients of a collinear triple are +-1 (or under the explicitly broader free-scaling model F); otherwise rho in [1,3] in model C. Theorem 2 wording, the scope note (c), the checker label and the summary labels changed consistently (checker output label only; no count changed).
  2. Theorem 4: 'no gate has two row inputs' was too strong. What the triple-free hypothesis excludes is a ROW-OUTPUT gate with two row inputs; arbitrary gates with two row inputs producing a non-row helper are allowed and counted in T. The counting proof is unchanged (they were already covered by the temporaries count).
  3. This gate history and the status set to PROVED-pending-delta2-gate.
- Gate 3 PASS (2026-09-29) and cold no-memory Opus referee ACCEPT (2026-09-29): see below.

### Gate 3 (2026-09-29) — PASS
Stateful Codex referee (neutral prompt, fresh review of the repaired document): Lemmas T, U, L and Theorems 1-4 hold in their stated models; the rank-48 intervals are correctly [5, 17/3] (arbitrary circuits) and [5.25, 17/3] (staged Model S only); Theorem 2/3 model-C qualifications confirmed. No further repairs required.

### Cold referee (2026-09-29) — ACCEPT
A Claude Opus subagent with no repository memory read this document cold and re-derived the claims: Lemma T verified on 300 random circuits with duplicated outputs and weights 1,2,3 (transposition identity exact; needs the no-dead-gate hypothesis, now added); Lemma U, Theorems 1-4, Lemma L verified; the checker's Strassen and rank-48 outputs reproduced. Verdict: accept as PROVED for the scopes stated. Four over-strong sentences and four stale status lines were corrected.
