# Two-axis complexity: multiplications and precision bits

Date: 2026-08-01

Status: **ANALYTICAL ROUND COMPLETE; NEW THEOREMS ARE PROOF PRESENT AND INDEPENDENT-REVIEW PENDING.** Exact finite computations are labeled separately.

Scope: standard termwise evaluation of displayed bilinear degenerations, exact CPU inspection of the repository's authenticated coefficient tables, and no GPU, formatter, solver, MP-93 sweep, or census access.

Labels used below are literal:

- **PROVED (prior/standard):** a result already covered by the repository's accepted frame or a standard identity used only as background;
- **PROOF PRESENT; REVIEW PENDING:** a new derivation in this unit, not promoted to repository `PROVED` before the independent proof-review gate;
- **EXACT COMPUTATION:** integer, rational, Laurent, or finite-field replay;
- **EMPIRICAL:** deterministic floating spectral computation on exact inputs;
- **CONJECTURE:** a proposed interpretation not implied by the derivation.

## 0. Verdict

The correct complexity object for an approximate bilinear scheme at requested error `epsilon` is a Pareto pair

\[
\mathcal C_\epsilon(\mathcal D)=\bigl(r,P_{\mathcal D}(\epsilon)\bigr),
\]

where `r` is the number of nonscalar bilinear products and `P_D(epsilon)` is the least working precision that guarantees the requested error in a declared evaluation model. A scalar `r M(P)` may be formed only after naming the bit-complexity `M(p)` of a `p`-bit multiplication and the linear-operation schedule.

The central conjecture survives, but in a more exact form:

> **PROOF PRESENT in the declared evaluation model.** If a degeneration has approximation order `q` and effective cancellation pole order `h`, then its total working precision has leading term `(1+h/q) log2(1/epsilon)`. The border tax is the extra `(h/q) log2(1/epsilon)` bits. Border rank moves cost from products to bits; it does not erase it.

For the authenticated Smirnov-46 table, `q=1` and `h=3`. At the stated real accuracy

\[
\epsilon=2^{-53},
\]

the displayed one-level scheme needs at least **213 bits**, and **217 bits** suffice in the idealized one-rounding-per-displayed-summand model. An actual straight-line implementation needs additional schedule guard bits. DPS-48 has zero border-approximation tax: its baseline is 53 bits plus its ordinary schedule/stability guard. AlphaTensor-47 is exact at one bit over `F2`, but it is not an algorithm for 53-bit real matrix multiplication.

The crossover verdict is decisive for allocation:

- **locally tuned black-box recursion:** Smirnov-46 never beats DPS-48 in the declared bit model; every recursive level makes the ratio worse;
- **optimally flattened shared-parameter tensor powers:** Smirnov can win only asymptotically. At 53 target bits, the crossover is bracketed between depths `143--147` even under the impossible-to-beat linear `M(p)=p` envelope, `169--173` under `M(p)=p log2 p`, and `324--332` under schoolbook `M(p)=p^2`. The upper endpoints are certified by the sufficient bound and correspond to matrix dimensions about `10^88.50`, `10^104.16`, and `10^199.88`.

**EMPIRICAL/DECISION VERDICT.** Border-46 is exponent-only for every practical scale supported by this accounting. The practical 4x4 hunt should therefore be exact-first. Border work remains relevant to asymptotic theory and to finding degenerations with a materially smaller ratio `h/q`; it is not a deployable competitor to exact-48 as presently instantiated.

This verdict is favorable to Smirnov: it omits its 245 additions, 120 cached parameter scales, parameter setup, encode/decode traffic, and executable schedule guard bits. Including them cannot create an earlier crossover.

## 1. Exact definition of the pair

Let `T` be a bilinear tensor in a fixed coefficient basis and let a displayed length-`r` degeneration be

\[
B(t)=\sum_{i=1}^{r} U_i(t),\qquad U_i(t)=a_i(t)\otimes b_i(t)\otimes c_i(t),
\]

with Laurent-polynomial or convergent Laurent-series factors for real `0<t<=t_0`. This document uses coefficient max norm

\[
\|X\|_\infty=\max_j |X_j|.
\]

The choice makes the 4,096 Brent coordinates auditable and does not hide a dimension constant in the Smirnov calculation. Another norm changes constants, not the logarithmic slope, provided the dimension is fixed.

### 1.1 Approximation order and cancellation pole order

The **approximation order** is `q>=1` when constants `0<a<=A` exist with

\[
a t^q\le \|B(t)-T\|_\infty\le A t^q\quad(0<t\le t_0).
\]

The displayed termwise **cancellation mass** is

\[
\kappa_{\mathcal D}(t)=\max_j\sum_{i=1}^{r}|(U_i(t))_j|.
\]

The **effective pole order** is `h>=0` when constants `0<c<=C` exist with

\[
c t^{-h}\le\kappa_{\mathcal D}(t)\le C t^{-h}.
\]

This is more precise than taking the largest pole appearing anywhere. A formal pole that a fixed exact preprocessing identity removes before numerical evaluation is not charged. A pole that remains in the displayed termwise cancellation mass is charged. The definition is representation- and schedule-specific on purpose: precision is a property of a realization, not of border rank alone.

### 1.2 Standard termwise evaluation model

Let unit roundoff be `u_p=2^{-p}`. In the ideal termwise model, each displayed rank-one summand is perturbed independently:

\[
\widehat U_{i,j}(t)=U_{i,j}(t)(1+\delta_i),\qquad |\delta_i|\le u_p.
\]

Therefore the worst permitted roundoff in coefficient `j` is exactly `u_p sum_i |U_{i,j}(t)|`, and the global roundoff is bounded by `u_p kappa_D(t)`. In an operation-level straight-line program, replace `u_p` by the standard `gamma_s=s u_p/(1-s u_p)` for the relevant chain length `s`. That adds `log2 s+O(1)` guard bits but does not change the leading slope.

Define

\[
P_{\mathcal D}(\epsilon)=\min\left\{p:\ \exists t\in(0,t_0]\text{ such that every allowed rounding perturbation gives }\|\widehat B(t)-T\|_\infty\le\epsilon\right\}.
\]

The honest pair is

\[
\boxed{\mathcal C_\epsilon(\mathcal D)=(r,P_{\mathcal D}(\epsilon)).}
\]

Pairs are partially ordered coordinatewise. Rank alone is the projection onto the first axis. Given a named integer/real multiplication cost `M(p)`, the product-bit proxy is

\[
\operatorname{PB}_\epsilon(\mathcal D)=rM(P_{\mathcal D}(\epsilon)).
\]

It is not a full implementation cost until additions, scalar constants, encode/decode, memory, and the actual error schedule are also charged.

## 2. Precision-transfer theorem

### Theorem 2.1 -- necessary precision slope

**PROOF PRESENT; REVIEW PENDING.** Under Sections 1.1--1.2, any evaluation that guarantees coefficient error at most `epsilon` for every permitted rounding perturbation must satisfy

\[
P_{\mathcal D}(\epsilon)\ge\left(1+\frac hq\right)\log_2\frac1\epsilon+\log_2 c+\frac hq\log_2 a.
\tag{2.1}
\]

Ceil the right side for an integral bit count.

**Derivation.** A guarantee against zero rounding error first forces the exact truncation below `epsilon`, hence

\[
a t^q\le\epsilon,\qquad t\le(\epsilon/a)^{1/q}.
\]

Because rounding perturbations are independent, an adversary may align every perturbation with the sign of the summand in a coordinate attaining the cancellation mass. The guarantee therefore also forces

\[
u_p c t^{-h}\le\epsilon.
\]

Combining the two inequalities gives

\[
u_p\le c^{-1}a^{-h/q}\epsilon^{1+h/q}.
\]

Taking negative base-two logarithms proves (2.1). The derivation is a lower bound for the displayed standard evaluation model, not for every symbolic reformulation of the same border identity.

### Theorem 2.2 -- matching sufficient slope

**PROOF PRESENT; REVIEW PENDING.** The error bound

\[
E(t,p)\le A t^q+C u_p t^{-h}
\tag{2.2}
\]

has its interior optimum where

\[
t^{h+q}=\frac{h C u_p}{q A}.
\]

Equivalently, allocating the optimum truncation share gives

\[
t^q=\frac{h\epsilon}{A(h+q)}
\]

and it suffices that

\[
u_p\le\frac{qA}{hC}\left(\frac{h\epsilon}{A(h+q)}\right)^{(h+q)/q}.
\tag{2.3}
\]

Thus

\[
P_{\mathcal D}(\epsilon)=\left(1+\frac hq\right)\log_2\frac1\epsilon+O(1).
\tag{2.4}
\]

The leading term includes the target's own `log2(1/epsilon)` bits. The additional border tax is

\[
\boxed{\Delta P(\epsilon)=\frac hq\log_2\frac1\epsilon+O(1).}
\]

This corrects the loose phrase “`h log(1/epsilon)` bits”: for first-order contact (`q=1`), that is the extra tax, while total working precision is `(h+1) log2(1/epsilon)+O(1)`.

### What `h` alone cannot prove

**PROOF PRESENT by counterexample shape.** Formal pole order alone is insufficient. One may write an exact tensor as `t^-h X-t^-h X+T`; a compiler that cancels the first two terms symbolically has no precision tax, while literal termwise evaluation does. A rigorous claim therefore needs all three items:

1. approximation order `q`;
2. a lower bound on the cancellation mass of the declared realization; and
3. an evaluation schedule or explicit perturbation model.

The two-axis conjecture should be stated using effective cancellation order, not raw Laurent degree alone.

## 3. Smirnov-46: exact pole and precision certificate

The exact source object is `research/smirnov46_laurent_table.py`, SHA-256 `1ebd908cc6db1e6cbb7629356783c6ef4a19852ee70751bf7c51218e39e9ea13`. The independent extractor/replayer is `research/verify_smirnov46.py`, SHA-256 `ef445d5f91960617609cd9c851242bbec84bcee5f21939a4a8173e9832ce803c`.

### 3.1 Pole anatomy

**EXACT COMPUTATION.** In source factor order `(gamma,alpha,beta)`:

| factor-valuation triple | summands | complete-summand pole order |
|---|---:|---:|
| `(-1,-1,-1)` | 35 | 3 |
| `(0,-1,-1)` | 6 | 2 |
| `(-1,-1,0)` | 5 | 2 |

The 35 pole-order-three labels are

`2,3,4,5,6,7,8,9,11,13,14,15,19,20,21,22,25,26,27,29,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46`.

The raw complete-summand Laurent range is `[-3,6]`. After exact cancellation and subtraction of `T4`, the residual range is `[1,6]`, with nonzero coefficient counts

| degree | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|
| residual coefficients | 37 | 50 | 37 | 3 | 4 | 2 |

At negative degrees `-3,-2,-1`, respectively `34,32,37` tensor coordinates contain displayed contributions. Their total absolute coefficient masses are `68,64,74`, and every coefficient sum cancels exactly to zero. At degree `-3` the coordinate `(gamma,alpha,beta)=(14,15,1)` contains term 3 with `+t^-3` and term 37 with `-t^-3` (zero-based flattened coordinate indices).

### 3.2 Exact norm laws

**EXACT COMPUTATION.** For every real `0<t<=1/2`, direct Laurent inspection gives

\[
\|B(t)-T_4\|_\infty=t.
\tag{3.1}
\]

The upper bound follows because every residual coordinate polynomial has absolute value at most `t`; equality is witnessed by coordinate `(15,12,11)`, whose residual is exactly `t`.

For the displayed complete-summand evaluation,

\[
\kappa_{\rm Sm46}(t)=2t^{-3}.
\tag{3.2}
\]

The upper bound follows by multiplying every coordinate's absolute Laurent mass by `t^3` and checking its nonnegative polynomial on `(0,1/2]`; the maximum is 2. Equality is attained by the explicit `+t^-3,-t^-3` cancellation above. Consequently Smirnov has exact `q=1`, `h=3`, `a=A=1`, and `c=C=2` in the declared model.

### 3.3 Required bits

The ideal termwise error bound is

\[
E(t,p)\le t+2u_p t^{-3}.
\tag{3.3}
\]

The necessary bound from the two witness coordinates is

\[
u_p\le\frac{\epsilon^4}{2},\qquad P_{\rm Sm46}(\epsilon)\ge\left\lceil4\log_2\frac1\epsilon+1\right\rceil.
\tag{3.4}
\]

Optimizing (3.3) chooses

\[
t_*=\frac{3\epsilon}{4}
\]

and gives the sufficient condition

\[
u_p\le\frac{27}{512}\epsilon^4,\qquad P_{\rm Sm46}(\epsilon)\le\left\lceil4\log_2\frac1\epsilon+\log_2\frac{512}{27}\right\rceil
\tag{3.5}
\]

inside the ideal termwise model. For `epsilon=2^-53`, (3.4)--(3.5) give

\[
213\le P_{\rm Sm46}(2^{-53})\le217.
\]

The upper endpoint uses the exact dyadic parameter `t_*=3*2^-55`. It is not a certified executable upper bound for the published straight-line algorithm: operation-level rounding replaces `u_p` by `gamma_s` and requires schedule guard bits. The exact table supplies the slope and ideal constants; it does not supply a fully optimized numerical SLP.

## 4. First honest 4x4 rank/precision/effective-cost table

Stated real target: `epsilon=2^-53`. `M_1(p)=p` is an information-transfer lower envelope, and `M_2(p)=p^2` is schoolbook integer mantissa multiplication. The effective columns count only nonscalar bilinear products; they are lower bounds on total implementation work.

| scheme | coefficient domain | rank/products | border `(q,h)` | working bits | extra border bits | `r M_1(p)` | `r M_2(p)` | honest scope |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Smirnov-46 | real evaluation of `Q(t)` | 46 | `(1,3)` | 213 necessary; 217 ideal sufficient | 160--164 plus schedule guard | 9,982 at `p=217` | 2,166,094 at `p=217` | approximate `T4`; displayed termwise model |
| DPS-48 | exact `Q`, dyadic coefficients | 48 | exact | 53 baseline plus ordinary schedule guard | 0 approximation tax | 2,544 | 134,832 | exact when 2 is invertible; no optimized SLP assumed here |
| AlphaTensor-47 | exact `F2` | 47 | exact | 1 field bit | 0 | 47 | 47 | exact parity arithmetic only; not a real-accuracy competitor |

At one level, even the linear lower-envelope ratio is

\[
\frac{46\cdot217}{48\cdot53}=3.923742\ldots,
\]

and the quadratic ratio is `16.065133...`. The 6.25% reduction from 48 to 46 products does not compensate for the approximately fourfold mantissa.

**CONJECTURE, not established by this table.** A search objective that jointly minimizes rank and effective cancellation order `h/q` may find a useful border scheme. Rank-only search has no mechanism to prefer such a point.

## 5. Recursion and the practical exponent

There are two mathematically different meanings of “apply recursively.” They must not be conflated.

### 5.1 Local black-box recursion

Suppose every recursive call independently selects its parameter to meet the accuracy requested by its parent. Let `L_j` be the bits requested at level `j`. Theorem 2.1 gives the recurrence

\[
L_{j+1}\ge\left(1+\frac hq\right)L_j+O(1).
\tag{5.1}
\]

Hence

\[
L_d=\Omega\left(\left(1+\frac hq\right)^d L_0\right).
\tag{5.2}
\]

If the border scheme uses `r` products and an exact comparator uses `R`, then for `M(p)=Theta(p^alpha)` the product-bit ratio is multiplied per level by

\[
\boxed{f_{\rm local}(h,q,\alpha)=\frac rR\left(1+\frac hq\right)^\alpha.}
\tag{5.3}
\]

This is the requested theorem shape “recursive bit-cost multiplies by `f(h)`,” with the missing dependencies `q`, `r/R`, and the bit-arithmetic exponent made explicit.

For Smirnov versus DPS,

\[
f_{\rm local}=\frac{46}{48}4^\alpha.
\]

It equals `23/6=3.8333...` for linear bit cost and `46/3=15.3333...` for quadratic bit cost. **PROOF PRESENT in this local model:** no accuracy or depth produces a crossover.

### 5.2 Flattened shared-parameter tensor power

Local recursion is not optimal. One can first tensor the algebraic identity to depth `d` and then evaluate one shared parameter. The flattened object has `r^d` products, pole order `dh`, and approximation order still at least `q`. Its precision grows linearly in `d` at fixed target bits, rather than as `(1+h/q)^d`. This is the strongest honest case for border rank.

For Smirnov, (3.1)--(3.2) give

\[
\|B(t)^{\otimes d}-T_4^{\otimes d}\|_\infty\le(1+t)^d-1,
\]

and flattened cancellation mass

\[
\kappa_d(t)=(2t^{-3})^d.
\]

For fixed `d`, maximize the admissible roundoff

\[
u(t)=\frac{\left[\epsilon-((1+t)^d-1)\right]t^{3d}}{2^d}.
\]

The unique interior optimizer `t_d` satisfies

\[
(1+t_d)^{d-1}(3+4t_d)=3(1+\epsilon).
\]

The sufficient idealized precision is therefore

\[
p_d=\left\lceil d+3d\log_2\frac1{t_d}+\log_2\frac1{\epsilon-((1+t_d)^d-1)}\right\rceil,
\qquad L=\log_2\frac1\epsilon.
\tag{5.4}
\]

Since `t_d=epsilon/d+o(epsilon/d)` and the roundoff budget is asymptotic to `epsilon/(3d+1)`,

\[
p_d=(1+3d)L+3d\log_2 d+O(d)
\tag{5.5}
\]

for small `epsilon`. The ratio

\[
\mathcal R_d[M]=\left(\frac{46}{48}\right)^d\frac{M(p_d)}{M(L)}
\tag{5.6}
\]

eventually falls below one because an exponential rank advantage beats the polynomial-in-`d` precision factor. “Eventually” is galactic.

### 5.3 Crossover computation

**EXACT FORMULAS / HIGH-PRECISION NUMERICAL ROOT SEARCH.** The tensor power has a coefficient whose truncation error is exactly `t`: use the residual-only coordinate `(15,12,11)` in one block and any of the 64 target coordinates, all of which the Smirnov table computes exactly as 1, in the other blocks. Together with `kappa_d=(2t^-3)^d`, every guaranteed evaluation must therefore obey

\[
p_d^{\rm nec}\ge (1+3d)L+d.
\tag{5.7}
\]

The table scans integer depths from 1 upward. “First possible” is where the necessary-cost lower bound first drops below DPS; no realization can cross earlier. “First certified” is where the optimized sufficient precision (5.4) first drops below DPS. The actual ideal-model crossover lies inside the bracket. `log10 n=d log10 4` at the certified endpoint for `n=4^d`.

| target bits `L` | bit multiplication model | first possible `d` (`p_nec`) | first certified `d` (`p_suf`) | certified `log10 n` |
|---:|---|---:|---:|---:|
| 24 | `M(p)=p` | 143 (10,463) | 151 (14,337) | 90.9111 |
| 24 | `M(p)=p log2 p` | 173 (12,653) | 182 (17,420) | 109.5749 |
| 24 | `M(p)=p^2` | 324 (23,676) | 341 (33,536) | 205.3025 |
| 53 | `M(p)=p` | 143 (22,933) | 147 (26,759) | 88.5028 |
| 53 | `M(p)=p log2 p` | 169 (27,093) | 173 (31,603) | 104.1564 |
| 53 | `M(p)=p^2` | 324 (51,893) | 332 (61,526) | 199.8839 |
| 113 | `M(p)=p` | 143 (48,733) | 145 (52,547) | 87.2987 |
| 113 | `M(p)=p log2 p` | 166 (56,553) | 168 (60,970) | 101.1461 |
| 113 | `M(p)=p^2` | 324 (110,273) | 328 (119,869) | 197.4757 |

At 53 bits, selected shared-parameter ratios are:

| depth | `p_d` | ratio for `M(p)=p` | ratio for `M(p)=p^2` |
|---:|---:|---:|---:|
| 1 | 217 | 3.9237 | 16.0651 |
| 5 | 894 | 13.6347 | 229.9886 |
| 10 | 1,760 | 21.6972 | 720.5092 |
| 100 | 18,056 | 4.8306 | 1,645.7004 |
| 147 | 26,759 | 0.9686 | 489.0209 |
| 332 | 61,526 | 0.000848 | 0.9842 |

The practical conclusion does not depend on whether one chooses linear, quasilinear, or quadratic mantissa multiplication. The optimistic crossover dimension is already beyond physical relevance, and actual linear work moves it later.

## 6. Entropy-defect fingerprint

For every nonzero displayed factor matrix `M`, define in bits

\[
\rho_M=\frac{M^*M}{\operatorname{tr}(M^*M)},\qquad S(M)=-\operatorname{tr}(\rho_M\log_2\rho_M),
\]

\[
H_{\rm diag}(M)=-\sum_j(\rho_M)_{jj}\log_2(\rho_M)_{jj},\qquad \Delta_S(M)=H_{\rm diag}(M)-S(M)\ge0.
\]

**PROVED (standard Schur--Horn/Schur concavity).** The defect is nonnegative. It is invariant under nonzero scalar rescaling of a factor, so the ordinary per-term tensor gauge does not affect it.

**PROOF PRESENT; important limitation.** It is not invariant under a change of the matrix's column basis. Choosing right singular vectors diagonalizes `M^*M` and makes `Delta_S(M)=0` for every `M`. It is also not invariant under general tensor-scheme basis equivalence. Therefore the table below is a **displayed-coordinate fingerprint**, not an intrinsic complexity invariant, and “entropy defect equals exploitable structure” is too strong without a restricted transform class and a cost for the transform. Schoolbook's all-zero row is the simplest counterwarning.

### 6.1 Aggregation contract

For a rank-`R` scheme with displayed factor matrices `(A_i,B_i,C_i)`, compute the defect of all `3R` matrices in their repository row/column orientation and report means, median, maximum, and the count numerically zero within `1e-12`. Rectangular factors use the same `M^*M` definition. Logarithms are base two.

The finite-table scope includes every canonical exact coefficient table in the repo's current compact census plus every finite dataset in `b402_decomposition_data.py`, the large exact data modules, the authenticated AlphaTensor-47 object, the authenticated Smirnov curve, and schoolbook baselines `n=2..6`. The two Johnson--McLoughlin families in `b409_jm_data.py` are exact *parametric families* with no canonical parameter specialization; assigning them one scalar fingerprint would be arbitrary, so their fingerprints remain functions of the parameters. Transient generated search states without canonical full factor tables are likewise excluded.

For Smirnov, the accuracy-matched slice is the exact rational `t=3*2^-55`. For AlphaTensor-47, the table uses the authenticated canonical real `0/1` lift; this is a coordinate fingerprint, not a quantum entropy intrinsic to `F2`.

### 6.2 Fingerprint table

**EMPIRICAL on exact inputs.** Spectra were computed deterministically in float64 after per-matrix scaling. Independent SVD and `eigvalsh(M^*M)` paths agreed on the three central 4x4 schemes to at most `1.11e-15` defect bits.

| scheme | field | dims | R | matrices | mean `S` | mean `Hdiag` | mean defect | median defect | max defect | zero defects |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strassen-7 | `Z` / arbitrary ring | 2x2x2 | 7 | 21 | 0.142857143 | 0.428571429 | 0.285714286 | 0.000000000 | 1.000000000 | 15 |
| Laderman-23 | `Z` | 3x3x3 | 23 | 69 | 0.161070431 | 0.468978120 | 0.307907689 | 0.000000000 | 1.000000000 | 39 |
| `SMIRNOV_333_23` | ternary `Z` | 3x3x3 | 23 | 69 | 0.224644603 | 0.538883454 | 0.314238851 | 0.000000000 | 1.000000000 | 41 |
| `BALLARD_BENSON_333_23_142` | ternary `Z` | 3x3x3 | 23 | 69 | 0.221488083 | 0.469373061 | 0.247884978 | 0.000000000 | 1.584962501 | 46 |
| `BALLARD_BENSON_333_23_143` | ternary `Z` | 3x3x3 | 23 | 69 | 0.173752713 | 0.490229878 | 0.316477165 | 0.000000000 | 1.584962501 | 43 |
| `BALLARD_BENSON_333_23_144` | ternary `Z` | 3x3x3 | 23 | 69 | 0.212238285 | 0.560121285 | 0.347883001 | 0.000000000 | 1.584962501 | 40 |
| `BALLARD_BENSON_333_23_152` | ternary `Z` | 3x3x3 | 23 | 69 | 0.187902289 | 0.554652928 | 0.366750640 | 0.000000000 | 1.584962501 | 38 |
| `BALLARD_BENSON_333_23_221` | ternary `Z` | 3x3x3 | 23 | 69 | 0.253629679 | 0.835448243 | 0.581818563 | 0.666666667 | 1.584962501 | 23 |
| `SMIRNOV_336_40` | `Q` | 3x3x6 | 40 | 120 | 0.720887909 | 1.477713752 | 0.756825843 | 0.808465470 | 1.395750649 | 0 |
| `SMIRNOV_363_40` | `Q` | 3x6x3 | 40 | 120 | 0.720887909 | 1.442844883 | 0.721956975 | 0.653838112 | 1.395750649 | 0 |
| `SMIRNOV_633_40` | `Q` | 6x3x3 | 40 | 120 | 0.720887909 | 1.217282199 | 0.496394290 | 0.584962501 | 1.000000000 | 0 |
| `COURTOIS_BARD_HULME_333_23` | ternary `Z` | 3x3x3 | 23 | 69 | 0.271824042 | 0.635047735 | 0.363223693 | 0.000000000 | 1.584962501 | 40 |
| `OH_KIM_MOON_333_23` | ternary `Z` | 3x3x3 | 23 | 69 | 0.227866687 | 0.649104171 | 0.421237484 | 0.368248074 | 1.584962501 | 29 |
| `SMIRNOV_TERNARY_LINZ_333_23` | ternary `Z` | 3x3x3 | 23 | 69 | 0.224644603 | 0.538883454 | 0.314238851 | 0.000000000 | 1.000000000 | 41 |
| `ALPHATENSOR_333_23_STANDARD` | integral | 3x3x3 | 23 | 69 | 0.172441527 | 0.565140666 | 0.392699139 | 0.000000000 | 1.584962501 | 37 |
| AlphaTensor-49 standard | ternary `Z` | 4x4x4 | 49 | 147 | 0.166339202 | 0.835466242 | 0.669127040 | 0.736496149 | 2.000000000 | 51 |
| DPS-48 | dyadic `Q` | 4x4x4 | 48 | 144 | 0.302642637 | 1.173837384 | 0.871194748 | 1.000000000 | 2.000000000 | 44 |
| MP-153 | `Z` | 6x6x6 | 153 | 459 | 0.406681537 | 1.225491699 | 0.818810162 | 0.899123963 | 2.584962501 | 88 |
| MP-93 | `Z` | 5x5x5 | 93 | 279 | 0.368960067 | 1.058299676 | 0.689339609 | 0.649022500 | 2.000000000 | 85 |
| Perminov-486 | ternary `Z` | 9x9x9 | 486 | 1,458 | 0.627148563 | 1.258832779 | 0.631684216 | 0.499855008 | 2.321928095 | 410 |
| KMW-153 | `Q` | 6x6x6 | 153 | 459 | 0.404408900 | 1.191220968 | 0.786812068 | 0.899123963 | 2.000000000 | 69 |
| AlphaTensor-47 canonical `0/1` lift | `F2` lifted to `R` | 4x4x4 | 47 | 141 | 0.542226268 | 0.886108939 | 0.343882671 | 0.368248074 | 1.584962501 | 52 |
| Smirnov-46 at `t=3*2^-55` | `Q(t)` evaluated in `Q` | 4x4x4 | 46 | 138 | 0.127899589 | 0.199346196 | 0.071446607 | 0.000000000 | 1.584962501 | 128 |
| schoolbook `n=2` | arbitrary ring | 2x2x2 | 8 | 24 | 0 | 0 | 0 | 0 | 0 | 24 |
| schoolbook `n=3` | arbitrary ring | 3x3x3 | 27 | 81 | 0 | 0 | 0 | 0 | 0 | 81 |
| schoolbook `n=4` | arbitrary ring | 4x4x4 | 64 | 192 | 0 | 0 | 0 | 0 | 0 | 192 |
| schoolbook `n=5` | arbitrary ring | 5x5x5 | 125 | 375 | 0 | 0 | 0 | 0 | 0 | 375 |
| schoolbook `n=6` | arbitrary ring | 6x6x6 | 216 | 648 | 0 | 0 | 0 | 0 | 0 | 648 |

The Smirnov fingerprint is stable in the high-accuracy regime: its mean defect is `0.076457956` at the source display point `t=0.1`, `0.071446606` at the 24-bit accuracy-matched point, and `0.071446607` at both the 53-bit and 113-bit points. This does not cure the coordinate-invariance problem.

## 7. What is proved, measured, and still conjectural

| claim | label | exact boundary |
|---|---|---|
| precision slope is `(1+h/q)L` | **PROOF PRESENT; REVIEW PENDING** | standard displayed termwise perturbation model with two-sided truncation and cancellation-mass bounds |
| Smirnov has `q=1`, `h=3`, truncation `t`, and cancellation mass `2t^-3` | **EXACT COMPUTATION** | authenticated 46-term table, coefficient max norm, `0<t<=1/2` |
| Smirnov needs 213--217 idealized bits at `epsilon=2^-53` | **PROOF PRESENT + EXACT CONSTANTS** | lower/upper bracket in the declared ideal termwise model; executable guard bits excluded |
| locally recursive Smirnov never beats exact DPS in product-bit cost | **PROOF PRESENT; REVIEW PENDING** | recurrence (5.1), `M(p)=Theta(p^alpha)`, `alpha>=1` |
| flattened Smirnov crossover brackets are those in Section 5.3 | **EXACT LOWER BOUND + HIGH-PRECISION EVALUATION OF THE SUFFICIENT FORMULA** | ideal shared-parameter product-bit model; additions and schedule guards excluded |
| border-46 is not practical against exact-48 | **EMPIRICAL/DECISION VERDICT** | all declared cost models and current artifacts; not a theorem against every future implementation |
| complexity should be a Pareto pair | **CONJECTURE / PROPOSED DEFINITION** | useful objective choice; not a theorem that every computational model reduces to two coordinates |
| entropy defect is nonnegative | **PROVED (standard)** | fixed displayed basis |
| entropy defect is an intrinsic measure of exploitable structure | **REFUTED AS STATED** | arbitrary right-basis choice always makes it zero |
| fingerprint table values | **EMPIRICAL ON EXACT INPUTS** | declared orientations, lifts, parameter slice, and numerical tolerance |

## 8. Allocation consequence and next theorem target

The present border route loses because `h/q=3`. The mission-relevant search variable is therefore not rank alone but the three-way frontier

\[
(r,\ h/q,\ \text{linear realization cost}).
\]

An exact rank-48 improvement has zero approximation tax and is the practical priority. A border successor should not be funded merely because `r<48`; it should first exhibit one of:

1. `h/q` small enough that `r(1+h/q)^alpha<48` in the local model;
2. a symbolic cancellation compilation that provably reduces effective `h`;
3. an interpolation/exact-recovery protocol whose full bit cost beats exact-48;
4. a useful restricted input family whose norm or coefficient structure suppresses the cancellation mass.

For Smirnov's `r=46`, the local necessary threshold is

\[
\frac hq < \left(\frac{48}{46}\right)^{1/\alpha}-1.
\]

That is less than `0.04348` even for linear bit cost and less than `0.02151` for quadratic bit cost, far below the authenticated value 3. This numerical gap, not the two-product saving, is the practical fact.

The entropy table suggests a separate correction to the owner's frame: retain entropy defect only after freezing a cheap transform class and charging the transform. A viable matrix-native fingerprint should be a cost-constrained orbit quantity, for example

\[
\Delta_{\mathcal G,C}(M)=\max_{g\in\mathcal G:\ \operatorname{cost}(g)\le C}\Delta_S(Mg),
\]

or a dispatch statistic calibrated against actual multiplication savings. Without the transform restriction, the minimum defect is identically zero and cannot guide structure discovery.

## 9. Reproduction bindings

No new runner or result file was created. All computation was read-only and printed to the shell. A fresh read-only Codex proof review was attempted after the self-audit but failed before repository read at the known in-process app-server `Access is denied` boundary; no independent PASS is claimed. Central inputs:

| input | SHA-256 |
|---|---|
| `research/MATRIX_NATIVE_INFORMATION_THEORY.md` | `bfeccd754c1eb3563bc9c39acd914d9975c27d6f96bdfa8e73ae9dcb8c267147` |
| `research/smirnov46_laurent_table.py` | `1ebd908cc6db1e6cbb7629356783c6ef4a19852ee70751bf7c51218e39e9ea13` |
| `research/verify_smirnov46.py` | `ef445d5f91960617609cd9c851242bbec84bcee5f21939a4a8173e9832ce803c` |
| `research/b410_rank48_data.py` | `12ae8b15e475050311add0f033a6d8bcd80e123d69718b8d8ceadc15818bb47b` |
| `research/external/alphatensor/at_f2_4x4_rank47_verified.npz` | `1c5907b6575ab75b49290502ca087309d4468f901aee313a395b27cee6171095` |
| `research/verify_b398_defect.py` | `2afd24125f1cef44c151700b27156faad982ebe9d01175b2a489904fc0b6b2de` |
| `research/verify_b400_laderman.py` | `26ce451d9ff5215984a30d4738d19adc175abc19d77856153da638133cfe8206` |
| `research/b402_decomposition_data.py` | `83387667f8210906d8c44f60be3f334d85a35b414c3d091d3da1fd1699e56a01` |
| `research/b415_mp153_data.py` | `d2f10cb3da42d5b4b0926126c0d50f3eaec4aeb2370fdea0847f1883f30dc739` |
| `research/b416_mp93_data.py` | `7ef2f5ec977044c2376962272a3452c5ff300aa436c91c52901333f4cce3f41c` |
| `research/b417_p486_data.py` | `3ff50613d05438d0cd321e02241cb40b664565286746a66330cdcdcb012179d1` |
| `research/b437_kmw153_data.py` | `582f2b9de68f33f25b54ccd17b7fb928a2ede845832abde700a2d0e543975cf2` |

The exact pole computation imported the authenticated Laurent tuples and expanded complete summands over integer Laurent arithmetic. The entropy pass reshaped every factor using its declared matrix dimensions, normalized each matrix before spectral evaluation, and never invoked the MP-93 sweep/census runner or read its protected result surface.

## 10. One-sentence mission narrative

**Honest headline:** Smirnov saves two 4x4 products by turning each requested answer bit into roughly four working bits, so the saving does not pay back before matrices have around `10^104` rows even under a favorable fast-bit-cost model.

This survives “isn't that trivial?” because the exact `q=1`, `h=3`, constants, recursion split, and crossover were derived from the authenticated real scheme, not inferred from rank. It does not itself make matrix multiplication cheaper; it redirects the practical hunt toward exact schemes or border schemes with a provably tiny `h/q`.



