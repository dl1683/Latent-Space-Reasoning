# Independent Proof Review 2 — ACDFT Area–Time Repair

**Review status:** COMPLETE (2026-07-14).

**Reviewer posture:** fresh, zero-context, adversarial mathematical review performed
only from the repository artifacts named in the work order. I had no prior involvement
with the theorem or its repair and used no private project memory as evidence.

**Review target:** only the repaired conditional area–time inference `(RAT)` and its
declared positive regime in `research/PROOF_REVIEW_PACKET_2.md`. Candidate Lemma 1,
Candidate Lemma 2, and `ACDFT-RCUT` are inherited as certified inputs, but their uses
here are being checked against their exact certified scopes. The geometry premise `(G)`
is an assumption, not a reviewed input.

**Citation convention:** `PACKET:Lx-Ly`, `ACDFT:Lx-Ly`, `REVIEW1:Lx-Ly`,
`MISSION:Lx-Ly`, and `OFFT:Lx-Ly` refer to current line numbers in
`research/PROOF_REVIEW_PACKET_2.md`, `research/ACDFT_LEMMA.md`,
`research/PROOF_REVIEW_1.md`, `research/MISSION_NARROWED.md`, and
`research/sources/OFFT_SOURCES.md`.

## Frozen claim and audit boundary

The submitted implication is

\[
T\kappa\sqrt A\ge T C_{\rm cut}(\gamma)\ge Q
\quad\Longrightarrow\quad
AT^2\ge \frac{([Q]^+)^2}{\kappa^2},
\]

under `A >= 0`, `T >= 0`, `0 < kappa < infinity`, the assumed geometry law
`C_cut(gamma) <= kappa sqrt(A)`, and the reviewed cut inequality
`T C_cut(gamma) >= Q` (`PACKET:L53-L100`). The packet expressly excludes proof of
the geometry law, an unconditional planar theorem, a physical or optical advantage,
and wider parameter regimes (`PACKET:L182-L193`).

## Findings

1. **SOUND — the nonnegative-part inference has the correct algebraic
   form (`PACKET:L102-L120`).** Multiplication of `(G)` by `T` preserves order because
   `T >= 0`. Since `kappa > 0` and `A >= 0`, the left endpoint
   `T kappa sqrt(A)` is nonnegative. Therefore it is at least `max(Q,0)`, after which
   squaring is monotone on the nonnegative reals and division by `kappa^2` is legal.
   This directly removes the defect found in `REVIEW1:L37-L45`.

2. **SOUND — a negative `Q` produces no residual positive claim
   (`PACKET:L63-L97`).** The outer positive part is applied to all of `Q`, after the
   `-h_2(delta)` term, rather than only to the packing bracket. Thus `Q < 0` gives
   exactly `AT^2 >= 0`.

3. **SOUND — `(R3)` is necessary and sufficient for strict positivity, within the
   inherited parameter domain (`PACKET:L122-L140`; `ACDFT:L20-L65`,
   `ACDFT:L288-L321`).** Put
   `L = log_2(c_V/rho)` and `X = (N/3)L - Gamma(rho)`. Since
   `0 <= delta < 1/2`, `1-delta > 0` and `h_2(delta) >= 0`. If `rho >= c_V`, then
   `L <= 0`, so `X < 0` because `Gamma(rho) > 0`, and hence `Q <= 0`. If
   `rho < c_V`, then `L > 0`, and
   `Q > 0` iff `X > h_2(delta)/(1-delta)`, which rearranges exactly to the strict
   inequality `(R3)`. For `delta=0`, `h_2(0)=0`, and the same strict inequality is
   equivalent to `X>0`; equality gives `Q=0`, not positivity.

4. **SOUND — the displayed sufficient regime really is sufficient, but deliberately
   narrower than the exact regime (`PACKET:L142-L154`).** Numerically,
   `c_V = 0.0082320112189...`, so `ceil(log_2(4/c_V))=9`. If `b>=9`, then
   `2^{-b} <= 2^{-9} < c_V/4`; combining this with `0<epsilon<c_V/4` gives
   `rho<c_V/2<c_V`. The remaining strict threshold is exactly `N>N_+`.

5. **SOUND — the positive regime is narrow but mathematically and asymptotically
   nonempty (`PACKET:L122-L154`).** Positivity is possible only if
   `2^{-b}<c_V`, hence only for `b>=7`. At `b=7` it requires
   `epsilon < c_V-2^{-7} = 0.000419511...`; at `b=8` the ceiling is
   `0.004325761...`; at `b=9` it is `0.006278886...`. A concrete fixed-parameter
   point is `b=9`, `epsilon=0.001`, `delta=0.10`: then
   `rho=0.002953125`, `N_+=22.833...`, and the first allowed power of two is
   `N=32`. Thus the regime contains a genuine constant-global-error, finite-precision
   infinite family; it is not an exact-word limit or asymptotically empty.

6. **SOUND — the asymptotic qualification is accurate (`PACKET:L150-L154`).** For
   fixed `rho<c_V` and fixed `delta<1/2`, once `N>N_+`, `Q` is an affine function of
   `N` with positive leading coefficient. Hence `(RAT)` is
   `Omega(N^2/kappa^2)` in the exact parameterized sense. It is `Omega(N^2)` only
   under a uniform `N`-independent upper bound on `kappa`. If `rho=rho_N` approaches
   `c_V` or `kappa=kappa_N` grows, no stronger scaling follows; the packet does not
   claim otherwise.

7. **SOUND — no certified input is used outside its reviewed scope
   (`PACKET:L30-L51`, `PACKET:L75-L87`; `ACDFT:L84-L93`, `ACDFT:L142-L169`,
   `ACDFT:L292-L321`).** The assumed geometry law supplies one balanced
   named-output cut `gamma`. `ACDFT-RCUT` is certified for *every* such cut, so it
   applies to that same `gamma`. Candidate Lemma 1 is certified for every input
   placement at every balanced named-output cut; Candidate Lemma 2 is certified for
   the better direction at that cut with analyst-only cut-dependent subcode selection.
   The repaired assembly does not select a new codebook, provide advice, change the
   phase quotient, or invoke either component independently of the already-certified
   `(RCUT)`. The positive regime remains inside `N=2^m` with `m>=1`, `b>=1`,
   `epsilon>0`, and `0<=delta<1/2`; nothing here extends to `epsilon=0`,
   `delta=1/2`, or arbitrary non-power-of-two `N`.

8. **SOUND — the consequence is Thompson-honest precisely because it remains
   conditional (`MISSION:L280-L304`; `OFFT:L39-L65`, `OFFT:L89-L120`).** Thompson's
   `AT^2` theorem assumes planar unit-spacing wiring, resident on-chip named inputs
   and outputs, exact finite-ring words, and fixed word/bit bandwidth. The ACDFT
   contract preserves resident named I/O and charges communication, but deliberately
   replaces exact words by a finite complex grid, global-`l_2` coherent approximation,
   randomized failure, and an operational bit-capacity certificate. It imports none
   of Thompson's constant `N^2/16`, word bandwidth, or planar geometry for free.
   Instead `(G)` is the explicit missing bridge. The packet repeatedly withholds proof
   of `(G)`, an unconditional planar theorem, and every physical correction hidden in
   `kappa` (`PACKET:L75-L100`, `PACKET:L172-L193`). This is an honest extension of
   the cut method, not a claim that Thompson already proved the approximate theorem.

9. **SOUND — approximate output is not silently changed back to exact output
   (`ACDFT:L292-L315`, `ACDFT:L341-L379`; `REVIEW1:L29-L35`).** The inherited
   `(RCUT)` applies to the worst-case success event
   `d_coh(y_hat,F_N x)<=epsilon`. Exact recovery occurs only for the finite hard
   message index after packing points more than `2 epsilon` apart, on that success
   event; it is a rate-distortion reduction, not an exact-output obligation imposed
   on the architecture. The area–time assembly uses only the resulting information
   inequality and introduces no new decoder, ring word, phase reference, or exactness
   premise.

10. **SOUND — the `Omega(N^2)` task lower bound is not a terminal-counting tautology
    (`MISSION:L242-L304`; `ACDFT:L84-L169`).** Merely having `N` resident named
    inputs and outputs, or even an invertible transform on an exponentially large
    finite set, does not force linear information across a balanced output cut: the
    identity or a locally placed permutation has the same terminal count and global
    input entropy, yet can pair each input with its output and avoid such traffic.
    Constant-size on-chip terminals give at most the simpler area count `Omega(N)`,
    not `AT^2=Omega(N^2)`. The stronger conclusion needs the DFT-specific stable
    cross-cut dimension plus finite approximate packing. Thus the submitted assembly
    is algebraically short once `(RCUT)` and `(G)` are assumed, but its lower-bound
    content is not implied by a simpler task-independent count.

11. **SOUND — vacuity and the missing physical bridge are not marketed as theorem
    substance (`PACKET:L122-L154`, `PACKET:L172-L193`; `ACDFT:L497-L519`).** Every
    `Q<=0` case is explicitly called vacuous, the exact positive regime is displayed,
    the sufficient constants are called narrow, and the `Omega(N^2)` form is tied to
    control of `kappa`. The artifact no longer says that all physical correction terms
    are exposed: `kappa` remains a placeholder for a separately sourced
    layer/pitch/channel/terminal-aware geometry lemma.

## Boundary checks

- `Q < 0`: `[Q]^+=0`, so `(RAT)` is exactly `AT^2>=0`; the old counterexample now
  returns only this vacuous truth (`REVIEW1:L37-L45`; `PACKET:L178-L180`).
- `Q = 0`: the same vacuous bound results. This includes equality in `(R3)` when
  that real threshold happens to coincide with an allowed `N=2^m`.
- `delta = 0`: `h_2(0)=0`, so positivity is exactly `rho<c_V` and
  `N>3 Gamma(rho)/log_2(c_V/rho)`. There is no lost endpoint.
- `A = 0` or `T = 0`: the left endpoint of the chain is zero. If `Q<=0`, `(RAT)`
  says only `AT^2>=0`; if `Q>0`, `(G)` and `(R2)` cannot simultaneously hold, so
  there is no hidden positive conclusion about such a degenerate architecture.
- `rho = c_V`: `log_2(c_V/rho)=0`, the inner expression is `-Gamma(rho)<0`, and
  `Q=-h_2(delta)<=0`; `(RAT)` is vacuous. Equivalently, at the error boundary
  `epsilon=c_V-2^{-b}` (where positive), strict positivity is correctly excluded.
- singleton packing: when the inner positive part is zero, `Q=-h_2(delta)<=0` and
  the outer positive part returns zero. For `delta=0`, it returns `Q=0`, still zero.
- small `b`: `b<=6` forces `rho>c_V`, so no positive claim is made. `b=7` is
  nonempty only in the very small error interval stated in Finding 5.
- small `N`: the certified family has `N=2^m`, `m>=1`. `N=2` can never be positive
  because `Gamma(rho)>log_2(c_V/rho)` for `rho<c_V`, so `N_+>3`; `N=4` is possible
  only at extremely small `rho` (about `3.62e-11` when `delta=0`). These cases are
  faithfully classified by `(R3)`, not advertised as useful.
- degenerate cuts: the geometry cut must still balance the named output terminals.
  Arbitrary, even one-sided, input-terminal placement is within Candidate Lemma 1's
  certified scope; a zero-capacity balanced cut is incompatible with `Q>0` by
  `(R2)`, and otherwise yields only the vacuous bound.

## Three strongest attacks mounted

1. **Old sign counterexample, strengthened at the zero-resource boundary — failed
   after repair.** Take a singleton-packing regime and `0<delta<1/2`, so
   `Q=-h_2(delta)<0`; the prior review exhibited an allowed zero-output,
   `T=0` architecture at large error (`REVIEW1:L43-L45`). The old square yielded a
   false positive lower bound. Here `[Q]^+=0`, and `(RAT)` gives only `0>=0`. Trying
   `Q=0`, `A=0`, or `T=0` produces the same result; if `Q>0`, the premises themselves
   exclude `A=0` and `T=0`.

2. **Scope-laundering and Thompson-model mismatch — failed.** I chose the geometry
   cut only after the architecture, allowed all inputs to lie on one side, retained
   feedback/noise/randomness, and asked whether the proof had silently returned to
   exact finite-ring output. `ACDFT-RCUT` holds for every balanced named-output cut,
   Candidate Lemma 1 covers every input placement, the hard subcode remains
   analyst-only, and exact message decoding is only the certified packing reduction.
   Planarity and wire/word capacity are not inferred: they remain entirely inside the
   unproved premise `(G)` and the operational capacity certificate.

3. **Simpler-counting domination — failed.** I removed all DFT-specific structure and
   retained only `N` named resident inputs, `N` named resident outputs, finite
   encodings, and global invertibility. Identity/permutation devices satisfy those
   counts without forcing `Omega(N)` information across every balanced output cut and
   admit `O(N)`-area, constant-time layouts under ordinary internal-terminal
   semantics. Hence terminal count or total entropy alone cannot yield the submitted
   `Omega(N^2)` shape. The stable-cut and projective-packing inputs are essential.

## Meaningfulness of the positive regime

The regime is **real but high-accuracy and narrow**. It is not asymptotically empty:
for every fixed `rho<c_V` and fixed `delta<1/2`, all sufficiently large allowed powers
of two satisfy `(R3)`. It is also not exact-word-equivalent, because `b`, `epsilon`,
and `delta` may all remain fixed as `N` grows. The concrete point
`(b,epsilon,delta)=(9,0.001,0.10)` becomes positive already at `N=32`.

The limitation is severe in error, not in asymptotic existence. Even as `b` tends to
infinity, the present constants require global `l_2` error below
`c_V≈0.008232` (less than about `0.8232%` of the unit norm bound); the packet's clean
sufficient regime asks for less than `0.2058%`. Low precision rapidly collapses the
window, and `b<=6` gives none. Moreover, physical significance remains conditional on
a sourced, uniformly controlled `kappa`. The honest assessment is therefore:
mathematically substantive as a fixed-constant approximate-output family and clearly
better than an exact-only theorem, but not a broad practical-error theorem and not yet
a physical area–time result.

## Verdict

**UPHELD.** The nonnegative-part repair is correct and complete. No sign error,
boundary counterexample, positive-regime error, certified-scope escape,
Thompson-model substitution, approximate-to-exact task change, or simpler-counting
domination survived review. All findings are `SOUND`; there is no repair list.

## Exact promotion boundary

The following statement, and only this statement, is certified for a `PROVED-AT-SCOPE`
label:

Let `N=2^m` with `m>=1`, `b>=1`, `epsilon>0`,
`rho=epsilon+2^{-b}`, and `0<=delta<1/2`. Let an admitted architecture satisfy the
frozen worst-case coherent-output promise on `X_{N,b}`, terminate by finite time
`T>=0`, and possess the reviewed finite operational interactive cut-capacity
certificate for every balanced named-output cut. Assume `A>=0` and, separately, a
geometry lemma that supplies a balanced named-output cut `gamma` and a finite
`kappa>0` satisfying

\[
C_{\rm cut}(\gamma)\le\kappa\sqrt A.
\]

Define

\[
Q=(1-\delta)
\left[\frac N3\log_2\frac{c_V}{\rho}-\Gamma(\rho)\right]^+
-h_2(\delta).
\]

Then

\[
AT^2\ge\frac{([Q]^+)^2}{\kappa^2}.
\]

The right side is strictly positive exactly when

\[
\rho<c_V
\quad\text{and}\quad
N>
\frac{3\left(\Gamma(\rho)+h_2(\delta)/(1-\delta)\right)}
{\log_2(c_V/\rho)}.
\]

This certification permits promotion of the repaired conditional consequence
`(RAT)` / `(ACDFT-AT|G-REPAIRED)` only. It does **not** certify `(G)`, an
unconditional planar `ACDFT-AT` theorem, the repository token
`ACDFT-LOWER-BOUND-REVIEWED`, any Thompson constant or bit/word specialization,
source-final novelty, optical advantage, a physical CAN theorem, or mission
completion.
