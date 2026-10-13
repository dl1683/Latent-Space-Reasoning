# ACDFT Regime-Widening Unit - Result and Geometry Fallback

**Date:** 2026-07-14

**Registration:** `research/ACDFT_REGIME_WIDENING.md`

**Inherited theorem:** `research/ACDFT_LEMMA.md`, PROVED-AT-SCOPE only at the
scope certified by `research/PROOF_REVIEW_1.md` and
`research/PROOF_REVIEW_2.md`.

**This artifact:** **PROVED-AT-SCOPE AFTER INDEPENDENT PROOF REVIEW 3** for exactly
`(R-PACK)`, `(R-W-PACK)`, `(R-RCUT)`, `(R-AT|G)`, the delimited proof-template
obstruction in section 3.1, and `(G-T)`, under the quantifiers and exclusions
stated below and certified in `research/PROOF_REVIEW_3.md`. Source occupation,
real-device admission, novelty, and mission interpretations retain their explicit
EMPIRICAL or HEURISTIC labels.

> **UNIT RETURN - REVIEWED TARGET MISS, PROVED RESIDUAL, AND PROVED GEOMETRY
> SUBCLASS.**
> `ACDFT-WIDE-TARGET-MISS` fires: the proof below does not establish the frozen
> \(\rho\le 1/16\) target. It does give a proved residual through
> \[
> \rho\le \rho_*:=
> \frac{5}{42\,2^{1/32}\sqrt{3\pi e}}
> =0.0230160462\ldots,
> \]
> about \(2.80\) times the old \(c_V\), and it proves a residual plus (G) for a strictly
> narrower Thompson graph-embeddable, edge-capacitated subclass. Proof Review 3
> upheld both at their exact scopes. The registered wide target and broad planar
> geometry law remain unproved.

## 0. Token return

| Frozen token | Disposition |
|---|---|
| `ACDFT-WIDE-GATE0-CLEAR` | **ISSUE - HEURISTIC CHECKED-RECORD VERDICT.** See `research/ACDFT_REGIME_WIDENING_GATE0.md`; source-incomplete items remain explicit. |
| `ACDFT-WIDE-PACK-PROVED-AT-SCOPE` | **WITHHOLD.** The registered \(\rho\le1/16\) statement was not proved. |
| `ACDFT-WIDE-RCUT-PROVED-AT-SCOPE` | **WITHHOLD.** Its packing premise was not proved at the registered radius. |
| `ACDFT-WIDE-TARGET-MISS` | **FIRES.** The best completed argument in this unit reaches only \(\rho\le\rho_*<1/16\). |
| `ACDFT-WIDE-CONSTANT-BARRIER` | **DOES NOT FIRE.** No certified block/grid family with subexponential packing at \(\rho=1/16\) was found or proved. |
| `ACDFT-WIDE-GRID-FAIL` | **DOES NOT FIRE.** The residual keeps the unchanged product grid and analyst-only cut-dependent subcode. |
| `ACDFT-WIDE-PHASE-FAIL` | **DOES NOT FIRE.** The residual packs directly under \(d_{\rm coh}\) and restores no absolute phase. |
| `WORD-PRECISION-ONLY` | **DOES NOT FIRE FOR THE RESIDUAL.** It contains fixed-\(b\), fixed-error positive families, but not the registered six-bit/\(3/64\) boundary. |
| `INFINITY-LEAK` | **DOES NOT FIRE IN THE STATED RESIDUAL OR THOMPSON SUBCLASS.** The latter explicitly charges layers, pitch, and finite per-edge capacity. |
| `VOID` | **DOES NOT FIRE.** No frozen task, grid, metric, failure rule, terminal rule, or capacity definition changed. |
| `ACDFT-RESIDUAL-PACK-PROVED-AT-SCOPE` | **ISSUE - PROOF REVIEW 3.** Only `(R-PACK)` at its full displayed scope and `(R-W-PACK)` when \(\rho\le\rho_*\). |
| `ACDFT-RESIDUAL-RCUT-PROVED-AT-SCOPE` | **ISSUE - PROOF REVIEW 3.** Only the inherited admitted-machine/Fano/interactive-capacity scope, \(0\le\delta<1/2\), and \(\rho\le\rho_*\); `(R-AT|G)` still requires a separately valid same-cut `(G)`. |
| `ACDFT-THOMPSON-G-PROVED-AT-SCOPE` | **ISSUE - PROOF REVIEW 3.** Only the exact finite routed graph subclass in section 4.2, with the output-set adaptation of Thompson's proof and all \(N\)-dependence retained. |

The target-miss rule forbids renaming \(\rho_*\) as success. The reviewed old
theorem remains unchanged.

## 1. Frozen objects

Retain the notation of `research/ACDFT_LEMMA.md`. Let \(A\) be the better
directional DFT cross block. Let \(E\) be its \(k\)-complex-dimensional stable
right-singular subspace, real dimension \(d=2k\ge N/3\), so that

\[
\|Aw\|_2\ge \frac12\|w\|_2\qquad(w\in E).
\tag{1}
\]

Let \(r=2^{-b}\), \(\rho=\epsilon+r\), \(a=1/\sqrt{2N}\), and

\[
K=E\cap[-a,a]^{2|I|}.
\]

The ambient finite set remains \(\mathcal X_{N,b}\). The proof analyst may select
a hard subcode after seeing the architecture and cut; the architecture receives
neither that subcode nor \(E\) as advice. These are inherited scope rules, not
new relaxations.

## 2. Residual packing lemma

### 2.1 Statement

Define

\[
\alpha=\frac{42}{5},\qquad
c_\alpha=\frac{1}{\alpha\sqrt{3\pi e}}
=\frac{5}{42\sqrt{3\pi e}}.
\]

**LEMMA ACDFT-RESIDUAL-PACK - PROVED-AT-SCOPE, PROOF REVIEW 3.** For every frozen
\(N=2^m\), \(b\ge1\), \(\epsilon>0\), and every certified better block \(A\),
the unchanged grid contains an analyst-only, cut-dependent subcode
\(\mathcal C_\gamma\) with

\[
d_{\rm coh}(Ax,Ax')>2\epsilon
\quad(x\ne x'\in\mathcal C_\gamma)
\tag{2}
\]

and

\[
\log_2|\mathcal C_\gamma|
\ge
\left[
d\log_2\!\left(\frac{c_\alpha}{\rho}\right)
-\log_2\!\left(3+\frac{40\pi}{\rho}\right)
\right]^+.
\tag{R-PACK}
\]

Consequently, when

\[
\rho\le\rho_*:=
\frac{c_\alpha}{2^{1/32}}
=\frac{5}{42\,2^{1/32}\sqrt{3\pi e}},
\tag{3}
\]

then \(\rho_*<1/16\) analytically: \(\pi>3\), \(e>2\), and
\(42\sqrt{18}>80\) already give the strict comparison, without using the
floating-point audit.

the same subcode obeys the registered cardinality form

\[
\log_2|\mathcal C_\gamma|
\ge
\left[
\frac d{32}-\log_2\!\left(\frac{128}{\rho}\right)
\right]^+.
\tag{R-W-PACK}
\]

This is a residual with a smaller radius, not `(W-PACK)`.

### 2.2 Derivation

**Step 1 - sharper continuous separation.** Vaaler's central-section inequality,
already used and independently reviewed in the inherited proof, gives
\(\operatorname{vol}_d(K)\ge(2a)^d\). Take a maximal
\(\tau\)-separated set \(\mathcal P\subseteq K\) for

\[
\tau=\alpha\rho=\frac{42}{5}\rho.
\]

The same covering-volume calculation as in the reviewed Lemma 2 gives

\[
|\mathcal P|
\ge
\left(\frac{1}{\alpha\sqrt{3\pi e}\,\rho}\right)^d
=
\left(\frac{c_\alpha}{\rho}\right)^d.
\tag{4}
\]

This step changes only the chosen separation scale from \(24\rho\) to
\((42/5)\rho\).

**Step 2 - unchanged coordinate rounding.** Round every real coordinate of
\(w\in\mathcal P\) to the same grid as in the reviewed proof and call the result
\(\widetilde w\). The inherited rounding estimate is

\[
\|\widetilde w-w\|_2\le r.
\tag{5}
\]

For two distinct rounded points, (1), the defining
\(\alpha\rho\)-separation of \(\mathcal P\), (5), and
\(\|A\|_{\rm op}\le1\) give Euclidean output separation at least

\[
\begin{aligned}
s
&:=\frac{\alpha\rho}{2}-2r\\
&=\frac{21}{5}\epsilon+\frac{11}{5}r.
\end{aligned}
\tag{6}
\]

In particular,

\[
s-4\epsilon
=\frac{\epsilon+11r}{5}
\ge\frac{\rho}{5}>0.
\tag{7}
\]

Thus rounding does not identify two members.

**Step 3 - prune conflicts at the required projective radius.** On the rounded
outputs \(z=A\widetilde w\), join \(z,z'\) by an edge exactly when
\(d_{\rm coh}(z,z')\le2\epsilon\). For fixed \(z\), its phase orbit has
Euclidean length at most \(2\pi\), because \(\|z\|_2\le1\). Set

\[
\eta=\frac{s-4\epsilon}{4}.
\]

The orbit can be covered by at most \(1+2\pi/\eta\) Euclidean
\(\eta\)-balls. Every conflict neighbor lies within \(2\epsilon+\eta\) of
one cover center. Two distinct conflict neighbors cannot use the same center:
their mutual Euclidean distance would be at most

\[
4\epsilon+2\eta
=\frac{s+4\epsilon}{2}<s,
\]

contradicting (6). By (7), the maximum conflict degree is therefore at most
\(1+40\pi/\rho\). Keeping the same conservative integer slack used by the
reviewed proof, greedy pruning retains a fraction at least

\[
\left(3+\frac{40\pi}{\rho}\right)^{-1}.
\tag{8}
\]

An independent set has strict projective separation \(>2\epsilon\), because
the graph included all pairs at distance \(\le2\epsilon\). Combining (4) and
(8), with the trivial singleton when the logarithmic expression is negative,
gives `(R-PACK)`.

**Step 4 - recover the registered rate/correction on the smaller interval.**
If \(\rho\le\rho_*\), then

\[
\log_2(c_\alpha/\rho)\ge\frac1{32}.
\]

Also \(\rho\le\rho_*<1/16\), and

\[
3+\frac{40\pi}{\rho}
\le\frac{128}{\rho},
\tag{9}
\]

since \(3\rho+40\pi<128\) on this interval (the elementary bound
\(\pi<22/7\) already suffices). Equations (R-PACK) and (9) give
`(R-W-PACK)`.

### 2.3 Reviewed downstream assembly

**PROVED-AT-SCOPE - PROOF REVIEW 3.** Substituting `(R-W-PACK)` into the already
reviewed Fano/capacity proof gives, only for \(\rho\le\rho_*\),

\[
TC_{\rm cut}\ge
(1-\delta)
\left[
\frac N{96}-\log_2\!\left(\frac{128}{\rho}\right)
\right]^+
-h_2(\delta)
=:Q_{\rm res}.
\tag{R-RCUT}
\]

If a separately valid (G) holds, the already reviewed nonnegative-part inference
would then give

\[
AT^2\ge\frac{([Q_{\rm res}]^+)^2}{\kappa^2}.
\tag{R-AT|G}
\]

The logical assembly is inherited. Proof Review 3 certified `(R-RCUT)` only at the
same admitted-machine, worst-case coherent-output, per-input failure, every-balanced-
named-output-cut, and finite operational interactive-capacity scope as the inherited
`(RCUT)`, with \(0\le\delta<1/2\) and \(\rho\le\rho_*\). It certified `(R-AT|G)`
only for the same cut when \(A,T\ge0\) and a separately valid `(G)` supplies a
finite \(\kappa>0\). The outer nonnegative part is mandatory.

## 3. Why the registered target was missed

### 3.1 Obstruction to this proof template

**PROVED DELIMITED ROUTE OBSTRUCTION - PROOF REVIEW 3.** Consider the specific
template that:

1. obtains only Vaaler's volume lower bound for a \(\tau\)-packing of \(K\);
2. rounds each point with per-vector error at most \(r\); and
3. controls the number of projective conflicts only by comparing two
   \(2\epsilon\)-neighbors with the post-rounding Euclidean minimum separation.

Writing \(\tau=\alpha\rho\), this template has only

\[
s\ge\frac{\alpha\rho}{2}-2r.
\]

Its triangle-inequality collision certificate needs \(s>4\epsilon\). Uniformly
over all decompositions \(\rho=\epsilon+r\), the limit \(r/\rho\to0\) therefore
forces \(\alpha\ge8\) asymptotically. At the registered boundary
\(\rho=1/16\), Vaaler's resulting exponential base is at most

\[
\frac{1}{8\rho\sqrt{3\pi e}}
=\frac{2}{\sqrt{3\pi e}}
=0.3951365\ldots<1.
\tag{10}
\]

So this exact volume-rounding-minimum-separation certificate cannot prove a
positive exponential rate at \(1/16\). This is not an upper bound on the actual
packing number. A discrete small-ball or structure-sensitive proof could bypass
(10).

### 3.2 Route B and counterexample search

**HEURISTIC / OPEN.** The closest checked theorem for a different route is
Rudelson-Vershynin's dimension-exponential small-ball bound for arbitrary
orthogonal projections of independent-coordinate vectors. Its universal constant
is not explicit in the statement used here, and it does not perform the global
phase pruning with the registered correction. It therefore did not certify the
frozen constants.

**HEURISTIC NEGATIVE SEARCH RESULT.** No certified stable DFT block/grid family
with only \(2^{o(d)}\) projective packing at \(\rho=1/16\) was found or constructed.
Rankin/Levenshtein-type and Grassmannian upper bounds concern freely selected
sphere or manifold codes, not the image of the frozen product grid under every
certified block. Therefore there is no matching obstruction and
`ACDFT-WIDE-CONSTANT-BARRIER` remains unavailable.

## 4. Geometry premise (G)

### 4.1 What (G) actually assumes

The inherited conditional theorem assumes the following existential statement
for the same balanced named-output vertex cut used in the information proof:

\[
\text{there exists }\gamma=(L,R),\quad
|O_L|=|O_R|=N/2,\quad
C_{\rm cut}(\gamma)\le\kappa\sqrt A.
\tag{G}
\]

Here \(C_{\rm cut}\) is the operational two-way interactive information-capacity
certificate, not a wire count. For an asymptotic constant-\(\kappa\) specialization,
the architecture family must control every physical quantity absorbed into
\(\kappa\) independently of \(N\).

**REVIEWED SCOPE FINDING.** The frozen broad planar class does not imply (G) from
its current admission clauses alone. It declares a finite capacity for each cut,
but it does not require that communication factor through a bounded-overlap graph
of finite-capacity edges or that a cut's joint capacity be additive over those
edges. Finiteness cut by cut supplies no \(\sqrt A\) scaling law.

### 4.2 Thompson-subclass theorem

Let \(q\) be a finite positive integer giving the declared number of routing
layers, \(p>0\) the physical pitch, and \(0\le\bar c<\infty\) a total two-way
reliable information-rate bound per routed graph edge. Define the following
stricter admission packet:

1. **Graph realization.** All input-dependent communication is represented by a
   finite Thompson communication graph; the \(N\) named outputs occupy \(N\)
   distinct distinguished vertices.
2. **Embedding.** The graph has a legal \(q\)-layer Thompson embedding in physical
   area \(A\) at pitch \(p\). Under Thompson's multilayer correspondence it maps
   into at most \(q^2A/p^2\) occupied unit squares of a one-layer graph.
3. **Cut fidelity.** Removing the graph edges between a vertex partition removes
   every input-dependent physical communication path between its sides; no
   unrepresented broadcast, shared field, wireless link, or hyperedge bypasses
   the graph.
4. **Capacity cutset.** For every vertex cut \(\gamma\),
   \[
   C_{\rm cut}(\gamma)
   \le\sum_{e\in\partial\gamma}c_e,
   \qquad c_e\le\bar c.
   \tag{11}
   \]
   Correlated or interactive signaling is allowed only when the displayed joint
   cutset certificate remains valid.

**LEMMA ACDFT-THOMPSON-G - PROVED-AT-SCOPE, PROOF REVIEW 3.** Every
architecture satisfying this packet has a balanced named-output vertex cut
\(\gamma\) such that

\[
C_{\rm cut}(\gamma)
\le
\frac{2q\bar c}{p}\sqrt A.
\tag{G-T}
\]

For \(0<\bar c<\infty\), (G) holds on this subclass with the finite positive rate
constant

\[
\kappa=\frac{2q\bar c}{p}.
\tag{12}
\]

**Derivation - reviewed at the stated scope.** Let \(w\) be the minimum number of
graph edges whose removal bisects the distinguished output vertices. Thompson
states Theorem 2 for source nodes, but first defines bisection for an arbitrary
distinguished vertex set; the proof uses only membership and balance. Re-running
the proof of Theorem 2 with the even set of \(N\) distinct named-output vertices
gives \(w^2/4\) as a lower bound on the occupied one-layer graph area. Thompson's
multilayer mapping represents the physical layout in at most \(q^2A/p^2\) such
squares. Therefore

\[
\frac{w^2}{4}\le\frac{q^2A}{p^2},
\qquad
w\le\frac{2q}{p}\sqrt A.
\tag{13}
\]

Choose a minimum output-bisecting edge cut and its induced vertex partition.
The cut-fidelity and capacity-cutset assumptions turn (13) into

\[
C_{\rm cut}(\gamma)
\le\bar c\,w
\le\frac{2q\bar c}{p}\sqrt A.
\]

This derives `(G-T)` without using exact DFT algebra. Thompson supplies the
geometry through the output-set adaptation of his proof; the separately declared
operational edge-capacity certificate supplies the information units. If
\(\bar c=0\), `(G-T)` remains true and forces \(C_{\rm cut}=0\), but the displayed
\(\kappa\) is zero and may not be used as a divisor in `(R-AT|G)`. Combining that
branch directly with `(R-RCUT)` instead forces \(Q_{\rm res}\le0\). The positive
area-time formula below therefore uses exactly \(0<\bar c<\infty\).

### 4.3 Where the geometry derivation applies

**PROVED FORMAL CLASS; HEURISTIC REAL-DEVICE ADMISSION VERDICT.** `(G-T)` is a
theorem for the exact formal packet above. Conventional routed electronic VLSI
may inherit it only when its signal paths are faithfully reduced to a
bounded-layer, nonvanishing-pitch graph with finite per-edge reliable rate. It
also admits analog wires only after their finite operational rates and the joint
cutset inequality are proved.

It does not automatically cover free-space optics, shared wave fields, wireless
or capacitive global coupling, broadcast buses represented as free hyperedges,
unbounded wavelength/time multiplexing, growing layer count, shrinking pitch, or
any device whose joint cut capacity is not bounded by a sum of represented edge
capacities. Such a device may satisfy a different sourced geometry theorem; it
does not inherit this one by being called planar.

Hence the geometry fallback has two outcomes:

- **PROVED-AT-SCOPE discharge on the exact Thompson graph subclass**; and
- **(G) remains an explicit admission premise for the broader frozen class.**

## 5. Closure/admission and infinity-leak audit

**PROGRAM-LAW APPLICATION - HEURISTIC.**

- The task, full coherent output, global phase quotient, grid, error, failure,
  input residency, and terminal materialization are unchanged on both sides.
- No singular subspace, hard subcode, phase reference, or arbitrary real reaches
  the architecture for free.
- The Thompson specialization charges \(q\), \(p\), \(\bar c\), area, time,
  terminal realization, and the graph/capacity certificate. If any varies with
  \(N\), that dependence remains in \(\kappa_N=2q_N\bar c_N/p_N\).
- A comparison with an optical, analog, or other non-graph device requires a
  same-task admission proof or a charged uniform task-preserving compiler into
  the Thompson subclass. No such compiler is asserted.
- Excluding a device that lacks `(G-T)` is not a lower bound on that device.

No infinity leak was found inside the stated residual or subclass. The broader
class cannot be promoted by silently treating finite capacity as a geometric
capacity-density theorem.

## 6. Boundary audit

**EMPIRICAL ARITHMETIC, NOT PROOF.** CPU-only audit `ACDFT-WIDE-NUM-1` recorded:

\[
\begin{array}{rcl}
c_V&=&0.0082320112189\ldots,\\
c_\alpha&=&0.0235200320541\ldots,\\
\rho_*&=&0.0230160462676\ldots,\\
\rho_* / c_V&=&2.79592017739\ldots.
\end{array}
\]

At the frozen target boundary
\((b,\epsilon,\delta)=(6,3/64,1/3)\), \(\rho=1/16>\rho_*\), so the
residual says nothing new. The registered target's positivity threshold remains
\(N>1188.2346\ldots\), first allowed power \(N=2048\), but no packing proof was
obtained there. Equality in `(W-POS)` would make the unsquared Fano lower bound
zero; strict \(N>\) remains necessary for positive content.

The residual is nevertheless nonempty at fixed finite precision and error. For
example, \(b=7\) and \(\epsilon=1/128\) give
\(\rho=1/64<\rho_*\); this witness is outside the old
\(\rho<c_V\) interval. This is a residual-scope fact, not registered-target
success.

The computation is logged in `EXPERIMENTS.md` and
`experiments/ledger.jsonl`. No GPU was used.

## 7. Falsification ledger

| Attempted claim | Result | Corrected disposition | Surviving kernel |
|---|---|---|---|
| The registered \(\rho\le1/16\), \(d/32\), \(128/\rho\) target follows by optimizing the inherited volume/rounding/phase proof | **FALSIFIED FOR THE COMPLETED TEMPLATE; PROOF REVIEW 3 UPHELD THE DELIMITED OBSTRUCTION.** The collision certificate forces asymptotic \(\alpha\ge8\), where the Vaaler base at \(1/16\) is below one. | `ACDFT-WIDE-TARGET-MISS`; no wide token. | **PROVED-AT-SCOPE residual** through \(\rho_*=0.023016\ldots\). |
| The proof-template failure is a fundamental packing barrier | **NOT ESTABLISHED.** No matching block/grid upper bound or counterexample family was proved. | Withhold `ACDFT-WIDE-CONSTANT-BARRIER`. | Discrete small-ball/structure-sensitive route remains open. |
| Grassmannian/projective code bounds solve the target off the shelf | **FALSIFIED BY TYPECHECK IN THE CHECKED RECORD.** Their code centers are freely designed and may be shared with the device. | Gate-0 clearance only; source-final incompletes remain. | Their phase metric and entropy methods are adjacent ingredients. |
| Finite cut capacity plus planarity alone implies \(C_{\rm cut}=O(\sqrt A)\) | **FALSIFIED AS AN INFERENCE FROM THE FROZEN CLAUSES.** No graph factorization, capacity density, or additive cutset law is present. | Keep (G) assumed for the broad class. | **PROVED-AT-SCOPE `(G-T)`** for the stricter Thompson graph subclass. |
| Thompson wire bisection alone proves operational (G) | **FALSIFIED WITHOUT A CAPACITY BRIDGE.** Wire count has no information unit by itself. | Require finite \(\bar c\) and joint inequality (11). | Geometry plus the explicit cutset certificate yields \(\kappa=2q\bar c/p\). |
| A Thompson-subclass theorem adjudicates broad photonic/analog acceleration | **FALSIFIED BY ADMISSION TYPE.** Shared fields, nonlocal couplings, or nonadditive cut capacity need another theorem or compiler. | No broad device verdict. | Routed finite-capacity electronic subclass only. |

## NARRATIVE

**Honest one-sentence story - PROVED-AT-SCOPE + TARGET MISS.** *Even after finite
precision, approximation, random failure, and one free global phase are allowed, a
full Fourier output must pay a quadratic area-time price at sufficiently large
allowed sizes on every admitted finite-capacity Thompson-routed device in a narrow
high-accuracy regime.*

**"Isn't that trivial?" attack - HEURISTIC.** The numerical constant is not the
point by itself. The useful delta is the exact projective conflict pruning on the
unchanged finite grid and the identification of the proof-template wall without
mislabeling it as a theorem barrier.

**"So what?" attack - HEURISTIC.** Even after review, the residual would still
constrain only a high-accuracy full-coherent-DFT class. `(G-T)` gives that class
unconditional physical teeth for the exact routed Thompson-style subclass, not for
photonic or analog accelerators generally.

**Mission standing - HEURISTIC.** This unit completes the ACDFT proof chain only
for the certified routed subclass and therefore advances Milestone D at that scope;
the broader frozen planar class remains conditional. It supplies no new computation
model, positive CAN theorem, photonic advantage, or Turing-scale mission result.
