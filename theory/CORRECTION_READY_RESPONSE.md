# Memory that preserves the ability to change a world

Analytical construction, 2026-10-14. **Scoped mathematical proof review passed subject to the listed definition/numerical clarifications, applied here; no learned or economic result.**
Independent Codex proof review 2026-10-15: PASS-WITH-LISTED-FIXES; both listed fixes (grid legality, transfer boundary) are applied here. Review: docs/council_2026_10_15_correction_proof_codex.md.
This is an exact quadratic-world design, not a learned latent artifact, a new
computational model, a novelty claim or an observed efficiency result. Three rounds
of design dialogue selected this object; agreement does not establish its claims.

## Question and contract

What must a compact memory preserve so that future interventions have the right
consequences? Preserving today's answer is insufficient. The first construction
below preserves both a response and its intervention susceptibility. It gives an
explicit admission boundary for approximate memory and paid source recovery.

The world is an immutable symmetric positive definite matrix, partitioned into
boundary and interior coordinates:

$$A=\begin{pmatrix}A_{BB}&A_{BI}\\A_{IB}&A_{II}\end{pmatrix}.$$

Its energy is $\tfrac12(u,z)^TA(u,z)$. Minimizing over the interior gives the
boundary response $S=A_{BB}-A_{BI}A_{II}^{-1}A_{IB}$. Fix a versioned port matrix
$C$. Retain

$$F=A_{BI}A_{II}^{-1}C,\qquad G=C^TA_{II}^{-1}C.$$

A legal current correction is $A_{II}+CDC^T$, with $0\preceq D\preceq I$.
Retain cumulative $D$ exactly on a declared finite rational grid. Legality is
$D_{\mathrm{new}}\in\mathcal D_{\mathrm{grid}}\cap[0,I]$: satisfying the matrix inequalities is not
enough (for example $D_{\mathrm{new}}=I/3$ meets them but violates a declared dyadic grid).
Signed supplied updates are legal only when the resulting matrix remains in that domain.
Named-event retraction also needs a charged event record; subtracting a supplied
matrix does not establish that a historical event occurred. Raw sources, factors,
versions and history are separate retained resources, not hidden in the capsule.

The algebraic identities allow real matrices. **Finite storage and exact full
promotion additionally require an exact rational source contract:** every acquired
entry of $S,F,G$ has numerator/denominator encoding length at most the declared
$g$, source requests finish within their stated budgets, and source versions and
domain promises are validated or explicitly trusted. $g$ refers to the derived
susceptibility entries, not merely the input world's entries. Rational inversion
can increase their bit lengths. For a real source returning only approximations,
entry uncertainty and spectral bounds must be enclosed; full acquisition then
need not give a singleton. That additional implementation is not supplied here.

The normalized scalar contract below uses $S=1$, $F=f^T$, $f=e_1$,
$3I/4\preceq G\preceq5I/4$, unit diagonal and a supplied off-diagonal envelope.
Outputs are scalar absolute-error answers. Interior reconstruction, relative
error, trajectories, arbitrary nonlinear worlds and historical queries are outside
this contract. Any threshold decision needs its own margin condition.

## 1. Exact continuation state

For any symmetric $M\succeq0$ define

$$R_M(D)=D^{1/2}(I+D^{1/2}MD^{1/2})^{-1}D^{1/2}
=D(I+MD)^{-1}=(I+DM)^{-1}D.$$

All three expressions remain valid because $I+D^{1/2}MD^{1/2}\succ0$, including for singular $D$ and singular or zero $M$ without requiring full column rank on $C$. The corrected response is

$$S(D)=S+F R_G(D)F^T.$$

**Derivation.** Apply the inverse-update identity to $A_{II}+CDC^T$, using
$CD^{1/2}$ as the update factor. Substitution into the Schur complement gives
the formula without inverting $D$. The push-through identity gives the two
rational forms. No commutativity is assumed.

Two worlds demonstrate why $S,F$ alone are insufficient:

$$A_1=\begin{pmatrix}2&1\\1&1\end{pmatrix},\qquad
A_2=\begin{pmatrix}3&2\\2&2\end{pmatrix},\qquad C=1.$$

Both have $S=F=1$ and the same minimizing interior $z=-u$. Their susceptibilities
are $G_1=1$ and $G_2=1/2$. The same interior correction $D=1$ produces responses
$3/2$ and $5/3$. This witness precedes, and is separate from, the normalized
spectral family above.

## 2. A sound approximate response

Let $J=\{i:D_{ii}>0\}$ be the **current** support. PSD implies that zero diagonal
entries have zero rows, so restriction to $J$ is exact. Cache an exact principal
block on $K\subseteq J$ and use

$$Q=G_{KK}\oplus I_{J\setminus K}.$$

Principal spectral bounds make $Q$ safely SPD. Filling arbitrary individually
known entries into identity does not supply that guarantee. All remaining
quantities in this section are restricted to $J$. Put $E=G-Q$, $x=R_Q(D)f_J$,
and supply symmetric entrywise nonnegative $H$ with $|E|\le H$. A zero in $H$
requires an acquired entry or a valid structural promise.

**Certificate.** Set

$$q=H|x|,\quad k=|x|^TH|x|,\quad
b=\min\{(4/7)\|q\|_2^2,\ q^T|D|q\}.$$

For $p=S+f_J^Tx$ and the true response $y$,

$$y\in[p-k,\ p+k+b].$$

**Proof.** The two ordered first-resolvent identities imply

$$R_G=R_Q-R_QER_Q+R_QER_GER_Q.$$

Consequently $y-p=-x^TEx+(Ex)^TR_G(Ex)$. Inverse order gives

$$0\preceq R_G\preceq D,\qquad
R_G\preceq D(I+3D/4)^{-1}\preceq(4/7)I.$$

Since $|Ex|\le q$, the first term lies in $[-k,k]$ and the second in $[0,b]$.
The entrywise absolute matrix $|D|$ need not be PSD: its nonnegative entries
justify the absolute-value quadratic bound. If $t=x^TEx$ is known, the sharper
enclosure is $[p-t,p-t+b]$. This does not permit guessing missing signs.

## 3. Large coordinate corrections can be cheap

Let $m=2d$, with $d$ a perfect square, and

$$G=\begin{pmatrix}I_d&W/(32\sqrt d)\\
W^T/(32\sqrt d)&I_d\end{pmatrix},\qquad
W\in\{-1,1\}^{d\times d},\quad\|W\|_2\le8\sqrt d.$$

These premises imply the required spectral bounds. They are construction
premises, not free validation of an arbitrary supplied matrix. For every diagonal
$0\preceq D\preceq I$, the identity proxy has $x=\beta e_1$,
$\beta=d_1/(1+d_1)\le1/2$. With $E=G-I$, $E_{11}=0$ and
$\|Ee_1\|^2=1/1024$. The preceding identity therefore proves

$$0\le S_G(D)-\left(1+\frac{d_1}{1+d_1}\right)\le\frac1{7168}.$$

This holds uniformly over the whole diagonal alphabet, including full-rank
corrections touching every port. At tolerance $1/4096$ the proxy answer leaves
$3/28672$ numerical headroom. Exact $D$ is still retained for action legality and
retraction. The structural nonzero count is essential; a generic entry envelope
alone does not supply this particular row-norm bound.

Mixing directions can expose information that diagonal operations barely affect.
For $G_\pm=\begin{pmatrix}1&\pm1/32\\\pm1/32&1\end{pmatrix}$ and
$D=\tfrac12(1,1)(1,1)^T$, the responses are $81/65$ and $79/63$, separated by
$32/4095$. This is a finite witness, not a quadratic memory lower-bound proof.
Rank and touched support alone are inadequate memory-admission criteria.

Every SPD susceptibility $G$ used above is realizable by an actual energy world:
take $C=I$, $A_{II}=G^{-1}$, $A_{BI}=f^TG^{-1}$ and
$A_{BB}=1+f^TG^{-1}f$. For
$P=\begin{pmatrix}1&0\\f&I\end{pmatrix}$, this is
$A=P^T\operatorname{diag}(1,G^{-1})P\succ0$, with $S=1,F=f^T$.
This is a constructive realization, not a free preprocessing step; exact derived
entry lengths and factor construction remain charged.

## 4. Acquisition does not monotonically improve this certificate

Let $v=\mathbf1\in\mathbb R^8$, $D=vv^T/8$,
$G=I-(e_1e_2^T+e_2e_1^T)/4$, and declare only unit diagonal, spectral bounds
and the generic off-diagonal envelope $1/4$. Initially $K=\{1\}$ gives
$x=v/16$, $k=7/128$, $b=7/128$. Promote port 2. The new proxy happens to equal
the true world, but the reader has not acquired its other structural zeros.
Then $x=2v/31$, $k=54/961$, $b=366/6727$. Thus

$$k+b:\quad7/64\longrightarrow744/6727,\qquad
744/6727-7/64=527/430528>0.$$

Actual proxy error falls from $1/496$ to zero while the certificate worsens.
The indistinguishable completions $G\pm(e_3e_4^T+e_4e_3^T)/4$ have responses
$17/16$ and $16/15$, separated by $1/240>2/4096$. Under this transcript, a
uniform answer at tolerance $1/4096$ is impossible. Analyst knowledge that the
proxy equals the true matrix is not available information for the reader.

**Repair:** freeze the full query identity (world/ports versions, baseline,
readout and exact current $D$). Intersect valid output intervals for that same
identity, retaining two endpoints. Every interval contains $y$, so the
intersection is nonempty and never widens. Emit its midpoint only when certified
halfwidth plus output rounding fits tolerance. A changed correction/readout/version
starts a new interval. Nesting guarantees safety, neither strict progress nor
economical acquisition. Empty intersections diagnose a violated premise or check.

## 5. Finite operation and numerical boundary

An admissible controller performs initial certification, at most one partial
promotion, then at most one full-current-support promotion if access and capacity
permit. Otherwise it returns an enclosure, uses a priced source fallback, or
refuses. Full exact promotion makes $H=0$ and yields a singleton interval.
There are at most three proxy systems/stages; numerical attempts have a
separate finite budget, not constant work or an unconditional latency guarantee.
Source timeouts also need finite budgets.

For an approximate solve $\widehat x$, enclose its residual
$r=Df_J-\widehat x-DQ\widehat x$. Similarity through $Q^{1/2}$ gives
$\|(I+DQ)^{-1}\|_2\le\sqrt{5/3}<2$, so an outward bound
$\rho\ge2\|r\|_2$ implies $\|x-\widehat x\|_2\le\rho$. Residual enclosure must
use the exact intended $D,Q,f_J$ (or enclosing representations of them).
Use $a=|\widehat x|+\rho\mathbf1$ instead of $|x|$ in the certificate.
Require $\eta\ge|\widehat p-(S+f_J^Tx)|$, for which an outward bound on
$|\widehat p-(S+f_J^T\widehat x)|+\|f_J\|_2\rho$ is sufficient; arithmetic
rounding alone is insufficient. Outward endpoints are then
$[\widehat p-\eta-\bar k,\widehat p+\eta+\bar k+\bar b]$, with certificate bounds
and endpoints computed outward.
Intersect **already widened** intervals; a latest-solve allowance does not validate
old endpoints. Exact promotion removes representation uncertainty, not rounding.
This note supplies a numerical construction, not a verified floating-point solver.

## Costs, strongest control, and the next proof

With $s=|J|$, $k=|K|$, current-entry precision $B$ and source precision $g$,
dense retained state costs $O(s^2B+s\log m)$ bits and cache
$O(k^2g+k\log m)$ bits. Source construction/factorization, validation, exact
intermediate bit growth, history, cache eviction/reacquisition, readout and fallback
are additional charges. A solve takes up to $O(s^3)$ field operations; certificates
up to $O(s^2)$. Field-operation counts are not bit-complexity claims.
Promoting $r$ ports reads $kr+r(r-1)/2$ unknown off-diagonal entries when diagonals
are promised. A retained factor source may instead compute a directional
contraction with one solve: rank-one $D=\alpha vv^T$ gives

$$y=S+\frac{\alpha(f^Tv)^2}{1+\alpha v^TGv}.$$

**The control is an ordinary dynamic Schur/resolvent engine with identical access,
cache, factors, precision and history. It can implement these formulas and tie
this construction. No advantage over that control is established.**

The decision-changing next question is acquisition for output ambiguity. Among
all admissible worlds consistent with a transcript, let $w_*$ be the width of the
true scalar-output range. Compactness and continuity give extrema. Its optimal
uniform absolute-error radius is $w_*/2$; acquiring true entries cannot widen
that range. Computing it need not be cheap. Start with rank-one corrections,
where uncertainty reduces to $v^T Gv$, and prove or refute a cost-competitive
acquisition rule against a certifying extension with the same access. Direct
directional acquisition must remain an alternative. A failed competitive bound
should narrow the admissible world/action family or change the constructor.

For the unified program, this is a concrete bridge question: can a learned
constructor preserve intervention-conditioned consequences, with recovery,
while removing recurring backbone computation? Nothing here establishes that
bridge, trains a model or launches a separate research queue.

**Transfer boundary (independent review, 2026-10-15).** No operation-preserving map from
this quadratic world to a dependency ledger is proved or claimed. E1 and E2 in
`experiments/cloud/RESEARCH_QUEUE.md` are independent graph constructions with their own
correctness contract: the exact versioned graph evaluator is the oracle, and no result about
them inherits this document's identities. A learned quadratic constructor would be the
direct alternative bridge. The finite checker verifies witnesses only; it does not validate
its expected outputs or the general domain premises.

## Prior mathematics and verification

Inverse updates and Schur response are classical. Dörfler and Bullo give explicit
interior-edge response updates under graph-Laplacian premises in
[Kron Reduction, Section III.D/Theorem III.14](https://motion.me.ucsb.edu/pdf/2011d-db.pdf).
The signed SPD family here is not automatically a graph realization.
[Andoni et al., Section 5.1/Theorem 5.1](https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/ackqwz16.pdf)
establish a general PSD simultaneous quadratic-sketch obstruction; their randomized
fixed-query guarantees have different quantifiers. No specialized asymptotic
lower bound or novelty is admitted here.

The [stdlib exact checker](../experiments/analysis/correction_capsule_exact.py)
and its [fixed configuration](../experiments/analysis/correction_capsule_exact.json)
check finite arithmetic witnesses. They do not prove universal propositions.
Independent proof review and execution receipts are recorded separately from
design dialogue. All inherited empirical claims and unrun controls remain intact.

### Verification record

- **Root execution:** The [stdlib exact checker](../experiments/analysis/correction_capsule_exact.py) and its [fixed configuration](../experiments/analysis/correction_capsule_exact.json) verified 12 exact arithmetic test cases (PASS).
- **Independent proof audit:** Reviewer conducted universal proof reading and executed 22 in-memory exact-rational assertions (PASS), covering singular/noncommuting identities, the ordered resolvent identity, finite witness values, and tolerance separation.
- **Audit execution provenance:** An initial CLI attempt was incomplete; the fresh retry audit was executed using the actual REAL Codex CLI with clean disk context (`internal/analysis_2026_10_14/codex_proof_retry_audit.md`).
- **Epistemic boundary:** Finite arithmetic checks verify finite witnesses only; they never constitute universal proofs or novelty claims.
