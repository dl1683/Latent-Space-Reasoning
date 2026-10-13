**Seven-cycle verdict: the best frontier is a joint law for finite erasure, retained randomness, and gate work. It has a concrete proof seed, but useful efficiency and novelty remain open.** Two further successors survive as research hypotheses: online distributed cleanup and randomness-efficient testing against adaptive erasures. None establishes the moonshot.

I read the requested repository materials, AGENTS.md, and the [dual-loop skill](<USER_HOME>/.claude/skills/dual-loop/SKILL.md) in full. I also read the newly available atomic-erasure definition and compiler. I made no file edits or git writes, spawned no agents or ongoing jobs, and used no private memory as evidence. The experiments below were bounded, in-memory CPU smokes.

The repository’s current standing remains **no verified new-paradigm pair**. HANDOFF §17 and SYNTHESIS_GRAPH §29 leave semantic obligations open; TERMINAL_GATE_2 records NO. The DFT result remains scoped secondary mathematics. EBC’s entropy inequality does not establish its physical interpretation, novelty, or claimed comparison with conventional computation.

All new mathematical arguments below are **CANDIDATE PROOFS**, pending independent review. Proposed extensions are **CONJECTURES**. Experiment observations are **EMPIRICAL**.

**Entry conditions for all seven cycles**

These are inherited constraints, not counted as new objections:

- A conventional simulator may retain the source state and compute every derived field internally. Publishing deterministic history fields creates no semantic power.
- Both sides must satisfy the same output, cleanup, accuracy, initialization, and access contract.
- Internal state remains permitted. Resource bounds must charge it rather than prohibit it.
- Finite gate syntax, program generation, routing or addressing, initialization, random bits, and precision must appear in the accounting.
- An occupied resource framework can still contain a new theorem. Renaming its objects, or adding independent bounds together, does not establish one.

### The seven cycles

**1. Atomic erasure under an identity continuation**

**Steelman.** The current atomic candidate improves on EBC’s endpoint inequality. For \(M\) equiprobable initial atoms and largest output fibre \(q\), it proposes the exact all-depth feasibility threshold

\[
K_{\min}=\frac{\phi(q)}{M},
\qquad
\phi(q)=q\log_2q-(q-1)\log_2(q-1).
\]

Determinism prevents merged atoms from separating later. The charged compiler realizes selective merges using NOT, CNOT, Toffoli, and RESET. This is a concrete candidate theorem pair, not an arbitrary-merge oracle.

**New hardest objection.** Append \(r\) independent live bits \(Z\), and require them to survive unchanged:

\[
(X,Z)\longmapsto(f(X),Z).
\]

The information erased remains \(H(X)-H(f(X))\). Nevertheless, the largest final fibre still contains only \(q\) atoms, while the initial sample space contains \(M2^r\). Applying the candidate argument gives

\[
K_{\min}^{\mathrm{return}}
=\frac{\phi(q)}{M2^r}.
\]

Thus an identity continuation can divide the peak threshold by \(2^r\), without changing total information loss.

**Reconstruction.** This defeats any claim that peak erasure is a context-independent property of \(f\). It supports a richer resource vector: peak loss, gate work, width, and retained uncertainty. Retained uncertainty can spread erasure over a longer schedule.

**Control.** CONFIRM the scaling only with \(Z\) preserved pointwise and all other workspace cleared. KILL any “intrinsic threshold of \(f\)” claim that ignores this extension. VOID calculations that omit the appended register or change its distribution.

**Survivor:** a compositional tradeoff problem, not an intrinsic scalar invariant.

---

**2. Retained randomness versus randomness that must be cleaned**

**Steelman.** Random padding appears to soften atomic granularity. Every instance still has finite syntax and finitely many atoms.

**New hardest objection.** Returning the padding and cleaning it are different tasks. If the final state is only \(f(X)\), then each original fibre expands to \(q2^r\) atoms:

\[
K_{\min}^{\mathrm{clean}}
=\frac{\phi(q2^r)}{M2^r},
\qquad
L_{\mathrm{clean}}=H(X)-H(f(X))+r.
\]

If \(Z\) is returned unchanged, the threshold is instead \(\phi(q)/(M2^r)\), and total loss gains no \(r\)-bit term.

Both thresholds can become small as \(r\) grows. Only the returned-register contract avoids the extra cleanup debt.

**Reconstruction.** Randomness has at least three distinct roles: sampled and consumed, sampled and returned, or already present as live input. Their acquisition and final-state obligations must be explicit. “Randomization removes the bound” is too coarse.

This is close to established catalytic-space and entropy-catalysis work. Catalytic space already permits auxiliary memory that must be restored; entropy catalysis already studies transformations assisted by preserved catalysts. Neither observation establishes the proposed finite-gate peak/work frontier. [Buhrman et al., *Computing with a full memory*](https://ir.cwi.nl/pub/22659), [Wilming, *Entropy and reversible catalysis*](https://arxiv.org/abs/2012.05573).

**Control.** Run paired contracts with identical initial distributions and gate bases. CONFIRM only separately reported resource vectors. KILL any advantage obtained by moving random registers from “must clean” to “may retain” on one side. VOID uncharged random initialization.

**Survivor:** paid randomness with an explicit lifecycle.

---

**3. Feasibility-optimal erasure versus a useful succinct compiler**

**Steelman.** The atomic compiler closes an important gap: selective merging is implementable using constant-arity gates, with finite reusable scratch.

**New hardest objection.** Its truth-table construction can spend exponential work acquiring the function and identifying individual atoms. Exact feasibility does not establish an efficient capability on succinct useful tasks. A truth-table circuit for AND would conceal this issue because AND itself has a short description.

**Reconstruction.** A compact capability seed exists for AND with returned random bits. Compute AND reversibly, copy its result, reverse the computation, and then erase input bits separately for each value of \(Z\). Each reset affects probability mass at most \(2^{-r}\). The construction uses

\[
G=O\!\left(n+2^r(n+r)\right),
\qquad W=O(n+r),
\qquad K\le 2^{-r}.
\]

Choosing \(r=\lceil\log_2(1/K)\rceil\) gives a succinct upper bound of approximately

\[
G=O\!\left(\frac{n+\log(1/K)}{K}\right).
\]

This is useful as an explicit compiler control. It leaves a much harder question: can similarly small gate work be achieved with only the retained randomness required by the atomic feasibility bound?

Bennett’s reversible compute/copy/reverse construction and later energy-aware algorithms occupy much of the foundation. The new obligation is the simultaneous peak-loss, randomness, width, and gate-work frontier. [Bennett, *Logical Reversibility of Computation*](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html), [Demaine et al., *Energy-Efficient Algorithms*](https://arxiv.org/abs/1605.08448).

**Control.** CONFIRM only a uniform generator with charged construction work and explicit gates. KILL a near-optimal-work conjecture with a certified circuit lower bound. VOID an abstract partition schedule presented as a gate implementation.

**Survivor:** successor H1 below.

---

**4. Continuation-dependent retention meets local cleanup**

**Steelman.** Online future requests can make information valuable now even when a currently selected output does not use it.

**New hardest objection.** Global sufficiency does not determine the cost of clearing geographically separated copies. A site may possess a locally uncertain bit that is redundant given another site’s state. Exploiting that redundancy requires a charged interaction.

The bare “retain enough information to answer every continuation” formulation remains Myhill–Nerode or INDEX. The bare “erase correlated local records” formulation is already closely occupied by thermodynamics of modularity. Optimal predictive internal states are also part of that literature. [Boyd, Mandal, and Crutchfield, *Thermodynamics of Modularity*](https://arxiv.org/abs/1708.03030).

**Reconstruction.** Require online service, local memory limits, communication, and eventual cleanup in one contract. Permit arbitrary internal encodings. Seek a bound on their joint feasible region, rather than another lower bound on sufficient-state cardinality.

A finite seed gives the exact static relation

\[
E_{\min}(C)=\max(n,2n-C)
\]

for clearing two identical \(n\)-bit records under the local cost model specified below. That seed is not a new paradigm; it supplies a control for a genuinely online extension.

**Control.** CONFIRM the seed against arbitrary permitted local computation. KILL the successor if its online theorem reduces to INDEX plus independent local-erasure accounting. VOID any uncharged remote conditioning.

**Survivor:** successor H2, conditional on a new online composition lemma.

---

**5. Linear requests do not justify linear controllers**

**Steelman.** Uniform binary sources and linear requests suggest a finite subspace calculus. Gaussian elimination gives explicit algorithms, and rank gives useful lower bounds.

**New hardest objection.** A universal CANNOT theorem cannot optimize only over linear internal encodings. Nonlinear controllers may use biased states, coded intermediates, or interactive transformations. Linear demands alone do not prove linear optimality. Network coding supplies an established warning: nonlinear codes can outperform all linear codes in general networks. [Dougherty, Freiling, and Zeger, *Insufficiency of Linear Coding in Network Information Flow*](https://citeseerx.ist.psu.edu/document?doi=2c80ca4786e43d4af0516e43662187c4652c048b&repid=rep1&type=pdf).

**Reconstruction.** Restrict the *task family* initially to two sites and a public request tree, while retaining arbitrary Boolean controllers. Formulate linear-state optimality as a conjecture to attack, not as a machine axiom.

A useful theorem must capture compatibility across branches: the policy before a request is revealed must support every permitted continuation. Optimizing each leaf independently is insufficient.

**Control.** Enumerate small nonlinear machines alongside affine policies. Any strictly better nonlinear policy is KILL for the proposed normal form. A timeout is VOID. Agreement on small instances permits further work but does not establish universality.

**Survivor:** a fragile, falsifiable normal-form hypothesis.

---

**6. Short random seeds meet an adaptive erasure adversary**

**Steelman.** Property testing offers useful sublinear tasks with concrete CAN/CANNOT results under online input degradation.

**New hardest objection.** Replacing independent queries with a short seeded walk can expose future query support. If the next query lies among at most \(t\) publicly predictable possibilities, an adversary permitted \(t\) erasures can erase all of them. A constant-degree expander walk therefore fails this control once \(t\) reaches its degree, irrespective of its ordinary mixing properties.

The same issue threatens algebraic generators whose future becomes predictable after enough outputs. Private internal state is allowed; its remaining unpredictability must be analyzed conditional on the public transcript.

**Reconstruction.** Generate fresh private choices within a structured family that preserves parity witnesses. Track query count, private random bits, retained state, and bit work jointly.

Online erasure testing is already an established model, with optimal query results for linearity and later improvements across erasure regimes. Improving query complexity alone would reoccupy that theory. [Kalemaj, Raskhodnikova, and Varma](https://theoryofcomputing.org/articles/v019a001/), [Ben-Eliezer et al.](https://arxiv.org/abs/2311.16566), [Arora, Kelman, and Meir](https://arxiv.org/abs/2411.14431).

**Control.** Include an adversary that erases the highest-probability next-query support after every response. KILL constructions justified only by unconditional marginals. VOID tests against an offline erasure set substituted for the online adversary.

**Survivor:** successor H3.

---

**7. Structured parity witnesses meet false local consistency**

**Steelman.** A small algebraic query family might preserve linearity witnesses while using substantially fewer random bits than independent domain samples.

**New hardest objection.** Perfect consistency on every allowed local subspace need not imply global linearity.

A finite counterexample is available. In \(\mathbb F_{16}\), let

\[
W=\mathbb F_4=\{0,1,6,7\}
\]

using the representation defined by \(x^4+x+1\). Set \(f(1)=f(6)=1\), and set \(f=0\) elsewhere. On each multiplicative copy \(aW\), the restriction is \(\mathbb F_2\)-linear: the nonzero copies partition the field into five triples, and the only nonzero restriction has values \(1,1,0\).

Yet \(f\) is globally \(1/8\)-far from linear. Its distance from zero is \(2/16\); every nonzero linear function has eight ones and differs from it on at least six points. Every parity witness confined to one \(aW\) passes.

Avoiding proper subfields removes this counterexample, but does not establish robust soundness. Exact kernel dimension is also insufficient: a far function might violate only a vanishing fraction of constraints.

**Reconstruction.** Use a specified power-basis subspace rather than a subfield, and demand a robust global soundness lemma plus survival under adaptive erasures. Ordinary randomness-efficient linearity testing is occupied, and online randomness complexity already has substantial theory. [Ben-Sasson et al., *Randomness-efficient Low Degree Tests*](https://people.seas.harvard.edu/~salil/research/epsbias.pdf), [Kelman, Linder, and Raskhodnikova, *Online versus Offline Adversaries*](https://arxiv.org/abs/2411.18617).

**Control.** The proper-subfield construction must fail as a negative control. KILL the main sampler with a far function having insufficient rejection probability. VOID a positive result based only on exact consistency or nonadaptive sampling.

**Survivor:** H3’s specific adaptive composition lemma remains open.

### Three specific successor hypotheses

**H1 — A succinct peak-erasure/work frontier with returned randomness**

**Status: CONJECTURE, with an explicit CANDIDATE-PROOF capability seed.**

**Finite syntax and cost model.** Fixed-width Boolean circuits over NOT, CNOT, Toffoli, and one-bit RESET. Programs are generated uniformly from finite integers and a rational architectural cap \(K\). Charge gate count \(G\), width \(W\), program generation and storage, initialization, and

\[
K=\max_j\bigl(H(S_j)-H(S_{j+1})\bigr).
\]

All data and controller state enter \(S_j\). Execution schedules are input-independent. No physical heat or clock-duration interpretation is assumed.

**Useful common-contract task.** For uniform independent \(X\in\{0,1\}^n\) and \(Z\in\{0,1\}^r\), return

\[
(\operatorname{AND}(X),Z)
\]

and clear every other register, for every \(x,z\). This models aggregate computation followed by workspace cleanup while preserving an unrelated live record. Both sides receive the same initial registers.

**Explicit uniform CAN algorithm.**

1. Compute AND with a reversible Toffoli tree.
2. Copy its output and reverse the tree.
3. For each \(z\in\{0,1\}^r\), reversibly compute \(d=[Z=z]\).
4. For each input bit \(x_i\), use a zero flag \(b\):
   \[
   b\leftarrow b\oplus(d x_i),\quad
   x_i\leftarrow x_i\oplus b,\quad
   b\leftarrow0.
   \]
5. Uncompute \(d\) and reuse the scratch.

The first two operations in step 4 are reversible. RESET merges states only when \(Z=z\). Its joint entropy loss is at most

\[
\Pr[Z=z]=2^{-r}.
\]

All inputs eventually clear; the output and \(Z\) survive exactly. Predicate construction uses explicit constant-arity gates.

The resulting seed has

\[
G=O(n+2^r(n+r)),\qquad W=O(n+r).
\]

**Universal CANNOT technique.** Two arguments apply to every deterministic realization under this contract, regardless of internal encoding:

\[
G K\ge n-h_2(2^{-n}),
\]

by telescoping full-state entropy, and

\[
K\ge
\frac{\phi(2^n-1)}{2^{n+r}},
\]

by indivisible full-state fibres. The second is conditional on independent review of the atomic proof; it cannot be bypassed by reversible recoding or extra zero scratch.

**Load-bearing proposed strict delta.** For every feasible rational \(K\in(0,1]\), can a uniform compiler achieve

\[
G=O\!\left(n+r+\frac{n-h_2(2^{-n})}{K}\right),
\qquad W=O(n+r),
\]

including when \(r\) is only the minimum needed for atomic feasibility?

The existing truth-table compiler does not establish this. The explicit seed above establishes an efficient region with more retained randomness. Closing the gap would connect granularity, retained information, and executable work rather than merely juxtaposing two lower bounds.

**Nearest occupants.** Bennett’s reversible simulation, energy-aware algorithms, Frank’s conditional reversibility, catalytic space, and compositional generalized reversible computing. The last already treats partitioned states and resource-theory composition. The proposed delta must survive translation into that framework. [Frank](https://arxiv.org/abs/1806.10183), [Chanavat and Srinivasan](https://arxiv.org/abs/2511.02425).

**Cheap CPU falsification.** Exhaustively search fixed-gate circuits for \(n=2,3\), \(r=0,1,2\), with small width and gate limits. Maintain exact integer state multiplicities; certify entropy comparisons with rational intervals or exact product comparisons. Search both the proposed efficient compiler and arbitrary permitted circuits.

- **CONFIRM gate:** correct clean circuits in the predicted region, followed by a uniform construction argument.
- **KILL:** a certified asymptotic obstruction to the proposed work bound, or a counterexample to the atomic bound.
- **VOID:** abstract merges, omitted flags, omitted initialization, or an incomplete search reported as impossibility.

**So what:** determine when additional live uncertainty buys lower peak erasure at an honestly priced execution cost.

---

**H2 — Online distributed cleanup with arbitrary internal encodings**

**Status: CONJECTURE. The static proof seed below is complete enough for immediate independent attack.**

**Finite syntax and cost model.** Two finite sites hold initial linear views \(AX\) and \(BX\) of uniform binary \(X\). A finite public request tree specifies requested linear outputs, their destination sites, memory caps at designated boundaries, and clean final registers.

Local computation may use arbitrary NOT/CNOT/Toffoli circuits and RESET. A cross-site CNOT costs one communication unit. Charge:

- \(E\): sum of decreases in the reset site’s full marginal entropy;
- \(C\): cross-site gates;
- site widths and local gate work;
- request-tree, matrix, and controller representation.

Public requests carry no hidden information about \(X\). Outputs retained for later use count toward memory. Controllers may retain any information that fits the stated resources.

**Useful common-contract task.** Serve later aggregation or parity requests from replicated or sharded records while reducing local storage, then clear unauthorized workspace at termination. This is an exact service-and-cleanup contract, without a cryptographic deletion claim.

**Explicit uniform CAN algorithm.** Enumerate branch-compatible affine state policies under the finite memory and communication budgets. Track their Pareto costs. Compile feasible policies through charged local Gaussian elimination, cross-site CNOTs, and resets; retain a basis spanning every still-required request.

This is finite and uniform, but potentially expensive. It supplies an algorithm for the linear policy class, not a universal optimality theorem.

**Universal CANNOT technique and finite proof seed.**

Start with identical records \(S_A=S_B=X\), where \(X\) has \(n\) uniform bits. End with both sites completely zero.

Let

\[
F=H(S_A)+H(S_B).
\]

Initially \(F=2n\); finally \(F=0\).

- Local reversible computation preserves \(F\).
- A local reset decreases \(F\) by exactly its charged local loss.
- A cross-site CNOT changes only one target bit, so it decreases \(F\) by at most one bit.

Consequently,

\[
E+C\ge2n.
\]

Global joint entropy must decrease from \(n\) to zero. Local reset loss bounds its joint entropy loss from above, so

\[
E\ge n.
\]

For an integer communication budget \(c\), take \(q=\min(c,n)\). Use \(q\) cross-site CNOTs to clear \(q\) duplicated bits at site B. Reset B’s remaining \(n-q\) bits and all \(n\) bits at A. This attains

\[
E=2n-q,
\]

hence

\[
\boxed{E_{\min}(c)=\max(n,2n-c)}.
\]

The lower argument permits arbitrary local nonlinear computation and scratch. It therefore does not obtain the separation by banning internal state.

**Load-bearing proposed strict delta.** For two-site public linear request trees, conjecture that arbitrary Boolean controllers cannot improve the \(E,C,\text{memory}\) frontier over branch-compatible affine policies.

The required theorem must expose a crossed online constraint: some request family must cost strictly more than the maximum of independently optimized leaf costs, and that excess must involve cleanup and communication jointly. A cardinality bound on retained state, or the static formula above alone, fails the novelty gate.

**Nearest occupants.** Thermodynamics of modularity and predictive-state machines; circuit thermodynamics; communication complexity; reversible computation; nonlinear network coding. These occupy the ingredients. The proposed two-site online normal form is a separate, currently unsupported claim. [Wolpert and Kolchinsky, *The thermodynamics of computing with circuits*](https://arxiv.org/abs/1806.04103).

**Cheap CPU falsification.** For \(n=2,3\) and two-level request trees, enumerate complete state maps generated by both affine and nonlinear gates, including scratch and controller bits. Compare feasible regions with the affine dynamic program.

- **CONFIRM gate:** finite agreement plus an explicit branch-crossing witness worth proving.
- **KILL:** any nonlinear policy improving the proposed frontier; or a reduction showing the entire result is known INDEX, pebbling, or ordinary online storage accounting.
- **VOID:** uncounted outputs, free remote information, leaf-specific preprocessing, or incomplete enumeration.

**So what:** characterize when future service obligations prevent economical distributed cleanup.

---

**H3 — A private structured sampler for online-erasure linearity testing**

**Status: CONJECTURE. Both the upper composition lemma and proposed joint lower bound are open.**

**Finite syntax and cost model.** Input is a finite Boolean table \(f:\mathbb F_2^d\to\mathbb F_2\), accessed through the same indexed oracle on both sides. After each answer, an adversary observes the public transcript and may erase up to \(t\) entries.

Charge queries \(Q\), private random bits \(R\), retained bits \(S\), and bit-operation work. Address construction costs \(d\)-bit work. Field arithmetic uses a supplied, charged irreducible polynomial and ordinary finite algorithms. No physical routing advantage is asserted.

**Useful common-contract task.** With fixed \(\varepsilon=1/8\), distinguish a linear \(f\) from one \(\varepsilon\)-far from every linear function, with one-sided error and success at least \(2/3\), against every permitted online adversary.

**Explicit uniform CAN baseline.** Existing testers sample independent uniform base points and privately select parity relations among them. A tester can choose its private subset mask first and stream the selected-address XOR and response parity, avoiding storage of the whole address bank. This preserves the established capability; the small-memory implementation itself is not a novelty claim.

**Proposed uniform CAN algorithm.** In a finite field representation of \(\mathbb F_2^d\), fix

\[
W=\operatorname{span}_{\mathbb F_2}
\{1,\beta,\ldots,\beta^{r-1}\},
\]

with \(\beta\) and its defining polynomial explicitly specified. Choose a private uniform nonzero multiplier \(a\). Generate base points \(x_i=a z_i\), with fresh \(z_i\) uniform in \(W\). Stream parity witnesses selected by private masks.

Use \(q=\Theta(\log(t+2))\) base points and \(r=q+O(\log q)\), in the regime \(d\) sufficiently larger than \((\log t)^2\). The intended bounds are

\[
Q=O(\log t),\qquad
R=d+O((\log t)^2),\qquad
S=O(d+\log t).
\]

Bounded rejection sampling and all exceptional events must consume a preallocated error budget.

**Load-bearing proposed strict delta.** Prove that this small family retains a constant probability of a *global* linearity violation and that a violation survives every adaptive erasure strategy. The two conclusions must hold together; unconditional soundness plus marginal anti-erasure estimates is insufficient.

**Universal CANNOT technique.**

The established span-erasure/Yao technique gives an \(\Omega(\log t)\) query requirement without restricting internal memory.

A second, elementary support argument supplies a randomness control. Under the all-zero answer transcript, a \(Q\)-query machine with at most \(R\) random bits visits at most \(Q2^R\) addresses over all seeds. If this union is sufficiently small, a counting argument supplies an \(\varepsilon\)-far function agreeing with zero everywhere visited. The machine cannot distinguish it from the zero linear function. This yields

\[
R\ge d-\log_2 Q-O(1)
\]

for constant proximity and sufficiently large domains.

Those separate bounds are occupied-style controls. A stronger **CONJECTURE** is that achieving \(Q=O(\log t)\) requires

\[
R\ge d-O(\log\log t)+\Omega((\log t)^2).
\]

The proposed technique is an adversary that repeatedly erases the highest-probability conditional next-query support, combined with a hard input distribution forcing enough surviving independent information. This lower bound is not established.

**Nearest occupants.** The KRV, BKMR, and Arora–Kelman–Meir online testers; Ben-Sasson–Sudan–Vadhan–Wigderson derandomized linearity tests; and Kelman–Linder–Raskhodnikova’s online randomness separations. The target is a specific joint query/randomness theorem for linearity, not a new name for their model.

**Cheap CPU falsification.**

For small \(d\), construct the parity-check matrix generated by all allowed scaled-subspace witnesses.

1. Check whether its exact kernel consists only of global linear functions.
2. Measure rejection probability for far functions; include glued local solutions.
3. Solve tiny adaptive erasure games for \(t=1,2\).
4. Run the proper-subfield counterexample as a mandatory negative control.

- **CONFIRM gate:** the negative control fails, and the specified main family survives both robust-soundness and adaptive controls.
- **KILL:** an extra nonlinear kernel element, vanishing rejection probability on far functions, or an adversary defeating the claimed success bound.
- **VOID:** testing only random inputs, static erasures, or exact consistency without robustness.

**So what:** determine the private randomness cost of preserving useful algebraic witnesses while access deteriorates adaptively.

### Experiments completed in this batch

These were precommitted finite smokes, not independently reviewed outcome experiments. They warrant **EMPIRICAL smoke observations**, not PROVED labels or full evidence-gate approval.

| Smoke | Finite coverage | Observation |
|---|---|---|
| Local cleanup seed | One duplicated bit; communication caps 0, 1, 2 | Minimum local losses 2, 1, 1 |
| Clean AND control | \(n=2,4,8\); layer caps \(K=1,2,4\) | All nine cases produced correct clean states and respected the cap |
| Returned-randomness AND | \(n=2,4,8\); \(r=1,2\) | All six cases preserved \(Z\), cleared other workspace, and respected \(2^{-r}\) |

For the returned-randomness construction, the complete results were:

| \(n\) | \(r\) | Exhaustive input rows | Width | Gates | Maximum joint entropy drop | Cap |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 8 | 7 | 23 | 0.344361 | 0.5 |
| 2 | 2 | 16 | 8 | 51 | 0.172180 | 0.25 |
| 4 | 1 | 32 | 11 | 39 | 0.467246 | 0.5 |
| 4 | 2 | 64 | 12 | 79 | 0.233623 | 0.25 |
| 8 | 1 | 512 | 19 | 71 | 0.498041 | 0.5 |
| 8 | 2 | 1,024 | 20 | 135 | 0.249021 | 0.25 |

Correctness and cleanup were checked for every enumerated input. Entropy used full-state multiplicities with floating-point evaluation; future boundary tests require certified comparisons. The local search used two data cells and did not empirically establish arbitrary-workspace universality. Its broader claim rests on the candidate potential proof.

### Failure synthesis and novelty uncertainty

**HEURISTIC synthesis.** The graveyard does not show that fundamental progress is impossible. It shows that the search repeatedly selected descriptions whose apparent advantages disappeared when ordinary implementations, preparation costs, or controlling task contracts were restored.

The recurring failures are:

- **Derived semantics:** deterministic fields internalize into an ordinary simulator.
- **Task drift:** output-only computation was compared with clean-output computation.
- **Preparation drift:** tables, routing, calibrated devices, or random reservoirs entered without their acquisition costs.
- **Resource renaming:** sufficient states, erased bits, queue occupancy, and pebble schedules acquired new vocabulary without a new theorem.
- **Factorization:** independent lower bounds were presented as a composition result.
- **Restricted universality:** impossibility covered an authored representation or algorithm class.
- **Premature completion:** a scoped theorem or finite experiment was promoted to a paradigm verdict.

Two additional traps emerge here. Peak expected entropy is sensitive to retained spectator uncertainty, so its composition law matters. Structured randomness can preserve every permitted local identity while missing globally far inputs, so exact algebraic consistency is not enough.

The relevant literature is broader than the older program map. Compositional generalized reversible computing is now an explicit nearby occupant. Noisy reversible computation also has longstanding capability and impossibility results; a “fresh purity” model would need a specific theorem beyond those foundations. [Aharonov et al., *Limitations of Noisy Reversible Computation*](https://arxiv.org/abs/quant-ph/9611028).

**Novelty remains uncertain for all three successors.** H1 may be an unrecognized conditional-reversibility scheduling theorem. H2 may collapse into known information-flow optimization, or fail immediately against nonlinear encodings. H3 may follow from existing derandomized tests, or fail its robust soundness lemma. None may claim novelty from unsuccessful source searches.

### Ranked next gates

1. **Review the atomic proof and the returned-register extension together.** Attack the complete-state partition argument, compiler scratch, exact entropy comparisons, initialization, and spectator scaling. This resolves the strongest immediate mathematical foundation.

2. **Attack H1’s succinct work conjecture.** Start with small low-randomness gate searches and a uniform constructive attempt near the feasibility boundary. Seek executable efficiency, rather than another feasibility-only theorem.

3. **Try to kill H2 with nonlinear controllers.** Test small request forks before investing in the subspace formalism. Require a cleanup/communication constraint crossing future branches; reject a disguised INDEX or storage theorem.

4. **Run H3’s algebraic negative controls before adaptive experiments.** An extra kernel element or weak robustness ends the proposed sampler quickly. Only surviving families deserve an online-erasure proof attempt.

5. **Close source equivalence for the surviving theorem, then run registered experiments.** Match contracts and costs against the nearest occupants. Obtain fresh independent proof/code review and raw-output review before promotion.

6. **Record and commit through the authorized work loop.** This read-only batch supplies the report only. A useful same-contract pair, an earned strict delta, and a fresh whole-repository adversarial review are still required for a milestone claim.

The strongest immediate result is a finite construction showing how returned uncertainty can spread cleanup across a charged schedule. The research target is the optimal joint law governing that capability and its limits. **The home-run mission remains open.**