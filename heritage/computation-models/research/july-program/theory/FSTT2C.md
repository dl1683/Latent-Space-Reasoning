# FSTT-2C Gate 0 and Death Certificate

**Date:** 2026-07-14  
**Role:** Work-loop worker, Batch 4  
**Frozen registration:** `research/REFRAME_ROUND3.md` Part 3  
**Batch result — HEURISTIC:** `TELESCOPE-ONLY`  
**Corroborating deaths — HEURISTIC:** `NO-LOAD-BEARING-CANNOT` and
`MISSION-KILL`  
**Endgame consequence — HEURISTIC:** `NEITHER-AGAIN`; escalate the proposed
mission narrowing to the supervisor and Devansh.

`FSTT-CLEAR-TO-PROVE` did not issue. No conditional theorem proof was opened,
no proof-review packet was assembled, and no statement created in this batch is
labelled `PROVED`.

---

## 0. Claim discipline and binding outcome

- **PROVED BY REVIEW** is used only for the finite-kernel TV telescope already
  attacked and upheld in `research/REVIEW_1_FINDINGS.md`.
- **PROVED (literature)** means the cited primary source states the result in the
  cited scope; this batch did not independently reproduce the source proof.
- **EMPIRICAL SOURCE FINDING** means a bounded search or direct source inspection,
  never literature-wide absence.
- **CANDIDATE DERIVATION — UNREVIEWED** means a complete-looking mathematical route
  used to test Gate 0. It is not a repository theorem.
- **DEFINITION — GATE-0 FROZEN** fixes the object adjudicated here; it is not a
  novelty claim.
- **HEURISTIC** covers token decisions, novelty judgments, interpretations, and
  mission consequences. Unless a paragraph is given a stronger label, it is
  **HEURISTIC**.

**HEURISTIC — EXACT BINDING VERDICT.** The exact conjunction was not found verbatim
in the bounded source closure, and both narrow constructors can be made uniform,
finite, and non-oracular. Those facts are insufficient for clearance. Once the
constructor-specific established results are imported, the common FSTT-2C proof has
only three moves: convert each constructor into finite sampled kernels, apply the
elementary TV telescope, and total the endpoint/resource ledger. That is exactly the
frozen `TELESCOPE-ONLY` death. The CANNOT then follows because polynomial-time
finite samplers and a polynomial number of sampled transitions were inserted into
the admission rule; no premise is shown necessary for efficient classical sampling.
The theorem unit therefore stops at Gate 0.

**HEURISTIC — ALTERNATIVE INTERPRETATION CONSIDERED.** One can value a single
preparation-to-readout contract spanning clocked stochastic circuits and CTMJs as an
audit standard. That is the project's floor. It does not change the mathematics of
either constructor, identify a new capability boundary, or defeat the registration's
anti-checklist death.

---

## 1. Iteration 1 — exact registration

### 1.1 Common finite task and transducer syntax

**DEFINITION — GATE-0 FROZEN.** Let \(x\in\{0,1\}^n\) and let the requested accuracy
be \(2^{-p}\), with complexity measured in \(n+p\). A finite stochastic transducer
instance is

\[
\mathcal A_{x,p}=(S_{x,p},P_{x,p},T_{x,p},
K_{x,p,1},\ldots,K_{x,p,T_{x,p}},R_{x,p},O_{x,p}),
\]

where:

1. \(S_{x,p}\subseteq\{0,1\}^{m(x,p)}\) is a finite encoded state space;
2. \(P_{x,p}:\{\star\}\rightsquigarrow S_{x,p}\) is the preparation kernel;
3. \(T_{x,p}\in\mathbb N\) is the declared discrete horizon;
4. every \(K_{x,p,t}:S_{x,p}\rightsquigarrow S_{x,p}\) is a finite Markov
   kernel;
5. \(O_{x,p}\subseteq\{0,1\}^{\leq \ell(x,p)}\) is a finite output set; and
6. \(R_{x,p}:S_{x,p}\rightsquigarrow O_{x,p}\) is the finite stochastic readout.

The ideal delivered law is

\[
\nu_{x,p}=P_{x,p}K_{x,p,1}\cdots K_{x,p,T_{x,p}}R_{x,p}.
\]

**DEFINITION — GATE-0 FROZEN.** One deterministic uniform **meta-constructor
interface** receives \((x,1^p)\). Each substrate lemma supplies a uniform branch of
that interface; the branch selector is the declared finite syntax class, not advice.
The selected branch produces the finite descriptions above plus bounded-time
fair-bit sampling programs \(P',K'_t,R'\). Succinct local circuit/generator
descriptions are allowed, but their row evaluation and sampler programs must be
uniform and polynomial; an exponentially materialized transition table is not
silently free. The branch also produces checkable component bounds

\[
\operatorname{TV}(P,P')\leq\epsilon_{\rm prep},\qquad
\sup_z\operatorname{TV}(K_t(z,\cdot),K'_t(z,\cdot))\leq\epsilon_t,
\qquad
\sup_z\operatorname{TV}(R(z,\cdot),R'(z,\cdot))\leq\epsilon_{\rm read},
\]

When the task declares a target law \(\tau_{x,p}\) distinct from the exact transient
law \(P K_1\cdots K_T R\), the constructor must additionally derive
\(\operatorname{TV}(P K_1\cdots K_T R,\tau_{x,p})\leq
\epsilon_{\rm target}\). With

\[
\epsilon_{\rm target}+\epsilon_{\rm prep}+\sum_t\epsilon_t+
\epsilon_{\rm read}\leq2^{-p},
\]

the D-LSN execution task uses \(\epsilon_{\rm target}=0\), while the CTMJ Gibbs task
uses the generated mixing allocation \(\epsilon_{\rm target}=\epsilon_{\rm mix}\).
The constructor must derive these bounds from the generated syntax. A user-supplied
statement that a preparation, terminal law, mixing time, or readout is correct is not
admitted as a certificate.

**DEFINITION — GATE-0 FROZEN.** Sampling success means outputting a finite string
law within \(2^{-p}\) TV of the declared target law \(\tau_{x,p}\), taking the exact
transient law as the target when no separate task target is declared. Decision
success means that the target law
has a declared \(2/3\) versus \(1/3\) acceptance gap and that the approximation
budget is small enough to preserve bounded error. No exact continuous sample,
infinite output stream, equilibrium oracle, or asymptotic-only limit is a delivered
answer.

### 1.2 Charged resource vector

**DEFINITION — GATE-0 FROZEN.** The common charged vector is

\[
\mathbf C=(D,C_g,C_v,P,W,\rho,B,M,H,J,\kappa,I,O),
\]

with the following coordinates:

| Coordinate | Charged content |
|---|---|
| \(D\) | finite graph/state/kernel/rate/readout descriptions and their encodings |
| \(C_g\) | uniform constructor and certificate-generation work |
| \(C_v\) | certificate verification work |
| \(P\) | preparation work, bits, and randomness |
| \(W\) | ordinary transition/event-generation and sampler work |
| \(\rho\) | all fair random bits consumed by bounded-time samplers |
| \(B\) | coefficient, probability, clock, arithmetic, and error-budget precision bits |
| \(M\) | encoded state and peak ancillary memory |
| \(H\) | discrete horizon or finite truncation/event budget |
| \(J\) | CTMJ total uniformization-rate–time product; zero/not applicable for D-LSN |
| \(\kappa\) | generated numerical transition-perturbation budget and its verification; physical calibration only if a finite procedure is separately admitted |
| \(I\) | finite readout work, communication, and confidence handling |
| \(O\) | delivered output bits |

**DEFINITION — GATE-0 FROZEN.** A family has polynomial FSTT cost only when every
coordinate is bounded by one polynomial in \(n+p\), the constructor is uniform, and
every sampler halts within its declared polynomial worst-case work and random-bit
budget. Expected-finite rejection loops, arbitrary real constants, nonuniform
per-instance certificates, and an uncharged physical calibration premise do not
qualify.

### 1.3 Composition core and attempted counterexamples

**PROVED BY REVIEW.** Markov contraction and telescoping give

\[
\operatorname{TV}(PK_1\cdots K_TR,P'K'_1\cdots K'_TR')
\leq\epsilon_{\rm prep}+\sum_{t=1}^T\epsilon_t+
\epsilon_{\rm read}.
\]

The proof does not require a global trajectory-sensitivity schedule. A stochastic
readout is another Markov kernel and cannot increase TV.

**CANDIDATE DERIVATION — UNREVIEWED.** When a separate task target \(\tau\) is
declared, the triangle inequality adds the generated
\(\epsilon_{\rm target}\) term. For CTMJ CAN-2 this term is exactly the
finite-burn-in mixing error \(\epsilon_{\rm mix}\); Poisson truncation, local sampler,
numerical perturbation, and readout errors remain on the implementation side of the
displayed telescope.

**HEURISTIC — COUNTEREXAMPLE PRESSURE.** Five superficially broader statements fail
before source search:

1. allowing arbitrary real transition probabilities lets one probability encode
   nonuniform advice;
2. allowing an unbounded or only almost-sure horizon makes the finite answer depend
   on an uncertified limit;
3. accepting a terminal-law or mixing-time promise lets the certificate contain the
   hard behavior;
4. allowing an exponentially large output string hides work at readout; and
5. charging only physical time while total CTMJ rate diverges lets infinitely many
   event opportunities enter one clock interval.

The frozen registration excludes all five. Excluding them makes the simulator
honest; it also makes the later ceiling close to an unfolding of admission.

---

## 2. Iteration 2 — primary-source novelty closure

### 2.1 Recheck of the ten-source local closure

**EMPIRICAL SOURCE FINDING — BOUNDED.** The ten full texts indexed in
`research/sources/INDEX.md` were re-searched for preparation, readout, finite-horizon
stochastic output, transition kernels, samplers, uniform families, constructors, and
TV. The direct relevant results remain exactly those isolated by Review 1:

| Source line | What is established | Why it does not occupy the full conjunction |
|---|---|---|
| Bournez–Graça–Pouly | **PROVED (literature):** deterministic ODE representation, initialization, robust evolution, output margins, and explicit numerical cost | deterministic and ODE-specific; no finite stochastic terminal law or CTMJ/D-LSN pair |
| Blanc–Bournez papers | **PROVED (literature):** model-specific robustness, reachability/evaluation, and time/length/precision complexity | no preparation-to-finite-stochastic-output compiler across substrates |
| Braverman–Grigo–Rojas | **PROVED (literature):** invariant-measure computation under explicit analytic regularity and mixing assumptions; Theorem 36 gives a quantitative truncated-operator approximation in its narrow regime | invariant law on its represented dynamics, not finite-horizon sampling, generated preparation/readout, or two constructors |
| Bournez–Pouly survey | **PROVED (literature summary):** hidden precision, time contraction, noise, representation, and physical-cost pathologies are established concerns | survey/audit coverage, not the registered theorem |

**EMPIRICAL SOURCE FINDING — REVIEW-INHERITED.** No one of the ten local primary
texts contains the exact fourfold conjunction of preparation, finite stochastic
output, a uniform constructor, and noncircular composition across more than one
substrate. This preserves an absence in a frozen source set; it is not
literature-wide novelty.

### 2.2 Fresh nearest-line search

**EMPIRICAL SOURCE FINDING — BOUNDED WEB CLOSURE.** Fresh searches were run over the
nearest finite-Markov, stochastic-cellular-automaton, probabilistic-program,
uniformization, Glauber-mixing, and stochastic-chemical-network lines. The following
primary results were located:

| Primary source | Result used at Gate 0 | Scope consequence |
|---|---|---|
| Abate, Redig, Tkachev, [“On the effect of perturbation of conditional probabilities in total variation” (2014)](https://doi.org/10.1016/j.spl.2014.01.009) | **PROVED (literature):** studies how per-kernel TV perturbations affect the product law generated by a sequence of conditional kernels, including linear finite-horizon bounds | independently occupies the common TV-product core; no constructor pair |
| Arrighi, Schabanel, Theyssier, [“Stochastic Cellular Automata: Correlations, Decidability and Simulations” (2013)](https://arxiv.org/abs/1304.7185) | **PROVED (literature):** gives a unified composable stochastic-CA formalism and intrinsic simulation tools | occupies a richer D-LSN neighboring syntax, not CTMJ preparation/readout |
| Sidje, Burrage, MacNamara, [“Inexact Uniformization Method for Computing Transient Distributions of Markov Chains” (2007)](https://doi.org/10.1137/060662629) | **PROVED (literature):** uniformization is a standard stable CTMC transient-distribution algorithm, with approximate matrix steps analyzed | occupies the CTMJ-to-DTMC bridge, not the cross-constructor wrapper |
| Gillespie, [“Exact stochastic simulation of coupled chemical reactions” (1977)](https://doi.org/10.1021/j100540a008) | **PROVED (literature):** exact event-by-event simulation for stochastic chemical reaction networks | occupies a CTMJ event-sampling route |
| Soloveichik, [“Robust Stochastic Chemical Reaction Networks and Bounded Tau-Leaping” (2009)](https://arxiv.org/abs/0803.1030) | **EMPIRICAL SOURCE FINDING — PRIMARY ABSTRACT:** reports a relation between robustness of CTMJ/CRN state probabilities, efficient approximate simulation, and explicit simulated time/concentration dependence | indicates that CTMJ robustness and simulation cost already meet in one substrate; exact theorem scope remains an unresolved full-text obligation below |
| Dyer, Goldberg, Jerrum, [“Dobrushin conditions and systematic scan” (2005)](https://eccc.weizmann.ac.il/report/2005/075/) | **PROVED (literature):** Dobrushin influence conditions imply rapid mixing for Glauber updates in the stated finite-spin regimes | supplies a locally checkable high-temperature certificate line |
| Chen, Feng, Yin, Zhang, [“Optimal Mixing Time for the Ising Model in the Uniqueness Regime” (2021)](https://arxiv.org/abs/2111.03034) | **PROVED (literature):** proves \(O(n\log n)\)-scale Glauber mixing in its stated uniqueness interior and a sampling algorithm | occupies the useful high-temperature CAN core |
| Sharma, Roy, [“Approximations in Probabilistic Programs” (2019)](https://arxiv.org/abs/1912.06791) | **PROVED (literature):** compiles stationary Markov-kernel constructs into approximate finite probabilistic programs under uniform ergodicity assumptions | occupies another preparation/finite-unrolling neighboring line; not the exact two-constructor theorem |
| Petrack, Doty, [“Exactly simulating stochastic chemical reaction networks in sub-constant time per reaction” (2025)](https://arxiv.org/abs/2508.04079) | **EMPIRICAL SOURCE FINDING — PRIMARY ABSTRACT:** states that, under its assumptions, exact stochastic CRN simulation can use sublinear time in the number of reactions | weakens any unsupported claim that total reaction count is a generally necessary digital cost; no theorem detail from the abstract is imported |

**EMPIRICAL SOURCE FINDING — SEARCH RESULT.** No located primary source states the
entire preparation + finite output + uniform charged constructor + TV composition
theorem for both D-LSN and CTMJ at materially the registered scope. Therefore
`OCCUPIED-CONJUNCTION` is not issued. This non-finding is not clearance: the search
was bounded, and the strict-delta gate is independently fatal.

### 2.3 Explicit unresolved source obligations

**SOURCE-INCOMPLETE — EMPIRICAL.** The following obligations remain unresolved and
are not inferred in the candidate's favor:

1. the direct Caltech full-text endpoint for the 2009 Soloveichik paper was
   unreachable during this batch; only the primary repository record and arXiv
   abstract were used for its scope;
2. no exhaustive closure of stochastic Petri nets, probabilistic process algebra,
   succinct CTMC complexity, stochastic circuits, or probabilistic-program compiler
   literature was possible;
3. no primary source for the standard uniform randomized-circuit characterization of
   BPP was needed for the kill and none is claimed as newly checked; and
4. exact absence of the full conjunction remains an open novelty obligation, not a
   pass token.

### 2.4 Strict-delta decomposition after the search

**HEURISTIC — FATAL DECOMPOSITION.** The source closure partitions every proposed
proof step as follows:

\[
\boxed{\text{D-LSN sampler unrolling}}
+\boxed{\text{CTMJ uniformization/event sampling}}
+\boxed{\text{known high-temperature mixing}}
+\boxed{\text{TV kernel telescope}}
+\boxed{\text{endpoint ledger}}.
\]

No remaining lemma relates the two substrates in a way not already accomplished by
turning each into sampled Markov kernels. This is the exact issue the
`TELESCOPE-ONLY` token was written to catch.

---

## 3. Iteration 3 — D-LSN constructor

### 3.1 Exact class

**DEFINITION — GATE-0 FROZEN.** A D-LSN instance consists of a uniformly generated
finite graph \(G=(V,E)\), polynomially many \(q\)-bit cells, a polynomial list of
clocked sublayers, and a polynomial list of bounded-footprint local update samplers.
For one update \(u\), a finite Boolean circuit

\[
S_u:\{0,1\}^{q|R_u|}\times\{0,1\}^{\rho_u}
\longrightarrow\{0,1\}^{q|W_u|}
\]

induces its local Markov kernel. Footprint conflicts are serialized by a supplied or
uniformly generated coloring exactly as in `research/BLSD.md`; disjoint footprints
execute in parallel. Preparation and readout are finite randomized circuits of the
same kind. Every route, copied bit, color, random bit, ancilla, and output buffer is
charged.

**DEFINITION — GATE-0 FROZEN.** The primary D-LSN syntax uses fair random inputs, so
its induced transition probabilities are dyadic and exact. A rational local kernel
without a bounded exact fair-bit sampler is admitted only through a generated dyadic
approximation and an explicit conditional TV bound. Unbounded rejection sampling is
not silently treated as one step.

### 3.2 Uniform compiler and cost

**CANDIDATE DERIVATION — UNREVIEWED.** The compiler concatenates the preparation
circuit, scheduled local sampler circuits, and readout circuit. Its state is the
encoded cell array plus declared ancillas; its horizon is the number of generated
sublayers; its work/size/randomness is the corresponding sum (or parallel maximum
for depth); and exact local samplers contribute zero kernel error. Approximate local
samplers contribute their generated conditional TV errors, which add by the reviewed
telescope. This is the BLSD-D candidate proof specialized to the common FSTT
semantics.

**CANDIDATE DERIVATION — UNREVIEWED.** A uniform polynomial-size probabilistic
circuit embeds by laying out its wires/gates in the D-LSN schedule, placing input and
zero ancillas in preparation, treating every random gate as a fair-bit sampler, and
reading its output bit. Routing and any geometry overhead remain charged. Thus the
registered BPP CAN route is available without advice.

### 3.3 Gate pressure

**HEURISTIC.** No preparation, mixing, calibration, or readout oracle is present:
the finite circuits are the syntax. No infinity leak is present when the generated
schedule is polynomial. The constructor is nevertheless an explicit probabilistic
circuit unrolling. Its mathematical core is the already frozen
`OCCUPIED-CORE / UNESTABLISHED WRAPPER` result in `research/BLSD.md`.

**HEURISTIC — `NO-USEFUL-CAN` PRESSURE.** D-LSN CAN-1 alone would be exactly a
probabilistic circuit simulating a probabilistic circuit and would not justify a
theorem unit. The token is not issued at this iteration only because the CTMJ
constructor still owes a useful continuous-time finite-output family.

---

## 4. Iteration 4 — CTMJ constructor

### 4.1 Exact rational finite-state class

**DEFINITION — GATE-0 FROZEN.** A CTMJ instance is a uniformly generated bounded-
degree graph \(G=(V,E)\), \(|V|=N\), with spin state
\(S=\{-1,+1\}^{V}\). Positive unary and pair factors are finitely encoded rationals:

\[
w(\sigma)=\prod_{v\in V}\phi_v(\sigma_v)
\prod_{\{u,v\}\in E}\phi_{uv}(\sigma_u,\sigma_v),
\qquad
\pi(\sigma)=\frac{w(\sigma)}{\sum_\tau w(\tau)}.
\]

At rate one per site, the site's clock rings and its spin is resampled from the exact
rational conditional \(\pi(\sigma_v\mid\sigma_{V\setminus\{v\}})\). The resulting
time-homogeneous generator \(Q\) has rational, locally evaluable off-diagonal rates,
total dominating rate \(\Lambda=N\), and finite encoded state. The declared horizon
\(t=t(x,p)\) is rational. The preparation is the deterministic all-\(+1\) state and
the readout returns the original \(N\) spins as a finite string.

**CANDIDATE DERIVATION — UNREVIEWED.** The local heat-bath form gives detailed
balance with the rational factor-graph Gibbs law by direct cancellation of the
single-site conditional factors. This derives the target stationary law from the
constructor syntax rather than supplying terminal-law correctness.

### 4.2 Generated Dobrushin/mixing certificate

**DEFINITION — GATE-0 FROZEN.** For adjacent \(u,v\), the compiler enumerates the
constant-size neighborhood of \(v\) and computes the exact rational influence

\[
c_{vu}=\max_{\eta,\eta':\eta\equiv\eta'\text{ off }u}
\left|\pi(\sigma_v=+1\mid\eta)-
\pi(\sigma_v=+1\mid\eta')\right|.
\]

It then computes

\[
\alpha=\max_v\sum_{u\in N(v)}c_{vu}
\]

and admits the high-temperature CAN family only when the generated rational witness
satisfies \(\alpha\leq1-\gamma\), where \(\gamma=1-\alpha\), and for one fixed class
constant \(c_0\), \(\gamma\geq(n+p)^{-c_0}\). Bounded degree makes enumeration and
exact rational comparison polynomial in the input bit length. Fixing \(c_0\) prevents
“some polynomial” from becoming instance-dependent advice.

**PROVED (literature, scoped).** Dobrushin influence conditions yield rapid mixing
for the stated finite-spin Glauber regimes, and stronger modern uniqueness-interior
results give \(O(N\log N)\)-scale mixing in their stated parameter ranges. The Gate-0
constructor uses only the simpler generated Dobrushin witness. It does not ask a user
to assert a spectral gap or mixing time.

**CANDIDATE DERIVATION — UNREVIEWED.** Standard path-coupling/Dobrushin contraction
then gives a generated burn-in of polynomial order

\[
t_{\rm mix}=O\!\left(\gamma^{-1}
(\log N+\log(1/\epsilon_{\rm mix}))\right)
\]

for rate-one clocks per site, uniformly from the deterministic preparation. The
exact constant is not needed for the Gate-0 death; a theorem branch would have had
to freeze and review it.

### 4.3 Uniformization, Poisson truncation, and finite sampling

**CANDIDATE DERIVATION — UNREVIEWED.** Uniformization writes the time-\(t\) kernel as

\[
e^{tQ}=e^{-\mu}\sum_{k=0}^{\infty}\frac{\mu^k}{k!}P^k,
\qquad
P=I+Q/\Lambda,
\qquad
\mu=\Lambda t=Nt.
\]

In the common syntax of §1.1, this constructor has \(T=1\) and ideal finite-state
kernel \(K_1=e^{tQ}\), succinctly described by the rational pair \((Q,t)\). The
truncated-uniformization procedure below is the charged bounded-time sampler
\(K'_1\); \(k_*\) is its internal event budget, not a replacement ideal dynamics or
an uncharged variable horizon.

For

\[
k_*=\left\lceil\max\left\{2e\mu,
\log_2(2/\epsilon_{\rm Pois})\right\}\right\rceil,
\]

the standard Poisson Chernoff bound
\(\Pr[Z\geq k]\leq(e\mu/k)^k\) makes the discarded tail at most
\(\epsilon_{\rm Pois}/2\). Because \(\mu\) is rational, the finite conditional weights
\(\mu^k/k!\) for \(0\leq k<k_*\) are rational with polynomial total bit length when
\(\mu\) and \(k_*\) are polynomial. The compiler normalizes this finite list,
approximates the resulting categorical law dyadically, and samples it with a bounded
number of fair bits. This derives the event-count sampler and its TV error without an
exact exponential or arbitrary-real clock.

**CANDIDATE DERIVATION — UNREVIEWED.** Conditional on the sampled event count, each
uniformized step chooses a site and performs the rational local heat-bath update.
The site choice and Bernoulli probability are implemented by bounded dyadic
approximations. Allocating per-event error at most
\(\epsilon_{\rm step}/k_*\) gives total transition error at most
\(\epsilon_{\rm step}\) by the same TV telescope. The work, random bits, and
precision are polynomial in \(N,t,p\), the factor bit lengths, and
\(\gamma^{-1}\).

### 4.4 Calibration/perturbation and finite readout

**CANDIDATE DERIVATION — UNREVIEWED.** If the generated finite implementation changes
one uniformized conditional kernel by at most \(\zeta\) in worst-case TV, then the
truncated execution changes by at most \(k_*\zeta\). Thus the compiler can select
finite rate/probability precision from
\(\zeta\leq\epsilon_{\rm cal}/k_*\), compute the dyadic local tables, and charge their
generation and verification. Identity readout on the original spin sites has zero
additional semantic error; an approximate readout would receive its own generated
TV budget.

Here \(\epsilon_{\rm cal}\) and the \(\kappa\) coordinate mean numerical transition-
approximation error in the abstract sampler. They are not evidence that a fabricated
device realizes \(Q\).

**HEURISTIC — PHYSICAL ADMISSION LIMIT.** This is an honest abstract CTMJ
constructor. It does not prove that fabricated hardware realizes the generated rate
table or lies inside a claimed local calibration interval. A physical FSTT admission
would need a finite calibration procedure and an explicit measurement/noise model;
supplying rate fidelity as a promise would trigger `CONSTRUCTOR-ORACLE`. No physical
efficiency or hardware-fidelity claim is made in this unit, so that unresolved
empirical obligation is excluded rather than laundered into clearance.

### 4.5 Why the two constructors are genuinely non-equivalent

**CANDIDATE DERIVATION — UNREVIEWED.** The exact classes are not mere renamings:

1. a bounded fair-bit D-LSN execution has a dyadic output law, whereas the one-spin
   member of the registered heat-bath class with equal positive unary factors,
   rate-one resampling, deterministic initial spin \(+1\), and horizon \(t=1\) has
   \(\Pr[X_1=-1]=(1-e^{-1})/2\), which is not dyadic; and
2. a finite-time time-homogeneous CTMJ transition matrix \(e^{tQ}\) is invertible and
   has positive determinant \(e^{t\operatorname{tr}Q}\), whereas a D-LSN may execute
   a singular reset or a two-state swap with determinant \(-1\).

Thus neither exact constructor class contains the other under identity state and one
finite exact execution.

**CANDIDATE DERIVATION — UNREVIEWED, SAME-TASK WITNESS.** On any admitted rational
Dobrushin factor graph, the CTMJ constructor evolves the heat-bath chain and reads
the spin string. The D-LSN constructor can lay out the generated finite
uniformization sampler—event-count circuit, site choices, local heat-bath circuits,
and identity readout—as a clocked local stochastic network. With the same generated
\(\epsilon_{\rm mix}\) and implementation budget, both therefore deliver a finite
spin-string law within \(2^{-p}\) of the same Gibbs target \(\pi\). The common task is
not inferred from the unrelated generic-BPP example.

**HEURISTIC — WHY THIS DOES NOT SAVE THE THEOREM.** FSTT asks only for approximate
finite-output simulation. Uniformization converts the CTMJ into a Poisson mixture of
ordinary discrete sampled kernels, and truncation makes that mixture finite. Exact
class non-equivalence is therefore real while the shared simulator remains the
standard discrete-kernel route. `FAKE-SECOND-CONSTRUCTOR` need not fire for
`TELESCOPE-ONLY` to fire.

### 4.6 Certification-dilemma adjudication

**HEURISTIC — NARROW PASS.** The formal CTMJ family avoids the certification oracle:

- the preparation is generated, not promised;
- the Dobrushin influence matrix and gap \(\gamma\) are computed from bounded-degree
  rational local factors;
- stationarity follows from the generated heat-bath syntax rather than a supplied
  target sample;
- the mixing horizon follows from the generated Dobrushin witness;
- Poisson truncation and finite precision follow from \(N,t,p\); and
- readout is finite and explicit.

The certificate excludes hard low-temperature behavior, but it does not encode the
answer or solve a hidden instance problem. This is a legitimate easy-regime
certificate, not evidence of a new boundary. For arbitrary CTMJs, global mixing or
hardware fidelity would return the certification dilemma and are not admitted.

---

## 5. Iteration 5 — Gate-0 evidence gate

### 5.1 Exact-conjunction occupation check

| Required conjunct | Gate-0 status |
|---|---|
| preparation kernels | **HEURISTIC — CONSTRUCTIBLE:** finite circuits for D-LSN; deterministic all-plus for CTMJ |
| finite stochastic output | **HEURISTIC — CONSTRUCTIBLE:** finite readout circuits / spin string |
| uniform family constructor | **CANDIDATE DERIVATION — UNREVIEWED:** explicit finite compilers above |
| noncircular TV composition | **PROVED BY REVIEW; ALSO OCCUPIED IN NEIGHBORING LITERATURE** |
| two exact non-equivalent constructors | **CANDIDATE DERIVATION — UNREVIEWED:** dyadic/determinant separation above |
| literature-wide strict absence | **SOURCE-INCOMPLETE:** no exact occupation found in the bounded closure; no clearance inferred |
| strict theorem delta | **HEURISTIC — FAIL:** only known constructor results plus telescope/ledger remain |

### 5.2 Infinity-leak and admission audit

| Leak channel | D-LSN | CTMJ | Verdict |
|---|---|---|---|
| arbitrary reals | fair-bit circuits/dyadic tables | rational factors/rates/time; exponential clock removed by finite approximation | **HEURISTIC — CLOSED** |
| state/output size | polynomial encoded state and finite output | \(N\) spins and \(N\)-bit output | **HEURISTIC — CLOSED** |
| horizon | polynomial schedule | generated \(t\), \(\mu=Nt\), and \(k_*\) polynomial | **HEURISTIC — CLOSED** |
| random bits | every circuit coin charged | count, event mark, and local coins charged | **HEURISTIC — CLOSED** |
| coefficient/transition precision | finite sampler circuits/tables | rational inputs and generated dyadic precision | **HEURISTIC — CLOSED** |
| preparation | generated circuit | fixed generated state | **HEURISTIC — CLOSED** |
| mixing/terminal law | no mixing premise for finite horizon | Dobrushin certificate and detailed-balance syntax generated | **HEURISTIC — CLOSED IN THE NARROW ABSTRACT FAMILY** |
| certificate/advice | constructor emits all syntax | bounded-degree certificate generated | **HEURISTIC — CLOSED** |
| calibration | finite table is syntax | finite implemented rates and perturbation budget generated | **HEURISTIC — CLOSED ABSTRACTLY; PHYSICAL FIDELITY OUTSIDE CLAIM** |
| readout/margin | finite circuit and gap charged | finite spin identity; decision margin separately required | **HEURISTIC — CLOSED** |
| failure probability | TV budget and decision gap charged | Poisson, step, mixing, calibration, readout budgets charged | **HEURISTIC — CLOSED** |

**HEURISTIC — INFINITY TOKEN.** `INFINITY-LEAK` does not fire for the two narrow
abstract constructors. It would fire if physical rate fidelity, arbitrary mixing, or
an exact exponential clock were reintroduced as free promises.

### 5.3 Symmetric closure

**HEURISTIC — BINDING.** The digital comparator receives exactly the generated
finite graph, factors/rates, certificate, preparation, error allocation, horizon,
and readout. It may run the same local circuits or the same uniformized heat-bath
steps. No physical-only advice, warm start, target sample, or implicit state is
withheld. Under this symmetric closure:

- D-LSN is already a finite randomized circuit execution; and
- CTMJ is sampled by the generated finite uniformization program.

The closure produces a correct ceiling, but it also exposes why the ceiling is not a
new impossibility theorem.

### 5.4 Strict-delta and CAN/CANNOT asymmetry attacks

**HEURISTIC — `TELESCOPE-ONLY`, COMPLETE ARGUMENT.** After importing the D-LSN
unrolling, standard CTMJ uniformization/event simulation, generated high-temperature
mixing, and finite local samplers, the common proof does not use a new cross-
substrate invariant or boundary lemma. It concatenates preparation, transitions,
and readout and applies the already-proved additive TV bound. The uniform constructor
and cost vector expose hidden work but do not derive a new mathematical restriction.
This satisfies the frozen death token literally.

**HEURISTIC — `NO-LOAD-BEARING-CANNOT`, INDEPENDENT COMPLETE ARGUMENT.** The claimed
SampBPP/BPP CANNOT assumes a uniform polynomial-time constructor that outputs
bounded-time finite samplers, a polynomial state encoding, and a polynomial
transition/event budget. A randomized Turing machine belongs to SampBPP/BPP simply by
running those samplers. Removing a premise does not yield a matched hard family:

- nonuniform certificates can encode advice but need not be necessary for every easy
  family;
- superpolynomial mixing can make this physical chain slow without proving the task
  is outside SampBPP;
- a superpolynomial event count is not a universal digital lower bound, and exact CRN
  simulation can sometimes batch events; and
- finite state/output alone says nothing about efficient sample generation.

No premise is proved necessary or tight for a natural task class. The CANNOT is an
admission ceiling, not a load-bearing impossibility theorem.

**HEURISTIC — CAN ATTACK.** CAN-1 is the standard probabilistic-circuit embedding.
CAN-2 is useful finite high-temperature Gibbs sampling, but the same generated
Dobrushin/uniqueness certificate already yields a polynomial classical Glauber
sampler. The two CANs establish nonemptiness and a useful example; they do not create
an asymmetric discovery. `NO-USEFUL-CAN` is not issued because finite spin samples
are useful. The stronger `MISSION-KILL` still fires because the useful family lies
entirely in a known digitally easy regime and no physical resource separation is
even targeted.

### 5.5 Token-by-token adjudication

| Token | Fired? | Reason |
|---|---:|---|
| `OCCUPIED-CONJUNCTION` | **NO CLAIM** | no exact full-conjunction source found in the bounded closure; absence remains unresolved |
| `CONSTRUCTOR-ORACLE` | **NO, NARROW ABSTRACT CLASSES** | both compilers generate their certificates; physical fidelity is explicitly outside the claim |
| `FAKE-SECOND-CONSTRUCTOR` | **NO AT EXACT-CLASS LEVEL** | exact-law and determinant witnesses separate the classes, though uniformization absorbs CTMJ for approximate finite output |
| `TELESCOPE-ONLY` | **YES — PRIMARY BINDING DEATH** | the cross-substrate proof reduces to known constructors, uniformization, TV telescoping, and endpoint accounting |
| `NO-LOAD-BEARING-CANNOT` | **YES — CORROBORATING DEATH** | the ceiling follows by executing admitted samplers; no natural necessity/tightness theorem remains |
| `NO-USEFUL-CAN` | **NO** | high-temperature finite spin sampling is useful, though already digitally easy |
| `INFINITY-LEAK` | **NO IN THE FROZEN ABSTRACT FAMILY** | all state, time, rate, precision, randomness, certificates, and output are finite and charged |
| `MISSION-KILL` | **YES — CORROBORATING DEATH** | the surviving result is an audit/interface theorem with no new useful capability boundary |
| `VOID` | **NO** | on the same admitted rational Dobrushin factor graph, CTMJ evolution and a D-LSN implementation of its generated uniformization sampler both approximate the same finite Gibbs spin-string target under one error/resource contract |

**HEURISTIC — EXACT GATE-0 RETURN: `TELESCOPE-ONLY`.** The one-strike rule ends
theorem work. `FSTT-CLEAR-TO-PROVE` is withheld. The two corroborating tokens do not
change the exact primary return.

---

## 6. Iterations 6–7 — registered death branch

### 6.1 Frozen theorem targets and dispositions

| Frozen target | Disposition after Gate 0 |
|---|---|
| D-LSN BPP CAN | **HEURISTIC — NO THEOREM WORK:** standard embedding/occupied core |
| CTMJ high-temperature Gibbs CAN | **HEURISTIC — NO THEOREM WORK:** constructor route is non-oracular but imported from known Glauber mixing plus uniformization |
| FSTT SampBPP/BPP CANNOT | **HEURISTIC — KILLED:** definitional admission ceiling; no load-bearing premise |
| robustness/composition lemma | **PROVED BY REVIEW only for the elementary TV telescope:** no broader worker theorem promoted |
| paired theorem | **HEURISTIC — NOT OPENED:** primary kill precludes proof and proof-review packet |

**HEURISTIC — NO RESCUE.** Adding a third constructor, a physical efficiency
coordinate, a different output task, a new certificate theory, or continuous-state
Langevin dynamics would change the frozen registration and trigger the round's
mission kill. None was added.

### 6.2 Complete falsification record

| Candidate claim | What survived | What failed | Scoped lesson |
|---|---|---|---|
| one FSTT theorem creates a strict cross-substrate result | both narrow constructors can be finite, uniform, non-oracular, and exactly non-equivalent | their shared finite-output proof has no content beyond uniformization, known local samplers/mixing, TV telescoping, and accounting | exact constructor diversity does not imply a new common theorem |
| polynomial FSTT cost gives a meaningful classical ceiling | executing the generated sampler does put the family in SampBPP/BPP | the inclusion is inserted by the admission rule and no premise is necessary/tight | a correct ceiling can still be definition-level |
| high-temperature CTMJ supplies an asymmetric CAN | a generated Dobrushin certificate yields a useful finite spin sample | the same certificate gives an ordinary classical sampler; no physical-over-digital frontier appears | non-oracular easy-regime certification is not novelty |
| a full end-to-end ledger rescues the conjunction | hidden preparation, randomness, precision, calibration, and readout are exposed | exposing them changes audit quality, not computational capability | retain as floor discipline only |

---

## 7. Iteration 8 — robustness and necessity attack

### 7.1 Representation changes

**CANDIDATE DERIVATION — UNREVIEWED.** A uniformly computable polynomial-overhead
bijection of finite state encodings conjugates every preparation, transition, and
readout kernel. TV is invariant under bijections, and encoding/decoding circuits add
only their charged work. This preserves the ceiling and the death; it creates no new
boundary.

**HEURISTIC.** An exponentially expanding or nonuniform encoding is not an allowed
representation change. Charging it is exactly the permanent admission law, not an
FSTT theorem delta.

### 7.2 Kernel grouping, readout factorization, and error allocation

**CANDIDATE DERIVATION — UNREVIEWED.** Grouping several finite kernels into one,
splitting one kernel into generated subkernels, or factorizing the readout only
changes where the reviewed telescope is applied. If component sampler cost is
polynomial in requested precision, reallocating a total \(2^{-p}\) budget across a
polynomial number of components changes precision by at most polynomial/logarithmic
overhead. None of these perturbations supplies a strict delta.

### 7.3 Clock/rate representation

**CANDIDATE DERIVATION — UNREVIEWED.** Multiplying all CTMJ rates by \(a>0\) and
dividing physical horizon by \(a\) leaves the uniformization product
\(J=\Lambda t\) unchanged. This blocks clock-speed laundering. It also confirms that
the load-bearing simulator coordinate is the generated finite event/kernel work,
which is already charged by definition.

### 7.4 Necessity failure

**HEURISTIC.** No converse was found or plausibly derived:

- a family can have a poor Dobrushin certificate yet be easy by cluster, tempering,
  direct, or problem-specific sampling;
- a CTMJ can have many reaction events yet admit batched/aggregated simulation in a
  structured regime;
- a high mixing time for one local dynamics does not lower-bound all algorithms for
  its output law; and
- a nonuniform or hard-to-generate certificate is an admission failure, not proof
  that the finite output is classically hard.

Therefore the CANNOT has no natural iff, tightness theorem, or named necessity
witness. Robustness checks strengthen the kill rather than repair it.

---

## 8. Iteration 9 — independent review gate

**HEURISTIC — REVIEW TARGET.** Because Gate 0 killed the theorem unit, there is no
worker proof eligible for a proof-review token. The independent gate instead asks a
fresh zero-context Codex reviewer to attack the death packet: find an unimported
strict theorem obligation, a false non-equivalence claim, a hidden constructor
oracle, a source-scope overclaim, or a token misapplication. This is an evidence
review only and cannot promote any mathematical statement to `PROVED`.

**EMPIRICAL — REVIEW RESULT: `PASS`.** A fresh zero-context independent Codex
reviewer read the frozen registration and the complete relevant repository record,
made no edits, and returned no material finding. It upheld `TELESCOPE-ONLY`,
`NO-LOAD-BEARING-CANNOT`, `MISSION-KILL`, and the scoped `NEITHER-AGAIN` consequence.
This was an evidence review, not a proof review, and promotes no statement to
`PROVED`.

**EMPIRICAL — MINOR REPAIRS APPLIED.** The reviewer required six local corrections:

1. replace the out-of-class absorbing-chain non-equivalence witness with the one-spin
   positive-factor heat-bath witness now in §4.5;
2. exhibit both constructors on the same Dobrushin Gibbs-sampling task;
3. include \(\epsilon_{\rm mix}\) in the full target-error allocation;
4. clarify that one meta-constructor interface has two uniform substrate branches;
5. distinguish abstract numerical transition error from fabricated-hardware
   calibration; and
6. update the canonical frontier and worklog from registered to completed status.

Items 1–5 are repaired in this document; item 6 is repaired in `research/PROGRAM.md`
and `WORKLOG.md`. An initial attempt to invoke a separate read-only `codex exec`
process was blocked before analysis by the worker sandbox's outbound transport. It
is not counted as the review; the successful separate zero-context Codex review is.

---

## 9. Iteration 10 — batch evidence, mission gate, and endgame packet

### 9.1 Batch verdict

**HEURISTIC — BINDING.** FSTT-2C returns `TELESCOPE-ONLY`; no positive theorem token
issues. The paired proof-review token `FSTT-PAIR-REVIEWED` is inapplicable. The
Turing-1936-scale target remains unachieved.

### 9.2 `NEITHER-AGAIN` failure synthesis

**HEURISTIC — BINDING CONSEQUENCE FROM ROUND 3.** `NEITHER-AGAIN` issues for the
current stochastic/physical search program.

1. **Failure pattern.** Single-dial candidates were symmetrically absorbed; task
   cards were occupied or directly substituted; the conjunction-first successor
   assembled genuine finite constructors but left only known simulator pieces plus
   an interface.
2. **Constraint on future work.** Constructor diversity, a larger cost vector, or a
   generated easy-regime certificate is insufficient. A future reopening needs a
   genuinely external mathematical object that yields a load-bearing CAN/CANNOT
   boundary not created by admission.
3. **What is not proved.** This is not a no-go theorem against new classical models,
   physical acceleration, or a Turing-scale result. It is a gate-level exhaustion of
   the present search program.
4. **What stops.** Do not generate another physical-resource, substrate, or task card
   in this domain absent an external object or explicit direction from Devansh.

### 9.3 Supervisor/endgame consequence

**HEURISTIC — ESCALATION PACKET.** Report to the supervisor:

- FSTT-2C died primarily to `TELESCOPE-ONLY`, independently reinforced by
  `NO-LOAD-BEARING-CANNOT` and `MISSION-KILL`;
- the abstract CTMJ constructor did not die to the certification dilemma—its local
  Dobrushin certificate can be generated—but that success only certifies a known
  digitally easy regime;
- the current stochastic/physical search returns `NEITHER-AGAIN`;
- the founding mission is unachieved and judged unavailable by the present search,
  not disproved; and
- the strongest concrete theorem surfaced by the record is the **proposed
  approximate coherent-DFT lower bound**: a same-task lower bound for approximate
  complex coherent DFT with named terminals, explicit norm/error/failure, finite
  precision, phase/output quotient, and matched layout/I/O assumptions.

**HEURISTIC — AUTHORITY LIMIT.** The approximate coherent-DFT direction is only
`PROPOSED MISSION NARROWING`. It must not be registered or started until Devansh
accepts the narrowing. OFFT and PFPWO remain dead and cannot be rescued by that
theorem.

### 9.4 CONTINUE / ESCALATE evaluation

**HEURISTIC — `ESCALATE`.** Do not autonomously continue theorem work on FSTT-2C and
do not rotate to another stochastic/resource card. Escalate the complete kill and
proposed narrowing to the supervisor/Devansh. The only in-scope continuation without
new direction is documentation repair demanded by the independent evidence review.

### 9.5 NARRATIVE

**Honest one-sentence story — HEURISTIC.** *Even when two genuinely different noisy
machines can generate every certificate and finite sample honestly, their supposed
shared theorem collapses to an old clock-conversion trick, an old error sum, and a
bill—so this search still has no new model of computation.*

**“Isn't that obvious?” test — HEURISTIC.** The dismissal is strong: a finite
stochastic circuit is randomized computation, and a bounded finite-state CTMJ is
uniformized into randomized steps. The nontrivial Gate-0 work was showing that the
continuous constructor could be honest and still fail strict delta; the kill is not
an oracle shortcut.

**Same-task / `VOID` test — HEURISTIC.** The registered rational Dobrushin Gibbs
instance supplies one target \(\pi\): the CTMJ reaches it after generated burn-in,
and the D-LSN executes the finite generated uniformization sampler for that same
chain. Their exact transition classes differ, but task, target, error budget,
description, and charged sampler rights agree.

**“So what?” test — HEURISTIC.** The result prevents an attractive cross-substrate
wrapper from being mistaken for a theorem boundary and closes the present search
program cleanly. That is useful research control, not a world-changing capability.

**Mission test — HEURISTIC.** A random person would say “that is sensible,” not
“that changes everything for me.” The batch is a negative gate result and floor
improvement. It does not satisfy the mission and cannot be reframed as the moonshot.

### 9.6 Execution integrity

- **EMPIRICAL:** `research/QUESTIONS.md` was read and not edited.
- **EMPIRICAL:** no GPU was touched.
- **EMPIRICAL:** no CPU experiment or significant computational run was needed; all
  work was analytic, source-oriented, and document validation.
- **EMPIRICAL:** no Git commit, staging, or commit retry was attempted, per
  `AGENTS.md` section 4.1.
- **EMPIRICAL:** no theorem work continued after the primary Gate-0 death.
- **EMPIRICAL:** unreachable sources and bounded-search limits are explicit and were
  not treated as novelty clearance.
- **EMPIRICAL:** no worker-created claim is labelled `PROVED`.

### 9.7 Intended commit messages

1. `research(work-loop): kill FSTT-2C at strict-delta gate`
   - Body: `Define the common finite transducer and both charged abstract constructors,
     derive the non-oracular CTMJ certificate route, and issue TELESCOPE-ONLY with
     no-load-bearing-ceiling and mission deaths.`
2. `research(program): record NEITHER-AGAIN stochastic endgame`
   - Body: `Close the present resource/substrate/task-card search, preserve the
     Turing-scale mission as unachieved rather than disproved, and escalate the
     approximate coherent-DFT lower bound only as a proposed Devansh-controlled
     mission narrowing.`
