# BLSD — Bounded Local Stochastic Dynamics Floor Unit

**Work-loop batch:** 2, iterations 9–10
**Date:** 2026-07-14
**Registration:** research/DESIGN_GATE_2.md sections 4–5
**Mission status — HEURISTIC:** floor/audit component only; never the positive mission

**EMPIRICAL — EXECUTION RECORD.** This unit used CPU-free analytic work and targeted
source retrieval. No experiment or significant computation ran, the GPU was not
touched, and no Git write was attempted.

## 0. Rigor labels

- **PROVED:** used only after a complete proof has survived the fresh independent
  Codex review recorded in this file.
- **CANDIDATE PROOF — UNREVIEWED:** a complete-looking worker proof that has not yet
  passed the required independent gate.
- **EMPIRICAL:** a source or execution fact in its stated scope.
- **CONJECTURE:** a precise mathematical statement not established by a
  review-surviving proof.
- **HEURISTIC:** an interpretation, scope decision, novelty judgment, or gate token.

---

## 1. Iteration 9 — BLSD-D novelty and theorem packet

**Historical pre-review formulation.** Sections 1.1–1.4 record the original worker
packet and retain their then-current candidate labels. The later self-contained
statement in section 5 is the only BLSD-D scope promoted after
`research/PROOF_REVIEW_1.md`; no earlier or broader wording inherits that promotion.

### 1.1 Named target model and process

**DEFINITION — HEURISTIC FORMALIZATION.** The target is a **uniform bounded-fan-in
geometry-local randomized circuit family on the same graph**.

Let \(G=(V,E)\) be a uniformly constructed graph with \(|V|=n\), maximum degree
\(\Delta\), and one \(q\)-bit encoding of a symbol in the finite alphabet
\(\Sigma\) at each vertex. A block update \(u\) has:

- a read set \(R_u\), write set \(W_u\), and declared physical circuit footprint
  \(F_u\), all contained in a radius-\(r\) neighborhood of a declared center;
- \(R_u\cup W_u\subseteq F_u\) and \(|F_u|\le s\);
- an ideal finite Markov kernel \(K_u:\Sigma^{R_u}\to
  \mathcal P(\Sigma^{W_u})\); and
- a bounded-fan-in circuit sampler \(S_u\) placed within that neighborhood.

A logical layer \(t\) declares a finite block set \(\mathcal B_t\). Its conflict
graph joins \(u,v\in\mathcal B_t\) whenever \(F_u\cap F_v\ne\varnothing\).
This deliberately
serializes read/read overlap as well as data hazards so that shared-state fanout is
not free. A declared proper coloring
\(\mathcal B_{t,1},\ldots,\mathcal B_{t,\chi_t}\) fixes the semantics:
colors execute in order; within one color all samplers read the same pre-sublayer
state and write in parallel. Because conflict is full-support overlap, read/read
overlap is also serialized; it is allowed only in different color sublayers.

For every local input \(a\), sampler \(S_u(a)\) approximates \(K_u(a,\cdot)\)
in total variation by at most \(\eta_u\). Its declared resources are depth
\(d_u\), size \(z_u\), fresh random bits \(\rho_u\), and ancillary/output-
buffer bits \(a_u\), including every local route, copy, finite decoding operation,
and output buffer needed before committing the write.

The preparation channel \(P\) and readout channel \(R\) are same-graph local
randomized circuits with approximation errors \(\eta_P,\eta_R\) and declared
resource tuples \((d_P,z_P,\rho_P,a_P)\) and
\((d_R,z_R,\rho_R,a_R)\). The ideal process is

\[
\mu_{\mathrm{out}}
=\mu_0
\left(\prod_{t=1}^{T}\prod_{c=1}^{\chi_t}
       \prod_{u\in\mathcal B_{t,c}}K_u\right)R,
\]

where \(\mu_0\) is the ideal preparation law, products are in the displayed
temporal order, and same-color kernels are the parallel product on
footprint-disjoint blocks.

### 1.2 Frozen theorem

**THEOREM BLSD-D — CANDIDATE PROOF, UNREVIEWED.** Under section 1.1, the
preparation, \(T\)-layer stochastic process, and readout compile uniformly to a
bounded-fan-in geometry-local randomized circuit on the same graph with:

\[
D\le d_P+
\sum_{t=1}^{T}\sum_{c=1}^{\chi_t}
\max_{u\in\mathcal B_{t,c}}d_u+d_R,
\]

\[
Z\le z_P+\sum_{t,c}\sum_{u\in\mathcal B_{t,c}}z_u+z_R,
\qquad
\rho\le\rho_P+\sum_{t,c}\sum_u\rho_u+\rho_R,
\]

and peak encoded working state at most

\[
qn+\max\left\{a_P,a_R,
\max_{t,c}\sum_{u\in\mathcal B_{t,c}}a_u\right\}.
\]

Its terminal law \(\widehat\mu_{\rm out}\) obeys

\[
\operatorname{TV}(\mu_{\rm out},\widehat\mu_{\rm out})
\le \eta_P+\sum_{t,c}\sum_{u\in\mathcal B_{t,c}}\eta_u+\eta_R.
\]

The output remains at the same registered graph sites. There is no central sample,
global route, or second materialization phase. If only the uncolored block list is
given and each conflict graph has maximum degree \(D_t^{\rm conf}\), a uniform
greedy compiler may use \(\chi_t\le D_t^{\rm conf}+1\); the coloring work and
schedule description are charged to the uniform constructor, not hidden in circuit
depth.

### 1.3 Candidate proof

**CANDIDATE PROOF — UNREVIEWED.**

1. Replace ideal preparation by its circuit sampler. By hypothesis this changes the
   prepared law by at most \(\eta_P\).
2. For one color class, the conflict definition makes complete physical footprints
   disjoint, removing data hazards, uncharged shared-input fanout, and overlap
   between local gates or routes. Place all samplers \(S_u\) in parallel on their
   declared footprints. The sublayer depth is the maximum \(d_u\), while size,
   random bits, and simultaneous ancillas add. No information travels farther than
   the local sampler circuits already charge.
3. Couple each approximate local sampler to its ideal kernel with failure probability
   at most \(\eta_u\), conditional on the common pre-sublayer state. A union bound
   over the color class gives total-variation error at most
   \(\sum_{u\in\mathcal B_{t,c}}\eta_u\) for that parallel product. Equivalently,
   telescope one disjoint-write kernel at a time and use contraction of TV under the
   remaining kernels.
4. Compose colors and logical layers in their declared order. Markov-kernel
   contraction and telescoping add the sublayer errors; they do not multiply them.
   Sequential depths add, while the other declared circuit resources add or take the
   stated peak.
5. Apply the approximate readout. TV contraction gives the accumulated process error,
   and replacing readout by its sampler adds at most \(\eta_R\).
6. The construction is uniform because the graph, block list, schedule, local sampler
   circuits, preparation, and readout are uniformly generated. If the coloring is
   not supplied, deterministic greedy coloring of the finite conflict graph supplies
   the stated \(D_t^{\rm conf}+1\) schedule and its construction cost is recorded.

These six steps give every displayed bound.

### 1.4 Edge cases and attack surface

| Attack | Disposition |
|---|---|
| Overlapping blocks | **SEALED BY DEFINITION + CANDIDATE PROOF:** every overlap of declared circuit footprints, including read/write hazards and local routes, is serialized by the conflict coloring. |
| Same-color shared reads | **EXCLUDED BY THE CONFLICT COLORING:** read/read overlap is serialized so no free fanout is assumed. |
| Correlated randomness | **OUTSIDE THE THEOREM:** samplers use declared fresh independent bits; a shared nonlocal noise source would require its own preparation circuit and cost. |
| Approximate kernel error | **CANDIDATE PROOF:** uniform conditional TV errors telescope to the displayed sum. |
| Routing | **SEALED BY SCOPE:** each sampler's local circuit includes its route; no global wire is introduced by compilation. |
| Global stopping/readout | **OUTSIDE THE THEOREM:** preparation and readout are explicitly local finite circuits. |
| Coloring as free advice | **SEALED:** schedule is declared input or uniformly computed and charged. |
| Materialization | **SEALED:** circuit state and terminal outputs occupy the same vertices. |

**HEURISTIC — ALTERNATIVE INTERPRETATION.** The theorem can be read as a definition
unfolding: once every physical update is already supplied as a finite local sampler
circuit and a conflict schedule, composition is ordinary circuit semantics. That
interpretation weakens novelty, not correctness.

### 1.5 Novelty and occupied-core check

**EMPIRICAL — BOUNDED PRIMARY-SOURCE CHECK.** Feng, Sun, and Yin study what Gibbs
distributions can be sampled in the LOCAL model and give local algorithms and
lower bounds ([PODC 2017](https://doi.org/10.1145/3087801.3087815)). Feng and Yin
relate local distributed sampling, inference, and counting
([PODC 2018](https://doi.org/10.1145/3212734.3212757)). Fischer and Ghaffari give a
parallel local Glauber dynamics with logarithmic-round regimes
([DISC 2018](https://doi.org/10.4230/LIPIcs.DISC.2018.26)). These sources establish
that local randomized updates and their round complexity already live naturally in
distributed/local computation.

**HEURISTIC — NOVELTY VERDICT.** The exact preparation-to-readout ledger above was
not found in the bounded search, but the mathematical core is elementary unrolling
of scheduled local stochastic kernels. Absence of an exact wrapper is not novelty.
The honest label is **OCCUPIED-CORE / UNESTABLISHED WRAPPER**.

**SOURCE-INCOMPLETE — EMPIRICAL.** No exhaustive search of interacting-particle,
probabilistic cellular automata, asynchronous-circuit, or Markov-chain simulation
literature was completed. Exact follow-up queries are:

- “probabilistic cellular automata randomized circuit simulation local kernel”;
- “block dynamics conflict graph parallel randomized circuit total variation”; and
- “local Markov process preparation readout circuit compilation”.

This incompleteness can only weaken novelty. It does not block review of the explicit
finite theorem.

### 1.6 Independent-review packet

The review target is exactly sections 1.1–1.4. The reviewer must try to find:

1. a data hazard not covered by the conflict graph;
2. a false parallel-product TV step;
3. an omitted routing, schedule, random-bit, state, or materialization cost;
4. a nonuniform object hidden in the constructor; or
5. a mismatch between the process semantics and the compiled circuit.

The reviewer must separately classify proof validity and novelty. Correctness cannot
promote the occupied core to a model theorem.

---

## 2. Iteration 10 — BLSD-CT causal-event attempt

### 2.1 Precise bounded graphical model

**DEFINITION — HEURISTIC FORMALIZATION.** Let \(G=(V,E)\), \(|V|=n\), be
bounded-degree. Let \(\mathcal E\) be a uniformly generated finite set of event
types. Each \(e\in\mathcal E\):

- reads and writes a nonempty support \(S_e\subseteq V\) of size at most \(s\)
  and bounded graph radius;
- has a state-dependent rate \(q_e(x)\le\bar q_e\);
- has a finite local update kernel; and
- has finite encoded rate and update descriptions.

The dominating rates satisfy the per-site incident bound

\[
\sum_{e:v\in S_e}\bar q_e\le\lambda
\quad\text{for every }v\in V.
\]

Use independent dominating Poisson clocks of rates \(\bar q_e\), followed by local
thinning with acceptance probability \(q_e(x)/\bar q_e\). Two potential events
conflict when their supports overlap. A causal chain of length \(h\) is a
chronologically increasing sequence of potential events
\((e_1,t_1),\ldots,(e_h,t_h)\) with
\(S_{e_i}\cap S_{e_{i+1}}\ne\varnothing\). Let \(H_t\) be the maximum chain
length in \([0,t]\). Accepted-event chains are a subset of these potential-event
chains.

### 2.2 Causal-chain tail

**LEMMA BLSD-CT-CHAIN — CANDIDATE PROOF, UNREVIEWED.** For every integer \(h\ge1\),

\[
\Pr[H_t\ge h]
\le
\frac{n\lambda(s\lambda)^{h-1}t^h}{h!}
\le
n\left(\frac{e s\lambda t}{h}\right)^h.
\]

Consequently, for \(0<\delta<1\),

\[
h_*=
\left\lceil
\max\left\{2es\lambda t,\log_2(n/\delta)\right\}
\right\rceil
\]

satisfies \(\Pr[H_t\ge h_*]\le\delta\).

**CANDIDATE PROOF — UNREVIEWED.** The total rate of possible first events is

\[
\sum_e\bar q_e
\le\sum_{v\in V}\sum_{e:v\in S_e}\bar q_e
\le n\lambda.
\]

For a fixed event \(e\), the total rate of event types whose supports overlap it is
at most

\[
\sum_{f:S_f\cap S_e\ne\varnothing}\bar q_f
\le\sum_{v\in S_e}\sum_{f:v\in S_f}\bar q_f
\le s\lambda.
\]

The factorial-moment formula for independent Poisson processes and the volume
\(t^h/h!\) of ordered times give expected chain count at most the first displayed
quantity. Markov's inequality bounds the probability that at least one chain exists.
Using \(h!\ge(h/e)^h\), \(s\ge1\), and dropping the harmless factor \(1/s\)
gives the second inequality. When \(h\ge2es\lambda t\), its ratio is at most
\(1/2\); when also \(h\ge\log_2(n/\delta)\), the bound is at most \(\delta\).

**HEURISTIC — SIGNIFICANCE.** Bounded local incident rate does derive a causal-depth
tail of order \(s\lambda t+\log(n/\delta)\) for the dominating graphical
construction. Wall time is not silently equated with one circuit step.

### 2.3 Event-count and finite-precision ledger

Let \(R=\sum_e\bar q_e\le n\lambda\). Global uniformization gives a potential
event count \(K\sim\operatorname{Poisson}(Rt)\). The same Poisson tail argument
gives

\[
k_*=
\left\lceil
\max\{2eRt,\log_2(1/\delta_K)\}
\right\rceil
\quad\Longrightarrow\quad
\Pr[K\ge k_*]\le\delta_K.
\]

**HEURISTIC — PARTIAL COMPILATION LEDGER.** Suppose finite-bit samplers approximate
the truncated Poisson count within \(\eta_K\), each event mark within
\(\eta_M\), each thinning coin within \(\eta_A\), and each accepted local
update within \(\eta_U\), uniformly over encoded local states. Then ordinary
TV telescoping gives the candidate terminal-error budget

\[
\eta_P+\delta_K+\eta_K+
k_*(\eta_M+\eta_A+\eta_U)+\eta_R.
\]

The explicit event budget is \(k_*\). The explicit causal-event execution depth on
an already generated and locally scheduled event DAG is at most
\(h_*d_U\), plus preparation, readout, event-generation, schedule-construction, and
routing depth. Random bits are the count-sampler bits plus at most \(k_*\) times
the mark, thinning, and update-sampler bits. No one of these samplers is assigned
unit cost.

### 2.4 Exact obstruction to the full theorem

**HEURISTIC — OBSTRUCTION, NOT A CANNOT THEOREM.** The chain tail is not yet a
uniform geometry-local circuit compiler. The missing step is a fixed,
bounded-fan-in, same-graph circuit that, from finite random bits:

1. samples the approximate local Poisson schedules without a free global clock or
   global event-index oracle;
2. orders only overlapping events and constructs their random dependency levels;
3. routes state through that random schedule at depth controlled by \(H_t\), not by
   the total global count \(K\);
4. evaluates state-dependent finite-bit rates and thinning probabilities at a
   declared bit cost; and
5. preserves the displayed TV budget without a global sort, nonlocal memory, or
   uncharged schedule-description advice.

Global uniformization makes the terminal law finite and simulable, but its single
global event stream can cost depth \(K=\Theta(n\lambda t)\). Independent local
clocks expose the desired \(H_t\) depth, but compiling their random relative order
into one fixed geometry-local circuit requires a schedule-generation/routing theorem
not supplied here. “Finite-bit computable rate” also does not alone give a polynomial
rate-evaluation or Bernoulli-sampling bound; those costs must be parameters or stronger
hypotheses.

Adding a global sorting network or a polynomial-time sampler would change the named
target model or add a premise after the attempt. Calling the event DAG an executable
circuit would assume the very continuous-time depth translation the registration
requires us to derive.

### 2.5 Iteration-10 BLSD-CT token

**FREEZE — HEURISTIC UNDER THE BINDING TWO-UNIT RULE.** A complete-looking
causal-chain tail was derived, but the finite-precision, uniform, geometry-local
compilation with explicit schedule/routing depth was not. The exact obstruction is
section 2.4. BLSD receives no third work-loop unit and the tail lemma is not promoted
to a full BLSD-CT theorem.

**SCOPE LIMIT — HEURISTIC.** This FREEZE says nothing about unbounded rates,
continuous-state dynamics, dense long-range interactions, nonlocal correlated noise,
global preparation/stopping/readout, or physical systems outside the declared
graphical construction. Those are exclusions, not impossibility results.

---

## 3. Review targets and pre-review tokens

| Unit | Worker status before independent review |
|---|---|
| BLSD-D statement and proof | **CANDIDATE PROOF — UNREVIEWED / OCCUPIED-CORE** |
| BLSD-CT chain lemma | **CANDIDATE PROOF — UNREVIEWED** |
| BLSD-CT full compiler | **FREEZE — INCOMPLETE** |

The fresh reviewer is asked to attack both complete-looking proofs but cannot turn
the incomplete BLSD-CT compiler into a pass.

## 4. Independent-review execution and final dispositions

**EMPIRICAL — REVIEW ATTEMPT.** A fresh zero-context Codex CLI session was launched
with the section 1 proof packet and section 2 chain lemma, xhigh reasoning, and an
explicit no-edit adversarial mandate. The process initialized a new session but the
model endpoint was unreachable in the restricted network. WebSocket and HTTPS
fallback retries exhausted before any reviewer response. The temporary reviewer home
and copied credentials were removed from the workspace after failure.

**HEURISTIC — GATE CONSEQUENCE.** Infrastructure failure is not proof rejection and
not proof acceptance. The precommitted positive token
**PROVED-AFTER-REVIEW / OCCUPIED-CORE** cannot fire. The theorem remains a
self-contained candidate proof and review packet; no statement in it receives the
PROVED label.

| Unit | Final batch-2 disposition |
|---|---|
| BLSD-D | **CANDIDATE PROOF — UNREVIEWED / OCCUPIED-CORE; REVIEW-GATE UNSATISFIED.** No positive theorem token issued. The packet is frozen for the supervisor or the next reachable fresh reviewer. |
| BLSD-CT chain lemma | **CANDIDATE PROOF — UNREVIEWED.** It may be useful after review but cannot promote the full unit. |
| BLSD-CT compiler | **FREEZE — HEURISTIC.** Section 2.4 is the exact obstruction; the binding two-unit rule forbids another BLSD work unit. |

**HEURISTIC — ITERATION-9 OUTPUT.** The honest output is **OCCUPIED-CORE CANDIDATE
PROOF + INDEPENDENT-REVIEW PACKET, NO PROVED TOKEN**. It is not
**REJECTED-BY-REVIEW**, because no reviewer reached the proof, and not VOID, because
the process class and target circuit model never changed.

## NARRATIVE

**Honest one-sentence story — HEURISTIC.** *A finite schedule of local random updates
really is just a local random circuit written sideways, and bounded local Poisson
rates limit causal event chains—but converting the whole continuous-time process at
that causal depth still needs a scheduling theorem this batch did not prove.*

**“Isn't that obvious?” test — HEURISTIC.** The discrete unrolling is obvious and
occupied in core. The derived chain-count tail is the nontrivial part, but it is only
a lemma until finite-precision schedule construction closes.

**“So what?” test — HEURISTIC.** BLSD-D can be a reusable audit boundary and the
BLSD-CT obstruction prevents “one second of physics” from becoming a free circuit
depth. Neither supplies a positive capability.

**Mission test — HEURISTIC.** This is floor infrastructure. It would earn “sensible,”
not “that changes everything.” The floor freezes after this batch exactly as
registered.

---

## 5. Batch 5 — standalone BLSD-D proof-review packet and disposition

**PROVED AT CERTIFIED SCOPE / OCCUPIED-CORE.** This section restates the BLSD-D
target from sections 1.1–1.4 as one self-contained packet. The fresh adversarial
review in `research/PROOF_REVIEW_1.md` section 2 upheld exactly this section's
uniform composition of already-supplied finite same-graph preparation, fully
footprint-scheduled local randomized sampler circuits, and readout, with bounds
(D3)–(D6) and the declared resource meanings in section 5.1. The exact repository
token is **`BLSD-D-PROVED-AFTER-REVIEW/OCCUPIED-CORE`**.

The review does not reopen BLSD-CT, alter the process or circuit model, establish
wrapper novelty, construct efficient local samplers from ideal kernels, certify a
continuous-time compiler, define a new computation model, or produce a
mission-level result.

### 5.1 Frozen process semantics and charge boundary

Let $G=(V,E)$ be a uniformly constructed graph, $|V|=n$, of maximum degree
$\Delta$. Each vertex stores one $q$-bit encoding of a symbol in a finite alphabet
$\Sigma$.

For each logical layer $t\in\{1,\ldots,T\}$, a uniformly generated finite block
set $\mathcal B_t$ is given. Every block update $u$ has:

1. a read set $R_u$, write set $W_u$, and complete physical circuit footprint
   $F_u$, all contained in the radius-$r$ neighborhood of a declared center;
2. $R_u\cup W_u\subseteq F_u$ and $|F_u|\le s$;
3. an ideal finite Markov kernel
   $K_u:\Sigma^{R_u}\to\mathcal P(\Sigma^{W_u})$;
4. a bounded-fan-in randomized circuit sampler $S_u$ placed within $F_u$; and
5. declared sampler depth $d_u$, gate size $z_u$, fresh independent random bits
   $\rho_u$, and ancillary/output-buffer bits $a_u$.

The footprint and resources include every local wire, copy, finite decoding
operation, random-bit input, and buffer needed to read the pre-sublayer state and
commit the write. Nothing outside the declared footprint is free.

For every encoded local input $a\in\Sigma^{R_u}$, let
$\widehat K_u(a,\cdot)$ be the sampler's output law and require the uniform
conditional bound

\[
\sup_a\operatorname{TV}\!\left(
K_u(a,\cdot),\widehat K_u(a,\cdot)
\right)\le\eta_u.
\tag{D1}
\]

The conflict graph on $\mathcal B_t$ joins distinct $u,v$ exactly when
$F_u\cap F_v\ne\varnothing$. Thus read/read overlap is serialized, not only
read/write and write/write hazards. A proper coloring

\[
\mathcal B_{t,1},\ldots,\mathcal B_{t,\chi_t}
\]

executes in displayed color order. Within one color, all samplers read the same
pre-sublayer state, use fresh mutually independent random bits, and commit their
writes in parallel. Full-footprint disjointness makes this a product kernel on
disjoint coordinates.

Preparation $P$ and readout $R$ are same-graph bounded-fan-in local randomized
circuits. Their ideal-to-implemented conditional TV errors are $\eta_P,\eta_R$ and
their declared resource tuples are
$(d_P,z_P,\rho_P,a_P)$ and $(d_R,z_R,\rho_R,a_R)$. The ideal terminal law is

\[
\mu_{\rm out}
=\mu_0
\left(\prod_{t=1}^{T}\prod_{c=1}^{\chi_t}
\prod_{u\in\mathcal B_{t,c}}K_u\right)R,
\tag{D2}
\]

where products are in displayed temporal order and the innermost product is the
same-color parallel product. Preparation begins and readout ends at the registered
graph sites; no central sample, global route, or second output materialization is
admitted.

Uniformity means one finite constructor, given the family parameter, generates
$G$, all block/footprint descriptions, every sampler circuit, preparation,
readout, and the schedule. A family of precomputed schedules not generated by that
constructor is outside the theorem.

### 5.2 Exact reviewed theorem

**THEOREM BLSD-D — PROVED AT THE SECTION 5.1 SCOPE.** Under section 5.1, there is a
uniform bounded-fan-in geometry-local randomized circuit on the same graph whose
terminal law $\widehat\mu_{\rm out}$ satisfies

\[
D\le d_P+
\sum_{t=1}^{T}\sum_{c=1}^{\chi_t}
\max_{u\in\mathcal B_{t,c}}d_u+d_R,
\tag{D3}
\]

\[
Z\le z_P+\sum_{t,c}\sum_{u\in\mathcal B_{t,c}}z_u+z_R,
\qquad
\rho\le\rho_P+\sum_{t,c}\sum_{u\in\mathcal B_{t,c}}\rho_u+\rho_R,
\tag{D4}
\]

and peak encoded working state at most

\[
qn+\max\left\{
a_P,a_R,\max_{t,c}\sum_{u\in\mathcal B_{t,c}}a_u
\right\}.
\tag{D5}
\]

Its output remains at the registered graph sites and

\[
\operatorname{TV}(\mu_{\rm out},\widehat\mu_{\rm out})
\le
\eta_P+\sum_{t,c}\sum_{u\in\mathcal B_{t,c}}\eta_u+\eta_R.
\tag{D6}
\]

If only the uncolored block and footprint lists are generated, let
$D_t^{\rm conf}$ be the maximum conflict-graph degree. The same uniform constructor
can deterministically greedy-color the graph and use
$\chi_t\le D_t^{\rm conf}+1$. Schedule generation is a constructor cost and the
resulting schedule description is part of the finite circuit description; it is
not counted as run-time depth in (D3), silently supplied as advice, or claimed to
obey the execution bounds (D3)–(D5).

### 5.3 Reviewed proof

**PROVED AFTER INDEPENDENT REVIEW AT THE SECTION 5.1 SCOPE.**

**Preparation.** Replace the ideal preparation law $\mu_0$ by its declared local
circuit law $\widehat\mu_0$. The hypothesis on $P$ gives

\[
\operatorname{TV}(\mu_0,\widehat\mu_0)\le\eta_P.
\tag{D7}
\]

**One same-color sublayer.** Fix $t,c$ and condition on one complete pre-sublayer
encoded state $x$. The sets $F_u$, $u\in\mathcal B_{t,c}$, are pairwise disjoint.
Consequently the ideal and implemented conditional sublayer kernels factor as

\[
K_{t,c}(x,\cdot)=
\bigotimes_{u\in\mathcal B_{t,c}}K_u(x_{R_u},\cdot),
\qquad
\widehat K_{t,c}(x,\cdot)=
\bigotimes_{u\in\mathcal B_{t,c}}\widehat K_u(x_{R_u},\cdot).
\tag{D8}
\]

Coordinates not written in the sublayer are retained identically in both kernels.
Sequentially replace the factors in (D8). Total variation is unchanged by tensoring
both measures with a common probability measure, and is contracted by every
subsequent Markov kernel. Therefore

\[
\operatorname{TV}\!\left(
K_{t,c}(x,\cdot),\widehat K_{t,c}(x,\cdot)
\right)
\le\sum_{u\in\mathcal B_{t,c}}\eta_u.
\tag{D9}
\]

This is a conditional statement for every $x$, so shared uncertainty in the
pre-sublayer state creates no independence assumption. Independence is required
only of the declared fresh sampler bits conditional on $x$. Integrating (D9)
against any input-state law preserves the same bound.

Physically, all same-color circuits occupy disjoint footprints. They may therefore
run in parallel with depth at most $\max_u d_u$; their sizes, fresh random bits, and
simultaneous ancillary/output-buffer bits add. All reads use the common
pre-sublayer state and all writes commit only after the sampler finishes, so the
parallel product has exactly the semantics in (D8).

**Full telescope.** Order all color sublayers lexicographically by $(t,c)$. For
probability laws $\nu,\widehat\nu$ and Markov kernels $K,\widehat K$,

\[
\operatorname{TV}(\nu K,\widehat\nu\widehat K)
\le
\operatorname{TV}(\nu,\widehat\nu)
+
\sup_x\operatorname{TV}(K(x,\cdot),\widehat K(x,\cdot)).
\tag{D10}
\]

The first term follows from Markov-kernel contraction and the second from replacing
the kernel at a common input law. Applying (D10) to preparation and every sublayer,
then applying the ideal readout contraction and paying $\eta_R$ for its
implementation, gives (D6).

Sequential sublayer depths add, giving (D3). Every sampler appears once in the
unrolled circuit, so gate size and random-bit counts add, giving (D4). Between
sublayers the persistent encoded state is $qn$ bits. During preparation, readout,
or one same-color sublayer, the relevant declared ancillary and output buffers are
simultaneously live; colors do not overlap in time. This gives (D5). Since each
sampler already includes its local routes and the compiled circuit uses the same
vertices for state and terminals, the construction introduces no uncharged route
or materialization.

**Uniform schedule construction.** If a coloring is absent, the constructor first
enumerates the finite block footprints, constructs the conflict graph by testing
finite-set intersections, fixes a deterministic vertex order, and greedily assigns
the least available color. At most $D_t^{\rm conf}+1$ colors are used. This is a
finite uniform construction procedure. Its own generation time may be large; this
theorem claims execution resources for the generated circuit, not a bound on
offline constructor running time.

Equations (D7)–(D10) and the resource accounting establish every displayed bound.
Proof Review 1 attacked the same-color semantics, conditional product, telescope,
resource ledger, and schedule construction and upheld them at this exact scope.

### 5.4 Occupied core and exact claimed delta

**EMPIRICAL + HEURISTIC NOVELTY LABEL.** Scheduled composition of already supplied
finite local randomized update circuits is the occupied core. The proof is ordinary
unrolling plus total-variation contraction; it is not offered as a new computation
model, a new local-sampling theorem, or a positive mission result.

The exact claimed delta is only the wrapper: one statement carries the uniform
conditional-TV product bound from preparation through readout while also exposing
the conflict schedule, full physical footprints, routing, fresh random bits,
ancillas/output buffers, constructor provenance, same-graph terminals, and output
materialization. The bounded source check in section 1.5 did not establish that this
exact ledger is novel. Its honest status remains
**OCCUPIED-CORE / UNESTABLISHED WRAPPER** even if the proof is valid.

### 5.5 Independent attack record and token

The fresh reviewer was required to attack, at minimum:

1. whether full-footprint coloring really excludes every shared read, write hazard,
   route collision, and free fanout;
2. whether (D8) is the correct same-color kernel when samplers share an uncertain
   pre-sublayer state;
3. whether the conditional product inequality (D9) needs an omitted independence
   or measurability hypothesis;
4. whether (D10) justifies the full TV sum for arbitrary intervening state laws;
5. whether preparation/readout, commit buffers, persistent state, routes, random
   bits, or terminal materialization are missing from (D3)–(D6);
6. whether greedy coloring is uniformly constructible from the frozen descriptions
   without schedule advice; and
7. whether any statement has silently changed the process, target circuit model,
   or geometry.

Validity and novelty receive separate dispositions. Proof Review 1 found every
listed attack soundly answered and returned `UPHELD`. Therefore the current exact
validity token is **`BLSD-D-PROVED-AFTER-REVIEW/OCCUPIED-CORE`**.

**Current disposition — PROVED AT CERTIFIED SCOPE / OCCUPIED-CORE.** The certified
statement is only the uniform composition of the already-supplied finite same-graph
preparation, fully footprint-scheduled local randomized sampler circuits, and
readout, with exact bounds (D3)–(D6) and the resource meanings frozen in section
5.1. The novelty label remains **OCCUPIED-CORE / UNESTABLISHED WRAPPER**. BLSD-CT,
sampler synthesis, continuous-time compilation, broader physical simulation, new-
model status, and mission success remain uncertified.
