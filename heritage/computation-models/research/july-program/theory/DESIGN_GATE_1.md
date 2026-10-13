# Design Gate 1 — Founding Dial and Impossibility Target

**Work-loop iteration:** 1
**Steering dialogue:** round 1
**Date:** 2026-07-14
**Scope:** dial selection only; no model definition and no theorem proof
**Gate verdict (HEURISTIC):** **D2-R — end-to-end finite-resolution robustness**

**EMPIRICAL (execution record).** No experiment or GPU computation was run; the GPU lock remains untouched.

## Rigor-label convention

- **PROVED (literature):** a cited work proves the stated result; this document does not independently re-prove it.
- **EMPIRICAL (sweep):** a literature-search or benchmark finding, necessarily limited by its scope.
- **CONJECTURE:** a formal target that has not been proved.
- **HEURISTIC:** a design judgment, reduction argument, or unreviewed mathematical inference.
- **SPECULATION:** a possible broader interpretation or payoff.

**HEURISTIC — decision.** Turn **D2**, sharpened from “precision as a costed resource” to **D2-R: end-to-end finite-resolution robustness**. Charge the resolution required to describe and prepare the device, represent input, maintain a correct trajectory under perturbation, make branch distinctions, sample stochastic dynamics, and extract output—not merely digits inside an ideal real-valued operation.

**HEURISTIC — rejection of the supervisor's primary lean.** Do not found the program on D1 as “polynomial time with \(k\) mind changes.” A known polynomial stabilization bound makes the eventual answer ordinarily computable by waiting to the bound; a polynomial bound only per stage/update leaves unbounded stages or waiting uncharged. D1 currently faces a clock-or-collapse trilemma.

**HEURISTIC — decision basis.** D2-R has the strongest conjunction of a first-class resource, a consequential cannot target, robustness prospects, a famous question—the nonquantum physical Extended Church–Turing Thesis—and an immediate floor-tier use. Its novelty is conditional and narrow; the first theorem push must discriminate novelty rather than begin a long build.

---

## 1. Dial decision and reasoning

### 1.1 What counts as one step

**HEURISTIC — founding cost principle.** An analog/physical transition requested at tolerance ε is not unit cost. Its provisional cost exposes

\[
\mathcal R=(V,L,B,D,I,S), \qquad B=\lceil\log_2(1/\varepsilon)\rceil,
\]

**HEURISTIC — provisional coordinates.** \(V\) is causal volume or locally interacting degrees of freedom, \(L\) is update count or trajectory length, \(B\) is required end-to-end resolution, \(D\) is uniform description/preparation cost, \(I\) is input/output conversion and measurement, and \(S\) is sampling/randomness/confidence.

**HEURISTIC — boundary, not model-building.** This vector is a constraint for the next iteration, not a model definition. Scalarization waits until composition rules show how coordinates add, multiply, or trade off.

**HEURISTIC — selected dial.** \(B\) is primary. \(V,L,D,I,S\) are mandatory side accounts sealing hidden resources, not coequal research programs.

### 1.2 Comparative gate

| Dial | Four-property assessment | Verdict |
|---|---|---|
| **D1 mind changes** | **PROVED (literature):** Ershov, inductive-inference, and Weihrauch hierarchies exist. **HEURISTIC:** ordinary stabilization time erases the proposed hierarchy or exposes free time; robustness depends radically on clock semantics. | **HEURISTIC:** do not select; reopen only after defeating the trilemma. |
| **D2 precision** | **PROVED (literature):** TTE/Ko cost precision; Bournez–Graça–Pouly characterize P by polynomial ODE length; Blanc–Bournez 2023–24 relate robustness to PTIME/PSPACE broadly. **HEURISTIC:** an end-to-end, preparation-to-readout theorem across architectures may remain open. | **HEURISTIC:** select conditionally as D2-R; kill if it becomes prior work plus checklist. |
| **D3 energy** | **PROVED (literature):** Landauer/Bennett and reversible computing already establish energy–time–space tradeoffs. **HEURISTIC:** energy alone does not seal precision, preparation, or relaxation time. | **HEURISTIC:** side account, not founding dial. |
| **D4 uncontrolled interaction** | **PROVED (literature):** IP/MIP and physical-device verification occupy nearby territory. **HEURISTIC:** an uncontrolled environment is a distribution, adversary, or advice/oracle until contracted sharply. | **HEURISTIC:** reject. |
| **D5 relaxation** | **EMPIRICAL (literature):** Tiunov et al. and Kowalsky et al. undercut broad advantage claims on tested families. **HEURISTIC:** “one relaxation” hides length, precision, volume, and readout. | **HEURISTIC:** demote; use as audit target. |

### 1.3 Why not D6

**HEURISTIC.** “Robustness radius,” “description information,” and “distinguishability” are required parts of honest D2, not a novel missed dial. Renaming them would not create novelty.

### 1.4 Decision risk

**PROVED (literature).** Blanc–Bournez is closer to D2-R than research/BIBLIOGRAPHY.md records: their 2023/2024 work covers general discrete, continuous, and hybrid systems, relates polynomial precision robustness to PSPACE in specified settings, and relates time/length robustness to PTIME. See [CSL 2024](https://doi.org/10.4230/LIPIcs.CSL.2024.17) and [ICALP 2024](https://doi.org/10.4230/LIPIcs.ICALP.2024.129).

**HEURISTIC.** Generic “finite precision kills analog magic” is therefore non-novel. D2-R survives only if one architecture-neutral composition theorem includes uniform description, preparation, evolution, perturbation, stochasticity, measurement, and conversion. Whether that gap exists is the first kill question.

---

### 1.5 Concurrent question-loop dissent

**EMPIRICAL (repository record).** research/QUESTIONS.md batch 1 was committed while this gate was in progress. It kills D1 as stated, treats D2 as a mandatory guardrail rather than the leading model, proposes D6 certified transduction/programmability as an interface, and ranks D5 relaxation as the first end-to-end testbed.

**HEURISTIC — work-loop response.** The D1 attack reinforces this gate. The D2 objection is serious but is encoded by the EFRS KILL/VOID criteria. D6 is indispensable normalization, yet it has no single primary resource, positive capability, or distinct impossibility target; it is built into D2-R's end-to-end boundary rather than crowned. D5 supplies a falsification target, but relaxation is a mechanism whose cost still depends on D2-R resolution semantics. Under the user's binding demand for one dial, D2-R remains the least ill-typed choice.

**HEURISTIC — round-2 conflict.** The supervisor must adjudicate D2-R versus the question loop's D5-testbed/D6-interface package. This document does not conceal that the autonomous loops currently disagree.

---

## 2. Infinity-leak audit

### 2.1 Precision leaks

| Leak | Entry point | Required seal |
|---|---|---|
| Exact states | **PROVED (literature):** a real state can encode infinitely many bits. | **HEURISTIC:** access states only through finite enclosures/samples; each requested bit increments \(B\); forbid equality on unresolved reals. |
| Real parameters | **PROVED (literature):** exact constants/weights can act as advice. | **HEURISTIC:** one uniform constructor produces parameter enclosures; charge its program, time, and requested precision in \(D,B\); forbid arbitrary per-length reals. |
| Initialization | **HEURISTIC:** a hard answer can hide in an easy-to-evolve initial state. | **HEURISTIC:** prepare from a fixed finite-description reference state; charge preparation work, resolution, and success probability; pre-solved states are advice. |
| Thresholds | **PROVED (literature):** exact sign/equality tests introduce discontinuous power. | **HEURISTIC:** branch only with a paid separation margin \(\rho\); charge \(\lceil\log_2(1/\rho)\rceil\); unresolved branches remain unresolved. |
| Primitive operations | **HEURISTIC:** unit-cost integration, minimization, equilibrium, or arbitrary transcendental maps can hide oracles. | **HEURISTIC:** use a finite uniform primitive library; expose local evaluation and modulus costs; global solution operators are not primitives. |
| Accumulated error | **PROVED (literature):** finite local precision does not imply finite end-to-end precision when errors amplify. | **HEURISTIC:** charge the smallest perturbation radius preserving the final decision through the whole trajectory. |
| Measurement/output | **HEURISTIC:** exponentially close outputs or rare events hide readout work. | **HEURISTIC:** charge separation, detector resolution, repetitions, confidence, and conversion in \(B,I,S\); require finite classical output or finite sample guarantee. |
| Noise law | **HEURISTIC:** uncomputable or infinitely precise distribution parameters are stochastic advice. | **HEURISTIC:** require a uniform finite sampler or adversarial uncertainty set; charge parameters, entropy samples, repetitions, and confidence. |
| Dynamic range | **HEURISTIC:** enormous amplitude can encode information or magnify tiny distinctions. | **HEURISTIC:** charge exponent/scale bits and creation/readout resources in \(B,D\); record energy as a side account. |

### 2.2 Infinite-time leaks

| Leak | Entry point | Required seal |
|---|---|---|
| Zeno acceleration | **PROVED (literature):** infinitely many logical events in finite observer time repackages infinity. | **HEURISTIC:** count causal events/trajectory length, not wall time alone; finite cost cannot contain infinitely many distinguishable transitions. |
| “One relaxation” | **HEURISTIC:** settling/mixing can take arbitrarily long. | **HEURISTIC:** charge mixing/convergence time or trajectory length at stated tolerance/confidence; relaxation is never atomic. |
| Clock rescaling | **HEURISTIC:** an ODE time rescaling compresses arbitrary work. | **HEURISTIC:** charge length and local bandwidth/energy scale; rescaling preserves cost up to polynomial factors. |
| Stiffness/metastability | **PROVED (literature):** convergence cost depends on conditioning hidden by idealized elapsed time. | **HEURISTIC:** expose condition, spectral gap, stiffness, or mixing through \(B,L\); kill a definition unable to do so. |
| Rare fluctuation | **HEURISTIC:** waiting for an exponentially rare event can be presented as free. | **HEURISTIC:** report expected and high-probability stopping cost; charge success and repetitions in \(S\). |

### 2.3 Unbounded-advice leaks

| Leak | Entry point | Required seal |
|---|---|---|
| Per-size hardware | **PROVED (literature):** nonuniform families can realize advice-class power. | **HEURISTIC:** require a uniform algorithm constructing the size-\(n\) device and parameter enclosures within charged resources. |
| Fabrication tolerance | **HEURISTIC:** the answer can be encoded in which side of a microscopic tolerance a component occupies. | **HEURISTIC:** correctness holds over the full declared tolerance set; no privileged unmeasured point. |
| Environment | **HEURISTIC:** a bath, stream, or field may supply instance-specific information. | **HEURISTIC:** access is uniformly sampled, explicitly charged as input/oracle access, or adversarial within a declared set. |
| Calibration/table | **HEURISTIC:** unlimited offline calibration can precompute the hard map. | **HEURISTIC:** include data size, acquisition cost, and update rule in \(D\); bound reusable advice rather than amortizing it to zero. |

### 2.4 Other finite disguises of infinity

**HEURISTIC — parallelism seal.** Charge \(V\): components, interaction edges, spatial extent, and locality. Exponentially many degrees of freedom do not become one step because they evolve simultaneously.

**HEURISTIC — I/O seal.** Charge encoding, loading, fan-out, sensing, digitization, and decoding. Excluding a dominating conversion cost invalidates the speedup.

**HEURISTIC — compositionality seal.** Each subdevice exports a finite error/robustness contract that composes. If two robust components require unbounded precision when connected, the definition fails.

**HEURISTIC — audit verdict.** D2-R has candidate seals for precision, time, and advice, but none passes until formalized without assuming global digital simulability. Efficient local-law evaluation may be an admissibility condition; efficient whole-device simulation may not.

---

## 3. Crisp question and four-property test

> **CONJECTURE — crisp Physical ECTT question.** Is every uniform, nonquantum physical computation using polynomial causal volume, trajectory length, end-to-end resolution, description/preparation, and I/O/sample cost simulable by a probabilistic Turing machine with polynomial overhead?

**HEURISTIC — plain language.** Can a nonquantum analog machine retain a superpolynomial advantage after paying for every bit of precision needed to build it, keep it on course, and read its answer?

**HEURISTIC — famous-question test.** This makes a candidate nonquantum physical Extended Church–Turing Thesis precise. D2-R is useful only if it replaces movable words—“reasonable,” “physical,” “robust,” “one analog operation,” “including I/O”—with a representation-invariant envelope.

1. **First-class resource.** **HEURISTIC:** end-to-end distinguishability \(B\) becomes explicit rather than exact preparation, preservation, and readout remaining free.
2. **Impossibility theorem.** **CONJECTURE:** §4 says no admissible device escapes efficient probabilistic simulation without a superpolynomial bill in an exposed coordinate.
3. **Robustness/equivalence.** **CONJECTURE:** dyadic intervals, rational balls, finite samplers, and equivalent norms/encodings define the same polynomial class up to polynomial overhead. Failure kills the model.
4. **Famous question.** **HEURISTIC:** the physical ECTT is restricted to nonquantum, finitely described, robust computation; quantum is explicitly outside the class.

---

## 4. Impossibility target identified before model-building

### 4.1 Target theorem

> **CONJECTURE — End-to-End Finite-Resolution Simulation Theorem (EFRS).** Let \(\{A_n\}\) be a uniform family of nonquantum physical computations with finite classical inputs/outputs. Suppose its local laws and noise samplers are uniformly computable to requested precision; correctness is invariant over declared perturbation sets for preparation, parameters, evolution, and measurement; and its resource vector \((V,L,B,D,I,S)\) is bounded. Then a probabilistic Turing machine reproduces its output distribution to total-variation error \(\delta\) in time polynomial in \(n,V,L,B,D,I,S,\log(1/\delta)\).

> **CONJECTURE — cannot corollary.** No admissible family with every coordinate polynomial decides a language outside BPP. Any claimed super-BPP advantage must expose superpolynomial causal volume, trajectory/mixing length, resolution/conditioning, nonuniform description/preparation, I/O, or sampling/confidence.

**HEURISTIC — deterministic specialization.** For deterministic dynamics with adversarial bounded perturbations, target P containment if simulation requires no stochastic sampler.

**HEURISTIC — schema only.** “Uniform,” “local law,” “causal volume,” “trajectory length,” and “perturbation set” remain obligations for the next design unit, not definitions smuggled into this no-model-building iteration.

### 4.2 Nontriviality and later obligations

**HEURISTIC.** EFRS is nontrivial only if one axiom system covers at least two genuinely different substrates—e.g. a finite-precision real-register/relaxation machine and a continuous polynomial-ODE machine—and derives global simulation from local computability, locality, robustness, and charged resources.

**HEURISTIC.** EFRS is trivial if “admissible” means globally efficiently digitally simulable, if it restates bit complexity for a fixed arithmetic circuit, or if every difficulty enters an unnamed condition number defined as “whatever makes simulation efficient.”

**PROVED (literature).** Bournez–Graça–Pouly already provide the finished template for polynomial ODE length and P. **PROVED (literature).** Blanc–Bournez connect precision robustness to PSPACE and time/length robustness to PTIME broadly. EFRS must strictly extend or compositionally reorganize them.

**CONJECTURE — later capability.** The eventual model must simulate ordinary probabilistic Turing computation with polynomial overhead, making its intended class BPP rather than an over-restricted subclass.

**CONJECTURE — later robustness.** Equivalent finite representations and constant-factor norm changes preserve the polynomial class.

---

## 5. Pre-committed first-push criteria

### **CONFIRM**

Return **CONFIRM** only if all hold:

1. **CONJECTURE:** a complete EFRS proof, or sharply delimited theorem with the same cannot corollary, passes fresh independent proof review.
2. **HEURISTIC:** no axiom assumes global efficient digital simulation.
3. **HEURISTIC:** verified embeddings cover at least two non-equivalent substrates, one continuous or stochastic, without changing cost semantics.
4. **HEURISTIC:** the simulator has an explicit polynomial bound; no “reasonable condition,” “fast mixing,” “bounded precision,” or “efficient readout” remains unpriced.
5. **HEURISTIC:** uniform parameters, preparation, trajectory, parallel volume, stochastic sampling, and readout are inside the theorem.
6. **EMPIRICAL:** primary-source review finds no materially identical theorem, including Blanc–Bournez 2023–24 and their sources.
7. **HEURISTIC:** at least one published claim translates into the coordinates without inventing vendor data; missing data yields “unverifiable.”

### **KILL**

Return **KILL** for D2-R as the middle/ceiling direction if any holds:

1. **HEURISTIC:** the only proof assumes the whole machine/global flow efficiently digitally simulable.
2. **HEURISTIC:** the theorem reduces to one known substrate or reformulates TTE/Ko, Bournez–Graça–Pouly, Braverman et al., or Blanc–Bournez.
3. **HEURISTIC:** a uniform countermodel satisfies all polynomial bounds yet defeats simulation for a reason not captured by a legitimate omitted charge.
4. **HEURISTIC:** composition requires a catch-all parameter equivalent to digital simulation cost.
5. **HEURISTIC:** the class excludes realistic noisy analog devices or ordinary probabilistic Turing computation.
6. **HEURISTIC:** the cannot corollary requires P ≠ NP or NP ⊄ BPP instead of unconditional simulation containment.

**HEURISTIC — surviving kernel.** The floor audit may survive KILL as a useful byproduct; it does not keep D2-R alive as the new-model direction or satisfy the moonshot.

### **VOID**

Return **VOID** without judging D2-R true or false only if:

1. **EMPIRICAL:** primary-source review finds EFRS already proved at the claimed breadth.
2. **HEURISTIC:** formal inconsistency/representation ambiguity makes the statement not truth-apt before proof.
3. **HEURISTIC:** a source was materially misread and the registered target must be restated before evidence counts.

**HEURISTIC — anti-goalpost rule.** Failed proof, counterexample, or narrower result is not VOID; it is KILL unless the registered statement itself was invalid.

---

## 6. Novelty positioning

### 6.1 D2-R against nearest work

| Existing line | Established territory | D2-R's only defensible novelty |
|---|---|---|
| **BSS** | **PROVED (literature):** exact real registers/unit arithmetic leave precision and physical robustness uncharged. | **HEURISTIC:** forbid exact access and charge preparation-to-readout robustness; criticizing BSS is not new. |
| **TTE / Ko** | **PROVED (literature):** real inputs are approximation streams and requested precision enters complexity. | **HEURISTIC:** add a compositional physical execution contract—manufacture, error growth, stochasticity, measurement, conversion—not a renamed precision parameter. |
| **Bournez–Graça–Pouly** | **PROVED (literature):** polynomial-length polynomial ODEs characterize P. | **HEURISTIC:** embed this as one instance of a substrate-independent theorem; do not rediscover “length is time.” |
| **Braverman et al.** | **PROVED (literature):** specified noise/closed-system assumptions destroy some idealized intractability. | **HEURISTIC:** unify adversarial finite-resolution and samplable stochastic contracts, including preparation/readout, across multiple dynamics families. |
| **Blanc–Bournez 2023–24** | **PROVED (literature):** robustness, precision, time, and length already characterize PTIME/PSPACE in broad dynamical settings. | **HEURISTIC:** only end-to-end resource composition spanning description, construction, initialization, evolution, noise, measurement, and I/O may remain. If already supplied, D2-R is VOID/KILL, not incrementally new. |
| **Gandy** | **PROVED (literature):** locality/bounded structure force Turing computability for discrete deterministic mechanisms. | **HEURISTIC:** EFRS targets a quantitative approximate complexity analogue for nonquantum continuous/stochastic mechanisms. “Continuous Gandy” is a target, not a result. |
| **Audit lineage** | **EMPIRICAL (sweep):** hidden-resource criticism/fair-benchmarking components exist; no unified checklist-grade standard was found. | **HEURISTIC:** the audit is new only if EFRS supplies theorem-backed rules. A better survey is neither model nor moonshot. |

### 6.2 D1 against all five fragments

**PROVED (literature) — Ershov.** Bounded changes already stratify \(\Delta^0_2\) approximations through constructive ordinals. **HEURISTIC:** a strict \(k\) versus \(k-1\) result without restrictive time interaction is Ershov with renamed classes.

**PROVED (literature) — Case–Smith / Freivalds–Smith.** Finite and ordinal mind-change hierarchies already cost retractions while leaving ordinary time unrestricted. **HEURISTIC:** a per-update polynomial clock is insufficient if arbitrary postponement remains.

**PROVED (literature) — Weihrauch.** Finite changes and limit operators already have robust structural degrees over represented spaces. **HEURISTIC:** D1 must produce a stable joint resource theorem not inherited from that lattice.

**PROVED (literature) — Nies.** Change-budget cost functions already grade computable approximations and connect to randomness. **HEURISTIC:** weighted changes are occupied; novelty requires honest finite-runtime or operational content.

**PROVED (literature) — Bei–Chen–Zhang.** Trial count is jointly costed with polynomial computation under a certifying violation oracle. **HEURISTIC:** Putnam/Gold uncertifiable convergence is semantically different, but removing feedback also removes a practical stopping signal; semantic difference alone is not usefulness.

### 6.3 Pitt 1989 and the unclaimed clearance

**PROVED (literature).** Pitt records these failures: target-size polynomial total time cannot read arbitrarily long early examples; polynomial update time permits exhaustive enumeration; update time plus polynomial mind changes can retain useless inconsistent hypotheses; a convergence bound depending on observed presentation length can be padded; and a polynomial convergence index is impossible under arbitrary presentation because decisive data may be delayed. See [Pitt 1989, pp. 5–7](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/pitt89.pdf).

**HEURISTIC — clock-or-collapse trilemma.**

1. **HEURISTIC:** Charge all ordinary steps through stabilization with a computable polynomial bound: a standard machine waits to the bound and outputs once, apparently collapsing the polynomial-time mind-change hierarchy.
2. **HEURISTIC:** Charge only each update/stage while allowing unbounded stages/gaps: time is cosmetic and power comes from free unbounded time.
3. **HEURISTIC:** Let external evidence trigger revisions: its schedule must be costed or declared input/oracle access, returning to Pitt, online learning, or certifiable feedback.

**HEURISTIC — Pitt verdict.** This round does not claim clearance because D1 is not selected. “Jointly time-and-mind-change-costed” is not a definition surviving Pitt; the exact clock must be supplied before D1 re-enters a gate.

---

## 7. Disagreements with the supervisor's lean

### Disagreement 1 — D1 should not be the founding dial

**HEURISTIC.** The literature gap is necessary but insufficient. It may exist because the desired properties are incompatible: limit power comes from unbounded, uncertifiable ordinary time, while a known polynomial stabilization bound removes that power. This warrants withholding selection, not later definitional cleanup.

### Disagreement 2 — the proposed hierarchy question is not yet viable

**HEURISTIC.** “Does polynomial time with \(k\) mind changes strictly exceed \(k-1\)?” is not crisp until time bounds stabilization, inter-output delay, per-stage work, accumulated work, or response to presentation. The first reading appears to collapse; others risk Ershov with a cosmetic clock or Pitt-style learning. The first D1 target should be the trilemma, not the advertised separation.

### Disagreement 3 — uncertifiable convergence must earn its keep

**HEURISTIC.** “Science as computation” is a strong narrative but not an unkillable usefulness argument. A process whose correct answer cannot be recognized and whose convergence time is unbounded may describe inquiry without delivering efficient usable computation. Philosophical resonance cannot substitute for operational advantage.

### Sharpening — the floor audit is not a coequal parallel program

**HEURISTIC.** The Extropic/Normal audit should be D2-R's first adversarial application, but only after a minimal resource interface exists. A free-standing checklist risks optimizing a methodology byproduct while the theorem is undefined. The audit is a falsifier and floor deliverable, not the moonshot or rescue for killed EFRS.

**HEURISTIC — agreement retained.** D5 should be demoted, and Bournez–Graça–Pouly should be the completeness standard: natural cost, classical characterization, and no hidden physical advantage.

---

## 8. Open items for round 2

1. **EMPIRICAL — blocking novelty check.** Does full reading of Blanc–Bournez 2023–24 and predecessors already give D2-R's end-to-end compositional theorem, or only reachability/ODE robustness?
2. **HEURISTIC — D1 challenge.** Supply a time definition counting all ordinary computation, revealing no stabilization deadline, leaving no unbounded waiting/stages free, and supporting a strict \(k\)-change hierarchy not inherited from Ershov/Case–Smith.
3. **HEURISTIC — theorem domain.** Begin EFRS with discrete-time noisy real registers then embed continuous systems, or start with substrate-neutral Gandy-style axioms? The former is more provable; the latter alone has ceiling-level novelty.
4. **HEURISTIC — randomness.** Choose deterministic P containment under adversarial perturbations or BPP containment with uniform noise sampling. Changing semantics after results violates the gate.
5. **HEURISTIC — condition accounting.** Identify non-circular primitives for sensitivity, stiffness, mixing, and output margin. A catch-all condition number is forbidden unless independently measurable.
6. **HEURISTIC — representation test.** Preselect representations/norms whose polynomial equivalence counts as robustness.
7. **HEURISTIC — audit sequencing.** If retaining D2-R, state the minimal formal interface before an Extropic/Normal dry run; unavailable vendor data yields “unverifiable,” not inference.
8. **HEURISTIC — resolution.** Round 2 accepts D2-R without weakening novelty, defeats the D1 trilemma and reopens D1, or rejects both for a new dial. “Explore both” is not a founding decision.

---

## 9. Narrative gate

**SPECULATION — honest story if EFRS is proved.** *Every nonquantum machine that looks magically faster must pay the missing bill in time, hardware, precision, setup, or readout—and this theorem makes hiding that bill impossible.*

**HEURISTIC — obviousness test.** The story is obvious unless one architecture-neutral theorem derives accounting across multiple models. A checklist or second ODE proof fails.

**HEURISTIC — so-what test.** The theorem would give researchers, funders, and buyers a falsifiable standard for physical-computing advantage. Until proved/applied, that payoff is conjectural.

**HEURISTIC — mission test.** An impossibility theorem plus audit is a necessary component and valuable floor result, not by itself the middle/ceiling moonshot. A capability theorem or positive useful task class remains owed.

---

## 10. Round-1 recommendation

**HEURISTIC — single recommendation.** Select **D2-R: end-to-end finite-resolution robustness**.

**CONJECTURE — first theorem target.** Push **EFRS**: polynomial-overhead probabilistic Turing simulation for every uniform nonquantum computation whose causal volume, evolution length, end-to-end resolution, description/preparation, I/O, and sampling costs are polynomial.

**HEURISTIC — pre-committed posture.** If EFRS cannot be architecture-neutral without circularly assuming simulation, or Blanc–Bournez already proves its core, return KILL or VOID under §5. Do not retreat to “a useful precision checklist” and call the founding model successful.

---

## 11. Design Gate 1 — Round 2 resolution

**Date:** 2026-07-14

**Steering dialogue:** round 2

**Dialogue verdict (HEURISTIC):** **CONVERGED**

**Design-gate verdict (HEURISTIC):** **FREEZE D2-R/EFRS, PROCEED ONLY TO THE BLOCKING NOVELTY CHECK**

**EMPIRICAL (execution record).** This round changed only the research record. It ran
no experiment or proof computation, and the GPU lock remained untouched.

### 11.1 Point-by-point adjudication

#### 1. D2-R versus the D5/D6 package — synthesis accepted

**HEURISTIC — accepted.** D2-R is the founding dial and EFRS is the theorem target.
D5 and D6 are absorbed as obligations of that target rather than treated as rival
founding dials:

- The first stochastic substrate must be a **discrete-time noisy real-register
  relaxation machine**, not merely an arbitrary real-register machine with decorative
  noise. Its mandatory nontrivial instance is a finite-resolution Langevin/Markov
  relaxation with charged step or trajectory length, mixing/stopping rule, noise
  sampling, output separation, and confidence. Calling registers “noisy” without this
  relaxation instance does not satisfy the two-substrate obligation.
- D6's certified-transduction content is carried by uniformity and the **D, I, S**
  coordinates together with preparation, evolution, and composition rules. The
  question loop's ten-item end-to-end normalization sheet becomes the admissibility
  checklist for the machine class; it is not a third model.
- The Extropic/Normal audit remains EFRS's first adversarial application, after the
  minimal formal interface exists. It is a falsifier and floor byproduct, never a
  rescue path if EFRS is killed.

**HEURISTIC — resolution of the loop conflict.** The question loop was right that
relaxation must force an end-to-end task and that certified transduction must prevent
cost migration. The work loop was right that neither fact displaces finite-resolution
robustness as the load-bearing resource. The conflict recorded in §1.5 is closed.

#### 2. Blocking novelty check — accepted, with a hard dependency

**HEURISTIC — accepted.** Work-loop batch 1 begins with full primary-source reading of
CSL 2024 LIPIcs 17, ICALP 2024 LIPIcs 129, and every predecessor on which their
PTIME/PSPACE, robustness, or substrate-generality claims materially depend. The output
must compare exact definitions and theorems against every preparation-to-readout
obligation in §§4–6; abstracts, search snippets, and the present bibliography are not
enough.

**HEURISTIC — worker tightening.** This is an outcome gate, not a two-iteration
deadline. Iterations 1–2 are reserved for it, but if the load-bearing predecessor
trace is still open, iteration 3 and later remain on novelty until the trace closes.
No calendar slot authorizes interface or proof work while novelty is unresolved.

The novelty gate has three procedural outcomes:

- **CLEAR-TO-FORMALIZE:** no §5 KILL or VOID condition is triggered by the source
  record. This is not CONFIRM and makes no theorem claim.
- **VOID:** a materially identical end-to-end theorem already exists, or the
  registered target is not truth-apt as written, exactly as §5 specifies.
- **KILL:** the only surviving contribution is a known-substrate reformulation,
  circular simulation condition, or checklist-grade composition of existing results,
  exactly as §5 specifies.

On VOID or KILL, EFRS work stops. The audit checklist is not promoted. The program
returns to dial selection with two reopened packages: the finite-observer D1 variant,
which must defeat the standing clock-or-collapse trilemma, and bounded causal reach,
which must survive absorption by circuit depth, distributed computation, and locality
bounds.

#### 3. First theorem domain — accepted, without assuming strictness

**HEURISTIC — frozen first domain.** The first formal substrate is the discrete-time
noisy real-register relaxation machine above. The second is the deterministic
polynomial-ODE/Bournez–Graça–Pouly setting. BGP is to be embedded into the common EFRS
interface through its local polynomial vector field, uniform descriptions, robustness
contract, and trajectory-length accounting—never by installing a unit-cost ODE-solution
oracle or assuming the global flow is efficiently simulable.

**HEURISTIC — scope correction.** The combined interface is *intended* to strictly
extend the deterministic BGP setting because it also covers stochastic relaxation;
strict containment is an owed result, not a design premise. If the two embeddings are
only notationally different or require changed cost semantics, §5 KILL applies.

**CONJECTURE — registered ceiling.** A substrate-neutral quantitative Gandy-style
axiomatization remains the later ceiling target. It is not part of the first proof
push and cannot delay the narrower theorem.

#### 4. Randomness semantics — accepted and frozen

**CONJECTURE — primary target.** EFRS targets BPP containment under uniformly samplable
noise. Noise parameters and samplers must have finite uniform descriptions, accept a
requested approximation tolerance, and expose their bit cost, sample count, and
confidence in **B, D, S**. An exact Gaussian, equilibrium sample, or transition kernel is
not a free primitive.

**CONJECTURE — specialization.** Deterministic dynamics with declared adversarial
perturbation sets target P containment. This is a specialization of the frozen
interface, not permission to swap stochastic and worst-case semantics after seeing a
result. A later semantic change triggers §5 VOID and re-registration.

#### 5. D1 challenge — accepted

**HEURISTIC.** D1 remains killed as the founding dial. The clock-or-collapse trilemma
in §6.3 is the program's standing answer. The finite-observer reconstruction remains
in question-loop custody and reopens only if EFRS is killed/voided or if it later
supplies an operational finite-horizon separation that defeats the trilemma and its
learning-theory absorption baselines.

#### 6. Work-loop batch 1 — registered

**HEURISTIC — cadence.** This is one autonomous ten-iteration batch before the next
ordinary supervisor check-in. Dependencies outrank slot numbers: unfinished novelty or
proof obligations slide later items rather than receiving a favorable verdict by
schedule.

1. **Iteration 1 — nearest-work theorem map.** Read CSL 2024 LIPIcs 17 and ICALP 2024
   LIPIcs 129 in full. Extract their exact machine domains, robustness quantifiers,
   precision/time/length measures, uniformity assumptions, noise semantics,
   preparation/readout boundaries, simulation conclusions, and load-bearing cited
   predecessors. Produce a claim-to-source map, not a survey summary.
2. **Iteration 2 — predecessor closure and novelty verdict.** Read the load-bearing
   predecessors in full; compare the resulting theorem closure against every EFRS
   coordinate and composition obligation. Return CLEAR-TO-FORMALIZE, VOID, or KILL
   with a written §5 criterion-by-criterion matrix. If closure is incomplete, record
   the exact missing sources and continue this item in later iterations; do not infer
   clearance from absence.
3. **Iteration 3 — machine class and contracts, conditional on clearance.** Define the
   discrete-time noisy real-register relaxation machine, finite input/output task
   contract, uniform constructor, local transition/noise interface, perturbation
   semantics, and nonquantum boundary. Include a concrete Langevin/Markov relaxation
   instance without granting equilibrium sampling as a primitive.
4. **Iteration 4 — resources, admissibility, and composition.** Give operational
   semantics for the resource vector **(V, L, B, D, I, S)**; instantiate all ten
   normalization-sheet items;
   preselect representations and norms; define sequential/parallel composition and
   amortization; rerun the infinity-leak audit. Keep the resource vector unscalarized
   and forbid a catch-all “condition number.”
5. **Iteration 5 — finite-resolution simulation lemmas.** Derive a paid precision
   schedule from local moduli, perturbation robustness, branch/output margins, and the
   finite trajectory contract. State explicit bounds and attack the possibility that
   admissibility already assumes the desired simulator.
6. **Iteration 6 — stochastic simulator and error budget.** Construct the probabilistic
   simulation; account explicitly for transition-kernel approximation, accumulated
   error, total-variation target, randomness, repetitions, and runtime. No appeal to
   “fast mixing” or “reasonable precision” is permitted.
7. **Iteration 7 — theorem closure and hostile self-audit.** Attempt the BPP containment
   and deterministic P specialization, ordinary BPP capability embedding, robustness
   under the preselected equivalent representations, and the explicit noisy-relaxation
   instance. Search for a polynomial-resource countermodel and apply §5 immediately if
   one survives.
8. **Iteration 8 — BGP/ODE substrate.** Embed the polynomial-ODE/length setting into the
   same interface without a solution oracle or altered resource semantics. Prove the
   inherited simulation bound or record the exact obstruction; failure of genuine
   two-substrate coverage triggers the registered KILL posture.
9. **Iteration 9 — independent proof-review gate registration.** Assemble the theorem,
   dependency graph, definitions, proof, source boundary, and unresolved obligations
   into a zero-context review packet. Precommit the fresh reviewer's attacks: circular
   admissibility, hidden global simulation, noise-oracle leakage, nonuniform advice,
   composition failure, representation dependence, and false substrate diversity. No
   claim is labeled PROVED before that independent review passes.
10. **Iteration 10 — audit dry-run translation.** Translate Extropic's approximately
    **10,000x** TSU-versus-GPU energy-efficiency claim into the resource vector
    **(V, L, B, D, I, S)** plus the side energy account, using a refreshed primary vendor
    statement and a matched output contract. Missing preparation, I/O, precision,
    confidence, amortization, or baseline data yields **UNVERIFIABLE**, not an inferred
    failure or success.

**HEURISTIC — VOID/KILL branch.** If iteration 2 or a later-completed novelty trace
returns VOID/KILL, iterations 3–10 above are canceled as EFRS work. The remaining
batch investigates only the two reopened dial packages: formalize the D1 trilemma
challenge and finite-use contract; trace causal reach against depth/distributed/locality
theory; normalize both end to end; run their absorption and infinity-leak attacks; and
end iteration 10 with a round-1 steering recommendation. Neither dial is selected
without the next supervisor dialogue.

### 11.2 Frozen tokens and remaining obligations

**HEURISTIC — tokens unchanged.** The CONFIRM, KILL, and VOID tokens in §5 are
incorporated here verbatim by reference and remain the only EFRS outcome criteria.
CLEAR-TO-FORMALIZE is merely permission to define and attempt the theorem; it is not a
fourth success verdict. No audit result, narrower one-substrate theorem, failed proof,
or narratively attractive vendor example may weaken those tokens.

**HEURISTIC — still owed after this gate.** The condition-accounting primitives and
representation equivalence class from open items 5–6 are work items, not settled
facts. The design is frozen enough to test them; their failure triggers the existing
KILL/VOID rules rather than silent redesign.

### 11.3 Narrative gate

**HEURISTIC — honest one-sentence story now.** *We have chosen the exact hidden bill we
will try to prove every nonquantum physical speedup must pay, but we do not yet know
whether the theorem is new or true.*

**HEURISTIC — obviousness/triviality test.** A design choice and a longer accounting
sheet are sensible, not world-changing. If the primary-source check finds the theorem
already present, or if admissibility assumes digital simulation, the story collapses
and the registered VOID/KILL branch fires.

**HEURISTIC — mission test.** This round is infrastructure for the moonshot, not the
moonshot. A negative theorem and vendor audit would remain floor/middle components;
the program still owes a positive useful capability result.

**CONJECTURE — unkillable target.** The story becomes nontrivial only if one
non-circular theorem survives independent proof review across both a genuine noisy
relaxation substrate and the BGP/ODE substrate, exposes every preparation-to-readout
cost, and is paired later with a useful capability theorem rather than promoted as an
audit framework.
