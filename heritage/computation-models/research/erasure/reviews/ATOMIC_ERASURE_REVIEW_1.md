**Code-gate verdict: REPAIR.** The circuit semantics and partition dynamic program survive review, but the runner needs a fail-closed check against optimized Python before the registered full run.

**Proof verdict: UPHELD AT THE FROZEN SCOPE.** The atomic lower bound is correct, including arbitrary depth, multiway merges, simultaneous merges, unequal fibres, and finite ancillary space. The matching partition upper bound is also correct. I additionally derive a charged gate implementation below; its general truth-table cost prevents an efficiency claim.

No registered full experiment was run. No files or git state were modified, no GPU was used, and no additional agents or ongoing loops were launched. Private memory supplied no evidence.

**Review boundary and provenance**

I read the repository constitution, the required dual-loop skill in full, README, HANDOFF §17, TERMINAL_GATE_2, SYNTHESIS_GRAPH §29, STEERING_ENDGAME, [CURRENT_STATE.md](<LOCAL_PROJECTS>/new-computation-model/research/CURRENT_STATE.md), [ATOMIC_ERASURE_FRONTIER.md](<LOCAL_PROJECTS>/new-computation-model/research/ATOMIC_ERASURE_FRONTIER.md), October registrations, the experiment ledger, the implementation, and the complete smoke data.

The initial-state and terminal-state assumptions in the frontier are essential: initially each atom has a distinct full state; finally the full state identifies \(f\) and nothing else.

Current fingerprints, checked again after the probes:

| Artifact | SHA-256 |
|---|---|
| `experiments/entropy_controls.py` | `202611d0babb23cf36860c2c853397b678d63a977e36265cf94101984a21d0ea` |
| `results/entropy_controls_smoke.json` | `05630d1115bb4f7252bd83ba111d83483f789364b0c14ff034cbc15b4cade50a` |

The smoke’s recorded source hash matches the inspected source. This establishes source/result consistency for that smoke, not full-run outcomes.

**Implementation findings and dispositions**

| Finding | Assessment | Disposition |
|---|---|---|
| Validation disappears under optimized Python | **Blocking robustness defect.** All substantive acceptance checks use `assert`; `-O` or inherited `PYTHONOPTIMIZE` removes them. | **REPAIR before full run.** Reject optimized execution explicitly. |
| “At” cap includes rounding slack | The implementation tests `predicted + 1e-60`, not the exact computed threshold. The slack is disclosed in the JSON. | Nonblocking numerical convention; describe the field as “at within disclosed slack.” It supplies no exact-equality proof. |
| NOT is absent from `apply_layer` | The declared basis includes NOT, but the registered AND construction emits only CNOT, Toffoli, and RESET. | No defect in these registered controls. The simulator does not cover the entire declared basis. |
| Failure verdict serialization | Ordinary assertion failures abort before producing a new positive result, but the runner does not emit structured KILL/VOID records. | The execution/evidence wrapper must classify and record failures; an older output file cannot stand for a failed rerun. |
| Mathematical scope of partition search | It searches serial binary merges of one uniform fibre, represented by integer partitions. | Correct registered scope. Do not promote its output to gate-level or general machine certification. |
| Source/smoke agreement | All saved AND traces and atomic case fields reproduced from the inspected source. | Consistent smoke plumbing; **EMPIRICAL**, not outcome evidence. |

The optimization defect is demonstrated, rather than hypothetical. In an isolated in-memory namespace, I replaced the circuit generator with five empty layers for \(N=2,K=1\):

- Normal compilation rejected the incorrect clean circuit with `AssertionError`.
- Compilation with `optimize=1` returned `verdict='CONFIRM'`, although terminal entropy remained **2 bits**, rather than \(h_2(1/4)\).

No source file was changed during this probe. The present interpreter had optimization disabled, so this does **not** invalidate the saved smoke.

A sufficient minimal repair for the registered invocation is an explicit module-level guard:

```python
if not __debug__:
    raise RuntimeError("Validation requires assertions enabled")
```

Then rerun the smoke and record its new source fingerprint before proceeding through the existing gates. This changes validation protection, not the frozen experiment’s range, objective, or arithmetic precision. I have not applied it.

The relevant checks and unconditional positive verdicts are in [entropy_controls.py](<LOCAL_PROJECTS>/new-computation-model/experiments/entropy_controls.py:63); full-run orchestration begins at line 161.

**What survived the implementation audit**

The AND generator is correct for the registered integer range:

- Odd frontiers carry their unmatched wire forward correctly.
- Every pair receives a fresh target. Exactly \(N-1\) scratch bits are allocated.
- Inputs, scratch, and output occupy exactly \(2N\) bits.
- Footprints within each layer are disjoint.
- Reading controls from the pre-layer state implements simultaneous-layer semantics correctly.
- Reversing the layer sequence correctly uncomputes the self-inverse Toffoli tree.
- Copying the root into the separate output before reversal preserves the result.
- RESET batches cover precisely the original input wires, without overlap or omission.
- Clean terminal-state comparison checks the entire packed state.
- Dirty mode retains the original inputs and deliberately retains tree garbage.

For \(h=\lceil\log_2N\rceil\), the clean depth is

\[
2h+1+\lceil N/K\rceil,
\]

including \(N=1\), where \(h=0\). The clean gate count is \(3N-1\); the dirty construction has depth \(h+1\) and \(N\) gates. The source’s rejection of \(K>N\) matches the registered range; the mathematical construction extends to larger \(K\) by using one batch containing all \(N\) resets.

Entropy uses full-state multiplicities:

\[
H=\log_2M-\frac1M\sum_c c\log_2c.
\]

These are counts of complete states, not marginal wire entropies. The double-precision tolerance is adequate for the frozen finite range, although numerical tolerance is not an analytic certificate.

The atomic dynamic program also matches its registered objective:

\[
V(P)=\min_{P\to P'}\max\{\operatorname{loss}(P,P'),V(P')\}.
\]

Every edge reduces block count by one, so the recurrence is acyclic. Sorting block sizes is a valid quotient because atoms are uniform, labels impose no additional constraints within this single fibre, and edge loss depends only on sizes. Deduplicating identical successor partitions cannot hide a different entropy cost: that cost is determined by the endpoint entropies.

The shortest-path search is ordinary BFS over admissible edges. Every successful binary-merge path necessarily contains \(q-1\) merges; therefore these depth fields primarily test **reachability under the cap**, not a variable-depth optimization. The balanced recurrence correctly retains the original denominator \(M=q\) throughout its subtrees.

**Cheap independent CPU verification — EMPIRICAL**

Probes used `python -B` with inline, in-memory code. The source was loaded through `runpy.run_path` without invoking `main`; nothing was written.

| Probe | Coverage | Result |
|---|---|---|
| Independent bit-vector circuit simulator | \(N=1,2,3,5,7,9\); dirty mode and selected \(K\in\{1,\lfloor N/2\rfloor,N\}\), deduplicated | 20 cases agreed on execution, output, cleanup, entropy cap, and gate count |
| Saved smoke replay | Six clean AND cases, three dirty cases, five atomic cases | Every saved case field reproduced |
| Independent partition-successor enumeration | Every integer partition for \(q\le7\) | 44 states and 62 distinct edges agreed |
| Labelled partition search allowing **every** coarsening | Single fibres through size five, plus unequal-fibre examples | Agreed with the proposed bound |
| Exact rational bottleneck calculation | \(q=1,\ldots,5\) | Exact optimum and exact-cap reachability agreed |
| Charged gate compiler microprobe | Six small task tables | Clean output and predicted peak agreed |

The source/smoke probe batch reported approximately **0.089 seconds** under Python **3.13.7**, optimization level zero. The exact-rational probe reported approximately **0.000181 seconds**. The saved smoke records **0.0029073 seconds** for its original execution; I did not independently establish that historical timing.

For exact comparison, a binary merge has

\[
2^{M\,\operatorname{loss}(a,b)}
=\frac{(a+b)^{a+b}}{a^ab^b}.
\]

Thus minimax comparisons can be performed using exact rational numbers, without evaluating logarithms. The independent smoke-range optima were:

| \(q\) | Exact \(2^{M K_*}\) | Binary depth at exact cap |
|---:|---:|---:|
| 1 | \(1\) | 0 |
| 2 | \(4\) | 1 |
| 3 | \(27/4\) | 2 |
| 4 | \(256/27\) | 3 |
| 5 | \(3125/256\) | 4 |

This is an independent numerical check; it does not replace the registered 80-digit Decimal experiment.

The search permitting all coarsenings also checked fibre sizes \((3,2)\) on \(M=5\), and \((3,2,1)\) and \((4,2)\) on \(M=6\). Their optimal peaks agreed with \(\phi(q)/M\). These small checks include multiway and simultaneous merges, which the registered binary search omits.

**Independent AND proof — UPHELD / PROVED AT SCOPE**

The tree has \(N-1\) internal nodes and height \(\lceil\log_2N\rceil\). A Toffoli writes the AND of two controls into a fresh zero target while retaining the controls. After root copying and tree reversal, the state is \((X,\operatorname{AND}(X),0_{\text{scratch}})\).

For a RESET batch erasing \(r\le K\) bits, write the pre-reset state as \((A,B)\), where \(A\) contains those bits and \(B\) contains everything retained. Its entropy loss is

\[
H(A,B)-H(B)=H(A\mid B)\le r\le K.
\]

This argument accounts for correlations; it does not sum marginal entropies.

Initially \(H(S_0)=N\). Under the clean contract,

\[
H(S_D)=h_2(2^{-N}).
\]

Telescoping therefore gives

\[
DK\ge N-h_2(2^{-N}).
\]

For \(N\ge2\), the right side is positive, so cap zero forbids clean computation at every finite depth. Dirty reversible output remains possible because the inputs are retained. The exclusion of \(N=1\) is correct: AND is then injective.

Finally, a wire’s dependency set can grow by a factor of at most three per layer in this basis. Since AND depends on all \(N\) inputs,

\[
D\ge\lceil\log_3N\rceil.
\]

These are valid control results using established reversible ingredients. They establish no novelty or physical cooling law.

**Independent atomic proof — UPHELD / PROVED AT SCOPE**

Let \(X\) range uniformly over \(M\) atoms, and let \(S_t(X)\) denote complete computational state at synchronous step \(t\). Define \(P_t\) by equality of those states. Assume initially \(P_0\) is discrete and terminal state is an injective encoding of \(f(X)\), with all other state fixed.

Write \(g(k)=k\log_2k\), with \(g(1)=0\).

1. **Partitions only coarsen.**  
   Determinism gives
   \[
   S_t(x)=S_t(x')\Rightarrow S_{t+1}(x)=S_{t+1}(x').
   \]
   Equal complete states cannot later separate.

2. **Every intermediate block lies inside a final fibre.**  
   If atoms with different \(f\)-values merged, they would remain equal at termination, contradicting exact output. Thus \(P_t\) refines the final fibre partition.

3. **Loss decomposes into nonnegative merge contributions.**  
   For uniform atoms,
   \[
   H(P)=\log_2M-\frac1M\sum_{B\in P}g(|B|).
   \]
   A new block formed from predecessor blocks of sizes \(a_1,\ldots,a_r\), with total \(s\), contributes
   \[
   \frac{g(s)-\sum_i g(a_i)}M
   =\frac{s}{M}H(a_1/s,\ldots,a_r/s)\ge0.
   \]
   Other simultaneous merges cannot offset this contribution.

4. **Use the first formation of a largest final fibre.**  
   Choose a fibre \(F\) of size \(q\ge2\). It is absent as a whole block initially and present at termination. At its first formation, it merges at least two positive-size predecessor blocks whose sizes sum to \(q\).

5. **The least destructive formation has sizes \(q-1,1\).**  
   Convexity of \(g\) implies that, for fixed \(r\), the maximum of \(\sum_i g(a_i)\) occurs at
   \[
   (q-r+1,1,\ldots,1).
   \]
   One can prove this directly by transferring one unit from a smaller nonsingleton part to a largest part: convexity makes the sum increase. Since \(r\ge2\),
   \[
   \sum_i g(a_i)\le g(q-1).
   \]
   Therefore this step loses at least
   \[
   \boxed{\frac{q\log_2q-(q-1)\log_2(q-1)}M
   =\frac{\phi(q)}M}.
   \]

Consequently, \(K<\phi(q)/M\) forbids every finite exact clean realization, irrespective of depth or finite ancillary width.

For the matching partition path, merge one singleton at a time into an accumulating block, handling fibres in separate steps. Growing a block from \(k-1\) to \(k\) loses \(\phi(k)/M\). Strict convexity makes these successive increments increasing. The peak is therefore exactly \(\phi(q)/M\), using

\[
\sum_F(|F|-1)=M-|\operatorname{image}(f)|
\]

merging steps.

Thus, over unrestricted partition-coarsening trajectories,

\[
\boxed{K_*=\phi(q)/M}
\]

for \(q\ge2\), with obstruction zero for injective \(f\).

**Counterexample attacks and scope limits**

| Attack | Disposition |
|---|---|
| Multiway formation | Covered: predecessor sizes \(a_1,\ldots,a_r\) are arbitrary. |
| Simultaneous merges elsewhere | Covered: their contributions are nonnegative. |
| Unequal final fibres | Covered: choose a largest fibre for the lower bound; serialize all fibres for the upper bound. |
| Arbitrarily much initialized scratch | Covered: deterministic scratch cannot split merged complete states or create new input atoms. |
| Controller and timing information | Must be included whenever it depends on input. Retaining it violates the clean contract unless it is determined by \(f\). A fixed circuit clock contributes no entropy. |
| Variable halting times | Use absorbing terminal states and the common finite maximum halting time. Input-dependent timing retained elsewhere must be accounted for. |
| Conditioning on currently active inputs | Changes the charged distribution. It is not the stated global joint-state cap. |
| Fibre already merged at an intermediate time | Charge its first formation earlier in the trajectory. |
| Fibre already merged initially | The theorem can fail: an initially completed constant fibre needs no further erasure. Initial injectivity is essential. |
| Nonuniform inputs | Outside scope. Two atoms of masses \(\epsilon,1-\epsilon\) can merge with loss \(h_2(\epsilon)\), arbitrarily below the uniform two-atom bound. |
| Random coins | Outside scope. Pre-sampled coins included in full initial state change the atom universe and \(M\); fresh coins invalidate deterministic coarsening on the original atoms. |
| Hidden environment or archive | Outside scope. Information stored there must be included and cleaned before applying the theorem. |

This is a genuine improvement over the telescoping condition. For example, constant output on four uniform atoms has total loss two bits. A cap \(K=0.75\) satisfies the telescoping inequality once \(D\ge3\), yet the atomic threshold is approximately \(0.811278\): **every finite depth remains impossible**.

**Charged gate-level construction — DERIVED IN THIS REVIEW**

The partition upper bound need not remain an arbitrary-merge primitive. It can be compiled into NOT, CNOT, Toffoli, and RESET with explicit costs.

This construction is new material supplied by this review. It is not implemented in the repository and has not received a further fresh proof review; its repository promotion should remain pending.

Take an explicit table for \(f\) on \(M\) distinct \(n\)-bit atom encodings, with \(m\)-bit outputs. Use registers \(d,y\), a flag \(b\), and clean comparator workspace.

First compute \(y=f(d)\) reversibly by processing table rows:

- Toggle \(b\) when \(d\) equals the row’s atom encoding.
- CNOT the row’s output-one positions into \(y\), controlled by \(b\).
- Reverse the equality test, restoring flag and workspace to zero.

An equality test is not free. After NOT gates convert zero controls, a Toffoli ladder computes a multi-controlled flag toggle and reverses its workspace. It uses \(O(n)\) gates and \(O(n)\) clean workspace.

For each fibre, choose representative \(u\). To absorb a remaining singleton representative \(v\):

1. Reversibly set \(b=[d=v]\), cleaning comparator workspace.
2. For each differing bit of \(u,v\), apply CNOT from \(b\) to that data bit.
3. RESET \(b\).

Immediately before RESET, the two relevant states are

\[
(u,f(u),0),\qquad (u,f(u),1).
\]

All other occupied states retain flag zero and distinct data encodings. RESET therefore merges exactly the accumulated \(u\)-block with the singleton \(v\). When the resulting block contains \(k\) atoms, its loss is exactly \(\phi(k)/M\). Every preceding gate is globally reversible.

After all fibres are merged, \(d\) is a deterministic representative function of \(y\). Clear it reversibly: test each output encoding, XOR its representative into \(d\), and uncompute the test from unchanged \(y\). Terminal state contains only \(f(X)\), with all ancillary registers zero.

Costs are explicit:

| Resource | Bound |
|---|---|
| Primitive gates and conservative serial depth | \(O(M(n+m))\) |
| Computational width | \(O(n+m)\) |
| RESET operations | Exactly \(M-|\operatorname{image}(f)|\) |
| Peak joint-state entropy loss | Exactly \(\phi(q)/M\) |
| Explicit input/output table | \(O(M(n+m))\) bits |
| Circuit description with encoded wire addresses | \(O(M(n+m)\log(n+m))\) bits |

The circuit generator is uniform from the explicit table. Obtaining that table from a succinct algorithm for \(f\) is additional charged work. For \(M=2^n\), these general costs are exponential in \(n\).

Independent small gate probes produced:

| Task | Width | Single-gate layers | RESETs | Peak bits |
|---|---:|---:|---:|---:|
| AND, two inputs | 4 | 45 | 2 | 0.688721876 |
| AND, three inputs | 6 | 159 | 6 | 0.517713681 |
| Parity, three inputs | 6 | 159 | 6 | 0.405639062 |
| Constant, three inputs | 6 | 160 | 7 | 0.543564443 |
| Eight atoms, fibres \((5,2,1)\) | 7 | 169 | 5 | 0.451205059 |
| Injective three-bit function | 8 | 216 | 0 | 0 |

All probes reached clean terminal states and matched the predicted threshold. These are **EMPIRICAL microchecks**, not proof certification.

This removes an absolute finite gate-realizability objection. Efficient realization remains unresolved. Indeed, for AND at its minimal feasible cap,

\[
K_*=\frac{\phi(2^N-1)}{2^N},
\]

the existing telescoping inequality already forces \(D=\Omega(2^N)\). Threshold attainment consequently supplies no fast-computation claim.

**Primary-literature audit and novelty disposition**

The nearest concrete antecedent found is Ebrahimi, Chen, and Khisti’s *Minimum Entropy Coupling with Bottleneck* (2024). Appendix A.3, Definition 1, equations 19–21 gives the same two-mass entropy-loss formula; Lemma 2 gives its monotonicity and concavity. Algorithm 1 repeatedly merges the largest masses along one search path. For uniform starting masses, that path becomes singleton absorption after its first merge. Their optimization objective differs from this all-depth feasibility question, but the merge formula and upper-path ingredients are established antecedents. [Primary paper](https://arxiv.org/html/2410.21666)

Further relevant primary sources are:

| Source | Relationship to this review |
|---|---|
| Bennett, *Logical Reversibility of Computation* (1973) | Compute, copy, reverse; distinguishes retaining input from fully erasing it. The AND ingredients carry no novelty claim. [Paper](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html) |
| Geiger and Kubin, *Information Measures for Deterministic Input-Output Systems* (2013), §II and Proposition 1 | Deterministic information loss as conditional entropy and additivity through cascades. [Paper](https://arxiv.org/pdf/1303.6409) |
| Frank, *Generalized Reversible Computing* (2018), §3.2, Theorem 6 | Support-sensitive reversibility and the role of initial probabilities. [Paper](https://arxiv.org/pdf/1806.10183) |
| Wolpert and Kolchinsky, *Thermodynamics of Computing with Circuits* | Fibre/island partitions, circuit constraints, and the distinction between joint entropy decrease and physical entropy flow. [Paper](https://arxiv.org/html/1806.04103) |
| Cicalese, Gargano, and Vaccaro, *Maximum Entropy Interval Aggregations* (2018) | Entropy optimization over aggregated probability masses, with a different endpoint and contiguity objective. [Paper](https://arxiv.org/pdf/1805.05375) |

**Novelty verdict: NOT ESTABLISHED.** The exact all-depth statement was not located in this bounded search, but that absence earns no novelty claim. The result has substantial absorption risk as an elementary consequence of existing merge-entropy analysis plus first-fibre formation. Any claimed strict delta must identify and survive comparison with these antecedents.

**Final dispositions**

| Dimension | Verdict |
|---|---|
| Pre-run implementation gate | **REPAIR:** reject optimized Python before the full run |
| AND candidate mathematics | **UPHELD / PROVED AT THE STATED CLEAN CIRCUIT SCOPE** |
| Atomic all-depth lower bound | **UPHELD / PROVED AT THE UNIFORM, EXACT, DETERMINISTIC FULL-STATE SCOPE** |
| Atomic partition upper bound | **UPHELD / PROVED AT PARTITION-COARSENING SCOPE** |
| Charged compiler supplied here | **DERIVED / PENDING FURTHER FRESH PROOF REVIEW** |
| Numerical evidence | **EMPIRICAL:** cheap probes and consistent smoke; registered full ranges unexecuted |
| Novelty | **NOT ESTABLISHED** |
| Practical usefulness | A precise cap-feasibility test and finite compiler; useful efficiency advantage unestablished |
| Mission significance | A mathematically sound component; no verified new-paradigm advantage or mission completion |

The honest one-sentence result is: **under a fixed uniform finite input distribution, exact deterministic cleanup has a minimum allowable entropy loss per step that additional running time cannot evade.**

That survives the correctness attack. It does not yet survive the novelty or mission-significance gate.

This final response is the complete report for the calling CLI’s separate output capture; I did not create or verify a capture file.