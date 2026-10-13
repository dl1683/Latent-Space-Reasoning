**A: PASS. B: PASS. Ordinary \(N=k\) lifetime: PASS. C: PASS — primary transfer VALID.** These are proof admissions at the binding machine’s scope, not mission completion.

I reviewed the named authority, registration, steering, source-comparison and actual boundary-review files, plus the primary floating-code definition and Theorem 2. Reviewed universal candidate SHA256: `288da6942fa7bcdaa4896fda00c49bb13d9ad4bc91c1e54e42df117686adce85`. The binding model matches its admitted hash.

All verdicts cover arbitrary nonlinear representations, initialization for every raw vector, arbitrary multi-bit endpoints and intermediate changes, and every history-bearing controller bit within the \(N\) cells. They require canonical volatile entry, terminating promised operations, unlimited read-only QUERYs, and strict exhaustion without wear feedback, as specified in the [binding model](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md:16). No algorithm receives the proof’s projection, wear records, or raw-input knowledge after initialization.

**A — PASS: universal fibre/projection bound, including initialization.**

\[
H\le L\min\!\left(N,2^{N-k+1}-1\right)-1.
\]

Canonical entry and wear-independent read-only QUERYs determine a single decoder on reachable boundary states. Correctness prevents one physical state from representing two logical vectors. Extending that decoder arbitrarily elsewhere produces a partition of all \(2^N\) strings, with every logical fibre nonempty.

Each fibre appears on exactly \(k\) logical-cube edges. Consequently, the average adjacent-pair union has size \(2^{N-k+1}\), so some adjacent pair has union size \(s\) no greater than that. A distinguishing coordinate projection needs at most \(\min(N,s-1)\) coordinates: each added coordinate splits an unresolved class, and no more than \(s-1\) splits are needed.

The two initializer endpoints have different projections. Choose the side whose projection contains a one. Its **actual** initialization consumes at least one projected wear unit from unworn zeros. Repeating the pair’s differing FLIP keeps completed endpoints in the two fibres; each completed update changes at least one projected coordinate. Intermediates may leave those fibres and undo changes, but only increase actual wear. Thus

\[
H+1\le L|P|,
\]

which proves the bound. The adjacent pair, projection and initial side are mathematical witness choices, not information supplied to the controller. The \(H=0\) case also survives.

**Ordinary \(N=k\) baseline — PASS at lifetime scope.**

A gives \(H\le L-1\). Direct storage attains that horizon: initialize each cell to its corresponding raw bit, then toggle the addressed cell on FLIP. For any input and any sequence of \(L-1\) flips, each cell incurs at most one initialization change plus \(L-1\) update changes. Queries remain valid after exhaustion because they only read.

The finite address, control and clearing routine can fit the declared scratch allowance with charged work. This establishes the ordinary wear optimum \(H_{\mathrm{opt}}=L-1\); it does not admit the journal construction or its command-cost claim.

**B — PASS: initializer packing plus all final branches.**

With

\[
t_0=\min\left\{t:\sum_{j=0}^{t}\binom Nj\ge2^k\right\},
\qquad b=\lceil\log_2(k+1)\rceil,
\]

the admitted ceiling is

\[
H\le\max(0,NL-t_0-b+1).
\]

Correct decoding makes initialization injective. Fewer than \(2^k\) strings have weight below \(t_0\), so some initializer endpoint has weight at least \(t_0\); its actual initialization consumes at least that much wear.

For \(H\ge1\), follow any \(H-1\) flips from that input. Each changes the logical vector and therefore consumes at least one physical change. At the resulting boundary, all \(k\) possible final FLIPs must succeed.

Let \(u\) cells retain positive wear. Exhausted cells cannot change on any valid branch. Hence those branches can reach at most \(2^u-1\) endpoints distinct from the current state. Their \(k\) logical neighbors are mutually distinct, forcing

\[
k\le2^u-1,\qquad u\ge b.
\]

Remaining wear is therefore at least \(b\), giving

\[
NL\ge t_0+H-1+b.
\]

These are alternative branches from one boundary; the proof does not execute all of them consecutively. Intermediate changes cannot enlarge the endpoint count. The argument requires \(H\ge1\); when its expression is negative, it excludes positive lifetime without excluding \(H=0\).

For \(N=4,k=2,L=2\), \(t_0=1\) and \(b=2\), yielding **\(H\le6\) as a proved symbolic ceiling**. This supplies neither solver evidence nor achievability.

**C — PASS: primary transfer VALID, with the zero-start wrapper made explicit.**

\[
H\le (N-k+1)L+\left\lfloor\frac{(k-1)L}{2}\right\rfloor.
\]

The primary starts both physical cells and logical variables at zero, uses stationary decoder/update functions, permits coordinatewise monotone endpoint increases, and defines lifetime by success for every rewrite sequence. Its Theorem 2 applies here with \(n=N\), logical alphabet size \(2\), and physical alphabet size \(Q=L+1\); \(N\ge k\) satisfies its first-case premise. [Primary definition and Theorem 2](https://people.engr.tamu.edu/ajiang/Floatingcodes.pdf).

The transfer survives the following attacks:

- **Zero default.** Choose the candidate’s permitted fixed raw input \(x_0=0^k\), and set the lifted zero-state decoder to \(0^k\). An arbitrary \(x_0\) also works after XOR-relabeling logical values by \(x_0\); merely assigning an arbitrary nonzero default would miss the source condition.
- **Paid first changing request.** At lifted state \(w=0\), perform the original initializer for \(0^k\), restore canonical volatile entry, then perform the requested FLIP. Add **all actual changes from both phases** to the lifted levels. This avoids invoking FLIP on a potentially unreachable original binary zero state.
- **Stationarity and merged histories.** Every changing request makes \(w\ne0\), and levels never decrease. Thereafter the physical bits are \(w\bmod2\). Canonical entry and absence of wear feedback make the original trace—and its change-count vector—a function of those bits and the command. Histories merging at the same \(w\) therefore require the same lifted update.
- **Full request alphabet.** A request assigning the current logical value returns \(w\) unchanged. Otherwise, binary assignment invokes FLIP. Thus every source sequence of at most \(H\) rewrites entails at most \(H\) original flips, including sequences with unchanged-value requests. No external invocation index is needed.
- **Total maps and irrelevant states.** Define the encoder from the finite terminating traces on required states and extend it elsewhere. Do not simulate a possibly divergent original invocation outside its promised domain. The source lifetime guarantee concerns sequences from zero through the guaranteed horizon; it does not require successful changing rewrites from every irrelevant or exhausted vector.

Every lifted endpoint is monotone and bounded by \(L\), and its parity matches the original endpoint even after multi-bit or undone intermediate changes. Therefore the resulting floating code guarantees at least \(H\) rewrites, allowing Theorem 2 to bound \(H\).

Observable cumulative levels are granted only to this **upper-bound relaxation**. They provide no wear knowledge or construction to the actual binary algorithm. C is a valid source-derived ceiling, not a novelty claim.

The honest result is universal lifetime ceilings plus the ordinary zero-redundancy optimum. Strict primary delta, useful-world significance, tight loss and low-redundancy achievability remain open. The positive journal’s **ELEMENTARY-CONTROLLER-COMPILATION** repair remains outstanding; this review neither bypasses it nor authorizes calibration. The keep-going objective remains active.

Read-only throughout: no edits, agents, GPU, solver, scientific implementation, probes or FULL run. Git status remained clean.