| Gate | Verdict |
|---|---|
| **LOWER** | **PASS — PROVED AT SCOPE** |
| **UPPER** | **PASS — PROVED AT SCOPE**, including arbitrary X/R correlation |
| **CODE** | **REPAIR — registered FULL serialization fails under the observed Python limit** |

CPU-only, read-only review. No edits, commits, additional agents, GPU use, or FULL execution.

**The blocking code finding is reproducible without running large circuits.** At the last R=0 cleanup RESET under uniform joint input, the two AND-zero classes have multiplicities \(a=2^{n-1}\) and \(a-1\). Their exact loss ratio is

\[
Q_n=\frac{(2^n-1)^{2^n-1}}
{a^a(a-1)^{a-1}}.
\]

The fraction is already reduced. Independent multiplicity arithmetic produced:

| n | Numerator digits | Denominator digits | `str(Fraction)` at observed limit 4,300 |
|---|---:|---:|---|
| 10 | 3,080 | 2,772 | succeeds |
| 11 | 6,778 | 6,162 | **ValueError** |
| 12 | 14,793 | 13,560 | **ValueError** |

[The trace serialization](/<LOCAL_PROJECTS>/new-computation-model/experiments/fair_bit_cleanup.py:82) therefore fails on valid registered cases. Python 3.13.7’s observed limit was 4,300; this behavior matches the [official conversion-limit documentation](https://docs.python.org/3.13/library/stdtypes.html#integer-string-conversion-length-limitation). An injected derived-ratio serialization failure exercised the actual main failure path: RUNNING → VOID, return 1, with the failing unit recorded.

A finite limit of **32,768** suffices for the frozen inventory: \(D\le8192\), and both ratio integers are bounded by \(D^D\), whose maximum decimal length is 32,059. Both large ratios round-tripped exactly under that limit. Incorporate and record a sufficient limit—or use an exact serialization without this restriction—cover the independent reader too, then repeat the bounded smoke and code gate. **FULL remains blocked at the reviewed launch contract.**

**LOWER.** Every data input must be a gate target: otherwise an input initially equal to one remains uncleared. The supplied R input must also be targeted, and each distinct output must be targeted because each can equal one. Every admitted elementary gate has exactly one target. Consequently,

\[
G\ge n+3,\qquad W\ge n+3.
\]

Together with the construction, this establishes \(\Theta(n)\) gate work and width. It does not establish optimal constant factors. The earlier no-R exponential theorem remains intact for its capped-X contract, **n≥3**; adding the returned fair marginal changes the statistical resource.

**UPPER.** Let \(P=[R=c]\), \(B\) be the current data bit, and \(D=0\) the disposable bit. The three-gate cleanup macro produces

\[
D=PB,\qquad B'=(1-P)B,
\]

then resets D. Let T contain every other register immediately before RESET. Complete-state entropy loss is exactly \(H(D\mid T)\).

At every RESET, the returned output contains the original R, so T determines P. For \(R\ne c\), D is identically zero. Hence

\[
\Delta H
=\Pr(R=c)\,H(D\mid T,R=c)
\le \Pr(R=c)=\tfrac12.
\]

This uses the **current complete state**, including outputs, predicates, and distinctions surviving previous erasures. Arbitrary initial correlation cannot alter the pointwise-preserved R marginal. No independence assumption enters the proof.

The two initial CNOTs transfer R and clear its input. The balanced AND tree uses \(n-1\) clean targets; copying its root and reversing the tree clears every tree target before `pool[0]` becomes the prefix flag. Prefix operations restore temporarily flipped R output bits before every RESET. Each macro clears its disposable; the two context sweeps clear every X bit; final uncomputation clears the prefix and sentinel. This holds pointwise for **every valid input encoding**, including zero-prior inputs.

Exact gate counts are:

| Kind | Count |
|---|---:|
| CNOT | \(2n+3\) |
| Toffoli | \(4n+2\) |
| NOT | 6 |
| RESET | \(2n\) |
| **Total** | **\(8n+11\)** |

The declared allocation is exactly **\(2n+7\)** bits, including unused allocated scratch. The half-bit bound is attained by this construction: give \((X,R)=(00,0),(01,0),(00,1)\) masses \(1/4,1/4,1/2\). Its first active cleanup RESET loses exactly \(1/2\) bit.

**The general r-bit paragraph separately passes at its explicit assumptions.** Every complete context must satisfy \(\Pr(R=z)\le K\). Compute-copy-uncompute f, then apply the same cleanup within each preserved context. The identical full-state argument bounds each RESET by that context’s mass. Reflected Gray updates have total changed-suffix length \(2^{r+1}-r-2\), so charged predicate maintenance is \(O(2^r+r)\). Thus the stated

\[
O(C_f+n2^r+r+2^r)
\]

bound follows, with f’s reversible implementation, workspace, and program work separately charged.

Fair coordinate marginals do **not** supply a \(2^{-r}\) joint-mass bound. Independently tested \(R=00\) or \(11\), each with probability \(1/2\): both coordinates are fair, but cleanup can lose \(1/2\) bit, violating a purported \(K=1/4\) guarantee. Nine separately constructed AND/parity/majority controls for r=1..3 passed their complete-state checks.

Vertex enumeration is also insufficient. Two fair-law vertices each gave zero loss; their midpoint gave \(1/2\) bit. Conditional entropy is concave, and its maximum need not occur at a vertex. The registered controls support falsification, not universal-law certification.

**Bounded implementation evidence.**

- Independently matched the saved smoke’s six law traces and 32 three-atom priors: **5,520 transitions and 190 RESET distributions**.
- Including additional small circuits, **28 non-dyadic rational laws** with denominators 6 and 30, and boundary witnesses: **29,880 transitions and 472 RESET distributions across 77 laws**.
- Rejected seven wrong-output/dirty-workspace/gate-stream attacks and seven arity, operand, width, and count defects.
- Verified the FULL inventory **without evaluating it**: n=2..12 and \(48+448+3840=\mathbf{4336}\) distinct registered priors.
- Verified all launcher dependency branches pass the captured byte objects to compilation and inject their hashes. Snapshot-read failure executed zero modules.
- Startup/end source mismatches produced VOID; injected case failures stopped at the first failing unit and preserved completed units. RUNNING replaced the modeled historical CONFIRM before computation.
- Actual `-O`/`-OO` processes and optimized code objects rejected execution. Actual invalid output paths rejected publication. Filesystem publication and hostile link races were not newly certified; shared IO remains within its previously reviewed trusted-host scope.

An actual child smoke, with publication confined to memory, exited **0** for attempt `f21c334d-205e-4f3f-a75b-44a899b2924e`. Another actual child published CONFIRM and then exited **1** after an injected stdout failure. Initial publication failure can leave historical CONFIRM; final publication failure can leave RUNNING. **Successful exit for the matching invocation remains mandatory.**

The [saved smoke receipt](/<LOCAL_PROJECTS>/new-computation-model/results/fair_bit_cleanup_smoke_exit_receipt.json) matches attempt `f3de6b97-6baf-46e9-9ce0-c65abd9efed7`, artifact bytes, launcher, dependency hashes, and inventory. Its zero exit is the saved external observation; this review verified its bindings.

**Exact SHA-256 fingerprints**, unchanged at the final check:

```text
experiments/fair_bit_cleanup.py
027281b66b018bd094ee7f4cde599b7469022d609b940a9738328a0075dd6aa0

experiments/succinct_erasure.py
a8c1d33a98822de83792f344842c9ac90b709134eb6cb10d012d71b6f3f0e71c

experiments/evidence_io.py
2bf4310800821fabf69f9a95bb71561ddf1a595d9263ec6f8a3fa90d1675386f

experiments/run_snapshot.py
c1e6877d532cd4ec7e17323a1a88af50bde0c73842df98a87027ad808610035b

research/RETURNED_FAIR_BIT_CANDIDATE.md
b5deba503b40fbd1aaa0744d2bacc149d104ecd1475123c772027a231a182525

results/fair_bit_cleanup_smoke.json
dc21e55099b82ed26c2bf0569c43023f3901b23dc36bce0aaa725c43746c90ba

results/fair_bit_cleanup_smoke_exit_receipt.json
9eb7d8cc28b8ea78491144a09eabb0ceda4b7be09b181bbc353b1fadefaf37de
```

The **old launcher** at both `6bf05cc` and `b5d79a9` hashes to:

```text
2681e422d7cb5ba7b50f61a7bb973b6e308cd7a4f9b8da7386437c48038406a1
```

That matches the old FULL environments and receipts. Today’s launcher is not attributed to those attempts.

**Infinity, triviality, and mission debt.** The fixed compiler has finite registers, bounded-arity gates, a fixed schedule, and no probability oracle, resampling, or uncharged predicate. Universal quantification over finite probability laws supplies no machine-accessible infinite precision. Preparation and authentication of the exact marginal remain explicit resource obligations; finite trials cannot certify that promise.

The strongest dismissal is **ensemble scheduling**: preserve one fair bit and spread irreversible operations across its two contexts. Conditional on the active context, a RESET may still discard one bit. Total logical loss remains \(H(X\mid\mathrm{AND}(X),R)\). The composition works beyond AND whenever f has a suitable reversible implementation, substantially limiting an AND-specific breakthrough interpretation.

Priority remains unresolved against existing [statistical-context reversibility](https://arxiv.org/abs/1806.10183) and [energy-aware computation](https://erikdemaine.org/papers/Energy_ITCS2016/paper.pdf). This review establishes no same-resource simulation separation, device-heat saving, ordinary computation speedup, or whole-repository completion. The result is a rigorous composition theorem and pressure test of the meter; useful-world and new-paradigm standing remain open.