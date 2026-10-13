**COMPACT-MODEL-DESIGN-PASS.** Stationary endpoint and existing lower-bound transfers are VALID at the exact compact-model scope. The fixed-support corollary also survives the separate attack below. **SOURCE OPEN; construction CAN remains unadmitted. No repair is required or consumed by this gate.**

Both hashes matched at entry and final recheck:

```text
MODEL
217c867350cd8d8d5be1820269639d9698cc6f0e5aa675f34242e6f03a1cf00d

PROTOCOL
8059b17fd88f34d82f7e9d6f61ccd474a94c7a5385c06815c78d80dc5ab108db
```

**1. MODEL / DESIGN / INFINITY — PASS at definition scope.**

The [frozen model](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_COMPACT_PROBE_MODEL_CANDIDATE.md) closes the relevant information and infinity leaks.

READ acquires exactly one persistent bit, with every acquisition and reread charged. Direct copy, Boolean operands, scans and indirect operands receive no free DATA access. WRITE costs one probe even when the value stays unchanged, supplies no old value or status, and therefore cannot hide read-modify-write acquisition.

Wear counts every actual value change, including INIT, transient changes and changes subsequently undone. Exhausted opposite writes are INVALID; they supply no saturation behavior, exception, timing branch or failure feedback. Hidden wear remains accounting state rather than controller memory.

One fixed finite, parameter-independent, deterministic computable program receives finite public parameters and current input. Parameter-growing ROM, installed policy tables, input advice, invocation ordinals, clocks, prior outputs and output positions are excluded. Every mutable address, operand, transcript, PC, input/program head, path frame and cleanup variable belongs to \(S=256q^2\) throughout preparation, INIT and commands.

Actual promised termination and actual restoration of the parameter-only canonical image are explicit obligations. Declaring local work free in the probe coordinates neither makes a divergent computation terminate nor clears its scratch. It changes price without enlarging computability or granting additional space.

The registered preparation-plus-INIT bound \(2^{20}Nq^6\) counts ordinary bit work, including source/input reads, local operations, control transitions and persistent probes. Fixed program provisioning must be reported. **The bound is an acceptance requirement; no routine meeting it is established here.**

The release of the old microstate, description and CODE-generation restrictions is explicit. Consequently, this definition does not reinterpret or confirm the closed old RAM unit.

The reserved workspace arithmetic is sound as **CAPACITY ONLY**:

\[
96q+4q(q+1)+8v+32q^2+64q
\le36q^2+204q+48\le256q^2
\quad(q\ge2).
\]

It does not establish that an actual program’s simultaneous temporaries, heads, control and clearing routine fit that inventory.

**2. LOWER TRANSFER — VALID, including arbitrary nonlinear programs.**

At completed boundaries, canonical local state, fixed program, identical current command and absent wear/history feedback leave DATA as the only history-bearing input. Thus promised QUERYs define one stationary decoder \(d(s)\); INIT defines \(I(x)\); promised FLIPs define stationary endpoint maps \(T_i(s)\).

Equal DATA states cannot represent different logical vectors, because their cold exact queries execute identically. Differing hidden wear cannot select different valid successors: any attempted exhausted opposite write would invalidate a promised execution.

For actual initialization wear \(w_0\) and command change-count vector \(c_t\),

\[
I(x)\le w_0,\qquad s_t\oplus s_{t+1}\le c_t.
\]

Endpoint accounting therefore relaxes actual wear. Every reachable wear record remains necessary; one representative or averaged wear vector per DATA state is insufficient. Arbitrary off-promise extensions are mathematical devices, without executable or termination claims. States reached only at depth \(H\) require decoding; states also reached earlier retain their promised stationary transitions.

The existing admitted arguments transfer:

- **A — VALID:**
  \[
  H\le L\min(N,2^{N-k+1}-1)-1.
  \]
  Adjacent-fibre averaging and a distinguishing coordinate projection apply to arbitrary decoding. Selecting an initializer with a projected one supplies the initialization charge; repeated FLIPs across that logical edge supply the remaining \(H\) charges.

- **B — VALID:** with
  \[
  t_0=\min\left\{t:\sum_{j=0}^{t}\binom Nj\ge2^k\right\},
  \qquad \beta=\lceil\log_2(k+1)\rceil,
  \]
  \[
  H\le\max(0,NL-t_0-\beta+1).
  \]
  Injective initialization supplies \(t_0\). At the final branching boundary, \(k\) distinct logical successors require at least \(\beta\) cells with remaining wear. These are alternative branches, not consecutively executed updates.

- **C — VALID, source-derived:**
  \[
  H\le(N-k+1)L+\left\lfloor\frac{(k-1)L}{2}\right\rfloor.
  \]
  Cumulative actual changes give monotone levels with DATA as their parity. At lifted zero, initialize raw \(0^k\), restore canonical entry and execute the first changing request, charging both phases. Later required traces depend only on parity and current request. Unchanged-value requests leave levels unchanged. This satisfies the primary zero-start contract with physical alphabet \(L+1\); it supplies no reverse construction granting readable wear to the binary controller. [Floating Codes, definition and Theorem 2](https://people.engr.tamu.edu/ajiang/Floatingcodes.pdf).

**Adaptive query law — VALID for every \(i\):**

\[
H+1\le L\min(N,2^{r_i}-1).
\]

Contract finite deterministic computation between persistent observations into a reachable binary decision tree. Its depth is at most \(r_i\), so its position union \(P_i\) contains at most \(\min(N,2^{r_i}-1)\) cells. Reachable states agreeing on that union produce the same answer. Some initializer has a one there; every subsequent FLIP\(_i\) changes the projected endpoint. Actual initialization and intermediate wear supply at least \(H+1\) changes within that support.

**Fixed-support / nonadaptive clause — VALID, separately attacked here:**

\[
H+1\le L\min(N,|P_i|).
\]

Here \(P_i\) is a fixed parameter/index-dependent support containing every possible QUERY\(_i\) acquisition. Agreement on this support forces equal answers, regardless of nonlinear local computation. Some initializer has nonzero projection; each repeated FLIP\(_i\) must change that projection. Summing **actual changes within the support** proves the inequality.

This does not assert that every support cell changes on every update, nor confuse total variation with a distinguishing-coordinate argument. A nonadaptive address sequence of length at most \(r_i\) has \(|P_i|\le r_i\). An adaptive reader taking \(r_i\) probes per invocation can have a much larger union across histories.

The admitted projection proof covers this exact corollary; **this fresh gate supplies its separate exact-scope attack, so another proof gate is unnecessary for this clause alone.** Stronger tradeoff or incumbent-separation claims require their own evidence.

The depth-two obstruction likewise transfers at \(N=3,k=1,L=2,H=5\): its decision-tree classification, wear saturation and stationary-walk argument use precisely these retained premises. The flat-log result remains an admitted **endpoint** capability and wear-accounting component; it does not become a compact-model CAN without the uniform program, space, preparation and cleanup certificates.

No unconditional write-amplification law transfers.

**3. SOURCE — OPEN, independently of model validity.**

Targeted primary inspection confirms substantial occupied ingredients:

- **DSST:** node copying, constant amortized allocation, controlled ancestor propagation without stored inverse pointers in trees, ordinary partial red-black color overwriting, and a write-once extension with an access-pointer exception are already present. Root publication must pay that exception; eliminating colors does not establish novelty. [DSST §§2.3, 4.1 and 6](https://www.cs.cmu.edu/~sleator/papers/making-data-structures-persistent.pdf).
- **Crit-bit ancestry:** external key leaves, critical-position internal nodes and compressed branching are established. Their use here needs the strict-wear construction certificate. [Bernstein’s primary description](https://cr.yp.to/critbit.html).
- **Flash logs and folds:** changed-bit index logging, parity storage and phase folding are explicit. Readable monotone cell levels do not directly implement hidden binary wear or the frozen cold probe/workspace contract. [Rewriting Codes §VI](https://yaakobi.net.technion.ac.il/files/2015/04/8_Rewriting_Codes_IT14.pdf).
- **TCC second copy:** the primary explicitly schedules an additional working copy to obtain worst-case locality under its Buffered-Hamming contract, while distinguishing the Prefix-Hamming limitation. It does not supply this all-history cumulative wear ledger. [Locally Updatable and Locally Decodable Codes §3.3](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/03/520.pdf).
- **Wear leveling:** paid labels coexist with surviving wear-free counters/randomness in the ISAAC construction. The DISC counter permits stale selection variables with slower recovery, so coldness requires translation rather than automatic exclusion. [ISAAC model, Theorem 3 and algorithms](https://drops.dagstuhl.de/storage/00lipics/lipics-vol181-isaac2020/LIPIcs.ISAAC.2020.65/LIPIcs.ISAAC.2020.65.pdf), [DISC §3](https://drops.dagstuhl.de/storage/00lipics/lipics-vol281-disc2023/LIPIcs.DISC.2023.38/LIPIcs.DISC.2023.38.pdf).
- **Write-efficient dictionaries:** asymmetric-memory search trees already achieve constant amortized word updates after searching. Comparison-based external-memory lower bounds retain their restricted element and block-transfer assumptions; they do not become arbitrary nonlinear bit-encoding lower bounds. [Asymmetric-memory §4](https://arxiv.org/pdf/1511.01038), [Brodal–Fagerberg model and Theorem 1.1](https://cs.au.dk/~gerth/papers/soda03.pdf).

An equivalent incumbent comparison must charge initialization, cold recovery, all persistent metadata, bounded private workspace, worst-case literal probes and every cell’s actual wear identically. Conventional adaptive incumbents remain included.

These targeted sections establish neither global novelty nor complete absorption. Failure to locate an equivalent theorem supplies **no DELTA**.

**4. CONSTRUCTION DESIGN OBLIGATIONS — coherent attempt; proof debt remains OPEN.**

The [frozen protocol](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_COMPACT_PROBE_PROTOCOL.md) specifies a credible bounded family, without supplying its construction theorem. The remaining debts are:

- **Effective fields and latest-root semantics:** specify box precedence and copy overflow with the requested change already installed in the fresh base. Historical NEW roots are publication records, not immutable snapshots. OLD’s final root and effective fields stay frozen during folding.
- **Paths and aliases:** prove increasing critical positions, correct first-difference splice, unique current parents/leaves, unchanged sibling sharing and validity of every retained private ancestor ID throughout bottom-up copying.
- **Allocation:** establish at most \(3j\) allocations for every prefix, including detached copies and intermediate allocation peaks. The filled-live-box potential is a plausible argument, conditional on the live-tree invariant. \(C=4\) is arena capacity, not a four-allocation per-command bound.
- **Exact layout and sizing:** account for every type/key/critical-position/child/parity/publication and box bit within \(v=d+e+3a+6\), plus all root records, base, ROLE and STARTED. Certify null/ID encodings, unused cells, zero-width fields, maximal power-of-two \(M\), and direct-base fallback. The existing sizing arithmetic reveals no design contradiction; neither \(ML\ge H_0\) nor fallback coverage is admitted as a construction theorem here.
- **Physical-ID folding:** distinguish internal nodes and unpublished slots; fold a leaf only when frozen OLD lookup returns that exact active physical ID. Prove one active leaf per touched key and exactly-once folding despite garbage copies. Retain zero-parity keys.
- **Cutoff and clearing:** establish the cutoff \(2Cj\), complete scanning by \(M/2\), and **no OLD acquisition after clearing starts**, including root, flag and allocator reads. Charge every clearing WRITE, including same-value writes.
- **Individual-cell wear and publication:** prove final-buffer writes, single-use boxes, staged flags/root publication, alternating fill/clear wear, base initialization plus later folds, STARTED’s single change, and **ROLE’s \(L\) exchanges**, including the final epoch. Maintenance, clearing and exchange finish before canonical acknowledgement.
- **Actual program resources:** supply the fixed uniform finite program, simultaneous workspace mapping, terminating routines, self-clearing control/head tail, numerical \(R_Q\) and \(R_F+W_F\) certificates, and preparation-plus-INIT bound. Asymptotic lookup/copy estimates do not discharge these numerical obligations.

No defect found here requires spending the unit’s sole repair. Root can proceed to the **one registered bounded construction attempt**, preserving this review verbatim. Fresh independent construction-proof review remains mandatory before any CAN promotion.

I read the full constitution and dual-loop skill, prescribed startup authority, compact documents, both actual outcome dialogues through rounds 1–3, named proof/review families, source pretests, fat-node sketch and targeted primary sections. CERT admission remains finite CANNOT provenance without H4 achievability; old RAM remains closed FAIL. This review performed no writes, agents, GPU work, solver or numerical outcome run, code construction, or terminal review.