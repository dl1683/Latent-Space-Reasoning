**SOURCE OPEN. Construction: `ENDURE-DESIGN-REPAIR`.**  
Named repair: **ELEMENTARY-CONTROLLER-COMPILATION**.

Neither `ENDURE-ABSORBED` nor `ENDURE-DOMINATED` is established. I found no concrete boundary-state counterexample warranting `ENDURE-ROUTE-KILL`; `ENDURE-VOID` does not fire. `ENDURE-OPEN-REGIME` remains applicable to the separate low-redundancy frontier.

These are **HEURISTIC design/source judgments**, not lifetime or theorem admissions.

I read the constitution, full dual-loop skill, current authority, actual October4 advisor/architect round1/2 responses, decision, registration, candidates, PRETEST, INITIAL_SOURCES and actual boundary review. The registration exists in both EXPERIMENTS and the ledger. Reviewed journal SHA256:

`705c445ccfcbe217c7efc783208e99ea35e043a548cb96c507666501221e23c0`

The boundary candidate still matches its admitted SHA256.

**The blocking repair**

The [compiler refinement](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_JOURNAL_DESIGN_CANDIDATE.md:165) removes multiplication at every node visit, which addresses a real cost threat. It does **not yet justify a uniform elementary implementation with the complete cubic tally**.

The reservation
\[
64(2q)+q(2q)+16q^2+16q^2
=34q^2+128q\le98q^2
\]
is numerically correct for \(q\ge2\). But reserved capacity is not an allocation showing that every live register, selector, instruction buffer, PC, return address and helper temporary fits.

Likewise, polynomial program size and an \(O(\log q)\)-bit PC establish neither constant-cost dispatch nor free access to an immediate private location. The draft leaves unspecified:

- how a selected scalar/frame bit is reached without an uncharged array operation;
- how instruction bits and operand locations are fetched and decoded;
- how PC changes, branches and helper returns are expanded;
- the simultaneous scratch allocation and clearing procedure;
- numerical program-size bounds and a declared fixed initialization/installation exponent \(c_{\rm init}\).

A straightforward implementation charging \(O(\log q)\) control overhead to each of a claimed \(O(q^3)\) elementary operations establishes only an \(O(q^3\log q)\) bound. That is **an attack on the supplied accounting, not a lower bound against every possible compiler**. A tighter decomposition could absorb control costs into persistent routing, or use a suitable generated execution schedule; neither is presently supplied.

Use the registered single repair to provide explicit instruction/helper schemas, private-location selection, code-fetch/PC accounting, a simultaneous numerical scratch map and the resulting worst-case tally. This can be a finite compiler specification; scientific implementation or synthesis is unnecessary for this repair. Until fresh regating, `ENDURE-DESIGN-CONFIRM` does not fire.

**What survived the design attack**

The [layout and schedule](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_JOURNAL_DESIGN_CANDIDATE.md:19) survived the following checks at design scope:

| Obligation | Finding |
|---|---|
| Fields and addresses | \(a\) bits suffice for physical pointers and counts. For journal start \(B>0\), slot \(u\) can have base \(B+(u-1)(3a+1)\); the paid layout bounds its fields inside the region. Null address zero remains distinct. |
| Exact allocation | Bottom-up allocation of precisely \(d+1\) slots makes slot \(j(d+1)\) the root. Null paths require no extra nodes; immutable sibling pointers stay within the same journal. |
| Key representation | Padded keys are never inserted. A touched leaf with parity zero must retain count one; otherwise hot-key repetitions could corrupt rank accounting. The draft preserves this distinction. |
| Publication | Prefix search recovers \(j\) without a persistent allocator/root word. Publishing the flag before maintenance finishes is permissible only because acknowledgement follows maintenance and exchange; no intermediate boundary or crash guarantee is claimed. |
| Folding | Sorted ranks advance over distinct touched keys, including zero-parity leaves. The rank-\(2j\) cutoff excludes exactly the folded prefix. Repeated hot keys and alternating keys introduce no double folding. |
| Clearing | \(M/2\) chunks of \(2P\) cover exactly \(MP=J\) bits. Once \(j\ge M/2\), QUERY must branch around **all** OLD root/count/flag reads, as the draft requires. |
| Startup and exchange | `started=0` excludes the initially empty OLD bank. At update \(M\), clearing finishes and role changes before acknowledgement. The next NEW bank has zero flags; the full journal becomes OLD. |
| Query interleavings | Queries perform no maintenance and preserve persistent state. Arbitrarily many queries therefore neither advance folding nor consume wear, including after flip \(H\). |

The rounding constants also survive. In the journal branch,
\[
M>\frac{k-2}{4P}\ge\frac{N}{16P}
\ge\frac{N}{320(\log_2N)^2}.
\]
Here \(d+1\le a\), \(P\le5a^2\), and \(a\le2\log_2N\). If \(M\ge4\) does not fit, \(k-2<8P\) gives \(N<324(\log_2N)^2\); consequently \(H=L-1\ge L/2\) covers the frozen \(1/1024\) target wherever that branch’s target domain is nonempty. The \(k=1\) XOR branch independently supports its proposed \(H=L\).

The **candidate** every-cell ledger is consistent with \(L\) epochs:

- Base: initialization contributes at most one change; epochs \(2,\ldots,L\) contribute at most one fold change each.
- Journal cells: fill and clear occur in alternating epochs, with at most one change per epoch.
- Role: exactly \(L\) changes through the final exchange.
- Started: at most one change.

This requires writing each fresh persistent field bit directly to its final value, with arithmetic performed privately. Incrementally constructing a persistent field could introduce extra intermediate changes and invalidate that ledger. Reads after exhaustion remain legal; thus queries after the final exchange introduce no wear problem. **The proposed \(H=ML\) remains unpromoted.**

**Source comparison**

The strongest threats are substantive:

| Primary | Occupied contribution and remaining transfer |
|---|---|
| Flash2014 §VI | Already records changed-bit indices and folds into a parity bank, with equation (8) giving roughly \((n-k)(Q-1)/\log_2k\) writes. Readable monotone levels cannot be replaced by hidden binary wear using \(Q=L+1\). Local queries, deamortized maintenance and paid boundary recovery still require construction. [Primary](https://yaakobi.net.technion.ac.il/files/2015/04/8_Rewriting_Codes_IT14.pdf). |
| ISAAC2020 | Theorem3 supplies near-total **word** lifetime with high probability, paid labels, expected/amortized access overhead and surviving wear-free words. Flattening words can conservatively transfer wear accounting, but does not remove randomness or the surviving controller, nor establish worst-case elementary command work. [Primary](https://drops.dagstuhl.de/storage/00lipics/lipics-vol181-isaac2020/LIPIcs.ISAAC.2020.65/LIPIcs.ISAAC.2020.65.pdf). |
| ELM | Current-state-only encoding, zero-error constructions and finite transition budgets are occupied. Definition1 supplies stage-indexed maps; Construction19 composes WOM codes. Paid stage recovery, local FLIP histories, execution costs and strict invalid-exhaustion semantics remain transfer obligations. [Primary](https://arxiv.org/pdf/2109.09932). |
| Floating/flash | Already provides current-state update maps and decoding for arbitrary logical flip sequences. Monotone levels, initialization and published scanning procedures prevent an automatic transfer to this retained contract. [Primary](https://arxiv.org/pdf/1210.7515). |
| Local rewriting | Bounds rewriting cost while masking stuck-at defects; it does not supply this cumulative hidden-wear lifetime guarantee. [Primary](https://arxiv.org/pdf/1602.01202). |
| AVR101 | Paid status rings recover progress after reset. Its scanning recovery and parameter-ring organization do not establish constant-redundancy pooling against arbitrary hot coordinates. [Primary](https://ww1.microchip.com/downloads/en/appnotes/doc2526.pdf). |
| Persistent counters / Start-Gap | The counter uses transactions and retained static selection state; its task differs. Start-Gap’s deterministic lower concerns one-to-one placement with one spare, while strong positive guarantees use randomness, workload restrictions or an unlimited-endurance cache. Neither supplies the full incumbent sought here. [Counter](https://drops.dagstuhl.de/storage/00lipics/lipics-vol281-disc2023/LIPIcs.DISC.2023.38/LIPIcs.DISC.2023.38.pdf), [Start-Gap](https://ics.uci.edu/~goodrich/pubs/J-89.pdf). |

Thus logging, folding and paid recovery are occupied ingredients. **No full-contract absorption or domination was established; this establishes no novelty.**

The honest surviving story is a plausible storage mechanism that spreads hostile updates while its controller shares the cells’ endurance, with elementary controller execution still unadmitted. It remains a candidate useful CAN component. The universal nonlinear ceiling, tight-loss and low-redundancy questions remain separate OPEN obligations; no matched CAN/CANNOT pair or mission completion follows.

Read-only throughout: no edits, agents, GPU, scientific implementation, solver, FULL run or theorem promotion. Git status was clean at both checks. Continue with the named repair and fresh design regate; retain the useful positive target and universal CANNOT objective.