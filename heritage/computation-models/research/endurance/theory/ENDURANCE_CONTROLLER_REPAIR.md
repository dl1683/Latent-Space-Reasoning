# ELEMENTARY-CONTROLLER-COMPILATION — repair failure

Date: 2026-10-04. Unit: ENDURANCE-REGIME-CAN-DESIGN-1.

**Disposition: REPAIR-FAILURE. The ONE registered repair did not supply a complete legal elementary compiler with the frozen cubic worst-command bound.**

Rigor: **HEURISTIC / DESIGN ACCOUNTING ONLY.** This is a failure to discharge the scaffold’s compilation obligation. It is neither a boundary-state counterexample nor an impossibility theorem about all compilers or all admitted machines. I therefore do not assert the registration’s concrete-counterexample meaning of `ENDURE-ROUTE-KILL`.

`ENDURE-DESIGN-CONFIRM` does not fire. No lifetime theorem, implementation admission, novelty, absorption, domination, or mission completion follows.

## 1. Authority and exact reviewed inputs

I read AGENTS.md, the full dual-loop skill, README, CURRENT_STATE, HANDOFF section 17, TERMINAL_GATE_2, SYNTHESIS_GRAPH section 29, STEERING_ENDGAME, ENDURANCE_DESIGN_EXPERIMENT, the exact journal candidate, its actual review, the admitted boundary model/review, the controller-repair packet, and the actual STEERING_OCTOBER_4_DECISION. I also read ENDURANCE_DESIGN_QUESTIONS, SOURCE_DESIGN_PRETEST and INITIAL_SOURCES.

Current file hashes checked during this invocation:

| File | SHA256 |
|---|---|
| ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md | `f0ab692cf629fd1f3c4f35d7e6600c3a09f5a6505b98f9ff776649b3c8d7ec97` |
| ENDURANCE_JOURNAL_DESIGN_CANDIDATE.md | `705c445ccfcbe217c7efc783208e99ea35e043a548cb96c507666501221e23c0` |
| ENDURANCE_JOURNAL_DESIGN_REVIEW_1.md | `388c86d9b265c6ce8ee8120f9d91b6c6a7b3506db86c23a93e8d8a37fd630f23` |

The boundary and journal hashes match their actual saved reviews. Ledger records 80–81 register the unit and record `ELEMENTARY-CONTROLLER-COMPILATION`, with `repairs_used=1` and `repairs_max=1`.

## 2. Frozen contract retained

The attempted repair retains:

- \(N=2k\), \(a=\lceil\log_2(N+1)\rceil\), \(q=\lceil\log_2(NL+2)\rceil\), and exactly \(S=256q^2\) private bits.
- Every initial \(x\), every promised FLIP sequence, unlimited read-only QUERY interleavings, and correct queries after the final guaranteed flip.
- Strict INVALID exhaustion, no wear observation, and charging every actual intermediate persistent change.
- Address formation charged separately from the \(a+1\) cost of each addressed persistent bit access.
- Canonical completed-command volatile boundaries, charged clearing, and parameter-only immutable code.
- The target domain and frozen target
  \[
  H\ge \frac{NL}{1024(\log_2N)^2},
  \qquad O(q^3)\text{ worst-case command work}.
  \]
- The same fixed trie, exactly \(d+1\) fresh nodes per FLIP, two journals, prefix flags, rank folding, clearing chunks and role exchange.
- The original two-cell and direct-base parameter branches.

No new private-memory access primitive, code-memory access primitive, word operation, external controller, or weaker target is introduced.

## 3. What remains supported at design scope

The actual review found no layout, schedule or candidate wear counterexample. This repair supplies no reason to retract those scoped findings.

Physical child bit-addresses remove the need to multiply a slot number by node width at every node visit. Exactly \(d+1\) fresh slots per version determine both the root slot and allocation extent. Latest-root, fresh-start and clearing-chunk address products occur only constantly many times per command.

Field and clearing accesses can use consecutive cursors. Each fresh persistent field bit must be written once, directly to its privately computed final value. Arithmetic, incremental counters and partial field construction must remain private.

These observations preserve a plausible route-level cubic decomposition. They do not implement private selection or instruction dispatch.

The proposed \(H=ML\), small-branch guarantees and \(1/1024\) target remain **CONJECTURE / UNADMITTED**.

## 4. The unresolved elementary selection operation

The required implementation includes operations of the following form:

```text
read scalar[r].bit[t]
write scalar[r].bit[t]
read frame[level].pointer.bit[t]
write frame[level].direction
fetch installed_code[PC]
execute the fetched instruction's selected operands
```

Here `r`, `t`, `level` and `PC` can depend on the current invocation.

The admitted primitive-cost clause permits finite-fan-in Boolean bit operations, bit copies, bounded branches, controller/PC/return-address manipulation and emitted bits. It explicitly supplies the addressed **persistent** bit operation and its routing charge. The code clause permits parameter-only generated code while requiring generation, installation and execution access accounting.

Neither clause licenses treating the operations above as free array accesses. In particular, the persistent \(a+1\) rule cannot simply be assigned to a separate private or code address space.

A generated instruction with an immediate private location avoids a runtime arithmetic calculation of that location. It does not, by itself, discharge:

1. selection and fetching of that instruction;
2. decoding of its operand locations;
3. access to those operands;
4. selection and execution of the controller that performs those actions.

I did not derive a complete implementation of those four stages from the frozen primitives.

This is an unmet construction obligation. It does not overturn the admitted boundary contract or prove that the clauses prohibit every suitable implementation.

## 5. Expansions attacked and rejected

### A. Charged branch-tree lookup

A tempting schema follows address bits through a binary decision tree and copies the selected bit at a leaf.

The address tests and encoded PC changes can be charged. The missing stage is how execution reaches each selected tree node and its immediate operand. Treating that reachability as automatic would leave precisely the instruction/operand selection being audited uncharged.

An abstract tree’s depth is insufficient evidence of its complete elementary execution cost. I therefore reject this schema as a completed repair.

### B. Explicit Boolean selection scans

Selection can be expressed using equality masks and finite-fan-in gates. For candidate locations \(b_u\), form a selector equality \(e_u\) and combine \(e_u\land b_u\). A selected write similarly computes a masked replacement for every candidate destination.

This exposes the selection work rather than concealing it. It also exposes the cost threat.

For example, a compiler that scans a \(\Theta(q^2)\)-bit private arena separately for each of \(\Theta(q^2)\) logical bit actions spends \(\Theta(q^4)\) selection work before complete controller accounting. Likewise, scanning a \(\Theta(q^2)\)-bit installed program at each of \(\Theta(q^2)\) dispatches produces a quartic subtotal.

These are conditional tallies for those scan expansions, **not lower bounds for private access or code fetch**. Batching a frame transfer, exploiting program structure, or another legal compiler could improve them. I did not supply and close such a compiler.

A fixed Boolean selection circuit also needs its own charged sequential execution schedule. Its gate count alone is not the requested installed-program execution certificate.

### C. A logarithmic factor on already cubic work

The expansion
\[
O(q^3)\times O(\log q)
\]
does not establish the frozen \(O(q^3)\) target.

The promising decomposition would instead separate persistent routing from a smaller private/control workload:
\[
W(q)
=
W_{\mathrm{persistent}}(q)
+
W_{\mathrm{private/control}}(q)
+
W_{\mathrm{clear}}(q).
\]

An \(O(q^2(\log q)^c)\) private/control subtotal, for fixed \(c\), could fit a cubic upper bound. But that argument requires the complete underlying elementary expansion. It cannot replace one.

The aggregate cursor bound on bit changes likewise does not charge all cursor reads, tests, selections and control transitions. I did not use it as a complete work bound.

## 6. Resource obligations that remain unclosed

The reservation
\[
64(2q)+q(2q)+16q^2+16q^2
=34q^2+128q\le98q^2
\]
is valid capacity arithmetic for \(q\ge2\). It remains a reservation.

This failed repair does **not** claim that it is a simultaneous live-allocation certificate.

| Required deliverable | Repair outcome |
|---|---|
| Actual named scalar fields, lifetimes and helper clobber rules | No complete executable allocation supplied |
| Selected scalar/frame access | No complete charged expansion supplied |
| Instruction, operand and selector buffers | No complete simultaneous bound supplied |
| PC, branch and helper-return expansion | No closed fetch/dispatch implementation supplied |
| Add/copy/compare/multiply helper schemas | Mathematical bit recurrences do not close their selected operand execution |
| Numerical installed program bit length | No complete compiler supplied from which to derive it |
| Uniform generation and installation resources | No complete generator/installer certificate supplied |
| Fixed initialization exponent \(c_{\rm init}\) | Not established |
| Clearing of every private/controller bit and canonical return | No complete executable clearing tail supplied |
| Worst-command \(O(q^3)\) tally | Not established |

Assigning unused reservation slots to names would not repair these omissions. Nor would a large guessed program-size constant or initialization exponent establish a uniform generator.

No scratch clearing work may disappear merely because storage is volatile. The final controller transition, including clearing input-dependent selectors, buffers and return state, belongs inside the command tally.

## 7. Bounded disposition and continuing obligations

The single registered repair is exhausted by this failed deliverable. It must not silently become another repair iteration or a replacement construction family within ENDURANCE-REGIME-CAN-DESIGN-1.

The appropriate recorded state is:

```text
ELEMENTARY-CONTROLLER-COMPILATION: REPAIR-FAILURE
repairs_used: 1 / 1
positive_design: NOT-CONFIRMED
elementary_compiler: NOT-SUPPLIED
cubic_command_bound: NOT-ESTABLISHED
positive_lifetime: NOT-ADMITTED
source_comparison: OPEN
```

There is no new layout/schedule/wear counterexample and no all-model lower bound.

A **different fresh independent actual Codex reviewer** must assess this artifact and its scope. This invocation supplies no independent regate and authorizes no implementation or FULL run.

INITIAL_SOURCES’ monotone wear-lift comparison was read but not developed or admitted. Universal limits, tight-loss, low redundancy and the strict primary-literature delta remain open. The positive target remains unchanged.

Execution was read-only: no file edits, agents, GPU use, solver, scientific implementation, FULL run or git mutation attempt. Root retains responsibility for saving this artifact and committing the record.

## 8. Narrative and mission standing

**HEURISTIC one-sentence story:** The storage schedule remains plausible, but this repair did not turn its controller into a fully charged elementary program.

The strongest dismissal is “this records a missing implementation.” That dismissal is fair. Identifying the missing work protects the research record; it is not a useful CAN theorem or a new computation model.

The useful finite CS/math CAN+CANNOT objective remains active. This bounded repair failure is not a terminal conclusion, and the keep-going invariants remain in force.