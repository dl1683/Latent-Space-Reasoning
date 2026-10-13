**CORE-DESIGN-PASS — component design evidence only.**  
**RAM-DESIGN-REPAIR — `CLOSED-BIT-LOOP-LOWERING`.**

The worker’s `RAM-REPAIR` disposition is justified. I found no functional counterexample in the supplied fetch and cleanup loops. The artifact still supplies no complete Johnson/journal compiler, emitted-length certificate, generator, simultaneous live allocation, unconditional command bound, or actual-machine lifetime CAN.

Rigor: **design-review judgments and conditional accounting**, with no execution or new PROVED label.

Reviewed build SHA-256:

`650ad0f4637d36ff8b55d8c5e8666ff52a2b02fe367c0d09119daf2e691b185c`

The indexed-model and journal-candidate hashes match their saved reviews.

**The component attack**

| Target | Finding |
|---|---|
| **252 states** | The ranges partition IDs 0–251 without overlap. The fetch, operand, jump, selected-lane and cleanup blocks fit their stated allocations. This supports a runtime design; the four unused encodings establish nothing about generator feasibility. |
| **Instruction decoding** | Four opcode bits and serial operands have bounded decoding paths. LP consumes exactly `c` bits; JZ consumes `b` bits before testing the preserved predicate. LANE consumes four selector and three action bits. End tags provide termination without an uncounted width counter. |
| **Latches and fetch clobbers** | The inventory totals sixteen. Fetch preserves L0/L1/O/D, returns X, and uses K. R names fourteen fixed fetch continuations; it supplies no bytecode CALL stack. Helpers cannot retain carry or predicates in X/K across fetch. |
| **PC increment and branches** | Fetch copies PC into C before acquisition, increments PC by explicit ripple, and restores both heads. TARGET→PC is a serial copy. Because `b=ceil(log₂(B+1))`, incrementing a legal final CODE address to B need not overflow the lane; a subsequent CODE acquisition at B would remain invalid. |
| **Dynamic lane selectors** | Ten fixed conditional rows select literal lane names. HOME and INC expand into tag reads, adjacent moves, bit writes and charged control. No numeric head or growing selector table is required. |
| **PRIVATE access** | PR/PW specify one paid access through formed P. They do not implement scalar/frame selection, field traversal, pointer saves or helper returns. Those remain lowering obligations. |
| **Finite description** | The row envelope and arithmetic support `690×68+252×5+4096=52,276` bits for the **runtime-description candidate**. They certify neither the complete compiler controller nor installed CODE length. |
| **Allocation** | The trial envelope includes `S_base`. At `q=2`, its totals are 1,012 or 1,020 within 1,024 bits. This is capacity arithmetic, not evidence that helper fields, continuations and generator state coexist legally. |
| **CLEAN preconditions** | The required `P=S_base`, `T3=S`, `L0=0` are essential and must be established by paid instructions. Fetching CLEAN preserves them. |
| **Control self-erasure** | GENERAL is erased before its controlling lanes. Equality and ripple use counted latches and heads. The ten lane sweeps restore DATA and heads; sixteen subsequent states clear the latches. |
| **Actual RETURN** | State 251 explicitly pays restoration to state 0, then invokes RETURN with the canonical-image check after those updates. This passes under the supplied ordering; no automatic reset or intervening fetch is imported. |

These findings concern the [runtime inventory and loops](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_RAM_COMPILER_BUILD.md:84), with the [cleanup preconditions and final transition](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_RAM_COMPILER_BUILD.md:222) preserved literally.

One bookkeeping correction belongs in the eventual literal table: the row-alphabet paragraph says **six** port actions, but the design needs **eight**—DATA read/write, PRIVATE read/write, CODE read, INPUT read, EMIT and RETURN. The eight-bit action field accommodates them. This does not expose a functional loop defect or require another repair family.

The cleanup sweep also writes zero to the left end-cell DATA. That preserves its required zero value; it does not authorize nonzero end-cell writes or tag changes.

**The cost deductions survive only with their hypotheses**

The supplied fetch bound includes copying, ripple increment, head restoration, decoding and the CODE tariff. The cleanup bound includes all `g=S−S_base` PRIVATE writes, comparisons, cursor manipulation, lane sweeps, latch clearing and final control restoration. It excludes the preceding instructions that establish CLEAN’s three preconditions; those must enter the eventual program ledger.

The quartic deduction has the correct conditional form:

\[
n_{\mathrm{inst}}\le Kq^2(c+2)
\quad\Longrightarrow\quad
W_{\mathrm{command}}\le(2^{40}K+2^{35})q^4.
\]

Using the supplied bounds, `c+2≤6.5q` and
`2903×6.5<2¹⁵`, so the coefficient leaves room for the stated instruction envelope. Cleanup contributes at most `2³⁵q⁴` for `q≥2`.

**The missing fact is the antecedent.** No fixed K has been derived. It must count dynamic PRIVATE selection, frame access, arithmetic loops, address formation, every branch operand fetch, continuation saves/restores and CLEAN setup. An additional q factor or a logarithmic multiplier on an already quartic subtotal cannot disappear into this deduction.

Likewise:

- `52,276` runtime-description bits do not establish useful CODE length `≤65536q²`.
- Paid padding is only one preparation obligation; it does not supply the append-only generator.
- `O(Nq⁶)` preparation remains a target without a derived coefficient.
- Raw-input initialization still needs its executable body, work bound and wear accounting. No initialization exponent is established merely by choosing a generous constant.

See the [conditional allocation and cost discussion](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_RAM_COMPILER_BUILD.md:320).

**The single repair’s acceptance condition**

`CLOSED-BIT-LOOP-LOWERING` must deliver executable specifications for **both Johnson and the journal**, using this same ISA/controller/cleanup family:

1. Literal instruction templates for scalar/frame selection, arithmetic, persistent field access, prefix recovery, lookup, rank selection, path copying, maintenance and all parameter branches.
2. Helper interfaces with actual locations, clobbers, saved predicates/carries, continuation fields and maximum nesting.
3. A simultaneous live map including base lanes, arithmetic widths, frames, continuations and preparation phases.
4. Exact template/site counts and parameter layout deriving useful CODE length within B.
5. An append-only emitter in S, with an actual finite controller table and branch-address construction that neither rereads output nor hides a label dictionary.
6. Numerical preparation, initialization, instruction and port ledgers, including CLEAN setup and its final transition.

The emitter must fit the model’s controller and description caps through an explicit construction. It cannot be justified by the unused four runtime state encodings.

The emitted journal must also preserve the reviewed trace: fresh fields written directly to final values; exactly `d+1` nodes per update; zero-parity keys retained in rank accounting; no OLD reads once clearing begins; maintenance and final clearing completed before acknowledgement and role exchange. Johnson’s conditional one-write trace cannot substitute for the primary journal target.

This is one missing executable lowering with several required certificates. **No separate functional core repair was identified.** The port-count wording should be corrected when that repair supplies the literal table.

At entry, the record reports **repairs used 0/1**. This fresh named gate assigns the sole correction to `CLOSED-BIT-LOOP-LOWERING`; its outcome awaits fresh regating. Recording this gate should allocate **1/1**. The proposed initial artifact was not itself a consumed repair, and the exhausted old cubic unit remains closed. No ledger was edited here.

**Exact coverage and limits**

Fully read:

| File | Lines |
|---|---:|
| AGENTS.md; README.md | 1–160; 1–64 |
| ENDURANCE_RAM_CONTROLLER_PROTOCOL.md | 1–55 |
| ENDURANCE_INDEXED_MODEL_CANDIDATE.md / REVIEW.md | 1–151 / 1–75 |
| ENDURANCE_RAM_COMPILER_BUILD.md | 1–450 |
| ENDURANCE_JOURNAL_DESIGN_CANDIDATE.md / REVIEW_1.md | 1–193 / 1–86 |
| STEERING_CONTROLLER_DECISION.md | 1–41 |
| ENDURANCE_CONTROLLER_REPAIR.md | 1–189 |
| ENDURANCE_PROBE_LAW_CANDIDATE.md | 1–93 |
| ENDURANCE_INITIAL_SOURCES.md / SOURCE_DESIGN_PRETEST.md | 1–123 / 1–78 |
| RAM compiler BUILD_PACKET / DESIGN_PACKET | 1–59 / 1–22 |
| TERMINAL_GATE_2.md; STEERING_ENDGAME.md | 1–342; 1–83 |
| Complete dual-loop SKILL.md | 1–526 |

Also read HANDOFF §17 and its narrative, lines 264–327; SYNTHESIS_GRAPH §29, lines 2028–2907; CURRENT_STATE lines 174–209; EXPERIMENTS lines 255–277; and matching ledger records at lines 81, 84, 94, 95 and 103.

All shell calls used `C:/Windows/System32/cmd.exe`, `login=false`. No code execution, solver, GPU, agents, writes or commit attempts occurred. This was a component review, not an entire-repository or terminal gate; source-comparison records were read without a new primary-literature admission.

The honest surviving story is that the runtime design can fetch bits and clear its own control, while its useful storage program remains missing. **Primary delta OPEN; compiler, lifetime CAN, novelty and mission completion UNADMITTED.**