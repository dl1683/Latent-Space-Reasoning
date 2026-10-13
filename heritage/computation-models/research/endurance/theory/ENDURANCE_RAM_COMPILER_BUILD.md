**RAM-REPAIR — CLOSED-BIT-LOOP-LOWERING. UNADMITTED.**

This construction attempt does **not** supply the complete compiler required for `RAM-DESIGN-CANDIDATE`. The precise blocking gap is the missing, continuation- and clobber-complete lowering of the journal routines into the bit ISA below. Without that lowering, there is no actual emitted-program inventory, append-only generator certificate, simultaneous journal-field certificate, or unconditional quartic command ledger.

The useful constructive result is a concrete **252-state runtime-controller design**, including serial literal decoding, charged fetch, selected lane actions, and a cleanup tail that clears its own control. It removes those mechanisms from the list of unspecified assembly promises. It does not include a completed generator or journal program.

Rigor throughout: **DESIGN COMPONENT / UNADMITTED**, or **CONDITIONAL** where explicitly stated. No numerical execution, scientific implementation, GPU use, or independent proof admission occurred. This is the initial construction report for the new unit; it proposes one named repair, not an additional repair of the exhausted cubic unit.

**Authority and scope.** I fully read AGENTS, the dual-loop skill, the RAM-controller protocol, indexed-model candidate and actual review, controller decision, journal candidate, and prescribed historical entry files. I also read the actual journal review, exhausted controller-repair report, Johnson candidate, and source-translation records.

The current candidate hashes match the saved reviews:

| Artifact | SHA-256 |
|---|---|
| Indexed model | `3cd1732070d1874c36c3ad9fd267a03f0b98edd76ad27faa0078954ea2fb3129` |
| Journal design | `705c445ccfcbe217c7efc783208e99ea35e043a548cb96c507666501221e23c0` |

The indexed review supplies `INDEXED-DESIGN-PASS` and VALID endpoint/A/B/C/query-law transfers. Those admissions do not supply this compiler.

**One proposed bit ISA.** Instructions have a four-bit opcode. Numeric operands are emitted least-significant bit first. Lane identifiers are the fixed order `A,P,C,I,PC,TARGET,T0,T1,T2,T3`, encoded `0,…,9`.

| Opcode | Instruction | Operand bits | Effect |
|---|---|---:|---|
| `0` | `LP` | `c` | Serial literal load into P |
| `1` | `PR` | 0 | PRIVATE[P] → L0 |
| `2` | `PW` | 0 | L0 → PRIVATE[P] |
| `3` | `NOT` | 0 | L0 ← ¬L0 |
| `4` | `COPY` | 0 | L1 ← L0 |
| `5` | `AND` | 0 | L0 ← L0 ∧ L1 |
| `6` | `XOR` | 0 | L0 ← L0 ⊕ L1 |
| `7` | `DRD` | 0 | DATA[A] → L0 |
| `8` | `DWR` | 0 | L0 → DATA[A] |
| `9` | `IRD` | 0 | INPUT[I] → L0 |
| `A` | `EMIT` | 0 | Emit L0 |
| `B` | `JZ` | `b` | Load TARGET serially; copy it to PC iff L0=0 |
| `C` | `LANE` | 4+3 | Named serial-lane action |
| `D` | `SET` | 1 | Literal bit → L0 |
| `E` | `JUMP` | 0 | Serial TARGET → PC |
| `F` | `CLEAN` | 0 | Enter the explicit cleanup tail |

`LANE` actions `0,…,7` are:

1. Read marked DATA into L0.
2. Write L0 into marked DATA.
3. Move HEAD left.
4. Move HEAD right.
5. Read LEFT tag into L0.
6. Read RIGHT tag into L0.
7. HOME using repeated tag reads and adjacent moves.
8. Increment using the explicit ripple routine below.

HOME and increment are interpreted finite routines. Their reads, moves, bit writes, Boolean operations and control changes are charged individually. They are not additional word primitives.

Lane identifiers `10,…,15`, outward moves, invalid addresses, illegal end-cell writes and increment overflow are outside promised executions. No recoverable signal is supplied.

There is no CALL opcode. A compiled helper must save and restore its continuation through ordinary charged instructions. **This is precisely where the unfinished lowering must become concrete.**

**All sixteen runtime latches.** The core uses:

```text
L0, L1                         2 value latches
O0,O1,O2,O3                    4 opcode/action bits
D0,D1,D2,D3                    4 lane-selector bits
R0,R1,R2,R3                    4 fixed fetch-continuation bits
X                              fetched CODE bit
K                              local read/ripple temporary
```

Fetch preserves L0, L1, O and D. It uses X and K and returns through the counted R bits. A bytecode routine cannot keep arithmetic carry in X or K across fetch. A compiler using those latches as surviving arithmetic state would be incorrect.

The four R bits select only fourteen fixed microcode continuations:

```text
0..3    opcode-bit continuations
4       LP operand-bit continuation
5       JZ operand-bit continuation
6..9    lane-selector-bit continuations
10..12  lane-action-bit continuations
13      SET literal-bit continuation
```

Setting R pays each assigned bit; selecting its continuation pays the Boolean decoding and microstate changes. No helper return is implicit.

**Runtime microstate inventory.** States use ordinary eight-bit encodings of the following integers:

| IDs | Count | Function |
|---|---:|---|
| 0 | 1 | Entry: request opcode bit 0 |
| 1–4 | 4 | Store successive opcode bits |
| 5 | 1 | Four-bit opcode dispatch |
| 6–38 | 33 | Fetch one CODE bit |
| 39–49 | 11 | LP literal decoding |
| 50–61 | 12 | JZ literal decoding and predicate |
| 62–81 | 20 | Serial TARGET → PC |
| 82–115 | 34 | LANE decoding and execution |
| 116–125 | 10 | PR/PW/NOT/COPY/AND/XOR/DRD/DWR/IRD/EMIT |
| 126–127 | 2 | Fetch and install SET’s literal |
| 128–164 | 37 | GENERAL clearing loop |
| 165–234 | 70 | Ten lane-clearing blocks |
| 235–250 | 16 | Clear the sixteen latches |
| 251 | 1 | Paid state restoration and RETURN |
| **Total** | **252** | **Runtime core only** |

Every transition invokes at most one named port or local lane action. Boolean operations, latch assignments and changed microstate bits are additional paid work. Selected-lane operations expand to ten fixed conditional rows, one per named lane; there is no hidden numeric head or parameter-growing lane table.

**Literal fetch microcode.** The following completely specifies the 33-state fetch block. `ML/MR` moves one adjacent cell. Reads copy the returned bit into the indicated latch.

```text
F0:  LEFT(PC)→K;                         goto F1
F1:  if K=1 goto F3, else F2
F2:  ML(PC);                            goto F0

F3:  LEFT(C)→K;                         goto F4
F4:  if K=1 goto F6, else F5
F5:  ML(C);                             goto F3

F6:  MR(PC);                            goto F7
F7:  MR(C);                             goto F8
F8:  RIGHT(PC)→K;                       goto F9
F9:  if K=1 goto F14, else F10
F10: DATA(PC)→K;                        goto F11
F11: K→DATA(C);                         goto F12
F12: MR(PC);                            goto F13
F13: MR(C);                             goto F8

F14: CODE[C]→X;                         goto F15

F15: LEFT(PC)→K;                        goto F16
F16: if K=1 goto F18, else F17
F17: ML(PC);                            goto F15
F18: MR(PC);                            goto F19
F19: RIGHT(PC)→K;                       goto F20
F20: if K=1 INVALID, else F21
F21: DATA(PC)→K;                        goto F22
F22: K←¬K;                              goto F23
F23: K→DATA(PC);                        goto F24
F24: if K=1 goto F26, else F25
F25: MR(PC);                            goto F19

F26: LEFT(PC)→K;                        goto F27
F27: if K=1 goto F29, else F28
F28: ML(PC);                            goto F26
F29: LEFT(C)→K;                         goto F30
F30: if K=1 goto F32, else F31
F31: ML(C);                             goto F29

F32: decode R into its fixed continuation
```

PC and C have identical width, so their copy-loop heads remain aligned. The copied DATA bits form C; CODE acquisition is charged `b+1`. Ripple increment uses K, preserving the fetched bit X. Both heads return to cell 0.

At most `28b+32` microstate visits occur per legal fetch. This includes worst-position HOME operations; it does not assume heads were already home.

LP uses three states to HOME P, one to move to its first bit, two to test RIGHT/request fetch, two to store X/move right, and three to HOME P afterward. Thus all `c` operand bits are decoded through fetch, without a width counter.

JZ applies the same eleven-state scheme to TARGET, then one state tests L0. The twenty-state jump block consists of two initial HOME routines, two first-bit moves, a six-state copy loop, and two final HOME routines.

The LANE block contains eleven decoding/dispatch states, six direct-action states, a three-state HOME routine and a fourteen-state increment routine. Increment performs:

```text
HOME lane;
MR;
repeat:
    read RIGHT tag; reject overflow;
    read marked DATA into K;
    K←¬K;
    write K;
    if K=1 finish;
    MR;
HOME lane.
```

The end tags bound every local loop.

**Finite description representation.** For this runtime core, a conditional-row representation needs at most 690 rows:

```text
252 states × at most two ordinary rows                  504
17 selected-lane action states × eight additional rows 136
opcode dispatch’s additional rows                       14
fetch-return dispatch’s additional rows                 12
lane-ID validation’s additional rows                     8
lane-action dispatch’s additional rows                   6
ten cleanup HOME tests’ third branch                    10
                                                        ---
                                                        690
```

A row has this fixed 68-bit encoding:

```text
latch-condition mask/value      16+16
action                           8
literal named lane               4
destination/source latch         4+4
R constant-assignment mask/value 4+4
next microstate                  8
```

Source state is implicit in the state’s row list. Five bits encode each list’s row count. The row action alphabet consists of NONE; named local read/write/moves/tag reads; the six indexed/input/output/return port actions; and one-bit COPY, NOT, AND, XOR, SET0 and SET1. R assignments accompany the action as explicitly charged constant latch assignments.

The rows and counts occupy at most

\[
690(68)+252(5)=48{,}180\text{ bits}.
\]

Allowing 4,096 bits for the fixed codec/action specification gives a **runtime-description candidate** of 52,276 bits. This is below the cap. **It is not a description certificate for the unfinished generator/compiler.**

To avoid treating dispatch logic as free, a conservative fixed decoder can test source-state patterns, masked latch conditions and gate the finite actions/updates. Fewer than \(2^{17}\) Boolean gates suffice for the listed row envelope; charging \(2^{18}\) per microstate visit covers those gates, latch copies, changed state bits and a local action. Indexed port tariffs are additional.

Consequently the fetch component has the conservative explicit bound

\[
W_{\rm fetchbit}
\le (28b+32)2^{18}+(b+1)
\le 2^{23}(b+2).
\]

This coefficient is intentionally loose and finite.

**Executable cleanup, including its own control.** CLEAN requires the preceding bytecode to establish:

```text
P     = S_base
T3    = S
L0    = 0
```

All other lane DATA and latch values may be dirty, subject to legal lane encoding. These preparations are paid instructions. They cannot be omitted from a compiler ledger.

GENERAL is exactly `[S_base,S)`. The clearing loop does not write any reserved lane marker, tag, latch or microstate through PRIVATE.

Its 37 states are:

```text
C0:      L1←1
C1..3:   HOME P
C4..6:   HOME T3
C7:      MR P
C8:      MR T3

C9:      RIGHT(P)→K
C10:     if K=1 goto C18, else C11
C11:     DATA(P)→X
C12:     DATA(T3)→K
C13:     K←K xor X
C14:     K←¬K
C15:     L1←L1 and K
C16:     MR P
C17:     MR T3; goto C9

C18..20: HOME P
C21..23: HOME T3
C24:     if L1=1, clear O3 and enter first lane block;
         otherwise goto C25
C25:     PRIVATE[P]←L0

C26:     MR P
C27:     RIGHT(P)→K
C28:     if K=1 INVALID, else C29
C29:     DATA(P)→K
C30:     K←¬K
C31:     K→DATA(P)
C32:     if K=1 goto C34, else C33
C33:     MR P; goto C27
C34..36: HOME P; goto C0
```

The comparison is serial equality of P and T3. Because P starts at `S_base`, increments once after each zero write, and stops at S, every PRIVATE write selects a legal GENERAL bit. T3 and all clearing control remain outside the arena being erased.

Each of the ten lane blocks uses seven states:

```text
H:  read LEFT tag into K
T:  if K=0 goto M;
    if K=1 and O3=0 goto R;
    if K=1 and O3=1:
        clear O3 and enter next lane block
M:  move left; goto H

R:  read RIGHT tag into K
B:  if K=1:
        O3←1; goto H
    else goto W
W:  write zero from L0 to marked DATA
V:  move right; goto R
```

O3 distinguishes the initial HOME from the final HOME. The sweep visits the left end and all width-many DATA cells, stops at RIGHT, and returns HEAD to cell 0. Legal executions already preserve zero end-cell DATA. Tags are never changed.

The lane order is A, P, C, I, PC, TARGET, T0, T1, T2, T3. GENERAL has already been erased before P or T3 is cleared. No CODE read occurs after entering CLEAN.

Sixteen successive states then assign zero to the sixteen latches. State 251 pays its changed microstate bits, restores state 0, and invokes RETURN **after** that restoration within the same declared base transition. RETURN’s canonical-image precondition is checked after the paid updates. There is no intermediate fetch from state 0.

This last ordering is part of the supplied transition specification. It must not be replaced by “RETURN automatically resets control.”

Let

\[
g=S-S_{\rm base},\qquad U=a+e+3b+5c.
\]

A conservative cleanup ledger is

\[
W_{\rm clean}\le
2^{18}\!\left[40(c+2)(g+1)+10U+177\right]
+g(c+1).
\]

The first term includes comparison, ripple, heads, Boolean logic, state changes and latch clearing; the second charges the \(g\) PRIVATE zero writes. Using the admitted dimension bounds gives, for \(q\ge2\),

\[
W_{\rm clean}\le2^{36}q^3.
\]

This is a cleanup-component bound under the stated entry preconditions.

**Space attack and a useful small-q correction.** The previous \(98q^2\) reservation cannot simply be added to an ignored base. A possible envelope for repairing this compiler is:

```text
base lanes, state and latches     S_base
64 q-bit scalar fields           64q
q frames of four q-bit fields    4q²
eight c-bit short fields          8c
eight b-bit continuation fields   8b
Boolean/control block             64
```

Its total satisfies

\[
S_{\rm trial}\le4q^2+168q+672\le256q^2
\quad(q\ge2).
\]

At \(q=2,c=11,b=19\), the non-base portion is exactly 448 bits. The actual base cases therefore give **1,012 or 1,020 bits**, within S=1,024.

This establishes a usable **capacity envelope**, not the missing simultaneous-live certificate. In particular, the eight continuation fields and short-field save rules have not been tied to completed bytecode routines. Calling this envelope a live allocation would repeat the earlier error.

A journal frame can hold old-node address, sibling address, sibling count and a direction bit with zero padding. The direction does not require a hidden key or lookup table. Since \(d+1\le a\le q\), q frames cover the proposed path. Arithmetic products requiring a wider temporary must occupy explicitly paired scalar fields.

**What the runtime core makes quartic accounting possible.** For a non-CLEAN instruction, the core gives the conservative bound

\[
W_{\rm inst}\le
2^{25}\!\left[(b+2)(b+c+16)+a+e+4\right].
\]

It includes opcode/operand fetch, literal formation, serial jump and the longest selected lane routine. Persistent and input ports are included within this envelope; a detailed program ledger should still expose their counts separately.

Put \(\ell=\lceil\log_2q\rceil\). Then

\[
c\le2\ell+9,\quad b\le2\ell+17,\quad
\ell^2\le3q,
\]

so

\[
W_{\rm inst}\le 2^{25}(2903q).
\]

Thus a repair establishing an actual bound

\[
n_{\rm inst}\le Kq^2(c+2)
\]

with a derived fixed K would obtain

\[
W_{\rm command}
\le(2^{40}K+2^{35})q^4,
\]

including cleanup. **K has not been established in this report.** This is a precise sufficient criterion for the repair, not an asserted command bound.

The lowering must count short-pointer formation, frame selection, arithmetic bit loops, branches and continuations inside \(n_{\rm inst}\). In particular, an inner JZ has a b-bit operand and pays b CODE fetches; it is not a constant-cost branch. No already-quartic subtotal may acquire an unbounded logarithmic multiplier.

**Same-family Johnson requirement.** The proposed first program remains the classical two-sweep cycle. Its register-level algorithm is:

```text
p ← DATA[N−1]
binary-search first j with DATA[j] xor p = 0

QUERY:
    emit parity(j) xor (p and parity(N))

FLIP:
    write DATA[j] ← 1−p
```

Initialize raw zero by leaving all cells zero; initialize raw one by setting cell 0. The changed positions then cycle through `0,…,N−1`. Through `NL−1` flips, including initialization, no cell changes more than L times. Queries change no persistent cell.

This wear argument remains **CONDITIONAL on its emitted program obeying that one-write trace and canonical cleanup**. No Johnson bytecode image was completed here. The journal must use the same ISA, controller, continuation mechanism and cleanup; no easier bootstrap interpreter is proposed.

The cycle’s ancestry is explicit: TI identifies the CD4017B/CD4022B as Johnson counters. Its hardware shifts and decoded outputs do not provide free operations in this machine. [TI primary description](https://www.ti.com/product/CD4017B).

**Journal trace and wear obligations preserved.** The repair must emit the fixed-trie schedule without persistent arithmetic or provisional field writes:

- Recover j from the valid prefix; derive physical root/fresh addresses using only constantly many products.
- Traverse with physical child pointers; save the actual path privately.
- Construct exactly \(d+1\) fresh nodes bottom-up. Compute each field privately and write each fresh persistent bit directly to its final value.
- Publish the valid flag, then perform the scheduled maintenance before acknowledging.
- During the first half, fold at most two OLD ranks, including zero-parity keys.
- During the second half, clear the prescribed \(2P\) consecutive OLD bits. Queries at these boundaries must bypass all OLD accesses.
- Complete the final clear before toggling role; set started at most once.

Under that exact trace, L epochs give the candidate per-cell ledger:

| Cell class | Changes through L epochs |
|---|---|
| Base | At most one initialization change plus one fold per subsequent epoch |
| Node/valid | At most one fill or one clear per epoch, in alternating roles |
| Role | Exactly one per completed epoch |
| Started | At most one total |
| Unused | Zero |

Hence the scaffold’s \(H=ML\) calculation remains consistent, including post-H queries. The direct-base and two-cell branches retain their separate guarantees and the already-derived \(1/1024\) target calculation. **None becomes an actual-machine CAN without the missing program.**

**The single named repair, precisely.** `CLOSED-BIT-LOOP-LOWERING` must supply one literal emitted-template list covering Johnson and journal execution, with:

1. Each helper’s input locations, output locations, clobbered latches/lanes, saved fields, continuation representation and maximum nesting.
2. Exact instruction bodies for short-pointer selection, word copy/add/subtract/compare/multiply, field read/write, prefix search, lookup, rank selection and bottom-up path copying.
3. Exact template/site counts and parameter literal layout, deriving useful CODE length ≤B.
4. An append-only emitter that obtains branch addresses without retaining an oversized label table or rereading emitted CODE; its actual controller, simultaneous space, source reads, emission, padding and installation must be charged.
5. The resulting numerical instruction/port ledger, including setup of CLEAN’s three preconditions.

These are consequences of one missing executable lowering, not evidence for several unrelated mathematical obstructions. The runtime core leaves four microstate encodings unused; that fact does **not** establish that a generator fits. The repair must integrate or explicitly account for the generator’s finite control.

No unconditional emitted length or \(O(Nq^6)\) preparation coefficient is supplied here. Choosing a generous constant without emitting the templates would not discharge either obligation.

**Primary comparisons, checked against the actual sources.**

| Primary | Occupied ingredient | Required transfer still missing |
|---|---|---|
| Flash2014 §VI | Changed-index logging, parity-bank folding and phase reuse; equation (8) gives the constant-rate write count | Readable monotone levels do not directly implement hidden binary wear, paid cold recovery or this worst-command bit program. [Primary §VI](https://yaakobi.net.technion.ac.il/files/2015/04/8_Rewriting_Codes_IT14.pdf) |
| ELM | Current-state encoding and zero-error constructions; Definition 1 and Construction 19 use stage-indexed maps | Stage recovery, correlated FLIP histories and elementary execution remain payable. The source’s exhausted cells refuse changes; importing that observable behavior would violate strict INVALID exhaustion. [Primary](https://arxiv.org/pdf/2109.09932) |
| ISAAC2020 | Strong randomized word lifetime with paid metadata labels | Flattening words does not remove surviving wear-free counters/random parameters or convert expected/amortized maintenance into this deterministic worst-command contract. [Theorem 3 and Algorithms 2–6](https://drops.dagstuhl.de/storage/00lipics/lipics-vol181-isaac2020/LIPIcs.ISAAC.2020.65/LIPIcs.ISAAC.2020.65.pdf) |

These sources prevent novelty claims based on logging, folding, hidden programming counts or Johnson ancestry. This inspection establishes neither full-contract domination nor priority.

**Disposition and narrative.** The result is `RAM-REPAIR`, with positive compiler, preparation, quartic command bound and actual-machine lifetime **UNADMITTED**. The useful next constructive fact is the explicit fetch/cleanup core and its finite cost criterion; the next required artifact is the literal lowering described above.

The honest story is: *the proposed interpreter can fetch bits and erase its own control within the finite envelope, but the journal still lacks its complete executable program.* That is a construction component, not the computation-model objective or a completed capability/limit pair.

All shell calls used `C:/Windows/System32/cmd.exe`, `login=false`. No repository writes, commit attempts, subagents, simulated reviewers, solver runs or scientific code execution occurred. Unrelated tiny-calibration result files changed externally during this read-only session.