# CLOSED-BIT-LOOP-LOWERING: executable source, generator unclosed

**RAM-FAIL-UNADMITTED.** The sole correction of the new quartic unit is used
here. The old cubic unit remains exhausted. This artifact supplies literal
Johnson and journal source bodies and a 252-state runtime table. It does not
supply the legal uniform installation generator, so it cannot confirm the
registered compiler or primary lifetime. No impossibility or counterexample
against other compilers follows.

Rigor: **SOURCE CONSTRUCTION / UNADMITTED**, **CHECKED COMPONENT** for the
specified small logical checks, and **CONDITIONAL / UNREVIEWED** for the loop
accounting below. Nothing here is PROVED, CAN, RAM-CONFIRM or FULL. Fresh actual
code/proof review belongs to the root. Primary novelty, full-contract delta,
tight loss, low redundancy and the mission pair remain OPEN.

## Executable artifacts and entry points

- `experiments/endurance_bit_microcode.py`: `TABLE`, `BY_STATE`, `Row.bits()`,
  `validate_table()`, `decode_rows()`, and a row-by-row component checker. Every state 0..251 has
  actual guards, actions and next-state targets. State 115 requests the first
  LANE selector bit; this makes the previously described 34-state LANE block
  literal. The action alphabet has **eight ports**, correcting the six-port
  wording. No new machine primitive is introduced.
- `experiments/endurance_bit_compiler.py`: `johnson_source`, `journal_source`,
  `Lower`, `Image.emit`, `source_audit`, `live_map`, `frame_address`.
  `Token.length()` specifies exact instruction lengths. BEGIN/END/LABEL tokens
  emit no bits. All emitted operations are in the frozen 16-opcode ISA.

These are actual bodies, not future helper promises. Source includes word
literal/copy/add/subtract/compare/equality, shifts and multiplication; formed
PRIVATE pointer access, lane/word transfer, persistent fields, prefix recovery,
lookup, rank, path capture/copy, folding, chunk clearing, base initialization,
two-cell/direct-base branches, and CLEAN setup. Calls in Python expand source
offline. There are **no bytecode CALLs, return words or implicit runtime stack**.

Example source construction, using cmd.exe with login=false:

```text
python -B experiments\endurance_bit_microcode.py
python -B experiments\endurance_bit_compiler.py --check
python -B experiments\endurance_bit_compiler.py --emit .codex_tmp\endurance_lowering_example.bits
python -B experiments\endurance_bit_compiler.py --family journal --N 16384 --k 8192 --L 256
python -B .codex_tmp\endurance_lowering_audit.py
```

`--emit` writes one byte per bit (values 0/1), useful image then paid-model
padding to B; the byte container is an offline artifact. It is not B extra
runtime bytes. Only the authorized scratch directory/prefix is accepted.

## Core and exact source inventory

The actual runtime table has **521 rows**. Each row encodes exactly 68 bits;
five count bits per state's row list give **36,688 encoded bits**. All counts
fit five bits. The former 690-row figure was an envelope. This count is for
the runtime rows, not the generator. It excludes a separately encoded finite
codec/decoder; the prior 4,096-bit codec allowance is not a newly certified
generator description.

Guard checks find no overlapping rows. Ordinary states cover all 16-bit latch
assignments; selected-action states cover the ten valid D selectors. D>=10 is
rejected at state 86 before any selected action. It remains outside the selected
states' promised domain. Every target is in 0..251. There is no ninth port or
parameter-growing local-action table.

`Image.inventory()` reports exact opcode counts and, for **each named template**,
its exact bit length, instruction count and source site count. Nested templates
are explicitly included, so their sizes must not be summed as disjoint regions.
The reproducible complete inventories are in authorized scratch:

- `.codex_tmp/endurance_lowering_check.txt` — Johnson and core check.
- `.codex_tmp/endurance_lowering_journal_inventory.json` — journal inventory.
- `.codex_tmp/endurance_lowering_audit.json` — branch/operand closure checks.
- `.codex_tmp/endurance_lowering_core_rows.json` — actual guarded row list.
- `.codex_tmp/endurance_lowering_core.bits` — encoded counts/rows.
- `.codex_tmp/endurance_lowering_johnson.bits` — actually emitted padded image.

| Source parameter point | Useful bits | Source instructions | Closed JZ sites |
|---|---:|---:|---:|
| Johnson N=1,k=1,L=1,q=2 | 24,511 | 2,159 | 326 |
| Primary journal N=16384,k=8192,L=256,q=23 | 515,388 | 40,244 | 5,872 |
| Direct base N=8,k=4,L=16 | 13,285 | 1,122 | 153 |
| Two cells N=2,k=1,L=1 | 10,716 | 953 | 138 |

These are source branches inspected for construction, **not a numerical command
grid**. The journal point has d=13, node width 46, P=645, M=4, J=2580. Its
immutable image cap is 34,668,544 bits; its S is 135,424 bits. The source audit
checks literal operand ranges, label uniqueness, branch closure, targets at real
instruction boundaries, template nesting and the instruction alphabet. The core
description codec round-trips every actual row in state-list order. It does
not establish execution invariants. The Johnson image actually emitted 24,511
useful bits and 237,633 zero-padding bits, total B=262,144.

## Simultaneous locations, clobbers and continuations

`WORDS` fixes 64 scalar fields. Field i occupies `[S_base+iq,S_base+(i+1)q)`.
The q frames immediately follow, each exactly 4q bits: old node pointer,
sibling pointer, sibling key count, direction word. Direction is the low bit;
the other q-1 bits are explicitly zeroed before capture. Twenty-one one-bit
flags follow the frames. `live_map(d)` enumerates every address; `frame_address`
checks all three indices and computes the actual bit location.

```text
base lanes + microstate + sixteen latches    S_base
64 scalar words                            64q
q frames, four q-bit fields                 4q²
named Boolean flags                        21
continuations / bytecode CALL stack         0
```

Total `S_base+64q+4q²+21 <= 4q²+136q+453 <=256q²`, q>=2.
Actual q=2 totals are **729/737**, including bases 564/572. At the journal
point the exclusive used endpoint is **4501**: frame start 2364, flag start
4480. Unused GENERAL is still scrubbed by CLEAN; allocation size does not
discount clearing work.

| Helper/stage | Inputs and outputs | Actual clobbers/live restrictions |
|---|---|---|
| Fetch | PC/C serial DATA; X=result | X,K,R,C,PC/control; L0/L1 and O/D preserved until instruction decoding |
| BUSCOPY | Named source/destination lane DATA | L0; destination DATA; source/destination heads homed; L1 preserved |
| PRIVATE_GET/PUT | T-lane formed pointer, L0=value | P,L0; PUT saves its value in counted L1, then restores via SET0/XOR; both involved heads homed |
| LANE_EQ | Equal-width lane values | EQ,P,L0/L1; values preserved, both heads homed |
| Word arithmetic | Named q-bit words | T0/T1 cursors, T3=end; FA/FB/FT/RES/CARRY/BORROW/ACC/EQ, P,L0/L1; no surviving carry in X/K |
| MUL | x,y -> dst, mod 2^q | MA,MB,MP plus word clobbers; aliases with those three temporaries rejected; address products must fit q bits |
| FIELD | CURSOR/address -> buffer or final DATA bits | A,T0/T3,P and word clobbers; independent address/buffer words required |
| Prefix | Bank base/limit, optional PAR -> J | LO/HI/MID/DIFF/ADDR and word clobbers; saved PAR/CMD survive |
| Lookup/rank | Root/key or root/RANK | NODE/LEFT/RIGHT/COUNT/LEVEL/SELECT/KEY/CURSOR; rank returns FOLDKEY/FOLDPAR and destroys RANK |
| Path copy | CMDKEY,NEWROOT,FRESH, parameters | Frames plus NODE/CHILD/SIBLING/SIBCOUNT/TOTAL/FRAME/LEVEL/SELECT/OFFSET; T2 walks each frame; no arithmetic helper clobbers T2 |
| Fold | OLDROOT,KCOUNT,INITI,CUT | Traversal scratch reused after path copy; next rank saved in INITI because rank destroys RANK |
| Clear | CURSOR=start, parameter 2P | A=physical cursor, T0=c-bit counter, T3=2P; LANE_EQ flags; no q-bit word addition per cleared bit |
| Command return | P=S_base,T3=S,L0=0 | GENERAL, all lane DATA/heads, all latches and microstate explicitly restored by the core |

Parameters are loaded by O(q)-bit WORD_LITERAL setup blocks. Inner word loops
use formed c-bit pointers and tag-bounded bus copies. No q-bit immediate is
repeated at every inner iteration. PRIVATE's LP literals address fixed flags;
they have c bits and their fetch cost is paid by the ISA ledger.

The journal branch rejects q<c rather than truncating a dynamic frame address.
In the registered primary domain this guard should be unreachable: M>=4 implies
`k-2>=8((d+1)(3a+1)+1)` and a>=d+1. For d=1..12 the right side exceeds
`2^d-2`, so d>=13, k>4096, N>8192. The primary L condition then gives q>=21;
c<=q follows for q>=21. This is an unreviewed parameter argument, not a new
domain or a fallback interpreter.

## Literal command structure

Johnson initialization writes raw x[0] directly to DATA[0]; no other persistent
write occurs. Query/flip first read DATA[N-1] into PRIVATE PAR, then execute
`PREFIX` over q-bit LO/HI/MID. MID is `LO+((HI-LO)>>1)`, so HI=N is a sentinel
that is never acquired. Predicate is DATA[MID] XOR PAR. Query emits J's low
bit XOR PAR when N is odd. Flip writes exactly one final bit, `1-PAR`, at J.
Both branches emit actual CLEAN setup, not a call to a host cleanup function.
The one-write/wear conclusion still requires the valid-cycle invariant and a
fresh attack on this emitted source. Johnson is not the primary confirmation.

The primary source initializes only base bits from raw x. Each runtime command
reads role/started, forms physical bank addresses and binary-searches NEW flags.
Root/fresh/chunk address products use at most three source MUL sites; none is
inside a trie-depth loop. Child pointers are physical addresses.

Path copy captures d frames, fills one fresh leaf, then fills d fresh ancestors
bottom-up. Each child/count field is first computed privately; FIELD_WRITE
writes each persistent bit directly to its final value. Leaves retain count one
when parity becomes zero. Internal parity is explicitly zero. Publication of
the valid flag precedes maintenance; CLEAN/RETURN follows maintenance.

First-half maintenance uses INITI=2*JNEW-1 and CUT=2*JNEW, selecting at most two
OLD ranks. Missing ranks exit; parity-zero ranks still advance INITI. Second-half
maintenance clears exactly 2P bits using A and a c-bit counter. At JNEW=M,
clearing completes before role toggles and started is written one. Query tests
STARTED and `J<HALF` **before every OLD persistent access**. For J=0 it looks up
all OLD keys; otherwise it applies the rank-2J cutoff only if 2J<KCOUNT.
Clearing-phase queries never consult an OLD root, node, count or flag.

There is no mutable CODE, runtime root cache, invocation counter, external
history, wear signal or retained output. Invalid exhaustion remains INVALID.

## Conditional instruction and work ledger

This section is a **source loop accounting candidate**, not fresh admission.
The checker does not substitute Python arithmetic for runtime word operations.

BUSCOPY has `8m+10` executed instructions for destination width m<=c.
LANE_EQ has `17c+15`. Ordinary word copy/add/subtract/compare/equality/shift
and lane transfer fit the conservative bound `256q(c+2)`; persistent field
read/write fit `1024q(c+2)`. MUL has <=q counted outer iterations, each made
of these word bodies, so `4096q²(c+2)` suffices at its interface. Frame access
uses q formed one-bit PRIVATE accesses with the same word envelope.

The actual full journal source has

```text
38,588 + 72q + (3e-14 if e>c else 0) source instructions.
```

The extra term is literal I=2 setup when I is wider than P. Journal e<=q, so
`40,000+80q` is an envelope. Johnson fits `2,200+10q`; small branches fit
`5,000+40q`. The formulas count source instructions, not dynamic execution.
Each instruction emits at most b+4 bits; using b+4<=2q+19 proves CODE fits
B for Johnson/small branches at q>=2 and journal at its q>=21 lower endpoint.
The exact per-parameter lengths/site counts are provided by inventory rather
than inferred from this loose envelope.

For command execution, discard WORD_LITERAL's <=80q setup instructions from
the fixed full-journal source count (<40,000). The deepest ordinary nesting
is one outer q-iteration prefix/trie/MUL loop, one q-bit word loop, and one
c-bit bus/equality loop. Rank folding repeats at most twice. No depth loop
contains MUL. Clearing has <=2P<=10q² iterations and a c-bit equality loop.
Thus each fixed instruction site is visited at most `10q²(c+2)` times; literal
word setup blocks occur at most once per command on a selected path. Counting
all fixed sites, even mutually exclusive branches and initialization-only
sites, is a conservative overcount. This gives the candidate numerical ledger

```text
n_inst <= 2^20 q²(c+2).
W_command <= (2^60 + 2^35) q^4,
```

using the reviewed conditional core inst/fetch bound and charging cleanup
separately. CLEAN's setup and all JZ operands are counted. Guard decoding is
not treated as a free branch. Raw base initialization uses <=k outer iterations
of the same word/access bodies and one cleanup; a conservative conditional
`W_init <=2^52 N q³` covers it. Johnson initialization is smaller. These bounds
are deliberately loose and await exact source/control proof review.

**Preparation including generation remains unbounded at the model contract.**
No `C Nq^6` coefficient is asserted for a missing legal generator.

## Append-only emitter and the precise unclosed gap

`Image.emit(write_bit)` takes only finite N,k,L, recomputes all derived
parameters, and never receives raw x or command history. It walks source in
order. At every JZ it reruns the same parameter-only walk with one sought
numeric label and one bit-length accumulator, obtaining the target's actual
bit offset. It retains no label/address dictionary and never reads or rewrites
output. Padding is emitted one zero at a time. One recomputation walk is needed
per branch site, not per emitted operand bit. This algorithm is executable
offline; Johnson generation was actually checked.

The host source expander nevertheless uses Python integers, objects, its fixed
word/flag maps, closures and call stack. Their work and simultaneous storage
have **not** been lowered into a controller with <=256 states and <=65536
description bits, using the same S workspace. The actual initialization of
N/k/L/q/d/M/layout offsets, source cursor, address accumulator, append cursor,
padding counter, local Boolean operations and emission remains to be encoded.
There are no generation-phase transition rows in TABLE. This is the precise
construction gap; four unused runtime encodings are no certificate.

The phase interface is explicit but also unadmitted: install/seal the one image
before exposing raw x; paid installation sets PC to the image's parameter-only
initialization entry while all other mutable bits are canonical. Initializer
reads only raw x, executes its own literal parameter setup and CLEANs to PC=0.
Subsequent calls enter PC=0, QUERY opcode bits (0,0) or FLIP (1,0), followed by
a index bits at positions 2..a+1. I[1] is reserved zero. The PC initialization
is one installation phase, never a runtime clock, history field, second CODE
image or externally selected command entry. `install_init_pc`/`run_initial`
are **host checking oracles** for this interface, not its legal implementation.
Generation must pay those PC assignments and seal/phase handoff under the caps.

Accordingly runtime bodies alone do not supply a complete admitted executable
specification of preparation. No actual-machine lifetime follows even if every
runtime body later survives fresh attack.

## Checks performed and limits

All shell calls used `C:/Windows/System32/cmd.exe`, login=false. Full governing
reads included AGENTS, the complete dual-loop skill, RAM controller protocol,
indexed candidate/review, actual compiler build/design review, journal
candidate/review, and the prescribed historical entry files/sections.

The fixed core check executes SET/COPY/XOR/EMIT followed by CLEAN, with harness-
prepared P=S_base, T3=S and dirty GENERAL. It acquired **31 CODE bits**, cleared
**460 GENERAL bits**, emitted the expected bit, and reached canonical RETURN
after **105,218 row transitions**. This tests actual rows and cleanup ordering;
it does not test bytecode CLEAN setup. Eight one-bit adder/borrow identities
were also checked. These checks all passed.

An earlier Johnson example initialized raw one at N=1,L=2 and reached canonical
RETURN. The following query hit the two-million-visit construction-check cap.
Its diagnostic is retained in `.codex_tmp/endurance_lowering_debug.txt`.
This is **uncompleted validation**, not a query failure/counterexample, timing
benchmark, successful Johnson command run, or evidence against the compiler
class. The final `--check` explicitly reports both Johnson and journal execution
unvalidated. No registered FULL, workload grid, solver, GPU or agent was run.

Only the two authorized new source files, this report, and the authorized
scratch prefix were written. No existing code, proof, protocol, ledger,
status or git was modified. Unrelated tiny-certificate result paths appeared
externally during this session and were left untouched. Suggested root commit:
`construction: emit bit-level Johnson/journal source and 252-state core; record generator gap`.

## Narrative

The honest story is that the storage candidate now has concrete bit programs
instead of unspecified assembly helpers, but its finite preparation controller
is still missing. That is a useful construction component; it does not change
what computation can presently be claimed to do. The strongest triviality
attack still stands: writing a host compiler is not proving an honestly priced
machine can install and execute it. The required fresh attack must confront
the generator and actual source invariants before any positive promotion.
