**RAM-FAIL. The ONE/ONE `CLOSED-BIT-LOOP-LOWERING` repair is exhausted. RAM-VOID does not fire.**

The required machine contract is defined, but this delivery does not construct its uniform generation/installation controller. `Image.emit` remains a host compiler, and [`install_init_pc` / `run_initial`](<LOCAL_PROJECTS>/new-computation-model/experiments/endurance_bit_compiler.py:815) remain host phase-interface oracles. The runtime table contains no generation-phase rows. Neither four unused state encodings nor an implicit source ROM, code-copy routine, or external controller closes that gap.

Consequently, no complete compiler, actual-machine lifetime CAN, unconditional preparation bound, or primary RAM confirmation follows. This is failure of this bounded construction; it establishes no all-compiler impossibility.

**Component findings**

| Target | Finding and surviving scope |
|---|---|
| Exact runtime description | **CHECKED COMPONENT:** 252 states, 521 rows, 68 bits per row, and 36,688 count/row bits. Every saved JSON row matches source; every encoded bit matches state-list serialization. Guards are disjoint, coverage matches the declared domains, targets are valid, and the maximum row count is 16. |
| Codec and controller-description cap | All actual rows round-trip through `decode_rows` in **state-list order**. This verifies serialization. It does not certify the separately encoded codec/guard decoder or complete preparation controller. The old 4,096-bit codec allowance remains uncertified. |
| Fetch, operands and branches | LP/JZ use serial `c`/`b` operands; LANE uses four selector and three action bits. Fetch preserves L0/L1/O/D, uses X/K, explicitly ripples PC, and homes PC/C. All source JZ targets at the four fixed report points resolve to instruction boundaries. |
| Arithmetic and helper clobbers | The lowered one-bit carry/borrow bodies pass all Boolean cases. Source inspection preserves carry/predicates privately across fetch. Actual call sites respect the relevant aliases: arithmetic clobbers T0/T1/T3; frame traversal retains T2; PRIVATE PUT saves its value in L1. This does **not** validate every complete word/helper execution. |
| Widths and allocation | The simultaneous allocation includes base lanes/control, 64 q-bit words, q four-word frames, and 21 flags. It gives 729/737 bits at the two q=2 cases and 4,501 at the journal point. Journal frame transfers require `q≥c`; the primary-domain argument supports that guard. Small I transfers explicitly zero-extend when `e>q`. |
| Johnson and small branches | **CONDITIONAL SOURCE FINDING:** the Johnson search avoids acquiring HI=N, including N=1. Initialization has one final-value DATA write. Two-cell and direct-base branches have literal bodies. Their complete commands remain unvalidated. |
| Journal structure | **CONDITIONAL SOURCE FINDING:** physical pointers, zero-padded directions, frame traversal, bottom-up allocation of exactly d+1 nodes, count-one parity-zero leaves, and zero internal parity match the intended scaffold. Fresh persistent fields are written from privately computed final buffers. |
| Fold, cutoff and clearing | INITI survives destructive RANK traversal; folding considers at most two ranks and advances through parity-zero keys. Query tests STARTED and `J<HALF` before OLD persistent access. Computing OLDROOT privately before that test is not an OLD acquisition. Chunk clearing precedes role exchange; publication precedes maintenance; acknowledgement follows maintenance/exchange. These remain source findings, without a complete journal trace. |
| CLEAN and self-control | **CHECKED COMPONENT:** emitted setup establishes `P=S_base`, `T3=S`, `L0=0`. GENERAL clears before its controlling lanes; lane DATA/heads and latches clear afterward. State 251 pays restoration to state zero before checking canonical RETURN. |
| Command/work ledger | Source counts and bounded-loop structure support a **conditional accounting candidate**. They do not constitute measured command visits or an unconditional actual-machine `O(q⁴)` certificate. The evaluator also leaves finite guard-decoder work unadmitted. Preparation remains unbounded at the contract because its controller is missing. |

The fixed source construction points reproduced exactly:

| Family; N,k,L | Useful bits | Source instructions | Closed JZ sites |
|---|---:|---:|---:|
| Johnson; 1,1,1 | 24,511 | 2,159 | 326 |
| Journal; 16384,8192,256 | 515,388 | 40,244 | 5,872 |
| Direct base; 8,4,16 | 13,285 | 1,122 | 153 |
| Two-cell; 2,1,1 | 10,716 | 953 | 138 |

These are **source construction checks**, not a numerical command grid. The journal point matches `38,588+72q=40,244`; its S/B are 135,424/34,668,544 bits. The complete saved journal inventory matches recomputation. There is no saved emitted journal CODE image among these artifacts.

**Actual probe coverage**

- Compared every saved core row and encoded bit; verified codec round-trip and guard coverage.
- Compared all Johnson opcodes, operands, resolved branch addresses and padding independently against `johnson.bits`. The actual host recomputation emitter also matched all 262,144 bits: 24,511 useful and 237,633 padding.
- Reproduced the saved core check: **105,218 row visits, 31 CODE reads, 460 GENERAL writes**, canonical RETURN.
- Executed emitted CLEAN setup from dirty legal heads, lane DATA, latches and GENERAL, with PC at its bytecode entry: **468,008 row visits, 1,333 CODE reads, 460 GENERAL writes**, canonical RETURN and unchanged DATA/wear. Its 239 source bits therefore must not be reported as 239 dynamic CODE acquisitions.
- Executed the lowered Boolean adder/borrow bodies for eight input triples per mode: **16 cases**, 92,541–94,705 visits each, **1,497,968 total visits**.
- Tested isolated fetch carries **31→32** and **262143→262144**, starting PC/C heads at their right ends: **429/559 visits**, one CODE acquisition each, preserved L0/L1/O/D and restored heads.

Two reviewer-harness issues did not supply artifact counterexamples: the first codec comparison incorrectly used TABLE insertion order instead of encoded state-list order; the first CLEAN-setup attempt reached its 200,000-visit cap at state 35/PC 83. Correct-order comparison and the original two-million-visit component cap completed successfully. **No source correction occurred.**

The earlier saved Johnson diagnostic remains incomplete validation: it reports the query cap at state 18/PC 1558, with 2,469,517 cumulative visits including initialization. It supplies neither successful final-source execution nor a functional/nontermination counterexample. **No complete Johnson/journal runtime execution or FULL admission was performed here.**

**Exact reviewed SHA-256 identities**

Research filenames below are under `research/`; component artifacts are under `results/endurance_ram_lowering_components/`.

| File | SHA-256 |
|---|---|
| `experiments/endurance_bit_compiler.py` | `7b802d5140ac272388761370bd0f3f5b25b50e8be946b39fcedc8029b60669e6` |
| `experiments/endurance_bit_microcode.py` | `9055798f4936868b13a44ee4e9f67128c3a69d25cdc9b05895affddf96f6fa49` |
| `ENDURANCE_RAM_LOWERING_REPAIR.md` | `34628271e1b8e8e3f51fb617f2a83adbd3daab657402091ee8d7782f3953800c` |
| `ENDURANCE_RAM_CONTROLLER_PROTOCOL.md` | `032d02f4dd3a63562bb114073c231d08361abb9240c59173937e108629e8df7a` |
| `ENDURANCE_INDEXED_MODEL_CANDIDATE.md` | `3cd1732070d1874c36c3ad9fd267a03f0b98edd76ad27faa0078954ea2fb3129` |
| `ENDURANCE_INDEXED_MODEL_REVIEW.md` | `bee22ae941b9e16b89603de48cde9f9cff9c88d2020dc85267ef2fafdb87d2ac` |
| `ENDURANCE_RAM_COMPILER_BUILD.md` | `650ad0f4637d36ff8b55d8c5e8666ff52a2b02fe367c0d09119daf2e691b185c` |
| `ENDURANCE_RAM_COMPILER_DESIGN_REVIEW.md` | `b0be16a9517585f0ff24410cf59cd816845be3d14c4ff404b9a40dd56bdea1db` |
| `ENDURANCE_JOURNAL_DESIGN_CANDIDATE.md` | `705c445ccfcbe217c7efc783208e99ea35e043a548cb96c507666501221e23c0` |
| `ENDURANCE_JOURNAL_DESIGN_REVIEW_1.md` | `388c86d9b265c6ce8ee8120f9d91b6c6a7b3506db86c23a93e8d8a37fd630f23` |
| `audit.json` | `777c905ab130d1009333eb5f6ded94615de911859edc4629e85bf8c966352abd` |
| `check.txt` | `37f7e27be0bea4f77d41fb6572855f103c989dee2da0ca7da335e8117ed7a71c` |
| `core.bits` | `740bf3ea702724eabba0daff9d566f1be892e3df167878aa404afe97eb38866e` |
| `core_rows.json` | `91fb9ee73e8105f517cff00a915fee6a903da195fc9ba3849692dff0a36a8d17` |
| `johnson.bits` | `d79bc610f043a5be30dd36b500ea64ec3a4ce90492fab9729af322df389d77ab` |
| `journal_inventory.json` | `97a2fe759170005cf31e2f9d57c2fe2745939f0b42c922fd9dfbdc755fa2c7a7` |

Earlier diagnostic `.codex_tmp/endurance_lowering_debug.txt`:
`100bb93298f49e9b1de9c3a37bff39ec3ce747cc45459b3893ed1210ea1f06d3`.

The two source files, lowering report and six component artifacts retained their hashes through the review. Governing files, full skill, prescribed startup authority, required candidates/reviews/reports, and both complete source files were read. Every shell used `cmd.exe`, `login=false`; Python used `-B`. No writes, agents, GPU, solver, FULL run, or numerical workload grid occurred. Other processes changed unrelated repository/status material during review.

**Direction debt for the next genuine advisor dialogue:** decide whether a *newly registered* construction can supply complete parameter initialization, source/control representation, append-only branch-address generation, padding/sealing and initialization-PC handoff within the same S/state/description caps and explicit `C·Nq⁶` preparation bound—or whether a different controller/cost-model direction is warranted. Complete runtime invariant and dynamic-work certificates remain subsequent obligations. This closed unit receives no second repair.

The surviving result is concrete runtime source plus narrowly checked components. **Positive CAN, primary delta, novelty, matched CAN/CANNOT pair and mission admission remain unadmitted.**