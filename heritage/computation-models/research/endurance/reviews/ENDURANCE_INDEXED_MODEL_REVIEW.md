**INDEXED-DESIGN-PASS.** I found no design-blocking infinity, storage, addressing, wear-feedback, or boundary defect in the exact new definition.

Reviewed candidate SHA-256:

`3cd1732070d1874c36c3ad9fd267a03f0b98edd76ad27faa0078954ea2fb3129`

**The finite primitives close interpreter regress as a model class.** The fixed base relation and ten named serial-head lanes are declared operations of this abstract machine. They require no additional interpreter. Their finite transition inventory, state count, description, and compiled routines remain construction debts; the numerical caps do not establish a positive compiler. This conclusion follows from the [base-relation and lane clauses](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_INDEXED_MODEL_CANDIDATE.md:42).

The requested design attacks resolve as follows:

- **State accounting survives.** All ten lanes charge four bits per cell, including end cells, onehot markers and both tags. The allocation includes eight microstate bits and sixteen latches. PC, target, address buffers, temporary lanes, source cursors, stacks and arithmetic must fit within `S`. There is no additional numeric head position.
- **The algebra survives, including `q=2`.** Summing lane widths gives `a+e+3b+5c`; twenty end cells give
  \[
  S_{\rm base}=4(a+e+3b+5c)+104.
  \]
  The stated estimates yield `S_base≤72q+432≤256q²` for every integer `q≥2`. At `q=2`, `c=11`, `b=19`; the possible parameter cases give `S_base=564` or `572`, below `S=1024`.
- **Validity restrictions supply no oracle.** Reserved markers/tags cannot be changed through PRIVATE writes; other control writes must preserve legal encoding. Invalid addresses, outward head moves and exhausted opposite writes provide no recoverable outcome. Promised executions must avoid them. Same-value writes still pay work.
- **Installation closes advice and workspace leaks.** CODE has exactly `B` bits, is parameter-only and sealed before raw `x`. The generator has the same `S` allowance, including its controller, and cannot reread append-only output as workspace. Its emission, padding, installation and source work require accounting.
- **The current-input port is literal and finite.** It returns one charged bit; its addressing state belongs to `S`. Raw initialization input becomes unavailable afterward. Current opcode/index bits supply no invocation count or history.
- **Cleanup is an obligation, not an operation.** RETURN requires the canonical image. Clearing lanes, heads, latches, PC, flags and the clearing routine’s own control must use explicit paid actions. No wide arithmetic, copy, helper return or automatic cleanup is admitted. See the [input and RETURN clauses](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_INDEXED_MODEL_CANDIDATE.md:105).

**Transfer verdicts:**

| Result | Verdict under the new definition |
|---|---|
| Endpoint relaxation | **VALID** |
| Universal A | **VALID** |
| Universal B | **VALID** |
| Universal C | **VALID** |
| Query law | **VALID** |

**Endpoint relaxation — VALID.** At a completed boundary, fixed code, identical canonical control and fixed current command leave only the `N` physical bits as history-bearing input. Hidden wear cannot distinguish executions: an exhausted opposite write would invalidate a promised execution rather than return information. Consequently QUERY defines one history-free decoder, and promised FLIPs define stationary endpoint maps.

Initialization exists for every raw vector. Coordinatewise, endpoint initialization wear and subsequent endpoint XORs undercount actual changes:
\[
I(x)\le w_{\rm init},\qquad
s_t\oplus s_{t+1}\le c_t.
\]
All reachable wear records must be retained. States reached only at depth `H` require decoding; states also reached earlier retain their update obligations. Arbitrarily many queries preserve the boundary. Nonlinear initialization/decoding, multi-bit endpoints and undone intermediate writes are covered. Midwrite crash or overlapping-command correctness remains outside the contract.

**A — VALID.**
\[
H\le L\min(N,2^{N-k+1}-1)-1.
\]
The adjacent-fibre averaging and distinguishing projection remain applicable. One selected initializer consumes at least one projected wear unit; each of the `H` repeated-coordinate flips consumes another. No linearity or single-write restriction enters.

**B — VALID.** Write \(\beta=\lceil\log_2(k+1)\rceil\), distinguishing it from the new CODE-address width `b`. With the original packing definition of \(t_0\),
\[
H\le\max(0,NL-t_0-\beta+1).
\]
Injective initialization forces one input to spend at least \(t_0\) wear. After `H−1` flips, the alternative final branches need at least \(\beta\) cells with remaining wear. Intermediates cannot enlarge their endpoint capacity.

**C — VALID.**
\[
H\le(N-k+1)L+\left\lfloor\frac{(k-1)L}{2}\right\rfloor.
\]
Cumulative actual wear gives monotone levels whose parity equals the binary endpoint. At lifted zero, initialize raw `0^k` and perform the first changing request, charging both phases. Thereafter the history-free controller supplies a stationary trace from parity; unchanged-value requests leave levels unchanged. Required domains can be extended without executing off-promise programs. These premises match the primary floating-code definition and Theorem 2 with physical alphabet `L+1`. [Primary definition and theorem](https://people.engr.tamu.edu/ajiang/Floatingcodes.pdf).

**Query law — VALID.** For every coordinate \(i\),
\[
H+1\le L\min(N,2^{d_i}-1).
\]
Between worn-bit observations, CODE, PRIVATE, local control and current-input reads introduce no history information. Contracting them produces the same reachable binary decision tree. Its observation-position union has at most `2^{d_i}−1` cells; initialization plus repeated FLIP(i) requires `H+1` wear there. Count every acquisition of a worn-cell value, including rereads. Preserve the prior review’s clarification: the logarithmic reformulation also needs the global condition `H+1≤LN`.

**Exact coverage:** Fully read `AGENTS.md`, the complete dual-loop skill, and:

- `ENDURANCE_INDEXED_MODEL_CANDIDATE.md`: lines 1–151.
- `STEERING_CONTROLLER_DECISION.md`: lines 1–41.
- `ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md` / `ENDURANCE_BOUNDARY_REVIEW.md`: lines 1–82 / 1–30.
- `ENDURANCE_UNIVERSAL_BOUND_CANDIDATE.md` / `ENDURANCE_UNIVERSAL_BOUND_REVIEW.md`: lines 1–134 / 1–88.
- `ENDURANCE_PROBE_LAW_CANDIDATE.md` / `ENDURANCE_PROBE_LAW_REVIEW.md`: lines 1–93 / 1–32.

Also read the prescribed README, HANDOFF §17, TERMINAL_GATE_2, SYNTHESIS_GRAPH §29, STEERING_ENDGAME and indexed review packet; independently checked the primary floating-code definition and Theorem 2.

No writes, solver, implementation, simulation, GPU use or subagents. The candidate hash remained unchanged; unrelated git state changed during review. **No positive compiler, `O(q⁴)` command cost, `O(Nq⁶)` preparation, novelty, old-kernel simulation, physical advantage or mission completion is admitted.**