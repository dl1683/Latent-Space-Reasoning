**DEPTH2-PROOF-PASS.** For \(N=3,k=1,L=2,H=5\), every admitted deterministic, cold, exact QUERY has worst-case observation depth **at least 3**, under both specified kernels. I found no counterexample or required proof repair.

The [candidate proof](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_DEPTH_TWO_CANDIDATE.md:17) survives independent reconstruction:

1. **Reachable-path extension is valid.** Canonical entry, fixed code and identical current QUERY input make the next observation address or answer deterministic from the observed transcript. Contract nonpersistent computation and replace unreachable branches with arbitrary leaves. Promised termination and depth \(\le2\) produce a total Boolean function \(f\) computed by a depth-two tree, agreeing with every reachable answer. This never requires executing QUERY on an off-promise state.

2. **At most two essential variables is impossible.** Let \(P\) be \(f\)’s essential-coordinate set. Initial answers differ, so some initializer has nonzero projection on \(P\), costing at least one change there. Five FLIPs each change that projection because their decoded answers alternate. Therefore
   \[
   6=H+1\le L|P|\le4,
   \]
   a contradiction. This works even when \(P\) differs from the original program’s observation-position union.

3. **Three essential variables force a signed multiplexer.** After root observation \(a\), each child has at most one observation remaining. Dependence on all three variables requires both restricted children to be nonconstant literals on distinct remaining coordinates \(b,c\). Constants, repeated root observations or the same child coordinate cannot supply three essential variables. Complementing/permuting coordinates gives the ordinary mux \(f(a,b,c)=b\) for \(a=0\), \(c\) for \(a=1\).

   Its bichromatic edges are exactly the two paths
   \[
   000\xleftrightarrow{b}010\xleftrightarrow{a}110\xleftrightarrow{c}111,
   \qquad
   011\xleftrightarrow{b}001\xleftrightarrow{a}101\xleftrightarrow{c}100.
   \]
   Thus the graph is \(P_4\sqcup P_4\). The cube isomorphism classifies the graph and preserves coordinate-edge identity; **it does not normalize the actual initializer or grant free complemented initialization.**

4. **Actual wear saturation forces the chosen trajectory’s single changes.** Exact initial decoding implies \(I(0)\ne I(1)\), so choose a nonzero endpoint. Let \(C_{\rm init}\) and \(C_j\) count all actual persistent changes, including intermediate and undone writes. Then
   \[
   6\ge C_{\rm init}+\sum_{j=1}^{5}C_j\ge1+5=6.
   \]
   Hence initialization and each FLIP consume exactly one actual change. Each FLIP therefore traverses a bichromatic cube edge. Single-change behavior is derived for this trajectory, not imposed on arbitrary updates. Same-value writes still pay work.

5. **Stationarity defeats the five-move walk.** Identical persistent bits and current FLIP input give identical executions: volatile/PC/control/port state is canonical, code is fixed, and hidden wear supplies no feedback. An exhausted opposite write is invalid, so it cannot select an alternative promised successor.

   The five transitions consequently form a stationary functional walk in one \(P_4\). A repeated vertex forces a directed cycle; changing moves exclude loops, and a tree permits only a two-vertex cycle. With four vertices, at most two moves precede that cycle. At least three of the five moves therefore traverse the same cycle edge, changing its physical coordinate at least three times—contradicting \(L=2\).

6. **The horizon is fully covered.** FLIPs are required at depths \(0,\ldots,4\), and exact QUERY at depth \(5\) supplies the final decoded change. No post-\(H\) FLIP or off-promise update obligation is used.

**Exact coverage:** fully read these current files, including every line:

| File | Lines |
|---|---:|
| `AGENTS.md` | 1–160 |
| `research/ENDURANCE_DEPTH_TWO_CANDIDATE.md` | 1–71 |
| `research/ENDURANCE_PROBE_LAW_CANDIDATE.md` | 1–93 |
| `research/ENDURANCE_PROBE_LAW_REVIEW.md` | 1–32 |
| `research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md` | 1–82 |
| `research/ENDURANCE_BOUNDARY_REVIEW.md` | 1–30 |
| `research/ENDURANCE_INDEXED_MODEL_CANDIDATE.md` | 1–151 |
| `research/ENDURANCE_INDEXED_MODEL_REVIEW.md` | 1–75 |

Also read the review packet, complete dual-loop skill, README, HANDOFF §17, TERMINAL_GATE_2, SYNTHESIS_GRAPH §29 and STEERING_ENDGAME. This was a scoped theory gate, not an entire-repository terminal review.

**PROVED at this scope:** depth two cannot support the stated lifetime, strengthening the general bound’s depth-two allowance here. No linearity, \(I(0)=0\), free wear or encoding-symmetry restriction enters. No paid Johnson CAN, all-\(N\)/all-\(L\) tightness, novelty or mission-completion admission follows.

Every shell used `C:/Windows/System32/cmd.exe`, `login=false`. Read-only throughout; no solver, numerical run, source writes, GPU or subagents.