# Round 31, droid

**HEURISTIC verdict:** No mission-level survivor. The recurring phenomenon
is *priced access to structure*: factors, residues, correct addresses,
history, certificates, and paths help, but an ordinary machine granted
the same access can run the same method. This does not rule out new models (`AI Moonshots/moonshot-capability-translation/research/FINAL_SYNTHESIS.md`).
**EMPIRICAL** below means exact finite computation at stated sizes, except
where CPU timing is explicit. Unreviewed all-size extensions
remain **CONJECTURE**, not PROVED. Sources are paths under `Projects/`.
M/R have source-tested costs; B/S/P motivate priced toys; none supplies an equal-access ordinary lower bound.
1. **M, structured factors.** Step: append/apply `H P H`, charge both
   q³-value blocks, all k factors and dense extraction. CAN: cheap
   factor-list apply; CANNOT candidate: generic regularly decoded
   product descriptor has dimension `min(q⁴,(k+1)q³-kq²)`.
   **EMPIRICAL:** tangent ranks q=2..5, k=1..3 match; GF(4)
   incidence replays q=4. **KILL:** ordinary factor list copies it;
   nonlinear closed families escape generic rank (`matrix-native-math/research/NONLINEAR_COMPOSITION_SCOPE_V1.md`).
   Source: `matrix-native-math/results/monarch_growth_v1/REPORT.md` (source-reviewed); Dao et al., *Monarch* (2022); *Structured matrix factorization length* (2026).
2. **R, modular product.** Step: multiply binary 2x2 factors
   modulo charged moduli; pay arithmetic, conversion, decoder/CRT.
   CAN: reconstruct when residues are injective; CANNOT candidate:
   smaller modulus aliases outputs. **EMPIRICAL:** exact minima
   t=2..15: 3,5,9,17,33,65,129,195,385,641,1282,2561,
   4099,8195. Full-matrix replay confirms through t=13;
   t=14,15 use exact row-state DP alone. Thus the tempting
   `2^(t-1)+1` all-t law **fails** at t=9..15; joint-matrix
   decoding gains nothing over coordinates at t=2..11.
   Signed {-1,0,1} control instead gives 5,9,15 at t=2..4.
   Decoder-table preparation is NOT free. **KILL:** ordinary CRT/lookup.
   Source: `matrix-native-math/research/DIGIT_GEMM_COMPLEXITY_V1.md`;
   Hayashi et al., *AWE* (2026); Uchino et al., *Ozaki-II* (2026).
3. **A, addressed records.** Step: pay for each keyed-record read.
   CAN: scan k records; CANNOT candidate: fixed first read fails
   on arbitrary key/value assignment. **EMPIRICAL:** exact failure
   fraction `(k-1)/k` at k=2..6, scan zero failures.
   **KILL:** ordinary keyed scan. Source:
   `AI Moonshots/moonshot-llm-genome/THINKING.md` T037.
4. **T, event memory.** Step: pay for append, timestamp, storage,
   and log reads. CAN: answer past-value queries; CANNOT candidate:
   latest-only snapshot loses past answers. **EMPIRICAL:** all
   `4^t` histories t=2..7, log zero failures, snapshot ambiguous.
   Independent forward-state oracle confirms every prefix.
   **KILL:** ordinary timestamped log.
   Source: `MapU/GLOBAL_MEMORY_BENCHMARK_STATUS.md`.
5. **C, commit certificate.** Step: pay one probe plus certificate
   bit. CAN: w probes identify w-bit command; CANNOT candidate:
   <w binary replies cannot certify all `2^w` commands.
   **EMPIRICAL:** exact adaptive minimax w=2..6 equals w.
   **KILL:** same ordinary decision tree. Source:
   `pramana-contract/docs/CONTROLLER_CERTIFICATES.md` (not a chip theorem).
6. **B, cancellation certificate.** Step: charge finite-bit
   rational sign-blind bound for products of `I +/- 0.1 I`.
   CAN: safe bound; CANNOT candidate: bound stays tight on
   alternating signs. **EMPIRICAL:** all sign words at
   d=2,4,6,8,10,12; no bound violation, overestimate
   21x to 36.54x. **KILL:** ordinary interval arithmetic.
   Source: `matrix-native-math/research/CERTIFICATE_LAW_V1.md` (not its real bound).
7. **S, aggregate interference.** Step: charge each signed
   dot product and aggregate operation. CAN: sum squared
   pairwise overlaps in `O(k²d)`; CANNOT candidate: *emitting
   every pair* takes `Omega(k²)` symbols (not the aggregate).
   **EMPIRICAL:** 1,155,664 exhaustive sign tuples match mean
   `choose(k,2)/d`. **KILL:** ordinary arithmetic copies it.
   Source: `Latent-Space-Reasoning/theory/HPE_AGGREGATE_INTERFERENCE.md`.
8. **P, prefix search.** Step: charge each membership probe
   AND acquisition of r hint bits. CAN: find marked leaf
   among `m=2^(h-r)`; CANNOT candidate: *positive verification*
   needs m worst-case probes. **EMPIRICAL:** all targets
   h=3..6, r=0..h; without positive verification and with
   a unique-mark promise, m-1 negatives suffice, defeating
   the broader CANNOT. **KILL:** ordinary equal-hint search.
   Source: `AI Moonshots/moonshot-address-dial/research/GATE_B_ACQUISITION_GAME_CONTRACT.md` (not LDLR).
9. **V, circuit path.** Step: charge edges, transmitting
   gates, masks, readout. CAN: identity-gate path carries bit;
   CANNOT candidate: no path means no input influence.
   **EMPIRICAL:** 161/256, 13,009/32,768 pathless
   masks at widths 2,3, exhaustive; width 4 exact count
   3,839,105/16,777,216. **KILL:** ordinary masked circuit.
   Source: `neuro-ai-lab/docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`.
10. **G, local gauge.** Step: charge within-bank comparison,
    read and calibration offsets. CAN: bank ordering is
    unchanged by bank-common shifts; CANNOT candidate:
    local orders alone cannot fix global argmax.
    **EMPIRICAL:** all 36 and 216 strict-local score arrays
    at 2 and 3 banks admit different global winners under
    allowed shifts. **KILL:** ordinary calibrated argmax.
    Source: `sangam/docs/ACQUISITION_AND_QUERY_ACCESS_V1_RESULT.md` §5.
11. **D, diffusion.** Step: parallel neighbor OR on N
    path cells, charge N cells, links, work, output.
    CAN: marker reaches distance t in t ticks; CANNOT
    candidate: farther sites remain unaffected.
    **EMPIRICAL:** all bit masks at N=8,12,16, t=1..3;
    serial frontier beats sweeps; one output reads <=2t+1 bits.
    **KILL:** equal-N parallel ordinary cellular automaton.
    Sources: `AI Moonshots/moonshot-yagna/README.md`;
    Abelson et al., *Amorphous Computing Manifesto* (1996); not a Yagna rerun.

**What survives:** scoped side facts, chiefly M's finite
dimension checks and R's counterexample to the extrapolated
modulus law. A single-thread NumPy CPU control at q=16,k=2
measured factor apply vs dense build+apply as 0.092 vs
5.27 ms for one query, but 14.60 vs 8.74 ms for 256.
Both are ordinary algorithms; times are NOT lower bounds.
Twenty receipts for eleven dead candidates repeat `docs/LESSONS.md` §10's apparatus problem.
Under `ROUNDTABLE/r23_droid.md`, M/R/B/S lower literally
to ordinary arithmetic (H2), A/T/P/G copy with equal
information (H1), C/V/D copy with equal hardware (H3);
H4 charges preparation, space, precision and output. A priced
unavailable primitive plus a matched useful lower bound remains an escape target.

**Verification/limits:** pre-registrations `r31_PREREG*.md`
preceded corresponding tests; `py ROUNDTABLE/work/droid/r31_verify.py` passes
20 source-hashed receipts. Direct R replay at t=6 stopped
at its 30-second cap (1.8M/16.8M paths); distinct-state
DP completed. A temporal checker bug and two expected-value
typos were corrected and rerun; a tautological log check got a fresh oracle. No independent proof review,
GPU timing or hardware test. Blackboard MCP was unavailable;
the local files are the record. No commit, since `.git` lies
outside the allowed work directory.
