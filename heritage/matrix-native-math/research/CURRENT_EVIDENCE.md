# Current evidence

## 2026-09-29: speedup claims relabelled (corrections; no new evidence rows)

No row below quotes the "workload pipeline 2.8-14x" or "Wedderburn instrument 1.9-5x" figures (they live in unit
documents and ledger rows 1138-1148 and 1162-1166). Cite them only as follows, per the single-thread re-measurements in
`../results/record_corrections_2/` (`code/record_corrections_2_checks.py`; Errata E11-E13 in
[RESEARCH_MONOGRAPH.md](../RESEARCH_MONOGRAPH.md)):

| Claim as previously worded | Corrected reading | Numbers (single thread) |
|---|---|---|
| Workload pipeline v2: compiled kernel 2.8-14x over 24-thread BLAS | Win over a dense GEMM chain only. The kernel is a direct O(N^2)-per-member DFT that returns eigenvalues only; `np.fft.fft2` (the script's own correctness reference) is the right specialist and is faster; the structure is domain-obvious | fft2 is 56x / 526x / 616x faster than the kernel and 59x / 1870x / 2240x faster than the dense chain (n_side 16 / 32 / 32, T = 16 / 16 / 64); still 18-451x faster than the dense chain when the full N x N product is materialised |
| Wedderburn instrument: fair advantage 1.9-5x from m >= 24, "over the strongest baseline" | Win over a dense GEMM chain only. FFT along the block axis plus one batched matmul per composition is faster than the emitter | specialist is 3.3x / 2.8x / 3.1x faster than the emitter at m x nb = 24x64 / 32x48 / 48x32 (N = 1536, T = 16); 5.4-9.3x if the complex128 cast is charged |
| Labeling recovery: joint diagonalization "dominates" recovery + FFT at n <= 384 | Small-n constant-factor observation (O(n^3) GEMM materialisation against an O(n^2) decode). The Middle-tier withdrawal stands because the route is classical | decode is cheaper from n of about 150; GEMM / decode = 0.32, 1.28, 2.4, 4.8, 10.5 at n = 96, 192, 384, 768, 1536 |

EMPIRICAL, ratios only, on a machine shared with unrelated load; an independent evidence gate is pending. The B382-B389
rows below compare against dense GEMM as their tables state and were not re-examined here.

## 2026-09-08: no new DPS48 comparison evidence

The [registered retained comparison](DPS48_RETAINED_RECONCILIATION_V1_DISPOSITION.md)
is independently closed EXECUTION_VOID after an initial monitor failure, before
source replay. No candidate comparison, H0/H1 result or speedup follows.
Its supporting proof/code gates remain conditional implementation evidence.
The [two-launch operational diagnostic](runtime_lifecycle_probe_v1/DISPOSITION.md)
records a console-host observation in a new trivial process, not a historical
child identification or a scientific result. No rerun authority remains.

The entries below retain earlier accepted evidence. Historical preparation and
successor descriptions do not override [CURRENT_FOCUS.md](CURRENT_FOCUS.md).

The [CRC approximation](CRC_APPROXIMATION_V1_DISPOSITION.md) has independent
PROOF GO for a dimension-uniform fixed-accuracy representation in its registered
family. Exact state growth does not imply fixed-accuracy hardness. No speedup.

The [CRC fixed-order theorem](CRC_FIXED_ORDER_CLOSURE_DISPOSITION.md) has independent
PROOF GO: width-one factors need growing exact merged state on the same row snake.
The retained word is already O(N); no runtime advantage or fixed-tolerance claim.
The separate [CR construction](CR_TWO_FACTOR_PROOF_DISPOSITION.md) has independent
PROOF GO for an explicit four-state realization per flow; no speedup is earned.

## 2026-09-07 update

The [Hilbert traversal obstruction](MESH_HILBERT_PROOF_DISPOSITION.md) and supporting G-SS lemma have independent component PROOF GO. The [receipt](MESH_HILBERT_PROOF_RECEIPT.json) binds exact proofs and review artifacts. This excludes the declared dyadic Hilbert traversal as a universal constant-width choice, with a shrinking precision bound. Literature completion, novelty, all-path implications and positive capability remain open.

For current execution and the accepted butterfly recovery proof gates, see
[CURRENT_FOCUS.md](CURRENT_FOCUS.md). The
[historical displacement timing audit](DISPLACEMENT_BENCHMARK_EVIDENCE_AUDIT.md)
extracts the saved chain measurements and qualifies their scope: dense product
materialization, not vector-application acceleration. It also records missing
historical ledger/default-thread references without fabricating replacements.

The [small-block exit](SMALL_BLOCK_RESTRICTION_EXIT_V1.md) now has independent
proof GO: no exact rational/complex M3 restriction into M2^3 plus scalar.
This is a closed template, not a general rank-22 or border obstruction.
The constructive successor overlaps local product identities and seeks exact
term sharing. Its local proof and known 23-product control passed review.
The [bounded search](OVERLAP_SEARCH_V1_DESIGN.md) has conditional design GO;
the [amendment](OVERLAP_SEARCH_V1_AMENDMENT.md) passed independent delta review.
The four odd correction products provably survive the specified merge rules.
The exact engine and runner are implemented: 30 targeted tests pass and all six
known-control schedules return 23 products. The [implementation receipt](OVERLAP_SEARCH_V1_IMPLEMENTATION_RECEIPT.json)
binds code, tests and raw smoke. The [literature receipt](OVERLAP_SEARCH_V1_LITERATURE_RECEIPT.json)
completes the pre-code gate. Independent code review passed; the full campaign
reports DOMAIN_NEGATIVE with best 23. Independent evidence review passed.
No new circuit or performance advantage was found.

The imported [CN122 source audit](REVIEW_CN122_ADD55_SOURCE.md) independently
replayed the published 23-product/55-addition circuit: all 729 integer
coefficient identities passed. Its [receipt](CN122_ADD55_SOURCE_RECEIPT.json)
binds the retained source and review. This is a stronger constructive comparator,
not a project discovery, runtime result or replay of the source optimality proof.
The [prior-art check](OVERLAP_SEARCH_V1_PRIOR_ART.md) also distinguishes the
known commutative scalar 21-product formula from ordered block multiplication.
[Pre-code diagnostics](OVERLAP_SEARCH_V1_PRECODE_CHECKS.md) passed independent
proof review: 2653 exact parameter tuples and a 1077 decomposition-mass bound.
The [receipt](OVERLAP_SEARCH_V1_PRECODE_PROOF_RECEIPT.json) preserves the reviewed
pending text and terminal GO. These checks do not authorize pruning or a run.


Earlier accepted gates: the [multivariate minor obstruction](MULTIVARIATE_MINOR_OBSTRUCTION_V1.md)
closes its fixed-grid global-polynomial contract for n=3 and fixed n>=5, with
n=4 inconclusive. The [unitary proof receipt](UNITARY_SEPARATOR_SUPPORT_V1_PROOF_RECEIPT.json)
records delta GO for the corrected complete family, fixed even n>=4 and shared
global algebraic covers. The original source denominator error and initial NO-GO
are retained. Neither result demonstrates a speedup. Both reviewers exited zero.
Successor selection rejected underspecified dissipative composition and closed
the exact small-block carrier. Its bounded-overlap successor was subsequently
implemented and closed DOMAIN_NEGATIVE, as recorded above; no new circuit or
advantage was established.

The foundational map is [REPRESENTATION_FRONTIER_V2.md](REPRESENTATION_FRONTIER_V2.md).
Its scoped statements passed independent proof and delta reviews; no new algorithm
or runtime result is claimed. It refutes the September landscape's G3 in the exact
complex bilinear model without changing historical B389/B392 measurements.
The [GIRS obstruction](GIRS_REPRESENTATION_OBSTRUCTION_V1.md) passed independent
review, refuting the literal all-graphs DV conjecture by exact existence. It has
no explicit numeric witness or mesh conclusion. The known-class
[tree composition unit](TREE_COMPOSITION_FRONTIER_V1.md) passed its local proof
gate and independent [implementation evidence gate](TREE_COMPOSITION_V1_IMPLEMENTATION.md):
26 fixtures, 260 comparisons, 14 exact cases, 12 small dyadic floating cases,
10 rejected contracts, and one large-integer promotion regression. The reviewer
replayed every saved row and used a second global-state oracle. No implementation
speedup or general numerical stability is claimed. The native reduction and
sharp-growth [theory control](TREE_NATIVE_REDUCTION_V1.md) passed independent proof review.

The [explicit GIRS successor](GIRS_EXPLICIT_APPROXIMATION_V1.md) also passed
independent proof review: a specified analytic family requires exact DV width
Omega(sqrt(n/log n)) on hypercubes yet has width-two DV approximants with
relative Frobenius and spectral errors <=2/n. This qualifies the practical
meaning of the exact obstruction. It supplies neither an explicit rational
matrix nor a floating stability, numerical-hardness, or new-algorithm result.

The [nonlinear scope audit](NONLINEAR_COMPOSITION_SCOPE_V1.md) passed mathematical
review; its sole documentary blocker, an instructor attribution, was corrected.
It withdraws the old unrestricted closure/algebra dichotomy using a classical
symmetric-power family, with conversion and integer bit growth accounted for.
The [separator support obstruction](SEPARATOR_SUPPORT_OBSTRUCTION_V1.md) passed
independent proof review. For the literal corrected real scalar-interpolation
separator and fixed n>=3, a globally required polynomial representation block
already exceeds the embedded matrix size. Removing unused blocks cannot make
that cover yield an exponent bound. Alternative functions, finite-support
replacements, other representations, and the special-unitary construction are
outside the claim. No runtime or general matrix-multiplication bound follows.

The earlier `code/tests` run passed 26 tests. Small direct probes found a complex
reconstruction failure in StructuredFamily and verified that the implemented
110-to-011 hopping move changes dipole moment. The old 3x3 component count of 211
was reproduced for that move. These are implementation/model scope findings.
See [LIVE_STATE.md](LIVE_STATE.md) for the current gate and continuation.

The tables below are dated historical evidence. Operational LIVE/RUNNING labels
do not establish a current process. Source receipts control each scientific claim.

Status: **LIVE — claim/evidence/cost register**
Reconciled: **2026-08-02, hybrid group-induced profile campaign**

This is the compact evidence surface from the B382 structured controls through
the current terminal gates. It does not replace immutable registrations,
results, reviews, or the append-only ledger. Full review lineage and exact
review hashes are in [REVIEW_LEDGER.md](REVIEW_LEDGER.md); registrations are in
[REGISTRATION_LEDGER.md](REGISTRATION_LEDGER.md).

## Claim firewall

No row below establishes a new arbitrary-dense matrix-multiplication algorithm
or an improved general exponent. `PROVED` applies only to the stated theorem
hypotheses; `EMPIRICAL` applies only to the stated implementation, population,
field, precision, and baseline. `VOID`, `KILL`, and `QUARANTINED` are not
negative theorems.

## PROVED / reviewed exact statements

| Claim | Exact scope | Canonical evidence | Cost and boundary |
|---|---|---|---|
| `sigma(T_2)=2`, `sigma(T_3)=3`, `sigma(T_4)=3` | Exact characteristic-zero witnesses plus stated lower bounds | B414 notebook SHA-256 `2f719b231f699caaea7798389da582190c2e1fc114f04a091bd35fa7b80e92aa`; B414 review SHA-256 `938a18d939f86ea23c0bd896a3a22098805de3e85462bd829bf5f45273b25ea3`; earlier exact reviews indexed in the review ledger | Finite certificate statements; no implementation-speed or general-exponent claim |
| `sigma(T_5) in {3,4}`, `sigma(T_6) in {3,4,5}`; `sigma(S)=3`, `sigma(S^2)=5` | Exact reviewed bounds/witnesses under the named tensor definitions | Combined B416/B418/B420 review SHA-256 `f2165bf39bae81d414bb0277895f98b5a3ad6cd16e3083afec56b270bcbfb3ef`; AMP review in the archive index | B417’s T9 arms are excluded: they closed resource-void-equivalent and prove no T9 bound |
| Binary no-child-collision / transversal creation law | Legal nondegenerate binary KM flips, all four signed variants, with the exact nonzero/projective hypotheses | [INVARIANTS.md](INVARIANTS.md); held source `INVARIANT_FORMULATION_R2.md` SHA-256 `1e31b4fe25f2dc0d50e3ce9105649efb967d2e901f0859ebdcab58adcaa11dc4` | Local law only; plus and general circuits prevent a grammar-wide depth theorem |
| Canonical 526-dimensional Goursat core | The quotient object and induced isomorphism for the frozen gauge-reduced Brent-compatible T4 relation | Theory SHA-256 `9b06a1bcdae09164041bb86d0778b40303810a29a339896bfa4319ac9747c80a`; zero-trust review SHA-256 `81093559d50e707b425693f5179562d59bd1edad48e00f347045de9bab9abe2a` | `PROVED` for the linear-relation theorem; the frozen GF(2) instance is exact finite evidence, not descent sufficiency |
| KMW seeded Brent V6 full-18 negative | No F2 assignment exists in the frozen 18-site restricted chart (778 raw factors, 3,654 active cubics); by spectator-anchor extension, all registered singleton/pair subcharts are also empty | [KMW_SEEDED_BRENT_SIMPLE_V6_RESULT.md](KMW_SEEDED_BRENT_SIMPLE_V6_RESULT.md) SHA-256 `709ae1d7432d4dc88e0089bf0f6ce511f69c625e34e5fc41de4665933b24ba16`; verdict `../results/kmw_seeded_brent_simple_v6_full18.out.json` SHA-256 `82a5c3dac3415bef14309a070aa6076de6eb1179fbeb35229953c9e4b9fe5d04`; accepted RUP `c32d7277...` | Exact finite F2 theorem for this frozen chart only; no unrestricted rank-152, exact-Q, wider-grammar, or matrix-rank lower-bound inference |
| KMW chart sweep V1 negative atlas | No F2 assignment exists in any of the 76 immutable registered KMW-153 charts: 36 wedge targets, 36 nearest-coordinate widened site hulls, 24/30/36 pair-term charts, and all pairs plus nine elementary remnants | [KMW_CHART_SWEEP_V1_RESULT.md](KMW_CHART_SWEEP_V1_RESULT.md) SHA-256 `2a6a87ff7285daf814b6ff22735a6c9ebd2216d32c1ef8f20d034b2c39addbd7`; atlas `../results/kmw_chart_sweep_v1_atlas.json` SHA-256 `03b7fb98f7248960368217af9769c4cd4f2bbffa924b4c8b1674763c98bfcabd`; fresh all-RUP audit `b3f84329...` | Exact finite F2 theorem for these charts only. The 76 labels comprise 40 distinct CNFs; 69 have hard constant contradictions, 72 have one-addition root RUPs, and four require longer proofs. No unrestricted rank-152, exact-Q, other-chart, or matrix-rank lower-bound inference |

## EMPIRICAL / exact finite evidence

| Unit | Evidence-bounded result | Canonical artifact and SHA-256 | Full accounting boundary |
|---|---|---|---|
| B382 | Diagonal/Fourier structured dense output reaches parity near `N=1536` and wins in recorded larger captures | `../results/b382_diagonal_dense_output_cpu_pinned1_validated.json` SHA-256 `6d41d5c64a1ae4827752d00032c6afac5de2c0b892f361b1fefe009a72e00f1b`; large capture `../results/b382_diagonal_dense_output_cpu_pinned1_large_validated.json` SHA-256 `8397ce18006dacb807cabd8e80071242a3a04568f2c277c1fc789e986917e029` | Promised structure only; transforms, conversion, output, precision, and thread policy remain charged |
| B383 | Real Walsh/Hadamard structured dense output wins in the pinned captures at `N=513,1025,2049` | `../results/b383_hadamard_dense_output_cpu_pinned1.json`, SHA-256 `168e4a21200193e9425910d8a93b145fbaa3ea6d4fa52a02c48c6687ee7bd973` | Promised Walsh structure; not arbitrary dense multiplication |
| B384 | Block-cyclic dense-matrix atoms provide a known structured control | `../results/b384_block_cyclic_cpu_pinned1_n2048.json`, SHA-256 `48089257b78bdcc454307ec35c5194600766e42a7c5d5d828e2d43bc0d49f137`; real pinned control `../results/b384_real_cpu_pinned1.json`, SHA-256 `7a1c288ee271218385345c6a55c5cdfd5bb6e8a114b2ac720432377b639f7b36` | Structured interface; known-control status, not novelty |
| B387/B388 | Radical and dual-channel controls have selected pinned wins and default/resident-specialist losses | `../results/b387_radical_channel_cpu_pinned1.json` SHA-256 `925ceea0ff3b88898255c6969dca357ba924bbce72e3ebaed0499c6a500a3b2b`; `../results/b387_radical_channel_cpu_default.json` SHA-256 `c9d68cd21f74aa79814dbfcefd4fce1013b281891507e16fa2a1b1c16a444085`; `../results/b388_dual_channel_cpu_pinned1.json` SHA-256 `1c8cd86b7878a88609af623fd1b0b628c1dd0cd256b8834f8965efa5d524b949`; `../results/b388_dual_channel_cpu_default.json` SHA-256 `4a491a0ac78be59d861d1558181bdeb08f3701033d9a14fe1cf5cea44ebe33f8` | Mixed baseline outcome; rank floors and conversion remain charged |
| B389 | Strongest practical result: optimized real block-cyclic interface beats pinned one-thread resident real GEMM by `16.36x–30.88x` at `n=2048`, error `<=6.85e-16` | `../results/b389_real_block_cyclic_optimized_pinned1.json`, SHA-256 `4a2c28a3765e58ac1f48c2ad579be801b3ff0254fd373bc290cd0034a4ff0f88` | Float64, promised block-circulant family, pinned one thread, full dense output; no arbitrary-dense claim |
| B392 | Exact low-grade Clifford dense-output product has an `O(n^2 log n)` arithmetic envelope but the current CPU realization loses after transforms/conversion | `../results/b392_clifford_cpu_pinned1_compiled.json` SHA-256 `39f933c27365ce006218dd3d0ef1655e989407c8cd67c72319f67563e176e95e`; large `../results/b392_clifford_cpu_pinned1_large.json` SHA-256 `f894f5c77aee021102cbc5974115bfafe9b179225834f714fefd15137501611d`; n1024 `../results/b392_clifford_cpu_n1024.json` SHA-256 `833939c64778081e2502a841617ee1c34dd1cd140c9c3222f2d5cbb09311640b` | Analytical control only; practical gate negative |
| B442 | Exact bounded motif search found no strict fusion event in its frozen 20-seed paired budget | `../results/b442_motif_search/final.json`, SHA-256 `f5502f7ffa0fd3be17ec44da72f40299c0f0347a5c85d9829b6f4dce2c631473` | `NO_STRICT_FUSION_WITHIN_BUDGET` plus low-event-rate inconclusiveness; not a nonexistence theorem |
| T4 global radius three | Complete frozen F2 plus-free schoolbook-T4 census found zero reduction-positive parents/children across 90,979,200 legal third-flip arcs | `../results/t4_global_r3_frontier.json`, SHA-256 `abfb750e48893e48de2e7ec9f826d5436394fbe392e1ad2e382d6ef8a650bb9d` | Exact finite F2 radius-at-most-three population only; no exact-Q or wider-grammar claim |
| T2 pullback oracle | Complete shortest-DAG candidate with publication replay over 33 edge shards and 248 three-way cases | `../results/t2_pullback_oracle.json`, SHA-256 `74d59347755085bc58539e37ebebf207bf6d844b36dd42c9cc76820eb88794bc` | Candidate complete; fresh zero-trust confirmation was not recorded by its terminal review |
| KMW transversal T01/T04 | `EMPIRICAL`: T01 and T04 are F2-empty at exact depths one and two in the frozen pure-transversal grammar; D3-D7 are unresolved | Time-box summary SHA-256 `6ab83330c012773920064de460ecc36d7f13ecc4339eae835f250e4c3330a689`; seal commit `7bbe1e3d`; ledger `KMW_TRANSVERSAL_WEDGE_TOP6_V1_PRODUCTION_SEAL`; adjudication `KMW_EVIDENCE_ADJUDICATION.md`; result-only amendment `KMW_RESULT_ONLY_SEAL_AMENDMENT.md` | Four finite empty cells only; no Q lower bound, other target, rank-152 candidate, or historical source replay |
| Depth-geometry spar | Refuted wedge-support `b=1` and the advertised present bidirectional cost escape; retained a narrowly sound stitched-forward construction question | `SPAR_DEPTH_GEOMETRY_R1.md`, SHA-256 `309cfa6a59c14a89a5f080be101d1f9b5d894d0a896f9ee93545e88b29558010` | Small CPU-only exact probes; no D5/D6 feasibility evidence and no search outcome |

## VOID / resource-stopped

| Unit | Honest terminal meaning | Evidence | Forbidden inference |
|---|---|---|---|
| T4 plus-seeded Stage B | `VOID_T4_PLUS_CENSUS_RESOURCE` after the complete-population projection exceeded the amended cap | `../results/t4_plus_seeded_predicate_v1/stage_B.json`, SHA-256 `3db10cac78b2a5487f691a217b9f1b214bcf3c9fd4d4470144d363d3b1da33f9` | Partial zero is not an empty-predicate result |
| T4 R6 stratified sample | `VOID_T4_R6_SAMPLE_RESOURCE` at the registered projection checkpoint | `../results/t4_targeting_v2_r6/evidence_review.json`, SHA-256 `cb5e4246c2b78c05fcc715bb79e0102b867ef4aa8d2bf243700662f2b8fd2adc` | The stratum-major partial shard is not an unbiased density estimate or global PARK result |
| KMW depths three through seven | `UNRESOLVED_DEFERRED_RESOURCE` for ten cells | Sealed ledger/worklog events | Not `EMPTY`, not a depth lower bound, and not evidence of impossibility |
| KMW GF3 18/all-36 charts V1 | Scientific 18-site chart is VOID: two UNSAT statuses exist but the required RUP certificate is absent after native proof-fallback failure; all-36 was not opened. The planted control passed. | [KMW_GF3_AND_MOONSHOT_V1_RESULT.md](KMW_GF3_AND_MOONSHOT_V1_RESULT.md) SHA-256 `de2d5cf510e72b61ad2a8112a6e5808b5893bba99230bd67c45ffc1fc323749d`; Stage 1 seal `../results/kmw_gf3_and_moonshot_v1/stage1.seal.out.json` SHA-256 `fa02a38f2bf586639868388fb9332cbafe35b1220edbc0bc4f3dc6a1d76e8069` | Not GF3 UNSAT, not a chart nonexistence theorem, and not evidence over Q |
| KMW fully free F2 moonshot V1 | Exact 3,221,748-variable / 11,286,864-clause CNF completed; Kissat404 exited with `0xC0000005` before a status at 1.57 GiB peak | Stage 2 seal `../results/kmw_gf3_and_moonshot_v1/stage2.seal.out.json` SHA-256 `0b6f6b2df1ae00e0d92ee254e7b803ba74e0cb9968d5c91a53a45c5bd8d0c995`; encoding-complete artifact SHA-256 `33e9d2a08c96c85059906490e73af822279abbb0665d98d21a07aa49ee0c74dd` | Not SAT, not UNSAT, not cap exhaustion, and not an unrestricted F2 lower bound |
| KMW GF3 18/all-36 charts V2 | Runner/registration schema mismatch raised `KeyError: choice` during planted-chart measurement before any solver call; no planted or scientific verdict exists | [KMW_GF3_AND_MOONSHOT_V2_RESULT.md](KMW_GF3_AND_MOONSHOT_V2_RESULT.md) SHA-256 `b6bebf522435965f532415f6d8ee03f07ca65980249e137ad7132fd130c5fadc`; Stage 1 seal `../results/kmw_gf3_and_moonshot_v2/stage1.seal.out.json` SHA-256 `fa8e719be275170849fe3f738464d8777af2bbd7f170f8dd363cfc21e363f9ae` | Not a GF3 solver result, not GF3 UNSAT, and not evidence for or against the parity-dodge axis |
| KMW fully free F2 CaDiCaL probe V2 | Retained full DIMACS loaded in 13.726 seconds; CaDiCaL interrupt did not return before the exact 600-second Job close, leaving no result or search-stat snapshot. Probe VOID kept the three-hour attempt closed. | Probe seal `../results/kmw_gf3_and_moonshot_v2/stage2.probe.seal.out.json` SHA-256 `075f03d25acfde92c7209b0b01e6fd480fb9ea1d63bc53d4efd42bc734a09606`; load-complete SHA-256 `e11be09cef9276ee8f3cc2dbbdfba6007810aaf91d4b12db3287e2733278c1b6`; peak Job memory 3,831,132,160 bytes | Full load is not stable probe PASS; no SAT/UNSAT, search-rate estimate, or unrestricted F2 lower bound |
| KMW GF3 18-site certification Portfolio V1 | Exact retained chart is CaDiCaL+Glucose UNSAT; fresh propagation-only RUP check accepted all 2,754 additions ending empty | [KMW_MOONSHOT_PORTFOLIO_V1_RESULT.md](KMW_MOONSHOT_PORTFOLIO_V1_RESULT.md); certified seal `../results/kmw_moonshot_portfolio_v1/stage1.certified.seal.out.json` SHA-256 `f1c7432d7f537150fff50234b4cd6366412090d954ba18b4f58fcf01fb97155b` | Exact GF3 18-site pair-block chart only; not all-36 GF3, fully-free F2, or Q |
| KMW fully-free F2 Portfolio V1 | Four complete-load independent attempts charged 21,600 seconds: CaDiCaL default, SAT-phase, UNSAT-phase, and Glucose all reached their 5,400-second wall caps without SAT/UNSAT | [KMW_MOONSHOT_PORTFOLIO_V1_RESULT.md](KMW_MOONSHOT_PORTFOLIO_V1_RESULT.md) SHA-256 `ad0dd0822b6fbad177fbcc2a8ff60857fae746dd3d4ba4d18083e8cf9c206bae`; Stage 2 seal `../results/kmw_moonshot_portfolio_v1/stage2.seal.out.json` SHA-256 `4330a025f7271d86d20b6ed709578d5f8e88ee9235e60f291d4655fd5e96d641` | `VOID / portfolio cap`; not SAT, not UNSAT, not a lift or candidate, and not an unrestricted F2 or Q lower bound |

## KILL / refuted routes

| Route | Disposition | Surviving value |
|---|---|---|
| B391 Heisenberg carrier count | Speed route killed because the carrier retains a full matrix block | Negative control against hiding ordinary MM inside representation accounting |
| Rank-22 Laderman artifact | Proposed 22-term certificate refuted | Exact refutation survives; it proves neither rank 22 nor a lower bound |
| Wedge-support depth law | Refuted by an exact legal `(W,H)=(2,1) -> (0,0)` flip | The projective zero predicate and local no-child theorem survive; `W` is not a distance |
| Current advertised bidirectional cost escape | Refuted as an independently terminal-rooted cone | Pre-run r1 returned NO-GO on the predecessor executor; the one allowed repair remains unreviewed and mandatory full forward stitch replay still governs |

## QUARANTINED / held

| Surface | Condition | Consequence |
|---|---|---|
| KMW r13 production lineage | Post-seal tuple with the same 4-empty/10-deferred/0-token scientific result | Permanent quarantine/archive provenance; not canonical and not deletion-authorized in Round 1 |
| `T4_STEP_TWO_WALL_V1` result root | Exact durability tuple received zero-context GO; detached CPU worker launched from seed 48 and first observed durable seal is seed position 49 | Running/checkpoint state is not a scientific outcome; no complete-census, closure, theorem, counterexample, or breaker inference |
| `KMW_BIDIRECTIONAL_MEET_V1` | Worklog records pre-run r1 NO-GO SHA-256 `a1cbf5e743415f373745f6ea299b3593ae7d2ca8e88a2b98111bac57c255b238`; stable root review path absent while one repair is live | R1 NO-GO is a gate result, not scientific evidence; no smoke, production, or candidate exists |

## How to extend this register

Add a claim only after its immutable result and required review exist. Record the
rigor label, exact scope, SHA-256, representation/encoding/precision/output
costs, matched baseline, and the token or theorem boundary. Put versioned review
and registration provenance in the two ledgers, not in a new round-named status
file.

## KMW jets soft-orbit atlas — 2026-07-29

`PROVED / exact finite`: the seven amended F2[eps]/(eps^2) wound charts at deletions `{61,63,65,69,73,75,77}` are empty. Both solvers returned UNSAT for every chart and fresh watched-literal replay accepted every CaDiCaL RUP addition against each persisted DIMACS. Authority: `KMW_JETS_CHARTS_V1_RESULT.md` (SHA-256 `6b0d28fd6481766714fc16ba73058860a71259c664fd302f4aa3f83b182c94b5`) and `../results/kmw_jets_charts_v1/self_check.final.out.json` (SHA-256 `3c0a55592492a715c573f3c108a70e580c6df9559fb8d84449ceb209ede807de`). This is not a rank or border-rank lower bound.

## T4 rank-46 quotient V1 — exact telemetry VOID

`PROVED artifact result / EMPIRICAL profiles / HEURISTIC interpretation`: four complete orbit-covering S46 x C3 rank-46 F2 Brent discovery arms charged exactly 28,800 seconds and returned no SAT/UNSAT. Z49 control passed. Final `../results/t4_rank46_quotient_v1/final.verdict.out.json`, SHA-256 `28ae91325407fa8f7bf6b94fe04ae9d5715d7adeda79aac66ff0e972a75a7270`; result [T4_RANK46_QUOTIENT_V1_RESULT.md](T4_RANK46_QUOTIENT_V1_RESULT.md), SHA-256 `b3f06a71d02b26801ab77a4ad6189ad0daf6da0b28375c0875c9c3b00d2e322b`; evidence review `18f56c30...` and gate `7183eb8f...`. Default and seed-17 are strongly trajectory-similar; ALS is deeper and propagation-heavy; SAT phase is shallower and conflict-heavy. Leading diagnosis is genuinely hard after the existing quotient, but residual symmetry is not excluded. No F2 rank bound follows.

## 2026-08-02 hybrid group-induced profile evidence

### PROVED  fixed-2-subgroup restriction

For a 2-subgroup `P` of a frozen index action over `F2`, put
`f=|Fix_Omega(P)|`. Any declared stabilizer profile admitting a multiplication
tensor decomposition must satisfy
`sum_H c_H |(G/H)^P| >= f^2`. Nonfixed `P`-term orbits cancel in
characteristic two; the surviving fixed terms must span the rank-`f^2`
A-versus-BC flattening of `M_f`. Together with the reviewed A5 fixed-slice
specialization, this closes exactly 1,426,427 of 10,420,930 declared profiles.
Scope is `F2`, the 15 frozen actions, and their exact subgroup/profile grammar.

Canonical evidence:
[HYBRID_PROFILE_CAMPAIGN.md](HYBRID_PROFILE_CAMPAIGN.md) and
[REVIEW_HYBRID_PROFILE_CAMPAIGN.md](REVIEW_HYBRID_PROFILE_CAMPAIGN.md);
obstruction summary SHA-256
`d439f75a3d5e621c6d78567b6e8d986157c721447d51d347de444e94a99447c4`.

### Exact finite selected-tranche evidence

The frozen 81-key CPU tranche has 14 RUP-certified UNSAT, five theorem UNSAT,
62 VOID, zero SAT, and zero KILL. All 81 receipt bindings and all 14 RUP proofs
were revalidated at closeout. Atlas SHA-256:
`3aef1efda1518612106b0dc0feb56502a4b43a7d63803c4d5d5e42886363d628`.
VOID is unresolved, not negative evidence. No rank-46/92 record, characteristic
zero result, non-equivariant lower bound, or general exponent claim exists.

2026-09-07 constructive design update: SMALL_BLOCK_SLICE_FILTER_V1 passed
independent proof review. Necessary encoder constraints and covering charts
are exact; a restriction, search advantage, or faster algorithm is not established.
The third direction review ended; its proposed cyclic identities remain unreviewed.

The [overlap result](OVERLAP_SEARCH_V1_RESULT.md) records the complete
35.06-second run and its sharing diagnostics. All 13 best endpoints reproduce
the source 23-product decomposition. Independent evidence reviewer session
01a07cd2-0326-76b1-8480-a22772a6c6c0 exited zero with EVIDENCE GO. Three possible successor
mechanisms are identified; no enlarged search or mission completion is claimed.

Historical question-loop proposal: [successor direction](OVERLAP_SUCCESSOR_DIRECTION_V1.md).
Both independent direction reviews are terminal; the disposition below controls.
Current work is routed through [CURRENT_FOCUS.md](CURRENT_FOCUS.md).

Current update: [three-term conic proof](THREE_TERM_REPARAMETRIZATION_V1.md) accepted conditionally; the registered exhaustive fixed-seed census found no eligible patch in 1,771 triples. [Proof receipt](THREE_TERM_REPARAMETRIZATION_V1_PROOF_RECEIPT.json). Both direction reviewers are terminal. The [two-round disposition](OVERLAP_SUCCESSOR_DIRECTION_V2_RECEIPT.json) rejects promotion of the parity bridge and classical Riordan control; constructive representation research remains open with no implementation successor selected. No successor experiment or mission completion.
