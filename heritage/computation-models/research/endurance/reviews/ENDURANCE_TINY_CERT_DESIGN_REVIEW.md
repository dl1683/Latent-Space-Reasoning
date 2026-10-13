**CERT-DESIGN-PASS — abstract formula and certificate design only.** No blocking design defect found. Implementation, native-library behavior, resource enforcement and raw admission remain gated.

Reviewed [certificate protocol](/<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_TINY_CERT_PROTOCOL.md), SHA256:
`4e622f4e89ae706cc6cd84acd1d3962d86fdf4f7212a172f470dc2c715c604aa`.

- **Complete endpoint encoding — PROVED at design scope.** Exactly-one initialization, transition and boundary-state selections admit arbitrary nonlinear policies. Root equivalence fixes `u=I(x)`. For the uniquely selected parent `s` and transition target `t`, `¬u_parent[s]∨¬T[i][s][t]∨u_child[t]` forces the exact successor; child onehot excludes alternatives. Physical-bit equivalences recover its binary coordinates. Decoder implications enforce every required logical bit. Conversely, any legal endpoint policy supplies these assignments and the fully equivalent auxiliaries. Total extension on irrelevant states adds no promised execution obligation.

- **Wear — PROVED at design scope.** Threshold pairs `(0,0),(1,0),(1,1)` encode precisely wear `0,1,2`; `w2→w1` excludes `(0,1)`. With `delta=0`, both thresholds stay unchanged. With `delta=1`, wear advances `0→1→2`, and `oldw2∧delta` forbids `2→3`. Root wear equals initializer bits. Full AND/OR/XOR equivalences prevent auxiliaries from evading changes. Each history retains separate wear variables even when physical states coincide; shared `d/I/T` enforce stationarity without merging wear.

- **Coverage and terminal scope — PASS.** All four raw inputs and every binary FLIP word are included:

  | Horizon | Boundary records | Update edges | Bit-query checks |
  |---|---:|---:|---:|
  | H6 | 508 | 504 | 1016 |
  | H5 | 252 | 248 | 504 |

  These follow from `4Σ₂⁰…ᴴ2ᵗ`, four roots, and two queries per record. Updates are required only before H; terminal decoding remains required indefinitely. A state reached earlier retains its earlier transition obligations. The admitted canonical-boundary model supports this reduction, including nonlinear decoding, nonpreferred initialization, multibit transitions and arbitrary read-only query interleavings.

- **Retaining deleted clauses — PROVED sound.** Let `F` be the original CNF and `D` contain it plus previously validated additions. Unit conflict in `D∧¬C` establishes `D⊨C`. Inductively, `F` entails every addition, so adding or retaining those clauses preserves the models of `F`. Ignoring deletions therefore remains sound, although it checks RUP against the retained database rather than reproducing the exact active DRUP database. Every addition must pass RUP; there is no RAT-only fallback. Acceptance requires a validated empty clause, not merely a nonempty list of validated clauses.

- **Installed APIs — inspected, runtime behavior NOT ESTABLISHED.** PySAT’s installed version is `1.9.dev7`. Its [Glucose3 wrapper](/<USER_HOME>/AppData/Local/Programs/Python/Python313/Lib/site-packages/pysat/solvers.py:4202) enables proof tracing **after** bootstrap clauses are loaded; `get_proof()` materializes the proof in memory. Its [MiniSat22 wrapper](/<USER_HOME>/AppData/Local/Programs/Python/Python313/Lib/site-packages/pysat/solvers.py:7439) returns a propagation-specific conflict flag, while clause insertion can separately latch root inconsistency. `get_status()` is not interchangeable with the propagation return. Native semantics require the registered code-gate validation; I did not execute either backend.

The mandatory code gate must concretely verify:

1. Original-DIMACS readback, complete mappings, onehot constraints and every gate’s full truth table; tiny CNF/SMT/replay agreement, overflow rejection and nonpreferred initialization.
2. Ordinary propagation conflicts, conflicting root units, an original empty clause, empty assumptions, and restoration after failed assumptions. The checker must add each proof clause **only after** validation and never call search.
3. Rejection of empty/truncated proofs, malformed additions and genuine non-RUP clauses. Inject invalid additions against a satisfiable control or before contradiction; after established inconsistency every clause is entailed. Altered-CNF controls must test artifact binding—arbitrary mutations need not invalidate a mathematical proof.
4. Actual Glucose proof production, including bootstrap root conflicts; proof parsing and a checked empty-clause conclusion. Unsupported behavior yields VOID.
5. The complete frozen H6-then-H5 pipeline within **four CPUs, 8 GiB aggregate memory and 3600 aggregate seconds**, including preparation, proof extraction, retained-clause growth, replay, hashing and publication. Proof temporary files belong in the repository attempt area. No adaptive horizon, portfolio or configuration fallback.

**Prior attempt remains VOID.** Attempt `10338110` reports UNSAT with `certificate:null`; its result hash matches the receipt. More decisively, [root_exit.json](/<LOCAL_PROJECTS>/new-computation-model/results/endurance_tiny_6_10338110-3a1b-4f68-bb2b-24117c80f314/root_exit.json) records launcher exit **1**, overriding worker exit **0**. Neither its CPU UNSAT nor any proof-clause count establishes impossibility.

Checked H6 UNSAT plus independently replayed H5 SAT would establish **finite stationary-endpoint optimum 5**: any longer guarantee restricts to H6. It would exclude actual admitted machines at H6 while leaving actual paid-machine achievability at H5 unadmitted. Compiled control, intermediate wear, cleanup, source/code and work accounting remain separate obligations.

**Exact review coverage:** the requested AGENTS, CERT protocol/packet, prior TINY6 protocol/design/code reviews, indexed-model and universal-bound reviews, and attempt result/receipt/root-exit files; prescribed startup authority and dual-loop skill; relevant model definitions, launcher regate, attempt environment, and installed PySAT wrapper/version records. No whole-repository or native-C++ verification claim.

All shells used `C:/Windows/System32/cmd.exe`, `login=false`. No solver execution, implementation, GPU, agents or writes. This is a finite calibration component; the positive journal/compiler objective remains active. No primary novelty or mission-terminal admission.