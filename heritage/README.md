# Inherited research library

These records support the unified [research program](../theory/RESEARCH_PROGRAM.md).
They are **historical reference material**, with their original scopes and corrections.
They establish no new result through consolidation. Historical "current", "active",
"ready", autonomy, model-policy and launch instructions inside them are superseded
by root [STATE.md](../STATE.md) and the current program.

| Source collection | Read first | Important companion |
|---|---|---|
| Matrix-native mathematics | [Monograph errata](matrix-native-math/RESEARCH_MONOGRAPH.md) | [Latest source snapshot](matrix-native-math/research/CURRENT_FOCUS.md), [evidence corrections](matrix-native-math/research/CURRENT_EVIDENCE.md) |
| Neural dissection / LLM Genome | [Current corrections first](../docs/UNIFIED_EVIDENCE.md#corrections-that-must-travel-with-any-reuse), then [pause snapshot](neural-dissection/PAUSED.md) | [Scoped memory hypotheses](neural-dissection/THE_LAWS_V3.md); the source's "laws" language does not imply universality |
| Computational models | [Result register](computation-models/docs/RESULTS.md) | [Latest classification review](computation-models/ROUNDTABLE/work/droid/r19_REVIEW.md), [latest negative synthesis](computation-models/ROUNDTABLE/work/droid/r31_FINAL.md) |

Use the [unified evidence map](../docs/UNIFIED_EVIDENCE.md) for corrected licensed
sentences and transfer obligations. The [import manifest](IMPORT_MANIFEST.json)
records source revision, original relative path and content hash for each file.
Public copies with personal path-prefix redactions have both original and public
hashes and an explicit transformation record; source line counts are preserved.
Contextual excerpts record their original line ranges and full source identity.
These are working-tree captures: a source Git HEAD alone does not identify dirty
or untracked bytes. No original Git history has been grafted into this repository.

Two corrections are especially important before reading the Genome pause snapshot:
the alleged post-wash geometry used installation/re-formation checkpoints, and the
optimizer lower-rung bracket is momentum SGD with that rung already holding.
Neither the washed-state wall nor an AdamW-versus-SGD kill-edge shift follows.
Original endpoint read observations and incomplete optimizer-cross obligations
remain separate; the current map supplies their corrected scopes.

Code here is **reference-only**. We inspected relevant implementations; we did not
rerun their experiments, install their environments, or establish portability.
Do not execute a historical queue or assume that a copied runner has its checkpoint,
data or import dependencies. Adopt a consumer into `experiments/` with a declared
dependency set, controls and a new run record when useful.

## Borrow by operation, with its scope attached

This small use layer is static navigation, not tested portability or permission to
run an old experiment. Exact source records remain unchanged.

| Intended reuse | Start at | Scope and minimal dependency/recovery route |
|---|---|---|
| Literal tree-operator application/composition | [TreeOperator.apply/compose (line 113)](matrix-native-math/code/tree_composition.py) | NumPy plus standard library; same tree/physical partition. Composition grows edge descriptors and performs no minimal-rank reduction. The module contains no experiment launch; adopt the needed consumer, not a speedup claim. |
| Representation/scheme conversion cost | [Representation automaton (line 15)](matrix-native-math/code/repr_automaton.py) | Imports omitted `code/framework_floor_checks.py` and references the original reproducibility pack. Both are catalogued in the matrix archive; discover them with source-relative prefixes before adoption. This is a restricted arithmetic model, not a general learned compiler. |
| Priced defect propagation | [Blockwise recurrence (line 68)](matrix-native-math/code/near_algebra_blockwise.py) | NumPy plus standard library. The recurrence maintains descriptor/bound state internally; the runner computes dense comparisons and returns measurement rows, not a reusable descriptor object; "true distance" is a projection-residual metric. Its CLI writes result files. Use the [current correction](../docs/UNIFIED_EVIDENCE.md#corrections-that-must-travel-with-any-reuse) before borrowing the denominator. |
| Parameter restoration or optimizer intervention | [Saved-baseline routine (line 965)](neural-dissection/lab/e299_embalming.py), [optimizer paths (line 561)](neural-dissection/lab/e347_optimizer_cross.py) | These runners import omitted `lab/e043_install.py`, `lab/g1_anchored_ball.py`, and `lab/e261_rank_ladder.py`; they are catalogued in the genome archive. Importing the runners can invoke hardware checks, create run directories and open logs. Read/extract the specific mechanism, recover explicit dependencies/assets, and review a new consumer before any execution. |
| Compare a new latent action interface with the old bottleneck | [Actual branch transition (line 154)](../experiments/cloud/lsr_jobs/branch.py), [P2 updater (line 60)](../experiments/cloud/lsr_jobs/p2.py) | Existing package dependencies remain in `experiments/cloud/lsr_jobs`; runtime uses model/adapter libraries and GPU code. These are frozen reference jobs, not a CPU smoke or adopted queue. They retain recurring backbone execution and do not establish its replacement. |
| Apply a conditional behavioral or restoration proposition | [Continuation scope/status (line 15)](../theory/CONTINUATION_REFINEMENT.md), [lease admission register (line 151)](computation-models/docs/RESULTS.md) | Read the exact observer family, horizon/alphabet, returned/borrowed-state promise and later correction/admission record. A historical proof-status label does not supersede its later disposition; this consolidation renews no proof gate. |

For omitted dependencies, `python tools/heritage_assets.py list <source> --prefix
<original-relative-prefix>` discovers the receipt members; selected restoration
preserves bytes and never imports or deserializes them. For an actual construction,
pin the dependency closure, input/checkpoint versions and import-time effects in its
new registration. Recovering an old module is not adopting its historical scheduler.

Full source histories, notebooks, data, checkpoints, private records, ignored files,
untracked work and administrative instructions are captured in verified local
archives. Check [consolidation status](../docs/CONSOLIDATION.md) for actual source
retirement and cleanup status. See [asset custody and recovery](../docs/ASSETS.md).
An old link absent from
this selective library refers to an archived source member, not a deleted insight.
Use its source-relative path with the asset tool. Exact historical absolute paths
describe the original run environment and are not current execution locations.

Owner-authored imported material is maintained here by the same owner. Third-party
literature, vendor/model weights and external dependency trees are not republished
as part of this import. Their terms remain separate from the repository's license.
Historical source citations and dependencies are not a license to redistribute them.
