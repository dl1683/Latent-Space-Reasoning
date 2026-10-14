# Asset custody and recovery

The unified program is preserving source insights without keeping three giant
active working trees. Its selective [research library](../heritage/README.md) is
public reference material; raw data, checkpoints, notebooks and histories are
captured in verified local archives. No source is retired before verified custody.
Current per-source status is in [CONSOLIDATION.md](CONSOLIDATION.md).

## What is retained

| Source | Scientific assets retained privately | Public starting packet |
|---|---|---|
| Matrix-native mathematics | Complete Git database and working tree; exact search databases/certificates, numerical outputs, original literature and local kernel/cache dependencies inside the tree | Monograph with errata; selected representation/composition/precision proofs, review text, reference code and small result records |
| Neural dissection / LLM Genome | Complete Git database; current dirty and committed states; every experiment's data, metrics and checkpoints; full NOTES/THINKING/reviews; interrupted optimizer state and draw state; the separately preserved consolidation copy | Historical scoped hypotheses, substrate/pause records and selected reference implementations, with corrections in the unified evidence map |
| Computational models | Complete Git database and current modified/untracked work; exact finite receipts, compressed and raw proof evidence, solver/search outputs and original literature | Result/lesson register, core theory and corrections, later group-classification review, selected reference code and latest negative synthesis |
| Existing LSR | Existing experiment/model/data caches and source-versioned evidence paths remain in place; remote raw shards remain in their existing store | Existing code, theory, experiment ledger and historical state archives |

Moving a file does not prove its original result. The importer records source
working-tree hashes separately from source Git revision. Full archive manifests
preserve dirty/untracked/ignored material that a clone or Git bundle would omit.

## Local recovery

All three verified source archives and their private receipts live under the owner's local
`lsr-research-archives/2026-10-13` directory outside synchronization. It is not part
of Git or a dependency that a public clone downloads. Use `--archives` to point at
a different local custody location. Full manifests contain private source filenames
and are deliberately not committed.

All three relocation and recovery exercises passed. Matrix-native and computation-
model source roots are removed; Genome's files are removed and its empty root awaits
release of an editor lock. Acceptance passed, but the final three-root retirement
receipt and disposable-copy cleanup are still pending. Recovery commands below are
available without reopening the original working trees.

The [asset tool](../tools/heritage_assets.py) lists receipt records or restores
explicitly selected ordinary file bytes with member SHA-256 verification:

```powershell
# Discover a selected experiment's archived files, without loading its model.
python tools/heritage_assets.py list llm-genome --prefix runs/e347

# Recover one data file. The destination is ignored assets/heritage/llm-genome/.
python tools/heritage_assets.py restore llm-genome runs/e347/metrics.json

# Recover a mathematical source omitted from the selective public packet.
python tools/heritage_assets.py list matrix-native-math --prefix research
```

Listing trusts the local receipt and archive size. Restoration verifies each selected
file before installation, refuses overwrite and never executes/deserializes a
checkpoint. It is per-file: if a later member fails, earlier verified recoveries
remain. It does not itself certify unselected archive members or restore all Windows
metadata. The separate private custody procedure records full-member verification,
directory/named-stream metadata and archival Git recovery checks.

Do not restore a whole historical instruction/automation tree and treat it as
current work. Adopt only a needed consumer and its complete dependency set into the
unified program, retaining source identities and a new validation/run record.

## External dependencies and permissions

- Shared pretrained-model/download caches outside the retired roots remain in
  their existing locations. They are referenced, not duplicated or deleted.
- Existing remote LSR artifact stores remain external custody. Consolidation neither
  downloads every shard nor deletes cloud storage. Current model/template revisions
  and result manifests remain the recovery authority for each experiment.
- Temporary compiled-kernel caches are local tool dependencies, not scientific
  results. Keep unique source/resume evidence; recreate reproducible compiled caches
  only when a future adopted consumer needs them.
- Dataset, model, literature and third-party software rights remain separate. The
  lab's dataset filename is not a provenance or language certificate. In particular,
  the source's `input_french.txt` currently identifies English translated text;
  cross-corpus observations do not thereby become cross-language evidence.
- No research environment has been made portable simply by copying reference code.
  Check relative imports, source layout, versions, data, state, hardware and scorer
  before using an inherited runner. Original absolute paths are historical.

The archive preserves local research content and Git history. It is verified local
custody, **not an established off-device disaster backup**. Read the private restore
receipt for what metadata and recovery types were actually tested.
