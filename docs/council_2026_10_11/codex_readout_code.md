**Fix the gate/reporting issues before smoke; binding and dimension claims need additional controls.** Static review only; no files edited. References below use `experiments/cloud/lsr_jobs/` filenames.

- **Extraction — no indexing fix needed.** `readout.py:37–51` correctly uses right-padding lengths and moves both indices onto each hidden tensor’s device, including sharded GPUs. `common.py:39–62` sets eval mode; config `readout_8b_smoke.json:27` pins the revision. **Before smoke:** reconcile layer conventions: `readout.py:130` clamps depth to `L−1`, whereas registration `theory/LATENT_BRANCH_CAPACITY.md:397` specifies depth. Verify whether the final returned hidden state is normalized rather than the registered residual.

- **Positive-control alignment — strengthen before smoke.** `native.py:316–329` asserts exact token prefixes and returns the value’s **last** token; `readout.py:156–158` correctly matches pooled slot states to flattened labels. It never asserts single-token colours in context. Closing-prefix checks cover only eight test blocks (`readout.py:144–147`). Assert intended value-token identity and complete alignment.

- **Permutation bookkeeping — no fix needed.** `readout.py:22–31,183–191` preserves names/multisets, excludes unchanged collections, aligns original indices correctly, and guarantees nonzero changed-slot denominators.

- **Histogram baseline — valid within scope.** IID colours (`native.py:31–32`) make mode Bayes-optimal for ordinary per-slot accuracy given only the histogram (`readout.py:109–112`). Changed-slot scoring conditions on the permutation; its posterior differs from histogram frequencies. **Before claims:** justify optimality under that conditional, collection-weighted estimand rather than transferring the ordinary argument.

- **Binding shortcuts — before binding claims, not mechanics smoke.** Slot-specific heads (`readout.py:94–96`) can recover ordered colours without representing name-addressable binding. Name-only population prediction is chance, but token lengths/positions remain uncontrolled (`native.py:19–37`). Add name/slot reorderings and length-matched controls; otherwise license only slot-colour decodability.

- **Splits — before full run.** Names and seed namespaces are separated; layer/decay selection uses dev (`readout.py:105,138–139,165`). However, k=1 has only 80 possible training examples, so train/dev overlap. Smoke/full also reuse test prefixes because seeds omit run identity.

- **Training — no initialization bug.** Train-only standardization, dev-best checkpointing, seeded random projection and zero heads are coherent (`readout.py:63–89`). Zero heads delay projection learning one update, not forever. CPU reader execution still requires explicit model device configuration (`19,125`).

- **Gates/dimension — fix before smoke.** `readout.py:203–208` emits `r_req` even when instrument/E gates fail, never evaluates E/E1, and merely logs smoke failure after completing ranks. Registration `413–420` forbids that dimension verdict. Best-test-seed selection is registered but optimistic; `425`’s “3k dimensions” is no necessity theorem.

- **Uncertainty/memory — no smoke blocker established.** Collection-paired bootstrap is correct (`115–121`), conditional on fitted readers. At full k=16,d=4096, value tensors alone occupy **1.375 GiB CPU**, plus concatenation/conversion copies; closing features about **832 MiB** (`149–159`). GPU retains model, all-layer activations and logits (`43`): measure peak memory before full launch.