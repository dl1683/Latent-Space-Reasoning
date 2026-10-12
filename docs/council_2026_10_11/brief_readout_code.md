# Correctness hunt before data: Registration R readout probe (READ-ONLY; do not edit any file)

Read in detail (file:line): experiments/cloud/lsr_jobs/readout.py (all), helpers from native.py it imports
(block, collection, value_positions, FAM, NAMES), common.load, experiments/cloud/configs/readout_8b_smoke.json, and
"Registration R" at the end of theory/LATENT_BRANCH_CAPACITY.md.
Hunt for bugs that would make the smoke wrong or the claims unsupported: closing-token extraction under right
padding and multi-GPU, value-token positive-control alignment, label/permutation bookkeeping in the binding control,
histogram-mode baseline correctness (is mode Bayes-optimal here?), leakage between train/dev/test (names, seeds,
layer selection, weight-decay selection), the reader training (standardization, early stopping on dev, learned
bottleneck init, seeds), bootstrap CIs, r_req definition, gate logic, memory at k=16 for n_train 8192 on CPU/GPU,
and mismatches between code and registration. Also: is "binding information" correctly separated from position-only
or name-only shortcuts (e.g., could a reader exploit token-length or slot-position artifacts)? Say for each issue
whether it must be fixed before the smoke. Under 450 words, file:line.
