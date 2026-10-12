# Correctness hunt before data: Registration N (rename x reorder) (STRICTLY READ-ONLY; do not edit any file)

Read in detail (file:line): experiments/cloud/lsr_jobs/bindq.py (all), imported helpers (native.FAM/COLORS/NAMES/
value_positions, q1pp.check_options, readout.DEVICE/boot_ci/hist_mode), common.load, configs/bindq_8b_smoke.json,
and "Registration N" at the end of theory/LATENT_BRANCH_CAPACITY.md.
Hunt bugs that would make the smoke wrong or the claims unsupported: name-pool generation and length matching
(is "2 tokens in-family" checked correctly for both W1 "The {n} box" and W2 "- {n}'s crate"?), the q(n) extraction
position, row/collection ordering (query_rows vs states vs value-token reshape vs view(-1,k)), quad construction
(bindings preserved, reorder non-identity, renamed names distinct), hist-mode baseline per quad, leakage (held-out
names in standardization/selection, seeds), reader training/selection, the superposition diagnostic, the colour-free
null (is "here" a fair colour-free substitute; could the null leak via token positions?), adequacy rule, gates and
N1/N2 logic vs the registration, memory at k=16 n_train 8192 on a 40GB A100, and runtime. Could the entity reader
solve the task WITHOUT name addressing (e.g., using h-only histogram info plus q-only priors)? Say for each issue
whether it must be fixed before the smoke. Under 450 words, file:line.
