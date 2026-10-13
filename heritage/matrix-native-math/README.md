# matrix-native-math

Rebuild arithmetic with the matrix as the atomic unit, treating scalar arithmetic as the degenerate 1x1 special case. The goal is to discover representations and native operations that make useful matrix computation fundamentally cheaper. This is a computer-science and mathematics research project centered on open-ended exploration: no manuscripts, no submission polish, and no paper-writing.

## Start here

Read in this exact order:

1. [AGENTS.md](AGENTS.md): working constitution, mission, success tiers, and process discipline.
2. [research/CURRENT_FOCUS.md](research/CURRENT_FOCUS.md): live front door (standing questions and resource constraints at the top; active state, ranked frontier, and verified result numbers).
3. [research/PROGRAM.md](research/PROGRAM.md): research map, spine questions, and prior art foundations.
4. [research/CURRENT_EVIDENCE.md](research/CURRENT_EVIDENCE.md): accepted evidence, rigor labels, and claim gates.
5. [RESEARCH_MONOGRAPH.md](RESEARCH_MONOGRAPH.md): technical record (read its Errata section first).
6. [research/external_review_2026_09/](research/external_review_2026_09/): outside audit, reviews, and constructive session record.
7. [WORKLOG.md](WORKLOG.md): operational chronology and session logs.
8. [experiments/ledger.jsonl](experiments/ledger.jsonl): append-only structured ledger of pre-registrations, runs, and hashes.
9. [archive/2026-10-01-retired/INDEX.md](archive/2026-10-01-retired/INDEX.md): index of retired threads moved on 2026-10-01 (CPU races, the rank-48 construction hunt, and the commutant/fragmentation physics line; note `code/tuned_race_v2.py` stayed in `code/`).

## Active units

All verified speedup figures and performance metrics live in [research/CURRENT_FOCUS.md](research/CURRENT_FOCUS.md).

- `ozaki_fair_engine_v1` ([research/OZAKI_FAIR_ENGINE_V1.md](research/OZAKI_FAIR_ENGINE_V1.md), [code/ozaki_engine.py](code/ozaki_engine.py), [results/ozaki_fair_engine_v1/](results/ozaki_fair_engine_v1/)): own INT8 residue (CRT) engine emulating FP64 GEMM vs vendor emulation.
- `digit_gemm_complexity_v1` ([research/DIGIT_GEMM_COMPLEXITY_V1.md](research/DIGIT_GEMM_COMPLEXITY_V1.md), [code/digit_gemm.py](code/digit_gemm.py), [results/digit_gemm_complexity_v1/](results/digit_gemm_complexity_v1/)): exact counts of low-precision products.
- [code/contract_plane_table.py](code/contract_plane_table.py) (+ [results/contract_plane_table_v1/](results/contract_plane_table_v1/)): contract-to-plane calculator.
- `floor_ldr_v1` ([research/FLOOR_LDR_V1.md](research/FLOOR_LDR_V1.md), [code/floor_ldr_checks.py](code/floor_ldr_checks.py), [results/floor_ldr_v1/](results/floor_ldr_v1/)): controlled map of structured-matrix classes (no new theorem claimed).
- `framework_floor_v1` and `repr_automaton_v1`, `rect_floor_score_v1` ([research/FRAMEWORK_FLOOR_V1.md](research/FRAMEWORK_FLOOR_V1.md), [research/REPR_AUTOMATON_V1.md](research/REPR_AUTOMATON_V1.md), [research/RECT_FLOOR_SCORE_V1.md](research/RECT_FLOOR_SCORE_V1.md), [results/framework_floor_v1/](results/framework_floor_v1/), [results/repr_automaton_v1/](results/repr_automaton_v1/), [results/rect_floor_score_v1/](results/rect_floor_score_v1/)): bilinear algorithm floor limits, representation automaton analysis, and rectangular scheme score catalogue.
- `beyond_bilinear_v1` ([research/BEYOND_BILINEAR_V1.md](research/BEYOND_BILINEAR_V1.md), [results/beyond_bilinear_v1/](results/beyond_bilinear_v1/)): closed partial.
- `h2 probe` ([code/h2_shared_basis_probe.py](code/h2_shared_basis_probe.py), [results/h2_shared_basis_probe_v1/](results/h2_shared_basis_probe_v1/)): operator-native shared-basis rank probe (inconclusive).
- [research/WORKLOAD_SURVEY_V1.md](research/WORKLOAD_SURVEY_V1.md): unverified survey of candidate scientific FP64 GEMM chain workloads.

## How to run the checks

Only verified executable entry points are listed:

```bash
# Contract-to-plane calculator self-tests
python code/contract_plane_table.py --selftest

# Shared-basis H2 operator rank probe self-test
python code/h2_shared_basis_probe.py --selftest

# Public repo cleanliness and leak audit self-test
python code/public_audit.py --selftest

# Fair Ozaki engine benchmark race report
python code/ozaki_engine.py report

# Full race replay for the fair Ozaki engine (requires an NVIDIA GPU)
bash results/ozaki_fair_engine_v1/replay.sh
```

## Standing rules

- Every claim carries a rigor label (PROVED / EMPIRICAL / CONJECTURE / HEURISTIC / SPECULATION).
- Pre-registration is hashed before any run (`python code/prereg.py`).
- Results are logged in `experiments/ledger.jsonl`.
- Commit format: 'one line, blank line, Committed by Devansh'.
