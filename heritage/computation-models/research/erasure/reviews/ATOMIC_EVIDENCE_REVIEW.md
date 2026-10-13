**PASS — bounded FULL evidence admission.** Read-only, CPU-only probes independently reproduce the outputs without calling either runner’s `main()`.

| Gate | Concrete validation |
|---|---|
| Clean/dirty AND | Exact ranges: **78 clean** `(N=1..12, K=1..N)` and **12 dirty** cases. Simulated 98,304 input runs and reproduced all 881 layer traces. Clean final states contain only AND; dirty states preserve every input bit. |
| Atomic minimax | **16 cases**, `q=M=1..16`. Independently evaluated 914 integer partitions and 3,380 transitions using 120-digit Decimal arithmetic. Maximum discrepancy: `1.3e-79`. Below-threshold reachability fails; threshold-plus-`1e-60` and above-threshold depths equal `q−1`. |
| Compiler | **308 cases**: 276 Boolean tables, 30 family cases, two bucket maps. Independent simulation checked 2,782 input runs, 2,026 selective merges/reset traces, scratch cleanup, final outputs, integer block masses, and reversible steps. |
| Counted operations | Independently derived **47,563 gates**: 14,230 NOT, 8,373 CNOT, 22,934 Toffoli, 2,026 RESET. Reconstructed schedules exactly match the reviewed sources. Maximum floating trace discrepancy: `1.78e-15 < 1e-12`. |

The equal-entropy separation also survives exact integer comparisons: both maps lose **1.5 bits** and have output entropy **2.5 bits**, while
`φ(4)/16 ≈ 0.2028195311 < 1/4 < φ(8)/16 ≈ 0.2717822216`.

Reviewed hashes, the frozen pre-run commit, the [pre-run record](<LOCAL_PROJECTS>/new-computation-model/results/atomic_full_run_environment.json), post-computation artifact hashes, and current source hashes agree. Outputs are complete; current smoke rows match their FULL counterparts. No audited registration, criterion, or scope changed; registry additions concern a separate experiment.

Exact-byte SHA256s:

```text
experiments/entropy_controls.py
c22765f9a084260d81dc274128939c57757545d23612f372f08ec10055044dad
experiments/atomic_compiler.py
40b24c94078b21a49ff6e4b7b73ff424ada096e06ea66bbb1ddcb86ba5327e4e

results/atomic_full_run_environment.json
b08e8c0cc40311b09536e20ac6fd4e06c5fe47fc096f9439238e0572996e8bfb
results/entropy_controls_full.json
dbb048089ec407e279954fdef003100c8918337900773b9168934063399c62f2
results/atomic_compiler_full.json
dc47469207b16cf24853e7697d54e567cf6b24f691fd8364ba3b514143017ee3
results/entropy_controls_smoke.json
f395bf1a66ad9aac552fb56f1e7eab6a2138abc149ac7f18f2c78eca0c420353
results/atomic_compiler_smoke.json
4d4ceb95add1e25d8ca1a4dc1e1cb9b6102ad08ed3915c002c35d50195255695
```

Registered outcomes remain **EMPIRICAL**. Exact general feasibility rests on the [independently reviewed proof](<LOCAL_PROJECTS>/new-computation-model/research/ATOMIC_COMPILER_REVIEW_2.md:23), not finite agreement or numerical equality. Historical provenance is hash-bracketed; no continuous execution trace is recorded. Construction bit work and physical behavior remain unmeasured. No files were modified.