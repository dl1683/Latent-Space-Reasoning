**CERT-CODE-REPAIR.** Actual runtime evidence identifies native proof buffering; marking this installed Glucose3 backend unsupported is unwarranted. **FULL remains gated.**

- **Original failure reproduced:** N1H2’s original 48-variable, 136-clause CNF has no insertion-time root conflict. MiniSat22 returns `propagate([]) == (True, [])`. The original synthetic `0` proof is rejected as `Non-RUP addition at index 0`.
- **Native buffering demonstrated:** with the installed Glucose3 binding, an anonymous pipe contained **0 readable bytes before native flushing and 80 afterward**. Python-side flushing did not expose them. The recovered trace contains five additions and four deletions, ending in the empty clause. Every addition passes the actual checker, with **zero MiniSat22 search calls**.
- **Correct CRT verified:** the loaded extension’s imported `fflush` address equals `ucrtbase.dll.fflush`. Upstream source corroborates the mechanism: MSVC disables `setlinebuf`, tracing uses a separate `fdopen` stream, and the Glucose patch comments out its closing operation. This upstream source is corroboration, not exact installed-build provenance. [Native binding source](https://raw.githubusercontent.com/pysathq/pysat/master/solvers/pysolvers.cc), [Glucose patch](https://raw.githubusercontent.com/pysathq/pysat/master/solvers/patches/glucose30.patch).
- **Shutdown failure isolated:** closing the descriptor while native bytes remain buffered produces normal output followed by raw exit `0xC0000409`. Flushing first produces exit **0**. This establishes a failure mechanism; the earlier launcher’s exact exit-1 causal path remains unproven.
- **Checker ordering is correct:** [check_endurance_drup.py](/<LOCAL_PROJECTS>/new-computation-model/experiments/check_endurance_drup.py:66) validates before insertion and separately tracks insertion-time root conflicts. Empty, invalid, truncated and mismatched-binding controls were rejected; genuine root contradictions were accepted.

**Concurrent changes appeared during review; I made no writes.** The updated source now flushes before proof extraction. I independently regenerated and matched all six saved tiny formulas, replayed all four SAT witnesses, and rechecked the newly saved certificates:

| Saved case | Checked additions | Ignored deletions |
|---|---:|---:|
| N1k1H2 | 5 | 4 |
| N2k2H2 | 46 | 24 |

These checks concern the new native traces. Nothing is admitted from the original bad `0` proof.

**Remaining code repair:** [endurance_tiny_cert.py](/<LOCAL_PROJECTS>/new-computation-model/experiments/endurance_tiny_cert.py:53) flushes only after successful `solve()`. Put flushing in a `finally` block **inside** the Glucose3 context, before its proof descriptor closes, covering loading/solving exceptions. The runtime evidence supports flushing the matched UCRT; the additional `msvcrt` flush was unnecessary for these controls. Preserve independent checking of any synthesized root-conflict certificate.

The concurrently produced `smoke.json` reports 256 SMT comparisons, 24 Boolean-gate controls and N4 generation counts. I did not rerun those searches or independently close the complete schema/threshold/onehot/resource gate. No smoke root-exit receipt was present. **H6/H5 searches remain unexecuted by this review; no FULL code-gate pass.**

Current SHA-256 snapshot, HEAD `623a7aecd419356766640adc40f3adae3584a6b4`:

```text
experiments/endurance_tiny_cnf.py
312e4927833150aeb1b43655f10bc23eb07f8b1fb8f06dac8c78cce887377ff5
experiments/check_endurance_drup.py
a3cc9768297ad04ea6aed081dab4a515ac2975a596db2fdece48e1cdd069a8a0
experiments/endurance_tiny_cert.py
5702e31975dd3f126328750830eceaf3ee2b160d23e0a70f1d5754f7e1400b17
experiments/validate_endurance_tiny.py
4f4ab3f2e4771d0bd848eaa0b870f48c2db8dfe9497587148848ea5ad894e7bb
experiments/run_endurance_tiny_cert.py [concurrently added]
868e7155a70c7525522b4ff75a5b423ac51f01a7b3d814d1aede26069e8723c1
research/ENDURANCE_TINY_CERT_PROTOCOL.md
4e622f4e89ae706cc6cd84acd1d3962d86fdf4f7212a172f470dc2c715c604aa
research/ENDURANCE_TINY_CERT_DESIGN_REVIEW.md
bf346819a24b82d338bd4240ea4f53843206d2b30eb8438af2cc0be9346f66cd
research/ENDURANCE_TINY_CERT_SMOKE_FAILURE.md
c3cea62d9f35fa8d88d05f0034136692abf11fd532dfe3e3767f709125ea33e4
installed pysat/solvers.py
253654d8efabae650a0d136ad2f2e6d30b57206b1fb70846c714197468a28f7e
installed pysolvers.cp313-win_amd64.pyd
1019bacdbb9400cc54fa89aa39294fefe1c63d5a67fdab35f473364529ec72dd
results/endurance_tiny_cert_smoke/n1k1h2/formula.cnf
601deab336be53ed2f46965314b4c5ebd85e0f13d0ba305c2576bde74cf475e7
results/endurance_tiny_cert_smoke/n1k1h2/proof.drup [new native trace]
4a0ddb904804b0be5bc88bc3623d3a861178d6f3a62023063618643b9a4c2d28
results/endurance_tiny_cert_smoke/smoke.json
0ddb2ad508cb2b8893ce7eab67806d2f85f2c2c354b344c50c6b1dfe9a0acf4b
```

All shells used `C:/Windows/System32/cmd.exe`, `login=false`. No source writes, FULL N4 solve, other-backend search, GPU or agents.