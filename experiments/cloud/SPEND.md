# LSR GCP spend ledger

Estimate snapshot 2026-10-14T11:58:30+00:00; authorization synchronized 2026-10-15. Estimated from VM lifetimes x all-in Spot rate (Cloud Billing catalog, 2026-10-11). The former 5,000 USD cap is historical: owner direction 2026-10-15 removes per-program spend caps. Compute credit only; "Scale Y1 - AI" expires 2026-10-17.

| run | VMs launched | est. USD |
|---|---:|---:|
| diag-k1 | 4 | 1.04 |
| f8b | 12 | 7.83 |
| gate-eval-conc | 3 | 12.33 |
| gate-eval-cur | 12 | 65.21 |
| gate-pilot | 1 | 0.53 |
| gate-q06 | 3 | 106.88 |
| gate-q06-cur | 3 | 66.15 |
| gemma4-controls-2026-10-15 | 2 | 1.96 |
| gemma4-controls-v2-2026-10-15 | 1 | 0.58 |
| p1-s11 | 10 | 77.59 |
| p1b-pilot | 2 | 0.53 |
| p1b-s11 | 10 | 47.76 |
| p2 | 29 | 96.04 |
| p2-pilot | 2 | 0.81 |
| p3 | 3 | 0.65 |
| q1p-8b-smoke | 2 | 0.48 |
| q1pp-8b-smoke | 2 | 0.63 |
| q1pp-8b-sweep | 2 | 2.73 |
| q8b | 2 | 0.76 |
| q8b-diag | 1 | 0.14 |
| q8b-validate | 1 | 0.16 |
| q8b2 | 2 | 1.88 |
| readout-14b-smoke | 2 | 0.66 |
| readout-32b-smoke | 2 | 2.39 |
| readout-72b-smoke | 2 | 8.84 |
| readout-8b-full | 6 | 2.21 |
| readout-8b-smoke | 2 | 0.49 |
| recur-confirm | 4 | 1.12 |
| recur-pilot | 5 | 2.51 |
| recur-sw-b | 6 | 2.21 |
| recur-sw-s | 28 | 87.96 |
| search-confirm | 69 | 117.06 |
| search-confirm-r32 | 32 | 112.23 |
| search-dev-b | 16 | 44.03 |
| search-dev-s | 64 | 39.94 |
| search-pilot | 4 | 0.30 |
| search-temps | 6 | 3.62 |
| smoke1 | 4 | 0.12 |
| **total** | 361 | **918.39** |

Live check 2026-10-14T16:46:12+00:00 under the sc-bench config: live LSR VMs 0; burn rate $0.0/h.

Gemma 4 31B instrument controls (2026-10-14 UTC clock; run names carry 2026-10-15): 3 VM launches, a2-highgpu-2g Spot at 4.42 USD/h, europe-west4-a. Run 1 (gemma4-controls-2026-10-15): first VM crashed on a config-attribute bug, relaunched after the fix, done 10:55:54Z; estimated 0.44 VM-h = 1.96 USD (conservative: both launches charged to the done marker). Run 2 (v2, five controls): 11:09:58Z to 11:17:51Z, 0.13 VM-h = 0.58 USD. The recorded control attempts were reaped; snapshot live VMs 0. Rows added by hand: `lsr_cloud.py spend` could not be re-run cleanly (the dispatcher has uncommitted edits by another agent), so regenerate before relying on this figure.

Last ledger entries: gemma4-controls-2026-10-15 and gemma4-controls-v2-2026-10-15 (1.96 + 0.58 USD, above); no launches since. Pending, not launched: E1 alignment/hook smoke and S0 memory-port smoke, each gated on a ready queue row and a fresh Codex LAUNCHABLE review. A100-80GB quota is 0; candidate shapes are a2-highgpu-2g (2 x A100-40GB, 4.42 USD/h planning rate) or a3-highgpu-2g (2 x H100, rate to verify). Figures are hand-maintained; regenerate with `lsr_cloud.py spend` once the dispatcher edits settle.
