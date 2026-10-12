**Verdict: F is INVALID under its registered positive control.** No seed passes both staged orders; forward-staged seed 2 alone passes. Forward gaps are +2.25–2.94 pp; reversed gaps +2.13–2.75 pp, below 15 pp. Re-running aggregation reproduced the summary. ([registration](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/LATENT_BRANCH_CAPACITY.md:278))

- **Moment separation is confounded with effective step size.** Buffers update every fourth step, but bias correction uses global \(t\). For identical buffers, neglecting epsilon, the update multiplier relative to correction by buffer-update count \(k\) is  
  \[
  \frac{1-.9^k}{1-.9^t}\sqrt{\frac{1-.999^t}{1-.999^k}}.
  \]
  At \(t=200,k=50\), this is 1.918; at \(1800,450\), 1.518. It is not uniformly larger initially. Thus the observed advantage cannot isolate moment separation. Also, the custom “AdamW” implements no weight decay. ([formation.py:133–143](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/formation.py:133))

- **Reset is a plausible collapse mechanism, not an established cause.** Consolidation zeros moments and restarts bias correction, while global warmup leaves lr at full strength. Seed 3 falls from 0.990 to 0.744, but already dipped during main training at step 1400; checkpoints cannot localize collapse to early consolidation steps. ([reset](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/formation.py:153), [raw checkpoints](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/f8b/shards/011/out/results.jsonl:7))

- **Denominators and artifacts:** 200 entities × two wordings = 400 scores/family; 800 entities and 1,600 scores overall. Balanced gold makes the constant-answer baseline 12.5%. Early checkpoints exhibit constant choices; final rows do not. Training excludes the literal test templates but includes their facts and overlapping answer-clause language: paraphrase transfer only. Batch contents are deterministically matched across arms. Entity-level predictions are absent, preventing registered clustered paired uncertainty. ([evaluation](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/formation.py:64), [batches](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/formation.py:105))

- **Aggregator bug:** missing staged orders are skipped by `all()`, permitting false passes on incomplete runs; all orders are present here. ([aggregate.py:235](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/aggregate.py:235))

**Licensed sentences:** “Sequential training shows substantial forgetting; interleaving retains more before consolidation.” “Interleaved pre-consolidation **means** are 0.879–0.901; individual families span 0.790–0.940.” “The moments recipe reaches means 0.990–0.996 before consolidation, with unresolved optimizer confounds.”

**Never say:** “Concurrent formation penalty demonstrated”; “moment separation caused the gain”; “reset caused collapse”; “unseen-fact generalization”; “native latent mathematics discovered.”

Continue the program; highest leverage here is matched-effective-step controls and reset/warmup ablations. This remains conventional optimizer scaffolding, not native vocabulary.

Blackboard state/synthesis calls required approval unavailable under this session’s policy.