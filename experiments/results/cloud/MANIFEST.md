# GCP results manifest

Raw per-shard outputs (results.jsonl, model_info.json, shard.json, startup logs, checkpoints) live in
`gs://sc-bench-iqidis-artifacts/lsr/runs/<run>/shards/<i>/out/` and are pulled locally with
`python experiments/cloud/lsr_cloud.py pull <run>` (git-ignored). Committed here: the canonical aggregate
summaries produced by `python experiments/cloud/aggregate.py {search|recur|branch} <runs...>`.

| Summary | Runs |
|---|---|
| search_dev_summary.txt | search-dev-s, search-dev-b, search-temps (probe B dev screen) |
| search_confirm_summary.txt | search-confirm, search-confirm-r32 (probe B confirmation) |
| recur_summary.txt | recur-sw-s, recur-sw-b (probe C screen) |
| recur_confirm_summary.txt | recur-confirm (probe C confirmation) |
| gate_eval_cur_summary.txt | gate-eval-cur (locked admission gate, curriculum arm) |
| gate_eval_conc_summary.txt | gate-eval-conc (locked admission gate, concurrent arm) |
