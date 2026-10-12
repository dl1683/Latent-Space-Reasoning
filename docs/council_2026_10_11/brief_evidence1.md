# Evidence gate 1 (Tier-3 auditor role): verdicts recorded 2026-10-11/12

You are a fresh, unprimed auditor hunting overclaims IN BOTH DIRECTIONS (claims too strong, and negatives stated too broadly). Read: theory/LATENT_BRANCH_CAPACITY.md (pre-registration, amendments, Deviation 1, P1/P2, Amendment 3), experiments/EXPERIMENTS.md (top four sections), the tail of experiments/ledger.jsonl (last ~15 rows), and these summaries: experiments/results/cloud/gate_eval_cur_summary.txt, experiments/results/cloud/search_confirm_summary.txt, experiments/results/cloud/search_dev_summary.txt, experiments/results/cloud/recur_confirm_summary.txt, experiments/results/cloud/recur_summary.txt. Code: experiments/cloud/lsr_jobs/ and experiments/cloud/aggregate.py.

For each recorded verdict (gate FAIL; probe B FAIL; probe C FAIL; the gate geometry OBSERVATION):
1. Do the metrics justify the stated verdict under the frozen rules? Are comparisons fair and reproducible, any leakage/confound (termination, empty answers, constant answers, batch shape, selection on dev, test reuse)?
2. Is any sentence an overclaim or an over-broad negative? Give the exact replacement sentence (the licensed sentence) and a never-say list.
3. Is anything logged insufficiently (ledger fields, provenance)?
4. Should this program continue, and is the current pivot (P1' storage capacity vs width; P2 update closure) the highest-leverage thing in the remaining ~4 days? If not, what is?
Under 800 words, markdown. No experiments, no edits.
