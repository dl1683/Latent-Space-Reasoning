Strategy council for Latent-Space-Reasoning (LSR), 2026-10-12. Read-only: do not edit any file other than your output. You are a goal-aligned adversary. Assume the project's mission is true and attack the METHOD and STRATEGY. Never close a direction or declare the program dead; if a line is stalling, say what would work instead.

READ FIRST, in detail, citing file:line: README.md, STATE.md, CLAUDE.md, AGENTS.md, NOTEBOOK.md (tail), experiments/EXPERIMENTS.md, experiments/ledger.jsonl (last ~40 rows), theory/CROSS_PROJECT_INSIGHTS.md, docs/council_2026_10_11/, and the newest hunts in docs/ (codex_nameabl_hunt.md, codex_slotctl_hunt.md, codex_bindq_*). Also skim git log -30.

User direction (2026-10-11/12):
- They want something far more radical and earth-shattering than incremental probes. Risk is fine; explore fearlessly.
- Fold three lines into LSR, each kept in its own repository: high-dimensional / matrix-native math, the LLM Genome project (dissections, in the Moonshots repo), and latent-space reasoning.
- Run at scale on GCP with real budget: about $9K of credit remains, shared; the LSR cap is $5K, with ~$916 spent.

Recent findings, to be verified against the files:
- Qwen3-8B's closing state supports slot-indexed colour readout (slot-query readers 0.99 at 4 facts, 0.62-0.73 at 16). Name-query readers were inconclusive in the N smoke; the N-control does not establish absence of name binding. [Corrected 2026-10-12 after council launch: the earlier 'not keyed by name' wording overstated the result.]
- The 20-vs-300 training-name ablation: held-out readout is higher with 300 names at both fixed layers (drops 0.06-0.10, CIs above zero); only W2 at layer 9 reaches the 0.10 k=4 bar, so the prediction is not replicated. [Corrected 2026-10-12 after council launch.]
- Admission gate FAIL on curriculum and concurrent arms; storage 2-4; updates 0.84-0.86.
- NOTE: by user policy, Chinese-origin models are now excluded from the Stateful product. Say whether that should also apply to LSR research subjects (Qwen3-8B is the current subject) or whether research use is fine. Propose non-Chinese subjects (Gemma 4, gpt-oss, Muse Glimmer 30B, Nemotron, Mistral) if a switch helps.

Questions:
1. Distance from claim: what is LSR's central bet, and does the last 48h of work test it directly, or is it measurement-of-measurement? Give the ratio honestly.
2. What are the 3-5 most promising radical directions that could produce a striking, publishable result within the budget? Each must combine the genome dissections, matrix-native math and latent reasoning. For each: the hypothesis, the smallest decisive experiment, the compute, and what success or failure would show.
3. What is going wrong strategically right now, if anything, and what should change?
4. Concrete next 2 weeks: the ordered experiments, GCP allocation, and the stop and pivot rules.
At most 10 numbered points. Be concrete and bold.
