# Re-entry council, round 1: latent-reasoning at scale on Google Cloud

You are one of two independent advisors (the other is a different AI engine). There is no arbiter: you two must converge over at most 3 rounds. Be a goal-aligned adversary: attack and sharpen, never close the mission.

## Read first (fresh, from disk)
- AGENTS.md (process rules; standing orders, evidence gates, think-before-you-run), CLAUDE.md, README.md, top of STATE.md.
- legacy/README.md (the prior latent-perturbation program and its CORRECTION: the headline capability claim was withdrawn because thinking-mode truncation at a token cap, not reasoning, drove the benchmark; learn from this).
- theory/HPE_AGGREGATE_INTERFERENCE.md (the repo's own HD-math results: exact quadratic aggregate interference for k superposed random unit features crosses order one at k ~ sqrt(d); participation-ratio generalisation k ~ sqrt(PR(Sigma)); fourth-moment sharpness law). These are the theorems Bet 2 would test in a real reasoner.

## The situation (new)
The project was CPU/laptop-bound (sustained GPU crashed the machine). It now has Google Cloud: project with Spot GPUs (H100/H200/B200 8-GPU nodes, A100, L4; Spot quota up to 64 accelerators per region; 8xB200 Spot is about 40 USD/h, a single L4 Spot well under 1 USD/h). Budget: about 5,000 USD total. HARD DEADLINE: the credit expires 2026-10-17, so we have roughly 5.5 days of wall clock from 2026-10-11. Everything must produce a decisive result inside that window.

User mandate, verbatim: restart latent space reasoning on Google Cloud and run it at scale; we can now be really aggressive with testing. Offered incremental options, the user said none of these excite them; they want something more latent and more earth shattering. Risky is fine; explore fearlessly.

Hard constraints: no Gemini, no OpenAI models of any kind (including OpenAI open-weight models). Allowed: open-weight models such as Qwen3 / Qwen2.5 family (0.5B to 72B), Llama 3.x, Mistral/Mixtral, OLMo 2, SmolLM2, DeepSeek-R1-distill, Phi. Avoid Gemma to be safe.

## The four candidate bets
1. Latent-state search: a frozen model's hidden state, optimised directly (gradient or evolutionary search over the residual stream, scored by a verifier), solves problems the same model fails at in text, even with chain-of-thought, at matched compute.
2. Superposed hypotheses (flagship candidate): high-dimensional geometry (near-orthogonal packing, concentration of measure; the repo's own HD-math theorems) lets latent reasoning carry many candidate solutions in one state and search them in parallel. Train COCONUT-style latent reasoners at several widths, measure the branch capacity inside the hidden state, and test whether it scales with d as the theory predicts (sqrt(d) or sqrt(PR)).
3. Recurrent depth on frozen pretrained models: looping middle layers at inference, no retraining, improves hard reasoning — a zero-token think-longer knob; then a light fine-tune to stabilise the loops.
4. Model-independent reasoning geometry: a partial reasoning state from model A, mapped into model B's residual stream, lets B finish the problem.

## Your job this round
1. Attack each bet: prior art (COCONUT, Huginn/recurrent-depth, soft thinking, latent BFS evidence, cross-model stitching, test-time latent optimisation papers you know of), the cheapest confound that would fake a positive (matched compute, truncation, verifier leakage, answer-in-the-prompt, best-of-N with a verifier being the real driver), and whether a positive would actually be consequential.
2. Add better bets if they exist — more latent, more consequential, still decisive in 5 days.
3. Pick a portfolio: 2 or 3 FAST PROBES (hours each, can run wide across many models/sizes/seeds in parallel) plus ONE FLAGSHIP (the most consequential bet that can still be decisive by 2026-10-16).
4. For each chosen bet give: a falsifiable central claim in one sentence; pre-registered predictions with numbers; kill criteria (what result ends that bet — not the mission); the cheapest decisive first experiment; models; GPU type and hours; estimated USD.
5. Name the baselines that make a positive meaningful (for Bet 1: best-of-N sampling and self-consistency at equal verifier calls and equal FLOPs; for Bet 3: same-FLOP extra-token CoT; etc.).
6. Give a one-sentence so-what that a non-expert finds exciting for the flagship.

Write your answer to the output file given to you (markdown, concise, numbered). Do not run experiments or spend money. Do not edit tracked repo files.
