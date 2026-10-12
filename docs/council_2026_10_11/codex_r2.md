# Codex advisor — round 2: amended joint pre-registration

## 1. Signature and decision

**I do not sign `droid_r2.md` unchanged. I sign the consolidated specification below, with six amendments.** Droid has not yet signed these amendments; this is my proposed final common text, not a joint endorsement.

Continue: causal, selectable branch maintenance is the highest-leverage flagship; B and C run broadly in parallel; Bet 4 stays deferred.

1. Specify the graph’s 16 edges and answer balancing exactly.
2. Require branch-addressed edits and restoration. Whole-state swaps between inputs differing in one graph cannot alone establish independently editable branches.
3. Refine the capacity boundary: a powers-of-two grid cannot substantiate a narrow ratio prediction.
4. Define participation ratio, admissibility, and uncertainty before declaring a scaling refutation.
5. Freeze training and selection rules; distinguish maintained computational state from stored bits and simultaneous search.
6. Add a funded larger-model panel.

I accept the operator’s compromise: **frozen A is an instrument check; trained Qwen3-0.6B A is the flagship admission gate.** A frozen null cannot veto trainable branching. All numbers below are predictions or ceilings, not results.

*So what:* One hidden state might keep several possible solutions usable, update one without damaging the others, and answer a question chosen afterward.

## 2. Exact flagship task and implementation

**Generator.** Each branch is a labelled DAG with source `s`, four two-vertex layers `a,b,c,d`, and sink `t`. Its 16 possible edges are:

| Indices | Edges |
|---|---|
| 0–1 | `s→a` |
| 2–5 | Complete bipartite `a→b` |
| 6–9 | Complete bipartite `b→c` |
| 10–13 | Complete bipartite `c→d` |
| 14–15 | `d→t` |

Use lexicographic vertex order within each group. Sample independent Bernoulli(1/2) edge bits. A collection contains k graphs labelled 1…k.

Sample two distinct edge indices before encoding. Each subsequent instruction toggles its edge in every graph. **After both updates**, sample query branch J uniformly, independently of graph contents and updates. Score exact source-to-sink reachability.

For each fixed update pair, rejection-sample each graph to balance final reachable/unreachable outcomes. Oracle answers never enter prompts or updates. Also report an unconditioned Bernoulli panel. Reject duplicate ordered graph collections across splits.

Assign the 120 unordered update pairs deterministically to **72 training, 24 development, 24 test pairs**, using seed **20261011**; randomize order within pairs. Lock composition IDs, labels/renderings, and an update-sensitive panel balanced between answer changes and no changes. Oracle-based panel selection is declared evaluation stratification, never search feedback.

**Persistent bottleneck.** Freeze Qwen3-1.7B backbone weights; train rank-8 LoRA on attention/MLP projections and shared linear maps:

- E: hidden dimension → 1024.
- D: 1024 → hidden dimension.

No branch-indexed slots, auxiliary bit labels, frontier labels, or externally solved candidate tables.

For each training seed, choose a fixed random orthogonal basis Q and nested masks of **64, 256, 1024** coordinates. Every compression computes

`z = normalize(mask_r(Q E(h_last)))`

to fixed norm 1; inactive coordinates remain exactly zero. Expand through `D(Qᵀz)`. Parameter count and dense execution remain identical across widths.

Prefill graph text, compress the final residual, and discard the source tokens, accessible activations, and **entire KV cache**. Every update starts a fresh forward context containing only the expanded state token and update instruction, with `past_key_values=None`, `use_cache=False`, and fresh positions. Compress again and discard that context.

Answer from a fresh context containing only the final expanded state and J. A shared binary head produces YES/NO. The primary capacity assay generates no reasoning tokens. Text controls use the same final binary head.

Verify tensor dependencies and context lengths. Changing deleted prompt/cache buffers must leave continuation identical. No undeclared buffer or graph text may survive.

**Training freeze.** Training seeds **11, 23, 47**; AdamW, learning rate **2e-4**, effective batch **64**; uniform width exposure and log-uniform k over **1/2/4/8/16/32/64**. Train with randomly selected 0/1/2 updates and a late selector, using final-answer loss only.

Include equal numbers of global and branch-addressed toggles. A branch-addressed instruction names an independently sampled branch, never the future query branch. Use at most **100,000 training collections and three passes**. Select checkpoints on development accuracy only.

One repair is permitted: double exposure within those caps and halve learning rate. Record both attempts. Baselines receive equal data and tuning allowances. The 0.6B gate has a **20-L4-GPU-hour** cap. Insufficient throughput yields a deadline-limited verdict, not a scientific negative.

**Causal tests.** At k≥4, apply a branch-addressed toggle to the shared state and query unchanged branches separately. On oracle-certified answer-flip witnesses require:

- Targeted answer-change effect **≥20 percentage points**.
- Collateral accuracy loss **≤5 points**.
- Reversal recovering **≥80%** of induced loss.

Compare equal-token/FLOP sham edits. Apply these operations **before two further global updates**: undo-then-advance must restore future answers; edit-then-advance must track the edited graph.

Follow the proximal staircase: training items at zero delay, short delay, held-out labels, unseen wording, then held-out compositions. Whole-state swaps, self-swaps, shuffled states, and zero states remain custody controls, not sufficient evidence of selective editing.

Compare identity-plus-shared-displacement updates, sham recurrence, learned pause tokens, matched text scratchpads, and independent per-graph states charged at their actual k-fold cost.

Include a strong **storage/delayed-computation control** that stores graph bits and update indices but performs reachability only at final readout, matched for state budget, parameters, training, and FLOPs. Query after 0/1/2 updates diagnostically.

Passing selectivity establishes operational maintenance. Beating the storage control strengthens a computation claim. Neither alone proves simultaneous parallel search, random-feature superposition, or new mathematics absent from Rⁿ. Graph-only success licenses a graph-task result.

## 3. Capacity and square-root prediction

**Capacity C(r)** is the largest tested k attaining **≥90% balanced exact-answer accuracy for every selector position**, with no failure at a smaller tested k. Exhaust selectors on each collection; report joint correctness separately. The flagship additionally requires the k=4 causal thresholds.

Run the coarse grid **1/2/4/8/16/32/64** on development. Include every integer between its last passing and first failing k in the locked test grid. Freeze that grid before test scoring.

Evaluate **1,024 independent collections per tested k per training seed**. Cluster all selectors and intervention variants within collections. Bootstrap collections, preserving paired widths, for capacity and ratio intervals. Boundaries outside the frozen refinement bracket are interval-censored. Failed k=1 competence, ceilings at 64, and unresolved thresholds yield no precise capacity ratio.

Measure PR on **branch-variation state differences** from paired collections differing in one graph, at **k=4 after two global updates**, removing the mean difference:

\[
\mathrm{PR}(\Sigma)=\frac{(\operatorname{tr}\Sigma)^2}{\operatorname{tr}(\Sigma^2)}.
\]

Freeze centering and normalization on development. Report fourth moments, common components, and shared-index dependencies. Aggregate-state covariance PR is a separate diagnostic. Bootstrap PR ratios by collection. Connecting these learned differences to the theorem’s random-feature ensemble remains an empirical hypothesis.

**Frozen numerical prediction:**

| Comparison | Admissible measured PR ratio | Nominal capacity ratio | Prediction band |
|---|---:|---:|---:|
| 64→256 | 3.2–4.8 | **2×** | **1.5–2.7×** |
| 256→1024 | 3.2–4.8 | **2×** | **1.5–2.7×** |
| 64→1024 | 12.8–19.2 | **4×** | **2.25–7.29×** |

The proposed exponent is explicitly **1/2**: \(C\propto\sqrt{\mathrm{PR}}\). No constants are fitted to held-out outcomes.

Support requires admissible PR, uncensored boundaries, both adjacent point estimates in-band, and consistent direction across three training seeds.

**Refute the operational bridge** when an admissible, uncensored comparison’s multiplicity-adjusted 95% ratio interval lies wholly outside its band. Use a max-statistic bootstrap across the three comparisons. Overlapping, unresolved intervals are inconclusive. Bad PR, floors, ceilings, competence failures, or custody failures fail to test the bridge.

Give **mechanism, utility, and scaling separate verdicts**. Scaling failure does not refute the specified quadratic random-feature interference theorem. No earlier G4 exponent claim enters this preregistration.

## 4. Breadth, tasks, and compute accounting

**Frozen A:** Qwen3 **0.6/1.7B**, Qwen2.5 **0.5/1.5B**, SmolLM2 **360M/1.7B**, OLMo 2 **1B**. First test direct last-residual feedback with source KV removed, without a trained random projection. Its negative can reflect interface failure. The trained gate then tests the declared bottleneck.

**Core sweep for both B and C:**

- Qwen3 **0.6/1.7/4/8B**.
- Qwen2.5 **0.5/1.5/3/7B**.
- SmolLM2 **360M/1.7B**.
- OLMo 2 **1/7B**.
- DeepSeek-R1-distill-Qwen **1.5/7B**.
- Mistral **7B**; Phi-3-mini **3.8B**.
- Optional with existing access: Llama-3.2 **1/3B**, Llama-3.1 **8B**.

OLMo 2 1B exists; pin `allenai/OLMo-2-0425-1B` or its instruct variant consistently. [Official release](https://allenai.org/olmo2).

**Larger panel:** Qwen3 **14/32B**, Qwen2.5 **14/32/72B**, DeepSeek-R1-distill-Qwen **14/32B**, OLMo 2 **13/32B**. Use A100-80GB/B200 as memory requires; retain BF16 for confirmation.

Pin repository/revision, base versus instruct variant, and template. No Gemma, Gemini, or OpenAI models.

Use task/decoding seeds **101, 202, 303**, with **128 development problems per task**. Promote at most four checkpoints per probe using development gain, including a second family. Confirm on **512 new problems per task**, paired across methods and seeds. Report every screened cell, exclusion, and failed instrument. Freeze task-macro-average comparisons; apply Holm adjustment across promoted checkpoint/budget hypotheses.

**B tasks and method:** generated executable constraint puzzles and Countdown. Every arm receives identical public constraints and candidate-validity feedback; scoring answers remain inaccessible. Select a norm-constrained evolutionary search over the final prompt-position residual, rerunning downstream computation after each perturbation. Compare verifier-guided text repair, best-of-N, self-consistency, and random perturbations. Freeze two budgets equivalent to **8 and 32 ordinary complete attempts**, expressed in measured FLOPs.

**C tasks and method:** those puzzles plus generated graph reachability and fresh GSM8K-style arithmetic. Development selects one four-layer block near **¼, ½, or ¾ depth**, **1/2/4/8 total applications**, and damping in **{0.25,0.5,1}**. No learned weights. Use separate repeated-block KV slots. Compare extra-token CoT, pause tokens, sham loops, repeated ordinary passes, and soft-token thinking where runnable.

**Matched-FLOP contract:** Count prefill, state construction, projections, repeated layers, KV attention, decoding, rejected proposals, search arithmetic, verifier work, and backward passes. Report model FLOPs, verifier work/calls, wall time, and cost separately. Match model FLOPs within **5%**, verifier calls exactly, and disclose auxiliary work. Include failed attempts. Give text controls source access where the useful task permits it; report strict state-only compression separately. Match training exposure and report training cost separately. Run thinking-mode, format, and budget remedies first; Qwen3 exposes `enable_thinking=False`. [Official model card](https://huggingface.co/Qwen/Qwen3-1.7B). Report termination beside accuracy; baseline termination below **95%** blocks capability interpretation. Measure fixed-input numerical variation on each hardware/precision configuration, cluster by problem, and manually inspect successes and failures.

## 5. Predictions, stopping rules, and resources

| Bet | Claim, first experiment, prediction | Stopping rule | Hardware ceiling; USD |
|---|---|---|---|
| **A** | Frozen: 300 collections/checkpoint at k=1/2/4, three seeds; ≥2 selectable branches and ≥5-point lift over shuffled state. Trained 0.6B: ≥90% single-branch accuracy; ≥2 editable branches meeting **20/5 points/80%** thresholds. | Frozen null ends frozen branching only. Trained proximal failure after one repair blocks flagship escalation. | **48+20 L4 GPU-hours; $34** |
| **B** | ≥5-point gain over strongest matched text arm at both budgets, positive adjusted paired intervals on two checkpoints spanning two families. Freeze on development, confirm on 512/task. | Fixed sweep misses prediction, text absorbs gain, or feedback leaks answers. Uncertain intervals remain inconclusive. | **160 L4 GPU-hours; $79**, plus shared large panel |
| **C** | ≥3-point gain over strongest matched arm, positive adjusted paired interval, second-family replication. Freeze block/damping/loops before confirmation. | Matched controls absorb gain, fixed sweep misses prediction, or instability survives declared damping choices. Ends frozen looping, not trained recurrence. | **128 L4 GPU-hours; $63**, plus shared large panel |
| **Flagship** | After trained A: k≥4 causal maintenance through two further updates; ≥8-point gain over strongest equal-FLOP control, positive paired interval, consistent direction across three seeds; scaling predictions above. | Custody failure invalidates result; storage-only behavior blocks computation claim; absent benefit blocks utility claim. Scaling verdict remains separate. | **72 wall-hours on 8×B200 = 576 GPU-hours; $2,880** |

Optional Qwen3-4B flagship replication shares the flagship cap.

The shared larger-model panel gets **120 A100-80GB GPU-hours with a $360 all-in cap**, plus **8 additional 8×B200 node-hours, $320**. Allocate at most half each envelope to screening. Run independent small jobs on separate GPUs; avoid idle eight-GPU nodes.

**Allocated ceilings: $3,736. Reserve: $1,264. Hard maximum: $5,000.**

Use the brief’s operator-verified planning figures of approximately **$0.49/L4-hour** and **$40/8×B200-node-hour**. A100 accelerator prices exclude host costs; enforce its dollar cap including those charges. Charge storage, networking, verifier CPU work, and preemption against reserve.

Hours are ceilings, not throughput promises. Enforce spending through an active job controller; billing alerts alone are insufficient. Stop launching jobs before accrued plus committed costs could exceed $5,000.

## 6. Deadline and final interpretation

- **October 11–12:** freeze manifests; complete frozen A and trained admission; run broad B/C development screens.
- **October 13–14:** flagship training and causal controls; B/C confirmation in parallel. If A misses October 12, close flagship escalation for this window.
- **October 15:** finish locked evaluation and replication; no test-set tuning.
- **October 16:** independent adversarial review, separate mechanism/utility/scaling verdicts, evidence export, and compute shutdown.

The material disagreement is that whole-state swaps and post-toggle reachability can still be storage with delayed computation. **Branch-addressed edit/undo/advance tests and the storage/delayed-computation control settle that distinction operationally.**

Even a complete pass does not automatically establish Nobel/Fields-level novelty or native mathematics. The useful vocabulary is **retained future, selective edit, restoration, and update-closed answerability**; PR remains scaffolding.

Saved to [codex_r2.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/docs/council_2026_10_11/codex_r2.md). No experiments, cloud actions, spend, or tracked-file edits occurred.

`bb_list` succeeded. Blackboard state access and synthesis were rejected because they require approval while the approval policy is `never`; no board entry or synthesis is claimed.

