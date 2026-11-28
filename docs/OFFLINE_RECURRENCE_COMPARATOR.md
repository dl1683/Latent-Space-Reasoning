# Offline recurrence: a learned comparator for persistent-world maintenance

28 November 2026. Classification: mission-directed literature and experimental-design review, **not an LSR experiment or implementation result**. Reviewed at `3f1ab425`. The [registered sequence](LATENT_WORLD_SYSTEM_DESIGN.md#7-research-sequence-and-decision-gates), frozen reader gate and separate held operation panel remain unchanged. No new model run is proposed before qualification and authorization.

## A close external result and its important boundary

Lee et al., [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference, v3, 5 June 2026](https://arxiv.org/html/2605.26099v3), propose learned recurrent updates to SSM fast weights before evicting attention context, then use single-pass prediction. This is a stronger conceptual competitor than a cache or ordinary recurrent read: extra computation is spent organizing memory before prediction.

The paper contains two materially different evidence regimes:

- Section 6.2 (Depo): the shuffled directed cycle is split across four context windows. Both start node and hop count vary, and queries follow the evicted graph. Four consolidation passes improve held-out loss on deeper traversals relative to one pass. The graph law and query family are specified during training.
- Section 6.3 (GSM-Infinite): the **question precedes the context**. The fine-tuned Jet-Nemotron 2B and Ouro 1.4B can therefore specialize consolidation to that question. For eight-operation problems the reported Jet accuracy is 0.351 versus 0.388 (one versus six passes); Ouro is 0.210 versus 0.272 (one versus four). This is not query-blind natural-language maintenance. These experiments require roughly 1–2 H100 GPU-days per run (Section 6). Neither backbone meets LSR's current 0.6B experiment preference.

The paper supports feasibility of trained offline recurrence under its task distributions; it does not establish versioned semantic revision, evidence custody, arbitrary later queries, edge economics, or useful learned LSR operations. Do not cite the GSM results as evidence that a worker can anticipate unknown evidence needs.

Two additional reference controls separate the benefits:

- Lin et al., [Sleep-time Compute](https://arxiv.org/abs/2504.13171), April 2025: offline textual reasoning before a query. The authors report roughly fivefold less query-time compute at similar accuracy on their stateful math variants and a 2.5-fold amortized cost reduction on their multi-query benchmark. Query predictability correlates with benefit. A query-blind text worker that writes reusable notes into a versioned store is therefore a serious control for the value of offline planning itself.
- Eyuboglu et al., [Cartridges](https://arxiv.org/abs/2506.06266), already cited in [Architecture](ARCHITECTURE.md): self-study trains corpus-specific compact caches. Baek et al., [RestoreKV](https://arxiv.org/abs/2608.01247), August 2026, instead learn a shared restoration mechanism to construct compact, context-specific supplementary KV entries. The latter reports improvements over matched eviction budgets, not persistent operation discovery. These are controls for reusable representations and retained information rather than the full world mission.

These external methods are reference comparisons; no checkpoint, training recipe or result has been locally reproduced or qualified.

## Discriminating the eventual learned maintenance claim

The existing [same-pin intervention](WHOLE_WORLD_EVALUATION.md) can establish that a retained contribution is causally used. It does not by itself establish that spending the compute *before* an unknown query was better than spending it after the query or storing its output in text. Distinguish three claims: learned consolidation improves retention, query-blind consolidation amortizes useful computation, and revision-aware latent operations outperform conventional incremental computation.

After the gates qualify, for a separately registered development study (not additional arms retrofitted to the frozen panel):

1. Publish an initial world and revisions while hiding tasks. Give the same source bytes, allowed access and base reader to a single-pass latent retention arm, a multi-pass latent maintenance arm and a query-blind textual sleep worker. The textual worker may maintain links, derived notes, contradictions and unresolved questions, not merely summaries. Preserve versioned dependencies and compare at matched budgets.
2. Give a query-time extra-compute arm the same initial representation and access rights; it spends its extra budget after the late task arrives. Compare equal total resource budgets over the declared workload as well as matched answer latency. A question-known consolidation arm, if used, is explicitly an upper-bound diagnostic, not a maintenance result.
3. Sample independent tasks after publication from both familiar and novel operation families, with a changed relation or rule, a necessary withdrawal, a restoration, an unaffected finding, and a missing premise. Do not let the task sampler or gold obligations drive maintenance. Include a matched exact warm incremental control and the same reader with memory omitted as already required by the design.
4. Remove and restore the proposed computation at the same source pin. Keep base representation, reasoning configuration and authorized fallback rights fixed. Measure no-reopen accuracy separately from recovered answers. The extra matched-compute controls isolate maintenance value; the intervention isolates the contribution actually used.
5. Charge every offline pass, discarded proposals, text tokens, state bytes, verification, invalidation, fallback and recovery. Report resource dimensions separately on actual hardware; a low wake-time latency alone is not total efficiency.

For one publication interval with M paid maintenance cost above a baseline, and constant positive matched-quality per-query saving d, the incremental total saving after n queries is n*d-M. Strict break-even requires n>M/d; if d is nonpositive there is no break-even under these assumptions. Revisions may terminate the interval before that threshold, so include publications with zero queries and report the observed distribution of reuse, not only high-reuse worlds. This is ordinary amortization accounting, not a new theorem. With different query costs use the actual sum of savings, and count all maintenance paid between revisions.

**Decision:** preserve the reader-first path. For later mechanism selection, a latent advantage over worker-off alone is insufficient; compare against equally resourced query-blind textual reasoning and task-time reasoning. If a latent worker only wins when the question is provided in advance, classify the result as query-conditioned inference, not autonomous world maintenance. If it wins only on predictable operation families, restrict the claim to that family. If the exact warm control matches quality at lower cost, stop the latent-specific economic claim. This tests whether the same retained world can discover useful computation under changing evidence before it knows what will be asked, rather than only whether context can be compacted.
