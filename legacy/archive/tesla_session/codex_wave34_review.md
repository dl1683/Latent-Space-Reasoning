**18-32 Review**

18. **DO-BoN**: CDE component. New: measured diversity gating. Borrowed: best-of-N. Flaw: fingerprint diversity is not correctness diversity.

19. **ETR**: CDE router/component. New: cheap early basin pruning with KV resume. Borrowed: clustering/speculative expansion. Flaw: early tokens may misclassify late-diverging reasoning.

20. **VFR**: Domain component, not general architecture. New: perturb-regenerate only failed spans. Borrowed: process verification. Flaw: parsing and wrong-approach errors; only works where steps are verifiable.

21. **DRCG**: CDE generation component. New: online duplicate abort. Borrowed: DO-BoN + ETR. Flaw: threshold tuning; can reject useful similar starts.

22. **Gated-Attention Probe**: Not architecture. It is a falsification test. Critical, but does not belong in the catalog as a design.

23. **DBTR**: Independent speculative architecture. New: using frozen AR LLM as denoiser. Borrowed: text diffusion/COCONUT. Critical flaw: decoder causal mask and OOD embeddings make it likely nonfunctional.

24. **FVI**: Selector component. New: formal verifier inside CDE selection loop. Borrowed: symbolic verification. Flaw: formalization/parsing gap; can make arithmetic benchmark trivial.

25. **IPF**: Search strategy inside CDE/GGIO. New: property-objective prefix search. Borrowed: inverse prompting/GGIO. Critical flaw: circular when property equals answer.

26. **CSLA**: Speculative independent probe. New: attention-pattern feedback token. Borrowed: reflection/recurrent transformer ideas. Flaw: untrained recurrence plus attention extraction overhead; likely worse than more samples.

27. **CPH**: Neuro-symbolic subarchitecture. New: perturb-diverse constraint extraction. Borrowed: SMT/OR-Tools neuro-symbolic pipeline. Flaw: extraction/formalization is still the hard part.

28. **MMNE**: CDE operator, not separate architecture. New only in combining model diversity with CDE allocation. Borrowed: standard ensembles. Flaw: weak second model may waste budget.

29. **CRR**: Postprocessing component. New: multi-compression as cheap answer voting. Borrowed: summarization/refinement. Flaw: compression usually loses signal; low priority.

30. **PDR**: CDE prompt operator. New: persona × perturbation grid. Borrowed: role prompting. Flaw: correlated personas, format drift.

31. **CPME**: Dataset-level memory component. New: verified memory injection. Borrowed: few-shot/in-context learning. Flaw: contamination and order dependence.

32. **AMR**: CDE allocation component. New: tiny learned operator router. Borrowed: routing/mixture-of-experts. Flaw: 25 tasks is not enough data.

**Ranking**

No change: **CDE > ALM > DDC > Neuro-Symbolic > CIR**.

VFR/FVI can dominate arithmetic, but that is domain tooling, not a general reasoning architecture. MMNE is the only serious candidate to challenge the list, but it is naturally absorbed by CDE as another operator.

**Catalog Status**

Stop designing. The catalog is complete enough and already over-fragmented. The missing “paradigm” is not another architecture; it is **measurement discipline**: fixed budget, operator decorrelation, selector accuracy, marginal utility per token, and ablations.

Claude should now measure. Any new design without CDE-compatible metrics is architectural noise.tokens used
20,754
**18-32 Review**

18. **DO-BoN**: CDE component. New: measured diversity gating. Borrowed: best-of-N. Flaw: fingerprint diversity is not correctness diversity.

19. **ETR**: CDE router/component. New: cheap early basin pruning with KV resume. Borrowed: clustering/speculative expansion. Flaw: early tokens may misclassify late-diverging reasoning.

20. **VFR**: Domain component, not general architecture. New: perturb-regenerate only failed spans. Borrowed: process verification. Flaw: parsing and wrong-approach errors; only works where steps are verifiable.

21. **DRCG**: CDE generation component. New: online duplicate abort. Borrowed: DO-BoN + ETR. Flaw: threshold tuning; can reject useful similar starts.

22. **Gated-Attention Probe**: Not architecture. It is a falsification test. Critical, but does not belong in the catalog as a design.

23. **DBTR**: Independent speculative architecture. New: using frozen AR LLM as denoiser. Borrowed: text diffusion/COCONUT. Critical flaw: decoder causal mask and OOD embeddings make it likely nonfunctional.

24. **FVI**: Selector component. New: formal verifier inside CDE selection loop. Borrowed: symbolic verification. Flaw: formalization/parsing gap; can make arithmetic benchmark trivial.

25. **IPF**: Search strategy inside CDE/GGIO. New: property-objective prefix search. Borrowed: inverse prompting/GGIO. Critical flaw: circular when property equals answer.

26. **CSLA**: Speculative independent probe. New: attention-pattern feedback token. Borrowed: reflection/recurrent transformer ideas. Flaw: untrained recurrence plus attention extraction overhead; likely worse than more samples.

27. **CPH**: Neuro-symbolic subarchitecture. New: perturb-diverse constraint extraction. Borrowed: SMT/OR-Tools neuro-symbolic pipeline. Flaw: extraction/formalization is still the hard part.

28. **MMNE**: CDE operator, not separate architecture. New only in combining model diversity with CDE allocation. Borrowed: standard ensembles. Flaw: weak second model may waste budget.

29. **CRR**: Postprocessing component. New: multi-compression as cheap answer voting. Borrowed: summarization/refinement. Flaw: compression usually loses signal; low priority.

30. **PDR**: CDE prompt operator. New: persona × perturbation grid. Borrowed: role prompting. Flaw: correlated personas, format drift.

31. **CPME**: Dataset-level memory component. New: verified memory injection. Borrowed: few-shot/in-context learning. Flaw: contamination and order dependence.

32. **AMR**: CDE allocation component. New: tiny learned operator router. Borrowed: routing/mixture-of-experts. Flaw: 25 tasks is not enough data.

**Ranking**

No change: **CDE > ALM > DDC > Neuro-Symbolic > CIR**.

VFR/FVI can dominate arithmetic, but that is domain tooling, not a general reasoning architecture. MMNE is the only serious candidate to challenge the list, but it is naturally absorbed by CDE as another operator.

**Catalog Status**

Stop designing. The catalog is complete enough and already over-fragmented. The missing “paradigm” is not another architecture; it is **measurement discipline**: fixed budget, operator decorrelation, selector accuracy, marginal utility per token, and ablations.

Claude should now measure. Any new design without CDE-compatible metrics is architectural noise.
