**DESIGN-CONTRACT: REPAIR.**  
**ENDPOINT-RELAXATION: REPAIR as written; the reduction is valid once promised-operation termination is explicit.**

These are design judgments, with no new theorem promotion. The frozen accounting and strict-write repairs survive this attack. The remaining essential ambiguity is liveness.

The [lifetime clause](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md:41) guarantees correctness of “each completed query,” but does not explicitly require initialization or promised commands to complete. “QUERY emits” and “FLIP toggles” suggest the intended requirement; the admission quantifier should state it directly. Otherwise a partial-correctness reading permits divergence where the reduction needs an endpoint or decoded answer.

The sufficient repair is:

> Initialization terminates for every initial \(x\). At every boundary reachable by a valid prefix containing at most \(H\) flips, every QUERY terminates correctly. At every such boundary reached with fewer than \(H\) flips, every FLIP terminates correctly. Each promised invocation respects wear and private-space limits and finishes in the canonical volatile boundary state. No termination requirement applies to irrelevant states or unpromised invocations.

This adds no mid-command crash guarantee and imposes no unearned efficient work bound.

With that clause, the coverage argument is sound as a **design reduction**:

1. **Static decoding exists.** Let \(B\) be the persistent bit states reached at completed boundaries through depth \(H\). For \(s\in B\), define \(d(s)_i\) by running QUERY(\(i\)) from canonical entry. It terminates, changes no persistent cells, and observes no wear. If two histories reach the same \(s\), their queries execute identically, so correctness forces the same logical vector. Different hidden wear vectors cannot create different decoding.

2. **Initialization exists without a normalization assumption.** Define \(I(x)\) as the completed initialization endpoint. Then \(d(I(x))=x\). From unworn zeros, its actual initialization wear \(w^0\) satisfies
   \[
   I(x)\le w^0
   \]
   coordinatewise. Neither \(I(0)=0\) nor a preferred encoding follows.

3. **Endpoints are stationary on their required domain.** Define \(T_i(s)\) wherever \(s\) occurs at a boundary before depth \(H\). Canonical entry, fixed immutable code and current-command-only input determine the execution from \(s,i\). Every guaranteed execution avoids INVALID writes, so hidden wear supplies no branch or feedback. Consequently equal \(s,i\) give the same endpoint across histories and depths. Intermediate writes may be numerous and may leave the boundary encoding fibres.

4. **Endpoint wear undercounts actual wear.** For a completed command with endpoint \(s'\), let \(c\) count its actual changes coordinatewise. Then
   \[
   s\oplus s'\le c.
   \]
   Associate each relaxed history with the same actual flip history. Starting from \(v_0=I(x)\), induction gives
   \[
   v_{t+1}=v_t+(s_t\oplus T_i(s_t))
   \le w_t+c=w_{t+1}\le L\mathbf1.
   \]
   Thus every admitted program supplies a feasible endpoint policy. Read-only queries can be inserted arbitrarily without altering this correspondence.

5. **Total extensions add no obligations.** Extend \(d\) arbitrarily outside \(B\), and extend \(T_i\) arbitrarily outside states occurring before depth \(H\). Induction from every \(I(x)\) shows that exploration through depth \(H\) follows actual program endpoints. These extensions therefore introduce no additional required histories. A state appearing only at depth \(H\) needs decoding, not another flip; a state also appearing earlier must use its already fixed transition.

This argument requires arbitrary nonlinear \(d,I,T_i\), multi-bit endpoint changes, and every reachable relaxed wear record. It uses neither a fixed logical parity decoder, one-bit transitions, one history per state, nor depth-dependent controller policies. Physical wear parity is an accounting invariant, not a decoder restriction.

The [future formula and certificate gate](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md:67) remains separate. A certificate establishes UNSAT of its formula; independent encoding review must establish coverage of the entire relaxation. Complete UNSAT would exclude admitted machines for that tuple and horizon. SAT still requires finite compilation within the charged machine, followed by instruction-level replay.

For compilation, distinguish offline synthesis resources from runtime resources. Report synthesis work/space, generator work/private space, installed code bits, installation work, code-access implementation, and actual PC/controller widths. A parameter-only compiled table is not input advice, but its lookup is not free ROM access. Offline solver memory does not demonstrate compliance with runtime \(S=256q^2\), and a finite witness supplies no scalable polylogarithmic uniform algorithm.

I read the required authority files, full dual-loop skill, actual pre-test, repaired candidate, design questions, and both available October 4 round-1 steering reports. Only the round-2 **packet**, not a returned round-2 report, was present. Reviewed candidate SHA256: `0da12066832969935ff809ab674ff7cec65c5a4af5cda93ff159be55ad4a29fe`.

No edits, agents, GPU work, implementation, solver runs or synthesis occurred. The lifetime ceiling remains **CONJECTURE**; the journal construction remains unadmitted pending its own paid layout, rotation, threshold, boundary, wear and source design. This review selects no strategy and establishes no priority or completed CAN/CANNOT pair. The useful finite CS and mathematics objective and keep-going invariants remain active.