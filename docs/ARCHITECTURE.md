# Architecture and research program

Canonical destination; targets are not implementation results. Current status and operating boundaries are in [README](../README.md); measurements are in [RESULTS](RESULTS.md).

## The program and architecture

[TARGET] "Reason over everything you have; do not depend on finding it."
Encode the ENTIRE corpus once into the same latent space where reasoning happens.
Reason there instead of writing tokens at every step; reuse and update computation already paid for.
Latent memory and latent reasoning are meant to become ONE system.
Whole-corpus completeness is the commercially central question: being in the dump must mean being noticed.

[TARGET] ONE multimodal latent reasoning model: I/O encoders/decoders at the edges, latent computation at the core.
Memory is an OPTIONAL PARAMETER of the same call, not a separate mode; without memory the model still reasons in latent space.
Role/type and time-horizon signals distinguish attached evidence from input, output and authorized instructions; memory content must never acquire instruction authority.
Text, image, audio and video, context trees and long-context reasoning belong to the destination.
Leave latent space only for exact source text or external tools, encode returns back into memory, preserve source pointers, and keep memory current.
Use multiscale/fractal levels that track computational relevance so complete sweeps can remain cheap.
A frontier agent queries it as a tool and receives natural language with source pointers.
Reuse pretrained encoders, decoders, specialists and base models heavily; embed hierarchical agent memory as latents later.
Efficient construction and mid-pack recognized intelligence performance are TARGETS, not achieved results.

[HYPOTHESIS] Cheap full latent sweeps, useful compression and reliable stopping may make repeated work cheaper and more complete.
These need causal task evidence and complete costs, not vector geometry alone.
"Infinite context" means expandable memory, not constant total storage or a universal exact fixed-size encoding.

## Definition boundaries

> 1. LATENT MEMORY was described correctly: "infinite context in latent space": encode the ENTIRE corpus once into the same latent space where reasoning happens, so the question flips from "did I find everything?" to "everything is in; which parts matter?"; a full pass is cheap; leave latent space only to pull exact source text or call external tools, encoding what comes back into memory; keep provenance pointers to source; keep memory current; completeness is the commercially central question; "not in memory" detection; being in the dump must mean being noticed; multiscale (fractal) levels so sweeps stay cheap. Memory is an OPTIONAL PARAMETER of the same call of ONE multimodal latent reasoning model (I/O encoder/decoder at the edges, latent core), with role/type and time-horizon signals so injected memory is never confused with input/output or treated as instructions; a frontier agent queries it like a tool and gets natural language with source pointers; heavy reuse of pretrained parts; hierarchical agent memory is to be embedded as latents later.

> 2. LATENT-SPACE REASONING: the GOAL is reasoning in latent space instead of writing tokens at every step. The MECHANISM IS UNKNOWN AND OPEN. Judge-scored candidate chains, evolutionary search, prune-before-commit and decode-once are ONE CANDIDATE mechanism to be tested against alternatives, NOT the definition. We do not know what will work. Latent memory and latent reasoning are meant to become one system.

> 3. The memory-prefix / "receiving" prototype (a trained prefix lets a small Gemma stem answer questions about content not in its context, vs a storage-only control) is ONLY a first slice (central bet 1: a prefix can influence and train through the model). It is not the program and a pass would not mean the vision is achieved. The other claims (selectable futures, survives selective revision, beats storage-only control on cost, whole-corpus completeness, "not in memory", provenance, update/invalidation, multiscale, latent reasoning over memory, tool interface) are separate and not yet tested.

## Slice, program, mechanism

[TARGET] **Program:** the whole destination above, including completeness, useful latent computation, validity and full-cost reuse.
**Slice:** a bounded construction or test that advances a named part; a slice PASS cannot become program completion.
**Mechanism:** a candidate way to construct the object, replaceable when evidence favors another.
The memory-prefix/receiving prototype is **SLICE 1 of the program, not the program**: can a prefix influence and train through the selected model?
Its bounded trained readout and restart demo do not establish the remaining claims.
Judge-scored candidate chains, evolutionary search, prune-before-commit and decode-once form ONE candidate family, not the definition.

[HYPOTHESIS] Keep at least two reasoning mechanisms alive: trained recurrent latent-state transitions through reused blocks; hidden-state-fed continuous steps with learned stopping.
A now has executable reused-block workspace mechanics and B has executable fixed-depth continuous-feedback mechanics, both checked on random CPU Gemma. Actual pretrained A operation passed its bounded no-fit mechanical gates; its first acquisition and single lower-LR remedy failed TRAIN and DEV with constant answers. Useful learned consequence use remains unestablished. Neither has admitted pretrained consequence evidence or an A/B comparison; the historical draft is design only. Judge-guided population search remains another candidate. Compare useful held-out consequences at matched complete compute against textual reasoning and no-step controls.
A no-op, wrong-state or shuffled-transition control must break the claimed consequence; a decoder solving independently cannot earn the latent core's claim.
Memory routes (prefix, native KV, cross-attention, fast weights) are a separate choice from reasoning mechanisms.
The two construction tracks and shared matrix/neural/computational mathematics retain their full ambition; no separate launch queues or retrieval-only product replaces it.

## Latent space and maintenance swarm

The corpus becomes a persistent state that numerical operators can write, combine and revise. A shared blackboard records source versions, candidate changes, operator versions and decisions. A maintenance worker runs against a declared snapshot and proposes a change with its dependencies. A commit checks those dependencies again so that concurrent revisions cannot silently overwrite or revive superseded work. The query model reads an admitted snapshot and emits an answer with source witnesses. Memory remains optional in the same model call.

The architecture research swarm and the maintenance swarm have different timescales. The first proposes executable weight edits, new operators and changes to their wiring, using inspected pretrained blocks where they preserve a demonstrated operation. The second refines the corpus state using an admitted operator version while remaining in continuous coordinates between I/O steps. A candidate operator must retain earlier capabilities after installation and under subsequent maintenance. A short supervised calibration can fit a narrow adapter or update rule after that operation and the receiver are defined. The run budget should follow the information gained by the smallest adequate control.

### Recurrent numerical revision candidate

[SLICE] The NumPy [latent field](../experiments/memory/latent_field.py) implements an initial operation for revising coupled numerical state. Each cell holds a drive $b_i$ and a width-$d$ state $z_i$. A directed weight block $W_{ij}$ maps a message from cell $j$ into cell $i$. The shared synchronous transition computes $z_i' = \tanh(b_i + \sum_j W_{ij}z_j)$ coordinatewise. The caller supplies the drives and weights. Relation provenance and source text stay outside the transition.

For fixed weights, the maximum aggregate absolute row sum $q$ bounds the infinity-norm Lipschitz constant of the transition. For $q < 1$, the transition has a unique fixed point. Its real-arithmetic error bound is $r/(1-q)$ for a global bound $r$ on the residual. After changing drives, the code warm-starts and recalculates only the forward closure of the changed cells. Cells outside the closure receive the same inputs and keep their old values. The new global residual bound is the larger of the recomputed closure residual and the inherited bound. A small local residual alone can hide earlier error, even with two cells. Float64 roundoff is outside the mathematical bound.

A field uses synchronous sweeps and immutable full arrays for each accepted snapshot. A reconfiguration can append cells, replace observation drives, and write or remove weight blocks in one batch. Replacing an ordered pair replaces all of its parallel blocks. The new field seeds revision with appended and edited cells and with the targets of changed or deleted links, even when a deleted link is absent from the new topology. It then follows the new forward closure. If a higher contraction bound makes the inherited residual too large for the requested tolerance, it refreshes the entire field. The old field and snapshot remain available even if the batch fails.

Local revisions reduce the set of recomputed messages. Whole-state copying, retained snapshot bytes, topology rebuilds and setup still count. Growing the coordinate basis, learning links, resolving conflicts, natural-language readout and swarm scheduling remain open. Strict contraction can also attenuate distant influence. If a change first reaches a cell after $h$ edges, the worst-case norm bound contains the factor $q^h/(1-q)$. Long-distance and competing-path consequences must therefore be tested before adopting the geometry.

### Relative observation revision

[SLICE] The [relative observation field](../experiments/memory/relative_observations.py) compiles weighted scalar differences and absolute anchors into the recurrent field's optional identity activation. A relative source row measures $x_t-x_s=d_e$ with weight $w_e$, and an anchor measures $x_i=y_a$ with weight $w_a$. Both the affine recurrence and an independent direct incidence solve target

$$\min_x \lambda\|x\|_2^2 + \sum_e w_e(x_t-x_s-d_e)^2 + \sum_a w_a(x_i-y_a)^2.$$

Here $\lambda>0$ is a declared zero-centered ridge. A cell's diagonal is the ridge plus its incident and anchor weights. The drive divides its signed measurement sum by that diagonal, with $-w_e d_e$ on the source and $+w_e d_e$ on the target. A neighboring cell sends its state with coefficient $w_e$ divided by the receiver's diagonal. Parallel sources contribute their own terms. The row sum is below one in exact arithmetic, giving the existing fixed-point residual bound. The implementation rejects nonfinite affine updates and exhausted iteration budgets. Ridge changes the estimate, and a weak ridge can give slow convergence. The residual bound governs the compiled transition and excludes rounding and measurement uncertainty.

A named source edit recompiles only its old and new endpoints using their retained incident rows. A change in normalized links uses the field's existing topology replacement. Relaxation warm-starts the affected components and preserves earlier snapshots. The bidirectional links can require revisiting the entire connected component. Source-index copies, topology rebuilds, support refreshes and retained snapshots still count. Readout returns an absolute estimate when the queried cell is connected to at least one observation anchor. A detached component returns `None` even if its regularized internal state is finite. This is a structural support policy for this task.

The sparse conventional baseline applies diagonally preconditioned conjugate gradients to the same objective. Each matrix product accumulates weighted differences along undirected edges, plus each cell's ridge and anchor diagonal. Both solvers use the same source index, support and snapshot interface. The sparse baseline starts from the selected snapshot and solves the revised components reached from edited endpoints. Bridge deletion reaches both descendants, and an unsupported component still gets a numerical update for possible reconnection. Values outside the affected components stay fixed.

Each accepted conventional solve recomputes the residual at the returned state, including a warm start that needs zero iterations. In exact arithmetic the row dominance margin is at least $\lambda$, giving $\lVert x-x^\star\rVert_\infty\le\lVert b-Ax\rVert_\infty/\lambda$. The implementation evaluates the edge weights, right-hand side and residual in Float64, scaled by the largest active diagonal. It refuses scaling that erases the ridge, an aggregated edge or anchor weight, or a nonzero compiled right-hand side. The bound excludes roundoff and retains the prior snapshot's bound when some components remain untouched. A failed solve publishes no snapshot. Iterations, matrix products and edge-endpoint accumulations are separate work counts. Full comparisons must also charge compilation, traversal, vector reductions, copies and storage.

The direct reference assembles weighted incidence and identity rows, solves by dense least squares, and obtains support independently from source connectivity. Agreement checks the construction on conflicting paths, fixed-topology corrections, and bridge removal and restoration. The sparse baseline and an incremental factor-graph solver need end-to-end comparison under equal source, history and revision access. Learned acquisition, calibrated absence and comparative cost remain open.

### Multiscale relative observation revision

[SLICE] The [region hierarchy](../experiments/memory/relative_hierarchy.py) applies the same weighted objective with two levels of coordinates. The caller supplies one region per cell. Current cross-region measurements identify the boundary cells, called ports. Each observation belongs to one region or the coarse problem. Each ridge term belongs to the region containing its cell.

A region factors its weighted rows using QR with columns ordered as interior $i$, ports $p$, and right-hand side. The upper block gives $R i+B p=c$ for a fixed boundary. The lower block gives a coarse residual $C p-d$. Retain the latter block and the affine map $i=R^{-1}(c-Bp)$, computed by solving with $R$. Stacking the regional coarse factors and adding the cross-region observations preserves the original minimizer in exact arithmetic. A read of one interior coordinate evaluates only its row of the affine map. Discarded constant residuals would matter for reporting objective values or calibrating disagreement. The elimination uses the same square-root principle applied in bundle adjustment. [Demmel et al.](https://openaccess.thecvf.com/content/CVPR2021/papers/Demmel_Square_Root_Bundle_Adjustment_for_Large-Scale_Reconstruction_CVPR_2021_paper.pdf)

Cross-region edits on existing ports reuse every regional factor. Internal edits rebuild their region. A new cross-region endpoint promotes a cell to a port and rebuilds its region from retained sources. Promotions persist until explicit compaction. Branches share unmodified region factors and retain their own sources, ports and coarse values. Structural support tracks connected components within each region and joins them using active cross-region measurements. A bridge can disappear and withdraw support while its ports are still retained. An unchanged region without ports keeps its read operator and values.

`compact` is a bounded maintenance operation on a selected snapshot. It picks regions with the most ports unused by any current bridge, breaking ties by region index, and limits the number of regions per call. For retired coordinates $q$ and surviving ports $p$, a new QR of the retained regional factor yields $q=h+Tp$. The old readout $i=a+B_q q+B_p p$ becomes $i=(a+B_qh)+(B_p+B_qT)p$. Readouts for $q$ become additional interior rows. This rewrite preserves the source objective up to discarded constants and projects the parent solution onto the surviving coordinates in exact arithmetic. It reuses the source ledger and skips both fine reconstruction from sources and the coarse solve. It can check supplied protected reads for exact support agreement and an absolute numeric tolerance before returning the child. Older branches keep their original factors and ports. If a retired cell becomes a bridge endpoint again, its region rebuilds from sources.

The default coarse backend has one independent numerical block per group of regions connected by active bridges. All ports in a region share a numerical block. Structural support still uses the finer observation connectivity within each region. Each block's stacked square-root system has a thin SVD, $A=U\Sigma V^\top$. Its solve applies $V((U^\top b)/\sigma)$ with scaled operands. A value-only edit can reuse the SVD when ordered ports, regional coefficient columns, and bridge identities, endpoints and weights match. An unchanged block reuses its accepted values. The cache belongs to the selected parent snapshot, and a split or merge uses new blocks. Compaction keeps unchanged blocks, projects changed blocks' values and leaves their new SVDs to the next source edit that changes those blocks.

Numerical rank is checked against each block's own largest singular value and dimensions. Separate block scales can resolve systems a single globally scaled rank cutoff would reject. A rank loss or nonfinite solve can abort a revision with the parent available. Compaction can accumulate roundoff, and selected read checks leave other coordinates uncertified. The regional factor and read operator store $n(n_p+1)$ scalar entries for $n$ cells and $n_p$ ports, so retiring $d$ ports removes $nd$ entries from the new region. A cached $m$-row, $p$-port SVD adds $mp+p^2+p$ scalar entries. Held ancestors keep their factors.

Passing a complete `elimination_order` to `RelativeObservationHierarchy` selects the persistent sparse elimination backend. The order is filtered to the current ports. Regional factors separate by exact local observation components. The source topology identifies structural zero coefficients across those components even when floating-point QR introduces small nonzero entries. Each resulting factor and each active bridge row starts at the first port in its scope. For each port, QR of the incoming rows retains a conditional read operator for that port and sends the remaining square-root factor to a later port. Constant residuals may be dropped for recovering the minimizer. Nonfinite factor construction or an unresolved pivot relative to the local input scale can abort the revision. This pivot check gives a different numerical acceptance rule from the default backend's block rank cutoff.

Sparse snapshots retain an index of inputs to each elimination bucket. A source batch stages replacements for each rebuilt region's complete factor set and each changed bridge before processing the queue. It schedules newly promoted ports, and it visits affected buckets in elimination order. If a reduced factor changes its destination, both the old and new bucket receive an update. An unchanged conditional can be reused when all its input objects match. An exactly equal reduced factor keeps its old identity and ends that update path. The index preserves the full sweep's numerical row order, including the order of incoming reduced factors.

Sparse source revisions publish these conditionals without recovering their values. A reused conditional can acquire a different value when its dependencies change, so reads resolve those dependencies in the selected snapshot.

Structural support also has a snapshot-owned index. Active named bridges join local observation components, and per-region incidence identifies bridges to reproject after a local split, merge or anchor change. Edge counts distinguish parallel bridge sources from the connectivity they provide. A change in edge presence or local anchor status invalidates the touched old groups, then traversal finds their surviving or newly connected parts. Numerical reweights can share the graph. Removing a parallel source can also leave reachability unchanged. Compaction and reordering share the graph because they preserve active bridges and local components. A component with its own anchor needs no bridge to give supported reads.

`support_witness(cell)` returns a cell path, the named relative observations along it, and the named anchor at the final cell. It searches the component graph for an anchored component with the fewest bridge hops, then uses retained local adjacency to connect actual bridge endpoints. Component and cell IDs break path ties, and the smallest current source name breaks parallel-source ties. It preserves the recorded orientation of each row and retrieves its payload from the selected snapshot. An unsupported cell returns `None`. Witness lookup can succeed even if numerical substitution overflows. The path proves structural anchoring. The weighted estimate can depend on other sources, and the path itself provides no measurement validation or attribution of numeric influence. A search can traverse a large connected group, and continued revisions retain the local named adjacency.

`source_gains(cell, source_ids)` returns the requested named measurement gains for a supported read. Hold the ridge, weights and observation endpoints fixed. For row $k$, use $a_k=e_{t_k}-e_{s_k}$ for a relative measurement and $a_k=e_{t_k}$ for an anchor. Let $H=\lambda I+\sum_k w_k a_k a_k^\top$. A single response $u$ to $H u=e_{cell}$ gives $g_k=w_k a_k^\top u$. The exact-arithmetic change in the read from any combination of value edits is $\sum_k g_k\Delta d_k$. A source outside the structural witness can have a nonzero gain, and a source in a disconnected component has zero gain.

The response uses the same regional transfer map as readout. An interior query adds one local triangular solve from the affected region's source design. The coarse response uses its retained SVD or sparse elimination pivots and conditionals, with forward accumulation and dependency-based recovery. A compacted snapshot can rebuild a needed coarse factor privately. This leaves the source ledger and historical read state intact. Numerical range and conditioning still bound the computation. Charge the local factorization, forward dependency closure, requested endpoint recovery and any cold rebuild.

`predict_source_edits(cell, changes)` predicts one read from a batch of value changes, positive reweights and removals of existing sources. A replacement keeps its original oriented endpoints. Additions and endpoint moves use a revision. After validating the whole batch, the prediction ignores sources outside the query's original connected group. Removal preserves support if the current named witness survives. If that witness breaks, a search over the remaining named local and bridge rows checks whether any path to an anchor remains, including paths through parallel rows and backup anchors. An unsupported read returns `None`.

Let $x$ be the original estimate, $q$ the query cell, and $A$ the matrix with one incidence column $a_i$ per relevant edit. Let $D=\operatorname{diag}(w'_i-w_i)$ and $v_i=w'_i(d'_i-d_i)+(w'_i-w_i)(d_i-a_i^\top x)$. Removal uses $w'_i=0$ and omits the first term. In exact arithmetic the new estimate solves $H'(x'-x)=Av$ with $H'=H+ADA^\top$. Set $M=A^\top H^{-1}A$ and $r=A^\top H^{-1}e_q$. Solving $(I+DM)z=v$ gives the predicted change $r^\top z$. For rows with unchanged weight, $z_i=v_i$ directly. Symmetry gives the necessary rows of $M$ from one response per weight change, so the dense correction solve has dimension equal to the number of changed weights.

Prediction reads the original query and the endpoints of relevant weight edits. One query response and the changed-weight responses share local and coarse factor preparation within the call. The call can still traverse the affected support group and coarse dependencies, and it publishes no revision. The implementation refuses a correction when its smallest singular value is at most $\sqrt{\epsilon}\,k\max(1,\sigma_{\max})$, where $\epsilon$ is Float64 machine epsilon and $k$ counts weight edits. The guard includes an absolute scale because forming $I+DM$ can cancel values of order one. It can refuse an accurate prediction and gives no forward-error guarantee for accepted predictions. The measured evidence covers supplied Float64 fixtures. Comparing full prediction cost with a private refactor or ordinary revision remains open.

#### Local source patch

[SLICE] `preview_source_patch(cell, changes, max_cells=...)` tests a factor-independent preview for maintenance workers. It accepts the same named edits as a full revision, including additions, deletions and endpoint moves. It overlays those edits on the selected snapshot and visits up to the given number of cells in breadth-first order, with stable source-name ordering. It retains every active row incident to those cells, including rows crossing the patch boundary. An anchor inside the connected patch proves support. A patch with neither an anchor nor a crossing row proves detachment. With a crossing row and no anchor yet, support is unresolved. Only the anchored case returns a numerical value. The preview writes no revision.

For the edited precision matrix $H'$, let $P$ be the patch and $C=-H'[P,\bar P]$ the nonnegative coupling to the outside. The preview solves the restricted square-root least-squares system with exterior values set to zero. Its response to setting all exterior values to one is $h=H'[P,P]^{-1}C\mathbf{1}$. The inverse is nonnegative by the discrete maximum principle. At a negative minimum, the strictly positive row sum and nonpositive off-diagonals make the corresponding equation negative, contradicting a nonnegative right-hand side. Thus if all exterior coordinates have magnitude at most $B$, the error due to truncating the domain at query $q$ is at most $B h_q$ in exact arithmetic. The implementation solves for the zero-boundary estimate and $h$ as two right-hand sides of one square-root system. [Discrete Dirichlet problems](https://web.stanford.edu/~lindrew/18.676.pdf)

For $n$ cells and observation magnitudes at most $D$, $B=nD$ suffices regardless of positive weights. If a positive estimate exceeded $nD$, the sorted positive coordinates and zero would contain a gap wider than $D$. Sum the normal equations over the coordinates above that gap. Internal edges cancel. Every crossing edge pushes in the same direction, and the ridge and any anchors on the upper side contribute positively. This contradicts stationarity. Reflecting all values gives the negative bound. The snapshot maintains a conservative historical maximum for $D$ and includes replacement values in the preview. Unsupported or disconnected regions can still inflate that maximum.

The reported `boundary_error` accounts for omitted coordinates in the mathematical model. Float64 rank and range checks can refuse a solve. This includes a zero computed allowance on an open connected patch with a positive envelope, since the exact boundary influence is strictly positive there. Roundoff is outside the returned quantity, including when a closed patch reports zero boundary error. The cell cap bounds solved coordinates, while a touched region can still require a scan of its local and incident bridge rows. The dense patch design also charges every incident observation. The patch may be small with strongly grounded coordinates and may need the full connected domain with weak grounding or remote anchors. This is a source-edit computation on supplied numerical observations. Learned encoders and late semantic effects remain to be tested.

`read_many(cells)` returns results in request order, including duplicate requests and `None` for structurally unsupported cells. A sparse read follows the nonzero coefficients of the requested regional readout and its conditional dependencies. Iterative reverse substitution recovers each required port once within the call. The memo belongs to that call. A narrow request can recover a strict subset of the ports, and a wide or poorly ordered request can still recover them all. A read with an unrepresentable value in its closure can fail after factor construction succeeded. The parent and published numerical operators remain available after a failed read.

Sparse compaction projects the selected snapshot's surviving ports, including numerically defined values that public reads suppress. It recovers their dependencies before projecting, which can traverse most of the parent graph. The compacted snapshot stores the projected values and drops the sparse cache. The next source edit reconstructs it from the compacted factors.

A poor elimination order can enlarge intermediate factors or place a frequently edited bridge early enough to invalidate a long sequence. A value-only edit can still propagate through the sparse factors while the default SVD reuses its design. Indexed updates avoid visiting unaffected elimination buckets and can reuse whole support groups. A single deletion can still require traversing a large old support group, and snapshot map copies and port position construction traverse retained state. Cold initialization, reordering and the first source edit after sparse compaction build the elimination index anew. Charge initial elimination, regrouping, compaction and re-promotion, coefficient checks, fill, factorization, source copies, support, planning, reverse substitution, readout and history. The measurable claim is a reduction in complete revision and late-read cost at matched accuracy. Compare the backends with sparse warm starts and static coarsening, including changing boundaries and repeated or wide reads.

#### Elimination order maintenance

An elimination order belongs to its snapshot. Reordering keeps the selected sources, ports and regional factors and recompiles the sparse coarse conditionals. Source revisions and compaction inherit that branch's order, including after compaction drops the coarse cache.

The in-process workspace evaluates a caller-bounded sequence of proposed orders on one pinned snapshot. It checks protected reads at the starting revision, then applies the same sequence of hypothetical source edits and late-read batches to the current and candidate orders. A candidate must preserve support and keep all checked values within the supplied absolute tolerance. It must improve at least one replay count while matching or improving the others. Qualifying candidates rank first by rebuilt buckets, then recovered ports, then the peak number of entries in retained conditionals and outgoing factors. Equal counts retain the first candidate. The receipt reports the selected order's conversion buckets separately from replay counts. Elapsed time and total memory remain unmeasured.

Publication installs the reordered starting snapshot. Hypothetical edits remain on temporary branches. The workspace compares the pinned parent identity with its current tip under a lock immediately before publication. A concurrent source revision or layout change invalidates the earlier candidate. Older snapshots remain readable with their own layouts.

`maintain_generated` derives a symbolic graph from current ports and the bridge endpoints promoted by the declared replay. It makes each local observation component a clique on its ports and adds current and anticipated bridges. It merges components for anticipated local links and conservatively keeps deleted links in the union. Greedy minimum-fill offers up to two distinct alternatives. It first tries a tie-break that delays ports touched often by declared edits and reads, then considers geometry alone. The full permutation preserves positions of cells outside the symbolic graph. The port cap bounds the vertex count of this search. Baseline replay, input scans and actual factorization have separate costs.

Automatic admission adds the candidate's conversion buckets to its replay rebuild count before applying the same comparison. Its guarded improvement must hold after that charge. Manual admission continues to report conversion separately. Bucket counts vary in numerical cost, and the charge excludes rejected trials, search, verification and retained history.

`JournaledRelativeHierarchyWorkspace` gives local processes a shared durable tip for sparse source revisions, generated layouts and compaction. A creator explicitly requests `create=True`. Workers open the existing SQLite WAL file with the same initial hierarchy and named source rows. A missing file fails an ordinary open. The root digest binds the protocol, membership, ridge, initial order, exact Float64 source inputs, NumPy version and error modes. Replay assumes the same numerical environment. The journal stores named source events, accepted orders with work counts, and compaction caps. Hypothetical edits and protected reads stay out of the journal. Recovery reconstructs admitted operations and verifies unseen event digests. Reopening from the root verifies the whole recorded chain. Its integrity checks assume trusted local writers and append-only history.

Each worker reads a numbered, digest-bound snapshot, prepares its change outside the write transaction, and then compares that exact tip in a write transaction before appending. A stale operation, including a no-op, cannot report success on a different tip. A process interrupted after commit can reopen or catch up to recover the accepted state. Cached workers verify their current digest and replay subsequent events. New workers reconstruct from the supplied root inputs. Replay reconstructs the committed layout and compaction without re-running their original guard workload. Source replacement follows the hierarchy's equality rule, which treats positive and negative zero as equal. Include cold replay, per-worker caches, numerical work and retained history in any cost comparison. Scheduling, workload prediction, distributed coordination and held-out usefulness remain open.

### Continuous feedback candidate

[SLICE] The [feedback kernel](../experiments/memory/latent_feedback.py) retains immutable observation coordinates and prepares their keys once. For a query state $h_t$, it projects a key query and attends over both those observation rows and the previously generated hidden rows. The two row families have separate key and value maps. The next state is the coordinatewise tanh of a projected carry from $h_t$ plus the weighted values. The new state is appended and participates in later steps. The seed remains separate, and the same call accepts an empty observation snapshot.

The observation bank is immutable and can serve multiple trajectories. A source revision uses a newly prepared bank and starts a fresh trajectory. The intervention method deliberately substitutes a generated row with all later rows held fixed, so tests can isolate an earlier state's role. An observation-value ablation with the latest state and routing keys fixed checks whether the late result uses numerical content. These controls address early-answer and placeholder effects documented in [causal analyses of latent chains](https://proceedings.mlr.press/v306/li26c.html).

This candidate can link distant observations on consecutive steps without a fixed propagation topology. Full attention scores $TN+T(T-1)/2$ rows over $T$ steps with $N$ observations, and the code retains the generated vectors and repacks their projected keys and values each step. This count omits other work including one observation-value projection per step and retaining the coordinate bank. Compare against direct keyed composition and a sufficient-statistics implementation on suitable cases. For a useful learned test, keep the total evidence and readout fixed, disclose how revisions invalidate prior work, and establish that a missing or replaced intermediate changes a held-out consequence. Selective retention and reusable partial trajectories remain hypotheses.

### Multiscale softmax feedback

[SLICE] [Tree feedback](../experiments/memory/tree_feedback.py) combines the existing continuous transition with a persistent binary tree of encoded observations. The caller assigns rows to stable integer slots. Each internal node retains a row count, mean payload, coordinatewise projected key bounds and its children. Row insertion, deletion and replacement copy the affected paths. An operator edit on the value map shares the same memory. Source edits and operator proposals use one revision check, so a source edit invalidates a proposal measured on the old bank. Queries pin their original snapshot, and the same call accepts an empty bank. Separate processes can recover the tree from its durable journal.

For node $G$ with count $n_G$ and key bounds $\ell_G,u_G$, a projected query $q$ gives midpoint logit $c_G=q^\top(u_G+\ell_G)/2$ and radius $r_G=|q|^\top(u_G-\ell_G)/2$. Each observation logit in the node lies in $[c_G-r_G,c_G+r_G]$. The collapsed node contributes mass $n_Ge^{c_G}$ and its mean payload. Each generated row has an individual exact score and participates in the same softmax normalization. The search maintains a frontier whose groups partition the entire observation bank. It expands the eligible group with largest $n_Ge^{c_G+r_G}(1-e^{-2r_G})$, subject to a frontier group cap and a per-group logit-radius tolerance. Compatible keys can remain grouped with zero approximation width. This approach shares the use of multiplicity-weighted representatives with [Multipole Attention](https://proceedings.neurips.cc/paper_files/paper/2025/file/359f442022ef31bcd9e41959576107ef-Paper-Conference.pdf).

Let $L$ and $U$ be the sums of lower and upper possible mass across all groups and generated rows. For each observation row, both its dense and collapsed normalized probabilities are at least $e^{c_G-r_G}/U$. A generated row has the same lower contribution in both cases. Summing their common lower probabilities gives $L/U$. The joint probability vectors therefore differ by at most $2(1-L/U)$ in $\ell_1$. Since observation coordinates lie in $[-1,1]$, this also bounds the maximum coordinate error of the observation mixture for the same query and generated state. The implementation scores the extrema directly and shifts midpoint and upper logits separately before adding group counts. The bound applies in exact arithmetic at one step. Its Float64 estimate excludes roundoff, weight-map amplification and accumulation through later recurrent steps.

With frontier cap $K$, each nonempty call scores at most $2K-1$ tree nodes and still scores all generated rows. A caller can use `with_frontier` and replay from the same seed and memory under a larger cap. Replay preserves the bank and recalculates the generated states. Traversal, heap operations, mixture construction, leaf and summary storage, slot assignment and history all contribute to cost. The fixed slot layout works well when keys useful to the same queries stay together. It can require full refinement when layout and key geometry disagree. Compare it with dense softmax and revisable feature moments on identical source edits and unseen queries. A learned layout must demonstrate that it preserves later consequences after replacement and deletion and that its repair cost fits the saved query work. [Differentiable sparse hierarchy as an architectural rival](https://arxiv.org/abs/2605.18753)

### Feedback with revisable feature moments

[SLICE] [Linear feedback](../experiments/memory/linear_feedback.py) replaces the softmax with a different attention rule. Each signed key or query coordinate contributes its positive part and its negative magnitude as separate features. Only aligned signs contribute. For each bank, let $x_j$ be the integer payload row, $f_j$ the key feature vector, $M=\sum_j f_jx_j^\top$ and $z=\sum_j f_j$. At each step the query feature vector $g$ reads observation and generated banks with one shared normalizer $D=g^\top(z_{\mathrm{obs}}+z_{\mathrm{gen}})$. The decoded mixture from a bank is $g^\top M/(127D)$. Separate value maps, carry and conditional terms then act on the mixtures before tanh. If no occupied feature aligns with the query, its routed contribution is zero. This is an arithmetic outcome. Factual absence requires its own evidence. Associative summation follows [linear attention](https://proceedings.mlr.press/v119/katharopoulos20a.html), and the capacity question is related to [fast weight programming](https://proceedings.mlr.press/v139/schlag21a.html).

The operator can add products for a caller-selected list of pairs of signed features from distinct coordinates. With pairs present, scaling unary key features by 127 puts them on the same fixed-point scale as products. Each unary query feature saturates at one before products are formed. At read time the complete feature vector can be rescaled by a common factor for numerical stability. In exact arithmetic, existing rows retain their relative unnormalized scores when rows are added.

Coordinates and key-map entries use a fixed signed grid with step $1/127$ on $[-1,1]$. Key projection uses an integer dot product, rescaling and clipping to that grid. Query projection and the final transition use Float64. Identical row coordinates yield identical integer contributions when inserted in bulk and retracted individually. With the state width and each bank's row count capped at $2^{31}-1$, even pair-feature cells have absolute value at most $(2^{31}-1)127^3$. Revise a snapshot with the caller's known current rows, then start a fresh trajectory. Generated states from an earlier bank may still depend on what was removed. A source ledger must attest membership because aggregate bounds alone cannot determine which row was present. An earlier generated-state intervention applies the add and retract rule while holding later rows fixed.

With state width $d$, key width $r$ and $p$ selected pairs, each bank stores $(2r+p)(d+1)$ Int64 entries independently of observation or feedback row count. Summary scoring and numerical update cost $O((r+p)d)$ per step alongside projections. The implementation retains one generated vector per step for interventions and protected-prefix checks, and its immutable history tuple copies $O(T)$ references at step $T$. Building the observation bank costs $O(N(r+p)d)$ for $N$ rows. Retaining snapshots and source records adds memory. The shared workspace can reuse the exact bank through a value-map edit and retract its current source contributions by ID. Changing the encoded key map or selected pairs requires rebuilding or separately maintaining an alternate representation.

Loss of within-feature distinctions is a failure mode for corpus memory. Two banks with the same distinct keys and opposite two-coordinate bindings can produce identical unary moments. One selected pair distinguishes the constructed case. Identical keys with opposing payloads can still produce the same first moments as repeated neutral payloads, even with pair features.

For selected observation coordinates, the bank can retain positive and negative payload magnitudes separately, using two more columns per coordinate. Contract each column with the query features and divide by the observation-only mass and the grid scale. The returned disagreement score is twice the smaller of the two normalized magnitudes. Zero can represent agreement, a neutral payload or no aligned observation. Generated rows never contribute to the score at a fixed query. An optional fixed projection of the score enters the recurrent transition before tanh. Weight programs account for that projection when fitting the observation value map. With $a$ tracked axes the observation bank adds $2a(2r+p)$ Int64 entries. The source journal pins the axis order and projection and replays sign statistics through the same row revisions. The generated bank keeps its original shape.

Opposing signs distinguish the equal-key cancellation fixture from neutral payloads. A disagreement score can also arise from different compatible bindings that share a query, and it cannot recover individual witnesses, joint signs or factual contradiction. The normalized score omits total evidence support and depends on where the representation places zero. Observation variance is a direct rival for this channel. Equal means and variances can have different opposing sign scores.

An optional `payload_branch=(payload_axis, selector_axis)` retains two additional banks of full payload moments. Each key feature keeps separate contributions from observation rows with positive and negative quantized values on the payload axis. At each step the previous hidden state's selector coordinate chooses the sign and the strength $s=|\operatorname{clip}(h_{\mathrm{selector}},-1,1)|$. The observation numerator blends the full bank with the selected partition, weighted by $1-s$ and $s$. The denominator keeps the unfiltered routing mass of observations and generated rows. This lets an unsupported branch fade to zero at a fixed query without amplifying the remaining rows or changing generated values. Neutral rows contribute through the unfiltered fraction. Disagreement still measures the unfiltered observations before selection.

The extra payload partitions require $2(2r+p)(d+1)$ Int64 entries. They are revised with the same retractions as the full bank, including consistency checks for each sign and the implied neutral partition. The source journal binds both axes. The branch can recover the association between the chosen payload sign and another coordinate. It still merges any distinctions that share its retained moments. Compare against direct row scoring for arithmetic, and against dense softmax feedback, sparse recurrent memory and indexed exact composition for learned utility. Charge construction, source custody, revision, readout and fallback, and test on held-out late consequences.

### Reuse of one pretrained attention head

[SLICE] The [attention head adapter](../experiments/memory/attention_head.py) takes bias-free Linear weight matrices as arrays. For query head $h$ and shared key/value group $g$, it builds a rectangular query map $Q_h^\top/\sqrt{d_h}$, a key map $K_g^\top$, and factored value application $V_g^\top O_h^\top$. It treats the output block $O_h$ as the selected columns of the full output projection. It uses the same factors for observation and generated rows. Stored coordinates stay in the donor's input width. Projected keys have the head width, and normal value applications retain both rectangular factors. Generated states and cached generated values still retain the full state width. A constrained value edit can materialize the square map, so its storage, arithmetic and replay costs differ from the compact starting point.

The equality checked is the isolated head contribution before tanh with equal query/key positions and a matched bank of rows. Explicit seed inclusion in the bank is supported. The feedback call uses supplied coordinates, omits rotary position differences, causal masking, residuals, normalization and the other heads, and adds its own tanh and generated history. These are architecture choices around the reused linear maps. The reference layout is the [Llama attention implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

A candidate donor is [SmolLM2-135M-Instruct at revision 12fd25f7](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac). Its public checkpoint header contains 120 BF16 attention projection weight entries across 30 layers. The bias-free configuration has 576 input coordinates, nine query heads, three shared key/value heads and 64 coordinates per head.

The [arithmetic smoke runner](../experiments/memory/attention_head_smoke.py) verifies the complete staged file against the published size and SHA256 before mapping the verified descriptor. It reads the four layer-zero attention projection arrays, the input RMSNorm scale, and embedding rows in order for token IDs 128, 1024, 4096, 8192 and 16384. It casts to Float64 and applies rowwise input RMSNorm with epsilon $10^{-5}$. The first row seeds the trajectory and the next four are its fixed observation bank. Query heads 2 and 3 test opposite sides of a key/value sharing boundary. For two steps in each head, the runner compares logits, softmax probabilities, pre-tanh contributions and resulting state against independent direct projection arithmetic, with absolute tolerance $10^{-9}$ and relative tolerance $10^{-12}$. The second step appends the previous generated state directly. A first-step control requires identical routing with zeroed observation values, an exactly zero ablated contribution and an intact contribution with maximum absolute coordinate at least $10^{-6}$. The receipt contains aggregate numerical errors, control magnitudes and elapsed time. A pass would establish selected-head arithmetic and a numerical observation effect. Any later language or latent-maintenance test must establish receiver competence, useful intervention effects and complete costs against an equally informed comparator.

Pending authorization, the proposed first run is one process on an existing Linux cloud CPU worker with Python 3.11 or later, NumPy, PyTorch, safetensors and at least 12 GiB free RAM. Stage the pinned checkpoint in a worker-owned directory with a read-only file and no concurrent writer. Run once with one CPU math thread, a ten-minute external wall limit inclusive of hashing, and zero GPU, training or provisioning. Record the source commit, runtime versions, worker allocation and exit status with the receipt. End on the first failure, timeout or completed receipt. The runner fetches nothing and publishes aggregate measurements only.

```bash
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  timeout --kill-after=5s 595s python -B -m experiments.memory.attention_head_smoke \
  --checkpoint /path/to/staged/model.safetensors
```

### Weight programs with preservation constraints

[SLICE] The [weight installer](../experiments/memory/weight_program.py) constructs a new linear map from an existing map $W$, target examples $(X,Y)$, protected activation rows $A$ and protected readout columns $B$. It solves for a correction $\Delta$ in the complementary input and output subspaces, where $A\Delta=0$ and $\Delta B=0$. Truncated singular-value decompositions select the editable rank. Evaluation uses the requested targets and the materialized map $W+\Delta$ in Float64. Edits that violate the tolerance or exceed the Frobenius norm budget are refused. Null-space protection has precedent in [AlphaEdit](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html).

Changing a feedback value map changes the output of a weighted mixture of observations. Protect the mixture the actual transition saw. Protecting selected raw rows can miss contributions from other rows. Protecting the full mixture can preserve a step while leaving other directions editable, and later queries depend on earlier states. The kernel now factors observation values out of the retained bank. If $X$ contains observation rows and $a_t$ is their share of the joint observation/history softmax, then $a_t(XW_v)=(a_tX)W_v$ in exact arithmetic. This lets a value-only edit reuse $X$ and its projected keys. Generated states and their keys and values remain pinned to their full, immutable kernel version. Public seed and hidden-state arrays use independent read-only views, so metadata changes on one returned view cannot reshape another trajectory. Factoring trades a dense projection of every observation during preparation or a value edit for one matrix-vector operation each step. It changes rounding order, and value arithmetic failures can now occur during a step rather than during memory preparation.

The replay-based installer takes a one-based step and a desired hidden vector strictly inside the tanh range. It uses the actual previous observation mixtures as protected inputs. At the target step it subtracts carry and generated-history contributions from the inverse-tanh target to specify the new value-map output. It returns an edited kernel only after replaying the seed against the same bank and verifying every earlier state along with the requested target. Cost comparisons must charge both replays and the edit solve. A fixed bank makes no guarantee about later attention queries because those depend on generated states. The fixture alters the second step while preserving the full first state, with the same observation bank and routing maps. Its constraints apply to the specified map and activation subspaces. Model-level preservation requires checking additional trajectories and deployment precision.

Conflicting targets and protected directions expose a limit of editing the chosen map. A later architecture proposal can add a separately gated operator, move the edit to another layer, or change the coordinate system and refit the installer. Compare those choices against ordinary constrained least squares and direct keyed execution with equal observations. Charge verification, projection, replay and accumulated edits.

### Programming observation routing

[SLICE] `program_observation_route(state, step, preferred_slot, other_slot, log_odds, ...)` changes which source a later hidden state consults. The [feedback kernel](../experiments/memory/latent_feedback.py) retains its shared query map for generated-history scores and installs a separate observation query map $Q_o$. Both sets of scores still share a softmax denominator. Generated-history scores at a fixed hidden state stay fixed, and their normalized mass can change. A call with empty or omitted memory skips the observation map entirely. The new kernel reuses the same bank, projected keys and value operators, including factored values. [Query-key and output-value circuits](https://transformer-circuits.pub/2021/framework/#splitting-attention-head-terms-into-query-key-and-output-value-circuits)

For target input $h$, current observation query $q=hQ_o$, key contrast $c=k_a-k_b$, and requested log odds $g$, the closest target query in exact arithmetic is $q^\star=q+(g-q\cdot c)c/\lVert c\rVert^2$. A first edit starts from the shared map. The installer fits $hQ'_o=q^\star$ and preserves projected queries at every earlier target input and through the final transition of each explicitly protected run on the same bank. It applies the Frobenius update cap to the installed map. The candidate replays the target prefix and all protected runs, then checks the target step's logit difference from the replayed hidden input. Dense feedback checks the scored logits. A tree requires zero logit tolerance and frontier capacity for every occupied slot. The tree checks the stored leaf keys at their named slots, including when equal-score leaves share a frontier node. Floating-point tolerances still apply.

The key and current hidden state must supply editable contrast. If $h_\perp$ is the part of $h$ outside the protected-input span, the minimum exact update norm for one requested gap is $|g-q\cdot c|/(\lVert h_\perp\rVert\lVert c\rVert)$ when both norms are nonzero. Source collisions and near-saturated preservation constraints can prevent a useful edit. A third source can dominate both selected sources even when their pairwise odds match. Compare actual late consequences under source interventions as well as route scores. Candidate rivals are editing the shared query map, installing a state-gated query residual, and editing the value map or a conditional payload term. Use the same source changes and protected inputs, and charge fitting, replay and ongoing projection costs.

An edited tree separates whole-route reuse by the observation query map and keys retained node scores by the projected observation query. The journal can reopen with the fork supplied in its initial kernel and binds those weights in the root digest. A declared routing goal can also carry a proposed observation query through the shared journal and repair queue.

#### Programming joint source mass

[SLICE] `program_observation_mass(state, step, slot, min_mass, ...)` programs a floor on one occupied source slot's share of the same joint softmax. At the target input $h$, keep generated-history logits $c_\ell$ fixed. For source key $k_s$, the log mass at observation query $q$ is $k_s\cdot q-\log(\sum_j\exp(k_j\cdot q)+\sum_\ell\exp(c_\ell))$. Its gradient is $k_s-\sum_j P_j k_j$ with observation probabilities $P_j$ taken from the joint distribution. The objective is concave in $q$, so its derivative along a line cannot increase. [Log-sum-exp convexity](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)

The default search chooses one unit direction from the initial gradient. With update cap $M$ and target input component $h_\perp$ outside the protected span, the ideal query displacement cap is $M\lVert h_\perp\rVert$. The existing weight installer checks the materialized edit and protected inputs under its Float64 rank and norm policy. The line search brackets the bounded peak if needed and bisects toward the first observed feasible crossing. Each search uses at most two blocks of 60 bisections and allows numerical slack in the proposed displacement. A finite search can refuse near the threshold or miss a feasible direction. [Shifted softmax arithmetic](https://arxiv.org/abs/1909.03469)

The target replay checks the actual slot probability against the requested floor. A full-resolution tree uses the executed probability of the frontier group containing the slot divided by its number of leaves. The fitting pass scans every observation key and includes generated-history competition. Every scalar search evaluation still scans the precomputed scores and slopes. The floor applies to the replayed step and pinned bank. A source intervention checks whether a change in that source also changes the late result.

`search="multivariate"` also searches outside the initial gradient direction. For log mass $f(x)$ at query displacement $x$ and floor $t$, concavity places all feasible candidates in the halfspace $\nabla f(x)\cdot(y-x)\ge t-f(x)$. Each iteration projects zero onto the intersection of a new tangent halfspace and the retained halfspace $x\cdot(y-x)\ge0$. In exact arithmetic, the projection inequality preserves containment of the target set when the cuts contain it. The implementation uses a small margin that shrinks with the shortfall to approach the requested floor. It checks the two halfspace residuals before keeping a projection. An above-radius proposal can be tested at the radius as a final candidate. This uses at most 256 projections. Refusals can reflect finite scans, narrow feasibility near the floor or nearly parallel normals. The Float64 search supplies a candidate for strict replay, without a global optimality or infeasibility certificate. [Outer approximation](https://pcombet.math.ncsu.edu/sicon1.pdf)

The multivariate fit recomputes scores and gradients across the source bank. Its dense work is proportional to the number of occupied slots times key width per projection. The one-direction fit caches scalar slopes after its first pass. Compare costs with the initial ray and with a fixed-radius Frankâ€“Wolfe fit. A state-gated query residual remains a separate candidate when the target input lies in the protected linear span. Compare shared and observation-only maps at matched source corrections and protected runs. Include fitting scans, replay and projection in the cost.

### Conditional architecture growth

[SLICE] The feedback kernel can expand by appending a new term to its computation. A signed [conditional term](../experiments/memory/latent_feedback.py) adds $(h\cdot u)(m\cdot v)w$ before tanh, using previous hidden state $h$ and observation mixture $m$. Two different hidden states that read the same mixture can require different value contributions. With other maps fixed, changing the shared value map alone gives the same contribution to both.

The signed installer projects the target hidden state onto directions orthogonal to the previously protected transition inputs, then normalizes that projection to obtain $u$. It takes $v$ along the target observation mixture and computes $w$ from the desired inverse-tanh value minus the old preactivation. It refuses zero observation mixtures, negligible state contrast, saturated targets and terms outside the norm cap. Old operators contribute to the old preactivation. Replay checks the target and the preserved trajectories. With observations omitted the term is inactive, and opposite state or observation directions reverse its contribution in the controlled fixture. [FiLM](https://arxiv.org/abs/1709.07871) and [WeightCode](https://arxiv.org/abs/2611.25476) provide precedents for feature conditioning and selectively active weight programs.

A one-sided term replaces the signed state factor with $\max(0,h\cdot u-b)$ for a nonnegative threshold $b$. The installer approximates projection of the target state onto the convex hull of the protected transition inputs and zero, using at most 32 Frank–Wolfe updates. It takes the unit residual direction, sets $b$ to the highest protected projection or zero, and explicitly checks the target margin. A positive margin makes the term inactive at the protected inputs and active at the target. Replay then checks actual trajectories. A full linear span can still allow this separation. A target inside the protected convex hull cannot pass with a single gate constrained to be inactive throughout that hull, and the finite search can miss separators outside it. Guard targets stay outside synthesis. An active-side guard can reject a hinge after fitting. A compact-support gate is a competing design when protected behavior lies on both sides of a target. [Frank–Wolfe](https://proceedings.mlr.press/v28/jaggi13.html)

The workspace can admit one data-only term under an explicit term-count cap and the same fixed probe and revision policies. `allow_hinge=True` permits one-sided terms in manual admission, automated search and journal replay. Signed and one-sided terms count against the same cap and edit norm budget. The journal binds that permission and records a one-sided term's exact threshold. It checks unit input axes before multiplying factor norms, preventing scaling from understating an update through Float64 underflow. A stack of $K$ terms adds $O(Kd)$ arithmetic per step at width $d$, plus retention and proposal costs. The one-sided search adds up to 32 scans of the protected states before replay. Compare both families against constrained bilinear and nonlinear regressors and equally informed direct conditional operators. Generalization and semantic binding require separate evidence.

### Maintenance workspace for numerical operators

[SLICE] The [feedback workspace](../experiments/memory/feedback_workspace.py) pins the current observation bank, operator and measured errors against a fixed probe policy. A worker selects an unresolved probe and prepares a data-only value-map edit or, when permitted, a conditional operator using the current bank. The installer protects the complete observed prefixes of already satisfied probes on that bank. The workspace constructs and replays the proposed operator independently. It requires the chosen target to enter the fixed tolerance, keeps previously satisfied probes within that tolerance, and refuses error increases on other unresolved probes. It checks the installed change against its own per-edit norm cap. Admission also checks the chosen target's pre-edit prefix and the retained prefixes of satisfied probes. Prefix anchors stay fixed across edits on the same observation bank. An authorized observation replacement recomputes the anchors.

Replays for new proposals run before the write transaction. Publication compares the complete base revision with the current head, so neither an intervening operator edit nor an observation replacement can be overwritten by stale work. Observation replacement is a separate caller-authorized operation that re-evaluates all probes with the current kernel. Both kinds of update preserve live earlier revisions for callers holding them.

For a shared workspace, initialize and reopen [`JournaledFeedbackWorkspace`](../experiments/memory/feedback_journal.py) with the same initial kernel, observations, probes, limits and path to a trusted local SQLite file. Keep the file and its sidecars in private runtime storage on one host. Call `snapshot()` to catch up on admitted events, then use the same `propose`, `publish` and `replace_observations` methods as the in-process workspace. The initial fingerprint binds the complete numerical representation and probe policy, including factored value maps. Each event records the revision, previous digest through a hash chain, change kind, exact Float64 payload and selected probe for operator edits. The selected probe matters because its pre-edit prefix is protected even when the same edit satisfies other probes. Reopen replays these decisions through the original admission rules. Pin the code and numerical environment across workers because borderline decisions depend on floating-point evaluation. A change to event encoding or replay rules requires a journal version change.

With a linear feedback kernel, pass `source_ids` in the same order as the initial observation rows. The root binds the ID-to-row association, the selected feature pairs, disagreement channels and any payload branch. Use `revise_sources(base, changes)` with a mapping from IDs to bounded replacement vectors, or use `None` to delete a known ID. The workspace reads the old rows for just those IDs from a shared SQLite index on the checked base. The source revision updates that index and appends one event with the actual before and after rows in the same transaction. Probe results and anchors are recomputed on the revised summary. A moment query on a retained snapshot uses the summary directly. Source IDs establish numerical row custody within this workspace. Source authentication, text witnesses and answer provenance require separate interfaces.

An atomic insertion compares the checked parent revision number and digest with the current durable tip. Dense and moment journals use SQLite WAL with full synchronization before acknowledgment. A process interrupted after commit can recover through `snapshot()` and see that its prior base is stale. Cold recovery starts from the supplied original inputs. Live workers replay only unseen events, and the moment source index is stored once per journal. Count all replay, serialization, hashing and retained payloads when comparing coordination costs. A moment source event stores before and after rows for its changes. A full softmax observation replacement stores its entire bank, and a dense edit stores its entire installed map. Compare with a single admission process to identify when duplicated worker replay costs more than serialized ownership. Checkpoint compaction, distributed operation and worker scheduling remain open.

Fixed targets and anchored prefixes bound the checked trajectories. Total parameter movement and behavior on unseen inputs still require separate controls. A disagreement between acceptance constraints can exhaust the editable subspace and call for a different operator or geometry. Compare sequential proposals against batch constrained least squares and an equally informed direct executor. Include proposal generation, unsuccessful and superseded replays, and all retained observations and revisions in the cost. Use independent held-out consequences before treating this numerical protocol as useful learned maintenance.

Optional `guards` add fixed validation endpoints. Operator synthesis uses the working probes and their protected trajectories. Admission independently measures the guards on the same source bank and rejects guard error increases above the greater of the base error and tolerance. Guards preserve endpoint errors. Working probes also enforce their earlier-state anchors. A source revision recomputes both working and guard errors, and later operator candidates inherit that new baseline. The journal binds guard definitions and order in the root and reapplies them on replay. These guards are repeatedly consulted during selection. Independent scientific evaluation requires separate unconsulted cases.

Call `maintain(max_candidates=n)` for one bounded operator search. It takes one revision, orders unresolved working probes by descending error, and tries value-map, signed conditional and, when enabled, one-sided edits in that order. Conditional proposals require remaining term capacity. Synthesis failures and admission refusals each count against the cap. Among the admitted candidates it tests, the worker prefers more satisfied working probes, then lower summed working error, then fewer conditional terms. It publishes at most one edit. The local receipt records the base, tested alternatives, refusals or ranking keys, and any committed revision. A source or operator commit racing with search can make the result stale. An exhausted search with untested alternatives returns a continuation that `maintain(max_candidates=n, resume=receipt)` can use on the same live revision. Other outcomes distinguish no admissible tested constructor, satisfied working and guard probes, and unresolved guards when no working target remains.

Set `max_workers` above one to test distinct candidates concurrently in the same process. The default is one worker. A single coordinator checks the shared revision before each group of at most `max_workers` candidates. Candidate threads read the same immutable revision, and each candidate runs synthesis and independent admission on it once. The coordinator processes results in candidate order, so finishing order does not alter the tie-break. With a change detected at a group boundary or before publication, the call returns stale after retaining the completed attempts. Already dispatched computations finish. An unexpected error drains outstanding work and propagates before publication. Multiple unexpected failures in one group are reported in candidate order. Candidate threads inherit the caller's NumPy floating-point error policy. Caller-installed callbacks can execute concurrently. Array operations still use the pinned numerical environment and its own library threads. [NumPy thread safety](https://numpy.org/doc/stable/reference/thread_safety.html)

For separate processes, the journal can produce a bounded `plan_repairs(max_candidates=n)`. The plan pins the root, tip, revision number, ordered candidate range, protocol and data-only NumPy error policy. Plans accept policies without NumPy callback modes. A caller assigns disjoint half-open ranges within `plan.candidates` to trusted workers that opened the same journal. Each calls `synthesize_repairs(plan, start=first, stop=end)` and returns a shard with a byte payload or refusal per position. Synthesis checks the working trajectories. A worker using two-resolution tree repair runs admission locally to decide whether a draft needs refinement. The coordinator applies its own admission checks to every returned candidate. Journal opening and historical replay still incur their ordinary guard checks. The protocol assumes workers report their assigned outcomes honestly and use a pinned numerical environment.

The coordinator calls `publish_repair_shards(plan, shards)` and requires exact coverage of the planned range. It reconstructs proposals from fixed-size Float64 payloads and evaluates them through normal admission, including guard and prefix checks. It computes ranks itself and uses the earliest candidate on a tie. Current ancestry checks and the final compare-and-swap fence publication, and stale plans raise `StaleFeedbackRevision`. Plans and shard reports can cross process boundaries as data. The coordinator can use `plan_repairs(max_candidates=n, resume=receipt)` to continue an exhausted search with the same revision object in the same workspace instance. The journal durably retains accepted updates.

This round is callable alongside a source worker. A successful commit describes its committed revision, including when a subsequent source change reopens an error. After an interruption at commit, recover the durable journal before treating an outcome as definite. The candidate and worker caps bound dispatch. Cross-process handoff adds startup, catch-up, payload storage, transport and coordinator replay. A dense replacement sends $8d^2$ bytes at width $d$, while signed and one-sided terms send $24d$ and $24d+8$ bytes. Separate coordinators can still choose overlapping allocations. A matched-cost swarm comparison must include duplicated and stale work, compare with a single worker and a batch constrained fit, and test independent late consequences. Continuous scheduling and learned issue discovery remain open.

#### Tree repair at two resolutions

[SLICE] Pass `max_reference_frontier` to the in-process workspace with a tree kernel. The supplied kernel's cap and logit tolerance define the drafting policy. The workspace serves, measures and admits against one full-resolution tree view with zero logit-radius tolerance. The explicit reference cap must cover the number of occupied slots. Initial preparation and later source revisions check that capacity before measuring probes. In exact arithmetic this cap allows refinement to every occupied slot when the logits require it. The tree's Float64 limitations still apply.

Each search position covers one probe and edit family. A worker first drafts with the smaller view over the same weights and bank. A successful draft faces the full-resolution target, norm, working-probe, guard and protected-prefix checks. If draft synthesis or full-resolution admission refuses it, the worker tries synthesis at full resolution and applies the same checks to that proposal. An admitted draft ends that position. The full-resolution synthesizer could produce a different admissible proposal with a better rank. Rank reflects the candidates actually checked. Manual proposals use the same fallback, and manual publication replays admission against the current base. Both paths preserve the workspace revision check.

The receipt records resolution, frontier cap and refusal stage for each attempted synthesis and its admission. At most two syntheses and two admission replays occur per position. The reference view supplies the working-probe order, ranking, history anchors and published state. Approximate queries can start from the same bank with a smaller cap when the caller chooses that policy. Source edits preserve the tree by path copying and invalidate stale operator work. In a parallel wave, repeated stale notices describe one revision change. Unexpected worker errors still propagate.

This construction tests whether cheaper drafting can preserve repairs on a bank small enough for full replay. Wide attention intervals can add the cost of both drafts and repeated admission. Compare with full-resolution-only synthesis, and with selective replay that can justify skipping fine work while protecting later trajectories. A one-step probability estimate alone supplies no such guarantee. [Biased low-fidelity evaluations as a related search problem](https://proceedings.neurips.cc/paper_files/paper/2016/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html)

#### Reusing attention across candidate replays

[SLICE] The in-process tree workspace can set `reuse_tree_routes=True` to retain each working and guard probe's measured attention on its current revision. An entry holds the unprojected observation mixture, the weighted generated-history value and the resulting hidden vector. Whole-route replay requires the same immutable bank, routing maps, frontier policy and binary seed. The candidate computes each new hidden vector with its own operator and projects that vector for future feedback. If the new hidden vector differs by even one bit from the recorded vector, the replay rebuilds each remaining attention route. Matching a later hidden state alone would leave the earlier generated rows unverified. This transfers the idea of exact prefix reuse to branching latent operators. [Prefix reuse in SGLang](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)

The stored routes share no operator-specific carry or observation-value application. Identical past hidden vectors and the same feedback projections determine the same generated keys and values under the declared numerical environment. The candidate's replay still compares its actual working outcomes, guard errors and prefix anchors with the original tolerances. The selected base prefix comes from that revision's measured trace. Source or bank edits build fresh normalized routes, and stale replay remains fenced by the ordinary revision check. Workspace synthesis uses its own existing replay. Each journal worker builds its own route traces and cache.

For width $d$, each recorded step retains up to three $d$-vectors in addition to object metadata. Candidates can share immutable route entries while preserving their own resulting states. Retaining old revisions retains their traces. Score counts exclude byte comparisons, trace allocation, transition and feedback projections. A short common prefix or high probe churn can reduce reuse. Compare with full replay, batching independent candidates over shared memory, and selective refinement on the same held-out late tasks.

#### Reusing node scores after source changes

[SLICE] Set a positive `max_retained_tree_scores` on the in-process tree workspace to enable route recording and retain up to that many visited node scores per probe step. The revised traversal can use a score from the corresponding old probe step if the projected query bytes match and the immutable scored node is the same object. A path-copied slot edit preserves many node identities. A freshly prepared bank generally preserves none. The stored score covers only the node's bounding logits and width for that query. It does not include its place in the frontier, normalized probability or any generated-history contribution.

The new walk starts at the revised root, chooses a frontier with fresh queue order, rescores generated history and rebuilds the normalization and output mixture. This also permits node-score reuse after the generated prefix diverges when the projected query repeats exactly. Whole-route replay keeps its stricter gate. Only nodes visited and recorded in the new traversal enter the new step's cache. Lookup, traversal, queueing, normalization and retained score records still contribute to cost. Query changes and small retention caps can eliminate the savings. Compare with fresh traversal and a cache organized by frequently recurring query directions, charging lookup, updates, memory and stale work. [Self-adjusting computation](https://arxiv.org/abs/1106.0478)

#### Durable tree workspace

[SLICE] Create a tree journal with `JournaledFeedbackWorkspace(..., create=True)`. Ordinary opens require the existing initialized file. Supply the standard tree kernel and the same initial rows, working probes, guards and repair policy in each trusted process on one host. A distinct tree root binds row order, weights, depth, both resolution policies, NumPy version and NumPy error modes. The tree journal checks that policy on access and accepts data-only error modes. It also binds the original probe and repair parameters. Workers choose their own route and node-cache limits. The journal stores the changes and admitted operators, and workers rebuild the numerical tree and their selected caches by replay.

`revise_observation_slots(base, changes)` appends one event with ascending integer slots. The tree builds each affected summary from the batch's final children, so serialization order cannot introduce a temporary numerical state. Each record stores an eight-byte slot ID and a one-byte replacement tag. A replacement carries one little-endian Float64 row, and a deletion carries only the header. Thus a batch of $k$ changes with $a$ replacement rows of width $d$ uses $9k+8ad$ payload bytes, excluding the SQLite row and digest. Whole-bank replacement gives the supplied rows consecutive slots again. Slot and operator events share the same revision-number and parent-digest fence. Workers that lose the fence reopen or catch up before preparing another edit.

Tree journals use SQLite `DELETE` rollback mode with full synchronization. This avoids a known WAL-reset race on the tested SQLite 3.50.4 runtime. Existing dense and moment journals retain their format and journaling mode. Tree catch-up pins the durable ending revision and digest, then reads one event at a time and releases its SQLite read lock before numerical replay. A writer can append during a slow replay. The reader returns the pinned prefix, and the next catch-up sees later events. The process installs its new local head only after the entire prefix reaches the pinned digest. [SQLite WAL-reset bug](https://sqlite.org/wal.html#walreset) · [SQLite isolation](https://sqlite.org/isolation.html)

Source slots establish revision identity for numerical rows. They do not supply an external source witness or authorize arbitrary access. Scope is cooperating local processes and an append-only journal on a trusted filesystem. Compare a central owner with per-worker replay against shared immutable checkpoint pages on the same insertions, deletions and delayed repair probes. Charge startup replay, duplicate trees and caches, refused drafts, and writer contention. Cross-host storage and recovery from a machine-level power loss remain open.

#### Leasing repair work

[SLICE] [FeedbackRepairQueue](../experiments/memory/feedback_repair_queue.py) shares the tree journal's SQLite file. A coordinator enqueues a validated, finite `FeedbackSearchPlan` with `enqueue(plan)`. The queue identifies the window by its canonical plan digest. Re-enqueuing that window is idempotent, and overlapping windows on the same revision are refused. Workers open the same journal and call `claim(lease_seconds=...)` to obtain one candidate position. They synthesize it through `synthesize_repairs(lease.plan, start=lease.position, stop=lease.position+1)` and return its shard through `complete(lease, shard)`. The queue gives unclaimed positions priority over expired ones. [SQLite transaction semantics](https://sqlite.org/lang_transaction.html)

Claim, renewal and completion use a write transaction to check the current journal tip and assignment together. A worker can extend a live assignment with `renew(lease, lease_seconds=...)`. A reclaimed position gets a new token. The first completion requires the live token before its recorded deadline. On the same revision, a retry with the completion token and identical result acknowledges the already committed result without changing it. A different result cannot replace it. Deadlines use the host wall clock, so clock changes can shift recovery time. If a source or operator revision changes the tip, old positions become unclaimable and their results cannot be published. Enqueuing the next revision removes the obsolete assignments and results from the queue.

The coordinator can use `ready_plans()` after reopening to discover complete windows on the current tip, including windows with a recorded refusal. Calling `publish(plan)` returns `None` while positions remain incomplete. Ordinary shard admission ranks candidates within one complete window and publishes at most one through the journal fence. Callers may enqueue disjoint windows on one tip and choose which complete window to admit first. An accepted edit makes all other windows on that tip stale. The caller controls polling, renewal timing, cancellation of in-flight CPU work and future-round selection. Compare worker pull and assigned ranges on identical revision traces and window boundaries, charging lease transactions, expired work, duplicated checks, process startup and total memory.

`receipt(plan)` reads a recorded decision for a known window, including one on a past tip. `latest_receipt()` discovers the last recorded decision after reopening. When a proposal wins, the journal appends its operator edit and stores the receipt in the same write transaction. The receipt identifies the selected attempt and its published revision and digest. Reading it checks those identifiers against the recorded event. Calling `publish(plan)` again after a committed edit raises `StaleFeedbackRevision`. A refusal on the current tip yields the stored attempts and continuation offset, if one remains, together with the reopened workspace's live base. It skips numerical admission on that retry. [SQLite atomic commit](https://sqlite.org/atomiccommit.html)

Decision metadata remains readable after obsolete leases and candidate payloads are pruned. Its storage grows with the windows and attempts recorded. Coordinators can still duplicate admission work when they race. The existing journal fence limits them to one committed edit. These recovery contracts apply to trusted local processes sharing one journal.

#### Leasing source routing edits

[SLICE] A working `FeedbackProbe` can declare `routing=FeedbackRoutingGoal(slot, min_mass)` with a floor strictly between zero and one. Its final transition measures the endpoint and the executed probability of that slot in the same trajectory. A missing source measures zero. A source revision can delete or restore a slot and establishes a new baseline for both quantities. Routing goals require standard dense softmax or a tree with zero logit tolerance and frontier capacity for all occupied slots. Guards still declare endpoints. Working probes with routing goals replay directly, while ordinary working probes and guards can use the existing optional tree caches.

A selected working probe must meet its endpoint tolerance and its declared floor, if present. Independently, every floor attained on the base stays at or above its threshold through an operator edit. The workspace keeps endpoint error, anchor and guard checks, including an established prefix anchor when only the floor remains unresolved. The numerical mass is an admission criterion. Source causality and generalization still require independent interventions.

The planner adds a `route` candidate after the other kinds for each working probe whose floor is unmet. `propose_routing(base, probe)` fits the observation query using the multivariate search at the full reference resolution. The query fitter targets the floor. The coordinator separately checks the output and all admission constraints. The result can be refused even with a valid mass increase. An opt-in joint query and value proposal can meet both parts of the contract.

A journal with routing goals binds their ordered probe indices, slots and floors in its initial fingerprint and uses repair-plan version 2. A `route` payload holds the full rectangular little-endian Float64 query map. The root-bound shape and exact length govern decoding. The same checked append handles the operator event and its durable queue decision. Reopening replays the original admission against its source revision. Journals without routing goals keep version 1. Leased work remains caller-started. [SQLite isolation](https://sqlite.org/isolation.html)

#### Repairing query and value together

[SLICE] Set `allow_joint=True` on a workspace with a working routing goal to offer one additional candidate after `route` on each unmet source floor. `propose_routing(base, probe, joint=True)` first fits the query on the full reference bank. It holds that draft query fixed while fitting the value map to the requested endpoint. Both fits use the selected probe's prefix and the endpoint-correct other working trajectories. The admission replay uses the original saved anchors and independently checks the goal, previously attained floors and the guards.

The two installed deltas share one cap, $\operatorname{hypot}(\|\Delta Q_o\|_F,\|\Delta W_v\|_F)$. Measure them from the effective observation query and dense equivalent value map at the pinned revision. The value fit receives only the unused part of the cap. Fitting the value map may materialize inherited factors. Charge their changed storage and inference costs even if the numerical update is small.

The journal binds explicit joint permission in its root and uses plan version 3 only when enabled. The `joint` payload stores the rectangular query followed by the square value map in little-endian Float64. For state width $d$ and key width $r$ this is $8(dr+d^2)$ payload bytes. Both maps publish through one event and one receipt on the pinned parent. Existing version-1 and version-2 roots, plan layouts and receipt encoding still apply when joint permission is absent.

The default single-trial constructor chooses the query from the routing floor before seeing the required value movement. A larger or differently directed query can make a much cheaper value correction possible. Compare the staged fit with a bounded query search that prices the required value solution at each draft, and with a joint fit over both blocks. With a frozen prefix and routing, the value fit is linear before tanh. That structure motivates searching the nonlinear query coordinates with the value fit inside the objective. Changing the prefix changes the routing, so replay remains the admission test. [Variable projection as a related method](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)

With joint edits enabled, `joint_trials=N` can try more than one source floor in the same bounded candidate. The first level is the declared floor. Each later level moves halfway toward one, and sampling stops before a level rounds to one. Each fit uses the same pinned original maps and cap. The worker retains the smallest combined installed norm among the pairs that replay the original selected endpoint, floor and prefix. Ties keep the earlier level. The submitted pair still faces all independent coordinator checks. Guard outcomes stay outside trial selection, so a different sampled pair could pass a guard that refuses the submitted one.

More than one trial binds the count in the journal root and uses plan version 4. The order of work items, joint map bytes and receipt encoding stay the same. The worker checks for a stale revision between fits and before returning, including when an earlier fit succeeded. Charge up to $N$ query programs, value fits and target replays for one work item. Finite mass levels can miss a feasible interval or a better direction at the same mass. Compare against adaptive placement of levels and direct query search with the value fit inside the objective.

#### Admitting joint trials separately

With `joint_trials>1`, set `separate_joint_trials=True` to give each sampled mass its own work position. It follows the other constructors for that probe. The positions `joint:0`, `joint:1` and so on use the same level sequence as the combined trial search. `propose_routing(base, probe, joint=True, trial=i)` fits one such position. A worker checks the selected output, declared floor, original prefix and combined cap without consulting guards. The coordinator applies full admission to every returned pair in the completed window, then uses the usual behavioral rank with plan order for exact ties.

Separate positions use plan version 5 and a distinct root binding. They publish the existing `joint` map payload and keep the trial index in the receipt's plan position. Receipt recovery checks the canonical event and the selected position's recorded rank. Versions 1 through 4 keep their plans, events and receipt layouts under their original configurations. In version 5 each sampled level consumes a candidate slot. Put the alternatives in the same window when they should compete. An earlier commit can make another window stale.

Separate positions retain one full map payload for each successful fit until queue cleanup and repeat coordinator replay for each pair. The winning journal event still holds one pair. Compare with a bounded shortlist from one worker or the original local selection on matched mass levels and window boundaries, including lease traffic, process setup and stale work. [SQLite transactions](https://sqlite.org/lang_transaction.html)

#### Refining source-share trials

With separate joint trials enabled, set `joint_refinements=R` to offer one additional leased candidate after the $L$ fixed levels when there is a representable midpoint between any two levels. This position prices the fixed levels on the same pinned bank. While pricing, the query and value updates each have the original cap. A pair may exceed the combined cap and still give a finite price if it replays the original selected endpoint, floor and prefix. Failed fits have infinite price. Only a pair within the combined cap can be returned for coordinator admission.

Each of at most $R$ additional fits bisects a sampled interval. Intervals with a representable midpoint are ordered by the smaller endpoint cost, then the larger endpoint cost, then descending width and earlier position. The worker retains the smallest eligible norm it observes, with earlier fits winning exact ties. The coordinator independently checks the selected pair and the other drafts in its window. The refinement prices no guard outcomes. Its single submission can still hide an adaptive alternative that would pass.

A positive refinement count is root-bound and uses plan version 6. The extra position uses `joint:L` and the same event and receipt formats as fixed positions. The earlier defaults keep their original plan versions. The extra worker runs up to $L+R$ fit attempts. When the fixed and refinement positions all run, total joint fitting can reach $2L+R$ attempts per probe before retries. Charge repeated anchors, rejected replays, lease expiry and window delay. Compare with predeclared interior levels and search over query coordinates at matched total cost. The bounded midpoint search can still miss feasible intervals.

### Architectural exploration and review

The project should try new representations and forms of computation early. Treat coordinate geometry, transition rules, multiscale structure, decoder coupling and the network wiring as variables. Weight-space programming can reuse pieces of a pretrained network or install computations the original network lacked. Seek constructions that remove expensive work from the maintenance and query path while preserving useful consequences. Keep the current field replaceable. Its contractivity gives us a checkable starting point, and its limited reach makes genuinely different designs worth trying early.

Reviewers should work from the shared goal and propose alternatives independently. For each major direction, identify the assumption most worth breaking, propose a materially different construction, and specify the cheapest observation that would change the decision. Examples include persistent recurrent states with explicit long-range summaries, continuous feedback with selective retention, or coordinates whose structure changes as new dependencies appear. Each proposal should specify what survives a revision, why its computation could be cheaper at comparable quality, and how an intervention would show that the state caused the result. Keep failed swings and their transferable lessons in the existing ledgers so the next candidate can start further ahead.

One assumption to challenge is that every revision must propagate at the original resolution. The two-level numerical candidate makes that question executable for supplied relative observations. A learning candidate could choose regions or transfer operators and refine locally when a late question requires detail. Trainable candidates would need to preserve the effect of deleting or restoring a bridge on the coarse state and the eventual answer, alongside unrelated and historical reads. Reviewers should propose replacements for the mechanism whenever they expect a different representation, transition or wiring change to preserve more useful consequences per unit of work.

Challenge the assumption that a maintenance worker needs access to the entire original factorization to preview an edit. The local source patch makes an initial comparison possible. A further rival would maintain local response summaries and low-rank cross-region couplings, refining those summaries only near a proposed change. Another is to grow the patch where residual influence is largest using a sparse diffusion step. Compare these with retained factors and warm-started conjugate gradients on the same mixed edits and hidden late queries. Vary route strength after removing a strong bridge with a weak alternate route, and measure wrong consequences, unresolved support, and the cost to reach a useful estimate. Include summary repair and drift across successive edits. [Local diffusion solvers](https://arxiv.org/abs/2410.21634)

Use the shared review branch for code, evidence summaries and competing proposals. Pull and inspect external changes before each working batch and publication. Commit coherent milestones with explicit paths. Record decisions in this architecture document and update the results and failure ledgers when their evidence changes.

### Co-evolution and the first decision

Keep two mechanisms in contention. The recurrent candidate revises a bounded field via repeated local transitions. The continuous-feedback rival passes newly generated hidden states back as inputs, retaining the states that future operations can still use. Compare them with identical observation access and readout, inspect which past errors each retains, and charge both history and repeated computation. Weight-space programs can propose an interpretable basis, roles and transition blocks. They can also modify the wiring itself. Compare programmed initialization and bounded calibration to intact pretrained features, randomized features and the strongest equally informed non-neural implementation. [WeightCode](https://arxiv.org/abs/2611.25476) motivates retaining a reusable installer alongside its resulting parameters.

The first scientific decision is whether latent updates improve a late consequence beyond a competent conventional computation at matched information and cost. Use revisions of noisy relative observations with conflicting paths, genuinely necessary combinations of observations and a disconnected component. Ask the late question after ingestion. Change one observation with topology held fixed, remove a necessary premise, restore it, and separately demand the historical and unrelated values. Compare against an optimized warm-start solver for the same equations and an [incremental factor-graph solver](https://dspace.mit.edu/entities/publication/0cea54ca-7a8b-47f7-b4e9-337ab9295bad) for the underlying task. Withhold witness content from the answer computation. A useful output must change with the relevant latent intervention and survive permutations that break incidental identifiers. A future calibration brief must pin access, data, controls, resource limits and stop rules.

## Claims and evidence boundary (2026-10-15; not a new experiment)

| Claim | Tag and current boundary | What is still required |
|---|---|---|
| C1: retained content causally influences a trainable receiver | MEASURED: scoped intervention controls and synthetic fast-weight PASS_BOUNDED; pretrained-prefix receiving/fit acceptance remains unadmitted | Actual prefix smoke then frozen held-out trained readout/restart; no information-absence conclusion from instrument failure |
| C2: independently selectable futures and continued consequences | TARGET: not established by the prefix slice; bounded categorical reads are not general futures | Late selection, legal continued operations, causal controls |
| C3: selective revision preserving unrelated and historical uses | MEASURED: synthetic host re-encoding/current-history reads only; learned/source-free updating unestablished | Selective invalidation/repair, unrelated retention, immutable history; charge source reopening |
| C4: better complete quality-cost frontier than storage-only | TARGET: unestablished; synthetic passing payload exceeds exact dictionary bytes | Strong equally exposed storage/cache/retrieval comparator; setup, training, maintenance, readout and fallback |
| C5: whole-corpus completeness; stored means noticed | TARGET: untested at the program scale | Versioned corpus manifest and eligible evidence universe; report ingestion coverage, compression retention, access recall and answer exhaustiveness separately; held-out query-specific omission criteria/reference answers, rare/interacting/conflicting evidence |
| C6: reliable "not in memory", stopping and tool fallback | TARGET: untested | Absent-evidence controls, calibrated abstention, stop failures and encoded tool returns |
| C7: provenance and update/invalidation | TARGET: host pointers/versioning are components, not complete grounded answering or semantic validity | Retained source/dependency records, faithfully grounded answers and tested invalidation; origin laundering, instruction-looking evidence, event/ingestion time and decode role tests |
| C8: multiscale cheap sweeps without missed relevant details | MEASURED: bounded synthetic hierarchy recipes failed; general claim untested | Quality/cost against optimized flat scans; detail loss and coverage |
| C9: latent reasoning over memory and without memory | TARGET: mechanism open; no admitted matched-compute general advantage | At least two reasoning mechanisms, causal transitions and strong textual comparator |
| C10: one multimodal call and agent tool interface | TARGET: not established by text-only synthetic APIs | Modal evidence necessity, role/time behavior, optional-memory parity, grounded text/source returns |
| C11: build economics and pretrained reuse | TARGET: efficient incremental construction; unestablished | Separate inherited resources, failed attempts, adaptation, maintenance, human work, storage and serving; charged dependencies and complete lifetime costs |
| C12: recognized general intelligence | TARGET: mid-pack Artificial Analysis intelligence performance; unestablished | Pin comparison population/date, evaluation version and inference policy; evaluate actual artifact rather than inherit a base score |
| Scoped mathematical scaffolding | MATHEMATICAL: exact results only under their stated assumptions | A proved numerical/continuation contract does not establish learned memory, completeness or economics |

[MEASURED] Evidence is read from existing records, not rerun: [experiment record](RESULTS.md), [bounded explorer note](RESULTS.md), [explorer summary](RESULTS.md), [prefix acceptance slice](RESULTS.md).
The explorer uses familiar synthetic addresses and categorical values, no prose or pretrained language model; revision re-encodes source; restart parity covers one saved world; some tensor controls have scalar receipts only.
[MATHEMATICAL] The [evidence map](RESULTS.md) and [research program](ARCHITECTURE.md) retain each theorem's assumptions and transfer boundary; they do not prove this architecture.

## Success criterion (owner, 2026-10-17)
We do NOT have to beat plain text. It is a WIN if the latent memory can extract ALL of the same information the plain-text memory holds (same questions answered, same facts and source provenance, same completeness and "not in memory" behaviour), AND does it WITHOUT constantly encoding and decoding the text (the memory is encoded once and queried in latent space, no re-encoding or decoding of the corpus per query). Report parity to plain text (coverage and exactness, per question type) together with the per-query encode/decode cost avoided, as the headline result. Beating text on accuracy is a bonus, not required. Still never claim more than the numbers show; a tie with a stated cost advantage and full coverage is the target.


## Research program and shared foundations

## The ambition

Existing models embody a large investment in learning structure. We want to make that
structure persistent, revisable, portable, and economical to compute with. A model may
construct or interpret a representation; a smaller engine may maintain and manipulate
it; language may communicate the result. This component division is a hypothesis within the owner-directed single multimodal latent model with optional memory; it is not a separate-mode architecture.

The long-term ambition includes new mathematical primitives for high-dimensional
spaces and neural networks, new ways to build and train models, and potentially a
different computational framework for intelligence. Matrices, tensors, vectors,
graphs, programs, recurrent dynamics, and interacting or evolving model ecosystems
are candidate mechanisms. No single carrier is mandatory. A useful new model of
computation could emerge through the mathematics and constructions rather than being
declared in advance. Neither a new notation nor a different physical encoding earns
that claim: an operational model needs explicit primitives, resources, capabilities,
and limits, compared with an equally resourced conventional implementation.

Math, construction, and empirical exploration develop together. Formalizing an
operation can reveal what to build; building can reveal a missing primitive; neural
dissection can reveal why an operation fails. We do not wait for a complete new
mathematics before constructing the system, nor abandon foundational questions for
an application benchmark.

## Track 1: latent memory and long-context computation

**Question:** can a bounded active workspace preserve useful access to a growing
history, answer later questions, simulate consequences, and repair its understanding
without repeatedly processing the whole history?

The proposed object is a persistent computational memory, not one lossless vector.
Its possible layers are immutable evidence/events; local exact identities, values,
versions and source pointers; learned semantic or relational objects; and a small
active workspace. The reasoner can reopen raw evidence when compressed information
is insufficient. Hierarchical retrieval alone is a legitimate component, but does
not establish direct latent reasoning.

Memory operations to make concrete:

- **Write:** admit an observation without silently replacing existing knowledge.
- **Address and recover:** find the right entity, event, binding, or source when a
  previously minor detail becomes relevant.
- **Compute:** derive a new consequence from retained information, including a
  question or rule revealed only after the memory was built.
- **Branch and simulate:** retain alternatives and apply a declared hypothetical
  change; distinguish simulated consequences from external observations.
- **Revise:** retract or replace a premise and repair its downstream influence while
  preserving unrelated information and the historical record.
- **Compose:** combine separately prepared memories without assuming their
  coordinates or operations are interchangeable.

**Long-history computation** preserves a declared family of future questions and
actions over an expanding history, including late relevance and revision, under an
explicit access/recovery contract. **World simulation** additionally supplies a world
state, transition rule, interventions and observables, and tests rollout fidelity
over a horizon. Answering historical questions alone does not establish a simulator.

For simulation, specify reference transitions, retained-state transitions, encoding,
interpretation and observable families. Compare consequences after legal action
sequences, not only initial reconstruction. Exact trajectories, observable
distributions and approximate task consequences are different targets. Price branch
copying, merging, rejected rollouts and synchronization with external observations.

Current-view consequences and historical observations need separate contracts.
Absorbing a replaced write can be correct for the current view while an earlier-
version query intentionally distinguishes the histories. The projected current-view
actions must also factor through that view; history-sensitive rollback cannot be
silently admitted under its absorption law. Such an action retains the required
distinction or crosses a separately declared, priced recovery/refinement boundary.
For approximate rollouts,
relate representation error to the intended observables, decision margins and its
propagation over the horizon. Unobserved external transition mechanisms remain
uncertain; internal simulation does not supply missing observations.

"Infinite context" is an ambition for expandable external memory and bounded active
processing. It promises neither bounded total storage, constant worst-case query
work, nor exact fixed-size compression of arbitrary histories. Report ingestion,
storage growth, compaction, query, reopening and replay separately.

A memory result must expose late-emerging relevance, similar identities, negation,
exact numbers, superseded instructions, conflicting sources, and corrections after
substantial computation. Distinguish retained information, competent access, and
competent computation. Compare with full replay where feasible and strong caching,
retrieval, database/graph, and small-model memory systems at equivalent access.

The evidence interface can eventually include text, conversations, images, audio
and video. Cheap global descriptors, local features, exact values/timestamps and
raw-source reopening have different roles. Semantic compression does not replace
provenance or exact facts. Multimodal expansion earns its place through preserved
decision-relevant information and complete processing/reopening cost.

## Track 2: new model construction, training, and inference

**Question:** can different learned representations and layer operations produce
useful capability with less total training or execution work?

This track changes the model and learning procedure, not only the application
around an unchanged language backbone. Directions include reusable pretrained
encoders/embeddings, learned readers and updaters, recurrent or matrix memories,
conditional execution, representation changes between layers, computational
capability reuse across models, and training objectives that preserve future
operations rather than only today's answer.

### The owner's pretrained-embedding and layer-operation hypothesis

Start from a pretrained representation. Learn a new computation over that
representation, potentially at successive layers, that changes predictions and
supports token-by-token output when useful. What does each operation read, preserve,
change, and emit? Which original layers can it replace, skip, amortize, or execute
less often? Which pretrained knowledge remains accessible after doing so?

An encoder plus a newly trained decoder, an operator inserted between layers, a
recurrent latent core, and a learned route among representations are different
experiments. We must define and compare them separately. They are proposed options,
not a commitment to a particular implementation. An embedding can lose distinctions
needed for generation, and a frozen receiver may lack the operation needed to use
it. An added analysis module is not an efficiency gain if the original expensive
backbone still performs all the work.

A pretrained resource may be a token table, contextual encoder, selected layer
interface or connected donor. Specify sequence positions, masks, positional
information and retained KV state. A final vector and a full contextual sequence
are different resources. Study two distinct behavior contracts:

- **Continuation-preserving replacement:** reproduce declared existing behavior
  through a cheaper continuation, identifying the actual layers/passes removed.
- **Operation-adding construction:** learn a useful new action over the resource,
  testing held-out rules, queries and composition rather than only a taught decoder.

The new analysis reads a declared representation, persistent state and action, then
emits a changed representation/state and possibly a route. For token generation,
define legal inputs and state updates at each token, free-running continuation,
termination and recovery. Teacher-forced prediction alone does not establish stable
generation. Compare competent direct receivers and equally exposed learners; test
whether the resource supplies reusable knowledge or the receiver learns afresh.

Trainable representations should be judged by held-out consequences, bindings,
new rules, composition and correction. Compare frozen and adapted interfaces, and
attribute which component supplies the capability. Charge data preparation,
pretraining dependencies, incremental training, optimization state, distillation,
search, preprocessing, fallback, and deployment execution. Report both cold-start
and repeated-use economics; do not hide construction behind amortization.

Training-dynamics hypotheses from the small-network lab remain scoped to the
tested architectures, optimizers, data and interventions. Cross optimizers,
architecture, loss and scale when a mechanism claim requires it. Dose and effective
update sizes must be matched before interpreting an optimizer difference.

For broader latent-world systems, construction includes deciding what an unfamiliar
problem needs: internal computation, a new intermediate representation, external
evidence, a consequential clarification or a conditional answer. Missing observations
and unspecified objectives cannot be computed away. Train and evaluate causal
investigation episodes and same-prefix changed-evidence continuations without future
information leakage or a task-specific agenda supplied at inference. Compress
execution after capability is demonstrated, not in place of demonstrating it.
These are training/control hypotheses from the larger vision, not an adopted
architecture, funded curriculum or claim of self-formulation.

## Shared foundations, one selection process

| Foundation | What it contributes to both tracks |
|---|---|
| High-dimensional mathematics | Interference, concentration, finite precision, approximation, operational distinguishability, and failure boundaries |
| Mathematics of neural networks and dissection | Causal read/write interfaces, routing, distributed computation, training dynamics, forgetting and repair |
| Matrix-native and representation-changing computation | Structured operators, composition closure and growth, conversion costs, precision/error propagation, and representation selection |
| Capability interoperability | Transfer of useful distinctions and operations, competent receiving interfaces, jointly retained capability, and donor-removal costs |
| Computational-model and resource theory | Explicit step semantics, information and resource limits, equal-access translations, and tests for genuinely new computational advantage |

These remain substantive research areas. They share the current mission and
experiment selection; they are not renamed silos with independent queues.
Pure mathematical work is appropriate when it states its connection to an
operation, representation, limit or constructive escape relevant to the program.
Exploration can also uncover connections not known in advance.

That connection may be prospective, or the question may have independent
foundational importance. It need not promise an immediate deployment benefit.

The unification preserves the full foundational ambition: arithmetic with
structured objects as primitives, economical composition and representation
changes, and the invention of a computational model with substantive capability
and limit theorems. A theorem about general matrix computation can matter beyond
AI; a new computational model need not begin as a neural architecture. The two
tracks organize current construction priorities without reducing those ambitions
to engineering a better chatbot.

## Working objects and bridge obligations

These are provisional operational definitions, not novelty claims.

| Object | Required declaration |
|---|---|
| State | What persists: parameters, activations, KV entries, external records, operator descriptors, or a hybrid; all auxiliary retained resources |
| Memory | Which observations and later operations remain available, for how long, at what tolerance and access cost |
| Operation | Permitted input/action, state change, observable consequence, and implementation cost |
| Future | A legal sequence of actions and queries, including questions not known when state was constructed |
| Representation change | Encoding/decoding map, preserved operations, approximation, precision, and conversion cost |
| Transfer | A source computation or distinction made usable through a competent receiving interface |
| Connected growth | Capability reuse with a retained donor/service; ongoing calls and storage remain charged |
| Assimilation | Declared useful behavior after donor removal, with acquisition and retention costs measured |
| Simulation | Specified world state, transition rules and observables, with fidelity/error over a declared horizon |
| Computational model | Machine class, primitive steps, access/precision/resource contracts, capabilities and limits |

The lifetimes of those resources also matter:

| State lifetime | Complete object to account for | What matching it does not establish |
|---|---|---|
| Learning/adaptation | Parameters, optimizer moments/buffers, schedule, random/draw state, objective/masks and teaching/replay resources | Matching weights or a readout need not match the next learning transition |
| Runtime | Active/recurrent state, KV, addresses, operator descriptors, branches, randomness and callable components | Matching today's answer need not preserve the next action or a newly selected query |
| Evidence/recovery | Source identities, versioned observations, original values, histories, baselines and recovery material | Matching current beliefs need not preserve historical observations or pay recovery costs |

These may be physically colocated. A training buffer becomes a runtime resource
only if the proposed machine actually uses it there. The distinction prevents a
weight-space maintenance result from silently becoming an activation-memory law.

Each cross-foundation proposal identifies the source object, destination object,
map, operation that should be preserved, missing control, and falsifying outcome.
For example, a matrix descriptor-growth theorem does not become a memory-capacity
law until a state/update interface satisfies its premises. Weight-memory transport
does not become activation-update failure without a discriminating experiment.
Shared terminology or the same square-root scale is insufficient.

## Representation changes and evolving systems

**Representation-changing computation:** an algorithm may retain a computation in
one form, choose another because the next operation becomes cheaper, convert,
execute, and continue with a preserved-operation/error contract. Search both among
representations of a fixed algorithm and among equivalent algorithms with different
internal structures. The choice policy, cost estimation, failed conversions,
precision and final output all count. A fixed-graph transform optimizer is a useful
restricted instance, not a compiler for arbitrary learned computation.

**Evolving computational ecosystems, open hypothesis:** a growing system may
introduce, specialize, adapt, retire or replace interacting components while
preserving useful joint operations. Define variation and selection/adaptation,
component interfaces, shared state, objectives and who provides supervision. Adding
a donor to a router is connected growth, not autonomous evolution. Ask whether local
adaptation produces reusable joint computation without recurrent full-system
retraining or unpriced human/teacher work. Compare equally resourced mixtures,
routers, incremental learners and retained donors. Selection improving its own
proxy while joint capability or retained operations fail is a falsifying outcome.

These are serious research questions, not established ecosystems or new queues.

Standalone state-transfer claims require source removal. Connected systems may
remain useful; retain and account for their donor resources rather than enforcing
removal as a universal architectural restriction. Useful neural architectures
need not beat ordinary computability; a claimed new computational-model advantage
does need a precise equally resourced comparator.

## The native-mathematics question

Take the learned space as a world. What would a denizen need to invent to identify
the same place, make a move, measure effort, keep a map, and predict consequences?
Which distinctions must survive a new question or update? Which operations compose?

PCA, cosine, Euclidean norms, spectra and ordinary linear algebra are welcome tools.
Their successful application does not by itself establish native mathematical
structure. An operational vocabulary earns its place by predicting causal behavior,
useful composition, or a cheaper computation at a stated tolerance. Ask whether the
measurement imposes the structure it claims to discover. Proven holes constrain the
next representation or architecture; a failed interface is not a proof that all
latent reasoning is impossible.

## What the inherited evidence permits

Read [RESULTS](RESULTS.md) and its source correction qualifications before reusing inherited evidence.

- LSR has conditional continuation/overwrite mathematics and scoped high-dimensional
  interference results. Its latest compressed-state recipes did not establish
  selective, update-closed reasoning; operational width scaling remains untested.
  Readable closing states and local value interventions are not a portable memory.
- Neural dissection supplies concrete training/readout hypotheses and instruments.
  Its small-network observations do not establish universal laws or arbitrary
  capability transfer. Its optimizer-cross experiment is partial and remains paused.
- Matrix-native work supplies composition/descriptor/precision constraints and
  constructions. Known structured methods and engineering speedups do not establish
  a new general matrix arithmetic; specialist baselines materially changed claims.
- Computation-model work supplies scoped erasure, endurance and lease mathematics
  plus a large negative translation record. It establishes no new computational
  paradigm or verified advantage over an equally resourced standard model.

None of these conclusions changes merely because the repositories are unified.
Their combination motivates new hypotheses, not a composite empirical result.



## Navigator design contracts

Accepted design only; runtime qualification, economic success and launch authority remain separate. Full source notes are preserved in the external recovery archive described in [RESULTS](RESULTS.md).

## Intended object and evidence boundary

The destination remains one multimodal latent reasoner with memory optional in the same call: encode the whole corpus once using generic pretrained parts, reason over retained representations, and recover verified source evidence. Completeness, reliable absence, revision, provenance, multiscale access, useful latent computation and full-cost economics are separate claims. A navigation tool can advance the program without establishing the full model. Preserve both reused-block recurrent workspace and growing continuous hidden-state feedback as distinct reasoning alternatives; graph lookahead is not a substitute for comparing them.

The durable tool contract is `context(question, corpus, snapshot, intent, budget, readings=[], continuation=None) -> ContextPack`, served by the same warm backend through package, CLI and MCP. The product baseline is a regular unrestricted competent agent with its normal files/search/shell tools. Current owner requirements additionally call for traditional dense top-k RAG as an external baseline. Stateful remains a later integration opportunity, not a dependency or current test harness. No UI is required.

## Accepted construction design, not a claim of current implementation

The final architecture is:

`immutable native-source register -> occurrence/container-scoped incidences -> replaceable frozen adapters -> bounded compatible-binding workspace -> exact-source packer`.

At approximately 1M tokens, begin with flat coarse access, independent lexical/structural/semantic routes and ordinary streaming joins. Keep genuine containment and references, but do not make hierarchy an exclusive gate. A secondary entity layer is a source-witnessed evidence graph, not a complete knowledge graph. A witnessed assertion need not be true or correctly interpreted. Preserve occurrence and container scopes, reversible alias hypotheses, arguments, polarity, qualifiers, time, versions, permissions, dependency identities and explicit unknowns. Co-occurrence and geometry propose access; they do not establish semantic support.

The core is modality-neutral. Source identities and typed native coordinates remain independent of text/code/table/image/audio adapters; unsupported objects stay inventoried and explicitly unsearched semantically. Keep unaligned feature spaces separate. Encoder windows differ from coherent delivery units; account for actual truncation and preserve native mappings/fallback. Structural inclusion is not modal understanding. A source hash proves byte identity, not relevance, truth, authority, relation entailment or completeness.

The bounded workspace retains snapshot/auth epoch, original question, at most eight readings, eight needs per reading and 128 compatible tuples per reading; cached features/scores, frontier/cursors, proposed and verified witnesses, verified buffers, budget and overflow. A transition extends one compatible assignment. Utility concerns needs covered within ONE feasible tuple; independent slot maxima cannot establish a compatible answer. Original questions remain accessible; inferred facets cannot hard-prune. Actions include open/zoom, follow a witnessed reference, enumerate, rebind and probe an independent region. Lookahead uses retained information only; actual verified source opens alone add observations. Numerical scores are priorities, not probabilities.

Design caps are 2,048 transitions, 4,096 charged compatibility probes per decision, 64 unique opens, 64 MiB verified reads, 6,000 delivered tokens and a five-second navigation deadline. Optional depth-two lookahead costs 32 + 32*4 = 160 transitions per complete decision, permitting at most 12 such decisions and 49,152 probes at the per-decision maximum. Charge attempted join pairs before allocation; do not spend one transition allowance twice. These bounds are not throughput measurements or an end-to-end time guarantee.

One call returns readable exact contexts/native pointers, mechanical access reasons, unresolved/conflicting bindings, separate inventory/adapter/search coverage, overflow, stop reason, costs and resumable search state. Preserve complete tables, qualifiers and both parents of necessary joins. Share source reads while keeping interpretations separate. Keep delivery paging distinct from search refinement: a paging cursor drains previously selected groups; it does not reroute. Bind search handles to principal/auth epoch, snapshot, original query, policy/adapter revisions and source dependencies; reject stale state rather than silently rebase.

Budget/frontier exhaustion, low entropy, complete postings, convergence or stalled improvement never certify unrestricted semantic absence. A bounded negative needs an explicit eligible universe, validated membership recognizer, complete traversal and scoped claim. Keep ingestion coverage, adapter processing, discovered supports, delivered supports, semantic reading coverage and answer quality separate.

## Delivery and resource contracts that must survive consolidation

- Preserve verified anchors and fitting seeds when a bridge fails. Keep necessary parent groups atomic; explicitly omit oversized groups instead of clipping them into apparent completeness. Once access stops, use only previously verified cached evidence.
- Deliver an immutable verified buffer, never reconstruct quoted bytes from a mutable path. Check snapshot/current authority and authorization at a serialized release commit. Committed historical evidence is not a promise of latest authority.
- Authorize before candidate generation, counts, descriptors and cache lookup. Bind handles to principal/auth epoch and serialize release with revocation. Quotes, paths, counts, exports and errors must not reveal denied objects. Shared deployment remains gated; zero unauthorized disclosure is accepted.
- First slice admits one build/query worker, with exclusive ownership, bounded queue/cancellation and owner-checked cleanup; do not steal uncertain live leases. Test process death and cancellation. Measure cold build, warm query and snapshot-swap peaks, charging FP32 backing, originals, strings/indexes/runtime, copies and overlapping old/new snapshots. Preserve the standing 12-GiB free-RAM floor; sampled observations cannot guarantee continuous compliance.

These are agreed mitigations and mandatory qualification gates, not already passed production guarantees. The 32,033,560-byte 1M subtotal plus FP32 is conditional arithmetic before variable overhead, not measured deployment memory.

## Decision-changing construction and controls

Build in the existing modules: qualify delivery/fallback first; add immutable bound delivery tickets and separate search sessions; implement ordinary compatible joins and greedy scheduling; then test optional lookahead. No mandatory reader, generative answerer, scorer, per-corpus training, curator, new service or immediate hierarchy rebuild. Corrected source expansion remains selectable.

Use three identical-information component arms: H = corrected expansion, J = conventional compatible-binding joins/greedy, S = same-map lookahead. Equalize sources, seeds, features, actions, packer and budgets. H->J tests representation; J->S tests lookahead. The four public DEV cases proposed in the final design are Datadog 011/012/014/016, conditional on evaluator-owner confirmation that alternative complete supports fit. These are selection/falsification cases, not confirmation. S earns admission only if it adds at least one complete support set without losing another OR reduces verified-read work at least 25% without support loss, AND total measured navigator cost does not rise. Otherwise leave S off and retain cheaper H/J.

Preserve controls that can change the decision: remove either necessary witness, swap a binding, restore; remove/shuffle numerical features while preserving incidences; move the sole discriminator beyond lookahead depth; test the discarded 129th binding, quadratic joins and an incomplete membership recognizer. A gain surviving feature removal belongs to structure/access, not latent computation. Test nested aliases in one file, negation across window seams, unit/quantifier/role swaps, limited amendments retaining unaffected assertions and a genuinely necessary translated/image/table witness through the SAME core. Qualify corruption, revocation during paging/commit, snapshot replacement, admission/death/cancellation and cold/warm/swap resource peaks on actual operations.

Measure complete agent trajectories to establish substitution: complete compatible dated bundles and scoped enumeration registers should remove later discovery/reread decisions. A same-evidence crossover compares existing JSON against readable/compact delivery under the same consumer policy and normal verification tools; score supported answers/citations, provider input, turns, source rereads and complete cost. Smaller local payloads alone do not prove savings. Charge build/update, query encoding, checks, rejected work, retries, serialization, transport, client waiting and fallback. With added setup F and recurring saving delta>0, break-even requires Q>F/delta; delta<=0 cannot be amortized away.

Separately, a meaningful latent-reasoning comparison requires a competent receiver and task headroom beyond no-step, identical evidence/supervision/readout and matched complete compute across fixed recurrent workspace, growing feedback and text. Remove/swap/restore midpoint state and operators; retain a latent-removed side-channel control. Charge growing history, attention, KV, readout and fallback. Intervention sensitivity without useful consequences is insufficient; exact verifier/executor success belongs to that conventional component.

## Evaluation boundary

The [evaluation protocol](CONTEXT_TOOL.md) is canonical for the proposed A/V/B confirmation and acceptance gates. No benchmark launch is authorized by design closure.

## Durable lessons and live opportunities

Historical lessons to retain with their bounded evidence: source-bound bridges can improve delivered support without establishing a uniquely latent mechanism; wider branching/geometry can cost more without improving the pack; pooled marginals lose bindings; exact side channels can explain apparent latent successes; native ordered KV reuse preserves both computation and reader errors; independent-cache positional repair cannot recreate missing cross-passage dependencies; aggregate accuracy ties can conceal individual lost correct answers. Qualify readers on present AND nonempty-insufficient evidence, and distinguish formatting/type correctness from relation support.

Retain the 15 open design proposals as opportunities, not results or a queue. Highest immediate value: per-turn substitution/cost attribution; identical-evidence readable/compact delivery; entity/discriminator/asserted-link census with wrong-binding controls; paired present/nonempty-absent gates; declared search-scope replay; paired routing-compression answer/loss roster; volunteered reading/event-phase scoring; equal-metadata routes with flat bypass; naturally answerable cross-document/no-rare-handle questions; person/series/effective-date edition chains; scoped-relation/boilerplate filtering; and descriptor-to-shard routing versus BM25/lexical centroids. The retained folding/template-denominator studies are methodological opportunities; the candidate-window crossover is conditional, not a measured admission boundary.

Keep corrected source details if examples are carried forward: filing date differs from ownership-effective date (the November 2024 example uses September 30, so universal December is false); newest values require issuer/person/series scope; duplicate clauses are not independent witnesses; the old 75-KB/14.7-MB KV claim confused text size with 75 tokens; the acquisition disclosure is approximately $260M cash AND stock with about 25% deferred, not an exact $195M/$65M cash split. Prior co-founder attribution, universal-date rule and unrun 65% savings must not return through condensation.

Do not merge different evidence snapshots: the initial pilot and later Stage 1 are separate protocols. The historical pilot "tie" wording overstates what its interval can show: [-4.63,+1.39] points does not establish a -3-point noninferiority gate. Stage 1's later cost regression is a separate reported result, not the original +66.5% pilot. Hypothesized JSON/history attribution, net savings, useful H/J/S, modal retention, semantic completeness/absence and recurrent-versus-feedback benefit remain unestablished.


## Existing methods to adopt and distinguish

The durable finding of the 44-family scan is that encode-once reuse, retrieval, cache reuse, structured memory and latent reasoning already have different established mechanisms; none of the inspected sources establishes LSR's complete joint contract. Use retained BM25, generic dense retrieval and rank fusion as components, give the text comparator the same reranker, graph, iterative access and prefix caching, and compare consumed calls/tokens only at qualified answer quality. Cached text retrieval also performs zero corpus re-encodes per query.

- **Native KV / [CAG](https://arxiv.org/abs/2412.15605):** conventional same-reader prefill reuse for a prefix that fits context and KV budgets. [CacheBlend](https://arxiv.org/abs/2405.16444) composes cached chunks with selective recomputation; reuse is not zero work. Platform integration and invalidation remain costs.
- **[ColBERTv2](https://arxiv.org/abs/2112.01488)/[PLAID](https://arxiv.org/abs/2205.09707):** strong late-interaction retrieval comparator; the historical local exhaustive MaxSim adapter was not PLAID. [FAISS](https://arxiv.org/abs/2401.08281), DiskANN and BM25 provide existing search/indexing machinery before custom structures are justified.
- **Graph/temporal/multiscale comparators:** [HippoRAG 2](https://arxiv.org/abs/2502.14802) for multihop retrieval, [Graphiti/Zep](https://arxiv.org/abs/2501.13956) for validity/invalidation/history, [RAPTOR](https://arxiv.org/abs/2401.18059) for hierarchical summaries, and [Hindsight](https://arxiv.org/abs/2512.12818)/[SimpleMem](https://arxiv.org/abs/2601.02553) for fused agent-memory workflows. Their extracted facts/summaries can omit information; APIs for update/delete do not establish selective dependent-answer repair. Eligible local substitutions and matched local quality were untested.
- **Compression research:** [Cartridges](https://arxiv.org/abs/2506.06266) uses corpus-specific training; [Attention Matching](https://arxiv.org/abs/2602.16284) fits attention outputs and mass. Published compression ratios are task/configuration results, not arbitrary-fact parity, complete physical-memory savings or complete lifecycle economics. Surveyed checkpoints/configurations were not qualified by matched local tests; the legacy reader interface is unsupported and withdrawn. Code, checkpoint and data rights remain separate.
- **Two reasoning alternatives:** [Coconut](https://arxiv.org/abs/2412.06769) feeds hidden outputs back as embeddings; [Huginn/recurrent depth](https://arxiv.org/abs/2502.05171) reuses recurrent blocks. These are distinct from memory-port choices. Their useful matched-compute comparison in LSR remains unrun.

## Durable design cautions and open hypotheses

**Identity and semantics are separate.** The old compressed consumer reproduced 100/100 selections and 1,000/1,000 span checks, yet retrieval remained incomplete. Content deduplication had discarded duplicate source-instance identity; later pointers cannot reconstruct lost lineage. Preserve source instance, body version, span and dependencies separately from shared vectors. SHA256 authenticates chosen bytes, not derivation, source/index alignment, truth or current authority. A caller-approved same-count BM25 archive with permuted rows is not detected by the ladder adapter.

**Source-access software has real charges.** Reader-packet setup hashes the full packed file; each proposed leaf verification rereads its full source body, including repeated proposals. Prepared tokens are not consumed model tokens. The latent front-end's packed8/refinement path retains and charges the full FP32 bank; it is not a total-storage compression result. The retained BM25 provider allocates a full score vector and sorts the rung, cannot open `source:<filename>`, and is not yet a complete strong text harness. Incompatible later consumer versions are not qualified by older fake-reader tests.

**No retrieval is not no information.** Mixed near-miss labels invalidated the old semantic-absence interpretation. Keep `UNKNOWN` separate from scoped `ABSENT_WITHIN(universe, criterion, version)`; a full vector sweep or literal scan does not certify semantic absence. Retain competing readings and conflicting sources instead of collapsing them into one exact-match answer.

**Neuroscience supplies fallible design hypotheses.** Cue/indexing, pattern separation/completion, metamemory, gist plus protected exceptions, multiscale relational maps and surprise-guided access motivate identity-safe proposals and coverage-aware stopping. Biological familiarity can be wrong; predictable dates, negations and ordinary details may be essential. Much of the literature review used indexed abstracts/excerpts, not full-text methods/effect-size verification. Representative original records are [indexing theory](https://pubmed.ncbi.nlm.nih.gov/3008780/), [feeling-of-knowing](https://pubmed.ncbi.nlm.nih.gov/5825050/), [conceptual gridlike codes](https://pmc.ncbi.nlm.nih.gov/articles/PMC5248972/) and [prediction error and learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC5381756/). They do not prove a machine-memory algorithm.

**Continue the central program, stop multiplying access apparatus.** The decision-changing construction remains a useful source-sensitive retained consequence, with late selection, missing-premise/binding/null interventions, answer-changing revision/current/history/unrelated checks, and an operation-competent equally exposed cached-text/KV/direct-execution comparator. General reader perfection is not a prerequisite. Completeness, calibrated absence, plural exploration, useful latent computation and full costs remain separate unestablished claims.
