# Post-Paper Research Programs — IDEATOR, 2026-09-26 (CPU-only thinking)

Inputs: DAY_THREE_REPORT.md, scratch/novelty_inventory.md,
scratch/frontier_research_20260925.md, THINKING.md T029-T036, the P-A paper
draft, QUEUE.md. Premise: the expression-gap paper is submission-ready; what
follows must be PROGRAMS (5-15 experiments each with phases and kill
criteria), not single follow-ups. Every first experiment is eval-only /
zero-GPU on existing checkpoints or a ≤0.84M training cell; nothing leaves
the lab envelope (≤100M, prefer ≤1-10M, single steps ≤30 min).

**Decision summary:** three programs proposed — P1 COORDINATE (top pick,
immediate start, eval-only), P2 IMMUNOLOGY (constructive turn on the frozen
basis, starts from the queued e060/e062/e063 ramp), P3 CACHE WEATHER (the
deciders are already registered; the provenance split is the sleeper hit).
One wildcard sketched: P4 THE ERASER — the paper's own sequel (suppression
mechanism), held back only because circuit-level work in char transformers
is the highest-variance bet of the four.

---

## P1. COORDINATE — the positional-binding program (TOP PICK)

### Organizing question

**Is position the indexing substrate of knowledge — and can the address
space be censused, predicted, moved, and rebound?** If a fact is an address
(T035: "the knowledge is an address, not a steering direction") bound to a
positional coordinate (T032: wpe-130, one-char knife-edge 0.556→0.12), then
the next questions are about the coordinate system itself: how many
addresses exist, what is their geometry, whether they are portable rows or
distributed keys, whether NATURAL (trained) knowledge binds the same way,
and whether the long-context cliff is the same knife-edge at scale.

### Why NOW

- T032/T035 made coordinates first-class objects. Before the one-char-shift
  probe, content-vs-position was an open confound (novelty inventory #2);
  now it is a measured dissociation with an instrument (the knife-edge) we
  own.
- e056b left a loaded gun on the table: the **mean-donor relay direction**
  beats every individual donor (d5 0.912 vs 0.494) — averaging denoises
  toward a canonical ADDRESS DIRECTION in state space. Nobody has asked
  whether that direction IS the wpe row lifted into the residual stream.
  Closing that loop (position row ↔ state direction) would upgrade the
  paper's phenomenology into a mechanism sketch.
- The frontier scan (§1d) states verbatim: "No causal component-patching
  study separating positional vs content subspace failure exists at any
  scale." NoLiMa/Context-Rot are behavioral; we have causal tools and a
  measured knife-edge. Candidate B (the cliff) is this program's phase 3.
- The paper's biggest admitted limitation is single-family,
  install-protocol specificity. The natural-knowledge leg (phase 2) is the
  external-validity test, and it reuses the e049 retrieval corpus already
  in `runs/`.

### Program arc (~11 experiments)

- **Phase 1 — census (3, all zero-GPU):** close the loop (relay direction
  vs wpe row); address census by single-row perturbation; address
  prediction from corpus position statistics (our corpora are ours —
  where a fact sat at train time is known; can we predict where it will
  bind?).
- **Phase 2 — surgery + natural knowledge (4):** rebinding by row copy;
  renumbering/permutation tolerance of whole context windows; shift-vs-
  substitute binding probe on the NATURAL retrieval circuit (e049 corpus);
  address generality across fact families (names vs refrains vs syntax).
- **Phase 3 — the cliff link (2-3):** train KV-recall at ctx-512, sweep
  eval 512→2048, patch positional vs content components of the retrieval
  path (scan candidate B, registered prediction already drafted); does the
  knife-edge predict the cliff location?
- **Phase 4 — training-time (2):** position-jitter during install —
  dose-response of binding breadth (can installs be made position-robust?
  the cheapest possible "fix" for elicitation failure); wpe-row transplant
  across seeds (a one-row organ — the minimal graft, and the natural
  bridge experiment into P2).

### First 3 experiments

1. **e065 — close the loop (zero-GPU, minutes):** compute
   cos(e056b mean-donor relay direction at d5, wpe-130 row / its projected
   stream component) from `scratch/e055_traj_cache.pt` — registered:
   |cos| ≥ 0.4 closes the position-row→state-direction loop; ≈0 means two
   distinct objects both called "address."
2. **e066 — address census (eval-only, 0.84M + 2.7M install family):**
   single-row wpe perturbation sweep × install battery — registered:
   sparse code (≤5 rows carry ≥80% of the collapse) vs dense entangled
   code; is wpe-130 alone or the peak of an address cluster?
3. **e067 — rebinding surgery (eval-only):** copy wpe-130's row to position
   k (and two-row swaps) — durable expression at the new position ⇒ the
   address is a portable row; census-sparsity + rebinding-failure ⇒ a
   distributed key grown in downstream weights, and "move the address"
   requires weight surgery, not row surgery.

### Paper / workshop / dead end

- **Main-conference paper:** causal account of positional-vs-content
  indexing spanning INSTALLED and NATURAL knowledge + a working rebind or
  renumbering that moves function + the cliff link ("the addressing
  substrate of a language model"). This is the scan's §1d gap filled with
  our own instruments, and it upgrades the expression-gap paper from a
  case study to the first panel of a theory.
- **Workshop paper:** census + rebinding on installed knowledge only, or
  the cliff link without the natural-knowledge leg (single-family caveat
  survives).
- **Dead end:** dense entangled wpe code (no row is individually
  load-bearing) AND natural retrieval is content-bound — then
  position-binding is an artifact of the install protocol. Salvage: the
  negative + the knife-edge methods note. Kill signal: e066 shows
  distributed code AND e069 (shift-vs-substitute on retrieval) shows
  content dominance in ≥2 nets.

### Visualization (VISUALIZER thread)

**v013 "Address atlas":** a wpe matrix heatmap (position × dimension) with
per-fact causal-load overlays from e066, annotated with training-corpus
geometry (where each fact family lived at train time); side panel: ridgeline
plot of knife-edge curves (p(Z) vs shift size) per fact family. One image:
the address space, its sparsity, and its provenance.

---

## P2. IMMUNOLOGY — the organ-reliance / universal-donor program

### Organizing question

**What makes two networks representationally compatible — and can
compatibility be ENGINEERED (universal donors, tolerant hosts, crossmatch
prediction) rather than inherited (same init)?** The three-run chain
(T024-T027) established the negative: the stream basis is written once at
init and selection cannot see it through the organ-load dominant
(r=0.807, T026). The open door is the constructive one nobody has tried:
bypass selection and engineer compatibility directly.

### Why NOW

- T026 decomposed the assay: "compatibility" was a black box; now we know
  the phenotype is host-side organ criticality with a real L2-localized
  geometry residue (r=0.747). For the first time we can select, predict,
  and intervene on the RIGHT trait.
- T029's dissociation (criticality ≠ basis-specificity: L0 is the most
  vital organ at 2.7M yet a foreign L0 plugs in cleanly) is exactly the
  separation a "transplant medicine" needs: vitality and rejection are
  different axes. No universality/LMC paper has this dissociation
  causally (scan §3 verdict: the lane is empty).
- The applied hook is hot: model merging / soup / re-Basin-style
  permutation alignment is engineering-first with no causal account of
  WHY permutations help and what they cannot fix. Our framework predicts:
  permutations fix interface alignment but cannot fix organ-reliance
  mismatch — a registerable, falsifiable prediction in the merging
  literature's own terms.
- The entry ramp is ALREADY QUEUED (e060/e062/e063) — the program is a
  reframing of parked single experiments into a coherent arc, which is
  exactly what the parking-lot promotion policy asks for.

### Program arc (~11 experiments)

- **Phase 1 — the ramp (3, queued):** e060 (A-residualized selection
  lineage — the evolvability verdict on the correct trait), e062
  (subspace-cosine predictor), e063 (organ-load setpoint/homeostasis) +
  the second lineage/donor replication P-B already demands.
- **Phase 2 — constructive medicine (4):** trained tolerance (host-only
  fine-tune under graft; tolerance specificity: same donor vs seed family
  vs unrelated — learned remapping or global dampening?); donor
  conditioning (pre-align the donor interface with a cheap transform,
  organ computation fixed — splits interface-mismatch from
  host-criticality rejection); immunosuppression (downweight the host's
  interference-peak site during graft); the healed-host contrast (e056's
  fragility axis).
- **Phase 3 — the crossmatch matrix (2):** 6×6 host×donor grid with
  residualized damage + predictor head-to-head — is there a cheap
  pre-graft compatibility test (the practical deliverable)?
- **Phase 4 — the merging bridge (2):** does the crossmatch predictor
  predict weight-averaging merge quality in-family? Post-permutation
  residual damage tracks organ-reliance, not basis distance (the
  registered merging-literature prediction).

### First 3 experiments

1. **e060 (registered, queued):** lineage selection on A-residualized
   graft damage — if even the correct trait fails to move alignment,
   FROZEN is complete and the program pivots fully to the constructive
   leg; if it moves, compatibility is evolvable and everything ignites.
2. **e068 — trained tolerance (0.84M, short fine-tunes ≤30 min):** graft a
   foreign L-organ, fine-tune HOST weights only at small LR; measure
   rejection decay + specificity across donor relatedness — registered:
   tolerance is donor/seed-specific (learned remapping) rather than
   uniform (dampening).
3. **e069 — crossmatch grid (eval + grafts, no training):** pre-graft
   stream-cosines per site → predict A-residualized damage across a
   6-host × 6-donor matrix; a cheap compatibility test with real
   predictive power (registered: pooled r ≥ 0.5 held-out) is itself a
   result the merging literature lacks.

### Paper / workshop / dead end

- **Main-conference paper (crossover):** "representational immunology" —
  the causal compatibility account (organ-reliance dominant + geometry
  residue + criticality≠basis-specificity) + an engineered
  tolerance/donor recipe that beats naive grafting + the crossmatch
  predictor + the evolvability verdict. Audience: mech-interp AND the
  model-merging line.
- **Workshop paper:** the negative chain alone (FROZEN complete at full
  strength) or predictor correlation without any constructive win —
  publishable (scan: "publishable as-is") but modest.
- **Dead end:** residual selection fails AND tolerance is trivial
  (indistinguishable from ordinary retraining) AND the predictor flops
  (r < 0.3 held-out) AND the L2 residue fails to replicate in lineage 2 —
  four independent kill signals, each cheap to check in phase 1. Salvage
  in the worst case: the assay-method note (graft damage reads
  organ-load) is a citable methodological corrective.

### Visualization

**v014 "Crossmatch table":** a host × donor crossmatch grid (residualized
damage as color), rows/columns ordered by seed-family dendrogram, flanked
by per-host organ-reliance bars and interference-peak site markers. The
picture of the claim: rejection clusters by host (columns), not donor
(rows) — and the one donor column that beats the trend is the universal
donor, if it exists.

---

## P3. CACHE WEATHER — the negative-utility / pruning-frontier program

### Organizing question

**Why does a model maintain cached state that actively harms it — and how
much of a KV cache can be causally deleted or reweighted for GAIN?**
e053/e053b/T031 established: last-~7-token spike + shoulder + near-zero
plateau; sink dead at generation (5/5); 13-32% of old positions IMPROVE
when lesioned. The program asks the mechanism (why junk persists), the
provenance (whose tokens are the junk), and the exploit (the causal
pruning/compression frontier).

### Why NOW

- The curve exists nowhere else at any scale (scan §1a), and it is
  RECONCILED but instrument-fragile — the two registered deciders
  (ctx-512; more sequences) are cheap, already specified, and gate every
  downstream claim.
- T031's reframe is a thesis waiting for its experiments: "the win isn't
  finding the useful old entries — it's that almost nothing old is
  useful." Nobody has tested whether ACTING on that (oracle pruning,
  reweighting) improves generation. If it does even at char scale, the
  claim "13-32% of your context is fighting you — and here is the causal
  fix" lands directly in the long-context serving conversation
  (AgentKV/Leyline line reports edited caches accumulating error, only
  behaviorally).
- The sleeper: the provenance split. If negative-utility entries are
  disproportionately the model's OWN generated tokens aging out, then
  cache junk = accumulated exposure bias — unifying the cache thread with
  the expression-gap thread (the lab's two best assets) under one
  mechanism: free-run drift poisons the cache that free-run must read.

### Program arc (~11 experiments)

- **Phase 1 — deciders (3, registered):** the ctx-512 cell (absolute vs
  proportional onset + the instrument conflict); n-scale-up (≥16
  sequences, CIs); sequential-vs-one-shot pruning gap (does the one-shot
  lesion delta predict utility under the model's own continued
  dynamics?).
- **Phase 2 — mechanism (4):** dilution-vs-poison repair split
  (attention-renorm vs content-neutralization per negative-utility
  entry); K-drop/V-zero dissociation mapped across the full age axis; the
  provenance split (prompt-provided vs self-generated entries' utility-vs
  -age curves); junk genealogy (do today's negative-utility entries
  become tomorrow's plateau, i.e., is hurt a transient of middle age?).
- **Phase 3 — exploit (3):** oracle pruning frontier vs recency and
  sink-preserve baselines (registered bar: ≥0.02 nats/char improvement at
  ≥50% cache reduction); reweighting (attention temperature on the old
  block) vs deletion; shoulder compression (can the shoulder+plateau be
  replaced by a summary vector — cache distillation — without loss?).
- **Phase 4 — training interaction (2):** which instrument lies (resolve
  the onset-direction conflict at scale); cache-lesion regularization
  during training — can the useful window be GROWN (robustness-to-prunin
  as a training objective)?

### First 3 experiments

1. **e070 — ctx-512 decider (registered):** the timeline at ctx-512 with
   n≥16 sequences — settles absolute-vs-proportional onset, tests both
   instruments, doubles the curve's scope. Gate for everything else.
2. **e071 — dilution vs poison (eval-only):** for each negative-utility
   entry, repair by attention-renormalization (kill the mass, keep
   content elsewhere) vs content-neutralization (keep the mass, kill the
   payload) — registered: renorm recovers most (mass-stealers) with a
   poison minority; yields the two-type junk taxonomy.
3. **e072 — provenance split (eval-only, existing trajectories):**
   utility-vs-age curves split by entry provenance (prompt-provided vs
   model-generated) — registered: self-generated entries decay faster and
   dominate the negative-utility set; if confirmed, cache junk is
   accumulated exposure bias.

### Paper / workshop / dead end

- **Main-conference paper (inference-flavored):** the causal curve at 2-3
  context lengths + junk taxonomy + the provenance law + an oracle-pruned
  generation WIN. The provenance result alone, if clean, is the
  mech-interp version of "exposure bias compounds in the cache" — a
  bridge claim two literatures would each cite.
- **Workshop paper:** shape + taxonomy + sink verdict with no actionable
  pruning win, or pruning wins that vanish under sequential pruning.
- **Dead end:** oracle pruning loses CE at all rates in ≥2 nets (utility
  is non-stationary; the one-shot lesion delta predicts nothing about
  sustained pruning) AND ctx-512 deepens rather than settles the
  instrument conflict — then only the shape survives as a workshop note.
  Salvage either way: the sink-dead-at-generation result is permanent.

### Visualization

**v015 "Cache weather map":** generation-step × cache-age heatmap of
per-entry causal utility, negative-utility cells hatched in a second hue,
live-window boundary as a contour line, sink row marked, provenance
(prompt vs self-generated) as a subtle overlay texture. Animated version:
the weather rolling forward as generation proceeds (the v007 funnel-film
idiom applied to the cache). This single figure is the paper's Figure 1.

---

## P4 (wildcard, sketched). THE ERASER — the suppression-mechanism sequel

The deep-insight answer to "(d) what neither of us listed": the paper
LOCALIZED where a known answer dies (d1-peak/d2-crash, mid-stack causal
locus, depth-4 re-injectability) but never identified the AGENT. Program:
head/MLP-level ablation census at the crash site; the active-eraser vs
passive-overwrite discriminator (feed TF-clean states head-by-head through
blocks 1-2 — if ablating one head restores the address, suppression is a
CIRCUIT, not a property); the A-rev symmetry unification (does the
machinery that actively kills battery readout off-geometry, 0.775→0.08,
coincide with free-run suppression? — two suppression phenomena, one
possible eraser); cross-net conservation of the eraser; and the payoff:
can a one-head edit un-suppress knowledge without any state transplant —
an edit CHEAPER than the paper's d4 write? Viz: **v016 "Crime scene"** —
the address's survival traced through the stack with per-head ablation
columns and the killer highlighted. Paper if a conserved eraser circuit
exists ("the brain's forgetting circuit" at char scale); workshop note if
suppression is diffuse; highest-variance bet of the four — hold until P1's
census says whether the address is sparse enough for head-level logic to
attach to.

---

## Sequencing and cross-program notes

- **Start P1 today:** e065 is minutes on cached trajectories; e066/e067
  are eval-only. Zero training until phase 3.
- **P3's phase 1 is already registered** (ctx-512 was the T031 decider) —
  run it in parallel; it shares the lesion harness with nothing else in
  flight.
- **P2 waits on nothing but discipline:** e060 is queued; run the ramp,
  then let the A-residualized verdict decide whether the constructive
  leg or the negative-closure leg becomes the spine.
- **Cross-links are features, not accidents:** the wpe-row transplant
  (P1 phase 4) is a one-row organ graft — the minimal test of P2's
  interface hypothesis; the provenance split (P3) and the eraser (P4)
  are the same exposure-bias coin from two sides; the cliff link (P1
  phase 3) is scan candidate B and the only place a tiny-lab result
  touches the long-context flagship literature directly.
- **Review gates:** each program's kill signals are cheap and front-loaded
  (phase 1); no program should survive two consecutive phase-1 kill
  signals — that is the parking-lot promotion policy applied in reverse.
