# Historical journal context — selected source passages

These are source excerpts, **not current claims**. Read the correction map in
[UNIFIED_EVIDENCE](../../../docs/UNIFIED_EVIDENCE.md) before interpretation.
Post-wash geometry, the lower-rung comparator and broad mechanism language
are specifically corrected there. Full journal bytes remain in private custody.

Original source: `llm-genome/THINKING.md`; SHA-256 `527393a625d86e314355db6f45e3583a51bd4169d53536555dd0ef42d435413d`.
Numbers below are original source line numbers, not current file line numbers.

## Original lines 1–111

```text
000001 # Thinking Journal — the 80%
000002 
000003 The lab's operating ratio is 80% thinking / 20% doing. Every result gets an
000004 entry here BEFORE the next experiment that builds on it: what it could mean
000005 (multiple hypotheses), how the hypotheses differ, which cheap observation
000006 would discriminate them, and a registered prediction so we can't retrofit.
000007 New experiments are gated on this file: if the latest result has no
000008 interpretation entry, the next heartbeat thinks instead of runs.
000009 
000010 ## T349 — e347 (partial, folded at the lab pause): the grammar's first crossing — dose transports everywhere, texture breaks at SIGN, and step-one is nobody's (2026-10-13, datetime.now ~06:35Z; NON-ADJUDICATED — the composite verdict stays OPEN; no lean tick; P3/P4/P5 unrun)
000011 
000012 The optimizer cross died twice (the quota kill ~17:24Z Oct-12; the ~1h-window kill
000013 ~22:32Z Oct-12) and was resumed once with the three-proof standard PROVEN for a
000014 real death (19 arms bit-identical across the kill; draw-record md5 conserved). What
000015 landed before the pause is the cell's first two phases, and they already carry the
000016 crossing's shape:
000017 
000018 (i) THE LADDERS TRANSPORT: all four optimizers — SGD, MSGD, RMSPROP, SIGN — found
000019 canonical lrs landing IN-BAND (reads 0.213-0.257 vs the AdamW anchor 0.265; write
000020 norms 6.6-9.8 vs 9.2; the dose-normalized 5-point ladders worked at lr spreads of
000021 FOUR orders of magnitude, SGD@3.0 down to SIGN@0.0003). DOSE IS THE CURRENCY across
000022 optimizer families — Law 5's economics transports, and the canonical_lr_rule
000023 machinery is certified reusable (the BENCH's backbone).
000024 
000025 (ii) THE TEXTURE BREAKS AT EXACTLY ONE OPTIMIZER: SGD/MSGD/RMSPROP HOLD the
000026 formation texture (in_own_room 0.930/0.957/0.943 vs anchor 0.944) — SIGN MOVES
000027 (0.826, the only inroom_in_band False). Sign descent is the one rule in the set
000028 whose update has EQUAL magnitude in every coordinate by construction — its writes
000029 are isotropic in coordinate space. TWO EXPLANATIONS ON PAPER: (a) ISOTROPY ITSELF
000030 (any equal-magnitude rule spreads the write out of the formation's own room) vs
000031 (b) THE SIGN-CORRELATION STRUCTURE (sign() synchronizes coordinate directions —
000032 the write's cross-coordinate correlation, not its magnitude profile, is what the
000033 own-room geometry measures). DISCRIMINATOR (rides the Muon cross): spectral
000034 whitening is isotropy-ADJACENT but correlation-PRESERVING — if Muon's texture
000035 HOLDS in-room while SIGN's moved, isotropy is not the cause and (b) wins; if
000036 Muon's texture also moves, (a) gains its first leg. REGISTERED PREDICTION: (b) —
000037 the own-room measure has tracked write CORRELATION structure at every crossing
000038 (the name-blind writes; the compact-in-room feedback finding of x51).
000039 
000040 (iii) STEP-ONE IS NOBODY'S: s1 DIE for ALL FOUR at canonical lr — from THE BASE,
000041 nothing happens at step one under any optimizer (reads 1.3e-5 class). P-e347a
000042 clause 2 FALSIFIED: the s1 behavior is not AdamW-specific; it is substrate-uniform.
000043 The cold-shock saga's step-one chapter closes as a BASE property, not an optimizer
000044 property. NOTE THE FORK THIS LEAVES: the warm-substrate s1 shock (e335's class,
000045 reads executed at s1 on formed states) is a DIFFERENT object from this cold-BASE
000046 s1 silence — the bracket rungs (P3, unrun) were to speak to the former.
000047 
000048 (iv) W061'S CASCADE GETS ITS FIRST BOUND: SGD has no slow variable (like SIGN) yet
000049 HOLDS texture; SIGN has none and MOVES it — the texture axis does NOT track
000050 timescale coverage. Whatever own-room geometry measures, it is not which timescale
000051 variable parks the write. (The retention axis may still be W061's; the two axes
000052 are now empirically separated.)
000053 
000054 RESUME POINT (exact): P3 bracket mid-rung (rung_SGD_0.0008x chunk3, resume ckpt
000055 saved); P4 maintenance (tonly/conly) and P5 unstarted; bars frozen in the birth
000056 registration; the composite (TRANSPORT-INVARIANT / OPTIMIZER-SPECIFIC / MIXED)
000057 cannot fire until P3+P4 land. The working-tree metrics clobber (OneDrive, second
000058 instance) was restored from the committed b7501f9 — the resume reads the committed
000059 state, not the disk accident.
000060 
000061 [CORRECTION BANNER (2026-10-13 ~07:05Z): the "~1h-window kill class, third instance" claim below is WRONG —
000062 the executor did NOT die at ~22:32Z; it STALLED ~8 hours (22:32Z -> 06:24Z, the heartbeat-gap class) and
000063 RESUMED ITSELF at 06:24Z as a fresh process re-verifying checkpoints. One real death (the quota kill), one
000064 stall. The cell is COMPLETING under the pause directive's carve-out ("let them finish"); the composite
000065 verdict may land before the pause finalizes — the fold below describes the P1+P2 state accurately and
000066 stands as the interpretation of record for those phases; P3-P5's fold appends when the process exits.]
000067 
000068 [FINAL ADDENDUM AT THE PAUSE (2026-10-13 ~07:35Z; the last session's data harvested after the
000069 agent task was STOPPED at the owner's pause): the timeline is now precise — the agent's own commit
000070 (61d8e76) names the mechanism: each ~59-minute session KILLS ITS BACKGROUND RUN (two sessions:
000071 21:33Z and 06:24Z Oct-12/13, both dead at ~59 min — the ~1h-window class CONFIRMED as the session
000072 limit); restart #2 PROVEN bit-identical (21 arms + P2 vs the b7501f9 baseline; draw md5 conserved
000073 again — the three-proof standard held across BOTH restarts). THE CELL'S LAST LANDED DATUM (bd33af6):
000074 P3 rung_SGD@0.0008x — the Law-2b LOWER EDGE under SGD at exact matched budget — HOLDS (ratio_t400
000075 0.8775, g0_t400 0.2322; G_ORTH 4.6e-17, G_BUDGET exact, G_BUFSEP 400/400). Under AdamW 0.0008 IS
000076 the kill onset; SGD's first bracket point does not kill there. THE DECISIVE RUNG (0.004x, the
000077 upper edge) died mid-flight — the resume's first question is exactly that rung. FINAL STATE: P1+P2
000078 complete (folded above), P3 1-of-N rungs (0.0008x HOLDS; 0.004x ckpt at chunk5/step ~245), P4/P5
000079 unrun; composite verdict OPEN; the cell counts NOTHING until adjudication (no lean tick).]
000080 
000081 ## T348 — x66: the oracle's honest answer — the wash is a uniform wall, the menu is a K-dial, and the barriers vanish exactly where Hopfield said (2026-10-12, datetime.now ~15:40Z)
000082 
000083 The era's first sweep-scale cell returned four verdicts, two of them structural.
000084 REGRESSION-SILENT — but read carefully: the retention target itself was DEGENERATE
000085 (every powered formation died under the unbiased 100-step wash into a 0.018-IQR
000086 band, missing the 0.02 floor by 11%): THE WASH IS A UNIFORM WALL at this class.
000087 The oracle's answer is not 'your features are weak'; it is 'there is nothing to
000088 predict' — the first-step collapse dominates everything, and any retention-prediction
000089 program needs either a gentler wash or a longer-horizon retention measure. The
000090 ablation is untested-not-refuted (a silent full set cannot carry T344's reading —
000091 the frozen clause fired honestly).
000092 
000093 THE MENU IS A K-DIAL: the jittered menu deepens EVERY seed (0.196→0.461, 0.145→0.394,
000094 0.259→0.553), the LACK seed carries under it, and the random-context menu is
000095 die-mode on every seed. T346's per-seed coin is menu-contingent; the K-companion
000096 regression (Q² −0.64) confirms: process features explain neither K nor retention —
000097 the residual is SEED × MENU. The two-dial picture (the seed dial, the menu dial,
000098 the menu bigger) is the sweep's constructive gift: the formation protocol's
000099 deepening lever exists and it is the menu's variety structure.
000100 
000101 BARRIERS-VANISH-AT-K16 — the Hopfield import's prediction lands exactly (0/15
000102 interior kills, the both-alive point at α=0.500 to three decimals), with the
000103 companion finding that the mid-band co-occupants are OWN-BASIN (path-plane
000104 residuals 0.63-0.66): mid-band co-occupancy is not interpolation between saturated
000105 states — it is its own configuration. The Hopfield frame is now a measured object
000106 in the lab: capacity, load catastrophe, and own-basin mid-bands.
000107 
000108 PRELUDE-SILENT with the knife-edge disclosed (the top statistic clears r but
000109 misses permutation by 0.0022) and NARROWS-TO-S200 exact — R88's registered
000110 prediction hit precisely. The emergent-key reading now stands at sweep scale: no
000111 trajectory marginal carries the key's depth; the seed × menu interaction does.
```

## Original lines 913–1104

```text
000913 ## W059 — the uniform wall: attractor, floor, or measurement compression? (2026-10-12, datetime.now ~21:10Z; the heartbeat's card under the quota hold)
000914 
000915 x66's honest answer (T348) left the strangest object in the record: every powered
000916 formation died into an 0.018-IQR band — REGRESSION-SILENT not because the features
000917 were weak but because NOTHING VARIES at the far side of the wash. A wall that tight
000918 is itself a datum. THREE READINGS, no bars: (a) DYNAMICAL ATTRACTOR — the wash's
000919 dose arithmetic forces every post-formation state to the same floor regardless of
000920 where it started (the wall is WHAT THE WASH IS); (b) INSTRUMENT-CLASS FLOOR — the
000921 0.86M battery saturates: at this class everything washes out to the same residual
000922 texture and only scale separates worlds; (c) MEASUREMENT COMPRESSION — our pooled
000923 summary statistics collapse genuinely different post-wash states into one number,
000924 and the structure is there, unread, one layer down.
000925 
000926 DISCRIMINATING OBSERVATIONS, ON PAPER FIRST (all cheap, none dispatched): (1) if
000927 the post-wash states were persisted in runs/x66, a PER-LAYER/PER-FEATURE re-read of
000928 the wall's members is pure desk — structure appearing there kills (c) and the wall
000929 becomes a real object; (2) the SUBSTRATE PROGRAM is the wall's first true
000930 perturbation — under Muon or the constructive loss the wall should MOVE if (a)
000931 (it is dynamics-bound) and STAY PUT if (b) (it is class-bound); register wall-width
000932 as a first-class column in the SUBSTRATE BENCH, not a constant of the instrument;
000933 (3) the K-dial already whispers: the menu moves seeds through K BEFORE the wash —
000934 if the dial's spread survives the wash measured in K-units, the wall is a property
000935 of the pooled currency, not of the states (leaning (c)); if K collapses too, (a)
000936 gains its first leg.
000937 
000938 [DESK ANSWER (2026-10-12 ~21:25Z, same beat as the card's mint; CPU 4 threads,
000939 ~1 min; scratch/_w059_reread.py + _w059_scale.py committed for reproducibility):
000940 DISCRIMINATOR #1 RAN — READING (c) MEASUREMENT-COMPRESSION IS DEAD. The 20
000941 persisted wall members (every menu {authored, jittered, random, coo, half} x 10
000942 seeds x doses 100-800 x both hosts) were re-read per-layer in parameter space:
000943 per-layer norm CV across members 0.6-2.9% (no layer varies); inter-layer agreement
000944 on who-is-close-to-whom median Spearman 0.983 [0.891, 1.000] over 325 layer pairs
000945 (no layer tells a different story — exactly the structure (c) predicted, and it is
000946 absent); member-to-member distance / layer norm median 0.185 [0.082, 0.315] — a
000947 TIGHT CLUSTER sharing most of its direction, ~18% relative spread. THE WALL IS A
000948 REAL OBJECT IN PARAMETER SPACE: the pooled read hid nothing. THE FORK NARROWS TO
000949 (a) attractor vs (b) class-floor — discriminator #2 (does the wall MOVE under
000950 Muon / the constructive loss) upgrades from curiosity to the BENCH's first-class
000951 column. FREE TEXTURE: the cluster's small spread is NOT uniform — it concentrates
000952 in wpe.weight (0.315, the largest) and the late mlp.2 projections (0.24-0.31) —
000953 the position table and the late-MLP carrier layer are where washed worlds still
000954 disagree; the read dies long before those layers forget each other.]
000955 
000956 The joy here: the lab's first REGRESSION-SILENT verdict may turn out to be its
000957 first instrument-grade discovery — walls tell you what the space of deaths looks
000958 like, and nobody has ever mapped that space under varied substrates.
000959 
000960 ## W060 — THE WALL'S SYMMETRY CLASS (for the attractor-geometry focus): is the wash a BASIN COLLAPSE — the 20 wall members ONE terminal point modulo permutation — with the non-permutable wpe table the wall's only genuine fiber?
000961 
000962 SOURCE-FIELD: loss-landscape symmetry / permutation-invariant linear mode connectivity —
000963 independently trained nets often occupy one basin modulo neuron relabeling (Entezari et al.,
000964 arXiv:2011.01512; Ainsworth/Hayase/Srinivasa, Git Re-Basin, arXiv:2202.13965).
000965 
000966 TRANSLATIONAL QUESTION: are the wall's members the SAME washed solution up to hidden-unit/
000967 head permutation — so the wash funnels every powered formation through a symmetry collapse —
000968 making the 0.315 wpe spread (position rows are NOT permutable; T053's own note) the entire
000969 genuine content of "washed worlds still disagree", i.e. the wall's fiber is exactly the
000970 position axis the late-MLP carrier reads through?
000971 
000972 CHEAPEST DISCRIMINATING OBSERVATION (desk, persisted x66 members, CPU-minutes):
000973 weight-match the permutable blocks (mlp hidden units, attention heads) between member pairs
000974 (iterative straight-through); PERMUTATION-COMPACT = the 0.24-0.31 late-mlp spread collapses
000975 toward ~0 while wpe sits still (one basin mod symmetries; positions carry all residual
000976 structure) vs EXTENDED-MANIFOLD = alignment leaves the spread standing (a genuinely flat
000977 terminal region, not a point-plus-symmetry). RIDER (W059's fork gains its third shape for
000978 free on the same states): run any 3 members 20 further wash steps, project the displacement
000979 onto the inter-member spread subspace — displacement ALONG the spread = a SLOW MANIFOLD (the
000980 wall is where the flow crawls, and the wpe/mlp.2 disagreement is the crawl's direction);
000981 displacement off it or ~zero = point attractor / class floor.
000982 
000983 EXTENDS: W059's desk answer (the exact ~18% object and its wpe/mlp.2 texture); the lab's
000984 interpolation-barrier machinery (new object: the wall's INTERNAL geometry); feeds the
000985 SUBSTRATE BENCH's wall-width column with a symmetry-corrected definition (width after
000986 alignment, not before).
000987 
000988 [DESK ANSWER (2026-10-12 ~22:00Z, CPU 4 threads; scratch/_w060_align2.py committed;
000989 HONESTY NOTE: the v1 aligner (_w060_align.py) was a NO-OP BUG - x[perm] applies the
000990 permutation instead of scattering unit i to its matched slot j - and its identical
000991 before/after numbers were meaningless; the v2 sanity gate (a KNOWN random shuffle
000992 recovered 1.4090 -> 0.0000 exact, Hungarian via scipy) certifies the machinery
000993 before any member claim): EXTENDED-MANIFOLD LANDS ON THE MLP LEG. Optimal
000994 hidden-unit alignment of all 19 members to the reference leaves the late-mlp.2
000995 distance UNCHANGED (median 0.2404, range [0.1439, 0.3177], identical at 4 decimals
000996 before and after) - the members are ALREADY in best-possible basis correspondence;
000997 their differences are genuine same-coordinate differences, not neuron relabelings.
000998 BOUNDS DISCLOSED: head-permutation untested (irrelevant to the mlp.2 object - head
000999 perms leave the residual-stream coordinates mlp.2 reads untouched - but named for
001000 the full card); wpe untouched by construction (the 0.333 control). CONSEQUENCES:
001001 (1) W059b does NOT need the align-before-score birth amendment (the fold note's
001002 condition fired only on PERMUTATION-COMPACT); raw parameter distances are
001003 meaningful on this wall. (2) The wall's fiber is NOT pure wpe: the late-mlp.2
001004 spread survives optimal alignment, so "washed worlds still disagree" extends beyond
001005 the position axis into the carrier layer's own coordinates. (3) The slow-manifold
001006 rider (displacement along the spread subspace under further wash) stays OPEN -
001007 GPU, behind e347's lane.]
001008 
001009 ## W061 — THE OPTIMIZER AS CASCADE (for the optimizer x memory-grammar focus): is a fact's wash-survival set by WHICH TIMESCALE VARIABLE the write landed in — are the optimizer's buffers the net's synaptic cascade?
001010 
001011 SOURCE-FIELD: computational neuroscience of memory under ongoing plasticity — cascade models
001012 (Fusi/Drew/Abbott 2005) and Benna & Fusi 2016 (arXiv:1603.01354): memory lifetime is set by a
001013 hierarchy of variable timescales; a memory is consolidated when written into the SLOW ones.
001014 
001015 TRANSLATIONAL QUESTION: do the optimizer state variables — AdamW's v (slow, per-parameter),
001016 momentum (fast), Muon's instantaneously-whitened update (no slow channel at all), AdEMAMix's
001017 beta3~0.9999 EMA (a literal third, very slow synapse) — constitute the 2.74M net's cascade,
001018 so that "which optimizer facts survive a wash" is really "where does each optimizer park the
001019 write" (slow coordinates pin, fast coordinates die), and optimizer x memory-grammar becomes a
001020 measurable timescale-coverage question no optimizer paper asks (the survey's 0-hit is about
001021 fact-level retention; this names the mechanism the literature would need)?
001022 
001023 CHEAPEST DISCRIMINATING OBSERVATION (desk-first): if any committed states persist optimizer
001024 buffers, correlate per-matrix wash retention with the formation write's mass on slow vs fast
001025 coordinates (small-v coordinates = fast = should wash first, the cascade signature); else a
001026 rider on e347's resumed arms — persist the buffers at formation end, one extra read — or,
001027 paper-only, check whether the s1 shock magnitude across e347's four optimizers orders by each
001028 rule's slowest timescale (1/(1-beta2)-class).
001029 
001030 EXTENDS: the survey's AdEMAMix rank-3 entry (the cell was ranked; the import supplies its
001031 theory and prediction SHAPE — lifetime ratios, not speed); e273's finding that Adam's
001032 normalization builds the low-rank formation block (a cascade-flavored datum re-read: v's
001033 per-parameter timescale structure IS the block's origin candidate); W059 discriminator #2
001034 (the wall moving under Muon gains a mechanism noun: an update rule with no slow variable
001035 cannot park mass — does its wall sit elsewhere, or not form?).
001036 
001037 ## W062 — THE FACT AS HEDGE (for the constructive-loss focus): is elicitation decorrelation the fact's BET-HEDGE — and should the constructive loss therefore be SPECTRAL, driving the context-response correlation's top eigenvalue to its floor?
001038 
001039 SOURCE-FIELD: evolutionary bet-hedging / stochastic switching — diversity pays exactly when
001040 the selection environment fluctuates unpredictably (Kussell & Leibler 2005, Science) — with
001041 the quantitative handle borrowed from portfolio risk: a spread of carriers survives a crash
001042 to the degree their correlation matrix's TOP EIGENVALUE is small.
001043 
001044 TRANSLATIONAL QUESTION: is the lab's key (decorrelated elicitation families) the net's
001045 bet-hedge — storage spread across carriers whose responses do not co-move because the
001046 post-formation context distribution is unpredictable — and is the constructive loss's
001047 correct form therefore not a flat DeCov-style covariance penalty but an EIGENVALUE TARGET:
001048 drive lambda-1 of the cross-context next-token correlation matrix toward the Barlow-Twins
001049 identity floor while the read-height terms hold (diversity with quality — the hedge must
001050 still pay out)?
001051 
001052 CHEAPEST DISCRIMINATING OBSERVATION (desk, on disk): over the committed states that have
001053 both per-context correlation matrices and retention numbers (x44's offset panels, x45's walk
001054 rungs, x66's menu arms), regress wash retention on lambda-1 of the context-response
001055 correlation vs mean-correlation vs write-norm — HEDGE-PRICED = the eigenvalue carries
001056 retention the others do not (the portfolio prediction, one regression, no new runs); FREE
001057 ALREADY-LANDED DATUM = T348's K-dial (the jittered menu = a forced hedge) deepening every
001058 seed while the random-context menu dies — the frame's first supporting point, retroactively.
001059 
001060 EXTENDS: the survey's constructive-loss entry (machinery DeCov/Barlow/VICReg cited — the
001061 import supplies the WHY diversity pays here and the loss's target form); the 2026-10-10
001062 wildcard's "correlation eigen-structure as probe feature" (there a regressor input; here
001063 the trained-for quantity); prior-era eigenvalue work was curvature spectra (Fisher/Jacobian
001064 e265-class) — a different object, disclosed. If HEDGE-PRICED, e349's constructive arm gets
001065 its lambda dial pre-registered in eigenvalue units.
001066 
001067 [Folded verbatim from scratch/novelty_2026-10-12T2140.md by the heartbeat 2026-10-12 ~21:50Z (the ideator minted; the W-series ledger entry is the heartbeat's fold). Connections at fold time: W060's PERMUTATION-COMPACT outcome would make preg_w059b's parameter-distance metric symmetry-blind (align before scoring — a birth amendment owed if W060 runs first); W061 hands e347's resumed arms a free rider (persist the optimizer buffers at formation end) and gives W059b its mechanism noun (SGD has NO slow variable — where does its wall sit?); W062's lambda-1 dial pre-registers e349's constructive arm in eigenvalue units, with T348's K-dial as its already-landed supporting datum.]
001068 
001069 [DESK-FEASIBILITY ANSWER (2026-10-12 ~22:05Z, heartbeat probe of runs/x66/metrics.json):
001070 the minted regression is NOT EXECUTABLE on disk — the sweep arms persist scalar
001071 features only (menu flags, read_t0, dip_ratio, write_growth, seed_K, name_prior);
001072 NO per-context correlation matrices anywhere in x66, and the needed intersection
001073 (matrix at formation AND wash retention on the same states) exists in no committed
001074 cell — x44/x45 instruments pooled into scalar r values by design. RE-KEY: the
001075 lambda-1-vs-retention regression becomes a RIDER on W059b's twin washes (the
001076 cheapest cell that will hold both sides): log the cross-context response
001077 correlation matrix at formation end and at each wash panel; lambda-1(t) alongside
001078 retention(t) falls out for free. W062's already-landed supporting datum (the
001079 K-dial) is untouched by this — only the cheapest-observation route moved.]
001080 
001081 ## W063 — the stream's fixed point: is the wall a property of the WASH or of the STREAM? (2026-10-12, datetime.now ~22:10Z; minted from the R90 agenda's sharpest attack — the confound deserves its own card)
001082 
001083 Every wall member was washed by the SAME stream (x66's certified wash corpus, draws,
001084 and order — the instrument's constant by design). The tight ~18% cluster may therefore
001085 be THE STREAM'S FIXED POINT — the attractor of that specific continued-training input —
001086 rather than a wash-general property of forgetting (W059's "the wash is the wall" reading
001087 quietly assumes stream-invariance nobody has tested). W060's EXTENDED-MANIFOLD sharpens
001088 the stake: the terminal region is wide and coherent in the same coordinates; if it is
001089 stream-shaped, its width and direction are the STREAM's signature on the substrate.
001090 
001091 DISCRIMINATING OBSERVATION (the trivial generalization of W059b's twin-wash design,
001092 same cost): twin washes from ONE certified formed state under TWO STREAMS (the canon
001093 stream vs a permuted/shifted corpus variant) at matched update-mass — STREAM-SHAPED =
001094 the endpoints separate beyond the 2x-cluster threshold (and the separation direction is
001095 stream-specific, reproducible across seeds) / STREAM-BLIND = both land in the same
001096 cluster (the wall is input-content-free at this dose — pure dose arithmetic, the
001097 strongest form of W059 reading (a)).
001098 
001099 THE BRACKET THIS COMPLETES: W059b perturbs the OPTIMIZER axis of the wall; this card
001100 perturbs the STREAM axis; together they bracket the wall's provenance (arithmetic vs
001101 geometry vs input) before the BENCH freezes its wall-width column. W061's cascade frame
001102 supplies the mechanism noun either way: the stream writes through the optimizer's
001103 coordinates — if the wall is stream-shaped, the slow coordinates (AdamW's v) are where
001104 the stream's signature parks.
```

## Original lines 2125–2174

```text
002125 ## T304 — X32: the net0 confound — the fork was two loading conventions, and nothing ever happened at step one (2026-10-10, datetime.now ~07:35Z)
002126 
002127 The atlas went hunting for the surge's geometry and found instead that
002128 the surge never existed. The canon lineage installs from THE BASE; the
002129 census lineage installs from the g1c ROOT — a root whose standing read
002130 is 0.745. The "surge" is a standing read staying; the "no-surge" is a
002131 prior staying. At step one, in both conventions, NOTHING HAPPENS — no
002132 ray crosses any transition, and the read is indifferent to the first
002133 gradient's sign pattern. The bimodal wheel was a tale of two loaders.
002134 
002135 THE HONEST LEDGER OF WHAT THIS MEANS: the lab spent a night and a
002136 morning measuring "two formation modes" that were two starting states.
002137 The correction is sharp but not nihilistic — everything measured is
002138 still real, re-worded: BASE-FORMED writes (slow, pruned, localized,
002139 norm ~9) versus ROOT-FORMED writes (from a warm start, bulky,
002140 diffuse, norm ~27) are genuinely different formation OUTCOMES from
002141 different starting states — a legitimate and interesting comparison
002142 (W052's float survives as the start-state contrast). What dies: the
002143 claim that ONE protocol stochastically forks (gen-tracked); the die-
002144 share; "rare"; the empty band as a mode gap (it is the prior-to-
002145 standing gap). What re-words: e326's cross-class bracket (the root-
002146 formed write's transport — STILL conservative-direction and still
002147 valid), e334's controller (held a root-formed fact — still a genuine
002148 second substrate), the doc's Law 3 scope block (fully re-written
002149 below).
002150 
002151 THE META-LESSON (the third of its kind this week): the lab's eras
002152 carry silent convention drift — the anti's misbind (R68), the
002153 stamp discipline (R65+), and now the loaders. The monoculture clause
002154 was the right scent: the "second branch" smelled like a new
002155 instrument because it WAS one. W054's law holds: instruments age —
002156 and so do the conventions between them. The new standing rule: ANY
002157 cross-era comparison gate-checks the STARTING STATE first (net0's
002158 class boundary recorded in every install's metrics from now on).
002159 
002160 THE FORMATION FRONTIER RE-AIMED: the live question is no longer "why
002161 does the fork fork" but "what does the start-state buy" — the root's
002162 warm start yields bulky diffuse writes at 3x the norm; the base's
002163 cold start yields pruned localized ones. e328's swap (running, with
002164 its prediction registered) now reads on the formation axes. And the
002165 deepest open: WHERE did the root's standing read come from — the
002166 root is itself an artifact of the g1c ladder's history. The lab's
002167 oldest substrate has a memory, and the census was reading it.
002168 
002169 ## T303 — X34: the kill that never was — the die texture dissolves into no-surge, and the surge is the anomaly (2026-10-10, datetime.now ~05:55Z)
002170 
002171 > [T304 CORRECTION 2026-10-10]: one step further than this card knew — the no-surge/surge
002172 > axis was itself the net0 confound (base-start vs root-start). The card's prior-ratio
002173 > arithmetic survives and was the thread that led x32 to the confound.
002174 
```

## Original lines 14339–14371

```text
014339 **P1 CONFIRMED, absolutely:** expression count = 0 across ALL arms —
014340 unseeded, seeded (induction route), greedy diagnostic, T ∈ {0.7, 1.0,
014341 1.3}, and dose × {400, 800, 1600} steps — while battery accuracy holds
014342 0.92-0.96 throughout. P2 (prior threshold) refuted: neither seeding nor
014343 temperature unlocks a single occurrence. R8 RELABEL: P3's dose legs were BIT-IDENTICAL across s400-1600 (dose_probes AND generations) — same-checkpoint readout, INSTRUMENT SUSPECT; dose-response is OPEN, not refuted. P1 (zero expression) survives independently (its arms vary-verified).
014344 
014345 **C7 FINAL — the edit law, complete:** there are FOUR separable faculties:
014346 ADDRESS (rows — surgically removable, attractor-surviving), ABILITY
014347 (distributed body usage — train-only), EXPRESSION (free-generation
014348 surfacing — requires free-generation-shaped exposure; teacher-forced
014349 install never confers it at any dose), and HISTORY (the re-learned memory
014350 differs from the original in route and surgical-resistance). An exposure
014351 protocol that never shows the name in FREE contexts installs answer-
014352 knowledge that is constitutionally silent.
014353 
014354 **Lab-doctrine note:** this is also a warning about the battery method
014355 itself — continuation-battery accuracy is NOT evidence of usable
014356 knowledge; free-generation probes are the honesty check for any install
014357 claim, at any dose.
014358 
014359 **FULL-REPORT REFINEMENT (01:05Z):** the expression failure DECOMPOSES —
014360 (a) **geometry binding:** the installed address fires only in the trained
014361 continuation window (p(Z) 0.556 in battery geometry → 0.0898 at the same
014362 terminal text in a 120-char free prompt → 1.7e-6 in uniform contexts;
014363 deleting the 10 oldest context chars collapses it 6×); (b) **sub-argmax
014364 prior:** even where prior exists it never wins argmax. The single
014365 expression event of the day (1 in ~30-35k chars across 13-14 cells (R7: denominator bookkeeping imprecise; direction unaffected)) came
014366 from the DIRECT-TRAINED control at its home slots (p(Z) 0.27 there,
014367 argmax 25%) — natural learning puts the name in the free-generation
014368 prior; install does not, at any dose. Uniform-floor battery shows the
014369 install at 0.286 vs 0.974 host-trained (40× floor, 3.4× below claim —
014370 battery-overfit partial; the direct net shows the same floor shape, so
014371 part is CORRECT slot-gating). Open edge: content-vs-position binding.
```

## Original lines 14434–14484

```text
014434 ## T016 — C6 DEMOTED: two-factor erasure does not replicate (E046, 2026-09-24T23:28Z)
014435 
014436 **The replication verdict (2 hosts, 4 cells, full honesty battery):**
014437 - **Neither the in-run-discovered head (B43: L4H4; BDO: L3H1) nor B's
014438   L3H5 recipe reaches Bar-2 on any other net** — JULIET stays at 13-17%
014439   accuracy after row-zero + top-head lesion (B43 own-head 15.7%,
014440   uniform-floor 15.1%; BDO 13.9/16.4%). B's specific recipe transfers as
014441   predicted-NO (its site is inert in B43: −0.14).
014442 - **What DOES replicate:** row surgery alone damages to the same 13-17%
014443   band at ~+0.001 corpus cost, class-exact — on every net. The ADDRESS
014444   half of the doctrine is general; the "one body head" COMPLETION was a
014445   B-specific coincidence.
014446 - **C6 demoted to:** "entity-row surgery is a general, cheap, class-exact
014447   DAMAGER (~500× selectivity); COMPLETE erasure was achieved once (B,
014448   rows+L3H5) but the second factor did not generalize — the residual after
014449   address-burning is net-specific in structure."
014450 - **Pattern note for the card:** the day's positive mechanism claims are
014451   systematically dying under replication/audit (rare-token heads, stage
014452   invariance, now two-factor erasure) while its NEGATIVE claims (ascent
014453   fails, no far retrieval on natural data, install-not-surgical) hold
014454   everywhere. The asymmetry-laws survive; the mechanism-stories don't.
014455   The Review-6 panel must weigh a REPLICATION SWEEP of remaining positive
014456   claims vs new arcs.
014457 
014458 ## T015 — Edit asymmetry law (E043; audited 21:30Z, AMENDED by full report 22:05Z)
014459 
014460 **AMENDMENT (full report):** the earlier "exposure installs transiently then
014461 decays" line was a partial-trajectory read. The truth is better:
014462 - **P2 FALSIFIED in the good direction: anchored exposure installs CHEAPLY
014463   and SELECTIVELY.** 7 guarded cells reach Bar-I2; best Dmix@s400: 0.09
014464   NLL / 0.974 acc at ΔCE +0.0505, S_install 144-289, incumbents ≤0.06.
014465   The earlier CE-climb belonged to a different anchor trajectory.
014466 - **Two new caveats define the law's real shape:** (1) **the expression
014467   gap** — 0.974 battery accuracy yet ZERO ZEPHYRA in 2,800 generated
014468   chars: teacher-forced install never surfaces in free generation;
014469   "installed" ≠ "expressed." (2) **protocol fragility** — the onset wall
014470   was anchor-manufactured (0.00 through s1000 under fully-paired anchors
014471   at CE +3.40 vs 0.82 by s400 under the mix at CE +0.05): install is
014472   protocol-fragile where erase was protocol-robust.
014473 - **P3(b) beautiful:** the installed name rides the INCUMBENTS' shared
014474   L0-MLP (+9.18 nats, top head only 10% of the block) — no private
014475   circuit ever grows; new knowledge reuses shared machinery (consistent
014476   with e042). Deep installs mildly reshuffle incumbent rankings (ρ→0.66).
014477 
014478 **The law (final form, C7):** REMOVAL is surgical — 384 parameters,
014479 robust, class-exact. INSTALLATION requires plasticity — never surgical
014480 (best graft closes 5.9% at 2.4× the guard; B43 diff-init rows install
014481 nothing while damaging +1.92) — but plasticity is cheap and highly
014482 selective when anchored; its limits are expression (silent under teacher
014483 forcing) and protocol-fragility. Address is concentrated; ability is
014484 distributed and trained; expression is a third thing entirely.
```
