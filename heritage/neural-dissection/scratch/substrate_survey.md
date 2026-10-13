# THE SUBSTRATE SURVEY — researcher, 2026-10-12T02:42Z (datetime.now UTC; CPU desk)

Feeds THE SUBSTRATE PROGRAM (owner directive ~02:30Z). Builds on: e347 (the classic-four
optimizer cross, RUNNING on GPU — SGD/momentum/RMSProp/sign), e348 (architecture ladder,
READY: depth/width/norm-placement/no-residual), e349 (loss ladder, READY: CE/MSE/hinge/
constructive). This survey covers what those cells do NOT: the modern optimizer field,
the target-geometry losses, the architecture variants beyond e348's frozen list, and the
novelty scan. Ranking metric = information about the LAB's questions / implementation
cost (the certified harness = plain PyTorch, no new deps).

## 1. THE OPTIMIZER LANDSCAPE 2024-2026 (beyond e347's classic four)

The field's own calibration first: Wen et al., "Fantastic Pretraining Optimizers and
Where to Find Them" (arXiv:2509.02046, Sep 2025) fair-tuned the claimants and found
AdamW hard to beat; the genuine winners are all MATRIX preconditioners — Muon, SOAP,
Kron — at 1.2-1.4x (not the claimed 1.4-2x). Muon is best at 1-4x Chinchilla compute
(= exactly our regime); SOAP/Kron win at >=8x overtraining. Scalar variants (Lion,
MARS, NAdamW, Cautious) land within ~1.1x. For the lab this is GOOD news: we do not
care about perplexity speedups; we care about which OPTIMIZER FACTS the memory grammar
turns on — and the fair-benchmark hierarchy tells us which optimizers are genuinely
DIFFERENT animals rather than re-tuned Adam.

Ranked catalog (mechanism / small-scale evidence / implementability / cost of a fair
trial at 2.74M = 3-4 lr points x 2 seeds x ~2-3 min, i.e. ~20-40 GPU-min per optimizer,
CPU probes on top):

1. **MUON — rank 1 (information/cost).** Orthogonalizes the momentum of each 2D weight
   matrix via ~5 Newton-Schulz iterations, then scales by sqrt(fan-out/fan-in);
   embeddings/lm_head stay AdamW (Jordan et al. arXiv:2412.16793; modded-nanogpt
   records). Proven at scale: Moonlight/Kimi — 48.4% compute reduction at 3B/1.52T
   tokens, MuonClip (arXiv:2502.16982), adopted in Kimi K2; and per Wen et al. the best
   small-compute optimizer. Implementability HIGH: ~60-80 lines, matmul-only, no deps.
   Why it beats a speedup for US: e273's barrel (b) found the low-rank formation block
   is ADAM'S NORMALIZATION; Muon is anti-low-rank BY DESIGN (spectral whitening of the
   update). A Muon canon formation directly asks: does the seat form in a different
   basis when the update geometry is rotated? Does the kill bracket [0.0008, 0.004]
   (Law 2b, AdamW-measured) re-express in Muon's update units? Does the cold-start
   sign-shock (e335's S1 class) exist at all without sign normalization?
2. **SCHEDULE-FREE ADAMW — rank 2.** Replaces the lr schedule with Polyak iterate
   averaging (Defazio et al., "The Road Less Scheduled", arXiv:2405.15682, NeurIPS
   2024); the ONLY submission to beat the AlgoPerf self-tuning baseline (~8%; MLCommons
   Aug 2024; analysis arXiv 2506 Kasimbeg et al.). Implementability VERY HIGH: the
   z/y/x interleave is ~30 lines on top of torch.optim.AdamW. Information: the lab's
   wash laws are entangled with cold-AdamW starts and cosine+warmup history (e341's
   annealing, the e185 start-up-shock saga); a schedule-free canon removes the horizon
   and the momentum/average conflation — the anneal/warm controls the supervisor asked
   for nine times, bought in one cell.
3. **AdEMAMix — rank 3.** Adds a second very-slow gradient EMA (beta3~0.9999) to
   Adam's fast momentum; 1.6-4.2x token-matching at 101M-1.3B (Pagliardini et al.,
   arXiv:2409.03137). Implementability TRIVIAL (~5 lines). Information: the supervisor's
   standing β2=0.95 vs moment-window questions (resurrection stickiness ~ 1/(1-beta2),
   v-inflation windows) get their cleanest probe — a slow-momentum channel is a
   different memory of the gradient than v.
4. **LION — rank 4.** Sign of the momentum (EMA of g then EMA of update), update-then-
   momentum order, no second moment (Chen et al., arXiv:2302.06675; found by program
   search). Fair-tune delta small (~1.1x class). Implementability TRIVIAL (~10 lines,
   lr 3-10x smaller, wd larger). Information: e347 already runs plain sign-descent;
   Lion is the EMA-coupled sign — the pair isolates what the momentum (not the sign)
   contributes to wash kinetics.
5. **SOAP — rank 5 (second wave).** Runs Adam in the eigenvector basis of the Shampoo
   preconditioner; >40% iteration / >35% wall-clock vs AdamW, NeurIPS 2024 spotlight
   (Vyas et al., arXiv:2409.11321); the external-tuning AlgoPerf winner was a Shampoo
   variant. Implementability MEDIUM: needs torch.linalg.eigh bookkeeping + basis
   rotation caching (~150 lines, no deps, but easy to get silently wrong — budget a
   verification run against the reference behavior on a toy problem). At >=8x-compute
   regimes SOAP overtakes Muon — not our regime, so it loses rank despite bigger wins.
6. **SOPHIA — rank 6.** Diagonal-Hessian (Hutchinson or Gauss-Newton-Bartlett) with
   element-wise update-ratio clipping; claimed 2x at GPT-2 125-354M (arXiv:2305.14342),
   later softened by fair-tune studies and the authors' own benchmark work.
   Implementability MEDIUM (HVP every k steps — a second backward through
   torch.autograd.grad; cheap at 2.74M). Information: the CLIPPING mechanism (cap the
   update/curvature ratio) is a different dial than anything above — worth one cell
   only if the bracket's edges move under it.
7. **PSGD-Kron / Grokfast — noted, not ranked.** Kron is a genuine third matrix-
   preconditioner per Wen et al., but its implementation (cyclic whitening, trust
   region) is the highest-dev-cost item here. Grokfast (arXiv:2405.20233) amplifies
   low-frequency gradient components — more an analysis lens for the wash's
   slow/fast decomposition than a trial candidate.

Our-own-optimizer opening (the owner's ask): the lab's instruments suggest a concrete
candidate family — a wash-aware optimizer that down-weights updates on coordinates
whose gradient sign conflicts with a maintained-fact's own gradient (the "unmaintained"
coordinate definition, check-in 5's sign-conflict probe). No published optimizer does
fact-conditioned preconditioning; e347's per-optimizer dose normalization gives the
protocol. Flag: design card owed before any cell (W-card + optimizer spec), not in this
survey's scope.

## 2. THE LOSS / ERROR-FUNCTION LANDSCAPE (what changes the TARGET GEOMETRY)

The lab's key observable = elicitation decorrelation (x45's fingerprint: r across
contexts). Ranked by information about formation dynamics / cost (all CPU-trainable at
0.86M arms; GPU at 2.74M):

1. **LABEL SMOOTHING — rank 1 (cheapest discriminator).** CE toward (1-e)·onehot +
   e/uniform. Known formation effect: collapses intra-class penultimate geometry into
   tight class-direction clusters AND ERASES within-class similarity information —
   Müller, Kornblith, Hinton, "When Does Label Smoothing Help?" (arXiv:1906.02629);
   calibration improves (Pereyra 2017's confidence penalty, arXiv:1701.06548, is the
   entropy-flavored twin). Implementability TRIVIAL (one line in the loss).
   Prediction the lab can register: LS's cluster-collapse should REDUCE elicitation
   decorrelation (contexts collapse toward class means) — and it moves logit margins,
   so the kill bracket and the read-height ladder (x39) should shift. One cell, three
   readouts.
2. **THE CONSTRUCTIVE LOSS (e349's own term) — the program's novel question.** Can
   response diversity be PAID FOR in the loss? The existing machinery, all portable
   with no deps: DeCov — penalize squared off-diagonal of the hidden-activation
   covariance (Cogswell et al., arXiv:1510.02468); Barlow Twins — drive off-diagonal
   of the cross-view correlation to zero (Zbontar et al., arXiv:2103.03230); VICReg —
   variance hinge + covariance penalty (Bardes/Ponce/LeCun, arXiv:2105.04906); NCL —
   correlation penalty on ensemble errors (Liu & Yao 1999). The lab's version: CE +
   lambda · covariance penalty on the model's next-token DISTRIBUTIONS (or context-
   pooled hidden states) across DISTINCT elicitation contexts of the same fact —
   i.e., write the key (decorrelation) directly into training. Search verdict: this
   exact application has no counterpart found — SSL uses it across augmentation
   views, ensembles across members; nobody rewards cross-CONTEXT response diversity
   in an LM. COST: ~30 lines (a batch reshaped into context groups); the honest risk
   is trivial-solution leakage (decorrelate by degrading the read) — the battery
   guards against that for free.
3. **MSE-ON-LOGITS — rank 3.** Regression onto one-hot (or teacher) logits: no softmax
   tail-chasing, bounded gradients, different saturation (the distillation lineage,
   Hinton arXiv:1503.02531, uses soft targets for the same reason). Implementability
   TRIVIAL. Information: the lab's margin view of memory (check-in 4/5) predicts
   wash-retention is margin-geometry; MSE forms flat, marginless readouts — a clean
   two-ended test (if retention tracks margin, MSE-formed facts should wash FAST).
   e349 already lists MSE; the survey adds the registered prediction.
4. **R-DROP — rank 4, and the constructive loss's CONTROL.** Two dropout forward
   passes + bidirectional KL between them (Liang et al., NeurIPS 2021,
   arXiv:2106.14448); consistency = anti-decorrelation by construction (it PENALIZES
   response variance across passes). Running it as e349's fifth arm makes the ladder
   two-sided: constructive term should RAISE elicitation decorrelation, R-Drop should
   LOWER it, same instrument — the strongest single-cell design available. Cost: one
   extra forward per step (dropout is already in the harness if enabled; else trivial).
5. **FOCAL / token-difficulty losses — rank 5.** Down-weight easy tokens, up-weight
   hard/rare ones (Lin arXiv:1708.02002; for LM pretraining: FL-Tuning, Su et al.
   2024, ACL). Implementability TRIVIAL. Information: reads directly on the
   rehearsal-dose/frequency laws (e179-class) — a focal-trained canon shifts the
   effective dose of every fact; prediction: the rehearsal threshold moves, the
   passive death clock does not.
6. **Hinge/margin-on-logits — rank 6.** Bounded drive once the target margin is met
   (no asymptotic softmax saturation). The lab already owns the margin-hinge dial at
   the CONTROLLER level (calm-v3, QUEUE line ~709); training-level hinge is its
   pre-training twin — the registered bar transfers: >=99% retention with ECE within
   1.05x, i.e. calm != weak. Multi-class hinge on char logits is ~10 lines.

Not ranked (survey found no small-scale LM evidence or no target-geometry delta):
calibration-only losses beyond LS/CP (FL/LS hybrids); contrastive alignment losses
(need a view generator = a different program); z-loss (logit-norm regularizer, mostly
a stability tool — revisit with QK-norm below).

## 3. THE ARCHITECTURE AXIS at 2-100M (beyond e348's frozen list)

e348 already owns depth/width, norm placement, no-residual. What the survey adds,
ranked by what each discriminates about the SEAT GRAMMAR (capacity-two's architecture-
dependence) / implementability:

1. **POSITION-ENCODING LADDER: learned-absolute (canon) vs NoPE vs RoPE/ALiBi.** The
   supervisor's check-in 3 suggestion was never run (grep: no NoPE/RoPE cell in
   NOTES/QUEUE/lab). NoPE: causal attention alone carries position; BEATS explicit PE
   on OOD length generalization at small scale (Kazemnejad et al., NeurIPS 2023,
   arXiv:2305.19466); RoPE arXiv:2104.09864; ALiBi arXiv:2108.12409. Information:
   Law 9's coordinate-keying lives in wpe rows (wpe-129 the portable row); a NoPE net
   has NO address rows — does coordinate-keying die, migrate, or was it never about
   rows? This is the single sharpest architecture question the instruments can ask.
   Implementability HIGH: delete wpe / add rotary (rotary = ~30 lines).
2. **QK-NORM (+ RMSNorm block norm).** LayerNorm on queries and keys pre-dot-product;
   kills attention-logit blowup and softmax saturation (Henry et al. 2020,
   arXiv:2010.04345); the ViT-22B/Chameleon stability recipe (arXiv:2302.05442 /
   2405.09818, with z-loss). Information: the lab's sink/massive-activation thread
   (position-0 sink, the L5-calibrator, the 5/5 late-MLP carrier) is exactly the
   logit-explosion channel QK-norm suppresses — prediction: sinks form later/weaker,
   and seat formation (the write's in_room geometry) re-addresses. ~10 lines.
3. **DENSE / VALUE-RESIDUAL BLOCKS.** DenseFormer: learned depth-weighted average of
   all prior hidden states, ~1 param/layer, lower loss at matched params at 70M-1B
   (Pagliardini et al., arXiv:2402.02622). VaLOR/ResFormer: value residuals from
   layer-0, ~16% params saved at equal loss (arXiv:2410.17816). Information: they
   break the "one write interface, strictly-serial stream" premise the unified
   question (T037) is built on — the read policy gains literal copies of old states.
   Prediction: the address census (wpe-keying share) shifts toward content-keying.
   Implementability HIGH (~20-40 lines).
4. **SOFT-MoE AT 2.74M.** The literature is thin exactly here — MoE studies sit at
   >=1B (Switch arXiv:2101.03961; upcycling arXiv:2212.05055); no tiny-scale
   specialization study found (survey hit). That makes it OUR ground: parameter-
   matched 4-expert soft-MoE MLP (tokens mixed across experts — routing is dense and
   stable, ~30 lines) vs the dense canon. Does capacity-two move when "rooms" become
   literal expert slots? Does the name-blind write become expert-partial?
5. **LINEAR-ATTENTION HYBRID (GLA or Mamba block in the L3-4 slot).** Established
   deficit: fixed-state recurrence is worse at copying/associative recall — "Repeat
   After Me" (Arora et al., arXiv:2402.01032); the 8B hybrid study (Waleffe et al.,
   arXiv:2406.07887) prescribes interleaving full-attention. GLA's chunked form is
   plain PyTorch (~80 lines). Information: the lab's far-retrieval law (task-elicited
   retrieval heads, the late-attention slot) predicts the hybrid's attention layers
   INHERIT the seats while the recurrent layers hold context statistics — the seat
   census on a hybrid is the cleanest architecture-dependence read available.
6. **GQA / head-count at fixed params** (arXiv:2305.13245) — cheap rider on any of the
   above: the seat's head-geometry (does capacity-two live per-head or per-stream?).

## 4. THE NOVELTY SCAN (what the instruments uniquely enable)

- **Wash-retention as a cross-substrate benchmark.** Searched: comparative
  catastrophic-forgetting studies exist at TASK level (backward transfer metrics; a
  2026 comparative-empirical arXiv hit; LLM 1-7B fine-tuning forgetting studies) —
  but NO benchmark found that compares architectures/optimizers by FACT-LEVEL DECAY
  UNDER A CERTIFIED WASH (destructive fine-tuning with a replay harness). The lab's
  wash + battery + certified-replay stack is a ready-made benchmark no one else has.
  Verdict: FRONTIER-NOVEL. Concretely: one script (canon formation + wash + battery
  parameterized by optimizer/architecture/loss) = a publishable instrument.
- **The elicitation-decorrelation reader as a formation-history instrument.** No
  counterpart found: prompt-sensitivity literature measures eval variance, not
  training-path fingerprints via context-family correlation structure. Closest
  neighbors are training-data-inference and seed-variance studies. FRONTIER-NOVEL;
  the loss axis gives it a causal handle nobody else has (the constructive term).
- **Seat-occupancy ledger / capacity-two.** The frame exists — superposition and
  interference (Elhage et al., "Toy Models of Superposition", arXiv:2209.10652);
  Hopfield-capacity lineage — but no fact-level, per-slot occupancy census on a
  char-LM found. PARTIALLY-KNOWN (frame) / the instrument is new.
- **Optimizer x memory-grammar.** "Fantastic" (2509.02046) benchmarks SPEED only; no
  work found crossing optimizer identity against forgetting/seat/wash measures. e347 +
  the Muon cell would be the first data points. FRONTIER-NOVEL.

## 5. RECOMMENDED FIRST WAVE after e347 (3-4 cells, information-ordered)

1. **THE MUON CROSS (e350-candidate, GPU):** canon formation + kill bracket + seat
   probe under Muon (matrices) + AdamW (embeddings/head), lr mini-sweep; registered
   predictions: (a) the low-rank formation block (e273's Adam-normalization finding)
   does not survive spectral orthogonalization; (b) the S1 cold-shock class weakens or
   changes sign-structure; (c) Law 2b's bracket re-expresses in Muon update units.
   ~30-40 GPU-min + CPU probes. Dev ~half-day.
2. **e348 AS QUEUED, with two riders added before dispatch (executor's call):** the
   NoPE arm (position ladder — the wpe-keying question) and the QK-norm arm (the
   sink-formation question). Both are ~10-30 lines inside e348's existing
   parameter-matched protocol; both convert e348 from "shape ladder" to "shape +
   geometry ladder" without a new cell.
3. **e349 AS QUEUED, with the R-Drop arm added (the two-sided design):** constructive
   term (VICReg/DeCov-style covariance penalty across elicitation contexts) vs R-Drop
   (consistency) on the SAME decorrelation instrument — opposite predicted signs in
   one cell is the strongest formation-dynamics test the loss axis can buy.
4. **THE SUBSTRATE BENCH (CPU-first):** the wrapper that runs wash+battery+ledger
   under any {optimizer, architecture, loss} — mostly composing certified tools
   (x66's sweep machinery + e347's dose normalization + the battery). Zero new
   science, but it is the novelty-scan deliverable: the first fact-level
   forgetting-under-wash substrate benchmark. CPU-feasible at 0.86M arms.

Sources: arXiv 2412.16793, 2502.16982, 2509.02046, 2405.15682, 2409.03137, 2302.06675,
2409.11321, 2305.14342, 1906.02629, 1701.06548, 1510.02468, 2103.03230, 2105.04906,
2106.14448, 1708.02002, 1503.02531, 2305.19466, 2104.09864, 2108.12409, 2010.04345,
2302.05442, 2405.09818, 2402.02622, 2410.17816, 2101.03961, 2212.05055, 2402.01032,
2406.07887, 2305.13245, 2209.10652, 2405.20233; MLCommons AlgoPerf 2024 results;
Liu & Yao 1999 (Ensemble learning via negative correlation, Neural Networks 22(9)).
