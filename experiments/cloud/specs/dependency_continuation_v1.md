# Execution spec: dependency-continuation memory, stage E1/E2 (DESIGN)

2026-10-14. Status: **DESIGN ONLY.** This document specifies the exact CPU
instrument and the design of the E1/E2 model stages. An E1 runner prototype
(`experiments/cloud/lsr_jobs/e1_ledger.py`, untracked) exists but is NOT-READY
(`docs/council_2026_10_15_e1_runner_codex.md`); the 2026-10-14 council decision
is recorded in section 4A and governs where it conflicts with sections 3 to 7.
Nothing here is registered or ready. Owner authorization (2026-10-15) allows
cloud GPU experiments on compute credit with caps removed, but every entry still
needs its smoke, pins, stop conditions and a LAUNCHABLE review. Stem
`google/gemma-4-31B` revision `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89` is
pinned by amendment G4; the extraction layer, dependency lock, image, region
and per-attempt deadline are still to be pinned by the execution agent
(`TO_BE_PINNED_BY_EXECUTION_AGENT`) and listed in the queue.

Selection grounding: `internal/analysis_2026_10_14/zcode_research_brief.md`
(owner execution assignment) and
`internal/analysis_2026_10_14/droid_r4_cloud.md` (Droid round 4: earn
sufficiency before cheaper continuation). Program contracts:
`theory/RESEARCH_PROGRAM.md` ("How we choose the next work"); current/history
view identities: `docs/UNIFIED_EVIDENCE.md`. This revision incorporates the
2026-10-14 three-subagent adversarial review (code correctness, brief
compliance, instrument-strength/confound hunt).

## 1. Scientific aim

Distinguish **equal present answers whose consequences separate after a later
rule or correction**. Two worlds whose current answer is identical ("does `s`
depend on `t`?" → yes) must diverge after the same retraction, because their
*dependency structure* differs. This operationalises the response-versus-
susceptibility lesson of `theory/CORRECTION_READY_RESPONSE.md` (preserving
today's answer does not preserve intervention consequences) without importing
its SPD guarantees, and replaces the failed pooled/compressed-state interface
(`experiments/EXPERIMENTS.md` P1/P1′/P2: storage competence never became
update closure; geometric closeness did not become functional answers).

E1 asks: **what survives construction?** — does a pretrained representation,
frozen and indexed per record, still answer after a late policy reveal,
retraction and historical version query? E2, only after E1 competence, asks:
**can recurring backbone work disappear?** — a learned updater versus
re-encoding versus an ordinary incremental engine.

**What counts as state-attributed evidence** (fixed here, from the review):
only the paired separating queries — the pre-retraction canonical query, the
post-retraction canonical query, and the historical query at the same
post-retraction snapshot, jointly correct in both worlds. Everything else a
reader answers from granted inputs (policy tokens, action tokens, record
identities) is reader competence over metadata, not memory evidence.

## 2. Exact task instrument (implemented, validated)

- Canonical stdlib-only runner: `experiments/memory/dependency_continuation.py`
- Literal frozen config: `experiments/memory/dependency_continuation_v1.json`
- Run (CPU, deterministic — byte-identical reports across hash seeds; exit 0
  iff all checks pass):

```bash
python experiments/memory/dependency_continuation.py \
    --config experiments/memory/dependency_continuation_v1.json
```

World: ≤8 packages, ≤12 immutable directed records per episode
(`rid, src, dst, etype ∈ {hard, soft, test}, ver`), event-sourced ledger with
an explicit active view and a separately retained event history. Operations:
INGEST (records only — never queries, policies, roles or consequences;
enforced by schema check plus case-insensitive forbidden-token scan),
late POLICY reveal (allowed etype set + horizon ≤ 3), QUERY (nonempty
directed path of ≤h allowed-type edges), RETRACT/REINSERT (invalid membership
refuses; state unchanged on refusal; REINSERT bumps `ver`), historical QUERY
(labelled event version evaluated under its own supplied policy).

Validated by 15 literal fixtures in 7 clusters. Hardened against the
instrument review's shortcuts:

- **Pair discipline:** pair members share byte-identical canonical public
  prefixes AND byte-identical event scripts (expectations excluded); the
  branch records differ; survival of the canonical query after the same
  retraction differs. The prior version's event-script asymmetry (a stateless
  grammar cheat scored ~95%) is closed; the per-query majority baseline on
  the frozen set is 0.75.
- **Paired history:** both worlds are queried historically at the same
  post-retraction snapshot (world A: no, world B: yes), so "retraction label
  ⇒ no" fails; labels are opaque (v2..v6).
- **Verifiable alternates:** dead-end second out-edges (out-degree shortcut
  falsified), soft alternates under a hard policy, length-3 alternates under
  horizon 2 — world identity requires route verification under policy, not a
  single-slot feature.
- **Action variety:** retraction targets include the a→t support, the b→t
  support, the branch record and both supports in sequence (double
  retraction); reinsert-one-of-two is tested; refusals leave state unchanged.
- **Renaming:** a role-neutral renaming (permuted name tokens, no role
  letters) shares the core cluster.
- **Diagnostics:** every query carries a role (pre / affected / unaffected /
  policy_flip / post_reinsert / historical); no role, cluster or view is
  all-constant (gated); constant-answer accuracy is reported beside
  correctness (frozen set: const-yes 0.645, const-no 0.355 over 155
  queries); no-state table nulls are reported and must be imperfect
  (query-majority 0.748, conditioned on (query, policy, horizon) 0.761 —
  optimistic in-sample fits bounding only tables with exactly those keys);
  every query is cross-checked by an independent simple-path enumeration
  evaluator, and every historical query is runner-checked to address a
  version other than the current view.

This validates an instrument only; it establishes no latent memory, model
result or algorithmic novelty. The frozen fixtures are oracle validation, not
a statistical sample and not a model corpus. E1 corpus generation must
additionally: use role-neutral, length- and frequency-matched package names
with permuted renamings per cluster; vary the separating retraction target
across pair families (the core pair keeps the brief's `retract a→t` witness);
balance the pre-retraction canonical answer across pairs where the action family
allows it (both members share it — that equality is the design; for retract-only
separating pairs it is necessarily positive, see 4A); ask
historical queries only at non-current versions (so they cannot duplicate the
current answer); include paired refusal, double-action and policy-flip
families so those operations enter the paired statistic; select queries after
policy reveal; and split by cluster only.

## 3. E1 arms (DESIGN; identical episodes, byte-accounted)

| Arm | Object | Role |
|---|---|---|
| E1-X | Exact versioned graph evaluator (immutable events, latest-value index, dependency-versioned answers) spending its full byte budget on exact records | Conventional comparator and task positive control; never padded or crippled |
| E1-R | Competent full-text replay of the episode transcript | Charged upper reference |
| E1-S | Indexed frozen per-record contextual states: 12 slots × 128-dim FP16 + 1 policy slot (3,328 payload bytes; illustrative, the runner uses 6+1 = 1,792 bytes, see 4A) from one pinned layer/constructor pass | The construction under test |
| E1-T | Same-budget noncontextual token features (no contextual pass) | Separates contextual value from lexical identity |
| E1-C | Shuffled-state and omitted-state controls | Causality of the state |
| E1-N | **Conditioned no-state null:** per-(src, dst, policy, action-state) majority table fitted on training clusters | Per-role accuracy floor; it cannot distinguish pair members by construction (they share all conditioned features), so the paired separating statistic is protected instead by fresh held-out clusters and the Δ_pair gate (§4) |

**E1-S interface split (fixed after the Antigravity sparring pass).** The
frozen-slots-plus-action-tokens design hides a coherence paradox: if the
harness never touches the slots, the reader must parse action tokens and
mentally mask slots — an in-context program executor over static slots, whose
reader cannot serve as E2's snapshot reader. Both coherent versions are
declared, with different claims:

- **E1-S/A (correction-aware reader):** frozen per-record slots + policy and
  action tokens + query, one pinned constructor pass. Tests whether pretrained
  per-record states support *correction-conditioned reading*. The historical
  role here is transcript indexing (diagnostic only). A pass licenses no E2
  interface claim.
- **E1-S/M (masked-state hybrid):** the harness maintains an exact
  slot-availability/version index (charged under §5 as ordinary exact-engine
  computation); the reader sees `(state with version mask, query)` only —
  exactly E2's snapshot-reader interface, so E1 passes the baton cleanly.
  Success is a **connected hybrid** result (exact index + learned reader),
  not standalone latent memory; the mask maintenance is priced and reported.
  The historical role is meaningful here: the mask selects the queried
  version, exactly as a conventional versioned store would.

**Policy/action channel (pinned):** the reader receives the policy and each
action as **raw input tokens** — exactly the channel E2's updater receives.
For S/A these are inputs; for S/M the action is executed by the charged exact
index and the reader never sees the action string. Any other precomputed
state mask, version map or curated policy embedding is either forbidden, or
charged identically to every arm as exact engine computation. If a backbone
pass encodes the policy, that pass is charged to the arm that uses it; a
hand-written embedding makes the slot a decoy and the byte accounting must
say so. Deciding which slots a retraction masks **is** revision computation.

Readers: bounded attention readers over the slots, trained on training
clusters, selected on development clusters. Exact-input readers must execute
the **complete** late-policy / correction / history task, not fact retrieval.
E2's snapshot reader is included on reconstructed states as a diagnostic from
the start (Droid r4).

`TO_BE_PINNED_BY_EXECUTION_AGENT`: stem model + revision (open weights; any
owner-approved stem — the non-Chinese / no-Gemma / no-OpenAI exclusions and
the frozen model panel come from a separate still-unapproved proposal and a
closed historical registration respectively, and are **not inherited** by
E1/E2 unless the owner adopts them; see `theory/FIRST_CONSTRUCTION_PROPOSAL.md`,
`theory/LATENT_BRANCH_CAPACITY.md`), extraction layer, tokenizer/template,
hardware/region, spend ceiling. No command in this spec pretends an unbuilt
adapter exists. Extraction/custody adapts `experiments/cloud/lsr_jobs/bindq.py`
(record positions, token alignment asserts, fresh identities, no KV
retention); scoring adapts `readout.py`; the recurring-backbone pattern of
`branch.py` (update/read call the model again) is rejected for E2 claims.

## 4. Statistics and gates (proposed; freeze before any run)

Reconciliation (2026-10-14; RESOLVED by the council in 4A below; the paragraph is kept as history): a concurrent, unregistered prototype
by another worker (`experiments/cloud/lsr_jobs/e1_ledger.py`, config
`e1_gemma4_smoke.json`; not this spec's product) implements a different
contract — exact action flags, learned policy embeddings and a narrower
fixed-target family. Droid r4 and the queue keep 64 held-out clusters with
per-view/joint-0.90/replay-deficit gates; this spec proposes ≥128 clusters
with Δ_pair. Root/Droid choose one contract before any model work freezes;
nothing below is accepted until then.

- Unit of resampling: **base-world cluster** (pair, renaming, policy and
  trajectory stay together); seeds are not tasks. Clustered percentile
  bootstrap (10k resamples, seed frozen); report BCa intervals as diagnostic.
- **Primary statistic — Δ_pair (recalibrated after the Antigravity pass):**
  per pair, Δ = I(B post-retraction correct AND A post-retraction correct
  with their differing expected answers) on the canonical separating query.
  A genuine route-verifying system has E[Δ]=1; any world-blind system
  (table null, constant answer, ingest-independent heuristic) has E[Δ] at
  its guessing rate. Gate: mean Δ_pair ≥ 0.85 with one-sided 95% clustered
  LCB ≥ 0.75 over ≥128 held-out clusters. The all-correct binary per cluster
  on the full triple (pre / post / historical at the same non-current
  snapshot, both members) is the secondary statistic, reported with its
  interval but not gated at the 0.90 ceiling (the previous 62/64 step
  function rejected competent systems over 2–3 noise failures).
- **Secondary gates:** unaffected-preservation and historical-at-non-current-
  version accuracy each ≥ 0.90 one-sided 95% LCB; all other roles (pre,
  affected, policy_flip, post_reinsert) reported with intervals, not gated.
  The replay deficit (memory arm vs competent full replay) is reported with
  its interval; ≤ 0.05 remains the target Droid r4 proposed, not a hard
  gate. Constant-answer and E1-N table-null baselines are reported beside
  every arm and every role.
- **Fresh-context rule:** every reader answers each query from a fresh
  context — its own earlier predictions and query outputs are never fed
  back, so pre-retraction queries cannot leak world identity through the
  answer channel to later queries.
- Termination/diversity: structured outputs first; any generative readout
  reports termination (≥95% to interpret), empty and constant-answer rates,
  and fixed-input noise + batch-shape checks before difference claims.
- Replay competence is established on training clusters **first**; a replay
  model that is simply weak routes to a different remedy (instrument/task
  escalation or model change) than an instrument defect, and a replay arm
  may itself use transcript shortcuts — its deficit comparisons are against
  a possibly-shortcutting upper reference and are labelled as such.
- Consequences of failure: exact-input or replay failure after competence =
  instrument repair (no claim). Competent controls pass + memory arm fails =
  **one** addressing/constructor remedy, then preserve the negative. No
  capacity impossibility is inferred; pass/fail changes constructor/updater/
  routing decisions only.

## 4A. Council decision (2026-10-14 13:05 UTC) -- supersedes the "reconciliation pending" status above

Transcripts: `internal/supervision/council_20261014T1226Z/` (Codex r1, real Droid r2, Codex r3). Everything below is **PROPOSED**; nothing is registered, and no result exists. Thresholds freeze only after the calculations marked CALC are done and before any confirmation data is touched.

**Interface (adopted): S/M.** `construct -> update -> snapshot(version) -> read(snapshot, query)`. The reader sees the snapshot state and query only: no raw ledger, no stem, no action string, no effective-retraction flags, no requested-version argument, no old answers. Inactive records are excluded from attention by a mask (zeroing features is insufficient because position embeddings and projection biases remain). Policy and horizon are encoded into the snapshot by a declared, charged encoder (the policy slot is filled at reveal time; a hand-written lookup would make it a decoy and must be charged as exact-engine computation). Current (v3) and historical (v2) snapshots are selected at the same post-action execution time, outside the reader. The exact index that maintains membership and version is charged ordinary computation. S/M is a connected hybrid; it claims neither learned revision nor causal erasure (later contextual slots still carry residue of retracted records) nor pretrained-computation reuse. **S/A** (raw action/policy tokens to the reader) is an optional diagnostic that locates a transition bottleneck; it is not a prerequisite for E2, and the existing flag-conditioned prototype may be retained only as an explicitly labelled S/A-style diagnostic.

**Slots.** The runner uses 6 record slots + 1 policy slot of 128-dim FP16 (1,792 payload bytes); the 12-slot / 3,328-byte figures in section 3 and the queue were illustrative (Droid r4) and are superseded unless the repaired family needs more records, in which case the declared schema and byte charge change before freeze (never truncate a route to keep six). The exact comparator buffer is 36 bytes plus dictionary: the learned payload is about 49.8x larger, so this is not a compression result and cannot support "beats storage-only" economics on a six-record task.

**Samples.** 64 independent base-world clusters = diagnostic pilot only; 128 fresh clusters from a frozen, shortcut-repaired family mixture = confirmation. Pair members, variants, policies and trajectories never become independent rows. 128 pooled is not 128 per family; family-specific claims need family-specific sizes (CALC).

**Primary promotion statistic (confirmation).** One registered canonical trajectory per cluster. A cluster succeeds when both pair members are correct at the pre-retraction, post-retraction and historical (non-current) reads: six decisions. Mean >= 0.90 and one-sided 95% exact-binomial LCB >= 0.85, which requires 116/128 (60/64 in the pilot). Exact operating characteristics (Codex r3, exact binomial summation): pass probability at true cluster-trace success 0.85 / 0.90 / 0.95 / 0.97 = 2.8% / 22.1% / 78.4% / 95.7% (64 clusters) and 4.3% / 48.1% / 98.8% / 99.99% (128). With independent per-decision accuracy 0.97 the trace success is 0.833 and the gate passes ~1.3%; at 0.99 per decision, 68% / 96%. Reading: 128 is adequate for a competent artifact with ~0.95 whole-trace success; it is not a power guarantee for anything weaker, and coherent versus isolated errors behave differently.

**Diagnostics and secondary gates.** Canonical post-action pair correctness (the old Delta_pair): mean >= 0.85, one-sided LCB >= 0.75; passing it alone does not promote the receiver. Unrelated and non-current-history preservation: one-sided LCB >= 0.90 on fixed auxiliary queries (CALC on the actual fractional scores). Deterministic legal/refused-action semantics, version selection and refusal-state preservation: exact. Exact-input and reconstructed-snapshot learned receiver: >= 0.95 per required role, with the intervention predictions, labels and scores attributed separately (not the original metrics). Percentile bootstrap stays for fractional diagnostics; binary cluster outcomes use the exact binomial bound.

**Required causal control: frozen-reader within-pair donor swap.** Freeze the successful reader; swap all record-content slots between the two members of a pair while preserving identical keys, masks, policy and every other permitted input, in both directions; score against the donor world's consequences. Proposed donor-follow mean >= 0.90, LCB >= 0.85, with recipient accuracy and omission on the frozen reader reported separately (CALC). Retrained omitted, shuffled and noncontextual arms remain as alternative-system controls and do not substitute. Falsifier thresholds still open (Droid D2): mask-only null (proposal <= 0.30) and active-slot permutation collapse (proposal <= 0.35).

**Replay.** Per-view development competence >= 0.95 is instrument validity for any replay-relative statement, with replay selected independently on development data (not at the memory-selected layer; full-text final-layer decoding or a bounded supervised reader is admissible). The replay deficit (paired upper bound <= 0.05) gates a replay-quality replacement claim only; it does not veto an operation-adding construction that meets its own declared quality and beats the strong structured comparator (theory/RESEARCH_PROGRAM.md:318-323,380-386).

**Noise.** Proposed: 100% decision agreement on a finite development preflight set across repeat, singleton and reordered batch execution, covering ledger slots, isolated name keys, replay features and the combined deployed input path; finite features and logits throughout; raw differences reported. Failure changes batch policy or precision and is not evidence against memory.

**Generator requirements added by the council.** (1) Remove the source-out-degree shortcut (e1_ledger.py:120-122,132-169): degree-matched dead-end, wrong-type and too-long alternate routes, with independent path enumeration for every label and an assertion that out-degree-only and conditioned-table predictors cannot separate matched pairs; (2) vary the structural retraction target (not always the a->t bridge); (3) CURRENT and HISTORICAL must differ in snapshot and label at the same post-action time; (4) include at least one pre-existing older-snapshot fixture if birth-version filtering is claimed, otherwise narrow the claim; (5) **retract-only separating pairs cannot balance the pre-retraction canonical answer**: under monotone reachability deletion, equal pre-answers for a separating pair force both to be positive, so section 2's balance requirement applies only to a second action family (reinsert) and the retract-only family states that its canonical pre-answers are positive; (6) policy replacement and reinsert get their own registered intervention checks and quality criteria before any readiness claim, and a retract-only pass licenses retract-only progression.

**Open calculations (Codex, in memory, <= 5 CPU minutes each, before freeze):** operating characteristics of the donor-swap gate, the fractional preservation gates and the replay-deficit gate on their actual cluster score definitions; per-family sample sizes; the finite proof that snapshot(T_a X, v) equals U_a(snapshot(X, .)) under declared version and history resources; the deterministic world-blind null for identical permitted non-state inputs.

**E2 dependency.** E2 via a frozen reader needs a competent S/M receiver; a jointly constructed route (for example Q1, `quadratic_continuation_cpu_v1.md`) need not wait for this benchmark. Horizon 4 only after the exact oracle's cap is extended and the dual-evaluator fixtures re-run at h=4.

## 5. Byte and cost accounting (reported per arm)

Payload (slots × dim × precision), token/dimension dictionaries, slot masks,
record addresses, version counters, source custody pointers, and any harness
computation performed on the arm's behalf (policy encoding, mask/version-map
construction — charged or forbidden per §3); ingestion (backbone passes,
tokens), construction, readout, any replay/fallback, and — for E2 —
maintenance and historical-recovery access. Full contextual sequences are
richer diagnostics, not equal-byte evidence.

## 6. E2 (DESIGN; only after E1 competence)

Question: can recurring backbone work disappear? Freeze the competent E1
snapshot reader. The updater receives the retained state and a declared
action **as input tokens** (same channel as E1's reader); the snapshot reader
then receives **the updated state and the query only** — no correction
payloads, action logs, old answers or history bypass. Controls: shuffled
action, no-op action, state swap. Operations: retract, reinsert, policy
replacement; horizons 1/2/3, plus horizon 4 only after the exact oracle's
validated horizon cap is extended and the dual-evaluator fixtures re-run at
h=4; queries selected after updating; compositions held out. Comparators
with identical access: actual re-encode, learned updater, ordinary
incremental evaluator. Pass requires E1 quality + paired action dependence +
lower complete cost at a declared reuse count (training, ingestion, action
encoding, maintenance, readout, fallback). Quality failure changes the
updater/objective; cost failure favours reconstruction or the ordinary
engine. Success admits this artifact, not an economic law.

## 7. Cheapest separate cloud smoke (PROPOSAL, not authorized)

256 training / 64 development / ≥128 held-out clusters (raised from Droid
r4's 64 for Δ_pair power), five packages / six records, one pinned approved
stem/layer, cached features, small readers; readers trained on training
clusters and selected on development clusters.
Preconditions: paired traces verified against the exact oracle with the §2
pair discipline at generation scale; complete-task exact-input and snapshot
competence ≥ 0.95; exact-update→snapshot-read intervention effects;
alignment, repeat and batch-shape noise checks; state-omission failure.
Requires a bounded adapter/config build, the pins listed in the queue and a
LAUNCHABLE review. Owner authorization for cloud runs on compute credit exists
(2026-10-15, caps removed); it does not make this entry ready. The first cloud
step is the separate alignment/hook smoke, `e1_alignment_hook_smoke_v1.md`.

## 8. Explicit non-claims

The task is synthetic and familiar (a dynamic dependency graph); it supplies
no algorithmic novelty. Instrument validation is not a model result. The
frozen fixtures are oracle validation, not a model corpus; E1 corpus
generation has its own requirements (role-neutral names, post-policy query
selection, cluster-level splits). E1/E2 remain DESIGN until the adapter,
pinned revisions, a separate passing smoke and execution bounds exist.
Nothing here launches training, cloud jobs, adopts a model policy, or revives
historical automation. Root reviews before any scientific admission or
publication.
