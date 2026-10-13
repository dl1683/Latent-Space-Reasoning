**CONTROL-DESIGN-PASS — empirical design only. Construction CAN and empirical outcomes remain unadmitted.**

I read the constitution, full dual-loop skill, prescribed startup authority, current state, complete control protocol, final construction/reference/submission/receipt, frozen model/protocol/model review, and launcher helpers. All 16 construction-receipt rows independently match their recorded sizes and hashes. The following identities also matched on final recheck:

```text
Construction
38619efa3c0568e4f11575333c28fee34912aa8c1cbb6ca4de16d947b7385ec8
Reference
f4df03c354829b09e1acf504f2bde601142b8ea716d0f9525e9510d8d87aa645
Construction receipt
c1a69cbfc61af99fa41a23bf5a6d3d451e58d935fff92d61062d77cb9fd3d456
Control protocol
84080b7baf4dc42885d59974c303e7222a66d49173f40284812af98f6fa2e60c
Reviewed helper source
59974ac0015a71cc40bb754469bb4d5c401386772b6c8b470f0a80c4b13df10f
```

This is a static adversarial design review. No construction commands, workloads, solver, enforcement tests, agents, GPU, Git subprocesses, or writes were executed.

**1. Frozen coverage survives the attack.**

The four small tuples require respectively 2, 2, 32, and 216 independent raw/history executions: **252 executions, 746 FLIPs, and 4,286 QUERY calls** under the prescribed initial, post-FLIP, and two terminal passes. The length-zero history is one empty word. Enumerating every length-`H` word covers every shorter word as a prefix.

Each execution must start with fresh zero DATA and zero wear. The `N=6,k=3,L=4` tuple is outside the primary real-log domain and remains a fallback diagnostic. Its success cannot enlarge that domain; its failure there is not a primary-domain counterexample.

Independent integer arithmetic gives the trie dimensions:

| N | k | L | v | b | M | H=ML |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 4096 | 169 | 64 | 271 | 4 | 676 |
| 16384 | 8192 | 196 | 68 | 288 | 8 | 1568 |
| 32768 | 16384 | 225 | 72 | 305 | 16 | 3600 |
| 65536 | 32768 | 256 | 76 | 322 | 32 | 8192 |

The two registered pairings therefore require **28,072 trie FLIPs**. They are eight cases, not a cross-product of both raw vectors with both histories.

The named events have concrete witnesses within these frozen workloads, conditional on the reference executing its specified transitions:

- `N8192`, PATH epoch `E=0`, update 3 copies a full leaf box. After update 4, the old leaf copy is garbage and the active leaf has parity zero. Epoch `E=1`’s first scan encounters both the garbage skip and the active zero-parity fold, including an unchanged base WRITE.
- `N16384`, PATH epoch `E=2` begins `[0,1,2,4]` and then repeats 4. Updates 6 and 8 copy the leaf; update 8 also copies its full-box internal ancestor.
- `N65536`, PATH epoch `E=1` contains the complete critical-position chain. QUERY of key 0 after its first 16 insertions exercises the full chain.

These are static workload witnesses, **not observed coverage**. Actual event counts and trace witnesses remain mandatory. Missing named coverage is `INCOMPLETE/CONTROL-VOID`; no replacement workload is authorized.

For every registered trie `k`, `65537 mod k = 1`. SPREAD-AND-REUSE consequently uses consecutive first-half keys and repeats them in the second half. Every touched key has final epoch parity zero. This supplies deterministic insertion/reuse and zero-parity-fold controls; it supplies no randomized, uniform, full-keyspace, or arbitrary-history coverage.

**2. Freeze the exact ordering before smoke.**

Implement the [control protocol](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_COMPACT_CONTROL_PROTOCOL.md:22) with these operational conventions:

- Small tuples run in listed order; raw words and FLIP words use a documented deterministic enumeration.
- Trie cases run by increasing `N`, PATH followed by SPREAD.
- Let `E` be the zero-based completed-command epoch, `t=0..M-1`, `m=M/2`, and `chain=[0,1,2,4,…,2^(d-1)]`. PATH mode 2 repeats `chain[(m-1) mod len(chain)]` after its first half. All modes restart their within-epoch position at zero.
- Deduplicate the four routine query keys in first-occurrence order. At positions `1,m,M-1,M`, append the ascending unique touched-key union for the executing epoch and its immediately previous epoch. These are additional queries; retain overlaps with the routine list.
- At position `M`, that touched-key union belongs to the just-completed epoch, despite ROLE already exchanging. Preserve both final ascending all-coordinate passes.

Record exact generated query counts before execution. These conventions resolve ordering without adding configurations or changing horizons.

**3. Oracle and cold-interface obligations are mandatory.**

Maintain an independent logical vector initialized directly from raw input. A requested FLIP changes **only that coordinate** by XOR with one. Expected QUERY answers come exclusively from this vector. Neither trie decoding, base folding, controller output, nor physical DATA may update it.

Use one controller and one reused reference tape per raw/history execution or trie case. Check tape identity, length, and every tape bit after INIT and every command, including terminal queries. The harness must inspect cleanup, never perform cleanup on the controller’s behalf. Controller-retained fields must remain parameter-only apart from that zero tape.

Pass a fresh facade exposing only `read(address)->bit` and `write(address,bit)->None`. Raw input is supplied only to INIT; later controller calls receive only their requested index and ports. Harness history, epoch, expected vector, counters, diagnostics, and prior outputs stay outside the controller.

These checks establish reference behavior only. Python heap, frame disposal, tape size, and source inspection do not establish ordinary-machine `S`, complete control/head cleanup, or preparation bit-work.

**4. Port accounting must include traffic and intermediate wear.**

Count every literal READ and WRITE, including repeated acquisitions, zero writes, and unchanged writes. INIT contributes its actual changes to the same individual-cell wear ledger.

Validate addresses and supplied bits before access. An opposite write with exhausted wear invalidates execution and aborts the command; do not clamp, saturate, return status, or permit continuation. Keep a sticky invalidity record so an exception cannot be converted into an admitted controller result.

QUERY must have zero DATA writes. Report INIT traffic separately. For each parameter tuple, compute observed maxima independently:

```text
max READ_Q <= 2^20 q^3
max READ_F + max WRITE_F <= 2^20 q^3
```

The second expression uses two separate maxima, potentially from different executions. `max(READ_F+WRITE_F)` is insufficient.

Check the direct branch’s actual base-only layout separately; do not reserve ROLE/STARTED cells there. For trie cases, independently verify `k+2+2Mb<=N` and the complete field layout.

**5. Observer and phase accounting must be explicit.**

Implement observer traversal independently from the reference’s decoder. Count every backing-DATA observation separately from controller probes; count tape observations separately as well.

At boundaries, verify full publication prefixes, allocator `D<=3j`, pointer validity, unique current parents/leaves, absence of cycles, routing constraints, increasing critical positions, and exact touched-key/parity agreement—including retained parity-zero leaves. Traverse the latest current root and frozen OLD final root. Earlier NEW root records are publication records, not historical snapshots.

Capture ROLE at command entry. For a FLIP, classify maintenance using `n=j_entry+1`:

- `n<=m`: folding; OLD reads are permitted.
- `n>m`: clearing; zero acquisitions from the entry-time OLD arena.
- `n=M`: still clearing under the entry-time roles, even though roles exchange before return.

A post-fold QUERY at `j=m` must acquire zero OLD bits. After exchange, QUERY at `j=0` may read the newly frozen OLD tree.

Observer traversal of frozen OLD is permitted only before its first clearing write. Cache any required independent active-leaf information then; do not traverse partially cleared OLD afterward. Verify clearing through the write trace and subsequently inspect the cleared arena as NEW after exchange.

Record publication-last ordering, immutable-field protection, single-use boxes, full-box copies, ancestor copies, garbage skips, and active folds from literal traces and independently decoded state—not controller-supplied diagnostic labels. Complete every exchange, including epoch `L`: ROLE must accumulate exactly `L` changes and end at `L mod 2`; STARTED changes once.

**6. Resource intent is sound, with implementation checks owed.**

The frozen [helper source](<LOCAL_PROJECTS>/new-computation-model/experiments/run_endurance_tiny.py:64) sets affinity to at most four permitted CPUs, aggregate job memory to 8 GiB, active processes to two, and kill-on-job-close, without breakaway permission. Its identity matches the prior reviewed helper. That historical review is provenance; this turn performed no live enforcement test.

The new launcher must bind the job before substantive preparation, keep its handle alive, and execute cases serially in **one inherited child**. Launcher plus child consume the two process slots. No worker pool, native Git fallback, detached process, or per-case budget reset is allowed.

Reuse only the reviewed helper functions; do not invoke the old launcher main or solver. One implementation trap needs explicit handling: `git_head()` derives `ROOT` from its module location. Loading a helper snapshot unchanged would point it at the snapshot tree. Bind its metadata root correctly in the parent or load the digest-verified original helper. Do not manufacture snapshot Git metadata or fall back to native Git.

Start the aggregate 3600-second deadline before preparation and keep it effective through hashing, output publication, and launcher receipt completion. Separately bind parent-observed child exit and root-observed launcher exit. Any external root receipt must also satisfy the aggregate deadline; a cancelled launcher timer alone cannot certify that later publication.

Freeze executable/runtime/environment and all source identities, including helpers. The child must load repository algorithm/driver/monitor code exclusively from the snapshot and write outcomes to the explicit confined results path. Enforcement, import, identity, or publication failure is VOID.

**7. Admission and next steps remain narrowly bounded.**

Preserve complete attempt identities, command/query ordering, outcomes, controller and observer counters, intermediate write/change evidence, event witnesses, final DATA, and per-cell final wear. Stream evidence within the same resource envelope. Independent raw admission must replay every finite execution or independently validate every emitted output/change sequence and the frozen coverage; summaries alone are insufficient.

Allowed next work is driver/monitor/launcher implementation, followed by exactly the four-small-plus-two-epoch smoke, fresh actual CODE review, fixed FULL suite, and independent complete raw admission. This design PASS authorizes no FULL launch now and grants no `CONTROL-PASS`.

A valid finite counterexample is `CONTROL-FAIL` at its exact scope. Missing coverage, caps, compromised monitoring, or incomplete evidence is `CONTROL-VOID`. Construction-source defects consume the construction unit’s existing ONE repair allowance. Monitor/runner repairs require regates and cannot weaken source, cases, horizons, or requirements.

No construction CAN, universal correctness, ordinary-space/preparation certificate, H4 achievement, solver result, grid result, practical performance, novelty, incumbent delta, or terminal inference is admitted. **No construction repair is required or consumed by this design review.**