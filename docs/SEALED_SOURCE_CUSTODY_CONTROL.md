# Source custody without routine corpus re-verification

26 November 2026 — infrastructure for edge/lifecycle economics, not learned reasoning.
Reviewed against 883e89c7 and the custody discussion in issue #6 comments
7203050534 and 7203310093. No model or held task was executed.

## Concrete boundary

NativeTaskWorldPin.bind currently reads every pinned source body and checks
its entire native span partition, on every episode. This is correct for
VerifiedSourceTool's mutable filesystem leaves. Its snapshot digest identifies
the expected versions, not the actual bytes currently stored at those paths.
A cache keyed only on the manifest, inode, file size or modification time is
not a substitute for byte verification. File permissions and content-addressed
names alone do not make bytes immutable.

A possible controller-side reuse mechanism is to separate two proofs:

1. **Content proof:** trusted controller owns an immutable byte object; once
   it has verified the full SHA256 and ordered contiguous span partition,
   this content/partition proof remains valid as long as that exact immutable
   object stays in trusted custody. Different partition or changed bytes need
   new verification. Full production keys also include the schema/validation
   policy and any span coordinate mappings being certified.
2. **Epoch/publication proof:** on each attach, bind the independently selected
   source snapshot, source IDs, versions, generations, types, native span
   catalogue, derived catalogue, runtime/reader identity and sealed native
   publication to the current requested pin. Content reuse is not permission
   to reuse a previous epoch's admission, provenance, derived state, or
   task receipt. A withdrawal/reinsert may reuse a retained byte proof but
   must establish a new epoch and revalidate dependent work. Untrusted
   serialized cache entries or loss of custody require re-verification.
   Continue checking the actual native publication and model independently.

Neither proof establishes semantic truth, complete evidence discovery, or
correct derivation. Do not disable the existing full verification on the
strength of this control. All arms should receive the same trusted custody
and source verification opportunity, including warm exact and text.

## Executed OS control

Run `python -m experiments.memory.sealed_source_custody_control` on Linux
with memfd sealing support. Local standalone run, kernel 6.18.57 x86_64.
The script does not import native reader code; the source bytes are small
synthetic buffers. It owns sealed file descriptors and caches only
successful content/partition checks; no actual epoch or native publication
certificate is implemented.

| Check | Observed |
| --- | ---: |
| Same-inode, same-size, restored-mtime mutable overwrite | Undetected by that metadata key; detected by digest |
| Initial source-partition verification | 12,288 B |
| 100 repeated two-source verification calls | 0 additional source bytes |
| One 4,096-byte correction; unchanged second source | 4,096 additional source bytes |
| Historical source-partition recheck | 0 additional source bytes |
| Wrong hashes / wrong version / incomplete partition | 3/3 rejected |
| Different valid partition | Rechecked all 4,096 B |
| Writes, shrink and growth on sealed descriptor and duplicate | 6/6 blocked with EPERM |

The byte counters cover only the content verifier, not initial construction,
hashing at add, copying into memfd, the final explicit byte read in the attack
check, native publication attachment, metadata scans, memory, or time.
Memfd is volatile and retains bytes in a memory-backed filesystem; it is
not the durable edge design. A durable alternative (e.g. authenticated
immutable storage such as fs-verity under a trusted controller) needs its
own platform, persistence, threat-model and full-cost qualification.
No lifecycle advantage or actual PEP corpus speedup was measured.

For R episodes on a fixed corpus of B bytes, the current source-partition
verification reads R*B. With the above assumptions, a reused source-local
proof reads B once, then only changed/unseen byte-partition pairs; metadata,
publication checks, custody setup and evidence reopening remain additional
charges. This identifies an avoidable repeated cost conditional on a stronger
storage contract. It is not a wall-time estimate.

## Priority

The main experiment remains a chronological independent source witness
and a separately chosen late task under a frozen publication, not another
synthetic admission sweep. Compact delivery is a useful matched control:
measure actual model tokens and grounded completion only when authorized.
For now keep full verification charged in the production adapter. If the
corpus-scale comparison makes controller verification material, implement
the explicit trusted custody contract with per-epoch validation and a
separate cost ledger rather than a manifest-only shortcut.
