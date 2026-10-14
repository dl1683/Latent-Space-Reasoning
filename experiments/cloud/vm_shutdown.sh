#!/bin/bash
# Pending receipt + separately published finalized receipt; failed always wins.
set -uo pipefail
OUT=${LSR_OUT:-/opt/lsr/out}
mkdir -p "$OUT" || exit 1
for field in job created bundle_hash run shard attempt shard_hash; do
    case "$field" in job) key=LSR_JOB;; created) key=LSR_CREATED_EPOCH;; bundle_hash) key=LSR_BUNDLE_SHA256;;
        run) key=LSR_RUN;; shard) key=LSR_SHARD;; attempt) key=LSR_ATTEMPT;; shard_hash) key=LSR_SHARD_SHA256;; esac
    if [[ -z "${!key:-}" && -f "$OUT/runtime_$field" ]]; then
        value=$(cat "$OUT/runtime_$field") || exit 1
        printf -v "$key" '%s' "$value"
    fi
done
# Persisted fields are shared with startup and verified runtime; reject malformed arithmetic.
for field in LIFETIME_SECONDS WORK_STOP_SECONDS SHUTDOWN_RESERVE_SECONDS CREDIT_GUARD_EPOCH; do
    key=LSR_$field
    if [[ -z "${!key:-}" && -f "$OUT/runtime_$field" ]]; then
        value=$(cat "$OUT/runtime_$field") || exit 1
        printf -v "$key" '%s' "$value"
    fi
    value=${!key:-}
    [[ "$value" =~ ^[1-9][0-9]{0,9}$ ]] || exit 1
done
[[ "$LSR_LIFETIME_SECONDS" == 3600 || "$LSR_LIFETIME_SECONDS" == 10800 || "$LSR_LIFETIME_SECONDS" == 14400 ]] || exit 1
(( LSR_WORK_STOP_SECONDS + LSR_SHUTDOWN_RESERVE_SECONDS == LSR_LIFETIME_SECONDS && LSR_SHUTDOWN_RESERVE_SECONDS >= 900 && LSR_CREDIT_GUARD_EPOCH <= 1792195200 )) || exit 1
# Fresh preemption hook stops the recorded work session before any storage call.
if [[ -f "$OUT/runtime_work_pgid" ]]; then
    PGID=$(cat "$OUT/runtime_work_pgid") || exit 1
    if ! [[ "$PGID" =~ ^[1-9][0-9]{0,8}$ ]]; then exit 1; fi
    kill -KILL -- "-$PGID" 2>/dev/null || true
fi
NOW=$(date +%s)
CREATED=${LSR_CREATED_EPOCH:-0}
RC=${LSR_EXIT_CODE:-1}
REASON=${LSR_END_REASON:-preemption}
case "$REASON" in bootstrap|deadline|shutdown|preemption) ;; *) REASON=shutdown;; esac
if ! [[ "$RC" =~ ^(0|[1-9][0-9]{0,2})$ ]] || (( RC > 255 )); then RC=1; fi
if ! [[ "$NOW" =~ ^[1-9][0-9]{0,9}$ ]]; then NOW=0; RC=1; fi
if ! [[ "$CREATED" =~ ^[1-9][0-9]{0,9}$ ]]; then CREATED=0; RC=1; fi
if (( CREATED + LSR_LIFETIME_SECONDS > LSR_CREDIT_GUARD_EPOCH )); then RC=1; fi
if (( CREATED > NOW )); then CREATED=0; RC=1; fi
BUNDLE=${LSR_BUNDLE_SHA256:-}
[[ "$BUNDLE" =~ ^[0-9a-f]{64}$ ]] || { BUNDLE=''; RC=1; }
RUN=${LSR_RUN:-}; SHARD=${LSR_SHARD:-0}; ATTEMPT=${LSR_ATTEMPT:-}; SHARD_HASH=${LSR_SHARD_SHA256:-}
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { RUN=''; RC=1; }
[[ "$ATTEMPT" =~ ^[A-Za-z0-9_-]+$ ]] || { ATTEMPT=''; RC=1; }
[[ "$SHARD" =~ ^(0|[1-9][0-9]{0,8})$ ]] || { SHARD=0; RC=1; }
[[ "$SHARD_HASH" =~ ^[0-9a-f]{64}$ ]] || { SHARD_HASH=''; RC=1; }
# Worker has been stopped above. Bind these final bytes BEFORE payload custody;
# neither this shell nor the worker may subsequently modify runtime_receipt.json.
RUNTIME_HASH=''
if [[ -f "$OUT/runtime_receipt.json" && ! -L "$OUT/runtime_receipt.json" ]]; then
    RUNTIME_HASH=$(sha256sum "$OUT/runtime_receipt.json") || RC=1
    RUNTIME_HASH=${RUNTIME_HASH%% *}
fi
[[ "$RUNTIME_HASH" =~ ^[0-9a-f]{64}$ ]] || { RUNTIME_HASH=''; RC=1; }
LEFT=$((CREATED + LSR_LIFETIME_SECONDS - NOW - 35))
(( LEFT <= LSR_SHUTDOWN_RESERVE_SECONDS - 40 )) || LEFT=$((LSR_SHUTDOWN_RESERVE_SECONDS - 40))
if (( CREATED == 0 || LEFT <= 0 )); then RC=1; REASON=deadline; LEFT=0; fi
if [[ "${LSR_JOB:-}" != gs://*/shards/* ]]; then RC=1; LEFT=0; fi
write_receipt() {
    local target=$OUT/shutdown_receipt.json
    if ! printf '{"status":"%s","reason":"%s","rc":%d,"finalized":%s,"identity":{"run":"%s","shard":%d,"attempt":"%s"},"hashes":{"bundle_sha256":"%s","shard_sha256":"%s","runtime_receipt_sha256":"%s"},"clock":{"created_epoch":%d,"now_epoch":%d,"work_stop_epoch":%d,"lifetime_end_epoch":%d,"reserve_seconds":%d}}\n' \
        "$1" "$REASON" "$RC" "$2" "$RUN" "$SHARD" "$ATTEMPT" "$BUNDLE" "$SHARD_HASH" "$RUNTIME_HASH" "$CREATED" "$NOW" "$((CREATED+LSR_WORK_STOP_SECONDS))" "$((CREATED+LSR_LIFETIME_SECONDS))" "$LSR_SHUTDOWN_RESERVE_SECONDS" > "$OUT/shutdown_receipt.tmp"; then return 1; fi
    if ! mv "$OUT/shutdown_receipt.tmp" "$target"; then return 1; fi
}
fallback() {
    RC=1; REASON=receipt
    printf '{"status":"failed","reason":"receipt","rc":1,"finalized":false,"hashes":{},"clock":{}}\n' > "$OUT/shutdown_receipt.json" || true
}
STATUS=failed
[[ "$RC" != 0 ]] || STATUS=completed
CUSTODY_PID=
cleanup_custody() {
    if [[ -n "$CUSTODY_PID" ]]; then
        kill -KILL -- "-$CUSTODY_PID" 2>/dev/null || true
        wait "$CUSTODY_PID" 2>/dev/null || true
    fi
}
trap cleanup_custody EXIT
trap 'exit 1' TERM INT
custody() {
    local code=0
    setsid timeout --kill-after=5 "$LEFT" "$@" &
    CUSTODY_PID=$!
    wait "$CUSTODY_PID" || code=$?
    cleanup_custody
    CUSTODY_PID=
    return "$code"
}
write_receipt "$STATUS" false || { fallback; exit 1; }
(( LEFT > 0 )) || exit 1
if ! custody gcloud storage rsync -r "$OUT" "$LSR_JOB/out"; then
    RC=1; REASON=custody
    write_receipt failed false || fallback
    exit 1
fi
# Publish the final receipt alone after payload custody. Partial rsync is not final.
write_receipt "$STATUS" true || { fallback; exit 1; }
LEFT=$((CREATED + LSR_LIFETIME_SECONDS - $(date +%s) - 35))
(( LEFT <= LSR_SHUTDOWN_RESERVE_SECONDS - 40 )) || LEFT=$((LSR_SHUTDOWN_RESERVE_SECONDS - 40))
if (( LEFT <= 0 )) || ! custody gcloud storage cp "$OUT/shutdown_receipt.json" "$LSR_JOB/out/shutdown_receipt.json"; then
    RC=1; REASON=custody
    write_receipt failed false || fallback
    exit 1
fi
exit "$RC"
