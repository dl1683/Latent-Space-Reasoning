#!/bin/bash
# Parent owns the work session; custody starts only after that session is dead.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl --connect-timeout 3 --max-time 10 -fsS -H 'Metadata-Flavor: Google' "$MD/$1"; }
gpu_enabled() { [[ -n "$1" && "$1" != false ]] && command -v nvidia-smi >/dev/null 2>&1; }
gpu_checks() {
    if gpu_enabled "$1"; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || return 1
        timeout --foreground 150 bash "$W/automon.sh" x 1 0 selftest > "$OUT/automon_selftest.log" 2>&1 || return 1
        grep -q 'IDLE_POWEROFF ready=0 reason=never_ready' "$OUT/automon_selftest.log" || return 1
        bash "$W/automon.sh" "$OUT/heartbeat" 30 45 > "$OUT/automon.log" 2>&1 &
    else
        echo 'CPU_BRANCH no GPU automon'
    fi
}
if [[ "${1:-}" == --branch-library ]]; then return 0 2>/dev/null || exit 0; fi
W=${LSR_ROOT:-/opt/lsr}; OUT=$W/out
export W OUT LSR_OUT="$OUT"
mkdir -p "$OUT"
work_budget() {
    NOW=$(date +%s)
    if ! [[ "$NOW" =~ ^[1-9][0-9]{0,9}$ ]]; then return 1; fi
    (( NOW >= LSR_CREATED_EPOCH )) || return 1
    LEFT=$((LSR_STOP_EPOCH - NOW))
    (( LEFT > 0 ))
}
if [[ "${1:-}" == --owned-work ]]; then
    export LSR_WORK_GROUP=1 LSR_CUSTODY_DEFERRED=1
    # Fork/scheduling latency must not admit bootstrap after the absolute cutoff.
    work_budget || exit 1
    RUN=${LSR_JOB%/shards/*}
    work_budget || exit 1
    timeout --foreground --kill-after=5 "$(( LEFT < 300 ? LEFT : 300 ))" gcloud storage cp "$RUN/bundle/code.tar.gz" "$W/bundle.tar.gz"
    printf '%s  %s\n' "$LSR_BUNDLE_SHA256" "$W/bundle.tar.gz" | sha256sum -c -
    tar -xzf "$W/bundle.tar.gz" -C "$W"
    [[ -d "$W/experiments/memory" && -f "$W/lsr_jobs/s0_runtime.py" ]]
    work_budget || exit 1
    timeout --foreground --kill-after=5 "$(( LEFT < 60 ? LEFT : 60 ))" gcloud storage cp "$LSR_JOB/shard.json" "$W/shard.json"
    cd "$W"
    export PYTHONPATH="$W${PYTHONPATH:+:$PYTHONPATH}"
    ACCELERATOR=$(md attributes/lsr-has-gpu || true)
    if ! gpu_enabled "$ACCELERATOR"; then export CUDA_VISIBLE_DEVICES=''; fi
    MIRROR=$(md attributes/lsr-weights-mirror-uri || true)
    if [[ -n "$MIRROR" ]]; then
        WEIGHTS=$(md attributes/lsr-weights-local-path)
        [[ "$MIRROR" == gs://* && "$WEIGHTS" == /opt/lsr/weights ]]
        mkdir -p "$WEIGHTS"
        LEFT=$((LSR_CREATED_EPOCH + LSR_WORK_STOP_SECONDS - $(date +%s)))
        (( LEFT > 0 ))
        timeout --foreground --kill-after=5 "$LEFT" gcloud storage rsync -r "$MIRROR" "$WEIGHTS"
        [[ -f "$WEIGHTS/config.json" ]]
    fi
    # Worker Landlock custody permits writes only beneath the attempt output.
    export HF_HOME=$OUT/offline-hf TMPDIR=$OUT/tmp
    mkdir -p "$TMPDIR"
    mkdir -p "$HF_HOME"
    gpu_checks "$ACCELERATOR"
    touch "$OUT/heartbeat"
    python3 -B -m lsr_jobs.s0_runtime --run --shard "$W/shard.json" --out "$OUT" \
        --created-epoch "$LSR_CREATED_EPOCH" --bundle-sha256 "$LSR_BUNDLE_SHA256" --job "$LSR_JOB" \
        --expected-run "$LSR_RUN" --expected-shard "$LSR_SHARD" --expected-attempt "$LSR_ATTEMPT" \
        --expected-shard-sha256 "$LSR_SHARD_SHA256"
    exit $?
fi
exec > >(tee -a "$OUT/startup.log") 2>&1
WORK_PID=; WATCHDOG=
stop_work() {
    if [[ -n "$WORK_PID" ]]; then
        kill -KILL -- "-$WORK_PID" 2>/dev/null || true
        wait "$WORK_PID" 2>/dev/null || true
    fi
    if [[ -n "$WATCHDOG" ]]; then
        kill -KILL -- "-$WATCHDOG" 2>/dev/null || true
        wait "$WATCHDOG" 2>/dev/null || true
    fi
}
finish() {
    local rc=$?
    trap - EXIT TERM INT
    stop_work
    export LSR_EXIT_CODE=$rc LSR_END_REASON=${LSR_END_REASON:-bootstrap}
    if [[ -s "$W/shutdown.sh" ]]; then
        bash "$W/shutdown.sh" || rc=1
    else
        rc=1
        printf '{"status":"failed","reason":"bootstrap_metadata","rc":1,"finalized":false,"hashes":{},"clock":{}}\n' > "$OUT/shutdown_receipt.json" || true
    fi
    shutdown -h now || rc=1
    exit "$rc"
}
trap finish EXIT
trap 'exit 1' TERM INT
LSR_CREATED_EPOCH=$(md attributes/lsr-created-at)
# Reject noncanonical/oversized text BEFORE any Bash arithmetic or persistence.
if ! [[ "$LSR_CREATED_EPOCH" =~ ^[1-9][0-9]{0,9}$ ]]; then exit 1; fi
NOW=$(date +%s)
if ! [[ "$NOW" =~ ^[1-9][0-9]{0,9}$ ]]; then exit 1; fi
if (( LSR_CREATED_EPOCH > NOW )); then exit 1; fi
export LSR_CREATED_EPOCH
LSR_JOB=$(md attributes/lsr-job); export LSR_JOB
[[ "$LSR_JOB" == gs://*/shards/* ]]
LSR_BUNDLE_SHA256=$(md attributes/lsr-bundle-sha256); export LSR_BUNDLE_SHA256
[[ "$LSR_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]]
printf '%s\n' "$LSR_JOB" > "$OUT/runtime_job"
printf '%s\n' "$LSR_CREATED_EPOCH" > "$OUT/runtime_created"
printf '%s\n' "$LSR_BUNDLE_SHA256" > "$OUT/runtime_bundle_hash"
md attributes/shutdown-script > "$W/shutdown.tmp"
mv "$W/shutdown.tmp" "$W/shutdown.sh"
# Persist independent launch authority for a fresh shutdown hook. The dispatcher
# must transport this journal hash; missing metadata is a terminal refusal.
LSR_RUN=$(md attributes/lsr-run); export LSR_RUN
LSR_SHARD=$(md attributes/lsr-shard); export LSR_SHARD
LSR_ATTEMPT=$(md attributes/lsr-attempt); export LSR_ATTEMPT
LSR_SHARD_SHA256=$(md attributes/lsr-shard-sha256); export LSR_SHARD_SHA256
[[ "$LSR_RUN" =~ ^[A-Za-z0-9_-]+$ && "$LSR_ATTEMPT" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$LSR_SHARD" =~ ^(0|[1-9][0-9]{0,8})$ && "$LSR_SHARD_SHA256" =~ ^[0-9a-f]{64}$ ]]
printf '%s\n' "$LSR_RUN" > "$OUT/runtime_run"
printf '%s\n' "$LSR_SHARD" > "$OUT/runtime_shard"
printf '%s\n' "$LSR_ATTEMPT" > "$OUT/runtime_attempt"
printf '%s\n' "$LSR_SHARD_SHA256" > "$OUT/runtime_shard_hash"
LSR_LIFETIME_SECONDS=$(md attributes/lsr-lifetime-seconds)
LSR_WORK_STOP_SECONDS=$(md attributes/lsr-work-stop-seconds)
LSR_SHUTDOWN_RESERVE_SECONDS=$(md attributes/lsr-shutdown-reserve-seconds)
LSR_CREDIT_GUARD_EPOCH=$(md attributes/lsr-credit-guard-epoch)
for value in "$LSR_LIFETIME_SECONDS" "$LSR_WORK_STOP_SECONDS" "$LSR_SHUTDOWN_RESERVE_SECONDS" "$LSR_CREDIT_GUARD_EPOCH"; do
    [[ "$value" =~ ^[1-9][0-9]{0,9}$ ]] || exit 1
done
[[ "$LSR_LIFETIME_SECONDS" == 3600 || "$LSR_LIFETIME_SECONDS" == 10800 || "$LSR_LIFETIME_SECONDS" == 14400 ]] || exit 1
(( LSR_WORK_STOP_SECONDS + LSR_SHUTDOWN_RESERVE_SECONDS == LSR_LIFETIME_SECONDS && LSR_SHUTDOWN_RESERVE_SECONDS >= 900 )) || exit 1
(( LSR_CREDIT_GUARD_EPOCH <= 1792195200 && LSR_CREATED_EPOCH + LSR_LIFETIME_SECONDS <= LSR_CREDIT_GUARD_EPOCH )) || exit 1
export LSR_LIFETIME_SECONDS LSR_WORK_STOP_SECONDS LSR_SHUTDOWN_RESERVE_SECONDS LSR_CREDIT_GUARD_EPOCH
for field in LIFETIME_SECONDS WORK_STOP_SECONDS SHUTDOWN_RESERVE_SECONDS CREDIT_GUARD_EPOCH; do
    key=LSR_$field
    printf '%s\n' "${!key}" > "$OUT/runtime_$field"
done
export LSR_STOP_EPOCH=$((LSR_CREATED_EPOCH + LSR_WORK_STOP_SECONDS))
# Metadata and persistence may have consumed the entire remaining work budget.
if ! work_budget; then export LSR_END_REASON=deadline; exit 1; fi
setsid bash "$0" --owned-work &
WORK_PID=$!
printf '%s\n' "$WORK_PID" > "$OUT/runtime_work_pgid"
# This watchdog only stops work. The surviving parent performs bounded custody.
setsid bash -c '
    now=$(date +%s)
    if [[ "$now" =~ ^[1-9][0-9]{0,9}$ ]] && (( now >= LSR_CREATED_EPOCH )); then
        left=$(($1 - now))
        if (( left > 0 )); then sleep "$left"; fi
    fi
    kill -KILL -- "-$2"
' watchdog "$LSR_STOP_EPOCH" "$WORK_PID" &
WATCHDOG=$!
RC=0
wait "$WORK_PID" || RC=$?
stop_work
if (( $(date +%s) >= LSR_CREATED_EPOCH + LSR_WORK_STOP_SECONDS )); then
    RC=1; export LSR_END_REASON=deadline
else
    if [[ -f "$OUT/runtime_receipt.json" ]]; then
        export LSR_END_REASON=shutdown
    else
        export LSR_END_REASON=bootstrap
    fi
fi
exit "$RC"
