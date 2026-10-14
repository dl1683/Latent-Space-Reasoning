#!/bin/bash
# Create-then-verify for an LSR attempt that was ALREADY launched through lsr_cloud.py (operator tooling; launches nothing).
# Usage: CLOUDSDK_ACTIVE_CONFIG_NAME=sc-bench bash experiments/cloud/verify_launch.sh <run> <n_shards> [max_minutes=10]
# Checks every 20 s: each shard's VM is RUNNING, a boot_*.json marker exists in GCS, no startup_fail_*.log, no traceback in
# startup.log. Exit 0 = all shards booted; exit 1 = a shard failed/missing at the deadline (stop and report; NO automatic retry).
RUN=$1; N=${2:?n_shards}; MAXMIN=${3:-10}
G="/c/Users/devan/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud"
export CLOUDSDK_ACTIVE_CONFIG_NAME=${CLOUDSDK_ACTIVE_CONFIG_NAME:-sc-bench}
B=gs://sc-bench-iqidis-artifacts/lsr/runs/$RUN/shards
end=$(( $(date +%s) + MAXMIN*60 ))
while :; do
  ok=0; bad=0
  vms=$("$G" compute instances list --project=iqidis-developer --filter="labels.run=${RUN,,} AND labels.role=lsr" --format="value(name,status)" 2>/dev/null)
  for i in $(seq 0 $((N-1))); do
    s=$(printf '%03d' $i); vm="sc-bench-lsr-${RUN,,}-$s"
    st=$(echo "$vms" | awk -v v="$vm" '$1==v{print $2}')
    out=$("$G" storage ls "$B/$s/out/" 2>/dev/null)
    booted=$(echo "$out" | grep -c "boot_")
    failed=$(echo "$out" | grep -c "startup_fail_")
    tb=$("$G" storage cat "$B/$s/out/startup.log" 2>/dev/null | grep -c -E "Traceback|AssertionError|AUTOMON_SELFTEST FAIL|FAIL ")
    echo "$(date -u +%T) shard $s vm=${st:-ABSENT} boot_marker=$booted fail_log=$failed traceback_lines=$tb"
    if [ "$failed" -gt 0 ] || [ "$tb" -gt 0 ]; then bad=$((bad+1)); elif [ "$st" = RUNNING ] && [ "$booted" -gt 0 ]; then ok=$((ok+1)); fi
  done
  [ $ok -eq $N ] && { echo "VERIFY_OK all $N shards booted"; exit 0; }
  [ $bad -gt 0 ] && { echo "VERIFY_FAIL $bad shard(s) show failure; stop and report, no retry"; exit 1; }
  [ $(date +%s) -ge $end ] && { echo "VERIFY_TIMEOUT after ${MAXMIN} min: $ok/$N booted; stop and report, no retry"; exit 1; }
  sleep 20
done
