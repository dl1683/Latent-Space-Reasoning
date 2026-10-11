#!/bin/bash
# GPU auto-off monitor (v2 semantics: never-ready counts as idle).
# Usage: automon.sh <heartbeat-file> <idle-min> <grace-min> [selftest]
# A minute is IDLE when summed GPU utilisation < 5% AND the job heartbeat is older than 5 min
# (or absent after the boot grace). IDLE for <idle-min> consecutive minutes -> power off.
# A heartbeat stale for > 3*idle-min powers off regardless of GPU utilisation (hung job).
# selftest: 5 s ticks, idle 1 min, grace 0, missing heartbeat, power-off replaced by echo.
HB=$1; IDLE_MIN=${2:-30}; GRACE_MIN=${3:-45}; MODE=${4:-live}
TICK=60; OFF="poweroff_now"
if [ "$MODE" = selftest ]; then TICK=5; IDLE_MIN=1; GRACE_MIN=0; HB=/nonexistent/hb; OFF="echo"; fi
poweroff_now() {
  [ -x /opt/lsr/final_sync.sh ] && bash /opt/lsr/final_sync.sh "automon:$1"
  NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
  ZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet >/dev/null 2>&1 || shutdown -h now
}
start=$(date +%s); idle_since=""
while sleep $TICK; do
  now=$(date +%s)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  if [ -f "$HB" ]; then hb_age=$(( now - $(stat -c %Y "$HB") )); ready=1; else hb_age=$(( now - start )); ready=0; fi
  reason=""
  if [ $ready = 1 ] && [ $hb_age -gt $(( 3 * IDLE_MIN * 60 )) ]; then reason=hung_job; fi
  in_grace=$(( now - start < GRACE_MIN * 60 ))
  if [ -z "$reason" ]; then
    if [ "$util" -lt 5 ] && [ $hb_age -gt 300 -o "$MODE" = selftest ] && [ $in_grace = 0 ]; then
      [ -z "$idle_since" ] && idle_since=$now
      if [ $(( now - idle_since )) -ge $(( IDLE_MIN * 60 )) ]; then
        [ $ready = 1 ] && reason=idle || reason=never_ready; fi
    else idle_since=""; fi
  fi
  if [ -n "$reason" ]; then
    echo "$(date -u +%FT%TZ) IDLE_POWEROFF ready=$ready reason=$reason util=$util hb_age=$hb_age"
    $OFF "$reason"; exit 0
  fi
done
