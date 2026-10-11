#!/bin/bash
# Every 10 min: reap powered-off VMs, relaunch preempted/unfinished shards of the runs in active_runs.txt.
cd "$(dirname "$0")"
while true; do
  { date -u +%FT%TZ; python lsr_cloud.py reap; for r in $(cat active_runs.txt); do python lsr_cloud.py relaunch "$r" 2>&1 | grep -E "relaunching|launched|NO CAPACITY|Error"; done; } >> watchdog.log 2>&1
  sleep 600
done
