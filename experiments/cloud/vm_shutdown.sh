#!/bin/bash
# Runs on preemption, max-run-duration deletion and self-delete: record an end marker and sync results.
[ -x /opt/lsr/final_sync.sh ] && timeout 25 bash /opt/lsr/final_sync.sh shutdown
