#!/bin/bash
# sc-bench-lsr-* VM boot: fetch bundle + shard from GCS, verify auto-off (self-test), run the shard,
# sync results, delete self (fallback: power off). Resumable: prior partial results are pulled first.
set -uo pipefail
export HOME=/root DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_XET_HIGH_PERFORMANCE=1
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }
NAME=$(md name); JOB=$(md attributes/lsr-job); IDLE=$(md attributes/lsr-idle-min || echo 30)
GRACE=$(md attributes/lsr-grace-min || echo 45)
W=/opt/lsr; OUT=$W/out; mkdir -p $OUT; exec > >(tee -a $OUT/startup.log) 2>&1
echo "BOOT $(date -u +%FT%TZ) $NAME job=$JOB"
printf '{"vm":"%s","boot":"%s"}\n' "$NAME" "$(date -u +%FT%TZ)" > $OUT/boot.json
gcloud storage cp $OUT/boot.json "$JOB/out/boot_$(date +%s).json" >/dev/null 2>&1
RUN=${JOB%/shards/*}
fail() { echo "FAIL $1"; gcloud storage cp $OUT/startup.log "$JOB/out/startup_fail_$(date +%s).log"; shutdown -h now; exit 1; }
gcloud storage cp -r "$RUN/bundle/*" $W/ || fail bundle
gcloud storage cp "$JOB/shard.json" $W/shard.json || fail shard
gcloud storage rsync -r -x ".*startup.log$|.*automon.*" "$JOB/out" $OUT >/dev/null 2>&1 && [ -f $OUT/results.jsonl ] && echo "RESUME $(wc -l < $OUT/results.jsonl) rows, ckpt: $(ls $OUT/*.pt 2>/dev/null | tr '
' ' ')"
cat > $W/final_sync.sh <<EOS
#!/bin/bash
echo "{\"end\":\"\$(date -u +%FT%TZ)\",\"why\":\"\$1\"}" > $OUT/end_\$(date +%s).json
gcloud storage rsync -r $OUT "$JOB/out" >/dev/null 2>&1
EOS
chmod +x $W/final_sync.sh
# auto-off self-test (dry run) BEFORE any work is routed
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
timeout 150 bash $W/automon.sh x 1 0 selftest > $OUT/automon_selftest.log 2>&1
if grep -q "IDLE_POWEROFF ready=0 reason=never_ready" $OUT/automon_selftest.log; then echo "AUTOMON_SELFTEST PASS"
else bash $W/final_sync.sh selftest_fail; fail automon_selftest; fi
nohup bash $W/automon.sh $OUT/heartbeat "$IDLE" "$GRACE" > $OUT/automon.log 2>&1 &
( while sleep 300; do gcloud storage rsync -r $OUT "$JOB/out" >/dev/null 2>&1; done ) &
PY=$(command -v python3); [ -x /opt/conda/bin/python ] && PY=/opt/conda/bin/python
$PY -m pip uninstall -y -q torchaudio 2>&1 | tail -1
$PY -m pip install -q -r $W/requirements.txt 2>&1 | tail -3
$PY -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"
touch $OUT/heartbeat
cd $W && $PY -u -m lsr_jobs.run $W/shard.json $OUT; RC=$?
echo "JOB_EXIT rc=$RC $(date -u +%FT%TZ)"
printf '{"rc":%d,"end":"%s"}\n' $RC "$(date -u +%FT%TZ)" > $OUT/done.json
bash $W/final_sync.sh job_done
gcloud compute instances delete "$NAME" --zone "$(md zone | awk -F/ '{print $NF}')" --quiet || shutdown -h now
