#!/bin/bash
# LSR asset pre-staging VM (CPU only; operator tooling, not an experiment). Runs as a VM startup-script on an
# sc-bench-lsr-stage-* Spot VM created from the SAME image family as the job VMs. It:
#   1. records the image's own pip freeze, runs the dispatcher's exact install, records the post-install freeze (the LSR lock);
#   2. downloads the pinned model revision from Hugging Face ON THIS VM (weights never touch a local machine);
#   3. copies it to the project bucket with a sha256 manifest and a _COMPLETE marker, then deletes itself.
# Metadata attributes: stage-model, stage-rev, stage-dest (gs:// destination dir), stage-tag (receipt suffix),
# stage-requirements (space-separated pip requirement strings, as in a dispatcher config's `requirements`).
set -uo pipefail
export HOME=/root DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }
NAME=$(md name); ZONE=$(md zone | awk -F/ '{print $NF}')
MODEL=$(md attributes/stage-model); REV=$(md attributes/stage-rev); DEST=$(md attributes/stage-dest)
TAG=$(md attributes/stage-tag); REQS=$(md attributes/stage-requirements)
PINS=gs://sc-bench-iqidis-artifacts/lsr/pins
LOG=/tmp/stage.log; exec > >(tee -a $LOG) 2>&1
echo "STAGE_BOOT $(date -u +%FT%TZ) $NAME model=$MODEL rev=$REV dest=$DEST"
finish() { gcloud storage cp $LOG "$PINS/stage_log_${TAG}.txt" >/dev/null 2>&1; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || shutdown -h now; exit "${1:-0}"; }
PY=$(command -v python3); [ -x /opt/conda/bin/python ] && PY=/opt/conda/bin/python
echo "PYTHON $($PY --version 2>&1) $PY"; cat /etc/os-release | grep PRETTY_NAME
echo "IMAGE_SOURCE $(md disks/0/image 2>/dev/null)"
$PY -m pip freeze > /tmp/freeze_image_base.txt 2>/dev/null
gcloud storage cp /tmp/freeze_image_base.txt "$PINS/lsr_freeze_image_base_${TAG}.txt" >/dev/null 2>&1
# the dispatcher's exact boot install (vm_startup.sh): uninstall torchaudio, then pip install -r requirements
$PY -m pip uninstall -y -q torchaudio 2>&1 | tail -1
printf '%s\n' $REQS > /tmp/requirements.txt
$PY -m pip install -q -r /tmp/requirements.txt 2>&1 | tail -3
$PY -m pip freeze > /tmp/freeze_lsr_lock.txt 2>/dev/null
cp /tmp/requirements.txt /tmp/requirements_used.txt
gcloud storage cp /tmp/freeze_lsr_lock.txt "$PINS/lsr_freeze_lock_${TAG}.txt" >/dev/null 2>&1
gcloud storage cp /tmp/requirements_used.txt "$PINS/lsr_requirements_used_${TAG}.txt" >/dev/null 2>&1
echo "LOCK_WRITTEN $(wc -l < /tmp/freeze_lsr_lock.txt) lines"
# weights: HF -> this VM -> bucket
$PY -m pip install -q hf_transfer 2>&1 | tail -1
mkdir -p /data/model
HF_HUB_ENABLE_HF_TRANSFER=1 $PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', revision='$REV', local_dir='/data/model')
print('SNAPSHOT', p)
" || { echo "DOWNLOAD_FAILED"; finish 1; }
( cd /data/model && find . -type f -not -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum ) > /tmp/manifest_sha256.txt
echo "{\"model\":\"$MODEL\",\"revision\":\"$REV\",\"staged\":\"$(date -u +%FT%TZ)\",\"vm\":\"$NAME\",\"files\":$(wc -l < /tmp/manifest_sha256.txt),\"bytes\":$(du -sb --exclude=.cache /data/model | cut -f1)}" > /tmp/_COMPLETE
gcloud storage rsync -r --exclude='^\.cache/.*' /data/model "$DEST" || { echo "UPLOAD_FAILED"; finish 1; }
gcloud storage cp /tmp/manifest_sha256.txt "$DEST/_manifest_sha256.txt"
gcloud storage cp /tmp/_COMPLETE "$DEST/_COMPLETE"
echo "STAGE_DONE $(date -u +%FT%TZ)"
finish 0
