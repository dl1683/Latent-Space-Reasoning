#!/bin/bash
# LSR custom-image bake (operator tooling; CPU-only staging VM startup script). Starting from the pinned public image, it
# uninstalls torchaudio, installs the pinned lock, verifies the installed environment against the lock, uploads the
# verification output and then STOPS the VM (the operator images the stopped disk and deletes the VM).
# Metadata: bake-tag, bake-lock (gs:// lock object), bake-dest (gs:// verification prefix).
set -uo pipefail
export HOME=/root DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }
TAG=$(md attributes/bake-tag); LOCK=$(md attributes/bake-lock); DEST=$(md attributes/bake-dest); NAME=$(md name)
LOG=/tmp/bake.log; exec > >(tee -a $LOG) 2>&1
echo "BAKE_BOOT $(date -u +%FT%TZ) $NAME tag=$TAG"
PY=$(command -v python3); echo "PY=$PY $($PY --version 2>&1)"; echo "SYS_EXECUTABLE=$($PY -c 'import sys;print(sys.executable)')"
gcloud storage cp "$LOCK" /tmp/lock.txt || { echo BAKE_FAIL_LOCK; gcloud storage cp $LOG "$DEST/bake_fail_${TAG}.txt"; shutdown -h now; exit 1; }
echo "LOCK_SHA256=$(sha256sum /tmp/lock.txt | cut -c1-64) lines=$(wc -l < /tmp/lock.txt)"
$PY -m pip freeze > /tmp/before.txt 2>/dev/null
$PY -m pip uninstall -y -q torchaudio 2>&1 | tail -1
# The lock is a full pip freeze that also lists OS-managed distributions (python-apt, PyGObject, systemd-python ...) which cannot
# be built from PyPI, so a wholesale `pip install -r` fails. Install ONLY the delta: pins not already satisfied, exact, no dependency resolution.
$PY - <<'PYEOF' > /tmp/todo.txt
import re
from importlib import metadata
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
have = {}
for d in metadata.distributions():
    n = d.metadata.get("Name")
    if n and norm(n) not in have: have[norm(n)] = d.version
for line in open("/tmp/lock.txt", encoding="utf-8").read().splitlines():
    m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
    if have.get(norm(m[1])) != m[2]: print(line)
PYEOF
echo "== delta to install ($(wc -l < /tmp/todo.txt) pins)"; cat /tmp/todo.txt
$PY -m pip install -q --no-deps -r /tmp/todo.txt 2>&1 | tail -5
$PY -m pip freeze > /tmp/after.txt 2>/dev/null
echo "== pip freeze vs lock (exact line diff)"; diff <(sort /tmp/lock.txt) <(sort /tmp/after.txt) && echo "FREEZE_EQUALS_LOCK" || echo "FREEZE_DIFFERS_FROM_LOCK"
echo "== runtime-style comparison via importlib.metadata (first dist per name wins; pip/setuptools/wheel excluded)"
$PY - <<'EOF'
import re
from importlib import metadata
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
lock = {}
for line in open("/tmp/lock.txt", encoding="utf-8").read().splitlines():
    m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
    assert m, line
    lock[norm(m[1])] = m[2]
inst, shadow, nameless = {}, [], 0
for d in metadata.distributions():
    n = d.metadata.get("Name")
    if not n:
        nameless += 1; continue
    k = norm(n)
    if k in inst: shadow.append((k, d.version)); continue
    inst[k] = d.version
ign = {"pip", "setuptools", "wheel"}
missing = sorted(set(lock) - set(inst)); unexpected = sorted(set(inst) - set(lock) - ign)
mism = {k: (lock[k], inst[k]) for k in lock if k in inst and lock[k] != inst[k]}
print("locked", len(lock), "installed", len(inst), "nameless", nameless, "shadowed", shadow[:5])
print("missing", missing); print("unexpected", unexpected); print("mismatched", mism)
print("ENV_MATCHES_LOCK" if not (missing or unexpected or mism) else "ENV_DIFFERS_FROM_LOCK")
EOF
echo "== installed pip/setuptools/wheel"; $PY -m pip list 2>/dev/null | grep -i -E "^(pip|setuptools|wheel) "
rm -rf /root/.cache/pip /tmp/before.txt; sync
gcloud storage cp /tmp/after.txt "$DEST/lsr_image_freeze_${TAG}.txt" >/dev/null 2>&1
echo "BAKE_DONE $(date -u +%FT%TZ)"
gcloud storage cp $LOG "$DEST/lsr_image_verify_${TAG}.txt" >/dev/null 2>&1
rm -f /tmp/after.txt /tmp/lock.txt $LOG
sync; shutdown -h now
