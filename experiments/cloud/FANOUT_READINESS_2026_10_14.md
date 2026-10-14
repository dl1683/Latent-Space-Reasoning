# LSR wide fan-out readiness (operator receipt, 2026-10-14)

Scope: preparation only. Nothing experimental was launched. One CPU staging VM (`sc-bench-lsr-stage-gemma-base`, n2-standard-16
Spot, pd-standard, 2 h hard limit, self-deleting) pre-stages the pinned lock and the pinned base weights. All gcloud calls use
`CLOUDSDK_ACTIVE_CONFIG_NAME=sc-bench`. Model weights were never downloaded to a local machine.

## 1. Dispatcher (`experiments/cloud/lsr_cloud.py`, working tree as read today)

Confirmed in code:
- Parallel creation (`ThreadPoolExecutor(16)`), Spot, `--instance-termination-action=DELETE`, `--maintenance-policy=TERMINATE`.
- Per-VM wall limit: `--max-run-duration` = `cost_cap_usd / usd_per_hour`, or `max_hours - shutdown_margin_hours` when set.
- Zone fallback list per VM on stockout/quota errors; stale same-name VM replacement; registry (`launches.jsonl`); bundle upload;
  resume/relaunch/reap/status with per-shard states.
- Auto-stop: VM deletes itself when the job ends; `automon.sh` powers off an idle or never-ready node. On a CPU VM there is no
  `nvidia-smi`, so utilisation reads 0 and the monitor keys only on the heartbeat: CPU jobs must call `out.beat()` at least every
  5 minutes or they are shut down after `idle_min`.

Gaps for wide fan-out (all in files that Z Code owns; nothing was edited here):
1. BLOCKER `boot_disk_type`: the dispatcher hard-codes `pd-balanced`, which counts against SSD quota. us-central1 SSD_TOTAL_GB is
   40,650 / 40,960 used (170 pd-balanced disks = 38,950 GB + pd-ssd 1,700 GB). About 310 GB remain, enough for one VM. Needed: a
   `boot_disk_type` config key (`pd-standard` has 102,230 GB free under DISKS_TOTAL_GB; `hyperdisk-balanced` is used by 4 existing
   disks). Terminated nodes by other programs also hold SSD quota; only their owners may delete them.
2. `nic_type`: running A3 nodes use `GVNIC` and local scratch SSD; the dispatcher passes no NIC type. A3 creation without
   `nic-type=GVNIC` is untested here and likely to be refused. Add a config key before the first A3 launch.
3. No create-then-verify: after `instances create` returns there is no check that the VM reaches RUNNING, writes its
   `boot_*.json` marker or survives the first minutes (Spot preemption). `status <run>` shows it manually. Needed: `launch --verify`.
4. No aggregate ceiling: `cost_cap_usd` is per VM; `launch` prints the total but does not enforce a per-run or program ceiling or
   consult SPEND.md / the guard.
5. Local load: `create_vm` calls `list_vms(run)` for every VM (O(N^2) gcloud calls) and up to 16 gcloud subprocesses run at once
   on the operator machine, which crashed from memory overload earlier. Run wide launches from a cloud operator VM, or use at
   most 4 workers locally.
6. No local-weights path: `common.load` fetches from Hugging Face at job start. A bucket mirror needs a pre-job sync step in
   `vm_startup.sh` and a `model_local` option.

## 2. Quota headroom (us-central1; running fleet as evidence)

| item | usage / limit or evidence |
|---|---|
| PREEMPTIBLE_CPUS (Spot CPU) | 0 / 5,000: up to 78 x n2-standard-64 (4,992 vCPU) or 312 x n2-standard-16 |
| N2_CPUS (on-demand) | 320 / 3,000 (other programs) |
| IN_USE_ADDRESSES | 2 / 575 |
| DISKS_TOTAL_GB (pd-standard) | 170 / 102,400 |
| SSD_TOTAL_GB (pd-balanced, pd-ssd) | 40,650 / 40,960 (BLOCKER above) |
| NVIDIA_A100 (40GB) | 0 / 16 on demand, 0 / 64 preemptible: up to 32 x a2-highgpu-2g; 80GB: 0 / 0 |
| H100 / A3 | no quota metric is listed for the region or project. Evidence from the running fleet under sc-bench: 10 x a3-highgpu-1g (9 in zone a, 1 in b) and 1 x a3-highgpu-2g (a), i.e. 12 H100 Spot GPUs running now; also 11 a3-highgpu-2g and many 1g/4g/8g TERMINATED in zone a. Headroom beyond 12 GPUs is unknown until a create is refused. |

Cost at full quota (rates per supervisor): n2-standard-64 Spot 1.864 USD/h x 78 = about 145 USD/h; a3-highgpu-2g Spot 13.1986 USD/h
x 10 nodes = about 132 USD/h.

## 3. Pre-staging

- Image: family `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` resolves to `...-v20261001` (100 GB, READY). Public image; no custom
  image is needed because the job VMs load weights from the bucket, not from the image.
- Pinned package lock from the LSR image itself (staging VM ran the dispatcher's exact install: uninstall torchaudio, then pip install):
  `gs://sc-bench-iqidis-artifacts/lsr/pins/lsr_freeze_lock_20261014a.txt` (full freeze), `lsr_freeze_image_base_20261014a.txt`
  (image before install), `lsr_requirements_used_20261014a.txt`. Key versions: torch 2.9.1+cu129, transformers 5.19.0,
  accelerate 1.15.0, tokenizers 0.23.3, huggingface_hub 2.2.0, numpy 2.2.6, Jinja2 3.1.6, safetensors 0.8.0, hf-xet 1.7.0,
  hf_transfer 0.1.9. These replace the unrecovered pins in the S0 section. They differ from the HAL node freeze
  (`lsr/pins/hal_gem31_pip_freeze_20261014.txt`: torch 2.13.0+cu130, transformers 5.17.0, huggingface_hub 1.33.0), which
  is a different image and must not be mixed in.
- Weights: the existing mirror `models/ladder/gem31` is `google/gemma-4-31B-it` revision `842da379...`, NOT the pinned base.
  The pinned base `google/gemma-4-31B` revision `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89` (10 files, 62,578,664,883 bytes) is being
  copied to `gs://sc-bench-iqidis-artifacts/lsr/models/gemma-4-31B/5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89/` with
  `_manifest_sha256.txt` and a `_COMPLETE` marker written last. Do not read the directory as usable until `_COMPLETE` exists.
- Staging tool: `experiments/cloud/stage_assets.sh` (startup script; metadata `stage-model`, `stage-rev`, `stage-dest`, `stage-tag`,
  `stage-requirements`).

## 4. Budget and credit flags

- Credit per supervisor reading: 9,169.48 USD remaining on 2026-10-14, expiring 2026-10-17. The planned LSR ceiling of about
  3,000 USD fits inside it. Billing lags a day or more, so the ledger can understate spend; the dispatcher has no live ceiling (gap 4).
- Spending 3,000 USD before expiry needs an average of at least about 50 USD/h over the remaining ~60 hours; the shapes above
  allow far more, so the guard, not the quota, is the binding limit.
- Flag, possible spend beyond credit (UNVERIFIED): this account cannot read the bucket's location (`storage.buckets.get` denied), so
  it is assumed US. If so, a VM outside North America (for example europe-west4) reading it would incur network egress (about 62 GB
  per model load); keep launches in us-central1. Not-credit-offset charges are otherwise
  limited to persistent disks left behind by stopped (not deleted) nodes.
