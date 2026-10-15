"""Offline production dispatch checks. CPU imports only; no pretrained model or network execution."""
import argparse
import copy
from contextlib import ExitStack, redirect_stdout, nullcontext
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import tempfile
import time
import shutil
import sys

_REAL_RUN, _REAL_POPEN = subprocess.run, subprocess.Popen
_REAL_SLEEP = time.sleep
from unittest.mock import patch

from .. import lsr_cloud as cloud
from . import run

PRODUCTION_RATE_TABLE = copy.deepcopy(cloud.RATE_TABLE)


def forbidden(*args, **kwargs):
    raise AssertionError("real external effect/time forbidden in offline checks")


def fixture():
    reqs = ["transformers==5.12.1", "accelerate==1.10.0", "jinja2==3.1.6"]
    # Synthetic pins exercise validation only; they are NOT operator receipts.
    return dict(provisioning_model="SPOT", run="offline-a01", kind="s0_memory_port", machine="a3-highgpu-2g", has_gpu=True,
                guest_accelerators=[{"type":"nvidia-h100-80gb","count":2}],
                weights_manifest_uri="gs://offline-mirror/model/manifest.json", weights_manifest_sha256="c"*64,
                usd_per_hour=13.1986, cost_cap_usd=52.7944, aggregate_cost_ceiling_usd=52.8,
                zones=["us-central1-a"], disk_gb=300, boot_disk_type="pd-balanced", nic_type="GVNIC",
                image_name="offline-exact-image", image_project="offline-image-project", image_id="123456789", release=False, requirements=reqs,
                requirements_lock_sha256=hashlib.sha256(("\n".join(reqs)+"\n").encode()).hexdigest(),
                weights_mirror_uri="gs://offline-mirror/model/revision", weights_local_path="/opt/lsr/weights",
                allow_hf_download=False, rate_verified=False, shape_verified=False, lifetime_seconds=14400,
                work_stop_seconds=13500, shutdown_reserve_seconds=900, shards=[{}],
                common={"model": "google/gemma-4-31B", "revision":"d"*40, "tokenizer_revision":"e"*40,
                        "runtime_control_command":["python","-B","-m","offline.operation","--control"], "job_stop_seconds": 13500})


def rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError, KeyError):
        return
    raise AssertionError("negative case did not fail")


class FakeCloud:
    """Finite response table; every unregistered command raises."""
    def __init__(self, cfg, contract):
        self.cfg, self.contract, self.calls = cfg, contract, []
        self.scripts={"startup-script":(cloud.HERE/"vm_startup.sh").read_bytes(),
                      "shutdown-script":(cloud.HERE/"vm_shutdown.sh").read_bytes()}
        self.mutate = lambda a: None
        self.existing, self.objects = [], []
        self.fail_describe = False
        self.fail_create = False

    def assert_create(self, args):
        # Independent request oracle: no production argv-builder reused here.
        c = self.cfg
        assert args[:4] == ("compute", "instances", "create", cloud.vm_name(c["run"],0))
        flags = {}
        for arg in args[4:]:
            assert arg.startswith("--") and "=" in arg, arg
            k,v = arg.split("=",1)
            assert k not in flags, k
            flags[k] = v
        expected = {
            "--zone":c["zones"][0], "--machine-type":c["machine"], "--service-account":cloud.SA,
            "--scopes":"https://www.googleapis.com/auth/cloud-platform", "--image":c["image_name"],
            "--image-project":c["image_project"], "--boot-disk-size":f"{c['disk_gb']}GB",
            "--boot-disk-type":c["boot_disk_type"], "--network-interface":f"nic-type={c['nic_type']}",
            "--labels":f"app=stateful-bench,role=lsr,run={c['run']}",
            "--provisioning-model":c["provisioning_model"], "--instance-termination-action":"DELETE",
            "--maintenance-policy":"TERMINATE",
            "--metadata-from-file":flags.get("--metadata-from-file")}
        if c.get("accelerator"):
            typ,count=c["accelerator"].split(":")
            expected["--accelerator"]=f"type={typ},count={count}"
        assert set(flags)==set(expected)|{"--metadata","--termination-time"}, set(flags)^set(expected)
        for key,value in expected.items():
            assert flags[key]==value, (key, flags[key], value)
        metadata=dict(item.split("=",1) for item in flags["--metadata"].split(","))
        created=int(metadata["lsr-created-at"])
        contract=cloud.dispatch_contract(c,created,metadata["lsr-bundle-sha256"])
        if "lsr-shard-sha256" in metadata:
            contract["shard_sha256"] = metadata["lsr-shard-sha256"]
        if self.contract:
            assert contract==self.contract
        expected_meta={"lsr-job":f"{cloud.BUCKET}/runs/{c['run']}/shards/000",
            "lsr-idle-min":str(c.get("idle_min",30)),"lsr-grace-min":str(c.get("grace_min",45)),
            "lsr-created-epoch":str(created),"lsr-accelerator":c.get("accelerator") or ("integrated" if c["has_gpu"] else ""),
            "lsr-model-mirror":c.get("weights_local_path") or "/opt/lsr/offline-hf"}
        expected_meta.update({"lsr-"+k.replace("_","-"):str(v).lower() if isinstance(v,bool) else str(v)
                              for k,v in contract.items() if v is not None})
        assert metadata==expected_meta
        assert dt.datetime.fromisoformat(flags["--termination-time"]).timestamp()==created+c.get("lifetime_seconds",14400)
        # Capture the exact referenced script bytes; the described metadata must match.
        self.script_bytes={key:Path(path).read_bytes() for key,path in
                           (item.split("=",1) for item in flags["--metadata-from-file"].split(","))}
        assert self.script_bytes==self.scripts

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if args[:3] == ("compute", "instances", "list"):
            return json.dumps(self.existing)
        if args[:2] == ("storage", "ls"):
            return "\n".join(self.objects)
        if args[:3] == ("compute", "instances", "create"):
            if self.fail_create:
                raise RuntimeError("name exists")
            self.assert_create(args)
            self.creation = args
            return ""
        if args[:3] == ("compute", "instances", "describe"):
            if self.fail_describe:
                raise RuntimeError("describe denied")
            def flag(name):
                return next(x.split("=",1)[1] for x in self.creation if x.startswith(name+"="))
            metadata = dict(x.split("=",1) for x in flag("--metadata").split(","))
            metadata.update({key:Path(path).read_text() for key,path in
                             (item.split("=",1) for item in flag("--metadata-from-file").split(","))})
            actual = dict(name=args[3], machineType=flag("--machine-type"), zone=flag("--zone"),
                          scheduling=dict(provisioningModel=flag("--provisioning-model"),instanceTerminationAction=flag("--instance-termination-action"),
                                          onHostMaintenance=flag("--maintenance-policy"),terminationTime=flag("--termination-time")),
                          serviceAccounts=[{"email":flag("--service-account"), "scopes":[flag("--scopes")]}],
                          labels=dict(x.split("=",1) for x in flag("--labels").split(",")),
                          metadata={"items":[{"key":k,"value":v} for k,v in metadata.items()]},
                          networkInterfaces=[{"nicType":flag("--network-interface").split("=",1)[1]}],
                          guestAccelerators=[{"acceleratorType":a["type"],"acceleratorCount":a["count"]} for a in self.cfg["guest_accelerators"]],
                          disks=[{"boot":True,"source":"offline-disk"}])
            self.mutate(actual)
            return json.dumps(actual)
        if args[:3] == ("compute", "disks", "describe"):
            flags = dict(x.split("=",1) for x in self.creation[4:])
            return json.dumps(dict(type=flags["--boot-disk-type"],sizeGb=int(flags["--boot-disk-size"].removesuffix("GB")),
                sourceImage=f"projects/{flags['--image-project']}/global/images/{flags['--image']}",
                sourceImageId=self.cfg.get("image_id", "123456789")))
        if args[:3] == ("compute", "instances", "delete"):
            return ""
        raise AssertionError(f"unregistered mocked command: {args}")


def creation(cfg, mutate=None, fail_describe=False, fail_create=False):
    contract = cloud.dispatch_contract(cfg, 100000, "a"*64)
    contract["shard_sha256"]="b"*64
    fake = FakeCloud(cfg, contract)
    if mutate:
        fake.mutate = mutate
    fake.fail_describe, fake.fail_create = fail_describe, fail_create
    with tempfile.TemporaryDirectory() as td, patch.object(cloud,"gcloud",fake), \
            patch.object(cloud.time,"time",return_value=100060), patch.object(cloud,"REGISTRY",Path(td)/"registry"):
        prefix = f"{cloud.BUCKET}/runs/{cfg['run']}/shards/000"
        if mutate or fail_describe or fail_create:
            rejects(cloud.create_vm,cfg,0,prefix,contract)
            deleted = any(c[:3] == ("compute","instances","delete") for c in fake.calls)
            assert deleted == (not fail_create), "mismatch must delete; creation collision must preserve"
            assert not (Path(td)/"registry").exists()
        else:
            assert cloud.create_vm(cfg,0,prefix,contract) == cfg["zones"][0]
            row=json.loads((Path(td)/"registry").read_text())
            assert row["dispatch"] == contract and row["rate_status"] == "UNVERIFIED"
    return fake


def options():
    c=fixture()
    for disk in ("pd-standard","pd-balanced","pd-ssd"):
        for nic in ("GVNIC","VIRTIO_NET"):
            d=copy.deepcopy(c);d.update(boot_disk_type=disk,nic_type=nic)
            fake=creation(d)
            assert f"--boot-disk-type={disk}" in fake.creation
    for field,bad in (("boot_disk_type","unknown"),("nic_type","unknown")):
        d=copy.deepcopy(c);d[field]=bad;rejects(cloud.validate_config,d)
        d.pop(field);rejects(cloud.validate_config,d)


def create_verify():
    c=fixture();creation(c)
    mutations=[lambda a:a.update(name="wrong"),lambda a:a.update(machineType="wrong"),
               lambda a:a.update(zone="wrong"),lambda a:a["scheduling"].update(provisioningModel="STANDARD"),
               lambda a:a["scheduling"].update(instanceTerminationAction="STOP"),
               lambda a:a["scheduling"].update(terminationTime="2100-01-01T00:00:00Z"),
               lambda a:a.update(serviceAccounts=[{"email":"wrong"}]),
               lambda a:a["serviceAccounts"][0].update(scopes=[]),lambda a:a.update(labels={}),
               lambda a:a.update(metadata={"items":[]}),
               lambda a:a["networkInterfaces"][0].update(nicType="wrong"),
               lambda a:a["disks"][0].update(source="wrong")]
    for mutate in mutations:
        if mutate is mutations[-1]:
            # Disk type is obtained from the actual disk describe, not echoed from request.
            contract=dict(cloud.dispatch_contract(c,100000,"a"*64), shard_sha256="b"*64);fake=FakeCloud(c,contract)
            original=fake.__call__
            def wrong_disk(*args,**kwargs):
                if args[:3]==("compute","disks","describe"):
                    return json.dumps({"type":"pd-ssd","sizeGb":300,"sourceImage":c["image_name"]})
                return original(*args,**kwargs)
            with patch.object(cloud,"gcloud",wrong_disk),patch.object(time,"time",return_value=100060):
                rejects(cloud.create_vm,c,0,f"{cloud.BUCKET}/runs/{c['run']}/shards/000",contract)
            assert any(x[:3]==("compute","instances","delete") for x in fake.calls)
        else:
            creation(c,mutate)
    creation(c,fail_describe=True);creation(c,fail_create=True)


def aggregate_cost():
    c=fixture();assert cloud.launch_plan(c)[1]==52.7944
    c["shards"]=[{},{}];rejects(cloud.launch_plan,c)
    c["aggregate_cost_ceiling_usd"]=105.6
    assert cloud.launch_plan(c)[1]==105.5888
    rejects(cloud.launch_plan,c,"0,0");rejects(cloud.launch_plan,c,"-1")
    for bad in (0,-1,float("inf"),float("nan"),True):
        d=fixture();d["usd_per_hour"]=bad;rejects(cloud.launch_plan,d)
    with tempfile.TemporaryDirectory() as td:
        c=fixture();c["aggregate_cost_ceiling_usd"]=1
        path=Path(td)/"cfg.json";path.write_text(json.dumps(c))
        # Global raising gcloud guard proves no preflight side effect.
        rejects(cloud.launch,path)


def entry_handling():
    c=fixture();c["kind"]="e1_ledger"
    assert run.assemble_shard(c,0)["entry_defaulted"]
    for where in (c,c["common"],c["shards"][0]):
        where["entry"]="alignment_hook_smoke"
        assert run.assemble_shard(c,0)["entry"]=="alignment_hook_smoke"
        assert not run.assemble_shard(c,0)["entry_defaulted"]
        del where["entry"]
    c["entry"]="alignment_hook_smoke";c["common"]["entry"]="main"
    rejects(run.assemble_shard,c,0)
    del c["common"]["entry"];c["entry"]="bogus";rejects(run.assemble_shard,c,0)
    del c["entry"];c["bogus_top_key"]=1;rejects(cloud.validate_config,c)
    for name in ("e1_gemma4_alignment_hook_v1.json","s0_gemma4_memory_port_v1.json"):
        cfg=json.loads((cloud.HERE/"configs"/name).read_text())
        cfg["common"]["runtime_control_command"]=["missing-operation-control"]
        assert run.assemble_shard(cfg,0)["entry"] == ("alignment_hook_smoke" if name.startswith("e1") else "main")
    # Execute the whole production launch with finite, explicit mocked effects.
    c=fixture();c.update(kind="e1_ledger",entry="alignment_hook_smoke")
    fake=FakeCloud(c,{});shards=[];expected_records=[];objects={}
    def effect(*args,**kwargs):
        if args[:2]==("storage","cp"):
            if args[2].startswith("gs://"):
                Path(args[3]).write_bytes(objects[args[2]]);return ""
            assert "--if-generation-match=0" in args
            objects[args[3]]=Path(args[2]).read_bytes()
            if args[3].endswith("shard.json"):
                raw=Path(args[2]).read_bytes();shard=json.loads(raw);shards.append(shard)
                rows=[json.loads(line) for line in cloud.REGISTRY.read_text().splitlines()]
                expected=cloud.expected_attempt(c["run"],0);expected_records.append(expected)
                assert rows[0]["record_type"]=="attempt_expected"
                assert expected==cloud.attempt_manifest(shard,raw)
                assert expected["shard_sha256"]==hashlib.sha256(raw).hexdigest()
            return ""
        if args[:3]==("compute","instances","create"):
            metadata=dict(x.split("=",1) for x in next(a.split("=",1)[1] for a in args if a.startswith("--metadata=")).split(","))
            assert metadata["lsr-shard-sha256"]==expected_records[-1]["shard_sha256"]
            for key in ("run","shard","attempt","bundle_sha256"):
                assert metadata["lsr-"+key.replace("_","-")]==str(expected_records[-1][key])
            assert metadata["lsr-created-epoch"]==str(expected_records[-1]["created_epoch"])
        return fake(*args,**kwargs)
    with tempfile.TemporaryDirectory() as td,patch.object(cloud,"gcloud",effect), \
            patch.object(time,"time",return_value=100000),patch.object(cloud,"now",return_value="offline-clock"), \
            patch.object(cloud,"REGISTRY",Path(td)/"registry"):
        path=Path(td)/"config.json";path.write_text(json.dumps(c))
        cloud.launch(path)
    assert len(shards)==1 and shards[0]["entry"]=="alignment_hook_smoke"
    assert shards[0]["dispatch"]["work_stop_at"]==113500
    assert shards[0]["requirements"]==c["requirements"]
    for key in ("requirements_lock_sha256","weights_manifest_uri","weights_manifest_sha256"):
        assert shards[0]["dispatch"][key]==c[key]
    assert shards[0]["dispatch"]["model_revision"]==c["common"]["revision"]
    assert expected_records[0]["attempt"]==shards[0]["attempt"]


def deadlines():
    c=fixture();d=dict(cloud.dispatch_contract(c,100000,"a"*64), shard_sha256="b"*64)
    assert d["work_stop_at"]==113500 and d["shutdown_at"]==114400
    assert d["shutdown_at"]-d["work_stop_at"]==900
    fake=creation(c)
    deadline=next(x.split("=",1)[1] for x in fake.creation if x.startswith("--termination-time="))
    assert dt.datetime.fromisoformat(deadline).timestamp()==114400
    assert not any("image-family" in x for x in fake.creation)
    for field in ("lifetime_seconds","work_stop_seconds","shutdown_reserve_seconds"):
        bad=fixture();bad[field]-=1;rejects(cloud.validate_config,bad)
    with patch.object(time,"time",return_value=113500):
        rejects(cloud.create_vm,c,0,f"{cloud.BUCKET}/runs/{c['run']}/shards/000",d)
    bad=fixture();bad["common"]["job_stop_seconds"]=12960;rejects(cloud.validate_config,bad)


def collisions():
    c=fixture();fake=FakeCloud(c,{})
    with patch.object(cloud,"gcloud",fake):
        cloud.refuse_collisions(c,[0])
        fake.existing=[{"name":cloud.vm_name(c["run"],0)}];rejects(cloud.refuse_collisions,c,[0])
        fake.existing=[];fake.objects=[f"{cloud.BUCKET}/runs/{c['run']}/"];rejects(cloud.refuse_collisions,c,[0])
    assert not any(x[:3]==("compute","instances","delete") or x[:2]==("storage","rm") for x in fake.calls)
    rejects(cloud.vm_name,"a"*80,0);rejects(cloud.vm_name,"Upper_Case",0)
    assert cloud.vm_name("a-b",0)!=cloud.vm_name("ab",0)
    rejects(cloud.relaunch,"prior")
    # Atomic run reservation collision must prevent bundle upload/VM creation.
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"cfg.json";path.write_text(json.dumps(c));calls=[]
        def race(*args,**kwargs):
            calls.append(args)
            if args[:3]==("compute","instances","list"):
                return "[]"
            if args[:2]==("storage","ls"):
                return ""
            if args[:2]==("storage","cp") and args[3].endswith("claim.json"):
                assert "--if-generation-match=0" in args
                raise RuntimeError("generation precondition failed")
            raise AssertionError("race allowed a later write")
        with patch.object(cloud,"gcloud",race),patch.object(cloud,"now",return_value="offline-clock"):
            rejects(cloud.launch,path)
        assert len(calls)==3


def bundle():
    paths=cloud.bundle_paths();names=[cloud.bundle_name(p) for p in paths]
    assert "lsr_jobs/run.py" in names and "lsr_jobs/common.py" in names and "automon.sh" in names
    assert "experiments/memory/associative_continuation_cpu.py" in names
    assert len(names)==len(set(names))
    assert all(not ({".git","__pycache__","weights","checkpoints"}&set(Path(n).parts)) for n in names)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);jobs=root/"experiments"/"cloud"/"lsr_jobs";mem=root/"experiments"/"memory"
        jobs.mkdir(parents=True);mem.mkdir(parents=True);(jobs/"run.py").write_text("pass")
        (mem/"task.py").write_text("pass");(root/"experiments"/"cloud"/"automon.sh").write_text("true")
        for folder in (".git","weights","checkpoints","__pycache__"):
            (mem/folder).mkdir();(mem/folder/"secret.py").write_text("exclude")
        (mem/"model.pt").write_text("exclude")
        (root/"experiments/cloud/vm_startup.sh").write_text("true")
        (root/"experiments/cloud/vm_shutdown.sh").write_text("true")
        with patch.object(cloud,"HERE",root/"experiments"/"cloud"),patch.object(cloud,"REPO",root):
            names=[cloud.bundle_name(p) for p in cloud.bundle_paths()]
            assert names==["automon.sh","lsr_jobs/run.py","experiments/memory/task.py"]
            captured=[]
            def upload(*args,**kwargs):
                assert args[:2]==("storage","cp")
                if args[2].startswith("gs://"):
                    Path(args[3]).write_bytes(captured[0]);return ""
                assert "--if-generation-match=0" in args
                captured.append(Path(args[2]).read_bytes())
                with tarfile.open(args[2]) as tar:
                    assert sorted(tar.getnames())==sorted(names+["experiments/cloud/lsr_jobs/run.py",
                          "requirements.txt","config.json","vm_startup.sh","vm_shutdown.sh","bundle_manifest.json"])
                    assert json.load(tar.extractfile("config.json"))==fixture()
                return ""
            with patch.object(cloud,"gcloud",upload),patch.object(time,"time",return_value=100000):
                assert len(cloud.upload_bundle("offline-a01",fixture()))==64
            # Missing memory tree is a failing bundle, not a partial upload.
            with patch.object(cloud,"REPO",root/"missing"):
                rejects(cloud.bundle_paths)


def pins_mirror():
    c=fixture();cloud.validate_config(c)
    for field in ("image_name","requirements_lock_sha256","weights_mirror_uri"):
        bad=fixture();bad.pop(field);rejects(cloud.validate_config,bad)
    for field,value in (("requirements",["transformers>=5.12"]),("allow_hf_download",True)):
        bad=fixture();bad[field]=value;rejects(cloud.validate_config,bad)
    with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{},clear=True):
        cfg={"model":"google/gemma-4-31B","dispatch":{"weights_local_path":td}}
        rejects(run.prepare_runtime,cfg)
        (Path(td)/"config.json").write_text("{}")
        rejects(run.prepare_runtime,cfg)  # even an existing arbitrary root is untrusted
        with patch.object(run.Path, "is_dir", return_value=True), patch.object(run.Path, "is_file", return_value=True):
            cfg={"kind":"latent_memory_prototype", "model":"reviewed/model", "dispatch":{"weights_local_path":"/opt/lsr/weights"}}
            run.prepare_runtime(cfg)
            assert cfg["model"]=="reviewed/model" and cfg["model_local"]=="/opt/lsr/weights"
            assert os.environ["HF_HUB_OFFLINE"]=="1"


def s0_config():
    c=json.loads((cloud.HERE/"configs/s0_gemma4_memory_port_v1.json").read_text())
    assert (c["machine"],c["usd_per_hour"],c["zones"])==("a3-highgpu-2g",13.1986,["us-central1-a"])
    assert c["rate_verified"] is False and c["shape_verified"] is False
    assert c["usd_per_hour"]*4==52.7944
    assert c["common"]["job_stop_seconds"]==13500
    cloud.validate_release_pins(c)
    shard=run.assemble_shard(c,0)
    assert shard["runtime_control_command"] == ["python","-B","-m","lsr_jobs.s0_memory_port",
                                               "--config","/opt/lsr/config.json","--check","s0-production-contract"]
    assert shard["phase"] == "smoke" and shard["dtype"] == "bfloat16"
    assert (shard["model"], shard["width"], shard["n_layers"]) == ("google/gemma-4-E2B", 1536, 35)
    assert shard["allowed_horizons"] == [0] and shard["recurrence"] is False
    assert shard["model_local"] == c["weights_local_path"] and shard["allow_hf_download"] is False
    assert shard["missing_operator_inputs"] and shard["smoke_cases"] is None and shard["fit_cases"] is None
    rejects(cloud.validate_config,c)  # Measured placement and pinned cases/backend are MISSING.
    with patch.object(cloud,"validate_placement",return_value=None):
        try:
            cloud.validate_config(c)
        except ValueError as exc:
            assert str(exc).startswith("MISSING-operator:"), str(exc)
        else:
            raise AssertionError("missing S0 inputs admitted")
        # Removing the label cannot make the null cases into a release input.
        bad=copy.deepcopy(c);bad["common"].pop("missing_operator_inputs")
        rejects(cloud.validate_config,bad)
    bad=fixture();bad["machine"]="";rejects(cloud.validate_config,bad)


def s0_identity():
    """Real assembly/prepare/execute; receiver entry and loader are recording doubles."""
    import types
    c=json.loads((cloud.HERE/"configs/s0_gemma4_memory_port_v1.json").read_text())
    shard=run.assemble_shard(c,0)
    shard["dispatch"]=cloud.dispatch_contract(c,100000,"a"*64)
    original=copy.deepcopy(shard)
    seen=[]
    def recording_entry(cfg,out):
        for field in ("model","model_source","revision","tokenizer_revision","model_local",
                      "placement","placement_contract","dtype","attn_implementation"):
            assert cfg[field] == original[field], field
        assert cfg["model"] == "google/gemma-4-E2B"
        seen.append(copy.deepcopy(cfg))
        raise RuntimeError("recording loader stopped before pretrained assets")
    class Output:
        done={}
        def __init__(self,path): pass
        def meta(self,name,cfg): assert name == "shard.json"
    with patch.object(run.Path,"is_dir",return_value=True),patch.object(run.Path,"is_file",return_value=True), \
         patch.dict(sys.modules,{run.__package__+".common":types.SimpleNamespace(Out=Output)}), \
         patch.object(run.importlib,"import_module",return_value=types.SimpleNamespace(main=recording_entry)):
        try:
            run.execute(copy.deepcopy(shard),"unused-output")
        except RuntimeError as exc:
            assert str(exc) == "recording loader stopped before pretrained assets"
        else:
            raise AssertionError("raising loader not reached")
        assert len(seen)==1
        for field,value in (("model","/opt/lsr/weights"),("model_source","different/identity"),
                            ("revision","f"*40),("tokenizer_revision","f"*40),
                            ("model_local","/untrusted")):
            bad=copy.deepcopy(shard);bad[field]=value
            rejects(run.execute,bad,"unused-output")
        assert len(seen)==1, "invalid identity/revisions reached the receiver"
    print("PASS S0 real assembly/prepare/execute preserves pinned identity/revisions/path/placement; "
          "identity rewrite and alias/revision/root conflicts rejected before recording loader; no weights")


def verify_shipped_cpu_evidence(cfg):
    """Retain historical byte/hash custody; obsolete packets cannot authorize launch."""
    raw=cloud.RELEASE_CPU_OPERATOR_EVIDENCE[cfg["operator_evidence_sha256"]]
    assert hashlib.sha256(raw).hexdigest()==cfg["operator_evidence_sha256"]
    receipt=json.loads(raw)
    path=cloud.HERE/"receipts"/(cfg["run"].replace("-","_")+"_operator_evidence.json")
    assert path.read_bytes()==raw
    historical=copy.deepcopy(cfg);historical.pop("provisioning_model")
    assert cloud.cpu_packet_sha256(historical)==receipt["packet_sha256"]
    assert receipt["credit_rate"]["machine"]==cfg["machine"]
    assert receipt["credit_rate"]["spot_usd_per_hour"]==cfg["usd_per_hour"]
    rejects(cloud.verify_cpu_operator_evidence,cfg)
    # Even restoring the historical source binding cannot revive the old schema/packet.
    with patch.object(cloud,"cpu_source_sha256",return_value=receipt["source_sha256"]):
        rejects(cloud.verify_cpu_operator_evidence,cfg)
    print("PASS historical CPU receipt bytes/hash/packet/rate custody; new packet/schema requires operator refresh")


def cpu_dry_assembly():
    path=cloud.HERE/"configs/associative_continuation_cpu_smoke.json"
    c=json.loads(path.read_text())
    with patch.object(cloud,"upload_bundle",side_effect=forbidden), \
         patch.object(cloud,"create_vm",side_effect=forbidden):
        cloud.launch(path,dry=True)
        for hours, name in ((1,"associative_continuation_cpu_smoke.json"),
                            (3,"associative_continuation_cpu_smoke_3h.json")):
            variant=json.loads((cloud.HERE/"configs"/name).read_text())
            idx,cost=cloud.launch_plan(variant)
            assert idx==[0] and abs(cost-(cloud.rate_entry(variant)["usd_per_hour"]*hours+cloud.ancillary(variant)))<1e-9
            assert variant["boot_disk_type"]=="pd-standard"
            assert variant["common"]["job_stop_seconds"]==hours*3600-900
            assert variant["credit_guard_epoch"]==1792193400
            if hours == 1:
                verify_shipped_cpu_evidence(variant)
                with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{}):
                    rejects(cloud.verify_cpu_operator_evidence,variant)
                print("PASS 1h CPU evidence removed in memory: rejected")
            else:
                rejects(cloud.verify_cpu_operator_evidence,variant)
                receipt=json.loads(cloud.RELEASE_CPU_OPERATOR_EVIDENCE[c["operator_evidence_sha256"]])
                # Isolate lifetime/packet rejection from the edited source hash.
                with patch.object(cloud,"cpu_source_sha256",return_value=receipt["source_sha256"]):
                    rejects(cloud.verify_cpu_operator_evidence,variant)
                print("PASS 3h CPU packet not covered by shipped evidence: rejected")
            shard=run.assemble_shard(variant,0)
            contract=cloud.dispatch_contract(variant,100000,"a"*64)
            shard.update(dispatch=contract,requirements=variant["requirements"])
            raw=(json.dumps(shard,sort_keys=True,separators=(",",":"))+"\n").encode()
            trusted=cloud.attempt_manifest(shard,raw)
            assert trusted["shard_sha256"]==hashlib.sha256(raw).hexdigest()
            assert trusted["bundle_sha256"]=="a"*64  # Explicit mock, not a released bundle.
            assert contract["requirements_lock_sha256"]==cloud.RELEASE_PINS["requirements_lock_sha256"]
            assert shard["requirements_lock_sha256"]==contract["requirements_lock_sha256"]
            assert contract["has_gpu"] is False and shard["model"] is None
            assert all(contract[k] is None for k in ("weights_local_path","weights_mirror_uri",
                                                    "weights_manifest_uri","weights_manifest_sha256"))
            assert (contract["lifetime_seconds"],contract["work_stop_seconds"],
                    contract["shutdown_reserve_seconds"])==(hours*3600,hours*3600-900,900)
            assert contract["work_stop_at"]==100000+hours*3600-900
            assert contract["shutdown_at"]==100000+hours*3600
            print(f"PASS CPU dry assembly {hours}h: plan-shaped only, {len(idx)} shard, ${cost:.4f}, "
                  f"lifetime/work/reserve={hours*3600}/{hours*3600-900}/900; "
                  "release image/lock and null assets preserved; mock bundle/clock; no upload; shape/rate flags from packet")


def cpu_bundle():
    """Execute frozen control from the REAL archive, no checkout on child path.

    Windows stages its installed interpreter as python3.exe; Linux requires a CPU-capable torch
    interpreter. A WSL without torch delegates this same archive check to Windows.
    This checks packaging, not pinned-image lock or Landlock compatibility.
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        executable = os.environ.get("LSR_CPU_CHECK_PYTHON")
        if not executable:
            raise RuntimeError("CPU bundle control requires installed torch; no install/network fallback")
        with patch.object(subprocess, "Popen", _REAL_POPEN), patch.object(time, "sleep", _REAL_SLEEP):
            result = _REAL_RUN([executable, "-B", "-m", __package__ + ".dispatch_checks",
                               "--check", "cpu-bundle", "--offline"], cwd=str(cloud.REPO), timeout=180)
        assert result.returncode == 0
        return
    cfg = json.loads((cloud.HERE/"configs/associative_continuation_cpu_smoke.json").read_text())
    raw = cloud.build_bundle(cfg)
    assert raw == cloud.build_bundle(cfg), "unchanged packet must produce identical archive bytes"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)/"opt"/"lsr"
        root.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            tar.extractall(root, filter="data")
        manifest = json.loads((root/"bundle_manifest.json").read_text())
        for name, digest in manifest.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
        assert (root/"lsr_jobs/s0_memory_port.py").read_bytes() == (root/"experiments/cloud/lsr_jobs/s0_memory_port.py").read_bytes()
        out = root/"out";out.mkdir()
        for folder in ("tmp", "offline-hf"):
            (out/folder).mkdir()
        # No inherited PYTHONPATH, user site, HOME, credentials or checkout cwd.
        env = {k: v for k, v in os.environ.items() if k.lower() in
               {"path", "systemroot", "windir", "localappdata"}}
        env.update(PYTHONPATH=str(root), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", PIP_NO_INDEX="1", LSR_OUT=str(out),
                   HF_HOME=str(out/"offline-hf"), TMPDIR=str(out/"tmp"), TEMP=str(out/"tmp"), TMP=str(out/"tmp"))
        if os.name == "nt":
            # Windows Store python3 alias is not an interpreter. Preserve the
            # configured argv with a local copy of the actual installed executable.
            # PYTHONHOME names only its stdlib/site-packages, never a repo checkout.
            bin_dir=Path(td)/"bin";bin_dir.mkdir()
            shutil.copyfile(sys.executable,bin_dir/"python3.exe")
            env["PYTHONHOME"]=sys.prefix
            env["PATH"]=str(bin_dir)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+env["PATH"]
        # Exact configured argv. Interpreter PATH is the local installed runtime;
        # operator must repeat this on the pinned image before releasing it.
        command = cfg["common"]["runtime_control_command"]
        def execute(argv):
            with patch.object(subprocess, "Popen", _REAL_POPEN), patch.object(time, "sleep", _REAL_SLEEP):
                executable=shutil.which(argv[0],path=env["PATH"])
                if executable is None:
                    raise RuntimeError("configured interpreter not found in clean PATH")
                return _REAL_RUN([executable,*argv[1:]], cwd=root, env=env, capture_output=True, text=True, timeout=150)
        control = execute(command)
        assert control.returncode == 0, (control.returncode, control.stdout[-3000:], control.stderr[-1000:])
        assert '"status": "PASS"' in control.stdout
        probe = execute([command[0], "-B", "-c",
            "from pathlib import Path; import sys; "
            "from experiments.cloud.lsr_jobs import s0_memory_port as p; "
            "from experiments.cloud.lsr_jobs import s0_envelope,gemma4_receiver; "
            "from experiments.memory import associative_continuation_cpu as c; "
            "assert Path(p.__file__).resolve().is_relative_to(Path.cwd()); "
            "assert callable(c.boundary_read); "
            "assert all('Latent-Space-Reasoning' not in x for x in sys.path); "
            "print('PASS isolated work imports; no fit/model')"])
        assert probe.returncode == 0, probe.stderr
        print(probe.stdout.strip())
        (root/"experiments/cloud/lsr_jobs/s0_memory_port.py").unlink()
        negative = execute(command)
        assert negative.returncode != 0 and "FileNotFoundError" in negative.stdout
        print("PASS extracted actual bundle exact-controls exit=0; removed required helper exit="
              + str(negative.returncode) + "; archive_sha256=" + hashlib.sha256(raw).hexdigest())


def cpu_progress():
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        executable=os.environ.get("LSR_CPU_CHECK_PYTHON")
        if not executable:
            raise RuntimeError("CPU progress check requires torch")
        with patch.object(subprocess,"Popen",_REAL_POPEN),patch.object(time,"sleep",_REAL_SLEEP):
            result=_REAL_RUN([executable,"-B","-m",__package__+".dispatch_checks",
                              "--check","cpu-progress","--offline"],cwd=cloud.REPO,timeout=60)
        assert result.returncode==0
        return
    import torch
    with tempfile.TemporaryDirectory() as td,patch.object(run.time,"time",return_value=100060):
        root=Path(td);value=torch.nn.Parameter(torch.tensor(1.0));optimizer=torch.optim.SGD([value],lr=0.1)
        with run.cpu_work_progress(root):
            assert not (root/"heartbeat_work").exists(), "no progress before real step"
            value.grad=torch.tensor(1.0);optimizer.step()
            first=json.loads((root/"heartbeat_work").read_text());assert first["optimizer_steps"]==1
            value.grad=torch.tensor(1.0);optimizer.step()
            second=json.loads((root/"heartbeat_work").read_text())
            assert second==dict(stage="work",epoch=100060,optimizer_steps=2,source="optimizer-step-completed")
        value.grad=torch.tensor(1.0);optimizer.step()
        assert json.loads((root/"heartbeat_work").read_text())==second, "hook leaked after exit"
    print("PASS actual CPU optimizer post-step progress; silent before steps; counters advance; hook removed")


def cpu_operator_gate():
    cfg=json.loads((cloud.HERE/"configs/associative_continuation_cpu_smoke.json").read_text())
    # Shipped operator evidence is now present: acceptance is the positive case.
    digest=cfg["operator_evidence_sha256"]
    raw=(cloud.HERE/"receipts"/(cfg["run"].replace("-","_")+"_operator_evidence.json")).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==digest
    assert cloud.RELEASE_CPU_OPERATOR_EVIDENCE[digest]==raw
    shipped_receipt=json.loads(raw)
    with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{}), tempfile.TemporaryDirectory() as td:
        rejects(cloud.verify_cpu_operator_evidence,cfg)
        path=Path(td)/"config.json";path.write_text(json.dumps(cfg))
        rejects(cloud.launch,path)  # Global raising cloud mocks must never be reached.
    with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{digest:raw+b" "}):
        rejects(cloud.verify_cpu_operator_evidence,cfg)
    # Synthetic current-source copies isolate semantic negatives even when editing
    # this bundled check invalidates the shipped receipt's source binding.
    current_receipt=copy.deepcopy(shipped_receipt)
    current_receipt["source_sha256"]=cloud.cpu_source_sha256()
    # Rehash mutations so each semantic rejection passes the outer hash check.
    for field in ("packet_sha256","source_sha256",*sorted(cloud.CPU_PRELAUNCH_CHECKS)):
        bad=copy.deepcopy(current_receipt)
        if field in cloud.CPU_PRELAUNCH_CHECKS:
            bad["checks"][field]="FAIL"
        else:
            bad[field]="0"*64 if bad[field]!="0"*64 else "1"*64
        altered_raw=json.dumps(bad,sort_keys=True).encode()
        altered_digest=hashlib.sha256(altered_raw).hexdigest()
        altered=copy.deepcopy(cfg);altered["operator_evidence_sha256"]=altered_digest
        with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{altered_digest:altered_raw}):
            rejects(cloud.verify_cpu_operator_evidence,altered)
    for field in ("rate_verified","shape_verified"):
        altered=copy.deepcopy(cfg);altered[field]=False
        bad=copy.deepcopy(current_receipt);bad["packet_sha256"]=cloud.cpu_packet_sha256(altered)
        altered_raw=json.dumps(bad,sort_keys=True).encode()
        altered_digest=hashlib.sha256(altered_raw).hexdigest()
        altered["operator_evidence_sha256"]=altered_digest
        with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{altered_digest:altered_raw}):
            rejects(cloud.verify_cpu_operator_evidence,altered)
    print("PASS empty evidence/hash mismatch/flipped checks/"
          "false rate/false shape/changed packet/changed source rejected in memory")
    for tag in ("a01","a02","a03","a04","a05"):
        historical_raw=(cloud.HERE/"receipts"/f"assoc_cpu_smoke_{tag}_operator_evidence.json").read_bytes()
        historical_receipt=json.loads(historical_raw)
        assert set(historical_receipt["checks"])==cloud.CPU_PRELAUNCH_CHECKS
        assert set(historical_receipt["checks"].values())=={"PASS"}
        rate=historical_receipt["credit_rate"]
        assert rate["machine"]=="n2-standard-64"
        assert rate["spot_usd_per_hour"]==PRODUCTION_RATE_TABLE[("n2-standard-64","SPOT")]["usd_per_hour"]
        assert rate["on_demand_usd_per_hour"]==PRODUCTION_RATE_TABLE[("n2-standard-64","STANDARD")]["usd_per_hour"]
        assert all(rate[k]==v for k,v in cloud.CREDIT_RATE_CONTEXT.items())
        for artifacts in historical_receipt["artifacts"].values():
            for artifact in artifacts:
                assert hashlib.sha256((cloud.REPO/artifact["path"]).read_bytes()).hexdigest()==artifact["sha256"]
        packet=copy.deepcopy(cfg);packet["run"]=f"assoc-cpu-smoke-{tag}"
        h=hashlib.sha256(historical_raw).hexdigest();packet["operator_evidence_sha256"]=h
        with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{h:historical_raw}):
            rejects(cloud.verify_cpu_operator_evidence,packet)
        print(f"PASS historical {tag} credit/rate/check/artifact custody; new schema/packet refused")
    verify_shipped_cpu_evidence(cfg)
    cfg.update(rate_verified=True,shape_verified=True)
    obligations=cloud.CPU_PRELAUNCH_CHECKS
    receipt={"packet_sha256":cloud.cpu_packet_sha256(cfg), "source_sha256":cloud.cpu_source_sha256(),
             "credit_rate":cloud.rate_entry(cfg), "checks":{k:"PASS" for k in obligations},
             "artifacts":{k:[{"path":"synthetic-offline-only", "sha256":"a"*64}]
                          for k in obligations - {"credit_rate"}}}
    raw=json.dumps(receipt,sort_keys=True).encode();digest=hashlib.sha256(raw).hexdigest()
    cfg["operator_evidence_sha256"]=digest
    with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{digest:raw}):
        cloud.verify_cpu_operator_evidence(cfg)
        for key,value in (("usd_per_hour",2.0),("boot_disk_type","pd-balanced"),
                          ("ancillary_allowance_usd",0),("operator_evidence_sha256","f"*64)):
            bad=copy.deepcopy(cfg);bad[key]=value;rejects(cloud.verify_cpu_operator_evidence,bad)
        archive=cloud.build_bundle(cfg)
        contract=cloud.dispatch_contract(cfg,100000,hashlib.sha256(archive).hexdigest())
        contract["shard_sha256"]="b"*64
        fake=FakeCloud(cfg,contract)
        with tempfile.TemporaryDirectory() as td,patch.object(cloud,"gcloud",fake), \
             patch.object(time,"time",return_value=100060),patch.object(cloud,"REGISTRY",Path(td)/"journal"):
            prefix=f"{cloud.BUCKET}/runs/{cfg['run']}/shards/000"
            assert cloud.create_vm(cfg,0,prefix,contract)==cfg["zones"][0]
            assert "--boot-disk-type=pd-standard" in fake.creation
            count=len(fake.calls)
            with patch.object(cloud,"build_bundle",return_value=b"altered staged archive"):
                rejects(cloud.create_vm,cfg,0,prefix,contract)
            assert len(fake.calls)==count, "altered packet reached cloud boundary"
    # Each of the five still needs evidence; neither runtime item is required.
    assert set(receipt["checks"]) == {"credit_rate", "shape_disk", "ancillary_retention",
                                      "image_lock_interpreter", "iam_scopes_prefix"}
    for key in obligations:
        bad=copy.deepcopy(receipt);bad["checks"].pop(key)
        raw_bad=json.dumps(bad,sort_keys=True).encode();h=hashlib.sha256(raw_bad).hexdigest()
        altered=copy.deepcopy(cfg);altered["operator_evidence_sha256"]=h
        with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{h:raw_bad}):
            rejects(cloud.verify_cpu_operator_evidence,altered)
    for key,value in (("source_sha256","f"*64),("credit_rate",{}),("artifacts",{})):
        bad=copy.deepcopy(receipt);bad[key]=value
        raw_bad=json.dumps(bad,sort_keys=True).encode();h=hashlib.sha256(raw_bad).hexdigest()
        altered=copy.deepcopy(cfg);altered["operator_evidence_sha256"]=h
        with patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{h:raw_bad}):
            rejects(cloud.verify_cpu_operator_evidence,altered)
    print("PASS five prelaunch checks only; omitted checks/source/rate/artifact negatives; runtime items absent")
    # Late-window exposure rejected at the actual registered guard.
    rejects(cloud.dispatch_contract,cfg,1792190700,"a"*64)
    assert cloud.dispatch_contract(cfg,1792189800,"a"*64)["shutdown_at"]==1792193400
    print("PASS independent CPU operator gate: missing/self-declared/tampered evidence and late cutoff rejected")


def bundle_readback_negative():
    cfg=json.loads((cloud.HERE/"configs/associative_continuation_cpu_smoke.json").read_text())
    calls=[]
    def effect(*argv,**kw):
        assert argv[:2]==("storage","cp")
        calls.append(argv)
        if argv[2].startswith("gs://"):
            Path(argv[3]).write_bytes(b"wrong remote object bytes")
        else:
            assert "--if-generation-match=0" in argv
        return ""
    with patch.object(cloud,"gcloud",effect):
        rejects(cloud.upload_bundle,cfg["run"],cfg)
    assert len(calls)==2
    print("PASS corrupt bundle readback refuses release digest; mocked storage only")


def terminal_reconciliation():
    with tempfile.TemporaryDirectory() as td:
        registry=Path(td)/"journal"
        row={"run":"offline-a01","shard":0,"dispatch":{"shutdown_at":100100}}
        registry.write_text(json.dumps(row)+"\n")
        def effect(*argv,**kw):
            assert argv[:2]==("storage","ls");return ""
        with patch.object(cloud,"REGISTRY",registry),patch.object(cloud,"gcloud",effect), \
             patch.object(time,"time",return_value=100101):
            assert cloud.shard_state("offline-a01",0).startswith("failed/unknown")
        row["vm"]=cloud.vm_name("offline-a01",0);registry.write_text(json.dumps(row)+"\n")
        with patch.object(cloud,"REGISTRY",registry),patch.object(cloud,"gcloud",effect), \
             patch.object(time,"time",return_value=100050),patch.object(cloud,"list_vms",return_value=[]):
            assert cloud.shard_state("offline-a01",0).startswith("failed/unknown")
    print("PASS expired/disappeared missing custody requires operator reconciliation; never success")


def cpu_support():
    c=fixture();c.update(machine="n2-standard-64",usd_per_hour=1.864192,cost_cap_usd=7.456768,
                         aggregate_cost_ceiling_usd=7.456768,has_gpu=False,guest_accelerators=[],
                         common={"runtime_control_command":["offline-control"]},weights_mirror_uri=None)
    assert "accelerator" not in c
    assert cloud.launch_plan(c)[1]==7.456768
    assert run.assemble_shard(c,0)["entry"]=="main"
    fake=creation(c)
    assert not any(x.startswith("--accelerator") for x in fake.creation)
    assert "lsr-has-gpu=false" in next(x for x in fake.creation if x.startswith("--metadata="))
    c["accelerator"]="nvidia-l4:1";rejects(cloud.validate_config,c)



def request_mutations():
    # Deleting OR changing every requested flag must trip the independent oracle.
    configs=[fixture()]
    attached=fixture();attached.update(machine="n1-standard-8",accelerator="nvidia-tesla-t4:1",
        guest_accelerators=[{"type":"nvidia-tesla-t4","count":1}]);configs.append(attached)
    count=0
    for c in configs:
        good=creation(c).creation
        for index in range(4,len(good)):
            for changed in (False,True):
                fake=FakeCloud(c,dict(cloud.dispatch_contract(c,100000,"a"*64), shard_sha256="b"*64))
                def effect(*args,**kwargs):
                    if args[:3]==("compute","instances","create"):
                        args=list(args)
                        if changed: args[index]=args[index].split("=",1)[0]+"=MUTATED"
                        else: del args[index]
                    return fake(*args,**kwargs)
                with patch.object(cloud,"gcloud",effect),patch.object(time,"time",return_value=100060):
                    try:
                        cloud.create_vm(c,0,f"{cloud.BUCKET}/runs/{c['run']}/shards/000",fake.contract)
                    except (AssertionError, ValueError, KeyError):
                        count+=1
                    else:
                        raise AssertionError(f"request mutation escaped: {good[index]}")
        for extra in ("--unknown=x",good[4]):
            try: FakeCloud(c,{})(*good,extra)
            except AssertionError: pass
            else: raise AssertionError("unknown/duplicate flag accepted")
    print(f"PASS request mutations: {count} deleting/changing critical-flag cases rejected")


def image_accelerators():
    c=fixture()
    for mutate in (lambda a:a.pop("guestAccelerators"),
                   lambda a:a["guestAccelerators"][0].update(acceleratorType="nvidia-wrong"),
                   lambda a:a["guestAccelerators"][0].update(acceleratorCount=1)):
        creation(c,mutate)
    contract=dict(cloud.dispatch_contract(c,100000,"a"*64), shard_sha256="b"*64);fake=FakeCloud(c,contract)
    def wrong_project(*args,**kwargs):
        value=fake(*args,**kwargs)
        if args[:3]==("compute","disks","describe"):
            disk=json.loads(value);disk["sourceImage"]=f"projects/wrong-project/global/images/{c['image_name']}"
            return json.dumps(disk)
        return value
    with patch.object(cloud,"gcloud",wrong_project),patch.object(time,"time",return_value=100060):
        rejects(cloud.create_vm,c,0,f"{cloud.BUCKET}/runs/{c['run']}/shards/000",contract)
    for wrong in (None, "987654321", 123456789):
        def wrong_id(*args,**kwargs):
            value=fake(*args,**kwargs)
            if args[:3]==("compute","disks","describe"):
                disk=json.loads(value)
                if wrong is None: disk.pop("sourceImageId")
                else: disk["sourceImageId"]=wrong
                return json.dumps(disk)
            return value
        with patch.object(cloud,"gcloud",wrong_id),patch.object(time,"time",return_value=100060):
            rejects(cloud.create_vm,c,0,f"{cloud.BUCKET}/runs/{c['run']}/shards/000",contract)
    for wrong in (None, "MISSING", "", 123456789):
        bad=fixture();bad["image_id"]=wrong;rejects(cloud.validate_config,bad)
    bad=fixture();bad.pop("image_id");bad["release"]=True;rejects(cloud.validate_config,bad)
    bad["release"]=False;cloud.validate_config(bad)
    cpu=fixture();cpu.update(machine="e2-standard-8",has_gpu=False,guest_accelerators=[])
    creation(cpu)
    creation(cpu,lambda a:a.update(guestAccelerators=[{"acceleratorType":"nvidia-tesla-t4","acceleratorCount":1}]))
    for field,value in (("has_gpu",None),("guest_accelerators",[]),("machine","unreviewed-8")):
        bad=fixture();bad[field]=value;rejects(cloud.validate_config,bad)


def control_pins():
    c=fixture();shard=run.assemble_shard(c,0);contract=dict(cloud.dispatch_contract(c,100000,"a"*64), shard_sha256="b"*64)
    assert contract["runtime_control_sha256"]==hashlib.sha256(json.dumps(shard["runtime_control_command"],separators=(",",":")).encode()).hexdigest()
    assert shard["runtime_control_sha256"]==contract["runtime_control_sha256"]
    assert shard["requirements_lock_sha256"]==c["requirements_lock_sha256"]
    assert shard["weights_manifest_sha256"]==c["weights_manifest_sha256"]
    for field in ("attempt", "runtime_control_sha256", "requirements_lock_sha256", "weights_manifest_sha256"):
        bad=fixture();bad["shards"][0][field]="forged";rejects(run.assemble_shard,bad,0)
    for value in (None,[],[""],[" "],"python control",[1]):
        bad=fixture();bad["common"]["runtime_control_command"]=value;rejects(cloud.validate_config,bad)
    for field in ("weights_manifest_uri","weights_manifest_sha256","requirements_lock_sha256"):
        bad=fixture();bad.pop(field);rejects(cloud.validate_config,bad)
    for field in ("revision","tokenizer_revision"):
        bad=fixture();bad["common"].pop(field);rejects(cloud.validate_config,bad)
    for field in ("cost_cap_usd","usd_per_hour"):
        bad=fixture();bad["common"][field]=1;rejects(cloud.validate_config,bad)
    for name in ("e1_gemma4_smoke.json","e1_gemma4_alignment_hook_v1.json"):
        cfg=json.loads((cloud.HERE/"configs"/name).read_text())
        assert cfg["cost_cap_usd"]==cfg["common"]["cost_cap_usd"]==cfg["usd_per_hour"]*4


def exact_shutdown(out, expected, rc, fault, total=14400, work=13500):
    """Actual production shell, private PATH stubs, bounded local bash only."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ("mkdir", "mv", "cat", "sha256sum", "setsid", "timeout", "flock", "python3"):
            target = shutil.which(name)
            assert target, name
            path = root/name
            path.write_text('#!/bin/bash\nexec "' + target + '" "$@"\n')
            path.chmod(0o755)
        for name, body in (("date", "echo 100060"), ("gcloud", "exit " + ("1" if fault == "custody" else "0"))):
            path = root/name
            path.write_text("#!/bin/bash\n" + body + "\n")
            path.chmod(0o755)
        argv = [shutil.which("bash"), str(cloud.HERE/"vm_shutdown.sh")]
        def allowed(argv_actual, **kw):
            assert argv_actual == argv and kw.get("env", {}).get("PATH") == str(root)
            return _REAL_POPEN(argv_actual, **kw)
        env = {**os.environ, "PATH":str(root), "LSR_OUT":str(out), "LSR_CUSTODY_STATE":str(root/"custody"), "LSR_JOB":"gs://offline/job/shards/000",
               "LSR_CREATED_EPOCH":str(expected["created_epoch"]), "LSR_BUNDLE_SHA256":expected["bundle_sha256"],
               "LSR_RUN":expected["run"], "LSR_SHARD":str(expected["shard"]), "LSR_ATTEMPT":expected["attempt"],
               "LSR_SHARD_SHA256":expected["shard_sha256"], "LSR_EXIT_CODE":str(rc), "LSR_END_REASON":"shutdown",
               "LSR_LIFETIME_SECONDS":str(total), "LSR_WORK_STOP_SECONDS":str(work), "LSR_SHUTDOWN_RESERVE_SECONDS":"900",
               "LSR_CREDIT_GUARD_EPOCH":str(cloud.CREDIT_GUARD_EPOCH)}
        with patch.object(subprocess, "Popen", allowed):
            result = _REAL_RUN(argv, env=env, capture_output=True, timeout=10)
        assert result.returncode == (1 if fault == "custody" else rc), result.stderr
        return (out/"shutdown_receipt.json").read_text()


def receipt_integration():
    from . import s0_runtime as runtime
    import types
    runtime_path=Path(runtime.__file__)
    runtime_hash=hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    c=json.loads((cloud.HERE/"configs/associative_continuation_cpu_smoke.json").read_text())
    cloud.validate_config(c)
    contract=dict(cloud.dispatch_contract(c,100000,"a"*64), shard_sha256="b"*64)
    for fault in ("success","control","custody","probe","work_stall","swapped-identity","missing-authority"):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/"out";out.mkdir();shard=run.assemble_shard(c,0)
            shard.update(dispatch=contract,requirements=c["requirements"])
            path=root/"shard.json";path.write_bytes(json.dumps(shard).encode("utf-8"));(out/"heartbeat").touch();os.utime(out/"heartbeat",(100060,100060))
            expected=cloud.attempt_manifest(shard,path.read_bytes())
            registry=root/"registry"
            registry.write_text(json.dumps(dict(record_type="attempt_expected",run=c["run"],shard=0,expected=expected))+"\n")
            children=[];remote={}
            class Child:
                def __init__(self,argv):
                    children.append(list(argv));self.index=len(children)
                def poll(self):
                    return 1 if fault=="control" and self.index==1 else None if fault=="work_stall" and self.index==2 else 0
                def stop(self): pass
            def bounded(argv,budget):
                if "--custody-exec" in argv:
                    assert argv[argv.index("--")+1:argv.index("--")+4] == ["gcloud","storage","cp"]
                    assert argv[-2:] == ["gs://offline/job/shards/000/out/control_upload_probe.json", "--if-generation-match=0"]
                    assert 0 < budget <= 60
                    if fault == "probe":
                        raise OSError("injected restricted VM-account upload failure")
                    return
                assert argv[:3] in (["gcloud","storage","rsync"],["gcloud","storage","cp"])
                assert 0<budget<=870
                if argv[2]=="rsync":
                    remote.update({p.name:p.read_text() for p in out.glob("*_receipt.json")})
                else: remote["runtime_receipt.json"]=(out/"runtime_receipt.json").read_text()
            lock=root/"requirements.txt"
            lock.write_bytes(("\n".join(c["requirements"])+"\n").encode())
            installed=[types.SimpleNamespace(metadata={"Name":line.split("==")[0]},version=line.split("==")[1]) for line in c["requirements"]]
            args=types.SimpleNamespace(shard=str(path),out=str(out),created_epoch=expected["created_epoch"],
                bundle_sha256=expected["bundle_sha256"],job="gs://offline/job/shards/000",
                expected_run=expected["run"],expected_shard=expected["shard"],
                expected_attempt=expected["attempt"] if fault!="swapped-identity" else "other-attempt",
                expected_shard_sha256=None if fault=="missing-authority" else expected["shard_sha256"],requirements_path=str(lock))
            with patch.object(runtime.metadata,"distributions",return_value=installed), \
                 patch.object(runtime,"Child",Child),patch.object(runtime,"bounded_command",bounded), \
                 patch.object(runtime.time,"time",return_value=100060),patch.object(runtime.time,"sleep",forbidden), \
                 patch.dict(os.environ,{"LSR_CUSTODY_DEFERRED":"0"}):
                # Runtime defaults captured wall time at import; inject the production class clock.
                base=runtime.Runtime
                tick=[100060]
                def sleep(_):
                    assert fault == "work_stall"
                    tick[0] += 301
                def factory(*a,**kw): return base(*a,clock=lambda:tick[0],sleep=sleep,**kw)
                with patch.object(runtime,"Runtime",factory): rc=runtime.run(args)
            terminal=json.loads((out/"runtime_receipt.json").read_bytes())
            if fault in ("swapped-identity","missing-authority"):
                assert rc==1 and not children and terminal["reason"]==("shard_identity" if fault=="swapped-identity" else "shard_hash"), terminal
                print(f"PASS runtime {fault}: nonzero refusal; zero children")
                continue
            assert children, f"runtime spawned no control; rc={rc}, receipt={terminal}"
            assert children[0]==shard["runtime_control_command"]
            control_receipt=json.loads((out/"control/terminal_receipt.json").read_text())
            assert control_receipt["status"] == ("failed" if fault=="control" else "completed")
            assert control_receipt["hashes"]["shard_sha256"]==expected["shard_sha256"]
            assert control_receipt["hashes"]["bundle_sha256"]==expected["bundle_sha256"]
            assert control_receipt["verification"]["runtime_control_sha256"]==shard["runtime_control_sha256"]
            assert len(children)==(1 if fault in ("control", "probe") else 2)
            assert rc==(1 if fault in ("control", "probe", "work_stall") else 0)
            if fault == "probe":
                assert terminal["reason"] == "landlock_control_upload"
                assert terminal["verification"]["landlock_control_upload"] == "FAIL"
                assert terminal["verification"]["work_progress"] == "NOT_RUN"
            if fault == "work_stall":
                assert terminal["reason"] == "stall"
                assert terminal["verification"]["work_progress"] == "FAIL"
            if fault == "success":
                assert terminal["verification"]["landlock_control_upload"] == "PASS"
                assert terminal["verification"]["work_progress"] == "PASS"
            terminal=json.loads((out/"runtime_receipt.json").read_text())
            assert terminal["hashes"]["shard_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
            # Bind a shell-protocol fixture to the exact production runtime bytes.
            # Shell execution is covered by the sibling runtime suite, not this check.
            assert terminal["identity"]=={k:expected[k] for k in ("run","shard","attempt")}
            assert remote["runtime_receipt.json"]==(out/"runtime_receipt.json").read_text()
            # No receipt augmentation: consume bytes emitted by the actual shell.
            remote["shutdown_receipt.json"] = exact_shutdown(out, expected, rc, fault)
            shutdown = json.loads(remote["shutdown_receipt.json"])
            def storage(*argv,**kw):
                if argv[:2]==("storage","ls"):return "\n".join("gs://offline/"+n for n in remote)
                if argv[:2]==("storage","cat"):return remote[argv[2].rsplit("/",1)[-1]]
                raise AssertionError(argv)
            with patch.object(cloud,"gcloud",storage),patch.object(cloud,"REGISTRY",registry):
                state=cloud.shard_state(c["run"],0)
                def status_storage(*argv,**kw):
                    if argv[:2]==("storage","cat") and argv[2].endswith("config.json"):
                        return json.dumps(c)
                    return storage(*argv,**kw)
                capture=io.StringIO()
                with patch.object(cloud,"gcloud",status_storage),patch.object(cloud,"list_vms",return_value=[]),redirect_stdout(capture):
                    cloud.status(c["run"])
                assert f"shard 000: {state}" in capture.getvalue()
            assert (state=="done")== (fault=="success"),state
            assert fault=="success" or state.startswith("failed("),state
            # Pull uses the same parser against files actually copied by its raising mock.
            with tempfile.TemporaryDirectory() as dst:
                def copy(*argv,**kw):
                    assert argv[:3]==("storage","rsync","-r")
                    target=Path(argv[-1]);(target/"config.json").write_text(json.dumps(c))
                    output=target/"shards/000/out";output.mkdir(parents=True)
                    for n,v in remote.items():(output/n).write_bytes(v.encode("utf-8"))
                    return ""
                capture=io.StringIO()
                with patch.object(cloud,"REPO",Path(dst)),patch.object(cloud,"gcloud",copy),patch.object(cloud,"REGISTRY",registry),redirect_stdout(capture):
                    cloud.pull(c["run"])
                assert f"shard 000: {state}" in capture.getvalue()
            assert cloud.receipt_state({})=="pending"
            assert cloud.receipt_state({"runtime_receipt.json":remote["runtime_receipt.json"]}).startswith(("incomplete", "failed"))
            for name in remote:
                for malformed in ("{bad", "[]", json.dumps({"status":"completed","rc":0}),
                                  json.dumps(dict(json.loads(remote[name]),hashes=[]))):
                    broken=dict(remote);broken[name]=malformed
                    assert cloud.receipt_state(broken,expected=expected).startswith("failed(malformed")
            if fault=="control":
                # A successful outer wrapper cannot erase an inner work failure.
                outer=dict(shutdown,status="completed",rc=0,finalized=True)
                assert cloud.receipt_state(dict(remote,**{"shutdown_receipt.json":json.dumps(outer)})).startswith("failed(runtime")
            if fault=="success":
                shutdown["finalized"]=False;pending=dict(remote,**{"shutdown_receipt.json":json.dumps(shutdown)})
                assert cloud.receipt_state(pending,expected=expected)=="incomplete(custody-unfinalized)"
                for key,bad in (("hashes",{"bundle_sha256":"f"*64}),("clock",{"created_epoch":100001})):
                    mismatch=dict(json.loads(remote["shutdown_receipt.json"]));mismatch[key]=bad
                    assert cloud.receipt_state(dict(remote,**{"shutdown_receipt.json":json.dumps(mismatch)})).startswith("failed(receipt-")
            print(f"PASS actual runtime + exact shutdown shell bytes -> status/pull: {fault} = {state}")
    assert hashlib.sha256(runtime_path.read_bytes()).hexdigest()==runtime_hash, "runtime source changed during integration"


def receipt_binding():
    # Independent prospective protocol fixtures; no production manifest builder.
    c=fixture()
    expected=dict(run=c["run"],shard=0,attempt="sc-bench-lsr-offline-a01-000",
                  shard_sha256="b"*64,bundle_sha256="a"*64,created_epoch=100000)
    identity={k:expected[k] for k in ("run","shard","attempt")}
    runtime=dict(status="completed",rc=0,reason="completed",finalized=False,
                 identity=identity,hashes=dict(shard_sha256="b"*64,bundle_sha256="a"*64),
                 clock=dict(created_epoch=100000))
    def pair(r, s=None):
        raw=json.dumps(r)
        if s is None:
            s=copy.deepcopy(r);s.update(finalized=True,reason="shutdown")
        s=copy.deepcopy(s)
        s["hashes"]["runtime_receipt_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
        return {"runtime_receipt.json":raw,"shutdown_receipt.json":json.dumps(s)}
    good=pair(runtime);cases=[("matched",good,"done")]
    for key,bad in (("run","other-run"),("shard",1),("attempt","other-attempt")):
        r=copy.deepcopy(runtime);r["identity"][key]=bad
        cases.append(("swapped-"+key,pair(r),"failed(receipt-identity-binding)"))
    for key,bad in (("shard_sha256","f"*64),("bundle_sha256","f"*64)):
        r=copy.deepcopy(runtime);r["hashes"][key]=bad
        cases.append(("equal-wrong-"+key,pair(r),"failed(receipt-attempt-binding)"))
    r=copy.deepcopy(runtime);r["clock"]["created_epoch"]=100001
    cases.append(("equal-wrong-epoch",pair(r),"failed(receipt-attempt-clock-binding)"))
    r=copy.deepcopy(runtime);r.pop("identity")
    cases.append(("missing-identity",pair(r),"failed(receipt-identity-binding)"))
    r=copy.deepcopy(runtime);r["hashes"].pop("shard_sha256")
    cases.append(("missing-shard-hash",pair(r),"failed(receipt-attempt-binding)"))
    r=copy.deepcopy(runtime);r["identity"]["shard"]=False
    cases.append(("boolean-shard",pair(r),"failed(receipt-identity-binding)"))
    for key in ("identity","hashes","clock"):
        s=copy.deepcopy(json.loads(good["shutdown_receipt.json"]))
        field,bad={"identity":("attempt","other-attempt"),"hashes":("shard_sha256","f"*64),
                   "clock":("created_epoch",100001)}[key]
        s[key][field]=bad
        cases.append(("swapped-outer-"+key,pair(runtime,s),"failed(receipt-"))
    for value in (None,"f"*64):
        broken=copy.deepcopy(good);s=json.loads(broken["shutdown_receipt.json"])
        if value is None:s["hashes"].pop("runtime_receipt_sha256")
        else:s["hashes"]["runtime_receipt_sha256"]=value
        broken["shutdown_receipt.json"]=json.dumps(s)
        cases.append(("custody-digest",broken,"failed(receipt-custody-binding)"))
    broken=dict(good);broken["runtime_receipt.json"]+="\n"
    cases.append(("changed-runtime-bytes",broken,"failed(receipt-custody-binding)"))
    assert cloud.receipt_state(good)=="failed(expected-attempt-missing)"
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);registry=root/"registry"
        row=dict(record_type="attempt_expected",run=c["run"],shard=0,expected=expected)
        registry.write_text(json.dumps(row)+"\n")
        with patch.object(cloud,"REGISTRY",registry):
            assert cloud.expected_attempt(c["run"],0)==expected
            assert cloud.expected_attempt("other-run",0) is None
            assert cloud.expected_attempt(c["run"],1) is None
            registry.write_text((json.dumps(row)+"\n")*2)
            assert cloud.expected_attempt(c["run"],0) is None
            registry.write_text(json.dumps(row)+"\n")
            for label,remote,want in cases:
                def storage(*argv,**kw):
                    if argv[:2]==("storage","ls"):return "\n".join("gs://offline/"+n for n in remote)
                    if argv[:2]==("storage","cat"):
                        if argv[2].endswith("config.json"):return json.dumps(c)
                        return remote[argv[2].rsplit("/",1)[-1]]
                    if argv[:3]==("storage","rsync","-r"):
                        dst=Path(argv[-1]);(dst/"config.json").write_text(json.dumps(c))
                        out=dst/"shards/000/out";out.mkdir(parents=True,exist_ok=True)
                        for n,v in remote.items():(out/n).write_bytes(v.encode())
                        return ""
                    raise AssertionError(argv)
                with patch.object(cloud,"gcloud",storage),patch.object(cloud,"list_vms",return_value=[]):
                    state=cloud.shard_state(c["run"],0);assert state.startswith(want),(label,state)
                    capture=io.StringIO()
                    with redirect_stdout(capture):cloud.status(c["run"])
                    assert f"shard 000: {state}" in capture.getvalue()
                    capture=io.StringIO()
                    with patch.object(cloud,"REPO",root),redirect_stdout(capture):cloud.pull(c["run"])
                    assert f"shard 000: {state}" in capture.getvalue()
    print(f"PASS receipt binding: {len(cases)-1} negative pairs rejected by status and pull; matched fixture done")


def cpu_template():
    c=json.loads((cloud.HERE/"configs/associative_continuation_cpu_smoke.json").read_text())
    cloud.validate_release_pins(c)
    valid=copy.deepcopy(c)
    cloud.validate_config(valid)
    contract=cloud.dispatch_contract(valid,100000,"a"*64)
    assert contract["weights_mirror_uri"] is None and contract["weights_manifest_uri"] is None
    # Remove the shipped receipt in memory and refuse before any effect.
    # The independently authorized synthetic CPU create is in cpu_operator_gate.
    contract["shard_sha256"]="b"*64
    with ExitStack() as stack:
        stack.enter_context(patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{}))
        guards=[]
        for obj,name in ((cloud,"gcloud"),(cloud,"now"),(subprocess,"run"),(subprocess,"Popen"),
                         (socket,"create_connection"),(socket,"socket"),(time,"time"),(time,"sleep")):
            guards.append(stack.enter_context(patch.object(obj,name,side_effect=forbidden)))
        rejects(cloud.create_vm,valid,0,f"{cloud.BUCKET}/runs/{valid['run']}/shards/000",contract)
        assert all(guard.call_count==0 for guard in guards), "missing receipt reached an external effect"
    print("PASS CPU template missing evidence refused before all raising external-effect mocks")
    shard=run.assemble_shard(c,0)
    assert shard["kind"]=="associative_continuation_cpu" and shard["cpu_check"]=="receiver"
    assert shard["runtime_control_command"][-2:]==["--check","exact-controls"]
    assert run.KINDS[shard["kind"]]=="experiments.memory.associative_continuation_cpu"
    # Execute the production CPU routing using a finite fake module; no torch/models.
    called=[]
    mod=type("CPU",(),{"_cli":staticmethod(lambda argv:called.append(argv) or 0)})
    shard["dispatch"]={"weights_local_path":None}
    with tempfile.TemporaryDirectory() as td,patch.object(run.importlib,"import_module",return_value=mod), patch.object(run,"cpu_work_progress",return_value=nullcontext()):
        assert run.execute(shard,td)==0
    assert called==[["--config","experiments/memory/associative_continuation_cpu.json","--check","receiver"]]


def spend_bound():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);registry=root/"launches.jsonl"
        row=dict(run="offline-a01",shard=0,vm="offline-vm",created="2026-10-14T00:00:00+00:00",
                 usd_per_hour=2,rate_status="UNVERIFIED",dispatch={"lifetime_seconds":14400})
        registry.write_text(json.dumps(row)+"\n")
        def effect(*argv,**kw):
            assert argv[:3]==("storage","ls","-l")
            return ""
        with patch.object(cloud,"HERE",root),patch.object(cloud,"REGISTRY",registry), \
             patch.object(cloud,"list_vms",return_value=[]),patch.object(cloud,"gcloud",effect), \
             patch.object(cloud,"now",return_value="offline-clock"):
            cloud.spend()
            receipt=(root/"SPEND.md").read_text()
            assert "| offline-a01 | 1 | 8.00 | UNVERIFIED |" in receipt
            row.pop("dispatch");registry.write_text(json.dumps(row)+"\n")
            rejects(cloud.spend)
    print("PASS disappeared VM charges full four-hour bound; unbounded legacy = UNKNOWN-HIGH")

def real_release_pins():
    lock=(cloud.HERE/"lsr_freeze_lock_20261014a.txt").read_bytes()
    manifest=(cloud.HERE/"lsr_gemma4_31b_manifest_sha256.txt").read_bytes()
    assert hashlib.sha256(lock).hexdigest()==cloud.RELEASE_PINS["requirements_lock_sha256"]
    assert hashlib.sha256(manifest).hexdigest()==cloud.RELEASE_MANIFEST_SHA256
    # Exercise each real sibling entry's asset pins independently of its still
    # missing operation/allocation fields. The exact existing lock/image bytes
    # come from S0; they are not written into the prototype templates here.
    base=json.loads((cloud.HERE/"configs/s0_gemma4_memory_port_v1.json").read_text())
    for tag in ("e2b", "e4b"):
        sibling=json.loads((cloud.HERE/f"configs/allin_{tag}_a01.json").read_text())
        candidate=copy.deepcopy(base)
        candidate["common"].update({k:sibling["common"][k] for k in
                                   ("model", "revision", "tokenizer_revision")})
        for key in ("weights_mirror_uri", "weights_manifest_uri", "weights_manifest_sha256"):
            candidate[key]=sibling[key]
        cloud.validate_release_pins(candidate)
        bad=copy.deepcopy(candidate);bad["weights_manifest_sha256"]="f"*64
        rejects(cloud.validate_release_pins,bad)
        bad=copy.deepcopy(candidate);bad["common"]["tokenizer_revision"]="f"*40
        rejects(cloud.validate_release_pins,bad)
        rejects(cloud.validate_config,sibling)
    print("PASS E2B/E4B operator asset pins; wrong manifests/tokenizer revisions rejected; incomplete prototype templates rejected")
    negatives=0
    for name in ("s0_gemma4_memory_port_v1.json","e1_gemma4_smoke.json",
                 "e1_gemma4_alignment_hook_v1.json","associative_continuation_cpu_smoke.json"):
        cfg=json.loads((cloud.HERE/"configs"/name).read_text())
        cloud.validate_release_pins(cfg)
        assert ("\n".join(cfg["requirements"])+"\n").encode()==lock
        fields={**{k:"wrong-valid-value" for k in ("image_name","image_project")},
                "image_id":"1805511023404495092","requirements_lock_sha256":"f"*64}
        if cfg["common"].get("model"):
            fields.update(weights_manifest_sha256="f"*64,weights_mirror_uri=cloud.RELEASE_MIRROR+"wrong",
                          weights_manifest_uri=cloud.RELEASE_MIRROR+"/wrong.txt")
        for field,bad in fields.items():
            for missing in (False,True):
                changed=copy.deepcopy(cfg)
                if missing:changed.pop(field)
                else:changed[field]=bad
                rejects(cloud.validate_release_pins,changed);negatives+=1
        if cfg["common"].get("model"):
            for field in ("revision","tokenizer_revision"):
                for missing in (False,True):
                    changed=copy.deepcopy(cfg)
                    if missing:changed["common"].pop(field)
                    else:changed["common"][field]="f"*40
                    rejects(cloud.validate_release_pins,changed);negatives+=1
        changed=copy.deepcopy(cfg);changed["requirements"]=list(reversed(cfg["requirements"]))
        rejects(cloud.validate_release_pins,changed)
        # Attempt the real launch_plan/dry path under the global raising guards.
        # Missing operation controls are not invented to manufacture an assembly PASS.
        try:
            idx,estimate=cloud.launch_plan(cfg)
        except ValueError as exc:
            assert str(exc) in ("nonempty runtime_control_command argv required", "31B autograd placement/headroom UNVERIFIED", "explicit provisioning_model SPOT or STANDARD required", "unreviewed machine/provisioning_model rate"), str(exc)
            print(f"BLOCKED dry assembly {name}: {exc}; release pins PASS")
        else:
            for i in idx:
                contract=cloud.dispatch_contract(cfg,100000,"a"*64,i)
                # Release VM creation must refuse absent/malformed independent hash
                # before reaching any of the globally raising cloud effects.
                prefix=f"{cloud.BUCKET}/runs/{cfg['run']}/shards/{i:03d}"
                rejects(cloud.create_vm,cfg,i,prefix,contract)
                rejects(cloud.create_vm,cfg,i,prefix,{**contract,"shard_sha256":"wrong"})
                shard=run.assemble_shard(cfg,i)
                shard.update(requirements=cfg["requirements"],dispatch=cloud.dispatch_contract(cfg,100000,"a"*64,i))
                raw=(json.dumps(shard,sort_keys=True,separators=(",",":"))+"\n").encode()
                expected=cloud.attempt_manifest(shard,raw)
                assert expected["shard_sha256"]==hashlib.sha256(raw).hexdigest()
                assert shard["dispatch"]["weights_manifest_sha256"] is None
            print(f"PASS dry assembly {name}: {len(idx)} shard; estimate {estimate}; no upload/cloud")
    print(f"PASS real release pins: four configs; {negatives} wrong/missing pin cases rejected; exact lock bytes")


def lifetime_variants():
    from . import s0_runtime as rt
    for hours in (1,3,4):
        total, work = hours*3600, hours*3600-900
        c=fixture(); c.update(lifetime_seconds=total, work_stop_seconds=work,
                             cost_cap_usd=13.1986*hours+2, aggregate_cost_ceiling_usd=13.1986*hours+2,
                             ancillary_allowance_usd=2)
        c["common"]["job_stop_seconds"]=work
        assert abs(cloud.launch_plan(c)[1] - (13.1986*hours+2)) < 1e-8
        fake=creation(c)
        assert dt.datetime.fromisoformat(next(x.split("=",1)[1] for x in fake.creation if x.startswith("--termination-time="))).timestamp()==100000+total
        bad=copy.deepcopy(c);bad["cost_cap_usd"]=13.1986*hours+1;rejects(cloud.validate_config,bad)
        bad=copy.deepcopy(c);bad["aggregate_cost_ceiling_usd"]=13.1986*hours+1;rejects(cloud.launch_plan,bad)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,args,installed=rt.deployment_fixture(root)
            cfg["dispatch"].update(lifetime_seconds=total,work_stop_seconds=work,shutdown_reserve_seconds=900,
                                   work_stop_at=1000+work,shutdown_at=1000+total)
            rt.save_fixture(cfg,args)
            runtime=rt.Runtime(root,1000,{},clock=lambda:1001,heartbeat=lambda:1001)
            with patch.object(rt.metadata,"distributions",return_value=installed):
                rt.verify_deployment(cfg,Path(args.shard).read_bytes(),args,runtime)
            assert (runtime.lifetime,runtime.work_stop,runtime.reserve)==(total,work,900)
            runtime.clock=lambda:1000+work
            try:runtime.guard("work")
            except rt.Stop as exc:assert str(exc)=="deadline"
            else:raise AssertionError("late work admitted")
            expected=dict(run="offline",shard=0,attempt="offline-0",created_epoch=100000,bundle_sha256="a"*64,shard_sha256="b"*64)
            (root/"runtime_receipt.json").write_text('{}')
            shell=json.loads(exact_shutdown(root,expected,0,"success",total,work))
            assert shell["clock"]["work_stop_epoch"]==100000+work
            assert shell["clock"]["lifetime_end_epoch"]==100000+total
        guard=cloud.CREDIT_GUARD_EPOCH
        d=cloud.dispatch_contract(c,guard-total,"a"*64)
        assert d["shutdown_at"]==guard
        rejects(cloud.dispatch_contract,c,guard-total+1,"a"*64)
        print(f"PASS {hours}h: creation/platform/runtime/work/shutdown; compute={13.1986*hours:.4f}; ancillary=2; credit guard")
    for field,value in (("lifetime_seconds",True),("work_stop_seconds",0),("shutdown_reserve_seconds",899)):
        c=fixture();c[field]=value;rejects(cloud.validate_config,c)


def placement_evidence():
    """Synthetic gate fixtures only; never register fabricated measurements."""
    c=json.loads((cloud.HERE/"configs/allin_31b_a01.json").read_text())
    c["release"]=True;c["ancillary_allowance_usd"]=0
    s=c["common"];s["runtime_control_command"]=["offline-control"]
    with tempfile.TemporaryDirectory() as td, patch.object(cloud,"REPO",Path(td)):
        root=Path(td);runner=root/"experiments/memory/prototype/runner.py"
        runner.parent.mkdir(parents=True);runner.write_bytes(b'# synthetic runner fixture\n')
        config=root/"config.json";config.write_bytes(b'{}')
        s["config"]=str(config);s["config_sha256"]=hashlib.sha256(b'{}').hexdigest()
        p=dict(status="VERIFIED",autograd_tested=True,device_map={"":"cuda:0","layer":"cuda:1"},
               max_memory={"0":70000000000,"1":70000000000},allow_offload=False,
               min_device_headroom_fraction=0.1,evidence_sha256="a"*64)
        s["placement_contract"]=p
        binding=dict(model=s["model"],revision=s["revision"],tokenizer_revision=s["tokenizer_revision"],
            machine=c["machine"],guest_accelerators=c["guest_accelerators"],kind=c["kind"],phase=s["phase"],
            config_sha256=s["config_sha256"],runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
            requirements_lock_sha256=c["requirements_lock_sha256"],weights_manifest_sha256=c["weights_manifest_sha256"],
            device_map=p["device_map"],max_memory=p["max_memory"],allow_offload=False,
            effective_workload_sha256=cloud.effective_workload_sha256(s),min_device_headroom_fraction=0.1)
        e=dict(status="PASS",binding=binding,headroom_fraction={"0":0.2,"1":0.15},
               procedure=dict(load=True,backward=True,decode=True,command=["offline-fixture"],input_sha256="b"*64))
        def authorize(evidence):
            raw=json.dumps(evidence,sort_keys=True).encode();h=hashlib.sha256(raw).hexdigest()
            s["placement_contract"]["evidence_sha256"]=h
            return {h:raw}
        with patch.object(cloud,"RELEASE_PLACEMENT_EVIDENCE",authorize(e)):
            cloud.validate_config(c);cloud.launch_plan(c)
            for key,value in (("status","UNVERIFIED"),("autograd_tested",False),("device_map",None),
                ("device_map","auto"),("device_map",{"":"cpu"}),("max_memory",None),
                ("allow_offload",True),("evidence_sha256",None),("evidence_sha256","f"*64),
                ("min_device_headroom_fraction",0.01)):
                bad=copy.deepcopy(c);bad["common"]["placement_contract"][key]=value
                rejects(cloud.validate_config,bad)
            for release in (False,True):
                bad=copy.deepcopy(c);bad["release"]=release
                bad["common"]["placement_contract"].update(status="UNVERIFIED",autograd_tested=True)
                rejects(cloud.validate_config,bad)
            config.write_bytes(b'{"altered":1}');rejects(cloud.validate_config,c);config.write_bytes(b'{}')
            runner.write_bytes(b'# changed\n');rejects(cloud.validate_config,c);runner.write_bytes(b'# synthetic runner fixture\n')
        for key in binding:
            bad=copy.deepcopy(e);bad["binding"][key]=None
            with patch.object(cloud,"RELEASE_PLACEMENT_EVIDENCE",authorize(bad)):
                rejects(cloud.validate_config,c)
        for key,value in (("status","FAIL"),("headroom_fraction",{"0":0.01,"1":0.2}),
                          ("procedure",dict(load=True,backward=False,decode=True))):
            bad=copy.deepcopy(e);bad[key]=value
            with patch.object(cloud,"RELEASE_PLACEMENT_EVIDENCE",authorize(bad)):
                rejects(cloud.validate_config,c)
        with patch.object(cloud,"RELEASE_PLACEMENT_EVIDENCE",{}):
            rejects(cloud.validate_config,c)
        # Recording API checks transport, not actual loader behavior.
        shard=run.assemble_shard(c,0);shard["dispatch"]=cloud.dispatch_contract(c,100000,"a"*64)
        calls=[]
        import types
        with patch.object(run.Path,"is_dir",return_value=True),patch.object(run.Path,"is_file",return_value=True), \
             patch.object(run.importlib,"import_module",return_value=types.SimpleNamespace(run_dispatch=lambda **kw:calls.append(kw) or 0)):
            assert run.execute(shard,td)==0
            assert calls[0]["effective_config"]["model_local"]=="/opt/lsr/weights"
            assert calls[0]["effective_config"]["placement_contract"]==p
            assert calls[0]["config_bytes"]==b'{}'
            for key,value in (("model_local","/tmp/untrusted"),("placement_contract",{"device_map":"auto"}),
                              ("model","another/model"),("revision","f"*40)):
                raw=json.dumps({key:value}).encode();config.write_bytes(raw)
                bad=copy.deepcopy(shard);bad["config_sha256"]=hashlib.sha256(raw).hexdigest()
                rejects(run.execute,bad,td);assert len(calls)==1
    print("PASS placement synthetic authority; contradictory/provenance/headroom/config/runner negatives; effective GPU API transport")


def s0_placement():
    """Import S0: real main validation -> real receiver placement, no loading."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        executable = os.environ.get("LSR_CPU_CHECK_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("S0 import check requires installed CPU-capable torch")
        with patch.object(subprocess, "Popen", _REAL_POPEN), patch.object(time, "sleep", _REAL_SLEEP):
            result = _REAL_RUN([executable, "-B", "-m", __package__ + ".dispatch_checks",
                               "--check", "s0-placement", "--offline"], cwd=str(cloud.REPO), timeout=60)
        assert result.returncode == 0
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from . import s0_memory_port as s0, gemma4_receiver as receiver
    import torch
    c=json.loads((cloud.HERE/"configs/s0_gemma4_memory_port_v1.json").read_text())
    s=c["common"];s.pop("missing_operator_inputs")
    s.update(attn_implementation="eager", smoke_cases=[{"fixture":"smoke"}],
             fit_cases=[{"fixture":i} for i in range(4)])
    for field in ("smoke_cases", "fit_cases"):
        s[field+"_sha256"]=hashlib.sha256(json.dumps(s[field],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    s["placement"]={"device_map":{"":"cuda:0"},"measurement_id":"synthetic-offline-only","min_free_fraction":0.1}
    p=dict(status="VERIFIED",autograd_tested=True,device_map={"":"cuda:0"},
           max_memory={"0":70000000000,"1":70000000000},allow_offload=False,
           min_device_headroom_fraction=0.1)
    s["placement_contract"]=p
    with tempfile.TemporaryDirectory() as td:
        config=Path(td)/"recipe.json";config.write_bytes(b'{}')
        s["config"]=str(config);s["config_sha256"]=hashlib.sha256(b'{}').hexdigest()
        binding=dict(model=s["model"],revision=s["revision"],tokenizer_revision=s["tokenizer_revision"],
            machine=c["machine"],guest_accelerators=c["guest_accelerators"],kind=c["kind"],phase=s["phase"],
            config_sha256=s["config_sha256"],runner_sha256=hashlib.sha256(Path(s0.__file__).read_bytes()).hexdigest(),
            requirements_lock_sha256=c["requirements_lock_sha256"],weights_manifest_sha256=c["weights_manifest_sha256"],
            device_map=p["device_map"],max_memory=p["max_memory"],allow_offload=False,
            effective_workload_sha256=cloud.effective_workload_sha256(s),min_device_headroom_fraction=0.1)
        e=dict(status="PASS",measurement_id=s["placement"]["measurement_id"],binding=binding,
               headroom_fraction={"0":0.2,"1":0.2},procedure=dict(load=True,backward=True,decode=True,
                   command=["synthetic-offline-only"],input_sha256="b"*64))
        raw=json.dumps(e,sort_keys=True).encode();h=hashlib.sha256(raw).hexdigest();p["evidence_sha256"]=h
        class AtLoad(RuntimeError): pass
        seen=[]
        def boundary(cfg):
            seen.append(receiver._placement(cfg))
            raise AtLoad("real S0 validation and real placement accepted; pretrained load forbidden")
        class Output:
            def write(self, value): forbidden()
        with patch.object(cloud,"RELEASE_PLACEMENT_EVIDENCE",{h:raw}), \
             patch.object(receiver,"load_receiver",side_effect=boundary), \
             patch.object(receiver,"_load_original",side_effect=forbidden), \
             patch.object(torch.cuda,"_lazy_init",side_effect=forbidden):
            cloud.validate_config(c)
            shard=run.assemble_shard(c,0)
            assert cloud.effective_workload_sha256(shard)==binding["effective_workload_sha256"]
            try:s0.main(shard,Output())
            except AtLoad:pass
            else:raise AssertionError("S0 failed to reach placement boundary")
            assert seen==[s["placement"]]
            for key,value in (("attn_implementation","sdpa"),("dtype","float32"),
                              ("smoke_cases",[{"changed":True}]),("fit_cases",[{"changed":True}]*4),
                              ("memory_positions",17),("placement",dict(s["placement"],measurement_id="changed")),
                              ("placement",dict(s["placement"],min_free_fraction=0.2))):
                bad=copy.deepcopy(c);bad["common"][key]=value
                rejects(cloud.validate_config,bad)
            for mapping in ({"":0},{"":"cuda:1"},{"":"cpu"},{"":"disk"},"auto"):
                bad=copy.deepcopy(c);bad["common"]["placement"]["device_map"]=mapping
                bad["common"]["placement_contract"]["device_map"]=mapping
                rejects(cloud.validate_config,bad)
            bad=copy.deepcopy(c);bad["common"]["placement_contract"]["allow_offload"]=True
            rejects(cloud.validate_config,bad)
        with patch.object(cloud,"RELEASE_PLACEMENT_EVIDENCE",{}):
            rejects(cloud.validate_config,c)
    print("PASS S0 imported main config validation -> real receiver _placement; canonical cuda:N; "
          "changed device/backend/workload/measurement/headroom/offload rejected; evidence-free autograd rejected; no weights/CUDA")


def prototype_cpu_config():
    import types
    c=fixture();c.update(kind="latent_memory_prototype",machine="e2-standard-8",has_gpu=False,guest_accelerators=[],
        weights_mirror_uri=None,weights_manifest_uri=None,weights_manifest_sha256=None,weights_local_path=None,
        common={"model":None,"phase":"prepare","config":"experiments/memory/prototype/pinned.json","config_sha256":hashlib.sha256(b'{}').hexdigest(),"runtime_control_command":["offline-control"]})
    calls=[]
    def load(name):
        assert name=="experiments.memory.prototype.runner", name
        return types.SimpleNamespace(run_dispatch=lambda **kw:calls.append(kw) or 0)
    with tempfile.TemporaryDirectory() as td, patch.object(run.importlib,"import_module",load):
        path=Path(td)/"config.json";path.write_bytes(b'{}');c["common"]["config"]=str(path)
        cloud.validate_config(c)
        shard=run.assemble_shard(c,0);shard["dispatch"]=cloud.dispatch_contract(c,100000,"a"*64)
        assert run.execute(shard,td)==0
        assert calls==[dict(effective_config=shard,config_bytes=b'{}',out_dir=str(Path(td)/"prepare"))]
        path.write_bytes(b'{"changed":true}')
        rejects(run.execute,shard,td);assert len(calls)==1
        path.write_bytes(b'{}')
        with patch.object(run.importlib,"import_module",return_value=types.SimpleNamespace(main=lambda _: forbidden())):
            rejects(run.execute,shard,td)
    for phase in ("unknown","train"):
        bad=copy.deepcopy(c);bad["common"]["phase"]=phase;rejects(run.assemble_shard,bad,0)
    bad=copy.deepcopy(c);bad["shards"][0]["kind"]="smoke";rejects(run.assemble_shard,bad,0)
    bad=copy.deepcopy(c);bad["common"]["config_sha256"]=None;rejects(run.assemble_shard,bad,0)
    for name in ("e2b","e4b","31b","cpu"):
        cfg=json.loads((cloud.HERE/f"configs/allin_{name}_a01.json").read_text())
        assert cfg["release"] is False
        rejects(cloud.validate_config,cfg)
    print("PASS prototype CPU exact config bytes/prepare API, tampered config and legacy API rejected, model-free assets, offline environment; no common.load/CUDA/network")



def provisioning_models():
    """Real six-entry rate policy, isolated from legacy synthetic regression rates."""
    base=json.loads((cloud.HERE/"configs/associative_continuation_cpu_smoke_a06.json").read_text())
    with patch.object(cloud,"RATE_TABLE",PRODUCTION_RATE_TABLE):
        assert cloud.launch_plan(base)[0]==[0] and abs(cloud.launch_plan(base)[1]-1.026944)<1e-12
        assert base["operator_evidence_sha256"] is None
        rejects(cloud.verify_cpu_operator_evidence,base)
        for (machine,model),table in PRODUCTION_RATE_TABLE.items():
            c=copy.deepcopy(base)
            c.update(machine=machine,provisioning_model=model,usd_per_hour=table["usd_per_hour"],
                     cost_cap_usd=50,aggregate_cost_ceiling_usd=50)
            gpu=machine.startswith("a3-")
            if gpu:
                c.update(kind="s0_memory_port",release=False,has_gpu=True,
                         guest_accelerators=[{"type":"nvidia-h100-80gb","count":1}],
                         common=copy.deepcopy(fixture()["common"]))
                c["common"]["job_stop_seconds"]=2700
                for k in ("weights_manifest_uri","weights_manifest_sha256","weights_mirror_uri","weights_local_path"):
                    c[k]=fixture()[k]
            cloud.validate_config(c)
            bad=copy.deepcopy(c);bad["usd_per_hour"]+=0.01;rejects(cloud.validate_config,bad)
            expected=cloud.rate_entry(c)
            assert expected["derived"] is table["derived"]
            if gpu:
                bad=copy.deepcopy(c);bad["kind"]="associative_continuation_cpu"
                rejects(cloud.verify_cpu_operator_evidence,bad)
                c["rate_verified"]=False
                fake=creation(c)
            else:
                c.update(rate_verified=True,shape_verified=True)
                obligations=cloud.CPU_PRELAUNCH_CHECKS
                receipt=dict(packet_sha256=cloud.cpu_packet_sha256(c),source_sha256=cloud.cpu_source_sha256(),
                    credit_rate=expected,checks={k:"PASS" for k in obligations},
                    artifacts={k:[dict(path="synthetic-offline-only",sha256="a"*64)] for k in obligations-{"credit_rate"}})
                def registered(r,packet):
                    raw=json.dumps(r,sort_keys=True).encode();h=hashlib.sha256(raw).hexdigest()
                    packet["operator_evidence_sha256"]=h
                    return patch.object(cloud,"RELEASE_CPU_OPERATOR_EVIDENCE",{h:raw})
                for key,value in (("usd_per_hour",0),("provisioning_model","WRONG"),("derived",not expected["derived"])):
                    wrong=copy.deepcopy(receipt);wrong["credit_rate"][key]=value
                    with registered(wrong,c): rejects(cloud.verify_cpu_operator_evidence,c)
                with registered(receipt,c):
                    cloud.verify_cpu_operator_evidence(c)
                    contract=cloud.dispatch_contract(c,100000,hashlib.sha256(cloud.build_bundle(c)).hexdigest())
                    contract["shard_sha256"]="b"*64
                    fake=FakeCloud(c,contract)
                    with tempfile.TemporaryDirectory() as td,patch.object(cloud,"gcloud",fake),patch.object(time,"time",return_value=100060),patch.object(cloud,"REGISTRY",Path(td)/"journal"):
                        prefix=f"{cloud.BUCKET}/runs/{c['run']}/shards/000"
                        cloud.create_vm(c,0,prefix,contract)
                        fake.mutate=lambda a:a["scheduling"].update(provisioningModel="STANDARD" if model=="SPOT" else "SPOT")
                        rejects(cloud.create_vm,c,0,prefix,contract)
                        assert fake.calls[-1][:3]==("compute","instances","delete")
            assert f"--provisioning-model={model}" in fake.creation
            assert "--preemptible" not in fake.creation
            assert sum(x[:3]==("compute","instances","create") for x in fake.calls) <= 2
            if gpu:
                creation(c,lambda a:a["scheduling"].update(provisioningModel="STANDARD" if model=="SPOT" else "SPOT"))
            if model=="STANDARD":
                wrong=tuple("--provisioning-model=SPOT" if a.startswith("--provisioning-model=") else a for a in fake.creation)
                try: FakeCloud(c,{}).assert_create(wrong)
                except AssertionError: pass
                else: raise AssertionError("STANDARD request with SPOT flag accepted")
                try: FakeCloud(c,{}).assert_create((*fake.creation,"--preemptible=true"))
                except AssertionError: pass
                else: raise AssertionError("STANDARD request with Spot-only preemptible accepted")
            print(f"PASS {machine}/{model}: exact rate {table['usd_per_hour']}, derived={table['derived']}; mocked argv/readback/delete")
        for value in (None,"spot","UNKNOWN",True,[]):
            c=copy.deepcopy(base);c["provisioning_model"]=value;rejects(cloud.validate_config,c)
        c=copy.deepcopy(base);c.pop("provisioning_model");rejects(cloud.validate_config,c)
        c=copy.deepcopy(base);c["machine"]="n2-standard-32";rejects(cloud.validate_config,c)
        c=copy.deepcopy(base);c["zones"]=["europe-west1-b"];rejects(cloud.validate_config,c)
        print("PASS required provisioning model, unknown pair/region rejected; a06 full-hour estimate $1.026944, provisional ancillary $0.25")

CHECKS={"provisioning-models":provisioning_models,"bundle-readback-negative":bundle_readback_negative,"cpu-progress":cpu_progress,"cpu-bundle":cpu_bundle,"cpu-operator-gate":cpu_operator_gate,"terminal-reconciliation":terminal_reconciliation,"placement-evidence":placement_evidence,"lifetime-variants":lifetime_variants,"prototype-cpu-config":prototype_cpu_config,"real-release-pins":real_release_pins,"options":options,"create-verify":create_verify,"aggregate-cost":aggregate_cost,
        "entry-handling":entry_handling,"deadlines":deadlines,"collisions":collisions,
        "bundle":bundle,"pins-mirror":pins_mirror,"s0-config":s0_config,"s0-identity":s0_identity,
        "cpu-dry-assembly":cpu_dry_assembly,"cpu-support":cpu_support, "request-mutations":request_mutations,
        "image-accelerators":image_accelerators,"control-pins":control_pins,
        "receipt-integration":receipt_integration,"receipt-binding":receipt_binding,"cpu-template":cpu_template,"spend-bound":spend_bound}


def main(argv=None):
    CHECKS["s0-placement"] = s0_placement
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check",required=True,choices=["all",*CHECKS])
    ap.add_argument("--offline",required=True,action="store_true")
    args=ap.parse_args(argv)
    if os.name == "nt" and args.check not in {"s0-placement", "cpu-bundle", "cpu-progress"}:
        repo = cloud.REPO.as_posix()
        posix = "/mnt/" + repo[0].lower() + repo[2:]
        python_posix = "/mnt/" + sys.executable[0].lower() + sys.executable[2:].replace("\\", "/")
        return _REAL_RUN(["wsl", "-d", "Ubuntu", "--cd", posix, "env", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2",
                          "LSR_CPU_CHECK_PYTHON=" + python_posix,
                          "PYTHONDONTWRITEBYTECODE=1", "python3", "-B", "-m", __package__ + ".dispatch_checks",
                          "--check", args.check, "--offline"]).returncode
    with ExitStack() as stack:
        stack.enter_context(patch.dict(cloud.RATE_TABLE, {
            ("a3-highgpu-2g", "SPOT"): {"usd_per_hour": 13.1986, "derived": False},
            ("n1-standard-8", "SPOT"): {"usd_per_hour": 13.1986, "derived": False},
            ("e2-standard-8", "SPOT"): {"usd_per_hour": 13.1986, "derived": False},
        }))  # Offline-only synthetic entries, never installed in production.
        for obj,name in ((cloud,"gcloud"),(cloud,"now"),(subprocess,"run"),(subprocess,"Popen"),
                         (socket,"create_connection"),(socket,"socket"),(time,"time"),(time,"sleep")):
            stack.enter_context(patch.object(obj,name,forbidden))
        for name in CHECKS if args.check=="all" else [args.check]:
            try:
                CHECKS[name]()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
                return 1
            print(f"PASS {name}: positive path and rejecting negative cases")
    print("PASS dispatch-checks: offline mocks only; no launchability or cloud execution claim")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
