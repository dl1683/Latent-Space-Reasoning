"""Canonical finite CUDA attention experiment; RUN_DIR PHASE, external supervision required."""
import os
if os.environ.get('LSR_ATTENTION_SUPERVISED') != '1':
    raise SystemExit('external resource_guard required')
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(sys.argv[1]).resolve()
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
phase = sys.argv[2]
assert phase in ('tiny', 'smoke', 'smoke_swap', 'panel')
pins = json.loads((ROOT/'pins.json').read_text())
pin_digest=hashlib.sha256((ROOT/'pins.json').read_bytes()).hexdigest()
execution_id=os.environ['LSR_EXECUTION_ID']
assert os.environ['LSR_PIN_DIGEST']==pin_digest
cfg = json.loads((ROOT/'config.json').read_text())
thermal=cfg.get('thermal',dict(stop=72,cool=60,burst_seconds=35,cool_seconds=30))
assert all(type(thermal[k]) is int for k in ('stop','cool','burst_seconds','cool_seconds'))
assert 0<thermal['cool']<thermal['stop']<=82 and thermal['cool']<=76
assert 0<thermal['burst_seconds']<=35 and thermal['cool_seconds']==30
assert cfg.get('helper_job') and cfg.get('helper_resume'), 'missing required pin coverage: helper paths must be explicit'
required=[Path(__file__),REPO/'experiments/memory/attention_completeness.py',
    REPO/'experiments/memory/prior_art_gpu.py',ROOT/'config.json',ROOT/'resource_guard.py',
    ROOT/'evaluate.py',ROOT/'bounded_monitor.py',ROOT/'PREREG.md',
    ROOT/'data_v3/reader.json',ROOT/'data_v3/labels.json',
    Path(cfg['helper_job']),
    Path(cfg['helper_resume'])]
for model_root in [Path(cfg['dense'])]:
    required.extend(p for p in model_root.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model'))
assert required and all(str(p.resolve()) in pins for p in required),'missing required pin coverage'
required.extend(Path(x)for x in cfg['reader_row_paths'])
for directory in set(str(Path(x).parent)for x in cfg['reader_row_paths']):
    required.extend(Path(directory)/name for name in ['config.json','pins.json','pinned_worker_preimage.py','panel_resources.json','data_v3/reader.json'])
assert all(str(p.resolve())in pins for p in required),'resume pin coverage'
for filename, expected in pins.items():
    digest = hashlib.sha256()
    with Path(filename).open('rb') as source:
        for piece in iter(lambda: source.read(1024*1024), b''):
            digest.update(piece)
    assert digest.hexdigest() == expected, filename

import torch
import psutil

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
def activity(status):
    row=dict(status=status,since=time.monotonic(),execution_id=execution_id,phase=phase)
    target=ROOT/(phase+'_activity.json');temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(row));temporary.replace(target)
    with (ROOT/(phase+'_activity.jsonl')).open('a') as log:log.write(json.dumps(row)+'\n')

activity('work')
torch.cuda.set_per_process_memory_fraction(.75)
burst_started=time.monotonic()

def temperature():
    until=time.monotonic()+30
    paused=False
    while True:
        path=Path(cfg.get('thermal_handoff_root',str(ROOT)))/(phase+'_external_thermal.jsonl')
        with path.open('rb') as source:
            source.seek(0,2);size=source.tell();source.seek(max(0,size-8192))
            complete=source.read().split(b'\n')[:-1]
        assert complete,'no complete thermal sample'
        row=json.loads(complete[-1])
        assert row['phase']==phase and row['execution_id']==execution_id,'thermal identity mismatch'
        assert row['uuid']==os.environ['LSR_GPU_UUID'],'thermal GPU mismatch'
        if time.monotonic()-row['monotonic']+row['vendor_age']<=2.5:
            if paused:activity('work')
            return row['temperature']
        torch.cuda.synchronize()
        if not paused:activity('cooling');paused=True
        assert time.monotonic()<until,'stale vendor thermal sample after bounded idle'
        time.sleep(.05)

def check():
    assert psutil.virtual_memory().available >= 20*2**30, 'RAM floor'
    assert temperature()<thermal['stop'], 'thermal stop'
    # Growing DynamicCache concatenations can strand unused allocator blocks.
    # Return those blocks before testing actual physical use; live KV is retained.
    torch.cuda.empty_cache()
    free,total=torch.cuda.mem_get_info()
    assert total-free<=19.5*2**30, f'CUDA physical VRAM stop: used={total-free}, allocated={torch.cuda.memory_allocated()}, reserved={torch.cuda.memory_reserved()}'

def cool():
    global burst_started
    start = time.monotonic()
    if start-burst_started>=thermal['burst_seconds'] or temperature()>=min(thermal['stop']-2,thermal['cool']+6):
        torch.cuda.synchronize()
        activity('cooling')
        since=None
        while True:
            check()
            assert time.monotonic()-start<150,'mandatory burst cooldown timeout'
            if temperature()<=thermal['cool']:
                since=since or time.monotonic()
                if time.monotonic()-since>=thermal['cool_seconds']:break
            else:since=None
            time.sleep(.5)
        burst_started=time.monotonic()
        activity('work')
    check()

def save(name, value):
    with (ROOT/name).open('x', encoding='utf8') as output:
        json.dump(dict(value,phase=phase,execution_id=execution_id,pins_sha256=pin_digest), output, allow_nan=False)

if phase == 'tiny':
    check()
    assert torch.cuda.mem_get_info()[0]>=20*2**30,'initial CUDA free VRAM reserve'
    x = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8,8)
    answer = x @ torch.eye(8,device='cuda')
    torch.cuda.synchronize()
    assert torch.equal(x,answer)
    save('tiny_operation.json', dict(passed=True, torch=torch.__version__,
        cuda=torch.version.cuda, architecture=torch.cuda.get_device_capability(),
        device=torch.cuda.get_device_name(), peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_free_total=list(torch.cuda.mem_get_info())))
    raise SystemExit(0)

from transformers import AutoTokenizer,AutoModel
from safetensors import safe_open
import numpy as np

data=json.loads((ROOT/'data_v3/reader.json').read_text())
origin=Path(cfg['resume_from'])
manifest=json.loads((origin/'prefix_checkpoint/manifest.json').read_text())
smoke=json.loads((origin/'smoke_measurements.json').read_text())
rows=[]
for index,filename in enumerate(cfg['reader_row_paths']):
    path=Path(filename);producer=path.parent;row=json.loads(path.read_text())
    source_pins=json.loads((producer/'pins.json').read_text())
    source_resource=json.loads((producer/'panel_resources.json').read_text())
    source_config=json.loads((producer/'config.json').read_text())
    assert source_resource['cleanup'] and row['execution_id']==source_resource['execution_id']
    assert row['pins_sha256']==hashlib.sha256((producer/'pins.json').read_bytes()).hexdigest()
    assert row['id']==data['questions'][index]['id']
    assert (producer/'data_v3/reader.json').read_bytes()==(ROOT/'data_v3/reader.json').read_bytes()
    assert source_config['reader']==cfg['reader'] and source_config['dense']==cfg['dense']
    canonical=str((REPO/'experiments/memory/attention_completeness_run.py').resolve())
    assert hashlib.sha256((producer/'pinned_worker_preimage.py').read_bytes()).hexdigest()==source_pins[canonical]
    row['producer_execution_id']=row.pop('execution_id')
    row['producer_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    row.pop('phase');row.pop('pins_sha256');rows.append(row)
assert len(rows)==60
count=0
for path in Path(cfg['dense']).glob('*.safetensors'):
    with safe_open(path,framework='pt',device='cpu')as f:
        for key in f.keys():count+=__import__('math').prod(f.get_slice(key).get_shape())
assert 0<count<4_000_000_000
check();assert torch.cuda.mem_get_info()[0]>=20*2**30
enc_started=time.monotonic()
tokenizer=AutoTokenizer.from_pretrained(cfg['dense'],local_files_only=True)
model=AutoModel.from_pretrained(cfg['dense'],local_files_only=True,dtype=torch.float32).cuda().eval()
assert model.config.model_type=='bert' and model.config.hidden_size==384
assert (tokenizer.cls_token_id,tokenizer.sep_token_id)==(101,102)
def embed(ids):
    cool();assert len(ids)<=254
    tokens=torch.tensor([[tokenizer.cls_token_id]+ids+[tokenizer.sep_token_id]],device='cuda')
    with torch.inference_mode():
        h=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),token_type_ids=torch.zeros_like(tokens)).last_hidden_state
        vector=torch.nn.functional.normalize(h.float().mean(1),dim=-1)
    assert vector.shape==(1,384) and torch.isfinite(vector).all()
    return vector
vectors=[];owners=[]
for owner,text in enumerate(data['blocks']):
    ids=tokenizer.encode(text,add_special_tokens=False)
    for start in range(0,len(ids),240):
        vectors.append(embed(ids[start:start+254]));owners.append(owner)
        if start+254>=len(ids):break
bank=torch.cat(vectors);dense_rows={};query_vectors=[]
for question in data['questions']:
    ids=tokenizer.encode(question['question'],add_special_tokens=False)
    vector=embed(ids);query_vectors.append(vector.float().cpu().numpy()[0])
    similarity=(vector@bank.T)[0].float().cpu().tolist();scores=[-2.]*60
    for owner,score in zip(owners,similarity):scores[owner]=max(scores[owner],score)
    dense_rows[question['id']]=scores
save('dense.json',dict(scores=dense_rows,windows=len(owners)))
np.savez(ROOT/'dense_vectors.npz',bank=bank.float().cpu().numpy(),owners=np.asarray(owners),queries=np.stack(query_vectors),question_ids=np.asarray([q['id']for q in data['questions']]))
save('panel_measurements.json',dict(rows=rows,corpus_tokens=manifest['length'],precision='bf16 reader / fp32 encoder',reader_parameters=smoke['reader_parameters'],encoder_parameters=count,prefix_payload_bytes=manifest['payload_bytes'],corpus_encoded_once=True,prefill_wall_seconds_including_cooldown=None,source_prefill_seconds=None,resume_from=str(origin),reader_collection='60unique reader questions;62 answer attempts including2 cold replays across3 preserved aborted jobs; this completed allocation encodes dense baseline only',dense_build_query_seconds_including_cooldown=time.monotonic()-enc_started,peak_allocated=torch.cuda.max_memory_allocated(),dense_vectors_sha256=hashlib.sha256((ROOT/'dense_vectors.npz').read_bytes()).hexdigest()))
print(json.dumps(dict(scope='dense-only recovery, reader rows retained',rows=len(rows),windows=len(owners),parameters=count)),flush=True)
