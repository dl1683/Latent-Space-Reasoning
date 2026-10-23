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
for model_root in [Path(cfg['reader']),Path(cfg['dense'])]:
    required.extend(p for p in model_root.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model'))
assert required and all(str(p.resolve()) in pins for p in required),'missing required pin coverage'
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

from transformers import AutoTokenizer, AutoModel, DynamicCache
from experiments.memory.attention_completeness import attention_query
from experiments.memory.prior_art_gpu import stream_model

check()
assert torch.cuda.mem_get_info()[0]>=20*2**30,'initial CUDA free VRAM reserve'
tokenizer = AutoTokenizer.from_pretrained(cfg['reader'], local_files_only=True)
if cfg.get('small_reader'):
    from experiments.memory.attention_completeness import stream_small_model
    model=stream_small_model(cfg['reader'], long_horizon=phase=='panel')
else:
    raise RuntimeError('the historical >=4B reader interface is unsupported and withdrawn')
window=getattr(model.config,'sliding_window',None)
reader_parameters=model._lsr_parameter_count
assert window is None or window>=12000,'sliding window masks declared context'
data = json.loads((ROOT/'data_v3/reader.json').read_text())
if phase.startswith('smoke'):
    values = ['LAVENDER','MARIGOLD','CORAL','VERMILION','TURQUOISE','ULTRAMARINE']
    if phase=='smoke_swap':
        values[1],values[4]='SAFFRON','INDIGO'
    blocks = [f'At SmokeStation{i}, the access code is {values[i]}.' for i in range(6)]
    if phase=='smoke_swap':
        blocks=[blocks[j] for j in (1,0,2,4,3,5)]
    prefix_ids, spans = [], []
    for text in blocks:
        ids = tokenizer.encode(text+'\n', add_special_tokens=False)
        spans.append([len(prefix_ids), len(prefix_ids)+len(ids)])
        prefix_ids.extend(ids)
    questions = [dict(id=f'S{i}', question=f'What is the access code at SmokeStation{i}?')
                 for i in (1,4)]
    questions.append(dict(questions[0], id='S1repeat'))
    if cfg.get('one_smoke'):questions=questions[:1]
else:
    smoke_root=Path(cfg.get('resume_from',str(ROOT)))
    smoke = json.loads((smoke_root/'smoke_verdict.json').read_text())
    assert smoke['passed'] and smoke['pins_sha256'] == hashlib.sha256((smoke_root/'pins.json').read_bytes()).hexdigest()
    for filename,expected in smoke['artifact_sha256'].items():
        assert hashlib.sha256((smoke_root/filename).read_bytes()).hexdigest()==expected,'smoke artifact drift'
    prefix_ids, spans, questions = data['ids'], data['spans'], data['questions']

# Native Phi instruction framing is held fixed across smoke and panel.
# Corpus token IDs are not decoded/reencoded to construct the prompt.
header = tokenizer.encode(
    '<|user|>\nAnswer the question using the corpus below. Give only the answer; '
    'if it is missing, say NOT IN MEMORY.\nCorpus:\n', add_special_tokens=False)
prefix_ids = header + prefix_ids
spans = [[a+len(header), b+len(header)] for a,b in spans]

resume_rows=[]
restore_seconds=0
if phase=='panel' and cfg.get('resume_from'):
    origin=Path(cfg['resume_from'])
    manifest=json.loads((origin/'prefix_checkpoint/manifest.json').read_text())
    original_pins=json.loads((origin/'pins.json').read_text())
    original_cfg=json.loads((origin/'config.json').read_text())
    assert all(cfg[k]==original_cfg[k] for k in ('reader','dense','small_reader','prefill','allocator_fraction'))
    assert manifest['pins_sha256']==hashlib.sha256((origin/'pins.json').read_bytes()).hexdigest()
    assert manifest['prefix_ids_sha256']==hashlib.sha256(json.dumps(prefix_ids).encode()).hexdigest()
    assert manifest['length']==len(prefix_ids)
    for name in ['reader.json','labels.json']:
        assert (ROOT/'data_v3'/name).read_bytes()==(origin/'data_v3'/name).read_bytes()
    assert hashlib.sha256((origin/'pinned_worker_preimage.py').read_bytes()).hexdigest()==original_pins[str(Path(__file__).resolve())]
    files=[Path(x)for x in cfg.get('resume_row_paths',[])] or sorted(origin.glob('panel_row_*.json'))
    for index,path in enumerate(files):
        row=json.loads(path.read_text())
        producer=path.parent
        producer_pins=json.loads((producer/'pins.json').read_text())
        producer_cfg=json.loads((producer/'config.json').read_text())
        assert all(producer_cfg[k]==original_cfg[k]for k in ('reader','dense','small_reader','prefill','allocator_fraction'))
        assert (producer/'data_v3/reader.json').read_bytes()==(origin/'data_v3/reader.json').read_bytes()
        assert hashlib.sha256((producer/'pinned_worker_preimage.py').read_bytes()).hexdigest()==producer_pins[str(Path(__file__).resolve())]
        if producer.resolve()!=origin.resolve():
            assert Path(producer_cfg['resume_from']).resolve()==origin.resolve()
            replay=json.loads((producer/'resume_replay.json').read_text())
            assert replay['mass_max_error']<=1e-6 and replay['answer_equal']
        assert row['id']==questions[index]['id'] and row['pins_sha256']==hashlib.sha256((producer/'pins.json').read_bytes()).hexdigest()
        assert row['execution_id']==json.loads((producer/'panel_resources.json').read_text())['execution_id']
        for key in ('phase','execution_id','pins_sha256'):row.pop(key)
        row['producer_execution_id']=json.loads((producer/'panel_resources.json').read_text())['execution_id']
        row['producer_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        resume_rows.append(row)
    assert len(resume_rows)==cfg['resume_completed_rows']>0
    restored=time.monotonic()
    def pairs():
        for member in manifest['members']:
            path=origin/'prefix_checkpoint'/member['path']
            assert str(path.resolve()) in pins and pins[str(path.resolve())]==member['sha256']
            cool();pair=torch.load(path,map_location='cpu',weights_only=True)
            assert len(pair)==2 and all(t.dtype==torch.bfloat16 and tuple(t.shape)==(1,32,len(prefix_ids),96) for t in pair)
            yield tuple(t.to('cuda') for t in pair)
            del pair
    cache=DynamicCache(ddp_cache_data=pairs(),config=model.config)
    torch.cuda.synchronize();restore_seconds=time.monotonic()-restored
    prefill_started=time.monotonic()
else:
    cache = DynamicCache(config=model.config)
    torch.cuda.synchronize()
    prefill_started = time.monotonic()
    with torch.inference_mode():
        # Fixed long-horizon frequencies allow bounded chunks without regime drift.
        chunk=128
        for offset in range(0, len(prefix_ids), chunk):
            cool()
            out = model(input_ids=torch.tensor([prefix_ids[offset:offset+chunk]], device='cuda'),
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
            del out
            torch.cuda.synchronize()
            time.sleep(.20)
assert cache.get_seq_length() == len(prefix_ids)
torch.cuda.synchronize()
prefill_seconds = time.monotonic()-prefill_started
prefix_payload_bytes=sum(t.numel()*t.element_size() for pair in cache for t in pair[:2])
if phase=='panel' and cfg.get('checkpoint_prefix') and not cfg.get('resume_from'):
    # Layer-wise host staging bounds RAM; this is native KV, not compact memory.
    directory=ROOT/'prefix_checkpoint';directory.mkdir(exist_ok=False)
    members=[]
    for index,(k,v,*_) in enumerate(cache):
        cool();pair=(k.detach().cpu(),v.detach().cpu())
        path=directory/f'layer_{index:02}.pt';torch.save(pair,path);del pair
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for piece in iter(lambda:source.read(1024*1024),b''):digest.update(piece)
        members.append(dict(path=path.name,sha256=digest.hexdigest(),bytes=path.stat().st_size))
    (directory/'manifest.json').write_text(json.dumps(dict(members=members,
        prefix_ids_sha256=hashlib.sha256(json.dumps(prefix_ids).encode()).hexdigest(),
        pins_sha256=pin_digest,length=len(prefix_ids),payload_bytes=prefix_payload_bytes)),encoding='utf8')

def cache_digest():
    h=hashlib.sha256()
    for k,v,*_ in cache:
        assert k.shape[-2]==len(prefix_ids) and v.shape[-2]==len(prefix_ids)
        for x in (k,v):
            h.update(x.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

initial_cache_digest=cache_digest() if phase.startswith('smoke') else None
rows = list(resume_rows)
for i, question in enumerate(questions):
    if resume_rows and i<len(resume_rows)-1:continue
    cool()
    suffix = tokenizer.encode('\nQuestion: '+question['question']+'<|end|>\n<|assistant|>\n', add_special_tokens=False)
    started = time.monotonic()
    scores, logits = attention_query(model, cache, suffix, spans)
    row = dict(id=question['id'], scores=scores, query_tokens=len(suffix),
               query_seconds=time.monotonic()-started)
    if i == 0 or (phase=='panel' and i%10==0):
        # Repeated identical input BEFORE any diversity/selection interpretation.
        again, repeated = attention_query(model, cache, suffix, spans)
        row['repeat_logit_max_error'] = max(abs(a-b) for a,b in zip(logits,repeated))
        row['repeat_mass_max_error'] = max(abs(a-b) for x,y in zip(scores,again) for a,b in zip(x,y))
        if phase.startswith('smoke'):
            model.config._attn_implementation = 'eager'
            with torch.inference_mode():
                direct = model(input_ids=torch.tensor([prefix_ids+suffix], device='cuda'),
                               use_cache=False, logits_to_keep=1).logits[0,-1].float().cpu().tolist()
            model.config._attn_implementation = 'sdpa'
            row['cached_direct_logit_max_error'] = max(abs(a-b) for a,b in zip(logits,direct))
    del logits
    # Full-context QA uses the entire SAME corpus prefix. This is the strong
    # prefix-cache incumbent; there is no fictional full-prefill saving versus it.
    qa_started=time.monotonic()
    with torch.inference_mode():
        cool()
        out = model(input_ids=torch.tensor([suffix], device='cuda'),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        if i==0 and phase.startswith('smoke'):
            with torch.inference_mode():
                raw=model(input_ids=torch.tensor([prefix_ids+suffix],device='cuda'),
                          use_cache=False,logits_to_keep=1).logits[0,-1]
            row['sdpa_cached_direct_max_error']=float((out.logits[0,-1]-raw).float().abs().max())
            row['sdpa_cached_direct_greedy_equal']=int(out.logits[0,-1].argmax())==int(raw.argmax())
            del raw
        answer = []
        terminated = False
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        for _ in range(32):
            token = int(out.logits[0,-1].argmax())
            if token in eos:
                terminated = True
                break
            answer.append(token)
            cool()
            out = model(input_ids=torch.tensor([[token]], device='cuda'),
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
        del out
        cache.crop(len(prefix_ids))
    torch.cuda.synchronize()
    row.update(answer=tokenizer.decode(answer), answer_tokens=len(answer), terminated=terminated,
               qa_wall_seconds_including_cooldown=time.monotonic()-qa_started)
    row['producer_execution_id']=execution_id
    if resume_rows and i==len(resume_rows)-1:
        prior=resume_rows[-1]
        error=max(abs(a-b) for ha,hb in zip(prior['scores'],row['scores']) for a,b in zip(ha,hb))
        assert error<=1e-6 and row['answer']==prior['answer'] and row['terminated']==prior['terminated'],'cold-cache replay mismatch'
        rows[-1]=row
        save('resume_replay.json',dict(id=row['id'],mass_max_error=error,answer_equal=True,source_execution=prior['producer_execution_id']))
    else:rows.append(row)
    if phase.startswith('smoke'):
        assert cache_digest()==initial_cache_digest,'cache content changed after attention/QA'
        if row['id']=='S1repeat':
            assert row['answer']==rows[0]['answer'] and row['scores']==rows[0]['scores'],'A-B-A restoration failed'
    save(f'{phase}_row_{i:02}.json', row)
    print(json.dumps(dict(phase=phase, i=i, id=row['id'], answer=row['answer'])), flush=True)

if phase == 'panel':
    del model, cache
    torch.cuda.empty_cache()
    cool()
    # Dense embeddings are built ONCE per window, exhaustively covering every
    # block; max window similarity avoids silently truncating 400-token blocks.
    dense_tok = AutoTokenizer.from_pretrained(cfg['dense'], local_files_only=True)
    dense = AutoModel.from_pretrained(cfg['dense'], local_files_only=True).cuda().eval()
    def embed(ids):
        cool()
        tokens=torch.tensor([[dense_tok.cls_token_id]+ids+[dense_tok.sep_token_id]],device='cuda')
        batch=dict(input_ids=tokens,attention_mask=torch.ones_like(tokens),token_type_ids=torch.zeros_like(tokens))
        with torch.inference_mode():
            h = dense(**batch).last_hidden_state
            mask = batch['attention_mask'][...,None].float()
            return torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1), dim=-1)
    vectors, owners = [], []
    for block, text in enumerate(data['blocks']):
        ids = dense_tok.encode(text, add_special_tokens=False)
        for start in range(0,len(ids),240):
            vectors.append(embed(ids[start:start+254]))
            owners.append(block)
            if start+254 >= len(ids):
                break
    bank = torch.cat(vectors)
    dense_rows = {}
    query_vectors=[]
    for question in questions:
        ids = dense_tok.encode(question['question'], add_special_tokens=False)
        assert len(ids) <= 256
        vector=embed(ids)
        query_vectors.append(vector.detach().float().cpu().numpy()[0])
        similarity = (vector @ bank.T)[0].cpu().tolist()
        scores = [-2.0]*60
        for owner, score in zip(owners,similarity):
            scores[owner] = max(scores[owner], score)
        dense_rows[question['id']] = scores
    save('dense.json', dict(scores=dense_rows,windows=len(owners)))
    import numpy as np
    np.savez(ROOT/'dense_vectors.npz',bank=bank.detach().float().cpu().numpy(),owners=np.asarray(owners),queries=np.stack(query_vectors),question_ids=np.asarray([q['id'] for q in questions]))
save(phase+'_measurements.json', dict(rows=rows, prefill_wall_seconds_including_cooldown=prefill_seconds,
    corpus_tokens=len(prefix_ids), precision='bf16', corpus_encoded_once=True,
    prefix_payload_bytes=prefix_payload_bytes,reader_parameters=reader_parameters,
    cache_digest=initial_cache_digest, cache_restore_seconds=restore_seconds,
    resume_from=cfg.get('resume_from'), inherited_rows=len(resume_rows),
    source_prefill_seconds=None if resume_rows else prefill_seconds,
    reader='native Phi user/corpus; Question plus end/assistant suffix',
    peak_allocated=torch.cuda.max_memory_allocated()))
