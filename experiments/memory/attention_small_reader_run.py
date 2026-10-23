"""Finite <4B native-Phi CUDA attention experiment; RUN_DIR PHASE, external supervision required."""
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
    REPO/'experiments/memory/small_reader_gpu.py',
    REPO/'experiments/memory/gpu_record_memory.py',
    REPO/'experiments/memory/record_native_memory.py',REPO/'experiments/memory/record_memory_input.py',ROOT/'config.json',ROOT/'resource_guard.py',
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
burst_started=time.monotonic()

def thermal_row():
    until=time.monotonic()+.075
    while True:
        try:
            row=json.loads((ROOT/(phase+'_external_thermal.json')).read_text())
            break
        except PermissionError:
            if time.monotonic()>=until:raise
            time.sleep(.002)
    assert row['phase']==phase and row['execution_id']==execution_id,'thermal identity mismatch'
    assert row['uuid']==os.environ['LSR_GPU_UUID'],'thermal GPU mismatch'
    assert time.monotonic()-row['monotonic']+row['vendor_age']<=1.25,'stale vendor thermal sample'
    return row

def temperature():
    return thermal_row()['temperature']

def check():
    assert psutil.virtual_memory().available >= 20*2**30, 'RAM floor'
    assert temperature()<thermal['stop'], 'thermal stop'
    free,total=torch.cuda.mem_get_info()
    assert total-free<=19.5*2**30,'CUDA physical VRAM stop'

def cool():
    global burst_started
    start = time.monotonic()
    if start-burst_started>=thermal['burst_seconds'] or temperature()>=thermal['cool']+2:
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
from experiments.memory.attention_completeness import attention_query as raw_attention_query
from experiments.memory.small_reader_gpu import stream_model
from experiments.memory.gpu_record_memory import query_tokens

assert psutil.virtual_memory().available>=20*2**30,'RAM floor'
assert temperature()<thermal['stop'],'thermal stop'

tokenizer = AutoTokenizer.from_pretrained(cfg['reader'], local_files_only=True)
def loader_checkpoint():
    # Each transfer is one parameter; wait/check between transfers. CPU file time
    # is not accumulated as a long GPU compute burst.
    torch.cuda.synchronize()
    check()
    if temperature()>=thermal['cool']+2:cool()
    time.sleep(.025)
model = stream_model(cfg['reader'], checkpoint=loader_checkpoint)
model_parameter_count = model.lsr_parameter_count
# Blocking thermal repair: at most one decoder layer queues before cooling.
# This preserves the full native long-RoPE prefill and all token positions.
def layer_boundary(module, args):
    global burst_started
    torch.cuda.synchronize()
    completed=time.monotonic()
    if args[0].shape[-2]>=512:
        # Fixed duty cycle for long prefills, regardless of the burst timer.
        time.sleep(1)
        until=time.monotonic()+1.25
        while True:
            row=thermal_row()
            if row['monotonic']-row['vendor_age']>=completed:break
            if time.monotonic()>=until:raise TimeoutError('no post-layer thermal observation')
            time.sleep(.025)
        burst_started=time.monotonic()  # fixed idle ended this compute burst
    cool()
layer_cooling_handles=[layer.register_forward_pre_hook(layer_boundary)
                       for layer in model.model.layers]

window=getattr(model.config,'sliding_window',None)
assert window is None or window>=12288,'sliding window masks declared context'

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
    smoke = json.loads((ROOT/'smoke_verdict.json').read_text())
    assert smoke['passed'] and smoke['pins_sha256'] == hashlib.sha256((ROOT/'pins.json').read_bytes()).hexdigest()
    for filename,expected in smoke['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected,'smoke artifact drift'
    prefix_ids, spans, questions = data['ids'], data['spans'], data['questions']

# Resolve trusted native-template markers BEFORE inserting preencoded corpus IDs.
framed = tokenizer.apply_chat_template([dict(role='user', content=
    'Answer the question using the corpus below. Give only the answer; '
    'if it is missing, say NOT IN MEMORY.\nCorpus:\n__CORPUS_SLOT__'
    '\n\nQuestion:\n__LATE_QUESTION__')], tokenize=False, add_generation_prompt=True)
before, after = framed.split('__CORPUS_SLOT__')
between, suffix_end = after.split('__LATE_QUESTION__')
assert between.endswith('Question:\n'),'native question anchor changed'
header = tokenizer.encode(before, add_special_tokens=False)
between_ids = tokenizer.encode(between[:-1], add_special_tokens=False)
# Corpus IDs are inserted unchanged; surrounding native framing is encoded once.
corpus_ids = prefix_ids
prefix_ids = header + corpus_ids + between_ids
spans = [[a+len(header), b+len(header)] for a,b in spans]
# Include trusted trailing framing as a separate temporary attention bucket,
# then discard that bucket so corpus block masses exclude question framing.
framing_span = [spans[-1][1], len(prefix_ids)]
def attention_query(model, cache, query, spans):
    scores, logits = raw_attention_query(model, cache, query, [*spans, framing_span])
    return [head[:-1] for head in scores], logits
assert len(prefix_ids)+32<=12288,'complete framed corpus exceeds token bound'

def late_ids(question):
    ids=query_tokens(tokenizer,question,suffix_end)
    assert len(prefix_ids)+len(ids)+32<=12288,'complete source/query/generation exceeds 12288'
    return ids

cache = DynamicCache(config=model.config)
torch.cuda.synchronize()
prefill_started = time.monotonic()
with torch.inference_mode():
    # One prefill selects a consistent long-RoPE regime for this known horizon.
    chunk=len(prefix_ids)
    for offset in range(0, len(prefix_ids), chunk):
        cool()
        out = model(input_ids=torch.tensor([prefix_ids[offset:offset+chunk]], device='cuda'),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        del out
assert cache.get_seq_length() == len(prefix_ids)
torch.cuda.synchronize()
if len(prefix_ids)>=512:
    completed=time.monotonic()
    time.sleep(1)
    until=time.monotonic()+1.25
    while True:
        row=thermal_row()
        if row['monotonic']-row['vendor_age']>=completed:break
        if time.monotonic()>=until:raise TimeoutError('no final post-prefill thermal observation')
        time.sleep(.025)
    burst_started=time.monotonic()
    cool()
prefill_seconds = time.monotonic()-prefill_started

def cache_digest():
    h=hashlib.sha256()
    for k,v,*_ in cache:
        assert k.shape[-2]==len(prefix_ids) and v.shape[-2]==len(prefix_ids)
        for x in (k,v):
            h.update(x.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

initial_cache_digest=cache_digest() if phase.startswith('smoke') else None
rows = []
for i, question in enumerate(questions):
    cool()
    suffix = late_ids(question['question'])
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
    rows.append(row)
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
        batch = dense_tok.prepare_for_model(ids, return_tensors='pt')
        batch = {k:v.cuda()[None] if v.ndim==1 else v.cuda() for k,v in batch.items()}
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
    for question in questions:
        ids = dense_tok.encode(question['question'], add_special_tokens=False)
        assert len(ids) <= 256
        similarity = (embed(ids) @ bank.T)[0].cpu().tolist()
        scores = [-2.0]*60
        for owner, score in zip(owners,similarity):
            scores[owner] = max(scores[owner], score)
        dense_rows[question['id']] = scores
    save('dense.json', dict(scores=dense_rows,windows=len(owners)))
save(phase+'_measurements.json', dict(rows=rows, prefill_wall_seconds_including_cooldown=prefill_seconds,
    corpus_tokens=len(prefix_ids), precision='bf16', corpus_encoded_once=True,
    cache_digest=initial_cache_digest,
    reader='native Phi user corpus/question and assistant generation framing',
    parameters=model_parameter_count,
    peak_allocated=torch.cuda.max_memory_allocated()))
