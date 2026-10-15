"""One-load smoke -> fit -> DEV generation. Offline checks use RANDOM weights.

No dispatcher, cloud, download, readiness or scientific-acceptance authority.
The restarted-reader boundary is deliberately outside this in-process entry.
"""
import argparse
import copy
from contextlib import contextmanager
import json
import hashlib
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from . import semantic as sem, data, evaluate
from experiments.cloud.lsr_jobs import gemma4_receiver as receiver_api


def write(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sem.file_hash(path)



class Progress:
    """Operation-bound progress, on the supervisor's absolute work clock."""
    def __init__(self, cfg, out):
        settings = cfg.get('resident', {})
        dispatch = cfg.get('dispatch', {})
        self.root = Path(settings.get('runtime_output_root', out))
        if not self.root.is_dir():
            raise ValueError('runtime output root must already exist')
        self.stage = settings.get('runtime_stage', 'work')
        if self.stage not in ('work', 'control'):
            raise ValueError('resident runtime stage must match work/control')
        real = cfg['semantic']['stem']['kind'] != 'tiny-random'
        end = dispatch.get('work_stop_at', settings.get('work_stop_at'))
        shutdown = dispatch.get('shutdown_at', settings.get('shutdown_at'))
        if end is None and not real:
            end, shutdown = time.time()+3600, time.time()+4500
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (end, shutdown)):
            raise ValueError('absolute work and shutdown deadlines required')
        if shutdown-end < 900:
            raise ValueError('at least 900 seconds custody reserve required')
        if real and (not settings.get('runtime_output_root') or 'runtime_stage' not in settings):
            raise ValueError('real resident needs explicit runtime progress root and stage')
        if real and (not self.root.is_absolute() or self.root.is_symlink()
                     or not Path(out).resolve().is_relative_to(self.root.resolve())):
            raise ValueError('resident output must be inside its absolute runtime progress root')
        self.end = end
        self.count = 0
        self.phase = 'load'
        self.last_emit = 0.
        self.last_phase = None

    def guard(self):
        if time.time() >= self.end:
            raise TimeoutError('absolute resident work deadline; custody reserve retained')

    def __call__(self, event):
        self.guard()
        self.count += 1
        now = time.time()
        if now-self.last_emit < 1 and self.last_phase == self.phase:
            return  # Deadline guarded above; emission only follows actual work.
        self.last_emit, self.last_phase = now, self.phase
        write(self.root / ('heartbeat_'+self.stage), {'stage': self.stage,
              'phase': self.phase, 'event': event, 'sequence': self.count,
              'epoch': time.time(), 'work_stop_at': self.end})



@contextmanager
def loading_progress(progress):
    """Instrument real hash blocks and real tensor loading, never timer activity.

    Wrappers preserve the pinned loader's validation and restore globals on every
    exit. One resident session owns loading in this process. The core loader
    source hash is retained with the extraction receipt.
    """
    if progress is None:
        yield
        return
    from transformers import core_model_loading
    original_hash, original_bar = receiver_api._sha256, core_model_loading.tqdm

    def hash_blocks(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(chunk)
                progress('load_hash_block_complete')
        return digest.hexdigest()

    class TensorProgress:
        def __init__(self, *args, **kwargs):
            self.bar = original_bar(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.bar, name)

        def __enter__(self):
            self.bar.__enter__()
            return self

        def __exit__(self, *args):
            return self.bar.__exit__(*args)

        def __iter__(self):
            for item in self.bar:
                progress('load_tensor_start')
                yield item
                progress('load_tensor_complete')

    receiver_api._sha256 = hash_blocks
    core_model_loading.tqdm = TensorProgress
    try:
        yield
    finally:
        receiver_api._sha256 = original_hash
        core_model_loading.tqdm = original_bar


def tick(handle, event):
    callback = handle.get('_progress')
    if callback:
        callback(event)


def clock(handle):
    device = handle['receiver'].stem.get_input_embeddings().weight.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.monotonic()


def headroom(handle, out, event):
    tick(handle, event)
    receipt = placement(handle, out)
    write(out / ('placement_'+event+'.json'), receipt)
    if receipt['status'] not in ('PASS', 'MECHANICS_ONLY'):
        raise ValueError('headroom/offload violation: '+event)
    return receipt


class Journal:
    """Every completed row survives a later exception; manifest never implies PASS."""
    def __init__(self, out, name, identity):
        self.path = out / (name+'.jsonl')
        self.manifest = out / (name+'_manifest.json')
        self.identity = identity
        self.hash = hashlib.sha256()
        self.count = 0
        self.arms = {}
        with self.path.open('xb') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        self.publish('INCOMPLETE')

    def append(self, row):
        raw = (json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode()
        with self.path.open('ab') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.hash.update(raw)
        self.count += 1
        if 'arm' in row:
            self.arms[row['arm']] = self.arms.get(row['arm'], 0)+1
        self.publish('INCOMPLETE')

    def publish(self, status):
        write(self.manifest, {'status': status, 'all_pass': False, 'rows': self.count,
              'completed_rows_by_arm': self.arms, 'bytes': self.path.stat().st_size,
              'sha256': self.hash.hexdigest(), 'identity': self.identity})


def checkpoint(handle, cfg, out, step, cases_hash, losses):
    from safetensors.torch import save_file
    path = out / ('connector_initial.safetensors' if step == 0 else 'connector.safetensors')
    pending = path.with_name(path.name+'.pending')
    identity = {'bindings': bindings(cfg), 'cases_sha256': cases_hash, 'step': step,
                'seed': cfg['seed'], 'torch_rng_sha256': sem.digest(torch.get_rng_state().tolist()),
                'diagnostic_only': True, 'resumable_training': False}
    save_file({k: v.detach().cpu().contiguous() for k,v in handle['connector'].state_dict().items()},
              str(pending), metadata={'identity': json.dumps(identity, sort_keys=True)})
    with pending.open('r+b') as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    write(out / (path.stem+'_manifest.json'), dict(identity, sha256=sem.file_hash(path),
          bytes=path.stat().st_size, losses=losses, status='INCOMPLETE', all_pass=False))
    return sem.file_hash(path)


def preflight(cfg):
    if cfg['semantic']['stem']['kind'] == 'tiny-random':
        return
    settings, rc = cfg.get('resident', {}), real_config(cfg)
    if cfg.get('offline') is not False or cfg['semantic'].get('cloud_stems', {}).get('E2B', {}).get('enabled') is not True:
        raise ValueError('real resident requires explicit enabled cloud stem')
    if settings.get('thresholds_frozen') is not True or settings.get('generator_sha256') != sem.file_hash(data.__file__):
        raise ValueError('frozen registration/generator required before load')
    panel = settings.get('prior_panel')
    if not isinstance(panel, list) or len(panel) != 20 or any(
            not isinstance(c, dict) or any(not isinstance(c.get(k), str) or not c[k].strip()
            for k in ('query', 'evidence', 'answer')) for c in panel):
        raise ValueError('20 nonempty registered prior cases required before load')
    if not isinstance(rc.get('placement'), dict):
        raise ValueError('explicit placement required before load')
    if cfg['semantic'].get('train_documents') != 2048 or cfg['semantic'].get('dev_documents') != 128:
        raise ValueError('full registered TRAIN/DEV counts required')
    if cfg['semantic'].get('budget') not in (8, 16) or cfg['semantic'].get('batch_size') != 8:
        raise ValueError('registered decoding budget/effective batch required')


def answer_ids(handle, query):
    prefix = sem.prompt(query)
    before = handle['tok'](prefix, add_special_tokens=False)['input_ids']
    ids = []
    for word in data.WORDS:
        joint = handle['tok'](prefix+' '+word, add_special_tokens=False)['input_ids']
        if joint[:len(before)] != before or len(joint) <= len(before):
            raise ValueError('answer boundary not prefix-stable for '+word)
        ids.append(joint[len(before)])
    return ids


def bindings(cfg):
    here = Path(__file__).parent
    paths = [here / n for n in ('resident.py', 'semantic.py', 'writer.py',
                               'data.py', 'evaluate.py', 'trainer.py', 'store.py')]
    paths += [Path(receiver_api.__file__), Path(sem.port.__file__), Path(sem.env.__file__)]
    return {'config': sem.digest(cfg), 'code': {str(p.resolve()): sem.file_hash(p) for p in paths}}


def real_config(cfg):
    rc = copy.deepcopy(cfg['semantic']['stem']['receiver_config'])
    pin = cfg['semantic'].get('cloud_stems', {}).get('E2B', {})
    rc.setdefault('weights_manifest_sha256', pin.get('manifest_sha256'))
    return rc


def verify_assets(rc):
    # Validate paths and immutable identity BEFORE any pretrained loader is called.
    root = Path(rc.get('model_local', ''))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('staged mirror missing; no HF fallback')
    if rc.get('model') != 'google/gemma-4-E2B' or rc.get('allow_hf_download') is not False:
        raise ValueError('explicit E2B identity and no-download policy required')
    for key in ('revision', 'tokenizer_revision'):
        if re.fullmatch('[0-9a-f]{40}', str(rc.get(key))) is None:
            raise ValueError('immutable revision pins required')
    if rc['revision'] != rc['tokenizer_revision']:
        raise ValueError('tokenizer/stem revision mismatch')
    return receiver_api._validate_staged(rc, root)


def load(cfg, progress=None):
    if cfg['semantic']['stem']['kind'] == 'tiny-random':
        return sem.load_handle(cfg)
    preflight(cfg)
    rc = real_config(cfg)
    if progress:
        progress('asset_verify_start')
    with loading_progress(progress):
        verify_assets(rc)
    if progress:
        progress('asset_verify_complete')
    if rc.get('dtype') != 'bfloat16':
        raise ValueError('real stem requires bf16')
    devices = set((rc.get('placement') or {}).get('device_map', {}).values())
    if len(devices) != 1 or not all(re.fullmatch('cuda:[0-9]+', d) for d in devices):
        raise ValueError('one explicitly placed GPU required; no CPU/disk offload')
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(cfg['seed'])
    if progress:
        progress('model_load_start')
    with loading_progress(progress):
        tok, original, audit = receiver_api._load_original(rc)  # exactly one staged load
    if progress:
        progress('model_load_complete')
    stem = receiver_api._extract_text(original)
    # Compare actual loaded conditional outputs before releasing its towers.
    probe = tok(sem.prompt('what is the north key of Ada?'), add_special_tokens=False,
                return_tensors='pt')
    device = stem.get_input_embeddings().weight.device
    probe = {k: v.to(device) for k, v in probe.items()}
    with torch.no_grad():
        original_logits = original(**probe, use_cache=False).logits
        extracted_logits = stem(**probe, use_cache=False).logits
    parity = torch.equal(original_logits, extracted_logits)
    if not parity:
        raise ValueError('actual original/extracted text mapping parity failed')
    del original, original_logits, extracted_logits
    if progress:
        progress('extraction_parity_complete')
    c = stem.config
    torch.manual_seed(cfg['seed'])
    connector = receiver_api.MemoryResampler(receiver_api.ResamplerConfig(
        memory_dim=rc.get('memory_dim', c.hidden_size), hidden_size=c.hidden_size,
        attention_dim=rc.get('attention_dim',64), num_heads=rc.get('num_heads',4)))
    connector.to(device=device)
    receiver = receiver_api.Gemma4Receiver(stem, connector)
    info = receiver_api._stem_info(rc, stem, audit)
    from transformers import core_model_loading
    info['load_observer_backend_sha256'] = sem.file_hash(core_model_loading.__file__)
    info['actual_weight_parity'] = 'PASS actual original/extracted logits (one loaded checkpoint)'
    handle = {'tok': tok, 'receiver': receiver, 'connector': connector, 'stem_info': info}
    handle['connector'].float()
    dtype = handle['receiver'].stem.get_input_embeddings().weight.dtype
    # FP32 connector parameters/Adam state, differentiable cast at the stem boundary.
    handle['connector'].register_forward_hook(lambda module, args, output: output.to(dtype=dtype))
    c = handle['receiver'].stem.config
    if (handle['stem_info']['D_memory'] != c.hidden_size or c.hidden_size != rc['width']
            or c.num_hidden_layers != rc['n_layers']):
        raise ValueError('loaded dimensions disagree with pinned recipe')
    return handle


def documents(split, count, seed):
    if split not in ('train', 'dev') or count <= 0 or count % 2:
        raise ValueError('resident admits even TRAIN/DEV only; TEST is locked')
    pairs = data.make_split(split, count // 2, seed)
    return [{'key': str(p['pair']) + suffix, 'document': p['doc_' + suffix],
             'reads': p['reads_' + suffix]} for p in pairs for suffix in 'ab']


def view(doc, read):
    records = copy.deepcopy(doc['document']['records'])
    if read['snapshot'] == 'v2':
        replacement = doc['document']['revision']
        records[replacement['record_id']] = replacement
    return next(r for r in records if r['slot'] == read['query']['slot'])


def competence(handle, dev, cfg):
    # Exactly two cases per value, selected independently of model outputs.
    started = clock(handle)
    by_value = {w: [] for w in data.WORDS}
    for doc in dev:
        for read in doc['reads']:
            by_value[read['answer']].append((doc, read))
    if any(len(v) < 2 for v in by_value.values()):
        raise ValueError('direct-evidence panel must cover every value twice')
    chosen = [pair for word in data.WORDS for pair in by_value[word][:2]]
    budget = cfg['semantic'].get('budget', 8)
    rows = []
    for doc, read in chosen:
        result = sem.generate(handle, read['query']['text'], evidence=view(doc, read)['text'], budget=budget)
        rows.append(dict(result, expected=read['answer'], doc=doc['key'], op=read['op']))
    panel = cfg.get('resident', {}).get('prior_panel', [])
    prior = []
    for case in panel:
        result = sem.generate(handle, case['query'], evidence=case['evidence'], budget=budget)
        prior.append(dict(result, expected=case['answer']))
    def correct(r):
        return evaluate.exact_match(r['generated'], r['expected']) and r['terminated'] and bool(r['generated'])
    hits = sum(map(correct, rows))
    termination = sum(r['terminated'] and bool(r['generated']) for r in rows) / 64
    prior_hits = sum(map(correct, prior))
    return {'rows': rows, 'correct_terminated': hits, 'total': 64,
            'value_coverage': len(by_value), 'termination': termination,
            'decode_seconds': clock(handle)-started, 'decode_rows': len(rows)+len(prior),
            'prior_panel': prior, 'prior_correct_terminated': prior_hits,
            'pass': hits >= 61 and termination >= .95 and len(prior) == 20 and prior_hits >= 19}


def placement(handle, out):
    devices = {str(p.device) for p in handle['receiver'].stem.parameters()}
    tiny = handle['stem_info'].get('scope') == 'tiny-random'
    rows = []
    if not tiny:
        for device in devices:
            if not device.startswith('cuda:'):
                raise ValueError('offloaded stem')
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            current_reserved = torch.cuda.memory_reserved(device)
            # Include non-PyTorch device occupancy in a conservative peak bound.
            peak_free = max(0, free - max(0, peak_reserved - current_reserved))
            rows.append({'device': device, 'total_bytes': total, 'free_bytes': free,
                         'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
                         'conservative_peak_headroom_fraction': peak_free / total})
    receipt = {'status': 'MECHANICS_ONLY' if tiny else ('PASS' if len(rows) == 1 and all(
        r['conservative_peak_headroom_fraction'] >= .1 for r in rows) else 'FAIL'),
        'devices': sorted(devices), 'measurements': rows,
        'memory_plan': {'weights_estimate_gb': 10.3, 'expected_peak_gb_range': [16, 22],
                        'minimum_headroom_fraction': .1, 'microbatch': 1,
                        'effective_batch': 8, 'precision': 'bf16 frozen stem / fp32 connector',
                        'note': 'Estimate only; includes multimodal load peak, eager activations and vocabulary logits. Actual measured headroom gates fit.'}}
    write(out / 'placement.json', receipt)
    return receipt


def controls(handle, cfg, out):
    receipt = sem.smoke(handle, cfg, out / 'tensor_smoke.json')
    docs = sem.slice_documents('train', 2, cfg['seed'])
    native = sem.states(handle, docs)
    memory = native[docs[0]['key']]['v1']
    read = docs[0]['reads'][0]
    inputs = sem.answer_batch(handle, [(read['query']['text'], read['answer'])])
    checks = dict(receipt['checks'])
    model = handle['receiver'].stem
    device = model.get_input_embeddings().weight.device
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        none = sem.port.forward(handle, dict(inputs, labels=None))['logits']
        checks['base_none_exact'] = torch.equal(base, none)
        cache = sem.port.forward(handle, {'input_ids': ids, 'use_cache': True}, memory)
        token = cache['logits'][:, -1].argmax(-1, keepdim=True)
        cached = handle['receiver'].decode(token, cache['past_key_values'])['logits'][:, -1]
        full_ids = torch.cat((ids, token), dim=1)
        full = sem.port.forward(handle, {'input_ids': full_ids}, memory)['logits'][:, -1]
        delta = float((cached-full).abs().max())
        # Frozen numerical tolerance for dtype rounding, with separate argmax gate.
        cache_tolerance = max(1e-2, 4*torch.finfo(base.dtype).eps)
        checks['cache_recompute'] = torch.equal(cached.argmax(-1), full.argmax(-1)) and torch.allclose(cached, full, atol=cache_tolerance, rtol=cache_tolerance)
        padded = torch.cat((torch.full((1,2), handle['tok'].pad_token_id, device=device), ids), dim=1)
        padded_mask = torch.cat((torch.zeros((1,2), dtype=torch.long, device=device), torch.ones_like(ids)), dim=1)
        continued = sem.port.forward(handle, {'input_ids': padded, 'attention_mask': padded_mask, 'use_cache': True}, memory)
        continuation_ok = True
        for _ in range(3):
            next_token = continued['logits'][:, -1].argmax(-1, keepdim=True)
            decoded = handle['receiver'].decode(next_token, continued['past_key_values'])
            padded = torch.cat((padded, next_token), dim=1)
            padded_mask = torch.cat((padded_mask, torch.ones_like(next_token)), dim=1)
            recomputed = sem.port.forward(handle, {'input_ids': padded, 'attention_mask': padded_mask}, memory)['logits'][:, -1]
            continuation_ok &= torch.equal(decoded['logits'][:, -1].argmax(-1), recomputed.argmax(-1)) and torch.allclose(decoded['logits'][:, -1], recomputed, atol=cache_tolerance, rtol=cache_tolerance)
            continued = decoded
        checks['left_padding_three_cached_tokens'] = bool(continuation_ok)
        singleton = sem.port.forward(handle, {'input_ids': ids}, memory)['logits']
        repeat = sem.port.forward(handle, {'input_ids': ids}, memory)['logits']
        batched = sem.port.forward(handle, {'input_ids': ids.expand(2, -1)},
                                  [native[docs[1]['key']]['v1'], memory])['logits'][1:]
        batch_noise = float((singleton-batched).abs().max())
        repeat_noise = float((singleton-repeat).abs().max())
        changed = sem.port.forward(handle, {'input_ids': ids}, native[docs[1]['key']]['v1'])['logits']
        answer_slice = answer_ids(handle, read['query']['text'])
        answer_position = len(handle['tok'](sem.prompt(read['query']['text']), add_special_tokens=False)['input_ids']) - 1
        effect = float((singleton[:, answer_position, answer_slice]-changed[:, answer_position, answer_slice]).abs().max())
        checks['answer_slice_content_use'] = effect > max(1e-7, 10*repeat_noise, 10*batch_noise)
        checks['batch_argmax_agreement'] = torch.equal(singleton.argmax(-1), batched.argmax(-1))
        origins = copy.deepcopy(memory['meta']['origin'][:2])
        origins[0]['valid_until'] = 1
        origins[1].update(event_time=1, ingestion_time=1, valid_from=1)
        payload = sem.env.to_torch(memory['payload'])[:2]
        temporal = sem.env.make_envelope(payload, origin=origins,
            temporal_horizon={'as_of': None, 'max_age': None},
            expected_width=handle['stem_info']['D_memory'], expected_schema=sem.env.WIDTH_SCHEMA_VERSION)
        observed = []
        hook = handle['connector'].register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().cpu().clone()))
        try:
            sem.port.forward(handle, {'input_ids': ids}, temporal, conditioning={'branch':'main', 'as_of':1, 'max_age':None, 'permissions':['read']})
        finally:
            hook.remove()
        checks['temporal_survivor_row1'] = len(observed) == 1 and observed[0].shape[1] == 1 and torch.equal(observed[0][0, 0].float(), payload[1].float())
    # Restore every diagnostic optimizer move; no fit may inherit a smoke update.
    initial = copy.deepcopy(handle['connector'].state_dict())
    optimizer = torch.optim.SGD(handle['connector'].parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    loss = sem.port.forward(handle, inputs, memory)['loss']
    loss.backward()
    optimizer.step()
    checks['diagnostic_step_changed'] = any(not torch.equal(v, initial[k]) for k, v in handle['connector'].state_dict().items())
    handle['connector'].load_state_dict(initial)
    handle['receiver'].zero_grad(set_to_none=True)
    checks['diagnostic_restored'] = all(torch.equal(v, initial[k]) for k, v in handle['connector'].state_dict().items())
    # Real adversarial output replacements: constant logits and digest-echo fields
    # must not pass the same returned answer-logit content-effect test.
    from unittest.mock import patch
    negatives = {}
    for kind in ('constant', 'digest_echo'):
        original = handle['receiver'].forward
        def bad(*args, **kwargs):
            result = original(*args, **kwargs)
            result['logits'] = torch.zeros_like(result['logits'])
            if kind == 'digest_echo':
                result['received_digest'] = kwargs.get('assembly_digest')
            return result
        with patch.object(handle['receiver'], 'forward', side_effect=bad), torch.no_grad():
            a = sem.port.forward(handle, {'input_ids': ids}, memory)['logits']
            b = sem.port.forward(handle, {'input_ids': ids}, native[docs[1]['key']]['v1'])['logits']
            negatives[kind] = not (float((a[:, answer_position, answer_slice]-b[:, answer_position, answer_slice]).abs().max()) > max(1e-7, 10*batch_noise))
    checks.update({k+'_receiver_rejected': v for k, v in negatives.items()})
    return {'checks': checks, 'pass': all(checks.values()), 'cache_max_error': delta,
            'repeat_noise': repeat_noise, 'batch_noise': batch_noise, 'answer_slice_effect': effect,
            'cache_tolerance': cache_tolerance, 'cache_criterion': 'argmax equal and allclose at max(.01, 4*dtype epsilon)'}


def require_smoke(handle, cfg, out):
    receipt = json.loads((out / 'smoke.json').read_text())
    if (receipt.get('status') != 'PASS' or receipt.get('bindings') != bindings(cfg)
            or receipt.get('stem_sha256') != sem.state_hash(handle['receiver'].stem)
            or receipt.get('connector_initial_sha256') != sem.state_hash(handle['connector'])):
        raise ValueError('failed/stale smoke blocks fit')
    tiny = handle['stem_info'].get('scope') == 'tiny-random'
    if receipt.get('mechanics_only') is not tiny or receipt.get('controls', {}).get('pass') is not True or not all(receipt['controls']['checks'].values()):
        raise ValueError('missing/failed receiving controls block fit')
    if receipt.get('placement_sha256') != sem.file_hash(out/'placement.json'):
        raise ValueError('placement receipt mismatch blocks fit')
    measured = json.loads((out/'placement.json').read_text())
    if not tiny and (receipt.get('competence', {}).get('pass') is not True or measured.get('status') != 'PASS' or receipt.get('registration_frozen') is not True):
        raise ValueError('competence, placement and frozen registration required before real fit')
    for name in ('smoke_cases', 'fit_cases'):
        if receipt.get(name+'_sha256') != sem.file_hash(out / (name+'.json')):
            raise ValueError('case hash mismatch blocks fit')
    return receipt


def fit(handle, cfg, out):
    smoke = require_smoke(handle, cfg, out)
    writer_before = copy.deepcopy(handle.get('_writer_costs', {}))
    docs = json.loads((out / 'fit_cases.json').read_text())
    settings = cfg['semantic']
    effective = settings.get('batch_size', 8)
    micro = settings.get('microbatch', 1)
    if micro != 1 or effective != 8:
        raise ValueError('resident recipe requires microbatch 1 / effective batch 8')
    started = time.monotonic()
    checkpoint(handle, cfg, out, 0, smoke['fit_cases_sha256'], [])
    native = sem.states(handle, docs)
    ingestion_seconds = time.monotonic()-started
    params = list(handle['connector'].parameters())
    optimizer = torch.optim.Adam(params, lr=settings['lr'])
    examples = [(d['key'], r) for d in docs for r in d['reads']]
    rng = random.Random(cfg['seed'])
    losses = []
    journal = Journal(out, 'fit_losses', {'bindings': bindings(cfg), 'cases_sha256': smoke['fit_cases_sha256']})
    for step in range(settings['steps']):
        tick(handle, 'optimizer_step_start')
        if time.monotonic()-started >= settings['max_fit_seconds']:
            raise TimeoutError('fit deadline; partial fit cannot become COMPLETE')
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.
        for key, read in rng.sample(examples, effective):
            tick(handle, 'optimizer_microbatch')
            inputs = sem.answer_batch(handle, [(read['query']['text'], read['answer'])])
            loss = sem.port.forward(handle, inputs, native[key][read['snapshot']])['loss']
            if not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError('invalid answer-only production loss')
            (loss / effective).backward()
            accumulated += float(loss.detach()) / effective
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params) or not any(p.grad.abs().sum() > 0 for p in params):
            raise ValueError('invalid connector gradients')
        if any(p.grad is not None or p.requires_grad for p in handle['receiver'].stem.parameters()):
            raise ValueError('stem must remain frozen')
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        losses.append(accumulated)
        checkpoint(handle, cfg, out, step+1, smoke['fit_cases_sha256'], losses)
        journal.append({'step': step+1, 'loss': accumulated, 'status': 'COMPLETED_STEP'})
        headroom(handle, out, 'fit_step')
    journal.publish('COMPLETE_DIAGNOSTIC')
    if not losses or smoke['stem_sha256'] != sem.state_hash(handle['receiver'].stem):
        raise ValueError('empty fit or changed stem')
    from safetensors.torch import load_file
    checkpoint_path = out / 'connector.safetensors'
    handle['connector'].load_state_dict(load_file(str(checkpoint_path)), strict=True)
    return {'status': 'COMPLETE', 'steps': len(losses), 'losses': losses,
            'microbatch': micro, 'effective_batch': effective,
            'writer_costs': {k: v-writer_before.get(k, 0) for k, v in handle.get('_writer_costs', {}).items()},
            'ingestion_revision_history_seconds': ingestion_seconds,
            'fit_seconds': time.monotonic()-started, 'checkpoint_sha256': sem.file_hash(checkpoint_path),
            'smoke_receipt_sha256': sem.file_hash(out/'smoke.json')}


def eval_dev(handle, cfg, dev, initial, out):
    writer_before = copy.deepcopy(handle.get('_writer_costs', {}))
    started = clock(handle)
    index = {doc['key']: {read['op']: view(doc, read) for read in doc['reads']} for doc in dev}
    write(out/'dictionary_index.json', index)
    index_build_seconds = clock(handle)-started
    started = clock(handle)
    index = json.loads((out/'dictionary_index.json').read_text(encoding='utf-8'))
    index_load_seconds = clock(handle)-started
    index_retrieval_seconds = 0.
    journal = Journal(out, 'dev_outputs', {'bindings': bindings(cfg), 'cases_sha256': sem.digest(dev),
                                          'expected_rows_by_arm': {a:128 if a=='donor' else 768 for a in evaluate.REQUIRED_ARMS}})
    started = clock(handle)
    native = sem.states(handle, dev)
    ingest_seconds = time.monotonic()-started
    trained = copy.deepcopy(handle['connector'].state_dict())
    by_key = {d['key']: d for d in dev}
    rows, truth, donor_truth = [], {}, {}
    for doc in dev:
        for read in doc['reads']:
            truth[doc['key'], read['op']] = read['answer']
    arm_seconds = {}
    for arm in evaluate.REQUIRED_ARMS:
        tick(handle, 'evaluation_arm_'+arm)
        handle['connector'].load_state_dict(initial if arm == 'untrained' else trained)
        start = clock(handle)
        for doc in dev:
            partner = doc['key'][:-1] + ('b' if doc['key'][-1] == 'a' else 'a')
            for read in doc['reads']:
                if arm == 'donor' and read['op'] != 'late_target':
                    continue
                memory = native[doc['key']][read['snapshot']] if arm in ('candidate', 'untrained') else None
                tick(handle, 'evaluation_row_start')
                lookup_start = clock(handle)
                record = index[doc['key']][read['op']]
                index_retrieval_seconds += clock(handle)-lookup_start
                evidence = record['text'] if arm == 'storage' else None
                if arm == 'donor':
                    memory = native[partner]['v1']
                    assert memory['meta'] == native[doc['key']]['v1']['meta']
                    donor_truth[doc['key'], read['op']] = view(by_key[partner], read)['value']
                if arm == 'dictionary':
                    result = {'generated': record['value'], 'terminated': True,
                              'finish_cause': 'exact_lookup', 'budget_exhausted': False}
                else:
                    result = sem.generate(handle, read['query']['text'], memory, evidence,
                                          budget=cfg['semantic'].get('budget', 8))
                row = dict(result, arm=arm, doc=doc['key'], op=read['op'], status='COMPLETED_ROW')
                journal.append(row)
                rows.append(row)
                if len(rows) % 32 == 0:
                    headroom(handle, out, 'evaluation')
        arm_seconds[arm] = clock(handle)-start
        print(json.dumps({'completed_arm': arm, 'rows': sum(r['arm'] == arm for r in rows)}), flush=True)
    handle['connector'].load_state_dict(trained)
    evaluate.validate_table(rows)
    for arm in evaluate.REQUIRED_ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        expected_keys = set(donor_truth) if arm == 'donor' else set(truth)
        if {(r['doc'], r['op']) for r in selected} != expected_keys or not all(evaluate._decode_evidence(r) for r in selected):
            raise ValueError('invalid manifest/decode roster: '+arm)
    summary = {arm: evaluate.score_arm(rows, arm, donor_truth if arm == 'donor' else truth)
               for arm in evaluate.REQUIRED_ARMS}
    for arm in evaluate.REQUIRED_ARMS:
        summary[arm]['capped'] = sum(r.get('finish_cause') == 'budget' for r in rows if r['arm'] == arm)
    intervals = {arm: evaluate.bootstrap_deficit(rows, 'candidate', arm, truth)
                 for arm in ('storage', 'omission', 'untrained', 'dictionary')}
    docs_ok, pairs_ok = evaluate.complete_units(rows, 'candidate', truth)
    c = summary['candidate']
    gates = {'dictionary_exact': summary['dictionary']['correct'] == 768,
             'candidate_total': c['correct'] >= 692,
             'each_operation': all(h >= 116 and n == 128 for h, n in c['by_op'].values()),
             'complete_documents': len(docs_ok) >= 103, 'complete_pairs': len(pairs_ok) >= 52,
             'candidate_termination': c['terminated'] >= 761,
             'donor_follow': summary['donor']['correct'] >= 116,
             'storage_termination': summary['storage']['terminated'] >= .95*768,
             'storage_noninferiority': intervals['storage']['lcb95_points'] >= -5}
    for arm in ('omission', 'untrained'):
        gates[arm+'_termination'] = summary[arm]['terminated'] >= .95*768
        gates['contribution_'+arm] = (gates[arm+'_termination'] and (c['correct']-summary[arm]['correct'])/768 >= .2
                                     and intervals[arm]['lcb95_points'] > 0)
    for arm in evaluate.REQUIRED_ARMS:
        if arm != 'dictionary':
            gates[arm+'_nonconstant'] = summary[arm]['distinct_generated'] > 1
    gates['storage_competent_above_constant'] = summary['storage']['correct']/768 >= summary['storage']['majority_constant_baseline'] + .2
    # DEV has familiar phrasing only. Do not invent a novel stratum or promote
    # an in-process packet into restarted-reader/final acceptance.
    journal.publish('COMPLETE_ROSTER_DIAGNOSTIC')
    outputs_hash = write(out / 'dev_outputs.json', rows)
    native_bytes = sum(len(sem.env.serialize_envelope(m, expected_width=handle['stem_info']['D_memory'],
                       expected_schema=sem.env.WIDTH_SCHEMA_VERSION).encode()) for snapshots in native.values() for m in snapshots.values())
    dictionary_bytes = (out/'dictionary_index.json').stat().st_size
    dictionary_serialization_seconds = index_build_seconds + index_load_seconds
    fit_receipt = json.loads((out/'fit.json').read_text())
    smoke_receipt = json.loads((out/'smoke.json').read_text())
    costs = {'dev_ingestion_revision_history_seconds': ingest_seconds,
             'writer_costs': {k: v-writer_before.get(k, 0) for k, v in handle.get('_writer_costs', {}).items()},
             'fit_including_ingestion_seconds': fit_receipt['fit_seconds'],
             'smoke_seconds': smoke_receipt['seconds'],
             'dictionary_serialization_seconds': dictionary_serialization_seconds,
             'persisted_index_build_seconds': index_build_seconds, 'persisted_index_load_seconds': index_load_seconds,
             'persisted_index_retrieval_seconds': index_retrieval_seconds,
             'stem_parameter_bytes_common_to_neural_arms': sum(p.numel()*p.element_size() for p in handle['receiver'].stem.parameters()),
             'connector_parameter_bytes': sum(p.numel()*p.element_size() for p in handle['connector'].parameters()),
             'reuse_time_projection_seconds_not_repeated_measurements': {str(k): {
                 'candidate': fit_receipt['fit_seconds'] + smoke_receipt['seconds'] + ingest_seconds + k*arm_seconds['candidate'],
                 'storage_generated': dictionary_serialization_seconds + k*arm_seconds['storage'],
                 'dictionary': dictionary_serialization_seconds + k*arm_seconds['dictionary']} for k in (1,5,20,100)}, 'readout_seconds_by_arm': arm_seconds,
             'retained_native_envelope_bytes': native_bytes, 'retained_dictionary_json_bytes': dictionary_bytes,
             'connector_checkpoint_bytes': (out/'connector.safetensors').stat().st_size,
             'source_case_file_bytes': (out/'fit_cases.json').stat().st_size + (out/'smoke_cases.json').stat().st_size,
             'output_bytes': (out/'dev_outputs.json').stat().st_size,
             'reuses_to_price': [1, 5, 20, 100],
             'complete_cost_status': 'INCOMPLETE: device energy, transient KV, complete host accounting and quality-matched reuse economics not measured; no efficiency claim'}
    return {'status': 'COMPLETE_DEV_DIAGNOSTIC', 'summary': summary, 'gates': gates,
            'dev_numeric_gates_pass': all(gates.values()), 'intervals': intervals,
            'cluster_units': 64, 'bootstrap_resamples': 10000, 'outputs_sha256': outputs_hash,
            'bet_failure_condition_met_numerically': not all(gates['contribution_'+a] for a in ('omission', 'untrained')) or c['correct'] <= summary['dictionary']['correct'],
            'all_pass': False, 'test': 'UNTOUCHED: frozen complete DEV gates, restarted reader and manual review not satisfied',
            'novel_stratum': 'not present in DEV',
            'manual_review_examples': [dict(next(r for r in rows if r['arm']=='candidate' and r['doc']==str(i)+'a' and r['op']==data.READ_OPS[i%6]),
                expected=truth[str(i)+'a', data.READ_OPS[i%6]], outputs_sha256=outputs_hash) for i in range(12)], 'manual_review': 'PENDING coordinator review of fixed outputs',
            'restarted_reader': 'UNRUN here; first real DEV arms intentionally in-process', 'costs': costs}


def session(cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    terminal = {'status': 'FAIL', 'phase': 'load', 'bindings': bindings(cfg), 'model_loads': 0}
    handle, progress = None, None
    try:
        progress = Progress(cfg, out)
        progress('preflight_start')
        preflight(cfg)
        load_started = time.monotonic()
        handle = load(cfg, progress)
        handle['_progress'] = progress
        # Exact model forward boundaries also cover ingestion, smoke and each decode call.
        handle['receiver'].stem.register_forward_pre_hook(lambda *_: progress('stem_forward_start'))
        handle['receiver'].stem.register_forward_hook(lambda *_: progress('stem_forward_complete'))
        progress('load_complete')
        headroom(handle, out, 'load')
        terminal['load_seconds'] = time.monotonic()-load_started
        terminal['model_loads'] = 1
        tiny = handle['stem_info'].get('scope') == 'tiny-random'
        terminal['scope'] = 'RANDOM stand-in mechanics only' if tiny else 'real E2B, in-process DEV'
        terminal['phase'] = progress.phase = 'smoke'
        progress('smoke_start')
        smoke_started = time.monotonic()
        initial = copy.deepcopy(handle['connector'].state_dict())
        settings = cfg['semantic']
        dev = documents('dev', 128, cfg['seed'])
        if not tiny and settings.get('train_documents') != 2048:
            raise ValueError('real resident fit requires 2048 TRAIN documents')
        fit_cases = documents('train', settings.get('train_documents', 2048), cfg['seed'])
        smoke_cases_hash = write(out/'smoke_cases.json', dev)
        fit_cases_hash = write(out/'fit_cases.json', fit_cases)
        checkpoint(handle, cfg, out, 0, fit_cases_hash, [])
        for query in sorted({r['query']['text'] for d in fit_cases+dev for r in d['reads']}):
            answer_ids(handle, query)
            progress('answer_context_validated')
        control = controls(handle, cfg, out)
        competent = competence(handle, dev, cfg)
        measured = headroom(handle, out, 'smoke')
        registration = cfg.get('resident', {})
        frozen = registration.get('generator_sha256') == sem.file_hash(data.__file__) and registration.get('thresholds_frozen') is True
        admitted = control['pass'] and (tiny or (competent['pass'] and measured['status'] == 'PASS' and frozen))
        smoke = {'seconds': time.monotonic()-smoke_started, 'status': 'PASS' if admitted else 'FAIL', 'scope': terminal['scope'],
                 'mechanics_only': tiny, 'competence_required_for_fit': not tiny,
                 'controls': control, 'competence': competent, 'registration_frozen': frozen,
                 'placement_sha256': sem.file_hash(out/'placement.json'),
                 'smoke_cases_sha256': smoke_cases_hash, 'fit_cases_sha256': fit_cases_hash,
                 'bindings': bindings(cfg), 'stem_info': handle['stem_info'],
                 'stem_sha256': sem.state_hash(handle['receiver'].stem),
                 'connector_initial_sha256': sem.state_hash(handle['connector'])}
        write(out/'smoke.json', smoke)
        if not admitted:
            write(out/'fit.json', {'status': 'BLOCKED', 'reason': 'failed smoke'})
            write(out/'dev_eval.json', {'status': 'BLOCKED', 'reason': 'failed smoke'})
            raise ValueError('failed smoke blocks fit; direct prompt/budget remedy must be separately frozen')
        if not tiny:
            seconds_per_decode = competent['decode_seconds']/competent['decode_rows']
            projection = seconds_per_decode*3200
            write(out/'decode_budget.json', {'status':'DIAGNOSTIC_ESTIMATE', 'measured_seconds_per_row':seconds_per_decode,
                  'dev_decode_projection_seconds':projection, 'remaining_work_seconds':progress.end-time.time(),
                  'limitations':'direct-evidence panel only; prefix/ingestion costs can be larger'})
            if projection+settings['max_fit_seconds'] >= progress.end-time.time():
                raise TimeoutError('measured decode time does not fit remaining absolute work budget')
        terminal['phase'] = progress.phase = 'fit'
        progress('fit_start')
        fit_receipt = fit(handle, cfg, out)
        write(out/'fit.json', fit_receipt)
        headroom(handle, out, 'fit_complete')
        terminal['phase'] = progress.phase = 'evaluate_dev'
        progress('evaluation_start')
        evaluated = eval_dev(handle, cfg, dev, initial, out)
        if terminal['bindings'] != bindings(cfg):
            raise ValueError('code/config changed during resident session')
        evaluated['scope'] = terminal['scope']
        evaluated['fit_receipt_sha256'] = sem.file_hash(out/'fit.json')
        headroom(handle, out, 'evaluation_complete')
        write(out/'dev_eval.json', evaluated)
        terminal.update(status='COMPLETE_MECHANICS_ONLY' if tiny else 'COMPLETE_DEV_DIAGNOSTIC', phase='complete',
                        scientific_acceptance='INCOMPLETE', dev_numeric_gates_pass=evaluated['dev_numeric_gates_pass'])
        return terminal
    except Exception as exc:
        terminal['error'] = repr(exc)
        for phase, name in (('smoke', 'smoke.json'), ('fit', 'fit.json'), ('evaluate_dev', 'dev_eval.json')):
            if not (out/name).exists():
                write(out/name, {'status': 'FAIL' if terminal['phase'] == phase else 'BLOCKED', 'reason': repr(exc)})
        raise
    finally:
        if not (out/'placement.json').exists():
            write(out/'placement.json', {'status':'BLOCKED', 'reason':'no admitted resident placement measurement'})
        if progress:
            terminal['progress_sequence'] = progress.count
            terminal['work_stop_at'] = progress.end
        # Preserve existing incomplete row/checkpoint manifests; never synthesize PASS.
        terminal['partial_artifacts'] = {p.name: {'sha256': sem.file_hash(p), 'bytes': p.stat().st_size}
            for p in out.iterdir() if p.suffix in ('.jsonl', '.safetensors')}
        terminal['receipt_hashes'] = {name: sem.file_hash(out/name) for name in
            ('placement.json', 'smoke.json', 'fit.json', 'dev_eval.json') if (out/name).exists()}
        write(out/'terminal.json', terminal)


def check(name):
    from unittest.mock import patch
    import socket
    cfg = json.loads((Path(__file__).parent/'contract.json').read_text())
    cfg['semantic'].update(steps=1, train_documents=16, batch_size=8, microbatch=1, max_fit_seconds=120)
    def forbidden(*args, **kwargs):
        raise AssertionError('offline CPU acceptance forbids external effects')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(torch.cuda, '_lazy_init', forbidden), \
            patch('transformers.PreTrainedModel.from_pretrained', forbidden), \
            patch('transformers.AutoTokenizer.from_pretrained', forbidden):
        if name == 'dims':
            recipe = json.loads((Path(__file__).parent.parents[1]/'cloud/configs/allin_e2b_prototype_recipe.json').read_text())
            rc = real_config(recipe)
            assert rc['width'] == rc['memory_dim'] == 1536 and rc['n_layers'] == 35
            cfg['semantic']['stem']['dimensions'] = {'hidden_size': rc['width']//16, 'num_hidden_layers': 35,
                'hidden_size_per_layer_input': 16, 'num_kv_shared_layers': 20,
                'num_attention_heads': 8, 'num_key_value_heads': 1, 'head_dim': 256,
                'attention_k_eq_v': True,
                'layer_types': ['full_attention' if (i+1)%6==0 or i==34 else 'sliding_attention' for i in range(35)]}
            handle = load(cfg)
            with tempfile.TemporaryDirectory() as tmp:
                receipt = controls(handle, cfg, Path(tmp))
                assert receipt['pass'], receipt
                handle['receiver'].stem.bfloat16()
                handle['connector'].register_forward_hook(lambda module, args, output: output.to(torch.bfloat16))
                mixed = controls(handle, cfg, Path(tmp))
                assert mixed['pass'], mixed
            assert handle['stem_info']['D_stem'] == 96 and handle['stem_info']['layers'] == 35
            return {'status': 'PASS_DIMS_MECHANICS_ONLY', 'scaled_dimensions': cfg['semantic']['stem']['dimensions'],
                    'real_config_dimensions': {'hidden_size': 1536, 'layers': 35, 'PLE': 256, 'shared_KV_layers': 20},
                    'checks': receipt['checks'], 'bf16_stem_fp32_connector_checks': mixed['checks'],
                    'cache_tolerances': {'fp32':receipt['cache_tolerance'], 'bf16':mixed['cache_tolerance']},
                    'layer_pattern_scope': 'library default 5 sliding:1 full plus final full; exact pinned operator pattern not present locally'}
        if name != 'stand-in-end-to-end':
            raise ValueError('unknown check')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'session'
            with patch.object(sem, 'load_handle', wraps=sem.load_handle) as loader:
                result = session(cfg, out)
                assert loader.call_count == 1
            evaluation = json.loads((out/'dev_eval.json').read_text())
            result['dev_summary'] = evaluation['summary']
            result['gates'] = evaluation['gates']
            result['costs'] = evaluation['costs']
            handle = load(cfg)
            # Every negative calls the actual fit gate. There is no second fit.
            smoke = json.loads((out/'smoke.json').read_text())
            smoke['status'] = 'FAIL'
            write(out/'smoke.json', smoke)
            for attack in ('failed_smoke', 'case_tamper'):
                if attack == 'case_tamper':
                    smoke['status'] = 'PASS'
                    write(out/'smoke.json', smoke)
                    (out/'fit_cases.json').write_text('[]')
                try:
                    fit(handle, cfg, out)
                except ValueError:
                    pass
                else:
                    raise AssertionError(attack+' admitted fitting')
            missing = copy.deepcopy(cfg)
            missing['semantic']['stem'] = {'kind': 'staged-real', 'receiver_config': {'model_local': str(Path(tmp)/'missing')}}
            try:
                verify_assets(missing['semantic']['stem']['receiver_config'])
            except ValueError as exc:
                assert 'missing' in str(exc)
            else:
                raise AssertionError('missing mirror admitted')
            root = Path(tmp)/'synthetic_manifest'
            root.mkdir()
            (root/'_COMPLETE').write_text('synthetic')
            (root/'_manifest_sha256.txt').write_text('bad manifest')
            rc = {'model_local': str(root.resolve()), 'model': 'google/gemma-4-E2B',
                  'allow_hf_download': False, 'revision': 'a'*40, 'tokenizer_revision': 'a'*40,
                  'weights_manifest_sha256': '0'*64}
            try:
                verify_assets(rc)
            except ValueError as exc:
                assert 'manifest pin mismatch' in str(exc)
            else:
                raise AssertionError('wrong manifest admitted')
            result['negatives'] = {'failed_smoke_blocks_fit': True, 'case_tamper_blocks_fit': True,
                                   'missing_weights_fail_closed': True, 'wrong_manifest_rejected': True}
            assert (out/'heartbeat_work').exists()
            beat = json.loads((out/'heartbeat_work').read_text())
            assert beat['stage']=='work' and beat['sequence']>0
            result['progress_files'] = ['heartbeat_work']
            # New sessions are finite, fresh, fault injected; no synthetic final PASS.
            for fault in ('completed_arm', 'optimizer_step'):
                fault_out = Path(tmp)/fault
                if fault == 'completed_arm':
                    calls = [0]
                    def crash_generate(*args, **kwargs):
                        calls[0] += 1
                        # Smoke has 64 direct evidence cases, then 768 candidate rows.
                        if calls[0] == 64+768+1:
                            raise RuntimeError('injected after completed candidate arm')
                        return {'generated': 'wrong'+str(calls[0]%2), 'terminated':False,
                                'tokens':[2]*8, 'token_count':8, 'max_new_tokens':8,
                                'eos_token_id':1, 'finish_cause':'budget', 'budget_exhausted':True}
                    context = patch.object(sem, 'generate', side_effect=crash_generate)
                else:
                    original_headroom = headroom
                    def fail_after_step(h, directory, event):
                        if event == 'fit_step':
                            raise RuntimeError('injected after one optimizer step')
                        return original_headroom(h, directory, event)
                    context = patch(__name__+'.headroom', side_effect=fail_after_step)
                from contextlib import nullcontext
                decode_context = patch.object(sem, 'generate', return_value={'generated':'wrong', 'terminated':False,
                    'tokens':[2]*8, 'token_count':8,'max_new_tokens':8,'eos_token_id':1,'finish_cause':'budget','budget_exhausted':True}) if fault=='optimizer_step' else nullcontext()
                with context, decode_context:
                    try:
                        session(cfg, fault_out)
                    except RuntimeError as exc:
                        assert 'injected' in str(exc)
                    else:
                        raise AssertionError('fault did not fail')
                failed = json.loads((fault_out/'terminal.json').read_text())
                assert failed['status']=='FAIL'
                if fault == 'completed_arm':
                    manifest = json.loads((fault_out/'dev_outputs_manifest.json').read_text())
                    retained = fault_out/'dev_outputs.jsonl'
                    assert manifest['rows']==768 and manifest['status']=='INCOMPLETE' and manifest['all_pass'] is False
                    assert manifest['sha256']==sem.file_hash(retained) and manifest['bytes']==retained.stat().st_size>0
                else:
                    manifest = json.loads((fault_out/'connector_manifest.json').read_text())
                    assert manifest['step']==1 and manifest['sha256']==sem.file_hash(fault_out/'connector.safetensors')
                    assert (fault_out/'fit_losses.jsonl').stat().st_size>0 and manifest['all_pass'] is False
                result['negatives'][fault+'_retained_bytes_fail_closed'] = True
            # Actual DEV verdict: correct candidate/storage, nonterminating baselines.
            gate_out = Path(tmp)/'gate_negative'
            gate_out.mkdir()
            write(gate_out/'fit.json', {'fit_seconds':0})
            write(gate_out/'smoke.json', {'seconds':0})
            (gate_out/'connector.safetensors').write_bytes(b'diagnostic')
            (gate_out/'fit_cases.json').write_text('[]')
            (gate_out/'smoke_cases.json').write_text('[]')
            dev = documents('dev', 128, cfg['seed'])
            truth = {(d['key'],r['op']): r['answer'] for d in dev for r in d['reads']}
            donors = {(d['key'], 'late_target'): view(next(x for x in dev if x['key']==d['key'][:-1]+('b' if d['key'][-1]=='a' else 'a')), d['reads'][0])['value'] for d in dev}
            roster = [(a,d,r) for a in evaluate.REQUIRED_ARMS for d in dev for r in d['reads'] if a!='dictionary' and (a!='donor' or r['op']=='late_target')]
            def fake_rows():
                for arm, doc, read in roster:
                    bad = arm in ('omission', 'untrained')
                    answer = 'wrong'+str(int(doc['key'][:-1])%2) if bad else (donors[doc['key'],read['op']] if arm=='donor' else read['answer'])
                    yield {'generated':answer, 'terminated':not bad, 'tokens':[2]*8 if bad else [2,1],
                           'token_count':8 if bad else 2,'max_new_tokens':8,'eos_token_id':1,
                           'finish_cause':'budget' if bad else 'eos','budget_exhausted':bad}
            generated = iter(fake_rows())
            with patch.object(sem, 'generate', side_effect=lambda *a,**k: next(generated)):
                negative = eval_dev(handle, cfg, dev, copy.deepcopy(handle['connector'].state_dict()), gate_out)
            assert not negative['dev_numeric_gates_pass']
            assert all(negative['gates']['contribution_'+a] is False for a in ('omission','untrained'))
            full_rows = json.loads((gate_out/'dev_outputs.json').read_text())
            final = evaluate.evaluate_all(full_rows, truth, {str(i) for i in range(32)}, donors)
            assert not final['scoring_pass'] and all(final['gates'][a+'_deficit_ge_20pts_cluster_excl_0'] is False for a in ('omission','untrained'))
            result['negatives']['failed_baseline_termination_blocks_each_contribution_DEV_and_final'] = True
            terminating = copy.deepcopy(full_rows)
            for row in terminating:
                if row['arm'] in ('omission','untrained'):
                    row.update(tokens=[2,1], token_count=2, terminated=True,
                               finish_cause='eos', budget_exhausted=False)
            positive = evaluate.evaluate_all(terminating, truth, {str(i) for i in range(32)}, donors)
            assert positive['scoring_pass'] and positive['all_pass'] is False
            result['negatives']['terminating_low_accuracy_controls_still_admitted_for_scoring_only'] = True
            # Deadline and actual loader observer probes: temporary synthetic bytes only.
            probe_root = Path(tmp)/'progress_probe'
            probe_root.mkdir()
            probe_cfg = copy.deepcopy(cfg)
            probe_cfg['dispatch'] = {'work_stop_at':1000.,'shutdown_at':1900.}
            with patch.object(time, 'time', return_value=999.):
                probe_progress = Progress(probe_cfg, probe_root)
                probe_progress('actual_boundary')
                with loading_progress(probe_progress):
                    synthetic = probe_root/'synthetic.txt'
                    synthetic.write_bytes(b'synthetic control')
                    assert receiver_api._sha256(synthetic)==sem.file_hash(synthetic)
                    from transformers import core_model_loading
                    assert list(core_model_loading.tqdm([1,2], disable=True))==[1,2]
            assert probe_progress.count >= 6
            with patch.object(time, 'time', return_value=1000.):
                try:
                    probe_progress('expired')
                except TimeoutError:
                    pass
                else:
                    raise AssertionError('expired operation admitted')
            assert json.loads((probe_root/'heartbeat_work').read_text())['epoch']==999.
            result['negatives']['deadline_and_real_hash_tensor_progress_observers'] = True
            # Inject ingestion failure at fit entry after one actual document.
            ingest_out = Path(tmp)/'interrupted_ingestion'
            original_states = sem.states
            def interrupted_states(h, docs):
                if h.get('_progress') and h['_progress'].phase=='fit':
                    original_states(h, docs[:1])
                    raise RuntimeError('injected ingestion interruption')
                return original_states(h, docs)
            # Decode competence here is a raising-mock-safe diagnostic, not model quality.
            synthetic_decode = {'generated':'wrong', 'terminated':False,'tokens':[2]*8,
                   'token_count':8,'max_new_tokens':8,'eos_token_id':1,'finish_cause':'budget','budget_exhausted':True}
            with patch.object(sem, 'states', side_effect=interrupted_states), patch.object(sem, 'generate', return_value=synthetic_decode):
                try:
                    session(cfg, ingest_out)
                except RuntimeError as exc:
                    assert 'ingestion' in str(exc)
                else:
                    raise AssertionError('ingestion interruption admitted')
            assert json.loads((ingest_out/'terminal.json').read_text())['status']=='FAIL'
            assert (ingest_out/'connector_initial.safetensors').stat().st_size>0
            assert json.loads((ingest_out/'fit.json').read_text())['status']=='FAIL'
            result['negatives']['interrupted_ingestion_retains_initial_checkpoint'] = True
            result['status'] = 'PASS_STAND_IN_MECHANICS_ONLY'
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    parser.add_argument('--config')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.check and args.check not in ('dims', 'stand-in-end-to-end'):
        parser.error('unknown check: '+args.check)
    if not args.check and (not args.config or not args.out):
        parser.error('--config and --out required for resident session')
    try:
        result = check(args.check) if args.check else session(json.loads(Path(args.config).read_text()), args.out)
    except Exception as exc:
        print(json.dumps({'status': 'FAIL', 'error': repr(exc)}), flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
