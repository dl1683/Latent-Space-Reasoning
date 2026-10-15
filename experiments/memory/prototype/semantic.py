"""Production-path construction slice. Random tiny Gemma results are mechanics only.

The existing runner/trainer are concurrently owned; run_phase is the integration
API for their owner. No pretrained loading is possible in the offline CLI.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from . import data, writer
from experiments.cloud.lsr_jobs import s0_envelope as env, s0_memory_port as port
from experiments.cloud.lsr_jobs.gemma4_receiver import Gemma4Receiver, MemoryResampler, ResamplerConfig


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(cfg):
    here = Path(__file__).resolve().parent
    jobs = here.parents[1] / 'cloud' / 'lsr_jobs'
    paths = [here / n for n in ('semantic.py', 'writer.py', 'data.py', 'reader.py')]
    paths += [jobs / n for n in ('s0_memory_port.py', 's0_envelope.py', 'gemma4_receiver.py')]
    return {'config': digest(cfg), 'code': {str(p): file_hash(p) for p in paths}}


def load_handle(cfg):
    """Config-driven stem; staged real models must use the production loader."""
    stem_cfg = cfg['semantic']['stem']
    if stem_cfg['kind'] != 'tiny-random':
        if cfg.get('offline', True):
            raise ValueError('real stem requires cloud operator; offline CLI refuses weights')
        from experiments.cloud.lsr_jobs.gemma4_receiver import load_receiver
        return load_receiver(stem_cfg['receiver_config'])
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    # Fixed vocabulary includes all split names but no bindings or examples.
    lexicon = set(data.WORDS + data.SLOTS)
    for names in data.NAME_POOLS.values():
        lexicon.update(names)
    templates = data.TEMPLATES_FAMILIAR + data.TEMPLATES_NOVEL + data.QUESTIONS_FAMILIAR + data.QUESTIONS_NOVEL
    import re
    lexicon = set(re.findall(r'\w+|[^\w\s]', ' '.join(sorted(lexicon) + templates + ['answer evidence'])))
    tokens = ['<pad>', '<eos>', '<unk>'] + sorted(lexicon - {'<pad>', '<eos>', '<unk>'})
    vocab = {word: i for i, word in enumerate(tokens)}
    core = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    core.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    torch.manual_seed(stem_cfg['seed'])
    dims = stem_cfg.get('dimensions', {})
    width = dims.get('hidden_size', 64)
    layers = dims.get('num_hidden_layers', 2)
    model_cfg = Gemma4TextConfig(vocab_size=len(vocab), hidden_size=width,
        intermediate_size=2*width, num_hidden_layers=layers, num_attention_heads=dims.get('num_attention_heads',4),
        num_key_value_heads=dims.get('num_key_value_heads',2), head_dim=dims.get('head_dim',16), global_head_dim=dims.get('head_dim',16),
        num_global_key_value_heads=dims.get('num_key_value_heads',2), attention_k_eq_v=dims.get('attention_k_eq_v',False),
        num_kv_shared_layers=dims.get('num_kv_shared_layers', 0), hidden_size_per_layer_input=dims.get('hidden_size_per_layer_input', 0),
        vocab_size_per_layer_input=len(vocab), max_position_embeddings=256,
        sliding_window=64, layer_types=dims.get('layer_types', ['sliding_attention' if i % 2 == 0 and i != layers-1 else 'full_attention' for i in range(layers)]),
        pad_token_id=0, eos_token_id=1)
    model_cfg._attn_implementation = 'eager'
    with torch.device('cpu'):
        stem = Gemma4ForCausalLM(model_cfg).float().eval()
        connector = MemoryResampler(ResamplerConfig(width, width, attention_dim=64, num_heads=4)).float()
    receiver = Gemma4Receiver(stem, connector).eval()
    return {'tok': tok, 'receiver': receiver, 'connector': connector,
            'stem_info': {'D_memory': width, 'D_stem': width, 'layers': layers, 'scope': 'tiny-random',
                          'seed': stem_cfg['seed'], 'vocabulary_sha256': digest(vocab)}}


def slice_documents(split, count, seed):
    """Independent TRAIN/DEV construction fixture; never constructs TEST truth.

    Uses the current record/question contract without the concurrently edited
    build_corpus balance sampler. No binding conditioning on outcomes.
    """
    if split not in ('train', 'dev') or count % 2:
        raise ValueError('only even-sized train/dev slices admitted')
    rng = random.Random(f'semantic-fix1:{seed}:{split}')
    docs = []
    for pair in range(count // 2):
        name = data.NAME_POOLS[split][pair]
        address = rng.randrange(4)
        a = rng.sample(data.WORDS, 5)
        b = rng.sample(data.WORDS, 5)
        while any(x == y for x, y in zip(a, b)):
            b = rng.sample(data.WORDS, 5)
        for member, values in zip('ab', (a, b)):
            doc, reads = data._make_doc(name, values, False, rng, address)
            docs.append({'key': f'{pair}{member}', 'document': doc, 'reads': reads})
    return docs


def prompt(question, evidence=None):
    base = '' if evidence is None else f'evidence: {evidence}\n'
    return base + question + '\nanswer:'


def answer_batch(handle, examples):
    tok = handle['tok']
    sequences, labels = [], []
    for question, answer in examples:
        prefix = prompt(question)
        # One joint tokenization; prefix stability is asserted instead of assumed.
        ids = tok(prefix + ' ' + answer, add_special_tokens=False)['input_ids']
        before = tok(prefix, add_special_tokens=False)['input_ids']
        if ids[:len(before)] != before or len(ids) <= len(before):
            raise ValueError('tokenizer merges the answer boundary; offset-aware masking required')
        ids += [tok.eos_token_id]
        sequences.append(ids)
        labels.append([-100] * len(before) + ids[len(before):])
    length = max(map(len, sequences))
    return {'input_ids': torch.tensor([x + [tok.pad_token_id]*(length-len(x)) for x in sequences]),
            'attention_mask': torch.tensor([[1]*len(x) + [0]*(length-len(x)) for x in sequences]),
            'labels': torch.tensor([x + [-100]*(length-len(x)) for x in labels]), 'use_cache': False}


def states(handle, docs):
    encode = writer.ContextualWriter(handle)
    result = {}
    for row in docs:
        if handle.get('_progress'):
            handle['_progress']('ingestion_document_start')
        doc = row['document']
        old = encode.write(doc['records'])
        before = env.serialize_envelope(old, expected_width=handle['stem_info']['D_memory'], expected_schema=env.WIDTH_SCHEMA_VERSION)
        new = encode.revise(old, doc['revision']['record_id'], doc['revision'])
        changed = doc['revision']['record_id']
        old_rows, new_rows = env.to_torch(old['payload']), env.to_torch(new['payload'])
        keep = [i for i in range(16) if i//4 != changed]
        assert torch.equal(old_rows[keep], new_rows[keep]), 'revision changed unrelated rows'
        assert env.serialize_envelope(old, expected_width=handle['stem_info']['D_memory'], expected_schema=env.WIDTH_SCHEMA_VERSION) == before
        result[row['key']] = {'v1': old, 'v2': new}
        if handle.get('_progress'):
            handle['_progress']('ingestion_document_complete')
    return result


def state_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def smoke(handle, cfg, output):
    """Separate terminal receipt through the exact writer/fit path before fitting."""
    docs = slice_documents('train', 2, cfg['seed'])
    native = states(handle, docs)
    one = docs[0]
    read = one['reads'][0]
    memory = native[one['key']]['v1']
    inputs = answer_batch(handle, [(read['query']['text'], read['answer'])])
    before = state_hash(handle['receiver'].stem)
    calls = []
    hook = handle['connector'].register_forward_hook(lambda m, args, out: calls.append(out.shape))
    result = port.forward(handle, inputs, memory)
    hook.remove()
    loss = result['loss']
    loss.backward()
    grads = [p.grad for p in handle['connector'].parameters() if p.requires_grad]
    checks = {'one_resampling_call': calls == [torch.Size([1, 8, handle['stem_info']['D_stem']])],
              'tensor_loss': isinstance(loss, torch.Tensor) and loss.requires_grad,
              'answer_eos_only': int((inputs['labels'] != -100).sum()) == len(handle['tok'](prompt(read['query']['text']) + ' ' + read['answer'], add_special_tokens=False)['input_ids']) - len(handle['tok'](prompt(read['query']['text']), add_special_tokens=False)['input_ids']) + 1,
              'finite_nonzero_gradients': all(g is not None and torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0,
              'frozen_stem': all(not p.requires_grad and p.grad is None for p in handle['receiver'].stem.parameters()) and state_hash(handle['receiver'].stem) == before}
    handle['receiver'].zero_grad(set_to_none=True)
    with torch.no_grad():
        a = port.forward(handle, inputs, memory)['logits']
        repeat = port.forward(handle, inputs, memory)['logits']
        changed = port.forward(handle, inputs, native[docs[1]['key']]['v1'])['logits']
        none = port.forward(handle, inputs)['logits']
        empty = env.empty_envelope(expected_width=handle['stem_info']['D_memory'], expected_schema=env.WIDTH_SCHEMA_VERSION)
        empty_logits = port.forward(handle, inputs, empty)['logits']
        noise = float((a-repeat).abs().max())
        effect = float((a-changed).abs().max())
        checks['content_effect'] = effect > max(noise*10, 1e-7)
        checks['none_empty_parity'] = torch.equal(none, empty_logits)
    # A disconnected boundary must produce no intended connector gradients.
    detached = copy.deepcopy(memory)
    with torch.no_grad():
        slots = handle['connector'](env.to_torch(detached['payload']).unsqueeze(0).to(next(handle['connector'].parameters())))
    device = handle['receiver'].stem.get_input_embeddings().weight.device
    disconnected = handle['receiver'](inputs['input_ids'].to(device), prefix_states=slots, labels=inputs['labels'].to(device))['loss']
    checks['disconnected_loss_rejected'] = not disconnected.requires_grad
    receipt = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
               'noise': noise, 'effect': effect, 'identities': identities(cfg),
               'stem_sha256': before, 'connector_initial_sha256': state_hash(handle['connector']),
               'scope': 'production receiving mechanics only; actual competence is gated separately by resident smoke'}
    path = Path(output)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return receipt


def generate(handle, query, memory=None, evidence=None, budget=8):
    if handle.get('_progress'):
        handle['_progress']('decode_start')
    ids = handle['tok'](prompt(query, evidence), add_special_tokens=False, return_tensors='pt')['input_ids']
    result = port.generate(handle, {'input_ids': ids}, memory,
                          budget={'max_new_tokens': budget, 'max_total_forward_calls': budget},
                          eos_ids=(handle['tok'].eos_token_id,))
    if handle.get('_progress'):
        handle['_progress']('decode_complete')
    result['generated'] = handle['tok'].decode(result['tokens'], skip_special_tokens=True).strip()
    result['finish_cause'] = 'eos' if result['terminated'] else 'budget'
    result['token_count'] = len(result['tokens'])
    result['max_new_tokens'] = budget
    result['eos_token_id'] = handle['tok'].eos_token_id
    return result


def fit(handle, cfg, docs, native, smoke_path):
    receipt = json.loads(Path(smoke_path).read_text(encoding='utf-8'))
    if receipt['status'] != 'PASS' or receipt['identities'] != identities(cfg) \
            or receipt['stem_sha256'] != state_hash(handle['receiver'].stem) \
            or receipt['connector_initial_sha256'] != state_hash(handle['connector']):
        raise ValueError('missing/failed/stale separate smoke blocks fitting')
    params = list(handle['connector'].parameters())
    optimizer = torch.optim.Adam(params, lr=cfg['semantic']['lr'])
    examples = [(d['key'], r) for d in docs for r in d['reads']]
    rng = random.Random(cfg['seed'])
    started = time.monotonic()
    losses = []
    frozen = state_hash(handle['receiver'].stem)
    for step in range(cfg['semantic']['steps']):
        if time.monotonic()-started >= min(1200, cfg['semantic']['max_fit_seconds']):
            break
        batch = rng.sample(examples, cfg['semantic']['batch_size'])
        inputs = answer_batch(handle, [(r['query']['text'], r['answer']) for _, r in batch])
        memory = [native[k][r['snapshot']] for k, r in batch]
        optimizer.zero_grad(set_to_none=True)
        loss = port.forward(handle, inputs, memory)['loss']
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad or not torch.isfinite(loss):
            raise ValueError('invalid returned production tensor loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params) or not any(float(p.grad.abs().sum()) > 0 for p in params):
            raise ValueError('missing/nonfinite intended gradient')
        if any(p.grad is not None or p.requires_grad for p in handle['receiver'].stem.parameters()):
            raise ValueError('stem gradient violation')
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            print(json.dumps({'step': step, 'loss': losses[-1], 'fit_seconds': time.monotonic()-started}), flush=True)
    if not losses or frozen != state_hash(handle['receiver'].stem):
        raise ValueError('empty fit or mutated stem')
    return {'steps': len(losses), 'first_loss': losses[0], 'last_loss': losses[-1],
            'fit_seconds': time.monotonic()-started, 'frozen_stem_sha256': frozen}


def run_phase(cfg, out, handle=None):
    """Integration adapter for canonical runner; tiny slice is not full acceptance."""
    if cfg.get('_phase') in ('train', 'evaluate'):
        return {'status':'INCOMPLETE', 'prototype_acceptance':'INCOMPLETE',
                'reason':'Only one bounded overfit construction is implemented; train/evaluate never silently refit.'}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    handle = load_handle(cfg) if handle is None else handle
    if handle['stem_info'].get('scope') != 'tiny-random':
        raise ValueError('only bounded random construction slice is implemented here')
    from safetensors.torch import save_file, load_file
    initial = copy.deepcopy(handle['connector'].state_dict())
    smoke_receipt = smoke(handle, cfg, out / 'smoke.json')
    if smoke_receipt['status'] != 'PASS':
        return {'status': 'FAIL', 'smoke': smoke_receipt}
    train = slice_documents('train', 16, cfg['seed'])
    native = states(handle, train)
    fit_result = fit(handle, cfg, train, native, out / 'smoke.json')
    trained = copy.deepcopy(handle['connector'].state_dict())
    train_outputs = []
    for doc in train:
        for read in doc['reads']:
            row = generate(handle, read['query']['text'], native[doc['key']][read['snapshot']])
            train_outputs.append(dict(row, expected=read['answer'], doc=doc['key'], op=read['op']))
    train_correct = sum(r['generated'] == r['expected'] for r in train_outputs)
    save_file({k: v.contiguous() for k, v in trained.items()}, str(out / 'connector.safetensors'),
              metadata={'stem_sha256':state_hash(handle['receiver'].stem),
                        'config_sha256':digest(cfg), 'train_fixture_sha256':digest(train),
                        'model_scope':'tiny-random'})
    # Reload identity check before querying; no source is needed to restore connector.
    handle['connector'].load_state_dict(load_file(str(out / 'connector.safetensors')), strict=True)
    dev = slice_documents('dev', 128, cfg['seed'])
    dev_native = states(handle, dev)
    rows = []
    for arm in ('candidate', 'omission', 'untrained', 'storage', 'donor'):
        handle['connector'].load_state_dict(initial if arm == 'untrained' else trained)
        for doc in dev:
            records = doc['document']['records']
            revised = copy.deepcopy(records)
            replacement = doc['document']['revision']
            revised[replacement['record_id']] = replacement
            partner = doc['key'][:-1] + ('b' if doc['key'][-1] == 'a' else 'a')
            donor_doc = next(d for d in dev if d['key'] == partner)
            for read in doc['reads']:
                if arm == 'donor' and read['op'] != 'late_target':
                    continue
                view = records if read['snapshot'] == 'v1' else revised
                target = next(r for r in view if r['slot'] == read['query']['slot'])
                expected = read['answer']
                memory = dev_native[doc['key']][read['snapshot']] if arm in ('candidate', 'untrained') else None
                evidence = target['text'] if arm == 'storage' else None
                if arm == 'donor':
                    memory = dev_native[partner]['v1']
                    expected = next(r['value'] for r in donor_doc['document']['records'] if r['slot'] == read['query']['slot'])
                    # Valid digest comes from the donor envelope, recipient metadata is identical.
                    assert memory['meta'] == dev_native[doc['key']]['v1']['meta']
                result = generate(handle, read['query']['text'], memory, evidence)
                rows.append(dict(result, arm=arm, doc=doc['key'], op=read['op'], expected=expected))
        print(json.dumps({'completed_arm': arm, 'rows': sum(r['arm'] == arm for r in rows)}), flush=True)
    summary = {}
    for arm in ('candidate', 'omission', 'untrained', 'storage', 'donor'):
        selected = [r for r in rows if r['arm'] == arm]
        summary[arm] = {'correct': sum(r['generated'] == r['expected'] for r in selected),
                        'total': len(selected), 'terminated': sum(r['terminated'] for r in selected),
                        'empty': sum(not r['generated'] for r in selected),
                        'capped': sum(r['budget_exhausted'] for r in selected),
                        'distinct_outputs': len(set(r['generated'] for r in selected))}
    summary['dictionary'] = {'correct': sum(next(r['value'] for r in
        (d['document']['records'] if q['snapshot'] == 'v1' else [d['document']['revision'] if r['record_id'] == d['document']['revision']['record_id'] else r for r in d['document']['records']])
        if r['slot'] == q['query']['slot']) == q['answer'] for d in dev for q in d['reads']), 'total': 768}
    report = {'status': 'COMPLETE_DIAGNOSTIC', 'prototype_acceptance': 'INCOMPLETE',
              'scope': 'tiny random Gemma stand-in only; independent train/dev slice; no test',
              'fit': fit_result, 'train_correct': train_correct, 'train_total': len(train_outputs),
              'train_fixture_sha256':digest(train), 'dev_fixture_sha256':digest(dev),
              'train_complete_documents':sum(all(r['generated']==r['expected'] for r in train_outputs if r['doc']==d['key']) for d in train),
              'overfit_pass': train_correct >= .9 * len(train_outputs), 'summary': summary,
              'diagnosis': None if train_correct >= .9 * len(train_outputs) else 'Frozen random stem has no learned question-to-record/answer readout; answer loss decreased without sufficient greedy semantic trace recovery. No retuning attempted.',
              'identities': identities(cfg), 'connector_sha256': file_hash(out / 'connector.safetensors'),
              'limitations': ['Restricted reader restart not yet integrated', 'No full acceptance or pretrained competence', 'No quality-matched economics or manual review', 'Canonical runner owner must wire run_phase']}
    (out / 'outputs.json').write_text(json.dumps({'train': train_outputs, 'dev': rows}, indent=2), encoding='utf-8')
    (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--check', default='trained-prefix')
    args = parser.parse_args(argv)
    if args.check not in ('trained-prefix', 'production-smoke'):
        print('UNKNOWN_CHECK')
        return 2
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # External effects are raising mocks, never merely advisory environment flags.
    from unittest.mock import patch
    import socket
    def forbidden(*args, **kwargs):
        raise RuntimeError('offline construction forbids external/model-weight effects')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    with patch.object(socket.socket, 'connect', forbidden), patch.object(torch.cuda, '_lazy_init', forbidden), \
         patch('transformers.AutoTokenizer.from_pretrained', forbidden), \
         patch('transformers.PreTrainedModel.from_pretrained', forbidden):
        if args.check == 'production-smoke':
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=False)
            result = smoke(load_handle(cfg), cfg, out / 'smoke.json')
        else:
            result = run_phase(cfg, args.out)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get('status') in ('PASS', 'COMPLETE_DIAGNOSTIC') else 1


if __name__ == '__main__':
    raise SystemExit(main())
