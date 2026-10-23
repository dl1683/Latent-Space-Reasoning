"""Native attention-state memory comparator; ordinary prompt caching, not a fit.

CPU is the default; CUDA needs the explicit external resource_guard. No model on import.
Saved states contain only tensors and position count; provenance is host supplied.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class NativeMemory:
    layers: tuple
    length: int

    @classmethod
    def capture(cls, cache):
        return cls(tuple((k.detach().cpu().clone(), v.detach().cpu().clone())
                         for k, v, *_ in cache), cache.get_seq_length())

    def cache(self, *, zero=False, device=None):
        from transformers import DynamicCache
        def copy(t):
            t = t.clone()
            if zero:
                t.zero_()
            return t.to(device) if device is not None else t
        return DynamicCache([(copy(k), copy(v)) for k, v in self.layers])

    @property
    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for pair in self.layers for t in pair)

    def digest(self):
        import torch
        h = hashlib.sha256(str(self.length).encode())
        for pair in self.layers:
            for t in pair:
                h.update(str(t.dtype).encode())
                h.update(t.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def save(self, path):
        import torch
        torch.save(dict(layers=self.layers, length=self.length), path)

    @classmethod
    def load(cls, path):
        import torch
        obj = torch.load(path, map_location='cpu', weights_only=True)
        if set(obj) != {'layers', 'length'} or type(obj['length']) is not int or obj['length'] <= 0:
            raise ValueError('invalid native memory schema')
        pairs = obj['layers']
        if not pairs:
            raise ValueError('empty native memory')
        for k, v in pairs:
            if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1
                    or k.shape[-2] != obj['length'] or k.dtype not in (torch.float32, torch.bfloat16)
                    or v.dtype != k.dtype
                    or not torch.isfinite(k).all() or not torch.isfinite(v).all()):
                raise ValueError('invalid native attention tensors')
        return cls(tuple(pairs), obj['length'])


def forward(model, tokens, memory=None, *, zero=False):
    """One call, optional ordinary KV. Caller owns tokenization/model identity."""
    import torch
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
        raise ValueError('one nonempty token sequence required')
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    prefix = 0 if memory is None else memory.length
    cache = None if memory is None else memory.cache(zero=zero, device=device)
    with torch.inference_mode():
        return model(input_ids=tokens, past_key_values=cache, use_cache=True,
                     attention_mask=torch.ones((1, prefix + tokens.shape[1]), dtype=torch.long, device=device),
                     position_ids=torch.arange(prefix, prefix + tokens.shape[1], device=device)[None])


def greedy(model, tokens, memory=None, *, limit=16):
    result = forward(model, tokens, memory)
    return decode(model, result, limit=limit)


def decode(model, result, *, limit=16):
    import torch
    device = next(model.parameters()).device
    stop = model.generation_config.eos_token_id
    stops = set(stop if isinstance(stop, list) else [stop])
    output = []
    cache = result.past_key_values
    for _ in range(limit):
        if not torch.isfinite(result.logits[0, -1]).all():
            raise ValueError('nonfinite decoder logits')
        token = int(result.logits[0, -1].argmax())
        output.append(token)
        if token in stops:
            return output, True
        length = cache.get_seq_length()
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                           attention_mask=torch.ones((1, length + 1), dtype=torch.long, device=device),
                           position_ids=torch.tensor([[length]], device=device), use_cache=True)
        cache = result.past_key_values
    return output, False


def choice_scores(model, tokenizer, output, choices):
    """Mean conditional token log probability, without an option-letter mapping."""
    import torch
    prompt_memory = NativeMemory.capture(output.past_key_values)
    scores = []
    for text in choices:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            raise ValueError('nonempty answer candidate required')
        first = output.logits[0, -1].float().log_softmax(-1)[tokens[0]]
        total = float(first)
        if len(tokens) > 1:
            tail = forward(model, torch.tensor([tokens[:-1]]), prompt_memory)
            logp = tail.logits[0].float().log_softmax(-1)
            total += float(logp[torch.arange(len(tokens) - 1), torch.tensor(tokens[1:])].sum())
        scores.append(total / len(tokens))
    return scores


def qualify_binary(cached, reference):
    """Finite experiment gate in score units, independent of reference labels.

    The raw-source reference is paid calibration work, not available to arbitrary
    new source-denied questions. This is not a general abstention guarantee.
    """
    import math
    if (len(cached) != 2 or len(reference) != 2
            or not all(math.isfinite(x) for x in [*cached, *reference])):
        return dict(answer='UNCERTAIN', passed=False)
    error = max(abs(x - y) for x, y in zip(cached, reference))
    margin = abs(cached[0] - cached[1])
    choice = max(range(2), key=lambda i: cached[i])
    agreed = choice == max(range(2), key=lambda i: reference[i])
    passed = agreed and error <= .1 and margin > 2 * error + .05
    return dict(answer='AB'[choice] if passed else 'UNCERTAIN', passed=passed,
                candidate_error=error, margin=margin)


def finalize_binary(row):
    """Expose a qualified answer; cold use without calibration fails closed.

    A=Yes/B=No are experiment labels. Raw choices remain explicitly named and
    cannot become qualified negatives merely because reference scores are absent.
    """
    arms = row['arms']
    if 'cached' in arms and 'full' in arms:
        qualification = qualify_binary(arms['cached']['logits'], arms['full']['logits'])
    else:
        qualification = dict(answer='UNCERTAIN', passed=False,
                             reason='permitted source calibration unavailable')
    for arm in arms.values():
        if 'answer' in arm:
            arm['raw_answer'] = arm.pop('answer')
    row['qualification'] = qualification
    row['answer'] = qualification['answer']
    return row


def load_streamed_mistral(path, *, dtype):
    """Guarded CUDA loader: release each CPU tensor/map before opening the next.

    No quantization or CPU offload. All checkpoint parameters must be assigned;
    tiny-model parity against the ordinary loader is a separate prerequisite.
    """
    import os
    if os.environ.get('LSR_GPU_GUARDED') != '1':
        raise ValueError('streamed CUDA load requires the external resource_guard')
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    root = Path(path).resolve()
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    if config.model_type != 'mistral' or config.tie_word_embeddings:
        raise ValueError('this finite loader supports untied Mistral only')
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, attn_implementation='eager')
    index = root / 'model.safetensors.index.json'
    if index.exists():
        weights = json.loads(index.read_text(encoding='utf8'))['weight_map']
    else:
        with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as f:
            weights = {name: 'model.safetensors' for name in f.keys()}
    names = dict(model.named_parameters())
    if set(weights) != set(names):
        raise ValueError('checkpoint/model parameter membership differs')
    for name in names:
        file = (root / weights[name]).resolve()
        if file.parent != root:
            raise ValueError('weight shard outside pinned checkpoint')
        with safe_open(file, framework='pt', device='cpu') as f:
            source = f.get_tensor(name)
            value = source.to(device='cuda:0', dtype=dtype)
            del source
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, torch.nn.Parameter(value, requires_grad=False))
        del value
    for name, value in list(model.named_buffers()):
        if value.is_meta:
            raise ValueError('unmaterialized model buffer')
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, value.to(device='cuda:0'))
    if any(x.is_meta or x.device.type != 'cuda' for x in model.parameters()):
        raise ValueError('incomplete CUDA checkpoint load')
    model.generation_config = GenerationConfig.from_pretrained(root, local_files_only=True)
    return model.eval()


def main():
    import argparse
    import os
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    device = cfg.get('device', 'cpu')
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('explicit CPU or guarded CUDA device required')
    if device != 'cpu' and os.environ.get('LSR_GPU_GUARDED') != '1':
        raise ValueError('CUDA requires the finite external resource resource_guard')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device != 'cpu':
        import uuid
        expected = uuid.UUID(os.environ['LSR_GPU_UUID'].removeprefix('GPU-')).hex
        actual = torch.cuda.get_device_properties(0).uuid
        actual = uuid.UUID(bytes=actual).hex if isinstance(actual, bytes) else uuid.UUID(str(actual).removeprefix('GPU-')).hex
        if expected != actual:
            raise ValueError('CUDA/NVML UUID mismatch')
        torch.cuda.set_per_process_memory_fraction(.85, 0)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)
    if cfg.get('streamed_cuda'):
        if device != 'cuda:0':
            raise ValueError('streamed load is a CUDA-only explicit protocol')
        model = load_streamed_mistral(cfg['model'], dtype=getattr(torch, cfg.get('dtype', 'bfloat16')))
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
                                                   dtype=getattr(torch, cfg.get('dtype', 'float32')), device_map=device,
                                                   attn_implementation='eager').eval()
    if device != 'cpu':
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    if cfg.get('denied_paths'):
        import sys
        denied = {str(Path(x).resolve()).casefold() for x in cfg['denied_paths']}
        def deny(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                if str(Path(os.fsdecode(args[0])).resolve()).casefold() in denied:
                    raise PermissionError('planned source access denied')
        sys.addaudithook(deny)
        for name in cfg['denied_paths']:
            try:
                Path(name).read_bytes()
            except PermissionError:
                continue
            raise AssertionError('source denial inactive')
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    def ids(text): return tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids
    candidate_ids = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    assert all(len(x) == 1 for x in candidate_ids)
    candidate_ids = [x[0] for x in candidate_ids]
    for q in cfg['questions']:
        if q.get('candidates') == ['Yes', 'No']:
            if any(len(tokenizer.encode(x, add_special_tokens=False)) != 1 for x in q['candidates']):
                raise ValueError('frozen binary candidates must each be one token')
    results = dict(model=cfg['model'], load_seconds=load_seconds, worlds={},
                   torch=torch.__version__, transformers=__import__('transformers').__version__)
    for world in cfg['worlds']:
        # Cold mode reads only serialized native memory, not source or prefix IDs.
        if cfg.get('cold'):
            if hashlib.sha256(Path(world['bank']).read_bytes()).hexdigest() != world['bank_sha256']:
                raise ValueError('native bank differs from external byte pin')
            bank = NativeMemory.load(world['bank'])
            prefix = None
            encode_seconds = None
        else:
            prefix = (torch.tensor([world['prefix_token_ids']], dtype=torch.long)
                      if 'prefix_token_ids' in world else ids(world['prefix']))
            if cfg.get('max_total_tokens') and len(prefix[0]) > cfg['max_total_tokens']:
                raise ValueError('complete source prefix exceeds registered bound')
            started = time.perf_counter()
            bank = NativeMemory.capture(forward(model, prefix).past_key_values)
            encode_seconds = time.perf_counter() - started
            bank.save(world['bank'])
        digest = bank.digest()
        rows = []
        for q in cfg['questions']:
            if q.get('world_ids') and world['id'] not in q['world_ids']:
                continue
            suffix = (torch.tensor([q['token_ids']], dtype=torch.long)
                      if 'token_ids' in q else ids(q['suffix']))
            if cfg.get('max_total_tokens'):
                longest = max([len(tokenizer.encode(x, add_special_tokens=False)) for x in q.get('candidates', ['A'])])
                if bank.length + len(suffix[0]) + max(longest, cfg.get('limit', 16)) > cfg['max_total_tokens']:
                    raise ValueError('complete source/query/continuation exceeds registered token bound')
            row = dict(id=q['id'], arms={})
            arms = ['cached'] if cfg.get('cold') else cfg.get('arms', ['cached', 'full', 'none', 'zero', 'empty'])
            for arm in arms:
                started = time.perf_counter()
                if arm == 'full': output = forward(model, torch.cat([prefix, suffix], dim=1))
                elif arm == 'none': output = forward(model, suffix)
                elif arm == 'empty':
                    empty = (torch.tensor([world['empty_prefix_token_ids']], dtype=torch.long)
                             if 'empty_prefix_token_ids' in world else ids(world['empty_prefix']))
                    output = forward(model, torch.cat([empty, suffix], dim=1))
                elif arm == 'reload': output = forward(model, suffix, NativeMemory.load(world['bank']))
                else: output = forward(model, suffix, bank, zero=(arm == 'zero'))
                last = output.logits[0, -1].float()
                logits = ([] if q.get('readout') == 'source_set' else
                          choice_scores(model, tokenizer, output, q['candidates'])
                          if 'candidates' in q else [float(last[i]) for i in candidate_ids])
                answer = 'ABCD'[max(range(len(logits)), key=lambda i: logits[i])] if logits else 'NOT_A_CHOICE'
                row['arms'][arm] = dict(answer=answer, logits=logits,
                                        finite_logits=bool(torch.isfinite(last).all()),
                                        score_kind='mean_answer_token_logp' if 'candidates' in q else 'first_letter_logit',
                                        seconds=time.perf_counter() - started,
                                        unconstrained_first=tokenizer.decode([int(last.argmax())]))
                if arm == 'cached': cached_logits = last.clone()
                if arm == 'full' and 'cached' in row['arms']:
                    row['full_cache_max_abs_error'] = float((last - cached_logits).abs().max())
                if arm == 'reload' and 'cached' in row['arms']:
                    row['repeat_cache_max_abs_error'] = float((last - cached_logits).abs().max())
                if q.get('readout') == 'source_set' and arm in ('cached', 'full', 'empty'):
                    generated, terminated = decode(model, output, limit=cfg.get('limit', 16))
                    row['arms'][arm]['generation'] = dict(text=tokenizer.decode(generated, skip_special_tokens=True),
                                                          ids=generated, terminated=terminated)
                    row['arms'][arm]['seconds'] = time.perf_counter() - started
                del output
            binary = q.get('candidates') == ['Yes', 'No']
            if binary:
                finalize_binary(row)
            if cfg.get('generate') and q.get('readout') != 'source_set':
                generated, terminated = greedy(model, suffix, bank, limit=cfg.get('limit', 16))
                row['generation'] = dict(text=tokenizer.decode(generated, skip_special_tokens=True),
                                         ids=generated, terminated=terminated)
                if prefix is not None:
                    full_generated, full_terminated = greedy(model, torch.cat([prefix, suffix], dim=1),
                                                             limit=cfg.get('limit', 16))
                    row['full_generation'] = dict(text=tokenizer.decode(full_generated, skip_special_tokens=True),
                                                  ids=full_generated, terminated=full_terminated)
            rows.append(row)
            assert bank.digest() == digest, 'query modified saved memory'
            display = (dict(answer=row['answer'], raw_answers={a:r['raw_answer'] for a,r in row['arms'].items()})
                       if binary else dict(answers={a:r['answer'] for a,r in row['arms'].items()}))
            print(json.dumps(dict(world=world['id'], query=q['id'], **display)), flush=True)
        results['worlds'][world['id']] = dict(rows=rows, prefix_tokens=bank.length,
                                             payload_bytes=bank.payload_bytes, digest=digest,
                                             encode_seconds=encode_seconds,
                                             serialized_bytes=Path(world['bank']).stat().st_size)
    (root / cfg.get('result_name', 'outputs.json')).write_text(json.dumps(results, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
