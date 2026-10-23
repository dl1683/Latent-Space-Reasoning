"""Guarded complete-record native KV writer; conventional cache, no fitted prefix.

The caller owns model/tokenizer pins, exclusive GPU custody, telemetry and limits.
All copied record fields enter the encoder. No model or CUDA context on import.
"""
import hashlib
import os

from experiments.memory.record_native_memory import RecordMemory, SYSTEM, prefix_text as _prefix_text
from experiments.memory.record_memory_input import encode_records


def prefix_text(tokenizer, records, *, user_only=False):
    """Default framing is unchanged; user_only supports templates without system.

    For user_only=True, the prefix ends at 'Question:' with its final newline
    deferred to the late query. Continue with query_tokens(), not independent
    tokenization of question + suffix: SentencePiece may add a dummy whitespace.
    """
    if not user_only:
        return _prefix_text(tokenizer, records)
    text = tokenizer.apply_chat_template(
        [dict(role='user', content=SYSTEM + '\n\nQuoted JSON evidence:\n'
              '__CORPUS_SLOT__\n\nQuestion:\n__LATE_QUESTION__')],
        tokenize=False, add_generation_prompt=True)
    # Resolve only trusted markers before inserting potentially marker-like data.
    before, after = text.split('__CORPUS_SLOT__')
    between, suffix = after.split('__LATE_QUESTION__')
    if not between.endswith('Question:\n'):
        raise ValueError('template does not preserve the trusted question boundary')
    payload = encode_records(records).decode('utf8')
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e')
    return before + payload + between[:-1], suffix


def query_tokens(tokenizer, question, suffix):
    """Late continuation IDs for user_only=True; never tokenize the corpus again.

    Tokenize a fixed trusted anchor with the query, then remove its IDs. This
    avoids standalone SentencePiece dummy whitespace and retains the newline.
    Callers pin/check tokenizer identity; arbitrary tokenizers may merge this
    boundary and are rejected instead of silently sending different token IDs.
    """
    anchor = tokenizer.encode('Question:', add_special_tokens=False)
    tokens = tokenizer.encode('Question:\n' + question + suffix, add_special_tokens=False)
    if not anchor or tokens[:len(anchor)] != anchor:
        raise ValueError('tokenizer does not preserve the trusted question boundary')
    return tokens[len(anchor):]


def _state_dtype(value):
    import torch
    if value in ('float32', torch.float32):
        return torch.float32
    if value in ('bfloat16', torch.bfloat16):
        return torch.bfloat16
    raise ValueError('record state dtype must be float32 or bfloat16')


def write(model, tokenizer, records, *, chunk_tokens=64, max_tokens=4096,
          state_dtype='float32', user_only=False):
    """Encode complete records once on guarded CUDA, retaining host source links.

    FP32 remains the default. BF16 requires state_dtype='bfloat16' (or the torch
    dtype), matching caller weights and native cache; no dtype conversion occurs.
    This does not authenticate sources or establish model access to every field.
    """
    if os.environ.get('LSR_GPU_GUARDED') != '1':
        raise ValueError('CUDA record writing requires the external resource_guard')
    import torch
    from experiments.memory.local_native_kv import NativeMemory
    dtype = _state_dtype(state_dtype)
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('CUDA model required')
    if next(model.parameters()).dtype != dtype:
        raise ValueError('model dtype must match the explicit record state dtype')
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError('positive integer chunk size required')
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('positive integer input bound required')
    prefix, _ = prefix_text(tokenizer, records, user_only=user_only)
    tokens = tokenizer(prefix, add_special_tokens=False, return_tensors='pt').input_ids
    if tokens.shape[1] > max_tokens:
        raise ValueError('complete record input exceeds registered bound')
    cache = None
    with torch.inference_mode():
        for start in range(0, tokens.shape[1], chunk_tokens):
            end = min(tokens.shape[1], start + chunk_tokens)
            out = model(input_ids=tokens[:, start:end].to(device), past_key_values=cache,
                        attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                        position_ids=torch.arange(start, end, device=device)[None],
                        use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            del out
    native = NativeMemory.capture(cache)
    if any(t.dtype != dtype for pair in native.layers for t in pair):
        raise ValueError('native state dtype must match the record state dtype')
    links = tuple(dict(local_id=f'D{i}', record_id=r.get('record_id'),
                       source_id=r.get('source_id'), revision=r.get('revision'),
                       source_pointer=r.get('source_pointer')) for i, r in enumerate(records))
    return RecordMemory(native, hashlib.sha256(prefix.encode()).hexdigest(), links)


def load(path, *, expected_sha256, expected_dtype):
    """Load checksum-bound state on CPU, without casting or changing host links.

    The caller must pin the expected dtype as well as the external file checksum.
    This checks artifact structure, not model/tokenizer identity or source truth.
    The original CPU RecordMemory loader remains FP32-only.
    """
    import io
    from pathlib import Path
    import torch
    from experiments.memory.local_native_kv import NativeMemory
    dtype = _state_dtype(expected_dtype)
    if dtype == torch.float32:
        return RecordMemory.load(path, expected_sha256=expected_sha256)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('record memory checksum mismatch')
    obj = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if (type(obj) is not dict
            or set(obj) != {'layers', 'length', 'input_sha256', 'source_links'}):
        raise ValueError('invalid record memory schema')
    if (type(obj['length']) is not int or obj['length'] <= 0
            or not isinstance(obj['input_sha256'], str)
            or len(obj['input_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in obj['input_sha256'])
            or type(obj['layers']) not in (tuple, list) or not obj['layers']
            or type(obj['source_links']) not in (tuple, list) or not obj['source_links']):
        raise ValueError('invalid record memory dimensions/binding')
    for pair in obj['layers']:
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise ValueError('expected key/value pair')
        k, v = pair
        if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
                or k.layout != torch.strided or v.layout != torch.strided
                or k.device.type != 'cpu' or v.device.type != 'cpu'
                or k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1
                or any(size <= 0 for size in k.shape)
                or k.shape[-2] != obj['length'] or k.dtype != dtype or v.dtype != dtype
                or not torch.isfinite(k).all() or not torch.isfinite(v).all()):
            raise ValueError('invalid record attention tensors')
    for i, link in enumerate(obj['source_links']):
        if (type(link) is not dict
                or set(link) != {'local_id', 'record_id', 'source_id', 'revision', 'source_pointer'}
                or type(link['local_id']) is not str
                or link['local_id'] != f'D{i}'):
            raise ValueError('invalid source link')
    return RecordMemory(NativeMemory(tuple(tuple(pair) for pair in obj['layers']), obj['length']),
                        obj['input_sha256'], tuple(obj['source_links']))
