"""Finite planted-evidence attention probe; not a completeness certificate.

Preparation/scoring import no model. CUDA execution requires exclusive external
supervision. Gold block IDs are used only after model measurements are written.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25(blocks, question):
    docs = [Counter(words(x)) for x in blocks]
    average = sum(map(lambda x: sum(x.values()), docs)) / len(docs)
    result = [0.0] * len(docs)
    for word in set(words(question)):
        count = sum(word in doc for doc in docs)
        idf = math.log(1 + (len(docs) - count + .5) / (count + .5))
        for i, doc in enumerate(docs):
            frequency = doc[word]
            result[i] += idf * frequency * 2.5 / (
                frequency + 1.5 * (.25 + .75 * sum(doc.values()) / average))
    return result


def ranked(scores):
    if not scores or not all(math.isfinite(x) for x in scores):
        raise ValueError("empty or nonfinite scores")
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def choose_heads(dev, labels, count=5):
    """One shared head set, chosen on DEV only, by mean reciprocal gold rank."""
    if set(dev) != set(labels) or not dev:
        raise ValueError("DEV measurement/label membership mismatch")
    shape = len(next(iter(dev.values())))
    quality = [0.0] * shape
    for key, scores in dev.items():
        if len(scores) != shape:
            raise ValueError("head shape mismatch")
        for h, row in enumerate(scores):
            quality[h] += 1 / (1 + ranked(row).index(labels[key]))
    return ranked(quality)[:count]


def summarize(measurements, labels, heads):
    if set(measurements) != set(labels) or not measurements:
        raise ValueError("held-out measurement/label membership mismatch")
    rows = []
    for key, scores in measurements.items():
        combined = aggregate(scores,heads)
        order = ranked(combined)
        rows.append(dict(id=key, gold=labels[key], top3=order[:3],
                         recall3=labels[key] in order[:3],
                         rank=1 + order.index(labels[key])))
    return rows


def aggregate(scores, heads):
    # Normalize corpus mass per head before combining; question self-mass is
    # deliberately excluded. Raw masses must still be retained beside ranks.
    if not scores or not heads:
        raise ValueError('empty heads or scores')
    combined = [0.0] * len(scores[0])
    for head in heads:
        total=sum(scores[head])
        if total<=0 or not math.isfinite(total):
            raise ValueError('head has no finite corpus attention')
        if len(scores[head])!=len(combined) or any(x<0 or not math.isfinite(x) for x in scores[head]):
            raise ValueError('invalid block masses')
        for i,score in enumerate(scores[head]):combined[i]+=score/total
    return combined


def prepare(corpus_root, tokenizer_path, output, block_tokens=400):
    """Build exactly 60 x 400 tokens, with synthetic probes in real background.

    This deliberately does not claim naturally occurring semantic completeness.
    B has zero non-stopword overlap with its own planted statement. Entity
    disambiguation for C uses six records differing only in entity and value.
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / 'tokenizer.json'))
    rng = random.Random(20261017)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources, background = [], []
    for group in ('agent-tooling', 'mapu', 'latent-space'):
        for path in sorted((Path(corpus_root) / 'text' / group).glob('*.txt')):
            raw = path.read_bytes()
            text = raw.decode('utf8')
            if re.search(r'(?i)(api[_-]?key\s*[:=]|BEGIN .*PRIVATE KEY|Bearer\s+[A-Za-z0-9])', text):
                continue
            # Prevent account emails from entering this private copied corpus.
            text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL REDACTED]', text)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                tokens=len(ids)))
            background.extend(ids)
    if len(background) < 60*block_tokens:
        raise ValueError('not enough unique real background tokens')
    # Tier B intentionally lacks a shared nonce/entity between question and text.
    # Twenty distinct semantic pairs are reused only as independent questions,
    # not treated as independent factual worlds in uncertainty estimates.
    pairs = [
        ('The physician parked the automobile beside the dwelling.', 'Where did the doctor leave a car?', 'beside the dwelling'),
        ('The infant gripped a crimson sphere.', 'What did the baby hold?', 'a crimson sphere'),
        ('The canine concealed a bone beneath the couch.', 'Where did the dog hide food?', 'beneath the couch'),
        ('The aviator purchased a violet suitcase.', 'What did the pilot buy?', 'a violet suitcase'),
        ('The monarch selected the eastern chamber.', 'Which room did the king choose?', 'the eastern chamber'),
        ('The burglar abandoned the jewels inside a barrel.', 'Where did the thief leave valuables?', 'inside a barrel'),
        ('The educator dispatched a parcel on Tuesday.', 'When did the teacher send a package?', 'on Tuesday'),
        ('The seaman repaired a vessel near the glacier.', 'Where did the sailor fix a boat?', 'near the glacier'),
        ('The merchant concealed currency behind the portrait.', 'Where did the trader hide money?', 'behind the portrait'),
        ('The chef obtained aubergines from the bazaar.', 'Where did the cook get eggplants?', 'from the bazaar'),
        ('The equine consumed fodder at dawn.', 'When did the horse eat hay?', 'at dawn'),
        ('The feline slept atop the cupboard.', 'Where did the cat nap?', 'atop the cupboard'),
        ('The attorney inspected the testament at noon.', 'When did the lawyer examine a will?', 'at noon'),
        ('The adolescent deposited footwear under the bench.', 'Where did the teenager put shoes?', 'under the bench'),
        ('The custodian unlocked the entrance with brass.', 'What material opened the janitor\'s door?', 'brass'),
        ('The journalist observed a conflagration beyond the ridge.', 'Where did the reporter see fire?', 'beyond the ridge'),
        ('The laborer constructed a fence using cedar.', 'What wood did the worker build with?', 'cedar'),
        ('The angler discovered a cavern beside the waterfall.', 'Where did the fisherman find a cave?', 'beside the waterfall'),
        ('The proprietor stored beverages within the cellar.', 'Where did the owner keep drinks?', 'within the cellar'),
        ('The passenger commenced the voyage during winter.', 'When did the traveler begin a journey?', 'during winter'),
    ]
    stop = set('a an the did do does where what when which with from of in on at to for s'.split())
    planted = [[] for _ in range(60)]
    questions = []
    places = list(range(60)); rng.shuffle(places)
    def nonce():
        return ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(9))

    for tier in 'ABC':
        for i in range(20):
            target = places['ABC'.index(tier) * 20 + i]
            if tier == 'A':
                entity, value = f'LexicalStation{i:02}', nonce()
                statement = f'The access code for {entity} is {value}.'
                question = f'What is the access code for {entity}?'
            elif tier == 'B':
                statement, question, value = pairs[i]
                overlap = (set(words(statement)) & set(words(question))) - stop
                if overlap:
                    raise ValueError(f'Tier B leakage: {i}: {overlap}')
            else:
                target_suffix = rng.randrange(6)
                entity, value = f'DecoyStation{i:02}R{target_suffix}', nonce()
                statement = f'The access code for {entity} is {value}.'
                question = f'What is the access code for {entity}?'
                for j in range(6):
                    if j == target_suffix:
                        continue
                    planted[(target + (j+1) * 7) % 60].append(
                        f'The access code for DecoyStation{i:02}R{j} is {nonce()}.')
            planted[target].append(statement)
            questions.append(dict(id=f'{tier}{i:02}', tier=tier,
                                  split='dev' if i < 10 else 'test',
                                  question=question, answer=value, gold_block=target,
                                  planted_statement=statement))
    blocks, spans, cursor = [], [], 0
    all_ids = []
    for i in range(60):
        text = '\n' + '\n'.join(planted[i]) + '\n'
        prefix = tokenizer.encode(text, add_special_tokens=False).ids
        if len(prefix) >= block_tokens:
            raise ValueError('planted content exceeds block')
        take = block_tokens - len(prefix)
        # Random interior placement removes a universal first-line shortcut.
        cut = rng.randrange(take+1)
        ids = (background[cursor:cursor + cut] + prefix
               + background[cursor + cut:cursor + take])
        cursor += take
        spans.append([len(all_ids), len(all_ids) + len(ids)])
        all_ids.extend(ids)
        blocks.append(tokenizer.decode(ids, skip_special_tokens=False))
    reader = dict(ids=all_ids, spans=spans, blocks=blocks,
                  questions=[{k: q[k] for k in ('id', 'question')} for q in questions])
    labels = dict(questions=questions, sources=sources,
                  design='synthetic planted probes in unique real-text background')
    for name, data in [('reader.json', reader), ('labels.json', labels)]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
    return dict(blocks=len(blocks), tokens=len(all_ids), questions=len(questions),
                real_background_tokens=cursor, sources=len(sources))


def stream_small_model(path, *, long_horizon=False):
    """Instrumentable native Phi3, metadata count before any weight allocation."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig,AutoModelForCausalLM,GenerationConfig
    root=Path(path)
    config=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    if config.model_type!='phi3' or config.tie_word_embeddings:
        raise ValueError('this small-reader loader supports untied native Phi3')
    if long_horizon:
        # Every prefix chunk uses the same long-context frequencies as a whole
        # >4096-token prefill; otherwise crossing the threshold changes old keys.
        assert config.rope_parameters['rope_type']=='longrope'
        config.rope_parameters['short_factor']=list(config.rope_parameters['long_factor'])
    mapping=json.loads((root/'model.safetensors.index.json').read_text(encoding='utf8'))['weight_map']
    count=0
    for shard in sorted(set(mapping.values())):
        with safe_open(root/shard,framework='pt',device='cpu') as f:
            for name in f.keys():count+=math.prod(f.get_slice(name).get_shape())
    if count>=4_000_000_000:raise ValueError('this small-reader loader supports models below 4B parameters only')
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,
                                              attn_implementation='sdpa',trust_remote_code=False)
    if set(mapping)!=set(dict(model.named_parameters())):raise ValueError('parameter membership mismatch')
    import psutil
    for name in list(dict(model.named_parameters())):
        if psutil.virtual_memory().available<20*2**30:raise RuntimeError('RAM floor')
        with safe_open(root/mapping[name],framework='pt',device='cpu') as f:
            value=f.get_tensor(name).to(device='cuda',dtype=torch.bfloat16)
        parent,attribute=name.rsplit('.',1)
        setattr(model.get_submodule(parent),attribute,torch.nn.Parameter(value,requires_grad=False))
        del value
    for name,value in list(model.named_buffers()):
        if value.is_meta:raise ValueError('meta buffer')
        parent,attribute=name.rsplit('.',1)
        setattr(model.get_submodule(parent),attribute,value.to('cuda'))
    model.generation_config=GenerationConfig.from_pretrained(root,local_files_only=True)
    model._lsr_parameter_count=count
    return model.eval()


def attention_query(model, prefix, query_ids, spans):
    """Capture block mass one layer at a time; never retain full attention maps.

    Caller restores a pristine corpus prefix for EVERY question. Cache cropping
    must be tested by the operation smoke; this function doesn't own the cache.
    """
    if os.environ.get('LSR_ATTENTION_SUPERVISED') != '1':
        raise RuntimeError('exclusive external CUDA supervision required')
    import torch
    import psutil
    if psutil.virtual_memory().available < 20 * 2**30:
        raise RuntimeError('20 GiB RAM floor')
    n = prefix.get_seq_length()
    if n != spans[-1][1] or len(query_ids) > 128:
        raise ValueError('prefix alignment or question length')
    masses = []
    handles = []

    def capture(module, inputs, output):
        attention = output[1]
        expected = (1, model.config.num_attention_heads, len(query_ids), n+len(query_ids))
        if attention is None or tuple(attention.shape) != expected:
            raise ValueError('eager attention missing/misaligned')
        if not torch.isfinite(attention).all() or attention.min() < 0:
            raise ValueError('invalid attention probabilities')
        mass = attention[0].float().mean(dim=1)
        masses.append(torch.stack([mass[:, a:b].sum(-1) for a, b in spans], -1).cpu().tolist())
        return output[0], None

    old = model.config._attn_implementation
    try:
        model.config._attn_implementation = 'eager'
        handles = [layer.self_attn.register_forward_hook(capture) for layer in model.model.layers]
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([query_ids], device='cuda'),
                        past_key_values=prefix, use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1].float().cpu().tolist()
    finally:
        for handle in handles:
            handle.remove()
        model.config._attn_implementation = old
        prefix.crop(n)
    if len(masses) != len(model.model.layers):
        raise ValueError('missing layer hooks')
    return [head for layer in masses for head in layer], logits
