"""Answer-loss trainer (Z Code B, offline mechanics).

Carries the answer-loss CONTRACT and its offline mechanics: joint
prompt+answer labeling with query/pad/prefix positions masked to -100;
trainable resampler vs explicitly frozen stem; finite nonzero intended
gradients; trainer API that cannot open test truth (mechanical guard).
The tiny random smoke verifies plumbing and gradients ONLY — it is
never a prototype PASS and never a model claim. Real fitting runs on
the cloud with the pinned stem passed by deployment.
"""
import json
import math
import os

import torch
import torch.nn.functional as F

from . import data, writer

IGNORE = -100  # masked-label convention for query/pad/prefix positions


class TrainerGuard(Exception):
    pass


def open_split(path, split):
    """The ONLY data-opening door in the trainer. Refuses anything that
    smells of test truth: the trainer may see train/dev answers, never
    test answers."""
    low = str(path).lower().replace("\\", "/")
    if "test_truth" in low or "test-truth" in low:
        raise TrainerGuard(f"trainer refused test-truth path: {path}")
    if split == "test":
        raise TrainerGuard("trainer may not open the test split")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mask_answer_loss(prompt_len, answer_ids, pad_to=None):
    """Labels for one jointly-tokenized prompt+answer: positions inside
    the prompt (query/prefix) and padding are IGNORE (-100); answer
    positions carry their ids. Returns (labels, n_active)."""
    labels = [IGNORE] * prompt_len + list(answer_ids)
    if pad_to is not None:
        if len(labels) > pad_to:
            raise ValueError("sequence longer than pad target")
        labels += [IGNORE] * (pad_to - len(labels))
    return labels, len(answer_ids)


def answer_loss(logits, labels):
    """Mean cross-entropy over non-ignored positions; 0.0 with no
    active positions is a contract violation, returned as None."""
    active = labels != IGNORE
    if not active.any():
        return None
    return F.cross_entropy(logits[active], labels[active])



def tiny_smoke(cfg, seed=0, verbose=False):
    """Compatibility entry, now exercising the real S0 tensor-loss path.

    One bounded diagnostic optimizer move is measured and then undone. This
    smoke never changes a fitted checkpoint or supplies semantic competence.
    """
    import copy
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import socket
    from . import semantic
    from experiments.cloud.lsr_jobs import s0_memory_port as port
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        if torch.get_num_interop_threads() != 2:
            raise
    if 'semantic' not in cfg:
        cfg=json.loads((Path(__file__).parent/'contract.json').read_text())
    if cfg['semantic']['stem']['kind'] != 'tiny-random':
        raise ValueError('local smoke requires explicit tiny random stem')
    def forbidden(*args, **kwargs):
        raise RuntimeError('offline smoke forbids external effects')
    scratch=Path(__file__).parent/'scratchpad'
    scratch.mkdir(exist_ok=True)
    with patch.object(socket.socket,'connect',forbidden), patch.object(torch.cuda,'_lazy_init',forbidden), \
            patch('transformers.PreTrainedModel.from_pretrained',forbidden), \
            patch('transformers.AutoTokenizer.from_pretrained',forbidden):
        handle=semantic.load_handle(cfg)
        initial=copy.deepcopy(handle['connector'].state_dict())
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            control=semantic.smoke(handle,cfg,Path(tmp)/'smoke.json')
        docs=semantic.slice_documents('train',2,cfg['seed'])
        memory=semantic.states(handle,docs)[docs[0]['key']]['v1']
        read=docs[0]['reads'][0]
        inputs=semantic.answer_batch(handle,[(read['query']['text'],read['answer'])])
        optimizer=torch.optim.SGD(handle['connector'].parameters(),lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        before=port.forward(handle,inputs,memory)['loss']
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after=port.forward(handle,inputs,memory)['loss']
        handle['connector'].load_state_dict(initial)
        handle['receiver'].zero_grad(set_to_none=True)
    try:
        open_split('test_truth.locked.json','train')
        refused=False
    except TrainerGuard:
        refused=True
    checks=control['checks']
    result={'masking_fixture':checks['answer_eos_only'], 'guard_refused_test_truth':refused,
            'resampler_grads_finite_nonzero':checks['finite_nonzero_gradients'],
            'frozen_stub_zero_grads':checks['frozen_stem'],
            'losses_finite':math.isfinite(float(before.detach())) and math.isfinite(float(after)),
            'loss_decreased':float(after)<float(before.detach()),
            'first_loss':float(before.detach()),'last_loss':float(after),'steps':1,
            'production_checks':checks,'production_status':control['status'],
            'interpretation':'actual production S0/receiver with RANDOM tiny Gemma; mechanics only; one diagnostic move restored'}
    if verbose:
        print(json.dumps(result))
    return result
