"""One locked CPU conditional-port diagnostic; no fitting or sequence search."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

SEQUENCE = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
PAIRS = [(10, 30), (20, 40), (30, 50), (40, 60)]
REVISION = 'c731040fcd8d73dceaa04b0a8e6329b345b0f5df'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.snapshot.name == REVISION and len(SEQUENCE) == 76
    assert not args.output.exists(), 'Do not overwrite a prior run'
    start = time.monotonic()
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.snapshot, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager').cpu().eval()
    assert next(model.parameters()).device.type == 'cpu'
    base = tokenizer(SEQUENCE, return_tensors='pt')
    ids = base['input_ids']
    assert ids.shape == (1, 78)
    assert tokenizer.convert_ids_to_tokens(ids[0, 1:-1]) == list(SEQUENCE)
    result = {
        'probe': 'conditional_port_v1', 'status': 'running',
        'sequence': SEQUENCE, 'pairs_one_based': PAIRS,
        'revision': REVISION, 'device': 'cpu', 'dtype': 'float32',
        'attention_implementation': 'eager', 'cpu_threads': 2,
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'primary_edge_tolerance_nats': 0.1, 'sensitivity_tolerances_nats': [0.05, 0.2],
        'runner_sha256': digest(Path(__file__)),
        'snapshot_sha256': {p.name: digest(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()},
        'training_runs': 0, 'forward_calls': 0, 'input_rows': 0, 'rectangles': [],
        'scope': 'Local conditional compatibility only; no quality, grafting or biological claim.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        result['wall_seconds'] = time.monotonic() - start
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(args.output)

    seen = {}
    shared_deltas = []
    checkpoint()
    for i, j in PAIRS:
        ai = tokenizer.convert_tokens_to_ids('G' if SEQUENCE[i-1] == 'A' else 'A')
        aj = tokenizer.convert_tokens_to_ids('G' if SEQUENCE[j-1] == 'A' else 'A')
        ri, rj = int(ids[0, i]), int(ids[0, j])
        assert ai != ri and aj != rj
        batch = ids.repeat(4, 1)
        # Positive edge differences: i|j=ref, j|i=alt, i|j=alt, j|i=ref.
        batch[0, i] = tokenizer.mask_token_id
        batch[1, i], batch[1, j] = ai, tokenizer.mask_token_id
        batch[2, i], batch[2, j] = tokenizer.mask_token_id, aj
        batch[3, j] = tokenizer.mask_token_id
        assert ((batch == tokenizer.mask_token_id).sum(dim=1) == 1).all()
        for k, changed in enumerate([{i}, {i, j}, {i, j}, {j}]):
            actual = set((batch[k] != ids[0]).nonzero().flatten().tolist())
            assert actual == changed
        positions, alternatives, references = [i, j, i, j], [ai, aj, ai, aj], [ri, rj, ri, rj]
        repeats = []
        with torch.inference_mode():
            for _ in range(2):
                logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
                repeats.append([float(logits[k, p, a] - logits[k, p, r])
                                for k, (p, a, r) in enumerate(zip(positions, alternatives, references))])
                result['forward_calls'] += 1
                result['input_rows'] += 4
        edges = repeats[0]
        circulation = edges[0] + edges[1] - edges[2] - edges[3]
        for k in range(4):
            key = (tuple(batch[k].tolist()), positions[k], alternatives[k], references[k])
            if key in seen:
                shared_deltas.append(abs(edges[k] - seen[key]))
            seen[key] = edges[k]
        result['rectangles'].append({
            'sites_one_based': [i, j], 'positive_edge_logits': repeats,
            'orientation': [1, 1, -1, -1], 'circulation_nats': circulation,
            'isolated_minimax_edge_correction_nats': abs(circulation)/4,
            'mean_absolute_edge_nats': sum(abs(x) for x in edges)/4,
            'max_repeat_edge_difference_nats': max(abs(a-b) for a, b in zip(*repeats)),
            'input_token_ids': batch.tolist(),
        })
        checkpoint()
    result['shared_input_checks'] = len(shared_deltas)
    result['max_shared_input_edge_difference_nats'] = max(shared_deltas, default=0)
    result['status'] = 'complete'
    checkpoint()
    print(json.dumps({k: v for k, v in result.items() if k != 'rectangles'}))
    print(json.dumps([{k: v for k, v in row.items() if k not in ['input_token_ids', 'positive_edge_logits']}
                      for row in result['rectangles']]))


if __name__ == '__main__':
    main()
