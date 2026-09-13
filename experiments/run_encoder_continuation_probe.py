"""Locked S26 CPU pilot: no fitting, search, or automatic retry."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
REVISION = 'c731040fcd8d73dceaa04b0a8e6329b345b0f5df'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SOURCES = {
    'AMFR_HUMAN_Tsuboyama_2023_4G3O.csv': '20f0688dc628f11cc944ce79e109fc91bb73ca08a9683cc7362f61ab4b7b3f99',
    'ARGR_ECOLI_Tsuboyama_2023_1AOY.csv': 'd456b5e596c932371f39f162789b482a828ad2be6ca465544106264bdecc5591',
    'BBC1_YEAST_Tsuboyama_2023_1TG0.csv': 'cf8e8bbddb924fd15c9981523acd8675f283bd164638b0acaf62969d9af4e9b0',
    'CBX4_HUMAN_Tsuboyama_2023_2K28.csv': 'e224843da3c2bf7ffee68b4436294ca9975c00c40a993118f9a164ac3a387639',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references(folder):
    parents = []
    for name, expected in SOURCES.items():
        path = folder / name
        assert digest(path) == expected, name
        restored, count = set(), 0
        with path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                sequence, positions = list(row['mutated_sequence']), set()
                for mutation in row['mutant'].split(':'):
                    match = re.fullmatch(r'([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])', mutation)
                    assert match, mutation
                    before, position, after = match.groups()
                    position = int(position) - 1
                    assert position not in positions and 0 <= position < len(sequence)
                    assert sequence[position] == after
                    positions.add(position)
                    sequence[position] = before
                restored.add(''.join(sequence))
                count += 1
        assert len(restored) == 1
        reference = restored.pop()
        assert set(reference) <= set(ALPHABET)
        sites = [k * (len(reference) + 1) // 9 for k in range(1, 9)]
        assert len(set(sites)) == 8 and min(sites) >= 1 and max(sites) <= len(reference)
        parents.append({'source': name, 'source_sha256': expected, 'source_rows': count,
                        'reference': reference, 'pairs': list(zip(sites[::2], sites[1::2]))})
    return parents


def choice(probabilities):
    return max(range(20), key=probabilities.__getitem__)


def margin(probabilities):
    ordered = sorted(probabilities, reverse=True)
    return ordered[0] - ordered[1]


def summarize(result):
    summaries = []
    for parent_index, parent in enumerate(result['parents']):
        totals = {name: {'both_correct': 0, 'second_correct': 0, 'second_baseline_correct': 0,
                         'both_corrected': 0, 'both_spoiled': 0}
                  for name in ['independent', 'increasing', 'reverse', 'confidence_first']}
        diagnostic = {'oracle_increasing_second_correct': 0, 'oracle_reverse_second_correct': 0,
                      'content_second_correct': 0, 'content_coincident_insertions': 0}
        for row in result['rows'][parent_index * 4:(parent_index + 1) * 4]:
            p = row['probabilities']
            a, b = [choice(x) for x in p['independent']]
            target = [ALPHABET.index(x) for x in row['targets']]
            forward, reverse = (a, choice(p['increasing'][0])), (choice(p['reverse'][0]), b)
            increasing_first = margin(p['independent'][0]) >= margin(p['independent'][1])
            policies = {'independent': ((a, b), 1), 'increasing': (forward, 1),
                        'reverse': (reverse, 0),
                        'confidence_first': (forward, 1) if increasing_first else (reverse, 0)}
            baseline = (a, b) == tuple(target)
            for name, (prediction, second) in policies.items():
                both = prediction == tuple(target)
                totals[name]['both_correct'] += int(both)
                totals[name]['second_correct'] += int(prediction[second] == target[second])
                totals[name]['second_baseline_correct'] += int((a, b)[second] == target[second])
                totals[name]['both_corrected'] += int(both and not baseline)
                totals[name]['both_spoiled'] += int(baseline and not both)
            diagnostic['oracle_increasing_second_correct'] += int(choice(p['oracle_increasing'][0]) == target[1])
            diagnostic['oracle_reverse_second_correct'] += int(choice(p['oracle_reverse'][0]) == target[0])
            diagnostic['content_second_correct'] += int(choice(p['content_control'][0]) == target[1])
            diagnostic['content_coincident_insertions'] += int(row['content_inserted'] == ALPHABET[a])
        summaries.append({'parent': parent['source'], 'n_pairs': 4, 'policies': totals, **diagnostic})
    gains = [(x['policies']['increasing']['both_correct'] - x['policies']['independent']['both_correct']) / 4 for x in summaries]
    macro = lambda values: sum(values) / 4
    both_gain = macro(gains)
    second_gain = macro([(x['policies']['increasing']['second_correct'] - x['policies']['independent']['second_correct']) / 4 for x in summaries])
    reverse_gain = macro([(x['policies']['reverse']['both_correct'] - x['policies']['independent']['both_correct']) / 4 for x in summaries])
    content_gain = macro([(x['policies']['increasing']['second_correct'] - x['content_second_correct']) / 4 for x in summaries])
    repeats = [(p, q) for row in result['rows'] for p, q in zip(row['probabilities']['independent'], row['probabilities']['repeat'])]
    stable = all(choice(p) == choice(q) for p, q in repeats)
    gates = {'both_gain_at_least_point_10': both_gain >= .10, 'second_gain_at_least_point_10': second_gain >= .10,
             'two_parents_positive_none_negative': sum(g > 0 for g in gains) >= 2 and min(gains) >= 0,
             'reverse_gain_at_least_minus_point_0625': reverse_gain >= -.0625,
             'content_control_gain_at_least_point_10': content_gain >= .10}
    return {'parents': summaries, 'primary_both_gain': both_gain, 'primary_second_gain': second_gain,
            'reverse_both_gain': reverse_gain, 'primary_second_vs_content_gain': content_gain,
            'repeat_argmax_agreement': stable,
            'max_repeat_probability_drift': max(abs(a - b) for p, q in repeats for a, b in zip(p, q)),
            'continuation_criteria': gates,
            'decision': 'inconclusive_unstable' if not stable else ('continue_local_study' if all(gates.values()) else 'close_this_procedure')}


def main():
    started = time.monotonic()
    timer = threading.Timer(180, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.snapshot.name == REVISION and not args.output.exists()
    parents = references(args.sources)
    rows = [{'parent_index': k, 'sites': [i, j], 'targets': [p['reference'][i - 1], p['reference'][j - 1]], 'probabilities': {}}
            for k, p in enumerate(parents) for i, j in p['pairs']]
    result = {'probe': 'encoder_continuation_v1', 'status': 'running', 'revision': REVISION,
              'device': 'cpu', 'dtype': 'float32', 'attention': 'eager', 'threads': 2,
              'alphabet': ALPHABET, 'parents': parents, 'rows': rows, 'forward_calls': 0, 'input_rows': 0,
              'training_runs': 0, 'runner_sha256': digest(Path(__file__)), 'deadline_seconds': 180,
              'snapshot_sha256': {p.name: digest(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()}, 'batches': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        result['wall_seconds'] = time.monotonic() - started
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(args.output)

    checkpoint()
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    result.update(torch_version=torch.__version__, transformers_version=transformers.__version__)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(args.snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').cpu().eval()
    assert next(model.parameters()).device.type == 'cpu' and model.config.token_dropout
    tokens = tokenizer([parents[r['parent_index']]['reference'] for r in rows], padding=True, return_tensors='pt')
    ids, attention = tokens['input_ids'], tokens['attention_mask']
    aa_ids = tokenizer.convert_tokens_to_ids(list(ALPHABET))
    assert len(set(aa_ids)) == 20 and tokenizer.unk_token_id not in aa_ids
    base = ids.clone()
    for n, row in enumerate(rows):
        reference = parents[row['parent_index']]['reference']
        assert tokenizer.convert_ids_to_tokens(ids[n, 1:len(reference) + 1]) == list(reference)
        base[n, row['sites']] = tokenizer.mask_token_id
    assert ((base == tokenizer.mask_token_id).sum(1) == 2).all()

    def batch(name, inputs, positions):
        begin = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=inputs, attention_mask=attention).logits
            assert torch.isfinite(logits).all()
            for n, row in enumerate(rows):
                row['probabilities'][name] = [torch.softmax(logits[n, position, aa_ids], dim=-1).tolist() for position in positions[n]]
        result['forward_calls'] += 1
        result['input_rows'] += len(rows)
        result['batches'].append({'name': name, 'seconds': time.monotonic() - begin, 'padded_tokens': inputs.numel(), 'nonpadding_tokens': int(attention.sum())})
        checkpoint()

    batch('independent', base, [r['sites'] for r in rows])
    batch('repeat', base, [r['sites'] for r in rows])
    initial = [[aa_ids[choice(p)] for p in r['probabilities']['independent']] for r in rows]
    for name, reverse, oracle, control in [('increasing', False, False, False), ('reverse', True, False, False),
                                         ('oracle_increasing', False, True, False), ('oracle_reverse', True, True, False),
                                         ('content_control', False, False, True)]:
        inputs, positions = base.clone(), []
        for n, row in enumerate(rows):
            first, second = (1, 0) if reverse else (0, 1)
            first_site, second_site = row['sites'][first], row['sites'][second]
            inserted = int(ids[n, first_site]) if oracle else initial[n][first]
            if control:
                inserted = initial[(n // 4) * 4 + (n % 4 + 1) % 4][0]
                row['content_inserted'] = tokenizer.convert_ids_to_tokens(inserted)
            inputs[n, first_site] = inserted
            assert int(inputs[n, second_site]) == tokenizer.mask_token_id
            positions.append([second_site])
        assert ((inputs == tokenizer.mask_token_id).sum(1) == 1).all()
        batch(name, inputs, positions)
    assert result['forward_calls'] == 7 and result['input_rows'] == 112
    result['summary'] = summarize(result)
    result['status'] = 'complete'
    checkpoint()
    timer.cancel()
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'], 'summary': result['summary']}))


if __name__ == '__main__':
    main()
