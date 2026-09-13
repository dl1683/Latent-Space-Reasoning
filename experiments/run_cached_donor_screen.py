"""Locked four-assay output-port screen using bounded official archive ranges."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import time
import urllib.request
import zipfile
import zlib

BASE = 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/'
ASSAYS = ['AMFR_HUMAN_Tsuboyama_2023_4G3O', 'ARGR_ECOLI_Tsuboyama_2023_1AOY',
          'BBC1_YEAST_Tsuboyama_2023_1TG0', 'CBX4_HUMAN_Tsuboyama_2023_2K28']
ARCHIVES = [('DMS_ProteinGym_substitutions.zip', 43021128, '"2907348-6329ff6970941"'),
            ('zero_shot_substitutions_scores.zip', 1911045703, '"71e83e47-633cd6a03f0f6"')]
DONORS = ['ESM2_8M', 'CARP_600K']
FIELDS = ['mutant', 'mutated_sequence', 'DMS_score'] + DONORS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_range(archive, start, length):
    name, total, etag = archive
    assert 0 < length < 4_000_000 and 0 <= start < start + length <= total
    request = urllib.request.Request(BASE + name, headers={
        'Range': f'bytes={start}-{start+length-1}', 'If-Match': etag})
    with urllib.request.urlopen(request, timeout=20) as response:
        assert response.status == 206, 'Server must honor bounded ranges'
        assert response.headers['Content-Range'] == f'bytes {start}-{start+length-1}/{total}'
        assert response.headers['ETag'] == etag
        raw = response.read(length+1)
    assert len(raw) == length
    return raw


def member_index(archive):
    tail_size = 65557
    tail = fetch_range(archive, archive[1]-tail_size, tail_size)
    with zipfile.ZipFile(io.BytesIO(tail)) as container:
        return {Path(x.filename).stem: {
            'name': x.filename, 'offset': x.header_offset+archive[1]-tail_size,
            'compressed_bytes': x.compress_size, 'bytes': x.file_size,
            'crc32': x.CRC, 'method': x.compress_type,
        } for x in container.infolist() if Path(x.filename).stem in ASSAYS}


def read_member(archive, info):
    header = fetch_range(archive, info['offset'], 30)
    h = struct.unpack('<4s5H3L2H', header)
    assert h[0] == b'PK\x03\x04' and h[3] == info['method'] == 8
    variable_size = h[-2] + h[-1]
    raw = fetch_range(archive, info['offset']+30, variable_size+info['compressed_bytes'])
    assert raw[:h[-2]].decode('utf-8') == info['name']
    compressed = raw[variable_size:]
    assert info['bytes'] < 8_000_000
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(compressed, info['bytes']+1)
    assert len(data) == info['bytes'] and decoder.eof and not decoder.unused_data
    assert zlib.crc32(data) == info['crc32']
    provenance = dict(info, archive_url=BASE+archive[0], archive_etag=archive[2],
                      compressed_sha256=sha(compressed), uncompressed_sha256=sha(data))
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))), provenance


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0]*len(values)
    start = 0
    while start < len(ordered):
        end = start+1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = start+end-1  # Doubled midrank: preserve exact ties before aggregation.
        start = end
    return result


def spearman(a, b):
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None  # Undefined correlation, never classify as zero or a loss.
    return statistics.correlation(ranks(a), ranks(b))


def scores(retained):
    target = [float(x['DMS_score']) for x in retained]
    donors = [[float(x[c]) for x in retained] for c in DONORS]
    single = [spearman(x, target) for x in donors]
    # Positive common scaling cannot affect ordering; avoid floating-point tie splitting.
    average = [a+b for a, b in zip(*(ranks(x) for x in donors))]
    combined = spearman(average, target)
    eligible = combined is not None and all(x is not None for x in single)
    return {'individual_spearman': dict(zip(DONORS, single)), 'combined_spearman': combined,
            'correlation_status': 'defined' if eligible else 'undefined_constant_column',
            'gain_vs_each': dict(zip(DONORS, [combined-x if combined is not None and x is not None else None for x in single])),
            'gain_vs_retrospective_better': combined-max(single) if eligible else None}


def recalculate(directory):
    path = directory/'result.json'
    original = path.read_bytes()
    result = json.loads(original)
    assert result['status'] == 'complete' and 'numerical_correction' not in result
    result['numerical_correction'] = {
        'original_result_sha256': sha(original), 'original_runner_sha256': result['runner_sha256'],
        'reason': 'Preserve mathematical ties using integer doubled-midrank sums',
        'source_commit': '0f61942', 'network_requests': 0, 'model_calls': 0}
    for assay in result['assays']:
        data = Path(assay['subset_path']).read_bytes()
        assert sha(data) == assay['subset_sha256']
        rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
        assay['original_combined_spearman'] = assay['combined_spearman']
        assay.update(scores(rows))
    result['runner_sha256'] = sha(Path(__file__).read_bytes())
    temporary = directory/'result.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps([{k: row[k] for k in ['assay', 'combined_spearman', 'gain_vs_retrospective_better']}
                      for row in result['assays']]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--recalculate', action='store_true', help='Apply reviewed tie correction offline to retained data')
    args = parser.parse_args()
    if args.recalculate:
        recalculate(args.output_dir)
        return
    assert not args.output_dir.exists(), 'Preserve prior attempts'
    start = time.monotonic()
    indices = [member_index(a) for a in ARCHIVES]
    assert all(set(index) == set(ASSAYS) for index in indices)
    args.output_dir.mkdir(parents=True)
    result = {'screen': 'cached_donor_screen_v1', 'status': 'running',
              'runner_sha256': sha(Path(__file__).read_bytes()), 'model_calls': 0,
              'training_runs': 0, 'donor_columns': DONORS, 'score_orientation': [1, 1],
              'rule': 'Equal average of within-assay ascending midrank percentiles; Spearman endpoint',
              'scope': 'Four descriptive assay units; no retention, internal acquisition, biological intervention or cost claim',
              'assays': []}

    def checkpoint():
        result['wall_seconds'] = time.monotonic()-start
        temporary = args.output_dir/'result.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(args.output_dir/'result.json')

    checkpoint()
    for assay in ASSAYS:
        (truth, truth_prov), (predictions, pred_prov) = [
            read_member(a, index[assay]) for a, index in zip(ARCHIVES, indices)]
        key = lambda row: (row['mutant'], row['mutated_sequence'])
        truth_map = {key(row): row for row in truth}
        pred_map = {key(row): row for row in predictions}
        assert len(truth_map) == len(truth) and len(pred_map) == len(predictions), 'Duplicate identity'
        assert all(column in predictions[0] for column in FIELDS)
        common = sorted(set(truth_map) & set(pred_map))
        retained, invalid = [], 0
        for item in common:
            row = pred_map[item]
            try:
                values = [float(row[c]) for c in FIELDS[2:]]
                target = float(truth_map[item]['DMS_score'])
            except (ValueError, TypeError):
                invalid += 1
                continue
            if not all(math.isfinite(x) for x in values+[target]):
                invalid += 1
                continue
            assert math.isclose(values[0], target, rel_tol=0, abs_tol=1e-7), 'Mismatched measured endpoint'
            retained.append({c: row[c] for c in FIELDS})
        assert len(retained) >= 10, 'Insufficient common support'
        metrics = scores(retained)
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(retained)
        subset = stream.getvalue().encode('utf-8')
        subset_path = args.output_dir/(assay+'.csv')
        subset_path.write_bytes(subset)
        result['assays'].append({
            'assay': assay, 'truth_rows': len(truth), 'prediction_rows': len(predictions),
            'common_identity_rows': len(common), 'joint_finite_rows': len(retained),
            'invalid_common_rows': invalid, 'truth_without_prediction': len(set(truth_map)-set(pred_map)),
            'prediction_without_truth': len(set(pred_map)-set(truth_map)),
            **metrics,
            'subset_path': subset_path.as_posix(), 'subset_sha256': sha(subset),
            'truth_provenance': truth_prov, 'prediction_provenance': pred_prov,
        })
        checkpoint()
        print(json.dumps({k: v for k, v in result['assays'][-1].items() if 'provenance' not in k}), flush=True)
    result['status'] = 'complete'
    checkpoint()


if __name__ == '__main__':
    main()
