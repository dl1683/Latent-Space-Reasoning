"""Bounded cached-export integrity probe. Never loads a model or fits a map."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile

import numpy as np

KNOWN_LAYOUTS = {
    'dee40062149055b2eb933961246ff64f4b5bb9010a49d77c7ac7917291af898e',
    '86077fc43515ff5df0bdaf9c44e6e80d5f7342c942268ac7d7fbad1a83c22162',
}


def inspect(path, rows, dim):
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Probe only accepts archives below 128 MiB")
    fields = {}
    query_ids = None
    with zipfile.ZipFile(path) as archive:
        meta_name, = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        metadata = archive.read(meta_name)
        if hashlib.sha256(metadata).hexdigest() not in KNOWN_LAYOUTS:
            raise ValueError('Unreviewed storage mapping, shape, offset or stride')
        if (rows, dim) != (13464, 1024):
            raise ValueError('Pinned metadata requires the reviewed shape')
        byteorder_name, = [n for n in archive.namelist() if n.endswith('/byteorder')]
        if archive.read(byteorder_name) != b'little':
            raise ValueError('Expected reviewed little-endian storage')
        globals_ = [arg for op, arg, _ in pickletools.genops(metadata)
                    if op.name == 'GLOBAL']
        if 'torch HalfStorage' not in globals_ or 'torch FloatStorage' not in globals_:
            raise ValueError("Expected inspected legacy half-feature/float-score schema")
        for field, storage, dtype, width in [
            ('query', '0', '<f2', dim), ('plan', '1', '<f2', dim),
            ('score', '2', '<f4', 1),
        ]:
            name, = [n for n in archive.namelist() if n.endswith('/data/' + storage)]
            expected = rows * width * np.dtype(dtype).itemsize
            if archive.getinfo(name).file_size != expected:
                raise ValueError(f"Unexpected storage shape for {field}")
            raw = archive.read(name)
            values = np.frombuffer(raw, dtype=dtype).reshape(rows, width)
            finite = np.isfinite(values)
            valid = finite.all(axis=1)
            stride = expected // rows
            hashes = [hashlib.sha256(raw[i*stride:(i+1)*stride]).hexdigest()
                      for i in range(rows)]
            counts = Counter(hashes)
            fields[field] = {
                'storage_sha256': hashlib.sha256(raw).hexdigest(),
                'finite_rows': int(valid.sum()),
                'any_nonfinite_rows': int((~valid).sum()),
                'all_nonfinite_rows': int((~finite).all(axis=1).sum()),
                'all_zero_rows': int((values == 0).all(axis=1).sum()),
                'unique_exact_rows': len(counts),
                'largest_exact_group': max(counts.values()),
                'finite_unique_exact_rows': len({h for h, ok in zip(hashes, valid) if ok}),
            }
            if field == 'query':
                query_ids = [h if ok else None for h, ok in zip(hashes, valid)]
    return {'path': path.as_posix(), 'bytes': path.stat().st_size,
            'metadata_sha256': hashlib.sha256(metadata).hexdigest(),
            'fields': fields}, query_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', type=Path, nargs=2, required=True)
    parser.add_argument('--rows', type=int, default=13464)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspected = [inspect(p, args.rows, args.dim) for p in args.archives]
    shared = [i for i, pair in enumerate(zip(*(x[1] for x in inspected)))
              if all(h is not None for h in pair)]
    partitions = []
    for _, ids in inspected:
        first = {}
        partitions.append([first.setdefault(ids[i], i) for i in shared])
    result = {'probe': 'cached_export_integrity_v1', 'model_calls': 0,
              'schema_validation': 'reviewed pickle metadata hashes and byteorder pinned',
              'rows': args.rows, 'dimension': args.dim,
              'archives': [x[0] for x in inspected],
              'both_finite_query_rows': len(shared),
              'matching_repeat_partition_rows_on_shared_finite':
                  sum(a == b for a, b in zip(*partitions)),
              'scope': 'Integrity only; no row identity, semantic or capability claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
