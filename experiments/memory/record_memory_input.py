"""Lossless admission envelope for copied structured evidence, before encoding.

This preserves recorded provenance; it does not authenticate the original source,
perform semantic reasoning, or give instructions in records execution authority.
"""
import hashlib
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def encode_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty copied record list required')
    rows = []
    for i, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('text'), str):
            raise ValueError('each record needs a text string')
        # Reject values JSON would silently transform (e.g. integer dict keys).
        raw = canonical(record)
        if json.loads(raw) != record:
            raise ValueError('record must round trip through JSON exactly')
        rows.append(dict(local_id=f'D{i}', record=record,
                         record_sha256=digest(raw)))
    return canonical(dict(schema='copied-evidence-v1', role='evidence', records=rows))


def decode_records(raw, *, expected_sha256):
    if digest(raw) != expected_sha256:
        raise ValueError('evidence envelope checksum mismatch')
    envelope = json.loads(raw, object_pairs_hook=_unique_keys)
    if (set(envelope) != {'schema', 'role', 'records'}
            or envelope['schema'] != 'copied-evidence-v1'
            or envelope['role'] != 'evidence' or not envelope['records']):
        raise ValueError('invalid evidence envelope')
    records = []
    for i, row in enumerate(envelope['records']):
        if (set(row) != {'local_id', 'record', 'record_sha256'}
                or row['local_id'] != f'D{i}'
                or digest(canonical(row['record'])) != row['record_sha256']):
            raise ValueError('invalid record binding')
        records.append(row['record'])
    # Check both schema validation and canonical form at this boundary.
    if encode_records(records) != raw:
        raise ValueError('noncanonical evidence envelope')
    return records
