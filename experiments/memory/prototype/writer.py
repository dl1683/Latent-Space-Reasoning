"""Question-blind native-state writer (Z Code B, offline mechanics).

The writer sees documents and revisions ONLY â€” never queries (enforced
by signature). It produces up to 16 contextual rows (four per record,
four records per document) plus value-free sidecars. On the cloud path
the encoder is the pinned frozen stem's last-four-record-token hidden
rows and the trained resampler compresses to eight receiver prefix
slots; offline this module carries the same STRUCTURE with a
deterministic hashed-feature encoder explicitly labelled
mechanics-only (never interpreted as a model path). Revision re-encodes
the replacement record's rows only (exact addressing and re-encoding
costs are charged; this is not learned revision).
"""
import hashlib
import json
import time

from . import data

ROWS_PER_RECORD = 4
MAX_ROWS = 16
PREFIX_SLOTS = 8
OFFLINE_WIDTH = 64
_ENCODER_LABEL = "offline-frozen-hashed-mechanics"


def _features(text, salt, width):
    """Deterministic float features from a byte hash. A fixed function
    of content (rows change iff content changes) holding no source
    text, tokens or KV."""
    out = []
    for j in range(width):
        h = hashlib.sha256(f"{salt}|{j}|{text}".encode()).digest()
        out.append((int.from_bytes(h[:4], "big") / 2 ** 32) * 2.0 - 1.0)
    return out


def encode_record(record, width=OFFLINE_WIDTH):
    """Four question-blind rows for one record."""
    return [_features(record["text"], f"r{record['record_id']}.{j}", width)
            for j in range(ROWS_PER_RECORD)]


def write_document(doc, width=OFFLINE_WIDTH):
    """Native state for a whole document (four records -> 16 rows)."""
    rows, meta = [], []
    for rec in doc["records"]:
        for j, row in enumerate(encode_record(rec, width)):
            if len(rows) >= MAX_ROWS:
                break
            rows.append(row)
            meta.append({"record_id": rec["record_id"], "row": j,
                         "kind": "record"})
    return {
        "encoder": _ENCODER_LABEL, "width": width,
        "doc_len": len(doc["records"]),
        "rows": rows, "row_meta": meta,
        "sidecars": {
            "n_records": len(doc["records"]),
            "n_rows": len(rows),
            "prefix_slots": PREFIX_SLOTS,
            "encoder_label": _ENCODER_LABEL,
        },
    }


def revise_document(prev_native, revision, width=OFFLINE_WIDTH):
    """Re-encode ONLY the replacement record's rows; every other row is
    carried over byte-identically (charged exact re-encoding)."""
    if prev_native is None:
        raise ValueError("revise needs the previous native state")
    rows = [list(r) for r in prev_native["rows"]]
    meta = [dict(m) for m in prev_native["row_meta"]]
    keep = [i for i, m in enumerate(meta)
            if m.get("record_id") != revision["record_id"]]
    new_rows = encode_record(revision, width)
    rebuilt = []
    for m, row in zip([meta[i] for i in keep], [rows[i] for i in keep]):
        rebuilt.append((m, row))
    for j, row in enumerate(new_rows):
        rebuilt.append(({"record_id": revision["record_id"], "row": j,
                         "kind": "revision"}, row))
    rebuilt.sort(key=lambda t: (t[0]["record_id"], t[0]["row"]))
    out_rows = [r for _, r in rebuilt]
    out_meta = [m for m, _ in rebuilt]
    side = dict(prev_native["sidecars"])
    side["n_rows"] = len(out_rows)
    side["revised_record"] = revision["record_id"]
    return {
        "encoder": _ENCODER_LABEL, "width": width,
        "doc_len": prev_native["doc_len"],
        "rows": out_rows, "row_meta": out_meta,
        "sidecars": side,
    }


def validate_no_retention(native_state, records=(), questions=(),
                          answers=()):
    """Reject any native state that retains source text, question text
    or answer values as literal strings (rows are floats; ids are hex;
    a clean state cannot contain any of the 32 value words). Returns
    [] (clean) or a list of findings."""
    blob = json.dumps(native_state, sort_keys=True)
    bad = []
    for rec in records or ():
        if rec.get("text") and rec["text"] in blob:
            bad.append(f"record text retained: {rec['record_id']}")
    for q in questions or ():
        if isinstance(q, dict):
            if q.get("text", "") in blob:
                bad.append("question text retained")
        elif isinstance(q, str) and q in blob:
            bad.append("question text retained")
    for a in answers or ():
        if isinstance(a, str) and len(a) >= 2 and a in blob:
            bad.append(f"answer value retained: {a!r}")
    if not answers:  # sweep the whole value vocabulary when unset
        for w in data.WORDS:
            if w in blob:
                bad.append(f"value word {w!r} inside native state")
                break
    return bad


def reader_bundle(native_state, question_text, records=()):
    """What a (restarted) reader process may receive: query + native
    state + value-free sidecars (deep-copied â€” a reader-side mutation
    cannot corrupt the writer's state). No records, tokens, KV or
    truth. The no-retention check runs on the native state BEFORE the
    query is attached (the query itself is the reader's own input)."""
    leaks = validate_no_retention(native_state, records=records,
                                  questions=[question_text])
    bundle = {
        "question": question_text,
        "rows": [list(r) for r in native_state["rows"]],
        "row_meta": [dict(m) for m in native_state["row_meta"]],
        "sidecars": json.loads(json.dumps(native_state["sidecars"])),
        "encoder": native_state["encoder"],
    }
    return bundle, leaks


class ContextualWriter:
    """Question-blind frozen-stem extraction; never retains source tokens/KV.

    The metadata is generated from physical addresses only. In particular a hash
    of a one-word fact is NOT exported as provenance (a dictionary attack leaks it).
    The canonical envelope digest binds the actual rows instead.
    """
    def __init__(self, handle):
        self.handle = handle

    def _clock(self):
        import torch
        device = self.handle['receiver'].stem.get_input_embeddings().weight.device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _charge(self, operation, started, tokens, rows):
        costs = self.handle.setdefault('_writer_costs', {})
        for key, value in ((operation+'_seconds', self._clock()-started),
                           (operation+'_input_tokens', tokens),
                           (operation+'_rows', rows), (operation+'_calls', 1)):
            costs[key] = costs.get(key, 0) + value

    def write(self, records):
        import torch
        from experiments.cloud.lsr_jobs import s0_envelope as envelope
        started = self._clock()
        stem = self.handle['receiver'].stem
        width = self.handle['stem_info']['D_memory']
        if width != stem.config.hidden_size or len(records) != 4:
            raise ValueError('writer requires four records and last-layer native width')
        rows, input_tokens = [], 0
        for address, record in enumerate(records):
            if self.handle.get('_progress'):
                self.handle['_progress']('ingestion_record_start')
            if type(record.get('record_id')) is not int or record['record_id'] != address:
                raise ValueError('record addresses must be canonical 0..3')
            tokens = self.handle['tok'](record['text'], add_special_tokens=False,
                                       return_tensors='pt')
            ids = tokens['input_ids'].to(stem.get_input_embeddings().weight.device)
            input_tokens += ids.shape[1]
            if ids.shape[1] < 4:
                raise ValueError('short record: at least four visible tokens required')
            with torch.no_grad():
                out = stem(input_ids=ids, attention_mask=torch.ones_like(ids),
                           output_hidden_states=True, use_cache=False, return_dict=True)
                rows.append(out.hidden_states[-1][0, -4:].detach().float().cpu())
            if self.handle.get('_progress'):
                self.handle['_progress']('ingestion_record_complete')
        payload = torch.cat(rows)
        # Same addresses/provenance for all counterfactual contents and revisions.
        origins = [{'origin_type': 'observation', 'modality': 'text',
                    'source_sha256': hashlib.sha256(f'address:{i//4}'.encode()).hexdigest(),
                    'locator': f'record:{i//4}:row:{i%4}', 'event_time': 0,
                    'ingestion_time': 0, 'valid_from': 0, 'valid_until': None,
                    'branch': 'main', 'dependencies': [], 'permissions': ['read']}
                   for i in range(16)]
        result = envelope.make_envelope(payload, origin=origins,
                    temporal_horizon={'as_of': 0, 'max_age': None},
                    expected_width=width, expected_schema=envelope.WIDTH_SCHEMA_VERSION)
        self._charge('write', started, input_tokens, 16)
        return result

    def revise(self, previous, record_id, replacement):
        """Host-addressed re-encoding; unaffected rows remain byte-identical."""
        import torch
        from experiments.cloud.lsr_jobs import s0_envelope as envelope
        started = self._clock()
        width = self.handle['stem_info']['D_memory']
        envelope.verify_envelope(previous, expected_width=width,
                                  expected_schema=envelope.WIDTH_SCHEMA_VERSION)
        if type(record_id) is not int or not 0 <= record_id < 4 \
                or replacement.get('record_id') != record_id:
            raise ValueError('invalid replacement address')
        stem = self.handle['receiver'].stem
        ids = self.handle['tok'](replacement['text'], add_special_tokens=False,
                    return_tensors='pt')['input_ids'].to(stem.get_input_embeddings().weight.device)
        if ids.shape[1] < 4:
            raise ValueError('short replacement')
        with torch.no_grad():
            new = stem(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       output_hidden_states=True, return_dict=True).hidden_states[-1][0, -4:]
        old_rows = envelope.to_torch(previous['payload']) if not isinstance(previous['payload'], torch.Tensor) else previous['payload']
        rows = old_rows.detach().clone()
        rows[record_id*4:(record_id+1)*4] = new.detach().to(rows)
        result = envelope.make_envelope(rows, origin=previous['meta']['origin'],
                    temporal_horizon=previous['meta']['temporal_horizon'],
                    expected_width=width, expected_schema=envelope.WIDTH_SCHEMA_VERSION)
        self._charge('revise', started, ids.shape[1], 4)
        return result
