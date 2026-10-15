"""Versioned memory store (Z Code B, model-free offline mechanics).

API per the brief: write(records)->id, revise(id, record_id,
replacement)->new_id, query(id, question, budget)->answer, save/load.
Immutable historical snapshots (revise never mutates its parent);
opaque value-free snapshot ids; provenance is operation kind + logical
counter only (no wall-clock, no value-derived ids).

Access model: reader access is bound by ROLE, not by name strings.
The records view (exact storage) is reserved for the dictionary and
storage roles; the candidate role can never register it regardless of
what a reader spec requests. Readers receive deep copies and the
snapshot digest is re-verified on every query, so a reader cannot
mutate or corrupt history through the dispatch path. A save file
carries a payload digest covering its allowlist; a restarted store
enforces the allowlist that travelled with it. Malformed state
rejects without mutation.
"""
import copy
import hashlib
import json
import os
import secrets


class StoreError(Exception):
    pass


VIEWS = ("records", "native_state", "sidecars")

# role-bound view permissions: the candidate path can never hold the
# exact records; only the exact-storage roles can. Unknown reader
# names MAY register native_state/sidecars views (diagnostic readers);
# the records view is the only reserved one — by design, since
# candidate-class views leak no record content.
VIEW_PERMISSIONS = {
    "dictionary": ("records",),
    "storage": ("records",),
    "candidate": ("native_state", "sidecars"),
}


class ReaderSpec:
    """Named reader with an explicit view allowlist, validated against
    its ROLE permissions at registration (fail closed)."""

    def __init__(self, name, views, fn):
        unknown = [v for v in views if v not in VIEWS]
        if unknown:
            raise StoreError(f"unknown views {unknown}")
        role_perms = VIEW_PERMISSIONS.get(name)
        if role_perms is not None:
            excess = [v for v in views if v not in role_perms]
            if excess:
                raise StoreError(
                    f"reader role {name!r} may not request views "
                    f"{excess} (permitted: {list(role_perms)})")
        elif "records" in views:
            # unlisted names fail closed: exact records are reserved
            # for the dictionary/storage roles
            raise StoreError(
                f"reader name {name!r} is not an exact-storage role; "
                "the records view is denied")
        self.name = name
        self.views = tuple(views)
        self.fn = fn


def dictionary_reader():
    """Exact per-slot lookup over the records view (the exact storage
    comparator). Question carries name+slot; the snapshot's structured
    record index answers directly."""

    def fn(view, question, budget):
        recs = view["records"]
        for rec in recs:
            if rec["slot"] == question["slot"]:
                return rec["value"]
        raise StoreError(f"no record for slot {question['slot']!r}")

    return ReaderSpec("dictionary", ("records",), fn)


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def _digest(payload):
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


# frozen native-state schema: anything outside these keys/types is a
# potential answer side channel (numeric indices, encoded strings,
# extra ids) and is rejected at the store door
NATIVE_TOP_KEYS = {"encoder": str, "width": int, "doc_len": int,
                   "rows": list, "row_meta": list, "sidecars": dict}
SIDECAR_SCHEMA = {
    "n_records": int, "n_rows": int, "prefix_slots": int,
    "encoder_label": str, "revised_record": int,
}
ROW_META_SCHEMA = {"record_id": int, "row": int, "kind": str}


def _validate_native_state(native_state):
    """Mechanics-level shape + schema + retention check (the cloud
    path rebinds shape checking to the canonical envelope API): rows
    of equal-width finite floats, aligned row_meta dicts, a FROZEN
    sidecar schema (unknown keys or non-scalar values rejected — no
    numeric index or encoded side channels), and NO stowaway source
    or answer strings anywhere in the state."""
    if not isinstance(native_state, dict):
        raise StoreError("native_state must be a dict")
    unknown_top = [k for k in native_state if k not in NATIVE_TOP_KEYS]
    if unknown_top:
        raise StoreError(
            f"native_state carries unknown fields {unknown_top} "
            "(potential side channel)")
    for key, kind in NATIVE_TOP_KEYS.items():
        if key in native_state and not isinstance(
                native_state[key], kind):
            raise StoreError(f"native_state.{key} must be {kind.__name__}")
    rows = native_state.get("rows")
    meta = native_state.get("row_meta")
    if not isinstance(rows, list) or not rows:
        raise StoreError("native_state.rows must be a non-empty list")
    width = len(rows[0])
    if width == 0:
        raise StoreError("empty native rows")
    for row in rows:
        if not isinstance(row, list) or len(row) != width:
            raise StoreError("ragged native rows")
        for x in row:
            if not isinstance(x, (int, float)) or x != x \
                    or x in (float("inf"), float("-inf")):
                raise StoreError("non-finite or non-numeric row entry")
    if meta is not None:
        if not isinstance(meta, list) or len(meta) != len(rows):
            raise StoreError("row_meta must align with rows")
        for m in meta:
            if not isinstance(m, dict):
                raise StoreError("row_meta entries must be dicts")
            bad = [k for k in m if k not in ROW_META_SCHEMA]
            if bad:
                raise StoreError(
                    f"row_meta carries unknown fields {bad}")
            for k, kind in ROW_META_SCHEMA.items():
                if k in m and not isinstance(m[k], kind):
                    raise StoreError(f"row_meta.{k} must be "
                                     f"{kind.__name__}")
    sidecars = native_state.get("sidecars")
    if sidecars is not None:
        bad_keys = [k for k in sidecars if k not in SIDECAR_SCHEMA]
        if bad_keys:
            raise StoreError(
                f"sidecars carry unknown fields {bad_keys} "
                "(value-free metadata only; numeric-index and "
                "encoded side channels are rejected)")
        for k, kind in SIDECAR_SCHEMA.items():
            v = sidecars.get(k)
            if k in sidecars and not isinstance(v, kind):
                raise StoreError(f"sidecars.{k} must be "
                                 f"{kind.__name__}")
            if kind is str and isinstance(v, str) and len(v) > 64:
                raise StoreError(f"sidecars.{k} string too long")
    # retention sweep at the store door: hex ids and numbers cannot
    # contain value words; any hit means source/answer text stowed away
    blob = _canonical(native_state)
    from . import data as _data
    for w in _data.WORDS:
        if w in blob:
            raise StoreError(
                f"value word {w!r} retained inside native_state")
    for value in _walk_strings(native_state):
        if len(value) > 64:
            raise StoreError("suspicious long string in native_state")


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_strings(item)


class MemoryStore:
    def __init__(self, readers=(), allowlist=None):
        self._snapshots = {}
        self._readers = {}
        for r in readers:
            self._readers[r.name] = r   # ReaderSpec validates its role
        # allowlist travels with save files; None = every registered name
        self._allowlist = set(allowlist) if allowlist is not None else None
        self._time = 0

    # -- snapshot internals -------------------------------------------
    def _new_id(self):
        # opaque random hex: carries no record content or value
        return "snap-" + secrets.token_hex(8)

    def _commit(self, snapshot):
        snap = dict(snapshot)
        body = {k: snap[k] for k in snap if k != "digest"}
        snap["digest"] = _digest(body)
        self._snapshots[snap["snapshot_id"]] = snap
        self._time += 1
        return snap["snapshot_id"]

    # -- public API ----------------------------------------------------
    def write(self, records, native_state=None):
        """records: list of {record_id, slot, value, text} (positional
        ids). native_state: writer-produced candidate rows (optional on
        the offline path; the dictionary/storage arms never need it)."""
        _validate_records(records)
        if not isinstance(records, list) or not records:
            raise StoreError("write needs a non-empty record list")
        ids = [r.get("record_id") for r in records]
        if ids != list(range(len(records))):
            raise StoreError("record ids must be positional 0..n-1")
        for r in records:
            if not {"slot", "value", "text"} <= set(r):
                raise StoreError("records need slot/value/text fields")
        if native_state is not None:
            _validate_native_state(native_state)
        snap = {
            "snapshot_id": self._new_id(),
            "parent_id": None,
            "version": 1,
            "logical_time": self._time,
            "provenance": {"op": "write", "record_id": None},
            "records": copy.deepcopy(records),
            "native_state": copy.deepcopy(native_state),
        }
        return self._commit(snap)

    def revise(self, snapshot_id, record_id, replacement, native_state=None):
        """replacement: the full replacement record ({record_id, slot,
        value, text}); it enters the writer only. Returns a NEW snapshot
        id; the parent is left byte-identical."""
        parent = self._get(snapshot_id)
        if not isinstance(record_id, int) \
                or not 0 <= record_id < len(parent["records"]):
            raise StoreError(f"bad record_id {record_id!r}")
        if not isinstance(replacement, dict) \
                or not {"slot", "value", "text"} <= set(replacement):
            raise StoreError("replacement must be a full record")
        if replacement.get("record_id") not in (None, record_id):
            raise StoreError("replacement record_id mismatch")
        if parent["native_state"] is not None and native_state is None:
            raise StoreError("candidate revision requires validated replacement native rows")
        if native_state is not None:
            _validate_native_state(native_state)
        records = copy.deepcopy(parent["records"])
        new_rec = dict(replacement)
        new_rec["record_id"] = record_id
        records[record_id] = new_rec
        _validate_records(records)
        snap = {
            "snapshot_id": self._new_id(),
            "parent_id": parent["snapshot_id"],
            "version": parent["version"] + 1,
            "logical_time": self._time,
            "provenance": {"op": "revise", "record_id": record_id},
            "records": records,
            "native_state": copy.deepcopy(native_state)
            if native_state is not None
            else copy.deepcopy(parent["native_state"]),
        }
        return self._commit(snap)

    def query(self, snapshot_id, question, budget=None, reader=None):
        """Dispatch to a NAMED reader (explicit reader is required),
        exposing ONLY the role-permitted views, deep-copied, after
        re-verifying the snapshot digest."""
        if reader is None:
            raise StoreError(
                "reader name required (no silent default dispatch)")
        snap = self._get(snapshot_id)
        body = {k: snap[k] for k in snap if k != "digest"}
        if _digest(body) != snap["digest"]:
            raise StoreError(
                f"snapshot {snapshot_id!r} digest mismatch on dispatch")
        if reader not in self._readers:
            raise StoreError(f"reader {reader!r} not registered")
        if self._allowlist is not None and reader not in self._allowlist:
            raise StoreError(f"reader {reader!r} not in restarted allowlist")
        spec = self._readers[reader]
        snap_copy = copy.deepcopy(snap)
        view = {}
        if "records" in spec.views:
            view["records"] = snap_copy["records"]
        if "native_state" in spec.views:
            view["native_state"] = snap_copy["native_state"]
        if "sidecars" in spec.views:
            view["sidecars"] = (snap_copy["native_state"] or {}).get(
                "sidecars")
        return spec.fn(view, question, budget)

    def snapshot_bytes(self, snapshot_id):
        """Canonical serialization for byte-immutability checks."""
        return _canonical(self._get(snapshot_id)).encode()

    def version_of(self, snapshot_id):
        return self._get(snapshot_id)["version"]

    def _get(self, snapshot_id):
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            raise StoreError(f"unknown snapshot {snapshot_id!r}")
        return snap

    # -- persistence ----------------------------------------------------
    def save(self, path):
        payload = {
            "format": "zcode-b-store/1",
            "allowlist": sorted(
                self._readers) if self._allowlist is None
            else sorted(self._allowlist),
            "time": self._time,
            "snapshots": list(self._snapshots.values()),
        }
        payload["payload_digest"] = _digest(
            {k: payload[k] for k in payload})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(_canonical(payload))
        os.replace(tmp, path)
        return payload["payload_digest"]

    @classmethod
    def load(cls, path, readers=()):
        """Restart: validate EVERYTHING into a scratch structure first;
        any malformed content raises without mutating anything. The
        stored allowlist (digest-protected) becomes binding for
        subsequent queries."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        payload = json.loads(raw)  # malformed JSON raises before commit
        try:
            if payload.get("format") != "zcode-b-store/1":
                raise StoreError("unknown store format")
            stored_digest = payload["payload_digest"]
            body = {k: payload[k] for k in payload
                    if k != "payload_digest"}
            snapshots = payload["snapshots"]
            allowlist = payload["allowlist"]
            time = payload["time"]
        except KeyError as exc:
            raise StoreError(f"store file missing field: {exc}") from exc
        if _digest(body) != stored_digest:
            raise StoreError("payload digest mismatch (tampered file?)")
        if not isinstance(snapshots, list) or not snapshots:
            raise StoreError("store file holds no snapshots")
        if not isinstance(allowlist, list):
            raise StoreError("allowlist must be a list")
        seen = {}
        for snap in snapshots:
            try:
                inner = {k: snap[k] for k in snap if k != "digest"}
                sid = snap["snapshot_id"]
                parent = snap["parent_id"]
            except KeyError as exc:
                raise StoreError(
                    f"snapshot missing field: {exc}") from exc
            if _digest(inner) != snap.get("digest"):
                raise StoreError(f"snapshot {sid!r} digest mismatch")
            if sid in seen:
                raise StoreError("duplicate snapshot id")
            seen[sid] = snap
        for snap in snapshots:
            if snap["parent_id"] is not None \
                    and snap["parent_id"] not in seen:
                raise StoreError("dangling parent link")
        if type(time) is not int or time < 0 or time != len(snapshots):
            raise StoreError('invalid logical clock')
        if len(set(allowlist)) != len(allowlist) or any(x not in VIEW_PERMISSIONS for x in allowlist):
            raise StoreError('invalid stored reader allowlist')
        clocks = set()
        for snap in snapshots:
            _validate_records(snap.get('records'))
            if snap.get('native_state') is not None:
                _validate_native_state(snap['native_state'])
            clock, version = snap.get('logical_time'), snap.get('version')
            if type(clock) is not int or not 0 <= clock < time or clock in clocks or type(version) is not int:
                raise StoreError('invalid snapshot clock/version')
            clocks.add(clock)
            parent = seen.get(snap['parent_id'])
            if parent is None:
                if version != 1 or snap.get('provenance') != {'op':'write','record_id':None}:
                    raise StoreError('invalid root snapshot')
            else:
                rid = snap.get('provenance', {}).get('record_id')
                if (version != parent['version'] + 1 or clock <= parent['logical_time']
                        or snap['provenance'].get('op') != 'revise'
                        or type(rid) is not int or not 0 <= rid < 4):
                    raise StoreError('invalid revision ancestry')
                if any(snap['records'][i] != parent['records'][i] for i in range(4) if i != rid):
                    raise StoreError('revision modified unrelated record')
                if parent['native_state'] is not None and snap['native_state'] is None:
                    raise StoreError('revision lost candidate native state')
        store = cls(readers=readers, allowlist=set(allowlist))
        store._snapshots = seen
        store._time = time
        return store


# Closed schemas validate values as well as types: a numeric vocabulary index
# cannot hide inside an otherwise legal counter/record field.
_validate_native_structure = _validate_native_state


def _validate_native_state(native_state):
    from . import writer
    _validate_native_structure(native_state)
    if set(native_state) != set(NATIVE_TOP_KEYS):
        raise StoreError('native state has missing required fields')
    if (type(native_state['width']) is not int or native_state['width'] != writer.OFFLINE_WIDTH
            or native_state['encoder'] != writer._ENCODER_LABEL
            or type(native_state['doc_len']) is not int or native_state['doc_len'] != 4
            or len(native_state['rows']) != 16
            or any(len(row) != writer.OFFLINE_WIDTH for row in native_state['rows'])):
        raise StoreError('native width/encoder/capacity identity mismatch')
    side = native_state['sidecars']
    required = {'n_records','n_rows','prefix_slots','encoder_label'}
    if set(side) not in (required, required | {'revised_record'}):
        raise StoreError('native sidecar schema mismatch')
    if (side['n_records'] != 4 or side['n_rows'] != 16 or side['prefix_slots'] != 8
            or side['encoder_label'] != writer._ENCODER_LABEL
            or any(type(side[k]) is not int for k in ('n_records','n_rows','prefix_slots'))):
        raise StoreError('encoded/noncanonical count metadata')
    if 'revised_record' in side and (type(side['revised_record']) is not int or not 0 <= side['revised_record'] < 4):
        raise StoreError('invalid revised address')
    for i, meta in enumerate(native_state['row_meta']):
        if (set(meta) != set(ROW_META_SCHEMA) or type(meta['record_id']) is not int
                or type(meta['row']) is not int or meta['record_id'] != i // 4
                or meta['row'] != i % 4 or meta['kind'] not in ('record','revision')):
            raise StoreError('noncanonical row address/role')


def _validate_records(records):
    from . import data
    if not isinstance(records, list) or len(records) != 4:
        raise StoreError('prototype requires exactly four records')
    for i, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != {'record_id','slot','value','text'}
                or type(record['record_id']) is not int or record['record_id'] != i
                or record['slot'] != data.SLOTS[i] or record['value'] not in data.WORDS
                or not isinstance(record['text'], str) or not record['text']):
            raise StoreError('invalid record shape/address/value')
