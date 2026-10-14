"""Typed data-only memory attachment. SHA-256 binds content, NOT authority.

Times are finite seconds in one host-chosen epoch; validity is [from, until).
Offsets remain unscaled seconds: refuse differences outside finite float64,
even with unbounded max_age. No saturation or implicit epoch conversion occurs.
The host owns policy; temporal_horizon is a selection receipt, not a grant.
Re-selection replaces that receipt, rather than intersecting prior horizons.
Each physical row has exactly one origin. Selection preserves physical order and
logical indices, and returns row-level refusals. No loader or device API is used.
"""
import argparse
import base64
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import sys

WIDTH = 5376
MEMORY_POSITIONS = 16
SCHEMA_VERSION = "s0-envelope/1"
WIDTH_SCHEMA_VERSION = "s0-envelope/2"
ROLE_IDS = {"observation": 0, "user_input": 1, "generated": 2, "derived": 3}
DTYPES = {"bool": 1, "uint8": 1, "int8": 1, "int16": 2, "int32": 4,
          "int64": 8, "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


class S0EnvelopeError(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise S0EnvelopeError(reason)


def _fields(obj, names, label):
    _require(isinstance(obj, dict) and set(obj) == set(names),
             label + ": missing or unknown fields")


def _time(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _offset(event_time, as_of):
    """Finite float64 seconds, shared by age policy and receiver conditioning."""
    try:
        result = float(event_time - as_of)
    except OverflowError as exc:
        raise S0EnvelopeError("nonfinite time offset") from exc
    _require(math.isfinite(result), "nonfinite time offset")
    return result


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _permissions(value):
    return (isinstance(value, list) and bool(value) and
            all(_text(v) for v in value) and len(set(value)) == len(value) and
            "read" in value)


@dataclass(frozen=True)
class NativeTensor:
    """Plain CPU tensor carrier, little-endian native IEEE/integer bytes."""
    dtype: str
    shape: tuple
    data: bytes

    def __post_init__(self):
        _require(isinstance(self.dtype, str) and self.dtype in DTYPES, "unsupported dtype")
        _require(isinstance(self.shape, tuple) and
                 all(type(n) is int and n >= 0 for n in self.shape), "bad shape")
        _require(isinstance(self.data, bytes) and
                 len(self.data) == math.prod(self.shape) * DTYPES[self.dtype],
                 "native byte length mismatch")


def native_tensor(tensor):
    """Snapshot dense strided logical values, resolving lazy negative views.

    Dtype and value bits are preserved; unsupported dtypes/layouts are refused.
    """
    if isinstance(tensor, NativeTensor):
        return tensor
    try:
        import torch
    except ImportError as exc:
        raise S0EnvelopeError("use NativeTensor when torch is unavailable") from exc
    _require(isinstance(tensor, torch.Tensor), "expected NativeTensor or torch.Tensor")
    _require(tensor.layout == torch.strided, "only dense strided tensors supported")
    _require(sys.byteorder == "little", "little-endian host required")
    t = tensor.detach().to(device="cpu").resolve_neg().contiguous()
    dtype = str(t.dtype).removeprefix("torch.")
    _require(dtype in DTYPES, "unsupported dtype")
    # Flatten BEFORE the byte view, including scalar and zero-element tensors.
    return NativeTensor(dtype, tuple(t.shape), bytes(t.reshape(-1).view(torch.uint8).tolist()))


def to_torch(tensor):
    """Reload a native carrier into an independent CPU torch tensor."""
    _require(sys.byteorder == "little", "little-endian host required")
    import torch
    t = native_tensor(tensor)
    dtype = getattr(torch, t.dtype)
    if not t.data:
        return torch.empty(t.shape, dtype=dtype, device="cpu")
    return torch.frombuffer(bytearray(t.data), dtype=dtype).clone().reshape(t.shape)


def _wire(t):
    t = native_tensor(t)
    return {"type": "native-tensor/1", "dtype": t.dtype, "shape": list(t.shape),
            "byte_order": "little", "data": base64.b64encode(t.data).decode("ascii")}


def _unwire(obj):
    _fields(obj, ("type", "dtype", "shape", "byte_order", "data"), "tensor")
    _require(obj["type"] == "native-tensor/1" and obj["byte_order"] == "little",
             "bad tensor type/byte order")
    _require(isinstance(obj["shape"], list) and isinstance(obj["data"], str),
             "bad tensor encoding")
    try:
        data = base64.b64decode(obj["data"], validate=True)
    except ValueError as exc:
        raise S0EnvelopeError("bad base64") from exc
    return NativeTensor(obj["dtype"], tuple(obj["shape"]), data)


def _json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def tensor_digest(tensor):
    return hashlib.sha256(_json(_wire(tensor)).encode("ascii")).hexdigest()


def _content(env):
    return {"meta": env["meta"], "payload": _wire(env["payload"]),
            "mask": env["mask"], "index_map": env["index_map"]}


def _digest(env):
    try:
        encoded = _json(_content(env)).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise S0EnvelopeError("invalid metadata encoding") from exc
    return hashlib.sha256(encoded).hexdigest()


def _origin(rec, branch):
    _fields(rec, ("origin_type", "modality", "source_sha256", "locator",
                  "event_time", "ingestion_time", "valid_from", "valid_until",
                  "branch", "dependencies", "permissions"), "origin")
    _require(isinstance(rec["origin_type"], str) and rec["origin_type"] in ROLE_IDS,
             "instruction/unknown origin forbidden")
    _require(rec["modality"] in ("text", "image", "video", "audio"), "bad modality")
    h = rec["source_sha256"]
    _require(isinstance(h, str) and len(h) == 64 and
             all(c in "0123456789abcdef" for c in h), "bad source digest")
    _require(_text(rec["locator"]), "missing source locator")
    _require(rec["branch"] == branch, "origin branch mismatch")
    _require(all(_time(rec[k]) for k in ("event_time", "ingestion_time", "valid_from")),
             "nonfinite/missing time")
    _require(rec["ingestion_time"] >= rec["event_time"], "ingestion before event")
    end = rec["valid_until"]
    _require(end is None or (_time(end) and end > rec["valid_from"]), "invalid validity interval")
    _require(isinstance(rec["dependencies"], list) and
             all(_text(v) for v in rec["dependencies"]), "bad dependencies")
    _require(_permissions(rec["permissions"]), "bad origin permissions")


def _contract(expected_width, expected_schema):
    _require(type(expected_width) is int and expected_width > 0, "invalid expected width")
    _require(expected_schema in (SCHEMA_VERSION, WIDTH_SCHEMA_VERSION), "invalid expected schema")
    _require(expected_schema != SCHEMA_VERSION or expected_width == WIDTH,
             "legacy schema requires 5376 width")


def verify_envelope(env, *, expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    """One strict validator for construction, direct calls and reload."""
    _contract(expected_width, expected_schema)
    _fields(env, ("meta", "payload", "mask", "index_map", "digest"), "envelope")
    m = env["meta"]
    _fields(m, ("schema", "type", "branch", "origin", "n_positions",
                "temporal_horizon", "compute_budget") +
            (("width",) if expected_schema == WIDTH_SCHEMA_VERSION else ()), "meta")
    _require(m["schema"] == expected_schema and m["type"] == "memory-data/1", "bad schema/type")
    if expected_schema == WIDTH_SCHEMA_VERSION:
        _require(type(m["width"]) is int and m["width"] == expected_width, "pinned width mismatch")
    _require(_text(m["branch"]), "invalid branch")
    t = native_tensor(env["payload"])
    _require(len(t.shape) == 2 and t.shape[1] == expected_width and
             0 <= t.shape[0] <= MEMORY_POSITIONS, "payload shape/capacity")
    n = t.shape[0]
    _require(type(m["n_positions"]) is int and m["n_positions"] == n, "position count")
    _require(isinstance(m["origin"], list) and len(m["origin"]) == n, "origin row alignment/loss")
    for rec in m["origin"]:
        _origin(rec, m["branch"])
    mask, ix = env["mask"], env["index_map"]
    _require(isinstance(mask, list) and len(mask) == n and
             all(type(v) is int and v in (0, 1) for v in mask), "invalid mask")
    _require(isinstance(ix, list) and len(ix) == n and
             all(type(v) is int and 0 <= v < MEMORY_POSITIONS for v in ix) and
             len(set(ix)) == n, "invalid index map")
    horizon = m["temporal_horizon"]
    _fields(horizon, ("as_of", "max_age"), "temporal horizon")
    _require(horizon["as_of"] is None or _time(horizon["as_of"]), "invalid as-of")
    _require(horizon["max_age"] is None or
             (_time(horizon["max_age"]) and horizon["max_age"] >= 0), "invalid max-age")
    if horizon["as_of"] is not None:
        now = horizon["as_of"]
        for i, rec in enumerate(m["origin"]):
            if not mask[i]:
                continue
            _require(max(rec["event_time"], rec["ingestion_time"], rec["valid_from"]) <= now,
                     "selected future evidence")
            _require(rec["valid_until"] is None or now < rec["valid_until"], "selected expired evidence")
            age = -_offset(rec["event_time"], now)
            _require(horizon["max_age"] is None or age <= horizon["max_age"],
                     "selected stale evidence")
    budget = m["compute_budget"]
    _fields(budget, ("latent_steps",), "compute budget")
    _require(type(budget["latent_steps"]) is int and budget["latent_steps"] >= 0, "invalid latent steps")
    _require(isinstance(env["digest"], str) and env["digest"] == _digest(env), "content digest mismatch")
    return m


def make_envelope(payload_rows=None, branch="main", origin=None, index_map=None,
                  mask=None, *, latent_steps=0, temporal_horizon=None,
                  expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    """Tensor [rows, WIDTH]; nonempty input requires one origin per row."""
    _contract(expected_width, expected_schema)
    t = NativeTensor("bfloat16", (0, expected_width), b"") if payload_rows is None else native_tensor(payload_rows)
    _require(len(t.shape) == 2, "payload must be a matrix")
    n = t.shape[0]
    env = {"meta": {"schema": expected_schema, "type": "memory-data/1", "branch": branch,
                    "origin": copy.deepcopy([] if origin is None and n == 0 else origin),
                    "n_positions": n, "temporal_horizon": copy.deepcopy(
                        {"as_of": None, "max_age": None} if temporal_horizon is None else temporal_horizon),
                    "compute_budget": {"latent_steps": latent_steps}},
           "payload": t, "mask": copy.deepcopy([1] * n if mask is None else mask),
           "index_map": copy.deepcopy(list(range(n)) if index_map is None else index_map), "digest": ""}
    if expected_schema == WIDTH_SCHEMA_VERSION:
        env["meta"]["width"] = expected_width
    try:
        env["digest"] = _digest(env)
    except (TypeError, ValueError) as exc:
        raise S0EnvelopeError("invalid metadata encoding") from exc
    verify_envelope(env, expected_width=expected_width, expected_schema=expected_schema)
    return env


def empty_envelope(*, expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    return make_envelope(expected_width=expected_width, expected_schema=expected_schema)


def serialize_envelope(env, *, expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    verify_envelope(env, expected_width=expected_width, expected_schema=expected_schema)
    return _json({**_content(env), "digest": env["digest"]})


def deserialize_envelope(blob, *, as_torch=False, expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    try:
        obj = json.loads(blob, object_pairs_hook=unique)
    except (ValueError, TypeError) as exc:
        raise S0EnvelopeError("malformed JSON: " + str(exc)) from exc
    _fields(obj, ("meta", "payload", "mask", "index_map", "digest"), "envelope")
    obj["payload"] = _unwire(obj["payload"])
    verify_envelope(obj, expected_width=expected_width, expected_schema=expected_schema)
    if as_torch:
        obj["payload"] = to_torch(obj["payload"])
    return obj


def select(envelopes, host_context, *, expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    """Pure row filter -> (selected envelopes, refusal dicts).

    Host permissions must contain ALL row permissions; read is mandatory.
    Future ingestion also refuses evidence even if its event timestamp is old.
    Invalid host context raises; malformed envelopes produce recorded refusals.
    Horizon receipts are replaced; callers must supply all current host policy.
    Unrepresentable offsets are row refusals even when max_age is unbounded.
    """
    _contract(expected_width, expected_schema)
    _fields(host_context, ("branch", "as_of", "max_age", "permissions"), "host context")
    h = host_context
    _require(_text(h["branch"]) and _time(h["as_of"]), "invalid branch/as-of")
    _require(h["max_age"] is None or (_time(h["max_age"]) and h["max_age"] >= 0), "invalid max-age")
    _require(_permissions(h["permissions"]), "invalid host permissions")
    selected, refusals = [], []
    for e, env in enumerate(envelopes):
        try:
            m = verify_envelope(env, expected_width=expected_width, expected_schema=expected_schema)
        except S0EnvelopeError as exc:
            refusals.append({"envelope": e, "row": None, "reasons": [str(exc)]})
            continue
        keep = []
        for i, rec in enumerate(m["origin"]):
            reasons = []
            if not env["mask"][i]:
                reasons.append("masked")
            if m["branch"] != h["branch"]:
                reasons.append("branch")
            if rec["event_time"] > h["as_of"] or rec["ingestion_time"] > h["as_of"]:
                reasons.append("future")
            if rec["valid_from"] > h["as_of"] or (rec["valid_until"] is not None and h["as_of"] >= rec["valid_until"]):
                reasons.append("validity")
            try:
                age = -_offset(rec["event_time"], h["as_of"])
            except S0EnvelopeError:
                reasons.append("nonfinite time offset")
            else:
                if h["max_age"] is not None and age > h["max_age"]:
                    reasons.append("max-age")
            if not set(rec["permissions"]) <= set(h["permissions"]):
                reasons.append("permission")
            if reasons:
                refusals.append({"envelope": e, "row": i, "reasons": reasons})
            else:
                keep.append(i)
        if keep:
            t = native_tensor(env["payload"])
            stride = expected_width * DTYPES[t.dtype]
            payload = NativeTensor(t.dtype, (len(keep), expected_width),
                                   b"".join(t.data[i*stride:(i+1)*stride] for i in keep))
            selected.append(make_envelope(payload, m["branch"], [m["origin"][i] for i in keep],
                                          [env["index_map"][i] for i in keep], [1]*len(keep),
                                          latent_steps=m["compute_budget"]["latent_steps"],
                                          temporal_horizon={"as_of": h["as_of"], "max_age": h["max_age"]},
                                          expected_width=expected_width, expected_schema=expected_schema))
    return selected, refusals


def role_time_tensors(selected, *, expected_width=WIDTH, expected_schema=SCHEMA_VERSION):
    """Return (int64 roles, float64 event offsets), physical prefix order.

    Offset = event_time - selected as_of, finite float64 seconds (no scaling
    or saturation). Unrepresentable differences refuse. No torch: return lists.
    Only selected, fully active envelopes are admitted here.
    """
    roles, times = [], []
    for env in selected:
        m = verify_envelope(env, expected_width=expected_width, expected_schema=expected_schema)
        as_of = m["temporal_horizon"]["as_of"]
        _require(as_of is not None and all(env["mask"]), "select before conditioning")
        for rec in m["origin"]:
            roles.append(ROLE_IDS[rec["origin_type"]])
            times.append(_offset(rec["event_time"], as_of))
    try:
        import torch
    except ImportError:
        return roles, times
    return (torch.tensor(roles, dtype=torch.int64, device="cpu"),
            torch.tensor(times, dtype=torch.float64, device="cpu"))


def _fixture(n=2):
    rec = {"origin_type": "observation", "modality": "text", "source_sha256": "a"*64,
           "locator": "fixture:1", "event_time": 10, "ingestion_time": 11,
           "valid_from": 10, "valid_until": 100, "branch": "main",
           "dependencies": [], "permissions": ["read"]}
    return make_envelope(NativeTensor("bfloat16", (n, WIDTH), b"\x00\x3f"*(n*WIDTH)),
                         origin=[copy.deepcopy(rec) for _ in range(n)])


def _reject(fn):
    try:
        fn()
    except S0EnvelopeError:
        return
    raise AssertionError("negative case accepted")


def check_origin():
    env = _fixture()
    assert deserialize_envelope(serialize_envelope(env))["meta"]["origin"] == env["meta"]["origin"]
    for origins in (None, [], [env["meta"]["origin"][0]]):
        _reject(lambda: make_envelope(env["payload"], origin=origins))
    for field, value in (("origin_type", "instruction"), ("branch", "other"),
                         ("source_sha256", ""), ("locator", ""), ("event_time", float("nan"))):
        bad = copy.deepcopy(env)
        bad["meta"]["origin"][0][field] = value
        _reject(lambda: verify_envelope(bad))
    bad = copy.deepcopy(env)
    bad["meta"]["origin"][0]["locator"] = "laundered"
    _reject(lambda: verify_envelope(bad))


def check_schema():
    env = _fixture()
    # Every top-level, metadata, origin and nested policy field is required.
    for path in ((), ("meta",), ("meta", "origin", 0),
                 ("meta", "temporal_horizon"), ("meta", "compute_budget")):
        obj = env
        for key in path:
            obj = obj[key]
        for field in obj:
            bad = copy.deepcopy(env)
            dst = bad
            for key in path:
                dst = dst[key]
            del dst[field]
            _reject(lambda: verify_envelope(bad))
        bad = copy.deepcopy(env)
        dst = bad
        for key in path:
            dst = dst[key]
        dst["unknown"] = 1
        _reject(lambda: verify_envelope(bad))
    for key, value in (("mask", [0, 1]), ("index_map", [1, 0]),
                       ("mask", None), ("index_map", None)):
        bad = copy.deepcopy(env)
        bad[key] = value
        _reject(lambda: verify_envelope(bad))
        blob = json.loads(serialize_envelope(env))
        blob[key] = value
        _reject(lambda: deserialize_envelope(_json(blob)))
    _reject(lambda: deserialize_envelope('{"meta":1,"meta":2}'))
    for key in ("mask", "index_map", "meta"):
        wire = json.loads(serialize_envelope(env))
        del wire[key]
        _reject(lambda: deserialize_envelope(_json(wire)))
    for key in ("type", "dtype", "shape", "byte_order", "data"):
        wire = json.loads(serialize_envelope(env))
        del wire["payload"][key]
        _reject(lambda: deserialize_envelope(_json(wire)))
    for ix in ([0, 0], [-1, 1], [0, MEMORY_POSITIONS]):
        _reject(lambda: make_envelope(env["payload"], origin=env["meta"]["origin"], index_map=ix))
    for steps in (-1, True, 1.5):
        _reject(lambda: make_envelope(env["payload"], origin=env["meta"]["origin"], latent_steps=steps))


def check_digest():
    env = _fixture()
    assert len(env["payload"].data) == 2 * 10752
    t = env["payload"]
    assert tensor_digest(t) != tensor_digest(NativeTensor(t.dtype, t.shape, t.data[:-1]+b"\x40"))
    assert tensor_digest(t) != tensor_digest(NativeTensor("int16", t.shape, t.data))
    assert tensor_digest(t) != tensor_digest(NativeTensor(t.dtype, (WIDTH, 2), t.data))
    changed = copy.deepcopy(env)
    changed["meta"]["origin"][0]["locator"] = "fixture:changed-source"
    assert _digest(changed) != env["digest"]
    try:
        import torch
    except ImportError:
        print("SKIP torch native views (torch unavailable)")
        return
    for dtype in DTYPES:
        original = torch.arange(2*WIDTH, device="cpu").reshape(2, WIDTH).to(getattr(torch, dtype))
        assert tensor_digest(original) == tensor_digest(native_tensor(original))
        recs = env["meta"]["origin"]
        e = make_envelope(original, origin=recs)
        loaded = deserialize_envelope(serialize_envelope(e), as_torch=True)
        assert loaded["digest"] == e["digest"]
        assert native_tensor(original) == native_tensor(loaded["payload"])
    # Noncontiguous, scalar, signed zero and NaN bit patterns are retained.
    x = torch.arange(12, device="cpu").reshape(3, 4).t()
    assert native_tensor(x) == native_tensor(to_torch(native_tensor(x)))
    assert native_tensor(torch.tensor(2, device="cpu")).shape == ()
    bits = NativeTensor("bfloat16", (1, WIDTH), b"\x00\x80\xc1\x7f"*(WIDTH//2))
    assert native_tensor(to_torch(bits)) == bits


def check_roundtrip():
    for env in (_fixture(), empty_envelope()):
        blob = serialize_envelope(env)
        assert serialize_envelope(deserialize_envelope(blob)) == blob
        assert deserialize_envelope(blob)["payload"] == env["payload"]


def check_selection():
    h = {"branch": "main", "as_of": 20, "max_age": 15, "permissions": ["read"]}
    env = _fixture()
    before = serialize_envelope(env)
    selected, refusals = select([env], h)
    assert len(selected) == 1 and not refusals and serialize_envelope(env) == before
    for field, value, reason in (("branch", "other", "branch"), ("as_of", 9, "future"),
                                  ("as_of", 100, "validity"), ("max_age", 5, "max-age")):
        context = {**h, field: value}
        got, refused = select([env], context)
        assert not got and reason in refused[0]["reasons"]
    origin = copy.deepcopy(env["meta"]["origin"])
    origin[0]["permissions"] = ["read", "private"]
    origin[1]["origin_type"] = "derived"
    mixed = make_envelope(env["payload"], origin=origin, index_map=[7, 3])
    got, refused = select([mixed], h)
    assert len(got) == 1 and got[0]["index_map"] == [3]
    assert got[0]["meta"]["origin"] == [origin[1]] and refused[0]["reasons"] == ["permission"]
    roles, times = role_time_tensors(got)
    assert list(roles) == [3] and list(times) == [-10]
    masked = make_envelope(env["payload"], origin=env["meta"]["origin"], mask=[0, 1])
    assert select([masked], h)[1][0]["reasons"] == ["masked"]
    for field, value in (("branch", ""), ("as_of", None), ("as_of", float("inf")),
                         ("max_age", -1), ("permissions", [])):
        _reject(lambda: select([env], {**h, field: value}))
    _reject(lambda: role_time_tensors([env]))
    for steps in (0, 7):
        e = make_envelope(env["payload"], origin=env["meta"]["origin"], latent_steps=steps)
        got, _ = select([e], h)
        assert got[0]["meta"]["compute_budget"]["latent_steps"] == steps
        assert got[0]["meta"]["temporal_horizon"] == {"as_of": 20, "max_age": 15}
    for as_of, age in ((9, None), (100, None), (20, 5)):
        _reject(lambda: make_envelope(env["payload"], origin=env["meta"]["origin"],
                                      temporal_horizon={"as_of": as_of, "max_age": age}))
    for field, value, reason in (("ingestion_time", 21, "future"),
                                  ("valid_from", 21, "validity")):
        recs = copy.deepcopy(env["meta"]["origin"])
        recs[0][field] = value
        e = make_envelope(env["payload"], origin=recs)
        got, refused = select([e], h)
        assert got[0]["meta"]["origin"] == [recs[1]] and reason in refused[0]["reasons"]
    bad = copy.deepcopy(env)
    del bad["mask"]
    assert select([bad], h)[0] == [] and select([bad], h)[1]


def check_offline():
    import builtins
    import socket
    import subprocess
    from unittest.mock import patch
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("offline plain-tensor control")
        return real_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("external effect forbidden")

    with patch("builtins.__import__", no_torch), patch.object(socket, "socket", forbidden), \
            patch.object(subprocess, "Popen", forbidden):
        env = _fixture()
        check_origin()
        check_schema()
        check_roundtrip()
        check_selection()
        check_numeric()
        selected, _ = select([env], {"branch": "main", "as_of": 20,
                                     "max_age": None, "permissions": ["read"]})
        assert role_time_tensors(selected) == ([0, 0], [-10, -10])


def check_numeric():
    env = _fixture(1)
    h = {"branch": "main", "as_of": 20, "max_age": None, "permissions": ["read"]}
    # M1: supported numeric types must fail through the validation exception,
    # including direct, hostile wire and mixed-invalid/valid selection paths.
    for field in ("event_time", "ingestion_time", "valid_from", "valid_until"):
        bad = copy.deepcopy(env)
        bad["meta"]["origin"][0][field] = 10**1000
        _reject(lambda: verify_envelope(bad))
        _reject(lambda: make_envelope(bad["payload"], origin=bad["meta"]["origin"]))
        wire = json.loads(serialize_envelope(env))
        wire["meta"]["origin"][0][field] = 10**1000
        _reject(lambda: deserialize_envelope(_json(wire)))
        got, refused = select([bad, env], h)
        assert len(got) == 1 and got[0]["meta"]["origin"] == env["meta"]["origin"]
        assert len(refused) == 1 and refused[0]["envelope"] == 0 and refused[0]["row"] is None
    for field in ("as_of", "max_age"):
        _reject(lambda: select([env], {**h, field: 10**1000}))
        bad = copy.deepcopy(env)
        bad["meta"]["temporal_horizon"][field] = 10**1000
        _reject(lambda: verify_envelope(bad))
    # Integer budgets are unbounded by policy, but JSON encoding has a runtime
    # limit. Encoding refusal must not abort subsequent envelope selection.
    bad = copy.deepcopy(env)
    bad["meta"]["compute_budget"]["latent_steps"] = 10**5000
    if sys.get_int_max_str_digits():
        _reject(lambda: verify_envelope(bad))
        got, refused = select([bad, env], h)
        assert len(got) == 1 and refused[0]["row"] is None

    # M2: finite inputs alone are insufficient. Refuse overflow, with and
    # without an age cap, while continuing to the ordinary legal envelope.
    for extreme in (-1e308, -int(sys.float_info.max)):
        recs = copy.deepcopy(env["meta"]["origin"])
        recs[0].update(event_time=extreme, ingestion_time=extreme,
                       valid_from=extreme, valid_until=None)
        far = make_envelope(env["payload"], origin=recs)
        assert deserialize_envelope(serialize_envelope(far)) == far
        now = 1e308 if type(extreme) is float else int(sys.float_info.max)
        for cap in (None, sys.float_info.max):
            context = {**h, "as_of": now, "max_age": cap}
            got, refused = select([far], context)
            assert not got and refused[0]["row"] == 0
            assert "nonfinite time offset" in refused[0]["reasons"]
            _reject(lambda: make_envelope(far["payload"], origin=recs,
                                          temporal_horizon={"as_of": now, "max_age": cap}))
            selected_bad = copy.deepcopy(far)
            selected_bad["meta"]["temporal_horizon"] = {"as_of": now, "max_age": cap}
            selected_bad["digest"] = _digest(selected_bad)
            _reject(lambda: role_time_tensors([selected_bad]))
        ordinary = copy.deepcopy(env["meta"]["origin"])
        ordinary[0].update(event_time=now, ingestion_time=now,
                           valid_from=now, valid_until=None)
        legal = make_envelope(env["payload"], origin=ordinary)
        got, refused = select([far, legal], {**h, "as_of": now})
        assert len(got) == 1 and len(refused) == 1
        assert list(role_time_tensors(got)[1]) == [0.0]
    # The largest finite negative offset is admitted, emitted without scaling.
    recs[0].update(event_time=-sys.float_info.max, ingestion_time=-sys.float_info.max,
                   valid_from=-sys.float_info.max, valid_until=None)
    boundary = make_envelope(env["payload"], origin=recs)
    got, refused = select([boundary], {**h, "as_of": 0})
    assert not refused and list(role_time_tensors(got)[1]) == [-sys.float_info.max]
    assert all(math.isfinite(v) for v in role_time_tensors(got)[1])
    # N1: the horizon is a receipt. Re-selection uses the new host policy.
    got, _ = select([env], {**h, "max_age": 10})
    again, refused = select(got, {**h, "as_of": 30, "max_age": 20})
    assert not refused and again[0]["meta"]["temporal_horizon"] == {"as_of": 30, "max_age": 20}
    assert got[0]["meta"]["temporal_horizon"] == {"as_of": 20, "max_age": 10}


def check_views():
    from unittest.mock import patch
    env = _fixture(1)
    # S2: wire byte order and simulated host policy are separate checks.
    wire = json.loads(serialize_envelope(env))
    wire["payload"]["byte_order"] = "big"
    _reject(lambda: deserialize_envelope(_json(wire)))
    with patch.object(sys, "byteorder", "big"):
        _reject(lambda: to_torch(env["payload"]))
        _reject(lambda: to_torch(empty_envelope()["payload"]))
    try:
        import torch
    except ImportError:
        print("SKIP torch lazy-negative views (torch unavailable)")
        return
    # S1: a genuinely lazy negative strided view agrees with materialization.
    x = torch.arange(2*WIDTH, dtype=torch.float32, device="cpu").reshape(1, 2*WIDTH)[:, ::2]
    lazy = torch._neg_view(x)
    assert lazy.is_neg() and not lazy.is_contiguous()
    assert native_tensor(lazy) == native_tensor(-x)
    # Contiguous lazy views must also resolve; contiguous() alone is insufficient.
    lazy = torch._neg_view(x.contiguous())
    assert lazy.is_neg() and lazy.is_contiguous()
    assert tensor_digest(lazy) == tensor_digest(-x)
    e = make_envelope(lazy, origin=env["meta"]["origin"])
    direct = {**e, "payload": lazy}
    got, refused = select([direct, env], {"branch": "main", "as_of": 20,
                                         "max_age": None, "permissions": ["read"]})
    assert len(got) == 2 and not refused
    assert got[0]["payload"] == native_tensor(-x)
    assert native_tensor(deserialize_envelope(serialize_envelope(direct), as_torch=True)["payload"]) == native_tensor(-x)
    with patch.object(sys, "byteorder", "big"):
        _reject(lambda: native_tensor(x))


def check_configurable_width():
    host = {"branch": "main", "as_of": 20, "max_age": None, "permissions": ["read"]}
    for width in (12, 32, 1536, WIDTH):
        contract = dict(expected_width=width, expected_schema=WIDTH_SCHEMA_VERSION)
        for dtype in DTYPES:
            stride = width * DTYPES[dtype]
            payload = NativeTensor(dtype, (2, width), b"\x01"*stride + b"\x02"*stride)
            origins = copy.deepcopy(_fixture()["meta"]["origin"])
            origins[0]["permissions"].append("private")
            env = make_envelope(payload, origin=origins, index_map=[7, 3], **contract)
            wire = serialize_envelope(env, **contract)
            loaded = deserialize_envelope(wire, **contract)
            got, refused = select([loaded], host, **contract)
            assert len(got) == 1 and refused[0]["reasons"] == ["permission"]
            assert got[0]["payload"] == NativeTensor(dtype, (1, width), b"\x02"*stride)
            assert got[0]["index_map"] == [3] and got[0]["meta"]["origin"] == [origins[1]]
            assert list(role_time_tensors(got, **contract)[1]) == [-10]
            _reject(lambda: deserialize_envelope(wire, expected_width=width+1,
                                                 expected_schema=WIDTH_SCHEMA_VERSION))
            _reject(lambda: deserialize_envelope(wire))
            bad = copy.deepcopy(env)
            bad["meta"]["width"] += 1
            bad["digest"] = _digest(bad)
            _reject(lambda: verify_envelope(bad, **contract))
        empty = empty_envelope(**contract)
        assert deserialize_envelope(serialize_envelope(empty, **contract), **contract) == empty
        assert empty["payload"].shape == (0, width)
        assert select([empty], host, **contract) == ([], [])
        _reject(lambda: deserialize_envelope(serialize_envelope(empty, **contract),
                                             expected_width=width+1, expected_schema=WIDTH_SCHEMA_VERSION))
    # The default contract still round-trips the original schema without rewriting it.
    legacy = _fixture()
    assert "width" not in legacy["meta"]
    assert deserialize_envelope(serialize_envelope(legacy)) == legacy
    _reject(lambda: empty_envelope(expected_width=12))
    print("widths=12,32,1536,5376 all 10 dtypes; survivor bytes/index/origin/time; empty/wrong-width/legacy PASS")


CHECKS = {"origin": check_origin, "schema": check_schema, "digest": check_digest,
          "roundtrip": check_roundtrip, "selection": check_selection, "offline": check_offline,
          "numeric": check_numeric, "views": check_views,
          "configurable-width": check_configurable_width}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=["all", *CHECKS])
    args = parser.parse_args()
    for name, check in CHECKS.items():
        if args.check in ("all", name):
            check()
            print("PASS " + name)


if __name__ == "__main__":
    main()
