"""S0 production optional memory port and bounded offline mechanics checks.

Typed native rows -> receiver-owned resampler -> eight slots -> complete frozen
Gemma stem; forward returns its live loss/logits/cache. Horizon is explicitly zero.
Private _legacy_forward and legacy envelope/connector fixtures retain historical
scaffold checks only; production never calls their hash connector or validator.
No pretrained assets, loader, CUDA operation or network call occurs on import.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

WIDTH = 5376          # Gemma 4 31B hidden size (pinned config)
N_LAYERS = 60
BLOCK = (24, 31)      # zero-based recurrence candidate (layer 29 is global)
WORKSPACE_POSITIONS = 8
MEMORY_POSITIONS = 16
SCHEMA_VERSION = "s0-memory-port/1"

_allowed_horizons = (0, 1, 2)      # additional workspace passes
_budget_keys = ("max_workspace_passes", "max_total_forward_calls",
                "max_new_tokens", "selective_logits")


class S0InputError(ValueError):
    """Invalid call; nothing is executed."""


class S0MemoryError(ValueError):
    """Unauthenticated/tampered/stale memory envelope."""


# --- authenticated memory envelope (full binding lands in S0-Z2) --------


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def empty_envelope():
    """Authenticated EMPTY snapshot: legal metadata, zero payload."""
    meta = {"schema": SCHEMA_VERSION, "branch": "main", "origin": [],
            "legal": {"policy": "read-only", "permissions": ["read"]},
            "n_positions": 0, "payload_sha256": _sha(b"")}
    return {"meta": meta, "auth": _auth_of(meta), "payload": None}


def _auth_of(meta):
    """Binding hash over the full canonical metadata: schema, branch,
    parent lineage, origin records (each with its own source hash and
    locator), legal block, validity interval, ingestion identity, payload
    digest and index/mask map digest (S0-Z2)."""
    return _sha(json.dumps(meta, sort_keys=True).encode())


def _norm_origin(origin):
    """Every retained unit carries origin, modality, source hash,
    locator, event time, ingestion time, validity interval, branch and
    dependencies (council custody contract)."""
    if origin is None:
        return None
    keys = ("origin_type", "modality", "source_sha256", "locator",
            "event_time", "ingestion_time", "valid_from", "valid_until",
            "branch", "dependencies")
    for rec in origin:
        missing = [k for k in keys if k not in rec]
        if missing:
            raise S0MemoryError(f"origin record missing {missing}")
        if rec["origin_type"] not in ("observation", "user_input",
                                      "instruction", "generated",
                                      "derived"):
            raise S0MemoryError(f"bad origin_type {rec['origin_type']!r}")
        if rec["modality"] not in ("text", "image", "video", "audio"):
            raise S0MemoryError(f"bad modality {rec['modality']!r}")
        if not _check_hex64(rec["source_sha256"]):
            raise S0MemoryError("source_sha256 must be a 64-hex digest")
        if rec["valid_until"] is not None and rec["valid_from"] is not None                 and rec["valid_until"] < rec["valid_from"]:
            raise S0MemoryError("valid_until precedes valid_from")
        if rec["ingestion_time"] is not None and rec["event_time"] is not None                 and rec["ingestion_time"] < rec["event_time"]:
            raise S0MemoryError("ingestion_time precedes event_time")
    branches = {rec["branch"] for rec in origin}
    if len(branches) > 1:
        raise S0MemoryError("origin records span multiple branches")
    return origin  # validated list; caller cross-checks the branch set


def serialize_envelope(env):
    """Canonical serialized form: canonical-JSON metadata + auth tag +
    base64 payload. Round-trips through deserialize_envelope with
    payload/index/mask alignment preserved."""
    import base64
    verify_envelope(env)
    payload_b64 = None
    if env.get("payload"):
        payload_b64 = base64.b64encode(b"".join(bytes(r) for r in env["payload"])).decode()
    return json.dumps({"meta": env["meta"], "auth": env["auth"],
                       "payload_b64": payload_b64,
                       "index_map": env.get("index_map"),
                       "mask": env.get("mask")}, sort_keys=True)


def deserialize_envelope(blob):
    """Authenticated reload. Rejects tampering with meta, auth, payload,
    index_map or mask; stale schema/branch; and index/mask
    misalignment (mask length must equal payload rows)."""
    import base64
    try:
        obj = json.loads(blob)
        meta, auth = obj["meta"], obj["auth"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise S0MemoryError(f"malformed serialized envelope: {exc!r}")
    if meta.get("schema") != SCHEMA_VERSION:
        raise S0MemoryError(f"stale schema {meta.get('schema')!r}")
    if _auth_of(meta) != auth:
        raise S0MemoryError("tampered envelope: auth mismatch")
    n = meta.get("n_positions", 0)
    payload = None
    if obj.get("payload_b64") is not None:
        try:
            raw = base64.b64decode(obj["payload_b64"], validate=True)
        except Exception as exc:
            raise S0MemoryError(f"malformed payload encoding: {exc!r}")
        if len(raw) % WIDTH != 0 or len(raw) // WIDTH != n:
            raise S0MemoryError("serialized payload/position misalignment")
        if _sha(raw) != meta["payload_sha256"]:
            raise S0MemoryError("serialized payload digest mismatch")
        payload = [raw[i * WIDTH:(i + 1) * WIDTH] for i in range(n)]
    elif n != 0:
        raise S0MemoryError("nonempty envelope serialized without payload")
    mask = obj.get("mask")
    index_map = obj.get("index_map")
    if mask is not None:
        if not isinstance(mask, list) or len(mask) != n                 or not all(isinstance(m, int) and m in (0, 1) for m in mask):
            raise S0MemoryError("mask/index misalignment on reload")
        if _sha(json.dumps(mask).encode()) != meta.get("mask_sha256"):
            raise S0MemoryError("tampered mask: digest mismatch")
    if index_map is not None:
        if not isinstance(index_map, list) or len(index_map) != n                 or len(set(index_map)) != n:
            raise S0MemoryError("index_map misalignment on reload")
        if _sha(json.dumps(index_map).encode()) != meta.get("index_map_sha256"):
            raise S0MemoryError("tampered index_map: digest mismatch")
    env = {"meta": meta, "auth": auth, "payload": payload}
    if obj.get("index_map") is not None:
        env["index_map"] = obj["index_map"]
    if mask is not None:
        env["mask"] = mask
    verify_envelope(env)
    return env


def make_envelope(payload_rows=None, branch="main", origin=None,
                  index_map=None, mask=None):
    """payload_rows: list of WIDTH-wide rows (0 < len <= MEMORY_POSITIONS),
    or None for the authenticated empty snapshot. origin records follow
    the council custody contract (validated). index_map maps payload
    rows to logical prefix positions; mask marks active rows (1)."""
    origin = _norm_origin(origin or [])
    if payload_rows is None:
        if branch != "main" or index_map is not None or mask is not None:
            raise S0MemoryError("empty envelope takes no branch/index/mask args")
        env = empty_envelope()
        env["meta"]["origin"] = origin
        env["auth"] = _auth_of(env["meta"])
        return env
    if not (0 < len(payload_rows) <= MEMORY_POSITIONS):
        raise S0MemoryError(f"payload rows must be 1..{MEMORY_POSITIONS}")
    for r in payload_rows:
        if not isinstance(r, (bytes, bytearray)) or len(r) != WIDTH:
            raise S0MemoryError(f"each payload row must be exactly {WIDTH} bytes")
    n = len(payload_rows)
    if mask is not None and (len(mask) != n or
                             not all(m in (0, 1) for m in mask)):
        raise S0MemoryError("mask must align with payload rows")
    if index_map is not None and (len(index_map) != n or
                                  len(set(index_map)) != n):
        raise S0MemoryError("index_map must be a permutation-aligned list")
    blob = b"".join(bytes(r) for r in payload_rows)
    meta = {"schema": SCHEMA_VERSION, "branch": branch,
            "origin": origin, "legal": {"policy": "read-only",
                                        "permissions": ["read"]},
            "n_positions": len(payload_rows), "payload_sha256": _sha(blob)}
    if index_map is not None:
        meta["index_map_sha256"] = _sha(json.dumps(index_map).encode())
    if mask is not None:
        meta["mask_sha256"] = _sha(json.dumps(mask).encode())
    env = {"meta": meta, "auth": _auth_of(meta), "payload": payload_rows}
    if index_map is not None:
        env["index_map"] = index_map
    if mask is not None:
        env["mask"] = mask
    return env


_META_REQUIRED = ("schema", "branch", "origin", "legal", "n_positions",
                  "payload_sha256")


def _check_hex64(s):
    return isinstance(s, str) and len(s) == 64         and all(c in "0123456789abcdef" for c in s)


def verify_envelope(env):
    """Strict verification on every path (construction, direct use,
    reload). Rejects tampering, stale schema/identity, malformed meta,
    bad origin custody, mask/index misalignment and digest mismatches.
    TRUST BOUNDARY: the binding hash is an unkeyed SHA-256 — it proves
    internal consistency against accidental corruption, not authenticity
    against an adversary who can rewrite the whole envelope; a keyed MAC
    or external signature is required before cross-process custody."""
    if not isinstance(env, dict) or not {"meta", "auth"} <= set(env):
        raise S0MemoryError("envelope missing meta/auth")
    meta = env["meta"]
    if not isinstance(meta, dict):
        raise S0MemoryError("meta must be a dict")
    missing = [k for k in _META_REQUIRED if k not in meta]
    if missing:
        raise S0MemoryError(f"meta missing {missing}")
    if meta.get("schema") != SCHEMA_VERSION:
        raise S0MemoryError(f"stale schema {meta.get('schema')!r}")
    if not isinstance(meta["branch"], str) or not meta["branch"]:
        raise S0MemoryError("branch must be a nonempty string")
    if type(meta["n_positions"]) is not int             or not (0 <= meta["n_positions"] <= MEMORY_POSITIONS):
        raise S0MemoryError("n_positions must be an int in 0..MEMORY_POSITIONS")
    if not _check_hex64(meta["payload_sha256"]):
        raise S0MemoryError("payload_sha256 must be a 64-hex digest")
    if _auth_of(meta) != env["auth"]:
        raise S0MemoryError("tampered envelope: auth mismatch")
    _norm_origin(meta.get("origin", []))          # custody on every path
    obranches = {r["branch"] for r in (meta.get("origin") or [])}
    if obranches and meta["branch"] not in obranches:
        raise S0MemoryError("origin branch does not match envelope branch")
    n = meta["n_positions"]
    payload = env.get("payload")
    if n == 0:
        if payload is not None:
            raise S0MemoryError("empty envelope carries a payload")
    else:
        if not payload or len(payload) != n:
            raise S0MemoryError("payload/position mismatch")
        for r in payload:
            if not isinstance(r, (bytes, bytearray)) or len(r) != WIDTH:
                raise S0MemoryError(f"payload rows must be exactly {WIDTH} bytes")
        blob = b"".join(bytes(r) for r in payload)
        if _sha(blob) != meta["payload_sha256"]:
            raise S0MemoryError("payload digest mismatch")
    mask, index_map = env.get("mask"), env.get("index_map")
    if mask is not None and "mask_sha256" not in meta:
        raise S0MemoryError("mask present without a bound digest")
    if index_map is not None and "index_map_sha256" not in meta:
        raise S0MemoryError("index_map present without a bound digest")
    if mask is None and "mask_sha256" in meta:
        raise S0MemoryError("bound mask deleted from envelope")
    if index_map is None and "index_map_sha256" in meta:
        raise S0MemoryError("bound index_map deleted from envelope")
    if mask is not None:
        if not isinstance(mask, list) or len(mask) != n                 or not all(type(m) is int and m in (0, 1) for m in mask):
            raise S0MemoryError("mask misalignment")
        if _sha(json.dumps(mask).encode()) != meta.get("mask_sha256"):
            raise S0MemoryError("tampered mask: digest mismatch")
    if index_map is not None:
        if not isinstance(index_map, list) or len(index_map) != n                 or not all(type(v) is int and 0 <= v < MEMORY_POSITIONS
                           for v in index_map) or len(set(index_map)) != n:
            raise S0MemoryError("index_map misalignment")
        if _sha(json.dumps(index_map).encode()) != meta.get("index_map_sha256"):
            raise S0MemoryError("tampered index_map: digest mismatch")
    return meta


# --- the one-call interface -----------------------------------------------


def _legacy_forward(model, inputs, memory=None, horizon=1, budget=None,
           conditioning=None):
    """TEMPORAL HORIZON (conditioning.as_of/branch/max_age) selects legal
    evidence by time/branch and is DISTINCT from the LATENT-STEP BUDGET
    (budget.max_workspace_passes): filtering the attachment never changes
    how many workspace passes run, and workspace passes never change which
    evidence is legal. Illegal roles/types/permissions reject before any
    assembly."""
    """One call, one complete model/decoder.

    model:   the pinned stem handle (cloud path). None selects CPU PLAN
             MODE: the identical routing/validation runs and returns the
             execution plan; no loader is ever called.
    inputs:  {"input_ids": [[...]]} or {"text": ["..."]} (tokenizer is
             part of the model handle on the live path).
    memory:  None, or an authenticated envelope (verify_envelope).
    horizon: additional workspace passes through block 24-31 (0/1/2).
    budget:  dict with any of _budget_keys; all values positive ints.

    TRUE EMPTY BYPASS: memory=None and an authenticated empty envelope
    are behaviourally identical by construction — the same routing, the
    same positions, the same cache contract.
    """
    plan = _validate_and_route(inputs, memory, horizon, budget,
                               conditioning)
    if model is None:
        return {"mode": "cpu-plan", "plan": plan}

    # Live path: ONE REAL RECEIVING OPERATION (18:05 council items 1-2,
    # integrated with the parallel-staffing helpers by import per the
    # 19:25 portfolio council). The handle must supply a callable
    # receiver; we assemble (prefix_states, input_ids, mask,
    # logical_positions, query context) and return the receiver's ACTUAL
    # result. When the handle carries the gemma4_receiver runtime, the
    # native receiving interface (PLE, masks, positions, bypass, decode)
    # is delegated to it; when it carries a typed s0_envelope attachment,
    # the envelope module's own verification is honoured first. No
    # receipt synthesis on this path; an unsupported runtime raises
    # explicitly.
    m = model
    # integration point 1: typed native attachment (s0_envelope worker)
    _envelope_mod = None
    try:
        import importlib as _imp
        _envelope_mod = _imp.import_module(".s0_envelope", __package__)
    except ImportError:
        pass
    if _envelope_mod is not None and memory is not None             and not isinstance(memory, dict)             and hasattr(_envelope_mod, "to_view_payload"):
        memory = _envelope_mod.to_view_payload(memory, conditioning=conditioning)
    missing = [k for k in ("tok", "connector", "receiver") if m.get(k) is None]
    if missing:
        # integration point 2: the gemma4_receiver runtime supplies all
        # three via its production interface when requested
        try:
            import importlib as _imp2
            _recv_mod = _imp2.import_module(".gemma4_receiver", __package__)
            runtime = getattr(_recv_mod, "make_handle", None)
            if runtime is not None:
                m = runtime(m)
                missing = [k for k in ("tok", "connector", "receiver")
                           if m.get(k) is None]
        except ImportError:
            pass
    if missing:
        raise S0InputError(f"model handle missing {missing}")
    if not callable(m["receiver"]):
        raise S0InputError("receiver must be callable")
    if not callable(m["connector"]):
        raise S0InputError("connector must be callable")
    # 1) input encoding (tokenizer or pass-through of ids)
    if "text" in inputs:
        enc = m["tok"](inputs["text"], add_special_tokens=False)
        input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
    else:
        input_ids = inputs["input_ids"]

    # 2) memory attachment through the connector: ACTIVE rows only,
    #    assembled with their logical positions and mask; the empty
    #    bypass performs NO connector call at all (identical to None).
    prefix_states, logical_positions, attach_mask = None, None, None
    cond_tensors = None
    if plan["memory_positions"] > 0:
        meta = verify_envelope(memory)
        kept, n_dropped = _filter_origin_by_conditioning(
            meta["origin"], plan["_cond"])
        rows = memory["payload"]
        mask = memory.get("mask", [1] * len(rows))
        if n_dropped:
            # keep only rows whose origin survived filtering (row i maps
            # to origin record i by construction)
            rows = [r for i, r in enumerate(rows) if i < len(kept)]
            mask = [1] * len(rows)
        conn = m["connector"](rows)
        prefix_states = conn["prefix"]
        # honor the envelope's digest-bound index_map when present
        logical_positions = memory.get("index_map",
                                       list(range(len(prefix_states))))
        attach_mask = mask
        role_ix = (_ROLE_TYPES.index(plan["_cond"]["role"])
                   if plan["_cond"]["role"] in _ROLE_TYPES else -1)
        role_oh = [1.0 if i == role_ix else 0.0
                   for i in range(len(_ROLE_TYPES))]
        type_mh = [1.0 if t in (plan["_cond"]["types"] or []) else 0.0
                   for t in ("text", "image", "video", "audio")]
        cond_tensors = {"role": role_oh, "types": type_mh,
                        "as_of": plan["_cond"]["as_of"],
                        "n_dropped_by_temporal_filter": n_dropped}

    # 3) THE receiving call: everything assembled flows to the receiver;
    #    its return value IS forward's return value. Receiving contract
    #    (receipt-and-echo): the receiver must echo the assembly digest,
    #    proving digest->output dataflow and rejecting dropped/missing
    #    outputs. LIMITATION (documented, not hidden): echo does NOT prove
    #    the receiver used the prefix CONTENT; content-use is established
    #    per-receiver by donor-swap/omission controls, not by this gate.
    assembly = {"input_ids": input_ids,
                "attach_mask": attach_mask,
                "logical_positions": logical_positions}
    if prefix_states is not None:
        def _row_bytes(r):
            if isinstance(r, torch.Tensor):
                t = r.detach()
                if t.is_cuda:                      # BF16/CUDA-safe path
                    t = t.to("cpu", copy=True)
                return t.contiguous().to(torch.float32).numpy().tobytes()
            return bytes(r)
        assembly["prefix_digest"] = _sha(b"".join(_row_bytes(r) for r, mk in
                                                  zip(prefix_states,
                                                      attach_mask or [1] * len(prefix_states)) if mk))
    digest = _sha(json.dumps(assembly, sort_keys=True, default=str).encode())
    result = m["receiver"](prefix_states=prefix_states,
                           input_ids=input_ids,
                           attach_mask=attach_mask,
                           logical_positions=logical_positions,
                           horizon=horizon,
                           budget=dict(plan["budget"]),
                           conditioning=cond_tensors,
                           assembly_digest=digest,
                           query=inputs.get("text", inputs.get("input_ids")))
    if result is None or not isinstance(result, dict)             or result.get("received_digest") != digest:
        raise S0InputError("receiver violated the receiving contract: "
                           "output dropped or disconnected from the assembly")
    return result


_ROLE_TYPES = ("observation", "user_input", "instruction",
               "generated", "derived")


def _validate_conditioning(conditioning):
    """Role/type/time/permission conditioning contract (host-enforced)."""
    c = conditioning or {}
    bad = [k for k in c if k not in ("role", "types", "as_of", "branch",
                                      "max_age", "permissions")]
    if bad:
        raise S0InputError(f"unknown conditioning keys {bad}")
    role = c.get("role")
    if role is not None and role not in _ROLE_TYPES:
        raise S0InputError(f"illegal role {role!r}")
    types = c.get("types")
    if types is not None and (not isinstance(types, (list, tuple))
                              or not all(t in ("text", "image", "video",
                                               "audio") for t in types)):
        raise S0InputError("illegal modality types")
    perms = c.get("permissions")
    if perms is not None and (not isinstance(perms, (list, tuple))
                              or not set(perms) <= {"read"}):
        raise S0InputError("host permissions permit read-only access only")
    as_of = c.get("as_of")
    if as_of is not None and not isinstance(as_of, str):
        raise S0InputError("as_of must be an ISO-8601 string")
    max_age = c.get("max_age")
    if max_age is not None and (not isinstance(max_age, (int, float))
                                or max_age < 0):
        raise S0InputError("max_age must be a nonnegative number")
    return {"role": role, "types": list(types) if types else None,
            "as_of": as_of, "branch": c.get("branch"),
            "max_age": max_age,
            "permissions": list(perms) if perms else ["read"]}


def _filter_origin_by_conditioning(origin, cond):
    """Branch/as-of validity filtering over origin records. Host
    permissions are enforced here (read-only). Exact clock arithmetic
    belongs to the runtime worker; ISO strings of the same format order
    lexicographically."""
    if cond["permissions"] != ["read"]:
        raise S0InputError("memory access is read-only; requested "
                           f"{cond['permissions']}")
    kept, dropped = [], 0
    for rec in (origin or []):
        if cond["branch"] and rec.get("branch") != cond["branch"]:
            dropped += 1
            continue
        if cond["as_of"] and rec.get("valid_from") \
                and rec["valid_from"] > cond["as_of"]:
            dropped += 1
            continue
        kept.append(rec)
    return kept, dropped


def _validate_and_route(inputs, memory, horizon, budget, conditioning=None):
    cond = _validate_conditioning(conditioning)
    if not isinstance(inputs, dict):
        raise S0InputError("inputs must be a dict")
    has_ids = "input_ids" in inputs
    has_text = "text" in inputs
    if has_ids == has_text:
        raise S0InputError("exactly one of input_ids/text required")
    seqs = inputs["input_ids"] if has_ids else inputs["text"]
    if not isinstance(seqs, list) or not seqs:
        raise S0InputError("sequences must be a nonempty list")
    if has_ids and not all(isinstance(s, list) for s in seqs):
        raise S0InputError("input_ids must be a list of token-id lists")
    if has_text and not all(isinstance(s, str) and s for s in seqs):
        raise S0InputError("text must be a list of nonempty strings")
    if has_ids and not all(isinstance(t, int) and t >= 0 for s in seqs for t in s):
        raise S0InputError("token ids must be nonnegative ints")
    if horizon not in _allowed_horizons:
        raise S0InputError(f"horizon must be one of {_allowed_horizons}")
    budget = budget or {}
    bad = [k for k in budget if k not in _budget_keys]
    if bad:
        raise S0InputError(f"unknown budget keys {bad}")
    if any((not isinstance(v, int)) or v <= 0 for v in budget.values()):
        raise S0InputError("budget values must be positive ints")
    if budget.get("max_workspace_passes") is not None \
            and budget["max_workspace_passes"] < horizon:
        raise S0InputError("max_workspace_passes below requested horizon")

    meta = verify_envelope(memory) if memory is not None else verify_envelope(empty_envelope())
    # TRUE BYPASS: n_positions == 0 (None or authenticated empty) adds
    # zero memory prefix positions either way.
    memory_positions = meta["n_positions"]
    mask = (memory or {}).get("mask") if memory is not None else None
    active = sum(mask) if mask is not None else memory_positions
    return {
        "schema": SCHEMA_VERSION,
        "batch": len(seqs),
        "input_kind": "ids" if has_ids else "text",
        "memory_positions": memory_positions,
        "memory_positions_active": active,
        "memory_branch": meta["branch"],
        "workspace_positions": WORKSPACE_POSITIONS if horizon > 0 else 0,
        "workspace_passes": horizon,
        "recurrence_block": list(BLOCK) if horizon > 0 else None,
        "prefix_positions": memory_positions,   # bypass adds none when 0
        "budget": {k: budget.get(k) for k in _budget_keys if k in budget},
        "temporal_horizon": {"as_of": cond["as_of"],
                             "branch": cond["branch"],
                             "max_age": cond["max_age"],
                             "note": "evidence legality only; never changes "
                                     "workspace passes"},
        "latent_step_budget": budget.get("max_workspace_passes", horizon),
        "conditioning": {"role": cond["role"], "types": cond["types"],
                         "permissions": cond["permissions"]},
        "_cond": cond,
        "decoder": "complete-pretrained-decoder-retained",
        "empty_bypass": memory_positions == 0,
    }


# --- S0-Z3: native prefix states, connector contract, cache contract ----


class ResamplerConnector:
    """RETIRED hash scaffold, never a trainable or production connector.
    Retained only for historical fixture compatibility. Production uses
    gemma4_receiver.MemoryResampler. This obsolete fixture contract:

    - accepts 0..MEMORY_POSITIONS native-width rows (each WIDTH wide);
    - emits exactly len(rows) prefix positions (native width), an
      index_map (logical positions) and a per-position mask;
    - is deterministic given (rows, seed);
    - declares which decoder positions the prefix occupies (prepended
      BEFORE user input, per the architecture draft: memory attachment
      order "first");
    - never touches the model or its caches (pure tensor function).
    """

    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, rows):
        if not rows:
            return {"prefix": [], "index_map": [], "mask": []}
        if not (0 < len(rows) <= MEMORY_POSITIONS):
            raise S0MemoryError("connector accepts 1..MEMORY_POSITIONS rows")
        import hashlib
        out = []
        for i, r in enumerate(rows):
            h = hashlib.sha256(bytes([self.seed]) + bytes(r[:32])).digest()
            out.append(bytes(h[:4]) * (WIDTH // 4) if len(r) >= WIDTH
                       else bytes(h[:4]) * (WIDTH // 4))
        return {"prefix": out, "index_map": list(range(len(rows))),
                "mask": [1] * len(rows)}


CACHE_CONTRACT = {
    "evidence_kv": "immutable, layer-specific; written once at ingestion; "
                   "never mutated by workspace passes or decoding",
    "scratch": "replaced per workspace iteration; zeroed between calls; "
               "never persists across _legacy_forward() calls",
    "logical_positions": "fixed once allocated (index_map); a masked row "
                         "releases its logical position only via an "
                         "explicit new envelope, never silently",
    "local_windows": "sliding-window legality (1024-token window; 5-sliding/"
                     "1-full pattern) is the decoder's contract; prefix "
                     "positions count toward window occupancy",
    "downstream_finalization": "workspace changes at iteration t propagate "
                               "through downstream layers before decoding; "
                               "final states are the only decode-visible "
                               "ones",
}


def cache_contract_check(connector_rows=None):
    """Static verification that the connector output and cache contract
    are mutually consistent (S0-Z3 CPU fixture core)."""
    conn = ResamplerConnector(seed=0)
    rows = connector_rows or [bytes(WIDTH), bytes(WIDTH), bytes(WIDTH)]
    out = conn(rows)
    assert len(out["prefix"]) == len(rows) == len(out["index_map"]) == len(out["mask"])
    assert all(len(r) == WIDTH for r in out["prefix"])
    assert out["index_map"] == list(range(len(rows)))
    assert all(m == 1 for m in out["mask"])
    # determinism
    out2 = conn(rows)
    assert out == out2, "connector must be deterministic"
    # empty input -> empty prefix (true bypass compatible)
    empty = conn([])
    assert empty == {"prefix": [], "index_map": [], "mask": []}
    # cache contract fields documented
    assert set(CACHE_CONTRACT) == {"evidence_kv", "scratch",
                                   "logical_positions", "local_windows",
                                   "downstream_finalization"}
    return {"connector_rows": len(rows), "prefix_width": WIDTH,
            "deterministic": True, "empty_bypass_compatible": True,
            "cache_contract_keys": sorted(CACHE_CONTRACT)}


# --- S0-Z4 CPU-side prep: live protocol skeleton + receipt formats ----
# Actual-model execution is cloud-gated (queue preconditions + fresh Codex
# LAUNCHABLE review). Everything below is preparation the live path calls;
# CPU fixtures validate its structure without weights.


LIVE_SMOKE_PROTOCOL = {
    "positive_control_phase": [
        "tokenizer alignment on fixed synthetic episodes",
        "original vs None vs empty parity (frozen deterministic comparison)",
        "same-scaffold zero-extra-pass parity (separate check)",
        "eight paired live controls: content replacement, image necessity, "
        "decode-time conditioning, empty/None parity, cached/uncached, "
        "singleton/reordered batches",
        "fixed-input repeat and batch-shape noise (terminal tolerances)",
        "finite outputs; per-device placement and peak memory",
    ],
    "competence_phase": [
        "20 explicit-evidence cases (>=19 correct AND terminated)",
        "constant-answer baseline and diversity reported beside accuracy",
    ],
    "fit_probe_phase": [
        "10 optimizer steps, microbatch 1, effective batch 4 (accumulation)",
        "gradient-safe checkpointing; selective answer logits; no "
        "generation KV",
        "finite nonzero gradients on every intended trainable path; "
        "frozen-stem gradient status logged",
        "per-device peak allocated/reserved; >=10% headroom",
    ],
    "stop_conditions": [
        "any failed control/parity/alignment/cache check",
        "custody mismatch; OOM; unexpected CPU/disk offload",
        "nonfinite or absent intended gradients",
        "insufficient per-device headroom; stalled progress; work deadline",
    ],
}


# --- S0 workspace adapter: trainable-path contract (Z4 fit probe) ------
# The ten-step fit probe requires "finite nonzero gradients on every
# intended trainable path". This enumerates those paths and their shape
# contracts so the gradient gate has a frozen checklist. CPU fixtures
# verify the enumeration and shapes with a stub; the trained modules are
# fitted on the cloud path only (frozen stem per commitment 2).

TRAINABLE_PATHS = {
    "connector.resampler": {
        "role": "memory payload rows -> native-width prefix states",
        "input": f"(n_rows, {WIDTH})", "output": f"(n_rows, {WIDTH})",
        "frozen": False},
    "connector.input_proj": {
        "role": "prefix states -> decoder input embedding space",
        "input": f"(n_rows, {WIDTH})", "output": f"(n_rows, {WIDTH})",
        "frozen": False},
    "workspace.boundary_adapters": {
        "role": "entry/exit adapters around block 24-31 passes",
        "input": f"(8, {WIDTH})", "output": f"(8, {WIDTH})",
        "frozen": False},
    "workspace.iteration_embedding": {
        "role": "per-pass iteration index embedding",
        "input": "(8,) int", "output": f"(8, {WIDTH})",
        "frozen": False},
    "workspace.recurrence_gate": {
        "role": "gate blending recurrent vs passthrough states",
        "input": f"(8, {WIDTH}) x2", "output": f"(8, {WIDTH})",
        "frozen": False},
    "consequence_head.structure": {
        "role": "structured consequence readout",
        "input": f"(8, {WIDTH})", "output": "answer/sources/coverage/unresolved/cost",
        "frozen": False},
    "stem.*": {"role": "the complete pretrained stem/decoder",
               "input": "any", "output": "any", "frozen": True},
}


def trainable_path_report():
    """CPU fixture: the enumeration is complete (every live-path component
    appears), the stem is the only frozen path, and shapes are WIDTH-
    consistent. The fit probe's gradient gate consumes exactly these keys."""
    paths = dict(TRAINABLE_PATHS)
    trainable = sorted(k for k, v in paths.items() if not v["frozen"])
    frozen = sorted(k for k, v in paths.items() if v["frozen"])
    assert frozen == ["stem.*"], "exactly the stem is frozen"
    for k in trainable:
        assert str(WIDTH) in str(paths[k]["input"]) or str(WIDTH) in str(paths[k]["output"]), k
    return {"trainable": trainable, "frozen": frozen,
            "n_trainable_paths": len(trainable),
            "gradient_gate_checklist": trainable + ["stem.requires_grad == False"]}


def live_receipt_template(stage, status="pending"):
    """Canonical receipt format for the cloud phases; written under
    experiments/results/cloud/s0-memory-port/<attempt-id>/."""
    return {"stage": stage, "status": status, "schema": SCHEMA_VERSION,
            "attempt_id": None, "pinned": {"model": None, "revision": None,
                                           "config_sha256": None,
                                           "code_sha256": None},
            "controls": {}, "metrics": {}, "metering": {},
            "interpretation": "S0 instrument evidence only; no reusable "
                              "memory, reasoning, transfer or economic claim"}


def validate_live_protocol():
    """CPU fixture: the protocol skeleton is complete and internally
    consistent (every stop condition maps to a phase; every phase has a
    receipt home)."""
    phases = set(LIVE_SMOKE_PROTOCOL) - {"stop_conditions"}
    stops = LIVE_SMOKE_PROTOCOL["stop_conditions"]
    assert len(phases) == 3 and all(LIVE_SMOKE_PROTOCOL[p] for p in phases)
    assert len(stops) >= 4
    templates = {ph: live_receipt_template(ph) for ph in phases}
    assert all(t["status"] == "pending" and t["schema"] == SCHEMA_VERSION
               for t in templates.values())
    return {"phases": sorted(phases), "stop_conditions": len(stops),
            "receipt_templates": sorted(templates)}


# --- CPU fixture checks (S0-Z1 acceptance) --------------------------------


def _receipt(check, status, cfg_path, detail=None, **extra):
    legacy = check in {"s0-interface", "s0-serialization", "s0-native-prefix",
                       "s0-live-prep", "s0-trainable-paths", "s0-callgraph"}
    return {"check": check, "status": status,
            "evidence_scope": ("RETIRED historical scaffold; not production acceptance"
                               if legacy else "bounded offline mechanics"),
            "config_path": str(cfg_path),
            "config_sha256": _sha(Path(cfg_path).read_bytes()),
            "code_sha256": _sha(Path(__file__).read_bytes()),
            "cuda_requested": False,
            "detail": detail or {}, **extra}


def _config_consistency(cfg_path):
    """Open-2 from the pre-review: config 'common' must match the module
    constants, so a wrong or drifted config cannot pass checks silently."""
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    c = cfg.get("common", {})
    expect = {"width": WIDTH, "n_layers": N_LAYERS,
              "recurrence_block": list(BLOCK),
              "workspace_positions": WORKSPACE_POSITIONS,
              "memory_positions": MEMORY_POSITIONS}
    mism = {k: (c.get(k), v) for k, v in expect.items() if c.get(k) != v}
    if mism:
        raise S0InputError(f"config/code constant mismatch: {mism}")
    return True


def _check_interface(cfg_path):
    """Retired scaffold check; run in a fresh process (asserts import absence)."""
    detail = {}

    _config_consistency(cfg_path)
    def expect_error(fn, *a, **k):
        try:
            fn(*a, **k)
            return False
        except (S0InputError, S0MemoryError):
            return True

    # 1) same reasoner + declared routing: plan is well-formed
    try:
        p = _legacy_forward(None, {"text": ["hello"]}, None, horizon=1)
        plan = p["plan"]
        assert p["mode"] == "cpu-plan"
    except Exception as exc:
        return _receipt("s0-interface", "FAIL", cfg_path,
                        {"reason": f"plan-mode forward failed: {exc!r}"},
                        exit_code=1), 1
    detail["plan_keys"] = sorted(plan)
    # 2) TRUE EMPTY BYPASS: None == authenticated empty, byte-identical
    pe = _legacy_forward(None, {"text": ["hello"]}, empty_envelope(), horizon=1)
    detail["empty_bypass_identical"] = json.dumps(pe["plan"], sort_keys=True) == \
        json.dumps(plan, sort_keys=True)
    assert detail["empty_bypass_identical"]
    assert plan["prefix_positions"] == 0 and plan["empty_bypass"]
    # non-empty memory adds exactly its positions
    row = bytes(WIDTH)  # real-width row (CRITICAL-1 fix enforces WIDTH)
    pm = _legacy_forward(None, {"text": ["hi"]}, make_envelope([row, row]), horizon=0)
    assert pm["plan"]["prefix_positions"] == MEMORY_POSITIONS // 8  # 2 rows
    detail["memory_positions_routed"] = True
    # 3) horizon/budget routing table
    routing = {}
    for h in (0, 1, 2):
        r = _legacy_forward(None, {"text": ["x"]}, None, horizon=h)["plan"]
        routing[h] = (r["workspace_passes"], r["workspace_positions"], r["recurrence_block"])
    detail["routing"] = {str(k): v for k, v in routing.items()}
    assert routing[0][1] == 0 and routing[2][0] == 2 and routing[1][2] == [24, 31]
    b = _legacy_forward(None, {"text": ["x"]}, None, horizon=1,
                budget={"max_workspace_passes": 2, "max_new_tokens": 8})["plan"]
    assert b["budget"]["max_workspace_passes"] == 2
    # 4) invalid inputs rejected
    rej = [
        expect_error(_legacy_forward, None, {}, None, 1),                       # both/neither kind
        expect_error(_legacy_forward, None, {"text": []}, None, 1),             # empty batch
        expect_error(_legacy_forward, None, {"text": ["x"]}, None, 3),          # bad horizon
        expect_error(_legacy_forward, None, {"text": ["x"]}, None, 1, {"nope": 1}),   # bad budget key
        expect_error(_legacy_forward, None, {"text": ["x"]}, None, 1, {"max_workspace_passes": 0}),  # <=0
        expect_error(_legacy_forward, None, {"text": ["x"]}, None, 2, {"max_workspace_passes": 1}),  # < horizon
        expect_error(_legacy_forward, None, {"input_ids": [[-1]]}, None, 1),    # negative id
    ]
    detail["invalid_inputs_rejected"] = all(rej)
    assert all(rej)
    # 5) tampered envelope rejected; stale schema rejected
    env = make_envelope([row])
    bad_env = dict(env, meta=dict(env["meta"], branch="evil"))
    detail["tamper_rejected"] = expect_error(_legacy_forward, None, {"text": ["x"]}, bad_env, 1)
    stale = dict(empty_envelope(), meta=dict(empty_envelope()["meta"],
                                             schema="s0-memory-port/0"))
    detail["stale_schema_rejected"] = expect_error(_legacy_forward, None, {"text": ["x"]}, stale, 1)
    assert detail["tamper_rejected"] and detail["stale_schema_rejected"]
    # 6) the model loader is NEVER called on fixture paths: assert the
    # loader module was not imported as a side effect of this check
    loader_mod = "experiments.cloud.lsr_jobs.common"
    detail["loader_module_absent_after_fixture"] = loader_mod not in sys.modules
    assert detail["loader_module_absent_after_fixture"]
    def _sanitize(o):
        if isinstance(o, bytes):
            return f"<{len(o)} bytes>"
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        return o
    return _receipt("s0-interface", "PASS", cfg_path, _sanitize(detail), exit_code=0), 0


def _check_serialization(cfg_path):
    """S0-Z2 acceptance: serialized reload preserves legal payload/index/
    mask alignment; authenticated empty round-trips to the true bypass;
    tampering with any component rejects; origin custody validates."""
    import base64
    import copy

    _config_consistency(cfg_path)
    def expect_reject(fn, *a, **k):
        try:
            fn(*a, **k)
            return False
        except (S0InputError, S0MemoryError):
            return True

    origin = [{"origin_type": "observation", "modality": "text",
               "source_sha256": _sha(b"doc-a"), "locator": "docs/a.md#L1",
               "event_time": "2026-10-15T00:00:00Z",
               "ingestion_time": "2026-10-15T01:00:00Z",
               "valid_from": "2026-10-15T00:00:00Z", "valid_until": None,
               "branch": "main", "dependencies": []}]
    rows = [bytes([i]) * WIDTH for i in range(1, 4)]     # 3 real-width rows
    env = make_envelope(rows, origin=origin,
                        index_map=[0, 2, 5], mask=[1, 0, 1])
    # 1) round-trip preserves payload, index, mask and legal metadata
    blob = serialize_envelope(env)
    rt = deserialize_envelope(blob)
    ok_rt = (rt["meta"] == env["meta"] and rt["auth"] == env["auth"]
             and rt["payload"] == env["payload"]
             and rt.get("index_map") == [0, 2, 5]
             and rt.get("mask") == [1, 0, 1])
    # 2) authenticated empty round-trips to the true bypass
    eb = serialize_envelope(empty_envelope())
    ert = deserialize_envelope(eb)
    p_none = _legacy_forward(None, {"text": ["x"]}, None, 1)["plan"]
    p_empty = _legacy_forward(None, {"text": ["x"]}, ert, 1)["plan"]
    ok_bypass = json.dumps(p_none, sort_keys=True) == json.dumps(p_empty, sort_keys=True)         and p_empty["empty_bypass"] and p_empty["prefix_positions"] == 0
    # 3) tamper battery: meta/auth/payload/index/mask each reject
    obj = json.loads(blob)
    tampered = []
    t1 = dict(obj, meta=dict(obj["meta"], branch="evil"))
    tampered.append(("meta", json.dumps(t1)))
    t2 = dict(obj, auth="0" * 64)
    tampered.append(("auth", json.dumps(t2)))
    raw = base64.b64decode(obj["payload_b64"])
    t3 = dict(obj, payload_b64=base64.b64encode(raw[:WIDTH] + bytes(WIDTH) + raw[2 * WIDTH:]).decode())
    tampered.append(("payload", json.dumps(t3)))
    t4 = dict(obj, index_map=[0, 2, 6])
    tampered.append(("index_map", json.dumps(t4)))
    t5 = dict(obj, mask=[1, 0])
    tampered.append(("mask", json.dumps(t5)))
    rejects = {name: expect_reject(deserialize_envelope, b) for name, b in tampered}
    ok_tamper = all(rejects.values())
    # 4) origin custody validation
    bad_origin = [dict(origin[0], origin_type="hallucination")]
    ok_origin = expect_reject(make_envelope, rows, origin=bad_origin)
    # 5) reload feeds _legacy_forward() and adds exactly 3 positions (2 masked -> active count from mask)
    plan = _legacy_forward(None, {"text": ["q"]}, rt, 0)["plan"]
    ok_positions = plan["memory_positions"] == 3
    return _receipt("s0-serialization", "PASS" if (ok_rt and ok_bypass and ok_tamper
                                                   and ok_origin and ok_positions) else "FAIL",
                    cfg_path,
                    {"roundtrip_preserved": ok_rt, "empty_bypass_after_reload": ok_bypass,
                     "tamper_rejections": rejects, "origin_custody_validated": ok_origin,
                     "reloaded_memory_positions": plan["memory_positions"],
                     "note": "payload/index/mask alignment enforced on reload; "
                             "authenticated empty takes the true bypass"},
                    exit_code=0 if (ok_rt and ok_bypass and ok_tamper
                                    and ok_origin and ok_positions) else 1),            (0 if (ok_rt and ok_bypass and ok_tamper and ok_origin and ok_positions) else 1)


def _check_native_prefix(cfg_path):
    """S0-Z3 acceptance: 16 5376-wide rows through the connector; shape/
    mask/index alignment; omitted/no-op/wrong payload rejected;
    source-free reload feeds the connector; cache contract documented."""
    def expect_reject(fn, *a, **k):
        try:
            fn(*a, **k)
            return False
        except (S0InputError, S0MemoryError):
            return True

    _config_consistency(cfg_path)
    # full-width 16-row envelope (real WIDTH bytes/row)
    rows16 = [bytes([i + 1]) * WIDTH for i in range(MEMORY_POSITIONS)]
    origin = [{"origin_type": "observation", "modality": "text",
               "source_sha256": _sha(bytes([i])), "locator": f"d/{i}",
               "event_time": "2026-10-15T00:00:00Z",
               "ingestion_time": "2026-10-15T00:00:00Z",
               "valid_from": "2026-10-15T00:00:00Z", "valid_until": None,
               "branch": "main", "dependencies": []} for i in range(MEMORY_POSITIONS)]
    env = make_envelope(rows16, origin=origin)
    blob = serialize_envelope(env)
    rt = deserialize_envelope(blob)   # source-free reload
    conn_info = cache_contract_check(list(rt["payload"]))
    # 17 rows rejected
    ok_cap = expect_reject(make_envelope, rows16 + [bytes(WIDTH)],
                           origin=origin + [origin[0]])
    # connector rejects >MEMORY_POSITIONS
    ok_conn_cap = expect_reject(ResamplerConnector(0), rows16 + [bytes(WIDTH)])
    # reload+connector+forward chain: plan sees 16 positions
    plan = _legacy_forward(None, {"text": ["q"]}, rt, 1)["plan"]
    ok_chain = plan["memory_positions"] == MEMORY_POSITIONS
    ok = conn_info["deterministic"] and conn_info["empty_bypass_compatible"]         and ok_cap and ok_conn_cap and ok_chain
    return _receipt("s0-native-prefix", "PASS" if ok else "FAIL", cfg_path,
                    {"connector": conn_info, "capacity_rejected": ok_cap,
                     "connector_capacity_rejected": ok_conn_cap,
                     "chain_memory_positions": plan["memory_positions"],
                     "cache_contract": CACHE_CONTRACT,
                     "note": "16 real-width rows -> serialized -> reloaded "
                             "source-free -> connector (deterministic stub) "
                             "-> forward plan; cache contract documented "
                             "verbatim for the S0-Z4 implementation"},
                    exit_code=0 if ok else 1), (0 if ok else 1)


def _check_live_prep(cfg_path):
    """S0-Z4 CPU-side prep acceptance: the live protocol skeleton and
    receipt formats are complete and consistent. NOT actual-model
    evidence (cloud-gated)."""
    _config_consistency(cfg_path)
    info = validate_live_protocol()
    ok = (set(info["phases"]) == {"competence_phase", "fit_probe_phase",
                                  "positive_control_phase"}
          and info["stop_conditions"] >= 4
          and set(info["receipt_templates"]) == set(info["phases"]))
    return _receipt("s0-live-prep", "PASS" if ok else "FAIL", cfg_path,
                    {**info,
                     "note": "preparation only; execution requires queue "
                             "preconditions + fresh Codex LAUNCHABLE review; "
                             "CPU/stub receipts are not actual-model evidence"},
                    exit_code=0 if ok else 1), (0 if ok else 1)

    _config_consistency(cfg_path)

def _check_trainable_paths(cfg_path):
    """S0-Z4 gradient-gate checklist: the intended trainable paths are
    enumerated, the stem is the only frozen path, shapes are WIDTH-consistent."""
    _config_consistency(cfg_path)
    rep = trainable_path_report()
    ok = (rep["n_trainable_paths"] >= 5 and rep["frozen"] == ["stem.*"]
          and "stem.requires_grad == False" in rep["gradient_gate_checklist"])
    return _receipt("s0-trainable-paths", "PASS" if ok else "FAIL", cfg_path,
                    {**rep,
                     "note": "the ten-step fit probe asserts finite nonzero "
                             "gradients on exactly these trainable paths and "
                             "zero stem gradients; frozen-stem status is logged"},
                    exit_code=0 if ok else 1), (0 if ok else 1)


def _check_callgraph(cfg_path):
    """18:05 council items 1-2: recording-receiver falsifier against the
    RETIRED _legacy_forward scaffold. Identity connector; receiver returns a unique
    sentinel object plus a value computed only from what it received.
    Passes iff: payload dependence (active-row change -> different
    received content and value), sentinel identity (forward returns the
    receiver's exact object), masked-row invariance, index-assignment
    sensitivity, None == authenticated-empty bypass, and rejection of
    disconnected/dropped-output variants."""
    _config_consistency(cfg_path)
    calls = []

    class Receiver:
        def __call__(self, **kw):
            self.last_kwargs = kw
            active = [p for p, m in zip(kw.get("prefix_states") or [],
                                        kw.get("attach_mask") or []) if m]
            # value derived ONLY from received active content + positions
            content = b"".join(bytes(r[:8]) for r in active)
            positions = tuple(kw.get("logical_positions") or ())
            self.sentinel = {"receiver_marker": id(self),
                             "received_digest": kw.get("assembly_digest"),
                             "value": _sha(content + repr(positions).encode())}
            calls.append(kw)
            return self.sentinel

    def identity_connector(rows):
        return {"prefix": list(rows), "index_map": list(range(len(rows))),
                "mask": [1] * len(rows)}

    def make_handle():
        return {"tok": lambda t, **k: {"input_ids": [[1, 2, 3]]},
                "connector": identity_connector,
                "receiver": Receiver()}

    W = WIDTH
    def row(b):
        return bytes([b]) * W

    origin = lambda n: [{"origin_type": "observation", "modality": "text",
                         "source_sha256": _sha(bytes([i])),
                         "locator": f"l{i}", "event_time": "t",
                         "ingestion_time": "t", "valid_from": "t",
                         "valid_until": None, "branch": "main",
                         "dependencies": []} for i in range(n)]
    envA = make_envelope([row(1), row(2)], origin=origin(2))
    envB = make_envelope([row(1), row(9)], origin=origin(2))  # one active row differs
    d = {}

    h1 = make_handle()
    r1 = _legacy_forward(h1, {"text": ["q"]}, envA, 1)
    h2 = make_handle()
    r2 = _legacy_forward(h2, {"text": ["q"]}, envB, 1)
    k1, k2 = h1["receiver"].last_kwargs, h2["receiver"].last_kwargs
    d["different_payload_different_received"] = (
        k1.get("prefix_states") != k2.get("prefix_states"))
    d["content_derived_value_differs"] = (r1["value"] != r2["value"])
    d["exact_receiver_sentinel_returned"] = (
        r1 is h1["receiver"].sentinel and r2 is h2["receiver"].sentinel)

    # masked-row invariance: change only a MASKED row
    envM1 = make_envelope([row(1), row(2)], origin=origin(2), mask=[1, 0])
    envM2 = make_envelope([row(1), row(7)], origin=origin(2), mask=[1, 0])
    h3, h4 = make_handle(), make_handle()
    # custom connector honoring the mask
    def masked_connector(rows, mask=None):
        return {"prefix": list(rows), "index_map": list(range(len(rows))),
                "mask": mask or [1] * len(rows)}
    # emulate masked assembly via env mask passed through forward
    r3 = _legacy_forward({"tok": make_handle()["tok"],
                  "connector": lambda rows: masked_connector(rows, envM1["mask"]),
                  "receiver": h3["receiver"]}, {"text": ["q"]}, envM1, 1)
    r4 = _legacy_forward({"tok": make_handle()["tok"],
                  "connector": lambda rows: masked_connector(rows, envM2["mask"]),
                  "receiver": h4["receiver"]}, {"text": ["q"]}, envM2, 1)
    # active content (row 1) identical -> same value
    d["masked_row_invariant"] = (r3["value"] == r4["value"])

    # index-assignment sensitivity
    envI1 = make_envelope([row(1), row(2)], origin=origin(2), index_map=[0, 1])
    envI2 = make_envelope([row(1), row(2)], origin=origin(2), index_map=[3, 1])
    h5, h6 = make_handle(), make_handle()
    r5 = _legacy_forward(h5, {"text": ["q"]}, envI1, 1)
    r6 = _legacy_forward(h6, {"text": ["q"]}, envI2, 1)
    d["index_assignment_changes_positions"] = (
        h5["receiver"].last_kwargs.get("logical_positions")
        != h6["receiver"].last_kwargs.get("logical_positions"))

    # None == authenticated-empty bypass: same call signature, no attachment
    h7, h8 = make_handle(), make_handle()
    _legacy_forward(h7, {"text": ["q"]}, None, 1)
    _legacy_forward(h8, {"text": ["q"]}, empty_envelope(), 1)
    k7, k8 = h7["receiver"].last_kwargs, h8["receiver"].last_kwargs
    d["none_empty_same_bypass"] = (
        k7.get("prefix_states") is None and k8.get("prefix_states") is None
        and k7.get("attach_mask") is None and k8.get("attach_mask") is None)

    # negative controls: disconnected/dropped-output variants rejected
    def broken_receiver(**kw):
        return None  # dropped output
    h9 = {"tok": make_handle()["tok"], "connector": identity_connector,
          "receiver": broken_receiver}
    try:
        out = _legacy_forward(h9, {"text": ["q"]}, envA, 1)
        d["dropped_output_rejected"] = False
    except (S0InputError, S0MemoryError):
        d["dropped_output_rejected"] = True
    # disconnected prefix: receiver ignores prefix kwarg entirely
    class Disconnected:
        def __call__(self, **kw):
            self.last_kwargs = kw
            self.sentinel = {"receiver_marker": id(self), "value": "const"}
            return self.sentinel
    h10 = {"tok": make_handle()["tok"], "connector": identity_connector,
           "receiver": Disconnected()}
    try:
        _legacy_forward(h10, {"text": ["q"]}, envA, 1)
        d["disconnected_prefix_rejected"] = False
    except (S0InputError, S0MemoryError):
        d["disconnected_prefix_rejected"] = True

    ok = all(v for v in d.values())
    return _receipt("s0-callgraph", "PASS" if ok else "FAIL", cfg_path,
                    {**d,
                     "note": "RETIRED legacy forward under test; identity "
                             "connector; receiver value derived only from "
                             "received active content and positions"},
                    exit_code=0 if ok else 1), (0 if ok else 1)


def _handle_contract(handle):
    """The loader owns the stem and the receiver owns the sole resampler."""
    from .gemma4_receiver import Gemma4Receiver
    if not isinstance(handle, dict) or not isinstance(handle.get("receiver"), Gemma4Receiver):
        raise S0InputError("production requires a loaded Gemma4Receiver handle")
    receiver = handle["receiver"]
    info = handle.get("stem_info", {})
    if (handle.get("connector") is not receiver.connector
            or info.get("D_memory") != receiver.connector.config.memory_dim
            or info.get("D_stem") != receiver.stem.config.hidden_size):
        raise S0InputError("verified stem_info/connector dimensions required")
    return receiver, info["D_memory"]


def _production_budget(budget):
    budget = {} if budget is None else budget
    if not isinstance(budget, dict) or set(budget) - set(_budget_keys):
        raise S0InputError("unknown budget keys")
    if any(type(v) is not int or v <= 0 for v in budget.values()):
        raise S0InputError("budget values must be positive integers")
    if "selective_logits" in budget:
        raise S0InputError("selective logits unsupported by the released receiver")
    return dict(budget)


def _native_attachment(handle, memory, conditioning, batch):
    """Select sealed metadata, but index the original live tensor for autograd.

    A dict is one sample; a list supplies one independently sealed envelope per
    sample. Raw [B,R,D] tensors are accepted only without host policy (no provenance
    to authorize selection). Stored NativeTensor carriers reload independently.
    """
    from . import s0_envelope as env
    receiver, width = _handle_contract(handle)
    parameter = receiver.connector.queries
    kwargs = {}
    if memory is None:
        if conditioning:
            # Even bypass validates supplied policy rather than silently dropping it.
            env.select([], conditioning, expected_width=width,
                       expected_schema=env.WIDTH_SCHEMA_VERSION)
        return kwargs, []
    if isinstance(memory, torch.Tensor):
        if conditioning:
            raise S0MemoryError("raw memory has no origin policy; use typed envelopes")
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[2] != width:
            raise S0MemoryError("raw memory must be [B,R,D_memory]")
        if not memory.is_floating_point() or not torch.isfinite(memory).all():
            raise S0MemoryError("native model memory must be finite floating values")
        if memory.shape[1] > MEMORY_POSITIONS:
            raise S0MemoryError("too many memory rows")
        return {"memory": memory.to(parameter), "memory_mask": None}, []
    envelopes = memory if isinstance(memory, list) else [memory]
    if len(envelopes) != batch:
        raise S0MemoryError("one typed envelope per sample required")
    rows, roles, types, times, indices, refusals = [], [], [], [], [], []
    for sample, attachment in enumerate(envelopes):
        if not isinstance(attachment, dict):
            raise S0MemoryError("expected typed envelope")
        schema = attachment.get("meta", {}).get("schema")
        contract = {"expected_width": width, "expected_schema": schema}
        meta = env.verify_envelope(attachment, **contract)
        if meta["n_positions"] > MEMORY_POSITIONS:
            raise S0MemoryError("too many memory rows")
        host = conditioning
        if host is None:
            # Never invent a wall clock: nonempty memory must carry an as_of receipt.
            host = {"branch": meta["branch"],
                    "as_of": meta["temporal_horizon"]["as_of"],
                    "max_age": meta["temporal_horizon"]["max_age"], "permissions": ["read"]}
            if meta["n_positions"] == 0 and host["as_of"] is None:
                host = dict(host, as_of=0)
        selected, rejected = env.select([attachment], host, **contract)
        refusals.extend(dict(r, sample=sample) for r in rejected)
        rejected_rows = {r["row"] for r in rejected}
        keep = [i for i in range(meta["n_positions"]) if i not in rejected_rows]
        payload = attachment["payload"]
        live = payload if isinstance(payload, torch.Tensor) else env.to_torch(payload)
        if not live.is_floating_point() or not torch.isfinite(live).all():
            raise S0MemoryError("native model memory must be finite floating values")
        live = live.index_select(0, torch.tensor(keep, device=live.device, dtype=torch.long))
        # select creates a detached integrity view; it never replaces live rows.
        role, time = env.role_time_tensors(selected, **contract)
        origins = [] if not selected else selected[0]["meta"]["origin"]
        rows.append(live.to(parameter))
        roles.append(role.to(device=parameter.device))
        times.append(time.to(device=parameter.device))
        types.append(torch.tensor([("text", "image", "video", "audio").index(r["modality"])
                                   for r in origins], device=parameter.device, dtype=torch.long))
        indices.append([] if not selected else selected[0]["index_map"])
        if indices[-1] != [attachment["index_map"][i] for i in keep]:
            raise S0MemoryError("selected logical indices lost alignment")
    if not any(len(r) for r in rows):
        return {}, refusals
    if any(len(r) == 0 for r in rows):
        raise S0MemoryError("mixed empty/nonempty samples require separate calls")
    count = max(len(r) for r in rows)
    def pad(values, value=0):
        return torch.stack([torch.cat((v, v.new_full((count-len(v), *v.shape[1:]), value)))
                            for v in values])
    kwargs = {"memory": pad(rows),
              "memory_mask": torch.tensor([[i < len(r) for i in range(count)] for r in rows],
                                          device=parameter.device, dtype=torch.bool),
              "role_ids": pad(roles), "type_ids": pad(types), "time_offsets": pad(times)}
    # Native logical indices select rows; eight learned slots have their own 0..7
    # positions. Never confuse the original row addresses with workspace positions.
    return kwargs, refusals


def forward(handle, inputs, memory=None, horizon=0, budget=None, conditioning=None):
    """Production optional memory port; live loss/cache objects belong to receiver.

    Query/padding labels are ignored using optional query_mask and attention_mask.
    Other prompt labels must already be -100. Training never enables KV caching.
    Nonzero recurrent horizons are unsupported, independently of temporal policy.
    """
    if type(horizon) is not int or horizon != 0:
        raise S0InputError("prefix prototype requires horizon=0")
    limits = _production_budget(budget)
    receiver, _ = _handle_contract(handle)
    if not isinstance(inputs, dict) or ("input_ids" in inputs) == ("text" in inputs):
        raise S0InputError("exactly one of input_ids/text required")
    allowed = {"input_ids", "text", "attention_mask", "labels", "query_mask", "use_cache"}
    if set(inputs) - allowed:
        raise S0InputError("unknown input fields")
    enc = inputs
    if "text" in inputs:
        if not callable(handle.get("tok")):
            raise S0InputError("text inputs require tokenizer")
        enc = {**handle["tok"](inputs["text"], add_special_tokens=False,
                               return_tensors="pt", padding=True), **inputs}
    device = receiver.stem.get_input_embeddings().weight.device
    ids = torch.as_tensor(enc["input_ids"], device=device)
    if ids.ndim != 2 or ids.dtype != torch.long or not ids.numel():
        raise S0InputError("input_ids must be nonempty [B,T] long")
    mask = None if enc.get("attention_mask") is None else torch.as_tensor(enc["attention_mask"], device=device)
    labels = None if enc.get("labels") is None else torch.as_tensor(enc["labels"], device=device)
    if labels is not None:
        if labels.shape != ids.shape or labels.dtype != torch.long:
            raise S0InputError("labels must be aligned long")
        query = enc.get("query_mask")
        if query is not None:
            query = torch.as_tensor(query, device=device)
            if query.shape != ids.shape or not ((query == 0) | (query == 1)).all():
                raise S0InputError("query_mask must be aligned binary")
            labels = labels.masked_fill(query.bool(), -100)
        if mask is not None:
            if mask.shape != ids.shape:
                raise S0InputError("attention_mask must be aligned")
            labels = labels.masked_fill(mask == 0, -100)
    use_cache = enc.get("use_cache", False)
    if type(use_cache) is not bool or (labels is not None and use_cache):
        raise S0InputError("training requires use_cache=False")
    attachment, _ = _native_attachment(handle, memory, conditioning, ids.shape[0])
    digest = None
    if isinstance(memory, dict):
        digest = memory.get("digest")  # custody only; never a content-use gate
    return receiver(input_ids=ids, attention_mask=mask, labels=labels,
                    use_cache=use_cache, horizon=horizon, budget=limits,
                    assembly_digest=digest, **attachment)


@torch.no_grad()
def generate(handle, inputs, memory=None, *, budget, conditioning=None, eos_ids=()):
    """Batch-one greedy inference; trim trailing pads before cache construction.

    Leading/internal masked positions retain their physical positions. Decode
    appends immediately after the final visible token, using that same cache.
    """
    limits = _production_budget(budget)
    if "max_new_tokens" not in limits or "max_total_forward_calls" not in limits:
        raise S0InputError("generation needs token and total-call bounds")
    if inputs.get("labels") is not None:
        raise S0InputError("generation must not receive evaluation labels")
    receiver, _ = _handle_contract(handle)
    if ("input_ids" in inputs) == ("text" in inputs):
        raise S0InputError("exactly one of input_ids/text required")
    if set(inputs) - {"input_ids", "text", "attention_mask", "query_mask", "labels", "use_cache"}:
        raise S0InputError("unknown input fields")
    inputs = dict(inputs)
    if "text" in inputs:
        if not callable(handle.get("tok")):
            raise S0InputError("text inputs require tokenizer")
        text = inputs.pop("text")
        inputs = {**handle["tok"](text, add_special_tokens=False,
                                  return_tensors="pt", padding=True), **inputs}
    ids = torch.as_tensor(inputs["input_ids"])
    if ids.ndim != 2 or ids.shape[0] != 1 or not ids.numel() or ids.dtype != torch.long:
        raise S0InputError("bounded generation requires nonempty batch-one long input_ids")
    mask = torch.ones_like(ids) if inputs.get("attention_mask") is None else torch.as_tensor(inputs["attention_mask"])
    if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all() or not mask.any():
        raise S0InputError("generation requires aligned binary mask with a visible token")
    end = int(torch.nonzero(mask[0], as_tuple=False)[-1].item()) + 1
    inputs["input_ids"] = ids[:, :end]
    inputs["attention_mask"] = mask[:, :end]
    if inputs.get("query_mask") is not None:
        query = torch.as_tensor(inputs["query_mask"])
        if query.shape != ids.shape:
            raise S0InputError("query_mask must be aligned")
        inputs["query_mask"] = query[:, :end]
    result = forward(handle, dict(inputs, use_cache=True), memory, budget=limits,
                     conditioning=conditioning)
    tokens, terminated, calls = [], False, 1
    if result["logits"].shape[0] != 1:
        raise S0InputError("bounded generation currently requires batch 1")
    for step in range(limits["max_new_tokens"]):
        token = result["logits"][:, -1].argmax(-1, keepdim=True)
        tokens.append(token.item())
        if token.item() in eos_ids:
            terminated = True
            break
        if step + 1 == limits["max_new_tokens"] or calls == limits["max_total_forward_calls"]:
            break
        result = receiver.decode(token, result["past_key_values"])
        calls += 1
    return {"tokens": tokens, "terminated": terminated, "empty": not tokens,
            "budget_exhausted": not terminated, "forward_calls": calls}


def _phase_case(handle, case):
    from . import s0_envelope as env
    if not isinstance(case, dict) or set(case) != {"inputs", "memory", "conditioning"}:
        raise S0InputError("case requires inputs, memory, conditioning")
    memory = case["memory"]
    if isinstance(memory, str):
        _, width = _handle_contract(handle)
        memory = env.deserialize_envelope(memory, expected_width=width,
                                         expected_schema=env.WIDTH_SCHEMA_VERSION)
    return case["inputs"], memory, case["conditioning"]


def _smoke(handle, cases):
    """Operation smoke on provided TRAIN/DEV cases, not semantic competence."""
    receiver, _ = _handle_contract(handle)
    if not cases:
        raise S0InputError("separate smoke_cases required")
    details = []
    for case in cases:
        inputs, memory, host = _phase_case(handle, case)
        if memory is None or inputs.get("labels") is None:
            raise S0InputError("smoke needs nonempty memory and answer labels")
        receiver.eval()
        receiver.zero_grad(set_to_none=True)
        out = forward(handle, inputs, memory, conditioning=host)
        loss = out["loss"]
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad or not torch.isfinite(loss):
            raise AssertionError("answer loss disconnected/nonfinite")
        loss.backward()
        norms = {n: 0.0 if p.grad is None else p.grad.float().norm().item()
                 for n, p in receiver.connector.named_parameters()}
        import math
        if any(not math.isfinite(v) or v <= 0 for v in norms.values()):
            raise AssertionError("intended connector path has absent/nonfinite gradient")
        if receiver.stem.training or any(p.requires_grad or p.grad is not None for p in receiver.stem.parameters()):
            raise AssertionError("stem must be frozen eval without gradients")
        with torch.no_grad():
            a = forward(handle, inputs, memory, conditioning=host)
            repeat = forward(handle, inputs, memory, conditioning=host)
            donor_memory = _content_replacement(memory, handle["stem_info"]["D_memory"])
            donor = forward(handle, inputs, donor_memory, conditioning=host)
            omitted = forward(handle, inputs)
            if not torch.equal(a["logits"], repeat["logits"]):
                raise AssertionError("repeat noise blocks content interpretation")
            # Bypass parity includes exact native returned loss, masks and ignored labels.
            empty = torch.empty((len(inputs["input_ids"]), 0, handle["stem_info"]["D_memory"]))
            e = forward(handle, inputs, empty)
            if not torch.equal(omitted["logits"], e["logits"]) or not torch.equal(omitted["loss"], e["loss"]):
                raise AssertionError("None/empty bypass differs")
            noise = _batch_noise(handle, inputs, memory, host, a["logits"], donor_memory)
            content_effect = _answer_effect(a["logits"], repeat["logits"], donor["logits"], inputs,
                                           noise_floor=noise)
            effect = _answer_effect(a["logits"], repeat["logits"], omitted["logits"], inputs,
                                    noise_floor=noise)
        details.append({"loss": loss.item(), "path_gradient_norms": norms,
                        "content_replacement": content_effect, "omission": effect})
    receiver.zero_grad(set_to_none=True)
    return {"cases": details, "scope": "mechanical smoke; semantic competence UNRUN"}


def _content_replacement(memory, width):
    """Change payload only; preserve shape, row policy, addresses and metadata."""
    from . import s0_envelope as env
    if isinstance(memory, torch.Tensor):
        tensor = memory.detach()
        return -tensor if torch.equal(tensor, torch.ones_like(tensor)) else 2 - tensor
    if isinstance(memory, list):
        return [_content_replacement(item, width) for item in memory]
    meta = env.verify_envelope(memory, expected_width=width,
                               expected_schema=memory["meta"]["schema"])
    payload = memory["payload"]
    tensor = payload.detach() if isinstance(payload, torch.Tensor) else env.to_torch(payload)
    # Choose a non-fixed replacement even for an all-ones carrier.
    changed = -tensor if torch.equal(tensor, torch.ones_like(tensor)) else 2 - tensor
    return env.make_envelope(changed, branch=meta["branch"], origin=meta["origin"],
                             index_map=memory["index_map"], mask=memory["mask"],
                             latent_steps=meta["compute_budget"]["latent_steps"],
                             temporal_horizon=meta["temporal_horizon"],
                             expected_width=width, expected_schema=meta["schema"])


def _batch_noise(handle, inputs, memory, host, singleton, changed=None):
    """Singleton/paired/reordered noise on the same inputs and legal attachments."""
    ids = torch.as_tensor(inputs["input_ids"])
    if ids.shape[0] != 1:
        raise S0InputError("S0 smoke requires size-one pretokenized cases")
    paired_inputs = {k: torch.as_tensor(v).repeat(2, 1) for k, v in inputs.items()
                     if k in ("input_ids", "attention_mask", "labels", "query_mask")}
    if isinstance(memory, torch.Tensor):
        pair = torch.cat((memory, memory if changed is None else changed), dim=0)
        reverse = pair.flip(0)
    else:
        def singleton_envelope(value):
            if isinstance(value, list):
                if len(value) != 1:
                    raise S0InputError("S0 smoke batch-one attachment requires exactly one envelope")
                return value[0]
            return value
        original = singleton_envelope(memory)
        replacement = original if changed is None else singleton_envelope(changed)
        pair = [original, replacement]
        reverse = list(reversed(pair))
    paired = forward(handle, paired_inputs, pair, conditioning=host)["logits"]
    reordered = forward(handle, paired_inputs, reverse, conditioning=host)["logits"].flip(0)
    if not torch.isfinite(paired).all() or not torch.isfinite(reordered).all():
        raise AssertionError("nonfinite batch-shape control")
    return max((paired[:1] - singleton).abs().max().item(),
               (paired - reordered).abs().max().item())


def _answer_effect(a, repeat, changed, inputs, tolerance=1e-6, noise_floor=0.0):
    """Receiver content changes must reach answer prediction logits, beyond noise."""
    labels = torch.as_tensor(inputs["labels"], device=a.device)
    query = inputs.get("query_mask")
    if query is not None:
        labels = labels.masked_fill(torch.as_tensor(query, device=a.device).bool(), -100)
    mask = inputs.get("attention_mask")
    if mask is not None:
        labels = labels.masked_fill(torch.as_tensor(mask, device=a.device) == 0, -100)
    target = labels[:, 1:] != -100
    if not target.any():
        raise AssertionError("smoke requires shifted answer positions for bypass comparator")
    values = [x[:, :-1][target] for x in (a, repeat, changed)]
    if not all(torch.isfinite(v).all() for v in values):
        raise AssertionError("nonfinite answer logits")
    noise = (values[0] - values[1]).abs().max().item()
    effect = (values[0] - values[2]).abs().max().item()
    if effect <= max(noise, noise_floor) + tolerance:
        raise AssertionError("constant/digest-echo receiver: no answer content effect")
    return {"answer_effect": effect, "repeat_noise": noise,
            "batch_noise_max_abs": noise_floor, "tolerance": tolerance}


def _fit_probe(handle, cases):
    """Reuse the released ten-step probe; every loss calls production forward."""
    from .gemma4_receiver import ten_step_backward_probe
    receiver, _ = _handle_contract(handle)
    class Port:
        stem, connector = receiver.stem, receiver.connector
        def train(self):
            receiver.train()
        def eval(self):
            receiver.eval()
        def __call__(self, input_ids, *, memory, conditioning, use_cache=False,
                     role_ids=None, type_ids=None, time_offsets=None, **inputs):
            # These probe declarations make its per-step intended-path gate include
            # the envelope's optional paths. forward reselects/revalidates the rows
            # and derives the same metadata; declarations cannot override policy.
            return forward(handle, dict(inputs, input_ids=input_ids, use_cache=use_cache),
                           memory, conditioning=conditioning)
    batches = []
    for case in cases:
        inputs, memory, host = _phase_case(handle, case)
        if memory is None or inputs.get("labels") is None:
            raise S0InputError("fit needs memory and answer labels")
        attachment, _ = _native_attachment(handle, memory, host, 1)
        batches.append({**inputs, "input_ids": torch.as_tensor(inputs["input_ids"]),
                        "memory": memory, "conditioning": host,
                        **{k: attachment.get(k) for k in ("role_ids", "type_ids", "time_offsets")}})
    detail = ten_step_backward_probe(Port(), batches)
    # Envelopes supply every optional categorical/time path; require every one.
    import math
    if any(not math.isfinite(v) or v <= 0 for step in detail["per_parameter_grad_norms"]
           for v in step.values()):
        raise AssertionError("fit has absent/nonfinite intended path gradient")
    detail["intended_gradient_paths"] = list(detail["per_parameter_grad_norms"][0])
    return detail


def main(cfg, out):
    """Registered S0 smoke/fit entry; one load, separate terminal phase receipts.

    No pretrained loading occurs on import. The operator/runtime owns launch and
    custody checks. No historical receipt is accepted as a substitute for smoke.
    Full prototype training remains the canonical trainer owner's responsibility.
    """
    from .gemma4_receiver import load_receiver
    if ((cfg.get("model_source") is not None and cfg.get("model") != cfg["model_source"])
            or ("model" in cfg and cfg["model"] not in
                ("google/gemma-4-31B", "google/gemma-4-E2B", "google/gemma-4-E4B"))):
        raise S0InputError("S0 receiver model identity rewritten: preserve model/model_source identity; "
                           "put staged /opt/lsr/weights only in model_local (dispatch owner fix)")
    if cfg.get("phase") not in ("smoke", "fit"):
        raise S0InputError("S0 phase must be smoke or fit")
    if out is None or not callable(getattr(out, "write", None)):
        raise S0InputError("phase receipts require Out.write")
    if not cfg.get("smoke_cases") or (cfg["phase"] == "fit" and len(cfg.get("fit_cases", [])) != 4):
        raise S0InputError("explicit smoke_cases and four fit_cases required before loading")
    handle = load_receiver(cfg)
    _handle_contract(handle)
    for phase in (("smoke",) if cfg["phase"] == "smoke" else ("smoke", "fit")):
        try:
            detail = (_smoke(handle, cfg["smoke_cases"]) if phase == "smoke"
                      else _fit_probe(handle, cfg["fit_cases"]))
        except Exception as exc:
            out.write({"key": f"s0|{phase}_fail", "phase": phase, "status": "FAIL",
                       "terminal": True, "reason": str(exc)})
            raise  # dispatch must exit nonzero; fit never follows failed smoke
        out.write({"key": f"s0|{phase}_pass", "phase": phase, "status": "PASS",
                   "terminal": True, "detail": detail,
                   "model_scope": handle["stem_info"].get("model_scope", "actual-staged")})
    return 0


def _integration_fixture(memory_width=32, *, ple=False):
    """Scaled random Gemma only; never calls the staged/pretrained loader."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from .gemma4_receiver import Gemma4Receiver, MemoryResampler, ResamplerConfig
    from . import s0_envelope as env
    torch.manual_seed(1701)
    config = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, global_head_dim=8 if ple else 16,
                             num_global_key_value_heads=2 if ple else 1,
                             attention_k_eq_v=not ple, num_kv_shared_layers=2 if ple else 0,
                             hidden_size_per_layer_input=4 if ple else 0,
                             vocab_size_per_layer_input=64, max_position_embeddings=128,
                             sliding_window=8, layer_types=["sliding_attention", "full_attention"]*2)
    config._attn_implementation = "eager"
    stem = Gemma4ForCausalLM(config).float().cpu().eval()
    connector = MemoryResampler(ResamplerConfig(memory_width, 32, 16, 4)).float().cpu()
    receiver = Gemma4Receiver(stem, connector).eval()
    handle = {"receiver": receiver, "connector": connector,
              "stem_info": {"D_memory": memory_width, "D_stem": 32, "model_scope": "random-tiny"}}
    payload = torch.randn(3, memory_width, requires_grad=True)
    origin = [{"origin_type": "generated" if i == 1 else "observation",
               "modality": "audio" if i == 1 else "text", "source_sha256": "a"*64,
               "locator": f"fixture:{i}", "event_time": t, "ingestion_time": t,
               "valid_from": t, "valid_until": None, "branch": "main",
               "dependencies": [], "permissions": ["read"]}
              for i, t in enumerate((12, 3, 4))]
    contract = {"expected_width": memory_width, "expected_schema": env.WIDTH_SCHEMA_VERSION}
    memory = env.make_envelope(payload, origin=origin, mask=[1, 1, 0], index_map=[8, 4, 9],
                               temporal_horizon={"as_of": None, "max_age": None}, **contract)
    # Factory is a persistence snapshot; replace only its byte-identical carrier
    # with the live tensor when training a writer through this port.
    memory["payload"] = payload
    inputs = {"input_ids": torch.tensor([[2, 9, 7, 13, 1, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
              "labels": torch.tensor([[2, 9, 7, 13, 1, 0]]),
              "query_mask": torch.tensor([[1, 1, 1, 0, 0, 0]])}
    host = {"branch": "main", "as_of": 10, "max_age": None, "permissions": ["read"]}
    return handle, {"inputs": inputs, "memory": memory, "conditioning": host}


def _assert_reject(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (ValueError, AssertionError):
        return True
    raise AssertionError("negative control accepted")


def _integration_contract():
    from . import s0_envelope as env
    from unittest.mock import patch
    handle, case = _integration_fixture()
    receiver = handle["receiver"]
    inputs, memory, host = _phase_case(handle, case)
    native_calls, stem_calls, outputs = [], [], []
    def capture_native(module, args, kwargs):
        native_calls.append(kwargs)
    def capture_stem(module, args, kwargs):
        stem_calls.append(kwargs)
    hooks = [receiver.register_forward_pre_hook(capture_native, with_kwargs=True),
             receiver.stem.register_forward_pre_hook(capture_stem, with_kwargs=True),
             receiver.register_forward_hook(lambda module, args, result: outputs.append(result))]
    try:
        with patch.object(receiver.connector, "forward", wraps=receiver.connector.forward) as resample:
            result = forward(handle, inputs, memory, conditioning=host)
            assert resample.call_count == 1
            assert result is outputs[-1] and result["loss"] is result["model_output"].loss
            native = native_calls[-1]
            assert torch.equal(native["memory"][0, 0], memory["payload"][1])
            assert native["memory"].shape == (1, 1, 32)
            assert native["memory_mask"].tolist() == [[True]]
            assert native["role_ids"].tolist() == [[env.ROLE_IDS["generated"]]]
            assert native["type_ids"].tolist() == [[3]]
            assert native["time_offsets"].tolist() == [[-7.0]]
            assert stem_calls[-1]["labels"].tolist() == [[-100]*8 + [-100]*3 + [13, 1, -100]]
            assert stem_calls[-1]["use_cache"] is False
            result["loss"].backward()
            grad = memory["payload"].grad
            assert grad is not None and grad[1].norm() > 0
            assert torch.count_nonzero(grad[0]) == torch.count_nonzero(grad[2]) == 0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                       for p in receiver.connector.parameters())
            assert all(p.grad is None and not p.requires_grad for p in receiver.stem.parameters())
        selected, rejected = env.select([memory], host, expected_width=32,
                                        expected_schema=env.WIDTH_SCHEMA_VERSION)
        assert selected[0]["index_map"] == [4]
        assert {r["row"] for r in rejected} == {0, 2}
        receiver.zero_grad(set_to_none=True)
        with torch.no_grad(), patch.object(receiver.connector, "forward", wraps=receiver.connector.forward) as resample:
            none = forward(handle, inputs)
            empty = forward(handle, inputs, env.empty_envelope(expected_width=32,
                                expected_schema=env.WIDTH_SCHEMA_VERSION))
            direct_labels = torch.tensor([[-100, -100, -100, 13, 1, -100]])
            baseline = receiver.stem(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                     labels=direct_labels, use_cache=False)
            for bypass in (none, empty):
                assert torch.equal(bypass["logits"], baseline.logits)
                assert torch.equal(bypass["loss"], baseline.loss)
            assert resample.call_count == 0
        # First-token answer is a legal shifted target behind the prefix.
        first = forward(handle, {"input_ids": [[2]], "labels": [[2]]}, memory, conditioning=host)
        first["loss"].backward()
        assert receiver.connector.queries.grad.norm() > 0
        _assert_reject(forward, handle, inputs, memory, horizon=1, conditioning=host)
        _assert_reject(forward, handle, dict(inputs, use_cache=True), memory, conditioning=host)
        _assert_reject(forward, handle, inputs, memory, budget={"answer": 13}, conditioning=host)
        _assert_reject(forward, handle, inputs, memory, budget={"max_new_tokens": True}, conditioning=host)
        _assert_reject(forward, handle, inputs, memory, conditioning=dict(host, permissions=["write"]))
        import copy
        bad = copy.deepcopy(memory)
        bad["index_map"][1] = 5
        _assert_reject(forward, handle, inputs, bad, conditioning=host)
        _assert_reject(forward, dict(handle, stem_info={"D_memory": 5376, "D_stem": 32}), inputs)
        # Same content repeat, donor swap and omission at shifted answer predictions.
        changed_payload = memory["payload"].detach().clone()
        changed_payload[1] = -changed_payload[1] + 2
        changed = env.make_envelope(changed_payload, origin=memory["meta"]["origin"],
                    mask=memory["mask"], index_map=memory["index_map"], expected_width=32,
                    expected_schema=env.WIDTH_SCHEMA_VERSION)
        with torch.no_grad():
            a = forward(handle, inputs, memory, conditioning=host)
            repeat = forward(handle, inputs, memory, conditioning=host)
            donor = forward(handle, inputs, changed, conditioning=host)
            noise = _batch_noise(handle, inputs, memory, host, a["logits"], changed)
            effect = _answer_effect(a["logits"], repeat["logits"], donor["logits"], inputs,
                                    noise_floor=noise)
        # Numerical counterfeit checks explicitly reject constant/digest-echo outputs.
        _assert_reject(_answer_effect, torch.zeros_like(a["logits"]), torch.zeros_like(a["logits"]),
                       torch.zeros_like(a["logits"]), inputs)
        with torch.no_grad():
            prefill_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            prefill = forward(handle, dict(prefill_inputs, use_cache=True), memory, conditioning=host)
            token = prefill["logits"][:, -1].argmax(-1, keepdim=True)
            cached = receiver.decode(token, prefill["past_key_values"])
            _assert_reject(receiver.decode, token, prefill["past_key_values"])
            full_inputs = {"input_ids": torch.cat((inputs["input_ids"], token), 1),
                           "attention_mask": torch.cat((inputs["attention_mask"], torch.ones_like(token)), 1)}
            full = forward(handle, full_inputs, memory, conditioning=host)
            cache_error = (cached["logits"][:, -1] - full["logits"][:, -1]).abs().max().item()
            assert cache_error < 1e-6
            generated = generate(handle, prefill_inputs, memory, conditioning=host,
                                 budget={"max_new_tokens": 3, "max_total_forward_calls": 2})
            assert len(generated["tokens"]) == generated["forward_calls"] == 2
            assert generated["budget_exhausted"]
        return {"labels_time_offsets_survivors": True, "returned_live_loss_gradients": True,
                "single_resampling": True, "none_typed_empty_native_exact": True,
                "first_token_loss": first["loss"].item(), "cache_max_abs_error": cache_error,
                "bounded_decode": generated, "donor_answer_effect": effect,
                "invalid_inputs_integrity_negatives": True}
    finally:
        for hook in hooks:
            hook.remove()


def _width_contract():
    from . import s0_envelope as env
    detail = {}
    for width in (32, 5376):
        handle, case = _integration_fixture(width, ple=width == 5376)
        case["memory"]["payload"].retain_grad()
        out = forward(handle, case["inputs"], case["memory"], conditioning=case["conditioning"])
        out["loss"].backward()
        assert case["memory"]["payload"].grad[1].norm() > 0
        contract = {"expected_width": width, "expected_schema": env.WIDTH_SCHEMA_VERSION}
        wire = env.serialize_envelope(case["memory"], **contract)
        reloaded = env.deserialize_envelope(wire, **contract)
        out2 = forward(handle, case["inputs"], reloaded, conditioning=case["conditioning"])
        assert torch.equal(out["logits"], out2["logits"])
        detail[str(width)] = {"logits_shape": list(out["logits"].shape), "reload_exact": True}
    # Legacy schema remains explicitly bound to 5376, native BF16 byte stride.
    origin = case["memory"]["meta"]["origin"]
    legacy = env.make_envelope(torch.randn(3, 5376).bfloat16(), origin=origin)
    selected, _ = env.select([legacy], case["conditioning"])
    assert env.native_tensor(selected[0]["payload"]).data == env.native_tensor(legacy["payload"]).data[10752:]
    _assert_reject(env.verify_envelope, legacy, expected_width=32,
                   expected_schema=env.WIDTH_SCHEMA_VERSION)
    detail["legacy_bf16_stride"] = 10752
    return detail


def _generation_contract():
    """Visible-boundary tokens and cached/full parity, with physical positions."""
    from unittest.mock import patch
    handle, case = _integration_fixture()
    host, memory = case["conditioning"], case["memory"]
    receiver = handle["receiver"]
    details = {}
    for name, ids, mask, end in (
            ("right_padded", [2, 9, 7, 13, 1, 0], [1, 1, 1, 1, 1, 0], 5),
            ("left_padded", [0, 0, 2, 9, 7, 13], [0, 0, 1, 1, 1, 1], 6),
            ("both_padded", [0, 2, 9, 7, 13, 0], [0, 1, 1, 1, 1, 0], 5)):
        inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}
        trimmed = {k: v[:, :end] for k, v in inputs.items()}
        with torch.no_grad():
            boundary = forward(handle, inputs, memory, conditioning=host)["logits"][:, end-1]
            expected = boundary.argmax(-1, keepdim=True)
            padded_last = forward(handle, inputs, memory, conditioning=host)["logits"][:, -1].argmax(-1)
            prefill = forward(handle, dict(trimmed, use_cache=True), memory, conditioning=host)
            cached = receiver.decode(expected, prefill["past_key_values"])
            full = forward(handle, {"input_ids": torch.cat((trimmed["input_ids"], expected), 1),
                            "attention_mask": torch.cat((trimmed["attention_mask"], torch.ones_like(expected)), 1)},
                           memory, conditioning=host)
            error = (cached["logits"][:, -1] - full["logits"][:, -1]).abs().max().item()
            assert error < 1e-6
            with patch.object(receiver, "forward", wraps=receiver.forward) as receive:
                result = generate(handle, inputs, memory, conditioning=host,
                                  budget={"max_new_tokens": 2, "max_total_forward_calls": 2})
                sent = receive.call_args.kwargs
                assert sent["input_ids"].shape[1] == end
                assert torch.equal(sent["attention_mask"], trimmed["attention_mask"])
            assert result["tokens"] == [expected.item(), full["logits"][:, -1].argmax(-1).item()]
            if name == "right_padded":
                assert padded_last.item() != expected.item(), "fixture must expose masked-tail bug"
            details[name] = {"visible_token": expected.item(), "masked_or_last_token": padded_last.item(),
                             "prefill_positions": end, "cache_max_abs_error": error,
                             "generated_tokens": result["tokens"]}
    _assert_reject(generate, handle, {"input_ids": [[0, 0]], "attention_mask": [[0, 0]]},
                   budget={"max_new_tokens": 1, "max_total_forward_calls": 1})
    details["all_masked_rejected"] = True
    return details


def _main_contract():
    from unittest.mock import patch
    handle, case = _integration_fixture()
    # Native carrier avoids backward through the same source graph in the fit probe.
    from . import s0_envelope as env
    case["memory"]["payload"] = env.native_tensor(case["memory"]["payload"])
    cfg = {"phase": "fit", "smoke_cases": [case], "fit_cases": [case]*4}
    class Out:
        def __init__(self):
            self.rows = []
            self.done = set()
        def meta(self, name, obj):
            assert name == "shard.json"
        def write(self, row):
            self.rows.append(row)
            self.done.add(row["key"])
    out = Out()
    with patch(__package__ + ".gemma4_receiver.load_receiver", return_value=handle) as loader:
        assert main(cfg, out) == 0
        assert loader.call_count == 1
    assert [(r["phase"], r["status"]) for r in out.rows] == [("smoke", "PASS"), ("fit", "PASS")]
    assert out.rows[-1]["detail"]["steps"] == 10
    results = {"single_resident_load": True, "separate_receipts": True,
               "fit": out.rows[-1]["detail"]}
    # Exercise the supported per-sample representation through the entire phase
    # path, including payload replacement and paired/reordered donor controls.
    list_handle, list_case = _integration_fixture()
    list_case["memory"]["payload"] = env.native_tensor(list_case["memory"]["payload"])
    list_case["memory"] = [list_case["memory"]]
    list_cfg = {"phase": "fit", "smoke_cases": [list_case], "fit_cases": [list_case]*4}
    list_out = Out()
    with patch(__package__ + ".gemma4_receiver.load_receiver", return_value=list_handle) as loader:
        assert main(list_cfg, list_out) == 0
        assert loader.call_count == 1
    assert [(r["phase"], r["status"]) for r in list_out.rows] == [("smoke", "PASS"), ("fit", "PASS")]
    assert list_out.rows[-1]["detail"]["steps"] == 10
    results["envelope_list_smoke_fit"] = {"separate_receipts": True,
                                         "smoke": list_out.rows[0]["detail"],
                                         "fit": list_out.rows[-1]["detail"]}
    # Neither source of a batch-one donor pair may be empty or multi-sample.
    envelope = list_case["memory"][0]
    replacement = _content_replacement(list_case["memory"], 32)
    with torch.no_grad():
        singleton = forward(list_handle, list_case["inputs"], list_case["memory"],
                            conditioning=list_case["conditioning"])["logits"]
        for bad in ([], [envelope, envelope]):
            _assert_reject(_batch_noise, list_handle, list_case["inputs"], bad,
                           list_case["conditioning"], singleton, replacement)
            _assert_reject(_batch_noise, list_handle, list_case["inputs"], list_case["memory"],
                           list_case["conditioning"], singleton, bad)
    results["envelope_list_donor_cardinality_negatives"] = True
    # Real dispatch, with all output writes replaced by an in-memory sink.
    # model-free dispatch avoids filesystem staging; the only loader is the
    # explicitly mocked random-model handle above. No dispatcher code is copied.
    from . import run
    dispatched = Out()
    with patch(__package__ + ".common.Out", return_value=dispatched), \
         patch(__package__ + ".gemma4_receiver.load_receiver", return_value=handle):
        assert run.execute({**cfg, "phase": "smoke", "kind": "s0_memory_port",
                            "dispatch": {}, "entry": "main"}, "NO_FILES_WRITTEN") == 0
    assert len(dispatched.rows) == 1 and dispatched.rows[0]["key"] == "s0|smoke_pass"
    results["real_dispatch_reaches_main_smoke"] = True
    real_forward = handle["receiver"].forward
    for name, form in ((name, form) for form in ("dict", "list")
                       for name in ("constant", "digest_echo", "disconnected_loss", "content_blind_prefix")):
        failed = Out()
        def counterfeit(*args, **kwargs):
            if name == "content_blind_prefix":
                kwargs = dict(kwargs)
                if kwargs.get("memory") is not None:
                    kwargs["memory"] = torch.ones_like(kwargs["memory"])
                return real_forward(*args, **kwargs)
            result = dict(real_forward(*args, **kwargs))
            if name == "disconnected_loss":
                result["loss"] = result["loss"].detach()
            else:
                result["logits"] = torch.zeros_like(result["logits"])
                if name == "constant":
                    result["received_digest"] = None
            return result
        with patch(__package__ + ".gemma4_receiver.load_receiver", return_value=handle), \
             patch.object(handle["receiver"], "forward", side_effect=counterfeit), \
             patch(__name__ + "._fit_probe", side_effect=AssertionError("fit must not start")) as fit:
            test_cfg = cfg if form == "dict" else dict(cfg, smoke_cases=[dict(case, memory=[case["memory"]])],
                                                       fit_cases=[dict(case, memory=[case["memory"]])]*4)
            _assert_reject(main, test_cfg, failed)
            assert fit.call_count == 0
        assert len(failed.rows) == 1 and failed.rows[0]["key"] == "s0|smoke_fail"
        results[("envelope_list_" if form == "list" else "") + name + "_rejected_before_fit"] = failed.rows[0]["reason"]
    with patch(__package__ + ".gemma4_receiver.load_receiver",
               side_effect=AssertionError("rewritten identity must fail before loading")) as loader:
        _assert_reject(main, dict(cfg, model="/opt/lsr/weights", model_source="google/gemma-4-31B"), Out())
        _assert_reject(main, dict(cfg, model="/opt/lsr/weights"), Out())
        assert loader.call_count == 0
    results["rewritten_identity_rejected_before_loader"] = True
    return results


def _dispatch_contract(cfg_path):
    """Read-only actual shard assembler; no cloud/staging/runtime execution."""
    from .run import assemble_shard
    import copy
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    try:
        assemble_shard(config, 0)
        supplied = {"status": "PASS", "detail": "assembled; other launch gates UNVERIFIED"}
    except ValueError as exc:
        supplied = {"status": "FAIL", "reason": str(exc)}
    fixture = copy.deepcopy(config)
    fixture.setdefault("common", {})["runtime_control_command"] = ["RAISING_MOCK_NEVER_EXECUTED"]
    fixture["common"]["phase"] = "smoke"
    shard = assemble_shard(fixture, 0)
    assert shard["entry"] == "main" and shard["phase"] == "smoke"
    bad = copy.deepcopy(fixture)
    bad["entry"] = "unknown-entry"
    _assert_reject(assemble_shard, bad, 0)
    bad = copy.deepcopy(fixture)
    bad["entry"], bad["common"]["entry"] = "main", "unknown-entry"
    _assert_reject(assemble_shard, bad, 0)
    return {"fixture_status": "PASS", "real_assemble_shard": True, "default_main_phase_preserved": True,
            "unknown_conflicting_entries_rejected": True, "supplied_config": supplied,
            "fixture_control": "not executed; mock argv only"}


def _check_dispatch_contract(cfg_path):
    try:
        detail = _dispatch_contract(cfg_path)
        status, code = (("PASS", 0) if detail["supplied_config"]["status"] == "PASS"
                        else ("SUPPLIED_CONFIG_FAIL", 1))
    except Exception as exc:
        detail = {"reason": f"{type(exc).__name__}: {exc}"}
        status, code = "FAIL", 1
    return _receipt("s0-dispatch-contract", status, cfg_path, detail, exit_code=code), code


def _offline_integration_check(cfg_path, check):
    import os
    from contextlib import ExitStack
    from unittest.mock import patch
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        if torch.get_num_interop_threads() > 2:
            raise
    detail = {}
    try:
        with ExitStack() as guards:
            guards.enter_context(patch("socket.socket.connect", side_effect=AssertionError("network forbidden")))
            guards.enter_context(patch("socket.create_connection", side_effect=AssertionError("network forbidden")))
            guards.enter_context(patch("subprocess.Popen", side_effect=AssertionError("process launch forbidden")))
            guards.enter_context(patch("os.system", side_effect=AssertionError("shell forbidden")))
            guards.enter_context(patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")))
            guards.enter_context(patch("transformers.PreTrainedModel.from_pretrained",
                                       side_effect=AssertionError("pretrained loader forbidden")))
            guards.enter_context(patch("transformers.AutoTokenizer.from_pretrained",
                                       side_effect=AssertionError("tokenizer loader forbidden")))
            if check in ("s0-production-contract", "labels-time-width", "single-resampling"):
                detail["production"] = _integration_contract()
                detail["generation"] = _generation_contract()
            if check in ("s0-production-contract", "labels-time-width"):
                detail["widths"] = _width_contract()
            if check in ("s0-production-contract", "s0-main-phases"):
                detail["main"] = _main_contract()
            if check == "s0-production-contract":
                detail["dispatch"] = _dispatch_contract(cfg_path)
                # A repaired fixture must never erase a supplied-config failure.
                broken = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
                for part in [broken.setdefault("common", {}), *broken["shards"]]:
                    part.pop("runtime_control_command", None)
                with patch.object(Path, "read_text", return_value=json.dumps(broken)):
                    failed_receipt, failed_code = _check_dispatch_contract(cfg_path)
                assert failed_code == 1 and failed_receipt["status"] == "SUPPLIED_CONFIG_FAIL"
                assert failed_receipt["detail"]["fixture_status"] == "PASS"
                detail["supplied_config_failure_distinct_from_fixture_pass"] = True
        status, code = "PASS", 0
    except Exception as exc:
        detail["reason"] = f"{type(exc).__name__}: {exc}"
        status, code = "FAIL", 1
    detail["scope"] = "CPU random-tiny mechanics; actual weights/competence/launch UNRUN"
    return _receipt(check, status, cfg_path, detail, exit_code=code), code


CHECKS = {"s0-interface": _check_interface, "s0-serialization": _check_serialization,
          "s0-native-prefix": _check_native_prefix,
          "s0-live-prep": _check_live_prep,
          "s0-trainable-paths": _check_trainable_paths,
          "s0-callgraph": _check_callgraph,
          "s0-dispatch-contract": _check_dispatch_contract,
          "s0-production-contract": lambda p: _offline_integration_check(p, "s0-production-contract"),
          "labels-time-width": lambda p: _offline_integration_check(p, "labels-time-width"),
          "single-resampling": lambda p: _offline_integration_check(p, "single-resampling"),
          "s0-main-phases": lambda p: _offline_integration_check(p, "s0-main-phases")}


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="S0 memory-port CPU checks.")
    ap.add_argument("--config",
                    default=str(Path(__file__).parents[1] / "configs" / "s0_gemma4_memory_port_v1.json"))
    ap.add_argument("--check", required=True,
                    help="one of: " + ", ".join(sorted(CHECKS)))
    args = ap.parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(json.dumps({"check": args.check, "status": "FAIL",
                          "detail": {"reason": f"config not found: {cfg_path}"},
                          "exit_code": 2}, indent=2))
        return 2
    fn = CHECKS.get(args.check)
    if fn is None:
        print(json.dumps({"check": args.check, "status": "UNKNOWN_CHECK",
                          "known_checks": sorted(CHECKS), "exit_code": 2}, indent=2))
        return 2
    receipt, code = fn(cfg_path)
    print(json.dumps(receipt, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(_cli())
