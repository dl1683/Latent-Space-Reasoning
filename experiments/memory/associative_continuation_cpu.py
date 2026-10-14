"""P-T3: learned additive arithmetic memory, CPU only.

Exact IDs route edits; they do not supply readout values. A learned nonlinear
constructor encodes (key, value), and snapshots retain its contributions plus
their sum. Replacement subtracts one contribution and inserts a new one.
A trained connector and bilinear reader answer late, multi-key linear queries.
The receiving tensor/loss path is shared by fitting and the S0 adapter.
This is a bounded synthetic construction, not a pretrained-model result.
receiver, continuation and trained-trace are aliases of one full original trace,
not independent experiments. trained-trace-v2 runs the two successor arms once.
"""
import argparse
import copy
import hashlib
import io
import json
import os
import random
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DTYPE = torch.float64
N_KEYS = 4
TOL = 1e-4
CHECKS = ("exact-controls", "receiver", "continuation", "trained-trace", "trained-trace-v2",
          "confirm-v2-linear")
ORIGINAL_CONFIG_SHA256 = "e3fb60f343d8594a3db9a1b5b64e0d76b4a7759f4f953b28d5a10ecd52e3d885"
SUCCESSORS = ("recipe_v2", "recipe_v2_linear")


def packed(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def freeze_config(path):
    raw = path.read_bytes()
    cfg = json.loads(raw)
    original = {key: value for key, value in cfg.items() if key not in (*SUCCESSORS, "confirmation")}
    if sha(packed(original)) != ORIGINAL_CONFIG_SHA256:
        raise ValueError("frozen original recipe fields changed")
    if cfg["tol"] != TOL or cfg["threads"] != 2:
        raise ValueError("tolerance must remain 1e-4; threads must be 2")
    if not 0 < cfg["fit_wall_seconds"] <= 1200:
        raise ValueError("fit wall time must be in (0, 1200]")
    if cfg["n_records"] != N_KEYS or cfg["dtype"] != "float64":
        raise ValueError("this recipe requires four records and float64")
    if cfg["optimizer"] != "LBFGS" or cfg["objective"] != "answer-only":
        raise ValueError("unsupported recipe; no implicit optimizer/objective fallback")
    if cfg["query_grid_denominator"] != 8 or cfg["latent_width"] != 8:
        raise ValueError("the S0 toy codec requires eighth-grid queries and width eight")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    return cfg, sha(raw)


def successor_config(cfg, name):
    result = {key: value for key, value in cfg.items() if key not in (*SUCCESSORS, "confirmation")}
    result.update(cfg[name])
    if result["constructor_kind"] not in ("relu", "linear-in-value"):
        raise ValueError("unknown successor constructor")
    if result["training_max_error_gate"] != TOL or result["fit_wall_seconds"] != 600:
        raise ValueError("successor training gate/budget changed")
    if (result["training_value_numerator_min"], result["training_value_numerator_max"],
        result["training_grid_denominator"]) != (-22, 20, 8):
        raise ValueError("successor must cover the full legal eighth grid")
    return result


def dyadic(value):
    value = Fraction(value)
    if value.denominator & (value.denominator - 1):
        raise ValueError("only exact dyadic values are legal")
    return value


def rational_json(value):
    return [value.numerator, value.denominator]


def resident_bytes(obj, seen=None):
    """Measured Python object sizes plus tensor storage, deduplicated by ID."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, torch.Tensor):
        return size + obj.untyped_storage().nbytes()
    if isinstance(obj, Fraction):
        return size + resident_bytes(obj.numerator, seen) + resident_bytes(obj.denominator, seen)
    if isinstance(obj, dict):
        return size + sum(resident_bytes(k, seen) + resident_bytes(v, seen)
                          for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return size + sum(resident_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return size + resident_bytes(vars(obj), seen)
    return size


class Costs:
    def __init__(self):
        self.rows = {}

    def run(self, name, fn, *args):
        start = time.perf_counter_ns()
        result = fn(*args)
        elapsed = time.perf_counter_ns() - start
        row = self.rows.setdefault(name, {"calls": 0, "total_ns": 0})
        row["calls"] += 1
        row["total_ns"] += elapsed
        return result

    def report(self):
        return {name: {**row, "mean_ns": row["total_ns"] / row["calls"]}
                for name, row in sorted(self.rows.items())}


class ExactMemory:
    """Indexed dyadic records and incrementally maintained per-key aggregates."""
    def __init__(self, records):
        self.records = {rid: (key, dyadic(value)) for rid, key, value in records}
        self.aggregate = [Fraction(0) for _ in range(N_KEYS)]
        for key, value in self.records.values():
            self.aggregate[key] += value
        self.events = {}
        self.current_event = {rid: f"initial:{rid}" for rid in self.records}
        self.predecessor_event = {rid: None for rid in self.records}

    def copy(self):
        return copy.deepcopy(self)

    def replace(self, event_id, rid, value):
        if event_id in self.events or rid not in self.records:
            return {"status": "REFUSED", "event_id": event_id,
                    "event_type": "replace", "reason": "duplicate-event-or-unknown-record"}
        key, old = self.records[rid]
        value = dyadic(value)
        previous, predecessor = self.current_event[rid], self.predecessor_event[rid]
        self.events[event_id] = (rid, key, old, value, previous, predecessor, True)
        self.current_event[rid], self.predecessor_event[rid] = f"event:{event_id}", previous
        self.records[rid] = (key, value)
        self.aggregate[key] += value - old
        return {"status": "APPLIED", "event_id": event_id, "event_type": "replace"}

    def retract(self, event_id):
        event = self.events.get(event_id)
        if event is None or not event[-1]:
            return {"status": "REFUSED", "event_id": event_id,
                    "event_type": "retract", "reason": "unknown-or-inactive-event"}
        rid, key, old, new, previous, predecessor, _ = event
        if self.current_event[rid] != f"event:{event_id}":
            return {"status": "REFUSED", "event_id": event_id,
                    "event_type": "retract", "reason": "superseded-event"}
        self.records[rid] = (key, old)
        self.aggregate[key] += old - new
        self.current_event[rid], self.predecessor_event[rid] = previous, predecessor
        self.events[event_id] = (rid, key, old, new, previous, predecessor, False)
        return {"status": "APPLIED", "event_id": event_id, "event_type": "retract"}

    def read(self, query):
        return sum((dyadic(q) * v for q, v in zip(query, self.aggregate)), Fraction(0))

    def serialize(self):
        return packed({
            "records": {rid: [key, rational_json(v)] for rid, (key, v) in self.records.items()},
            "aggregate": [rational_json(v) for v in self.aggregate],
            "current_event": self.current_event, "predecessor_event": self.predecessor_event,
            "events": {eid: [rid, key, rational_json(old), rational_json(new), previous, predecessor, active]
                       for eid, (rid, key, old, new, previous, predecessor, active) in self.events.items()}})

    @classmethod
    def reload(cls, blob):
        obj = json.loads(blob)
        result = cls([(rid, row[0], Fraction(*row[1])) for rid, row in obj["records"].items()])
        if result.aggregate != [Fraction(*v) for v in obj["aggregate"]]:
            raise ValueError("malformed exact aggregate")
        result.events = {eid: (row[0], row[1], Fraction(*row[2]), Fraction(*row[3]), *row[4:])
                         for eid, row in obj["events"].items()}
        result.current_event = obj["current_event"]
        result.predecessor_event = obj["predecessor_event"]
        return result


def oracle(records, query, edits=()):
    """Independent record replay, never called by the learned receiving path."""
    current = {rid: (key, dyadic(v)) for rid, key, v in records}
    for rid, value in edits:
        current[rid] = (current[rid][0], dyadic(value))
    return sum((dyadic(query[key]) * value for key, value in current.values()), Fraction(0))


def event_oracle(records, actions):
    """Independent stack replay specification, with no carrier/update helpers."""
    stacks = {rid: [(f"initial:{rid}", Fraction(value))] for rid, _, value in records}
    used, live, owners, receipts = set(), set(), {}, []
    for action in actions:
        kind, eid, *args = action
        reason = None
        if kind == "replace":
            rid, value = args
            if eid in used or rid not in stacks:
                reason = "duplicate-event-or-unknown-record"
            else:
                used.add(eid)
                live.add(eid)
                owners[eid] = rid
                stacks[rid].append((f"event:{eid}", Fraction(value)))
        elif kind == "retract":
            if eid not in live:
                reason = "unknown-or-inactive-event"
            elif stacks[owners[eid]][-1][0] != f"event:{eid}":
                reason = "superseded-event"
            else:
                stacks[owners[eid]].pop()
                live.remove(eid)
        else:
            raise ValueError("unknown oracle action")
        receipt = {"status": "REFUSED" if reason else "APPLIED",
                   "event_id": eid, "event_type": kind}
        if reason:
            receipt["reason"] = reason
        receipts.append(receipt)
    return {"receipts": receipts,
            "records": [(rid, key, stacks[rid][-1][1]) for rid, key, _ in records],
            "current_event": {rid: stack[-1][0] for rid, stack in stacks.items()},
            "predecessor_event": {rid: stack[-2][0] if len(stack) > 1 else None
                                  for rid, stack in stacks.items()}}


def event_metadata_bytes(memory):
    """JSON bytes for live identities plus retained undo ancestry (also in totals)."""
    return len(packed({"current_event": memory.current_event,
                       "predecessor_event": memory.predecessor_event,
                       "undo_ancestry": {eid: list(row[4:6]) for eid, row in memory.events.items()}}))


def event_semantics(cfg):
    torch.manual_seed(cfg["seed"])
    model = ArithmeticModel(cfg)
    records = [(f"semantic-{k}", k, Fraction(k, 8)) for k in range(N_KEYS)]
    rid = records[0][0]
    trajectories = {
        "equal_value_supersession": [
            ("replace", "older", rid, 1), ("replace", "newer", rid, 1),
            ("retract", "older"), ("retract", "newer"), ("retract", "older")],
        "ABA_and_predecessor_undo": [
            ("replace", "A", rid, 1), ("replace", "B", rid, 2),
            ("replace", "A-again", rid, 1), ("retract", "A"),
            ("retract", "A-again"), ("retract", "B"), ("retract", "A")],
        "refused_actions": [
            ("retract", "missing"), ("replace", "unknown", "absent", 1),
            ("replace", "once", rid, 1), ("replace", "once", rid, 2),
            ("retract", "once"), ("retract", "once"),
            ("replace", "once", rid, 1)],
    }
    # A reproducible additional mixed-record stack/reload stress control.
    rng = random.Random(cfg["train_seed"])
    trajectories["mixed_record_stress"] = [
        ("retract", f"e{rng.randrange(40)}") if rng.randrange(3) == 0 else
        ("replace", f"e{rng.randrange(40)}", records[rng.randrange(N_KEYS)][0],
         Fraction(rng.randrange(-22, 21), 8)) for _ in range(100)]
    failures, checks, refused, counterexample = [], 0, 0, None
    for name, actions in trajectories.items():
        exact, learned = ExactMemory(records), LearnedMemory(model, records)
        for i, action in enumerate(actions):
            expected = event_oracle(records, actions[:i + 1])
            for arm, memory in (("exact", exact), ("learned", learned)):
                before = memory.serialize()
                if action[0] == "replace":
                    actual = memory.replace(*action[1:]) if arm == "exact" else memory.replace(model, *action[1:])
                else:
                    actual = memory.retract(action[1])
                checks += 1
                if name == "equal_value_supersession" and i == 2:
                    counterexample = actual
                ok = actual == expected["receipts"][-1]
                if actual["status"] == "REFUSED":
                    refused += 1
                    ok &= before == memory.serialize()
                ok &= memory.current_event == expected["current_event"]
                ok &= memory.predecessor_event == expected["predecessor_event"]
                if arm == "exact":
                    reference = {rid: (key, value) for rid, key, value in expected["records"]}
                    ok &= memory.records == reference
                    ok &= memory.read([1] * N_KEYS) == sum((v for _, _, v in expected["records"]), Fraction())
                else:
                    with torch.no_grad():
                        codes = model.constructor(inputs(expected["records"]))
                    ok &= torch.allclose(memory.state, codes.sum(0), atol=1e-12, rtol=0)
                    ok &= all(torch.allclose(memory.records[rid][1], codes[j], atol=1e-12, rtol=0)
                              for j, (rid, _, _) in enumerate(expected["records"]))
                if not ok:
                    failures.append({"trajectory": name, "step": i, "arm": arm, "receipt": actual})
            # Continue exclusively from decoded event-bearing snapshots.
            exact = ExactMemory.reload(exact.serialize())
            learned = LearnedMemory.reload(learned.serialize(), cfg["latent_width"])
    return {"pass": not failures, "trajectories": list(trajectories), "action_checks": checks,
            "refused_byte_invariance_checks": refused, "failures": failures,
            "equal_value_counterexample_receipt": counterexample}


def worlds(n, seed, cfg):
    rng = random.Random(seed)
    result = []
    for w in range(n):
        keys = list(range(N_KEYS))
        rng.shuffle(keys)
        records = [(f"w{seed}-{w}-{rng.getrandbits(32):08x}", key,
                    Fraction(rng.randint(cfg["value_numerator_min"], cfg["value_numerator_max"]),
                             cfg["value_grid_denominator"])) for key in keys]
        # Late query is not stored in the constructed memory.
        query = [Fraction(rng.choice(cfg["query_numerators"]), cfg["query_grid_denominator"])
                 for _ in keys]
        result.append((records, query))
    return result


def inputs(records):
    rows = torch.zeros((len(records), 5), dtype=DTYPE)
    for i, (_, key, value) in enumerate(records):
        rows[i, key] = 1
        rows[i, 4] = float(value)
    return rows


class Constructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, cfg["hidden_width"], dtype=DTYPE),
                                 nn.ReLU(),
                                 nn.Linear(cfg["hidden_width"], cfg["latent_width"], dtype=DTYPE))

    def forward(self, rows):
        zeros = torch.cat((rows[..., :4], torch.zeros_like(rows[..., 4:])), -1)
        return self.net(rows) - self.net(zeros)


class LinearValueConstructor(nn.Module):
    """Injected phi(k,v)=v*u_k, not multiplication discovered from raw features.

    Both constructors store 456 raw trainable scalars. Constructor's eight final
    biases cancel under zero anchoring: only 448 can affect its output. All 456
    linear-arm scalars participate, but the function has only 32 effective key
    vector coordinates (four keys x eight outputs), with redundant factorization.
    Participation is not identifiable dimension. No budget-padding parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.key_lift = nn.Linear(N_KEYS, cfg["hidden_width"], dtype=DTYPE)
        self.gains = nn.Parameter(torch.ones(cfg["hidden_width"], dtype=DTYPE))
        self.basis = nn.Linear(cfg["hidden_width"], cfg["latent_width"], dtype=DTYPE)

    def forward(self, rows):
        key_vectors = self.basis(self.key_lift(rows[..., :N_KEYS]) * self.gains)
        return rows[..., N_KEYS:] * key_vectors


class TensorReader(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["latent_width"]
        self.frozen = nn.Linear(width, width, bias=False, dtype=DTYPE)
        with torch.no_grad():
            generator = None
            if "receiver_seed" in cfg:
                generator = torch.Generator(device="cpu").manual_seed(cfg["receiver_seed"])
            self.frozen.weight.copy_(torch.linalg.qr(
                torch.randn(width, width, dtype=DTYPE, generator=generator))[0])
        self.frozen.requires_grad_(False)
        self.readout = nn.Linear(width, N_KEYS, bias=False, dtype=DTYPE)

    def forward(self, prefix, query, target=None):
        values = self.readout(self.frozen(prefix))
        answer = (values * query).sum(-1)
        loss = None if target is None else (answer - target).square().mean()
        return {"answer": answer, "loss": loss}


class ArithmeticModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.constructor = (LinearValueConstructor(cfg) if cfg.get("constructor_kind") == "linear-in-value"
                            else Constructor(cfg))
        if "connector_seed" in cfg:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg["connector_seed"])
                self.connector = nn.Linear(cfg["latent_width"], cfg["latent_width"], bias=False, dtype=DTYPE)
        else:
            self.connector = nn.Linear(cfg["latent_width"], cfg["latent_width"], bias=False, dtype=DTYPE)
        self.reader = TensorReader(cfg)

    def forward(self, rows, query, target=None):
        state = self.constructor(rows).sum(-2)
        return self.reader(self.connector(state), query, target)


class LearnedMemory:
    """No original numeric facts or oracle at read time, only learned tensors."""
    def __init__(self, model, records):
        with torch.no_grad():
            codes = model.constructor(inputs(records))
        self.records = {rid: (key, codes[i].clone()) for i, (rid, key, _) in enumerate(records)}
        self.state = codes.sum(0)
        self.events = {}
        self.current_event = {rid: f"initial:{rid}" for rid in self.records}
        self.predecessor_event = {rid: None for rid in self.records}

    def copy(self):
        return copy.deepcopy(self)

    def replace(self, model, event_id, rid, value):
        if event_id in self.events or rid not in self.records:
            return {"status": "REFUSED", "event_id": event_id,
                    "event_type": "replace", "reason": "duplicate-event-or-unknown-record"}
        key, old = self.records[rid]
        with torch.no_grad():
            new = model.constructor(inputs([(rid, key, dyadic(value))]))[0]
        previous, predecessor = self.current_event[rid], self.predecessor_event[rid]
        self.events[event_id] = (rid, key, old.clone(), new.clone(), previous, predecessor, True)
        self.current_event[rid], self.predecessor_event[rid] = f"event:{event_id}", previous
        self.state = self.state + new - old
        self.records[rid] = (key, new.clone())
        return {"status": "APPLIED", "event_id": event_id, "event_type": "replace"}

    def retract(self, event_id):
        event = self.events.get(event_id)
        if event is None or not event[-1]:
            return {"status": "REFUSED", "event_id": event_id,
                    "event_type": "retract", "reason": "unknown-or-inactive-event"}
        rid, key, old, new, previous, predecessor, _ = event
        if self.current_event[rid] != f"event:{event_id}":
            return {"status": "REFUSED", "event_id": event_id,
                    "event_type": "retract", "reason": "superseded-event"}
        self.state = self.state + old - new
        self.records[rid] = (key, old.clone())
        self.current_event[rid], self.predecessor_event[rid] = previous, predecessor
        self.events[event_id] = (rid, key, old.clone(), new.clone(), previous, predecessor, False)
        return {"status": "APPLIED", "event_id": event_id, "event_type": "retract"}

    def omit(self, rid):
        result = self.copy()
        _, code = result.records.pop(rid)
        result.current_event.pop(rid)
        result.predecessor_event.pop(rid)
        result.state = result.state - code
        return result

    def serialize(self):
        content = {"state": self.state.tolist(),
                   "records": {rid: [key, code.tolist()] for rid, (key, code) in self.records.items()},
                   "current_event": self.current_event, "predecessor_event": self.predecessor_event,
                   "events": {eid: [rid, key, old.tolist(), new.tolist(), previous, predecessor, active]
                              for eid, (rid, key, old, new, previous, predecessor, active)
                              in self.events.items()}}
        return packed({"content": content, "sha256": sha(packed(content))})

    @classmethod
    def reload(cls, blob, width):
        obj = json.loads(blob)
        content = obj["content"]
        if sha(packed(content)) != obj["sha256"]:
            raise ValueError("malformed learned envelope: checksum mismatch")
        def tensor(values):
            result = torch.tensor(values, dtype=DTYPE)
            if result.shape != (width,) or not torch.isfinite(result).all():
                raise ValueError("malformed learned tensor")
            return result
        result = cls.__new__(cls)
        result.state = tensor(content["state"])
        result.records = {rid: (key, tensor(code)) for rid, (key, code) in content["records"].items()}
        result.events = {eid: (row[0], row[1], tensor(row[2]), tensor(row[3]), *row[4:])
                         for eid, row in content["events"].items()}
        result.current_event = content["current_event"]
        result.predecessor_event = content["predecessor_event"]
        if not torch.allclose(result.state, sum((v[1] for v in result.records.values()),
                                               torch.zeros(width, dtype=DTYPE)),
                              atol=1e-12, rtol=0):
            raise ValueError("malformed learned aggregate")
        return result


def boundary_read(model, memory, query, target=None):
    """Toy adapter into the actual S0 forward; this is not native BF16 data.

    Only the latent aggregate crosses the boundary. Neither source records nor
    oracle values enter the receiver. Byte decoding ends before the trainable
    connector, so S0's returned loss keeps that connector's autograd graph.
    """
    from experiments.cloud.lsr_jobs import s0_memory_port as port
    width = memory.state.numel()
    raw = struct.pack(f">{width}d", *memory.state.tolist())
    row = raw + bytes(port.WIDTH - len(raw))
    origin = [{"origin_type": "derived", "modality": "text", "source_sha256": sha(raw),
               "locator": "learned-aggregate", "event_time": "t", "ingestion_time": "t",
               "valid_from": "t", "valid_until": None, "branch": "main", "dependencies": []}]
    envelope = port.make_envelope([row], origin=origin, index_map=[0], mask=[1])
    coefficients = [int(dyadic(q) * 8) for q in query]
    if any(Fraction(c, 8) != q for c, q in zip(coefficients, query)):
        raise ValueError("query must lie on the registered eighth grid")
    tokens = [c + 32 for c in coefficients]

    def connector(rows):
        decoded = torch.tensor([struct.unpack(f">{width}d", bytes(r[:8 * width]))
                                for r in rows], dtype=DTYPE)
        return {"prefix": model.connector(decoded), "mask": [1], "index_map": [0]}

    def receiver(*, prefix_states, input_ids, attach_mask, assembly_digest, **kwargs):
        q = torch.tensor([(v - 32) / 8 for v in input_ids[0]], dtype=DTYPE)
        prefix = prefix_states[0] * attach_mask[0]
        result = model.reader(prefix, q, target)
        result["received_digest"] = assembly_digest
        return result

    handle = {"tok": lambda *a, **k: {"input_ids": [tokens]},
              "connector": connector, "receiver": receiver}
    return port.forward(handle, {"input_ids": [tokens]}, envelope, horizon=0)


def boundary_equivalence(model, cfg):
    """Assert direct and production-returned answers AND loss on identical inputs."""
    answer_errors, loss_errors = [], []
    for records, q in worlds(8, cfg["train_seed"], cfg):
        target = torch.tensor(float(oracle(records, q)) + 1, dtype=DTYPE)
        direct = model(inputs(records), torch.tensor([float(v) for v in q], dtype=DTYPE), target)
        deployed = boundary_read(model, LearnedMemory(model, records), q, target)
        for key in ("answer", "loss"):
            torch.testing.assert_close(direct[key], deployed[key], atol=1e-12, rtol=0)
        answer_errors.append(float((direct["answer"] - deployed["answer"]).abs().detach()))
        loss_errors.append(float((direct["loss"] - deployed["loss"]).abs().detach()))
    return {"pass": True, "cases": len(answer_errors), "assertion_atol": 1e-12,
            "answer_max_abs_error": max(answer_errors), "loss_max_abs_error": max(loss_errors)}


def exact_smoke(cfg):
    costs, gates, failures = Costs(), {}, []
    storage = {"max_serialized_bytes": 0, "max_resident_bytes": 0,
               "max_aggregate_payload_bytes": 0}
    refusals = []
    semantics = event_semantics(cfg)
    gates["independent_event_semantics"] = semantics["pass"]
    cases = worlds(cfg["heldout_worlds"], cfg["heldout_seed"], cfg)
    for w, (records, query) in enumerate(cases):
        base = costs.run("construct", ExactMemory, records)
        blob = costs.run("serialize", base.serialize)
        base = costs.run("reload", ExactMemory.reload, blob)
        a, b = costs.run("branch_copy", base.copy), costs.run("branch_copy", base.copy)
        rid_a, rid_b = records[0][0], records[1][0]
        edit_a = records[0][2] + Fraction(*cfg["edit_deltas"][0])
        edit_b = records[1][2] + Fraction(*cfg["edit_deltas"][1])
        costs.run("replace", a.replace, f"A-{w}", rid_a, edit_a)
        costs.run("replace", b.replace, f"B-{w}", rid_b, edit_b)
        a = costs.run("reload_edited", ExactMemory.reload, costs.run("serialize_edited", a.serialize))
        b = costs.run("reload_edited", ExactMemory.reload, costs.run("serialize_edited", b.serialize))
        old_blobs = [base.serialize(), a.serialize(), b.serialize()]
        for order in ((a, b), (b, a)):
            for view in order:
                edits = [(rid_a, edit_a)] if view is a else [(rid_b, edit_b)]
                prediction = costs.run("read", view.read, query)
                if prediction != oracle(records, query, edits):
                    failures.append({"world": w, "role": "edited-fork", "prediction": float(prediction)})
        for role, view, edits in (("late", base, ()), ("affected", a, ((rid_a, edit_a),)),
                                  ("unrelated", a, ((rid_a, edit_a),)), ("history", base, ())):
            q = query if role == "late" else [Fraction(int(k == (records[1][1] if role == "unrelated"
                                                                                else records[0][1])))
                                              for k in range(N_KEYS)]
            if costs.run("read", view.read, q) != oracle(records, q, edits):
                failures.append({"world": w, "role": role})
        if [base.serialize(), a.serialize(), b.serialize()] != old_blobs:
            failures.append({"world": w, "role": "read-mutation"})
        costs.run("retract", a.retract, f"A-{w}")
        for view, edits in ((a, ()), (b, ((rid_b, edit_b),)), (base, ())):
            if costs.run("read", view.read, query) != oracle(records, query, edits):
                failures.append({"world": w, "role": "post-retraction"})
        before = a.serialize()
        refusal = a.retract(f"A-{w}")
        refusals.append(refusal)
        if refusal != {"status": "REFUSED", "event_id": f"A-{w}",
                       "event_type": "retract", "reason": "unknown-or-inactive-event"} or a.serialize() != before:
            failures.append({"world": w, "role": "named-refusal"})
        if base.serialize() != old_blobs[0] or b.serialize() != old_blobs[2]:
            failures.append({"world": w, "role": "cross-fork-leak"})
        donor_records = cases[(w + 1) % len(cases)][0]
        donor = ExactMemory(donor_records)
        reduced_records = [r for r in records if r[0] != rid_a]
        omitted = ExactMemory(reduced_records)
        wrong_query = query[1:] + query[:1]
        for role, view, q, expected in (
                ("donor", donor, query, oracle(donor_records, query)),
                ("omission", omitted, query, oracle(reduced_records, query)),
                ("wrong_address", base, wrong_query, oracle(records, wrong_query)),
                ("no_op", base, query, oracle(records, query))):
            if costs.run("read", view.read, q) != expected:
                failures.append({"world": w, "role": role})
        for view in (base, a, b):
            storage["max_serialized_bytes"] = max(storage["max_serialized_bytes"], len(view.serialize()))
            storage["max_resident_bytes"] = max(storage["max_resident_bytes"], resident_bytes(view))
            storage["max_event_metadata_bytes"] = max(storage.get("max_event_metadata_bytes", 0),
                                                       event_metadata_bytes(view))
            aggregate_bytes = len(packed([rational_json(v) for v in view.aggregate]))
            storage["max_aggregate_payload_bytes"] = max(storage["max_aggregate_payload_bytes"], aggregate_bytes)
    bad = json.loads(blob)
    bad["aggregate"][0][0] += 1
    try:
        ExactMemory.reload(packed(bad))
        gates["malformed_envelope_rejected"] = False
    except ValueError:
        gates["malformed_envelope_rejected"] = True
    gates["independent_oracle_all_traces"] = not failures
    gates["payload_ceiling"] = storage["max_aggregate_payload_bytes"] <= cfg["payload_ceiling_bytes"]
    return {"pass": all(gates.values()), "gates": gates,
            "worlds": cfg["heldout_worlds"], "failures": failures,
            "questions": costs.rows["read"]["calls"], "max_abs_error": 0.0 if not failures else None,
            "named_refusal_sample": refusals[:2], "event_semantics": semantics, "storage": storage,
            "costs": costs.report(), "fits_attempted": 0}


def fit(cfg):
    """One fixed full-batch LBFGS recipe, no test-driven retry or retuning."""
    start = time.monotonic()
    deadline = start + cfg["fit_wall_seconds"]
    torch.manual_seed(cfg["seed"])
    model = ArithmeticModel(cfg)
    equivalence_before = boundary_equivalence(model, cfg)
    frozen_before = model.reader.frozen.weight.detach().clone()
    episodes = worlds(cfg["train_worlds"], cfg["train_seed"], cfg)
    rows = torch.stack([inputs(records) for records, _ in episodes])
    queries, targets = [], []
    for records, late in episodes:
        qs = [[Fraction(int(k == i)) for k in range(N_KEYS)] for i in range(N_KEYS)] + [late]
        queries.append([[float(v) for v in q] for q in qs])
        targets.append([float(oracle(records, q)) for q in qs])
    query = torch.tensor(queries, dtype=DTYPE)
    target = torch.tensor(targets, dtype=DTYPE)
    # Five questions share each constructed state, with no query in construction.
    repeated = rows[:, None].expand(-1, 5, -1, -1)
    optimizer = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"],
        max_iter=1, max_eval=cfg["max_line_search_evals"],
        history_size=cfg["lbfgs_history"], tolerance_grad=cfg["optimizer_tolerance_grad"],
        tolerance_change=cfg["optimizer_tolerance_change"], line_search_fn="strong_wolfe")
    counts, first_grads, losses, initial_losses = {"closures": 0, "steps": 0}, {}, [], []
    def closure():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed fit deadline reached")
        optimizer.zero_grad(set_to_none=True)
        result = model(repeated, query, target)
        loss = result["loss"]
        if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
            raise ValueError("returned loss must be a finite tensor")
        loss.backward()
        grads = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
                 for name, p in model.named_parameters()}
        if not first_grads:
            first_grads.update(grads)
            initial_losses.append(float(loss.item()))
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("nonfinite fit gradient")
        if model.reader.frozen.weight.grad is not None:
            raise ValueError("frozen receiver has a gradient")
        counts["closures"] += 1
        return loss
    timeout = False
    try:
        for _ in range(cfg["optimizer_steps"]):
            optimizer.step(closure)
            counts["steps"] += 1
            with torch.no_grad():
                losses.append(float(model(repeated, query, target)["loss"]))
    except TimeoutError:
        timeout = True
    elapsed = time.monotonic() - start
    model.eval()
    return model, {"fit_seconds": elapsed, **counts, "deadline_hit": timeout,
                   "initial_returned_loss": initial_losses[0] if initial_losses else None,
                   "final_returned_loss": losses[-1] if losses else None,
                   "first_gradient_l1": first_grads,
                   "frozen_parameters_unchanged": torch.equal(frozen_before, model.reader.frozen.weight),
                   "boundary_equivalence_before_fit": equivalence_before,
                   "train_worlds": len(episodes), "train_questions": target.numel(),
                   "optimizer_resident_bytes": resident_bytes(optimizer.state),
                   "training_tensor_bytes": sum(x.numel() * x.element_size() for x in (rows, query, target))}


def training_episodes_v2(cfg):
    """Training-only worlds, exhaustive key/value support and real undo targets."""
    train_cfg = {**cfg, "value_numerator_min": cfg["training_value_numerator_min"],
                 "value_numerator_max": cfg["training_value_numerator_max"]}
    episodes = worlds(cfg["train_worlds"], cfg["train_seed"], train_cfg)
    grid = list(range(cfg["training_value_numerator_min"], cfg["training_value_numerator_max"] + 1))
    rows, new_rows, queries, targets, signatures, replacement_signatures = [], [], [], [], [], []
    coverage = {k: set() for k in range(N_KEYS)}
    for w, (records, late) in enumerate(episodes):
        coverage_key = w % N_KEYS
        value = Fraction(grid[(w // N_KEYS) % len(grid)], 8)
        records = [(rid, key, value if key == coverage_key else old) for rid, key, old in records]
        episodes[w] = records, late
        for _, key, old in records:
            coverage[key].add(old.numerator * (8 // old.denominator))
        rid, key, _ = records[0]
        replacement = Fraction(grid[(w * 7 + 11) % len(grid)], 8)
        action = ("replace", f"train-{w}", rid, replacement)
        views = [event_oracle(records, actions)["records"]
                 for actions in ([], [action], [action, ("retract", f"train-{w}")])]
        qs = [[Fraction(int(k == i)) for k in range(N_KEYS)] for i in range(N_KEYS)] + [late]
        rows.append(inputs(records))
        new_rows.append(inputs([(rid, key, replacement)]))
        queries.append([[[float(v) for v in q] for q in qs]] * 3)
        targets.append([[float(oracle(view, q)) for q in qs] for view in views])
        signatures.append(tuple(sorted((key, float(value)) for _, key, value in records)))
        replacement_signatures.append(tuple(sorted((key, float(value)) for _, key, value in views[1])))
    if any(coverage[k] != set(grid) for k in range(N_KEYS)):
        raise ValueError("training worlds do not cover every key/grid value")
    grid_records = [[(f"grid-{k}-{n}", k, Fraction(n, 8))]
                    for k in range(N_KEYS) for n in grid]
    data = {"rows": torch.stack(rows), "new_rows": torch.stack(new_rows),
            "query": torch.tensor(queries, dtype=DTYPE), "target": torch.tensor(targets, dtype=DTYPE),
            "grid_rows": torch.stack([inputs(records) for records in grid_records]),
            "grid_query": torch.tensor([[float(k == records[0][1]) for k in range(N_KEYS)]
                                       for records in grid_records], dtype=DTYPE),
            "grid_target": torch.tensor([float(records[0][2]) for records in grid_records], dtype=DTYPE)}
    manifest = {"episode_tensors": {name: value.tolist() for name, value in data.items()},
                "world_signatures": signatures}
    # Independence is by complete world, not disjoint scalar support. Check
    # held-out base and edited worlds without consulting their quality.
    heldout_signatures = set()
    for records, _ in worlds(cfg["heldout_worlds"], cfg["heldout_seed"], cfg):
        heldout_signatures.add(tuple(sorted((key, float(value)) for _, key, value in records)))
        for index in (0, 1):
            rid = records[index][0]
            value = records[index][2] + Fraction(*cfg["edit_deltas"][index])
            heldout_signatures.add(tuple(sorted((key, float(value if record_id == rid else old))
                                                for record_id, key, old in records)))
    overlap = len(set(signatures + replacement_signatures) & heldout_signatures)
    if overlap:
        raise ValueError("training/held-out world overlap")
    return data, {"data_sha256": sha(packed(manifest)), "train_worlds": len(episodes),
                  "training_views": ["base", "replace", "retract"],
                  "each_key_grid_values": {str(k): len(coverage[k]) for k in coverage},
                  "isolated_key_value_targets": len(grid_records), "world_overlap": overlap,
                  "training_legal_domain": [-2.75, 2.5], "denominator": 8}


def episode_answers(model, data):
    codes = model.constructor(data["rows"])
    state = codes.sum(-2)
    old = codes[:, 0]
    new = model.constructor(data["new_rows"])[:, 0]
    edited = state - old + new
    retracted = edited - new + old
    views = torch.stack((state, edited, retracted), dim=1)
    return model.reader(model.connector(views)[:, :, None, :], data["query"])["answer"]


def fit_v2(cfg, data, manifest):
    """Single fixed schedule; no held-out quality enters optimization or stopping."""
    start = time.monotonic()
    deadline = start + cfg["fit_wall_seconds"]
    torch.manual_seed(cfg["seed"])
    model = ArithmeticModel(cfg)
    equivalence = boundary_equivalence(model, cfg)
    frozen_before = model.reader.frozen.weight.detach().clone()
    optimizer = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"],
        max_iter=1, max_eval=cfg["max_line_search_evals"], history_size=cfg["lbfgs_history"],
        tolerance_grad=cfg["optimizer_tolerance_grad"],
        tolerance_change=cfg["optimizer_tolerance_change"], line_search_fn="strong_wolfe")
    counts, first_grads, initial_loss = {"closures": 0, "steps": 0}, {}, []
    def predictions():
        return episode_answers(model, data), model(data["grid_rows"], data["grid_query"])["answer"]
    def loss_for(answers, grid_answers):
        residuals = torch.cat(((answers - data["target"]).flatten(),
                               grid_answers - data["grid_target"]))
        return residuals.square().mean()
    def closure():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed successor fit deadline reached")
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(*predictions())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite successor loss")
        loss.backward()
        if not first_grads:
            first_grads.update({name: 0.0 if p.grad is None else float(p.grad.abs().sum())
                                for name, p in model.named_parameters()})
            initial_loss.append(float(loss.detach()))
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("nonfinite successor gradient")
        if model.reader.frozen.weight.grad is not None:
            raise ValueError("frozen receiver gradient")
        counts["closures"] += 1
        return loss
    timeout = False
    try:
        for _ in range(cfg["optimizer_steps"]):
            optimizer.step(closure)
            counts["steps"] += 1
            if counts["steps"] % 1000 == 0:
                print(json.dumps({"phase": "successor-fit-progress", "recipe": cfg["recipe"],
                                  **counts, "seconds": time.monotonic() - start}), flush=True)
    except TimeoutError:
        timeout = True
    model.eval()
    with torch.no_grad():
        answers, grid_answers = predictions()
        episode_error = (answers - data["target"]).abs()
        grid_error = (grid_answers - data["grid_target"]).abs().reshape(N_KEYS, -1)
        max_error = max(float(episode_error.max()), float(grid_error.max()))
        final_loss = float(loss_for(answers, grid_answers))
    grid = range(cfg["training_value_numerator_min"], cfg["training_value_numerator_max"] + 1)
    elapsed = time.monotonic() - start
    return model, {"fit_seconds": elapsed, **counts, "deadline_hit": timeout,
                   "initial_returned_loss": initial_loss[0] if initial_loss else None,
                   "final_returned_loss": final_loss, "first_gradient_l1": first_grads,
                   "frozen_parameters_unchanged": torch.equal(frozen_before, model.reader.frozen.weight),
                   "boundary_equivalence_before_fit": equivalence,
                   "train_questions": data["target"].numel() + data["grid_target"].numel(),
                   "training_max_abs_error": max_error,
                   "training_max_error_gate": max_error <= cfg["training_max_error_gate"],
                   "episode_max_abs_error": float(episode_error.max()),
                   "per_key_value_abs_error": {
                       str(k): {str(Fraction(n, 8)): float(grid_error[k, j]) for j, n in enumerate(grid)}
                       for k in range(N_KEYS)},
                   "episode_roles": {role: {"max_abs_error": float(episode_error[:, i].max()),
                                           "failure_count": int((episode_error[:, i] > cfg["tol"]).sum())}
                                     for i, role in enumerate(("base", "replace", "retract"))},
                   "optimizer_resident_bytes": resident_bytes(optimizer.state),
                   "training_tensor_bytes": sum(x.numel() * x.element_size() for x in data.values()),
                   **manifest}


def gradient_gate(model, cfg):
    records, query = worlds(1, cfg["heldout_seed"], cfg)[0]
    memory = LearnedMemory(model, records)
    model.zero_grad(set_to_none=True)
    target = torch.tensor(float(oracle(records, query)) + 1, dtype=DTYPE)
    result = boundary_read(model, memory, query, target)
    loss = result["loss"]
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return {"pass": False, "loss_is_tensor_with_graph": False}
    loss.backward()
    connector = model.connector.weight.grad
    readout = model.reader.readout.weight.grad
    frozen = model.reader.frozen.weight.grad
    gates = {"loss_is_tensor_with_graph": True,
             "connector_gradient_finite_nonzero": connector is not None and
             bool(torch.isfinite(connector).all()) and float(connector.abs().sum()) > 0,
             "readout_gradient_finite_nonzero": readout is not None and
             bool(torch.isfinite(readout).all()) and float(readout.abs().sum()) > 0,
             "frozen_gradient_zero": frozen is None or bool((frozen == 0).all())}
    return {"pass": all(gates.values()), "gates": gates,
            "returned_loss": float(loss.item()),
            "connector_gradient_l1": 0.0 if connector is None else float(connector.abs().sum()),
            "readout_gradient_l1": 0.0 if readout is None else float(readout.abs().sum()),
            "frozen_gradient_l1": 0.0 if frozen is None else float(frozen.abs().sum())}


def learned_trace(model, cfg):
    costs, errors, examples, failures, gates = Costs(), {}, [], [], {}
    storage = {"max_serialized_bytes": 0, "max_resident_bytes": 0,
               "max_tensor_payload_bytes": 0}
    original_reads, restored_reads, memories, base_reads = [], [], [], []
    effects = {"donor": [], "omission": [], "wrong_address": []}
    strata = {domain: {"in_range": [], "out_of_range": []}
              for domain in ("initial_domain", "legal_edit_domain")}
    cases = worlds(cfg["heldout_worlds"], cfg["heldout_seed"], cfg)
    parameter_blob = io.BytesIO()
    torch.save(model.state_dict(), parameter_blob)
    checkpoint = parameter_blob.getvalue()
    # An independently initialized instance, no reused parameter tensors.
    torch.manual_seed(cfg["reload_seed"])
    restored = ArithmeticModel(cfg)
    restored.load_state_dict(torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True))
    restored.eval()
    reconstruction_error, recovery_errors, recovery_gates = 0.0, [], []
    def score(w, role, memory, q, expected, edits=(), source_records=None):
        with torch.no_grad():
            result = costs.run("boundary_read", boundary_read, model, memory, q)
        answer = float(result["answer"].item())
        error = abs(answer - float(expected))
        source_records = cases[w][0] if source_records is None else source_records
        replacements = dict(edits)
        relevant_values = [float(replacements.get(rid, value)) for rid, key, value in source_records if q[key]]
        for domain, bounds in (("initial_domain", (-2, 2)), ("legal_edit_domain", (-2.75, 2.5))):
            label = "in_range" if all(bounds[0] <= value <= bounds[1] for value in relevant_values) else "out_of_range"
            strata[domain][label].append(error)
        errors.setdefault(role, []).append(error)
        row = {"world": w, "role": role, "prediction": answer,
               "oracle": float(expected), "abs_error": error}
        if w == 0:
            examples.append(row)
        if not error <= cfg["tol"]:
            failures.append(row)
        original_reads.append(answer)
        with torch.no_grad():
            restored_reads.append(float(boundary_read(restored, memory, q)["answer"].item()))
        return answer
    for w, (records, query) in enumerate(cases):
        base = costs.run("construct", LearnedMemory, model, records)
        reconstructed = costs.run("reconstruct_reloaded_model", LearnedMemory, restored, records)
        reconstruction_error = max(reconstruction_error, float((base.state - reconstructed.state).abs().max()))
        assert base.serialize() == reconstructed.serialize()
        base = reconstructed
        blob = costs.run("serialize", base.serialize)
        base = costs.run("reload", LearnedMemory.reload, blob, cfg["latent_width"])
        base_answer = score(w, "late_multi_key", base, query, oracle(records, query))
        base_reads.append(base_answer)
        a, b = costs.run("branch_copy", base.copy), costs.run("branch_copy", base.copy)
        rid_a, rid_b = records[0][0], records[1][0]
        edit_a = records[0][2] + Fraction(*cfg["edit_deltas"][0])
        edit_b = records[1][2] + Fraction(*cfg["edit_deltas"][1])
        edits_a, edits_b = [(rid_a, edit_a)], [(rid_b, edit_b)]
        costs.run("replace", a.replace, restored, f"A-{w}", rid_a, edit_a)
        costs.run("replace", b.replace, restored, f"B-{w}", rid_b, edit_b)
        # Only serialized learned contributions/ancestry cross this recovery
        # boundary. No original value is needed by reload, read or named undo.
        edited_blobs = [costs.run("serialize_edited", view.serialize) for view in (a, b)]
        a, b = [costs.run("reload_edited", LearnedMemory.reload, blob, cfg["latent_width"])
                for blob in edited_blobs]
        assert [a.serialize(), b.serialize()] == edited_blobs
        old_blobs = [base.serialize(), a.serialize(), b.serialize()]
        fork_answers = {}
        for order_name, order in (("AB", (("A", a, edits_a), ("B", b, edits_b))),
                                  ("BA", (("B", b, edits_b), ("A", a, edits_a)))):
            for branch, view, edits in order:
                fork_answers[(branch, order_name)] = score(
                    w, f"fork_{branch}_{order_name}", view, query, oracle(records, query, edits), edits)
        gates.setdefault("both_query_orders_match", True)
        gates["both_query_orders_match"] &= all(
            abs(fork_answers[(branch, "AB")] - fork_answers[(branch, "BA")]) <= cfg["tol"]
            for branch in ("A", "B"))
        gates.setdefault("forks_have_distinct_edits_and_targets", True)
        gates["forks_have_distinct_edits_and_targets"] &= rid_a != rid_b and edit_a != records[0][2] and \
            edit_b != records[1][2] and a.serialize() != b.serialize()
        for role, view, key, edits in (
                ("revised_affected", a, records[0][1], edits_a),
                ("revised_unrelated", a, records[1][1], edits_a),
                ("history", base, records[0][1], ())):
            q = [Fraction(int(k == key)) for k in range(N_KEYS)]
            score(w, role, view, q, oracle(records, q, edits), edits)
        gates.setdefault("read_only_and_fork_isolation", True)
        gates["read_only_and_fork_isolation"] &= [base.serialize(), a.serialize(), b.serialize()] == old_blobs
        costs.run("retract", a.retract, f"A-{w}")
        score(w, "retracted_A", a, query, oracle(records, query))
        score(w, "post_retract_B", b, query, oracle(records, query, edits_b), edits_b)
        score(w, "post_retract_base", base, query, oracle(records, query))
        before = a.serialize()
        refusal = a.retract(f"A-{w}")
        gates.setdefault("named_event_refusal", True)
        gates["named_event_refusal"] &= refusal == {
            "status": "REFUSED", "event_id": f"A-{w}", "event_type": "retract",
            "reason": "unknown-or-inactive-event"} and a.serialize() == before
        gates["read_only_and_fork_isolation"] &= base.serialize() == old_blobs[0] and b.serialize() == old_blobs[2]
        # Valid donor in both directions; oracle is recomputed at the recipient query.
        donor_records = cases[(w + 1) % len(cases)][0]
        donor = LearnedMemory(model, donor_records)
        score(w, "donor_follows_donor_oracle", donor, query, oracle(donor_records, query),
              source_records=donor_records)
        effects["donor"].append(abs(float(oracle(donor_records, query) - oracle(records, query))))
        omitted = base.omit(rid_a)
        reduced_records = [r for r in records if r[0] != rid_a]
        score(w, "omission", omitted, query, oracle(reduced_records, query), source_records=reduced_records)
        effects["omission"].append(abs(float(oracle(reduced_records, query) - oracle(records, query))))
        wrong_query = query[1:] + query[:1]
        score(w, "wrong_address", base, wrong_query, oracle(records, wrong_query))
        effects["wrong_address"].append(abs(float(oracle(records, wrong_query) - oracle(records, query))))
        noop = score(w, "no_op", base, query, oracle(records, query))
        gates.setdefault("no_op_exact_repeat", True)
        gates["no_op_exact_repeat"] &= noop == base_answer
        # A was undone after reload above; independently undo B too, then
        # continue with new external actions through the reloaded constructor.
        continued = [a.copy(), b.copy()]
        assert continued[1].retract(f"B-{w}")["status"] == "APPLIED"
        for branch, view in enumerate(continued):
            rid, key, value = records[branch]
            replacement = value + Fraction(*cfg["edit_deltas"][branch])
            assert view.replace(restored, f"continued-{branch}-{w}", rid, replacement)["status"] == "APPLIED"
        before_continue_reads = [view.serialize() for view in continued]
        answers = {}
        for order in ((0, 1), (1, 0)):
            for branch in order:
                rid, _, value = records[branch]
                replacement = value + Fraction(*cfg["edit_deltas"][branch])
                expected = oracle(records, query, [(rid, replacement)])
                with torch.no_grad():
                    answer = float(costs.run("reloaded_continuation_read", boundary_read,
                                             restored, continued[branch], query)["answer"])
                recovery_errors.append(abs(answer - float(expected)))
                if branch in answers:
                    recovery_gates.append(answer == answers[branch])
                answers[branch] = answer
        recovery_gates.append([view.serialize() for view in continued] == before_continue_reads)
        for branch, view in enumerate(continued):
            assert view.retract(f"continued-{branch}-{w}")["status"] == "APPLIED"
            with torch.no_grad():
                answer = float(boundary_read(restored, view, query)["answer"])
            recovery_errors.append(abs(answer - float(oracle(records, query))))
        recovery_gates.append(base.serialize() == old_blobs[0] and b.serialize() == old_blobs[2])
        for view in (base, a, b, *continued):
            storage["max_serialized_bytes"] = max(storage["max_serialized_bytes"], len(view.serialize()))
            storage["max_resident_bytes"] = max(storage["max_resident_bytes"], resident_bytes(view))
            storage["max_event_metadata_bytes"] = max(storage.get("max_event_metadata_bytes", 0),
                                                       event_metadata_bytes(view))
            tensors = [view.state] + [r[1] for r in view.records.values()]
            tensors += [t for event in view.events.values() for t in event[2:4]]
            storage["max_tensor_payload_bytes"] = max(
                storage["max_tensor_payload_bytes"], sum(t.numel() * t.element_size() for t in tensors))
        memories.append(base)
    # Fixed-input numerical floor across the exact deployed attachment and the
    # shared receiving tensor path, in singleton, batch and reversed batch order.
    states = torch.stack([memory.state for memory in memories])
    queries = torch.tensor([[float(q) for q in query] for _, query in cases], dtype=DTYPE)
    with torch.no_grad():
        batched = model.reader(model.connector(states), queries)["answer"]
        reversed_batch = model.reader(model.connector(states.flip(0)), queries.flip(0))["answer"].flip(0)
    batch_error = max(abs(float(a) - b) for a, b in zip(batched, base_reads))
    order_error = float((batched - reversed_batch).abs().max())
    gates["fixed_input_batch_shape_noise"] = max(batch_error, order_error) <= cfg["tol"]
    gates["decision_changing_controls_present"] = all(
        any(effect > 10 * cfg["tol"] for effect in values) for values in effects.values())
    bad = json.loads(memories[0].serialize())
    bad["content"]["state"][0] += 1
    try:
        LearnedMemory.reload(packed(bad), cfg["latent_width"])
        gates["malformed_envelope_rejected_separately"] = False
    except ValueError:
        gates["malformed_envelope_rejected_separately"] = True
    reload_error = max(abs(a - b) for a, b in zip(original_reads, restored_reads))
    gates["saved_parameters_reload"] = reload_error <= cfg["tol"]
    gates["reloaded_model_reconstructs"] = reconstruction_error == 0.0
    gates["edited_fork_reload_continuation_structure"] = all(recovery_gates)
    gates["edited_fork_reload_continuation_targets"] = max(recovery_errors) <= cfg["tol"]
    gates["all_learned_targets"] = not failures
    gates["tensor_payload_ceiling"] = storage["max_tensor_payload_bytes"] <= cfg["payload_ceiling_bytes"]
    roles = {role: {"questions": len(values), "max_abs_error": max(values),
                    "mean_abs_error": sum(values) / len(values),
                    "within_tolerance": all(v <= cfg["tol"] for v in values)}
             for role, values in errors.items()}
    from experiments.cloud.lsr_jobs import s0_memory_port as port
    return {"pass": all(gates.values()), "gates": gates, "roles": roles,
            "value_domain_strata": {
                domain: {label: {"questions": len(values),
                                 "max_abs_error": max(values) if values else None,
                                 "failure_count": sum(value > cfg["tol"] for value in values)}
                         for label, values in split.items()} for domain, split in strata.items()},
            "worlds": len(cases), "questions": len(original_reads), "sample": examples,
            "failure_count": len(failures), "failures": failures[:32],
            "named_refusal_sample": refusal, "storage": storage, "costs": costs.report(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()
                                             if p.requires_grad),
            "frozen_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()
                                          if not p.requires_grad),
            "checkpoint_bytes": len(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "checkpoint_medium": "in-memory torch.save bytes, reloaded into an independent instance",
            "parameter_reload_max_abs_error": reload_error,
            "reloaded_model_reconstruction_max_abs_error": reconstruction_error,
            "edited_fork_reload_continuation": {
                "questions": len(recovery_errors), "max_abs_error": max(recovery_errors),
                "failure_count": sum(v > cfg["tol"] for v in recovery_errors),
                "both_orders_and_read_only": all(recovery_gates),
                "both_forks_reloaded_and_retracted": True},
            "singleton_vs_batch_max_abs_error": batch_error,
            "reordered_batch_max_abs_error": order_error,
            "control_oracle_effects": {
                name: {"decision_changing_cases": sum(v > 10 * cfg["tol"] for v in values),
                       "max_abs_effect": max(values)} for name, values in effects.items()},
            "zero_answer_null_max_abs_error": max(abs(float(oracle(records, query)))
                                                   for records, query in cases),
            "distinct_late_answers": len(set(base_reads)),
            "s0_padded_attachment_bytes_per_read": port.WIDTH,
            "s0_helper_sha256": sha(Path(port.__file__).read_bytes())}


def run_check(name, cfg, config_hash=None):
    # This separately executable exact smoke always completes before constructing
    # an optimizer, including when only receiver/continuation is requested.
    exact = exact_smoke(cfg)
    print(json.dumps({"phase": "exact-smoke-before-fit", **exact}), flush=True)
    if name == "exact-controls" or not exact["pass"]:
        return {"exact": exact, "pass": exact["pass"], "fit_attempted": False}
    if name == "trained-trace-v2":
        return run_successors(cfg, config_hash, exact)
    if name == "confirm-v2-linear":
        return run_confirmation(cfg, config_hash, exact)
    cfg = {key: value for key, value in cfg.items() if key not in (*SUCCESSORS, "confirmation")}
    print(json.dumps({"phase": "frozen-fit-recipe", "config": cfg,
                      "config_sha256": config_hash,
                      "config_semantic_sha256": sha(packed(cfg))}), flush=True)
    model, training = fit(cfg)
    equivalence = boundary_equivalence(model, cfg)
    gradients = gradient_gate(model, cfg)
    trace = learned_trace(model, cfg)
    fit_gates = {"deadline": not training["deadline_hit"] and training["fit_seconds"] <= cfg["fit_wall_seconds"],
                 "fixed_step_count": training["steps"] == cfg["optimizer_steps"],
                 "first_connector_gradient": training["first_gradient_l1"].get("connector.weight", 0) > 0,
                 "frozen_unchanged": training["frozen_parameters_unchanged"]}
    ok = all(fit_gates.values()) and gradients["pass"] and trace["pass"]
    return {"pass": ok, "exact": exact, "fit_attempted": True, "training": training,
            "recipe": cfg["recipe"], "check_aliases": ["receiver", "continuation", "trained-trace"],
            "original_recipe_semantic_sha256": sha(packed(cfg)),
            "fit_gates": fit_gates, "production_returned_loss_backward": gradients,
            "boundary_equivalence_after_fit": equivalence,
            "learned": trace,
            "bet_verdict": "BOUNDED_RECIPE_PASSES" if ok else "RECIPE_REJECTED",
            "claim_scope": "synthetic learned arithmetic memory; no lookup-only, pretrained reuse or efficiency claim"}


def run_successors(cfg, config_hash, exact):
    configs = {name: successor_config(cfg, name) for name in SUCCESSORS}
    data_by_arm = {name: training_episodes_v2(configs[name]) for name in SUCCESSORS}
    data_hashes = {name: manifest["data_sha256"] for name, (_, manifest) in data_by_arm.items()}
    if len(set(data_hashes.values())) != 1:
        raise ValueError("successor arms have different training data")
    mismatches = [key for key in configs[SUCCESSORS[0]]
                  if key not in ("recipe", "constructor_kind", "arithmetic_structure")
                  and configs[SUCCESSORS[0]][key] != configs[SUCCESSORS[1]][key]]
    if mismatches:
        raise ValueError(f"unmatched successor resource/data settings: {mismatches}")
    resources = {}
    for name, arm_cfg in configs.items():
        torch.manual_seed(arm_cfg["seed"])
        instrument = ArithmeticModel(arm_cfg)
        resources[name] = {
            "trainable_parameter_bytes": sum(p.numel() * p.element_size() for p in instrument.parameters()
                                             if p.requires_grad),
            "frozen_parameter_bytes": sum(p.numel() * p.element_size() for p in instrument.parameters()
                                          if not p.requires_grad),
            "frozen_receiver_sha256": sha(instrument.reader.frozen.weight.detach().numpy().tobytes()),
            "initial_connector_sha256": sha(instrument.connector.weight.detach().numpy().tobytes())}
        if arm_cfg["constructor_kind"] == "linear-in-value":
            with torch.no_grad():
                for k in range(N_KEYS):
                    unit = instrument.constructor(inputs([("unit", k, Fraction(1))]))[0]
                    for n in range(-22, 21):
                        actual = instrument.constructor(inputs([("unit", k, Fraction(n, 8))]))[0]
                        torch.testing.assert_close(actual, unit * (n / 8), atol=1e-12, rtol=0)
        if not event_semantics(arm_cfg)["pass"]:
            raise ValueError("successor event-semantics instrument failed")
    if resources[SUCCESSORS[0]] != resources[SUCCESSORS[1]]:
        raise ValueError("successor parameter/receiver/connector resources do not match")
    # Freeze both complete schedules, input tensors and code BEFORE either fit.
    # The output is an observed ordering receipt, not an external timestamp authority.
    hashes = {name: sha(packed(arm_cfg)) for name, arm_cfg in configs.items()}
    print(json.dumps({"phase": "frozen-successor-recipes-before-any-fit",
                      "config_sha256": config_hash, "recipe_sha256": hashes,
                      "data_sha256": data_hashes, "resources": resources,
                      "code_sha256": sha(Path(__file__).read_bytes()), "configs": configs}), flush=True)
    arms = {}
    for name, arm_cfg in configs.items():
        data, manifest = data_by_arm[name]
        model, training = fit_v2(arm_cfg, data, manifest)
        training_gate = training["training_max_error_gate"]
        print(json.dumps({"phase": "training-max-error-gate-before-heldout",
                          "recipe": name, "pass": training_gate,
                          "max_abs_error": training["training_max_abs_error"],
                          "tolerance": TOL}), flush=True)
        equivalence = boundary_equivalence(model, arm_cfg)
        gradients = gradient_gate(model, arm_cfg)
        trace = learned_trace(model, arm_cfg)  # Exactly one diagnostic panel per arm.
        fit_gates = {
            "deadline": not training["deadline_hit"] and training["fit_seconds"] <= arm_cfg["fit_wall_seconds"],
            "fixed_step_count": training["steps"] == arm_cfg["optimizer_steps"],
            "training_max_error": training_gate,
            "first_connector_gradient": training["first_gradient_l1"].get("connector.weight", 0) > 0,
            "frozen_unchanged": training["frozen_parameters_unchanged"]}
        ok = all(fit_gates.values()) and gradients["pass"] and trace["pass"]
        arms[name] = {"pass": ok, "recipe_sha256": hashes[name],
                      "structure_label": arm_cfg["arithmetic_structure"],
                      "training": training, "fit_gates": fit_gates,
                      "heldout_interpretation": "admissible" if training_gate else "diagnostic_only_training_gate_failed",
                      "boundary_equivalence_after_fit": equivalence,
                      "production_returned_loss_backward": gradients, "learned": trace,
                      "bet_verdict": "BOUNDED_RECIPE_PASSES" if ok else "RECIPE_REJECTED"}
    return {"pass": all(arm["pass"] for arm in arms.values()), "exact": exact,
            "fit_attempted": True, "fits_per_arm": 1, "heldout_panels_per_arm": 1,
            "arms": arms, "matched_resources": resources, "identical_training_data": data_hashes,
            "original_recipe": cfg["recipe"], "original_recipe_verdict": "RECIPE_REJECTED",
            "original_recipe_verdict_provenance": "preserved first report/review; not refitted by this check",
            "claim_scope": "bounded synthetic arithmetic; linear arm injects structure; no pretrained or efficiency claim"}


def world_signature(records):
    # Exact dyadic signatures omit random IDs; shared scalar support is allowed.
    return tuple(sorted((key, Fraction(value)) for _, key, value in records))


def confirmation_panel(cfg, arm_cfg):
    spec = cfg["confirmation"]
    if (spec["worlds"] < 96 or spec["edits_per_fork"] != 3 or
            spec["actions_per_fork"] != 7 or spec["tolerance"] != TOL or
            spec["wall_seconds"] != 1200):
        raise ValueError("confirmation contract changed")
    if spec["world_seed"] in (cfg["train_seed"], cfg["heldout_seed"]):
        raise ValueError("confirmation seed overlaps an earlier seed")
    forbidden = set()
    # Original training worlds, then successor coverage-adjusted training worlds
    # and replacement views; isolated targets have a different cardinality.
    forbidden.update(world_signature(r) for r, _ in worlds(cfg["train_worlds"], cfg["train_seed"], cfg))
    train_cfg = {**arm_cfg, "value_numerator_min": -22, "value_numerator_max": 20}
    grid = list(range(-22, 21))
    for w, (records, _) in enumerate(worlds(arm_cfg["train_worlds"], arm_cfg["train_seed"], train_cfg)):
        key, value = w % N_KEYS, Fraction(grid[(w // N_KEYS) % len(grid)], 8)
        records = [(rid, k, value if k == key else old) for rid, k, old in records]
        forbidden.add(world_signature(records))
        rid = records[0][0]
        edited = event_oracle(records, [("replace", "train", rid, Fraction(grid[(w * 7 + 11) % len(grid)], 8))])
        forbidden.add(world_signature(edited["records"]))
    training_count = len(forbidden)
    # Earlier primary and recovery panels use the same base and single-edit views.
    for records, _ in worlds(cfg["heldout_worlds"], cfg["heldout_seed"], cfg):
        forbidden.add(world_signature(records))
        for i in (0, 1):
            rid, _, value = records[i]
            edited = event_oracle(records, [("replace", "prior", rid, value + Fraction(*cfg["edit_deltas"][i]))])
            forbidden.add(world_signature(edited["records"]))
    panel, signatures, bases = [], set(), set()
    for w, (records, query) in enumerate(worlds(spec["worlds"], spec["world_seed"], cfg)):
        base_sig = world_signature(records)
        if base_sig in bases:
            raise ValueError("duplicate confirmation base world; no seed retry permitted")
        bases.add(base_sig)
        branches = []
        for branch in (0, 1):
            rid, other = records[branch][0], records[(branch + 1) % N_KEYS][0]
            values = [Fraction(*v) * (-1 if branch else 1) for v in spec["edit_values"]]
            e0, e1, e2 = [f"fresh-{w}-{branch}-{i}" for i in range(3)]
            actions = [("replace", e0, rid, values[0]), ("replace", e1, rid, values[1]),
                       ("retract", e0), ("retract", e1), ("replace", e2, other, values[2]),
                       ("retract", e2), ("retract", e0)]
            branches.append(actions)
            for step in range(len(actions) + 1):
                signatures.add(world_signature(event_oracle(records, actions[:step])["records"]))
        panel.append((records, query, branches))
    overlap = signatures & forbidden
    if overlap:
        raise ValueError(f"confirmation overlaps training/earlier panel in {len(overlap)} worlds; refuse, no retry")
    serial = [{"records": [(rid, k, rational_json(v)) for rid, k, v in r],
               "query": [rational_json(v) for v in q],
               "actions": [[[a[0], a[1], *([a[2], rational_json(a[3])] if len(a) == 4 else [])]
                            for a in actions] for actions in branches]} for r, q, branches in panel]
    return panel, {"panel_sha256": sha(packed(serial)), "confirmation_sha256": sha(packed(spec)),
                   "seed": spec["world_seed"], "world_clusters": len(panel),
                   "unique_confirmation_states": len(signatures), "training_states_checked": training_count,
                   "training_and_prior_states_checked": len(forbidden), "world_overlap": len(overlap),
                   "signature_policy": "exact key/value worlds without IDs, every action prefix; scalar support shared"}


def evaluate_confirmation(panel, cfg, model=None):
    costs, storage, rows, failures, gates = Costs(), {}, {}, [], {}
    arm = "exact" if model is None else "recipe_v2_linear"
    world_ok = []
    late_answers, null_errors = [], []
    def gate(name, condition):
        gates[name] = gates.get(name, True) and bool(condition)
    def charge(views):
        for view in views:
            blob = costs.run("serialize", view.serialize)
            numeric = (len(packed([rational_json(v) for v in view.aggregate])) if model is None else
                       sum(t.numel() * t.element_size() for t in
                           [view.state] + [v[1] for v in view.records.values()] +
                           [t for e in view.events.values() for t in e[2:4]]))
            for key, value in {"max_serialized_view_bytes": len(blob),
                               "max_resident_view_bytes": resident_bytes(view),
                               "max_numeric_payload_bytes": numeric,
                               "max_event_metadata_bytes": event_metadata_bytes(view)}.items():
                storage[key] = max(storage.get(key, 0), value)
        storage["max_resident_three_views_bytes"] = max(storage.get("max_resident_three_views_bytes", 0),
                                                       resident_bytes(views))
    def read(view, q):
        if model is None:
            return costs.run("read", view.read, q)
        with torch.no_grad():
            return float(costs.run("boundary_read", boundary_read, model, view, q)["answer"])
    def score(w, role, view, q, reference, tags):
        # Target is computed only AFTER the source-free deployed call returns.
        answer = read(view, q)
        expected = oracle(reference, q)
        error = abs(float(answer) - float(expected))
        ok = error <= TOL and (model is not None or answer == expected)
        relevant = [float(v) for _, k, v in reference if q[k]]
        labels = [role, *tags]
        for domain, lo, hi in (("initial", -2, 2), ("training_legal", -2.75, 2.5)):
            labels.append(domain + ("_in_range" if all(lo <= v <= hi for v in relevant) else "_out_of_range"))
        for label in labels:
            row = rows.setdefault(label, {"questions": 0, "failure_count": 0, "max_abs_error": 0.0})
            row["questions"] += 1
            row["failure_count"] += int(not ok)
            row["max_abs_error"] = max(row["max_abs_error"], error)
        if not ok:
            failures.append({"world": w, "role": role, "tags": tags, "answer": float(answer),
                             "target": float(expected), "abs_error": error})
        null_errors.append(abs(float(expected)))
        return float(answer), ok
    for w, (records, query, actions_by_branch) in enumerate(panel):
        base = costs.run("construct", ExactMemory, records) if model is None else costs.run("construct", LearnedMemory, model, records)
        branches = [costs.run("fork", base.copy) for _ in range(2)]
        base_blob = costs.run("serialize", base.serialize)
        answer, success = score(w, "late", base, query, records, [])
        late_answers.append(answer)
        for step in range(cfg["confirmation"]["actions_per_fork"]):
            references = []
            tags = []
            for b, actions in enumerate(actions_by_branch):
                action = actions[step]
                untouched = costs.run("serialize", branches[1-b].serialize)
                before = costs.run("serialize", branches[b].serialize)
                reference = event_oracle(records, actions[:step + 1])
                references.append(reference["records"])
                if action[0] == "replace":
                    args = action[1:] if model is None else (model, *action[1:])
                    actual = costs.run("replace", branches[b].replace, *args)
                else:
                    actual = costs.run("retract", branches[b].retract, action[1])
                gate("event_receipts", actual == reference["receipts"][-1])
                gate("event_identities", branches[b].current_event == reference["current_event"] and
                     branches[b].predecessor_event == reference["predecessor_event"])
                if actual["status"] == "REFUSED":
                    gate("refused_bytes_unchanged", before == branches[b].serialize())
                blob = costs.run("serialize", branches[b].serialize)
                branches[b] = (costs.run("reload", ExactMemory.reload, blob) if model is None else
                               costs.run("reload", LearnedMemory.reload, blob, cfg["latent_width"]))
                gate("reload_bytes", branches[b].serialize() == blob)
                gate("fork_isolation", branches[1-b].serialize() == untouched and base.serialize() == base_blob)
                tags.append(["fork_" + "AB"[b], "step_" + str(step + 1), action[0]])
            snapshots = [v.serialize() for v in branches]
            answers = {}
            for order_name, order in (("AB", (0, 1)), ("BA", (1, 0))):
                for b in order:
                    value, ok = score(w, "fork_late", branches[b], query, references[b],
                                      tags[b] + ["order_" + order_name])
                    success &= ok
                    if b in answers:
                        gate("order_invariance", answers[b] == value)
                    answers[b] = value
            for b in (0, 1):
                # Every key, including affected and unrelated coordinates.
                for k in range(N_KEYS):
                    q = [Fraction(int(k == j)) for j in range(N_KEYS)]
                    _, ok = score(w, "revised_key", branches[b], q, references[b], tags[b])
                    success &= ok
            _, ok = score(w, "historical", base, query, records, ["step_" + str(step+1)])
            success &= ok
            gate("read_only", snapshots == [v.serialize() for v in branches] and base.serialize() == base_blob)
            charge([base, *branches])
        world_ok.append(success)
    gate("numeric_payload_ceiling", storage["max_numeric_payload_bytes"] <= cfg["payload_ceiling_bytes"])
    return {"arm": arm, "pass": not failures and all(gates.values()), "gates": gates,
            "world_clusters": len(panel), "complete_worlds_pass": sum(world_ok),
            "questions": len(null_errors), "failure_count": len(failures), "failures": failures[:32],
            "strata": rows, "max_abs_error": max(v["max_abs_error"] for v in rows.values()),
            "costs": costs.report(), "storage": storage, "distinct_late_answers": len(set(late_answers)),
            "zero_answer_null_max_abs_error": max(null_errors),
            "resource_scope": "same three simultaneous views and action histories; object/tensor accounting, not peak RSS; timings include toy transport for learned reads"}


def run_confirmation(cfg, config_hash, exact):
    start = time.monotonic()
    arm_cfg = successor_config(cfg, "recipe_v2_linear")
    data, manifest = training_episodes_v2(arm_cfg)
    panel, panel_manifest = confirmation_panel(cfg, arm_cfg)
    frozen = {"phase": "confirmation-frozen-before-fit", **panel_manifest,
              "confirmation": cfg["confirmation"], "config_sha256": config_hash,
              "recipe_sha256": sha(packed(arm_cfg)), "recipe_section_sha256": sha(packed(cfg["recipe_v2_linear"])),
              "training_data_sha256": manifest["data_sha256"], "code_sha256": sha(Path(__file__).read_bytes())}
    print(json.dumps(frozen), flush=True)
    exact_panel = evaluate_confirmation(panel, cfg)
    print(json.dumps({"phase": "confirmation-exact-control-before-fit", "pass": exact_panel["pass"],
                      "questions": exact_panel["questions"], "max_abs_error": exact_panel["max_abs_error"]}), flush=True)
    if not exact_panel["pass"]:
        return {"pass": False, "fit_attempted": False, "frozen": frozen, "exact_panel": exact_panel}
    model, training = fit_v2(arm_cfg, data, manifest)
    equivalence = boundary_equivalence(model, arm_cfg)
    gradients = gradient_gate(model, arm_cfg)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    checkpoint = stream.getvalue()
    torch.manual_seed(cfg["reload_seed"])
    restored = ArithmeticModel(arm_cfg)
    restored.load_state_dict(torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True))
    restored.eval()
    learned = evaluate_confirmation(panel, cfg, restored)
    gates = {"training_max_error": training["training_max_error_gate"],
             "fixed_steps": training["steps"] == arm_cfg["optimizer_steps"],
             "fit_deadline": not training["deadline_hit"], "gradients": gradients["pass"],
             "frozen_unchanged": training["frozen_parameters_unchanged"],
             "fresh_trace": learned["pass"], "wall_budget": time.monotonic() - start <= cfg["confirmation"]["wall_seconds"]}
    return {"pass": all(gates.values()), "gates": gates, "frozen": frozen,
            "fits": 1, "fresh_panels_per_arm": 1, "training": training,
            "boundary_equivalence": equivalence, "production_loss_backward": gradients,
            "exact_smoke": exact, "exact_panel": exact_panel, "learned": learned,
            "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": len(checkpoint),
            "trainable_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad),
            "frozen_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters() if not p.requires_grad),
            "parameter_counts": {"raw_trainable": 552, "raw_constructor": 456,
                                 "nonlinear_constructor_output_affecting": 448,
                                 "linear_constructor_participating": 456,
                                 "linear_constructor_effective_key_vector_coordinates": 32,
                                 "note": "participation is not identifiable dimension; factor and downstream gauge redundancies remain"},
            "three_way_label": {"recipe_v2_linear": "injected arithmetic structure; see current fresh verdict",
                                "recipe_v2": "failed nonlinear successor; not fitted by confirmation",
                                "original": "failed original 250-step recipe; not fitted by confirmation"},
            "scope": "synthetic arithmetic only; no pretrained reuse or efficiency claim; exact comparator gets the same records/actions/queries; no target enters learned read",
            "seconds": time.monotonic() - start}


def _cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--check", required=True)
    args = parser.parse_args(argv)
    if args.check not in CHECKS:
        print(json.dumps({"check": args.check, "status": "UNKNOWN_CHECK", "exit_code": 2}))
        return 2
    start = time.perf_counter()
    receipt = {"check": args.check, "code_sha256": sha(Path(__file__).read_bytes())}
    try:
        cfg, config_hash = freeze_config(args.config.resolve())
        receipt.update(config_sha256=config_hash, frozen_tolerance=cfg["tol"],
                       threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                       device="cpu", python_version=sys.version.split()[0], torch_version=torch.__version__)
        helper_path = ROOT / "experiments/cloud/lsr_jobs/s0_memory_port.py"
        helper_hash = sha(helper_path.read_bytes())
        receipt["s0_helper_before_sha256"] = helper_hash
        detail = run_check(args.check, cfg, config_hash)
        receipt["detail"] = detail
        status = detail["pass"]
        receipt["config_unchanged_after_fit"] = sha(args.config.read_bytes()) == config_hash
        receipt["s0_helper_after_sha256"] = sha(helper_path.read_bytes())
        receipt["s0_helper_unchanged_during_check"] = receipt["s0_helper_after_sha256"] == helper_hash
        status &= receipt["config_unchanged_after_fit"] and receipt["s0_helper_unchanged_during_check"]
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        status = False
    receipt.update(status="PASS" if status else "FAIL", exit_code=0 if status else 1,
                   total_seconds=time.perf_counter() - start)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    sys.exit(_cli())
