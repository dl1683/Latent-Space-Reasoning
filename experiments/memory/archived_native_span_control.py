"""Reproduce the archived learned-query span check without models or labels."""

from fractions import Fraction
from hashlib import sha256
from json import dumps, loads
from math import sqrt
from pathlib import Path
from struct import unpack_from


_ARCHIVE = Path(__file__).with_name("results") / "native_world_addressing_p2_2026-11-26"
_INDEX_DIGEST = "0a8840c0c8716d6c51017db73883e73f14ba15e33680e9ed4340a04dd82f9994"
_READS_DIGEST = "523df8aa4f94490d9ffe21b53172315df6f6138e29132b3961f49a88c00b373c"
_WIDTH = 1024
_LEVELS = 127
_L1_LIMIT = 4


def _pinned_bytes(path, digest, *, normalize_newlines=False):
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"archive digest differs for {path.name}")
    return data


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def _span_distance_squared(query, keys):
    """Exact squared distance to the span of a full-rank integer key bank."""
    rhs = [_dot(row, query) for row in keys]
    gram = [
        [Fraction(_dot(row, other)) for other in keys] + [Fraction(target)]
        for row, target in zip(keys, rhs, strict=True)
    ]
    for column in range(len(keys)):
        pivot = next((index for index in range(column, len(keys)) if gram[index][column]), None)
        if pivot is None:
            raise ValueError("expected independent archived keys")
        gram[column], gram[pivot] = gram[pivot], gram[column]
        divisor = gram[column][column]
        gram[column] = [value / divisor for value in gram[column]]
        for index, row in enumerate(gram):
            if index != column and row[column]:
                factor = row[column]
                gram[index] = [value - factor * ref for value, ref in zip(row, gram[column], strict=True)]
    coefficients = [row[-1] for row in gram]
    return _dot(query, query) - _dot(rhs, coefficients)


def _read_bank(header, payload):
    source_keys = []
    offset = 0
    for _, spans in header["documents"]:
        count = len(spans)
        values_start, keys_start = offset, offset + 8 * _WIDTH * count
        for row in range(count):
            values = unpack_from(f"<{_WIDTH}d", payload, values_start + row * 8 * _WIDTH)
            key = unpack_from(f"{_WIDTH}b", payload, keys_start + row * _WIDTH)
            if key != tuple(round(value * _LEVELS) for value in values):
                raise ValueError("archived source key disagrees with its identity projection")
            source_keys.append(key)
        offset = keys_start + _WIDTH * count
    seeds = header["derived"]
    if len(source_keys) != 5 or len(seeds) != 1 or offset + 16 * _WIDTH != len(payload):
        raise ValueError("unexpected archived source or continuation inventory")
    address = unpack_from(f"<{_WIDTH}d", payload, offset)
    derived = tuple(round(value * _LEVELS) for value in address)
    return source_keys, seeds[0], derived


def run():
    index = loads(_pinned_bytes(_ARCHIVE / "publication_index.json", _INDEX_DIGEST))
    pin = index["pins"][-1]
    if pin["pin"] != "restored-p2" or index["runtime"]["kernel"] != "identity-query-key-value-feedback":
        raise ValueError("archived pin or its coordinate system differs")
    root = _ARCHIVE / "publications" / pin["pin"]
    header = loads(_pinned_bytes(root / "manifest.json", pin["publication_seal"]))
    payload = _pinned_bytes(root / "rows.bin", header["rows_sha256"])
    if (header["schema"] != "lsr-native-selective-v1" or len(payload) != header["rows_bytes"]
            or header["source"] != pin["source_seal"]
            or sha256(header["source_manifest"].encode()).hexdigest() != pin["source_seal"]):
        raise ValueError("archived publication framing differs")
    source_keys, seed, derived = _read_bank(header, payload)

    records = _pinned_bytes(_ARCHIVE / "reads.jsonl", _READS_DIGEST, normalize_newlines=True)
    active, = (row["read"] for row in map(loads, records.splitlines()) if row["arm"] == "active")
    if active["source_seal"] != pin["source_seal"] or active["publication_seal"] != pin["publication_seal"]:
        raise ValueError("active trace differs from the archived bank")
    rounding_bound_squared = _WIDTH * Fraction(1 + _L1_LIMIT, 2) ** 2
    banks = {"derived": [derived], "source_anchor": [source_keys[seed]], "all_six": [*source_keys, derived]}
    measured = []
    for index, (source_step, derived_step) in enumerate(
            zip(active["source_steps"], active["derived_steps"], strict=True), start=1):
        packed = bytes.fromhex(derived_step["query"])
        if len(packed) != _WIDTH or source_step["query"] != derived_step["query"]:
            raise ValueError("archived source and continuation queries must agree")
        query = unpack_from(f"{_WIDTH}b", packed)
        score = _dot(derived, query)
        if score > 0 or derived_step["receipt"]["status"] != "empty":
            raise ValueError("archived active trace unexpectedly routed a continuation")
        residuals = {name: _span_distance_squared(query, keys) for name, keys in banks.items()}
        measured.append({
            "step": index,
            "derived_dot": score,
            "query_l2": sqrt(_dot(query, query)),
            "residual_l2": {name: sqrt(value) for name, value in residuals.items()},
            "all_six_outside_rounding_bound": residuals["all_six"] > rounding_bound_squared,
        })
    if len(measured) != 2:
        raise ValueError("expected both active latent steps")
    worker = []
    for source_index, frontiers in header["attempted"]:
        for step_index, step in enumerate(frontiers, start=1):
            query = unpack_from(f"{_WIDTH}b", bytes.fromhex(step[4]))
            residual = _span_distance_squared(query, banks["all_six"])
            worker.append({
                "source_index": source_index,
                "step": step_index,
                "same_source_epoch": step[7] == pin["source_seal"],
                "derived_dot": _dot(derived, query),
                "is_archived_derived_key": query == derived,
                "all_six_residual_l2": sqrt(residual),
                "all_six_outside_rounding_bound": residual > rounding_bound_squared,
            })
    if len(worker) != 10:
        raise ValueError("expected all ten preserved source-only query steps")
    return {
        "rounding_bound_l2": sqrt(rounding_bound_squared),
        "active_task_steps": measured,
        "preserved_worker_steps": worker,
    }


if __name__ == "__main__":
    print(dumps(run(), indent=2, sort_keys=True))
