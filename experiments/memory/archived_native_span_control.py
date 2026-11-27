"""Reproduce the archived learned-query span check without models or labels."""

from argparse import ArgumentParser
from fractions import Fraction
from hashlib import sha256
from json import dumps, loads
from math import isqrt, sqrt
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


def _solve(matrix, rhs):
    """Solve a small invertible Gram system over rationals."""
    gram = [
        [*map(Fraction, row), Fraction(target)]
        for row, target in zip(matrix, rhs, strict=True)
    ]
    for column in range(len(gram)):
        pivot = next((index for index in range(column, len(gram)) if gram[index][column]), None)
        if pivot is None:
            raise ValueError("expected an independent Gram basis")
        gram[column], gram[pivot] = gram[pivot], gram[column]
        divisor = gram[column][column]
        gram[column] = [value / divisor for value in gram[column]]
        for index, row in enumerate(gram):
            if index != column and row[column]:
                factor = row[column]
                gram[index] = [value - factor * ref for value, ref in zip(row, gram[column], strict=True)]
    return [row[-1] for row in gram]


def _gram(vectors):
    return [[_dot(vector, other) for other in vectors] for vector in vectors]


def _span_distance_squared(query, keys):
    """Exact squared distance to the span of a full-rank integer key bank."""
    rhs = [_dot(row, query) for row in keys]
    coefficients = _solve(_gram(keys), rhs)
    return _dot(query, query) - _dot(rhs, coefficients)


def _independent_indices(gram):
    """Keep an ordered basis of vectors given their positive semidefinite Gram matrix."""
    remaining = [list(map(Fraction, row)) for row in gram]
    indices = []
    for index, row in enumerate(remaining):
        pivot = row[index]
        if not pivot:
            continue
        indices.append(index)
        for later in remaining[index + 1:]:
            factor = later[index] / pivot
            for column in range(index + 1, len(row)):
                later[column] -= factor * row[column]
    return indices


def _ceil_norm(vector):
    square = _dot(vector, vector)
    root = isqrt(square)
    return root + (root * root < square)


def _shared_split_bound(queries, keys, *, address_error, coefficient_limit):
    """Bound all orthogonal splits that put the keys and fixed queries at one address."""
    differences = [tuple(a - b for a, b in zip(vector, keys[0], strict=True))
                   for vector in (*keys[1:], *queries)]
    basis = [differences[index] for index in _independent_indices(_gram(differences))]
    basis_gram = _gram(basis)
    vectors = [*keys, *queries]
    raw_key_rank = len(_independent_indices(_gram(keys)))
    query_rank_gain = len(_independent_indices(_gram(vectors))) - raw_key_rank
    rhs = [[_dot(vector, direction) for direction in basis] for vector in vectors]
    weights = [_solve(basis_gram, row) for row in rhs]
    projected = [[_dot(row, coefficients) for coefficients in weights] for row in rhs]
    projected_key_gram = [row[:len(keys)] for row in projected[:len(keys)]]
    key_indices = _independent_indices(projected_key_gram)
    key_gram = [[projected[i][j] for j in key_indices] for i in key_indices]
    axis_differs = [any(column) for column in zip(*differences, strict=True)]
    operand_axes = [index for index, differs in enumerate(axis_differs) if differs]
    axis_keys = [[key[index] for index in operand_axes] for key in keys]
    axis_basis = [axis_keys[i] for i in _independent_indices(_gram(axis_keys))]
    max_key_norm = max(map(_ceil_norm, keys))
    exact, tolerant, axis = [], [], []
    for query_index, query in enumerate(queries, start=len(keys)):
        key_rhs = [projected[i][query_index] for i in key_indices]
        coefficients = _solve(key_gram, key_rhs)
        residual_squared = projected[query_index][query_index] - _dot(key_rhs, coefficients)
        exact.append(residual_squared)
        # Express the least-squares residual r as a linear combination of
        # original differences. If every address gap is at most t, its address
        # projection is at most t times the L1 norm of these weights.
        residual_weights = [weight - _dot((weights[i][dim] for i in key_indices), coefficients)
                            for dim, weight in enumerate(weights[query_index])]
        slack = address_error * sum(map(abs, residual_weights)) * (
            _ceil_norm(query) + coefficient_limit * max_key_norm
        )
        lower = max(Fraction(0), residual_squared - slack)
        tolerant.append(lower ** 2 / residual_squared if residual_squared else Fraction(0))
        axis.append(_span_distance_squared([query[index] for index in operand_axes], axis_basis))
    common_address = _dot(keys[0], keys[0]) - projected[0][0]
    operand_rounding_squared = len(keys[0]) * Fraction(1 + coefficient_limit, 2) ** 2
    return {
        "raw_key_rank": raw_key_rank,
        "query_rank_gain": query_rank_gain,
        "difference_rank": len(basis),
        "projected_key_rank": len(key_indices),
        "maximal_common_address_l2": sqrt(common_address),
        "nonzero_equal_axes": sum(value != 0 and not differs
                                 for value, differs in zip(keys[0], axis_differs, strict=True)),
        "axis_residual_l2": list(map(sqrt, axis)),
        "exact_orthogonal_residual_l2": list(map(sqrt, exact)),
        "exact_orthogonal_residual_squared": list(map(str, exact)),
        "bounded_address_residual_lower_l2": list(map(sqrt, tolerant)),
        "bounded_address_exceeds_operand_rounding": [
            value > operand_rounding_squared for value in tolerant
        ],
    }


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


def run(*, orthogonal_split=False):
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
    measured, queries = [], []
    for index, (source_step, derived_step) in enumerate(
            zip(active["source_steps"], active["derived_steps"], strict=True), start=1):
        packed = bytes.fromhex(derived_step["query"])
        if len(packed) != _WIDTH or source_step["query"] != derived_step["query"]:
            raise ValueError("archived source and continuation queries must agree")
        query = unpack_from(f"{_WIDTH}b", packed)
        queries.append(query)
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
    result = {
        "rounding_bound_l2": sqrt(rounding_bound_squared),
        "active_task_steps": measured,
        "preserved_worker_steps": worker,
    }
    if orthogonal_split:
        grid_l2 = isqrt(_WIDTH - 1) + 1
        result["shared_orthogonal_split"] = {
            "address_error_l2": grid_l2,
            "operand_rounding_l2": sqrt(rounding_bound_squared),
            "coefficient_l1_limit": _L1_LIMIT,
            "banks": {name: _shared_split_bound(queries, keys, address_error=grid_l2,
                                               coefficient_limit=_L1_LIMIT)
                      for name, keys in banks.items()},
            "derived_single_step_residual_l2": [
                _shared_split_bound((query,), banks["derived"], address_error=grid_l2,
                                    coefficient_limit=_L1_LIMIT)["exact_orthogonal_residual_l2"][0]
                for query in queries
            ],
        }
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--orthogonal-split", action="store_true",
                        help="also bound shared address and operand splits on the saved task states")
    print(dumps(run(orthogonal_split=parser.parse_args().orthogonal_split), indent=2, sort_keys=True))
