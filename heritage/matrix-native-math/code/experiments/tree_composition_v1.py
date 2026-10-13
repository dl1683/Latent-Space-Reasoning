"""Registered correctness control, not a performance experiment.

H0: index/shape/order defects may invalidate direct tree composition; standard
TQS arithmetic and retained chains remain the novelty/performance null.
H1 correctness: all registered fixtures agree with an independent unique-path
oracle. Exact cases require equality; floats require relative error <=1e-11.
One mismatch kills the gate. Do not replace seeds or overwrite attempts.
Full run requires independent code review after the two registered smoke cases.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from tree_composition import TreeOperator

REGISTRATION = ROOT / "research/TREE_COMPOSITION_V1_CORRECTNESS_REGISTRATION.json"
REGISTRATION_SHA256 = "744025ff0dd96828285600c09a31b5d6610dba29fb5ff09b6eba7f896f152b55"
AMENDMENT = ROOT / "research/TREE_COMPOSITION_V1_HOLDOUT_AMENDMENT.json"
AMENDMENT_SHA256 = "72e653c7f2c40b3a6a256fb66724ecb69d521f1550bf951a122b681dd28112aa"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(graph, graph_index, mode, mode_index, operand_offset, asymmetric=False):
    seed = 1009 + 37 * graph_index + mode_index + operand_offset
    rng = np.random.default_rng(seed)
    adjacency = [[] for _ in range(graph["nodes"])]
    edges = graph["edges"]
    if edges and isinstance(edges[0], int):  # Registered single-edge JSON form.
        edges = [edges]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    adjacency = [sorted(row) for row in adjacency]
    sizes = [1 + (i + graph_index) % 3 for i in range(len(adjacency))]
    if mode == "exact_rational" and len(adjacency) >= 4:
        sizes[1] = 0
    widths = {(i, j): (7 * i + 3 * j + seed) % 4
              for i, row in enumerate(adjacency) for j in row}
    if asymmetric:
        widths = {(i, j): (i + 2 * j + operand_offset // 10000) % 4
                  for i, row in enumerate(adjacency) for j in row}

    def draw(shape):
        values = rng.integers(-2, 3, size=shape)
        if mode == "exact_integer":
            return values.astype(object)
        if mode == "exact_rational":
            return np.array([Fraction(int(v), 3) for v in values.flat],
                            dtype=object).reshape(shape)
        if mode == "complex128":
            return (values + 1j * rng.integers(-2, 3, size=shape)) / 4
        return values / 4

    D = {i: draw((size, size)) for i, size in enumerate(sizes)}
    P = {(i, j): draw((widths[i, j], sizes[i])) for i, j in widths}
    Q = {(i, j): draw((sizes[i], widths[j, i])) for i, j in widths}
    T = {(i, j, k): draw((widths[i, j], widths[k, i]))
         for i, row in enumerate(adjacency) for j in row for k in row if j != k}
    return TreeOperator(adjacency, D, P, Q, T)


def path_oracle(op):
    """Reconstruct each block using its unique path, with no message sweeps."""
    dense = np.zeros(op.shape, dtype=op.dtype)
    for source in range(len(op.adjacency)):
        parent = {source: None}
        queue = [source]
        for vertex in queue:
            for neighbor in op.adjacency[vertex]:
                if neighbor not in parent:
                    parent[neighbor] = vertex
                    queue.append(neighbor)
        for target in range(len(op.adjacency)):
            if source == target:
                value = op.D[source]
            else:
                path = [target]
                while path[-1] != source:
                    path.append(parent[path[-1]])
                path.reverse()
                value = op.P[path[0], path[1]]
                for index in range(1, len(path) - 1):
                    value = op.T[path[index], path[index + 1], path[index - 1]] @ value
                value = op.Q[target, path[-2]] @ value
            dense[op.slices[target], op.slices[source]] = value
    return dense


def compare(actual, expected, exact, tolerance, label):
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape mismatch")
    if exact:
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{label}: exact mismatch")
        return 0.0
    error = float(np.linalg.norm(actual - expected))
    scale = float(np.linalg.norm(expected))
    relative = error / scale if scale else (0.0 if error == 0 else float("inf"))
    if not np.isfinite(relative) or relative > tolerance:
        raise AssertionError(f"{label}: relative error {relative}")
    return relative


def check_case(graph, gi, mode, mi, tolerance, asymmetric=False):
    exact = mode.startswith("exact_")
    a, b, c = (fixture(graph, gi, mode, mi, offset, asymmetric) for offset in (0, 10000, 20000))
    if asymmetric:
        assert any(a.P[i, j].shape[0] != a.P[j, i].shape[0] for i, j in a.P)
        assert any(a.P[e].shape[0] != b.P[e].shape[0] for e in a.P)
        assert any(op.P[e].shape[0] == 0 for op in (a, b, c) for e in op.P)
    dense_a, dense_b, dense_c = map(path_oracle, (a, b, c))
    errors = []
    x = (np.arange(a.shape[0] * 2).reshape(a.shape[0], 2) % 5 - 2).astype(
        object if exact else np.float64)
    for op, dense in ((a, dense_a), (b, dense_b), (c, dense_c)):
        errors.append(compare(op.apply(x), dense @ x, exact, tolerance, "input batch apply"))
        errors.append(compare(op.apply(x[:, 0]), dense @ x[:, 0], exact, tolerance, "input vector apply"))
    with patch.object(TreeOperator, "apply", side_effect=AssertionError("composition applied an input")):
        product = a.compose(b)
        left_chain = product.compose(c)
        right_chain = a.compose(b.compose(c))
    expected = dense_a @ dense_b
    errors.append(compare(path_oracle(product), expected, exact, tolerance, "AB descriptor"))
    errors.append(compare(product.apply(x), expected @ x, exact, tolerance, "AB apply"))
    expected_chain = expected @ dense_c
    errors.append(compare(path_oracle(left_chain), expected_chain, exact, tolerance, "(AB)C"))
    errors.append(compare(path_oracle(right_chain), expected_chain, exact, tolerance, "A(BC)"))
    for edge in a.P:
        assert product.P[edge].shape[0] == a.P[edge].shape[0] + b.P[edge].shape[0]
        assert left_chain.P[edge].shape[0] == sum(op.P[edge].shape[0] for op in (a, b, c))
    oracle_payload = json.dumps([[str(v) for v in row] for row in expected], separators=(",", ":"))
    return {"case": f"{graph['name']}/{mode}" + ("/directed_holdout" if asymmetric else ""),
            "status": "PASS", "exact": exact, "directed_holdout": asymmetric,
            "dimension": a.shape[0], "checks": len(errors), "max_relative_error": max(errors),
            "operand_scalar_counts": [a.scalar_count, b.scalar_count],
            "product_scalar_count": product.scalar_count,
            "oracle_sha256": hashlib.sha256(oracle_payload.encode()).hexdigest()}


def check_rejections():
    count = 0

    def rejects(fn):
        nonlocal count
        try:
            fn()
        except (ValueError, TypeError):
            count += 1
        else:
            raise AssertionError("Invalid contract was accepted")

    for adjacency in ([], [[0]], [[1], []], [[1, 2], [0, 2], [0, 1]],
                      [[1], [0], [3], [2]], [[1, 1], [0]]):
        rejects(lambda adjacency=adjacency: TreeOperator(adjacency, {}, {}, {}, {}))
    graph = {"name": "edge", "nodes": 2, "edges": [[0, 1]]}
    op = fixture(graph, 1, "exact_integer", 0, 0)
    rejects(lambda: op.apply(np.zeros(op.shape[0] + 1)))
    wrong_p = dict(op.P)
    wrong_p[0, 1] = np.zeros((1, 100))
    rejects(lambda: TreeOperator(op.adjacency, op.D, wrong_p, op.Q, op.T))
    other_partition = fixture(graph, 2, "exact_integer", 0, 0)
    rejects(lambda: op.compose(other_partition))
    different_tree = fixture({"name": "singleton", "nodes": 1, "edges": []},
                             0, "exact_integer", 0, 0)
    rejects(lambda: op.compose(different_tree))
    return count


def check_integer_promotion():
    """Regression for mixed signed/unsigned rounding, including the r=0 case."""
    values = [2**53 + 1, 2**63 + 1]
    D = {0: np.array([[values[0]]], dtype=np.int64),
         1: np.array([[values[1]]], dtype=np.uint64)}
    P = {(0, 1): np.zeros((0, 1), dtype=np.int64),
         (1, 0): np.zeros((0, 1), dtype=np.uint64)}
    Q = {(0, 1): np.zeros((1, 0), dtype=np.int64),
         (1, 0): np.zeros((1, 0), dtype=np.uint64)}
    op = TreeOperator([[1], [0]], D, P, Q, {})
    assert op.dtype == np.dtype(object)
    expected = np.array(values, dtype=object)
    assert np.array_equal(op.apply(np.ones(2, dtype=np.int64)), expected)
    assert np.array_equal(op.compose(op).apply(np.ones(2, dtype=np.int64)), expected * expected)
    return "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    assert sha(REGISTRATION) == REGISTRATION_SHA256, "Registration bytes changed"
    config = json.loads(REGISTRATION.read_text(encoding="utf-8-sig"))
    assert hashlib.sha256(config["null_prediction"].encode()).hexdigest() == config["null_prediction_sha256_utf8"]
    assert sha(AMENDMENT) == AMENDMENT_SHA256, "Amendment bytes changed"
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8-sig"))
    assert hashlib.sha256(amendment["null_prediction"].encode()).hexdigest() == amendment["null_prediction_sha256_utf8"]
    output = args.out or ROOT / "results/tree_composition_v1" / ("smoke.json" if args.smoke else "correctness.json")
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Artifacts must remain in this repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    intent = output.with_suffix(".intent.json")
    if output.exists() or intent.exists():
        raise FileExistsError("Attempt already exists; preserve it and use a new reviewed attempt")
    header = {"unit": config["id"], "smoke": args.smoke,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": REGISTRATION_SHA256,
              "amendment_sha256": AMENDMENT_SHA256,
              "runner_sha256": sha(Path(__file__)),
              "implementation_sha256": sha(ROOT / "code/tree_composition.py"),
              "numpy_version": np.__version__, "python_version": sys.version,
              "timing_claim": False, "novelty_claim": False, "mission_complete": False}
    with intent.open("x", encoding="utf-8") as handle:
        json.dump(header, handle, indent=2)
    results = []
    status, failure = "PASS", None
    try:
        for gi, graph in enumerate(config["graphs"]):
            for mi, mode in enumerate(config["modes"]):
                case = f"{graph['name']}/{mode}"
                if args.smoke and case not in config["smoke_cases"]:
                    continue
                results.append(check_case(graph, gi, mode, mi, config["relative_error_max"]))
        primary_count = len(config["smoke_cases"]) if args.smoke else config["full_cases"]
        assert len(results) == primary_count
        for holdout in amendment["holdouts"]:
            gi, graph = next((gi, graph) for gi, graph in enumerate(config["graphs"])
                             if graph["name"] == holdout["graph_name"])
            mode = holdout["mode"]
            mi = config["modes"].index(mode)
            results.append(check_case(graph, gi, mode, mi, config["relative_error_max"], asymmetric=True))
        assert len(results) == primary_count + amendment["additional_cases"]
        rejection_count = check_rejections()
        integer_promotion = check_integer_promotion()
    except Exception as error:
        status, failure, rejection_count, integer_promotion = "FAIL", repr(error), None, None
    result = dict(header, status=status, failure=failure, cases=results,
                  rejected_contracts=rejection_count, integer_promotion=integer_promotion,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": status, "cases": len(results), "failure": failure, "output": str(output)}))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
