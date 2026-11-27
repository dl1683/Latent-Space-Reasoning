"""Visible ARC-AGI-1 replay of a source-only factorized grid operation."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from os import link
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np


_RULE_VERSION = "arc-visual-geometry-palette-v1"
_ARC_REVISION = "399030444e0ab0cc8b4e199870fb20b863846f34"
_GEOMETRIES = tuple(
    (turns, reflected, mode, factor)
    for reflected in (False, True)
    for turns in range(4)
    for mode, factor in (("same", 1), ("repeat", 2), ("repeat", 3), ("tile", 2), ("tile", 3))
)
_EMPTY_PALETTE = (0,) * 10
_RULE_SHA256 = sha256(json.dumps((_RULE_VERSION, _GEOMETRIES), separators=(",", ":")).encode("ascii")).hexdigest()


def _write(path, value):
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    pending = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(raw)
        link(pending, path)
    finally:
        if pending is not None:
            pending.unlink()
    return sha256(raw).hexdigest()


def _read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def prepare(checkout, destination):
    """Separate demonstrations from late inputs and public development labels."""
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != _ARC_REVISION:
        raise ValueError("unexpected ARC-AGI-1 revision")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", "data/training"], check=True)
    files = sorted((checkout / "data" / "training").glob("*.json"))
    if len(files) != 400:
        raise ValueError("expected the complete ARC-AGI-1 public training split")
    sources, late, labels = {}, {}, {}
    for path in files:
        task = json.loads(path.read_bytes())
        sources[path.stem] = task["train"]
        late[path.stem] = [case["input"] for case in task["test"]]
        labels[path.stem] = [case["output"] for case in task["test"]]
    destination.mkdir()
    digest = _write(destination / "sources.json", sources)
    late_digest = _write(destination / "late.json", {"source_sha256": digest, "cases": late})
    _write(destination / "labels.json", {"late_sha256": late_digest, "cases": labels})
    return digest


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= size <= 30 for size in grid.shape)
            or grid.dtype.kind not in "iu" or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a rectangular ARC grid of colors 0 through 9")
    return grid


def _geometry(grid, index):
    turns, reflected, mode, factor = _GEOMETRIES[index]
    oriented = np.rot90(np.fliplr(grid) if reflected else grid, turns)
    if mode == "repeat":
        return np.repeat(np.repeat(oriented, factor, axis=0), factor, axis=1)
    if mode == "tile":
        return np.tile(oriented, (factor, factor))
    return oriented


def _constraint(left, right, index, work):
    work["demonstration_candidates"] += 1
    turns, _, _, factor = _GEOMETRIES[index]
    rows, cols = left.shape if turns % 2 == 0 else left.shape[::-1]
    if (rows * factor, cols * factor) != right.shape:
        return None
    transformed = _geometry(left, index)
    work["demonstration_cells"] += right.size
    palette = [0] * 10
    for code in np.unique(10 * transformed + right):
        source, target = divmod(int(code), 10)
        if palette[source]:
            return None
        palette[source] = 1 << target
    return tuple(palette)


def _merge(left, right):
    if left is None or right is None:
        return None
    union = tuple(a | b for a, b in zip(left, right, strict=True))
    return union if all(mask.bit_count() <= 1 for mask in union) else None


def _join(case_constraints):
    live = (_EMPTY_PALETTE,) * len(_GEOMETRIES)
    for constraint in case_constraints:
        live = tuple(_merge(left, right) for left, right in zip(live, constraint, strict=True))
    return live


def _candidate_rows(demonstration, work):
    left, right = (_grid(demonstration[role]) for role in ("input", "output"))
    return tuple(_constraint(left, right, index, work) for index in range(len(_GEOMETRIES)))


def _fresh_fit(demonstrations, work):
    """Reference starts over from the warm raw demonstrations and prunes as it goes."""
    live = (_EMPTY_PALETTE,) * len(_GEOMETRIES)
    for demo in demonstrations:
        left, right = (_grid(demo[role]) for role in ("input", "output"))
        live = tuple(_merge(palette, _constraint(left, right, index, work)) if palette is not None else None
                     for index, palette in enumerate(live))
    return live


def publish(source_path, publication_path):
    """Publish source-only evidence and a counterfactual witness withdrawal."""
    sources, source_sha = _read(source_path)
    if len(sources) != 400:
        raise ValueError("source roster differs from the public split")
    work, bank, states = Counter(), {}, {}
    for identity, demos in sources.items():
        if len(demos) < 2 or any(set(demo) != {"input", "output"} for demo in demos):
            raise ValueError("a source needs at least two paired demonstrations")
        constraints = tuple(_candidate_rows(demo, work) for demo in demos)
        bank[identity] = constraints
        states[identity] = _join(constraints)
    work["compiled_candidate_checks"] = len(_GEOMETRIES) * sum(map(len, bank.values()))
    return _write(publication_path, {
        "revision": _ARC_REVISION, "rule_sha256": _RULE_SHA256, "source_sha256": source_sha, "work": dict(work),
        "constraint_bank": bank, "states": states,
    })


def revise(base_path, base_sha, result_path, *, previous=None, previous_sha=None):
    """Withdraw the first demonstration, or restore it from the sealed bank."""
    base, actual_sha = _read(base_path)
    if (base_sha != actual_sha or base["revision"] != _ARC_REVISION or base["rule_sha256"] != _RULE_SHA256
            or (previous is None) != (previous_sha is None)):
        raise ValueError("opening publication differs from its pin")
    bank = base["constraint_bank"]
    if previous is None:
        action, parent = "withdraw", base_sha
        states = {identity: _join(constraints[1:]) for identity, constraints in bank.items()}
        retained_demos = sum(len(rows) - 1 for rows in bank.values())
    else:
        prior, parent = _read(previous)
        if (parent != previous_sha or prior["base_sha256"] != base_sha or prior["parent"] != base_sha
                or prior["action"] != "withdraw" or prior["states"].keys() != bank.keys()):
            raise ValueError("restoration requires the pinned withdrawal")
        action = "restore"
        states = {
            identity: tuple(_merge(left, right) for left, right in
                            zip(prior["states"][identity], constraints[0], strict=True))
            for identity, constraints in bank.items()
        }
        retained_demos = len(bank)
        if {identity: _wire_fit(rows) for identity, rows in states.items()} != base["states"]:
            raise ValueError("restored source result differs from the opening publication")
    return _write(result_path, {
        "base_sha256": base_sha, "parent": parent, "action": action, "states": states,
        "compiled_candidate_checks": len(_GEOMETRIES) * retained_demos,
    })


def _wire_fit(rows):
    return [None if palette is None else list(palette) for palette in rows]


def answer(grid, memory=None):
    """Apply the same read to an optional set of source-derived operations."""
    if memory is None:
        return "omitted", None
    grid = _grid(grid)
    outputs = set()
    for index, palette in enumerate(memory):
        if palette is None:
            continue
        if any(not palette[color] for color in np.unique(grid)):
            return "unbound_color", None
        result = _geometry(grid, index)
        colors = np.array([mask.bit_length() - 1 for mask in palette])
        outputs.add(tuple(map(tuple, colors[result].tolist())))
        if len(outputs) > 1:
            return "ambiguous", None
    if not outputs:
        return "no_program", None
    return "predicted", next(iter(outputs))


def grade(inputs):
    """Predict before opening public labels and compare with raw recomputation."""
    sources, source_sha = _read(inputs.sources)
    late, late_sha = _read(inputs.late)
    base, base_sha = _read(inputs.publication)
    withdrawal, withdrawal_sha = _read(inputs.withdrawal)
    restore, restore_sha = _read(inputs.restoration)
    if (base_sha != inputs.publication_sha256 or base["source_sha256"] != source_sha
            or late["source_sha256"] != source_sha or base["revision"] != _ARC_REVISION
            or base["rule_sha256"] != _RULE_SHA256
            or withdrawal_sha != inputs.withdrawal_sha256 or withdrawal["parent"] != base_sha
            or restore_sha != inputs.restoration_sha256 or restore["parent"] != withdrawal_sha
            or any(view["base_sha256"] != base_sha for view in (withdrawal, restore))
            or (withdrawal["action"], restore["action"]) != ("withdraw", "restore")
            or sources.keys() != late["cases"].keys()
            or any(view["states"].keys() != sources.keys() for view in (base, withdrawal, restore))
            or base["states"] != restore["states"]):
        raise ValueError("source, publication history or late roster differs from its pin")

    work = Counter()
    coverage = Counter()
    score_rows = {}
    for identity, grids in late["cases"].items():
        demos = sources[identity]
        active, off, restored = (view["states"][identity] for view in (base, withdrawal, restore))
        full_reference = _fresh_fit(demos, work)
        off_reference = _fresh_fit(demos[1:], work)
        if _wire_fit(full_reference) != active or _wire_fit(off_reference) != off:
            raise ValueError("maintained and raw-reference programs disagree")
        count = sum(palette is not None for palette in active)
        if count == 0:
            coverage["none"] += 1
        elif count == 1:
            coverage["one"] += 1
        else:
            coverage["multiple"] += 1
        score_rows[identity] = []
        for grid in grids:
            prediction, withdrawn, after_restore = (answer(grid, state) for state in (active, off, restored))
            if answer(grid) != ("omitted", None) or prediction != after_restore or (prediction, withdrawn) != (
                    answer(grid, full_reference), answer(grid, off_reference)):
                raise ValueError("maintained and recomputed readouts disagree")
            novel = all(grid != demo["input"] for demo in demos)
            shape = len(grid), len(grid[0])
            unseen_shape = all(shape != (len(demo["input"]), len(demo["input"][0])) for demo in demos)
            seen_colors = {color for demo in demos for row in demo["input"] for color in row}
            unseen_color = any(color not in seen_colors for row in grid for color in row)
            score_rows[identity].append((prediction, withdrawn, novel, unseen_shape, unseen_color))

    labels, labels_sha = _read(inputs.labels)
    if labels_sha != inputs.labels_sha256 or labels["late_sha256"] != late_sha or labels["cases"].keys() != score_rows.keys():
        raise ValueError("public labels do not match the frozen inputs")
    totals = Counter()
    withdrawal_changes, withdrawal_correctness = Counter(), Counter()
    for identity, rows in score_rows.items():
        task_correct = task_novel = True
        for row, expected, grid in zip(rows, labels["cases"][identity], late["cases"][identity], strict=True):
            (status, prediction), withdrawn, novel, unseen_shape, unseen_color = row
            expected = tuple(map(tuple, expected))
            correct = prediction == expected
            source_outputs = [tuple(map(tuple, demo["output"])) for demo in sources[identity]]
            totals.update({"cases": 1, status: 1, "correct": int(correct), "novel_inputs": int(novel),
                           "correct_novel_inputs": int(correct and novel),
                           "unseen_shape": int(unseen_shape), "correct_unseen_shape": int(correct and unseen_shape),
                           "unseen_color": int(unseen_color), "correct_unseen_color": int(correct and unseen_color),
                           "input_copy_correct": int(tuple(map(tuple, grid)) == expected),
                           "last_demo_output_copy_correct": int(source_outputs[-1] == expected),
                           "output_present_in_demos": int(expected in source_outputs),
                           "correct_new_outputs": int(correct and expected not in source_outputs)})
            task_correct &= correct
            task_novel &= novel
            if withdrawn != (status, prediction):
                withdrawal_changes.update({f"{status} -> {withdrawn[0]}": 1})
            withdrawal_correctness.update({f"{int(correct)} -> {int(withdrawn[1] == expected)}": 1})
        totals.update({"tasks": 1, "whole_tasks_correct": int(task_correct),
                       "whole_tasks_correct_all_novel": int(task_correct and task_novel),
                       "whole_tasks_correct_two_bindings": int(task_correct and len(rows) >= 2)})
    return {"totals": dict(totals), "program_coverage": dict(coverage), "publish_work": base["work"],
            "recompute_work_both_pins": dict(work),
            "revision_candidate_checks": {view["action"]: view["compiled_candidate_checks"]
                                          for view in (withdrawal, restore)},
            "withdrawal_changes": dict(withdrawal_changes),
            "withdrawal_correctness": dict(withdrawal_correctness), "publication_sha256": base_sha,
            "publication_bytes": {name: path.stat().st_size for name, path in
                                  (("opening", inputs.publication), ("withdrawal", inputs.withdrawal),
                                   ("restoration", inputs.restoration))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("checkout", type=Path)
    setup.add_argument("destination", type=Path)
    worker = commands.add_parser("publish")
    worker.add_argument("sources", type=Path)
    worker.add_argument("publication", type=Path)
    revision = commands.add_parser("revise")
    revision.add_argument("publication", type=Path)
    revision.add_argument("publication_sha256")
    revision.add_argument("destination", type=Path)
    revision.add_argument("--previous", type=Path)
    revision.add_argument("--previous-sha256")
    check = commands.add_parser("grade")
    for name in ("sources", "late", "publication", "withdrawal", "restoration", "labels"):
        check.add_argument(name, type=Path)
    for name in ("publication", "withdrawal", "restoration", "labels"):
        check.add_argument(f"{name}_sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.checkout, args.destination))
    elif args.command == "publish":
        print(publish(args.sources, args.publication))
    elif args.command == "revise":
        print(revise(args.publication, args.publication_sha256, args.destination,
                     previous=args.previous, previous_sha=args.previous_sha256))
    else:
        print(json.dumps(grade(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
