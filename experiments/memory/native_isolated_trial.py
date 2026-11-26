"""Replay a native latent publication across source-only and late-read Linux stages."""

import argparse
import json
import os
from pathlib import Path
from random import Random
from secrets import randbelow
from shutil import which
from stat import S_ISDIR, S_ISREG
import sys
from tempfile import TemporaryDirectory

from experiments.memory.native_isolated_stage import LABELS, MAX_BYTES
from experiments.memory.repository_publication_isolation import (
    _ACQUIRE_UID, _BOUNDARY, _STAGE_RECEIPT, _isolated, _json_record, _snapshot_code,
    _sync_directory, _write_record,
)
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.verified_source_tool import sha256


_MODULES = (
    "latent_feedback", "linear_feedback", "native_forecast_set", "native_isolated_stage", "native_latent_recurrence",
    "native_publication", "selective_continuations", "source_revision_feed",
    "verified_source_tool", "weight_program",
)
_STAGE = _BOUNDARY + """
def readonly(path):
    assert os.statvfs(path).f_flag & os.ST_RDONLY

def refuse_subprocess(event, arguments):
    if event == "subprocess.Popen":
        raise AssertionError("native stage attempted to launch a subprocess")

sys.addaudithook(refuse_subprocess)
absent("/code/.git/HEAD")
absent("/code/experiments/memory/native_isolated_trial.py")
if kind in ("publish", "revise"):
    absent("/task.txt")
    readonly("/source/updates.json")
    if kind == "publish":
        absent("/publication/manifest.json")
    else:
        readonly("/publication/manifest.json")
else:
    assert kind == "read"
    absent("/source/updates.json")
    absent("/output/publication/manifest.json")
    readonly("/task.txt")
    if args[2] == "none":
        absent("/publication/manifest.json")
    else:
        readonly("/publication/manifest.json")
""" + _STAGE_RECEIPT + """
from experiments.memory.native_isolated_stage import main
main(kind, args)
"""


def _retained_bytes(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError("native handoff requires bounded regular files with one link")
        data = handle.read(limit + 1)
        if len(data) != info.st_size:
            raise ValueError("native handoff size changed while retaining its bytes")
        return data


def _retain(candidate, bundle, receipt, source_seal, runtime_pin):
    if (not S_ISDIR(candidate.lstat().st_mode)
            or {p.name for p in candidate.iterdir()} != {"manifest.json", "rows.bin"}):
        raise ValueError("native publisher must leave exactly a manifest and numeric rows")
    manifest = _retained_bytes(candidate / "manifest.json", MAX_BYTES)
    rows = _retained_bytes(candidate / "rows.bin", MAX_BYTES - len(manifest))
    header = json.loads(manifest)
    if (sha256(manifest) != receipt["seal"] or header["source"] != source_seal
            or receipt["source_seal"] != source_seal or header["runtime"] != runtime_pin
            or header["rows_bytes"] != len(rows) or header["rows_sha256"] != sha256(rows)):
        raise ValueError("native publication bytes differ from their seals")
    bundle.mkdir(mode=0o755)
    _write_record(bundle / "manifest.json", manifest, mode=0o444)
    _write_record(bundle / "rows.bin", rows, mode=0o444)
    _sync_directory(bundle.parent)
    return len(manifest) + len(rows)


def _verify_reads(reads, scopes, allowed):
    spans = count = 0
    for read in reads:
        seal = read["observed_source_seal"]
        if seal not in allowed:
            raise ValueError("source span does not belong to an allowed epoch")
        for hit in read["selected"]:
            span = hit["span"]
            raw = scopes[seal][span["source_id"]].read_bytes()
            begin, end = span["start_byte"], span["end_byte"]
            if (not 0 <= begin < end <= len(raw) or sha256(raw) != span["version_sha256"]
                    or sha256(raw[begin:end]) != span["span_sha256"]):
                raise ValueError("raw source proof does not match the proposed span")
            spans += 1
            count += len(raw)
    return spans, count


def _selected(reads):
    return [hit["span"]["source_id"] for read in reads for hit in read["selected"]]


def _grade(reads, publications, scopes, paths, choice, answers):
    older, current, source_only, bare, restored = reads
    original, corrected = publications
    cue, middle, ending = paths[choice]
    history = original["source_seal"]
    seeds = original["seeds"]
    if (len(seeds) != len(paths) or seeds != corrected["seeds"]
            or (original["source_backbone_calls"], original["worker_backbone_calls"],
                corrected["source_backbone_calls"], corrected["worker_backbone_calls"]) != (6, 6, 2, 0)):
        raise ValueError("the correction did not preserve the prior source-only computations")

    checked = []
    for seed, path in zip(seeds, paths, strict=True):
        if seed["span"]["source_id"] != path[0] or _selected(seed["reads"]) != path[:2]:
            raise ValueError("a source-only computation does not follow its expected path")
        checked.append(_verify_reads(seed["reads"], scopes, {history}))
    for read, pin, answer in zip(reads, (original, corrected, corrected, None, corrected),
                                 (*answers, "", "", answers[1]), strict=True):
        if (not read["terminated"] or read["text"] != answer or read["latent_steps"] != 2
                or read["publication_seal"] != (None if pin is None else pin["seal"])
                or read["source_seal"] != (None if pin is None else pin["source_seal"])):
            raise ValueError("a late read missed its answer, budget, or pin control")
        traces = (*read["source_reads"], *read["derived_reads"])
        checked.append(_verify_reads(traces, scopes, {read["source_seal"]}))
    for read in (older, current, restored):
        if (read["applied_derived_steps"] != 1 or _selected(read["source_reads"]) != [cue, ending]
                or _selected(read["derived_reads"]) != [cue]):
            raise ValueError("attached answer lacks the expected causal path")
    if (current != restored or not source_only["derived_omitted"] or source_only["applied_derived_steps"]
            or source_only["derived_reads"] or _selected(source_only["source_reads"]) != [cue, middle]
            or bare["source_reads"] or bare["derived_reads"] or bare["applied_memory_steps"]):
        raise ValueError("source-only, omitted-memory or restoration control failed")
    return {key: sum(items) for key, items in zip(("verified_spans", "source_bytes_read"), zip(*checked))}


def _source_custody(root, documents):
    root.mkdir(mode=0o700)
    paths = {source: root / source for source in documents}
    for source, raw in documents.items():
        _write_record(paths[source], raw)
    return paths


def run(output, *, seed):
    tool = which("bwrap")
    runtime = Path(sys.prefix).resolve()
    output = output.resolve()
    if (not __debug__ or sys.platform != "linux" or os.geteuid() or not tool
            or not Path("/usr/bin/setpriv").is_file() or sys.prefix == sys.base_prefix
            or not (runtime / "bin/python").is_file()):
        raise RuntimeError("use Linux root, Bubblewrap and a dedicated CPU Python venv with assertions")
    if any(path.is_relative_to(mount) for path in (output, Path(__file__).resolve())
           for mount in (runtime, Path("/usr").resolve())):
        raise ValueError("the controller and custody must be outside mounts the reader receives")
    output.mkdir(mode=0o700)
    if output.stat().st_mode & 0o077:
        raise ValueError("custody directory requires private Linux filesystem permissions")
    _sync_directory(output.parent)

    with TemporaryDirectory(prefix="lsr-native-isolation-", dir="/tmp") as workspace:
        root = Path(workspace)
        code = root / "code"
        snapshots, _ = _snapshot_code(code, _MODULES)
        runtime_pin = sha256(json.dumps(snapshots, sort_keys=True).encode("utf8") + _STAGE.encode("utf8"))
        rng = Random(seed)
        labels = rng.sample(LABELS, 10)
        routes = (labels[:5], labels[5:])
        identities = sorted(f"{rng.getrandbits(128):032x}" for _ in range(6))
        paths = (identities[::2], identities[1::2])
        opening, correction = {}, {}
        for (start_id, middle_id, end_id), (cue, middle, ending, before, after) in zip(paths, routes, strict=True):
            opening[start_id] = f"link {cue} {middle}".encode()
            opening[middle_id] = f"link {middle} {ending}".encode()
            opening[end_id] = f"name {ending} {before}".encode()
            correction[end_id] = f"name {ending} {after}".encode()
        scopes, receipts = {}, []
        previous = None
        for mode, changes in (("publish", opening), ("revise", correction)):
            feed, worker = root / f"{mode}-feed", root / f"{mode}-worker"
            delta = write_source_delta(feed, changes, previous=previous)
            feed.chmod(0o755)
            worker.mkdir(mode=0o700)
            os.chown(worker, _ACQUIRE_UID, _ACQUIRE_UID)
            bindings = [("--ro-bind", feed, "/source"), ("--bind", worker, "/output")]
            arguments = [runtime_pin, delta.seal, "maintain"]
            if previous is not None:
                bindings.append(("--ro-bind", output / "publish", "/publication"))
                arguments.extend((receipts[0]["seal"], previous.seal))
            raw = _isolated(tool, runtime, code, mode, bindings, arguments, output,
                            f"{mode}-receipt.json", stage_script=_STAGE)
            receipt = json.loads(raw)
            receipt["publication_bytes"] = _retain(worker / "publication", output / mode,
                                                   receipt, delta.seal, runtime_pin)
            scopes[delta.seal] = _source_custody(root / f"{mode}-custody", opening | changes)
            receipts.append(receipt)
            previous = delta

        # The task choice does not exist until both writers have exited.
        choice = randbelow(len(routes))
        cue, _, _, before, after = routes[choice]
        task_path = output / "task.txt"
        task = f"follow {cue}".encode()
        _write_record(task_path, task, mode=0o444)
        raw_proposals = []
        for arm, mode, pin in (("history", "maintained", 0), ("current", "maintained", 1),
                               ("ablate", "source-only", 1), ("omit", "none", None),
                               ("restore", "maintained", 1)):
            bindings = [("--ro-bind", task_path, "/task.txt")]
            arguments = [runtime_pin, "-", mode]
            if pin is not None:
                bindings.append(("--ro-bind", output / ("publish", "revise")[pin], "/publication"))
                arguments.extend((receipts[pin]["seal"], receipts[pin]["source_seal"]))
            raw_proposals.append(_isolated(tool, runtime, code, "read", bindings, arguments, output,
                                           f"{arm}.json", stage_script=_STAGE))

        proposals = [json.loads(raw) for raw in raw_proposals]
        grade = _grade(proposals, receipts, scopes, paths, choice, (before, after))
        result = dict(seed=seed, task_choice=choice, runtime_pin=runtime_pin, stage_code_sha256=snapshots, stages=7,
                      task_sha256=sha256(task), publications=receipts, **grade,
                      proposals=[dict(text=read["text"], backbone_calls=read["backbone_calls"],
                                      sha256=sha256(raw))
                                 for read, raw in zip(proposals, raw_proposals, strict=True)])
        _json_record(output / "grade.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=55631)
    args = parser.parse_args()
    print(json.dumps(run(args.output, seed=args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
