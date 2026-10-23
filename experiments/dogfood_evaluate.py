"""Evaluator-only leakage receipts. Run AFTER immutable reader answers exist.

References schema: [{task_id, id, text}], evaluator-only. This module is never
imported by the source builder/reader. Semantic overlap is a conservative alert,
not proof of contamination. Unresolved alerts and missing checks withhold scores.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.dogfood import digest, encode, encoder_at, load, write
from experiments.dogfood_audit import audit_manifest, audit_payloads, make_canary_reference


def register_canary(references_path, output):
    references = load(references_path)
    if not references:
        raise ValueError("nonempty evaluator references required")
    canary = make_canary_reference(references[0]["text"])
    fake = Path(output).with_suffix(".gold_copy.json")
    write(fake, canary)
    write(output, dict(created_utc=datetime.now(timezone.utc).isoformat(),
        fake_gold=str(fake.resolve()), fake_gold_sha256=digest(fake.read_bytes()),
        reference_sha256=digest(Path(references_path).read_bytes())))


def register_gold_canary(gold_file, output):
    """Opaque custody copy before generation; never return golden content."""
    original = Path(gold_file).read_bytes()
    token = make_canary_reference("")["text"].splitlines()[-1]
    fake = Path(output).with_suffix(".gold_copy.txt")
    fake.write_bytes(original + b"\n" + token.encode("utf-8") + b"\n")
    write(output, dict(created_utc=datetime.now(timezone.utc).isoformat(),
        fake_gold=str(fake.resolve()), fake_gold_sha256=digest(fake.read_bytes()),
        source_gold_sha256=digest(original), token=token,
        custody="OPAQUE_BYTE_COPY_NOT_READER_INPUT"))


def _valid_id(value):
    return type(value) is int or (isinstance(value, str) and bool(value))


def audit(index, generation, references_path, encoder_path, output, registration=None):
    references = load(references_path)
    receipts = []
    generation_files = {p.name: p for p in Path(generation).glob("*.json")
                        if not p.name.endswith(".process.json")}
    records = {name: load(path) for name, path in generation_files.items()}
    if not records:
        raise ValueError("no frozen generations to audit")
    if not isinstance(references, list):
        raise ValueError("evaluator references must be a list")
    for record in [*references, *records.values()]:
        if not isinstance(record, dict) or not _valid_id(record.get("task_id")):
            raise ValueError("unsupported or missing task_id; require integer or nonempty string")
    reference_tasks = {r["task_id"] for r in references}
    if not {r["task_id"] for r in records.values()} <= reference_tasks:
        missing_pairs = [dict(task_id=r["task_id"], question_id=r.get("question_id"))
                         for r in records.values() if r["task_id"] not in reference_tasks]
        raise ValueError("references do not cover every generation task; missing_pairs=" +
                         json.dumps(missing_pairs, ensure_ascii=False))
    reference_bindings = {}
    all_reference_bindings = []
    for n, reference in enumerate(references):
        if not _valid_id(reference.get("id")):
            raise ValueError("unsupported or missing reference id; require integer or nonempty string")
        key = (reference["task_id"], reference["id"])
        # Hash the entire record as canonical UTF-8 JSON, not just its answer text.
        binding = dict(task_id=key[0], id=key[1], reference_index=n,
            detector_id=f"evaluator-reference-{n}",
            reference_record_sha256=digest(json.dumps(reference, sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        reference_bindings.setdefault(key, []).append(binding)
        all_reference_bindings.append(binding)
    missing_pairs = []
    for record in records.values():
        if not _valid_id(record.get("question_id")):
            raise ValueError("unsupported or missing question_id; require integer or nonempty string")
        key = (record["task_id"], record["question_id"])
        if key not in reference_bindings:
            missing_pairs.append(dict(task_id=key[0], question_id=key[1]))
    if missing_pairs:
        raise ValueError("references do not cover every generation question; missing_pairs=" +
                         json.dumps(missing_pairs, ensure_ascii=False))
    # Coverage and identity validation precede any encoder factory or audit output.
    encoder = encoder_at(encoder_path)
    embed = lambda texts: encode(encoder, texts)
    covered = set()
    prospective_canary = False
    if registration:
        registered = load(registration)
        fake = Path(registered["fake_gold"])
        if digest(fake.read_bytes()) != registered["fake_gold_sha256"]:
            raise ValueError("canary gold copy changed")
        if ("reference_sha256" in registered
                and registered["reference_sha256"] != digest(Path(references_path).read_bytes())):
            raise ValueError("canary reference binding mismatch")
        created = datetime.fromisoformat(registered["created_utc"]).timestamp()
        prospective_canary = (all(p.stat().st_mtime > created for p in generation_files.values())
            and not fake.resolve().is_relative_to(Path(index).resolve())
            and not fake.resolve().is_relative_to(Path(generation).resolve()))
        canary = (dict(id="canary", text=registered["token"]) if "token" in registered
                  else load(fake))
    else:
        canary = make_canary_reference("Disposable answer-only canary; never a source.")
        write(Path(output).with_suffix(".canary_gold.json"), canary)
    for directory in sorted(Path(index).iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").exists():
            continue
        manifest = load(directory / "manifest.json")
        task_bindings = [binding for binding in all_reference_bindings
                         if binding["task_id"] == directory.name]
        # Detector transport labels must be unique strings, including alternatives.
        # They never participate in the exact input-identity join above.
        refs = [{"id": binding["detector_id"],
                 "text": references[binding["reference_index"]]["text"]}
                for binding in task_bindings]
        if not refs:
            continue
        manifest_receipt = audit_manifest(manifest["files"], manifest["allowed_paths"])
        rows = load(directory / "rows.json")
        sources = []
        for n, row in enumerate(rows):
            sources.append(dict(label=f"source-{n}", text=row["text"]))
        # Each leaf/chunk and the corresponding geometry are pinned together.
        index_ok = all(digest((directory / name).read_bytes()) == manifest[key] for name, key in
            (("vectors.npy", "vector_sha256"), ("rows.json", "rows_sha256")))
        memory_receipt = audit_payloads(sources, refs + [canary], sources, embedder=embed)
        per_generation = []
        for name, filename in sorted(generation_files.items()):
            record = records[name]
            if record["task_id"] != directory.name:
                continue
            payloads, seen = [], set()
            for turn, event in enumerate(record["trace"]):
                for n, message in enumerate(event["messages"]):
                    text = message["content"]
                    if text not in seen:
                        payloads.append(dict(label=f"{turn}-{n}", text=text))
                        seen.add(text)
            check = audit_payloads(payloads, refs + [canary], sources, embedder=embed)
            token = canary["text"].splitlines()[-1]
            canary_absent = token.lower() not in json.dumps(record).lower()
            per_generation.append(dict(file=filename.name, generation_sha256=digest(filename.read_bytes()),
                task_id=record["task_id"], question_id=record["question_id"],
                reference_bindings=reference_bindings[(record["task_id"], record["question_id"])],
                overlap=check, canary_absent=canary_absent))
            covered.add(name)
        receipts.append(dict(task_id=directory.name, manifest=manifest_receipt,
            screened_reference_bindings=task_bindings,
            index_pins_match=index_ok, memory=memory_receipt, generation=per_generation))
    # Deliberate contamination proves detector sensitivity separately.
    sensitivity = audit_payloads([dict(label="deliberately_contaminated",
        text=canary["text"])], [canary], [], embedder=embed)
    def passed(receipt):
        return receipt.get("admission") == "PASS"
    ok = covered == set(generation_files) and bool(receipts) and all(
        passed(r["manifest"]) and r["index_pins_match"] and passed(r["memory"])
        and all(passed(g["overlap"]) and g["canary_absent"] for g in r["generation"])
        for r in receipts)
    write(output, dict(status="PASS" if ok and prospective_canary and not passed(sensitivity) else "WITHHELD",
        references_sha256=digest(Path(references_path).read_bytes()),
        tasks=receipts, canary_detector_control=sensitivity,
        isolation="SEPARATE_PROCESSES_AND_INPUT_ALLOWLIST_NOT_OS_SANDBOX",
        generation_hashes={name: digest(path.read_bytes()) for name, path in generation_files.items()},
        pretraining_exposure="UNKNOWN", wrong_task_outcomes="SEPARATE_NAMED_JUDGE_REQUIRED",
        canary_scope="PRE_GENERATION_GOLD_COPY" if prospective_canary else "POST_HOC_DETECTOR_ONLY",
        canary_registration=registration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-canary", action="store_true")
    for key in ("index", "generation", "references", "encoder", "output", "canary-registration", "gold-file"):
        parser.add_argument("--" + key)
    args = parser.parse_args()
    if args.register_canary:
        if args.gold_file:
            register_gold_canary(args.gold_file, args.output)
        else:
            register_canary(args.references, args.output)
    else:
        audit(args.index, args.generation, args.references, args.encoder, args.output,
              args.canary_registration)


if __name__ == "__main__":
    main()
