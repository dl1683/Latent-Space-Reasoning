"""Retokenize visible v7 request frames without loading weights or generating."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer, __version__ as transformers_version

ROOT = Path(__file__).resolve().parents[1] / "results/whole_world_planner_preflight_2026-11-26_v7"
MANIFEST_SHA256 = "cdce4e7953013e1092372106af919f93af7041408c1673c21ff0f1e050fa8ba1"
TEMPLATE_SHA256 = "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct"
    / "snapshots/a10cc1512eabd3dde888204e902eca88bddb4951"
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def packed_registry(needs):
    if not needs:
        return needs
    keys = list(needs[0])
    assert all(set(need) == set(keys) for need in needs)
    records = [[need[key] for key in keys] for need in needs]
    assert [dict(zip(keys, row, strict=True)) for row in records] == needs
    return [keys, records]


def delta_frame(frame, previous):
    delta = frame.copy()
    for key in ("need_registry", "receipts"):
        start = len(previous[key])
        assert frame[key][:start] == previous[key]
        values = frame[key][start:]
        delta[key] = {"from": start, "append": values}
        assert previous[key] + values == frame[key]
    return delta


def without_receipt_spans(receipts):
    slim = []
    for receipt in receipts:
        result = receipt["result"].copy()
        for key in ("candidates", "reads"):
            if key in result:
                result[key] = [
                    {field: value for field, value in record.items() if field != "span"}
                    for record in result[key]
                ]
        slim.append({"request": receipt["request"], "result": result})
    return slim


def collect(tokenizer):
    manifest = (ROOT / "SHA256SUMS.txt").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("The v7 archive manifest has changed")
    archive = {}
    for line in manifest.decode("ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        archive[name] = digest
    summary = {}
    assert len(archive) == 23
    for name, digest in archive.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"The v7 archive file has changed: {name}")
        if not name.endswith("__compact.jsonl"):
            continue
        previous = {"need_registry": [], "receipts": []}
        counts = {}
        out = []
        rows = [json.loads(line) for line in raw.splitlines()]
        for row in rows:
            data = row["data"]
            if row["kind"] == "response":
                out.append(data["output_tokens"])
                continue
            if row["kind"] != "request":
                continue
            (system_role, system), (user_role, original) = data["messages"]
            assert (system_role, user_role) == ("system", "user")
            frame = json.loads(original)
            assert compact(frame) == original
            need_ids = [{"id": need["id"]} for need in frame["need_registry"]]
            receipts_without_spans = without_receipt_spans(frame["receipts"])
            scenarios = {
                "baseline": original,
                "key_arrays": compact(frame | {"need_registry": packed_registry(frame["need_registry"])}),
                "omit_need_bodies": compact(frame | {"need_registry": need_ids}),
                "omit_receipt_spans": compact(frame | {"receipts": receipts_without_spans}),
                "omit_both": compact(frame | {
                    "need_registry": need_ids,
                    "receipts": receipts_without_spans,
                }),
                "deltas": compact(delta_frame(frame, previous)),
                "omit_registries_and_receipts": compact(frame | {"need_registry": [], "receipts": []}),
            }
            for scenario, content in scenarios.items():
                messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=False,
                )
                assert isinstance(ids, list) and ids and isinstance(ids[0], int)
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                assert tokenizer(rendered, add_special_tokens=False)["input_ids"] == ids
                counts.setdefault(scenario, []).append(len(ids))
            counts.setdefault("archived_input", []).append(len(data["input_ids"]))
            previous = frame
        assert len(out) == 7 and all(len(vals) == 7 for vals in counts.values())
        summary[name] = counts | {"archived_output": out}
    assert len(summary) == 11
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_snapshot, local_files_only=True, trust_remote_code=False)
    template_sha256 = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    if template_sha256 != TEMPLATE_SHA256:
        raise ValueError("The tokenizer chat template does not match the recorded diagnostic")
    episodes = collect(tok)
    totals = {}
    for scenario in next(iter(episodes.values())):
        rows = [episode[scenario] for episode in episodes.values()]
        sums = [sum(row) for row in rows]
        budget = 1792 if scenario == "archived_output" else 8192
        totals[scenario] = {
            "all": sum(sums), "mean": round(mean(sums), 1), "min": min(sums), "max": max(sums),
            "max_call": max(map(max, rows)), "fit_total": sum(total <= budget for total in sums),
            "phase_means": [round(mean(i), 1) for i in zip(*rows, strict=True)],
        }
    print(json.dumps({
        "chat_template_sha256": template_sha256,
        "transformers_version": transformers_version,
        "aggregate": totals, "episodes": episodes,
    }, indent=2))

if __name__ == "__main__":
    main()
