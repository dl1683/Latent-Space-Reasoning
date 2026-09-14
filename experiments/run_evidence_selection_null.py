"""Fixed lexical evidence-location diagnostic; no model or fitting."""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def sentences(abstract, full_text):
    section, cursor, result = None, 0, []
    parts = re.split(r"(?m)^(ABSTRACT\.[A-Z]+:+)\s*$", abstract)
    for part in parts:
        if re.fullmatch(r"ABSTRACT\.[A-Z]+:+", part):
            section = part.rstrip(":").split(".")[1]
            continue
        if section is None:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", part.strip()):
            if not sentence:
                continue
            start = full_text.find(sentence, cursor)
            if start < 0:
                raise ValueError("Abstract sentence cannot be bound to full source")
            cursor = start + len(sentence)
            result.append({"section": section, "start": start, "end": cursor,
                           "text": sentence})
    if not result:
        raise ValueError("No candidate sentences")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-cache")
    args = parser.parse_args()
    path = Path(args.panel)
    output = Path(args.output)
    if output.resolve() == path.resolve():
        raise ValueError("Output must differ from the reviewed input panel")
    panel = json.loads(path.read_text(encoding="utf-8"))
    cached = {}
    if args.source_cache:
        raw = json.loads(Path(args.source_cache).read_text(encoding="utf-8"))
        for record in raw["selected"]:
            for key in ("source", "released_abstract"):
                if key in record:
                    cached[record[key]["url"]] = record[key]["text"].encode("utf-8")

    def read(source):
        if source["url"] in cached:
            data = cached[source["url"]]
        else:
            response = requests.get(source["url"], timeout=20)
            response.raise_for_status()
            data = response.content
        if len(data) > 200000 or hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError("Source size or hash mismatch")
        return source["url"], data.decode("utf-8")

    sources = [a[k] for a in panel["articles"] for k in ("full_source", "abstract_source")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        texts = dict(pool.map(read, sources))
    candidates = {a["pmcid"]: sentences(texts[a["abstract_source"]["url"]],
                                        texts[a["full_source"]["url"]])
                  for a in panel["articles"]}
    rows = []
    for question in panel["queries"]:
        query_tokens = tokens(question["outcome"])
        for arm in ("all_abstract", "results_only"):
            pool = [s for s in candidates[question["pmcid"]]
                    if arm == "all_abstract" or s["section"] == "RESULTS"]
            if not pool:
                raise ValueError("Empty candidate pool")
            scores = [len(query_tokens & tokens(s["text"])) for s in pool]
            best = max(scores)
            tied = [dict(s, overlap_score=score) for s, score in zip(pool, scores) if score == best]
            rows.append({"prompt_id": question["prompt_id"], "pmcid": question["pmcid"],
                         "arm": arm, "candidate_count": len(pool), "top_score": best,
                         "selected": tied[0], "top_ties": tied,
                         "manual_adequacy": None})
    catalog = {}
    for row in rows:
        ids = []
        for sentence in row.pop("top_ties"):
            key = f"{row['pmcid']}:{sentence['start']}:{sentence['end']}"
            catalog[key] = {k: v for k, v in sentence.items() if k != "overlap_score"}
            ids.append(key)
        row.pop("selected")
        row["selected_candidate_id"], row["top_tied_candidate_ids"] = ids[0], ids
    panel["diagnostic"] = {"rows": rows, "sentences": catalog, "model_forwards": 0,
                           "status": "scored; manual evidence adequacy review required"}
    output.write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(panel["queries"]), "scored_rows": len(rows),
                      "model_forwards": 0, "status": "manual review required"}))


if __name__ == "__main__":
    main()
