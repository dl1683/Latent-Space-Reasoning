"""S37 fixed, CPU-only teaching-channel preflight; no training or generation."""
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  TORCH_DEVICE_BACKEND_AUTOLOAD="0", HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/agreement_teaching_preflight_v1/result.json"
SPECS = {
    "recipient": ("TinyStories-1M", "77f1b168e219585646439073245fe87e56b3023e",
                  "07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7"),
    "donor": ("TinyStories-8M", "8612e3b15c66ffa94eaa6ee0de5c96edd2d630af",
              "22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716"),
}
NOUNS = ("dog", "cat", "boy", "girl")
START = time.monotonic()
RESULT = {"status": "started", "scope": "forced-choice local preflight, not acquisition",
          "forwards": 0, "row_presentations": 0, "models": {},
          "started_utc": datetime.now(timezone.utc).isoformat()}


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT["wall_seconds"] = time.monotonic() - START
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(RESULT, indent=2), encoding="utf-8")
    temp.replace(OUT)
    print(RESULT["status"], RESULT["forwards"], flush=True)


def main():
    import requests
    import psutil
    RESULT.update(phase="download", phase_started_epoch=time.time())
    save()
    cache = Path.home() / ".cache/huggingface/hub"
    downloaded = 0
    paths = {}
    files = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.json", "merges.txt", "pytorch_model.bin")
    for role, (name, rev, expected) in SPECS.items():
        folder = cache / ("models--roneneldan--" + name) / "snapshots" / rev
        folder.mkdir(parents=True, exist_ok=True)
        for filename in files:
            dest = folder / filename
            if not dest.exists():
                url = f"https://huggingface.co/roneneldan/{name}/resolve/{rev}/{filename}"
                partial = dest.with_suffix(dest.suffix + ".s37partial")
                with requests.get(url, stream=True, timeout=20, headers={"Accept-Encoding": "identity"}) as response:
                    response.raise_for_status()
                    with partial.open("wb") as f:
                        while True:
                            remaining = 192 * 1024**2 - downloaded
                            if remaining <= 0:
                                raise RuntimeError("acquisition byte ceiling reached")
                            chunk = response.raw.read(min(1024 * 1024, remaining), decode_content=True)
                            if not chunk:
                                break
                            downloaded += len(chunk)
                            RESULT["downloaded_bytes"] = downloaded
                            if downloaded > 192 * 1024**2 or time.monotonic() - START > 120:
                                raise RuntimeError("acquisition ceiling exceeded")
                            f.write(chunk)
                partial.replace(dest)
                save()
            if time.monotonic() - START > 120:
                raise RuntimeError("acquisition time ceiling exceeded")
        digest = hashlib.sha256((folder / "pytorch_model.bin").read_bytes()).hexdigest()
        assert digest == expected, "checkpoint checksum mismatch"
        paths[role] = folder
        RESULT["models"][role] = {"model": name, "revision": rev, "weight_sha256": digest}
    RESULT.update(status="acquired", downloaded_bytes=downloaded,
                  phase="compute", phase_started_epoch=time.time())
    save()
    compute_start = time.monotonic()
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=False)
    RESULT["stage"] = "import_torch"
    save()
    import torch
    RESULT["stage"] = "import_transformers"
    save()
    import transformers
    RESULT["stage"] = "import_auto_classes"
    save()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    RESULT["stage"] = "tokenizer_alignment"
    save()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    RESULT["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    RESULT["execution"] = {"device": "cpu", "dtype": "float32", "attention": "eager",
                           "batch": 8, "threads": 2, "use_cache": False, "chat_template": "none"}
    tokenizer_hashes = [hashlib.sha256((p / "tokenizer.json").read_bytes()).hexdigest()
                        for p in paths.values()]
    assert len(set(tokenizer_hashes)) == 1, "tokenizer definitions differ"
    tok = AutoTokenizer.from_pretrained(paths["donor"], local_files_only=True)
    recipient_tok = AutoTokenizer.from_pretrained(paths["recipient"], local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    verb_ids = [tok.encode(v, add_special_tokens=False) for v in (" was", " were")]
    assert all(len(v) == 1 for v in verb_ids)
    verb_ids = [v[0] for v in verb_ids]
    RESULT["tokenizer_sha256"] = tokenizer_hashes[0]
    rows, lexical, plain = [], [], []
    for family, noun in enumerate(NOUNS):
        for number in (0, 1):
            subject = noun + "s" * number
            plain.append(f"Yesterday, the {subject}")
            for distractor_number in (0, 1):
                distractor = NOUNS[(family + 1) % 4] + "s" * distractor_number
                prefix = f"Yesterday, the {subject} near the {distractor}"
                replacement = NOUNS[(family + 2) % 4] + "s" * number
                lex = f"Yesterday, the {replacement} near the {distractor}"
                subject_ids = tok.encode(" " + subject, add_special_tokens=False)
                lex_ids = tok.encode(" " + replacement, add_special_tokens=False)
                pre_ids = tok.encode("Yesterday, the", add_special_tokens=False)
                ids = tok.encode(prefix, add_special_tokens=False)
                other = tok.encode(lex, add_special_tokens=False)
                assert len(subject_ids) == len(lex_ids) == 1
                pos = len(pre_ids)
                assert ids[:pos] == pre_ids and ids[pos] == subject_ids[0]
                assert other[pos] == lex_ids[0] and len(ids) == len(other)
                distractor_ids = tok.encode(" " + distractor, add_special_tokens=False)
                assert len(distractor_ids) == 1 and ids[-1] == distractor_ids[0]
                assert [j for j,(a,b) in enumerate(zip(ids,other)) if a != b] == [pos]
                rows.append({"id": len(rows), "family": noun, "number": number,
                             "distractor_number": distractor_number, "prefix": prefix,
                             "subject_position": pos, "tokens": ids})
                lexical.append(lex)
    for i, row in enumerate(rows):
        assert len(row["tokens"]) == len(rows[i ^ 2]["tokens"])
        assert row["subject_position"] == rows[i ^ 2]["subject_position"]
        assert [j for j,(a,b) in enumerate(zip(row["tokens"], rows[i ^ 2]["tokens"])) if a != b] == [row["subject_position"]]
    for text in [r["prefix"] for r in rows] + lexical + plain + [" was", " were"]:
        assert tok.encode(text, add_special_tokens=False) == recipient_tok.encode(text, add_special_tokens=False)
    RESULT["panel"] = rows
    RESULT["lexical_sources"] = lexical
    RESULT["plain_prefixes"] = plain
    peak_rss = 0

    def check():
        nonlocal peak_rss
        memory = psutil.Process().memory_info()
        peak_rss = max(peak_rss, getattr(memory, "peak_wset", memory.rss))
        if peak_rss > 3 * 1024**3 or time.monotonic() - compute_start > 180:
            raise RuntimeError("compute ceiling exceeded")
        assert RESULT["forwards"] <= 16 and RESULT["row_presentations"] <= 128

    for role in ("donor", "recipient"):
        check()
        RESULT["stage"] = "load_" + role
        save()
        model, loading = AutoModelForCausalLM.from_pretrained(
            paths[role], local_files_only=True, dtype=torch.float32,
            attn_implementation="eager", output_loading_info=True, weights_only=True)
        RESULT["stage"] = "score_" + role
        save()
        model.eval()
        assert all(p.device.type == "cpu" for p in model.parameters())
        assert not loading.get("missing_keys") and not loading.get("mismatched_keys"), loading
        RESULT["models"][role].update(parameters=sum(p.numel() for p in model.parameters()),
                                     loading_info=loading)

        def score(texts, patch=None, capture=False, offset=0, label="base"):
            scores, bank = [], []
            for start in range(0, len(texts), 8):
                check()
                inputs = tok(texts[start:start + 8], padding=True, return_tensors="pt")
                recorded = []

                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if capture:
                        recorded.append(h.detach().clone())
                    if patch is None:
                        return output
                    changed = h.clone()
                    for local in range(len(changed)):
                        idx = offset + start + local
                        changed[local, rows[idx]["subject_position"]] = patch[idx]
                    return (changed,) + output[1:] if isinstance(output, tuple) else changed

                handle = model.transformer.h[0].register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        logits = model(**inputs, use_cache=False).logits
                    RESULT["forwards"] += 1
                    RESULT["row_presentations"] += len(inputs["input_ids"])
                finally:
                    handle.remove()
                last = logits[torch.arange(len(logits)), inputs["attention_mask"].sum(1) - 1]
                selected = last[:, verb_ids]
                margin = selected[:, 1] - selected[:, 0]
                mass = torch.exp(torch.logsumexp(selected, 1) - torch.logsumexp(last, 1))
                for m, prob in zip(margin.tolist(), mass.tolist()):
                    scores.append({"margin_were_over_was": m, "p_were_given_pair":
                                   float(torch.sigmoid(torch.tensor(m))), "verb_mass": prob,
                                   "choice": " were" if m > 0 else " was"})
                if capture:
                    assert len(recorded) == 1
                    for local, state in enumerate(recorded[0]):
                        bank.append(state[rows[start + local]["subject_position"]].clone())
                check()
                RESULT.setdefault("completed_batches", []).append({"role": role, "arm": label,
                    "start": start, "scores": scores[-len(inputs["input_ids"]):],
                    "compute_seconds": time.monotonic() - compute_start, "peak_rss_bytes": peak_rss})
                save()
            return scores, bank

        texts = [r["prefix"] for r in rows]
        base, bank = score(texts, capture=(role == "donor"))
        old, _ = score(plain, label="plain")
        repeat, _ = score(texts[:8], label="repeat")
        result = {"base": base, "plain": old, "repeat": repeat}
        if role == "donor":
            lex_source, lex_bank = score(lexical, capture=True, label="lexical_source")
            identity, _ = score(texts, patch=bank, label="identity")
            changed, _ = score(texts, patch=[bank[i ^ 2] for i in range(16)], label="opposite_patch")
            lex_patch, _ = score(texts, patch=lex_bank, label="lexical_patch")
            result.update(lexical_source=lex_source, identity=identity,
                          opposite_patch=changed, lexical_patch=lex_patch)
        RESULT[role] = result
        RESULT.update(status=role + "_scored", peak_rss_bytes=peak_rss,
                      compute_seconds=time.monotonic() - compute_start)
        save()
        del model
        import gc
        gc.collect()
    donor, recipient = RESULT["donor"], RESULT["recipient"]
    correct = lambda scores: [int(s["choice"] == (" were" if r["number"] else " was"))
                              for r, s in zip(rows, scores)]
    dc, rc = correct(donor["base"]), correct(recipient["base"])
    mean = lambda xs: sum(xs) / len(xs)
    patch, lexical_effect, natural = [], [], []
    for i, row in enumerate(rows):
        sign = 1 if row["number"] == 0 else -1
        baseline = donor["base"][i]["margin_were_over_was"]
        patch.append(sign * (donor["opposite_patch"][i]["margin_were_over_was"] - baseline))
        lexical_effect.append(sign * (donor["lexical_patch"][i]["margin_were_over_was"] - baseline))
        natural.append(sign * (donor["base"][i ^ 2]["margin_were_over_was"] - baseline))
    drift = {}
    for role in ("donor", "recipient"):
        drift[role + "_repeat"] = max(abs(a["margin_were_over_was"] - b["margin_were_over_was"])
                                      for a, b in zip(RESULT[role]["base"], RESULT[role]["repeat"]))
    drift["identity"] = max(abs(a["margin_were_over_was"] - b["margin_were_over_was"])
                            for a, b in zip(donor["base"], donor["identity"]))
    cells = {f"subject{s}_distractor{d}": sum(dc[i] for i, r in enumerate(rows)
                                           if r["number"] == s and r["distractor_number"] == d)
             for s in (0, 1) for d in (0, 1)}
    directions = {str(s): mean([patch[i] for i, r in enumerate(rows) if r["number"] == s])
                  for s in (0, 1)}
    gap = sum(a and not b for a, b in zip(dc, rc))
    old_correct = sum(s["choice"] == (" were" if i % 2 else " was")
                      for i, s in enumerate(recipient["plain"]))
    verb_mass = mean([s["verb_mass"] for s in donor["base"]])
    specificity = mean([a - b for a, b in zip(patch, lexical_effect)])
    gates = {"mechanical": max(drift.values()) <= 1e-5,
             "donor_competent": sum(dc) >= 12 and min(cells.values()) >= 3 and verb_mass >= .05,
             "recipient_old_use": old_correct >= 6, "recipient_headroom": gap >= 2,
             "intervention": mean(patch) >= .5 and min(directions.values()) >= .25,
             "specificity": specificity >= .25}
    RESULT["summary"] = {"donor_correct": sum(dc), "recipient_correct": sum(rc),
                         "source_following_correct_out_of_16": sum(s["choice"] == (" was" if r["number"] else " were") for r,s in zip(rows,donor["opposite_patch"])),
                         "donor_cells_out_of_4": cells, "recipient_plain_correct_out_of_8": old_correct,
                         "donor_correct_recipient_wrong": gap, "mean_donor_verb_mass": verb_mass,
                         "patch_effect": mean(patch), "direction_effects": directions,
                         "lexical_effect": mean(lexical_effect), "specificity": specificity,
                         "natural_counterfactual_effect": mean(natural), "drift": drift,
                         "family_patch_means": {noun: mean(patch[4*i:4*i+4]) for i, noun in enumerate(NOUNS)},
                         "family_specificity_means": {noun: mean([patch[j]-lexical_effect[j]
                             for j in range(4*i,4*i+4)]) for i,noun in enumerate(NOUNS)},
                         "gates": gates, "fund_teaching_comparison": all(gates.values())}
    faulthandler.cancel_dump_traceback_later()
    RESULT["status"] = "complete"
    RESULT["interpretability"] = "valid local panel" if gates["mechanical"] else "inconclusive mechanical failure"
    save()


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit("Result already exists; never silently overwrite a prior attempt")
    import sys
    if "--worker" not in sys.argv:
        import subprocess
        import psutil
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"])
        process = psutil.Process(child.pid)
        reason = None
        peak = 0
        while child.poll() is None:
            try:
                info = process.memory_info()
                peak = max(peak, getattr(info, "peak_wset", info.rss))
                current = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
                phase = current.get("phase", "download")
                elapsed = time.time() - current.get("phase_started_epoch", time.time())
                if peak > 3 * 1024**3:
                    reason = "supervisor memory ceiling"
                elif elapsed > (120 if phase == "download" else 180):
                    reason = "supervisor phase deadline: " + phase
                elif time.monotonic() - START > 305:
                    reason = "supervisor total deadline"
                if reason:
                    child.kill()
                    child.wait()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.1)
        child.wait()
        current = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        current["supervisor"] = {"peak_rss_bytes": peak, "poll_seconds": .1,
                                 "exit_code": child.returncode, "abort_reason": reason}
        if reason or child.returncode:
            current.update(status="inconclusive_execution_failure", supervisor_error=reason)
        RESULT.clear()
        RESULT.update(current)
        save()
        raise SystemExit(child.returncode if not reason else 1)
    try:
        main()
    except Exception as error:
        RESULT.update(status="inconclusive_execution_failure", error=repr(error))
        save()
        raise
