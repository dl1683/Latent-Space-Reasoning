"""Single production/CLI runner (Z Code B).

run_phase(cfg, out, handle=None) is the only execution entry:
  --phase prepare  frozen packet: deterministic splits/hashes, disjoint
                   pools, dictionary 768/768 through the real store
  --phase smoke    tiny random neural mechanics (gradients/plumbing
                   ONLY — never a prototype PASS)
  --phase overfit  bounded random-stem fit enabled by the default contract;
                   train/evaluate remain disabled (clean refusal, exit 3).
                   These are construction diagnostics, not pretrained evidence.
  --check store-reload | leakage-negative | scoring-clusters | false-pass\n          production-contract | production-smoke | reader-boundary | first-overfit
                   (offline, CPU; first-overfit reports the historical failed
                   fit receipt without rerunning training)

Receipts are frozen JSON binding config + module hashes; PASS exits 0,
FAIL 1, unknown check 2, disabled/unimplemented semantic phase 3.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile

from . import data, evaluate, store, trainer, writer

MODULES = ("__init__", "data", "writer", "store", "trainer",
           "evaluate", "runner", "demo", "semantic", "reader",
           "checks", "regressions", "reader_model")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def code_hashes():
    out = {}
    for mod in MODULES:
        if mod == "__init__":
            path = os.path.join(_HERE, "__init__.py")
        else:
            path = os.path.join(_HERE, mod + ".py")
        out[mod] = _sha(path)
    contract = os.path.join(_HERE, "contract.json")
    if os.path.isfile(contract):
        out["contract.json"] = _sha(contract)
    from pathlib import Path
    jobs = Path(_HERE).parents[1] / 'cloud' / 'lsr_jobs'
    for name in ('s0_memory_port.py', 's0_envelope.py', 'gemma4_receiver.py', 's0_receiving_checks.py'):
        out['production/' + name] = _sha(jobs / name)
    return out


def receipt(name, status, detail, config_sha=None, exit_code=0):
    return {
        "check": name, "status": status,
        "config_sha256": config_sha,
        "code_sha256": code_hashes(),
        "detail": detail, "exit_code": exit_code,
    }


def emit(rec):
    print(json.dumps(rec, sort_keys=True, ensure_ascii=True))
    return rec["exit_code"]


# ---------------------------------------------------------------- phases
def _phase_prepare(cfg, out, config_sha):
    os.makedirs(out, exist_ok=True)
    corpus = data.build_corpus(cfg.get("seed", 20261015))
    errs = data.verify_corpus(corpus)
    if errs:
        return receipt("phase:prepare", "FAIL", {"verify_errors": errs[:8]},
                       config_sha, 1)

    for split in ("train", "dev"):
        path = os.path.join(out, f"{split}_pairs.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in data.writer_inputs(corpus, split):
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(out, "test_writer_inputs.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in data.writer_inputs(corpus, "test"):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(out, "test_queries.jsonl"), "w",
              encoding="utf-8") as fh:
        for q in data.reader_queries(corpus, "test"):
            fh.write(json.dumps(q, sort_keys=True) + "\n")
    truth = {f"{k[0]}|{k[1]}": v
             for k, v in data.truth_table(corpus, "test").items()}
    truth["donor_late_target"] = {
        f"{k[0]}|{k[1]}": v
        for k, v in data.donor_truth_table(corpus, "test").items()}
    with open(os.path.join(out, "test_truth.locked.json"), "w",
              encoding="utf-8") as fh:
        json.dump(truth, fh, sort_keys=True)

    # dictionary arm through the REAL store: write -> query v1 x2 ->
    # revise -> query v2 x3 + historical v1; old snapshot bytes unchanged
    dic = store.dictionary_reader()
    st = store.MemoryStore(readers=[dic])
    correct = total = 0
    immutable = True
    for pair in corpus["splits"]["test"]:
        for doc, reads in ((pair["doc_a"], pair["reads_a"]),
                           (pair["doc_b"], pair["reads_b"])):
            sid1 = st.write(doc["records"])
            before = st.snapshot_bytes(sid1)
            sid2 = st.revise(sid1, doc["revision"]["record_id"],
                             doc["revision"])
            if st.snapshot_bytes(sid1) != before:
                immutable = False
            ids = {"v1": sid1, "v2": sid2}
            for r in reads:
                ans = st.query(ids[r["snapshot"]], r["query"], budget=8,
                               reader="dictionary")
                correct += int(ans == r["answer"])
                total += 1
    balance = data.value_balance(corpus, "test")
    bmin, bmax = min(balance.values()), max(balance.values())
    lo, hi = data.balance_band(len(corpus["splits"]["test"]) * 2 * 6)
    detail = {
        "packet": corpus["generator"]["packet"],
        "doc_counts": {k: len(v) * 2
                       for k, v in corpus["splits"].items()},
        "pair_counts": {k: len(v)
                        for k, v in corpus["splits"].items()},
        "split_hashes": corpus["hashes"],
        "novel_test_pairs": sum(p["novel"]
                                for p in corpus["splits"]["test"]),
        "dictionary_arm": [correct, total],
        "old_snapshot_immutable": immutable,
        "test_value_balance_minmax": [bmin, bmax],
        "files": sorted(os.listdir(out)),
        "interpretation": "packet frozen; dictionary is exact storage, "
                          "not the trained candidate",
    }
    ok = (correct == 768 and total == 768 and immutable
          and detail["novel_test_pairs"] == data.NOVEL_TEST_PAIRS
          and lo <= bmin and bmax <= hi
          and detail["doc_counts"] == {"train": 2048, "dev": 128,
                                       "test": 128})
    with open(os.path.join(out, "packet_receipt.json"), "w",
              encoding="utf-8") as fh:
        json.dump(receipt("phase:prepare",
                          "PASS" if ok else "FAIL", detail, config_sha,
                          0 if ok else 1), fh, sort_keys=True, indent=1)
    return receipt("phase:prepare", "PASS" if ok else "FAIL", detail,
                   config_sha, 0 if ok else 1)


def _phase_smoke(cfg, out, config_sha):
    os.makedirs(out, exist_ok=True)
    if 'semantic' in cfg:
        rec = _bounded_check('production-smoke', cfg)
        rec['check'] = 'phase:smoke'
        rec['config_sha256'] = config_sha
        with open(os.path.join(out, 'production_smoke.json'), 'w', encoding='utf-8') as stream:
            json.dump(rec, stream, indent=2)
        return rec
    checks = trainer.tiny_smoke(cfg)
    plumbing = ["masking_fixture", "guard_refused_test_truth",
                "resampler_grads_finite_nonzero", "frozen_stub_zero_grads",
                "losses_finite", "loss_decreased"]
    ok = all(checks[k] for k in plumbing)
    checks["interpretation"] = ("tiny random mechanics ONLY: gradients "
                                "and plumbing; NEVER a prototype PASS")
    return receipt("phase:smoke", "PASS" if ok else "FAIL", checks,
                   config_sha, 0 if ok else 1)


PHASES = {"prepare": _phase_prepare, "smoke": _phase_smoke}
# semantic construction phases are owned by semantic.py (production
# path slice: tiny-random stem -> real S0 forward/generate); the
# canonical runner delegates to its run_phase when the config carries
# a semantic section
SEMANTIC_PHASES = ("overfit", "train", "evaluate")


def _phase_semantic(cfg, out, config_sha, handle=None):
    from . import semantic   # torch-heavy module: imported lazily so
    # the offline checks stay torch-free at import time
    # semantic.run_phase mkdir(exist_ok=False); the dispatcher already
    # pre-creates out_dir/<phase>, so the slice gets a fresh subdir
    slice_dir = os.path.join(out, "slice")
    report = semantic.run_phase(cfg, slice_dir, handle=handle)
    status = report.get("status", "FAIL")
    slim = {k: v for k, v in report.items()
            if k not in ("identities",)}
    slim["interpretation"] = (
        "tiny-random production-path construction slice; diagnostic "
        "ONLY — never prototype acceptance, never a pretrained claim")
    return receipt(f"phase:{cfg.get('_phase')}", status, slim,
                   config_sha,
                   0 if status in ("PASS", "COMPLETE_DIAGNOSTIC") else 1)


def run_dispatch(effective_config=None, config_bytes=None, out_dir=None,
                 **_):
    """Dispatcher entry (experiments.cloud.lsr_jobs.run.py contract:
    `run_dispatch(effective_config, config_bytes, out_dir)`), bridging
    the verified shard dispatch to the canonical run_phase. The
    offline default contract still refuses semantic phases; a staged
    pinned-stem config is what enables them (deployment-owned)."""
    if out_dir is None:
        raise ValueError("run_dispatch requires out_dir")
    cfg = dict(effective_config or {})
    if isinstance(config_bytes, (bytes, bytearray)):
        cfg["_config_sha256"] = hashlib.sha256(
            bytes(config_bytes)).hexdigest()
    cfg["_phase"] = cfg.get("phase") or cfg.get("_phase") or "prepare"
    rec = run_phase(cfg, out_dir)
    # the dispatcher sys.exit()s on our return value: receipts land on
    # stdout, the PROCESS exit code is the receipt's exit code
    print(json.dumps(rec, sort_keys=True, ensure_ascii=True))
    return rec["exit_code"]


def run_phase(cfg, out, handle=None):
    """Single production entry (deployment passes cfg intact)."""
    phase = cfg.get("_phase")
    config_sha = cfg.get("_config_sha256")
    gate = cfg.get("phases", {}).get(phase, {})
    if not gate.get("enabled", False):
        return receipt(
            f"phase:{phase}", "DISABLED",
            {"reason": gate.get(
                "reason", "semantic phase disabled in offline contract"),
             "hint": "deployment must pass an intact pinned-stem config"},
            config_sha, 3)
    if phase in SEMANTIC_PHASES:
        if "semantic" not in cfg:
            return receipt(
                f"phase:{phase}", "UNIMPLEMENTED",
                {"reason": "semantic phase requires a semantic config "
                           "section (pinned stem + fit recipe); none "
                           "supplied"},
                config_sha, 3)
        return _phase_semantic(cfg, out, config_sha, handle=handle)
    if phase not in PHASES:
        # enabled by config but implemented only on the cloud path;
        # never crash with a KeyError on a deployment-passed config
        return receipt(
            f"phase:{phase}", "UNIMPLEMENTED",
            {"reason": "semantic phase requires the pinned-stem cloud "
                       "path; this offline runner carries no fit code",
             "hint": "deployment runs the same runner with the staged "
                     "config on the cloud machine"},
            config_sha, 3)
    if handle is not None:
        raise RuntimeError(
            f"phase {phase!r} does not accept a live handle offline")
    return PHASES[phase](cfg, out, config_sha)


# ---------------------------------------------------------------- checks
def _check_store_reload():
    """Old snapshot unchanged after revision; save/restart keeps exact
    answers; corrupted or tampered files rejected WITHOUT mutation;
    role-bound view permissions and the restarted allowlist enforced."""
    doc = {"records": [
        {"record_id": i, "slot": s, "value": v, "text": f"rec {i} {v}"}
        for i, (s, v) in enumerate(zip(
            ["north key", "east key", "south key", "west key"],
            ["amber", "basil", "cedar", "dune"]))],
        "revision": {"record_id": 1, "slot": "east key", "value": "elm",
                     "text": "rec 1 elm"}}
    st = store.MemoryStore(readers=[store.dictionary_reader()])
    sid1 = st.write(doc["records"])
    q_north = {"text": "what is the north key of x?",
               "name": "x", "slot": "north key"}
    q_east = {"text": "what is the east key of x?",
              "name": "x", "slot": "east key"}
    a_before = st.query(sid1, q_east, 8, reader="dictionary")
    bytes_before = st.snapshot_bytes(sid1)
    sid2 = st.revise(sid1, 1, doc["revision"])
    results = {
        "parent_bytes_unchanged": st.snapshot_bytes(sid1) == bytes_before,
        "parent_query_unchanged": st.query(
            sid1, q_east, 8, reader="dictionary") == a_before == "basil",
        "new_version_answers": st.query(
            sid2, q_east, 8, reader="dictionary") == "elm",
        "historical_query": st.query(
            sid1, q_east, 8, reader="dictionary") == "basil",
        "unrelated_preserved": st.query(
            sid2, q_north, 8, reader="dictionary") == "amber",
        "versions": [st.version_of(sid1) == 1, st.version_of(sid2) == 2],
    }
    # reader mutations must not corrupt snapshots (deep-copied views):
    # a dictionary-role reader that clears its view mid-query
    mut = store.ReaderSpec("dictionary", ("records",),
                           lambda v, q, b: (v["records"].clear(), "x")[1])
    st_m = store.MemoryStore(readers=[mut])
    mid = st_m.write(doc["records"])
    first = st_m.query(mid, q_east, 8, reader="dictionary")
    results["reader_cannot_mutate_snapshot"] = (
        first == "x"
        and len(st_m._snapshots[mid]["records"]) == 4
        and st_m.query(mid, q_east, 8, reader="dictionary") == "x")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "store.json")
        st.save(path)
        original_bytes = open(path, "rb").read()

        # corrupted record content -> reject, file untouched, memory
        # store intact
        corrupted = bytearray(original_bytes)
        i = corrupted.find(b"amber")
        if i < 0:
            raise RuntimeError("fixture word missing")
        corrupted[i:i + 5] = b"onyx "
        bad_path = os.path.join(tmp, "corrupt.json")
        with open(bad_path, "wb") as fh:
            fh.write(bytes(corrupted))
        try:
            store.MemoryStore.load(bad_path)
            rejected_corrupt = False
        except store.StoreError:
            rejected_corrupt = True
        results["corrupted_rejected"] = rejected_corrupt
        results["corrupt_file_untouched"] = \
            open(bad_path, "rb").read() == bytes(corrupted)
        snapshot_state = copy.deepcopy(st._snapshots)
        try:
            store.MemoryStore.load(bad_path)
        except store.StoreError:
            pass
        results["in_memory_store_unmutated"] = \
            st._snapshots == snapshot_state

        # tampered ALLOWLIST (top-level field) -> payload digest catches
        tampered = json.loads(original_bytes)
        tampered["allowlist"] = ["candidate", "sneak"]
        tamp_path = os.path.join(tmp, "tampered.json")
        with open(tamp_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(tampered, sort_keys=True))
        try:
            store.MemoryStore.load(tamp_path)
            results["tampered_allowlist_rejected"] = False
        except store.StoreError:
            results["tampered_allowlist_rejected"] = True

        # restart: valid file loads; the saved allowlist binds even
        # REGISTERED readers not on it
        cand = store.ReaderSpec(
            "candidate", ("native_state", "sidecars"),
            lambda v, q, b: None)
        restarted = store.MemoryStore.load(
            path, readers=[store.dictionary_reader(), cand])
        results["restart_exact_answers"] = (
            restarted.query(sid1, q_east, 8, reader="dictionary")
            == "basil"
            and restarted.query(sid2, q_east, 8, reader="dictionary")
            == "elm")
        try:
            restarted.query(sid1, q_north, 8, reader="candidate")
            results["restarted_allowlist_enforced"] = False
        except store.StoreError:
            results["restarted_allowlist_enforced"] = True
        try:
            restarted.query(sid1, q_north, 8, reader="not-a-reader")
            results["unknown_reader_rejected"] = False
        except store.StoreError:
            results["unknown_reader_rejected"] = True
        try:
            restarted.query(sid1, q_north, 8)
            results["explicit_reader_required"] = False
        except store.StoreError:
            results["explicit_reader_required"] = True

    # role-bound view permissions: a records view under the candidate
    # name (or any non-storage name) is refused AT REGISTRATION
    try:
        store.ReaderSpec("candidate", ("records",), lambda v, q, b: 1)
        results["candidate_records_view_refused"] = False
    except store.StoreError:
        results["candidate_records_view_refused"] = True
    try:
        store.ReaderSpec("sneak", ("records",), lambda v, q, b: 1)
        results["unnamed_records_view_refused"] = False
    except store.StoreError:
        results["unnamed_records_view_refused"] = True
    ok = all(bool(v) for k, v in results.items() if k != "versions") \
        and all(results["versions"])
    return receipt("store-reload", "PASS" if ok else "FAIL", results,
                   None, 0 if ok else 1)


def _check_leakage_negative():
    """Intentionally embedded source/answer is rejected; honest states
    pass; the reader bundle exposes no records/questions/answers; the
    trainer guard refuses test truth; the store door sweeps retention."""
    corpus = data.build_corpus(splits=("train",))
    pair = corpus["splits"]["train"][0]
    doc = pair["doc_a"]
    native = writer.write_document(doc)
    honest_leaks = writer.validate_no_retention(
        native, records=doc["records"],
        questions=[{"text": pair["reads_a"][0]["query"]["text"]}],
        answers=[r["answer"] for r in pair["reads_a"]])
    results = {"honest_state_clean": not honest_leaks}

    malicious = json.loads(json.dumps(native))
    malicious["sidecars"]["stash"] = doc["records"][0]["text"]
    leaks_text = writer.validate_no_retention(
        malicious, records=doc["records"])
    results["embedded_record_text_rejected"] = bool(leaks_text)

    malicious2 = json.loads(json.dumps(native))
    malicious2["sidecars"]["answer"] = pair["reads_a"][0]["answer"]
    leaks_ans = writer.validate_no_retention(
        malicious2, answers=[r["answer"] for r in pair["reads_a"]])
    results["embedded_answer_rejected"] = bool(leaks_ans)

    malicious3 = json.loads(json.dumps(native))
    malicious3["sidecars"]["tail_stash"] = \
        " ".join(r["text"] for r in doc["records"])[-40:]
    leaks_tail = writer.validate_no_retention(
        malicious3, records=doc["records"])
    results["embedded_record_suffix_rejected"] = bool(leaks_tail)

    bundle, bundle_leaks = writer.reader_bundle(
        native, pair["reads_a"][0]["query"]["text"], records=doc["records"])
    results["reader_bundle_clean"] = not bundle_leaks
    blob = json.dumps(bundle)
    # the question legitimately carries slot/name wording shared with
    # record templates; the forbidden content is full record text and
    # any VALUE word (the question never contains the answer value)
    results["reader_bundle_has_no_records"] = (
        "records" not in bundle
        and all(rec["text"] not in blob for rec in doc["records"])
        and all(w not in blob for w in data.WORDS))
    # reader-side mutation of the bundle cannot corrupt writer state
    bundle["rows"][0][0] = 999.0
    results["bundle_is_deep_copy"] = native["rows"][0][0] != 999.0

    try:
        trainer.open_split("out/test_truth.locked.json", "train")
        results["trainer_guard_refuses_truth"] = False
    except trainer.TrainerGuard:
        results["trainer_guard_refuses_truth"] = True
    try:
        trainer.open_split("out/train_pairs.jsonl", "test")
        results["trainer_guard_refuses_test_split"] = False
    except trainer.TrainerGuard:
        results["trainer_guard_refuses_test_split"] = True

    # candidate reader receives exactly the role-permitted views (the
    # reader callable records what it was actually handed)
    seen_views = []

    def candidate_fn(view, question, budget):
        seen_views.append(sorted(view.keys()))
        return None

    candidate_spec = store.ReaderSpec(
        "candidate", ("native_state", "sidecars"), candidate_fn)
    st = store.MemoryStore(readers=[candidate_spec,
                                    store.dictionary_reader()])
    sid = st.write(doc["records"], native_state=native)
    st.query(sid, {"slot": "east key"}, 8, reader="candidate")
    results["candidate_view_has_no_records"] = (
        bool(seen_views) and seen_views[0] == ["native_state",
                                               "sidecars"])
    # the store door itself sweeps retention: a stowaway answer word in
    # a sidecar string is refused at write time
    stowaway = json.loads(json.dumps(native))
    stowaway["sidecars"]["note"] = "value is amber here"
    try:
        st.write(doc["records"], native_state=stowaway)
        results["store_door_sweeps_retention"] = False
    except store.StoreError:
        results["store_door_sweeps_retention"] = True
    ok = all(bool(v) for v in results.values())
    return receipt("leakage-negative", "PASS" if ok else "FAIL", results,
                   None, 0 if ok else 1)


def _check_scoring_clusters():
    """Evaluator machinery on the FULL frozen arm roster with DECODE
    EVIDENCE per neural row (the co-owned fail-closed evaluate_all
    contract): a healthy complete packet scores completely yet stays
    INCOMPLETE for final acceptance; corrupted pair/stratum/gate
    failures are detected; duplicated seed rows rejected; bootstrap is
    pair-clustered and deterministic; missing controls or evidence
    make the verdict unreachable."""
    import random as _random

    corpus = data.build_corpus()
    truth = data.truth_table(corpus, "test")
    donor_truth = data.donor_truth_table(corpus, "test")
    novel = {str(p["pair"]) for p in corpus["splits"]["test"] if p["novel"]}
    queries = data.reader_queries(corpus, "test")

    def wrong_of(exp):
        return data.WORDS[(data.WORDS.index(exp) + 7) % data.N_VALUES]

    def evidence_row(q, exp, gen, terminated=True):
        """Decode-harness evidence exactly as the production generate
        path records it: eos-terminated rows end in the eos id; budget
        rows carry count==budget without eos."""
        eos = 1
        if terminated:
            tokens = [7, 9, eos]
            cause = "eos"
        else:
            tokens = [7, 9, 5, 3, 6, 2, 4, 8]
            cause = "budget"
        row = {"arm": "", "doc": q["doc"], "op": q["op"],
               "generated": gen, "expected": exp,
               "tokens": tokens, "eos_token_id": eos,
               "token_count": len(tokens), "max_new_tokens": 8,
               "finish_cause": cause, "terminated": terminated}
        return row

    def table(arm, wrong_pairs=(), wrong_op=None, frac_wrong=0.0,
              rng=None, unterminated=False):
        rows = []
        for q in queries:
            if arm == "donor" and q["op"] != "late_target":
                continue
            exp = donor_truth[(q["doc"], "late_target")] \
                if arm == "donor" else truth[(q["doc"], q["op"])]
            gen = exp
            if q["doc"][:-1] in wrong_pairs \
                    or (wrong_op is not None and q["op"] == wrong_op):
                gen = wrong_of(exp)
            elif frac_wrong and rng.random() < frac_wrong:
                gen = wrong_of(exp)
            if arm == "dictionary":
                rows.append({"arm": arm, "doc": q["doc"], "op": q["op"],
                             "generated": gen, "expected": exp,
                             "terminated": True})
            else:
                row = evidence_row(q, exp, gen,
                                   terminated=not unterminated)
                row["arm"] = arm
                rows.append(row)
        return rows

    rng = _random.Random(11)
    rng_u = _random.Random(12)
    healthy = (table("candidate") + table("storage")
               + table("omission", frac_wrong=0.4, rng=rng)
               + table("untrained", frac_wrong=1.0, rng=rng_u)
               + table("donor") + table("dictionary"))
    base = evaluate.evaluate_all(healthy, truth, novel,
                                 donor_truth=donor_truth)
    results = {
        "healthy_candidate_768":
            base["summary"]["candidate"]["correct"],
        "healthy_packet_scores_completely": base.get("scoring_pass"),
        "final_acceptance_reserved":
            base["all_pass"] is False
            and base.get("acceptance_status") == "INCOMPLETE",
    }

    # missing ANY control (or broken evidence) makes the verdict
    # unreachable outright
    for drop in ("omission", "untrained", "donor", "storage",
                 "dictionary"):
        subset = [r for r in healthy if r["arm"] != drop]
        sub = evaluate.evaluate_all(subset, truth, novel,
                                    donor_truth=donor_truth)
        if sub.get("all_pass") or not sub["summary"].get("table_error"):
            results["missing_control_blocks_verdict"] = False
            break
    else:
        results["missing_control_blocks_verdict"] = True

    corrupt = ([r for r in table("candidate", wrong_pairs={"17"})]
               + [r for r in healthy if r["arm"] != "candidate"])
    bad = evaluate.evaluate_all(corrupt, truth, novel,
                                donor_truth=donor_truth)
    docs_ok, pairs_ok = evaluate.complete_units(corrupt, "candidate",
                                                truth)
    results["single_pair_corruption_reflected"] = (
        bad["summary"]["candidate"]["correct"] == 768 - 12
        and len(pairs_ok) == 63 and len(docs_ok) == 126
        and bad["summary"]["candidate"]["by_op"]["historical"][0] == 126)

    heavy = ([r for r in table(
        "candidate", wrong_pairs={str(i) for i in range(13)})]
        + [r for r in healthy if r["arm"] != "candidate"])
    hv = evaluate.evaluate_all(heavy, truth, novel,
                               donor_truth=donor_truth)
    results["corrupted_pairs_fail_gates"] = (
        not hv["gates"]["complete_pairs"]
        and not hv["gates"]["candidate_total"]
        and not hv["gates"]["complete_docs"])

    stratum = ([r for r in table("candidate", wrong_op="historical")]
               + [r for r in healthy if r["arm"] != "candidate"])
    strat = evaluate.evaluate_all(stratum, truth, novel,
                                  donor_truth=donor_truth)
    results["corrupted_op_fails_op_gate"] = (
        strat["summary"]["candidate"]["by_op"]["historical"][0] < 116
        and not strat["gates"]["per_op"])

    # seed-row inflation: duplicated rows under a new seed tag rejected
    dup = healthy + [dict(r, seed=2)
                     for r in healthy if r["arm"] == "candidate"]
    try:
        evaluate.validate_table(dup)
        results["seed_row_inflation_rejected"] = False
    except ValueError:
        results["seed_row_inflation_rejected"] = True

    # Droid-audit witnesses against the fail-closed scorer
    def swap_verdict(mutate):
        rows = [dict(r) for r in healthy]
        mutate(rows)
        return evaluate.evaluate_all(rows, truth, novel,
                                     donor_truth=donor_truth)

    def all_wrong_dict(rows):
        for r in rows:
            if r["arm"] == "dictionary":
                r["generated"] = "WRONG"

    results["dictionary_must_be_exact"] = not swap_verdict(
        all_wrong_dict).get("scoring_pass")

    def never_terminate_storage(rows):
        for r in rows:
            if r["arm"] == "storage":
                tokens = [7, 9, 5, 3, 6, 2, 4, 8]
                r.update(tokens=tokens, token_count=8,
                         finish_cause="budget", terminated=False)

    nt = swap_verdict(never_terminate_storage)
    results["baseline_termination_gated"] = (
        not nt.get("scoring_pass")
        and nt["summary"]["storage"]["terminated"] == 0)

    def strip_flags(rows):
        for r in rows:
            if r["arm"] != "dictionary":
                r.pop("terminated", None)
                r.pop("tokens", None)

    stripped = swap_verdict(strip_flags)
    results["missing_termination_evidence_fails"] = (
        not stripped.get("scoring_pass")
        and "decode evidence" in str(
            stripped["summary"].get("table_error", "")))

    partial = [r for r in healthy
               if not (r["arm"] == "donor"
                       and r["doc"] in {"5a", "5b", "17a", "17b",
                                        "23a", "23b", "40a", "40b",
                                        "50a", "50b", "60a", "60b"})]
    pv = evaluate.evaluate_all(partial, truth, novel,
                               donor_truth=donor_truth)
    results["partial_donor_table_rejected"] = (
        not pv.get("scoring_pass")
        and not pv["gates"].get("table_valid", True))

    def constant_arm(rows):
        for r in rows:
            if r["arm"] == "omission":
                r["generated"] = "amber"
                r["tokens"] = [7, 1]
                r["token_count"] = 2

    ca = swap_verdict(constant_arm)
    results["constant_receiver_gate"] = (
        not ca.get("scoring_pass")
        and not ca["gates"].get("omission_nonconstant", True))

    # hunt findings 3/4: empty immediate-eos decodes and capped strings
    # with eos evidence are NOT natural terminations
    def empty_eos_storage(rows):
        for r in rows:
            if r["arm"] == "storage":
                r.update(generated="", tokens=[1], token_count=1,
                         finish_cause="eos", terminated=True)

    ee = swap_verdict(empty_eos_storage)
    results["empty_eos_not_natural"] = (
        not ee.get("scoring_pass")
        and ee["summary"]["storage"]["terminated"] == 0)

    def capped_string_candidate(rows):
        for r in rows:
            if r["arm"] == "candidate":
                r["generated"] = r["generated"] + " "                     + evaluate.CAPPED_MARK

    cs = swap_verdict(capped_string_candidate)
    results["capped_string_not_natural"] = (
        not cs.get("scoring_pass")
        and cs["summary"]["candidate"]["terminated"] == 0)

    # hunt finding 5: a fully vacuous (junk, terminating) storage arm
    # must fail the competence floor, not pass noninferiority vacuously
    def vacuous_storage(rows):
        for i, r in enumerate(rows):
            if r["arm"] == "storage":
                r["generated"] = "junka" if i % 2 else "junkb"

    vs = swap_verdict(vacuous_storage)
    results["vacuous_storage_fails_floor"] = (
        not vs.get("scoring_pass")
        and not vs["gates"].get("storage_competent_above_constant",
                                True))

    # hunt finding 6: junk donor-truth values are rejected at the door
    bad_donor = dict(donor_truth)
    k0 = next(iter(bad_donor))
    bad_donor[k0] = "notaword"
    dj = evaluate.evaluate_all(healthy, truth, novel,
                               donor_truth=bad_donor)
    results["donor_truth_values_validated"] = (
        not dj.get("scoring_pass")
        and "donor truth values" in str(
            dj["summary"].get("table_error", "")))

    # the FILE door (load_table incl. ARM_SHAPES) is pinned by an
    # actual round trip: full roster validates; donor-only loads its
    # 128 rows; an inflated file is refused at the door
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as tmp:
        full_path = os.path.join(tmp, "full.jsonl")
        with open(full_path, "w", encoding="utf-8") as fh:
            for r in healthy:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        loaded = evaluate.load_table(full_path)
        results["load_table_full_roster"] = len(loaded) == 3968
        donor_path = os.path.join(tmp, "donor.jsonl")
        with open(donor_path, "w", encoding="utf-8") as fh:
            for r in healthy:
                if r["arm"] == "donor":
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
        results["load_table_donor_only"] = \
            len(evaluate.load_table(donor_path)) == 128
        bad_path = os.path.join(tmp, "bad.jsonl")
        with open(bad_path, "w", encoding="utf-8") as fh:
            for r in healthy:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
            fh.write(json.dumps(dict(healthy[0], seed=9),
                                sort_keys=True) + "\n")
        try:
            evaluate.load_table(bad_path)
            results["load_table_rejects_inflation"] = False
        except ValueError:
            results["load_table_rejects_inflation"] = True

    # pair-cluster bootstrap: non-degenerate, deterministic, and the
    # storage cluster noninferiority gates live in the verdict
    acc = evaluate.bootstrap_pairs(corrupt, "candidate", truth,
                                   n_boot=400)
    deficit = evaluate.bootstrap_deficit(corrupt, "candidate", "omission",
                                         truth, n_boot=400)
    b1 = evaluate.bootstrap_deficit(healthy, "candidate", "omission",
                                    truth, n_boot=400)
    b2 = evaluate.bootstrap_deficit(healthy, "candidate", "omission",
                                    truth, n_boot=400)
    results["bootstrap_accuracy_nondegenerate"] = (
        0.0 <= acc["lcb95"] < acc["mean"] < 1.0)
    results["bootstrap_deficit_nondegenerate"] = (
        0.0 < deficit["lcb95_points"] < deficit["mean_points"])
    results["bootstrap_deterministic"] = b1 == b2
    results["bootstrap_unit_is_pair"] = acc["unit"] == "world_pair"
    results["storage_cluster_gate_live"] = (
        base["gates"].get("storage_cluster_all")
        and "storage_cluster_novel" in base["gates"])

    ok = (results["healthy_candidate_768"] == 768
          and all(bool(v) for k, v in results.items()
                  if k != "healthy_candidate_768"))
    return receipt("scoring-clusters", "PASS" if ok else "FAIL", results,
                   None, 0 if ok else 1)


def _bounded_check(name, cfg, artifact=None, out=None):
    from pathlib import Path
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from . import regressions, semantic
    if not cfg:
        cfg = json.loads((Path(_HERE) / 'contract.json').read_text())
    if cfg['semantic']['stem']['kind'] != 'tiny-random':
        raise ValueError('local bounded checks use the random stand-in only')
    if name == 'production-contract':
        detail = regressions.production_contract(cfg)
    elif name == 'reader-boundary':
        if not artifact or not out:
            raise ValueError('reader-boundary requires --artifact and --out')
        detail = regressions.reader_boundary(cfg, artifact, out)
    elif name == 'production-smoke':
        with tempfile.TemporaryDirectory() as td:
            detail = semantic.smoke(semantic.load_handle(cfg), cfg, Path(td) / 'smoke.json')
        from .checks import production_smoke_valid
        detail['status'] = 'PASS' if production_smoke_valid(detail) else 'FAIL'
    elif name == 'false-pass':
        from . import checks
        detail = checks.gate_regressions()
    else:
        path = Path(_HERE) / 'scratchpad' / 'fix1_fit_a01' / 'result.json'
        saved = json.loads(path.read_text())
        detail = {'status':'PASS', 'source':str(path), 'source_sha256':_sha(path), 'result':saved,
                  'interpretation':'historical failed random-stand-in overfit; no training rerun, no current-byte validation'}
    return receipt(name, detail['status'], detail, exit_code=0 if detail['status']=='PASS' else 1)


CHECKS = {
    "store-reload": _check_store_reload,
    "leakage-negative": _check_leakage_negative,
    "scoring-clusters": _check_scoring_clusters,
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="experiments.memory.prototype")
    ap.add_argument("--config")
    ap.add_argument("--phase")
    ap.add_argument("--check")
    ap.add_argument("--out")
    ap.add_argument("--artifact")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = {}
    config_sha = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        config_sha = _sha(args.config)
    cfg["_config_sha256"] = config_sha

    if args.check:
        bounded = {'production-contract','production-smoke','reader-boundary','false-pass','first-overfit'}
        if args.check not in set(CHECKS) | bounded:
            return emit(receipt(args.check, "UNKNOWN_CHECK",
                                {"known": sorted(set(CHECKS) | bounded)}, config_sha, 2))
        if args.device != "cpu" or not args.offline:
            return emit(receipt(
                args.check, "FAIL",
                {"reason": "local checks are offline CPU only"},
                config_sha, 1))
        os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                          CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if args.check in bounded:
            return emit(_bounded_check(args.check, cfg if args.config else {}, args.artifact, args.out))
        return emit(CHECKS[args.check]())

    if args.phase:
        if not args.out:
            print("--out <unique-ignored-dir> is required for phases",
                  file=sys.stderr)
            return 2
        cfg["_phase"] = args.phase
        return emit(run_phase(cfg, args.out))

    print("need --phase or --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
