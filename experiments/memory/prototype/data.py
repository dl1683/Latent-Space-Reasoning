"""Synthetic revision-corpus generator v2 (Z Code B; Droid-audit
packet redesign, 2026-10-15).

Changes from packet v1 (ALLIN_DROID_AUDIT_OUT.md F2/F8/F9/F10):
- SHARED ENTITY IDENTITY per world pair: both documents of a pair
  carry the same name and the same slot layout; only the bindings
  (values) and the replacement values differ. The donor control is
  therefore a genuine content-only counterfactual: hold the recipient
  question fixed and score the DONOR's binding at the recipient's
  queried address (donor_truth_table).
- PRESERVATION TRACE: read 4 ("unrelated_again") re-queries the SAME
  slot as read 2 ("unrelated") across the revision; read 6 is a third
  unaffected slot.
- NOVEL QUESTION STRATUM: novel pairs use question phrasings absent
  from train/dev (both record and question templates split).
- DOCUMENT COUNTS ARE DOCUMENT COUNTS: train 2048 docs (1024 pairs),
  dev 128 docs (64 pairs), test 128 docs (64 pairs, exactly 32 novel).
- INDEPENDENT WORLD DRAWS: every document's five values are drawn by
  an independent rng.sample (no shared deck/cyclic support law);
  balance is achieved by declared deterministic rejection resampling
  and verified within tolerance; support-set diversity is verified.
The v1 scaffold packet's frozen prepare outputs stay untouched under
internal/analysis_2026_10_14/prototype_runs/ as historical evidence.
"""
import hashlib
import json
import random

N_VALUES = 32
WORDS = [
    "amber", "basil", "cedar", "dune", "ember", "flint", "grove",
    "haven", "indigo", "juniper", "kelp", "lumen", "marble", "nectar",
    "onyx", "petal", "quartz", "ridge", "saffron", "thistle", "umber",
    "velvet", "willow", "xenon", "yarrow", "zephyr", "anchor", "bramble",
    "clover", "drift", "elm", "fjord",
]
SLOTS = ["north key", "east key", "south key", "west key"]
TEMPLATES_FAMILIAR = [
    "{name} keeps the {slot} marked {value}.",
    "the {slot} of {name} is {value}.",
    "{name} filed the {slot} under {value}.",
]
TEMPLATES_NOVEL = [
    "per the ledger, {name}'s {slot} entry reads {value}.",
    "recorded for {name}: {slot} -> {value}.",
]
QUESTIONS_FAMILIAR = ["what is the {slot} of {name}?"]
QUESTIONS_NOVEL = [
    "which entry does {name} list for the {slot}?",
    "for the {slot}, what has {name} on record?",
]
READ_OPS = ["late_target", "unrelated", "revised_target",
            "unrelated_again", "historical", "unaffected"]
# document/pair counts are DOCUMENT counts; pairs are the world
# clusters (bootstrap units)
SPLIT_SIZES = {"train": 2048, "dev": 128, "test": 128}
PAIR_SIZES = {"train": 1024, "dev": 64, "test": 64}
NOVEL_TEST_PAIRS = 32


def balance_band(n_answers):
    """Declared per-split balance band for independent draws: the
    per-word answer count must sit within mean +/- max(4, 3.3 sigma)
    of the multinomial (a real check against systematic skew, not a
    tight constraint that would smuggle structure back in)."""
    mean = n_answers / N_VALUES
    sd = (mean * (1 - 1 / N_VALUES)) ** 0.5
    half = max(4.0, 3.3 * sd)
    return round(mean - half), round(mean + half)

_CONS = "bcdfghjklmnprstvwz"
_VOW = "aeiou"
_SYL = [c + v for c in _CONS for v in _VOW]          # 90 syllables
# one value-free name per world pair; disjoint pools across splits
NAME_POOLS = {
    "train": [_SYL[i // 90] + _SYL[i % 90] for i in range(1024)],
    "dev": [_SYL[i // 90] + _SYL[i % 90] for i in range(1024, 1088)],
    "test": [_SYL[i // 90] + _SYL[i % 90] for i in range(1088, 1152)],
}


def _hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def _check_values():
    """Offline lexical value check (tokenizer single-token check is a
    pinned-config obligation, never faked here)."""
    errs = []
    if len(WORDS) != N_VALUES or len(set(WORDS)) != N_VALUES:
        errs.append("value list must hold 32 distinct words")
    for w in WORDS:
        if not (w.isalpha() and w.islower() and 2 <= len(w) <= 8):
            errs.append(f"value not offline-checkable: {w!r}")
    return errs


def _make_doc(name, values, novel, rng, rid):
    """One document: 4 current records + its own revision.

    values: five distinct words, [v0..v3] current and values[4] the
    replacement; rid: the shared revised slot (fixed by the pair so
    the donor counterfactual addresses the same record in both
    worlds). Returns writer view + the six ground-truth reads."""
    v0, v1, v2, v3, vr = values
    templates = TEMPLATES_NOVEL if novel else TEMPLATES_FAMILIAR
    questions = QUESTIONS_NOVEL if novel else QUESTIONS_FAMILIAR
    records = [
        {"record_id": i, "slot": SLOTS[i], "value": (v0, v1, v2, v3)[i],
         "text": rng.choice(templates).format(
             name=name, slot=SLOTS[i], value=(v0, v1, v2, v3)[i])}
        for i in range(4)
    ]
    revision = {"record_id": rid, "slot": SLOTS[rid], "value": vr,
                "text": rng.choice(templates).format(
                    name=name, slot=SLOTS[rid], value=vr)}
    current = [v0, v1, v2, v3]
    after = list(current)
    after[rid] = vr
    # reader-side roles: the unrelated slot U is re-queried after the
    # revision (preservation); W is a third unaffected slot. Both are
    # drawn AFTER the writer view exists and never enter it.
    u, w = rng.sample([s for s in range(4) if s != rid], 2)

    def q(slot):
        return {"text": rng.choice(questions).format(
                    slot=SLOTS[slot], name=name),
                "name": name, "slot": SLOTS[slot]}

    reads = [
        {"op": "late_target", "snapshot": "v1", "query": q(rid),
         "answer": current[rid]},
        {"op": "unrelated", "snapshot": "v1", "query": q(u),
         "answer": current[u]},
        {"op": "revised_target", "snapshot": "v2", "query": q(rid),
         "answer": vr},
        {"op": "unrelated_again", "snapshot": "v2", "query": q(u),
         "answer": after[u]},
        {"op": "historical", "snapshot": "v1", "query": q(rid),
         "answer": current[rid]},
        {"op": "unaffected", "snapshot": "v2", "query": q(w),
         "answer": after[w]},
    ]
    writer_view = {"doc": name, "records": records, "revision": revision,
                   "novel": novel}
    return writer_view, reads


def _answer_counts(pairs):
    counts = {w: 0 for w in WORDS}
    for p in pairs:
        for reads in (p["reads_a"], p["reads_b"]):
            for r in reads:
                counts[r["answer"]] += 1
    return counts


def make_split(split, n_pairs, seed, novel_count=0):
    """Independent uniform five-value draws per document; declared
    deterministic rejection resampling on the TRUE per-word answer
    counts (independence first, balance by rejection, never a shared
    support law)."""
    names = NAME_POOLS[split]
    if n_pairs > len(names):
        raise ValueError(f"{split}: name pool too small")
    lo, hi = balance_band(n_pairs * 2 * 6)
    for attempt in range(64):
        rng = random.Random(f"{seed}|{split}|v2|{attempt}")
        draws = []
        for _ in range(n_pairs):
            first, second = rng.sample(WORDS, 5), rng.sample(WORDS, 5)
            while any(a == b for a, b in zip(first, second)):
                second = rng.sample(WORDS, 5)
            draws.append([first, second])
        novel_flags = [False] * n_pairs
        for i in rng.sample(range(n_pairs), novel_count):
            novel_flags[i] = True
        pairs = []
        for p in range(n_pairs):
            name = names[p]        # shared entity identity in the pair
            rid = rng.randrange(4)  # shared revised slot (same address)
            vals_a, vals_b = draws[p]
            wa, ra = _make_doc(name, vals_a, novel_flags[p], rng, rid)
            wb, rb = _make_doc(name, vals_b, novel_flags[p], rng, rid)
            pairs.append({"pair": p, "name": name,
                          "novel": novel_flags[p],
                          "doc_a": wa, "doc_b": wb,
                          "reads_a": ra, "reads_b": rb})
        counts = _answer_counts(pairs)
        if lo <= min(counts.values()) and max(counts.values()) <= hi:
            return pairs
    raise RuntimeError(
        f"{split}: balance rejection failed after 64 attempts")


def build_corpus(seed=20261015, splits=("train", "dev", "test")):
    """Build only the requested splits (the trainer path builds
    train/dev ONLY — the trainer process never constructs test
    truth). The default seed equals contract.json's packet seed: one
    frozen corpus everywhere (checks fixtures, --phase prepare and any
    config-less rebuild must never diverge — the v2 default of
    20261016 was a seed-drift defect)."""
    sizes = {"train": (0, 0), "dev": (0, 0),
             "test": (PAIR_SIZES["test"], NOVEL_TEST_PAIRS)}
    corpus = {
        "generator": {
            "packet": "v2-shared-identity", "seed": seed,
            "n_values": N_VALUES, "records_per_doc": 4,
            "reads_per_doc": 6, "read_ops": READ_OPS,
            "doc_counts": SPLIT_SIZES, "pair_counts": PAIR_SIZES,
            "balance_band": "mean +/- max(4, 3.3*sd) per split",
            "familiar_templates": TEMPLATES_FAMILIAR,
            "novel_templates": TEMPLATES_NOVEL,
            "familiar_questions": QUESTIONS_FAMILIAR,
            "novel_questions": QUESTIONS_NOVEL,
            "tokenizer_value_check": "deferred-to-pinned-tokenizer",
        },
        "splits": {
            name: make_split(name, PAIR_SIZES[name], seed,
                             novel_count=sizes[name][1])
            for name in splits
        },
    }
    corpus["hashes"] = {k: _hash(v) for k, v in corpus["splits"].items()}
    return corpus


def value_balance(corpus, split):
    """Answer-value counts for one split (docs of both kinds)."""
    counts = {w: 0 for w in WORDS}
    for pair in corpus["splits"][split]:
        for reads in (pair["reads_a"], pair["reads_b"]):
            for r in reads:
                counts[r["answer"]] += 1
    return counts


def verify_corpus(corpus):
    """Deterministic counts/hashes, shared-identity pairs, disjoint
    name pools, preservation trace, novel question separation,
    independent support diversity, balance, replacement rules,
    value-free ids. Returns a list of failures (empty == verified)."""
    errs = list(_check_values())
    for split, n_docs in SPLIT_SIZES.items():
        if split not in corpus["splits"]:
            continue
        pairs = corpus["splits"][split]
        if len(pairs) * 2 != n_docs or len(pairs) != PAIR_SIZES[split]:
            errs.append(f"{split}: {len(pairs)} pairs != doc count "
                        f"{n_docs}/2")
        if _hash(pairs) != corpus["hashes"][split]:
            errs.append(f"{split}: hash mismatch")
    names = {s: {p["name"] for p in corpus["splits"][s]}
             for s in corpus["splits"]}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if a in names and b in names and names[a] & names[b]:
            errs.append(f"name overlap {a}/{b}")
    if "test" in corpus["splits"]:
        novel = [p for p in corpus["splits"]["test"] if p["novel"]]
        if len(novel) != NOVEL_TEST_PAIRS:
            errs.append("test novel-pair count != 32")
    if "train" in corpus["splits"] and "dev" in corpus["splits"]:
        train_dev = json.dumps(corpus["splits"]["train"]) + \
            json.dumps(corpus["splits"]["dev"])
        for t in TEMPLATES_NOVEL + QUESTIONS_NOVEL:
            marker = t.split("{", 1)[0]
            if marker and marker in train_dev:
                errs.append("novel phrasing leaked into train/dev")
    for split in corpus["splits"]:
        supports = set()
        for p in corpus["splits"][split]:
            if p["doc_a"]["doc"] != p["name"] \
                    or p["doc_b"]["doc"] != p["name"]:
                errs.append(f"{split} pair {p['pair']}: pair members "
                            "do not share entity identity")
            ra = p["doc_a"]["revision"]
            rb = p["doc_b"]["revision"]
            if ra["record_id"] != rb["record_id"]:
                errs.append(f"{split} pair {p['pair']}: revised slots "
                            "differ (donor address mismatch)")
            first_values = [r['value'] for r in p['doc_a']['records']] + [ra['value']]
            second_values = [r['value'] for r in p['doc_b']['records']] + [rb['value']]
            if any(a == b for a, b in zip(first_values, second_values)):
                errs.append(f"{split} pair {p['pair']}: counterfactual binding unchanged")
            for tag, doc, reads in (("a", p["doc_a"], p["reads_a"]),
                                    ("b", p["doc_b"], p["reads_b"])):
                vals = {r["value"] for r in doc["records"]}
                if len(vals) != 4:
                    errs.append(f"{split} p{p['pair']}{tag}: bindings "
                                "not four-distinct")
                if doc["revision"]["value"] in vals:
                    errs.append(f"{split} p{p['pair']}{tag}: replacement"
                                " collides with current values")
                ids = [r["record_id"] for r in doc["records"]]
                if ids != [0, 1, 2, 3]:
                    errs.append("non-positional record ids")
                ops = [r["op"] for r in reads]
                if ops != READ_OPS:
                    errs.append("read sequence mismatch")
                u2 = reads[1]["query"]["slot"]
                u4 = reads[3]["query"]["slot"]
                if u2 != u4:
                    errs.append(f"{split} p{p['pair']}{tag}: "
                                "unrelated_again is not the preserved "
                                "slot")
                supports.add(frozenset(vals | {doc["revision"]["value"]}))
        n_docs = len(corpus["splits"][split]) * 2
        if len(supports) < 0.9 * n_docs:
            errs.append(f"{split}: support diversity {len(supports)}/"
                        f"{n_docs} below declared floor")
    for split in corpus["splits"]:
        balance = value_balance(corpus, split)
        lo, hi = balance_band(
            len(corpus["splits"][split]) * 2 * 6)
        if not (lo <= min(balance.values())
                and max(balance.values()) <= hi):
            errs.append(f"{split}: value balance outside declared "
                        f"tolerance [{lo},{hi}]: "
                        f"{min(balance.values())}..{max(balance.values())}")
    binding_tables = {}
    for split, pairs in corpus['splits'].items():
        binding_tables[split] = {tuple([r['value'] for r in p[key]['records']] + [p[key]['revision']['value']])
                                 for p in pairs for key in ('doc_a','doc_b')}
    for a, b in (('train','dev'),('train','test'),('dev','test')):
        if a in binding_tables and b in binding_tables and binding_tables[a] & binding_tables[b]:
            errs.append(f'binding table overlap {a}/{b}')
    fresh = build_corpus(corpus["generator"]["seed"],
                         splits=tuple(corpus["splits"]))
    if _hash(fresh["splits"]) != _hash(corpus["splits"]):
        errs.append("corpus rebuild not deterministic")
    return errs


def writer_inputs(corpus, split):
    """What the writer/store sees: docs + revisions, NEVER queries."""
    return [
        {"pair": p["pair"], "docs": [p["doc_a"], p["doc_b"]]}
        for p in corpus["splits"][split]
    ]


def reader_queries(corpus, split):
    """Query stream only (doc key, snapshot, op, question text)."""
    out = []
    for p in corpus["splits"][split]:
        for tag, reads in (("a", p["reads_a"]), ("b", p["reads_b"])):
            for r in reads:
                out.append({"pair": p["pair"], "doc": f"{p['pair']}{tag}",
                            "doc_name": r["query"]["name"],
                            "op": r["op"], "snapshot": r["snapshot"],
                            "question": r["query"]["text"],
                            "slot": r["query"]["slot"]})
    return out


def truth_table(corpus, split):
    """Evaluator-only answers, keyed (doc, op)."""
    out = {}
    for p in corpus["splits"][split]:
        for tag, reads in (("a", p["reads_a"]), ("b", p["reads_b"])):
            for r in reads:
                out[(f"{p['pair']}{tag}", r["op"])] = r["answer"]
    return out


def donor_truth_table(corpus, split):
    """Donor-control truth, keyed (doc, 'late_target'): memory is
    swapped between the SHARED-IDENTITY pair members while the
    recipient question stays fixed, so the correct donor-following
    answer is the DONOR's binding at the recipient's queried slot —
    the partner's late-target value at the same slot and address."""
    out = {}
    for p in corpus["splits"][split]:
        late_a = next(r for r in p["reads_a"]
                      if r["op"] == "late_target")
        late_b = next(r for r in p["reads_b"]
                      if r["op"] == "late_target")
        if late_a["query"]["slot"] != late_b["query"]["slot"]:
            raise ValueError("donor address mismatch (shared rid broken)")
        out[(f"{p['pair']}a", "late_target")] = late_b["answer"]
        out[(f"{p['pair']}b", "late_target")] = late_a["answer"]
    return out


# Interpretation note (v2, supersedes the v1 note): the donor swap is
# a content-only counterfactual — shared name/slot/revision address,
# differing bindings/replacements only; world labels stay outside
# reader inputs. unrelated_again re-queries the SAME unrelated slot
# (preservation through revision); the third unaffected slot keeps
# full stratum coverage. Novel pairs vary BOTH record and question
# phrasing; the question stratum is verified absent from train/dev.
