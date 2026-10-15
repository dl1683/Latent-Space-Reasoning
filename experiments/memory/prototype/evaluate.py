"""Separate string evaluator (Z Code B).

Scores GENERATED STRINGS only: exact match after whitespace
normalization (case kept), natural-termination accounting (a capped
output is not a natural termination), majority-constant baseline,
diversity, empty/capped counts, complete docs/pairs, and the frozen
acceptance gates from the brief. Cluster unit for all uncertainty is
the WORLD PAIR (both docs, twelve reads): bootstrap resamples pairs,
never rows; tables carrying duplicated or extra seed rows are rejected
before scoring (no seed-row inflation). Truth enters only here, never
the trainer.
"""
import json
import random
import re

from . import data

# frozen acceptance numbers (ALLIN_ZCODE_B_TRAINER_STORE.md)
GATES = {
    "candidate_total": (692, 768),
    "half_total": (346, 384),        # familiar and novel each
    "per_op": (116, 128),            # each of the six operations
    "complete_docs": (103, 128),
    "complete_pairs": (52, 64),
    "natural_terminations": (761, 768),
    "storage_deficit_max_points": 5.0,   # storage-generated vs candidate
    "donor_late_target": (116, 128),
    "control_deficit_min_points": 20.0,  # omission / untrained vs trained
}
CAPPED_MARK = "<capped>"
ARMS = ("candidate", "storage", "omission", "untrained", "donor",
        "dictionary")
# frozen required roster: the verdict is UNREACHABLE without every arm
# (no cherry-picking; donor is late-target-only 128 rows)
REQUIRED_ARMS = ("candidate", "storage", "omission", "untrained",
                 "donor", "dictionary")
ARM_SHAPES = {
    "donor": (128, {"late_target"}),   # 64 pairs x 2 docs x 1 op
}
DEFAULT_ARM_SHAPE = (768, None)        # 128 docs x 6 ops


def normalize(text):
    """Whitespace normalization ONLY (no case folding)."""
    return re.sub(r"\s+", " ", str(text)).strip()


def exact_match(generated, expected):
    return normalize(generated) == normalize(expected)


def natural_termination(generated, row=None):
    """Natural termination requires TRUSTED HARNESS EVIDENCE (the
    boolean flag from finish_reason/EOS) — a missing flag is NOT
    natural termination, and a flag contradicting a capped/empty
    string never counts either. The separate evaluator distrusts
    candidate-side and absent signals by construction."""
    g = normalize(generated)
    string_ok = bool(g) and CAPPED_MARK not in g
    if row is not None and isinstance(row.get("terminated"), bool):
        if row["terminated"] and not string_ok:
            return False
        return row["terminated"] and string_ok
    return False  # no harness evidence -> not a natural termination


def load_table(path):
    """Load + validate an answers table: one row per (arm, doc, op),
    768 rows per neural arm / 64 pairs / 128 docs / six ops, no
    duplicates, no extra seed rows. Raises ValueError on any
    violation."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    validate_table(rows)
    return rows


def validate_table(rows, expect_pairs=64, expect_ops=None):
    """Validate shape per arm: full arms carry 768 rows over the six
    ops; the donor arm carries exactly its 128 late-target rows. Any
    duplicate, unknown arm/op, partial or inflated arm raises."""
    expect_ops = expect_ops or data.READ_OPS
    seen = set()
    for r in rows:
        key = (r.get("arm"), r.get("doc"), r.get("op"))
        if key in seen:
            raise ValueError(f"duplicate row {key} (seed-row inflation?)")
        seen.add(key)
        if r.get("arm") not in ARMS:
            raise ValueError(f"unknown arm {r.get('arm')!r}")
        if r.get("op") not in expect_ops:
            raise ValueError(f"unknown op {r.get('op')!r}")
    docs = {r["doc"] for r in rows}
    pairs = {d[:-1] for d in docs}
    if len(pairs) != expect_pairs:
        raise ValueError(f"{len(pairs)} pairs != {expect_pairs}")
    if len(docs) != 2 * expect_pairs:
        raise ValueError(f"{len(docs)} docs != {2 * expect_pairs}")
    for arm in {r["arm"] for r in rows}:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n_expect, ops_expect = ARM_SHAPES.get(arm, DEFAULT_ARM_SHAPE)
        if len(arm_rows) != n_expect:
            raise ValueError(
                f"arm {arm}: {len(arm_rows)} rows != {n_expect} "
                f"(partial or inflated arm)")
        if ops_expect is not None:
            ops_seen = {r["op"] for r in arm_rows}
            if ops_seen != ops_expect:
                raise ValueError(
                    f"arm {arm}: ops {sorted(ops_seen)} != "
                    f"{sorted(ops_expect)}")
    return True


def score_arm(rows, arm, truth):
    """Per-arm summary over rows of one arm. truth: {(doc, op): str}
    (for the donor arm this is the DONOR truth table, not recipient
    truth)."""
    arm_rows = [r for r in rows if r["arm"] == arm]
    out = {
        "arm": arm, "n": len(arm_rows),
        "correct": 0, "terminated": 0, "empty": 0, "capped": 0,
        "termination_inconsistencies": 0,
        "by_op": {op: [0, 0] for op in data.READ_OPS},
        "distinct_generated": len({normalize(r["generated"])
                                   for r in arm_rows}),
    }
    for r in arm_rows:
        exp = truth[(r["doc"], r["op"])]
        hit = exact_match(r["generated"], exp)
        out["correct"] += hit
        flagged = r.get("terminated")
        out["terminated"] += natural_termination(r["generated"], r)
        g = normalize(r["generated"])
        string_ok = bool(g) and CAPPED_MARK not in g
        if flagged is True and not string_ok:
            out["termination_inconsistencies"] += 1
        out["empty"] += int(not g)
        out["capped"] += int(CAPPED_MARK in str(r["generated"]))
        out["by_op"][r["op"]][0] += hit
        out["by_op"][r["op"]][1] += 1
    if out["n"]:
        # best constant-answer accuracy against THIS arm's truth (the
        # chance-level comparator; generation-mode frequency alone is
        # reported separately as a diversity statistic)
        truth_counts = {}
        for r in arm_rows:
            exp = truth[(r["doc"], r["op"])]
            truth_counts[exp] = truth_counts.get(exp, 0) + 1
        out["majority_constant_baseline"] = \
            max(truth_counts.values()) / out["n"]
    return out


def complete_units(rows, arm, truth):
    """Docs with 6/6 exact and pairs with both docs 6/6."""
    docs_ok = set()
    for doc in {r["doc"] for r in rows if r["arm"] == arm}:
        hits = [exact_match(r["generated"], truth[(r["doc"], r["op"])])
                for r in rows if r["arm"] == arm and r["doc"] == doc]
        if hits and all(hits):
            docs_ok.add(doc)
    pairs_ok = {p for p in {d[:-1] for d in docs_ok}
                if f"{p}a" in docs_ok and f"{p}b" in docs_ok}
    return docs_ok, pairs_ok


def halves(rows, arm, novel_pairs, truth):
    """Familiar/novel half totals (each 64 docs = 384 reads)."""
    out = {}
    for half, sel in (("familiar", lambda p: p not in novel_pairs),
                      ("novel", lambda p: p in novel_pairs)):
        n = hit = 0
        for r in rows:
            if r["arm"] != arm or not sel(r["doc"][:-1]):
                continue
            n += 1
            hit += exact_match(r["generated"], truth[(r["doc"], r["op"])])
        out[half] = (hit, n)
    return out


def _per_pair_hits(rows, arm, truth):
    """hits/read-count per world pair for one arm (the arm's own row
    shape: 12 reads per pair for full arms, 2 for donor)."""
    pairs = sorted({r["doc"][:-1] for r in rows if r["arm"] == arm})
    per_pair = {}
    for p in pairs:
        hits = n = 0
        for r in rows:
            if r["arm"] == arm and r["doc"][:-1] == p:
                hits += exact_match(r["generated"], truth[(r["doc"],
                                                          r["op"])])
                n += 1
        per_pair[p] = (hits, n)
    return per_pair


def bootstrap_pairs(rows, arm, truth, n_boot=10000, seed=20261014):
    """Resample the 64 WORLD PAIRS with replacement; one-sided lower
    95% bootstrap bound for the arm's mean accuracy."""
    rng = random.Random(seed)
    per_pair = _per_pair_hits(rows, arm, truth)
    pairs = sorted(per_pair)
    means = []
    for _ in range(n_boot):
        sample = rng.choices(pairs, k=len(pairs))
        h = sum(per_pair[p][0] for p in sample)
        n = sum(per_pair[p][1] for p in sample)
        means.append(h / n if n else 0.0)
    means.sort()
    return {"lcb95": means[int(0.05 * len(means))],
            "mean": sum(means) / len(means), "unit": "world_pair",
            "n_boot": n_boot, "seed": seed}


def bootstrap_deficit(rows, candidate_arm, control_arm, truth,
                      n_boot=10000, seed=20261014):
    """Paired per-pair deficit bootstrap (candidate minus control, in
    accuracy POINTS), cluster unit = world pair. Returns the one-sided
    lower 95% bound of the deficit; a control that merely ties has
    lcb95 <= 0."""
    rng = random.Random(seed)
    cand = _per_pair_hits(rows, candidate_arm, truth)
    ctrl = _per_pair_hits(rows, control_arm, truth)
    pairs = sorted(set(cand) & set(ctrl))
    if not pairs:
        return {"lcb95_points": None, "mean_points": None,
                "unit": "world_pair", "n_boot": n_boot, "seed": seed}
    deficits = []
    for _ in range(n_boot):
        sample = rng.choices(pairs, k=len(pairs))
        dc = sum(cand[p][0] / cand[p][1] for p in sample) / len(sample)
        dk = sum(ctrl[p][0] / ctrl[p][1] for p in sample) / len(sample)
        deficits.append(100.0 * (dc - dk))
    deficits.sort()
    return {"lcb95_points": deficits[int(0.05 * len(deficits))],
            "mean_points": sum(deficits) / len(deficits),
            "unit": "world_pair", "n_boot": n_boot, "seed": seed}


def evaluate_all(rows, truth, novel_pairs, donor_truth=None):
    """Full summary + conjunctive gate booleans for the frozen test
    packet. The verdict requires the COMPLETE frozen arm roster
    (candidate/storage/omission/untrained/donor/dictionary): a missing
    control makes all_pass False outright, and the table itself is
    VALIDATED FIRST (partial or inflated arms raise, so failures can
    never disappear by dropping rows). The donor arm is scored against
    DONOR truth. The exact dictionary must be 768/768 and the declared
    baselines must terminate naturally >=95%. All gates are
    conjunctive; no cherry-picking."""
    try:
        validate_table(rows)
    except ValueError as exc:
        return {"summary": {"table_error": str(exc)},
                "gates": {"table_valid": False},
                "all_pass": False}
    summary = {arm: score_arm(rows, arm, truth)
               for arm in {r["arm"] for r in rows}
               if arm != "donor"}
    if any(r["arm"] == "donor" for r in rows):
        if donor_truth is None:
            raise ValueError(
                "donor arm present but donor_truth not supplied")
        summary["donor"] = score_arm(rows, "donor", donor_truth)
    gate = {"table_valid": True}
    missing = [a for a in REQUIRED_ARMS if a not in summary]
    gate["arm_roster_complete"] = not missing
    if missing:
        summary["missing_arms"] = missing
    # the exact dictionary comparator must be PERFECT: an incompetent
    # dictionary means the fixture or the storage view is broken
    if "dictionary" in summary:
        d = summary["dictionary"]
        gate["dictionary_exact_768"] = (
            d["correct"] == 768 and d["n"] == 768)
    # baseline natural termination >=95% on every declared baseline
    # arm (AGENTS evidence gates: below 95% blocks interpretation)
    for base in ("storage", "dictionary", "omission", "untrained"):
        if base in summary:
            b = summary[base]
            gate[f"{base}_termination_ge_95pct"] = (
                b["n"] > 0
                and b["terminated"] / b["n"] >= 0.95)
    if "candidate" in summary:
        c = summary["candidate"]
        gate["candidate_total"] = \
            c["correct"] >= GATES["candidate_total"][0]
        docs_ok, pairs_ok = complete_units(rows, "candidate", truth)
        gate["complete_docs"] = len(docs_ok) >= GATES["complete_docs"][0]
        gate["complete_pairs"] = \
            len(pairs_ok) >= GATES["complete_pairs"][0]
        gate["per_op"] = all(
            v[0] >= GATES["per_op"][0] for v in c["by_op"].values())
        gate["natural_terminations"] = \
            c["terminated"] >= GATES["natural_terminations"][0]
        halves_c = halves(rows, "candidate", novel_pairs, truth)
        gate["half_total"] = all(
            v[0] >= GATES["half_total"][0] for v in halves_c.values())
        summary["candidate_halves"] = halves_c
    if "storage" in summary and "candidate" in summary:
        # deficit of the CANDIDATE relative to storage: the candidate
        # must not trail the equally exposed storage control by more
        # than 5 points (a candidate ahead of storage passes)
        deficit = 100.0 * (summary["storage"]["correct"]
                           - summary["candidate"]["correct"]) / max(
                               summary["storage"]["n"], 1)
        gate["storage_deficit_within_5pts"] = \
            deficit <= GATES["storage_deficit_max_points"]
        summary["storage_deficit_points"] = deficit
    if "donor" in summary:
        donor_late = summary["donor"]["by_op"].get("late_target", [0, 0])
        gate["donor_late_target"] = \
            donor_late[0] >= GATES["donor_late_target"][0]
    for ctrl in ("omission", "untrained"):
        if ctrl in summary and "candidate" in summary:
            deficit = 100.0 * (summary["candidate"]["correct"]
                               - summary[ctrl]["correct"]) / max(
                                   summary["candidate"]["n"], 1)
            boot = bootstrap_deficit(rows, "candidate", ctrl, truth)
            summary[f"{ctrl}_deficit_points"] = deficit
            summary[f"{ctrl}_deficit_cluster"] = boot
            # >= 20 points AND one-sided 95% cluster interval excluding
            # zero (deficit's LCB strictly above 0)
            gate[f"{ctrl}_deficit_ge_20pts_cluster_excl_0"] = (
                gate.get(f"{ctrl}_termination_ge_95pct", False)
                and deficit >= GATES["control_deficit_min_points"]
                and boot["lcb95_points"] is not None
                and boot["lcb95_points"] > 0.0)
    return {"summary": summary,
            "gates": gate,
            "all_pass": bool(gate) and all(gate.values())}


# Final acceptance is deliberately distinct from partial score calculation.
# Keep this wrapper at the public boundary: callers cannot bypass key validation.
_score_complete_packet = evaluate_all


def _decode_evidence(row):
    if row.get('arm') == 'dictionary':
        return isinstance(row.get('generated'), str) and bool(normalize(row['generated']))
    tokens, eos = row.get('tokens'), row.get('eos_token_id')
    count, budget = row.get('token_count'), row.get('max_new_tokens')
    if (not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or type(eos) is not int or eos < 0
            or type(count) is not int or count != len(tokens)
            or type(budget) is not int or budget not in (8, 16)
            or count > budget or type(row.get('terminated')) is not bool
            or not isinstance(row.get('generated'), str) or row.get('error')):
        return False
    if row.get('finish_cause') == 'eos':
        return row['terminated'] and tokens[-1] == eos and eos not in tokens[:-1]
    if row.get('finish_cause') == 'budget':
        return not row['terminated'] and count == budget and eos not in tokens
    return False


def natural_termination(generated, row=None):
    """Evidence-gated termination WITH string consistency: a row only
    counts as naturally terminated when the harness evidence is valid
    AND (for eos rows) the decoded string is non-empty and carries no
    capped marker — an immediate-eos empty decode or a capped string
    flagged eos is exactly the degenerate pattern the >=95% baseline
    gate exists to block."""
    if row is None or not _decode_evidence(row):
        return False
    if row.get('arm') == 'dictionary':
        return True
    if row['finish_cause'] == 'eos' and row['terminated']:
        g = normalize(generated)
        return bool(g) and CAPPED_MARK not in g
    return False


def evaluate_all(rows, truth, novel_pairs, donor_truth=None):
    """Fail closed on malformed/absent arms; score at 64 world-pair units.

    all_pass is reserved for final scientific acceptance. The separate measured
    actual-model competence/persistence/manual-review consumer is not built yet;
    complete scoring alone therefore returns INCOMPLETE, never final PASS.
    """
    try:
        validate_table(rows)
        if not isinstance(truth, dict) or len(truth) != 768:
            raise ValueError('frozen truth must have exactly 768 keys')
        docs = {doc for doc, op in truth}
        pairs = {d[:-1] for d in docs}
        if len(pairs) != 64 or docs != {p+s for p in pairs for s in 'ab'}:
            raise ValueError('truth must contain 64 intact a/b pairs')
        roster = {(d, op) for d in docs for op in data.READ_OPS}
        if set(truth) != roster or any(v not in data.WORDS for v in truth.values()):
            raise ValueError('truth is not the full frozen document/operation roster')
        if len(set(novel_pairs)) != 32 or not set(novel_pairs) <= pairs:
            raise ValueError('novel stratum must contain exactly 32 of 64 pairs')
        if {r['arm'] for r in rows} != set(REQUIRED_ARMS):
            raise ValueError('missing required arm')
        donor_roster = {(d, 'late_target') for d in docs}
        if not isinstance(donor_truth, dict) or set(donor_truth) != donor_roster:
            raise ValueError('missing/partial frozen donor truth')
        if any(v not in data.WORDS for v in donor_truth.values()):
            raise ValueError('donor truth values are not frozen answers')
        for arm in REQUIRED_ARMS:
            selected = [r for r in rows if r['arm'] == arm]
            keys = {(r['doc'], r['op']) for r in selected}
            if keys != (donor_roster if arm == 'donor' else roster):
                raise ValueError('arm keys differ from frozen manifest: ' + arm)
            if not all(_decode_evidence(r) for r in selected):
                raise ValueError('missing/inconsistent decode evidence or crashed row: ' + arm)
        result = _score_complete_packet(rows, truth, novel_pairs, donor_truth)
        for arm in REQUIRED_ARMS:
            selected = [r for r in rows if r['arm'] == arm]
            summary = result['summary'][arm]
            # capped: harness budget-exhaustion OR a capped marker in
            # the string (whichever fires); never overwrite the
            # score_arm diagnostics with a zeroed counter
            summary['capped'] = max(
                sum(r.get('finish_cause') == 'budget' for r in selected),
                summary.get('capped', 0))
            if arm != 'dictionary':
                result['gates'][arm+'_nonconstant'] = summary['distinct_generated'] > 1
        # the storage control must be COMPETENT, not merely present: a
        # vacuous storage arm would make every noninferiority gate
        # content-free. Floor: storage accuracy must beat its own
        # best-constant baseline by at least 20 points.
        stor = result['summary']['storage']
        result['gates']['storage_competent_above_constant'] = (
            stor['n'] > 0
            and stor['correct'] / stor['n']
            >= stor.get('majority_constant_baseline', 1.0) + 0.20)
        result['summary']['storage_competence_margin_points'] = round(
            100.0 * (stor['correct'] / max(stor['n'], 1)
                     - stor.get('majority_constant_baseline', 1.0)), 2)
        # Candidate-storage noninferiority uses paired cluster lower bounds in
        # every stratum; candidate minus storage must stay above -5 points.
        for label, selected in [('all', rows),
                ('familiar', [r for r in rows if r['doc'][:-1] not in novel_pairs]),
                ('novel', [r for r in rows if r['doc'][:-1] in novel_pairs])]:
            interval = bootstrap_deficit(selected, 'candidate', 'storage', truth)
            result['summary']['storage_cluster_'+label] = interval
            result['gates']['storage_cluster_'+label] = interval['lcb95_points'] is not None and interval['lcb95_points'] >= -5.0
        result['scoring_pass'] = all(result['gates'].values())
        result['all_pass'] = False
        result['acceptance_status'] = 'INCOMPLETE'
        result['missing_obligations'] = ['bound actual-model direct-evidence control',
             'separate actual receiving smoke', 'source-excluded persistence receipt',
             '12-example coordinator manual-review provenance', 'validated final receipt consumer']
        return result
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {'summary': {'table_error': str(exc)}, 'gates': {'table_valid': False},
                'scoring_pass': False, 'all_pass': False, 'acceptance_status': 'INCOMPLETE'}
