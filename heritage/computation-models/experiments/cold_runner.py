"""Common-port cold-command runner with an independent monitor (BUILD_PLAN v1, unit 2, rev 2).

Spec: research/endurance/theory/COLD_RUNNER_SPEC.md.  CPU only.  A finite test harness for HONEST policy
code: it catches accidents and makes leaks auditable, it is NOT a sandbox against an adversarial policy
author (that needs process isolation).  A pass proves nothing about the universal CAN/CANNOT and does not
certify an ordinary bounded controller (Python heap wiping is not the bit-program).

Cold boundary is enforced mechanically: a FRESH policy object is built from (N, k, L) for every command,
so no instance attribute can carry history; class-level attributes and the ports facade are also checked.
The checker owns wear, the tape, the oracle and a latched violation flag.
"""

from __future__ import annotations
import argparse, copy, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endurance_compact_reference import ColdCompact  # frozen reference, not modified
from incumbent_flatlog import FlatLog, FlatLogBuffered, FlatLogRetained  # unit 3/3b incumbents


class Exhausted(Exception):
    """Value-changing write to a cell already at L changes (strict INVALID)."""


class QueryWrote(Exception):
    """A QUERY performed a write."""


class Violation(Exception):
    """Wrong answer, retained state, or other contract violation."""


class _Checker:
    """Checker-private state.  Never handed to a policy."""

    def __init__(self, N, L):
        self.N, self.L = N, L
        self.tape = [0] * N
        self.wear = [0] * N
        self.reads = self.writes = self.changes = 0
        self.in_query = False
        self.mem = None  # retained-state control only (unit 3b)
        self.latched = None  # first violation, set BEFORE raising so swallowing it does not help

    def latch(self, exc):
        if self.latched is None:
            self.latched = f"{type(exc).__name__}: {exc}"
        raise exc

    def reset_counters(self):
        self.reads = self.writes = self.changes = 0


class _Ports:
    """Policy-facing facade: read/write, plus `mem` (None unless the instance is a RETAINED-STATE control,
    unit 3b; `mem` is checker-owned and persists across commands, so such a run is NOT a cold run)."""
    __slots__ = ("read", "write", "mem")

    def __init__(self, ck: _Checker):
        def read(address):
            if not isinstance(address, int) or not 0 <= address < ck.N:
                ck.latch(Violation(f"read out of range {address}"))
            ck.reads += 1
            return ck.tape[address]

        def write(address, bit):
            if not isinstance(address, int) or not 0 <= address < ck.N or bit not in (0, 1):
                ck.latch(Violation(f"bad write {address},{bit}"))
            ck.writes += 1
            if ck.in_query:
                ck.latch(QueryWrote(f"QUERY wrote cell {address}"))
            if ck.tape[address] != bit:
                if ck.wear[address] >= ck.L:
                    ck.latch(Exhausted(f"cell {address} exhausted at {ck.L}"))
                ck.wear[address] += 1
                ck.tape[address] = bit
                ck.changes += 1

        object.__setattr__(self, "read", read)
        object.__setattr__(self, "write", write)
        object.__setattr__(self, "mem", ck.mem)

    def __setattr__(self, name, value):
        raise AttributeError("ports facade is read-only")


class DirectStorage:
    """Baseline: logical bit i stored in cell i.  Lifetime is L flips (L-1 if the hot bit starts at 1)."""

    def __init__(self, N: int, k: int, L: int):
        self.N, self.k, self.L = N, k, L

    def init(self, ports, raw):
        for i in range(self.k):
            ports.write(i, raw[i])

    def query(self, ports, i):
        return ports.read(i)

    def flip(self, ports, i):
        ports.write(i, 1 - ports.read(i))


_PLAIN = (int, float, str, bytes, bool, type(None))


def _plain(v):
    if isinstance(v, _PLAIN):
        return True
    if isinstance(v, (list, tuple, set, frozenset, bytearray)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(_plain(a) and _plain(b) for a, b in v.items())
    return False


def _class_state(cls):
    """Deep copy of every non-callable, non-dunder class attribute across the whole MRO (inherited
    state included).  Compared structurally across commands.  Caches stored OUTSIDE the policy class
    hierarchy (module globals, other classes) are not seen: those are audited by reading the policy (L3)."""
    out = {}  # callables, descriptors and dunders are excluded; plain data only, else rejected
    for base in cls.__mro__:
        if base is object:
            continue
        for k, v in vars(base).items():
            if callable(v) or isinstance(v, (staticmethod, classmethod, property)) or k.startswith("__"):
                continue
            if not _plain(v):  # fail closed: arbitrary objects are not structurally comparable
                raise Violation(f"unsupported class-level state {base.__qualname__}.{k} "
                                f"({type(v).__name__}); declare plain data or remove")
            out[(base.__qualname__, k)] = copy.deepcopy(v)
    return out


def _scratch_ok(pol, baseline):
    """Declared scratch (attribute `tape`, if present) must be all zero with unchanged shape at command end."""
    tape = getattr(pol, "tape", None)
    if baseline is None:
        return tape is None
    return tape is not None and len(tape) == baseline and not any(tape)


def run(policy_factory, name, N, k, L, raw, script, keep_trace=False, retained_bits=0):
    """INIT(raw) then script [('F',i)|('Q',i)...] against an oracle.  Stops at first violation.

    Returns a report dict.  outcome is PASS(instance) or FAILED-CONSTRUCTION(instance); both are finite
    statements about the tested history only.
    """
    ck = _Checker(N, L)
    if retained_bits:
        ck.mem = bytearray(retained_bits)
    oracle = list(raw)
    rep = dict(policy=name, N=N, k=k, L=L, commands=0, flips=0, lifetime_H=0, outcome=None, violation=None,
               init=None, worst=dict(query_reads=0, flip_reads=0, flip_writes=0, flip_changes=0),
               failing_command=None, ordinary_work="UNAVAILABLE (Python is not the bit-program; no proxy interface)",
               retained_bits=retained_bits,
               setup="policy constructor recomputed every command; its work is NOT charged",
               trace=[] if keep_trace else None)
    cls_state = None
    scratch_len = None

    def fail(label):
        rep["failing_command"] = dict(cmd=label, reads=ck.reads, writes=ck.writes, changes=ck.changes)

    def command(label, fn):
        nonlocal cls_state, scratch_len
        ck.reset_counters()
        try:
            pol = policy_factory(N, k, L)  # fresh object: cold boundary
            if cls_state is None:
                cls_state = _class_state(type(pol))
                tape0 = getattr(pol, "tape", None)
                scratch_len = len(tape0) if tape0 is not None else None
            out = fn(pol, _Ports(ck))
            if ck.latched:  # policy swallowed an INVALID operation
                raise Violation(f"policy swallowed a violation: {ck.latched}")
            if _class_state(type(pol)) != cls_state:
                raise Violation("class-level attribute changed across command boundary")
            if not _scratch_ok(pol, scratch_len):
                raise Violation("reference scratch not zeroed, resized or removed at command end")
        except (Exhausted, QueryWrote, Violation):
            fail(label)
            raise
        except Exception as exc:  # any non-contract failure inside a command is a failed construction
            fail(label)
            raise Violation(f"policy raised {type(exc).__name__}: {exc}") from exc
        if keep_trace:
            rep["trace"].append(dict(cmd=label, reads=ck.reads, writes=ck.writes, changes=ck.changes))
        return out

    if keep_trace:
        rep["init_raw"] = list(raw)
    try:
        command("INIT", lambda pol, p: pol.init(p, raw))
        rep["init"] = dict(reads=ck.reads, writes=ck.writes, changes=ck.changes)
        for op, i in script:
            rep["commands"] += 1
            if op == "F":
                command(f"F{i}", lambda pol, p: pol.flip(p, i))
                oracle[i] ^= 1
                rep["flips"] += 1
                rep["lifetime_H"] = rep["flips"]
                w = rep["worst"]
                w["flip_reads"] = max(w["flip_reads"], ck.reads)
                w["flip_writes"] = max(w["flip_writes"], ck.writes)
                w["flip_changes"] = max(w["flip_changes"], ck.changes)
            else:
                ck.in_query = True
                try:
                    got = command(f"Q{i}", lambda pol, p: pol.query(p, i))
                finally:
                    ck.in_query = False
                rep["worst"]["query_reads"] = max(rep["worst"]["query_reads"], ck.reads)
                if keep_trace:
                    rep["trace"][-1].update(answer=got, expected=oracle[i])
                if got != oracle[i]:
                    fail(f"Q{i}")
                    raise Violation(f"QUERY({i}) returned {got}, oracle {oracle[i]}")
    except (Exhausted, QueryWrote, Violation) as exc:
        rep["violation"] = f"{type(exc).__name__}: {exc}"
    rep["outcome"] = (f"FAILED-CONSTRUCTION({name},N={N},L={L})" if rep["violation"]
                      else f"PASS({name},N={N},L={L})")
    mw = max(ck.wear)
    rep["max_wear"] = mw
    rep["cells_at_max_wear"] = sum(1 for w in ck.wear if w == mw)
    rep["total_changes"] = sum(ck.wear)
    rep["wear_histogram"] = dict(sorted(Counter(ck.wear).items()))
    return rep


def _compact(N, k, L):
    return ColdCompact(N, k, L)


POLICIES = {"direct": DirectStorage, "compact": _compact, "flatlog": FlatLog}


def two_epoch_script(M, k):
    """Required diagnostic (spec rev 2): from all-ones, cancel pairs on bit 0, distinct bits, a repeated
    bit, and a hammered bit across >= 2*M flips (past folding/clearing/bank exchange for the compact
    reference), querying each touched index after EVERY flip, then terminal queries of all k bits."""
    pattern = [0, 0, 1, 0, 2, 2, 3, 0]
    flips = max(2 * M + 2, 12)
    script, touched = [], set()
    for t in range(flips):
        i = pattern[t % len(pattern)] % k
        script.append(("F", i))
        touched.add(i)
        script += [("Q", j) for j in sorted(touched)]
    script += [("Q", j) for j in range(k)]
    return script


class _WrongAnswer(DirectStorage):
    def query(self, ports, i):
        return 1 - ports.read(i)


class _QueryWriter(DirectStorage):
    def query(self, ports, i):
        b = ports.read(i)
        ports.write(i, b)  # same-value write still counts as a QUERY write
        return b


class _Swallower(DirectStorage):
    """Catches the INVALID exception and carries on: must still be flagged by the latch."""

    def flip(self, ports, i):
        try:
            super().flip(ports, i)
        except Exception:
            pass


class _ClassCache(DirectStorage):
    """Retains history in a class-level attribute: caught by the class fingerprint."""
    hist = 0

    def flip(self, ports, i):
        type(self).hist += 1
        super().flip(ports, i)


class _InheritedCacheBase:
    hist = []


class _InheritedCache(_InheritedCacheBase, DirectStorage):
    """Mutates a list stored on an inherited base class: caught by the MRO-wide deep comparison."""

    def flip(self, ports, i):
        _InheritedCacheBase.hist.append(i)
        super().flip(ports, i)


class _NoWipe(DirectStorage):
    """Declares a scratch tape and leaves it dirty: caught by the scratch-zero check."""

    def __init__(self, N, k, L):
        super().__init__(N, k, L)
        self.tape = bytearray(8)

    def flip(self, ports, i):
        self.tape[0] = 1
        super().flip(ports, i)


class _TapeNone(DirectStorage):
    def __init__(self, N, k, L):
        super().__init__(N, k, L)
        self.tape = bytearray(8)

    def flip(self, ports, i):
        self.tape[0] = 1
        self.tape = None
        super().flip(ports, i)


class _Crasher(DirectStorage):
    def flip(self, ports, i):
        ports.read(i)
        raise RuntimeError("boom")


class _NoopFlip(DirectStorage):
    def flip(self, ports, i):
        return None


def selftest():
    """Monitor must catch each injected fault; True iff every fault is detected and clean runs pass."""
    N, k, L = 16, 8, 3
    script2 = two_epoch_script(4, k)
    cases = {
        "wrong_answer": (_WrongAnswer, [("Q", 0)], [0] * k, "Violation"),
        "query_writes": (_QueryWriter, [("Q", 0)], [0] * k, "QueryWrote"),
        "retained_class_state": (_ClassCache, [("F", 0)], [0] * k, "Violation"),
        "inherited_class_cache": (_InheritedCache, [("F", 0)], [0] * k, "Violation"),
        "scratch_not_wiped": (_NoWipe, [("F", 0)], [0] * k, "Violation"),
        "scratch_replaced_by_none": (_TapeNone, [("F", 0)], [0] * k, "Violation"),
        "policy_crash": (_Crasher, [("F", 0)], [0] * k, "Violation"),
        "swallowed_exhaustion": (_Swallower, [("F", 0)] * (L + 1), [0] * k, "Violation"),
        "wear_overrun": (DirectStorage, [("F", 0)] * (L + 1), [0] * k, "Exhausted"),
        "noop_flip_vs_two_epoch": (_NoopFlip, script2, [1] * k, "Violation"),
    }
    ok = True
    for name, (factory, script, raw, want) in cases.items():
        rep = run(factory, name, N, k, L, raw, script)
        hit = rep["violation"] is not None and rep["violation"].startswith(want)
        print(f"selftest {name:24s} expect {want:10s} got {rep['violation']} -> {'OK' if hit else 'MISSED'}")
        ok &= hit
    clean = run(DirectStorage, "clean", N, k, 64, [1] * k, script2)
    print(f"selftest clean direct two-epoch  outcome={clean['outcome']} -> "
          f"{'OK' if clean['violation'] is None else 'FALSE-ALARM'}")
    return ok and clean["violation"] is None


def _workload(kind, k, n, qevery):
    script = []
    for t in range(n):
        idx = 0 if kind == "hammer" else t % k
        script.append(("F", idx))
        if (t + 1) % qevery == 0:
            script.append(("Q", idx))
    return script


def frontier(Ns, out=None):
    """Unit 3 frontier table.  For each N (k=N/2, L=(log2 N)^2): the compact reference, a flat log SIZED to the
    compact reference's guaranteed lifetime H0 = M*L, direct storage, and (N <= 8192) the full-size flat log.
    Each policy runs two workloads: HAMMER (always bit 0: the lifetime adversary) to first failure or a cap of
    2*H0+10 flips, and SPREAD (distinct indices: worst case for log folds) for the same number of flips.
    Reported lifetime = hammer lifetime (a lower bound if the cap was reached: marked '>=');
    worst-command probes = max over both workloads.  Finite measurements: not absorption, not asymptotic."""
    rows = []
    for N in Ns:
        k, L = N // 2, (N.bit_length() - 1) ** 2
        M = ColdCompact(N, k, L).p.M
        H0 = M * L
        cap = max(2 * H0 + 10, 40)
        policies = {
            "direct": DirectStorage,
            "compact": _compact,
            "flatlog_sized_to_H0": lambda n, kk, ll, h=H0: FlatLog(n, kk, ll, target_H=h),
            "flatlog_buffered_sized_to_H0": lambda n, kk, ll, h=H0: FlatLogBuffered(n, kk, ll, target_H=h),
            "RETAINED_flatlog_sized_to_H0": lambda n, kk, ll, h=H0: FlatLogRetained(n, kk, ll, target_H=h),
        }
        if N <= 8192:  # full-size log: lifetime ~ 26k flips at N=8192; hours in Python beyond that
            policies["flatlog_full"] = FlatLog
        for name, fac in policies.items():
            n_flips = cap if name != "flatlog_full" else FlatLog(N, k, L).lifetime + 40
            qevery = max(1, n_flips // 20)
            rb = fac(N, k, L).retained_bits if name.startswith("RETAINED") else 0
            ham = run(fac, name, N, k, L, [0] * k, _workload("hammer", k, n_flips, qevery), retained_bits=rb)
            spr = run(fac, name, N, k, L, [0] * k, _workload("spread", k, min(n_flips, ham["lifetime_H"] + 1), qevery),
                      retained_bits=rb)
            worst = {key: max(ham["worst"][key], spr["worst"][key]) for key in ham["worst"]}
            capped = ham["violation"] is None
            row = dict(policy=name, N=N, k=k, L=L, M_compact=M, H0=H0, hammer_flips=ham["lifetime_H"],
                       lifetime_lower_bound_only=capped, hammer_outcome=ham["outcome"],
                       spread_outcome=spr["outcome"], worst=worst, hammer_violation=ham["violation"],
                       spread_violation=spr["violation"],
                       M_flatlog=(fac(N, k, L).M if "flatlog" in name else None),
                       workspace_bits=(fac(N, k, L).workspace_bits if hasattr(fac(N, k, L), "workspace_bits") else None),
                       retained_bits=rb, allowed_S=256 * ((N * L + 1).bit_length()) ** 2)
            rows.append(row)
            print(f"N={N:6d} L={L:4d} H0={H0:6d} {name:20s} lifetime{'>=' if capped else '= '}{ham['lifetime_H']:7d} "
                  f"Q_reads={worst['query_reads']:6d} F_reads={worst['flip_reads']:9d} F_writes={worst['flip_writes']:6d} "
                  f"ws={row['workspace_bits']} ret={rb} "
                  f"hammer={ham['outcome'].split('(')[0]} spread={spr['outcome'].split('(')[0]}", flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1, default=str))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default=None, help="comma list of N for the unit-3 frontier table")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--L", type=int, default=169)
    ap.add_argument("--hammer", type=int, default=0, help="hot-bit FLIPs from all-zero")
    ap.add_argument("--workload", choices=("hammer", "spread"), default="hammer",
                    help="hammer: always bit 0.  spread: bit t mod k (distinct indices; worst case for log folds)")
    ap.add_argument("--query-every", type=int, default=1, help="query bit 0 after every Nth hammer flip")
    ap.add_argument("--policies", default=",".join(POLICIES), help="comma list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.frontier:
        frontier([int(x) for x in args.frontier.split(",")], args.out)
        return
    N, L = args.N, args.L
    k = N // 2
    M = ColdCompact(N, k, L).p.M
    results = []
    selected = {p: POLICIES[p] for p in args.policies.split(",")}
    for pname, factory in selected.items():
        results.append(dict(diag="two_epoch_all_ones", **run(factory, pname, N, k, L, [1] * k,
                                                             two_epoch_script(M, k))))
    if args.hammer:
        script = []
        for t in range(args.hammer):
            idx = 0 if args.workload == "hammer" else t % k
            script += [("F", idx)] + ([("Q", idx)] if (t + 1) % args.query_every == 0 else [])
        for pname, factory in selected.items():
            results.append(dict(diag=f"{args.workload}", **run(factory, pname, N, k, L, [0] * k, script)))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    for r in results:
        print(f"{r['diag']:20s} {r['policy']:8s} {r['outcome']:40s} H={r['lifetime_H']:5d} "
              f"maxwear={r['max_wear']:4d} worst={r['worst']} init={r['init']} "
              f"failing={r['failing_command']} violation={r['violation']}")


if __name__ == "__main__":
    main()
