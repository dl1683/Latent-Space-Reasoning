"""Input-only adapter for the frozen independent ordinary competence schema.

Pass decode_cases to run_ordinary_cpu_calibration. The caller must supply
and verify the published packet digest and obtain execution authorization.
No case bodies, grading criteria, scripted paths or model code live here.
"""

import json

SCHEMA = "lsr-independent-ordinary-competence-v1"
SCOPE = "ordinary-reader calibration only; separate from the frozen operation packet"
LIMITS = {
    "calls": 4, "candidates_per_search": 1, "evidence_bytes": 2048,
    "input_tokens_per_call": 4096, "max_feed_bytes": 65536,
    "max_new_tokens": 512, "max_passages": 16, "max_source_bytes": 65536,
    "passage_bytes": 512, "source_bytes": 2048, "steps": 8,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in calibration input")
        result[key] = value
    return result


def _nonempty_text(value):
    return type(value) is str and bool(value.strip())


def decode_cases(packet):
    """Normalize input exactly; no search, answer, or grade information added."""
    from experiments.memory.ordinary_cpu_calibration import OrdinaryCalibrationCase

    if type(packet) is not bytes:
        raise ValueError("expected frozen UTF-8 input bytes")
    data = json.loads(packet.decode("utf-8"), object_pairs_hook=_unique_object)
    if (type(data) is not dict or set(data) != {"schema", "scope", "limits", "cases"}
            or data["schema"] != SCHEMA or data["scope"] != SCOPE):
        raise ValueError("unexpected independent ordinary input envelope")
    limits = data["limits"]
    if (type(limits) is not dict or limits != LIMITS
            or any(type(v) is not int for v in limits.values())):
        raise ValueError("calibration limits differ from the frozen contract")
    cases = data["cases"]
    if type(cases) is not list or len(cases) != 4:
        raise ValueError("expected four independent cases")
    normalized = []
    seen = set()
    for case in cases:
        if (type(case) is not dict or set(case) != {"id", "task", "source_deltas", "read_epoch"}
                or not _nonempty_text(case["id"]) or not _nonempty_text(case["task"])
                or case["id"] in seen):
            raise ValueError("invalid or repeated calibration case")
        deltas, epoch = case["source_deltas"], case["read_epoch"]
        if (type(deltas) is not list or type(epoch) is not int or not 1 <= epoch <= len(deltas)
                or any(type(d) is not dict or any(not _nonempty_text(k)
                       or (v is not None and type(v) is not str) for k, v in d.items()) for d in deltas)):
            raise ValueError("invalid source delta history or read epoch")
        seen.add(case["id"])
        normalized.append(OrdinaryCalibrationCase(case["id"], case["task"], tuple(deltas), epoch))
    return tuple(normalized)
