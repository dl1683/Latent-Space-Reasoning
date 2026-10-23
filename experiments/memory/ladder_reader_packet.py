"""Join a finalized packed corpus rung to the existing verified-reader bridge.

No retrieval, model, tokenizer initialization or evaluation-label reads. The
caller supplies the trusted preparation digest and an already selected route.
Hash checks establish snapshot byte identity, not relevance or completeness.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import MappingProxyType

from experiments.memory.verified_reader_bridge import InputFrameOverflow, prepare_request
from experiments.memory.verified_source_tool import (
    SourceSpan, VerifiedSourceTool, require_digest, sha256,
)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


class LadderReaderPacket(VerifiedSourceTool):
    """Immutable label-free source snapshot, restricted to one available rung.

    Setup hashes the complete packed file once, including later-rung material;
    setup_bytes_read charges that work. Query verification reads each selected
    complete source body, not the whole packed corpus. No latest-authority claim
    follows from the externally trusted preparation manifest.
    """

    def __init__(self, root, preparation_sha256, target_tokens):
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("absolute copied-store path required")
        require_digest(preparation_sha256)
        if not _integer(target_tokens, 1):
            raise ValueError("positive integer rung required")
        self.setup_bytes_read = 0

        def read(name, expected):
            require_digest(expected)
            raw = (root / name).read_bytes()
            self.setup_bytes_read += len(raw)
            if sha256(raw) != expected:
                raise ValueError(f"changed prepared input: {name}")
            return json.loads(raw)

        preparation = read("prepare.json", preparation_sha256)
        if preparation.get("status") != "PREPARED":
            raise ValueError("completed preparation required")
        pins = preparation["pins"]
        # Explicit allowlist: never open evaluation_only.json or config/model paths.
        pointers = read("pointers.json", pins["pointers.json"])
        sources = read("sources.json", pins["sources.json"])
        rungs = read("rungs.json", pins["rungs.json"])
        questions = read("questions.json", pins["questions.json"])
        require_digest(pins["leaves.bin"])
        hasher = hashlib.sha256()
        with (root / "leaves.bin").open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                self.setup_bytes_read += len(block)
                hasher.update(block)
            leaf_size = source.tell()
        if hasher.hexdigest() != pins["leaves.bin"]:
            raise ValueError("changed prepared input: leaves.bin")
        source_rows = {}
        for row in sources:
            identity = row["source_id"]
            require_digest(identity)
            if (row["sha256"] != identity or identity in source_rows
                    or not _integer(row["start"]) or not _integer(row["end"], 1)
                    or not row["start"] < row["end"] <= leaf_size):
                raise ValueError("invalid source identity or packed extent")
            source_rows[identity] = {
                key: row[key] for key in ("source_id", "sha256", "start", "end")
            }
        ordered = sorted(source_rows.values(), key=lambda row: row["start"])
        if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("overlapping packed source bodies")
        matches = [row for row in rungs if row["target_tokens"] == target_tokens]
        if len(matches) != 1 or matches[0]["status"] != "AVAILABLE":
            raise ValueError("one available registered rung required")
        rung = matches[0]
        n = rung["chunks"]
        if not _integer(n, 1) or n > len(pointers):
            raise ValueError("invalid rung chunk count")
        spans, token_count = [], 0
        for row in pointers[:n]:
            source = source_rows[row["source_id"]]
            if (not _integer(row["start"]) or not _integer(row["end"], 1)
                    or not source["start"] <= row["start"] < row["end"] <= source["end"]
                    or not _integer(row["tokens"], 1)):
                raise ValueError("invalid rung pointer")
            spans.append(SourceSpan(
                row["source_id"], source["sha256"],
                row["start"] - source["start"], row["end"] - source["start"],
                row["span_sha256"]))
            token_count += row["tokens"]
        if (not _integer(rung["actual_tokens"], 1)
                or token_count != rung["actual_tokens"] or token_count < target_tokens
                or token_count - pointers[n - 1]["tokens"] >= target_tokens
                or len(set(spans)) != len(spans)):
            raise ValueError("rung must be the first unique whole-chunk prefix")
        question_rows = {}
        for question in questions:
            if (set(question) != {"id", "question", "type"}
                    or any(not isinstance(question[k], str) or not question[k].strip()
                           for k in question)
                    or question["id"] in question_rows):
                raise ValueError("unique label-free questions required")
            question_rows[question["id"]] = MappingProxyType(dict(question))
        versions = {span.source_id: span.version_sha256 for span in spans}
        super().__init__(root, versions)
        self._sources = MappingProxyType({
            key: MappingProxyType(value) for key, value in source_rows.items()
        })
        self.spans = tuple(spans)
        self._allowed = frozenset(spans)
        self.questions = MappingProxyType(question_rows)
        self.preparation_sha256 = preparation_sha256
        self.rung = MappingProxyType(dict(rung))

    def verify(self, pointer):
        """Verify full source version plus source-local span against pinned bytes."""
        if not isinstance(pointer, SourceSpan):
            raise TypeError("SourceSpan required")
        result = dict(pointer=asdict(pointer), source_bytes_read=0,
                      file="leaves.bin")
        expected = self.versions.get(pointer.source_id)
        if expected is None:
            return dict(result, status="UNKNOWN_SOURCE")
        if expected != pointer.version_sha256:
            return dict(result, status="STALE")
        if pointer not in self._allowed:
            return dict(result, status="OUTSIDE_RUNG")
        source = self._sources[pointer.source_id]
        try:
            with (self.root / "leaves.bin").open("rb") as stream:
                stream.seek(source["start"])
                raw = stream.read(source["end"] - source["start"])
        except FileNotFoundError:
            return dict(result, status="MISSING")
        result["source_bytes_read"] = len(raw)
        if (len(raw) != source["end"] - source["start"]
                or sha256(raw) != pointer.version_sha256):
            return dict(result, status="SOURCE_CHANGED")
        fragment = raw[pointer.start_byte:pointer.end_byte]
        if sha256(fragment) != pointer.span_sha256:
            return dict(result, status="SPAN_CHANGED")
        try:
            text = fragment.decode("utf8")
        except UnicodeDecodeError:
            return dict(result, status="INVALID_UTF8_SPAN")
        return dict(result, status="VERIFIED_SOURCE_BYTES", text=text,
                    returned_utf8_bytes=len(fragment), role="EVIDENCE",
                    verification="BYTE_IDENTITY_ONLY",
                    packed_start_byte=source["start"] + pointer.start_byte,
                    packed_end_byte=source["start"] + pointer.end_byte)

    def prepare(self, question_id, selected, tokenizer, *, full_context=False,
                context_tokens=12000, max_new_tokens=128):
        """Prepare one existing-owner reader call, never execute or score it.

        A full-context request must include every rung row in source order and
        fit without omissions; otherwise request is None and status is NOT_FIT.
        Regular retrieval may omit whole passages under the existing bridge's
        token budget, with exact admitted/omitted row identities retained.
        """
        if type(full_context) is not bool:
            raise ValueError("explicit full_context boolean required")
        question = self.questions[question_id]
        selected = tuple(selected)
        if any(not _integer(i) or i >= len(self.spans) for i in selected):
            raise ValueError("selected row outside registered rung")
        if full_context and selected != tuple(range(len(self.spans))):
            raise ValueError("full context requires every rung row in source order")
        try:
            request = prepare_request(
                self, [self.spans[i] for i in selected], question["question"], tokenizer,
                context_tokens=context_tokens, max_new_tokens=max_new_tokens,
                provenance_mode="handles")
        except InputFrameOverflow as exc:
            rejected = [row["proposal"] for row in exc.verification_checks
                        if row["status"] != "VERIFIED_SOURCE_BYTES"]
            return dict(
                status="SOURCE_REJECTED" if rejected else "NOT_FIT", request=None,
                framing_diagnostic=dict(reason="QUESTION_FRAMING_EXCEEDS_CONTEXT",
                    framing_question_tokens=exc.input_tokens,
                    verification_checks=list(exc.verification_checks)),
                question_id=question_id, question_type=question["type"],
                preparation_sha256=self.preparation_sha256, rung=dict(self.rung),
                selected_rows=list(selected), admitted_rows=[],
                budget_omitted_rows=[selected[i] for i in range(len(selected))
                                     if i not in rejected],
                rejected_rows=[selected[i] for i in rejected],
                source_bytes_read=exc.source_bytes_read, framed_input_tokens=None,
                consumed_reader_calls=0, consumed_reader_tokens=0,
                answer_quality="UNMEASURED", completeness="UNKNOWN",
                semantic_absence="UNKNOWN", full_context=full_context)
        admitted = [int(handle[1:]) for handle in request["citations"]]
        fits = not (request["budget_omissions"] or request["verification_rejections"])
        ready = fits or not full_context
        status = "PREPARED_READER_UNRUN"
        if not ready:
            status = ("SOURCE_REJECTED" if request["verification_rejections"]
                      else "NOT_FIT")
        return dict(
            status=status,
            request=request if ready else None,
            framing_diagnostic=request if not ready else None,
            question_id=question_id, question_type=question["type"],
            preparation_sha256=self.preparation_sha256, rung=dict(self.rung),
            selected_rows=list(selected),
            admitted_rows=[selected[i] for i in admitted],
            budget_omitted_rows=[selected[row["proposal"]]
                                 for row in request["budget_omissions"]],
            rejected_rows=[selected[i] for i, row in
                           enumerate(request["verification_checks"])
                           if row["status"] != "VERIFIED_SOURCE_BYTES"],
            source_bytes_read=request["source_bytes_read"],
            framed_input_tokens=request["input_tokens"],
            consumed_reader_calls=0, consumed_reader_tokens=0,
            answer_quality="UNMEASURED", completeness="UNKNOWN",
            semantic_absence="UNKNOWN", full_context=full_context)

    def read(self, question_id, selected, tokenizer, reader_call, **budgets):
        """Invoke an existing owner's backend once; no retries or scoring.

        The callback receives the existing request envelope, including actual
        tokenizer IDs. The owner supplies competence/identity/resource controls.
        Backend invocations are NOT necessarily model requests or forwards.
        Ollama-style response counters are retained separately from the framed
        logical input; missing counters remain UNKNOWN, never invented zeros.
        """
        if not callable(reader_call):
            raise TypeError("existing-owner reader callback required")
        packet = self.prepare(question_id, selected, tokenizer, **budgets)
        packet.update(backend_invocations_started=0, backend_invocations_returned=0,
                      backend_wall_seconds=0., raw_response=None, raw_answer=None,
                      processed_input_tokens=None, generated_tokens=None,
                      cached_input_tokens=None, terminated=None,
                      consumed_reader_calls=None, consumed_reader_tokens=None)
        if packet["request"] is None:
            # No backend invocation occurred. This is not zero answer accuracy.
            packet.update(consumed_reader_calls=0, consumed_reader_tokens=0)
            return packet
        started = time.perf_counter()
        packet["backend_invocations_started"] = 1
        try:
            response = reader_call(packet["request"])
        except Exception as exc:
            packet.update(status="BACKEND_ERROR", backend_error=type(exc).__name__)
        else:
            packet.update(backend_invocations_returned=1, raw_response=response,
                          status="BACKEND_RETURNED_UNGRADED")
            if isinstance(response, dict):
                message = response.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    packet["raw_answer"] = message["content"]
                for field, name in (("prompt_eval_count", "processed_input_tokens"),
                                    ("eval_count", "generated_tokens"),
                                    ("cached_input_tokens", "cached_input_tokens")):
                    value = response.get(field)
                    if _integer(value):
                        packet[name] = value
                reason = response.get("done_reason")
                if isinstance(reason, str):
                    packet["terminated"] = reason == "stop"
            # A provider response alone does not establish semantic quality,
            # cache-accounting semantics, number of model calls or termination.
        packet["backend_wall_seconds"] = time.perf_counter() - started
        return packet
