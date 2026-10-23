"""Verify proposed source bytes, not their relevance, truth or completeness.

The caller supplies a trusted version catalogue and ingest-time span hashes.
Neither a map nor a self-consistent hash authenticates that catalogue as latest.
No encoder, model, file access or network operation occurs on import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    version_sha256: str
    start_byte: int
    end_byte: int
    span_sha256: str
    start_character: int | None = None
    end_character: int | None = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source identity required")
        require_digest(self.version_sha256)
        require_digest(self.span_sha256)
        if (type(self.start_byte) is not int or type(self.end_byte) is not int
                or not 0 <= self.start_byte < self.end_byte):
            raise ValueError("nonempty integer byte span required")
        if not (self.start_character is None and self.end_character is None):
            if (type(self.start_character) is not int
                    or type(self.end_character) is not int
                    or not 0 <= self.start_character < self.end_character):
                raise ValueError("invalid mapped character span")

    def matches_map(self, source_id, version_sha256, start, end):
        return (self.source_id == source_id and self.version_sha256 == version_sha256
                and self.start_character == start and self.end_character == end)

    @classmethod
    def from_characters(cls, source_id, data, start, end):
        """Convert an encoder's character offsets once, at ingestion."""
        text = data.decode("utf8")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)):
            raise ValueError("invalid character span")
        a, b = len(text[:start].encode("utf8")), len(text[:end].encode("utf8"))
        return cls(source_id, sha256(data), a, b, sha256(data[a:b]), start, end)


class VerifiedSourceTool:
    """Immutable snapshot; returns EVIDENCE, never instruction authority.

    Retain the old instance for history; create a new instance when the trusted
    source catalogue changes. Old map pointers then fail current-version checks.
    A rejected pointer is not replaced by an unreported retrieval fallback.
    """

    def __init__(self, root, versions):
        self.root = Path(root).resolve()
        versions = dict(versions)
        for identity, version in versions.items():
            if not isinstance(identity, str) or not identity:
                raise ValueError("source identity required")
            require_digest(version)
        self.versions = MappingProxyType(versions)
        self.snapshot_sha256 = sha256(json.dumps(
            versions, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf8"))

    def verify(self, pointer: SourceSpan):
        if not isinstance(pointer, SourceSpan):
            raise TypeError("SourceSpan required")
        result = dict(pointer=asdict(pointer), source_bytes_read=0,
                      file=f"leaves/{pointer.version_sha256}.txt")
        expected = self.versions.get(pointer.source_id)
        if expected is None:
            return dict(result, status="UNKNOWN_SOURCE")
        if expected != pointer.version_sha256:
            return dict(result, status="STALE")
        path = self.root / "leaves" / (pointer.version_sha256 + ".txt")
        # Digest-only filenames prevent injected paths. Reject links escaping
        # the copied store too; this is not an OS sandbox against hostile races.
        if not path.resolve().is_relative_to(self.root):
            return dict(result, status="OUTSIDE_STORE")
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return dict(result, status="MISSING")
        result["source_bytes_read"] = len(data)
        if sha256(data) != pointer.version_sha256:
            return dict(result, status="SOURCE_CHANGED")
        if pointer.end_byte > len(data):
            return dict(result, status="INVALID_SPAN")
        if pointer.start_character is not None:
            try:
                source_text = data.decode("utf8")
            except UnicodeDecodeError:
                return dict(result, status="INVALID_UTF8_SOURCE")
            if (pointer.end_character > len(source_text)
                    or len(source_text[:pointer.start_character].encode("utf8"))
                    != pointer.start_byte
                    or len(source_text[:pointer.end_character].encode("utf8"))
                    != pointer.end_byte):
                return dict(result, status="MAP_SPAN_MISMATCH")
        fragment = data[pointer.start_byte:pointer.end_byte]
        if sha256(fragment) != pointer.span_sha256:
            return dict(result, status="SPAN_CHANGED")
        try:
            text = fragment.decode("utf8")
        except UnicodeDecodeError:
            return dict(result, status="INVALID_UTF8_SPAN")
        return dict(result, status="VERIFIED_SOURCE_BYTES", text=text,
                    returned_utf8_bytes=len(fragment), role="EVIDENCE",
                    verification="BYTE_IDENTITY_ONLY")

    def fetch(self, proposals):
        checks = [self.verify(pointer) for pointer in proposals]
        passages = [r for r in checks if r["status"] == "VERIFIED_SOURCE_BYTES"]
        return dict(
            status="VERIFIED_PROPOSALS" if len(passages) == len(checks)
            and checks else "INCOMPLETE_OR_UNKNOWN",
            proposals=len(checks), accepted=len(passages), passages=passages,
            rejections=[r for r in checks if r["status"] != "VERIFIED_SOURCE_BYTES"],
            source_bytes_read=sum(r["source_bytes_read"] for r in checks),
            returned_utf8_bytes=sum(r["returned_utf8_bytes"] for r in passages),
            snapshot_sha256=self.snapshot_sha256,
            version_eligibility="TRUSTED_CALLER_SNAPSHOT_ONLY",
            semantic_absence="UNKNOWN", completeness="UNKNOWN",
            relevance="UNASSESSED", authority="EVIDENCE_ONLY",
        )
