"""A replayable source-only boundary for workers on versioned corpora."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType

from experiments.memory.verified_source_tool import sha256


_SCHEMA = "lsr-source-delta-v1"
_OPAQUE_ID = re.compile(r"[a-f0-9]{32}")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


@dataclass(frozen=True, slots=True)
class SourceVersion:
    sha256: str
    generation: int


@dataclass(frozen=True, slots=True)
class SourceDelta:
    seal: str
    epoch: int
    manifest: MappingProxyType
    changes: MappingProxyType


def _advance(previous, changes):
    epoch = 1 if previous is None else previous.epoch + 1
    manifest = {} if previous is None else dict(previous.manifest)
    for source, raw in changes.items():
        if type(source) is not str or not _OPAQUE_ID.fullmatch(source):
            raise ValueError("source feed needs an opaque document identity")
        if raw is None:
            if source not in manifest:
                raise ValueError("source feed cannot withdraw an absent document")
            del manifest[source]
        else:
            if type(raw) is not bytes:
                raise TypeError("source feed needs committed UTF-8 bytes")
            raw.decode("utf8")
            digest = sha256(raw)
            current = manifest.get(source)
            if current is not None and current.sha256 == digest:
                raise ValueError("source update repeats the current bytes")
            manifest[source] = SourceVersion(digest, epoch)
    return epoch, manifest


def _manifest_record(previous, epoch, manifest, changes_bytes):
    return dict(schema=_SCHEMA, epoch=epoch, parent=None if previous is None else previous.seal,
                changes_sha256=sha256(changes_bytes),
                documents={key: [value.sha256, value.generation] for key, value in manifest.items()})


def write_source_delta(directory, changes, *, previous=None):
    """Publish one source revision, with no task or underlying repository metadata."""
    changes = dict(changes)
    epoch, manifest = _advance(previous, changes)
    scope = MappingProxyType(manifest)
    updates = _encode({key: None if raw is None else raw.decode("utf8") for key, raw in changes.items()})
    descriptor = _encode(_manifest_record(previous, epoch, scope, updates))
    root = Path(directory)
    root.mkdir(mode=0o700)
    (root / "updates.json").write_bytes(updates)
    pending = root / "manifest.pending"
    pending.write_bytes(descriptor)
    pending.rename(root / "manifest.json")
    return SourceDelta(sha256(descriptor), epoch, scope, MappingProxyType(changes))


def read_source_delta(directory, seal, *, previous=None):
    """Verify externally sealed source bytes and the ordered manifest transition."""
    root = Path(directory)
    descriptor = (root / "manifest.json").read_bytes()
    if sha256(descriptor) != seal:
        raise ValueError("source manifest seal differs")
    receipt = json.loads(descriptor)
    if type(receipt) is not dict:
        raise ValueError("source manifest needs an object")
    updates = (root / "updates.json").read_bytes()
    if sha256(updates) != receipt.get("changes_sha256"):
        raise ValueError("source update bytes differ from the manifest")
    decoded = json.loads(updates)
    if type(decoded) is not dict or any(value is not None and type(value) is not str for value in decoded.values()):
        raise ValueError("source updates need an object of UTF-8 strings or withdrawals")
    changes = {key: None if text is None else text.encode("utf8") for key, text in decoded.items()}
    epoch, manifest = _advance(previous, changes)
    expected = _manifest_record(previous, epoch, manifest, updates)
    if _encode(expected) != descriptor or _encode(decoded) != updates:
        raise ValueError("source projection does not match the retained scope and parent")
    return SourceDelta(seal, epoch, MappingProxyType(manifest), MappingProxyType(changes))
