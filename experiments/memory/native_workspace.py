"""Publish source-only native work to one recoverable snapshot history."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4

from experiments.memory.native_publication import (
    _enforce_limit, _read_native_publication, _write_native_publication, read_native_publication,
    write_native_publication,
)
from experiments.memory.native_workspace_rows import SharedNativeRows
from experiments.memory.verified_source_tool import require_digest, sha256


@contextmanager
def _preserve_source_tokenizer_settings(reader):
    # Fast tokenizers can clear backend padding and truncation while encoding
    # an unpadded, untruncated source. Legacy fingerprints include that state.
    backend = getattr(reader._tokenizer, "backend_tokenizer", None)
    settings = {}
    for name in ("padding", "truncation"):
        if (callable(getattr(backend, f"enable_{name}", None))
                and callable(getattr(backend, f"no_{name}", None))):
            settings[name] = getattr(backend, name)
    try:
        yield
    finally:
        for name, original in settings.items():
            if getattr(backend, name) != original:
                if original is None:
                    getattr(backend, f"no_{name}")()
                else:
                    getattr(backend, f"enable_{name}")(**original)


@dataclass(frozen=True, slots=True)
class NativeWorkspacePin:
    workspace_id: str
    revision: int
    source_seal: str
    publication_seal: str

    def __post_init__(self):
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("native workspace revision must be a nonnegative integer")
        for digest in (self.workspace_id, self.source_seal, self.publication_seal):
            require_digest(digest)


class StaleNativeWorkspace(RuntimeError):
    """The opening snapshot is no longer the head."""


class NativeWorkspace:
    """A local, trusted store for immutable native publications and their common head."""

    def __init__(self, directory, *, runtime_sha256, max_bytes):
        self._configure(directory, runtime_sha256, max_bytes)
        with closing(self._connect()) as db:
            self._workspace_id = self._binding(db)
            self._share_rows = self._has_shared_layout(db)
            if self._share_rows and not self._blocks.is_dir():
                raise ValueError("native shared numeric store is missing")
            if self._head(db) is None:
                raise ValueError("native workspace has no committed publication")

    @classmethod
    def create(cls, directory, reader, memory, *, runtime_sha256, max_bytes, share_rows=False):
        """Create a new workspace from an intact selective snapshot."""
        if type(share_rows) is not bool:
            raise ValueError("native workspace row sharing flag must be a boolean")
        workspace = cls.__new__(cls)
        workspace._configure(directory, runtime_sha256, max_bytes)
        workspace._directory.mkdir(mode=0o700)
        workspace._objects.mkdir(mode=0o700)
        workspace._share_rows = share_rows
        if share_rows:
            workspace._blocks.mkdir(mode=0o700)
        workspace._workspace_id = sha256(uuid4().bytes)
        with closing(workspace._connect(create=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE native_workspace (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                runtime TEXT NOT NULL, workspace_id TEXT NOT NULL
            )""")
            db.execute("INSERT INTO native_workspace VALUES (1, ?, ?)", (workspace._runtime, workspace._workspace_id))
            db.execute("""CREATE TABLE native_publications (
                revision INTEGER PRIMARY KEY CHECK (revision >= 0),
                source_seal TEXT NOT NULL, publication_seal TEXT NOT NULL UNIQUE,
                parent_seal TEXT REFERENCES native_publications(publication_seal),
                operation TEXT NOT NULL,
                CHECK ((revision = 0 AND parent_seal IS NULL AND operation = 'open') OR
                       (revision > 0 AND parent_seal IS NOT NULL AND operation IN ('maintain', 'revise')))
            )""")
            if share_rows:
                db.execute("CREATE TABLE native_numeric_layout (version INTEGER NOT NULL)")
                db.execute("INSERT INTO native_numeric_layout VALUES (1)")
            db.execute("COMMIT")
        workspace._commit(reader, memory, None, operation="open")
        return workspace

    def _configure(self, directory, runtime_sha256, max_bytes):
        require_digest(runtime_sha256)
        _enforce_limit(max_bytes)
        self._directory = Path(directory).resolve()
        self._objects = self._directory / "objects"
        self._blocks = self._directory / "numeric-blocks"
        self._database = self._directory / "publications.sqlite3"
        self._runtime = runtime_sha256
        self._max_bytes = max_bytes
        self._workspace_id = None
        self._share_rows = False

    def _connect(self, *, create=False):
        mode = "rwc" if create else "rw"
        db = sqlite3.connect(f"{self._database.as_uri()}?mode={mode}", uri=True,
                             isolation_level=None, timeout=5.0)
        try:
            if db.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                raise ValueError("native workspace requires SQLite rollback mode")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            db.close()
            raise
        return db

    @staticmethod
    def _has_shared_layout(db):
        present = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                             "AND name = 'native_numeric_layout'").fetchone()
        if present is None:
            return False
        if db.execute("SELECT version FROM native_numeric_layout").fetchall() != [(1,)]:
            raise ValueError("unknown native shared numeric layout")
        return True

    def _binding(self, db):
        record = db.execute("SELECT runtime, workspace_id FROM native_workspace WHERE id = 1").fetchone()
        if record is None or record[0] != self._runtime:
            raise ValueError("native workspace runtime differs")
        require_digest(record[1])
        if self._workspace_id is not None and self._workspace_id != record[1]:
            raise ValueError("native workspace identity changed")
        return record[1]

    def _head(self, db):
        workspace_id = self._binding(db)
        row = db.execute("SELECT revision, source_seal, publication_seal FROM native_publications "
                         "ORDER BY revision DESC LIMIT 1").fetchone()
        return None if row is None else NativeWorkspacePin(workspace_id, *row)

    def pin_at(self, revision):
        """Recover a committed historical pin by its revision number."""
        if type(revision) is not int or revision < 0:
            raise ValueError("native workspace revision must be a nonnegative integer")
        with closing(self._connect()) as db:
            return self._at(db, revision)

    def _at(self, db, revision):
        workspace_id = self._binding(db)
        row = db.execute("SELECT revision, source_seal, publication_seal FROM native_publications "
                         "WHERE revision = ?", (revision,)).fetchone()
        if row is None:
            raise ValueError("native workspace revision is not committed in this history")
        return NativeWorkspacePin(workspace_id, *row)

    def head(self):
        """Return the current publication pin without loading its tensors."""
        with closing(self._connect()) as db:
            return self._head(db)

    def attach(self, reader, *, pin=None):
        """Open the current publication or an exact committed historical pin."""
        with closing(self._connect()) as db:
            if pin is None:
                pin = self._head(db)
            elif type(pin) is not NativeWorkspacePin or pin != self._at(db, pin.revision):
                raise ValueError("native workspace pin is not committed in this history")
        return self._read(self._objects / pin.publication_seal, reader, pin)

    def _read(self, directory, reader, pin, *, shared=None):
        options = dict(seal=pin.publication_seal, source_seal=pin.source_seal,
                       runtime_sha256=self._runtime, max_bytes=self._max_bytes)
        if self._share_rows:
            if not self._blocks.is_dir():
                raise ValueError("native shared numeric store is missing")
            shared = shared if shared is not None else SharedNativeRows(self._blocks, self._max_bytes)
            return _read_native_publication(directory, reader, **options, read_rows=shared.read_rows)
        return read_native_publication(directory, reader, **options)

    @staticmethod
    def _expect(expected, current):
        if expected != current:
            raise StaleNativeWorkspace("native workspace head changed")

    def _opening(self, reader, expected):
        if type(expected) is not NativeWorkspacePin:
            raise TypeError("a native workspace opening pin is required")
        self._expect(expected, self.head())
        return self.attach(reader, pin=expected)

    def maintain(self, reader, expected, *, max_seeds, max_parent_scores=None):
        """Apply bounded query-blind maintenance to the specified opening pin."""
        before = self._opening(reader, expected)
        after = reader.maintain(before, max_seeds=max_seeds, max_parent_scores=max_parent_scores)
        if after is before:
            self._expect(expected, self.head())
            return expected
        return self._commit(reader, after, expected, operation="maintain",
                            opening_reader_sha256=before._reader_fingerprint_sha256)

    def revise(self, reader, expected, epoch_directory, source_seal, *, max_source_bytes):
        """Apply a direct source epoch and withdraw invalidated computations."""
        _enforce_limit(max_source_bytes)
        before = self._opening(reader, expected)
        with _preserve_source_tokenizer_settings(reader):
            after = reader.publish(epoch_directory, source_seal, previous=before, max_source_bytes=max_source_bytes)
        return self._commit(reader, after, expected, operation="revise",
                            opening_reader_sha256=before._reader_fingerprint_sha256)

    def _commit(self, reader, memory, expected, *, operation, opening_reader_sha256=None):
        revision = 0 if expected is None else expected.revision + 1
        parent_seal = None if expected is None else expected.publication_seal
        if operation == "maintain" and memory.seal != expected.source_seal:
            raise ValueError("native maintenance must keep its source seal")
        if operation == "revise" and json.loads(memory._scope.descriptor)["parent"] != expected.source_seal:
            raise ValueError("native revision must extend its opening source seal")
        with TemporaryDirectory(prefix=".draft-", dir=self._objects) as temporary:
            staged = Path(temporary) / "publication"
            options = dict(runtime_sha256=self._runtime, max_bytes=self._max_bytes)
            shared = None
            if self._share_rows:
                if not self._blocks.is_dir():
                    raise ValueError("native shared numeric store is missing")
                draft = Path(temporary) / "numeric"
                draft.mkdir(mode=0o700)
                shared = SharedNativeRows(self._blocks, self._max_bytes, draft=draft)
                seal = _write_native_publication(staged, reader, memory, **options, write_rows=shared.write_rows)
            else:
                seal = write_native_publication(staged, reader, memory, **options)
            pin = NativeWorkspacePin(self._workspace_id, revision, memory.seal, seal)
            verified = self._read(staged, reader, pin, shared=shared)
            if opening_reader_sha256 is not None and verified._reader_fingerprint_sha256 != opening_reader_sha256:
                raise ValueError("native workspace successor changed the opening reader")
            destination = self._objects / seal
            while True:
                # A prior process could have exited between rename and commit.
                found = destination.exists()
                if found:
                    self._read(destination, reader, pin)
                with closing(self._connect()) as db:
                    db.execute("BEGIN IMMEDIATE")
                    self._expect(expected, self._head(db))
                    if not found:
                        if destination.exists():
                            continue
                        if shared is not None:
                            shared.install()
                        staged.rename(destination)
                    db.execute("INSERT INTO native_publications VALUES (?, ?, ?, ?, ?)",
                               (pin.revision, pin.source_seal, pin.publication_seal, parent_seal, operation))
                    db.execute("COMMIT")
                    return pin
