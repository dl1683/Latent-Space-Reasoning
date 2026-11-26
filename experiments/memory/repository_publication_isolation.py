"""Run the visible repository publication with Linux stage isolation."""

from base64 import b64encode
from collections import Counter
from hashlib import sha1
import json
import os
from pathlib import Path
from shutil import copyfile, which
from stat import S_ISREG
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.memory.repository_import_memory import _git
from experiments.memory.verified_source_tool import sha256


_ACQUIRE_UID = 65534
_READ_UID = 65533
_STAGE_SECONDS = 45
_MAX_PACK_BYTES = 16 * 1024 * 1024
_CODE_MODULES = (
    "binding_feedback", "binding_links", "binding_memory", "binding_replay", "git_source",
    "corpus_binding_journal", "corpus_bindings", "disjoint_feedback", "disjoint_transition",
    "latent_feedback", "recurrent_world", "relational_feedback", "repository_import_memory",
    "repository_transport_control", "role_validity", "source_constraints", "transition_memory",
    "unit_program", "verified_source_tool", "weight_program", "world_transport",
)
_STAGE = """
import json
import os
from pathlib import Path
import socket
import sys

kind, uid, host_net, host_mnt, *args = sys.argv[1:]
assert os.geteuid() == os.getegid() == int(uid)
assert not os.getgroups()
status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
assert status["NoNewPrivs"].strip() == "1" and int(status["CapEff"], 16) == 0
assert str(os.stat("/proc/self/ns/net").st_ino) != host_net
assert str(os.stat("/proc/self/ns/mnt").st_ino) != host_mnt
assert socket.if_nameindex() == [(1, "lo")]

def absent(path):
    try:
        Path(path).open("rb").close()
    except FileNotFoundError:
        return
    raise AssertionError(f"unexpected authority for {path}")

for path in ("/grade/answers.json", "/answers.json", "/mnt/c", "/home", "/root"):
    absent(path)
sys.path.insert(0, "/code")

if kind == "publish":
    absent("/questions.json")
    absent("/publication/publication.json")
    assert os.statvfs("/source").f_flag & os.ST_RDONLY
else:
    assert kind == "read"
    absent("/source/HEAD")
    absent("/output/publication/publication.json")
    assert os.statvfs("/publication").f_flag & os.ST_RDONLY
    assert os.statvfs("/questions.json").f_flag & os.ST_RDONLY
    def refuse_subprocess(event, arguments):
        if event == "subprocess.Popen":
            raise AssertionError("isolated reader attempted to launch a subprocess")
    sys.addaudithook(refuse_subprocess)

fds = sorted(int(path.name) for path in Path("/proc/self/fd").iterdir() if path.is_symlink())
assert fds == [0, 1, 2]
mounts = sorted(line.split()[4] for line in Path("/proc/self/mountinfo").read_text().splitlines())
print(json.dumps({"mode": kind, "uid": int(uid), "groups": os.getgroups(), "fds": fds, "mounts": mounts,
                  "network_interfaces": socket.if_nameindex()}), file=sys.stderr, flush=True)
from experiments.memory.repository_transport_control import main
main(args)
"""


def _scope_objects(repo, pins, package):
    from experiments.memory.repository_transport_control import _tree

    parts = package.split(".")
    scope = "/".join(parts)
    objects = set()
    for pin in sorted(set(pins)):
        commit, modules = _tree(repo, pin, package)
        objects.add(commit)
        objects.update(_git(repo, "rev-parse", f"{commit}^{{tree}}",
                            *(f"{commit}:{'/'.join(parts[:index])}" for index in range(1, len(parts))))
                       .decode("ascii").splitlines())
        listing = _git(repo, "ls-tree", "-rtz", "--full-tree", commit, "--", scope)
        for line in listing.split(b"\0"):
            if line:
                metadata, _ = line.split(b"\t", 1)
                _, kind, oid = metadata.decode("ascii").split()
                if kind == "tree":
                    objects.add(oid)
        objects.update(blob for _, blob in modules.values())
    return sorted(objects)


def _project_git(repo, destination, pins, package, work):
    objects = _scope_objects(repo, pins, package)
    pack = _git(repo, "pack-objects", "--stdout", "--no-reuse-delta",
                data=("\n".join(objects) + "\n").encode("ascii"))
    if len(pack) > _MAX_PACK_BYTES:
        raise ValueError("scoped Git pack exceeds the visible control allowance")
    _git(work, "init", "-q", "--bare", "--template=", "--object-format=sha1", str(destination))
    _git(destination, "index-pack", "--stdin", data=pack)
    listed = _git(destination, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
    if sorted(listed.decode("ascii").splitlines()) != objects:
        raise ValueError("isolated Git pack differs from the scoped object allowlist")
    from experiments.memory.repository_transport_control import _tree
    for pin in set(pins):
        if _tree(repo, pin, package) != _tree(destination, pin, package):
            raise ValueError("isolated Git package listing differs from its pinned source")
    for path in (destination, *destination.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o444)
    return dict(git_pack_sha256=sha256(pack), git_pack_bytes=len(pack), git_objects=len(objects))


def _snapshot_code(destination):
    source = Path(__file__).resolve().parent
    package = destination / "experiments" / "memory"
    package.mkdir(parents=True)
    for path in (destination, package.parent, package):
        path.chmod(0o755)
    hashes, blob_ids = {}, {}
    for module in _CODE_MODULES:
        filename = f"{module}.py"
        target = package / filename
        copyfile(source / filename, target)
        target.chmod(0o444)
        raw = target.read_bytes()
        hashes[filename] = sha256(raw)
        blob_ids[filename] = sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    return hashes, blob_ids


def _code_overlap(repo, pins, package, blob_ids):
    from experiments.memory.repository_transport_control import _tree

    rows = []
    for pin in pins:
        _, modules = _tree(repo, pin, package)
        pinned = {Path(path).name: oid for path, oid in modules.values()
                  if Path(path).parent.as_posix() == "experiments/memory"}
        shared = pinned.keys() & blob_ids.keys()
        rows.append(dict(pin=pin, mounted_code_paths_in_pin=len(shared),
                         mounted_code_blobs_in_pin=sum(pinned[name] == blob_ids[name] for name in shared)))
    return rows


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_record(path, content, *, mode=0o600):
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as handle:
        os.fchmod(handle.fileno(), mode)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    pending.rename(path)
    _sync_directory(path.parent)


def _stage_capture(output, name, stdout, stderr, returncode):
    payload = dict(file=name, sha256=sha256(stdout), returncode=returncode,
                   stderr_sha256=sha256(stderr), stderr_base64=b64encode(stderr).decode("ascii"))
    _write_record(output / name, stdout)
    with (output / "stages.jsonl").open("a", encoding="utf8") as ledger:
        ledger.write(json.dumps(payload) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
    _sync_directory(output)


def _isolated(tool, runtime, code, mode, bindings, arguments, output, name):
    uid = _ACQUIRE_UID if mode == "publish" else _READ_UID
    command = [tool, "--unshare-ipc", "--unshare-pid", "--unshare-net", "--unshare-uts",
               "--die-with-parent", "--new-session", "--clearenv",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chmod", "1777", "/tmp", "--chdir", "/tmp",
               "--ro-bind", str(runtime), "/runtime", "--ro-bind", str(code), "/code",
               "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/tmp",
               "--setenv", "LANG", "C.UTF-8", "--setenv", "OPENBLAS_NUM_THREADS", "1",
               "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "GIT_CONFIG_NOSYSTEM", "1",
               "--setenv", "GIT_CONFIG_COUNT", "1", "--setenv", "GIT_CONFIG_KEY_0", "safe.directory",
               "--setenv", "GIT_CONFIG_VALUE_0", "/source"]
    for binding, host, guest in bindings:
        command.extend((binding, str(host), guest))
    command.extend(("/usr/bin/setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups",
                    "--no-new-privs", "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all",
                    "/runtime/bin/python", "-I", "-B", "-c", _STAGE, mode, str(uid),
                    str(os.stat("/proc/self/ns/net").st_ino), str(os.stat("/proc/self/ns/mnt").st_ino),
                    *arguments))
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                   close_fds=True, timeout=_STAGE_SECONDS, check=False)
    except subprocess.TimeoutExpired as failed:
        _stage_capture(output, name, failed.stdout or b"", failed.stderr or b"", None)
        raise
    _stage_capture(output, name, completed.stdout, completed.stderr, completed.returncode)
    if completed.returncode:
        detail = completed.stderr.decode("utf8", errors="replace").strip()
        raise RuntimeError(f"isolated {mode} failed: {detail}")
    return completed.stdout


def _retain_publication(candidate, bundle, limit):
    if candidate.is_symlink() or sorted(p.name for p in candidate.iterdir()) != ["corpus.sqlite", "publication.json"]:
        raise ValueError("publisher must leave exactly a journal and its manifest")
    bundle.mkdir()
    bundle.chmod(0o755)
    for path in candidate.iterdir():
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
                raise ValueError("publisher must provide bounded regular files with one link")
            data = handle.read(limit + 1)
            if len(data) > limit:
                raise ValueError("publisher file exceeded the transfer allowance")
        _write_record(bundle / path.name, data, mode=0o444)


def _json_record(path, payload):
    _write_record(path, (json.dumps(payload, indent=2) + "\n").encode("utf8"))


def _check_host_layout(repo, output, runtime):
    mounts = (runtime, Path("/usr").resolve())
    git_paths = _git(repo, "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir",
                     "--git-path", "objects").decode("utf8").splitlines()
    git_dir, common, objects = (Path(path).resolve() for path in git_paths)
    if any(path.is_relative_to(mount) for path in (repo, output, git_dir, common, objects) for mount in mounts):
        raise ValueError("source checkout, Git storage and custody must be outside stage runtime mounts")
    alternates = objects / "info" / "alternates"
    if alternates.exists() or any(path.is_symlink() for path in objects.rglob("*")):
        raise ValueError("isolated control requires a Git object store without alternates or symlinked entries")


def run_isolated(repo, output):
    from experiments.memory.repository_publication_trial import _ARMS, _PACKAGE, _PINS, _SEED, _author, _grade
    from experiments.memory.repository_transport_control import (
        _MAX_JOURNAL_BYTES, _bounded_read, _read_manifest, verify_publication,
    )

    if not __debug__:
        raise RuntimeError("the publication trial requires Python assertions")
    tool = which("bwrap")
    runtime = Path(sys.prefix).resolve()
    if (sys.platform != "linux" or os.geteuid() != 0 or not tool
            or not Path("/usr/bin/setpriv").is_file()
            or sys.prefix == sys.base_prefix or not (runtime / "bin/python").is_file()):
        raise RuntimeError("use a Linux root controller with bubblewrap and a dedicated Python venv")
    repo, output = repo.resolve(), output.resolve()
    _check_host_layout(repo, output, runtime)
    output.mkdir(mode=0o700)
    if output.stat().st_mode & 0o077:
        raise ValueError("custody directory requires private Linux filesystem permissions")
    _sync_directory(output.parent)
    with TemporaryDirectory(prefix="lsr-isolated-publication-", dir="/tmp") as scratch:
        root = Path(scratch)
        code, source, work = (root / name for name in ("code", "source", "worker"))
        snapshots, blob_ids = _snapshot_code(code)
        source_report = _project_git(repo, source, _PINS, _PACKAGE, root)
        overlap = _code_overlap(source, _PINS, _PACKAGE, blob_ids)
        work.mkdir(mode=0o700)
        os.chown(work, _ACQUIRE_UID, _ACQUIRE_UID)
        publication = _isolated(tool, runtime, code, "publish",
                                [("--ro-bind", source, "/source"), ("--bind", work, "/output")],
                                ["--publish", "--journal-only", "--repo", "/source", "--output", "/output/publication", *_PINS],
                                output, "publisher.json")
        receipt = json.loads(publication)
        bundle = output / "publication"
        _retain_publication(work / "publication", bundle, _MAX_JOURNAL_BYTES)
        manifest = _read_manifest(bundle, receipt["seal"])
        journal = _bounded_read(bundle / "corpus.sqlite", _MAX_JOURNAL_BYTES)
        if sha256(journal) != manifest["journal_sha256"] or receipt["content_sha256"] != manifest["content_sha256"]:
            raise ValueError("isolated publication receipt differs from its sealed files")
        expected_commits = _git(source, "rev-parse", *(f"{pin}^{{commit}}" for pin in _PINS)).decode("ascii").splitlines()
        if [pin[0] for pin in manifest["pins"]] != expected_commits:
            raise ValueError("isolated publication did not retain the requested commit order")
        verify_publication(bundle, receipt["seal"])
        _json_record(output / "publication-receipt.json", dict(
            **receipt, **source_report, journal_bytes=len(journal), journal_sha256=sha256(journal),
            stage_code_sha256=snapshots, code_overlap=overlap, worker_uid=_ACQUIRE_UID, reader_uid=_READ_UID,
        ))

        questions, gold = output / "questions.json", output / "answers.json"
        _author(source, questions, gold)
        questions.chmod(0o444)
        for path in (questions, gold):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _sync_directory(output)
        raw_proposals, records = [], []
        for number, arm in enumerate(_ARMS):
            name = f"proposal-{number}.json"
            proposal = _isolated(tool, runtime, code, "read",
                                 [("--ro-bind", bundle, "/publication"), ("--ro-bind", questions, "/questions.json")],
                                 ["--read", "/publication", receipt["seal"], "/questions.json", "--arm", arm],
                                 output, name)
            records.append(dict(file=name, arm=arm, sha256=sha256(proposal)))
            raw_proposals.append(proposal)
        _json_record(output / "proposals.json", dict(
            publication_seal=receipt["seal"], questions_sha256=sha256(questions.read_bytes()), proposals=records,
        ))
        outputs = [json.loads(proposal) for proposal in raw_proposals]
        checked = _grade(repo, outputs, questions, gold, receipt["seal"])
        result = dict(**receipt, **source_report, seed=_SEED, journal_bytes=len(journal),
                      questions_sha256=outputs[0]["questions_sha256"], checked_attached_witnesses=checked,
                      attached_statuses=Counter(row["status"] for row in outputs[0]["results"]),
                      validated_reads=sum(len(result["results"]) for result in outputs),
                      stages=1 + len(outputs), code_overlap=overlap)
        _json_record(output / "grade.json", result)
        return result
