"""Visible process-boundary canary; Linux/root controller, unprivileged stages.

This is a protocol control, not a sandbox for arbitrary hostile executables.
Run: python -m experiments.memory.sealed_process_control
"""
import base64
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class Receipt:
    source_digest: str
    publication_digest: str
    task_digest: str
    proposal_digest: str
    proposal: bytes
    accepted: bool


def digest(data):
    return sha256(data).hexdigest()


def _stage(command, payload, cwd, uid):
    if os.geteuid() != 0 or uid <= 0:
        raise RuntimeError("the visible UID-separation control needs a root controller and nonroot stages")
    result = subprocess.run(command, input=payload, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=cwd, user=uid, group=uid,
                            extra_groups=[], close_fds=True, timeout=15, check=True,
                            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                                 "HOME": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"})
    if len(result.stdout) > 16 * 1024 * 1024:
        raise ValueError("stage output exceeds visible control allowance")
    return result.stdout


def _packet(publication, task):
    return json.dumps([base64.b64encode(part).decode("ascii")
                       for part in (publication, task)]).encode("ascii")


def run_sealed(source, acquire, select, reason, check, *, worker_uid, reader_uid):
    """Select yields (task, gold) pairs after one publication; it receives no publication.

    Check receives immutable proposal bytes.

    Commands must be pre-registered and trusted not to fork background workers.
    Only source bytes cross the acquisition pipe; only sealed publication and
    late task bytes cross the reader pipe. The controller owns gold custody.
    """
    if type(source) is not bytes or worker_uid == reader_uid or min(worker_uid, reader_uid) <= 0:
        raise ValueError("immutable source and distinct nonroot stage identities required")
    with TemporaryDirectory(prefix="lsr-seal-") as directory:
        root = Path(directory)
        root.chmod(0o711)
        stages = []
        for name, uid in (("worker", worker_uid), ("reader", reader_uid)):
            stage = root / name
            stage.mkdir(mode=0o700)
            os.chown(stage, uid, uid)
            stages.append(stage)
        publication = _stage(acquire, source, stages[0], worker_uid)
        sealed = digest(publication)
        receipts = []
        for task, gold in select():
            if type(task) is not bytes or type(gold) is not bytes:
                raise TypeError("selector must yield immutable task and gold bytes")
            proposal = _stage(reason, _packet(publication, task), stages[1], reader_uid)
            raw_digest = digest(proposal)
            verdict = check(task, gold, proposal)
            if type(verdict) is not bool:
                raise TypeError("checker may accept or reject, never replace a proposal")
            receipts.append(Receipt(digest(source), sealed, digest(task), raw_digest, proposal, verdict))
        return tuple(receipts)


def self_test():
    # The source is a tiny visible control, not an independent held task set.
    # Gold exists before acquisition, in a controller-only directory.
    with TemporaryDirectory(prefix="lsr-gold-") as private:
        path = Path(private) / "gold"
        path.write_bytes(b"C")
        path.chmod(0o600)
        common = f"""import os, sys, json, base64
assert os.geteuid() != 0 and not os.getgroups()
try:
    open({str(path)!r}, 'rb').read()
except PermissionError:
    pass
else:
    raise AssertionError('gold was accessible')
assert 'LSR_PRIVATE_CANARY' not in os.environ
"""
        # Try to write into the other stage too; both must be denied.
        worker = common + """assert not os.access('../reader', os.R_OK | os.W_OK)
src = dict(line.split() for line in sys.stdin.buffer.read().decode().splitlines())
sys.stdout.write(json.dumps(src, sort_keys=True))
"""
        reader = common + """assert not os.access('../worker', os.R_OK | os.W_OK)
pub, task = map(base64.b64decode, json.load(sys.stdin))
links = json.loads(pub)
sys.stdout.write(links[links[task.decode()]])
"""
        args = dict(acquire=[sys.executable, "-c", worker],
                    reason=[sys.executable, "-c", reader],
                    worker_uid=65534, reader_uid=65533)
        gold_check = lambda task, gold, proposal: proposal == gold
        source = b"A B\nB C\nD E\nE F\n"
        a, b = run_sealed(source, select=lambda: ((b"A", path.read_bytes()), (b"D", b"F")),
                          check=gold_check, **args)
        c, = run_sealed(source.replace(b"B C", b"B G"), select=lambda: ((b"A", b"G"),),
                        check=gold_check, **args)
        restored, = run_sealed(source, select=lambda: ((b"A", b"C"),), check=gold_check, **args)
        assert a.accepted and b.accepted and c.accepted and restored == a
        assert a.publication_digest == b.publication_digest != c.publication_digest
        assert a.task_digest != b.task_digest and a.proposal == b"C" and c.proposal == b"G"
        refused, = run_sealed(source, select=lambda: ((b"A", b"Z"),), check=gold_check, **args)
        assert not refused.accepted and refused.proposal_digest == a.proposal_digest
        try:
            run_sealed(source, select=lambda: ((b"A", b"Z"),),
                       check=lambda *_: b"Z", **args)
        except TypeError:
            pass
        else:
            raise AssertionError("checker answer replacement was accepted")
        print("6 episodes: 4 accepted, 1 rejected, 1 invalid checker refused; "
              "5 acquisitions, 6 reads, 11 gold and 11 cross-stage access denials; "
              "publication stable across tasks and restoration")


if __name__ == "__main__":
    self_test()
