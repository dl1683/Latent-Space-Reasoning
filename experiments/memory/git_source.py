"""Read trusted local Git objects without loading a reasoning backend."""

from io import BytesIO
import os
import subprocess


def git_read(repo, *args, data=None):
    env = dict(os.environ)
    for name in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS",
                 "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"):
        env.pop(name, None)
    env["GIT_NO_LAZY_FETCH"] = "1"
    result = subprocess.run(["git", "--no-replace-objects", "--literal-pathspecs", "-C", str(repo), *args],
                            input=data, capture_output=True, check=False, env=env)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "Git command failed")
    return result.stdout


def read_blobs(repo, blobs):
    if not blobs:
        return {}
    response = BytesIO(git_read(repo, "cat-file", "--batch", data="\n".join((*blobs, "")).encode("ascii")))
    contents = {}
    for blob in blobs:
        header = response.readline().decode("ascii").split()
        if len(header) != 3 or header[:2] != [blob, "blob"]:
            raise ValueError("could not read a pinned Git blob")
        size = int(header[-1])
        contents[blob] = response.read(size)
        if len(contents[blob]) != size or response.read(1) != b"\n":
            raise ValueError("truncated pinned Git blob")
    return contents
