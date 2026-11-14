"""Files written whole: a partial file beside the target, put in place
when it is complete.

A writer that must leave either the whole file or none writes under a
fresh name beside the target and renames it over the target at the end
(`os.replace`, atomic on one file system), so that a failed write leaves
no file and an existing one stands until it is replaced.  The partial
file is created with the permissions a plain `open(path, "w")` would
give the target: those of the file already at `path`, kept, or for a
new file 0o666 less the process's umask, which the system applies as it
creates the partial file (not `tempfile.mkstemp`, whose files are 0o600
whatever the umask).  `partial_file`
makes the partial file; `written_whole` is the whole pattern, the
partial file's path handed to the writer and put in place or removed.
Only the standard library is used.
"""
from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["partial_file", "written_whole"]

#: The names tried beside a target before giving up.
_ATTEMPTS = 100


def partial_file(path: str | os.PathLike[str]) -> Path:
    """An empty file created under a fresh name beside `path`
    (`.<name>.<random hex>`), with the permissions a plain
    `open(path, "w")` would give `path`: those of the file at `path`
    where there is one, otherwise 0o666 less the umask."""
    path = Path(path)
    for _ in range(_ATTEMPTS):
        candidate = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        try:
            os.chmod(candidate, stat.S_IMODE(os.stat(path).st_mode))
        except FileNotFoundError:
            pass
        except BaseException:
            os.remove(candidate)
            raise
        return candidate
    raise FileExistsError(f"no name is free beside {path} to write it under")


@contextmanager
def written_whole(path: str | os.PathLike[str]) -> Iterator[Path]:
    """The path of a partial file beside `path` (`partial_file`) for the
    block to write, put in place at `path` with its permissions when the
    block ends without an error, and removed otherwise.  The permissions
    are set again before the rename, so that a writer that made the file
    anew under the partial name leaves them as `partial_file` gave
    them."""
    path = Path(path)
    partial = partial_file(path)
    mode = stat.S_IMODE(os.stat(partial).st_mode)
    try:
        yield partial
        os.chmod(partial, mode)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
