"""A data set's files, fetched once into a cache and checked every time.

`fetch(dataset)` gives the paths of a data set's files in its sources'
order (an archive's named members in place of the archive), from the
cache when they are there with the right sha256 and from the network
otherwise.  The cache is `PLANETMODEL_DATA` when it is set, else the
platform's per-user cache directory (`$XDG_CACHE_HOME/planetmodel` or
`~/.cache/planetmodel`, `~/Library/Caches/planetmodel`,
`%LOCALAPPDATA%\\planetmodel\\Cache`), one directory per data set.

A download streams into a temporary file beside its final name, its
sha256 computed as it comes and its length checked against what the
server announced, and is renamed into place only when whole and right,
so that an interrupted download leaves nothing behind; the source's
mirrors are tried in turn.  Requests carry a timeout and a User-Agent
naming the library, URLs are percent-encoded, and certificates are
always checked: a source on a host whose certificate fails is reached
through a mirror, or its file is placed in the cache by hand, where
`fetch` finds it and checks its hash.  A hash that differs from the
entry's is refused, naming the file and both hashes, unless
`accept_changed=True`, which keeps the file and warns with its hash,
for a source that rewrites its files in place.  An archive's named
members are extracted beside it, each checked by `tarfile`'s data
filter or by its path to stay inside its directory, regular files only,
and re-extracted when the archive changes.  `download=False`, or
`PLANETMODEL_OFFLINE` set, keeps everything off the network.  The
standard library alone.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import warnings
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any
from urllib.parse import quote, urlsplit, urlunsplit

from ..files import partial_file
from .entries import DatasetLike, as_dataset
from .registry import Source, archive_stem

__all__ = ["cache_directory", "fetch", "load", "sha256_of", "CACHE_VARIABLE",
           "OFFLINE_VARIABLE", "TIMEOUT"]

#: The environment variable that names the cache directory.
CACHE_VARIABLE = "PLANETMODEL_DATA"

#: The environment variable that, set to anything but "", "0", "false" or
#: "no", keeps `fetch` off the network.
OFFLINE_VARIABLE = "PLANETMODEL_OFFLINE"

#: Seconds a request may wait on the server before it fails.
TIMEOUT = 60.0

#: Bytes read and hashed at a time.
_CHUNK = 1 << 20

#: The file an archive's extraction records the archive's sha256 in.
_EXTRACTED = ".archive-sha256"


def cache_directory() -> Path:
    """The directory data sets are cached in: `PLANETMODEL_DATA` when set,
    else the platform's per-user cache directory, `planetmodel` in it."""
    named = os.environ.get(CACHE_VARIABLE)
    if named:
        return Path(named).expanduser()
    home = Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        local = Path(base) if base else home / "AppData" / "Local"
        return local / "planetmodel" / "Cache"
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "planetmodel"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base and Path(base).is_absolute()
            else home / ".cache") / "planetmodel"


def _offline() -> bool:
    """Whether `PLANETMODEL_OFFLINE` keeps fetching off the network."""
    return os.environ.get(OFFLINE_VARIABLE, "").strip().lower() not in ("", "0",
                                                                        "false", "no")


def _user_agent() -> str:
    """The User-Agent of every request: the library and its version."""
    from .. import __version__
    return f"planetmodel/{__version__} (+https://github.com/da380/planetmodel)"


def _quoted(url: str) -> str:
    """The URL with its path percent-encoded, escapes already there and
    colons kept: a colon is legal in a path and marks the drive of a
    Windows file URL, which an escaped one would hide."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=quote(parts.path, safe="/%~:")))


def sha256_of(path: Path) -> str:
    """The sha256 of a file, read a chunk at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _partial(directory: Path, name: str) -> Iterator[tuple[IO[bytes], Path]]:
    """A temporary file beside `name` in `directory`, removed unless the
    caller has moved it into place; it has the permissions a plain
    `open(path, "w")` would give the file (`files.partial_file`)."""
    path = partial_file(directory / name)
    try:
        with open(path, "wb") as out:
            yield out, path
    finally:
        path.unlink(missing_ok=True)


def _download(url: str, directory: Path, name: str,
              take: Callable[[Path, str], None]) -> str:
    """Stream `url` into a temporary file and hand it, with its sha256, to
    `take(path, sha256)`, which moves it into place or leaves it to be
    removed; an answer shorter than the length the server announced is
    an interruption.  Gives the sha256."""
    request = urllib.request.Request(_quoted(url),
                                     headers={"User-Agent": _user_agent()})
    with _partial(directory, name) as (out, path):
        digest, count = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            announced = response.headers.get("Content-Length")
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
                count += len(chunk)
        out.close()
        if announced is not None and count != int(announced):
            raise ConnectionError(f"{url}: the download stopped after {count} of "
                                  f"{announced} bytes")
        found = digest.hexdigest()
        take(path, found)
        return found


def _fetched(source: Source, directory: Path, *, accept_changed: bool
             ) -> tuple[Path, str]:
    """The source's file in `directory` and its sha256: downloaded from its
    URL or a mirror, checked, and moved into place; with `accept_changed`
    a file of another hash is kept only when no URL serves the entry's."""
    final = directory / source.name
    directory.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    changed: tuple[str, str] | None = None

    def take(path: Path, found: str) -> None:
        if found == source.sha256 or (accept_changed and changed is None):
            os.replace(path, final)

    for url in (source.url, *source.mirrors):
        try:
            found = _download(url, directory, source.name, take)
        except (OSError, http.client.HTTPException) as exc:
            failures.append(f"{url}: {exc}")
            continue
        if found == source.sha256:
            return final, found
        if changed is None:
            changed = (url, found)
    if changed is not None:
        url, found = changed
        if accept_changed:
            warnings.warn(f"{source.name} from {url} has sha256 {found}, not the "
                          f"entry's {source.sha256}; kept, as accept_changed asks",
                          stacklevel=4)
            return final, found
        raise ValueError(
            f"{source.name} from {url} has sha256 {found}, not the entry's "
            f"{source.sha256}: the source has changed or the download is not the "
            "file; accept_changed=True keeps it")
    raise ConnectionError(
        f"{source.name} could not be fetched:\n  " + "\n  ".join(failures)
        + f"\nplace the file by hand at {final} to use it")


def _cached(source: Source, directory: Path, *, download: bool,
            accept_changed: bool) -> tuple[Path, str]:
    """The source's file in the cache, fetched where it is missing, with
    its sha256."""
    path = directory / source.name
    if path.is_file():
        found = sha256_of(path)
        if found != source.sha256:
            if not accept_changed:
                raise ValueError(
                    f"the cached {path} has sha256 {found}, not the entry's "
                    f"{source.sha256}: remove it to fetch it again, or pass "
                    "accept_changed=True to keep it")
            warnings.warn(f"the cached {path} has sha256 {found}, not the entry's "
                          f"{source.sha256}; kept, as accept_changed asks",
                          stacklevel=4)
        return path, found
    if not download or _offline():
        raise FileNotFoundError(
            f"{path} is not in the cache and fetching is off the network; fetch "
            f"it from {source.url}, or place it there by hand")
    return _fetched(source, directory, accept_changed=accept_changed)


def _write_member(stream: IO[bytes], target: Path) -> None:
    """A member's bytes written beside `target` and moved into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with _partial(target.parent, target.name) as (out, path):
        shutil.copyfileobj(stream, out, _CHUNK)
        out.close()
        os.replace(path, target)


def _extract(archive: Path, source: Source, into: Path) -> None:
    """The source's members of `archive` written under `into`, each checked
    to be a regular file that stays inside it."""
    members = source.members or ()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in members:
                try:
                    info = zf.getinfo(name)
                except KeyError:
                    raise ValueError(f"{archive.name} has no member {name!r}") from None
                target = (into / name).resolve()
                if info.is_dir() or not target.is_relative_to(into.resolve()):
                    raise ValueError(f"the member {name!r} of {archive.name} is not "
                                     "a file inside its directory")
                with zf.open(info) as stream:
                    _write_member(stream, target)
        return
    with tarfile.open(archive) as tf:
        for name in members:
            try:
                info = tf.getmember(name)
            except KeyError:
                raise ValueError(f"{archive.name} has no member {name!r}") from None
            try:
                info = tarfile.data_filter(info, str(into))
            except tarfile.FilterError as exc:
                raise ValueError(f"the member {name!r} of {archive.name} is refused: "
                                 f"{exc}") from None
            stream = tf.extractfile(info) if info.isfile() else None
            if stream is None:
                raise ValueError(f"the member {name!r} of {archive.name} is not a "
                                 "regular file")
            with stream:
                _write_member(stream, into / name)


def _members(archive: Path, found: str, source: Source, directory: Path
             ) -> list[Path]:
    """The paths of an archive's members, extracted unless an extraction of
    this very archive is already whole."""
    into = directory / str(archive_stem(source.name))
    paths = [into / m for m in source.members or ()]
    marker = into / _EXTRACTED
    if (marker.is_file() and marker.read_text().strip() == found
            and all(p.is_file() for p in paths)):
        return paths
    marker.unlink(missing_ok=True)
    into.mkdir(parents=True, exist_ok=True)
    _extract(archive, source, into)
    marker.write_text(found + "\n")
    return paths


def fetch(dataset: DatasetLike, *, download: bool = True,
          accept_changed: bool = False) -> tuple[Path, ...]:
    """The paths of a data set's files, in its sources' order, each checked
    against its sha256, from the cache or fetched into it: an archive with
    members gives its members' paths, extracted beside it.

    `download=False` (or `PLANETMODEL_OFFLINE`) refuses the network, a
    missing file naming where to place it; `accept_changed=True` keeps a
    file whose hash differs from the entry's, with a warning.
    """
    d = as_dataset(dataset)
    directory = cache_directory() / d.name
    paths: list[Path] = []
    for source in d.sources:
        path, found = _cached(source, directory, download=download,
                              accept_changed=accept_changed)
        if source.members is None:
            paths.append(path)
        else:
            paths += _members(path, found, source, directory)
    return tuple(paths)


def load(dataset: DatasetLike, *, download: bool = True, accept_changed: bool = False,
         **options: Any) -> object:
    """The data set as the object it is: its files fetched, then read by
    its entry's reader, `options` going to the reader."""
    d = as_dataset(dataset)
    paths = fetch(d, download=download, accept_changed=accept_changed)
    return d.reader.read(paths, d, **options)
