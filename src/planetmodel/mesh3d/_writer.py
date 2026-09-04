"""_writer.py -- getting the mesh onto disk in the form MFEM reads.

MSH version 2.2, because that is what MFEM's gmsh reader wants.  gmsh
defaults to 4.1 and will happily write it, so the option is set
explicitly at every write rather than assumed to be still in place from
whatever ran before.
"""
from __future__ import annotations

from pathlib import Path

import gmsh

from ..io.manifest import beside
from ._session import session

__all__ = ["write_msh", "read_groups", "confirm_reread", "element_counts",
           "MSH_VERSION"]

MSH_VERSION = 2.2


def write_msh(path, *, binary: bool = False) -> Path:
    """Write the current model as MSH 2.2 and return the path."""
    path = beside(path, ".msh")
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", MSH_VERSION)
    gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
    gmsh.write(str(path))
    return path


def read_groups(path) -> dict:
    """Re-read a written mesh and report its physical groups.

    Used to check that what was written is what comes back: gmsh's
    writer drops entities that belong to no physical group, and a
    numbering that exists in memory but not in the file would be a
    silent loss.  The caller supplies a fresh session -- this only
    merges into whatever model is current.
    """
    gmsh.merge(str(path))
    out: dict[int, dict[int, str]] = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        out.setdefault(dim, {})[tag] = gmsh.model.getPhysicalName(dim, tag)
    return out


def confirm_reread(msh_path, manifest_path, dimension: int, layer_names,
                   interface_names) -> None:
    """Merge the written file in a fresh session and check its groups.

    Written and read in the same session, a mesh can look right for
    reasons that never reached the disk.  A fresh session merging the
    file is the only evidence a consumer's reader will have.  On a
    mismatch both files are removed, since a pair that failed this is
    exactly the pair nobody should find later.
    """
    msh_path, manifest_path = Path(msh_path), Path(manifest_path)
    with session(name="reread"):
        groups = read_groups(msh_path)
    d = dimension
    wanted = {d: [nm or f"layer_{i + 1}" for i, nm in enumerate(layer_names)],
              d - 1: [nm or f"interface_{i + 1}"
                      for i, nm in enumerate(interface_names)]}
    for dim, names in wanted.items():
        got = groups.get(dim, {})
        if [got.get(i + 1) for i in range(len(names))] != names:
            msh_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"the written mesh re-reads with dimension-{dim} groups {got}, "
                f"not {dict(enumerate(names, 1))}; both files were removed")


def element_counts(*, dimension: int = 3) -> dict:
    """Elements and nodes of the current model, by dimension.

    `elements` counts what the file will hold: the cells and the faces
    of `dimension`, which carry physical groups.  The seam curves and
    points OCC leaves on a sphere are meshed too, but never written.
    """
    counts: dict[str, int] = {}
    total = 0
    for dim in (0, 1, 2, 3):
        _, tags, _ = gmsh.model.mesh.getElements(dim)
        n = int(sum(len(t) for t in tags))
        if n:
            counts[f"dim{dim}"] = n
        if dim in (dimension, dimension - 1):
            total += n
    counts["elements"] = total
    counts["nodes"] = int(gmsh.model.mesh.getNodes()[0].size)
    return counts
