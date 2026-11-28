"""Getting the mesh onto disk in the form MFEM reads.

MSH version 2.2, because that is what MFEM's gmsh reader wants.  gmsh
defaults to 4.1, so the option is set explicitly at every write rather
than assumed to be still in place from whatever ran before.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import gmsh
import numpy as np

from ._session import session
from .manifest import MSH_VERSION, beside

__all__ = ["write_msh", "read_groups", "confirm_reread", "element_counts"]


def write_msh(path: str | Path, *, binary: bool = False) -> Path:
    """Write the current model as MSH 2.2 and return the path."""
    path = beside(path, ".msh")
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", MSH_VERSION)
    gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
    gmsh.write(str(path))
    return path


def read_groups(path: str | Path) -> dict[int, dict[int, str]]:
    """Re-read a written mesh and report its physical groups.

    gmsh's writer drops entities that belong to no physical group, so a
    numbering that exists in memory but not in the file would be a
    silent loss.  The caller supplies a fresh session; this only merges
    into whatever model is current.
    """
    gmsh.merge(str(path))
    out: dict[int, dict[int, str]] = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        out.setdefault(dim, {})[tag] = gmsh.model.getPhysicalName(dim, tag)
    return out


def confirm_reread(msh_path: str | Path, manifest_path: str | Path, dimension: int,
                   layer_names: Sequence[str | None],
                   interface_names: Sequence[str | None], *,
                   curves: Mapping[int, str] | None = None) -> None:
    """Merge the written file in a fresh session and check its groups.

    A fresh session merging the file is the only evidence a consumer's
    reader will have.  `interface_names` names every boundary group,
    the faults' after the interfaces', and `curves` the dimension-1
    groups by number, where a 3D mesh has any.  On a mismatch both files
    are removed, since a pair that failed this is exactly the pair
    nobody should find later.
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
    if curves:
        got = groups.get(1, {})
        if any(got.get(tag) != name for tag, name in curves.items()):
            msh_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"the written mesh re-reads with dimension-1 groups {got}, not "
                f"{dict(curves)}; both files were removed")


def element_counts(*, dimension: int = 3) -> dict[str, int]:
    """Elements and nodes of the current model, as the file will hold them.

    `elements` are the cells, of `dimension`; `boundary_elements` the
    faces of `dimension - 1`, which carry physical groups and which
    MFEM reads as boundary elements; `nodes` the nodes of the cells,
    which are the nodes written.  The seam curves and points OCC leaves
    on a sphere are meshed too, but never written, and the model may
    hold nodes that no cell uses.
    """
    def count(dim: int) -> int:
        _, tags, _ = gmsh.model.mesh.getElements(dim)
        return int(sum(len(t) for t in tags))

    cells = [np.asarray(n) for n in gmsh.model.mesh.getElements(dimension)[2]]
    nodes = np.unique(np.concatenate(cells)).size if cells else 0
    return {"elements": count(dimension), "boundary_elements": count(dimension - 1),
            "nodes": int(nodes)}
