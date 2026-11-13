"""A written mesh as plain arrays.

`read_mesh` reads a mesh file into a `MeshArrays`: the positions of its
nodes, its cells and its boundary elements with their attributes, its
curve groups and the names of its groups.  Every element is a row of
indices into the positions, its corners first and its nodes in the
order the file holds them, so a boundary element keeps the vertex order
that orients a fault's face.  This is what a consumer applies the
manifest's rules to (`manifest.fault_side`, `plus_cell`,
`fault_curves`) with no mesh library between, and what the library's
own checks read.  The positions are the file's: for the mesher's
output, reference positions, which the geometry's mapping moves.

It reads MSH files through gmsh, imported where a file is read: the
version the mesher writes (2.2) and any other gmsh reads.  An MFEM
delivery (`.mesh`) is read by MFEM, whose `Mesh` gives the same arrays
(`GetVertexArray`, `GetElementVertices`, `GetBdrElementVertices`,
`GetAttribute`, `GetBdrAttribute`), corners alone.

A cell is an element of the mesh's own dimension in a physical group,
its attribute the group's number (the layer); a boundary element one of
the dimension below (an interface's or a fault's face); a curve group
one of dimension one in a 3D mesh (a fault's tip, free edge or
junction).  An element in several groups of its dimension appears once
for each.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._extra import require_gmsh

__all__ = ["MeshArrays", "read_mesh"]


def _row_index(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The index of each row of `rows` among the rows of `table` (each row
    present once), or -1 where it is absent."""
    if not len(table) or not len(rows):
        return np.full(len(rows), -1)
    table = np.ascontiguousarray(table, dtype=np.int64)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    view = np.dtype((np.void, table.dtype.itemsize * table.shape[1]))
    keys, want = table.view(view).ravel(), rows.view(view).ravel()
    order = np.argsort(keys)
    at = np.clip(np.searchsorted(keys[order], want), 0, keys.size - 1)
    return np.where(keys[order][at] == want, order[at], -1)


@dataclass(frozen=True, eq=False)
class MeshArrays:
    """A mesh as plain arrays (module docstring); every element a row of
    node indices into `points`, corners first."""

    #: The mesh's dimension, 2 or 3.
    dimension: int
    #: Every node's position, high-order nodes included, (n, 3).
    points: np.ndarray
    #: The file's tag of each node, (n,), in increasing order.
    tags: np.ndarray
    #: The cells, (c, k): k = 4 for a linear tetrahedron, 10 quadratic.
    cells: np.ndarray
    #: The attribute of each cell: its layer, (c,).
    cell_attributes: np.ndarray
    #: The boundary elements, (b, m), in the vertex order the file holds.
    faces: np.ndarray
    #: The attribute of each boundary element: its interface or fault, (b,).
    face_attributes: np.ndarray
    _: KW_ONLY
    #: The curve groups of a 3D mesh, by number: each an (e, j) array of
    #: line elements; empty for a 2D mesh.
    curves: Mapping[int, np.ndarray] = field(default_factory=dict)
    #: The groups' names, by (dimension, number); empty where unnamed.
    names: Mapping[tuple[int, int], str] = field(default_factory=dict)

    def faces_with(self, attribute: int) -> np.ndarray:
        """The boundary elements of one attribute, in file order."""
        return self.faces[self.face_attributes == attribute]

    @cached_property
    def _sides(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every side of every cell, its corners sorted: the distinct
        sides, the index of each one's first cell in `owners`, how many
        cells hold it, and the cell of each sorted entry."""
        d = self.dimension
        corners = self.cells[:, :d + 1]
        sides = np.sort(np.concatenate([np.delete(corners, j, axis=1)
                                        for j in range(d + 1)]), axis=1)
        owners = np.tile(np.arange(len(corners)), d + 1)
        order = np.lexsort(sides.T[::-1])
        unique, first, counts = np.unique(sides[order], axis=0, return_index=True,
                                          return_counts=True)
        return unique, first, counts, owners[order]

    def face_cells(self, faces: ArrayLike) -> np.ndarray:
        """The cells holding each face, (m, 2), the smaller index first:
        two for a face inside the mesh (a fault's, or an interface
        between layers), one and -1 for a face on its outer boundary,
        and -1 twice for corners that are no face of a cell.

        `faces` are rows of node indices whose first `dimension` entries
        are the corners, (m, >= dimension), such as rows of `faces`.
        Refused where a face is held by more than two cells, which no
        conforming mesh has.
        """
        f = np.asarray(faces)
        d = self.dimension
        if f.ndim != 2 or f.shape[1] < d or not np.issubdtype(f.dtype, np.integer):
            raise ValueError(f"faces are rows of at least {d} node indices, got "
                             f"{f.shape} of {f.dtype}")
        unique, first, counts, owners = self._sides
        at = _row_index(np.sort(f[:, :d], axis=1), unique)
        found = at >= 0
        many = found & (counts[np.maximum(at, 0)] > 2)
        if np.any(many):
            raise ValueError(f"{int(np.count_nonzero(many))} faces are held by more "
                             "than two cells")
        out = np.full((len(f), 2), -1, dtype=np.int64)
        start = first[at[found]]
        out[found, 0] = owners[start]
        two = found.copy()
        two[found] = counts[at[found]] == 2
        out[two, 1] = owners[first[at[two]] + 1]
        out[two] = np.sort(out[two], axis=1)
        return out


def read_mesh(path: str | Path) -> MeshArrays:
    """The mesh in an MSH file as plain arrays (module docstring); gmsh is
    imported here, and its absence refused naming the `meshing` extra."""
    gmsh = require_gmsh()
    from ._session import session

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no mesh file at {path}")
    with session(name="read"):
        gmsh.merge(str(path))
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        order = np.argsort(tags)
        tags = tags[order]
        points = np.asarray(xyz, dtype=float).reshape(-1, 3)[order]
        groups = gmsh.model.getPhysicalGroups()
        if not groups:
            raise ValueError(f"{path} has no physical groups, so no attributes")
        dimension = max(dim for dim, _ in groups)

        def elements(dim: int) -> dict[int, np.ndarray]:
            out = {}
            for _, group in (g for g in groups if g[0] == dim):
                blocks = []
                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                    _, etags, enodes = gmsh.model.mesh.getElements(dim, entity)
                    blocks += [np.asarray(n, dtype=np.int64).reshape(len(t), -1)
                               for t, n in zip(etags, enodes) if len(t)]
                if len({b.shape[1] for b in blocks}) > 1:
                    raise ValueError(f"group {group} of dimension {dim} in {path} "
                                     "mixes element types")
                out[group] = (np.searchsorted(tags, np.concatenate(blocks))
                              if blocks else np.empty((0, dim + 1), dtype=np.int64))
            return out

        def stacked(by_group: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            if len({b.shape[1] for b in by_group.values() if len(b)}) > 1:
                raise ValueError(f"the groups of one dimension in {path} mix "
                                 "element types")
            rows = [b for b in by_group.values() if len(b)]
            if not rows:
                return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)
            return (np.concatenate(rows),
                    np.concatenate([np.full(len(b), g, dtype=np.int64)
                                    for g, b in by_group.items() if len(b)]))

        cells, cell_attributes = stacked(elements(dimension))
        faces, face_attributes = stacked(elements(dimension - 1))
        curves = elements(1) if dimension == 3 else {}
        names = {(dim, g): gmsh.model.getPhysicalName(dim, g) for dim, g in groups}
    return MeshArrays(dimension=dimension, points=points, tags=tags, cells=cells,
                      cell_attributes=cell_attributes, faces=faces,
                      face_attributes=face_attributes, curves=curves,
                      names={k: v for k, v in names.items() if v})
