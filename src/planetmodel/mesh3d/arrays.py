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

`cell_quality` reads the shape of every cell of an MSH file in the same
order, by the measure the mesher's checks take (`spec.QUALITY_MEASURE`,
gmsh's minSICN, which sees a curved element's curvature).

A cell is an element of the mesh's own dimension in a physical group,
its attribute the group's number (its layer's own attribute, or in a
mesh with faults one of its collars', the manifest's `layer_of` saying
whose); a boundary element one of the dimension below (an interface's,
a box's face's or a fault's face); a curve group one of dimension one
in a 3D mesh (a fault's tip, free edge, junction or face edge).  An
element in several groups of its dimension appears once for each.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from . import manifest as _manifest
from ._extra import require_gmsh
from .spec import QUALITY_MEASURE

__all__ = ["MeshArrays", "read_mesh", "cell_quality"]


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
    #: The element attribute of each cell, (c,): its layer's own, or one
    #: of its layer's collars' (`MeshManifest.layer_of`).
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

    def layers(self, manifest: _manifest.MeshManifest) -> np.ndarray:
        """Each cell's layer by its element attribute, as the layer's own
        attribute (`MeshManifest.layer_of`), (c,)."""
        out = np.empty(len(self.cell_attributes), dtype=np.int64)
        for a in np.unique(self.cell_attributes).tolist():
            out[self.cell_attributes == a] = manifest.layer_of(a)["attribute"]
        return out

    def layers_by_position(self, manifest: _manifest.MeshManifest) -> np.ndarray:
        """Each cell's layer by where its corners lie among the manifest's
        layers, as the layer's own attribute, (c,): the layer whose radii,
        or in a flat mesh depths, hold every corner (`MeshManifest.
        layered_coordinate` of the points the mesh's coordinates stand
        for), to a millionth of the mesh's `scale`.  The positions are
        taken as the reference ones they are in the mesher's file.

        A cell whose corners all lie on an interface fits the layers on
        both sides of it, and on a curved interface it need not be flat:
        where the interface's triangulation folds inward along an edge, a
        cell of the layer outside fills the fold with its corners on the
        sphere, and a cell inside spans a convex edge with its corners
        there too.  Its layer is that of its neighbours across
        its sides that are not an interface's boundary elements, which
        lie in one layer with it, the layers meeting at interfaces alone;
        a neighbour itself undecided is decided first.  Refused where a
        cell's corners span two layers, and where a cell's neighbours
        leave its layer undecided."""
        bounds = manifest.boundaries
        d = self.dimension
        X = np.zeros(self.points.shape[:-1] + (3,))
        X[..., list(manifest.mesh_axes(d))] = self.points[..., :d]
        r = manifest.layered_coordinate(X[self.cells[:, :d + 1]])
        tol = 1e-6 * manifest.scale
        fits = ((r.min(axis=1)[:, None] >= bounds[None, :-1] - tol)
                & (r.max(axis=1)[:, None] <= bounds[None, 1:] + tol))
        if not np.all(fits.any(axis=1)):
            raise ValueError(f"{int(np.count_nonzero(~fits.any(axis=1)))} cells have "
                             "corners in two layers")
        out = np.argmax(fits, axis=1) + 1
        open_ = np.count_nonzero(fits, axis=1) > 1
        if not np.any(open_):
            return out
        # the sides two cells hold that are no interface's: their cells pair
        unique, first, counts, owners = self._sides
        interfaces = self.boundary_of([e["attribute"] for e in manifest.interfaces])
        across = counts == 2
        at = _row_index(np.sort(interfaces, axis=1), unique)
        across[at[at >= 0]] = False
        start = first[across]
        pairs = np.stack([owners[start], owners[start + 1]], axis=1)
        pairs = pairs[open_[pairs].any(axis=1)]
        while np.any(open_):
            known = ~open_[pairs]
            settle = known[:, 0] != known[:, 1]
            if not np.any(settle):
                raise ValueError(
                    f"{int(np.count_nonzero(open_))} cells have their corners on an "
                    "interface and no neighbour across a side that is no "
                    "interface's to say which layer they are in")
            cell = np.where(known[settle, 0], pairs[settle, 1], pairs[settle, 0])
            out[cell] = out[np.where(known[settle, 0], pairs[settle, 0],
                                     pairs[settle, 1])]
            open_[cell] = False
        return out

    def walls(self, manifest: _manifest.MeshManifest) -> np.ndarray:
        """Every fault's walls on these arrays (`manifest.walls_of_faults`),
        the faults the manifest's in its order, (F, c): +1 in a fault's
        hanging wall, -1 in its footwall, 0 in neither, a shell's cells
        in neither; a box's faces (`MeshManifest.face_attributes`) are
        where a fault's face edge lies."""
        body = np.isin(self.layers(manifest),
                       [e["attribute"] for e in manifest.layers if e["in_geometry"]])
        out = np.zeros((len(manifest.faults), len(self.cells)), dtype=np.int8)
        out[:, body] = _manifest.walls_of_faults(
            self.points, self.cells[body, :4],
            [self.faces_with(e["attribute"])[:, :3] for e in manifest.faults],
            self.faces_with(manifest.surface_attribute)[:, :3],
            faces=self.boundary_of(manifest.face_attributes),
            names=[e["name"] for e in manifest.faults])
        return out

    def boundary_of(self, attributes: Sequence[int]) -> np.ndarray:
        """The corners of the boundary elements of any of the given
        attributes, (b, dimension), in file order."""
        rows = self.faces[np.isin(self.face_attributes, list(attributes))]
        return rows[:, :self.dimension] if len(rows) else np.empty(
            (0, self.dimension), dtype=np.int64)

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


def _groups(gmsh: object, dim: int, groups: Sequence[tuple[int, int]]
            ) -> dict[int, tuple[list[np.ndarray], list[np.ndarray]]]:
    """The elements of each physical group of dimension `dim` in the open
    gmsh model, in the order `read_mesh` takes them: by group, entity and
    block, each block's element tags (k,) and its node tags (k, j)."""
    out = {}
    for _, group in (g for g in groups if g[0] == dim):
        tags, blocks = [], []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
            _, etags, enodes = gmsh.model.mesh.getElements(dim, entity)
            for t, n in zip(etags, enodes):
                if len(t):
                    tags.append(np.asarray(t, dtype=np.int64))
                    blocks.append(np.asarray(n, dtype=np.int64).reshape(len(t), -1))
        out[group] = (tags, blocks)
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
            for group, (_, blocks) in _groups(gmsh, dim, groups).items():
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


def cell_quality(path: str | Path) -> np.ndarray:
    """The shape of every cell of the mesh in an MSH file, (c,), in the
    order of `read_mesh`'s cells: `spec.QUALITY_MEASURE`, gmsh's minSICN,
    1 for a regular tetrahedron (or triangle), near 0 for a flat one,
    negative for one folded; gmsh is imported here, its absence refused
    naming the `meshing` extra."""
    gmsh = require_gmsh()
    from ._session import session

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no mesh file at {path}")
    with session(name="quality"):
        gmsh.merge(str(path))
        groups = gmsh.model.getPhysicalGroups()
        if not groups:
            raise ValueError(f"{path} has no physical groups, so no cells")
        dimension = max(dim for dim, _ in groups)
        tags = [t for ts, _ in _groups(gmsh, dimension, groups).values() for t in ts]
        if not tags:
            return np.empty(0)
        return np.asarray(gmsh.model.mesh.getElementQualities(
            np.concatenate(tags), QUALITY_MEASURE), dtype=float)
