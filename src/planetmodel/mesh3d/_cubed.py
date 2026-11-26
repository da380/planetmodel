"""The cubed sphere: its cells, its map and its radial knots.

A ball is filled with hexahedra, with no mesher, as a central cube of
`edge`^3 cells and six blocks of `edge` x `edge` columns above its
faces, one cell thick for each radial level; a shell as the six blocks
alone; a disc and an annulus the same way in the plane, a central
square and four blocks of quadrilaterals.  The mesh conforms: every
face inside it is a face of exactly two cells.

It is first made in a parameter space of nested cubes.  A point P there
has cubic radius s = |P|_inf: s <= 1 is the central cube, and level l,
l = 1, 2, ..., is the layer of cells between the cube surfaces
s = l and s = 1 + l (`CubedTopology`).  In a shell there is no central
cube and level 0 is the inner boundary, s = 1.  The parameter mesh is
straight-sided, so nodes of any order placed in it are exact, and
`cubed_map` then takes it to the reference body: with t = tan(pi/4 .)
by component, g = t(P / s) and d = g / |g|,

    the cube      x = a t(P)                      a the cube's half-width
    the blend     x = (1 - w) a g + w R_1 d       w from 0 to 1
    above         x = R d                         R through the spheres

where R_1 is the first sphere, w runs through the blend's fractions,
one for each of its levels, and R through the spheres' radii, one for
each level above, linearly within a level.  The tangent makes the
angular grid equiangular: across a block every cell subtends the same
angle, pi / (2 edge), in each lateral direction.  Every radius from the
first sphere outward is a sphere of the mesh, the nodes on it on it
exactly, and the map is smooth within every cell; it is only continuous
across the boundaries of the blocks, which are cell boundaries.  The
blend, between the cube's surface and the first sphere, belongs with
the cube to the innermost layer of the body, and its levels are not
spheres.

The radial knots follow a target size (`size_at`): each interface gives
a size on it growing linearly away from it (`InterfaceSizing.at`), and
the target at a radius is the smallest of them.  A layer takes as many
cells as the target asks for across it, the integral of dr / h rounded
up and at least one, placed at equal shares of that integral
(`equal_shares`, `sphere_knots`).  The blend is divided the same way
along the ray through the centre of a face, where it is thickest, and
then more finely where its cells would be long needles pointing at the
centre: the lateral size of a cell is fixed by `edge` and shrinks with
radius, so one cell across the blend is about half of `edge` times
longer radially than it is wide (`blend_elongation`,
`blend_fractions`).

A cell's nodes are in MFEM's order, the corners of a quadrilateral
counter-clockwise and a hexahedron's bottom face then its top, and in a
block the last local direction is the radial one.  The faces of a
level's surface point away from the centre (in the plane, run
counter-clockwise about it), as the interfaces of every mesh of this
library do.

Nothing here needs more than numpy, and nothing knows about units.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .spec import InterfaceSizing

__all__ = ["CubedTopology", "cubed_topology", "cubed_map", "size_at",
           "equal_shares", "sphere_knots", "blend_fractions", "blend_elongation",
           "cell_lengths", "aspect_ratios", "cell_shapes", "shared_faces",
           "corner_jacobians", "DIMENSIONS", "SHARE_NODES", "CHUNK"]

#: The dimensions a cubed sphere is made in: a ball or a shell, and
#: their disc or annulus.
DIMENSIONS = (3, 2)

#: The number of radii across a span at which 1 / h is sampled to
#: integrate it.
SHARE_NODES = 4097

#: The number of cells measured at a time, which bounds the memory a
#: measure of a large mesh takes.
CHUNK = 65536

#: The corners of a cell as offsets on the lattice, in MFEM's order, by
#: dimension.
_CORNERS = {2: np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            3: np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])}

#: The edges of a cell along each of its local directions, as pairs of
#: its corners, by dimension.
_EDGES = {2: np.array([[[0, 1], [3, 2]], [[0, 3], [1, 2]]]),
          3: np.array([[[0, 1], [3, 2], [4, 5], [7, 6]],
                       [[0, 3], [1, 2], [4, 7], [5, 6]],
                       [[0, 4], [1, 5], [2, 6], [3, 7]]])}

#: The faces of a cell, each as its corners, by dimension.
_FACES = {2: np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
          3: np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])}

#: Each corner of a cell with its neighbours along the local directions,
#: so that the edges from it are a right-handed frame, by dimension.
_FRAMES = {2: np.array([[0, 1, 3], [1, 2, 0], [2, 3, 1], [3, 0, 2]]),
           3: np.array([[0, 1, 3, 4], [1, 2, 0, 5], [2, 3, 1, 6], [3, 0, 2, 7],
                        [4, 7, 5, 0], [5, 4, 6, 1], [6, 5, 7, 2], [7, 6, 4, 3]])}


def _require_dimension(dimension: int) -> int:
    """`dimension`, refused unless it is one of `DIMENSIONS`."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"a cubed sphere is made in 3 dimensions or 2, got "
                         f"{dimension}")
    return int(dimension)


@dataclass(frozen=True, eq=False)
class CubedTopology:
    """The cells of a cubed sphere in its parameter space (module
    docstring)."""

    #: The dimension, 3 or 2.
    dimension: int
    #: The number of cells along an edge of the central cube.
    edge: int
    #: The number of levels of cells above the cube's surface.
    levels: int
    #: Whether there is no central cube: a shell, or an annulus.
    hollow: bool
    #: The points in the parameter space, (n, dimension).
    points: np.ndarray
    #: The cells, (c, 8) hexahedra or (c, 4) quadrilaterals.
    cells: np.ndarray
    #: The level of each cell, (c,): 0 in the central cube, l in the
    #: l-th level above its surface.
    cell_levels: np.ndarray
    #: The points of the cube's surface, as indices into those of a
    #: full lattice, (p,).
    surface: np.ndarray
    #: The faces of one level's surface as indices into its points,
    #: (f, 4) or (f, 2), block by block.
    ring: np.ndarray

    def nodes_at(self, level: int) -> np.ndarray:
        """The points on the surface s = 1 + `level`, as indices into
        `points`, (p,), in the order `ring` indexes them."""
        if not 0 <= level <= self.levels:
            raise ValueError(f"the mesh has levels 0..{self.levels}, got {level}")
        p = self.surface.size
        if self.hollow:
            return level * p + np.arange(p)
        if level == 0:
            return self.surface
        return (self.edge + 1) ** self.dimension + (level - 1) * p + np.arange(p)

    def faces(self, level: int, *, inward: bool = False) -> np.ndarray:
        """The faces of the surface s = 1 + `level`, (f, 4) or (f, 2), as
        indices into `points`: pointing away from the centre, or towards
        it where `inward`."""
        faces = self.nodes_at(level)[self.ring]
        return faces[:, ::-1] if inward else faces

    @property
    def block(self) -> int:
        """The number of faces of a level's surface on one block."""
        return self.edge ** (self.dimension - 1)


def _ring(edge: int, dimension: int) -> np.ndarray:
    """The faces of the surface of the lattice of a cube of `edge` cells
    an edge, as lattice indices, (f, 2**(dimension - 1)), block by block,
    each pointing away from the centre."""
    n, d = edge, dimension
    strides = (n + 1) ** np.arange(d)
    cells = np.stack(np.meshgrid(*([np.arange(n)] * (d - 1)), indexing="ij"),
                     axis=-1).reshape(-1, d - 1)
    corners = _CORNERS[d - 1] if d == 3 else np.array([[0], [1]])
    out = []
    for axis in range(d):
        lateral = [(axis + k) % d for k in range(1, d)]
        for side in (0, 1):
            index = np.zeros((cells.shape[0], corners.shape[0], d), dtype=np.int64)
            index[..., axis] = side * n
            index[..., lateral] = cells[:, None, :] + corners[None, :, :]
            faces = index @ strides
            # the lateral axes follow the normal cyclically in three
            # dimensions, so a face on the far side points outward as it
            # stands; in the plane a side's tangent is counter-clockwise
            # on the far side of x and the near side of y
            outward = side == 1 if d == 3 else (side == 1) == (axis == 0)
            out.append(faces if outward else faces[:, ::-1])
    return np.concatenate(out)


def cubed_topology(edge: int, levels: int, *, dimension: int = 3,
                   hollow: bool = False) -> CubedTopology:
    """The cells of a cubed sphere of `edge` cells along an edge of its
    central cube and `levels` levels above the cube's surface (module
    docstring); `hollow` leaves the central cube out, and then there is
    at least one level."""
    d = _require_dimension(dimension)
    n, levels = int(edge), int(levels)
    if n < 1:
        raise ValueError(f"edge must be at least 1, got {edge}")
    if levels < (1 if hollow else 0):
        raise ValueError(f"a {'hollow ' if hollow else ''}cubed sphere has at "
                         f"least {1 if hollow else 0} levels, got {levels}")
    n1 = n + 1
    strides = n1 ** np.arange(d)
    ids = np.arange(n1 ** d)
    index = np.stack([(ids // s) % n1 for s in strides], axis=1)
    lattice = -1.0 + 2.0 * index / n
    surface = np.flatnonzero(np.any((index == 0) | (index == n), axis=1))
    place = np.full(n1 ** d, -1, dtype=np.int64)
    place[surface] = np.arange(surface.size)
    ring = place[_ring(n, d)]
    scales = 1.0 + np.arange(0 if hollow else 1, levels + 1)
    above = (scales[:, None, None] * lattice[surface][None]).reshape(-1, d)
    points = above if hollow else np.concatenate([lattice, above])

    base = np.stack(np.meshgrid(*([np.arange(n)] * d), indexing="ij"),
                    axis=-1).reshape(-1, d)
    cube = (base[:, None, :] + _CORNERS[d][None]) @ strides
    topology = CubedTopology(
        dimension=d, edge=n, levels=levels, hollow=bool(hollow), points=points,
        cells=np.empty((0, 2 ** d), dtype=np.int64),
        cell_levels=np.empty(0, dtype=np.int64), surface=surface, ring=ring)
    cells = [] if hollow else [cube]
    cell_levels = [] if hollow else [np.zeros(cube.shape[0], dtype=np.int64)]
    for level in range(1, levels + 1):
        below, on = topology.faces(level - 1), topology.faces(level)
        # a hexahedron stands on a face of the surface below; a
        # quadrilateral on a side of it, taken backwards so that its
        # corners run counter-clockwise
        cells.append(np.concatenate([below, on], axis=1) if d == 3
                     else np.concatenate([below[:, ::-1], on], axis=1))
        cell_levels.append(np.full(ring.shape[0], level, dtype=np.int64))
    return CubedTopology(
        dimension=d, edge=n, levels=levels, hollow=bool(hollow), points=points,
        cells=np.concatenate(cells), cell_levels=np.concatenate(cell_levels),
        surface=surface, ring=ring)


def cubed_map(points: ArrayLike, *, spheres: ArrayLike,
              blend: ArrayLike | None = None, half_width: float = 0.0
              ) -> np.ndarray:
    """Points of the parameter space as points of the reference body
    (module docstring), (n, d) to (n, d).

    `spheres` are the radii of the spheres of the mesh, increasing.
    With `blend`, the fractions w of the blend's levels from 0 on the
    cube's surface to 1 on the first sphere, the body is a ball: the
    cube of half-width `half_width` is s <= 1, the blend the levels
    1..B for its B + 1 fractions, and sphere k is s = 1 + B + k.
    Without it the body is a shell and sphere k is s = 1 + k.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    spheres = np.atleast_1d(np.asarray(spheres, dtype=float))
    s = np.abs(P).max(axis=1)
    quarter = 0.25 * np.pi
    if blend is None:
        fractions, a = np.ones(1), 0.0
    else:
        fractions, a = np.atleast_1d(np.asarray(blend, dtype=float)), float(half_width)
    x = a * np.tan(quarter * P)
    out = s > 1.0 if blend is not None else np.ones(s.size, dtype=bool)
    g = np.tan(quarter * P[out] / s[out, None])
    d = g / np.linalg.norm(g, axis=1, keepdims=True)
    level = s[out] - 1.0
    w = np.interp(level, np.arange(fractions.size), fractions)[:, None]
    R = np.interp(level - (fractions.size - 1), np.arange(spheres.size),
                  spheres)[:, None]
    x[out] = (1.0 - w) * a * g + w * R * d
    return x


# ------------------------------------------------------------ the knots

def size_at(r: ArrayLike, radii: Sequence[float],
            sizings: Sequence[InterfaceSizing]) -> np.ndarray:
    """The target size at the radii `r`: the smallest, over the
    interfaces at `radii`, of the size each one's sizing gives at its
    distance (`InterfaceSizing.at`)."""
    r = np.asarray(r, dtype=float)
    if not len(radii) or len(radii) != len(sizings):
        raise ValueError("a target size needs one sizing for each interface, and "
                         "at least one")
    return np.min([sizing.at(np.abs(r - float(radius)))
                   for radius, sizing in zip(radii, sizings)], axis=0)


def equal_shares(lo: float, hi: float, radii: Sequence[float],
                 sizings: Sequence[InterfaceSizing], *, count: int | None = None
                 ) -> np.ndarray:
    """The knots that divide [lo, hi] into cells of the target size.

    The number of cells the target asks for is N, the integral of
    dr / h across the span; the span takes N rounded up, at least one,
    or `count` where given, and its knots sit at equal shares of that
    integral, so that every cell is as many target sizes thick as every
    other.  Returns the knots from `lo` to `hi`, both included.
    """
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError(f"a span runs upward, got [{lo:g}, {hi:g}]")
    r = np.linspace(lo, hi, SHARE_NODES)
    density = 1.0 / size_at(r, radii, sizings)
    share = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(r))])
    if count is None:
        # a span a whole number of sizes thick takes that many cells,
        # whatever the last digit of the quadrature
        count = max(1, int(np.ceil(share[-1] - 1e-6)))
    knots = np.interp(share[-1] * np.arange(count + 1) / count, share, r)
    knots[0], knots[-1] = lo, hi
    return knots


def sphere_knots(boundaries: ArrayLike, radii: Sequence[float],
                 sizings: Sequence[InterfaceSizing]) -> np.ndarray:
    """The radii of the spheres of a mesh whose layers lie between
    `boundaries`, from the first sphere outward: every boundary from the
    first sphere on, and between each pair the knots the target size
    asks for (`equal_shares`).  The first sphere is the innermost
    boundary above the centre: the second boundary of a ball, whose
    innermost layer is the cube and the blend, and the first of a
    shell."""
    b = np.asarray(boundaries, dtype=float)
    first = 0 if b[0] > 0.0 else 1
    knots = [b[first:first + 1]]
    for lo, hi in zip(b[first:-1], b[first + 1:]):
        knots.append(equal_shares(lo, hi, radii, sizings)[1:])
    return np.concatenate(knots)


def blend_elongation(edge: int, fractions: ArrayLike, *, half_width: float,
                     radius: float, dimension: int = 3) -> float:
    """How long the blend's cells are radially against their width, at
    its worst: the largest, over the cells of the blend with the given
    `fractions`, of a cell's radial length over the smaller of its
    lateral ones (`cell_lengths`).  The cube has half-width `half_width`
    and the first sphere radius `radius`."""
    d = _require_dimension(dimension)
    w = np.asarray(fractions, dtype=float)
    ring = _ring(edge, d)[:edge ** (d - 1)]        # one block: all are alike
    n1 = edge + 1
    index = np.stack([(ring // s) % n1 for s in n1 ** np.arange(d)], axis=-1)
    below = -1.0 + 2.0 * index / edge              # (f, corners, d), on s = 1
    scales = 1.0 + np.arange(w.size)
    lo = scales[:-1, None, None, None] * below[None]
    hi = scales[1:, None, None, None] * below[None]
    corners = (np.concatenate([lo, hi], axis=2) if d == 3
               else np.concatenate([lo[:, :, ::-1], hi], axis=2))
    points = cubed_map(corners.reshape(-1, d), spheres=[radius], blend=w,
                       half_width=half_width)
    lengths = cell_lengths(points, np.arange(points.shape[0]).reshape(-1, 2 ** d))
    return float(np.max(lengths[:, -1] / lengths[:, :-1].min(axis=1)))


def blend_fractions(edge: int, *, half_width: float, radius: float,
                    radii: Sequence[float] | None = None,
                    sizings: Sequence[InterfaceSizing] | None = None,
                    elongation: float | None = None, dimension: int = 3
                    ) -> np.ndarray:
    """The fractions w of the blend's levels, from 0 on the cube's surface
    to 1 on the first sphere.

    With `radii` and `sizings`, the blend is divided as a layer is,
    along the ray through the centre of a face, from the cube's
    half-width to the first sphere (`equal_shares`); without them it is
    one cell thick.  Where `elongation` is given, cells are added, at
    equal shares still, until none is more than that many times longer
    radially than it is wide (`blend_elongation`).
    """
    a, R = float(half_width), float(radius)
    if not 0.0 < a < R:
        raise ValueError(f"the cube's half-width must lie between zero and the "
                         f"first sphere's radius {R:g}, got {a:g}")
    if elongation is not None and not elongation > 0.0:
        raise ValueError(f"the cap on the blend's elongation must be positive, "
                         f"got {elongation}")
    sized = radii is not None and sizings is not None

    def fractions(count: int | None) -> np.ndarray:
        if sized:
            knots = equal_shares(a, R, radii, sizings, count=count)
            return (knots - a) / (R - a)
        return np.linspace(0.0, 1.0, (count or 1) + 1)

    w = fractions(None)
    if elongation is None:
        return w
    while True:
        worst = blend_elongation(edge, w, half_width=a, radius=R,
                                 dimension=dimension)
        if worst <= elongation:
            return w
        # a cell's radial length falls about as one over the count, so
        # most of the way there in one step, then one cell at a time
        count = w.size - 1
        w = fractions(max(count + 1, int(0.8 * count * worst / elongation)))


# ------------------------------------------------------------ the cells

def _corners_of(points: ArrayLike, cells: ArrayLike
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """The points, the cells and their dimension, the cells refused unless
    they are quadrilaterals or hexahedra by the number of their corners."""
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    d = {4: 2, 8: 3}.get(cells.shape[1] if cells.ndim == 2 else -1)
    if d is None:
        raise ValueError(f"cells are quadrilaterals or hexahedra by their corners, "
                         f"rows of 4 or 8, got shape {cells.shape}")
    return points, cells, d


def _blocks(count: int) -> Iterator[slice]:
    """Slices covering `count` cells, `CHUNK` at a time."""
    for start in range(0, count, CHUNK):
        yield slice(start, min(start + CHUNK, count))


def cell_lengths(points: ArrayLike, cells: ArrayLike) -> np.ndarray:
    """The length of each cell along each of its local directions,
    (c, d): the mean length of its straight edges along that direction,
    four of a hexahedron's and two of a quadrilateral's.  `cells` holds
    the corners in MFEM's order."""
    points, cells, d = _corners_of(points, cells)
    out = np.empty((cells.shape[0], d))
    for rows in _blocks(cells.shape[0]):
        ends = points[cells[rows][:, _EDGES[d]]]    # (c, d, edges, 2, sdim)
        out[rows] = np.linalg.norm(ends[..., 1, :] - ends[..., 0, :],
                                   axis=-1).mean(axis=-1)
    return out


def aspect_ratios(points: ArrayLike, cells: ArrayLike) -> np.ndarray:
    """The aspect ratio of each cell, (c,): its greatest length over its
    least (`cell_lengths`), one for a cube."""
    lengths = cell_lengths(points, cells)
    return lengths.max(axis=1) / lengths.min(axis=1)


def _frames(points: np.ndarray, cells: np.ndarray, d: int) -> np.ndarray:
    """The edges leaving each corner of each cell along its local
    directions, as the rows of a right-handed frame, (c, corners, d, d)."""
    at = points[cells[:, _FRAMES[d]]]               # (c, corners, 1 + d, sdim)
    return at[:, :, 1:, :d] - at[:, :, :1, :d]


def corner_jacobians(points: ArrayLike, cells: ArrayLike) -> np.ndarray:
    """The Jacobian determinant of each straight-sided cell at each of
    its corners, (c, corners): the signed volume, or area, of the frame
    of its edges there, positive throughout a cell whose corners are in
    MFEM's order and which does not fold."""
    points, cells, d = _corners_of(points, cells)
    out = np.empty(cells.shape)
    for rows in _blocks(cells.shape[0]):
        out[rows] = np.linalg.det(_frames(points, cells[rows], d))
    return out


def cell_shapes(points: ArrayLike, cells: ArrayLike) -> np.ndarray:
    """The shape of each cell, (c,): the signed inverse condition number
    of the Jacobian J of the straight-sided cell through its corners,
    d det(J)^(2/d) / |J|_F^2 in d dimensions, at its worst over its
    corners, where J is the frame of the edges leaving one.  It is 1
    for a cube or a square, falls towards 0 as a cell is stretched or
    sheared, and is not positive where a corner is turned inside out."""
    points, cells, d = _corners_of(points, cells)
    out = np.empty(cells.shape[0])
    for rows in _blocks(cells.shape[0]):
        J = _frames(points, cells[rows], d)
        det = np.linalg.det(J)
        shape = d * np.sign(det) * np.abs(det) ** (2.0 / d) / (J ** 2).sum(axis=(2, 3))
        out[rows] = shape.min(axis=1)
    return out


def shared_faces(cells: ArrayLike) -> np.ndarray:
    """How many cells each face of the mesh is a face of, (f,), one entry
    for each distinct face: two inside a conforming mesh and one on its
    boundary."""
    _, cells, d = _corners_of(np.empty((0, 3)), cells)
    faces = np.sort(cells[:, _FACES[d]].reshape(-1, _FACES[d].shape[1]), axis=1)
    return np.unique(faces, axis=0, return_counts=True)[1]
