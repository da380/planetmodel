"""Sections of a simplex mesh: its elements cut by a plane, and drawn.

`plane_cut` cuts simplices by a plane through `origin` spanned by an
orthonormal pair (a, b), the normal n = a x b: a tetrahedron the plane
crosses leaves a triangle or a quadrilateral, the convex polygon
through the points where the plane crosses its edges, and a triangle
leaves a segment.  Each piece is in the plane's own coordinates
(s, t) = ((x - origin).a, (x - origin).b), so that the cells of a
volume mesh cut by a plane tile the section of the body without overlap
and the faces of an interface or a fault cut by the same plane are the
curves it leaves there.  A simplex is given by its corners alone, so an
element of order two or more is cut as the straight-sided one through
its corners, its curved edges taken as chords.  The points are
whichever the caller holds: a mesh's reference positions, or those
positions moved by a mapping.

A corner on the plane counts as lying on its negative side, opposite
the normal, and a crossing at such a corner is that corner exactly: a
face lying in the plane is left once, by the cell on the normal's side
of it.  A simplex that only touches the plane leaves nothing: a
triangle at a corner, a tetrahedron at a corner or along an edge,
which to rounding is a polygon whose area is at most `FLAT` times the
square of its diameter (an edge of the mesh lying in the plane, or a
hair off it, would otherwise be drawn once for every cell around it).
A tetrahedron's polygon is held as four vertices in order around it, a
triangle's last vertex repeated.

`mesh_section` draws a cut on plain axes of equal aspect: the pieces of
tetrahedra as polygons, each the colour given its simplex, their edges
the element edges in the section; the pieces of triangles as line
segments.  matplotlib is imported where a section is drawn.

A mesh that lies in a plane is its own section: `in_plane` takes its
triangles whole, as polygons, and its line elements whole, as
segments, into the same `PlaneCut`, so that `mesh_section` draws a
two-dimensional mesh (a disc, or a box's section in the (x, z) plane)
and the curves of its interfaces and faces as it draws a cut of a
three-dimensional one, and the faces of a surface of a volume mesh (a
box's top) seen square on.

A cutaway in three dimensions is drawn from the same pieces, put back
in space (`PlaneCut.in_space`), and the faces of the mesh kept beside
them; `clip_polygons` keeps the part of each polygon on one side of a
plane, so that the pieces and faces of a block cut from the body end on
its faces exactly instead of in whole elements.  `mesh_block` draws such
a block of a volume mesh on three-dimensional axes, the box (x0, x1) x
(y0, y1) x (z0, z1): the cells' sections by its three walls away from
the viewer, clipped to it and each the colour of its cell, with the
curves other simplices (an interface's faces) leave on them; whatever
the caller draws inside the block afterwards (a fault's faces, clipped
by `block_planes`, its curves) is seen against those walls, drawn in
the order drawn, matplotlib's own depth sorting off.

A fault's curves are drawn by kind (`fault_curves`), each kind in its
own line style (`FAULT_CURVE_STYLES`, one look for every picture): its
buried tip, where its slip is pinned, dashed; its free edge on the free
surface solid and heavy; its junction with another fault dotted; its
face edge, where it leaves a box through the base or a side, dash-dot.
Each kind's edges are joined end to end into polylines first, broken
where three or more meet, so that a dash runs along a curve rather
than starting again at every edge.  On three-dimensional axes they are
drawn in space, on plain axes projected onto a plane, the coordinates
`plane_cut` gives.

A cut shows a mesh's elements as the plane leaves them, not their
shape: a plane passing near a corner or along an edge of a well-shaped
tetrahedron cuts a thin shard from it, so a cut of any tetrahedral mesh
shows shards beside its polygons.  Its geometry is exact: its pieces
tile the section, and the curves of an interface or a fault cut by the
same plane run along their edges.  `plane_cut(..., kind="crinkle")`
shows the elements whole instead, as ParaView's crinkle slice does, the
cells the plane passes through seen from its positive side: the cells
with a corner strictly on its negative side are kept, those it passes
through and those behind it (a corner on the plane, here, is on
neither side), and the faces between a kept cell and one with no
corner behind the plane, with the faces on the mesh's boundary of the
kept cells the plane passes through, are the stepped surface of whole
faces over the section, which it covers to its edge.  They are
projected onto the plane, each a triangle held as `in_plane` holds
one, its kept cell its `index`; only the faces turned towards the
positive side are taken, in order from the farthest, so that a nearer
face is drawn over a farther one.  Its triangles are the elements'
faces as they are, foreshortened by their tilt to the plane, but its
surface is stepped, lying up to an element in front of the plane, so
the curves an interface or a fault leaves in the plane, drawn over it
from their own cut, do not run along its edges, and the colours of the
cells change along a jagged line beside them.  The shape of the
elements is measured, not read from either (`mesh3d.cell_quality`); the
faces of a mesh lying in a plane, a box's side or top, or a fault's own
faces seen square on (`in_plane`), are its elements as they are.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["FLAT", "PlaneCut", "plane_cut", "in_plane", "mesh_section",
           "clip_polygons", "block_planes", "mesh_block", "fault_curves",
           "FAULT_CURVE_STYLES", "BLOCK_VIEW"]

#: The area, as a fraction of the square of its diameter, at or under which
#: a piece cut from a tetrahedron is no polygon but a cell touching the
#: plane, to rounding, and is left out.
FLAT = 1e-9

#: The line style of each kind of a fault's curves (module docstring), by
#: the names of `mesh3d.manifest.FLAT_FAULT_CURVES`.
FAULT_CURVE_STYLES = {"tip": {"linestyle": (0, (3, 2)), "linewidth": 1.5},
                      "free_edge": {"linestyle": "-", "linewidth": 3.0},
                      "junction": {"linestyle": (0, (1, 1.2)), "linewidth": 3.0},
                      "face_edge": {"linestyle": (0, (6, 1.5, 1, 1.5)),
                                    "linewidth": 2.5}}

#: The (elevation, azimuth) in degrees a block is viewed from by default,
#: by `mesh_block` and by `flat.block`: from above its corner of least y
#: and greatest x, matplotlib's own default azimuth.
BLOCK_VIEW = (25.0, -60.0)


@dataclass(frozen=True, eq=False)
class PlaneCut:
    """Simplices cut by a plane (module docstring)."""

    #: A point of the plane, (3,): the origin of its coordinates.
    origin: np.ndarray
    #: The orthonormal pair (a, b) spanning the plane, (2, 3).
    axes: np.ndarray
    #: The pieces in the plane's coordinates (s, t): (p, 4, 2) polygons
    #: cut from tetrahedra, (p, 2, 2) segments cut from triangles; or, of
    #: a mesh in the plane (`in_plane`), its triangles and line elements;
    #: or the projected faces of a crinkle.
    pieces: np.ndarray
    #: The simplex each piece was cut from, as a row of those given, (p,):
    #: of a crinkle, the kept cell whose face it is.
    index: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        """The unit normal a x b, (3,): the plane's positive side."""
        return np.cross(self.axes[0], self.axes[1])

    def coordinates(self, points: ArrayLike) -> np.ndarray:
        """The plane's coordinates (s, t) of points, (..., 3) -> (..., 2):
        of their projections onto the plane."""
        x = np.asarray(points, dtype=float)
        if x.shape[-1:] != (3,):
            raise ValueError(f"points have shape (..., 3), got {x.shape}")
        return (x - self.origin) @ self.axes.T

    def in_space(self) -> np.ndarray:
        """The pieces as points in space, (p, k, 3): origin + s a + t b,
        for drawing a cut on three-dimensional axes."""
        return self.origin + self.pieces @ self.axes


def _plane_axes(axes: ArrayLike) -> np.ndarray:
    """The pair (a, b) made orthonormal, a's direction kept and b's
    component along it removed; refused where they are parallel."""
    pair = np.asarray(axes, dtype=float)
    if pair.shape != (2, 3) or not np.all(np.isfinite(pair)):
        raise ValueError(f"axes are two finite vectors, (2, 3), got {pair.shape}")
    a, b = pair
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("the axes are not zero vectors")
    a = a / na
    b = b - (b @ a) * a
    if np.linalg.norm(b) <= 1e-12 * nb:
        raise ValueError("the axes are parallel: they span no plane")
    return np.stack([a, b / np.linalg.norm(b)])


def plane_cut(points: ArrayLike, simplices: ArrayLike, origin: ArrayLike,
              axes: ArrayLike, *, kind: str = "cut") -> PlaneCut:
    """The simplices cut by the plane through `origin` spanned by `axes`
    (module docstring).

    `points` are the vertices, (n, 3); `simplices` the corners of each
    element as indices into them, (m, 4) tetrahedra or (m, 3) triangles
    (a curved element's corners: its first columns as a mesh holds
    them); `origin` a point of the plane, (3,); `axes` two vectors
    spanning it, (2, 3), made orthonormal with the first one's direction
    kept.  `kind` is "cut", the simplices cut by the plane, or, of
    tetrahedra, "crinkle", the cells it passes through seen whole from
    its positive side (module docstring).
    """
    if kind not in ("cut", "crinkle"):
        raise ValueError(f"kind is 'cut' or 'crinkle', got {kind!r}")
    x = np.asarray(points, dtype=float)
    s = np.asarray(simplices)
    o = np.asarray(origin, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points have shape (n, 3), got {x.shape}")
    if s.ndim != 2 or s.shape[1] not in (3, 4) or not (
            np.issubdtype(s.dtype, np.integer) or s.size == 0):
        raise ValueError(f"simplices are rows of 3 or 4 corner indices, got "
                         f"{s.shape} of {s.dtype}")
    s = s.astype(np.int64)
    if s.size and (s.min() < 0 or s.max() >= len(x)):
        raise ValueError(f"simplices index vertices outside 0..{len(x) - 1}")
    if o.shape != (3,) or not np.all(np.isfinite(o)):
        raise ValueError(f"origin is a finite point, (3,), got {o.shape}")
    ab = _plane_axes(axes)
    k = s.shape[1]
    if kind == "crinkle":
        if k != 4:
            raise ValueError("a crinkle is a section of tetrahedra; cut a surface's "
                             "triangles with kind='cut'")
        return _crinkle(x, s, o, ab)
    d = (x - o) @ np.cross(ab[0], ab[1])
    D = d[s]
    above = D > 0.0
    count = np.count_nonzero(above, axis=1)
    rows = np.flatnonzero((count > 0) & (count < k))
    S, D, above = s[rows], D[rows], above[rows]
    crossings = []
    for i, j in combinations(range(k), 2):
        # each crossing taken from the corner above the plane
        p = np.where(above[:, i], S[:, i], S[:, j])
        q = np.where(above[:, i], S[:, j], S[:, i])
        dp = np.where(above[:, i], D[:, i], D[:, j])
        dq = np.where(above[:, i], D[:, j], D[:, i])
        crosses = above[:, i] != above[:, j]
        w = np.divide(dp, dp - dq, out=np.zeros_like(dp), where=crosses)
        at = x[p] + w[:, None] * (x[q] - x[p])
        at = np.where((dq == 0.0)[:, None], x[q], at)
        crossings.append(np.where(crosses[:, None], at, np.nan))
    st = (np.stack(crossings, axis=1) - o) @ ab.T             # (r, pairs, 2)
    valid = ~np.isnan(st[..., 0])
    if k == 4:
        centre = (np.where(valid[..., None], st, 0.0).sum(axis=1)
                  / np.count_nonzero(valid, axis=1)[:, None])
        rel = st - centre[:, None]
        angle = np.where(valid, np.arctan2(rel[..., 1], rel[..., 0]), np.inf)
        order = np.argsort(angle, axis=1)[:, :4]
        pieces = np.take_along_axis(st, order[..., None], axis=1)
        three = np.count_nonzero(valid, axis=1) == 3
        pieces[three, 3] = pieces[three, 2]
        s, t = pieces[..., 0], pieces[..., 1]
        area = 0.5 * np.abs(np.sum(s * np.roll(t, -1, axis=1)
                                   - np.roll(s, -1, axis=1) * t, axis=1))
        spread = pieces[:, :, None] - pieces[:, None, :]
        diameter2 = np.max(np.sum(spread ** 2, axis=-1), axis=(1, 2),
                           initial=0.0)
        flat = area <= FLAT * diameter2
    else:
        order = np.argsort(~valid, axis=1, kind="stable")[:, :2]
        pieces = np.take_along_axis(st, order[..., None], axis=1)
        flat = np.all(pieces[:, 0] == pieces[:, 1], axis=1)
    return PlaneCut(origin=o, axes=ab, pieces=pieces[~flat], index=rows[~flat])


#: A tetrahedron's faces by its corners, and the corner opposite each.
_FACES = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
_OPPOSITE = np.array([3, 2, 1, 0])


def _crinkle(x: np.ndarray, cells: np.ndarray, o: np.ndarray,
             ab: np.ndarray) -> PlaneCut:
    """The crinkle of tetrahedra (module docstring), the arguments
    checked: the faces between a cell with a corner strictly on the
    plane's negative side and one with none, and the faces on the mesh's
    boundary of such a cell the plane passes through, turned towards the
    positive side, projected, farthest first."""
    n = np.cross(ab[0], ab[1])
    d = (x - o) @ n
    kept = np.any(d[cells] < 0.0, axis=1)
    crossed = kept & np.any(d[cells] > 0.0, axis=1)
    faces = cells[:, _FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(cells)), 4)
    apex = cells[:, _OPPOSITE].reshape(-1)
    key = np.sort(faces, axis=1)
    order = np.lexsort(key.T[::-1])
    key, faces, owner, apex = key[order], faces[order], owner[order], apex[order]
    # a face of two cells is two rows, adjacent once sorted
    same = np.all(key[1:] == key[:-1], axis=1)
    first = np.flatnonzero(same)
    step = kept[owner[first]] != kept[owner[first + 1]]
    row = np.where(kept[owner[first[step]]], first[step], first[step] + 1)
    alone = np.ones(len(key), dtype=bool)
    alone[first], alone[first + 1] = False, False
    row = np.concatenate([row, np.flatnonzero(alone & crossed[owner])])
    p = x[faces[row]]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", x[apex[row]] - p[:, 0], normal) > 0.0
    towards = np.where(inward, -1.0, 1.0) * (normal @ n) > 0.0
    depth = d[faces[row]].mean(axis=1)
    shown = np.flatnonzero(towards)
    shown = shown[np.argsort(depth[shown], kind="stable")]
    st = (p[shown] - o) @ ab.T
    return PlaneCut(origin=o, axes=ab, pieces=np.concatenate([st, st[:, 2:]], axis=1),
                    index=owner[row[shown]])


def in_plane(points: ArrayLike, simplices: ArrayLike, origin: ArrayLike,
             axes: ArrayLike, *, rtol: float = 1e-9) -> PlaneCut:
    """The simplices of a mesh lying in the plane through `origin` spanned
    by `axes`, each kept whole as a piece of a `PlaneCut` (module
    docstring): a triangle as a polygon, its last vertex repeated, a line
    element as a segment.

    `points` are the vertices, (n, 3); `simplices` the corners of each
    element as indices into them, (m, 3) triangles or (m, 2) line
    elements (a curved element's corners: its first columns as a mesh
    holds them); `origin` and `axes` as for `plane_cut`.  Refused where a
    corner lies off the plane by more than `rtol` times the largest
    distance between the corners given.
    """
    x = np.asarray(points, dtype=float)
    s = np.asarray(simplices)
    o = np.asarray(origin, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points have shape (n, 3), got {x.shape}")
    if s.ndim != 2 or s.shape[1] not in (2, 3) or not (
            np.issubdtype(s.dtype, np.integer) or s.size == 0):
        raise ValueError(f"simplices are rows of 2 or 3 corner indices, got "
                         f"{s.shape} of {s.dtype}")
    s = s.astype(np.int64)
    if s.size and (s.min() < 0 or s.max() >= len(x)):
        raise ValueError(f"simplices index vertices outside 0..{len(x) - 1}")
    if o.shape != (3,) or not np.all(np.isfinite(o)):
        raise ValueError(f"origin is a finite point, (3,), got {o.shape}")
    ab = _plane_axes(axes)
    corners = x[s]
    off = np.abs((corners - o) @ np.cross(ab[0], ab[1]))
    size = float(np.ptp(corners.reshape(-1, 3), axis=0).max()) if s.size else 0.0
    if s.size and off.max() > rtol * size:
        raise ValueError(f"a corner lies {off.max():.3g} off the plane: the "
                         "simplices are not in it")
    st = (corners - o) @ ab.T
    if s.shape[1] == 3:
        st = np.concatenate([st, st[:, 2:]], axis=1)
    return PlaneCut(origin=o, axes=ab, pieces=st, index=np.arange(len(s)))


def mesh_section(ax: Any, cut: PlaneCut, *, colours: Any = None,
                 **style: Any) -> Any:
    """Draw a cut on plain axes by the rules of the module docstring,
    setting their aspect equal and their limits to what is drawn.

    `colours` is one colour for every piece, or one for each simplex
    that was cut (a row for each row of `plane_cut`'s `simplices`),
    indexed by the piece's `index`; None leaves matplotlib's own.  The
    pieces of tetrahedra are a `PolyCollection`, `colours` their faces,
    and those of triangles a `LineCollection`, `colours` their lines;
    `style` goes to the collection.  Returns the collection drawn.
    """
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_rgba_array

    if getattr(ax, "name", "") != "rectilinear":
        raise ValueError("a mesh section is drawn on plain axes")
    if not isinstance(cut, PlaneCut):
        raise TypeError(f"expected a PlaneCut, got {type(cut).__name__}")
    polygons = cut.pieces.shape[1] == 4
    if colours is not None:
        rgba = to_rgba_array(colours)
        if len(rgba) != 1:
            if cut.index.size and cut.index.max() >= len(rgba):
                raise ValueError(f"{len(rgba)} colours for simplices up to row "
                                 f"{int(cut.index.max())}: give one colour, or one "
                                 "for each simplex cut")
            rgba = rgba[cut.index]
        key = "facecolors" if polygons else "colors"
        style = {key: rgba} | style
    collection = (PolyCollection(cut.pieces, **style) if polygons
                  else LineCollection(cut.pieces, **style))
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.autoscale_view()
    return collection


def _clipped(poly: np.ndarray, n: np.ndarray, offset: float) -> np.ndarray | None:
    """The part of one polygon where x . n <= offset, or None."""
    d = poly @ n - offset
    inside = d <= 0.0
    if np.all(inside):
        return poly
    if not np.any(inside):
        return None
    kept = []
    closed = len(poly) > 2
    for a in range(len(poly) if closed else 1):
        b = (a + 1) % len(poly)
        if inside[a]:
            kept.append(poly[a])
        if inside[a] != inside[b]:
            kept.append(poly[a] + d[a] / (d[a] - d[b]) * (poly[b] - poly[a]))
    if not closed and inside[1]:
        kept.append(poly[1])
    return np.array(kept)


def clip_polygons(polygons: Iterable[ArrayLike], normals: ArrayLike,
                  offsets: ArrayLike) -> tuple[list[np.ndarray], np.ndarray]:
    """The part of each polygon where x . n <= offset for every plane
    (n, offset) given: the polygon itself where it lies wholly there, the
    polygon cut by each plane it crosses in turn (Sutherland and
    Hodgman's rule), and nothing where it lies wholly beyond one.

    `polygons` are convex polygons or segments, each (k, d) vertices in
    order (k may differ between them); `normals` one plane's normal (d,)
    or several, (m, d), and `offsets` their offsets, () or (m,), so that
    a box is the six planes bounding it.  Returns the parts, in order,
    and for each the index of the polygon it was cut from, (p,): the
    polygons left out are skipped.
    """
    n = np.atleast_2d(np.asarray(normals, dtype=float))
    c = np.atleast_1d(np.asarray(offsets, dtype=float))
    if n.ndim != 2 or c.shape != (len(n),):
        raise ValueError(f"normals are (d,) or (m, d) and offsets () or (m,), got "
                         f"{np.shape(normals)} and {np.shape(offsets)}")
    parts, index = [], []
    for k, polygon in enumerate(polygons):
        poly = np.asarray(polygon, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != n.shape[1]:
            raise ValueError(f"a polygon is (k, {n.shape[1]}) vertices, got "
                             f"{poly.shape}")
        for normal, offset in zip(n, c):
            poly = _clipped(poly, normal, offset)
            if poly is None:
                break
        else:
            parts.append(poly)
            index.append(k)
    return parts, np.array(index, dtype=np.int64)



def _bounds(bounds: ArrayLike) -> np.ndarray:
    """A block's bounds as a (3, 2) array, each lower below its upper."""
    b = np.asarray(bounds, dtype=float)
    if b.shape != (3, 2) or not np.all(np.isfinite(b)) or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError(f"bounds are ((x0, x1), (y0, y1), (z0, z1)), each lower "
                         f"below its upper, got {np.asarray(bounds).tolist()}")
    return b


def block_planes(bounds: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The six planes bounding a block, as `clip_polygons` takes them: the
    outward normals (6, 3) and offsets (6,), so that a point x lies in the
    block where x . n <= offset for all six."""
    b = _bounds(bounds)
    normals = np.concatenate([-np.eye(3), np.eye(3)])
    offsets = np.concatenate([-b[:, 0], b[:, 1]])
    return normals, offsets


def _eye(view: tuple[float, float]) -> np.ndarray:
    """The unit vector towards the viewer of matplotlib's view (elevation,
    azimuth), in degrees."""
    elevation, azimuth = np.deg2rad(np.asarray(view, dtype=float))
    return np.array([np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def mesh_block(ax: Any, points: ArrayLike, cells: ArrayLike, bounds: ArrayLike, *,
               colours: Any = None, lines: ArrayLike | None = None,
               line_style: Mapping[str, Any] | None = None,
               view: tuple[float, float] = BLOCK_VIEW, zoom: float = 1.0,
               **style: Any) -> Any:
    """Draw a block of a volume mesh on three-dimensional axes, by the rules
    of the module docstring, setting the axes' limits to the block, their
    box aspect to its proportions and their view.

    `points` are the vertices, (n, 3); `cells` the corners of the
    tetrahedra, (m, 4) (a curved element's first four columns);
    `bounds` the block, ((x0, x1), (y0, y1), (z0, z1)); `colours` one
    colour, or one for each cell, of the pieces cut from it; `lines`
    triangles, (k, 3) corner indices, whose sections by the walls are
    drawn over them as curves in `line_style` (a dark line by default);
    `view` the (elevation, azimuth) in degrees, which also chooses the
    walls drawn, those whose outward normal points away from the viewer;
    `zoom` matplotlib's zoom of the box; `style` goes to the walls'
    `Poly3DCollection`.  The axes' panes and grid are hidden, their
    ticks and labels kept.  Returns that collection.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if getattr(ax, "name", "") != "3d":
        raise ValueError("a block is drawn on three-dimensional axes, made with "
                         "subplot_kw={'projection': '3d'}")
    x = np.asarray(points, dtype=float)
    c = np.asarray(cells)
    b = _bounds(bounds)
    if c.ndim != 2 or c.shape[1] < 4:
        raise ValueError(f"cells are rows of 4 corner indices, got {c.shape}")
    eye = _eye(view)
    normals, offsets = block_planes(b)
    walls, owner, curves = [], [], []
    for normal, offset in zip(normals, offsets):
        if normal @ eye >= 0.0:
            continue
        axis = int(np.flatnonzero(normal)[0])
        origin = np.zeros(3)
        origin[axis] = b[axis, 0] if normal[axis] < 0 else b[axis, 1]
        # the plane's normal into the block: a wall on a face of the mesh
        # is then left by the cells inside, on its positive side
        pair = np.eye(3)[[k for k in range(3) if k != axis]]
        if np.cross(*pair) @ normal > 0.0:
            pair = pair[::-1]
        cut = plane_cut(x, c[:, :4], origin, pair)
        parts, index = clip_polygons(cut.in_space(), normals, offsets)
        walls += parts
        owner.append(cut.index[index])
        if lines is not None and len(lines):
            rim = plane_cut(x, np.asarray(lines)[:, :3], origin, pair)
            curves += clip_polygons(rim.in_space(), normals, offsets)[0]
    owner = np.concatenate(owner) if owner else np.empty(0, dtype=np.int64)
    if colours is not None:
        from matplotlib.colors import to_rgba_array
        rgba = to_rgba_array(colours)
        if len(rgba) != 1:
            if len(rgba) != len(c):
                raise ValueError(f"{len(rgba)} colours for {len(c)} cells: give one "
                                 "colour, or one for each cell")
            rgba = rgba[owner]
        style = {"facecolors": rgba} | style
    ax.computed_zorder = False
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
    collection = Poly3DCollection(walls, **style)
    ax.add_collection3d(collection)
    if curves:
        lines_style = {"colors": "#52514e", "linewidths": 1.0} | dict(line_style or {})
        ax.add_collection3d(Line3DCollection(curves, **lines_style))
    ax.set(xlim=tuple(b[0]), ylim=tuple(b[1]), zlim=tuple(b[2]))
    ax.set_box_aspect(tuple(float(v) for v in b[:, 1] - b[:, 0]), zoom=zoom)
    ax.view_init(elev=float(view[0]), azim=float(view[1]))
    return collection


def _polylines(edges: np.ndarray) -> list[np.ndarray]:
    """Edges (e, 2) of node indices joined end to end: each polyline the
    nodes of a run of edges in order, broken at a node where three or
    more meet, a closed loop repeating its first node."""
    adjacent: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)
    used: set[tuple[int, int]] = set()
    out = []

    def walk(a: int, b: int) -> list[int]:
        line = [a, b]
        used.add((min(a, b), max(a, b)))
        while len(adjacent[b]) == 2:
            nxt = [c for c in adjacent[b] if (min(b, c), max(b, c)) not in used]
            if not nxt:
                break
            a, b = b, nxt[0]
            used.add((min(a, b), max(a, b)))
            line.append(b)
        return line

    # from every end and branch first, then round what is left: loops
    starts = [n for n, nbrs in adjacent.items() if len(nbrs) != 2]
    for n in starts + list(adjacent):
        for m in adjacent[n]:
            if (min(n, m), max(n, m)) not in used:
                out.append(np.array(walk(n, m)))
    return out


def fault_curves(ax: Any, points: ArrayLike,
                 curves: Mapping[str, ArrayLike | None], *,
                 plane: tuple[ArrayLike, ArrayLike] | None = None,
                 color: Any = "#1a1a19", **style: Any) -> dict[str, Any]:
    """Draw a fault's curves by kind, by the rules of the module docstring.

    `points` are the vertices, (n, 3); `curves` maps a kind of
    `FAULT_CURVE_STYLES` to its edges, rows of node indices whose first
    two are an edge's corners (a curve group of `MeshArrays.curves`, or
    an array of `mesh3d.manifest.CurveEdges`), a kind given None or no
    edges being skipped.  On three-dimensional axes the edges are drawn
    in space; on plain axes projected onto `plane`, (origin, axes) as
    `plane_cut` takes them, which they need.  Each kind's collection is
    labelled with its name, its underscore a space, for a legend; `color`
    and `style` go to every collection, over the kind's own style.  Its
    edges are drawn as polylines (module docstring).  Returns the
    collection drawn of each kind.
    """
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    unknown = sorted(set(curves) - set(FAULT_CURVE_STYLES))
    if unknown:
        raise ValueError(f"the kinds of a fault's curves are "
                         f"{list(FAULT_CURVE_STYLES)}, got {unknown}")
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points have shape (n, 3), got {x.shape}")
    space = getattr(ax, "name", "") == "3d"
    if not space:
        if plane is None:
            raise ValueError("on plain axes a fault's curves are projected onto a "
                             "plane: give plane=(origin, axes)")
        origin = np.asarray(plane[0], dtype=float)
        pair = _plane_axes(plane[1])
    out = {}
    for kind, edges in curves.items():
        if edges is None or not len(edges):
            continue
        e = np.asarray(edges)
        if e.ndim != 2 or e.shape[1] < 2 or not np.issubdtype(e.dtype, np.integer):
            raise ValueError(f"the {kind} is rows of node indices, its corners "
                             f"first, got {e.shape} of {e.dtype}")
        lines = [x[run] for run in _polylines(e[:, :2])]
        options = (FAULT_CURVE_STYLES[kind] | {"label": kind.replace("_", " ")}
                   | {"colors": color} | style)
        if space:
            collection = Line3DCollection(lines, **options)
            ax.add_collection3d(collection)
        else:
            collection = LineCollection([(line - origin) @ pair.T for line in lines],
                                        **options)
            ax.add_collection(collection)
            ax.autoscale_view()
        out[kind] = collection
    return out
