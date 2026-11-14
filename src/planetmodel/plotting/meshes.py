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

A cutaway in three dimensions is drawn from the same pieces, put back
in space (`PlaneCut.in_space`), and the faces of the mesh kept beside
them; `clip_polygons` keeps the part of each polygon on one side of a
plane, so that the pieces and faces of a block cut from the body end on
its faces exactly instead of in whole elements.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["FLAT", "PlaneCut", "plane_cut", "mesh_section", "clip_polygons"]

#: The area, as a fraction of the square of its diameter, at or under which
#: a piece cut from a tetrahedron is no polygon but a cell touching the
#: plane, to rounding, and is left out.
FLAT = 1e-9


@dataclass(frozen=True, eq=False)
class PlaneCut:
    """Simplices cut by a plane (module docstring)."""

    #: A point of the plane, (3,): the origin of its coordinates.
    origin: np.ndarray
    #: The orthonormal pair (a, b) spanning the plane, (2, 3).
    axes: np.ndarray
    #: The pieces in the plane's coordinates (s, t): (p, 4, 2) polygons
    #: cut from tetrahedra, (p, 2, 2) segments cut from triangles.
    pieces: np.ndarray
    #: The simplex each piece was cut from, as a row of those given, (p,).
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
              axes: ArrayLike) -> PlaneCut:
    """The simplices cut by the plane through `origin` spanned by `axes`
    (module docstring).

    `points` are the vertices, (n, 3); `simplices` the corners of each
    element as indices into them, (m, 4) tetrahedra or (m, 3) triangles
    (a curved element's corners: its first columns as a mesh holds
    them); `origin` a point of the plane, (3,); `axes` two vectors
    spanning it, (2, 3), made orthonormal with the first one's direction
    kept.
    """
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
