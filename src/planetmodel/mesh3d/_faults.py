"""Faults built into the layered mesh: pulled back, made CAD, fragmented
in with the shells or the boxes of the layers, and labelled by local
sides.

A fault is defined where it physically is and the mesh is the reference
one, so each patch enters as its preimage under the geometry's mapping
(`fault.pull_back`), in the mesh's own coordinates, and lands on the
fault when the export moves the nodes.  Each preimage becomes an OCC
surface through its spline (`TensorSpline`): a plane where the spline is
the bilinear one through four coplanar corners, so that its crossings
with the spheres are exact circles (with a box's planes, straight
lines), and otherwise the B-spline surface the spline is, knots and
coefficients as they are.  Either is the preimage exactly.

What depends on the body's shape is asked of its layout (`_layout`):
the parts of its boundary a fault may leave it through (`Exit`), the
free surface (the outer sphere, a box's top) and, of a box, which
stands for an unbounded body, its base and sides too; where else no
fault may reach (a sphere's centre, a hollow one's hole); and its CAD.
Where a point lies against an exit is read through the geometry's own
coordinates: its radius, or its depth or x or y in a box.

An edge of a patch whose preimage lies nowhere more than `reach` inside
the free surface, or nowhere inside a face of a box by more than the
relief of that face over the edge (how far the mapping moves the face
there along its own coordinate: none at a side under a vertical
stretch), is carried across it: the patch is continued beyond that edge
by its own spline, a plane as its plane and a structured patch by the
end pieces of its interpolation carried on (`TensorSpline`), until the
preimage of the new edge clears the boundary everywhere by a hundredth
of the patch's extent that way (0.02 of the parameter's range tried
first, then half as far again each time up to half of it, an edge not
clear by then refused), and the fragment trims it there, so the fault
meets the boundary exactly wherever the relief puts the reference
position of its edge.  `reach` defaults to the relief of the surface
over the edge, the largest displacement the mapping gives its points
there along the layered coordinate, plus the CAD kernel's tolerance:
under the identity only an edge at the boundary or beyond it is carried
across.  An edge is carried across one exit at most.  Under a mapping
every edge and trial edge is judged by its points pulled back one by
one, and the patch continued (`fault.Region`) is pulled back as a
whole, so that the part carried is the preimage of the fault's own
continuation, as close to it as the rest is to the fault; under the
identity a patch is its own preimage and its spline is continued.

Where the mapping is smooth, so is the preimage of a smooth patch, and
the bicubic of `pull_back` follows it to its tolerance.  Where the
mapping's derivative jumps, at a knot (a level of the layered coordinate
the mapping declares in `knots`: under a stack of reliefs interpolated
layer by layer, every interface inside the body), the preimage of a
patch crossing it is creased along the crossing, which no bicubic
follows closely.  Such a patch, continued where it is carried, is built
in pieces: the curve along which its preimage crosses each knot is
found in its own parameters, v as a function of u (the root on each
line of v by the safeguarded secant, at nodes along u doubled until the
cubic through them meets a thousandth of the fault's `rtol` of the
range of v between them), and the patch is cut along the curves into
regions (`fault.Region`) each within one span between knots, where the
mapping is smooth, each pulled back as a patch of its own.  The pieces
of a patch share one control grid, the finest their pull-backs choose,
so that the edge along which two join is one curve of both, its control
points in common; and that edge, which lies on the knot's level at its
control points, lies on it between them to the cubic's error, which is
held within `JOIN_ATOL` with more nodes along u where needed: within
it the CAD kernel takes the edge to lie on the surface the knot is (an
interface, in the CAD), and makes it a curve of both pieces and of that
surface, finding no other crossing of either piece with it.  So the
fault stays one surface: the join is a curve inside it, bounding faces
of both pieces, neither tip nor edge, and those faces lie in the layers
on either side, as the faces of any fault crossing an interface do.  A
crossing that does not run along the patch's strike, from its edge
u = -1/2 to its edge u = 1/2 and across every line of v once, is
refused: such a fault is given as patches that each cross so.

Before any CAD, every preimage is sampled densely, the pieces of a
patch together (where they join they cross the interface), and refused
by name where it reaches where no fault may (through a sphere's centre, below a
hollow domain's inner boundary), lies wholly outside the geometry, has
a local extremum of its layered coordinate, over it or along one of its
edges, within a tenth of the local element size of an interface,
whatever it does elsewhere, or likewise of its distance from a face of
a box, or, carried across an exit or lying beyond it, reaches no deeper
inside than that: the fragment would turn each into slivers, a
tangency, or a loop too small to mesh.  A fault therefore never ends
exactly on an interface or a face: it crosses it or stops clear of it,
and a fault meant to end on one (a crustal fault bottoming at the Moho)
is given ending a little short of it or continuing a little past it.

The fragment trims the faults at the geometry's outer boundary
(`build_concentric`) or at the box (`build_box`), and each piece left
is found in the layer whose solid embeds it; a piece made by two
surfaces (overlapping faults or patches) is refused.  The curves of a
fault are read from the CAD: a curve bounding one of its faces once is
an edge of it, one bounding two of its faces, or embedded in one, lies
inside it (a seam where it crosses a sphere or a plane, or where
another fault ends on it); any of these it shares with another fault is
a junction; an edge on the free surface is a free edge, one on a face
of a box, its base or a side, a face edge (`manifest.FLAT_FAULT_CURVES`);
every other edge is its buried tip.

Once the mesh is oriented and curved, each fault's walls are classified
on it by the manifest's rule (`manifest.walls_of_faults`, on the
corners, the faces' vertex order carrying the orientation) and the
cells of each collar are given their attribute (`manifest.collars`):
moved out of their layer's volume into a discrete volume of their own,
whose physical group is the collar's attribute, so that the MSH file
and MFEM's reading of it carry it as the cells' element attribute.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from collections.abc import Mapping as MappingOf
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import gmsh
import numpy as np
from scipy.interpolate import make_interp_spline

from ..fault import (PULL_BACK_CONTROLS, SQUARE, Preimage, Region, TensorSpline,
                     inverse_of, pull_back)
from ..mapping import Mapping
from . import manifest
from ._geometry import ConcentricGeometry, default_atol
from ._layout import Exit, Layout
from ._tagging import Tagging
from .spec import EmbeddedFault, InterfaceSizing, MeshSpec, NearPoints

if TYPE_CHECKING:
    from ..fault import Patch
    from ._box import BoxGeometry

__all__ = ["PreparedFault", "FaultTagging", "prepare_faults", "add_surfaces",
           "settle_faults", "apply_fault_groups", "fault_refinements",
           "fault_entries", "curve_names", "FaultGroups", "CLEARANCE", "GAP",
           "JOIN_ATOL",
           "CollarTagging", "assign_collars", "collar_name"]

#: How far a carried edge must clear the boundary it is carried across, as
#: a fraction of the patch's extent across that edge.
CLEARANCE = 0.01

#: The nearest a fault may come to an interface it does not cross, as a
#: fraction of the element size there.
GAP = 0.1

#: How near the knot it joins along the edge where two pieces of a patch
#: join must lie, in the layered coordinate: a tenth of gmsh's geometric
#: tolerance (`Geometry.Tolerance`, 1e-8), within which the CAD kernel
#: takes a curve to lie on a surface, absolute as the kernel's are.
JOIN_ATOL = 1e-9

#: The farthest an edge is carried, as a fraction of the parameter range.
_LONGEST_CARRY = 0.5

#: Samples along each edge, and each way over a patch, at the least.
_EDGE_SAMPLES = 65
_PATCH_SAMPLES = 33

#: The nodes each way of the sample a patch's crossings of the mapping's
#: knots are found on; and the nodes along u of a crossing curve, doubled
#: (less one) from the first until the curve meets its tolerance, a
#: thousandth of the fault's `rtol` of the patch's range of v, at the
#: midpoints between them, the last refused.
_CROSSING_SAMPLES = 65
_CROSSING_NODES = (33, 65, 129, 257, 513)
_CROSSING_RTOL = 1e-3

#: The most secant steps a crossing is found in (`_roots`); and the width
#: of the bracket it stops at, relative to the range of v.
_CROSSING_STEPS = 200
_CROSSING_XTOL = 1e-15


@dataclass(frozen=True)
class PreparedFault:
    """One fault ready for the CAD: its name and spec, the preimages of
    the surfaces it is built in through, one for each patch or for each
    piece of a patch (module docstring), the splines built from them in
    the mesh's coordinates, the patch each is of (an index into the
    fault's patches) and the exits it was carried across (by their
    names)."""

    name: str
    item: EmbeddedFault
    preimages: tuple[Preimage, ...]
    splines: tuple[TensorSpline, ...]
    patches: tuple[int, ...]
    carried: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class FaultTagging:
    """What the fragment made of the faults, fault by fault in the spec's
    order: their faces, the layers (attributes) those lie in, their tip,
    free-edge, junction and face-edge curves, and the faults each meets;
    for every face, the preimage it is a piece of; the free surface, the
    entity a free edge lies on; and the entities of a box's base and
    sides, where a face edge lies."""

    names: tuple[str, ...]
    faces: tuple[tuple[int, ...], ...]
    layers: tuple[tuple[int, ...], ...]
    tips: tuple[tuple[int, ...], ...]
    free_edges: tuple[tuple[int, ...], ...]
    junctions: tuple[tuple[int, ...], ...]
    meets: tuple[tuple[str, ...], ...]
    surfaces: MappingOf[int, Patch]
    surface: int
    _: KW_ONLY
    #: Each fault's curves on a box's base and sides.
    face_edges: tuple[tuple[int, ...], ...] = ()
    #: The entities of a box's base and sides; none on a sphere.
    boundary: tuple[int, ...] = ()

    def curves(self, k: int) -> dict[str, tuple[int, ...]]:
        """Fault k's curves (from zero) by kind, `manifest.FLAT_FAULT_CURVES`'
        names: a fault with no face edge on a sphere."""
        return {"tip": self.tips[k], "free_edge": self.free_edges[k],
                "junction": self.junctions[k],
                "face_edge": self.face_edges[k] if self.face_edges else ()}


@dataclass(frozen=True)
class CollarTagging:
    """The collars `assign_collars` gave the cells: the numbering, the
    faults each fault ends on (by index, `CurveEdges.ends_on`), and the
    name of each collar's physical group by its attribute."""

    numbering: manifest.Collars
    ends_on: tuple[tuple[int, ...], ...]
    names: MappingOf[int, str]


# -------------------------------------------------------------- before CAD

#: The bounds of a patch's parameters a surface is built over, (a, b, c, d):
#: u from a to b and v from c to d, the square or beyond it.
type Bounds = tuple[float, float, float, float]

#: The bounds of the parameter square.
_SQUARE: Bounds = (*SQUARE, *SQUARE)


def _edge(bounds: Bounds, side: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of n points along one edge of the bounds: `side` is "u0",
    "u1", "v0" or "v1", the low or high end of u or v."""
    a, b, c, d = bounds
    s = np.linspace(0.0, 1.0, n)
    along_u, along_v = a + (b - a) * s, c + (d - c) * s
    return {"u0": (np.full(n, a), along_v), "u1": (np.full(n, b), along_v),
            "v0": (along_u, np.full(n, c)), "v1": (along_u, np.full(n, d))}[side]


def _middle(bounds: Bounds, side: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of n points along the middle line of the bounds across one
    edge: the line of v at the middle of u for an edge of v, and the line
    of u at the middle of v for an edge of u."""
    a, b, c, d = bounds
    s = np.linspace(0.0, 1.0, n)
    return ((0.5 * (a + b) + 0.0 * s, c + (d - c) * s) if side[0] == "v"
            else (a + (b - a) * s, 0.5 * (c + d) + 0.0 * s))


def _widened(bounds: Bounds, side: str, t: float) -> Bounds:
    """The bounds widened by t beyond one edge."""
    a, b, c, d = bounds
    return {"u0": (a - t, b, c, d), "u1": (a, b + t, c, d),
            "v0": (a, b, c - t, d), "v1": (a, b, c, d + t)}[side]


def _carry(spline: TensorSpline, side: str, t: float) -> TensorSpline:
    """The spline continued by t beyond one edge."""
    widths = {"u0": {"u": (t, 0.0)}, "u1": {"u": (0.0, t)},
              "v0": {"v": (t, 0.0)}, "v1": {"v": (0.0, t)}}[side]
    return spline.extended(**widths)


def _relief(mapping: Mapping, points: np.ndarray, exit: Exit) -> float:
    """How far the mapping moves the exit's points over `points`, moved
    onto it along its coordinate, along that coordinate: the largest
    change of the exit's coordinate (a radius, a depth, the x or y of a
    side) between those points and their images, the part of the
    displacement that carries the exit towards or away from the body.
    A displacement along the exit (a vertical stretch at a side of a
    box) moves none of it."""
    X = exit.onto(points)
    moved = exit.level(np.asarray(mapping(X), dtype=float))
    return float(np.max(np.abs(moved - exit.value)))


def _reaches(X: np.ndarray, exit: Exit, *, reach: float | None, relief: float,
             atol: float) -> bool:
    """Whether an edge whose preimage is the points X lies nowhere more
    than its reach inside the exit (module docstring): `reach` on the free
    surface where given, otherwise the relief of the exit over it plus
    `atol`."""
    limit = reach if reach is not None and exit.free else relief + atol
    return float(np.max(exit.inside(X))) <= limit


def _carry_width(edge: Callable[[float], np.ndarray], exit: Exit, *,
                 extent: float, span: float, what: str) -> float:
    """How far an edge is carried beyond itself, in its parameter, to clear
    the exit by `CLEARANCE` of `extent` everywhere: the first of 0.02 of
    the parameter's range `span`, then half as far again each time, up to
    `_LONGEST_CARRY` of it, that does; `edge(t)` gives the preimage of the
    edge carried by t.  None clearing it is refused."""
    longest = _LONGEST_CARRY * span
    t = 0.02 * span
    while not np.all(exit.beyond(edge(t), CLEARANCE * extent)):
        if t >= longest:
            raise ValueError(
                f"{what} meets the {exit.where()} at too shallow an angle to be "
                "carried across it cleanly: continued by half its extent, its edge "
                "still does not clear it; end it farther inside the body, or give "
                "a smaller reach")
        t = min(1.5 * t, longest)
    return t


def _length(points: np.ndarray) -> float:
    """The length of the polyline through the points (n, 3)."""
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=-1)))


def _carried_across(spline: TensorSpline, *, exits: Sequence[Exit],
                    reach: float | None, atol: float, what: str
                    ) -> tuple[TensorSpline, frozenset[str]]:
    """Under the identity, where a patch is its own preimage: the patch's
    spline with every edge that reaches an exit carried across it (module
    docstring), the spline continued, and the names of the exits it was
    carried across."""
    carried: set[str] = set()
    for side in ("u0", "u1", "v0", "v1"):
        bounds = (*spline.bounds[0], *spline.bounds[1])
        X = spline(*_edge(bounds, side, _EDGE_SAMPLES))
        for exit in exits:
            if not _reaches(X, exit, reach=reach, relief=0.0, atol=atol):
                continue
            extent = _length(spline(*_middle(bounds, side, _EDGE_SAMPLES)))
            a, b, c, d = bounds
            span = (d - c) if side[0] == "v" else (b - a)

            def edge(t: float, *, side: str = side) -> np.ndarray:
                trial = _carry(spline, side, t)
                return trial(*_edge((*trial.bounds[0], *trial.bounds[1]), side,
                                    _EDGE_SAMPLES))
            t = _carry_width(edge, exit, extent=extent, span=span, what=what)
            spline = _carry(spline, side, t)
            carried.add(exit.name)
            break
    return spline, frozenset(carried)


def _continuation(patch: Patch, *, exits: Sequence[Exit], reach: float | None,
                  mapping: Mapping, inverse: Callable[[np.ndarray], np.ndarray],
                  atol: float, what: str) -> tuple[Bounds, frozenset[str]]:
    """Under a mapping: the bounds of the patch's parameters it is built
    over, each edge whose preimage reaches an exit (module docstring)
    carried across it, the patch continued past the edge by its own
    spline, as `fault.Region` continues it; and the names of the exits it
    was carried across.  Each edge and each trial edge is pulled back
    point by point (`inverse`, the mapping's own), so that what is judged
    is the preimage of the continued patch itself."""
    spline = patch.spline()

    def preimage(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(inverse(spline(u, v)), dtype=float)

    bounds, carried = _SQUARE, set()
    for side in ("u0", "u1", "v0", "v1"):
        X = preimage(*_edge(bounds, side, _EDGE_SAMPLES))
        for exit in exits:
            if not _reaches(X, exit, reach=reach, relief=_relief(mapping, X, exit),
                            atol=atol):
                continue
            extent = _length(preimage(*_middle(bounds, side, _EDGE_SAMPLES)))
            a, b, c, d = bounds
            span = (d - c) if side[0] == "v" else (b - a)

            def edge(t: float, *, side: str = side, bounds: Bounds = bounds
                     ) -> np.ndarray:
                return preimage(*_edge(_widened(bounds, side, t), side,
                                       _EDGE_SAMPLES))
            t = _carry_width(edge, exit, extent=extent, span=span, what=what)
            bounds = _widened(bounds, side, t)
            carried.add(exit.name)
            break
    return bounds, frozenset(carried)


def _roots(f: Callable[[np.ndarray, np.ndarray], np.ndarray], U: np.ndarray,
           c: float, d: float, *, ftol: float) -> np.ndarray:
    """The root in v, between c and d, of f(u, v) at each u of U, f taking
    opposite signs at the two ends: by the secant through the bracket's
    ends, the end on the side the new point falls kept and its value
    halved where it was kept the step before (the Illinois rule), which
    keeps both ends closing in where f bends at the root (a crossing of
    a knot), as the plain false position does not.  Each root is held to
    where |f| is at most `ftol` or the bracket is `_CROSSING_XTOL` of
    d - c wide."""
    lo, hi = np.full(U.shape, c), np.full(U.shape, d)
    flo, fhi = f(U, lo), f(U, hi)
    x = np.where(np.abs(flo) < np.abs(fhi), lo, hi)
    kept = np.zeros(U.shape, dtype=int)
    todo = np.flatnonzero((np.abs(flo) > ftol) & (np.abs(fhi) > ftol))
    x[np.abs(flo) <= ftol], x[np.abs(fhi) <= ftol] = c, d
    xtol = _CROSSING_XTOL * (d - c)
    for _ in range(_CROSSING_STEPS):
        if todo.size == 0:
            return x
        l, h, fl, fh = lo[todo], hi[todo], flo[todo], fhi[todo]
        t = l - fl * (h - l) / (fh - fl)
        t = np.where((t > l) & (t < h), t, 0.5 * (l + h))
        ft = f(U[todo], t)
        x[todo] = t
        low = np.sign(ft) == np.sign(fl)
        lo[todo] = np.where(low, t, l)
        flo[todo] = np.where(low, ft, np.where(kept[todo] == -1, 0.5 * fl, fl))
        hi[todo] = np.where(low, h, t)
        fhi[todo] = np.where(low, np.where(kept[todo] == 1, 0.5 * fh, fh), ft)
        kept[todo] = np.where(low, 1, -1)
        done = (np.abs(ft) <= ftol) | (hi[todo] - lo[todo] <= xtol)
        todo = todo[~done]
    raise RuntimeError(f"the crossings of a knot were not found at {todo.size} "
                       f"points in {_CROSSING_STEPS} steps")


def _crossing(f: Callable[[np.ndarray, np.ndarray], np.ndarray], bounds: Bounds,
              *, ftol: float, tol: float, what: str) -> np.ndarray:
    """The values of the patch's v where f(u, v) = 0 at uniform nodes over
    its u: their number doubled (less one) through `_CROSSING_NODES` until
    the not-a-knot cubic through them is within `tol` of the roots at the
    midpoints between them; the nodes then and the midpoints together."""
    a, b, c, d = bounds
    m = _CROSSING_NODES[0]
    values = _roots(f, np.linspace(a, b, m), c, d, ftol=ftol)
    for m in _CROSSING_NODES:
        u = np.linspace(a, b, 2 * m - 1)
        middle = _roots(f, u[1::2], c, d, ftol=ftol)
        off = float(np.max(np.abs(make_interp_spline(u[::2], values, k=3)(u[1::2])
                                  - middle)))
        both = np.empty(2 * m - 1)
        both[::2], both[1::2] = values, middle
        if off <= tol:
            return both
        values = both
    raise ValueError(f"{what} crosses a knot of the mapping along a curve the "
                     f"cubic through {_CROSSING_NODES[-1]} points of it follows "
                     f"only to {off:.3g} of v, short of {tol:.3g}: the crossing is "
                     "not smooth")


def _pieces(patch: Patch, bounds: Bounds, *, inverse: Callable[[np.ndarray],
                                                                np.ndarray],
            level: Callable[[np.ndarray], np.ndarray], knots: Sequence[float],
            label: str, rtol: float, what: str
            ) -> tuple[list[Region], list[float]] | None:
    """The patch over the bounds split along every curve where its preimage
    crosses a knot of the mapping, a level of the layered coordinate
    (`level` of reference points) where the mapping's derivative may jump
    (module docstring): the regions between them in order down v, each
    within one span between knots, and the knot each joins the next
    along; None where it crosses none.  A crossing that does not run from
    the edge u = a to the edge u = b, crossing every line of v once, is
    refused."""
    a, b, c, d = bounds
    spline = patch.spline()

    def preimage(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(inverse(spline(u, v)), dtype=float)

    U, V = np.meshgrid(np.linspace(a, b, _CROSSING_SAMPLES),
                       np.linspace(c, d, _CROSSING_SAMPLES), indexing="ij")
    L = level(preimage(U, V))
    crossed = [k for k in sorted(knots) if float(L.min()) < k < float(L.max())]
    if not crossed:
        return None
    scale = float(np.max(np.abs(L)))
    curves = []
    for k in crossed:
        above = L > k
        once = np.count_nonzero(above[:, 1:] != above[:, :-1], axis=1) == 1
        if not np.all(once & (above[:, 0] != above[:, -1])):
            raise ValueError(
                f"{what} crosses the reference {label} {k:g}, a knot of the "
                "mapping where its derivative may jump, along a curve that does "
                "not run along its strike from one end of the patch to the other, "
                "crossing each line down its dip once: its preimage, creased "
                "along the crossing, is built in pieces split along such a curve "
                "alone; give the fault as patches that each cross it so")

        def f(u: np.ndarray, v: np.ndarray, *, k: float = k) -> np.ndarray:
            return level(preimage(u, v)) - k
        curves.append((_crossing(f, bounds, ftol=1e-13 * scale,
                                 tol=_CROSSING_RTOL * rtol * (d - c), what=what), k))
    curves.sort(key=lambda gk: float(np.mean(gk[0])))
    edges: list[float | np.ndarray] = [c, *(g for g, _ in curves), d]
    return ([Region(patch, u=(a, b), lower=lo, upper=hi)
             for lo, hi in zip(edges[:-1], edges[1:])], [k for _, k in curves])


def _joined(pre: Sequence[Preimage], joins: Sequence[float], *,
            level: Callable[[np.ndarray], np.ndarray]) -> float:
    """How far the edges along which consecutive pieces join lie off the
    knots they join along, in the layered coordinate: the largest over a
    sample four times as fine as the control nodes along them."""
    worst = 0.0
    for p, k in zip(pre, joins):
        n_u = p.patch.positions.shape[0]
        u = np.linspace(*SQUARE, 4 * (n_u - 1) + 1)
        edge = p.patch.cartesian(u, np.full(u.shape, SQUARE[1]))
        worst = max(worst, float(np.max(np.abs(level(edge) - k))))
    return worst


def _shared(regions: Sequence[Region], joins: Sequence[float], mapping: Mapping,
            *, controls: int | tuple[int, int] | None, rtol: float,
            level: Callable[[np.ndarray], np.ndarray], label: str,
            what: str) -> list[Preimage]:
    """The regions pulled back through one control grid, so that pieces
    joined along an edge have its points in common: `controls` where
    given, otherwise the finest of the grids `pull_back` chooses for them
    one by one, its nodes along u doubled (less one) until each edge
    where two join lies within `JOIN_ATOL` of the knot it joins along
    (`_joined`), up to the last of `PULL_BACK_CONTROLS`.  A join farther
    off is refused: the CAD kernel would cut the pieces where they pass
    through the knot's surface beside it, into slivers."""
    if controls is not None:
        pre = [pull_back(r, mapping, controls=controls) for r in regions]
    else:
        pre = [pull_back(r, mapping, rtol=rtol) for r in regions]
        grid = max(p.controls for p in pre)
        pre = [p if p.controls == grid else pull_back(r, mapping, controls=grid)
               for p, r in zip(pre, regions)]
        while (_joined(pre, joins, level=level) > JOIN_ATOL
               and grid[0] < PULL_BACK_CONTROLS[-1]):
            grid = (min(2 * grid[0] - 1, PULL_BACK_CONTROLS[-1]), grid[1])
            pre = [pull_back(r, mapping, controls=grid) for r in regions]
    off = _joined(pre, joins, level=level)
    if off > JOIN_ATOL:
        n_u, n_v = pre[0].controls
        raise ValueError(
            f"{what} is built in pieces joined where it crosses the reference "
            f"{label} {', '.join(f'{k:g}' for k in joins)}, a knot of the mapping, "
            f"and through {n_u} by {n_v} control nodes their join lies {off:.3g} "
            f"off it, beyond {JOIN_ATOL:g}, within which the CAD kernel takes it "
            "to lie on that surface; give more control nodes along u"
            + ("" if controls is None else ", or none"))
    return pre


def _built_through(patch: Patch, spec: MeshSpec, item: EmbeddedFault, *,
                   exits: Sequence[Exit], atol: float, what: str
                   ) -> tuple[list[Preimage], frozenset[str]]:
    """Under a mapping: the preimages a patch is built in through (module
    docstring), and the exits it was carried across.  A patch crossing no
    knot and carried across no exit is pulled back as it stands."""
    g = spec.geometry
    mapping = g.mapping
    inverse = inverse_of(mapping, patch.cartesian(0.0, 0.0)[None])
    bounds, carried = _continuation(patch, exits=exits, reach=item.reach,
                                    mapping=mapping, inverse=inverse, atol=atol,
                                    what=what)
    coords = g.coordinates

    def level(X: np.ndarray) -> np.ndarray:
        return np.asarray(coords.split(X)[0], dtype=float)
    split = _pieces(patch, bounds, inverse=inverse, level=level,
                    knots=[float(k) for k in getattr(mapping, "knots", ())],
                    label=coords.labels[0], rtol=item.rtol, what=what)
    if split is None and not carried:
        return [pull_back(patch, mapping, controls=item.controls,
                          rtol=item.rtol)], carried
    regions, joins = split or ([Region(patch, u=bounds[:2], lower=bounds[2],
                                       upper=bounds[3])], [])
    return _shared(regions, joins, mapping, controls=item.controls, rtol=item.rtol,
                   level=level, label=coords.labels[0], what=what), carried


def _extrema(d: np.ndarray) -> np.ndarray:
    """The local extrema of a grid of values (1D or 2D): each point no
    greater, or no less, than every neighbour it has."""
    low = np.pad(d, 1, constant_values=np.inf)
    high = np.pad(d, 1, constant_values=-np.inf)
    least, most = np.full(d.shape, np.inf), np.full(d.shape, -np.inf)
    for shift in np.ndindex(*(3,) * d.ndim):
        if shift == (1,) * d.ndim:
            continue
        window = tuple(slice(k, k + n) for k, n in zip(shift, d.shape))
        least = np.minimum(least, low[window])
        most = np.maximum(most, high[window])
    return (d <= least) | (d >= most)


def _near_misses(d: np.ndarray, limit: float) -> np.ndarray:
    """Where a sampled surface comes within `limit` of a sphere and turns
    there, touching it, grazing it, running beside it, or crossing it in
    a loop too small to mesh: the points of the grid of signed distances
    d that are a local extremum of d over the surface, or along one of
    its edges, closer than `limit`.  A clean crossing has none there."""
    near = _extrema(d)
    for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        near[edge] |= _extrema(d[edge])
    return near & (np.abs(d) < limit)


def _sampled(splines: Sequence[TensorSpline]) -> np.ndarray:
    """A dense sample of a patch's surfaces, (n_u, n_v, 3): of its one
    spline, or of the splines of its pieces in order down v, which share
    their bounds of u and their control nodes along it, stacked along v,
    the edge each shares with the one before it once."""
    n_u = max(_PATCH_SAMPLES, 4 * splines[0].coefficients.shape[0])
    parts = []
    for k, spline in enumerate(splines):
        n_v = max(_PATCH_SAMPLES, 4 * spline.coefficients.shape[1])
        (a, b), (c, d) = spline.bounds
        U, V = np.meshgrid(np.linspace(a, b, n_u), np.linspace(c, d, n_v),
                           indexing="ij")
        X = spline(U, V)
        parts.append(X if k == 0 else X[:, 1:])
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


def _check_sample(splines: Sequence[TensorSpline], spec: MeshSpec,
                  sizes: MappingOf[int, InterfaceSizing], *, layout: Layout,
                  exits: Sequence[Exit], carried: Collection[str],
                  fine: float | None, what: str) -> None:
    """Refuse what the fragment could not cut cleanly (module docstring),
    on a dense sample of a patch's surfaces (`_sampled`), its pieces
    together, so that where they join, on a knot of the mapping, they
    cross it: by the layered coordinate against every interface, and by
    its distance from every face of a box."""
    X = _sampled(splines)
    g = spec.geometry
    coords = g.coordinates
    r = np.asarray(coords.split(X)[0], dtype=float)
    label = coords.labels[0]
    layout.check_reach(spec, r, atol=default_atol(layout.scale(spec)), what=what)

    def limit(size: float) -> float:
        return GAP * min(size, np.inf if fine is None else fine)

    # a face of a box is sized by no interface: the coarsest size there is
    coarsest = max(s.far_size for s in sizes.values())
    gaps = {e.name: limit(sizes[e.interface].size if e.interface in sizes else np.inf)
            if e.free else limit(coarsest) for e in exits}
    beyond = {e.name: e.beyond(X, gaps[e.name]) for e in exits}
    if np.all(np.logical_or.reduce(list(beyond.values()))):
        lo, hi = float(r.min()), float(r.max())
        raise ValueError(f"{what} lies wholly outside the geometry, at {label} "
                         f"{lo:.6g} to {hi:.6g}, beyond its "
                         f"{' or '.join(e.where() for e in exits)}; a fault's "
                         "positions are taken in the geometry's own lengths")

    # a near miss counts where the box keeps it: not beyond a face of it
    # (beyond its top too, for a face), the fragment trimming it there
    def kept(e: Exit | None) -> np.ndarray:
        keep = np.ones(r.shape, dtype=bool)
        for other in exits:
            if other is not e and (not other.free or (e is not None and not e.free)):
                keep &= ~beyond[other.name]
        return keep

    def refuse(e: Exit) -> None:
        # carried across an exit, or wholly beyond it: it must enter the body
        gap = gaps[e.name]
        level = e.level(X)
        if np.all(e.outward * level > e.outward * (e.value - e.outward * gap)):
            raise ValueError(
                f"{what} reaches no deeper than {float(np.max(e.inside(X))):.3g} "
                f"inside the {e.where()}, closer than {GAP:g} of the element size "
                f"there ({gap:.3g}): it touches the "
                f"{'surface' if e.free else 'face'} rather than entering the "
                "body, which the fragment cannot cut cleanly")

    def clean(d: np.ndarray, gap_limit: float, name: str, where: str,
              on: Exit | None, level: float) -> None:
        near = _near_misses(d, gap_limit) & kept(on)
        if not np.any(near):
            return
        at = np.flatnonzero(near.ravel())[np.argmin(np.abs(d)[near])]
        gap = float(np.abs(d).ravel()[at])
        if g.is_identity:
            which = f"{what} comes within {gap:.3g} of {name} ({where})"
        else:
            # the same two points mapped: the point of the preimage and the
            # boundary's point on the same line of its coordinate
            axis = 0 if on is None else on.axis
            P = X.reshape(-1, 3)[at]
            parts = list(coords.split(P)[:3])
            parts[axis] = np.asarray(level)
            pair = np.asarray(g.mapping(np.stack([P, coords.points(*parts)])),
                              dtype=float)
            apart = abs(float(np.diff(np.asarray(coords.split(pair)[axis]))[0]))
            along = coords.labels[axis]
            which = (f"{what} has a preimage under the geometry's mapping, the "
                     f"surface built into the reference mesh, that comes within "
                     f"{gap:.3g} of {name} at the reference {along} {level:g} "
                     f"(once mapped, {apart:.3g} from it along the {along}, the "
                     "mapping moving it too)")
        thing = "interface" if on is None or on.free else "face"
        raise ValueError(
            f"{which} and turns there without crossing it cleanly, closer "
            f"than {GAP:g} of the element size there ({gap_limit:.3g}): the "
            f"fragment would leave slivers or a tangency.  A fault does not "
            f"end on, touch or run along {'an' if thing == 'interface' else 'a'} "
            f"{thing}: stop it short of the {thing} by more than "
            f"{gap_limit:.3g}, or continue it past the {thing} by as much (a few "
            "elements either way keeps the elements there well shaped)"
            + ("; or give a reach to carry it across the surface"
               if on is not None and on.free else ""))

    by_interface = {e.interface: e for e in exits if e.free}
    for face in g.interfaces:
        rho = float(face.radius)
        name = face.name or f"interface_{face.index + 1}"
        e = by_interface.get(face.index)
        if e is not None:
            depth = e.inside(X)
            if e.name in carried or np.all(depth <= 0.0):
                refuse(e)
                continue
            clean(-depth, gaps[e.name], f"interface {name!r}", f"{label} = {rho:g}",
                  e, rho)
            continue
        size = sizes[face.index].size if face.index in sizes else np.inf
        clean(r - rho, limit(size), f"interface {name!r}", f"{label} = {rho:g}",
              None, rho)
    for e in exits:
        if e.free:
            continue
        depth = e.inside(X)
        if e.name in carried or np.all(depth <= 0.0):
            refuse(e)
            continue
        clean(-depth, gaps[e.name], f"face {e.name!r}", f"{e.label} = {e.value:g}",
              e, e.value)


def prepare_faults(spec: MeshSpec, sizes: MappingOf[int, InterfaceSizing], *,
                   layout: Layout) -> tuple[PreparedFault, ...]:
    """Pull every patch of every fault back through the geometry's
    mapping, in pieces where it crosses a knot of the mapping, carry the
    edges that reach an exit of the body (`layout`'s) across it, and
    refuse what the fragment could not cut cleanly (module docstring);
    nothing touches gmsh."""
    geometry = spec.geometry
    mapping = geometry.mapping
    identity = geometry.is_identity
    exits = layout.exits(spec) if spec.faults else ()
    atol = default_atol(layout.scale(spec))
    prepared = []
    for name, item in zip(spec.fault_names, spec.faults):
        fine = None if item.sizing is None else item.sizing.size
        preimages, splines, patches, carried = [], [], [], []
        for j, patch in enumerate(item.fault.patches):
            what = f"fault {name!r}, patch {j},"
            if identity:
                pre = pull_back(patch, mapping, controls=item.controls,
                                rtol=item.rtol)
                spline, was = _carried_across(pre.patch.spline(), exits=exits,
                                              reach=item.reach, atol=atol, what=what)
                built, made = [pre], [spline]
            else:
                built, was = _built_through(patch, spec, item, exits=exits,
                                            atol=atol, what=what)
                made = [p.patch.spline() for p in built]
            _check_sample(made, spec, sizes, layout=layout, exits=exits,
                          carried=was, fine=fine, what=what)
            preimages += built
            splines += made
            patches += [j] * len(made)
            carried += [was] * len(made)
        prepared.append(PreparedFault(name=name, item=item,
                                      preimages=tuple(preimages),
                                      splines=tuple(splines),
                                      patches=tuple(patches),
                                      carried=tuple(carried)))
    return tuple(prepared)


def _sample(spline: TensorSpline, spacing: float) -> np.ndarray:
    """Points of a spline over its bounds no farther apart than `spacing`
    along either parameter line, (n, 3)."""
    (a, b), (c, d) = spline.bounds
    m = max(_PATCH_SAMPLES, 4 * max(spline.coefficients.shape[:2]))
    U, V = np.meshgrid(np.linspace(a, b, m), np.linspace(c, d, m), indexing="ij")
    X = spline(U, V)
    along_u = float(np.max(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=-1),
                                  axis=0)))
    along_v = float(np.max(np.sum(np.linalg.norm(np.diff(X, axis=1), axis=-1),
                                  axis=1)))
    # a uniform parameter grid is only as fine as the fastest stretch of it
    speed_u = float(np.max(np.linalg.norm(np.diff(X, axis=0), axis=-1))) * (m - 1)
    speed_v = float(np.max(np.linalg.norm(np.diff(X, axis=1), axis=-1))) * (m - 1)
    n_u = int(np.ceil(max(along_u, speed_u) / spacing)) + 1
    n_v = int(np.ceil(max(along_v, speed_v) / spacing)) + 1
    U, V = np.meshgrid(np.linspace(a, b, n_u), np.linspace(c, d, n_v),
                       indexing="ij")
    return spline(U, V).reshape(-1, 3)


def fault_refinements(prepared: Sequence[PreparedFault]) -> tuple[NearPoints, ...]:
    """A refinement towards each fault that asks for one: `NearPoints` on
    a sampling of its splines no coarser than its size."""
    out = []
    for fault in prepared:
        sizing = fault.item.sizing
        if sizing is None:
            continue
        points = np.concatenate([_sample(s, sizing.size) for s in fault.splines])
        out.append(NearPoints(points, sizing.size, sizing.far_size,
                              sizing.decay_width))
    return tuple(out)


# --------------------------------------------------------------- the CAD

def _occ_surface(spline: TensorSpline) -> int:
    """An OCC surface that is the spline: a plane surface for four
    coplanar bilinear corners, the B-spline surface otherwise."""
    occ = gmsh.model.occ
    c = spline.coefficients
    if spline.degree == (1, 1) and c.shape[:2] == (2, 2):
        corners = [c[0, 0], c[1, 0], c[1, 1], c[0, 1]]
        scale = max(float(np.max(np.abs(c))), 1e-300)
        twist = float(np.dot(np.cross(c[1, 0] - c[0, 0], c[0, 1] - c[0, 0]),
                             c[1, 1] - c[0, 0]))
        if abs(twist) <= 1e-12 * scale ** 3:
            points = [occ.addPoint(*map(float, x)) for x in corners]
            lines = [occ.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]
            return occ.addPlaneSurface([occ.addCurveLoop(lines)])
    n_u, n_v = c.shape[:2]
    # OCC takes the control points with u running fastest
    tags = [occ.addPoint(*map(float, c[i, j])) for j in range(n_v)
            for i in range(n_u)]
    (tu, tv), (ku, kv) = spline.knots, spline.degree
    knots_u, mult_u = np.unique(tu, return_counts=True)
    knots_v, mult_v = np.unique(tv, return_counts=True)
    return occ.addBSplineSurface(tags, n_u, degreeU=ku, degreeV=kv,
                                 knotsU=knots_u.tolist(), knotsV=knots_v.tolist(),
                                 multiplicitiesU=mult_u.tolist(),
                                 multiplicitiesV=mult_v.tolist())


def add_surfaces(prepared: Sequence[PreparedFault]) -> tuple[int, ...]:
    """Every patch of every fault as an OCC surface, in order: the tools
    the fragment takes."""
    return tuple(_occ_surface(s) for fault in prepared for s in fault.splines)


def _owners(prepared: Sequence[PreparedFault]) -> list[tuple[int, int]]:
    """(fault index, surface index within the fault) of each surface
    `add_surfaces` made."""
    return [(k, j) for k, fault in enumerate(prepared)
            for j in range(len(fault.splines))]


def _curves_on(entities: Sequence[int]) -> set[int]:
    """The curves embedded in, or bounding, the given surfaces."""
    out: set[int] = set()
    for entity in entities:
        out |= {t for d, t in gmsh.model.mesh.getEmbedded(2, entity) if d == 1}
        out |= {abs(c) for _, c in gmsh.model.getBoundary(
            [(2, entity)], oriented=False, recursive=False)}
    return out


def settle_faults(cad: ConcentricGeometry | BoxGeometry, tagging: Tagging,
                  prepared: Sequence[PreparedFault], *, spec: MeshSpec,
                  exits: Sequence[Exit]) -> FaultTagging:
    """Keep the pieces of the faults inside the geometry's layers, remove
    the rest, refuse overlaps, and read every fault's curves (module
    docstring): its free edge on the free surface, the first of `exits`,
    and its face edge on the others, the faces of a box (the tagging's
    groups of their names)."""
    owners = _owners(prepared)
    names = [f.name for f in prepared]
    if cad.in_hole:
        raise ValueError("a fault reaches into the hole of the hollow domain; a "
                         "fault leaves the domain through its outer surface alone")
    made: dict[int, tuple[int, int]] = {}
    for owner, group in zip(owners, cad.pieces):
        for tag in group:
            if tag in made:
                (k0, j0), (k1, j1) = made[tag], owner
                j0, j1 = prepared[k0].patches[j0], prepared[k1].patches[j1]
                raise ValueError(
                    f"fault {names[k0]!r} (patch {j0}) and fault {names[k1]!r} "
                    f"(patch {j1}) overlap on a surface; faults and patches may "
                    "meet along curves, not share an area")
            made[tag] = owner
    n_geometry = spec.geometry.nlayers
    layer_of: dict[int, int] = {}
    for i, cell in enumerate(tagging.cells):
        for dim, tag in gmsh.model.mesh.getEmbedded(3, cell):
            if dim == 2 and tag in made:
                layer_of[tag] = i
    astray = [tag for tag in made if layer_of.get(tag, n_geometry) >= n_geometry]
    if astray:
        raise RuntimeError(
            f"fault pieces {astray} lie in no layer of the geometry after the "
            "fragment trimmed them at its surface; the CAD kernel did not cut "
            "the faults as expected")

    faces: list[list[int]] = [[] for _ in prepared]
    for tag in sorted(made):
        faces[made[tag][0]].append(tag)
    for k, group in enumerate(faces):
        if not group:
            raise ValueError(f"fault {names[k]!r} has no part inside the "
                             "geometry's layers")

    edges, inner = [], []
    for group in faces:
        count: dict[int, int] = {}
        within: set[int] = set()
        for f in group:
            for _, c in gmsh.model.getBoundary([(2, f)], oriented=False,
                                               recursive=False):
                count[abs(c)] = count.get(abs(c), 0) + 1
            within.update(t for d, t in gmsh.model.mesh.getEmbedded(2, f) if d == 1)
        edges.append({c for c, n in count.items() if n == 1})
        inner.append({c for c, n in count.items() if n > 1} | within)

    sphere = tagging.faces[next(e.interface for e in exits if e.free)]
    on_surface = _curves_on([sphere])
    named = {e.name for e in exits if not e.free}
    boundary = tuple(t for name, group in tagging.groups if name in named
                     for t in group)
    on_faces = _curves_on(boundary)

    tips, frees, junctions, cuts, meets = [], [], [], [], []
    for k in range(len(prepared)):
        own = edges[k] | inner[k]
        shared, others = set(), []
        for m in range(len(prepared)):
            if m != k and own & (edges[m] | inner[m]):
                shared |= own & (edges[m] | inner[m])
                others.append(names[m])
        free = (edges[k] & on_surface) - shared
        cut = (edges[k] & on_faces) - shared - free
        tips.append(tuple(sorted(edges[k] - shared - free - cut)))
        frees.append(tuple(sorted(free)))
        junctions.append(tuple(sorted(shared)))
        cuts.append(tuple(sorted(cut)))
        meets.append(tuple(others))

    surfaces = {tag: prepared[made[tag][0]].preimages[made[tag][1]].patch
                for tag in made}
    layers = tuple(tuple(sorted({layer_of[t] + 1 for t in group}))
                   for group in faces)
    return FaultTagging(names=tuple(names), faces=tuple(map(tuple, faces)),
                        layers=layers, tips=tuple(tips), free_edges=tuple(frees),
                        junctions=tuple(junctions), meets=tuple(meets),
                        surfaces=surfaces, surface=sphere,
                        face_edges=tuple(cuts) if boundary else (),
                        boundary=boundary)


def apply_fault_groups(faults: FaultTagging, *, first: int,
                       curves: Sequence[str] = manifest.FAULT_CURVES
                       ) -> dict[str, dict[str, int | None]]:
    """The faults' physical groups: fault k's faces as boundary group
    `first` + k - 1 under its name, and its curves of each kind of
    `curves` (the layout's: its tip, free edge and junction, and in a box
    its face edge) as the dimension-1 groups
    `manifest.fault_curve_groups(k, curves=curves)`, named `<name>_tip`,
    `<name>_free_edge`, `<name>_junction` and `<name>_face_edge`, where it
    has any.  Returns, by name, the attribute and the curve groups made."""
    out = {}
    for k, name in enumerate(faults.names, start=1):
        attribute = first + k - 1
        gmsh.model.addPhysicalGroup(2, list(faults.faces[k - 1]), attribute)
        gmsh.model.setPhysicalName(2, attribute, name)
        record: dict[str, int | None] = {"attribute": attribute}
        found = faults.curves(k - 1)
        for key, group in zip(curves, manifest.fault_curve_groups(k, curves=curves)):
            record[key] = None
            if found[key]:
                gmsh.model.addPhysicalGroup(1, list(found[key]), group)
                gmsh.model.setPhysicalName(1, group, f"{name}_{key}")
                record[key] = group
        out[name] = record
    return out


#: The groups `apply_fault_groups` made, by fault name: its boundary
#: attribute and its curve groups, each None where it has none.
type FaultGroups = MappingOf[str, MappingOf[str, int | None]]


def fault_entries(faults: FaultTagging, groups: FaultGroups,
                  prepared: Sequence[PreparedFault],
                  collared: CollarTagging) -> list[manifest.FaultEntry]:
    """The manifest's `faults[]` records."""
    entries = []
    for k, fault in enumerate(prepared):
        g = groups[fault.name]
        # each patch's grid, which every piece of it shares
        controls = [fault.preimages[fault.patches.index(j)].controls
                    for j in range(len(fault.item.fault.patches))]
        sizing = fault.item.sizing
        entries.append(manifest.FaultEntry(
            name=fault.name, attribute=int(g["attribute"]),
            layers=[int(a) for a in faults.layers[k]],
            hanging_wall=collared.numbering.hanging_wall(k),
            tip=g["tip"], free_edge=g["free_edge"], junction=g["junction"],
            meets=list(faults.meets[k]),
            ends_on=[faults.names[j] for j in collared.ends_on[k]],
            controls=(None if all(c is None for c in controls)
                      else [list(c) for c in controls]),
            offset=float(max(p.offset for p in fault.preimages)),
            sizing=None if sizing is None else {
                "size": float(sizing.size), "far_size": float(sizing.far_size),
                "decay_width": float(sizing.decay_width)},
            face_edge=g.get("face_edge")))
    return entries


def curve_names(faults: FaultTagging, groups: FaultGroups) -> dict[int, str]:
    """The dimension-1 physical groups made, by number, with their names."""
    out = {}
    for name in faults.names:
        for key, group in groups[name].items():
            if key != "attribute" and group is not None:
                out[int(group)] = f"{name}_{key}"
    return out


def collar_name(layer: str, faults: Sequence[str]) -> str:
    """The name of a collar's physical group: its layer's, and the faults
    whose hanging wall its cells are in, `<layer>:<fault>+<fault>`."""
    return f"{layer}:{'+'.join(faults)}"


def assign_collars(tagging: Tagging, faulted: FaultTagging, *,
                   layer_names: Sequence[str], body: int) -> CollarTagging:
    """Classify every fault's walls on the mesh as it stands and give the
    cells of each collar their attribute (module docstring), with a
    physical group named by `collar_name` from `layer_names` (each
    layer's group name); the walls are classified on the cells of the
    first `body` layers, the geometry's, a shell's cells being in no
    wall.  Run once the mesh is oriented: the walls are read from the
    corners and the faces' vertex order."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    points = np.asarray(coords, dtype=float).reshape(-1, 3)[order]

    def corners(dim: int, entities: Sequence[int]) -> np.ndarray:
        rows = []
        for entity in entities:
            _, etags, enodes = gmsh.model.mesh.getElements(dim, entity)
            rows += [np.asarray(n, dtype=np.int64).reshape(len(t), -1)[:, :dim + 1]
                     for t, n in zip(etags, enodes) if len(t)]
        rows = np.concatenate(rows) if rows else np.empty((0, dim + 1), np.int64)
        return np.searchsorted(node_tags, rows)

    # the cells, block by block as gmsh holds them, with their layers
    blocks = []
    for layer, entity in enumerate(tagging.cells, start=1):
        types, etags, enodes = gmsh.model.mesh.getElements(3, entity)
        blocks += [(layer, entity, int(t), np.asarray(e, dtype=np.int64),
                    np.asarray(n, dtype=np.int64).reshape(len(e), -1))
                   for t, e, n in zip(types, etags, enodes) if len(e)]
    cells = np.searchsorted(node_tags, np.concatenate([b[4][:, :4] for b in blocks]))
    layers = np.concatenate([np.full(len(b[3]), b[0]) for b in blocks])
    inside = layers <= body
    faces = [corners(2, group) for group in faulted.faces]
    curves = manifest.fault_curves(faces, corners(2, [faulted.surface]),
                                   faces=corners(2, faulted.boundary))
    walls = np.zeros((len(faces), len(cells)), dtype=np.int8)
    for k, (f, c, name) in enumerate(zip(faces, curves, faulted.names)):
        walls[k, inside] = manifest.fault_walls(points, cells[inside], f, c.pinned,
                                                name=name)
    numbering = manifest.collars(layers, walls, layer_count=len(tagging.cells))

    names = {}
    first = len(tagging.cells) + 1
    at = np.cumsum([0] + [len(b[3]) for b in blocks])
    for j, (layer, owners) in enumerate(numbering.combinations):
        attribute = first + j
        volume = gmsh.model.addDiscreteEntity(3)
        for (owner, entity, etype, etags, enodes), lo, hi in zip(blocks, at, at[1:]):
            moved = numbering.attributes[lo:hi] == attribute
            if owner != layer or not np.any(moved):
                continue
            gmsh.model.mesh.removeElements(3, entity, etags[moved])
            gmsh.model.mesh.addElements(3, volume, [etype], [etags[moved]],
                                        [enodes[moved].ravel()])
        gmsh.model.addPhysicalGroup(3, [volume], attribute)
        names[attribute] = collar_name(layer_names[layer - 1],
                                       [faulted.names[k] for k in owners])
        gmsh.model.setPhysicalName(3, attribute, names[attribute])
    return CollarTagging(numbering=numbering,
                         ends_on=tuple(c.ends_on for c in curves), names=names)
