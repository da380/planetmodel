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
stretch), is carried across it: the spline is
continued beyond that edge (`TensorSpline.extended`) until the new edge
clears the boundary everywhere by a hundredth of the patch's extent
that way, and the fragment trims it there, so the fault meets the
boundary exactly wherever the relief puts the reference position of its
edge.  `reach` defaults to the relief of the surface over the edge, the
largest displacement the mapping gives its points there along the
layered coordinate, plus the CAD kernel's tolerance: under the identity
only an edge at the boundary or beyond it is carried across.  An edge
is carried across one exit at most.  The part carried is the preimage
continued, its end pieces' polynomials carried on: an extrapolation,
which strays from the fault continued as the relief over the edge
grows towards the patch's extent across it, and an edge that does not
clear the boundary within half that extent is refused.  A fault meant
to meet a surface the relief raises well above its edge is given
reaching above it, and the fragment trims it on the surface exactly.

Before any CAD, every preimage is sampled densely and refused by name
where it reaches where no fault may (through a sphere's centre, below a
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

from collections.abc import Collection, Sequence
from collections.abc import Mapping as MappingOf
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import gmsh
import numpy as np

from ..fault import Preimage, TensorSpline, pull_back
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
           "CollarTagging", "assign_collars", "collar_name"]

#: How far a carried edge must clear the boundary it is carried across, as
#: a fraction of the patch's extent across that edge.
CLEARANCE = 0.01

#: The nearest a fault may come to an interface it does not cross, as a
#: fraction of the element size there.
GAP = 0.1

#: The farthest an edge is carried, as a fraction of the parameter range.
_LONGEST_CARRY = 0.5

#: Samples along each edge, and each way over a patch, at the least.
_EDGE_SAMPLES = 65
_PATCH_SAMPLES = 33


@dataclass(frozen=True)
class PreparedFault:
    """One fault ready for the CAD: its name and spec, its patches'
    preimages, and the splines built from them in the mesh's coordinates,
    each carried across the exits `carried` names (by their names)."""

    name: str
    item: EmbeddedFault
    preimages: tuple[Preimage, ...]
    splines: tuple[TensorSpline, ...]
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

def _edge(spline: TensorSpline, side: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of n points along one edge of a spline's bounds: `side` is
    "u0", "u1", "v0" or "v1", the low or high end of u or v."""
    (a, b), (c, d) = spline.bounds
    s = np.linspace(0.0, 1.0, n)
    along_u, along_v = a + (b - a) * s, c + (d - c) * s
    return {"u0": (np.full(n, a), along_v), "u1": (np.full(n, b), along_v),
            "v0": (along_u, np.full(n, c)), "v1": (along_u, np.full(n, d))}[side]


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


def _carried_across(spline: TensorSpline, *, exits: Sequence[Exit],
                    reach: float | None, mapping: Mapping, identity: bool,
                    atol: float, what: str) -> tuple[TensorSpline, frozenset[str]]:
    """The spline with every edge that reaches an exit carried across it
    (module docstring), and the names of the exits it was carried across."""
    carried: set[str] = set()
    for side in ("u0", "u1", "v0", "v1"):
        u, v = _edge(spline, side, _EDGE_SAMPLES)
        X = spline(u, v)
        for exit in exits:
            limit = (reach if reach is not None and exit.free
                     else (0.0 if identity else _relief(mapping, X, exit)) + atol)
            if float(np.max(exit.inside(X))) > limit:
                continue
            # the extent across the edge, along the middle line of the patch
            (a, b), (c, d) = spline.bounds
            s = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
            line = (spline(0.5 * (a + b) + 0.0 * s, c + (d - c) * s)
                    if side[0] == "v"
                    else spline(a + (b - a) * s, 0.5 * (c + d) + 0.0 * s))
            extent = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=-1)))
            span = (d - c) if side[0] == "v" else (b - a)
            t = 0.02 * span
            while True:
                trial = _carry(spline, side, t)
                u, v = _edge(trial, side, _EDGE_SAMPLES)
                if np.all(exit.beyond(trial(u, v), CLEARANCE * extent)):
                    break
                t *= 1.5
                if t > _LONGEST_CARRY * span:
                    raise ValueError(
                        f"{what} meets the {exit.where()} at too shallow an angle "
                        "to be carried across it cleanly: continued by half its "
                        "extent, its edge still does not clear it; end it farther "
                        "inside the body, or give a smaller reach")
            spline = trial
            carried.add(exit.name)
            break
    return spline, frozenset(carried)


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


def _check_sample(spline: TensorSpline, spec: MeshSpec,
                  sizes: MappingOf[int, InterfaceSizing], *, layout: Layout,
                  exits: Sequence[Exit], carried: Collection[str],
                  fine: float | None, what: str) -> None:
    """Refuse what the fragment could not cut cleanly (module docstring),
    on a dense sample of the spline: by the layered coordinate against
    every interface, and by its distance from every face of a box."""
    n_u, n_v = (max(_PATCH_SAMPLES, 4 * n) for n in spline.coefficients.shape[:2])
    (a, b), (c, d) = spline.bounds
    U, V = np.meshgrid(np.linspace(a, b, n_u), np.linspace(c, d, n_v),
                       indexing="ij")
    X = spline(U, V)
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
            P = spline(U.ravel()[at], V.ravel()[at])
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
    mapping, carry the edges that reach an exit of the body (`layout`'s)
    across it, and refuse what the fragment could not cut cleanly (module
    docstring); nothing touches gmsh."""
    geometry = spec.geometry
    mapping = geometry.mapping
    identity = geometry.is_identity
    exits = layout.exits(spec) if spec.faults else ()
    atol = default_atol(layout.scale(spec))
    prepared = []
    for name, item in zip(spec.fault_names, spec.faults):
        fine = None if item.sizing is None else item.sizing.size
        preimages, splines, carried = [], [], []
        for j, patch in enumerate(item.fault.patches):
            what = f"fault {name!r}, patch {j},"
            pre = pull_back(patch, mapping, controls=item.controls, rtol=item.rtol)
            spline, was = _carried_across(
                pre.patch.spline(), exits=exits, reach=item.reach,
                mapping=mapping, identity=identity, atol=atol, what=what)
            _check_sample(spline, spec, sizes, layout=layout, exits=exits,
                          carried=was, fine=fine, what=what)
            preimages.append(pre)
            splines.append(spline)
            carried.append(was)
        prepared.append(PreparedFault(name=name, item=item,
                                      preimages=tuple(preimages),
                                      splines=tuple(splines),
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
    """(fault index, patch index) of each surface `add_surfaces` made."""
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
        controls = [p.controls for p in fault.preimages]
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
