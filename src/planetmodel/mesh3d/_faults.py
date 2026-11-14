"""Faults built into the layered mesh: pulled back, made CAD, fragmented
in with the shells, and labelled by local sides.

A fault is defined where it physically is and the mesh is the reference
one, so each patch enters as its preimage under the geometry's mapping
(`fault.pull_back`), in the mesh's own coordinates, and lands on the
fault when the export moves the nodes.  Each preimage becomes an OCC
surface through its spline (`TensorSpline`): a plane where the spline is
the bilinear one through four coplanar corners, so that its crossings
with the spheres are exact circles, and otherwise the B-spline surface
the spline is, knots and coefficients as they are.  Either is the
preimage exactly.

An edge of a patch whose preimage lies nowhere more than `reach` below
the geometry's outer boundary is carried across it: the spline is
continued beyond that edge (`TensorSpline.extended`) until the new edge
clears the boundary everywhere by a hundredth of the patch's extent
that way, and the fragment trims it there, so the fault meets the
surface exactly wherever the relief puts the reference position of its
edge.  `reach` defaults to the relief of the outer boundary over the
edge, the largest displacement the mapping gives its points there, plus
the CAD kernel's tolerance: under the identity only an edge at the
surface or above it is carried across.

Before any CAD, every preimage is sampled densely and refused by name
where it passes through the centre, reaches below a hollow domain's
inner boundary (a fault leaves the domain through the outer surface
alone), lies wholly outside the geometry, has a local extremum of its
radius, over it or along one of its edges, within a tenth of the local
element size of an interface, whatever it does elsewhere, or, carried
across the outer boundary, reaches no deeper below it than that: the
fragment would turn each into slivers, a tangency, or a loop too small
to mesh.  A fault therefore never ends exactly on an interface: it
crosses it or stops clear of it, and a fault meant to end on one (a
crustal fault bottoming at the Moho) is given ending a little short of
it or continuing a little past it.

The fragment trims the faults at the geometry's outer boundary
(`build_concentric`), and each piece left is found in the layer whose
solid embeds it; a piece made by two surfaces (overlapping faults or
patches) is refused.  The curves of a fault are read from the CAD: a
curve bounding one of its faces once is an edge of it, one bounding two
of its faces, or embedded in one, lies inside it (a seam where it
crosses a sphere, or where another fault ends on it); any of these it
shares with another fault is a junction; an edge on the geometry's
outer boundary is a free edge; every other edge is its buried tip.

Once the mesh is oriented and curved, each fault's walls are classified
on it by the manifest's rule (`manifest.walls_of_faults`, on the
corners, the faces' vertex order carrying the orientation) and the
cells of each collar are given their attribute (`manifest.collars`):
moved out of their layer's volume into a discrete volume of their own,
whose physical group is the collar's attribute, so that the MSH file
and MFEM's reading of it carry it as the cells' element attribute.
"""
from __future__ import annotations

from collections.abc import Mapping as MappingOf
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gmsh
import numpy as np

from ..fault import Preimage, TensorSpline, pull_back
from ..geometry import InterfaceInfo
from ..mapping import Mapping
from . import manifest
from ._geometry import ConcentricGeometry, default_atol
from ._tagging import Tagging
from .spec import EmbeddedFault, InterfaceSizing, MeshSpec, NearPoints

if TYPE_CHECKING:
    from ..fault import Patch

__all__ = ["PreparedFault", "FaultTagging", "prepare_faults", "add_surfaces",
           "settle_faults", "apply_fault_groups", "fault_refinements",
           "fault_entries", "curve_names", "FaultGroups", "CLEARANCE", "GAP",
           "CollarTagging", "assign_collars", "collar_name"]

#: How far a carried edge must clear the outer boundary, as a fraction of
#: the patch's extent across that edge.
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
    carried across the outer boundary where `carried` says so."""

    name: str
    item: EmbeddedFault
    preimages: tuple[Preimage, ...]
    splines: tuple[TensorSpline, ...]
    carried: tuple[bool, ...]


@dataclass(frozen=True)
class FaultTagging:
    """What the fragment made of the faults, fault by fault in the spec's
    order: their faces, the layers (attributes) those lie in, their tip,
    free-edge and junction curves, and the faults each meets; for every
    face, the preimage it is a piece of; and the geometry's outer
    boundary, the surface a free edge lies on."""

    names: tuple[str, ...]
    faces: tuple[tuple[int, ...], ...]
    layers: tuple[tuple[int, ...], ...]
    tips: tuple[tuple[int, ...], ...]
    free_edges: tuple[tuple[int, ...], ...]
    junctions: tuple[tuple[int, ...], ...]
    meets: tuple[tuple[str, ...], ...]
    surfaces: MappingOf[int, Patch]
    surface: int


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


def _relief(mapping: Mapping, points: np.ndarray, radius: float) -> float:
    """The largest displacement the mapping gives the outer boundary's
    points over `points`, projected radially onto it."""
    X = radius * points / np.linalg.norm(points, axis=-1, keepdims=True)
    return float(np.max(np.linalg.norm(np.asarray(mapping(X), dtype=float) - X,
                                       axis=-1)))


def _carried_across(spline: TensorSpline, *, radius: float, reach: float | None,
                    mapping: Mapping, identity: bool, atol: float, what: str
                    ) -> tuple[TensorSpline, bool]:
    """The spline with every edge that reaches the outer boundary carried
    across it (module docstring), and whether any was."""
    carried = False
    for side in ("u0", "u1", "v0", "v1"):
        u, v = _edge(spline, side, _EDGE_SAMPLES)
        X = spline(u, v)
        r = np.linalg.norm(X, axis=-1)
        limit = (reach if reach is not None
                 else (0.0 if identity else _relief(mapping, X, radius)) + atol)
        if float(np.max(radius - r)) > limit:
            continue
        # the extent across the edge, along the middle line of the patch
        (a, b), (c, d) = spline.bounds
        s = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
        line = (spline(0.5 * (a + b) + 0.0 * s, c + (d - c) * s) if side[0] == "v"
                else spline(a + (b - a) * s, 0.5 * (c + d) + 0.0 * s))
        extent = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=-1)))
        span = (d - c) if side[0] == "v" else (b - a)
        want = radius + CLEARANCE * extent
        t = 0.02 * span
        while True:
            trial = _carry(spline, side, t)
            u, v = _edge(trial, side, _EDGE_SAMPLES)
            if float(np.min(np.linalg.norm(trial(u, v), axis=-1))) >= want:
                break
            t *= 1.5
            if t > _LONGEST_CARRY * span:
                raise ValueError(
                    f"{what} meets the outer boundary (r = {radius:g}) at too "
                    "shallow an angle to be carried across it cleanly: continued "
                    f"by half its extent, its edge still does not clear it; end "
                    "it farther below the surface, or give a smaller reach")
        spline, carried = trial, True
    return spline, carried


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
                  sizes: MappingOf[int, InterfaceSizing], *, carried: bool,
                  fine: float | None, what: str) -> None:
    """Refuse what the fragment could not cut cleanly (module docstring),
    on a dense sample of the spline."""
    n_u, n_v = (max(_PATCH_SAMPLES, 4 * n) for n in spline.coefficients.shape[:2])
    (a, b), (c, d) = spline.bounds
    U, V = np.meshgrid(np.linspace(a, b, n_u), np.linspace(c, d, n_v),
                       indexing="ij")
    r = np.linalg.norm(spline(U, V), axis=-1)
    lo, hi = float(r.min()), float(r.max())
    boundaries = spec.geometry.skeleton.boundaries
    inner, outer = float(boundaries[0]), float(boundaries[-1])
    atol = default_atol(spec.outer_radius)

    def limit(face: InterfaceInfo) -> float:
        size = sizes[face.index].size if face.index in sizes else np.inf
        return GAP * min(size, np.inf if fine is None else fine)

    surface = spec.geometry.interfaces[-1]
    if lo <= atol:
        raise ValueError(f"{what} passes through the centre, where the spherical "
                         "frame has no direction: a fault of a layered mesh "
                         "keeps clear of it")
    if inner > 0.0 and lo < inner:
        raise ValueError(f"{what} reaches r = {lo:.6g}, below the inner boundary "
                         f"of the hollow domain at r = {inner:g}: a fault leaves "
                         "the domain through its outer surface alone")
    if lo >= outer + limit(surface):
        raise ValueError(f"{what} lies wholly outside the geometry, at radii "
                         f"{lo:.6g} to {hi:.6g} beyond its surface at r = "
                         f"{outer:g}; a fault's positions are taken in the "
                         "geometry's own lengths")
    for face in spec.geometry.interfaces:
        rho, gap_limit = float(face.radius), limit(face)
        if rho == outer and (carried or lo >= outer):
            if lo > outer - gap_limit:
                raise ValueError(
                    f"{what} reaches no deeper than {outer - lo:.3g} below the "
                    f"outer boundary (r = {outer:g}), closer than {GAP:g} of the "
                    f"element size there ({gap_limit:.3g}): it touches the "
                    "surface rather than entering the body, which the fragment "
                    "cannot cut cleanly")
            continue
        near = _near_misses(r - rho, gap_limit)
        if np.any(near):
            at = np.flatnonzero(near.ravel())[np.argmin(np.abs(r - rho)[near])]
            gap = float(np.abs(r - rho).ravel()[at])
            name = face.name or f"interface_{face.index + 1}"
            if spec.geometry.is_identity:
                which = (f"{what} comes within {gap:.3g} of interface {name!r} (r = "
                         f"{rho:g})")
            else:
                # the same two points mapped: the point of the preimage and
                # the interface's point on the same radius
                X = spline(U.ravel()[at], V.ravel()[at])
                pair = np.asarray(spec.geometry.mapping(
                    np.stack([X, rho * X / np.linalg.norm(X)])), dtype=float)
                apart = abs(float(np.diff(np.linalg.norm(pair, axis=-1))[0]))
                which = (f"{what} has a preimage under the geometry's mapping, the "
                         f"surface built into the reference mesh, that comes within "
                         f"{gap:.3g} of interface {name!r} at the reference radius "
                         f"{rho:g} (once mapped, {apart:.3g} from it along the "
                         "radius, the mapping moving the interface too)")
            raise ValueError(
                f"{which} and turns there without crossing it cleanly, closer "
                f"than {GAP:g} of the element size there ({gap_limit:.3g}): the "
                "fragment would leave slivers or a tangency.  A fault does not "
                "end on, touch or run along an interface: stop it short of the "
                f"interface by more than {gap_limit:.3g}, or continue it past "
                "the interface by as much (a few elements either way keeps the "
                "elements there well shaped)"
                + ("; or give a reach to carry it across the surface"
                   if rho == outer else ""))


def prepare_faults(spec: MeshSpec, sizes: MappingOf[int, InterfaceSizing]
                   ) -> tuple[PreparedFault, ...]:
    """Pull every patch of every fault back through the geometry's
    mapping, carry the edges that reach the outer boundary across it,
    and refuse what the fragment could not cut cleanly (module
    docstring); nothing touches gmsh."""
    geometry = spec.geometry
    mapping = geometry.mapping
    identity = geometry.is_identity
    outer = float(geometry.skeleton.boundaries[-1])
    atol = default_atol(spec.outer_radius)
    prepared = []
    for name, item in zip(spec.fault_names, spec.faults):
        fine = None if item.sizing is None else item.sizing.size
        preimages, splines, carried = [], [], []
        for j, patch in enumerate(item.fault.patches):
            what = f"fault {name!r}, patch {j},"
            pre = pull_back(patch, mapping, controls=item.controls, rtol=item.rtol)
            spline, was = _carried_across(
                pre.patch.spline(), radius=outer, reach=item.reach,
                mapping=mapping, identity=identity, atol=atol, what=what)
            _check_sample(spline, spec, sizes, carried=was, fine=fine, what=what)
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


def settle_faults(cad: ConcentricGeometry, tagging: Tagging,
                  prepared: Sequence[PreparedFault], *, spec: MeshSpec
                  ) -> FaultTagging:
    """Keep the pieces of the faults inside the geometry's layers, remove
    the rest, refuse overlaps, and read every fault's curves (module
    docstring)."""
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

    outer = float(spec.geometry.skeleton.boundaries[-1])
    k_out = next(i for i, f in enumerate(spec.domain.interfaces)
                 if f.radius == outer)
    sphere = tagging.faces[k_out]
    on_surface = ({t for d, t in gmsh.model.mesh.getEmbedded(2, sphere) if d == 1}
                  | {abs(c) for _, c in gmsh.model.getBoundary(
                      [(2, sphere)], oriented=False, recursive=False)})

    tips, frees, junctions, meets = [], [], [], []
    for k in range(len(prepared)):
        own = edges[k] | inner[k]
        shared, others = set(), []
        for m in range(len(prepared)):
            if m != k and own & (edges[m] | inner[m]):
                shared |= own & (edges[m] | inner[m])
                others.append(names[m])
        free = (edges[k] & on_surface) - shared
        tips.append(tuple(sorted(edges[k] - shared - free)))
        frees.append(tuple(sorted(free)))
        junctions.append(tuple(sorted(shared)))
        meets.append(tuple(others))

    surfaces = {tag: prepared[made[tag][0]].preimages[made[tag][1]].patch
                for tag in made}
    layers = tuple(tuple(sorted({layer_of[t] + 1 for t in group}))
                   for group in faces)
    return FaultTagging(names=tuple(names), faces=tuple(map(tuple, faces)),
                        layers=layers, tips=tuple(tips), free_edges=tuple(frees),
                        junctions=tuple(junctions), meets=tuple(meets),
                        surfaces=surfaces, surface=sphere)


def apply_fault_groups(faults: FaultTagging, *, first: int
                       ) -> dict[str, dict[str, int | None]]:
    """The faults' physical groups: fault k's faces as boundary group
    `first` + k - 1 under its name, and its tip, free-edge and junction
    curves as the dimension-1 groups `manifest.fault_curve_groups(k)`,
    named `<name>_tip`, `<name>_free_edge` and `<name>_junction`, where it
    has any.  Returns, by name, the attribute and the curve groups made."""
    out = {}
    for k, name in enumerate(faults.names, start=1):
        attribute = first + k - 1
        gmsh.model.addPhysicalGroup(2, list(faults.faces[k - 1]), attribute)
        gmsh.model.setPhysicalName(2, attribute, name)
        record: dict[str, int | None] = {"attribute": attribute}
        for key, curves, group in zip(
                ("tip", "free_edge", "junction"),
                (faults.tips[k - 1], faults.free_edges[k - 1],
                 faults.junctions[k - 1]),
                manifest.fault_curve_groups(k)):
            record[key] = None
            if curves:
                gmsh.model.addPhysicalGroup(1, list(curves), group)
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
                "decay_width": float(sizing.decay_width)}))
    return entries


def curve_names(faults: FaultTagging, groups: FaultGroups) -> dict[int, str]:
    """The dimension-1 physical groups made, by number, with their names."""
    out = {}
    for name in faults.names:
        for key in ("tip", "free_edge", "junction"):
            if groups[name][key] is not None:
                out[int(groups[name][key])] = f"{name}_{key}"
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
    curves = manifest.fault_curves(faces, corners(2, [faulted.surface]))
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
