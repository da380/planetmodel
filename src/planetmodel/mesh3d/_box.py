"""A box cut by horizontal planes, or its section, and which entity is which.

A flat geometry's body is the box over its lateral domain's extent,
from the free surface z = 0 (or a hollow box's top) down to its base,
layer k between the planes z = -d_k and z = -d_(k+1) of its depths; over
a `Rectangle` it is the section of that body in the (x, z) plane, a
two-dimensional mesh whose coordinates are (x, z).  The construction
is one box (or rectangle) per layer, fragmented into one body so that
the layers share their interfaces, as the concentric construction
fragments its balls.

The surfaces of faults join that one fragment as tools: it cuts each
into pieces at the planes it crosses, embeds every piece inside the
box in the layer holding it and every crossing curve in its plane, a
side or the base, and makes the curves where surfaces meet common to
both, so that the mesh conforms to all of it.  The pieces outside the
box, beyond its top, its base or a side, lie in no solid and are
removed, which trims every fault at the box; what becomes of each
surface is the fragment's map, piece by piece (`BoxGeometry`).  No
duplicates are removed after it, which would renumber the pieces.  A
fault may cut a face of the box, its base or a side, into pieces, but
not a layer or an interface, its top included: each stays one entity,
which is the contract the tagging gives, and a fault that cuts one
right through is refused by name.

Identification is by bounding boxes, before meshing, and by matching,
never sorting, as for the spheres (`_tagging`).  With faults fragmented
in, the bounding boxes are taken from a triangulation of the entities
(gmsh's `Geometry.OCCBoundsUseStl`): OCC's own box of a B-spline surface
is that of its control net, which for a fault's preimage under a
mapping, carried across the top and trimmed there, reaches above the
box and would widen the box of the layer embedding it; a triangulation
is tight on the planes and lines of the box.  A cell is the layer
whose depths its box spans; a boundary entity flat along z is the
interface, or a piece of the base, at the depth asked for, and one flat
along a lateral axis is a piece of a side (x_min, ...) beside one layer,
whose depths hold its box, a side being one entity per layer unless a
fault cuts it.  Every place asked for must be matched, an interface by
exactly one entity, and every entity by one place; anything else is
refused with both lists, a near-match being the symptom of a CAD
failure.  The places recorded are the ones asked for.

Each boundary entity states its outward direction (`PlanarTagging`):
up for the top and every interface, out of the layer below as on the
sphere, down for the base, away from the body for a side.  The distance
to an interface is |z - z_k|, in the mesh's own coordinates (its second
in a section), and an interface's measured place the mean depth of its
nodes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from ._geometry import default_atol
from ._layout import PlanarFace
from ._tagging import Tagging

__all__ = ["PlanarTagging", "BoxGeometry", "build_box"]


@dataclass(frozen=True)
class BoxGeometry:
    """What the fragment made of the surfaces built into a box: `pieces`
    holds, for each surface in the order given, the faces of it inside
    the box, its parts outside removed.  A box has no hole, so `in_hole`,
    the pieces a hollow sphere leaves in its hole, is empty."""

    #: The faces each surface became inside the box.
    pieces: tuple[tuple[int, ...], ...]
    _: KW_ONLY
    #: The pieces left in a hole: none, a box having none.
    in_hole: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlanarTagging(Tagging):
    """Which CAD entity of a box is which layer, interface and face.

    `cells[i]` is layer i and `faces[k]` interface k from the surface
    down, `radii[k]` its depth; `groups` are the faces of the box, each
    with its entities (a side has one beside each layer, and a face more
    where a fault cuts it), numbered after the interfaces; `directions`
    the outward direction of every boundary
    entity in the mesh's coordinates; `axis` the index of the vertical
    among the mesh's coordinates, its last: 2 in a box, 1 in a section.
    """

    _: KW_ONLY
    axis: int = 2

    #: The name of the layered coordinate an interface is placed by.
    coordinate = "depth"

    def distance(self, k: int, *, centre: ArrayLike = (0.0, 0.0, 0.0)) -> str:
        """The distance to interface k, the plane of its depth, as an
        expression gmsh evaluates: |z + d_k|, z the mesh's vertical
        coordinate; `centre` is a sphere's and is not used."""
        z = "xyz"[self.axis]
        return f"Fabs({z}-({-float(self.radii[k])!r}))"

    def measure(self, tag: int) -> float:
        """The measured place of an interface's entity: its nodes' mean
        depth."""
        _, coords, _ = gmsh.model.mesh.getNodes(self.dimension - 1, tag,
                                                includeBoundary=True)
        if coords.size == 0:
            return float("nan")
        xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
        return float(np.mean(-xyz[:, self.axis]))

    def __repr__(self) -> str:
        return (f"PlanarTagging({len(self.cells)} layers, "
                f"depths {[round(r, 6) for r in self.radii]}, "
                f"faces {[name for name, _ in self.groups]})")


def _boxes(dimension: int, tags: Sequence[int]
           ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The bounding box of each entity, (lower, upper) corners in the
    mesh's coordinates."""
    out = {}
    for tag in tags:
        box = np.asarray(gmsh.model.getBoundingBox(dimension, tag), dtype=float)
        out[tag] = box[:3], box[3:]
    return out


def _fragment_surfaces(boxes: Sequence[int], surfaces: Sequence[int]
                       ) -> tuple[tuple[int, ...], ...]:
    """Fragment the boxes of the layers and the surfaces into one body and
    remove the surfaces' pieces outside it (module docstring): the pieces
    of each surface that remain."""
    occ = gmsh.model.occ
    _, mapped = occ.fragment([(3, boxes[0])], [(3, t) for t in boxes[1:]]
                             + [(2, s) for s in surfaces])
    pieces = [tuple(tag for _, tag in out) for out in mapped[len(boxes):]]
    occ.synchronize()
    held = {tag for _, v in gmsh.model.getEntities(3)
            for dim, tag in gmsh.model.mesh.getEmbedded(3, v) if dim == 2}
    excess = [(2, tag) for group in pieces for tag in group if tag not in held]
    if excess:
        occ.remove(excess, recursive=True)
        occ.synchronize()
    return tuple(tuple(tag for tag in group if tag in held) for group in pieces)


def build_box(boundaries: ArrayLike, *, coordinates: Sequence[str],
              bounds: Mapping[str, tuple[float, float]],
              faces: Sequence[PlanarFace], scale: float,
              surfaces: Sequence[int] = (), atol: float | None = None
              ) -> tuple[BoxGeometry, PlanarTagging]:
    """A box of the given depths over the lateral bounds, in the open gmsh
    session, the surfaces fragmented in and trimmed at it, and its
    entities identified (module docstring).

    `boundaries` are the skeleton's depths from the top down, the last
    the base; `coordinates` the physical axes of the mesh's coordinates,
    ("x", "y", "z") or ("x", "z"), the vertical last; `bounds` the extent
    along each lateral axis; `faces` the faces of the box, (name, axis,
    outward, value), in the order of their attributes; `scale` the size
    of the body, which the matching tolerance is relative to (`atol`, by
    default `_geometry.default_atol` of it); `surfaces` the tags of OCC
    surfaces already made in the session (a box alone, three
    dimensions).  Refused, as a ValueError: surfaces in a section, and
    surfaces that cut a layer or an interface into pieces.
    """
    b = [float(v) for v in np.atleast_1d(np.asarray(boundaries, dtype=float))]
    if len(b) < 2 or not all(hi > lo for lo, hi in zip(b[:-1], b[1:])) or b[0] < 0.0:
        raise ValueError(f"depths must start at or below zero and increase, got {b}")
    d = len(coordinates)
    if d not in (2, 3) or coordinates[-1] != "z":
        raise ValueError(f"a box's coordinates end with z, got {coordinates}")
    surfaces = tuple(int(s) for s in surfaces)
    if surfaces and d != 3:
        raise ValueError("surfaces are fragmented into a box alone, not a section")
    lateral = list(coordinates[:-1])
    tol = default_atol(scale) if atol is None else float(atol)
    occ = gmsh.model.occ

    lo = [bounds[a][0] for a in lateral]
    width = [bounds[a][1] - bounds[a][0] for a in lateral]
    made = []
    for top, bottom in zip(b[:-1], b[1:]):
        if d == 3:
            made.append(occ.addBox(lo[0], lo[1], -bottom, width[0], width[1],
                                   bottom - top))
        else:
            made.append(occ.addRectangle(lo[0], -bottom, 0.0, width[0], bottom - top))
    pieces: tuple[tuple[int, ...], ...] = ()
    if surfaces:
        pieces = _fragment_surfaces(made, surfaces)
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    elif len(made) > 1:
        occ.fragment([(d, made[0])], [(d, t) for t in made[1:]])
        occ.removeAllDuplicates()
    occ.synchronize()
    built = {tag for group in pieces for tag in group}

    v = d - 1                                    # the vertical among the coordinates
    layers = [(-bottom, -top) for top, bottom in zip(b[:-1], b[1:])]

    def near(x: float, y: float) -> bool:
        return abs(float(x) - float(y)) <= tol

    def layer_of(lower: np.ndarray, upper: np.ndarray) -> int | None:
        hits = [i for i, (zlo, zhi) in enumerate(layers)
                if near(lower[v], zlo) and near(upper[v], zhi)]
        return hits[0] if len(hits) == 1 else None

    def within(lower: np.ndarray, upper: np.ndarray) -> int | None:
        hits = [i for i, (zlo, zhi) in enumerate(layers)
                if lower[v] >= zlo - tol and upper[v] <= zhi + tol]
        return hits[0] if len(hits) == 1 else None

    cell_boxes = _boxes(d, [t for _, t in gmsh.model.getEntities(d)])
    if surfaces and len(cell_boxes) != len(layers):
        raise ValueError(
            f"the faults cut the box into {len(cell_boxes)} solids where its "
            f"{len(layers)} layers should stay whole: a fault must not cut a layer "
            "right through (it may end inside a layer, reach the surface, or leave "
            "through the base or a side, so long as some of the layer goes round "
            "its edge)")
    cells: list[int | None] = [None] * len(layers)
    for tag, (lower, upper) in cell_boxes.items():
        i = layer_of(lower, upper)
        if i is None or cells[i] is not None:
            raise RuntimeError(_mismatch("cell", tag, lower, upper, b, tol))
        cells[i] = tag
    if any(c is None for c in cells):
        raise RuntimeError(f"no cell spans the layers "
                           f"{[i for i, c in enumerate(cells) if c is None]} of "
                           f"depths {b}; the CAD kernel made {len(cell_boxes)} cells")

    up = np.zeros(3)
    up[v] = 1.0
    planes: dict[int, list[int]] = {}            # boundary index -> entities
    sides: dict[tuple[str, int], list[int]] = {}  # (face, layer) -> entities
    directions: dict[int, tuple[float, float, float]] = {}
    side_rows = [(name, axis, outward, value) for name, axis, outward, value in faces
                 if axis != "z"]
    boundary = [t for _, t in gmsh.model.getEntities(v) if t not in built]
    for tag, (lower, upper) in _boxes(v, boundary).items():
        if near(lower[v], upper[v]):
            hits = [j for j, depth in enumerate(b) if near(lower[v], -depth)]
            if len(hits) != 1 or (not surfaces and hits[0] in planes):
                raise RuntimeError(_mismatch("horizontal boundary", tag, lower,
                                             upper, b, tol))
            planes.setdefault(hits[0], []).append(tag)
            sign = -1.0 if hits[0] == len(b) - 1 else 1.0
            directions[tag] = tuple(float(x) for x in sign * up)
            continue
        found = None
        for name, axis, outward, value in side_rows:
            a = lateral.index(axis)
            if near(lower[a], upper[a]) and near(lower[a], value):
                i = within(lower, upper) if surfaces else layer_of(lower, upper)
                if i is not None and (surfaces or (name, i) not in sides):
                    found = (name, i, a, outward)
                break
        if found is None:
            raise RuntimeError(_mismatch("side", tag, lower, upper, b, tol))
        name, i, a, outward = found
        sides.setdefault((name, i), []).append(tag)
        out = np.zeros(3)
        out[a] = float(outward)
        directions[tag] = tuple(float(x) for x in out)

    cut = {j: len(planes[j]) for j in range(len(b) - 1) if len(planes.get(j, ())) > 1}
    if cut:
        raise ValueError(
            "the faults cut "
            + ", ".join(f"the interface at depth {b[j]:g} into {n} pieces"
                        for j, n in cut.items())
            + ": a fault cutting an interface, the box's top included, right through "
            "(from one side of the box to another) is not built; end it inside the "
            "box at one end at least")
    missing = [j for j in range(len(b)) if j not in planes]
    missing += [key for key in ((name, i) for name, _, _, _ in side_rows
                                for i in range(len(layers))) if key not in sides]
    if missing:
        raise RuntimeError(f"no boundary entity was found for {missing}; the box of "
                           f"depths {b} over {dict(bounds)} was not cut as asked")
    groups = []
    for name, axis, _, _ in faces:
        if axis == "z":
            groups.append((name, tuple(sorted(planes[len(b) - 1]))))
        else:
            groups.append((name, tuple(t for i in range(len(layers))
                                       for t in sorted(sides[(name, i)]))))
    tagging = PlanarTagging(d, tuple(cells),
                            tuple(planes[j][0] for j in range(len(b) - 1)),
                            tuple(b[:-1]), groups=tuple(groups),
                            directions=directions, axis=v)
    return BoxGeometry(pieces), tagging


def _mismatch(what: str, tag: int, lower: np.ndarray, upper: np.ndarray,
              depths: Sequence[float], tol: float) -> str:
    """A failure naming the entity, its bounding box and the depths asked
    for, since either may be at fault."""
    return (f"cannot match {what} {tag}, with bounding box {lower.tolist()} to "
            f"{upper.tolist()}, to the box asked for (depths {list(depths)}, "
            f"tolerance {tol:g}): exactly one place was required")
