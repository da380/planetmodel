"""Consistent element and boundary orientation, and curving.

MFEM repairs a mesh with inconsistent orientation on load and says so
with a warning; a warning that always appears stops being read, so the
mesh is delivered consistent instead.  Two things are checked.

Cells: every element must have positive signed volume (3D) or area
(2D).  Boundary faces: every interface of a concentric domain encloses
the origin, so "outward" means "away from the origin" and is
unambiguous; OCC orients a surface that bounds shells on both sides by
its own convention, which can point inward, and consistency here is
what lets a consumer treat interface i as the outer boundary of layer i
without inspecting normals itself.  The one exception is the inner
boundary of a hollow domain, which is the body's own boundary and
points out of the body, towards the centre: MFEM turns every element
on a mesh's true boundary that way as it reads it, so no other rule
could hold in a file MFEM has read, and a mesh is written as MFEM will
hold it.  A plane of a box encloses nothing,
so each states its own outward direction (`Directions`): up for the
free surface and every interface, which point out of the layer below
them as a sphere's do, down for the base, away from the body for a
side.  So stated, a boundary is oriented in two dimensions too, a line
element's normal being its direction turned clockwise, the body on its
left, as MFEM's elements run round theirs.

A fault's faces enclose nothing and take no outward rule: each is
oriented with the fault's own normal, from its footwall into its
hanging wall, read from the surface it was built from at the point of
it nearest the face's centroid.  That makes the order of a face's
vertices say which of its two cells is on the hanging-wall side, which
MFEM, reading an interior boundary element, neither checks nor
changes.

All of it but a circle's lines (`circles_wait`) runs at element order
1, before the order is raised:
permuting the vertices of a straight simplex is trivial, permuting a
curved element's interior nodes consistently is not.  `raise_order`
then curves the mesh and repairs what curving folded.
"""
from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from .spec import QUALITY_FLOOR, QUALITY_MEASURE

if TYPE_CHECKING:
    from ..fault import Patch

__all__ = ["OrientationReport", "node_positions", "signed_measures",
           "outward_dots", "facing_dots", "boundary_dots", "circles_wait",
           "fault_dots",
           "orient_cells",
           "orient_boundary", "orient_faults", "orient_mesh", "orient_circles",
           "raise_order",
           "element_quality", "Centres", "Directions", "FaultSurfaces"]

#: The point each surface encloses, by entity tag, where it is not the
#: origin: an offset inclusion encloses its own centre.
type Centres = Mapping[int, ArrayLike]

#: The outward direction of each boundary entity that states its own, by
#: entity tag, in the mesh's coordinates as (x, y, z): a plane of a box.
type Directions = Mapping[int, ArrayLike]

#: The surface each face of a fault was built from, by entity tag: the
#: patch whose normal the face is oriented with.
type FaultSurfaces = Mapping[int, Patch]


@dataclass(frozen=True)
class OrientationReport:
    """What orientation repair found and did."""

    _: KW_ONLY
    cells_checked: int = 0
    cells_flipped: int = 0
    faces_checked: int = 0
    faces_flipped: int = 0
    fault_faces_checked: int = 0
    fault_faces_flipped: int = 0

    @property
    def clean(self) -> bool:
        """Whether the cells and the boundary were already consistently
        oriented; a fault's faces have no orientation until it is given."""
        return self.cells_flipped == 0 and self.faces_flipped == 0

    def __repr__(self) -> str:
        faults = (f", fault faces {self.fault_faces_flipped}/"
                  f"{self.fault_faces_checked} flipped"
                  if self.fault_faces_checked else "")
        return (f"OrientationReport(cells {self.cells_flipped}/"
                f"{self.cells_checked} flipped, faces {self.faces_flipped}/"
                f"{self.faces_checked} flipped{faults})")


def node_positions() -> dict[int, np.ndarray]:
    """Map node tag to position, for the whole mesh."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    return {int(t): xyz[i] for i, t in enumerate(tags)}


def _corner_blocks(dim: int, tag: int, pos: Mapping[int, np.ndarray]
                   ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (element tags, corner coordinates) per element block.

    Only the first vertices of each element are read: for a simplex
    those are the corners, and orientation is decided by the corners
    alone whatever the element order.
    """
    n_corners = {1: 2, 2: 3, 3: 4}[dim]
    types, etags, enodes = gmsh.model.mesh.getElements(dim, tag)
    for etype, tags, nodes in zip(types, etags, enodes):
        if len(tags) == 0:
            continue
        per = len(nodes) // len(tags)
        conn = np.asarray(nodes, dtype=np.int64).reshape(-1, per)[:, :n_corners]
        pts = np.stack([np.stack([pos[int(n)] for n in row])
                        for row in conn])
        yield np.asarray(tags, dtype=np.int64), pts


def signed_measures(dim: int, tag: int, pos: Mapping[int, np.ndarray]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(element tags, signed volume or area) for one entity."""
    all_tags, all_vals = [], []
    for tags, pts in _corner_blocks(dim, tag, pos):
        if dim == 3:
            vals = np.einsum(
                "ij,ij->i",
                np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]),
                pts[:, 3] - pts[:, 0]) / 6.0
        elif dim == 2:
            e1, e2 = pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]
            vals = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2.0
        else:
            vals = np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1)
        all_tags.append(tags)
        all_vals.append(vals)
    if not all_tags:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(all_tags), np.concatenate(all_vals)


def orient_cells(dimension: int, *, pos: Mapping[int, np.ndarray] | None = None
                 ) -> tuple[int, int]:
    """Give every top-dimensional element positive signed measure."""
    pos = node_positions() if pos is None else pos
    checked = flipped = 0
    for _, tag in gmsh.model.getEntities(dimension):
        tags, vals = signed_measures(dimension, tag, pos)
        checked += tags.size
        bad = tags[vals < 0.0]
        if bad.size:
            gmsh.model.mesh.reverseElements(bad)
            flipped += bad.size
    return checked, flipped


def outward_dots(tag: int, pos: Mapping[int, np.ndarray], *,
                 centre: ArrayLike = (0.0, 0.0, 0.0), dim: int = 2
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(element tags, normal . (centroid - centre)) for a surface (`dim`
    2) or a curve of a two-dimensional mesh (`dim` 1), the normals as
    `facing_dots` has them.

    Positive means the normal points away from `centre`.  This is well
    posed only for a boundary that encloses `centre`: a concentric
    interface encloses the origin, an offset inclusion encloses its own
    centre and nothing else.
    """
    centre = np.asarray(centre, dtype=float)
    all_tags, all_dots = [], []
    for tags, pts in _corner_blocks(dim, tag, pos):
        normal = _normals(dim, pts)
        centroid = pts.mean(axis=1) - centre
        all_tags.append(tags)
        all_dots.append(np.einsum("ij,ij->i", normal, centroid))
    if not all_tags:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(all_tags), np.concatenate(all_dots)


def _normals(dim: int, pts: np.ndarray) -> np.ndarray:
    """The normal of each boundary element by its corners, (n, 3): a
    triangle's (v1 - v0) x (v2 - v0), and a line's its direction v1 - v0
    turned clockwise in the mesh's plane."""
    if dim == 2:
        return np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    t = pts[:, 1] - pts[:, 0]
    return np.stack([t[:, 1], -t[:, 0], np.zeros(len(t))], axis=1)


def facing_dots(dim: int, tag: int, pos: Mapping[int, np.ndarray],
                direction: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """(element tags, normal . direction) for a surface (`dim` 2) or a
    curve of a two-dimensional mesh (`dim` 1).

    A triangle's normal is (v1 - v0) x (v2 - v0); a line's is its
    direction v1 - v0 turned clockwise in the mesh's plane, so that it
    points away from a body on its left.  Positive means the normal
    points along `direction`.
    """
    direction = np.asarray(direction, dtype=float)
    all_tags, all_dots = [], []
    for tags, pts in _corner_blocks(dim, tag, pos):
        all_tags.append(tags)
        all_dots.append(_normals(dim, pts) @ direction)
    if not all_tags:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(all_tags), np.concatenate(all_dots)


def fault_dots(tag: int, patch: Patch, pos: Mapping[int, np.ndarray]
               ) -> tuple[np.ndarray, np.ndarray]:
    """(element tags, normal . n) for the faces of a fault's surface, n
    the patch's unit normal at its point nearest each face's centroid.

    Positive means the face's normal points, as the patch's does, from
    the footwall into the hanging wall.  The patch is the one the
    surface was built from, in the mesh's own coordinates.
    """
    all_tags, all_dots = [], []
    for tags, pts in _corner_blocks(2, tag, pos):
        normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        where = patch.locate(pts.mean(axis=1), strict=False)
        all_tags.append(tags)
        all_dots.append(np.einsum("ij,ij->i", normal,
                                  patch.normal(where.u, where.v)))
    if not all_tags:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(all_tags), np.concatenate(all_dots)


def orient_faults(faults: FaultSurfaces, *,
                  pos: Mapping[int, np.ndarray] | None = None) -> tuple[int, int]:
    """Point every face of every fault from its footwall into its hanging
    wall: `faults` maps each face's entity tag to the patch it was built
    from (`fault_dots`)."""
    pos = node_positions() if pos is None else pos
    checked = flipped = 0
    for tag, patch in faults.items():
        tags, dots = fault_dots(tag, patch, pos)
        checked += tags.size
        bad = tags[dots < 0.0]
        if bad.size:
            gmsh.model.mesh.reverseElements(bad)
            flipped += bad.size
    return checked, flipped


def circles_wait(dimension: int, tag: int, directions: Directions | None) -> bool:
    """Whether a boundary entity is turned only once the order is raised:
    a curve of a two-dimensional mesh with no direction of its own, which
    is a circle about a centre.  gmsh raises the order of a mesh whose
    lines on a closed curve were reversed into cells its optimiser cannot
    repair, and ends the process; a curved line it reverses whole, its
    nodes with it, so a circle's lines are turned afterwards."""
    return dimension == 2 and not (directions and tag in directions)


def boundary_dots(dimension: int, tag: int, pos: Mapping[int, np.ndarray], *,
                  centres: Centres | None = None,
                  directions: Directions | None = None,
                  inward: Collection[int] = ()
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """(element tags, how far each element of a boundary entity faces as
    the rule of the module docstring has it), positive where it does:
    along the entity's own direction where `directions` states one, else
    away from the centre it encloses (`centres`, the origin by default),
    or towards it where the entity is in `inward`, the inner boundary of
    a hollow domain.  None for an entity the rule does not reach: a
    curve without a direction of its own in a two-dimensional mesh whose
    boundaries state theirs, a flat one, which has no centre to face."""
    if directions and tag in directions:
        return facing_dots(dimension - 1, tag, pos, directions[tag])
    if directions and dimension != 3:
        return None
    centre = (centres or {}).get(tag, (0.0, 0.0, 0.0))
    tags, dots = outward_dots(tag, pos, centre=centre, dim=dimension - 1)
    return tags, (-dots if tag in inward else dots)


def orient_boundary(dimension: int, *, pos: Mapping[int, np.ndarray] | None = None,
                    centres: Centres | None = None,
                    directions: Directions | None = None,
                    inward: Collection[int] = (),
                    skip: Collection[int] = (), curved: bool = False
                    ) -> tuple[int, int]:
    """Turn every boundary element to face as the rule of the module
    docstring has it (`boundary_dots`), a surface's or, in two
    dimensions, a curve's.

    `centres` maps a boundary's entity tag to the point it encloses; the
    origin otherwise.  `inward` names the entities that face the centre
    they enclose, the inner boundary of a hollow domain.  `skip` names
    surfaces that enclose nothing, a fault's faces, which keep the
    orientation they have.

    The curves of a two-dimensional mesh that face a centre, its
    circles, are turned in a pass of their own, `curved`, made once the
    order is raised (`circles_wait`): every other entity is turned at
    order 1 and left alone in that pass.
    """
    pos = node_positions() if pos is None else pos
    checked = flipped = 0
    for _, tag in gmsh.model.getEntities(dimension - 1):
        if tag in skip or circles_wait(dimension, tag, directions) != curved:
            continue
        facing = boundary_dots(dimension, tag, pos, centres=centres,
                               directions=directions, inward=inward)
        if facing is None:
            continue
        tags, dots = facing
        checked += tags.size
        bad = tags[dots < 0.0]
        if bad.size:
            gmsh.model.mesh.reverseElements(bad)
            flipped += bad.size
    return checked, flipped


def orient_mesh(dimension: int, *, centres: Centres | None = None,
                directions: Directions | None = None,
                inward: Collection[int] = (),
                faults: FaultSurfaces | None = None) -> OrientationReport:
    """Repair cell and boundary orientation, and orient the faults' faces
    with their normals (`orient_faults`); call before raising the order,
    and `orient_circles` after it."""
    pos = node_positions()
    faults = faults or {}
    cells_checked, cells_flipped = orient_cells(dimension, pos=pos)
    faces_checked, faces_flipped = orient_boundary(dimension, pos=pos,
                                                   centres=centres,
                                                   directions=directions,
                                                   inward=inward, skip=faults)
    fault_checked, fault_flipped = orient_faults(faults, pos=pos)
    return OrientationReport(cells_checked=cells_checked,
                             cells_flipped=cells_flipped,
                             faces_checked=faces_checked,
                             faces_flipped=faces_flipped,
                             fault_faces_checked=fault_checked,
                             fault_faces_flipped=fault_flipped)


def orient_circles(dimension: int, *, centres: Centres | None = None,
                   directions: Directions | None = None,
                   inward: Collection[int] = (), skip: Collection[int] = ()
                   ) -> tuple[int, int]:
    """Turn the lines on the circles of a two-dimensional mesh to face as
    the rule of the module docstring has it, the pass `orient_boundary`
    leaves until the order is raised (`circles_wait`); call after raising
    it.  Nothing in three dimensions.  Returns (lines checked, turned)."""
    return orient_boundary(dimension, centres=centres, directions=directions,
                           inward=inward, skip=skip, curved=True)


def element_quality(dimension: int) -> tuple[float, int, int]:
    """(worst minSICN, invalid count, total) over the top-dimensional cells.

    gmsh's own validity measure, and the one that sees curvature: a
    straight element with positive volume can curve into a fold when
    the order is raised, and only a measure evaluated across the
    element notices.
    """
    tags: list[int] = []
    for _, tag in gmsh.model.getEntities(dimension):
        _, etags, _ = gmsh.model.mesh.getElements(dimension, tag)
        tags += [int(x) for block in etags for x in block]
    if not tags:
        return 0.0, 0, 0
    q = np.asarray(gmsh.model.mesh.getElementQualities(
        np.asarray(tags, dtype=np.int64), QUALITY_MEASURE), dtype=float)
    return float(q.min()), int((q <= 0.0).sum()), q.size


def raise_order(dimension: int, order: int, *, optimize: bool = True,
                quality_floor: float = QUALITY_FLOOR) -> dict[str, Any]:
    """Curve the mesh to `order`, then repair any element that folded or
    came out badly shaped.

    Raising the order moves the new nodes onto the CAD surface, and
    where an element is large compared with the local curvature that
    can invert it.  gmsh's high-order optimiser runs whenever order > 1
    and some element is invalid or the worst minSICN is below
    `quality_floor`.  It is not always sufficient, which is why the
    caller validates afterwards rather than trusting this to have
    worked.
    """
    if order < 1:
        raise ValueError(f"element order must be at least 1, got {order}")
    gmsh.model.mesh.setOrder(order)
    before = element_quality(dimension)
    report = {"order": order, "min_sicn_before": before[0],
              "invalid_before": before[1], "elements": before[2],
              "optimized": False}
    if order > 1 and optimize and (before[1] > 0 or before[0] < quality_floor):
        gmsh.model.mesh.optimize("HighOrder")
        after = element_quality(dimension)
        report.update(optimized=True, min_sicn=after[0], invalid=after[1])
    else:
        report.update(min_sicn=before[0], invalid=before[1])
    return report
