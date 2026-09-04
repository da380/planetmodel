"""_orient.py -- consistent element and boundary orientation.

MFEM reports "elements with wrong orientation (fixed)" when it has to
repair a mesh on load.  It is a warning rather than an error, and the
mesh works afterwards, which is exactly why it is worth eliminating at
the source: a warning that always appears stops being read, and the
next one -- about something that matters -- goes with it.

Two separate things are checked here.

**Cells.**  Every element must have positive signed volume (3D) or area
(2D).  gmsh 4.15 does not in fact produce negatives for these
geometries, but that is an observation about one version and one set of
algorithms, not a guarantee, so it is checked and repaired rather than
assumed.

**Boundary faces.**  Every interface bounds a region containing the
origin, so "outward" means "away from the origin" and is unambiguous.
gmsh inherits face orientation from the CAD surface, and OCC orients a
surface that bounds shells on both sides -- every interior interface --
by its own convention, which for the middle interface of a three-shell
body points *inward*.  Consistency here is what lets a consumer treat
interface i as the outer boundary of layer i without inspecting normals
itself.

Both run at element order 1, before setOrder: permuting the vertices of
a straight simplex is trivial, whereas permuting a curved element's
interior nodes consistently is not.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import gmsh
import numpy as np

__all__ = ["OrientationReport", "node_positions", "signed_measures",
           "orient_cells", "orient_boundary", "orient_mesh",
           "raise_order",
           "element_quality"]


@dataclass(frozen=True)
class OrientationReport:
    """What orientation repair found and did."""

    _: KW_ONLY
    cells_checked: int = 0
    cells_flipped: int = 0
    faces_checked: int = 0
    faces_flipped: int = 0

    @property
    def clean(self) -> bool:
        """Whether the mesh was already consistently oriented."""
        return self.cells_flipped == 0 and self.faces_flipped == 0

    def __repr__(self) -> str:
        return (f"OrientationReport(cells {self.cells_flipped}/"
                f"{self.cells_checked} flipped, faces {self.faces_flipped}/"
                f"{self.faces_checked} flipped)")


def node_positions() -> dict:
    """Map node tag to position, for the whole mesh."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    return {int(t): xyz[i] for i, t in enumerate(tags)}


def _corner_blocks(dim: int, tag: int, pos: dict):
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


def signed_measures(dim: int, tag: int, pos: dict):
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


def orient_cells(dimension: int, *, pos: dict | None = None) -> tuple[int, int]:
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


def outward_dots(tag: int, pos: dict, *, centre=(0.0, 0.0, 0.0)):
    """(element tags, normal . (centroid - centre)) for a 3D surface.

    Positive means the face normal points away from `centre`.  This is a
    well-posed question only for a surface that encloses `centre` -- a
    layered body's interfaces all enclose the origin, which is the
    default; an offset inclusion encloses its own centre and nothing
    else, and asking about the origin there reverses part of it.
    """
    centre = np.asarray(centre, dtype=float)
    all_tags, all_dots = [], []
    for tags, pts in _corner_blocks(2, tag, pos):
        normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        centroid = pts.mean(axis=1) - centre
        all_tags.append(tags)
        all_dots.append(np.einsum("ij,ij->i", normal, centroid))
    if not all_tags:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(all_tags), np.concatenate(all_dots)


def orient_boundary(dimension: int, *, pos: dict | None = None,
                    centres: dict | None = None) -> tuple[int, int]:
    """Point every surface's normals away from the centre it encloses.

    `centres` maps a surface's entity tag to the point it encloses; the
    origin otherwise.  2D is skipped: a curve's orientation carries no
    outward normal in the same sense, and no consumer asks for one.
    """
    if dimension != 3:
        return 0, 0
    pos = node_positions() if pos is None else pos
    centres = centres or {}
    checked = flipped = 0
    for _, tag in gmsh.model.getEntities(2):
        tags, dots = outward_dots(tag, pos, centre=centres.get(tag, (0.0, 0.0, 0.0)))
        checked += tags.size
        bad = tags[dots < 0.0]
        if bad.size:
            gmsh.model.mesh.reverseElements(bad)
            flipped += bad.size
    return checked, flipped


def orient_mesh(dimension: int, *, centres: dict | None = None
                ) -> OrientationReport:
    """Repair cell and boundary orientation; call before setOrder."""
    pos = node_positions()
    cells_checked, cells_flipped = orient_cells(dimension, pos=pos)
    faces_checked, faces_flipped = orient_boundary(dimension, pos=pos, centres=centres)
    return OrientationReport(cells_checked=cells_checked, cells_flipped=cells_flipped,
                             faces_checked=faces_checked, faces_flipped=faces_flipped)


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
        np.asarray(tags, dtype=np.int64), "minSICN"), dtype=float)
    return float(q.min()), int((q <= 0.0).sum()), q.size


def raise_order(dimension: int, order: int, *, optimize: bool = True,
                quality_floor: float = 0.05) -> dict:
    """Curve the mesh to `order`, then repair any element that folded or
    came out badly shaped.

    Raising the order moves the new nodes onto the CAD surface, and
    where an element is large compared with the local curvature that
    can invert it -- *before* any topography is applied.  So gmsh's
    high-order optimiser runs whenever order > 1 and some element is
    invalid, or the worst minSICN is below `quality_floor` (the level
    the validation report warns at).  It is not free, and it is not
    always sufficient, which is why the caller still validates
    afterwards rather than trusting this to have worked.  The numbers
    behind the floor are in `docs/notes/mesh_thresholds.md`.
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
