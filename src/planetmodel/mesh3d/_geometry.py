"""Concentric CAD, in 2D and 3D by the same code.

A layered domain is a set of concentric balls (3D) or discs (2D), cut
against each other so the shells between them become separate
entities.  The dimension is a parameter throughout: discs are the cheap
testing analogue of balls and must not fork the implementation.

The construction is: build every radius as its own solid, fragment the
outermost against all the others, remove duplicates, and, for a hollow
domain, remove the innermost solid while keeping its boundary.  What
comes back is one entity per shell and one per interface in arbitrary
tag order, which is why identification is a separate step (`_tagging`)
and never a matter of sorting by tag.
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np

__all__ = ["ConcentricGeometry", "build_concentric", "entity_radius",
           "outer_face_of"]


@dataclass(frozen=True)
class ConcentricGeometry:
    """The CAD entities of a layered domain, before any identification.

    `cells` are the shells (dimension d), `faces` the interfaces
    (dimension d-1), both in meaningless tag order.  `radii` are the
    interface radii asked for, innermost first: for a hollow domain the
    first is the inner boundary, so there is one more radius than cells.
    """

    dimension: int
    radii: tuple[float, ...]
    cells: tuple[int, ...]
    faces: tuple[int, ...]
    hollow: bool

    @property
    def n_layers(self) -> int:
        return len(self.cells)


def _add_ball(radius: float, dimension: int) -> int:
    """One solid of the given radius, as a disc or a ball."""
    if dimension == 3:
        return gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
    return gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)


def build_concentric(radii, *, dimension: int = 3) -> ConcentricGeometry:
    """Concentric shells from an increasing list of boundary radii.

    `radii` are the skeleton boundaries, innermost first, in the units
    the mesh will be written in.  A leading zero is the centre of a full
    ball and is not a surface; a positive innermost radius makes the
    domain hollow, with that boundary a face of the mesh.
    """
    radii = [float(r) for r in np.atleast_1d(np.asarray(radii, dtype=float))]
    if radii and radii[0] == 0.0:
        radii = radii[1:]
        hollow = False
    else:
        hollow = True
    if radii and radii[0] < 0.0:
        raise ValueError(f"radii must be non-negative, got {radii}")
    if len(radii) < (2 if hollow else 1):
        raise ValueError("need at least one shell: two boundary radii, or one "
                         "above a centre at zero")
    if not all(b > a for a, b in zip(radii[:-1], radii[1:])):
        raise ValueError(f"radii must be strictly increasing, got {radii}")
    if dimension not in (2, 3):
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")

    outer = _add_ball(radii[-1], dimension)
    tools = [(dimension, _add_ball(r, dimension)) for r in radii[:-1]]
    if tools:
        gmsh.model.occ.fragment([(dimension, outer)], tools)
        gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    if hollow:
        # The innermost solid is the hole: remove it, keep its boundary.
        cells = [tag for _, tag in gmsh.model.getEntities(dimension)]
        inner = [tag for tag in cells
                 if abs(entity_radius(dimension, tag) - radii[0])
                 <= 1e-6 * radii[-1]]
        if len(inner) != 1:
            raise RuntimeError(
                f"expected one solid of radius {radii[0]} to remove for the "
                f"hollow, found {len(inner)}")
        gmsh.model.occ.remove([(dimension, inner[0])], recursive=False)
        gmsh.model.occ.synchronize()

    cells = tuple(tag for _, tag in gmsh.model.getEntities(dimension))
    faces = tuple(tag for _, tag in gmsh.model.getEntities(dimension - 1))

    n_shells = len(radii) - (1 if hollow else 0)
    if len(cells) != n_shells:
        raise RuntimeError(
            f"fragment produced {len(cells)} cells for {n_shells} shells; "
            "the CAD kernel did not cut the shells as expected")
    if len(faces) != len(radii):
        raise RuntimeError(
            f"fragment produced {len(faces)} faces for {len(radii)} radii; "
            "the CAD kernel did not cut the shells as expected")
    return ConcentricGeometry(dimension, tuple(radii), cells, faces, hollow)


def entity_radius(dimension: int, tag: int) -> float:
    """The radius of a concentric CAD entity, from its bounding box.

    Exact for a sphere or circle centred on the origin, and available
    before any mesh exists, which is what lets identification happen on
    the geometry asked for rather than on node positions.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
        dimension, tag)
    return 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)


def outer_face_of(dimension: int, cell: int) -> int:
    """The bounding face of a shell with the largest radius.

    A shell is bounded by its inner and outer interfaces; a central ball
    has only one.  Taking the largest makes "interface i is the outer
    boundary of layer i" true by construction.
    """
    boundary = gmsh.model.getBoundary([(dimension, cell)], oriented=False,
                                      recursive=False)
    faces = [tag for d, tag in boundary if d == dimension - 1]
    if not faces:
        raise RuntimeError(f"cell {cell} has no bounding faces")
    return max(faces, key=lambda t: entity_radius(dimension - 1, t))
