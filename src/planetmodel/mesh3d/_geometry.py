"""_geometry.py -- concentric CAD, in 2D and 3D by the same code.

A layered body is a set of concentric balls (3D) or discs (2D), cut
against each other so the shells between them become separate entities.
The dimension is a parameter throughout: 2D discs are the cheap testing
analogue of 3D spheres and must not fork the implementation, or the
cheap tests stop testing the expensive path.

The construction is: build every radius as its own solid, fragment the
outermost against all the others, remove duplicates, synchronize.  What
comes back is one entity per shell and one per interface -- but in
*arbitrary* tag order, which is why identification is a separate step
(_tagging.py) and never a matter of sorting by tag.
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np

__all__ = ["ConcentricGeometry", "build_concentric"]


@dataclass(frozen=True)
class ConcentricGeometry:
    """The CAD entities of a layered body, before any identification.

    `cells` are the shells (dimension d), `faces` the interfaces
    (dimension d-1).  Tag order means nothing; `radii` records what was
    asked for, so tagging can match rather than guess.
    """

    dimension: int
    radii: tuple[float, ...]
    cells: tuple[int, ...]
    faces: tuple[int, ...]

    @property
    def n_layers(self) -> int:
        return len(self.cells)


def _add_ball(radius: float, dimension: int) -> int:
    """One solid of the given radius, as a disc or a ball."""
    if dimension == 3:
        return gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
    return gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)


def build_concentric(radii, *, dimension: int = 3) -> ConcentricGeometry:
    """Concentric shells from an increasing list of radii.

    `radii` are the interface radii, innermost first, in the units the
    mesh will be written in (the caller non-dimensionalises first).  A
    body whose innermost radius is zero is a ball at the centre rather
    than a hole, so a leading zero is dropped: the centre is not a
    surface.
    """
    radii = [float(r) for r in np.atleast_1d(np.asarray(radii, dtype=float))]
    if radii and radii[0] == 0.0:
        radii = radii[1:]
    if len(radii) < 1:
        raise ValueError("need at least one non-zero radius")
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

    cells = tuple(tag for _, tag in gmsh.model.getEntities(dimension))
    faces = tuple(tag for _, tag in gmsh.model.getEntities(dimension - 1))

    if len(cells) != len(radii):
        raise RuntimeError(
            f"fragment produced {len(cells)} cells for {len(radii)} radii; "
            "the CAD kernel did not cut the shells as expected")
    return ConcentricGeometry(dimension, tuple(radii), cells, faces)


def entity_radius(dimension: int, tag: int) -> float:
    """The radius of a concentric CAD entity, from its bounding box.

    Exact for a sphere or circle centred on the origin, and available
    before any mesh exists -- which is what lets identification happen
    before meshing, on the geometry the caller asked for rather than on
    node positions that only exist afterwards.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
        dimension, tag)
    return 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)


def outer_face_of(dimension: int, cell: int) -> int:
    """The bounding face of a shell with the largest radius.

    A shell is bounded by its inner and outer interfaces; the innermost
    cell is a ball and has only one.  Taking the largest makes
    "interface i is the outer boundary of layer i" true by
    construction, which is the numbering convention consumers rely on.
    """
    boundary = gmsh.model.getBoundary([(dimension, cell)], oriented=False,
                                      recursive=False)
    faces = [tag for d, tag in boundary if d == dimension - 1]
    if not faces:
        raise RuntimeError(f"cell {cell} has no bounding faces")
    return max(faces, key=lambda t: entity_radius(dimension - 1, t))
