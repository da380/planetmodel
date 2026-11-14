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

Surfaces built into the domain (faults) join the one fragment as
tools, which cuts each into pieces at the spheres it crosses, embeds
every piece in the solid holding it and every crossing curve in its
sphere, and makes curves where surfaces meet common to both, so the
mesh conforms to all of it.  A fragment already leaves the entities
conforming, so duplicates are not removed after it, which would
renumber the pieces the fragment's map names.  What becomes of each
surface is that map, piece by piece, read before the hole is removed:
a piece in the hole is reported, for the caller to refuse.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass

import gmsh
import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ConcentricGeometry", "build_concentric", "entity_radius",
           "outer_face_of", "default_atol"]


#: OCC pads every bounding box by roughly this much, absolutely: the
#: same 1e-7 whether the radius is 0.2 or 6e6.  A matching tolerance
#: must clear it, so a purely relative one would be too tight for a
#: domain of radius near one.
_OCC_BBOX_PAD = 1e-7


def default_atol(r_outer: float) -> float:
    """The matching tolerance for a domain of the given outer radius.

    Generous against OCC's absolute bounding-box padding, and still far
    tighter than any interface separation worth meshing.
    """
    return max(10.0 * _OCC_BBOX_PAD, 1e-9 * float(r_outer))


@dataclass(frozen=True)
class ConcentricGeometry:
    """The CAD entities of a layered domain, before any identification.

    `cells` are the shells (dimension d), `faces` the interfaces
    (dimension d-1), both in meaningless tag order.  `radii` are the
    interface radii asked for, innermost first: for a hollow domain the
    first is the inner boundary, so there is one more radius than cells.
    `pieces` holds, for each surface fragmented in, the faces it became,
    and `in_hole` the pieces that lay in the hole of a hollow domain,
    left behind when it was removed.
    """

    dimension: int
    radii: tuple[float, ...]
    cells: tuple[int, ...]
    faces: tuple[int, ...]
    hollow: bool
    _: KW_ONLY
    pieces: tuple[tuple[int, ...], ...] = ()
    in_hole: tuple[int, ...] = ()

    @property
    def n_layers(self) -> int:
        """The number of shells: one cell each."""
        return len(self.cells)


def _add_ball(radius: float, dimension: int) -> int:
    """One solid of the given radius, as a disc or a ball."""
    if dimension == 3:
        return gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
    return gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)


def _fragment_surfaces(radii: Sequence[float], surfaces: Sequence[int],
                       trim: float) -> tuple[tuple[int, ...], ...]:
    """Fragment the balls of `radii` and the surfaces, trimming the surfaces
    at the ball of radius `trim`: the pieces of each surface that remain.

    The surfaces are fragmented with the balls out to `trim` first, which
    leaves the pieces beyond it in no solid, where they can be removed,
    and then the balls beyond it are fragmented in with everything, the
    pieces mapped through.  Fragmented with every ball at once, a piece
    beyond `trim` would be embedded in a shell, and OCC removes nothing
    embedded in a solid.
    """
    occ = gmsh.model.occ
    balls = {r: _add_ball(r, 3) for r in radii}
    inside = [r for r in radii if r <= trim]
    beyond = [r for r in radii if r > trim]
    _, mapped = occ.fragment([(3, balls[inside[-1]])],
                             [(3, balls[r]) for r in inside[:-1]]
                             + [(2, s) for s in surfaces])
    pieces = [tuple(tag for _, tag in out) for out in mapped[len(inside):]]
    occ.synchronize()
    held = {tag for _, v in gmsh.model.getEntities(3)
            for dim, tag in gmsh.model.mesh.getEmbedded(3, v) if dim == 2}
    excess = [(2, tag) for group in pieces for tag in group if tag not in held]
    if excess:
        occ.remove(excess, recursive=True)
        occ.synchronize()
    pieces = [tuple(tag for tag in group if tag in held) for group in pieces]
    if beyond:
        flat = [tag for group in pieces for tag in group]
        volumes = [(3, v) for _, v in gmsh.model.getEntities(3)]
        tools = [(3, balls[r]) for r in beyond[:-1]] + volumes
        _, mapped = occ.fragment([(3, balls[beyond[-1]])],
                                 tools + [(2, tag) for tag in flat])
        renamed = {tag: tuple(t for _, t in out)
                   for tag, out in zip(flat, mapped[1 + len(tools):])}
        pieces = [tuple(t for tag in group for t in renamed[tag])
                  for group in pieces]
        occ.synchronize()
    return tuple(pieces)


def build_concentric(radii: ArrayLike, *, dimension: int = 3,
                     surfaces: Sequence[int] = (), trim: float | None = None
                     ) -> ConcentricGeometry:
    """Concentric shells from an increasing list of boundary radii.

    `radii` are the skeleton boundaries, innermost first, in the units
    the mesh will be written in.  A leading zero is the centre of a full
    ball and is not a surface; a positive innermost radius makes the
    domain hollow, with that boundary a face of the mesh.  `surfaces`
    are the tags of OCC surfaces already made in the session, fragmented
    in with the shells (3D only) and trimmed at the sphere of radius
    `trim`, one of `radii` (the outermost by default): their pieces
    beyond it are removed.  A surface that cuts a shell or an interface
    into separate pieces is refused.
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
    surfaces = tuple(int(s) for s in surfaces)
    if surfaces and dimension != 3:
        raise ValueError("surfaces are fragmented into a 3D domain only")

    pieces: tuple[tuple[int, ...], ...] = ()
    if surfaces:
        trim = radii[-1] if trim is None else float(trim)
        if trim not in radii:
            raise ValueError(f"surfaces are trimmed at one of the radii {radii}, "
                             f"not {trim}")
        pieces = _fragment_surfaces(radii, surfaces, trim)
    else:
        outer = _add_ball(radii[-1], dimension)
        tools = [(dimension, _add_ball(r, dimension)) for r in radii[:-1]]
        if tools:
            gmsh.model.occ.fragment([(dimension, outer)], tools)
            gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

    in_hole: tuple[int, ...] = ()
    if hollow:
        # The innermost solid is the hole: remove it, keep its boundary.
        cells = [tag for _, tag in gmsh.model.getEntities(dimension)]
        inner = [tag for tag in cells
                 if abs(entity_radius(dimension, tag) - radii[0])
                 <= default_atol(radii[-1])]
        if len(inner) != 1:
            raise RuntimeError(
                f"expected one solid of radius {radii[0]} to remove for the "
                f"hollow, found {len(inner)}")
        if pieces:
            in_hole = tuple(tag for dim, tag in gmsh.model.mesh.getEmbedded(
                dimension, inner[0]) if dim == 2)
        gmsh.model.occ.remove([(dimension, inner[0])], recursive=False)
        gmsh.model.occ.synchronize()

    made = {tag for group in pieces for tag in group}
    cells = tuple(tag for _, tag in gmsh.model.getEntities(dimension))
    faces = tuple(tag for _, tag in gmsh.model.getEntities(dimension - 1)
                  if tag not in made)

    n_shells = len(radii) - (1 if hollow else 0)
    if pieces and (len(cells) != n_shells or len(faces) != len(radii)):
        raise ValueError(
            f"the faults cut the domain into {len(cells)} solids and "
            f"{len(faces)} spherical faces where its {n_shells} layers and "
            f"{len(radii)} interfaces should stay whole: a fault must not cut "
            "a layer or an interface right through (it may end inside a layer "
            "or on the outer surface)")
    if len(cells) != n_shells:
        raise RuntimeError(
            f"fragment produced {len(cells)} cells for {n_shells} shells; "
            "the CAD kernel did not cut the shells as expected")
    if len(faces) != len(radii):
        raise RuntimeError(
            f"fragment produced {len(faces)} faces for {len(radii)} radii; "
            "the CAD kernel did not cut the shells as expected")
    return ConcentricGeometry(dimension, tuple(radii), cells, faces, hollow,
                              pieces=pieces, in_hole=in_hole)


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
