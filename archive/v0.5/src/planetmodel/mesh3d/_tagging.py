"""_tagging.py -- deciding which CAD entity is which layer.

The attribute numbers are the interface with every consumer: MFEM turns
physical groups into element attributes, and material selection,
SubMesh construction and boundary conditions are all done by number.
So getting this wrong is not a mesh that looks odd, it is a solve that
runs to completion on the wrong materials.

Two decisions follow.  First, identify from the CAD, before meshing:
bounding boxes give the radius of a concentric entity exactly, whereas
a node average depends on how the mesher happened to distribute nodes
and only exists after the fact.  Second, **match, do not sort**.  The
entities come back in arbitrary order, and sorting them by radius
always produces a plausible-looking answer -- including when the CAD
kernel silently dropped or duplicated a shell.  Matching each measured
radius against the radius that was asked for turns that from a wrong
mesh into a loud failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np

from ._geometry import ConcentricGeometry, entity_radius, outer_face_of

__all__ = ["Tagging", "identify", "apply_physical_groups",
           "mean_radius_of_entity", "default_atol"]


@dataclass(frozen=True)
class Tagging:
    """Which CAD entity is which layer, and which interface.

    Both lists run centre outwards, so index i is layer i and interface
    i -- the ordering the physical groups will carry.
    """

    dimension: int
    cells: tuple[int, ...]          # layer i -> CAD entity
    faces: tuple[int, ...]          # interface i -> CAD entity
    radii: tuple[float, ...]        # interface i -> requested radius

    def __repr__(self) -> str:
        return (f"Tagging({len(self.cells)} layers, "
                f"radii {[round(r, 6) for r in self.radii]})")


#: OCC pads every bounding box by roughly this much, *absolutely* --
#: measured at 1.0e-7 whether the radius is 0.2 or 6.371e6, so it is a
#: kernel gap tolerance and not a relative one.  Any matching tolerance
#: must clear it; a purely relative tolerance would be far too tight in
#: non-dimensional units, where radii are of order 1.
_OCC_BBOX_PAD = 1e-7


def default_atol(r_outer: float) -> float:
    """The matching tolerance for a body of the given outer radius.

    Generous against OCC's absolute bounding-box padding, and still far
    tighter than any interface separation worth meshing: PREM's
    thinnest span is 12 km in 6371, or 1.9e-3 non-dimensional, three
    orders above this floor.
    """
    return max(10.0 * _OCC_BBOX_PAD, 1e-9 * float(r_outer))


def identify(geometry: ConcentricGeometry, expected_radii, *,
             atol: float | None = None) -> Tagging:
    """Match CAD entities to the radii they were built from.

    `expected_radii` are the interface radii, innermost first, in the
    geometry's units.  Every expected radius must be matched by exactly
    one face and every face by exactly one radius; anything else raises
    with both lists printed, because a near-match is the symptom of a
    CAD failure and guessing past it produces a silently wrong mesh.

    The recorded radii are the ones asked for, not the measured ones:
    a bounding box is padded by the CAD kernel, and a manifest saying an
    interface sits at 1.0000001 would be recording that padding rather
    than the model.
    """
    expected = [float(r) for r in np.atleast_1d(
        np.asarray(expected_radii, dtype=float))]
    if expected and expected[0] == 0.0:
        expected = expected[1:]
    tol = atol if atol is not None else default_atol(max(expected))

    measured = {tag: entity_radius(geometry.dimension - 1, tag)
                for tag in geometry.faces}

    faces: list[int] = []
    unmatched = dict(measured)
    for r in expected:
        hits = [tag for tag, rad in unmatched.items() if abs(rad - r) <= tol]
        if len(hits) != 1:
            raise RuntimeError(_mismatch(expected, measured, r, hits, tol))
        faces.append(hits[0])
        del unmatched[hits[0]]
    if unmatched:
        raise RuntimeError(_mismatch(expected, measured, None,
                                     list(unmatched), tol))

    # A cell is identified by its own outer face, which makes
    # "interface i bounds layer i" true rather than assumed.
    by_face = {face: i for i, face in enumerate(faces)}
    cells: list[int | None] = [None] * len(faces)
    for cell in geometry.cells:
        face = outer_face_of(geometry.dimension, cell)
        if face not in by_face:
            raise RuntimeError(
                f"cell {cell} has outer face {face}, which matched no expected "
                f"interface radius; measured faces {measured}")
        i = by_face[face]
        if cells[i] is not None:
            raise RuntimeError(
                f"cells {cells[i]} and {cell} both claim interface {i} "
                f"(radius {expected[i]}) as their outer boundary")
        cells[i] = cell
    if any(c is None for c in cells):
        missing = [i for i, c in enumerate(cells) if c is None]
        raise RuntimeError(
            f"no cell has interfaces {missing} as its outer boundary; the CAD "
            f"kernel produced {len(geometry.cells)} cells for "
            f"{len(expected)} interfaces")

    return Tagging(geometry.dimension, tuple(cells), tuple(faces),
                   tuple(expected))


def _mismatch(expected, measured, radius, hits, tol) -> str:
    """A failure message carrying both lists, since either may be at fault."""
    lines = [
        "cannot match CAD entities to the requested geometry "
        f"(tolerance {tol:g}):",
        f"  expected radii : {[round(r, 9) for r in expected]}",
        f"  measured faces : "
        f"{ {t: round(r, 9) for t, r in measured.items()} }",
    ]
    if radius is not None:
        lines.append(
            f"  radius {radius:.9g} matched {len(hits)} faces {hits}; "
            "exactly one was required")
    else:
        lines.append(f"  faces {hits} matched no requested radius")
    return "\n".join(lines)


def apply_physical_groups(tagging: Tagging, *, layer_names=(),
                          interface_names=()) -> dict:
    """Number the layers and interfaces 1..N from the centre outward.

    The convention every consumer reads: physical group i is layer i
    counting out from the centre, and physical group i of the next
    dimension down is that layer's outer boundary.  Names are set where
    the body supplies them, and are advisory -- the numbering is the
    contract.
    """
    d = tagging.dimension
    gmsh.model.removePhysicalGroups()
    out: dict[str, dict[int, int]] = {"layers": {}, "interfaces": {}}

    for i, cell in enumerate(tagging.cells):
        attr = i + 1
        gmsh.model.addPhysicalGroup(d, [cell], attr)
        name = layer_names[i] if i < len(layer_names) and layer_names[i] else None
        gmsh.model.setPhysicalName(d, attr, name or f"layer_{attr}")
        out["layers"][attr] = cell

    for i, face in enumerate(tagging.faces):
        attr = i + 1
        gmsh.model.addPhysicalGroup(d - 1, [face], attr)
        name = (interface_names[i]
                if i < len(interface_names) and interface_names[i] else None)
        gmsh.model.setPhysicalName(d - 1, attr, name or f"interface_{attr}")
        out["interfaces"][attr] = face
    return out


def mean_radius_of_entity(dimension: int, tag: int) -> float:
    """The node-average radius of a meshed entity.

    The fallback for geometry that is not concentric -- the offset
    benchmark bodies -- where a bounding box says nothing useful.  It
    needs a mesh to exist and is biased by node density, which is why
    it is not the primary path.
    """
    _, coords, _ = gmsh.model.mesh.getNodes(dimension, tag, includeBoundary=True)
    if coords.size == 0:
        return 0.0
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    return float(np.mean(np.linalg.norm(xyz, axis=1)))
