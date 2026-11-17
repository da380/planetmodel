"""Deciding which CAD entity is which layer.

The attribute numbers are the interface with every consumer: MFEM turns
physical groups into element attributes, and material selection,
submesh construction and boundary conditions are all done by number.
Getting this wrong is not a mesh that looks odd, it is a solve that
runs to completion on the wrong materials.

Two decisions follow.  Entities are identified from the CAD, before
meshing: bounding boxes give the radius of a concentric entity exactly.
And entities are matched, not sorted: they come back in arbitrary
order, and sorting by radius always produces a plausible answer,
including when the CAD kernel dropped or duplicated a shell.  Matching
each measured radius against the radius that was asked for turns that
into a loud failure.

A tagging answers for what the sizing, the orientation and the checks
need of each entity: the distance to an interface as an expression gmsh
evaluates (`distance`), the measured place of an interface (`measure`),
and, where a boundary has one, its outward direction (`directions`); a
box's (`_box.PlanarTagging`) answers with planes where a concentric
domain's answers with spheres.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from ._geometry import ConcentricGeometry, default_atol, entity_radius, outer_face_of

__all__ = ["Tagging", "identify", "apply_physical_groups",
           "mean_radius_of_entity", "default_atol", "distance_to_sphere",
           "FaceGroups"]

#: Boundary groups beyond the interfaces, in the order of their
#: attributes: (name, the entities the group holds).
type FaceGroups = tuple[tuple[str, tuple[int, ...]], ...]


def distance_to_sphere(centre: ArrayLike, radius: float) -> str:
    """The distance | |x - c| - R | as an expression gmsh evaluates.

    Each number is written with the digits that round-trip and in
    brackets, so a sign or an exponent is read as part of it.
    """
    cx, cy, cz = (float(c) for c in np.asarray(centre, dtype=float))
    return (f"Fabs(Sqrt((x-({cx!r}))^2+(y-({cy!r}))^2+(z-({cz!r}))^2)"
            f"-({float(radius)!r}))")


@dataclass(frozen=True)
class Tagging:
    """Which CAD entity is which layer, and which interface.

    Both lists run in the skeleton's order, centre outwards here:
    `cells[i]` is layer i and `faces[k]` interface k, the ordering the
    physical groups will carry.  For a hollow domain `faces[0]` is the
    inner boundary and layer i's outer boundary is `faces[i + 1]`; for a
    full one it is `faces[i]`.  `groups` are boundary groups beyond the
    interfaces, numbered after them (a box's base and sides), and
    `directions` the outward direction of every boundary entity that has
    one in its own right, in the mesh's coordinates; a concentric domain
    has neither, its boundaries facing away from the centre they
    enclose.
    """

    dimension: int
    cells: tuple[int, ...]          # layer i -> CAD entity
    faces: tuple[int, ...]          # interface k -> CAD entity
    radii: tuple[float, ...]        # interface k -> requested radius
    _: KW_ONLY
    hollow: bool = False
    groups: FaceGroups = ()
    directions: Mapping[int, tuple[float, float, float]] = field(default_factory=dict)

    #: The name of the layered coordinate an interface is placed by.
    coordinate = "radius"

    def distance(self, k: int, *, centre: ArrayLike = (0.0, 0.0, 0.0)) -> str:
        """The distance to interface k, an expression gmsh evaluates: to
        the sphere of its radius about `centre`."""
        return distance_to_sphere(centre, self.radii[k])

    def measure(self, tag: int) -> float:
        """The measured place of an interface's entity: its nodes' mean
        radius (`mean_radius_of_entity`)."""
        return mean_radius_of_entity(self.dimension - 1, tag)

    def __repr__(self) -> str:
        return (f"Tagging({len(self.cells)} layers, "
                f"radii {[round(r, 6) for r in self.radii]})")


def identify(geometry: ConcentricGeometry, expected_radii: ArrayLike, *,
             atol: float | None = None) -> Tagging:
    """Match CAD entities to the radii they were built from.

    `expected_radii` are the interface radii, innermost first, in the
    geometry's units; for a hollow domain the first is the inner
    boundary.  Every expected radius must be matched by exactly one
    face and every face by exactly one radius; anything else raises
    with both lists printed, because a near-match is the symptom of a
    CAD failure and guessing past it produces a silently wrong mesh.

    The recorded radii are the ones asked for, not the measured ones: a
    bounding box is padded by the CAD kernel, and a manifest saying an
    interface sits at 1.0000001 would be recording the padding.
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

    # A cell is identified by its own outer face, which makes "interface
    # i bounds layer i" true rather than assumed.
    first = 1 if geometry.hollow else 0
    by_face = {face: k for k, face in enumerate(faces)}
    cells: list[int | None] = [None] * (len(faces) - first)
    for cell in geometry.cells:
        face = outer_face_of(geometry.dimension, cell)
        if face not in by_face:
            raise RuntimeError(
                f"cell {cell} has outer face {face}, which matched no expected "
                f"interface radius; measured faces {measured}")
        k = by_face[face]
        if k < first:
            raise RuntimeError(
                f"cell {cell} has the inner boundary (radius {expected[k]}) as "
                "its outer face; the hollow was not cut")
        i = k - first
        if cells[i] is not None:
            raise RuntimeError(
                f"cells {cells[i]} and {cell} both claim interface {k} "
                f"(radius {expected[k]}) as their outer boundary")
        cells[i] = cell
    if any(c is None for c in cells):
        missing = [i for i, c in enumerate(cells) if c is None]
        raise RuntimeError(
            f"no cell has layers {missing} bounded above; the CAD kernel "
            f"produced {len(geometry.cells)} cells for {len(expected)} "
            "interfaces")

    return Tagging(geometry.dimension, tuple(cells), tuple(faces),
                   tuple(expected), hollow=geometry.hollow)


def _mismatch(expected: Sequence[float], measured: Mapping[int, float],
              radius: float | None, hits: Sequence[int], tol: float) -> str:
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


def apply_physical_groups(tagging: Tagging, *,
                          layer_names: Sequence[str | None] = (),
                          interface_names: Sequence[str | None] = ()
                          ) -> dict[str, dict[int, int]]:
    """Number the layers and interfaces 1..N in the skeleton's order.

    Physical group i of the mesh dimension is layer i, counting out from
    the centre on a sphere and down from the surface in a box, and
    physical group k of the next dimension down is interface k.  Names
    are set where given and default to `layer_<i>` and `interface_<k>`;
    the numbering is the contract.  The tagging's other boundary groups
    (a box's faces) follow the interfaces, under their own names, and
    are returned under "faces" where there are any.
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

    if tagging.groups:
        out["faces"] = {}
    for j, (name, entities) in enumerate(tagging.groups):
        attr = len(tagging.faces) + j + 1
        gmsh.model.addPhysicalGroup(d - 1, list(entities), attr)
        gmsh.model.setPhysicalName(d - 1, attr, name)
        out["faces"][attr] = entities[0]
    return out


def mean_radius_of_entity(dimension: int, tag: int) -> float:
    """The node-average radius of a meshed entity.

    The measure for geometry that is not concentric, where a bounding
    box says nothing useful.  It needs a mesh to exist and is biased by
    node density, which is why it is not the primary path.
    """
    _, coords, _ = gmsh.model.mesh.getNodes(dimension, tag, includeBoundary=True)
    if coords.size == 0:
        return 0.0
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    return float(np.mean(np.linalg.norm(xyz, axis=1)))
