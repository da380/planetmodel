"""A generated mesh is checked before it is trusted.

A mesh that is subtly wrong (an interface away from where the geometry
puts it, a folded element among many good ones, an attribute numbering
that does not match its own manifest) produces a solve that runs to
completion and answers the wrong question.  Every check here exists
because its failure would otherwise be invisible.  The report lives in
`spec` so the result's type is public.
"""
from __future__ import annotations

from collections.abc import Sequence

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from ._orient import (Centres, element_quality, node_positions, outward_dots,
                      signed_measures)
from ._tagging import Tagging, mean_radius_of_entity
from .spec import QUALITY_FLOOR, ValidationReport

__all__ = ["check_interface_radii", "validate_mesh"]


def check_interface_radii(tagging: Tagging, expected: ArrayLike, *,
                          tolerance: float | None = None
                          ) -> tuple[float, list[str]]:
    """Measure each tagged interface against the radius asked for.

    Returns (worst_error, failures).  The tolerance defaults to a
    fraction of the thinnest span, the scale at which being wrong would
    matter: an error smaller than that cannot have put an interface in
    the wrong layer.  A non-finite expected radius marks a boundary
    that is not concentric and has nothing to be checked against.

    This is a property of the reference geometry, so for a physical
    delivery it runs before the nodes are displaced: afterwards the
    interfaces are supposed to be away from these radii.
    """
    expected = np.asarray(expected, dtype=float)
    if tolerance is None:
        spans = np.diff(np.concatenate(([0.0], expected[np.isfinite(expected)])))
        tolerance = 0.05 * float(spans.min()) if spans.size else float("inf")
    d = tagging.dimension
    worst, failures = 0.0, []
    for i, face in enumerate(tagging.faces):
        if not np.isfinite(expected[i]):
            continue
        got = mean_radius_of_entity(d - 1, face)
        err = abs(got - expected[i])
        worst = max(worst, err)
        if err > tolerance:
            failures.append(
                f"interface {i + 1} has mean radius {got:.9g}, expected "
                f"{expected[i]:.9g} (error {err:.3g} > {tolerance:.3g})")
    return worst, failures


def validate_mesh(tagging: Tagging, *, expected_radii: ArrayLike,
                  layer_names: Sequence[str | None] = (),
                  interface_names: Sequence[str | None] = (),
                  radius_tolerance: float | None = None,
                  radius_check: tuple[float, list[str]] | None = None,
                  quality_warn: float = QUALITY_FLOOR,
                  centres: Centres | None = None) -> ValidationReport:
    """Check a finished mesh against what was asked for.

    `expected_radii` are the interface radii the mesh was built at.
    `radius_check` is a (worst_error, failures) pair from
    `check_interface_radii` measured before the nodes were displaced;
    without it the radii are measured here, which is correct only while
    the mesh is still the reference one.  `centres` maps a surface's
    entity tag to the point it encloses, for surfaces not centred on
    the origin.
    """
    d = tagging.dimension
    rep = ValidationReport(dimension=d)

    expected = np.asarray(expected_radii, dtype=float)

    # -- physical groups match the tagging ---------------------------------
    for dim, what, wanted in ((d, "layers", len(tagging.cells)),
                              (d - 1, "interfaces", len(tagging.faces))):
        got = [tag for _, tag in gmsh.model.getPhysicalGroups(dim)]
        rep.group_counts[what] = len(got)
        if sorted(got) != list(range(1, wanted + 1)):
            rep.failures.append(
                f"{what}: physical groups are {sorted(got)}, expected "
                f"1..{wanted} numbered from the centre outward")

    names = {"layers": layer_names, "interfaces": interface_names}
    for dim, what in ((d, "layers"), (d - 1, "interfaces")):
        for i, want in enumerate(names[what]):
            if not want:
                continue
            got = gmsh.model.getPhysicalName(dim, i + 1)
            if got != want:
                rep.failures.append(
                    f"{what[:-1]} {i + 1} is named {got!r}, expected {want!r}")

    # -- interfaces sit where the geometry puts them -----------------------
    if radius_check is None:
        radius_check = check_interface_radii(tagging, expected,
                                             tolerance=radius_tolerance)
    rep.max_interface_radius_error = float(radius_check[0])
    rep.failures.extend(radius_check[1])

    # -- elements are valid ------------------------------------------------
    min_sicn, invalid, _ = element_quality(d)
    rep.min_sicn = min_sicn
    rep.negative_jacobians = invalid
    if invalid:
        rep.failures.append(
            f"{invalid} element(s) have non-positive Jacobians (worst minSICN "
            f"{min_sicn:.4g}); the mesh is folded somewhere")
    elif min_sicn < quality_warn:
        rep.warnings.append(
            f"worst element quality is minSICN {min_sicn:.4g}, below "
            f"{quality_warn}: usable but poorly shaped")

    # -- orientation is consistent -----------------------------------------
    pos = node_positions()
    rep.negative_cells = sum(
        int((signed_measures(d, tag, pos)[1] < 0.0).sum())
        for _, tag in gmsh.model.getEntities(d))
    if rep.negative_cells:
        rep.failures.append(
            f"{rep.negative_cells} cell(s) are negatively oriented; MFEM would "
            "report them as wrong orientation and silently fix them")
    if d == 3:
        centres = centres or {}
        rep.inward_faces = sum(
            int((outward_dots(tag, pos, centre=centres.get(tag, (0.0, 0.0, 0.0)))[1]
                 < 0.0).sum())
            for _, tag in gmsh.model.getEntities(2))
        if rep.inward_faces:
            rep.failures.append(
                f"{rep.inward_faces} boundary face(s) point inward")

    return rep
