"""A generated mesh is checked before it is trusted.

A mesh that is subtly wrong (an interface away from where the geometry
puts it, a folded element among many good ones, an attribute numbering
that does not match its own manifest) produces a solve that runs to
completion and answers the wrong question.  Every check here exists
because its failure would otherwise be invisible.  The report lives in
`spec` so the result's type is public.

A fault is held to two more: each of its faces is a face of exactly
two cells (the mesh conforms to it, and no node is split), and each
points from the footwall into the hanging wall.
"""
from __future__ import annotations

from collections.abc import Sequence

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from ._orient import (Centres, FaultSurfaces, element_quality, fault_dots,
                      node_positions, outward_dots, signed_measures)
from ._tagging import Tagging, mean_radius_of_entity
from .spec import QUALITY_FLOOR, ValidationReport

__all__ = ["check_interface_radii", "nonconforming_faces", "validate_mesh"]


def check_interface_radii(tagging: Tagging, expected: ArrayLike, *,
                          tolerance: float | None = None
                          ) -> tuple[float, list[str]]:
    """Measure each tagged interface against the radius asked for.

    Returns (worst_error, failures).  The tolerance defaults to a
    fraction of the thinnest span, the scale at which being wrong would
    matter: an error smaller than that cannot have put an interface in
    the wrong layer.  A non-finite expected radius marks a boundary
    that is not concentric and has nothing to be checked against.

    This is a property of the reference geometry, which the mesh is.
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


def _corners(dimension: int, tag: int) -> np.ndarray:
    """The corner nodes of an entity's simplices, (n, dimension + 1),
    whatever their order."""
    _, tags, nodes = gmsh.model.mesh.getElements(dimension, tag)
    blocks = [np.asarray(block, dtype=np.int64).reshape(len(t), -1)[:, :dimension + 1]
              for t, block in zip(tags, nodes) if len(t)]
    return (np.concatenate(blocks) if blocks
            else np.empty((0, dimension + 1), dtype=np.int64))


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The index of each row of `rows` among the rows of `table`, or -1
    where it is absent."""
    view = np.dtype((np.void, table.dtype.itemsize * table.shape[1]))
    keys = np.ascontiguousarray(table).view(view).ravel()
    want = np.ascontiguousarray(rows).view(view).ravel()
    if not keys.size:
        return np.full(want.size, -1)
    order = np.argsort(keys)
    at = np.clip(np.searchsorted(keys[order], want), 0, keys.size - 1)
    return np.where(keys[order][at] == want, order[at], -1)


def nonconforming_faces(dimension: int, faces: Sequence[int]) -> int:
    """How many elements of the given (d-1)-dimensional entities are not a
    face of exactly two d-dimensional cells, matched by their corners."""
    conn = np.concatenate([_corners(dimension, tag)
                           for _, tag in gmsh.model.getEntities(dimension)])
    sides = np.sort(np.concatenate([np.delete(conn, j, axis=1)
                                    for j in range(dimension + 1)]), axis=1)
    known, counts = np.unique(sides, axis=0, return_counts=True)
    bad = 0
    for tag in faces:
        own = np.sort(_corners(dimension - 1, tag), axis=1)
        at = _rows_in(own, known)
        bad += int(np.count_nonzero((at < 0) | (counts[np.maximum(at, 0)] != 2)))
    return bad


def validate_mesh(tagging: Tagging, *, expected_radii: ArrayLike,
                  layer_names: Sequence[str | None] = (),
                  interface_names: Sequence[str | None] = (),
                  radius_tolerance: float | None = None,
                  quality_warn: float = QUALITY_FLOOR,
                  centres: Centres | None = None,
                  faults: FaultSurfaces | None = None,
                  fault_names: Sequence[str] = ()) -> ValidationReport:
    """Check a finished mesh against what was asked for.

    `expected_radii` are the interface radii the mesh was built at.
    `centres` maps a surface's
    entity tag to the point it encloses, for surfaces not centred on
    the origin.  `faults` maps each face of a fault to the patch it was
    built from, and `fault_names` names the faults, whose boundary
    groups follow the interfaces'.
    """
    d = tagging.dimension
    rep = ValidationReport(dimension=d)
    faults = faults or {}

    expected = np.asarray(expected_radii, dtype=float)

    # -- physical groups match the tagging ---------------------------------
    for dim, what, wanted in ((d, "layers", len(tagging.cells)),
                              (d - 1, "interfaces",
                               len(tagging.faces) + len(fault_names))):
        got = [tag for _, tag in gmsh.model.getPhysicalGroups(dim)]
        rep.group_counts[what] = len(got)
        if sorted(got) != list(range(1, wanted + 1)):
            rep.failures.append(
                f"{what}: physical groups are {sorted(got)}, expected "
                f"1..{wanted} numbered from the centre outward"
                + (", the faults' after the interfaces'" if fault_names else ""))
    if fault_names:
        rep.group_counts["interfaces"] -= len(fault_names)
        rep.group_counts["faults"] = len(fault_names)

    m = len(tagging.faces)
    boundary = list(interface_names)[:m]
    boundary += [None] * (m - len(boundary)) + list(fault_names)
    names = {"layers": layer_names, "interfaces": boundary}
    for dim, what in ((d, "layers"), (d - 1, "interfaces")):
        for i, want in enumerate(names[what]):
            if not want:
                continue
            got = gmsh.model.getPhysicalName(dim, i + 1)
            if got != want:
                rep.failures.append(
                    f"{what[:-1]} {i + 1} is named {got!r}, expected {want!r}")

    # -- interfaces sit where the geometry puts them -----------------------
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
            for _, tag in gmsh.model.getEntities(2) if tag not in faults)
        if rep.inward_faces:
            rep.failures.append(
                f"{rep.inward_faces} boundary face(s) point inward")

    # -- every fault is an oriented interior surface -----------------------
    if faults:
        rep.misoriented_fault_faces = sum(
            int((fault_dots(tag, patch, pos)[1] <= 0.0).sum())
            for tag, patch in faults.items())
        if rep.misoriented_fault_faces:
            rep.failures.append(
                f"{rep.misoriented_fault_faces} fault face(s) do not point from "
                "the footwall into the hanging wall")
        rep.nonconforming_fault_faces = nonconforming_faces(d, list(faults))
        if rep.nonconforming_fault_faces:
            rep.failures.append(
                f"{rep.nonconforming_fault_faces} fault face(s) are not a face of "
                "exactly two cells: the mesh does not conform to the fault")

    return rep
