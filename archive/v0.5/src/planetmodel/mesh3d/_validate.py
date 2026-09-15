"""_validate.py -- a generated mesh is checked before it is trusted.

Validation is part of the product, not a debugging aid.  A mesh that is
subtly wrong -- an interface a kilometre from where the model puts it, a
folded element among a million good ones, an attribute numbering that
does not match its own manifest -- produces a solve that runs to
completion and answers the wrong question.  Every check here exists
because its failure would otherwise be invisible.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

import gmsh
import numpy as np

from ._orient import element_quality, node_positions, outward_dots, signed_measures
from ._tagging import mean_radius_of_entity

__all__ = ["ValidationReport", "check_interface_radii", "validate_mesh"]


@dataclass
class ValidationReport:
    """The outcome of every check, whether or not any of them failed."""

    dimension: int
    _: KW_ONLY
    negative_jacobians: int = 0
    min_sicn: float = 0.0
    negative_cells: int = 0
    inward_faces: int = 0
    max_interface_radius_error: float = 0.0
    knots_aligned: bool = True
    group_counts: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> "ValidationReport":
        """Raise with every failure listed, not just the first."""
        if self.failures:
            raise ValueError(
                "the generated mesh failed validation:\n  - "
                + "\n  - ".join(self.failures))
        return self

    def __repr__(self) -> str:
        state = "ok" if self.ok else f"{len(self.failures)} FAILED"
        return (f"ValidationReport({state}, minSICN {self.min_sicn:.4g}, "
                f"max interface error {self.max_interface_radius_error:.3g})")


def check_interface_radii(tagging, expected, *,
                          tolerance: float | None = None) -> tuple:
    """Measure each tagged interface against the radius the model gives it.

    Returns (worst_error, failures).  The tolerance defaults to a
    fraction of the thinnest span, which is the scale at which being
    wrong would matter: an error smaller than that cannot have put an
    interface in the wrong layer.

    A non-finite expected radius means the boundary is not concentric
    and there is nothing to check it against -- the offset benchmark
    bodies, whose inclusion has a mean radius that is a property of the
    offset rather than a promise the geometry made.  It is skipped
    rather than compared against a number invented for the purpose.

    This check is a property of the *reference* geometry -- did CAD and
    tagging put interface i at radius r_i -- so for a physical delivery it must run
    before the nodes are displaced: afterwards the interfaces carry
    their relief and are supposed to be away from these radii.
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


def validate_mesh(tagging, *, expected_radii, layer_names=(),
                  interface_names=(), mapping=None,
                  radius_tolerance: float | None = None,
                  radius_check: tuple | None = None,
                  quality_warn: float = 0.05,
                  centres: dict | None = None) -> ValidationReport:
    """Check a finished mesh against what was asked for.

    `expected_radii` are the interface radii in mesh units.
    `radius_check` is a (worst_error, failures) pair from
    check_interface_radii, measured on the reference mesh before the
    nodes were displaced; without it the radii are measured here, which
    is correct only while the mesh is still the reference one.
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

    # -- interfaces sit where the model puts them --------------------------
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

    # -- the mapping's kinks land on element boundaries --------------------
    if mapping is not None:
        # A scaled mapping exposes knots itself; a bare RadialStretch
        # keeps them on its displacement.  Both are checked, or a
        # non-dimensional body -- which needs no scaling -- would pass.
        declared = getattr(mapping, "knots", None)
        if declared is None:
            declared = getattr(getattr(mapping, "h", None), "knots", ())
        knots = [k for k in declared if 0.0 < k < float(expected[-1])]
        stray = [k for k in knots
                 if np.min(np.abs(expected - k)) > 1e-9 * float(expected[-1])]
        rep.knots_aligned = not stray
        if stray:
            rep.warnings.append(
                f"the displacement kinks at {[round(k, 9) for k in stray]}, "
                "which are not meshed interfaces: dh/dr jumps inside elements "
                "there, so quadrature will not see it. Insert interfaces at "
                "those radii, or accept the loss of accuracy.")

    return rep
