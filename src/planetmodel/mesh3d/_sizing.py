"""Telling gmsh how finely to resolve each interface.

One Distance field per interface, each wrapped in a Threshold that
grows the element size from `size` at the boundary to `far_size` over
`decay_width`, and a single Min over all of them so every point takes
the finest requirement that applies to it.  A single Distance field
over every boundary could only express one size for all of them, which
is what a layered domain does not want.
"""
from __future__ import annotations

from collections.abc import Mapping

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from ._tagging import Tagging
from .spec import InterfaceSizing

__all__ = ["apply_size_fields", "apply_mesh_options",
           "check_sizing_resolves_spans", "check_sizing_scale"]


def apply_size_fields(tagging: Tagging, sizes: Mapping[int, InterfaceSizing]) -> int:
    """Build the background size field from per-interface sizings.

    `sizes` maps interface index to InterfaceSizing, in the lengths the
    geometry is drawn in.  Returns the tag of the field set as the
    background.
    """
    if not sizes:
        raise ValueError("no sizing given: every interface needs one")

    entity_key = "SurfacesList" if tagging.dimension == 3 else "CurvesList"
    thresholds: list[float] = []

    for i, face in enumerate(tagging.faces):
        sizing = sizes.get(i)
        if sizing is None:
            raise ValueError(f"no sizing for interface {i}")
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, entity_key, [float(face)])

        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", sizing.size)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", sizing.far_size)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", sizing.decay_width)
        thresholds.append(float(thr))

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", thresholds)
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)
    return combined


def apply_mesh_options(*, order: int, algorithm_2d: int, algorithm_3d: int,
                       size_min: float, size_max: float) -> None:
    """Set the meshing options that go with a background size field.

    The three MeshSizeFrom* options are switched off: with a background
    field in place, sizes inherited from CAD points, from curvature, or
    extended from boundaries compete with the field and quietly win in
    places, so the mesh stops matching the sizing rule asked for.
    """
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", float(size_min))
    gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_max))
    gmsh.option.setNumber("Mesh.ElementOrder", int(order))
    gmsh.option.setNumber("Mesh.Algorithm", int(algorithm_2d))
    gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm_3d))


def check_sizing_resolves_spans(boundaries: ArrayLike,
                                sizes: Mapping[int, InterfaceSizing], *,
                                max_ratio: float = 10.0,
                                strict: bool = True) -> list[str]:
    """Refuse element sizes too coarse for the layers they must fill.

    A shell far thinner than the elements asked for cannot be
    tetrahedralised, and gmsh reports that as a PLC error a long way
    from the sizing rule that caused it.  `max_ratio` is the ratio of
    element size to layer thickness at which gmsh fails.

    `boundaries` are the skeleton boundaries, innermost first (a leading
    zero for a full domain); `sizes` maps interface
    index to InterfaceSizing, interface k sitting on boundary k for a
    hollow domain and on boundary k + 1 for a full one.  Each layer is
    judged by the finer of its two bounding interfaces.
    """
    b = np.asarray(boundaries, dtype=float)
    first = 0 if b[0] > 0.0 else 1          # the boundary interface 0 sits on
    problems = []
    for i in range(b.size - 1):
        thickness = float(b[i + 1] - b[i])
        bounding = [sizes[j].size for j in (i - first, i + 1 - first)
                    if j in sizes and j >= 0]
        h = min(bounding)
        if h > max_ratio * thickness:
            problems.append(
                f"layer [{b[i]:.6g}, {b[i + 1]:.6g}] is "
                f"{thickness:.3g} thick but the elements bounding it are "
                f"{h:.3g} ({h / thickness:.0f}x too coarse; gmsh fails "
                f"above about {max_ratio:.0f}x)")
    if problems and strict:
        raise ValueError(
            "the sizing is too coarse for this domain's thinnest layers, and "
            "gmsh would fail with an unrelated-looking PLC error:\n  - "
            + "\n  - ".join(problems)
            + "\nRefine the sizing, or coarsen the geometry so the thin "
              "layers are merged.")
    return problems


def check_sizing_scale(outer_radius: float, sizes: Mapping[int, InterfaceSizing], *,
                       floor: float = 1e-5, ceiling: float = 2.0) -> None:
    """Catch a sizing given at the wrong scale before gmsh does.

    Sizing rules return lengths in the geometry's own units, the ones
    its radii are in; a rule written for a unit ball and applied to one
    of radius 6.4e6 asks for elements a million times too small, and
    gmsh reports that as "identical points in triangulation", which
    names neither the scale nor the rule.  Both bounds are relative to
    the outer radius: a target element more than a hundred thousand
    times smaller than the domain, or larger than the domain itself, is
    not a resolution choice.
    """
    outer = float(outer_radius)
    smallest = min(s.size for s in sizes.values())
    largest = max(s.size for s in sizes.values())
    if smallest < floor * outer:
        raise ValueError(
            f"the smallest element size is {smallest:.3g} against an outer "
            f"radius of {outer:.3g}, a ratio of {smallest / outer:.1e}, "
            "which is not a resolution choice. Sizing rules take lengths in "
            "the geometry's own units, the ones its radii are in.")
    if largest > ceiling * outer:
        raise ValueError(
            f"the largest element size is {largest:.3g}, more than the outer "
            f"radius {outer:.3g}: nothing would be resolved.")
