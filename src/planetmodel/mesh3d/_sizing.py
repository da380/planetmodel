"""Telling gmsh how finely to resolve each interface.

One distance per interface, each wrapped in a Threshold that grows the
element size from `size` at the boundary to `far_size` over
`decay_width`, and a single Min over all of them so every point takes
the finest requirement that applies to it.  A single distance to every
boundary could only express one size for all of them, which is what a
layered domain does not want.

Every interface is a full circle or sphere when it is meshed, of radius
R about a centre c, so the distance to it is | |x - c| - R |, or a
plane across a box, |z - z_k|, and is given to gmsh in closed form, as
the tagging states it (`Tagging.distance`).  gmsh's own Distance field
is not used:
it measures the distance to a sampling of the entity by points, which
on the interface itself is zero at the points and about half their
spacing between them, and the Threshold turns that into an element size
that ripples along the interface and is on average coarser than the one
asked for, the more so the shorter the decay width.

Refinements go through gmsh's size callback instead of its field graph:
each is an arbitrary size function this library evaluates itself, and
gmsh hands the callback the size its other sources computed, so the
minimum over them is taken here, in one visible place, rather than by
options gmsh applies in an order of its own.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import gmsh
import numpy as np
from numpy.typing import ArrayLike

from ._orient import Centres
from ._tagging import Tagging
from .spec import InterfaceSizing, Refinement

__all__ = ["apply_size_fields", "apply_refinements", "apply_mesh_options",
           "check_sizing_resolves_spans", "check_sizing_scale"]


def apply_size_fields(tagging: Tagging, sizes: Mapping[int, InterfaceSizing], *,
                      centres: Centres | None = None) -> int:
    """Build the background size field from per-interface sizings.

    `sizes` maps interface index to InterfaceSizing, in the lengths the
    geometry is drawn in.  The distance to each interface is the
    tagging's (`Tagging.distance`); `centres` maps a sphere's entity tag
    to its centre where that is not the origin.  Returns the tag of the
    field set as the background.
    """
    if not sizes:
        raise ValueError("no sizing given: every interface needs one")

    centres = centres or {}
    thresholds: list[float] = []

    for i, face in enumerate(tagging.faces):
        sizing = sizes.get(i)
        if sizing is None:
            raise ValueError(f"no sizing for interface {i}")
        dist = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            dist, "F", tagging.distance(i, centre=centres.get(face, (0.0, 0.0, 0.0))))

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


def apply_refinements(refinements: Sequence[Refinement], *,
                      dimension: int) -> Callable[[], None]:
    """Register the refinements as gmsh's size callback, and return the
    check to call after meshing.

    gmsh computes a size from its other sources — here the background
    field, every competing option being off — and hands it to the
    callback as `lc`; what the callback returns is the size used.  The
    callback takes the minimum of `lc` and every refinement at the
    point, so a refinement can only refine.  gmsh reports an exception
    raised in the callback and meshes on, so the callback keeps the
    first one and returns `lc`, and the returned check raises it.  With
    no refinements no callback is registered and meshing is untouched;
    a registered one lives until its session closes.
    """
    refinements = tuple(refinements)
    failures: list[Exception] = []

    def check() -> None:
        if failures:
            raise RuntimeError("a refinement failed while gmsh meshed; the mesh "
                               "is not the one asked for") from failures[0]

    if not refinements:
        return check
    d = int(dimension)

    def size_at(dim: int, tag: int, x: float, y: float, z: float,
                lc: float) -> float:
        if failures:
            return lc
        point = np.array([(x, y, z)[:d]])
        try:
            return min(lc, *(float(fine(point)[0]) for fine in refinements))
        except Exception as error:
            failures.append(error)
            return lc

    gmsh.model.mesh.setSizeCallback(size_at)
    return check


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
                                faces: Sequence[int] | None = None,
                                max_ratio: float = 10.0,
                                strict: bool = True) -> list[str]:
    """Refuse element sizes too coarse for the layers they must fill.

    A shell far thinner than the elements asked for cannot be
    tetrahedralised, and gmsh reports that as a PLC error a long way
    from the sizing rule that caused it.  `max_ratio` is the ratio of
    element size to layer thickness at which gmsh fails.

    `boundaries` are the skeleton boundaries in its order; `sizes` maps
    interface index to InterfaceSizing.  `faces` gives the boundary each
    interface sits on, in order: on a sphere every boundary but a
    leading zero, the centre, and in a box every boundary but the last,
    the base.  None takes the sphere's: interface k on boundary k for a
    hollow domain and on boundary k + 1 for a full one.  Each layer is
    judged by the finer of its bounding interfaces, a layer with only
    one (at the centre, or above the base) by that one.
    """
    b = np.asarray(boundaries, dtype=float)
    if faces is None:
        first = 0 if b[0] > 0.0 else 1      # the boundary interface 0 sits on
        faces = range(first, b.size)
    on = {int(j): k for k, j in enumerate(faces)}   # boundary -> interface
    problems = []
    for i in range(b.size - 1):
        thickness = float(b[i + 1] - b[i])
        bounding = [sizes[on[j]].size for j in (i, i + 1)
                    if j in on and on[j] in sizes]
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
            + "\nRefine the sizing, cap it against thin layers "
              "(CappedInterfaces), or coarsen the geometry so the thin "
              "layers are merged.")
    return problems


def check_sizing_scale(outer_radius: float, sizes: Mapping[int, InterfaceSizing], *,
                       refinements: Sequence[Refinement] = (),
                       floor: float = 1e-5, ceiling: float = 2.0) -> None:
    """Catch a sizing given at the wrong scale before gmsh does.

    Sizing rules and refinements return lengths in the geometry's own
    units, the ones its radii are in; a rule written for a unit ball
    and applied to one of radius 6.4e6 asks for elements a million
    times too small, and gmsh reports that as "identical points in
    triangulation", which names neither the scale nor the rule.  Both
    bounds are relative to the outer radius: a target element more than
    a hundred thousand times smaller than the domain, or larger than
    the domain itself, is not a resolution choice.  A refinement is
    judged by its `size` floor.
    """
    outer = float(outer_radius)
    targets = ([s.size for s in sizes.values()]
               + [float(fine.size) for fine in refinements])
    smallest = min(targets)
    largest = max(targets)
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
