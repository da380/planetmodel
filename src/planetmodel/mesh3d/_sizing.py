"""_sizing.py -- telling gmsh how finely to resolve each interface.

One Distance field per interface, each wrapped in a Threshold that
grows the element size from `size` at the boundary to `far_size` over
`decay_width`, and a single Min over all of them so every point takes
the finest requirement that applies to it.

This is per-interface by construction.  A single Distance field over
every boundary at once -- the shortcut -- can only express one size for
all of them, which is precisely what a layered body does not want: the
inner core boundary and the Moho deserve different resolutions and the
whole point of a sizing rule is to say so.
"""
from __future__ import annotations

import gmsh

__all__ = ["apply_size_fields", "apply_mesh_options",
           "check_sizing_resolves_spans", "check_sizing_scale"]


def apply_size_fields(tagging, sizes: dict) -> int:
    """Build the background size field from per-interface sizings.

    `sizes` maps interface index to InterfaceSizing, in the geometry's
    units.  Returns the tag of the field set as the background.
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

    The three MeshSizeFrom* options are switched off deliberately: with
    a background field in place, sizes inherited from CAD points, from
    curvature, or extended from boundaries all compete with the field
    and quietly win in places, so the mesh stops matching the sizing
    rule that was asked for.

    (The names are the modern ones.  gmsh still accepts the older
    CharacteristicLength* spellings, which the C++ meshing code used,
    but they are deprecated and silently ignored in some builds.)
    """
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", float(size_min))
    gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_max))
    gmsh.option.setNumber("Mesh.ElementOrder", int(order))
    gmsh.option.setNumber("Mesh.Algorithm", int(algorithm_2d))
    gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm_3d))


def check_sizing_resolves_spans(radii, sizes, *, max_ratio: float = 10.0,
                                strict: bool = True) -> list:
    """Refuse element sizes too coarse for the layers they must fill.

    A shell far thinner than the elements asked for cannot be
    tetrahedralised, and gmsh reports that as "PLC Error: a segment and
    a facet intersect" -- true, unhelpful, and a long way from the
    sizing rule that caused it.  `max_ratio` is the measured refusal
    point (`docs/notes/mesh_thresholds.md`); element quality above it
    is left to the validation report's own warning.

    `radii` are the interface radii in mesh units, innermost first;
    `sizes` maps interface index to InterfaceSizing.  Each span is judged
    by the finer of its two bounding interfaces.
    """
    import numpy as np

    radii = np.asarray(radii, dtype=float)
    edges = np.concatenate(([0.0], radii))
    problems = []
    for i in range(len(radii)):
        thickness = float(edges[i + 1] - edges[i])
        bounding = [sizes[j].size for j in (i - 1, i) if j in sizes and j >= 0]
        h = min(bounding) if bounding else sizes[i].size
        if h > max_ratio * thickness:
            problems.append(
                f"span [{edges[i]:.6g}, {edges[i + 1]:.6g}] is "
                f"{thickness:.3g} thick but the elements bounding it are "
                f"{h:.3g} ({h / thickness:.0f}x too coarse; gmsh fails "
                f"above about {max_ratio:.0f}x)")
    if problems and strict:
        raise ValueError(
            "the sizing is too coarse for this body's thinnest layers, and "
            "gmsh would fail with an unrelated-looking PLC error:\n  - "
            + "\n  - ".join(problems)
            + "\nRefine the sizing, or coarsen the body first with "
              "keep_interfaces/drop_interfaces so the thin layers are merged.")
    return problems


def check_sizing_scale(outer_radius: float, sizes, *, floor: float = 1e-5,
                       ceiling: float = 2.0) -> None:
    """Catch a sizing given in the wrong units before gmsh does.

    Sizing rules return values in the *body's* units, and the builder
    converts them along with the geometry.  Passing values already in
    mesh units is therefore an easy mistake, and an expensive one: the
    sizes are divided a second time, and gmsh reports the result as
    "Identical points in triangulation: increase element size", which
    names neither the units nor the rule that caused it.

    A target element more than a hundred thousand times smaller than the
    body, or larger than the body itself, is not a resolution choice.
    """
    outer = float(outer_radius)
    smallest = min(s.size for s in sizes.values())
    largest = max(s.size for s in sizes.values())
    if smallest < floor * outer:
        raise ValueError(
            f"the smallest element size is {smallest:.3g} against an outer "
            f"radius of {outer:.3g} -- a ratio of {smallest / outer:.1e}, "
            "which is not a resolution choice. Sizing rules take lengths in "
            "the body's own units (metres for an SI body); values already in "
            "mesh units get divided a second time.")
    if largest > ceiling * outer:
        raise ValueError(
            f"the largest element size is {largest:.3g}, more than the outer "
            f"radius {outer:.3g}: nothing would be resolved.")
