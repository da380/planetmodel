"""The offset two-body geometries, for semi-analytic tests.

A small ball of radius `a`, its centre displaced by `d`, inside a body
of radius `b`: two regions, two boundaries, and an answer that can be
written down.  It is a separate generator rather than a MeshSpec
because the domain is not spherically layered, so there is no geometry
and no mapping of which it is the image.  It shares everything below
that: the session, the sizing fields, orientation repair, curving,
validation, the writer and the manifest.

Tagging differs.  Concentric geometry is identified from CAD bounding
boxes before meshing, which an offset sphere defeats, so the boundaries
are identified twice: by bounding-box size before meshing, because the
sizing fields need a handle, and by node-average radius afterwards,
which is the ordering the attributes take.  The two must agree, and a
disagreement is raised rather than resolved.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gmsh
import numpy as np

from ..geometry import InterfaceInfo
from . import manifest
from ._orient import OrientationReport, orient_mesh, raise_order
from ._session import session
from ._sizing import apply_mesh_options, apply_size_fields, check_sizing_scale
from ._tagging import Tagging, mean_radius_of_entity
from ._validate import validate_mesh
from ._writer import confirm_reread, element_counts, write_msh
from .layered import policy_name
from .spec import InterfaceSizing, MeshResult, SizingRule, ValidationReport

__all__ = ["build_offset_mesh"]


def build_offset_mesh(path: str | Path, *, inner_radius: float, outer_radius: float,
                      offset: float = 0.0, sizing: SizingRule | None = None,
                      dimension: int = 3, order: int = 2,
                      layer_names: Sequence[str] = ("inclusion", "matrix"),
                      interface_names: Sequence[str] = ("inclusion_boundary",
                                                        "surface"),
                      algorithm_2d: int = 6, algorithm_3d: int = 1,
                      validate: bool = True, verbose: bool = False
                      ) -> MeshResult:
    """Mesh a ball of radius `a` offset by `d` inside one of radius `b`.

    `offset` displaces the inner body along z in 3D and along x in 2D,
    within the plane the geometry is drawn in.  Zero is the concentric
    case.  `sizing` is a sizing rule, applied to the two boundaries.
    Every length is meshed in the numbers given.  The result has no
    geometry, spec or mapping.
    """
    path = Path(path)
    a, b, d = float(inner_radius), float(outer_radius), float(offset)
    if not 0.0 < a < b:
        raise ValueError(
            f"need 0 < inner_radius < outer_radius, got {a} and {b}")
    if abs(d) + a > b:
        raise ValueError(
            f"the inner body reaches radius {abs(d) + a:.6g}, outside the "
            f"outer radius {b:.6g}: it must be strictly enclosed")
    if dimension not in (2, 3):
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")
    if not 1 <= order <= 3:
        raise ValueError(f"element order must be 1..3, got {order}")
    if sizing is None:
        raise ValueError(
            "no sizing given: an offset mesh takes the same sizing rules as "
            "a layered geometry, applied to its two boundaries")
    timings: dict[str, float] = {}
    clock = time.perf_counter

    # The sizing rule sees the two boundaries as a layered geometry's
    # rule sees its interfaces.
    faces = [InterfaceInfo(0, a, (0, 1), name=interface_names[0]),
             InterfaceInfo(1, b, (1, -1), name=interface_names[1])]
    sizes = dict(sizing(faces, b))
    check_sizing_scale(b, sizes)

    with session(name=path.stem or "offset", verbose=verbose):
        t0 = clock()
        cells, cad_faces = _offset_geometry(a, b, d, dimension)
        tagging = Tagging(dimension=dimension, cells=cells, faces=cad_faces,
                          radii=(a, b))
        timings["geometry"] = clock() - t0

        t0 = clock()
        apply_size_fields(tagging, sizes)
        apply_mesh_options(
            order=1, algorithm_2d=algorithm_2d, algorithm_3d=algorithm_3d,
            size_min=min(s.size for s in sizes.values()),
            size_max=max(s.far_size for s in sizes.values()))
        gmsh.model.mesh.generate(dimension)
        timings["mesh"] = clock() - t0

        # "Outward" on the inclusion means away from its own centre.
        centre = (0.0, 0.0, d) if dimension == 3 else (d, 0.0, 0.0)
        centres = {tagging.faces[0]: centre}

        t0 = clock()
        orientation = orient_mesh(dimension, centres=centres)
        curving = raise_order(dimension, order)
        timings["orient"] = clock() - t0

        measured = _confirm_by_node_average(tagging)
        groups = _apply_groups(tagging, layer_names, interface_names)

        t0 = clock()
        # The outer boundary is a sphere about the origin with mean
        # radius b; the inner one is offset and has nothing to be
        # checked against.
        report = validate_mesh(
            tagging, expected_radii=(np.nan, b),
            layer_names=layer_names, interface_names=interface_names,
            centres=centres)
        if validate:
            report.raise_if_failed()
        timings["validate"] = clock() - t0

        t0 = clock()
        counts = element_counts(dimension=dimension)
        gmsh_version = gmsh.option.getString("General.Version")
        msh_path = manifest.beside(path, ".msh")
        card = _build_manifest(
            dimension=dimension, order=order, a=a, b=b, d=d, sizes=sizes,
            measured=measured, counts=counts,
            report=report, curving=curving, orientation=orientation,
            layer_names=layer_names, interface_names=interface_names,
            algorithm_2d=algorithm_2d, algorithm_3d=algorithm_3d,
            msh_path=msh_path, gmsh_version=gmsh_version,
            policy=policy_name(sizing))
        manifest.validate_against(card, layer_count=2, interface_count=2,
                                  groups={k: list(v) for k, v in groups.items()})
        manifest_path = manifest.write(path, card)
        try:
            write_msh(path)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            raise
        timings["write"] = clock() - t0

    confirm_reread(msh_path, manifest_path, dimension, layer_names,
                   interface_names)

    counts["layers"] = 2
    counts["interfaces"] = 2
    return MeshResult(msh_path=msh_path, manifest_path=manifest_path,
                      geometry=None, counts=counts, validation=report,
                      timings=timings)


def _offset_geometry(a: float, b: float, d: float, dimension: int
                     ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two nested bodies, the inner one displaced, fragmented into one part.

    Identified by bounding-box half-width, which separates them
    whatever the offset: the inner body's box is `a` across its own
    centre and the outer one's is `b` about the origin.
    """
    if dimension == 3:
        outer = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, b)
        inner = gmsh.model.occ.addSphere(0.0, 0.0, d, a)
    else:
        outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, b, b)
        inner = gmsh.model.occ.addDisk(d, 0.0, 0.0, a, a)

    gmsh.model.occ.fragment([(dimension, outer)], [(dimension, inner)])
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    cells = [tag for _, tag in gmsh.model.getEntities(dimension)]
    faces = [tag for _, tag in gmsh.model.getEntities(dimension - 1)]
    if len(cells) != 2 or len(faces) != 2:
        raise RuntimeError(
            f"the offset geometry fragmented into {len(cells)} region(s) and "
            f"{len(faces)} boundarie(s), not 2 and 2; the inner body may be "
            "touching the outer boundary")

    cells.sort(key=lambda t: _half_width(dimension, t))
    faces.sort(key=lambda t: _half_width(dimension - 1, t))
    return tuple(cells), tuple(faces)


def _half_width(dimension: int, tag: int) -> float:
    """Half the largest side of an entity's bounding box."""
    box = np.asarray(gmsh.model.getBoundingBox(dimension, tag), dtype=float)
    return 0.5 * float(np.max(box[3:] - box[:3]))


def _confirm_by_node_average(tagging: Tagging) -> tuple[float, ...]:
    """Re-identify the boundaries from the mesh, and insist they agree."""
    d = tagging.dimension
    measured = tuple(mean_radius_of_entity(d - 1, face)
                     for face in tagging.faces)
    if list(measured) != sorted(measured):
        raise RuntimeError(
            f"the boundaries order differently by bounding box and by node "
            f"average (mean radii {[round(r, 6) for r in measured]} in "
            "bounding-box order); the geometry is not what this generator "
            "assumes")
    return measured


def _apply_groups(tagging: Tagging, layer_names: Sequence[str],
                  interface_names: Sequence[str]) -> dict[str, dict[int, int]]:
    """Number both regions and both boundaries 1..2, innermost first."""
    d = tagging.dimension
    gmsh.model.removePhysicalGroups()
    out: dict[str, dict[int, int]] = {"layers": {}, "interfaces": {}}
    for i, (cell, name) in enumerate(zip(tagging.cells, layer_names)):
        gmsh.model.addPhysicalGroup(d, [cell], i + 1)
        gmsh.model.setPhysicalName(d, i + 1, name)
        out["layers"][i + 1] = cell
    for i, (face, name) in enumerate(zip(tagging.faces, interface_names)):
        gmsh.model.addPhysicalGroup(d - 1, [face], i + 1)
        gmsh.model.setPhysicalName(d - 1, i + 1, name)
        out["interfaces"][i + 1] = face
    return out


def _build_manifest(*, dimension: int, order: int, a: float, b: float, d: float,
                    sizes: Mapping[int, InterfaceSizing], measured: Sequence[float],
                    counts: Mapping[str, int], report: ValidationReport,
                    curving: Mapping[str, Any], orientation: OrientationReport,
                    layer_names: Sequence[str], interface_names: Sequence[str],
                    algorithm_2d: int, algorithm_3d: int, msh_path: Path,
                    gmsh_version: str, policy: str) -> manifest.MeshManifest:
    """The same schema a layered mesh ships, saying what is true here.

    `r_inner` and `r_outer` are the spheres a region lies between, which
    for a displaced inclusion describes its size and not its position;
    `geometry.offset` says where it is.
    """
    kind = "two_sphere" if dimension == 3 else "two_disc"
    layers = [
        manifest.LayerEntry(attribute=1, name=layer_names[0], r_inner=0.0,
                            r_outer=a, in_geometry=True),
        manifest.LayerEntry(attribute=2, name=layer_names[1], r_inner=a,
                            r_outer=b, in_geometry=True),
    ]
    interfaces = [
        manifest.InterfaceEntry(attribute=i + 1, name=interface_names[i],
                                mean_radius=float(measured[i]),
                                between_layers=[i, i + 1 if i == 0 else -1])
        for i in (0, 1)
    ]
    return manifest.MeshManifest.from_build(
        geometry=manifest.geometry_block(
            outer_radius=b, inner_radius=0.0, n_layers=2, kind=kind,
            inclusion_radius=a, offset=d),
        mesh=manifest.mesh_block(
            dimension=dimension, order=order, gmsh_version=gmsh_version,
            algorithm_2d=algorithm_2d, algorithm_3d=algorithm_3d,
            counts=counts, curving=curving),
        delivery="physical", layers=layers, interfaces=interfaces,
        mapping=manifest.mapping_block(None, applied_to_nodes=False),
        sizing=manifest.sizing_block(policy=policy, sizes=sizes),
        validation=manifest.validation_block(report, orientation),
        provenance=manifest.provenance_block(mesh_file=msh_path.name))
