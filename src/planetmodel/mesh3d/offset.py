"""offset.py -- the offset two-body geometries, for semi-analytic tests.

A small ball of radius `a`, its centre displaced by `d`, inside a body of
radius `b`.  Two regions, two boundaries, and an answer that can be
written down: this is the geometry a solver is checked against before
anyone trusts it on a planet.

It is a separate generator rather than a MeshSpec because the body is not
spherically layered -- displace the inner sphere and the whole
referential framework stops applying, since there is no radial mapping of
which this is the image.  What it does share is everything below that:
the session, the sizing fields, orientation repair, curving, validation,
the writer and the manifest, so an offset mesh is checked exactly as hard
as a planet is.

Tagging is the one place it must differ.  Concentric geometry is
identified from CAD bounding boxes before meshing, which an offset sphere
defeats -- its box says how big it is, not where its surface averages
out.  So the boundaries are identified twice here: by bounding-box size
before meshing, because the sizing fields need a handle, and by
node-average radius afterwards, which is the ordering the attributes
take.  The two must agree, and a disagreement is raised rather than
resolved, on the same principle as the layered tagging: an ordering
nobody can check is one nobody should trust.
"""
from __future__ import annotations

import time
from collections import namedtuple
from pathlib import Path

import gmsh
import numpy as np

from ..io import manifest
from ..registry import name_of
from ._orient import orient_mesh, raise_order
from ._session import session
from ._sizing import apply_mesh_options, apply_size_fields, check_sizing_scale
from ._tagging import Tagging, mean_radius_of_entity
from ._units import MeshUnits
from ._validate import validate_mesh
from ._writer import confirm_reread, element_counts, write_msh
from .spec import MeshResult

__all__ = ["build_offset_mesh"]

#: What a sizing rule needs of an interface: where it is and which it is.
_Boundary = namedtuple("_Boundary", "index radius name")


def build_offset_mesh(path, *, inner_radius: float, outer_radius: float,
                      offset: float = 0.0, sizing=None, dimension: int = 3,
                      order: int = 2, rref: float | None = None,
                      layer_names=("inclusion", "matrix"),
                      interface_names=("inclusion_boundary", "surface"),
                      algorithm_2d: int = 6, algorithm_3d: int = 1,
                      validate: bool = True, verbose: bool = False
                      ) -> MeshResult:
    """Mesh a ball of radius `a` offset by `d` inside one of radius `b`.

    `offset` displaces the inner body along z in 3D and along x in 2D --
    in both cases within the plane the geometry is drawn in.  Zero is
    the concentric case, which is legitimate and worth meshing: it is
    the same generator with an answer that is separable.

    `rref` is one mesh length unit in metres, for a problem posed in SI;
    without it the radii given are the coordinates written, which is
    what a synthetic test usually wants.
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
            "a layered body, applied to its two boundaries")

    units = (MeshUnits(divisor=float(rref), rref_m=float(rref)) if rref
             else MeshUnits.identity())
    a_nd, b_nd, d_nd = (float(units.to_mesh(x)) for x in (a, b, d))
    timings: dict[str, float] = {}
    clock = time.perf_counter

    # The sizing rule sees the boundaries in the body's own units, as a
    # layered body's rule does, and its answer is converted with them.
    sizes = {index: s.scaled(1.0 / units.divisor) for index, s in sizing(
        [_Boundary(0, a, interface_names[0]),
         _Boundary(1, b, interface_names[1])], b).items()}
    check_sizing_scale(b_nd, sizes)

    with session(name=path.stem or "offset", verbose=verbose):
        t0 = clock()
        cells, faces = _offset_geometry(a_nd, b_nd, d_nd, dimension)
        tagging = Tagging(dimension=dimension, cells=cells, faces=faces,
                          radii=(a_nd, b_nd))
        timings["geometry"] = clock() - t0

        t0 = clock()
        apply_size_fields(tagging, sizes)
        apply_mesh_options(
            order=1, algorithm_2d=algorithm_2d, algorithm_3d=algorithm_3d,
            size_min=min(s.size for s in sizes.values()),
            size_max=max(s.far_size for s in sizes.values()))
        gmsh.model.mesh.generate(dimension)
        timings["mesh"] = clock() - t0

        # "Outward" on the inclusion means away from *its* centre: a
        # face on the far side of an inclusion that does not contain the
        # origin points towards the origin while pointing out of the
        # inclusion, and orienting it about the origin would reverse it.
        centre = (0.0, 0.0, d_nd) if dimension == 3 else (d_nd, 0.0, 0.0)
        centres = {tagging.faces[0]: centre}

        t0 = clock()
        orientation = orient_mesh(dimension, centres=centres)
        curving = raise_order(dimension, order)
        timings["orient"] = clock() - t0

        measured = _confirm_by_node_average(tagging)
        groups = _apply_groups(tagging, layer_names, interface_names)

        t0 = clock()
        # The outer boundary is a sphere about the origin and its mean
        # radius is exactly b; the inner one is offset, so its node
        # average is a property of the offset rather than a promise the
        # geometry made, and there is nothing to check it against.
        report = validate_mesh(
            tagging, expected_radii=(np.nan, b_nd),
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
            dimension=dimension, order=order, a_nd=a_nd, b_nd=b_nd, d_nd=d_nd,
            units=units, sizes=sizes, measured=measured, counts=counts,
            report=report, curving=curving, orientation=orientation,
            layer_names=layer_names, interface_names=interface_names,
            algorithm_2d=algorithm_2d, algorithm_3d=algorithm_3d,
            msh_path=msh_path, gmsh_version=gmsh_version,
            policy=name_of("sizing", sizing) or type(sizing).__name__)
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
    return MeshResult(msh_path=msh_path, manifest_path=manifest_path, body=None,
                      counts=counts, validation=report, timings=timings,
                      units=units)


def _offset_geometry(a: float, b: float, d: float, dimension: int):
    """Two nested bodies, the inner one displaced, fragmented into one part.

    Identified by bounding-box half-width, which separates them cleanly
    whatever the offset: the inner body's box is `a` across its own
    centre and the outer one's is `b` about the origin, and the two can
    only be confused if the bodies are the same size, which the caller
    is already forbidden from asking for.
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
    """Re-identify the boundaries from the mesh, and insist they agree.

    The node average is the ordering the attributes of these bodies
    take, and it is available only once a mesh exists.  Checking it
    against the bounding-box ordering costs one pass over the boundary
    nodes and turns a silent mis-tagging -- an inclusion labelled as the
    exterior, and a test that disagrees with its analytic answer for no
    visible reason -- into a refusal.
    """
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


def _apply_groups(tagging: Tagging, layer_names, interface_names) -> dict:
    """Number both regions and both boundaries 1..2, innermost first."""
    d = tagging.dimension
    gmsh.model.removePhysicalGroups()
    out = {"layers": {}, "interfaces": {}}
    for i, (cell, name) in enumerate(zip(tagging.cells, layer_names)):
        gmsh.model.addPhysicalGroup(d, [cell], i + 1)
        gmsh.model.setPhysicalName(d, i + 1, name)
        out["layers"][i + 1] = cell
    for i, (face, name) in enumerate(zip(tagging.faces, interface_names)):
        gmsh.model.addPhysicalGroup(d - 1, [face], i + 1)
        gmsh.model.setPhysicalName(d - 1, i + 1, name)
        out["interfaces"][i + 1] = face
    return out


def _build_manifest(*, dimension, order, a_nd, b_nd, d_nd, units, sizes,
                    measured, counts, report, curving, orientation,
                    layer_names, interface_names, algorithm_2d, algorithm_3d,
                    msh_path, gmsh_version, policy):
    """The same schema a layered mesh ships, saying what is true here.

    `r_inner_nd` and `r_outer_nd` are the spheres a region lies between,
    which for a displaced inclusion describes its size and not its
    position -- `model.geometry.offset_nd` is what says where it is, and
    a consumer reading the radii alone would place it wrongly.
    """
    kind = "two_sphere" if dimension == 3 else "two_disc"
    layers = [
        manifest.LayerEntry(attribute=1, name=layer_names[0], r_inner_nd=0.0,
                            r_outer_nd=a_nd, state="solid", fields=[],
                            is_vacuum=False, law=None),
        manifest.LayerEntry(attribute=2, name=layer_names[1], r_inner_nd=a_nd,
                            r_outer_nd=b_nd, state="solid", fields=[],
                            is_vacuum=False, law=None),
    ]
    interfaces = [
        manifest.InterfaceEntry(attribute=i + 1, name=interface_names[i],
                                mean_radius_nd=float(measured[i]),
                                between_layers=[i, i + 1 if i == 0 else -1],
                                role="material")
        for i in (0, 1)
    ]
    return manifest.MeshManifest.from_build(
        model={"name": f"offset_{kind}", "source": None, "sha256": None,
               "rref_m": units.rref_m,
               "geometry": {"kind": kind, "inner_radius_nd": a_nd,
                            "outer_radius_nd": b_nd, "offset_nd": d_nd},
               "units": manifest.units_block(None, units.divisor,
                                             units.rref_m)},
        mesh=manifest.mesh_block(
            dimension=dimension, order=order, gmsh_version=gmsh_version,
            algorithm_2d=algorithm_2d, algorithm_3d=algorithm_3d,
            counts=counts, curving=curving),
        delivery="physical", layers=layers, interfaces=interfaces,
        sizing=manifest.sizing_block(policy=policy, sizes=sizes),
        validation=manifest.validation_block(report, orientation),
        provenance=manifest.provenance_block(mesh_file=msh_path.name))
