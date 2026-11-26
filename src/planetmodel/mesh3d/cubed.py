"""A CubedSphereSpec becomes a mesh on disk.

A spherical geometry is meshed with hexahedra (in two dimensions,
quadrilaterals) as a cubed sphere, by construction and with no mesher:
the cells are made in a parameter space of nested cubes (`_cubed`),
given the nodes of the chosen order there, where the mesh is
straight-sided and so exact, and the nodes are then moved by the
cube-to-ball map.  PyMFEM holds the mesh, so the nodes are MFEM's own
and those on a sphere are on it exactly, at any order.  The mesh is the
reference one, concentric spheres with the geometry's names as
attributes, as every builder here writes it: the geometry's mapping is
applied later, at the MFEM export.

The computational domain is the geometry followed by the spec's shells,
and its numbers are the mesh's.  The cells of layer i carry element
attribute i + 1; the innermost layer of a ball is the central cube and
the blend around it.  The faces on interface k are boundary elements of
attribute k + 1, inside the mesh as well as on its boundary, pointing
away from the centre, a shell's inner boundary pointing out of the body
towards it.

Before anything is written the mesh is checked: MFEM's own verdict on
the orientation of its elements and boundary elements; the Jacobian of
every cell at its nodes and quadrature points, none folded
(`export.jacobian_report`), and the shape of every cell through its
corners (`_cubed.cell_shapes`); each interface's faces facing
the right way and its nodes at its radius; each layer's volume against
that of the shell it fills (`cell_volumes`), which catches a cell given
to the wrong layer; and the cells' aspect ratios, a warning where one
is long and thin.  The manifest is that of any spherical mesh: nothing
in it says how the mesh was made, and the mesh file itself holds what
its elements are.  The file written is read back as a consumer reads it
and held to the counts and attributes it was written with.

With shells and a non-identity mapping the mapping must be defined and
orientation-preserving out to the outer boundary of the computational
domain, and a spherical outer boundary asks for the identity there,
both checked before anything is built (`spec`).

The order of the geometry is the spec's and need not be the order a
consumer solves at: 2 puts an interface within about 2e-5 of its radius
between the nodes at 16 cells an edge, and each order more gains two
orders of magnitude.  MFEM chooses its default quadrature rules from
the geometry's order too, so a consumer of a mesh of order 4 or more
sets its own.

PyMFEM, the `mfem` extra, is imported where the mesh is built.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from . import manifest
from ._cubed import (CubedTopology, aspect_ratios, cell_shapes, cubed_map,
                     cubed_topology)
from ._extra import require_mfem
from ._layout import Concentric
from .export import (MESH_READ_OPTIONS, PRECISION, jacobian_report, load_mesh,
                     mesh_order, node_array, write_vector)
from .spec import (ASPECT_WARNING, QUALITY_FLOOR, CubedSphereSpec, MeshResult,
                   ValidationReport, require_mapping_on_shells,
                   require_spherical_outer_boundary)

__all__ = ["build_cubed_sphere", "cell_volumes", "volume_tolerance", "RADIUS_RTOL"]

#: The distance an interface's nodes may lie from its radius, over the
#: outer radius of the mesh: they are put on it to rounding.
RADIUS_RTOL = 1e-12


def volume_tolerance(edge: int) -> float:
    """The relative error of a layer's volume the check allows: 0.6 of the
    square of the angle a cell subtends, pi / (2 edge).  A straight-sided
    mesh, whose faces are chords of the spheres, falls short of every
    shell by about 0.4 of that square, and a curved one by far less; a
    cell in the wrong layer changes a volume by far more."""
    return 0.6 * (0.5 * np.pi / edge) ** 2


def _attributes(spec: CubedSphereSpec, topology: CubedTopology
                ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Each cell's element attribute, and each interface's boundary
    attribute with its faces.

    The level of the first sphere is the top of the blend, or of a shell
    the cube's surface; the cells below it are the innermost layer's,
    and those between two spheres the layer's whose radii hold them.
    """
    domain = spec.domain
    spheres = spec.spheres
    first = 0 if spec.blend is None else spec.blend.size - 1
    boundaries = domain.skeleton.boundaries
    middles = 0.5 * (spheres[1:] + spheres[:-1])
    above = np.searchsorted(boundaries, middles)          # layer index + 1
    by_level = np.concatenate([np.ones(first + 1, dtype=np.int64), above])
    faces = []
    for k, face in enumerate(domain.interfaces):
        (at,) = np.flatnonzero(spheres == face.radius)
        level = first + int(at)
        inner = spec.hollow and level == 0
        faces.append((k + 1, topology.faces(level, inward=inner)))
    return by_level[topology.cell_levels], faces


def _assemble(topology: CubedTopology, attributes: np.ndarray,
              faces: list[tuple[int, np.ndarray]]) -> Any:
    """The straight-sided mesh of the parameter space as an `mfem.Mesh`."""
    mfem = require_mfem()
    d = topology.dimension
    count = sum(f.shape[0] for _, f in faces)
    mesh = mfem.Mesh(d, topology.points.shape[0], topology.cells.shape[0], count, d)
    for point in topology.points.tolist():
        mesh.AddVertex(*point)
    add = mesh.AddHex if d == 3 else mesh.AddQuad
    for cell, attribute in zip(topology.cells.tolist(), attributes.tolist()):
        add(cell, attribute)
    add = mesh.AddBdrQuad if d == 3 else mesh.AddBdrSegment
    for attribute, rows in faces:
        for face in rows.tolist():
            add(face, attribute)
    mesh.FinalizeTopology()
    mesh.Finalize(False, False)
    return mesh


def cell_volumes(mesh: Any) -> np.ndarray:
    """The volume of every cell of an MFEM mesh of one element geometry,
    in two dimensions its area, (cells,): its Jacobian's determinant
    integrated by a rule that is exact for it, of order d p - 1 for a
    mesh of order p in d dimensions."""
    mfem = require_mfem()
    rule = mfem.IntRules.Get(mesh.GetElementBaseGeometry(0),
                             mesh.Dimension() * mesh_order(mesh) - 1)
    weights = np.array([rule.IntPoint(i).weight for i in range(rule.GetNPoints())])
    mesh.DeleteGeometricFactors()
    factors = mesh.GetGeometricFactors(rule, mfem.GeometricFactors.DETERMINANTS)
    dets = np.array(factors.detJ.GetDataArray(), dtype=float)
    return dets.reshape(mesh.GetNE(), rule.GetNPoints()) @ weights


def _facing(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Positive for each face that points away from the origin: its normal
    on its centroid, or in the plane the turn of a side about the origin."""
    corners = points[faces]
    centre = corners.mean(axis=1)
    if faces.shape[1] == 2:
        side = corners[:, 1] - corners[:, 0]
        return centre[:, 0] * side[:, 1] - centre[:, 1] * side[:, 0]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    return np.einsum("ij,ij->i", normal, centre)


def _shell_volume(inner: float, outer: float, dimension: int) -> float:
    """The volume between two spheres, or the area between two circles."""
    if dimension == 3:
        return 4.0 / 3.0 * np.pi * (outer ** 3 - inner ** 3)
    return np.pi * (outer ** 2 - inner ** 2)


def _validate(spec: CubedSphereSpec, mesh: Any, topology: CubedTopology,
              corners: np.ndarray, attributes: np.ndarray,
              faces: list[tuple[int, np.ndarray]]) -> ValidationReport:
    """Every check of the built mesh (module docstring); `corners` are the
    mapped positions of the topology's points."""
    d = spec.dimension
    domain = spec.domain
    report = ValidationReport(dimension=d)
    fail = report.failures.append

    report.negative_cells = int(mesh.CheckElementOrientation(False))
    wrong_faces = int(mesh.CheckBdrElementOrientation(False))
    if report.negative_cells or wrong_faces:
        fail(f"MFEM finds {report.negative_cells} wrongly oriented elements and "
             f"{wrong_faces} wrongly oriented boundary elements")

    report.min_sicn = float(cell_shapes(corners, topology.cells).min())
    report.negative_jacobians = int(jacobian_report(mesh)["negative_jacobians"])
    if report.negative_jacobians or report.min_sicn <= 0.0:
        fail(f"{report.negative_jacobians} cells fold: their Jacobian is not "
             f"positive throughout them (worst shape {report.min_sicn:.3g})")
    elif report.min_sicn < QUALITY_FLOOR:
        report.warnings.append(
            f"the worst cell has shape {report.min_sicn:.3g}, below "
            f"{QUALITY_FLOOR}")

    tol = RADIUS_RTOL * spec.outer_radius
    for (attribute, rows), face in zip(faces, domain.interfaces):
        inner = spec.hollow and attribute == 1
        facing = _facing(corners, rows)
        report.inward_faces += int(np.count_nonzero(
            facing >= 0.0 if inner else facing <= 0.0))
        radius = float(np.linalg.norm(corners[np.unique(rows)], axis=1).mean())
        gap = abs(radius - face.radius)
        report.max_interface_radius_error = max(report.max_interface_radius_error,
                                                gap)
        if gap > tol:
            fail(f"interface {attribute} has mean radius {radius:.9g}, expected "
                 f"{face.radius:.9g}")
    if report.inward_faces:
        fail(f"{report.inward_faces} faces of the interfaces point the wrong way")

    boundaries = domain.skeleton.boundaries
    volumes = np.bincount(attributes, weights=cell_volumes(mesh),
                          minlength=domain.nlayers + 1)[1:]
    exact = np.array([_shell_volume(lo, hi, d)
                      for lo, hi in zip(boundaries[:-1], boundaries[1:])])
    errors = np.abs(volumes / exact - 1.0)
    report.max_layer_volume_error = float(errors.max())
    bound = volume_tolerance(spec.edge)
    for i in np.flatnonzero(errors > bound):
        fail(f"layer {i + 1} has volume {volumes[i]:.6g}, and the shell it fills "
             f"{exact[i]:.6g}: off by {errors[i]:.3g}, more than {bound:.3g}")

    report.max_aspect_ratio = float(aspect_ratios(corners, topology.cells).max())
    if report.max_aspect_ratio > ASPECT_WARNING:
        report.warnings.append(
            f"the worst cell is {report.max_aspect_ratio:.3g} times longer one way "
            f"than another, above {ASPECT_WARNING:g}")

    found = {"layers": np.unique(attributes).size, "interfaces": len(faces)}
    report.group_counts = found
    wanted = {"layers": domain.nlayers, "interfaces": len(domain.interfaces)}
    if found != wanted:
        fail(f"the mesh has {found}, the domain {wanted}")
    return report


def _confirm_reread(mesh_path: Path, manifest_path: Path, *, cells: np.ndarray,
                    faces: np.ndarray) -> None:
    """Read the written mesh back as a consumer reads it and hold it to
    the counts of each attribute it was written with; both files are
    removed where it is not."""
    try:
        mesh = load_mesh(mesh_path)
        got_cells = np.bincount([mesh.GetAttribute(e) for e in range(mesh.GetNE())])
        got_faces = np.bincount([mesh.GetBdrAttribute(b)
                                 for b in range(mesh.GetNBE())])
        if (got_cells.tolist() != np.bincount(cells).tolist()
                or got_faces.tolist() != np.bincount(faces).tolist()):
            raise ValueError(
                f"{mesh_path.name} read back has other elements than were written: "
                f"cells by attribute {got_cells.tolist()[1:]}, boundary elements "
                f"{got_faces.tolist()[1:]}")
    except Exception:
        mesh_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


def build_cubed_sphere(spec: CubedSphereSpec, path: str | Path) -> MeshResult:
    """Build, check and write the cubed sphere a CubedSphereSpec describes.

    `path` is the basename: `<path>.mesh`, the reference mesh in MFEM's
    format with its nodes of the spec's order, and `<path>.json`, its
    manifest, are written.  A delivery of it (`export_mfem_mesh`,
    `export_mfem`) takes another basename.
    """
    mfem = require_mfem()
    path = Path(path)
    timings: dict[str, float] = {}
    clock = time.perf_counter

    t0 = clock()
    geometry = spec.geometry
    domain = spec.domain
    if spec.shells and not geometry.is_identity:
        require_mapping_on_shells(spec)
    if spec.outer_boundary == "spherical" and not geometry.is_identity:
        require_spherical_outer_boundary(spec)
    spheres, blend = spec.spheres, spec.blend
    timings["resolve"] = clock() - t0

    t0 = clock()
    d = spec.dimension
    topology = cubed_topology(spec.edge, spec.levels, dimension=d, hollow=spec.hollow)
    attributes, faces = _attributes(spec, topology)
    mesh = _assemble(topology, attributes, faces)
    timings["cells"] = clock() - t0

    t0 = clock()
    half_width = 0.0 if blend is None else spec.half_width
    mesh.SetCurvature(spec.order, False, d, mfem.Ordering.byVDIM)
    nodes = mesh.GetNodes()
    write_vector(nodes, cubed_map(np.array(node_array(nodes), dtype=float),
                                  spheres=spheres, blend=blend,
                                  half_width=half_width))
    mesh.NodesUpdated()
    corners = cubed_map(topology.points, spheres=spheres, blend=blend,
                        half_width=half_width)
    timings["curve"] = clock() - t0

    t0 = clock()
    report = _validate(spec, mesh, topology, corners, attributes, faces)
    if spec.validate:
        report.raise_if_failed()
    timings["validate"] = clock() - t0

    t0 = clock()
    mesh_path = manifest.beside(path, ".mesh")
    card = manifest.MeshManifest.from_build(
        mesh=manifest.mesh_block(mesh_path, format="mfem", nodes="reference",
                                 read_options=MESH_READ_OPTIONS),
        meta=spec.meta, **Concentric().records(spec, collared=None))
    face_attributes = np.concatenate([np.full(rows.shape[0], a, dtype=np.int64)
                                      for a, rows in faces])
    # The manifest is checked and written before the mesh: a mesh on
    # disk with no manifest, or with one that disagrees, looks finished.
    manifest.validate_against(
        card, layer_count=domain.nlayers, interface_count=len(domain.interfaces),
        groups={"layers": np.unique(attributes).tolist(),
                "interfaces": np.unique(face_attributes).tolist()})
    manifest_path = manifest.write(path, card)
    try:
        mesh.Print(str(mesh_path), PRECISION)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    _confirm_reread(mesh_path, manifest_path, cells=attributes, faces=face_attributes)
    timings["write"] = clock() - t0

    counts = {"elements": mesh.GetNE(), "boundary_elements": mesh.GetNBE(),
              "nodes": nodes.FESpace().GetNDofs(), "order": spec.order,
              "levels": spec.levels, "layers": domain.nlayers,
              "interfaces": len(domain.interfaces)}
    return MeshResult(mesh_path=mesh_path, manifest_path=manifest_path,
                      geometry=geometry, counts=counts, validation=report,
                      timings=timings, spec=spec, mapping=domain.mapping)
