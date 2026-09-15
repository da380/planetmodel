"""The assembly line: a MeshSpec becomes a mesh on disk.

The computational domain is the geometry followed by its shells, and
its numbers are the mesh's numbers: every radius and every element
size reaches gmsh as the geometry gives it.  The mesh is always the
reference one, concentric spheres with the geometry's names as
attributes: gmsh meshes the skeleton and nothing else, and the
geometry's mapping is applied later, at the MFEM export, where a
displacement is a vector in the mesh's own nodal space.  Then CAD,
tagging, sizing, meshing at order 1, orientation, curving, validation,
and only then anything is written.  A mesh that fails its checks and
exists anyway looks finished, so nothing reaches disk before
validation.

With shells and a non-identity mapping the mapping must be defined and
orientation-preserving out to the outer boundary of the computational
domain, checked on a lattice before any meshing and refused by name
otherwise; without shells the geometry's own checks are the whole
guarantee.  The outer boundary of the domain is a boundary like any
other and may carry topography, unless the spec asks for a spherical
one, in which case the mapping must be the identity there: the case of
a buffer added so that a far-field condition can be applied on a sphere.
"""
from __future__ import annotations

import time
from pathlib import Path

import gmsh
import numpy as np

from ..frames import cartesian_points
from ..geometry import Geometry
from ..mapping import validity_lattice
from . import manifest
from ._geometry import build_concentric
from ._orient import orient_mesh, raise_order
from ._session import session
from ._sizing import (apply_mesh_options, apply_size_fields,
                      check_sizing_resolves_spans, check_sizing_scale)
from ._tagging import apply_physical_groups, identify
from ._validate import validate_mesh
from ._writer import confirm_reread, element_counts, write_msh
from .spec import MeshResult, MeshSpec

__all__ = ["build_layered_mesh", "require_mapping_on_shells",
           "require_spherical_outer_boundary", "policy_name"]


def require_mapping_on_shells(spec: MeshSpec) -> None:
    """Refuse a mapping that is not defined on the shells.

    The computational domain under the geometry's mapping must satisfy
    every invariant of a Geometry: knots on boundaries, orientation
    preserved on the validity lattice, continuity across every interior
    boundary, the geometry's own outer boundary included.
    """
    g = spec.geometry
    domain = spec.domain
    try:
        Geometry(domain.skeleton, mapping=g.mapping, rtol=g.rtol, check=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "with shells the mapping must be defined and orientation-preserving "
            f"on the whole computational domain, out to r = {spec.outer_radius:g}: "
            f"{exc}"
        ) from exc


def require_spherical_outer_boundary(spec: MeshSpec) -> None:
    """Refuse a mapping that moves the outer boundary of the domain.

    The mapping must be the identity on the outer boundary of the
    computational domain to `geometry.rtol` times that radius, checked
    on the validity lattice's angles.
    """
    g = spec.geometry
    outer = spec.outer_radius
    _, theta, phi = validity_lattice(spec.domain.skeleton)
    X = cartesian_points(outer, theta, phi)
    gap = float(np.max(np.linalg.norm(np.asarray(g.mapping(X), dtype=float) - X,
                                      axis=-1)))
    tol = g.rtol * outer
    if gap > tol:
        raise ValueError(
            "a spherical outer boundary was asked for, so the mapping must be the "
            f"identity on the outer boundary of the computational domain at r = "
            f"{outer:g}, and it moves points there by up to {gap:.3g} (tolerance "
            f"{tol:.3g}); make the displacement vanish there, or mesh with "
            "outer_boundary='free'")


def policy_name(rule: object) -> str:
    """The name a manifest records for a sizing rule: its class, or the
    function's name."""
    return getattr(rule, "__name__", None) or type(rule).__name__


def build_layered_mesh(spec: MeshSpec, path: str | Path, *, verbose: bool = False
                       ) -> MeshResult:
    """Build, check and write the mesh a MeshSpec describes.

    `path` is the basename: `<path>.msh` and `<path>.json` are written.
    """
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
    mapping = domain.mapping
    timings["resolve"] = clock() - t0

    d = spec.dimension
    boundaries = domain.skeleton.boundaries
    outer = spec.outer_radius
    interface_radii = [f.radius for f in domain.interfaces]
    layer_names = [lay.name for lay in domain.layers]
    interface_names = [f.name for f in domain.interfaces]

    sizes = dict(spec.sizing(domain.interfaces, outer))
    check_sizing_scale(outer, sizes)
    check_sizing_resolves_spans(boundaries, sizes)

    with session(name=path.stem or "planetmodel", verbose=verbose):
        t0 = clock()
        cad = build_concentric(boundaries, dimension=d)
        tagging = identify(cad, interface_radii)
        groups = apply_physical_groups(tagging, layer_names=layer_names,
                                       interface_names=interface_names)
        timings["geometry"] = clock() - t0

        t0 = clock()
        apply_size_fields(tagging, sizes)
        apply_mesh_options(
            order=1, algorithm_2d=spec.algorithm_2d,
            algorithm_3d=spec.algorithm_3d,
            size_min=min(s.size for s in sizes.values()),
            size_max=max(s.far_size for s in sizes.values()))
        gmsh.model.mesh.generate(d)
        timings["mesh"] = clock() - t0

        t0 = clock()
        orient_mesh(d)
        raise_order(d, spec.order)
        timings["orient"] = clock() - t0

        t0 = clock()
        report = validate_mesh(
            tagging, expected_radii=interface_radii,
            layer_names=layer_names, interface_names=interface_names)
        if spec.validate:
            report.raise_if_failed()
        timings["validate"] = clock() - t0

        t0 = clock()
        counts = element_counts(dimension=d)
        msh_path = manifest.beside(path, ".msh")
        card = _build_manifest(spec, msh_path)
        # The manifest is checked and written before the mesh: a mesh on
        # disk with no manifest, or with one that disagrees, looks finished.
        manifest.validate_against(card, layer_count=domain.nlayers,
                                  interface_count=len(domain.interfaces),
                                  groups={k: list(v) for k, v in groups.items()})
        manifest_path = manifest.write(path, card)
        try:
            write_msh(path)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            raise
        timings["write"] = clock() - t0

    confirm_reread(msh_path, manifest_path, d, layer_names, interface_names)

    counts["layers"] = domain.nlayers
    counts["interfaces"] = len(domain.interfaces)
    return MeshResult(msh_path=msh_path, manifest_path=manifest_path,
                      geometry=geometry, counts=counts, validation=report,
                      timings=timings, spec=spec, mapping=mapping)


def _build_manifest(spec: MeshSpec, msh_path: Path) -> manifest.MeshManifest:
    """The manifest of the mesh: its skeleton, its file, the spec's meta."""
    domain = spec.domain
    b = domain.skeleton.boundaries
    n_geometry = spec.geometry.nlayers

    layers = [manifest.LayerEntry.from_layer(
        lay, attribute=i + 1, r_inner=b[i], r_outer=b[i + 1],
        in_geometry=i < n_geometry)
        for i, lay in enumerate(domain.layers)]
    interfaces = [manifest.InterfaceEntry.from_interface(
        face, attribute=k + 1, radius=face.radius)
        for k, face in enumerate(domain.interfaces)]
    return manifest.MeshManifest.from_build(
        mesh=manifest.mesh_block(msh_path, format="msh", nodes="reference"),
        layers=layers, interfaces=interfaces, meta=spec.meta)
