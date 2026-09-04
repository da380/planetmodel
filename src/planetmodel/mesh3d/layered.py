"""The assembly line: a MeshSpec becomes a mesh on disk.

The computational domain is the geometry followed by its shells.  Every
length is divided by the spec's divisor and the mapping is conjugated
by the same factor with `ScaledMapping`, so gmsh sees coordinates of
order one whatever the geometry's units.  Then CAD, tagging, sizing,
meshing at order 1, orientation, curving; then, for a physical delivery
of a non-identity mapping, the node displacement; then validation; and
only then anything is written.  A mesh that fails its checks and exists
anyway looks finished, so nothing reaches disk before validation.

With shells and a non-identity mapping the mapping must be defined and
orientation-preserving out to the outer boundary of the computational
domain and the identity on that boundary, both checked on a lattice
before any meshing and refused by name otherwise.  Without shells the
geometry's own checks are the whole guarantee.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import gmsh
import numpy as np

from ..frames import cartesian_points
from ..geometry import Geometry
from ..mapping import validity_lattice
from . import manifest
from ._displace import apply_mapping
from ._geometry import build_concentric
from ._orient import orient_mesh, raise_order
from ._session import session
from ._sizing import (apply_mesh_options, apply_size_fields,
                      check_sizing_resolves_spans, check_sizing_scale)
from ._tagging import apply_physical_groups, identify
from ._validate import check_interface_radii, validate_mesh
from ._writer import confirm_reread, element_counts, write_msh
from .spec import MeshResult, MeshSpec

__all__ = ["build_layered_mesh", "require_mapping_on_shells", "policy_name"]

#: Outer radii outside this range, in mesh units, put gmsh far from the
#: coordinate magnitudes its absolute kernel tolerances are tuned for.
_GMSH_COMFORTABLE_RANGE = (1e-3, 1e4)


def require_mapping_on_shells(spec: MeshSpec) -> None:
    """Refuse a mapping that is not defined on the shells or moves their edge.

    The computational domain under the geometry's mapping must satisfy
    every invariant of a Geometry (knots on boundaries, orientation
    preserved on the validity lattice, continuity across every interior
    boundary), and the mapping must be the identity on the outer
    boundary to `geometry.rtol` times that radius.
    """
    g = spec.geometry
    domain = spec.domain
    outer = spec.outer_radius
    try:
        Geometry(domain.skeleton, mapping=g.mapping, rtol=g.rtol, check=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "with shells the mapping must be defined and orientation-preserving "
            f"on the whole computational domain, out to r = {outer:g}: {exc}"
        ) from exc
    _, theta, phi = validity_lattice(domain.skeleton)
    X = cartesian_points(outer, theta, phi)
    gap = float(np.max(np.linalg.norm(np.asarray(g.mapping(X), dtype=float) - X,
                                      axis=-1)))
    tol = g.rtol * outer
    if gap > tol:
        raise ValueError(
            "with shells the mapping must be the identity on the outer boundary "
            f"of the computational domain at r = {outer:g}, and it moves points "
            f"there by up to {gap:.3g} (tolerance {tol:.3g}); make the "
            "displacement vanish on the outermost shell, or mesh without shells")


def policy_name(rule) -> str:
    """The name a manifest records for a sizing rule: its class, or the
    function's name."""
    return getattr(rule, "__name__", None) or type(rule).__name__


def build_layered_mesh(spec: MeshSpec, path, *, verbose: bool = False
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
    divisor = spec.effective_divisor
    if spec.shells and not geometry.is_identity:
        require_mapping_on_shells(spec)
    nd = domain.scaled(1.0 / divisor)
    mapping = nd.mapping
    moved = spec.delivery == "physical" and not nd.is_identity
    timings["resolve"] = clock() - t0

    d = spec.dimension
    boundaries = nd.skeleton.boundaries
    outer_nd = float(boundaries[-1])
    lo, hi = _GMSH_COMFORTABLE_RANGE
    if not lo <= outer_nd <= hi:
        warnings.warn(
            f"the divided outer radius is {outer_nd:g}, outside [{lo:g}, {hi:g}]; "
            "gmsh's kernel tolerances are absolute and tuned for coordinates "
            "of order one, so expect degraded robustness. Check the divisor.",
            stacklevel=2)
    interface_radii = [f.radius for f in nd.interfaces]
    layer_names = [lay.name for lay in domain.layers]
    interface_names = [f.name for f in domain.interfaces]

    # Sizing is computed in the geometry's own lengths and divided with them.
    sizes = {i: s.scaled(1.0 / divisor)
             for i, s in spec.sizing(domain.interfaces, spec.outer_radius).items()}
    check_sizing_scale(outer_nd, sizes)
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
        orientation = orient_mesh(d)
        curving = raise_order(d, spec.order)
        timings["orient"] = clock() - t0

        perturbation = None
        radius_check = None
        if moved:
            t0 = clock()
            # The radius check is a property of the reference geometry, so
            # it is measured before the nodes carry the mapping.
            radius_check = check_interface_radii(tagging, interface_radii)
            perturbation = apply_mapping(mapping)
            timings["perturb"] = clock() - t0

        t0 = clock()
        report = validate_mesh(
            tagging, expected_radii=interface_radii,
            layer_names=layer_names, interface_names=interface_names,
            radius_check=radius_check)
        if spec.validate:
            report.raise_if_failed()
        timings["validate"] = clock() - t0

        t0 = clock()
        counts = element_counts(dimension=d)
        # Read inside the session: gmsh answers nothing once finalized.
        gmsh_version = gmsh.option.getString("General.Version")
        msh_path = manifest.beside(path, ".msh")
        card = _build_manifest(spec, nd, divisor, sizes, counts, report,
                               curving, orientation, perturbation, msh_path,
                               gmsh_version, moved=moved)
        # The manifest is checked and written before the mesh: a mesh on
        # disk with no manifest, or with one that disagrees, looks finished.
        manifest.validate_against(card, layer_count=nd.nlayers,
                                  interface_count=len(nd.interfaces),
                                  groups={k: list(v) for k, v in groups.items()})
        manifest_path = manifest.write(path, card)
        try:
            write_msh(path)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            raise
        timings["write"] = clock() - t0

    confirm_reread(msh_path, manifest_path, d, layer_names, interface_names)

    counts["layers"] = nd.nlayers
    counts["interfaces"] = len(nd.interfaces)
    return MeshResult(msh_path=msh_path, manifest_path=manifest_path,
                      geometry=geometry, counts=counts, validation=report,
                      timings=timings, spec=spec, divisor=divisor,
                      mapping=mapping)


def _build_manifest(spec, nd, divisor, sizes, counts, report, curving,
                    orientation, perturbation, msh_path, gmsh_version, *,
                    moved: bool):
    """Assemble the manifest from what the build did; `nd` is the
    computational domain in mesh units."""
    b = nd.skeleton.boundaries
    n_geometry = spec.geometry.nlayers

    layers = [manifest.LayerEntry.from_layer(
        lay, attribute=i + 1, r_inner_nd=b[i], r_outer_nd=b[i + 1],
        in_geometry=i < n_geometry)
        for i, lay in enumerate(nd.layers)]
    interfaces = [manifest.InterfaceEntry.from_interface(
        face, attribute=k + 1, mean_radius_nd=face.radius)
        for k, face in enumerate(nd.interfaces)]

    return manifest.MeshManifest.from_build(
        geometry=manifest.geometry_block(
            divisor=divisor, outer_radius_nd=b[-1], inner_radius_nd=b[0],
            n_layers=nd.nlayers, n_shells=len(spec.shells)),
        mesh=manifest.mesh_block(
            dimension=spec.dimension, order=spec.order,
            gmsh_version=gmsh_version, algorithm_2d=spec.algorithm_2d,
            algorithm_3d=spec.algorithm_3d, counts=counts, curving=curving),
        delivery=spec.delivery, layers=layers, interfaces=interfaces,
        mapping=manifest.mapping_block(nd.mapping, knots_nd=nd.knots(),
                                       applied_to_nodes=moved),
        sizing=manifest.sizing_block(policy=policy_name(spec.sizing),
                                     sizes=sizes),
        validation=manifest.validation_block(report, orientation),
        provenance=manifest.provenance_block(
            mesh_file=msh_path.name, perturbation=perturbation,
            meta=spec.meta))
