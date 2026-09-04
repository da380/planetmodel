"""layered.py -- the assembly line: a MeshSpec becomes a mesh on disk.

One function, and the order of its steps is the design.  Geometry
surgery first, in the one order that works (coarsen, set the outer
boundary, refine, extend, buffer) because each step assumes the
previous one: cutting after refining would drop inserted interfaces,
and buffering before extending would bury the buffer inside the body.
Then units are resolved once; then CAD, tagging, sizing, meshing at
order 1, orientation, curving; then -- for a physical delivery -- the
node displacement; then validation; and only then anything is written.

Nothing reaches disk before validation passes.  A mesh that fails its
checks and exists anyway is worse than no mesh: it looks finished.
"""
from __future__ import annotations

import time
from pathlib import Path

import gmsh

from ..io import manifest
from ..model.mapping import validity_lattice
from ..registry import lookup, name_of
from ._geometry import build_concentric
from ._orient import orient_mesh, raise_order
from ._displace import apply_mapping
from ._session import session
from ._sizing import (apply_mesh_options, apply_size_fields,
                      check_sizing_resolves_spans,
                      check_sizing_scale)
from ._tagging import apply_physical_groups, identify
from ._units import GeometryScaledMapping, resolve_mesh_units
from ._validate import check_interface_radii, validate_mesh
from ._writer import confirm_reread, element_counts, write_msh
from .spec import MeshResult, MeshSpec

__all__ = ["build_layered_mesh", "resolve_body"]


def resolve_body(spec: MeshSpec):
    """Apply the spec's geometry surgery, in the one order that works."""
    body = spec.body
    coarsening: dict = {}

    if spec.keep_interfaces is not None or spec.drop_interfaces is not None:
        body, cmap = body.coarsened(keep=spec.keep_interfaces,
                                    drop=spec.drop_interfaces)
        coarsening = {"kept_interfaces": list(cmap.kept_interfaces),
                      "dropped_interfaces": list(cmap.dropped_interfaces)}

    if spec.outer_radius is not None:
        body, grown = _boundary_at(body, float(spec.outer_radius),
                                   spec.outer_name)
        coarsening["grown_to_boundary"] = bool(grown)

    if len(spec.insert_radii):
        body = body.refined(spec.insert_radii,
                            names=list(spec.insert_names) or None,
                            role=spec.insert_role)

    if len(spec.extend_radii):
        body = body.extended(spec.extend_radii,
                             names=list(spec.extend_names) or None,
                             fields=spec.extend_fields, role=spec.extend_role)

    for buf in spec.buffers:
        body = body.with_buffer(ratio=buf.ratio, radius=buf.radius,
                                name=buf.name)

    for which, shape in spec.surfaces.items():
        body = body.with_surface(which, shape)

    coarsening["inserted_radii"] = [float(r) for r in spec.insert_radii]
    coarsening["extended_radii"] = [float(r) for r in spec.extend_radii]
    if spec.outer_radius is not None:
        coarsening["outer_radius"] = float(spec.outer_radius)
    return body, coarsening


def _refuse_a_folded_mapping(mapping, body) -> None:
    """Check the mapping analytically, before anything is meshed.

    The discrete check in _displace is the one that matters -- a curved
    element can fold between corners that are individually fine -- but
    it can only run on a finished mesh, and a mapping that folds
    analytically will certainly fail it.  Finding that out first turns
    a wasted mesh into an immediate refusal.

    The sample is laid per span, so a thin crust gets points even
    though it is a thousandth of the body: thin spans are exactly where
    dh/dr is largest.
    """
    verdict = mapping.is_valid(sample=validity_lattice(body.skeleton))
    if not verdict:
        raise ValueError(
            f"the mapping is not orientation-preserving on this body: "
            f"{verdict!r}. Lower the exaggeration, move the confining "
            "interface so the displacement has further to ramp over, or "
            "give the relief a thicker layer to act in.")


def _boundary_at(body, radius: float, name):
    """Put the body's outer boundary at `radius`, cutting or growing.

    Cutting is the usual direction.  Growing happens when a boundary the
    data supplies sits above where the model's own knots stop --
    CRUST-1.0's mean Moho is three kilometres above prem.nocrust's outer
    radius, so a deck with no crust has to reach up to meet it.

    Growing then *merges away* the old boundary. Extending alone leaves
    an interface with the same extrapolated material on both sides: no
    discontinuity, nothing for a solver to do with it, and a three
    kilometre shell that dictates the element size of the whole crust,
    since a span this thin refuses any sizing coarser than about thirty
    kilometres. The material is continuous across it, so the geometry
    should be too.
    """
    outer = float(body.skeleton.boundaries[-1])
    if radius <= outer:
        return body.truncated(radius, name=name), False
    body = body.extended([radius], fields="extrapolate", names=[name])
    body, _ = body.coarsened(drop=[len(body.interfaces) - 2])
    return body, True


def _reference_radius_for_sizing(spec: MeshSpec, body) -> float:
    """The radius a sizing rule scales by, in the body's units.

    The one the spec names, else the solid body's surface -- not the
    buffer's edge, which would make every element coarser by the buffer
    ratio.  A non-dimensional body without `rref` scales by its own
    outer solid boundary, which for a body cut at the Moho is the Moho
    rather than the length scale it was non-dimensionalised by; a rule
    whose `r_ref` was chosen for the whole planet should be given
    `rref` explicitly in that case.
    """
    if spec.rref:
        return float(spec.rref)
    solid = max(i for i, lay in enumerate(body.layers) if not lay.is_vacuum)
    return float(body.skeleton.boundaries[solid + 1])


def build_layered_mesh(spec: MeshSpec, path, *, verbose: bool = False
                       ) -> MeshResult:
    """Build, check and write the mesh a MeshSpec describes."""
    path = Path(path)
    timings: dict[str, float] = {}
    clock = time.perf_counter

    t0 = clock()
    body, coarsening = resolve_body(spec)
    units = resolve_mesh_units(body, spec.rref)
    # A rule reads the surfaces, so the mapping is built here and not by
    # the caller: before resolve_body the interfaces it attaches to may
    # not exist yet.
    base_mapping = (body.mapping(rule=spec.mapping_rule)
                    if spec.mapping_rule is not None else spec.mapping)
    if base_mapping is None and body.surfaces:
        raise ValueError(
            "the body carries surfaces but the spec names no mapping and no "
            "mapping_rule: the relief would never reach the mesh. Give "
            "mapping_rule=layer_linear(), or detach the surfaces.")
    if base_mapping is not None:
        _refuse_a_folded_mapping(base_mapping, body)
    timings["resolve"] = clock() - t0

    d = spec.dimension
    radii_mesh = units.to_mesh(body.skeleton.boundaries)
    interface_radii = radii_mesh[1:]

    # Sizing is computed in the body's own units and converted with the
    # geometry, so a rule never has to know which units it is in.
    sizes_body = spec.sizing(body.interfaces,
                             _reference_radius_for_sizing(spec, body))
    sizes = {i: s.scaled(1.0 / units.divisor) for i, s in sizes_body.items()}

    layer_names = [lay.name for lay in body.layers]
    interface_names = [f.name for f in body.interfaces]

    check_sizing_scale(float(interface_radii[-1]), sizes)
    check_sizing_resolves_spans(interface_radii, sizes)

    mapping = base_mapping
    if mapping is not None and units.divisor != 1.0:
        mapping = GeometryScaledMapping(mapping, units)

    with session(name=path.stem or "planetmodel", verbose=verbose):
        t0 = clock()
        geometry = build_concentric(radii_mesh, dimension=d)
        tagging = identify(geometry, interface_radii)
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
        if spec.delivery == "physical" and mapping is not None:
            t0 = clock()
            # The radius check is a property of the reference geometry,
            # so it is measured now: once the nodes carry the relief the
            # interfaces are *supposed* to be away from their reference
            # radii, and measuring afterwards would either fail for any
            # relief with a non-zero mean or need a tolerance loose
            # enough to hide real tagging errors.
            radius_check = check_interface_radii(tagging, interface_radii)
            perturbation = apply_mapping(mapping)
            timings["perturb"] = clock() - t0

        t0 = clock()
        # The mapping goes in whichever the delivery: a kink inside an
        # element misplaces quadrature in a referential delivery and
        # warps the element geometry in a physical one, so the knot
        # check applies to both.
        report = validate_mesh(
            tagging, expected_radii=interface_radii,
            layer_names=layer_names, interface_names=interface_names,
            mapping=mapping, radius_check=radius_check)
        if spec.validate:
            report.raise_if_failed()
        timings["validate"] = clock() - t0

        t0 = clock()
        counts = element_counts(dimension=d)
        # Read inside the session: gmsh answers nothing once finalized.
        gmsh_version = gmsh.option.getString("General.Version")
        msh_path = manifest.beside(path, ".msh")
        card = _build_manifest(spec, body, units, sizes, coarsening, counts,
                               report, curving, orientation, perturbation,
                               msh_path, gmsh_version, mapping=base_mapping)
        # The manifest is checked and written before the mesh: a mesh on
        # disk with no manifest, or with one that disagrees, looks finished.
        manifest.validate_against(card, layer_count=len(body.layers),
                                  interface_count=len(body.interfaces),
                                  groups={k: list(v) for k, v in
                                          (("layers", groups["layers"]),
                                           ("interfaces", groups["interfaces"]))})
        manifest_path = manifest.write(path, card)
        try:
            write_msh(path)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            raise
        timings["write"] = clock() - t0

    confirm_reread(msh_path, manifest_path, d, layer_names, interface_names)

    counts["layers"] = len(body.layers)
    counts["interfaces"] = len(body.interfaces)
    return MeshResult(msh_path=msh_path, manifest_path=manifest_path, body=body,
                      counts=counts, validation=report, timings=timings,
                      spec=spec, units=units, mapping=base_mapping)


def _build_manifest(spec, body, units, sizes, coarsening, counts, report,
                    curving, orientation, perturbation, msh_path,
                    gmsh_version, *, mapping=None):
    """Assemble the manifest from what the build actually did."""
    b = units.to_mesh(body.skeleton.boundaries)

    layers = [manifest.LayerEntry.from_layer(
        lay, attribute=i + 1, r_inner_nd=b[i], r_outer_nd=b[i + 1])
        for i, lay in enumerate(body.layers)]
    interfaces = [manifest.InterfaceEntry.from_interface(
        face, attribute=i + 1, mean_radius_nd=units.to_mesh(face.radius))
        for i, face in enumerate(body.interfaces)]

    # resolve_body records its lengths in the body's own units, since it
    # runs before the units are resolved; the manifest speaks nd only.
    coarsening = dict(coarsening)
    for key in ("inserted_radii", "extended_radii"):
        coarsening[key + "_nd"] = [float(units.to_mesh(r))
                                   for r in coarsening.pop(key, ())]
    if "outer_radius" in coarsening:
        coarsening["outer_radius_nd"] = float(
            units.to_mesh(coarsening.pop("outer_radius")))
        coarsening["outer_name"] = spec.outer_name

    source = body.meta.get("source")
    model = {
        "name": body.meta.get("name", "unnamed"),
        "source": source,
        "sha256": manifest.file_digest(source) if source else None,
        "rref_m": units.rref_m,
        "units": manifest.units_block(body.scales, units.divisor,
                                      units.rref_m),
    }

    mapping_block = None
    if mapping is not None:
        h = getattr(mapping, "h", None)
        mapping_block = {
            "kind": ("identity" if getattr(mapping, "is_identity", False)
                     else "radial_stretch"),
            "rule": _rule_block(spec.mapping_rule, h, units),
            "knots_nd": [float(units.to_mesh(k))
                         for k in getattr(h, "knots", ())],
            "surfaces": [_surface_entry(body, i, units,
                                        declared=spec.meta.get("surface_sources", {}))
                         for i in sorted(body.surfaces)],
            "applied_to_nodes": spec.delivery == "physical",
        }

    recipe_keys = {k: v for k, v in spec.meta.items()
                   if k in ("recipe", "recipe_sha256", "command")}
    return manifest.MeshManifest.from_build(
        model=model,
        mesh=manifest.mesh_block(
            dimension=spec.dimension, order=spec.order,
            gmsh_version=gmsh_version, algorithm_2d=spec.algorithm_2d,
            algorithm_3d=spec.algorithm_3d, counts=counts, curving=curving),
        delivery=spec.delivery, layers=layers, interfaces=interfaces,
        coarsening=coarsening, mapping=mapping_block,
        sizing=manifest.sizing_block(
            policy=(name_of("sizing", spec.sizing)
                    or type(spec.sizing).__name__),
            sizes=sizes),
        validation=manifest.validation_block(report, orientation),
        provenance=manifest.provenance_block(
            mesh_file=msh_path.name, perturbation=perturbation,
            **recipe_keys))


def _rule_block(rule, h, units) -> dict:
    """The displacement rule, named and parameterised well enough to rebuild.

    A registered name plus its parameters is the whole point: a
    consumer applying the mapping itself (a referential delivery) has
    to construct the same rule, and a name alone leaves out the two
    things that move the knots.  An unregistered rule cannot be named,
    so it is recorded as custom with a repr and reproducibility falls
    back honestly to the caller's script.
    """
    name = name_of("displacement_rule", rule) if rule is not None else None
    if name is None and rule is None and h is not None:
        # A mapping given ready-made still carries the name of the rule
        # that built its displacement, and that name is what a consumer
        # rebuilding the mapping needs; the parameters it cannot recover
        # from `h` are recorded as unknown.
        hname = getattr(h, "name", None)
        try:
            lookup("displacement_rule", hname)
            name = hname
        except (KeyError, TypeError):
            name = None
    if name is None:
        source = rule if rule is not None else h
        return {"name": getattr(h, "name", None) or type(source).__name__,
                "registered": False, "repr": repr(source)}
    block = {"name": name, "registered": True}
    taper = getattr(rule, "inner_taper_radius", None)
    block["inner_taper_radius_nd"] = (None if taper is None
                                      else float(units.to_mesh(taper)))
    block["control_radii_nd"] = [float(units.to_mesh(r))
                                 for r in getattr(rule, "control_radii", ())]
    return block


def _surface_entry(body, i, units, *, declared=None) -> dict:
    """One mapping-block surface: where it sits and what shaped it.

    A boundary shape reaches the mesher as a tree -- a sum of grids,
    scaled by an exaggeration factor -- and the files that made it are
    at the leaves, where from_xyz named each one after its path.
    Walking to them keeps the block reconstruction-grade for a shape
    built in Python; `declared` overrides the walk for a caller, the
    recipe reader among them, that knows its own inputs.
    """
    face = body.interfaces[i]
    topo = body.surfaces[i].topography
    sources, exaggeration, interpolation = _sources_of(topo)
    # A caller that knows its files says so: summing two grids on one
    # lat-lon mesh adds their values into a single shape, and no walk of
    # the result can name the files that went into it.
    named = (declared or {}).get(face.name)
    if named:
        sources = [dict(entry, sha256=manifest.file_digest(entry["file"]))
                   for entry in named]
    return {
        "interface": i + 1,
        "name": face.name,
        "mean_radius_nd": float(units.to_mesh(face.radius)),
        "topography": getattr(topo, "name", None) or type(topo).__name__,
        "exaggeration": exaggeration,
        "interpolation": interpolation,
        "sources": sources,
    }


def _sources_of(topo) -> tuple[list, float, str | None]:
    """The files a shape was built from, its exaggeration and interpolation."""
    p = manifest.provenance_of(topo)
    return p["files"], float(p["exaggeration"]), p["interpolation"]
