"""The assembly line: MeshSpec in, checked mesh and manifest out.

Almost everything here runs on a three-shell synthetic body of unit
radius, which meshes in under a second.  That is deliberate: these tests
ask whether the *pipeline* is correct -- surgery order, tagging,
validation, manifest contents, the units rule -- and none of those
questions is answered better by a two-million-element PREM, and nothing
here builds one: PREM appears only where a span guard is checked against
real spans, which needs no mesh.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from planetmodel import PREM, ReferenceBody, Skeleton, layer_linear
from planetmodel.io import manifest as sc
from planetmodel.mesh3d import (AngularResolution, BufferSpec, MeshSpec,
                           UniformInterfaces, build_layered_mesh)
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._sizing import (check_sizing_resolves_spans,
                                   check_sizing_scale)
from planetmodel.mesh3d._writer import read_groups
from planetmodel.mesh3d.spec import InterfaceSizing
from planetmodel.model.topography import AnalyticTopography

pytestmark = pytest.mark.gmsh

#: One mesh length unit, in metres, for the synthetic bodies.
RREF = 1.0e6


#: Sizing rules take lengths in the BODY's units, so these are metres.
COARSE = UniformInterfaces(0.15e6, 0.30e6, 0.30e6)


def relief(amp=0.02e6):
    """Relief of 2% of the body radius: visible, comfortably valid."""
    return AnalyticTopography(lambda t, p: amp * np.cos(t))


def three_shells():
    """A body whose spans and sizes are all of order one in mesh units."""
    sk = Skeleton([0.0, 0.2e6, 0.55e6, 1.0e6])
    return (ReferenceBody.from_fields(sk, {})
            .annotate(0, name="core")
            .annotate(1, name="mantle")
            .annotate(2, name="crust")
            .name_interface(0, "icb")
            .name_interface(1, "cmb")
            .name_interface(-1, "surface"))


@pytest.fixture(scope="module")
def body():
    return three_shells().with_surface("surface", relief()).with_buffer(
        ratio=0.2)


def spec_for(body, **kw):
    kw.setdefault("sizing", COARSE)
    kw.setdefault("rref", RREF)
    kw.setdefault("order", 1)
    return MeshSpec(body=body, **kw)


@pytest.fixture(scope="module")
def built(body, tmp_path_factory):
    path = tmp_path_factory.mktemp("build") / "shells"
    spec = spec_for(body, order=2, mapping=body.mapping(rule=layer_linear()))
    return build_layered_mesh(spec, path)


# ------------------------------------------------------------ end to end

def test_the_build_produces_a_mesh_and_a_manifest(built):
    assert built.msh_path.exists() and built.manifest_path.exists()
    assert built.counts["elements"] > 0
    assert built.counts["layers"] == 4          # 3 shells + buffer


def test_validation_passes_on_every_count(built):
    v = built.validation
    assert v.ok, v.failures
    assert v.negative_jacobians == 0
    assert v.negative_cells == 0 and v.inward_faces == 0
    assert v.min_sicn > 0.0
    assert v.knots_aligned


def test_interfaces_land_where_the_model_puts_them(built):
    """The check that catches a mesh built from the wrong geometry."""
    assert built.validation.max_interface_radius_error < 1e-3


def test_the_mesh_is_msh_2_2_with_groups_intact(built):
    header = built.msh_path.read_text(errors="ignore").splitlines()[:2]
    assert header[1].split()[0] == "2.2"
    with session(name="reread"):
        groups = read_groups(built.msh_path)
        assert sorted(groups[3]) == [1, 2, 3, 4]
        assert sorted(groups[2]) == [1, 2, 3, 4]


def test_names_reach_the_mesh_file(built):
    with session(name="names"):
        groups = read_groups(built.msh_path)
        assert groups[3][1] == "core"
        assert groups[2][2] == "cmb"


# --------------------------------------------------------------- manifest

def test_manifest_round_trips_and_matches_the_mesh(built):
    card = sc.read(built.manifest_path)
    assert card.schema == sc.SCHEMA
    sc.validate_against(card, layer_count=4, interface_count=4)
    assert card.rref_m == pytest.approx(RREF)
    assert card.delivery == "physical"


def test_manifest_names_the_buffer_so_a_solver_can_exclude_it(built):
    card = sc.read(built.manifest_path)
    assert card.vacuum_attributes == (4,)
    assert card.layers[-1]["is_vacuum"] is True
    assert card.layers[-1]["fields"] == []      # a void holds nothing


def test_manifest_lookup_by_name(built):
    card = sc.read(built.manifest_path)
    assert card.interface_attribute("surface") == 3
    assert card.layer_attribute("mantle") == 2
    with pytest.raises(KeyError, match="no interface named"):
        card.interface_attribute("moho")


def test_manifest_records_the_units_with_room_to_grow(built):
    """rref_m flat for the C++ reader, the full triple beside it."""
    units = json.loads(built.manifest_path.read_text())["model"]["units"]
    assert units["rref_m"] == pytest.approx(RREF)
    assert set(units["scales"]) == {"length_m", "mass_kg", "time_s"}
    assert units["scales"]["length_m"] is None      # an SI body sets none
    assert units["convention"] == "non-dimensional"


def test_manifest_mapping_is_reconstruction_grade(built):
    card = sc.read(built.manifest_path)
    assert card.mapping["kind"] == "radial_stretch"
    assert card.mapping["rule"]["name"] == "layer_linear"
    assert card.mapping["applied_to_nodes"] is True
    assert max(card.mapping["knots_nd"]) == pytest.approx(1.2, abs=1e-6)


def test_manifest_rejects_a_foreign_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "something/else"}')
    with pytest.raises(ValueError, match="schema"):
        sc.read(path)


def test_validate_against_catches_a_mismatch(built):
    card = sc.read(built.manifest_path)
    with pytest.raises(ValueError, match="lists 4 layers"):
        sc.validate_against(card, layer_count=9, interface_count=4)


# ---------------------------------------------------------- perturbation

def test_the_nodes_actually_moved(built):
    p = json.loads(built.manifest_path.read_text())["provenance"]["perturbation"]
    assert p["nodes"] > 0
    assert p["max_displacement_nd"] == pytest.approx(0.02, rel=0.1)


def test_the_referential_mode_leaves_the_nodes_alone(body, tmp_path):
    """Mode B: the reference mesh plus the mapping, unperturbed."""
    spec = spec_for(body, delivery="referential",
                    mapping=body.mapping(rule=layer_linear()))
    res = build_layered_mesh(spec, tmp_path / "ref")
    card = sc.read(res.manifest_path)
    assert card.delivery == "referential"
    assert card.mapping["applied_to_nodes"] is False
    assert json.loads(res.manifest_path.read_text())[
        "provenance"]["perturbation"] is None


def test_a_spherical_body_needs_no_mapping(tmp_path):
    res = build_layered_mesh(spec_for(three_shells()), tmp_path / "sphere")
    assert res.validation.ok
    assert sc.read(res.manifest_path).mapping is None


def test_an_invalid_mapping_stops_before_anything_is_written(tmp_path):
    """A half-displaced mesh that exists is worse than none: it looks
    finished."""
    huge = three_shells().with_surface("surface", relief(0.02e6) * 60.0)
    spec = spec_for(huge, mapping=huge.mapping(rule=layer_linear()))
    path = tmp_path / "folded"
    with pytest.raises(ValueError, match="orientation-preserving|folds|cross"):
        build_layered_mesh(spec, path)
    assert not path.with_suffix(".msh").exists()
    assert not path.with_suffix(".json").exists()


# ------------------------------------------------------------ the guards

def test_sizing_in_the_wrong_units_is_named_as_such(tmp_path):
    """The mistake this guard exists for -- and the one I made writing
    these tests.  Sizing rules take the body's units, so values already
    in mesh units get divided a second time, and gmsh reports the
    result as "identical points in triangulation"."""
    spec = spec_for(three_shells(), sizing=UniformInterfaces(0.15, 0.3, 0.3))
    with pytest.raises(ValueError, match="not a resolution choice"):
        build_layered_mesh(spec, tmp_path / "units")


def test_elements_larger_than_the_body_are_refused():
    sizes = {0: InterfaceSizing(5.0, 6.0, 1.0)}
    with pytest.raises(ValueError, match="nothing would be resolved"):
        check_sizing_scale(1.0, sizes)


def test_coarse_sizing_is_refused_with_a_diagnosis(tmp_path):
    """gmsh reports this as an unrelated-looking PLC error."""
    body = PREM(ocean=False).with_buffer(ratio=0.2)
    spec = spec_for(body, rref=6.368e6,
                    sizing=AngularResolution(h_ref=400e3, r_ref=6.368e6,
                                             h_far=900e3))
    with pytest.raises(ValueError, match="too coarse"):
        build_layered_mesh(spec, tmp_path / "coarse")


def test_the_span_threshold_is_where_gmsh_actually_breaks():
    """Measured: gmsh meshes at 8x too coarse and fails at 15x, so the
    refusal sits at 10x rather than at the first sign of coarseness."""
    radii = np.array([0.5, 0.52, 1.0])
    ok = {i: InterfaceSizing(0.1, 0.3, 0.3) for i in range(3)}
    assert check_sizing_resolves_spans(radii, ok, strict=False) == []
    coarse = {i: InterfaceSizing(0.3, 0.6, 0.3) for i in range(3)}
    assert check_sizing_resolves_spans(radii, coarse, strict=False)


def test_an_si_body_without_rref_is_refused(tmp_path):
    spec = MeshSpec(body=three_shells(), rref=None, sizing=COARSE)
    with pytest.raises(ValueError, match="needs rref"):
        build_layered_mesh(spec, tmp_path / "norref")


def test_a_nondimensional_body_needs_no_rref(tmp_path):
    """The other half of the units rule, end to end."""
    nd = three_shells().nondimensionalised()
    spec = MeshSpec(body=nd, rref=None, order=1,
                    sizing=UniformInterfaces(0.15, 0.30, 0.30))
    res = build_layered_mesh(spec, tmp_path / "nd")
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    assert card.rref_m == pytest.approx(nd.scales.length)
    units = json.loads(res.manifest_path.read_text())["model"]["units"]
    assert units["scales"]["length_m"] == pytest.approx(nd.scales.length)
    assert units["scales"]["mass_kg"] is not None      # the triple is filled
    assert units["gravitational_constant"] == pytest.approx(1.0, abs=1e-12)


def test_giving_both_rref_and_scales_is_refused(tmp_path):
    nd = three_shells().nondimensionalised()
    spec = MeshSpec(body=nd, rref=5.0e5,
                    sizing=UniformInterfaces(0.15, 0.30, 0.30))
    with pytest.raises(ValueError, match="one answer, not two"):
        build_layered_mesh(spec, tmp_path / "both")


# ---------------------------------------------------- 2D and the surgery

def test_two_dimensions_goes_through_the_same_path(tmp_path):
    spec = spec_for(three_shells(), dimension=2)
    res = build_layered_mesh(spec, tmp_path / "disc")
    assert res.validation.ok
    assert res.counts["layers"] == 3


def test_surgery_happens_in_the_documented_order(tmp_path):
    """Coarsen, truncate, refine, extend, buffer -- each assumes the last."""
    spec = spec_for(three_shells(),
                    drop_interfaces=[0],                  # merge core into mantle
                    insert_radii=[0.75e6], insert_names=["floor"],
                    insert_role="control",
                    buffers=[BufferSpec(ratio=0.2)])
    res = build_layered_mesh(spec, tmp_path / "surgery")
    card = sc.read(res.manifest_path)

    roles = {f["name"]: f["role"] for f in card.interfaces}
    assert roles["floor"] == "control"                    # inserted and marked
    assert card.vacuum_attributes == (len(card.layers),)  # buffer outermost
    radii = [round(f["mean_radius_nd"], 4) for f in card.interfaces]
    assert 0.75 in radii and 0.2 not in radii             # inserted; dropped


# ------------------------------------------------- review regressions

def test_a_kink_inside_an_element_is_flagged_in_physical_mode(tmp_path):
    """control_radii warn in Mode A too: the element geometry warps there."""
    body = three_shells().with_surface("surface", relief())
    rule = layer_linear(control_radii=(0.8e6,))
    spec = spec_for(body, mapping=body.mapping(rule=rule))
    res = build_layered_mesh(spec, tmp_path / "kink")
    card = sc.read(res.manifest_path)
    assert card.validation["knots_aligned_with_interfaces"] is False
    assert any("kink" in w for w in res.validation.warnings)


def test_the_manifest_speaks_nd_throughout(tmp_path):
    """Surgery lengths are recorded in mesh units like every other length,
    and the sizing policy under the name a recipe file would use."""
    spec = spec_for(three_shells(), insert_radii=[0.75e6],
                    insert_names=["floor"], insert_role="control")
    res = build_layered_mesh(spec, tmp_path / "ndsc")
    card = sc.read(res.manifest_path)
    assert card.coarsening["inserted_radii_nd"] == [pytest.approx(0.75)]
    assert card.sizing["policy"] == "uniform_interfaces"


def test_mapping_surfaces_name_their_topography(built):
    """A surface entry says what shaped it, with a hash where that is a
    file -- the part that makes the block reconstruction-grade."""
    (surf,) = sc.read(built.manifest_path).mapping["surfaces"]
    assert surf["name"] == "surface"
    assert surf["topography"] == "AnalyticTopography"
    assert surf["sources"] == []                  # nothing on disk to hash
    assert surf["exaggeration"] == 1.0


# ------------------------------------------------------------------ law

def test_the_manifest_names_the_law_behind_each_layers_moduli(tmp_path):
    """static on the lifted layers, maxwell on the mantle, null elsewhere."""
    from planetmodel import RadialField
    from planetmodel.model.character import DENSITY, SCALAR
    from planetmodel.model.fields.frequency import lifted_to_frequency
    from planetmodel.model.materials import ElasticField, Symmetry
    from planetmodel.model.rheology import maxwell
    from planetmodel.model.units import Dimensions
    from planetmodel.model.classes import ViscoelasticModel

    base = three_shells()
    sk = base.skeleton
    const = lambda v, name, dims: RadialField(  # noqa: E731
        sk, [lambda r, v=v: v + 0.0 * r] * sk.nlayers, name=name, dimensions=dims,
        character=DENSITY if name == "rho" else SCALAR)
    rho = const(5.0e3, "rho", Dimensions.DENSITY)
    kappa = const(1.0e11, "kappa", Dimensions.MODULUS)
    mu = const(6.0e10, "mu", Dimensions.MODULUS)
    eta = const(1.0e21, "viscosity", Dimensions.VISCOSITY)
    el = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu},
                      name="elastic_moduli")
    body = ReferenceBody.from_fields(sk, {"rho": rho, "kappa": kappa, "mu": mu,
                                     "elastic_moduli": el, "viscosity": eta})
    law = maxwell(body["elastic_moduli"], body["viscosity"])
    layers = [lay.with_field("viscoelastic_moduli",
                             law.restricted(i) if i == 1
                             else lifted_to_frequency(lay["elastic_moduli"]))
              for i, lay in enumerate(body.layers)]
    model = ViscoelasticModel(layers, interfaces=base.interfaces)
    model = model.with_surface("surface", relief()).with_buffer(ratio=0.2)
    spec = spec_for(model, mapping=model.mapping(rule=layer_linear()))
    result = build_layered_mesh(spec, tmp_path / "law")
    card = sc.read(result.manifest_path)
    assert [lay["law"] for lay in card.layers] == ["static", "maxwell", "static",
                                                    None]
    assert "viscoelastic_moduli" in card.layers[1]["fields"]
    assert card.layers[3]["fields"] == []


def test_the_knot_check_runs_for_a_nondimensional_body(tmp_path):
    # Declared non-dimensional (a unit length), so the mesher applies no
    # scaling wrapper -- the path on which the knot check fell through.
    body = (ReferenceBody.from_fields(Skeleton([0.0, 0.5, 1.0]), {})
            .name_interface(1, "surface").nondimensionalised(length=1.0))
    relief = AnalyticTopography(lambda t, p: 0.01 * np.cos(t))
    spec = MeshSpec(body=body, sizing=UniformInterfaces(0.15, 0.3, 0.3),
                    order=1, surfaces={"surface": relief},
                    mapping_rule=layer_linear(control_radii=(0.8,)))
    result = build_layered_mesh(spec, tmp_path / "nd")
    assert not result.validation.knots_aligned
    assert any("kink" in w for w in result.validation.warnings)



def test_element_counts_are_what_the_file_holds(tmp_path):
    from planetmodel.mesh3d._writer import read_groups
    body = ReferenceBody.from_fields(Skeleton([0.0, 0.5, 1.0]), {})
    result = build_layered_mesh(
        MeshSpec(body=body, sizing=UniformInterfaces(0.12, 0.25, 0.3), order=1,
                 rref=1.0), tmp_path / "count")
    import gmsh
    with session(name="count"):
        gmsh.open(str(result.msh_path))
        held = sum(len(t) for dim in (2, 3)
                   for t in gmsh.model.mesh.getElements(dim)[1])
        assert read_groups(result.msh_path)[3] == {1: "layer_1", 2: "layer_2"}
    assert result.counts["elements"] == held



def test_relief_centred_at_attachment_still_reports_its_exaggeration(
        tmp_path, write_relief_xyz):
    """Centring must not hide the exaggeration from the manifest.

    Centring wraps the shape in a sum with a scaled constant, and the
    walk that reads the exaggeration reads it from the outermost
    scalings -- which, after centring, is the sum, whose factor is a
    mean radius and not an exaggeration.  So `grid * 2` attached raw
    reached the manifest as exaggeration 1.0, and a Mode B consumer
    rebuilding the relief from the files would have applied half of it.
    Both ways of saying the same thing must report the same factor: the
    raw shape centred on attachment, and the surface centred first.
    """
    from planetmodel.model.surface import Surface
    from planetmodel.model.topography import GriddedTopography

    path = write_relief_xyz(tmp_path / "relief.xyz", offset_km=2.0,
                            amplitude_km=5.0)
    grid = GriddedTopography.from_xyz(path, scale=1.0e3)

    def entry(shape, out):
        spec = MeshSpec(
            body=ReferenceBody.from_fields(Skeleton([0.0, 0.5e6, 1.0e6]), {})
            .name_interface(1, "surface"),
            rref=1.0e6, order=1,
            sizing=UniformInterfaces(0.15e6, 0.30e6, 0.30e6),
            surfaces={"surface": shape}, mapping_rule=layer_linear(),
            delivery="physical")
        result = build_layered_mesh(spec, tmp_path / out)
        (found,) = sc.read(result.manifest_path).mapping["surfaces"]
        return found

    # Raw: the mean of 2 km, doubled, is removed at attachment.
    with pytest.warns(UserWarning, match="area-weighted mean of 4"):
        raw = entry(grid * 2.0, "raw")
    explicit = entry(Surface(1.0e6, topography=grid).centred().at(1.0e6) * 2.0,
                     "explicit")

    for found in (raw, explicit):
        assert found["exaggeration"] == 2.0
        assert [Path(s["file"]) for s in found["sources"]] == [path]
        assert found["sources"][0]["sha256"] == sc.file_digest(path)
        assert found["sources"][0]["scale_to_m"] == 1.0e3

