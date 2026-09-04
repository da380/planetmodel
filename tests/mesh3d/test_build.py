"""The assembly line: MeshSpec in, checked mesh and manifest out.

Every geometry here is unit sized and a few elements across."""
import json

import numpy as np
import pytest

import gmsh

from planetmodel import Geometry, ScaledMapping, Skeleton
from planetmodel.mesh3d import (InterfaceSizing, MeshSpec, Shell,
                                UniformInterfaces, build_layered_mesh,
                                manifest as sc)
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._writer import read_groups
from planetmodel.mesh3d.layered import require_mapping_on_shells

from conftest import COARSE, confined_flattening, flattening, full_geometry, \
    hollow_geometry

pytestmark = pytest.mark.gmsh


def node_radii(msh_path):
    """The radii of every node in a written mesh."""
    with session(name="nodes"):
        gmsh.merge(str(msh_path))
        _, coords, _ = gmsh.model.mesh.getNodes()
    return np.linalg.norm(coords.reshape(-1, 3), axis=1)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """Build once per (dimension, hollow, shells) with the identity mapping."""
    root = tmp_path_factory.mktemp("identity")
    cache = {}

    def _build(dimension, hollow, shells):
        key = (dimension, hollow, shells)
        if key not in cache:
            g = hollow_geometry() if hollow else full_geometry()
            spec = MeshSpec(g, COARSE, dimension=dimension, order=2,
                            shells=[Shell(ratio=0.25)] if shells else ())
            name = f"{dimension}d_{'hollow' if hollow else 'full'}_{int(shells)}"
            cache[key] = build_layered_mesh(spec, root / name)
        return cache[key]

    return _build


CASES = [(d, h, s) for d in (2, 3) for h in (False, True) for s in (False, True)]


@pytest.mark.parametrize("dimension,hollow,shells", CASES)
def test_identity_builds_produce_a_checked_mesh_and_manifest(build, dimension,
                                                             hollow, shells):
    res = build(dimension, hollow, shells)
    assert res.msh_path.exists() and res.manifest_path.exists()
    v = res.validation
    assert v.ok, v.failures
    assert v.negative_jacobians == 0 and v.negative_cells == 0
    assert v.inward_faces == 0 and v.min_sicn > 0.0
    assert v.max_interface_radius_error < 1e-3
    n_layers = (2 if hollow else 3) + int(shells)
    n_faces = n_layers + int(hollow)
    assert res.counts["layers"] == n_layers
    assert res.counts["interfaces"] == n_faces
    assert res.counts["elements"] > 0
    assert set(res.timings) >= {"resolve", "geometry", "mesh", "orient",
                                "validate", "write"}
    assert res.divisor == pytest.approx(1.25 if shells else 1.0)
    assert res.mapping.is_identity and res.spec.dimension == dimension
    assert res.geometry is res.spec.geometry


@pytest.mark.parametrize("dimension,hollow,shells", CASES)
def test_the_manifest_describes_the_domain(build, dimension, hollow, shells):
    res = build(dimension, hollow, shells)
    card = sc.read(res.manifest_path)
    assert card.schema == sc.SCHEMA and card.delivery == "physical"
    sc.validate_against(card, layer_count=res.counts["layers"],
                        interface_count=res.counts["interfaces"])
    assert card.mesh["dimension"] == dimension and card.mesh["element_order"] == 2
    assert card.mesh["n_elements"] == res.counts["elements"]
    assert card.geometry["divisor"] == res.divisor
    assert card.geometry["outer_radius_nd"] == pytest.approx(1.0)
    assert card.geometry["inner_radius_nd"] == pytest.approx(
        (0.5 if hollow else 0.0) / res.divisor)
    assert card.geometry["n_layers"] == res.counts["layers"]
    flags = [lay["in_geometry"] for lay in card.layers]
    assert flags == [True] * (2 if hollow else 3) + [False] * int(shells)
    assert card.shell_attributes == ((res.counts["layers"],) if shells else ())
    names = [lay["name"] for lay in card.layers]
    assert names[:2] == (["lower", "upper"] if hollow else ["core", "mantle"])
    if shells:
        assert names[-1] == "buffer"
        assert card.interfaces[-1]["name"] == f"interface_{res.counts['interfaces']}"
    between = [f["between_layers"] for f in card.interfaces]
    n = res.counts["layers"]
    first = 1 if hollow else 0
    assert between == [[k - first, k + 1 - first if k + 1 - first < n else -1]
                       for k in range(len(between))]
    assert card.mapping["kind"] == "IdentityMapping"
    assert card.mapping["applied_to_nodes"] is False
    assert card.mapping["knots_nd"] == []
    assert card.sizing["policy"] == "UniformInterfaces"
    assert card.provenance["perturbation"] is None
    assert card.provenance["mesh_file"] == res.msh_path.name
    assert card.files is None


def test_the_mesh_is_msh_2_2_with_names_intact(build):
    res = build(3, False, True)
    header = res.msh_path.read_text(errors="ignore").splitlines()[:2]
    assert header[1].split()[0] == "2.2"
    with session(name="reread"):
        groups = read_groups(res.msh_path)
    assert groups[3] == {1: "core", 2: "mantle", 3: "crust", 4: "buffer"}
    assert groups[2] == {1: "cmb", 2: "moho", 3: "surface", 4: "interface_4"}


def test_a_hollow_mesh_has_no_nodes_inside_the_hole(build):
    r = node_radii(build(3, True, False).msh_path)
    assert r.min() == pytest.approx(0.5, abs=1e-6)
    r = node_radii(build(2, True, True).msh_path)
    assert r.min() == pytest.approx(0.5 / 1.25, abs=1e-6)
    assert r.max() == pytest.approx(1.0, abs=1e-6)


def test_unnamed_layers_get_default_names(tmp_path):
    g = Geometry(Skeleton([0.0, 0.6, 1.0]))
    res = build_layered_mesh(MeshSpec(g, COARSE, dimension=2), tmp_path / "plain")
    with session(name="reread"):
        groups = read_groups(res.msh_path)
    assert groups[2] == {1: "layer_1", 2: "layer_2"}
    assert groups[1] == {1: "interface_1", 2: "interface_2"}
    card = sc.read(res.manifest_path)
    assert [lay["name"] for lay in card.layers] == ["layer_1", "layer_2"]


# ------------------------------------------------------------ the mapping

@pytest.fixture(scope="module")
def deformed(tmp_path_factory):
    """The flattened geometry, built physically and referentially in 3D."""
    root = tmp_path_factory.mktemp("deformed")
    g = full_geometry().with_mapping(flattening(0.05))
    return {d: build_layered_mesh(MeshSpec(g, COARSE, delivery=d), root / d)
            for d in ("physical", "referential")}


def test_the_physical_delivery_moves_the_nodes(deformed):
    res = deformed["physical"]
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    assert card.delivery == "physical"
    assert card.mapping["kind"] == "RadialStretch"
    assert "flattening" in card.mapping["repr"]
    assert card.mapping["applied_to_nodes"] is True
    p = card.provenance["perturbation"]
    assert p["nodes"] == res.counts["nodes"]
    assert p["max_displacement_nd"] == pytest.approx(0.05, rel=0.05)
    assert p["validity_margin"] > 0.0
    assert "perturb" in res.timings
    # the surface nodes carry the flattening: radii spread over 1 -+ 0.05
    r = node_radii(res.msh_path)
    assert r.max() == pytest.approx(1.025, abs=2e-3)
    assert r.min() == pytest.approx(0.0, abs=1e-12)
    # the interfaces were checked at their reference radii
    assert res.validation.max_interface_radius_error < 1e-3


def test_the_referential_delivery_leaves_the_nodes_alone(deformed):
    res = deformed["referential"]
    card = sc.read(res.manifest_path)
    assert card.delivery == "referential"
    assert card.mapping["applied_to_nodes"] is False
    assert card.provenance["perturbation"] is None
    assert node_radii(res.msh_path).max() == pytest.approx(1.0, abs=1e-6)
    assert isinstance(res.mapping, ScaledMapping) and res.mapping.k == 1.0


def test_two_dimensions_take_the_same_mapping(tmp_path):
    g = full_geometry().with_mapping(flattening(0.05))
    res = build_layered_mesh(MeshSpec(g, COARSE, dimension=2), tmp_path / "disc")
    assert res.validation.ok
    # in the plane theta = pi/2, P2 = -1/2, so every node moves out by 2.5 %
    r = node_radii(res.msh_path)
    assert r.max() == pytest.approx(1.025, abs=1e-6)


def test_a_mapping_that_moves_the_outer_boundary_is_refused_with_shells(tmp_path):
    g = full_geometry().with_mapping(flattening(0.05))
    spec = MeshSpec(g, COARSE, shells=[Shell(ratio=0.2)])
    path = tmp_path / "refused"
    with pytest.raises(ValueError, match="identity on the outer boundary"):
        build_layered_mesh(spec, path)
    assert not path.with_suffix(".msh").exists()
    assert not path.with_suffix(".json").exists()
    with pytest.raises(ValueError, match="identity on the outer boundary"):
        require_mapping_on_shells(spec)


def test_a_mapping_undefined_on_the_shells_is_refused(tmp_path):
    """The mapping folds beyond the geometry: valid inside, invalid on the shell."""
    def h(r, theta, phi):
        return -0.05 * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0) * (
            1.0 + 40.0 * np.maximum(r - 1.0, 0.0))
    from planetmodel import CallableDisplacement, RadialStretch
    m = RadialStretch(CallableDisplacement(h, knots=[1.0]), rmax=1.0)
    g = full_geometry().with_mapping(m)
    with pytest.raises(ValueError, match="whole computational domain"):
        build_layered_mesh(MeshSpec(g, COARSE, shells=[Shell(ratio=0.5)]),
                           tmp_path / "folded")


def test_a_displacement_confined_to_the_geometry_is_accepted_with_shells(tmp_path):
    g = full_geometry().with_mapping(confined_flattening(0.05))
    spec = MeshSpec(g, COARSE, shells=[Shell(ratio=0.2)], delivery="physical")
    res = build_layered_mesh(spec, tmp_path / "confined")
    assert res.validation.ok
    assert res.divisor == pytest.approx(1.2)
    card = sc.read(res.manifest_path)
    assert card.mapping["applied_to_nodes"] is True
    assert card.mapping["knots_nd"] == pytest.approx([0.8 / 1.2, 1.0 / 1.2])
    # the outer boundary of the shell stays a sphere of mesh radius one
    r = node_radii(res.msh_path)
    assert r.max() == pytest.approx(1.0, abs=1e-9)
    # the displacement is largest at the Moho's poles: 0.05 * 0.8, in mesh units
    p = card.provenance["perturbation"]
    assert p["max_displacement_nd"] == pytest.approx(0.04 / 1.2, rel=0.05)


# --------------------------------------------------------------- divisor

def test_the_divisor_scales_everything_handed_to_gmsh(tmp_path):
    g = full_geometry().with_mapping(flattening(0.05))
    spec = MeshSpec(g, UniformInterfaces(0.15, 0.3, 0.3), dimension=2, divisor=2.0,
                    delivery="referential")
    res = build_layered_mesh(spec, tmp_path / "half")
    assert res.divisor == 2.0
    card = sc.read(res.manifest_path)
    assert card.geometry["divisor"] == 2.0
    assert card.geometry["outer_radius_nd"] == pytest.approx(0.5)
    assert [lay["r_outer_nd"] for lay in card.layers] == pytest.approx([0.2, 0.4, 0.5])
    assert card.sizing["per_interface"][0]["size_nd"] == pytest.approx(0.075)
    assert node_radii(res.msh_path).max() == pytest.approx(0.5, abs=1e-6)
    # the result's mapping is the geometry's conjugated by the divisor
    X = np.array([[0.3, 0.1, 0.2]])
    assert np.allclose(res.mapping(X), g.mapping(2.0 * X) / 2.0)
    assert res.mapping.k == pytest.approx(0.5)


def test_a_divisor_far_from_order_one_warns(tmp_path):
    spec = MeshSpec(full_geometry(), COARSE, dimension=2, divisor=1e-5)
    with pytest.warns(UserWarning, match="outside"):
        res = build_layered_mesh(spec, tmp_path / "huge")
    assert sc.read(res.manifest_path).geometry["outer_radius_nd"] == \
        pytest.approx(1e5)


# ---------------------------------------------------------------- guards

def test_coarse_sizing_for_a_thin_layer_is_refused_before_meshing(tmp_path):
    g = Geometry(Skeleton([0.0, 0.5, 0.99, 1.0]))
    with pytest.raises(ValueError, match="too coarse"):
        build_layered_mesh(MeshSpec(g, COARSE), tmp_path / "thin")
    assert not list(tmp_path.iterdir())


def test_sizing_at_the_wrong_scale_is_refused(tmp_path):
    tiny = UniformInterfaces(1e-7, 1e-6, 1e-6)
    with pytest.raises(ValueError, match="not a resolution choice"):
        build_layered_mesh(MeshSpec(full_geometry(), tiny), tmp_path / "tiny")


def test_meta_and_a_function_rule_reach_the_manifest(tmp_path):
    def coarse(interfaces, outer_radius):
        return {f.index: InterfaceSizing(0.15, 0.3, 0.3) for f in interfaces}

    spec = MeshSpec(full_geometry(), coarse, dimension=2, order=1,
                    meta={"run": "a", "seed": 3})
    res = build_layered_mesh(spec, tmp_path / "meta")
    data = json.loads(res.manifest_path.read_text())
    assert data["provenance"]["meta"] == {"run": "a", "seed": 3}
    assert data["sizing"]["policy"] == "coarse"
    assert data["mesh"]["element_order"] == 1
    assert data["mesh"]["high_order_optimised"] is False


@pytest.mark.parametrize("order", [1, 3])
def test_other_element_orders_build(order, tmp_path):
    spec = MeshSpec(hollow_geometry(), COARSE, dimension=2, order=order)
    res = build_layered_mesh(spec, tmp_path / f"o{order}")
    assert res.validation.ok
    assert sc.read(res.manifest_path).mesh["element_order"] == order
