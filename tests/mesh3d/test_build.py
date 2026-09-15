"""The assembly line: MeshSpec in, checked mesh and manifest out.

Every geometry here is unit sized and a few elements across, but for
the one that checks an Earth-sized geometry is meshed in its own
numbers."""
import json

import numpy as np
import pytest

import gmsh

from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (InterfaceSizing, MeshSpec, Shell,
                                UniformInterfaces, build_layered_mesh,
                                manifest as sc)
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._writer import read_groups
from planetmodel.mesh3d.layered import (require_mapping_on_shells,
                                        require_spherical_outer_boundary)

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
    assert res.mapping is res.geometry.mapping and res.mapping.is_identity
    assert res.spec.dimension == dimension
    assert res.geometry is res.spec.geometry


@pytest.mark.parametrize("dimension,hollow,shells", CASES)
def test_the_manifest_describes_the_domain(build, dimension, hollow, shells):
    res = build(dimension, hollow, shells)
    card = sc.read(res.manifest_path)
    assert card.schema == sc.SCHEMA
    sc.validate_against(card, layer_count=res.counts["layers"],
                        interface_count=res.counts["interfaces"])
    assert card.mesh == {"file": res.msh_path.name, "format": "msh",
                         "nodes": "reference", "read_options": {},
                         "displacement": None}
    assert card.outer_radius == pytest.approx(1.25 if shells else 1.0)
    assert card.inner_radius == pytest.approx(0.5 if hollow else 0.0)
    assert len(card.layers) == res.counts["layers"]
    assert len(card.shell_attributes) == int(shells)
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
    assert card.fields == [] and card.scales is None and card.constants == {}
    assert card.meta == {}


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
    assert r.min() == pytest.approx(0.5, abs=1e-6)
    assert r.max() == pytest.approx(1.25, abs=1e-6)


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

def test_the_mesher_leaves_the_nodes_on_the_reference_spheres(tmp_path):
    """The mapping is carried on the result, never applied by gmsh."""
    g = full_geometry().with_mapping(flattening(0.05))
    res = build_layered_mesh(MeshSpec(g, COARSE), tmp_path / "deformed")
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    assert card.mesh["nodes"] == "reference"
    assert "perturb" not in res.timings
    assert node_radii(res.msh_path).max() == pytest.approx(1.0, abs=1e-6)
    assert res.validation.max_interface_radius_error < 1e-3
    assert res.mapping is res.geometry.mapping


def test_two_dimensions_carry_the_same_mapping(tmp_path):
    g = full_geometry().with_mapping(flattening(0.05))
    res = build_layered_mesh(MeshSpec(g, COARSE, dimension=2), tmp_path / "disc")
    assert res.validation.ok
    assert res.mapping is g.mapping
    assert node_radii(res.msh_path).max() == pytest.approx(1.0, abs=1e-6)


def test_a_mapping_may_move_the_outer_boundary_of_the_shells(tmp_path):
    """The outer boundary carries topography unless a spherical one is asked for."""
    g = full_geometry().with_mapping(flattening(0.05))
    spec = MeshSpec(g, COARSE, shells=[Shell(ratio=0.2)])
    require_mapping_on_shells(spec)
    res = build_layered_mesh(spec, tmp_path / "free")
    assert res.validation.ok
    assert sc.read(res.manifest_path).outer_radius == pytest.approx(1.2)


def test_a_spherical_outer_boundary_refuses_a_mapping_that_moves_it(tmp_path):
    g = full_geometry().with_mapping(flattening(0.05))
    spec = MeshSpec(g, COARSE, shells=[Shell(ratio=0.2)], outer_boundary="spherical")
    path = tmp_path / "refused"
    with pytest.raises(ValueError, match="identity on the outer boundary"):
        build_layered_mesh(spec, path)
    assert not path.with_suffix(".msh").exists()
    assert not path.with_suffix(".json").exists()
    with pytest.raises(ValueError, match="identity on the outer boundary"):
        require_spherical_outer_boundary(spec)
    # the rule is about the domain's outer boundary, shells or not
    with pytest.raises(ValueError, match="identity on the outer boundary"):
        build_layered_mesh(MeshSpec(g, COARSE, outer_boundary="spherical"),
                           tmp_path / "refused_too")


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
    spec = MeshSpec(g, COARSE, shells=[Shell(ratio=0.2)],
                    outer_boundary="spherical")
    res = build_layered_mesh(spec, tmp_path / "confined")
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    assert card.outer_radius == pytest.approx(1.2)
    assert card.mesh["nodes"] == "reference"


# ------------------------------------------------- the geometry's numbers

def test_the_geometry_is_meshed_in_its_own_numbers(tmp_path):
    """An Earth-sized geometry reaches gmsh, the manifest and the mesh
    file in metres: nothing is normalised."""
    a = 6.371e6
    g = Geometry(Skeleton([0.0, 3.48e6, 5.7e6, a]),
                 layer_names=["core", "mantle", "crust"],
                 interface_names=["cmb", "moho", "surface"])
    spec = MeshSpec(g, UniformInterfaces(1.5e6, 3e6, 3e6), dimension=2, order=2)
    res = build_layered_mesh(spec, tmp_path / "earth")
    assert res.validation.ok, res.validation.failures
    assert res.mapping is g.mapping
    card = sc.read(res.manifest_path)
    assert card.outer_radius == a
    assert card.inner_radius == 0.0
    assert [lay["r_outer"] for lay in card.layers] == [3.48e6, 5.7e6, a]
    assert [f["radius"] for f in card.interfaces] == [3.48e6, 5.7e6, a]
    assert res.validation.max_interface_radius_error < 1e-3 * a
    r = node_radii(res.msh_path)
    assert r.max() == pytest.approx(a, rel=1e-9)
    assert r.min() < 1.5e6                      # the core is meshed, not a hole
    # the nodes of the Moho sit on a circle of radius 5.7e6, in metres
    with session(name="moho"):
        gmsh.merge(str(res.msh_path))
        (curve,) = gmsh.model.getEntitiesForPhysicalGroup(1, 2)
        _, coords, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
    moho = np.linalg.norm(coords.reshape(-1, 3), axis=1)
    assert np.allclose(moho, 5.7e6, rtol=1e-9)
    # element sizes are in the same numbers: a coarse mesh, a few across
    assert 10 < res.counts["elements"] < 400


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


def test_meta_reaches_the_manifest_and_a_function_rule_builds(tmp_path):
    def coarse(interfaces, outer_radius):
        return {f.index: InterfaceSizing(0.15, 0.3, 0.3) for f in interfaces}

    spec = MeshSpec(full_geometry(), coarse, dimension=2, order=1,
                    meta={"run": "a", "seed": 3})
    res = build_layered_mesh(spec, tmp_path / "meta")
    data = json.loads(res.manifest_path.read_text())
    assert data["meta"] == {"run": "a", "seed": 3}
    assert res.validation.ok


@pytest.mark.parametrize("order", [1, 3])
def test_other_element_orders_build(order, tmp_path):
    spec = MeshSpec(hollow_geometry(), COARSE, dimension=2, order=order)
    res = build_layered_mesh(spec, tmp_path / f"o{order}")
    assert res.validation.ok
    assert res.spec.order == order
