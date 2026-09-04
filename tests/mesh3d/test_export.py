"""The MFEM export: 2D and 3D, hollow and full, both deliveries, and an
offset mesh, each read back the way the manifest says."""
import dataclasses

import numpy as np
import pytest

from planetmodel.mesh3d import (MeshSpec, Shell, build_layered_mesh,
                                build_offset_mesh, export_mfem_mesh,
                                manifest as sc)
from planetmodel.mesh3d.export import MESH_READ_OPTIONS, _node_array

from conftest import COARSE, confined_flattening, flattening, full_geometry, \
    hollow_geometry

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

#: name -> (dimension, geometry factory, order, shells)
CASES = {
    "full3": (3, lambda: full_geometry(), 2, ()),
    "hollow3": (3, lambda: hollow_geometry().with_mapping(flattening(0.05)), 2, ()),
    "shells3": (3, lambda: full_geometry().with_mapping(confined_flattening(0.05)),
                2, (Shell(ratio=0.2),)),
    "full2": (2, lambda: full_geometry().with_mapping(flattening(0.05)), 2, ()),
    "hollow2": (2, lambda: hollow_geometry(), 1, ()),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every case built referentially, once."""
    root = tmp_path_factory.mktemp("built")
    out = {}
    for name, (dimension, make, order, shells) in CASES.items():
        spec = MeshSpec(make(), COARSE, dimension=dimension, order=order,
                        shells=shells, delivery="referential")
        out[name] = build_layered_mesh(spec, root / name)
    out["offset3"] = build_offset_mesh(root / "offset3", inner_radius=0.4,
                                       outer_radius=1.0, offset=0.3, sizing=COARSE)
    out["offset2"] = build_offset_mesh(root / "offset2", inner_radius=0.4,
                                       outer_radius=1.0, offset=0.3, sizing=COARSE,
                                       dimension=2)
    return out


@pytest.fixture(scope="module")
def exported(built, tmp_path_factory):
    """Both deliveries of every build, written beside each other."""
    root = tmp_path_factory.mktemp("delivery")
    return {(name, d): export_mfem_mesh(res, root / f"{name}_{d}", delivery=d)
            for name, res in built.items() for d in ("physical", "referential")}


def load(export):
    """The mesh a consumer loads, constructed the way the manifest says."""
    o = export.files["mesh_read_options"]
    assert o == MESH_READ_OPTIONS
    return mfem.Mesh(str(export.mesh_path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])


def nodes_of(mesh):
    return np.array(_node_array(mesh.GetNodes()), dtype=float)


NAMES = list(CASES) + ["offset3", "offset2"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("delivery", ["physical", "referential"])
def test_mfem_reads_every_export_back_clean(built, exported, name, delivery):
    export = exported[(name, delivery)]
    res = built[name]
    assert export.delivery == delivery
    assert export.mesh_path.exists() and export.manifest_path.exists()
    mesh = load(export)
    assert mesh.CheckElementOrientation(False) == 0
    assert mesh.CheckBdrElementOrientation(False) == 0
    card = sc.read(export.manifest_path)
    assert mesh.Dimension() == card.mesh["dimension"]
    assert mesh.SpaceDimension() == card.mesh["dimension"]
    assert list(mesh.attributes.ToList()) == list(range(1, len(card.layers) + 1))
    assert list(mesh.bdr_attributes.ToList()) == \
        list(range(1, len(card.interfaces) + 1))
    assert mesh.GetNE() + mesh.GetNBE() == card.mesh["n_elements"]
    assert export.counts["elements"] == mesh.GetNE()
    assert export.counts["nodes"] == mesh.GetNodes().FESpace().GetNDofs()
    assert card.delivery == delivery
    assert card.files["mesh"] == export.mesh_path.name
    identity = res.mapping is None or res.mapping.is_identity
    assert card.mapping["applied_to_nodes"] is (delivery == "physical"
                                                and not identity)
    if delivery == "referential":
        assert export.displacement_path.exists()
        (entry,) = card.files["grid_functions"]
        assert entry["kind"] == "displacement" and entry["name"] == "displacement"
        assert entry["file"] == export.displacement_path.name
        assert entry["vdim"] == mesh.SpaceDimension()
        assert entry["fe_space"] == mesh.GetNodes().FESpace().FEColl().Name()
        gf = mfem.GridFunction(mesh, str(export.displacement_path))
        assert gf.Size() == mesh.GetNodes().Size()
    else:
        assert export.displacement_path is None
        assert card.files["grid_functions"] == []


@pytest.mark.parametrize("name", NAMES)
def test_the_physical_mesh_is_the_reference_mesh_plus_the_displacement(
        exported, name):
    ref = exported[(name, "referential")]
    phys = exported[(name, "physical")]
    mesh = load(ref)
    X = nodes_of(mesh)
    gf = mfem.GridFunction(mesh, str(ref.displacement_path))
    u = np.array(_node_array(gf), dtype=float, copy=True)
    x = nodes_of(load(phys))
    assert x.shape == X.shape == u.shape
    assert np.abs(x - (X + u)).max() < 1e-12


def test_the_displacement_is_the_mappings_own(built, exported):
    """On the surface of the flattened hollow ball, r = 1 + h(theta)."""
    export = exported[("hollow3", "physical")]
    mesh = load(export)
    surface = sc.read(export.manifest_path).interface_attribute("outer")
    xyz = np.asarray(mesh.GetVertexArray())
    on = sorted({v for i in range(mesh.GetNBE())
                 if mesh.GetBdrAttribute(i) == surface
                 for v in mesh.GetBdrElementVertices(i)})
    r = np.linalg.norm(xyz[on], axis=1)
    cos_t = xyz[on][:, 2] / r
    want = 1.0 - 0.05 * 0.5 * (3.0 * cos_t ** 2 - 1.0)
    assert np.abs(r - want).max() < 1e-9
    assert r.max() - r.min() > 0.05


def test_identity_and_offset_exports_move_nothing(exported):
    for name in ("full3", "hollow2", "offset3", "offset2"):
        ref = exported[(name, "referential")]
        mesh = load(ref)
        gf = mfem.GridFunction(mesh, str(ref.displacement_path))
        assert np.abs(np.asarray(_node_array(gf))).max() == 0.0
        assert np.array_equal(nodes_of(load(exported[(name, "physical")])),
                              nodes_of(mesh))


def test_two_dimensions_lift_and_drop_the_third_coordinate(exported):
    export = exported[("full2", "physical")]
    mesh = load(export)
    assert mesh.SpaceDimension() == 2
    x = nodes_of(mesh)
    assert x.shape[1] == 2
    # in the plane the flattening is a uniform 2.5 % expansion
    assert np.linalg.norm(x, axis=1).max() == pytest.approx(1.025, abs=1e-9)


def test_a_straight_sided_mesh_exports_too(exported):
    """Order 1: MFEM gives the mesh no nodal space until asked."""
    export = exported[("hollow2", "referential")]
    assert export.counts["order"] == 1
    mesh = load(export)
    assert mesh.GetNodes() is not None
    (entry,) = export.files["grid_functions"]
    assert entry["fe_space"] == "H1_2D_P1"


def test_a_mesh_already_displaced_is_refused(built, tmp_path):
    res = built["full2"]
    card = sc.read(res.manifest_path)
    card.mapping["applied_to_nodes"] = True
    moved = dataclasses.replace(res, manifest_path=sc.write(tmp_path / "moved", card))
    with pytest.raises(ValueError, match="physical mesh"):
        export_mfem_mesh(moved, tmp_path / "nope")
    with pytest.raises(ValueError, match="delivery must be"):
        export_mfem_mesh(res, tmp_path / "bad", delivery="halfway")


def test_the_default_delivery_is_the_builds(built, tmp_path):
    export = export_mfem_mesh(built["hollow2"], tmp_path / "default")
    assert export.delivery == "referential"
    assert export.displacement_path is not None


def test_a_physically_built_mesh_is_refused(tmp_path):
    spec = MeshSpec(full_geometry().with_mapping(flattening(0.05)), COARSE,
                    dimension=2, delivery="physical")
    res = build_layered_mesh(spec, tmp_path / "phys")
    with pytest.raises(ValueError, match="physical mesh"):
        export_mfem_mesh(res, tmp_path / "out")
