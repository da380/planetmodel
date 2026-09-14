"""The fields of a model to MFEM: read back at the dof coordinates in 2D
and 3D, full and hollow, with a shell outside the model; the model block
of the manifest; and what is refused."""
import copy

import numpy as np
import pytest

from planetmodel.character import DENSITY, ELASTIC
from planetmodel.fields import AnalyticField, constant_field
from planetmodel.model import Model
from planetmodel.catalogue import LayeredIsotropicElastic
from planetmodel.units import G_SI, Scales
from planetmodel.mesh3d import (MeshSpec, Shell, build_layered_mesh,
                                build_offset_mesh, export_mfem,
                                manifest as sc)
from planetmodel.mesh3d.export import MESH_READ_OPTIONS, _node_array

from conftest import COARSE, confined_flattening, flattening, full_geometry, \
    hollow_geometry

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

#: name -> (dimension, geometry factory, order, shells)
CASES = {
    "full3": (3, lambda: full_geometry(), 2, ()),
    "shells3": (3, lambda: full_geometry().with_mapping(confined_flattening(0.05)),
                2, (Shell(ratio=0.2),)),
    "full2": (2, lambda: full_geometry().with_mapping(flattening(0.05)), 2, ()),
    "hollow2": (2, lambda: hollow_geometry(), 1, ()),
}


def model_on(geometry) -> Model:
    """A model on `geometry` whose fields vary with radius and direction.

    The density is a formula of all three coordinates; the elastic
    tensor is the identity in the spherical frame on every layer but
    the first, so the Cartesian components vary with direction and the
    first layer lacks the field; `foo` is a name with no spec on the
    last layer only.
    """
    layers = []
    for lay in geometry.layers:
        iv = lay.interval
        fields = {"rho": AnalyticField(
            iv, lambda r, t, p: 1.0 + r * r + 0.1 * np.sin(t) * np.cos(p)
            + 0.05 * np.cos(t), character=DENSITY, name="rho")}
        if lay.index > 0:
            fields["elastic_moduli"] = constant_field(
                (lay.index + 1.0) * np.eye(6), iv, character=ELASTIC,
                name="elastic_moduli")
        if lay.index == geometry.nlayers - 1:
            fields["foo"] = constant_field(2.0, iv, name="foo")
        layers.append(fields)
    return Model(geometry, layers)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every case built once, with its model."""
    root = tmp_path_factory.mktemp("built")
    out = {}
    for name, (dimension, make, order, shells) in CASES.items():
        geometry = make()
        spec = MeshSpec(geometry, COARSE, dimension=dimension, order=order,
                        shells=shells)
        out[name] = (build_layered_mesh(spec, root / name), model_on(geometry))
    return out


@pytest.fixture(scope="module")
def exported(built, tmp_path_factory):
    """Every case exported referentially with every field."""
    root = tmp_path_factory.mktemp("fields")
    return {name: export_mfem(res, root / name, model=model,
                              delivery="referential")
            for name, (res, model) in built.items()}


def load(export):
    """The mesh a consumer loads, constructed the way the manifest says."""
    o = export.read_options
    assert o == MESH_READ_OPTIONS
    return mfem.Mesh(str(export.mesh_path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])


def read_back(mesh, path) -> np.ndarray:
    """A GridFunction's values as (ndof, vdim), copied while it is alive."""
    gf = mfem.GridFunction(mesh, str(path))
    return np.array(_node_array(gf), dtype=float, copy=True)


def dof_points(mesh, fes) -> np.ndarray:
    """Where every dof of `fes` sits, through MFEM's element transformations."""
    X = np.zeros((fes.GetNDofs(), mesh.SpaceDimension()))
    for e in range(mesh.GetNE()):
        pts = mfem.DenseMatrix()
        mesh.GetElementTransformation(e).Transform(fes.GetFE(e).GetNodes(), pts)
        X[np.asarray(fes.GetElementDofs(e), dtype=int)] = pts.GetDataArray().T
    return X


def dof_attributes(mesh, fes) -> np.ndarray:
    """The attribute of the element each dof of `fes` belongs to."""
    a = np.zeros(fes.GetNDofs(), dtype=int)
    for e in range(mesh.GetNE()):
        a[np.asarray(fes.GetElementDofs(e), dtype=int)] = mesh.GetAttribute(e)
    return a


def clipped(X, interval) -> np.ndarray:
    """The export's rule: a point pulled radially into its layer."""
    r = np.linalg.norm(X, axis=-1)
    return X * (np.clip(r, *interval) / r)[:, None]


def expected(model, name, X, attributes, card) -> np.ndarray:
    """The field at the (lifted) dof points, zero off the layers holding it."""
    X3 = np.zeros((X.shape[0], 3))
    X3[:, :X.shape[1]] = X
    out = None
    for entry in card.layers:
        dofs = np.flatnonzero(attributes == entry["attribute"])
        i = entry["attribute"] - 1
        if not entry["in_geometry"] or name not in model.layer(i):
            continue
        layer = model.layer(i)
        got = layer[name].evaluate_at(clipped(X3[dofs], layer.interval),
                                      frame="cartesian")
        got = got.reshape(dofs.size, -1)
        if out is None:
            out = np.zeros((X.shape[0], got.shape[1]))
        out[dofs] = got
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_every_field_reads_back_at_the_dof_coordinates(built, exported, name):
    res, model = built[name]
    export = exported[name]
    mesh = load(export)
    card = sc.read(export.manifest_path)
    assert set(export.field_paths) == set(model.field_names()) == \
        {"rho", "elastic_moduli", "foo"}
    entries = {e["name"]: e for e in card.fields if e["name"] != "displacement"}
    assert set(entries) == set(export.field_paths)
    for field_name, path in export.field_paths.items():
        assert path.exists() and entries[field_name]["file"] == path.name
        gf = mfem.GridFunction(mesh, str(path))
        fes = gf.FESpace()
        assert fes.FEColl().Name() == entries[field_name]["fe_space"] == \
            f"L2_{mesh.Dimension()}D_P{export.counts['order']}"
        assert fes.GetVDim() == entries[field_name]["vdim"]
        assert entries[field_name]["ordering"] == "byNODES"
        values = np.array(_node_array(gf), dtype=float)
        scalar = mfem.FiniteElementSpace(mesh, fes.FEColl(), 1)
        X = dof_points(mesh, scalar)
        attributes = dof_attributes(mesh, scalar)
        want = expected(model, field_name, X, attributes, card)
        assert values.shape == want.shape
        assert np.abs(values - want).max() < 1e-12 * max(1.0, np.abs(want).max())
    # the shell and the core carry zeros, the rest does not
    rho = read_back(mesh, export.field_paths["rho"])
    elastic = read_back(mesh, export.field_paths["elastic_moduli"])
    scalar = mfem.FiniteElementSpace(
        mesh, mfem.L2_FECollection(export.counts["order"], mesh.Dimension()), 1)
    attributes = dof_attributes(mesh, scalar)
    for a in card.shell_attributes:
        assert np.all(rho[attributes == a] == 0.0)
    assert np.all(elastic[attributes == 1] == 0.0)
    assert np.all(np.any(elastic[attributes == 2] != 0.0, axis=1))
    assert np.all(rho[attributes <= model.nlayers] > 0.5)
    assert elastic.shape[1] == 36


def test_the_elastic_field_is_not_constant_in_the_cartesian_frame(exported):
    export = exported["full3"]
    mesh = load(export)
    C = read_back(mesh, export.field_paths["elastic_moduli"]).reshape(-1, 6, 6)
    C = C[np.abs(C).sum(axis=(1, 2)) > 0.0]
    assert np.abs(C - C[0]).max() > 0.1
    assert np.abs(C - np.swapaxes(C, 1, 2)).max() < 1e-12


def test_a_physical_delivery_writes_the_same_referential_values(
        built, exported, tmp_path):
    res, model = built["full2"]
    phys = export_mfem(res, tmp_path / "phys", model=model, delivery="physical")
    assert phys.delivery == "physical" and phys.displacement_path is None
    ref = exported["full2"]
    for name in ref.field_paths:
        assert np.array_equal(read_back(load(ref), ref.field_paths[name]),
                              read_back(load(phys), phys.field_paths[name]))
    # the nodes moved all the same
    assert sc.read(phys.manifest_path).mesh["nodes"] == "physical"


def test_the_fields_scales_and_constants_round_trip(exported, built):
    res, model = built["shells3"]
    export = exported["shells3"]
    card = sc.read(export.manifest_path)
    sc.validate_structure(card)
    assert card.schema == sc.SCHEMA == "planetmodel.mesh.manifest/4"
    assert card.scales == {"length": 1.0, "mass": 1.0, "time": 1.0}
    assert card.constants == {"G": G_SI}
    assert [e["name"] for e in card.fields] == \
        ["displacement"] + list(model.field_names())
    rho = card.field_record("rho")
    assert rho == {"name": "rho", "file": export.field_paths["rho"].name,
                   "fe_space": "L2_3D_P2", "vdim": 1, "ordering": "byNODES",
                   "rank": 0, "weight": 1, "voigt": False, "unit": "kg m-3",
                   "layers": [1, 2, 3]}
    elastic = card.field_record("elastic_moduli")
    assert (elastic["rank"], elastic["weight"], elastic["voigt"]) == (4, 1, True)
    assert elastic["unit"] == "kg m-1 s-2" and elastic["layers"] == [2, 3]
    assert elastic["vdim"] == 36
    foo = card.field_record("foo")
    assert foo["unit"] == "unknown" and foo["layers"] == [3]
    # the displacement is a field like the others, on every layer, in metres
    u = card.field_record("displacement")
    assert card.mesh["displacement"] == "displacement"
    assert u["file"] == export.displacement_path.name
    assert (u["rank"], u["weight"], u["voigt"]) == (1, 0, False)
    assert u["unit"] == "m" and u["layers"] == [1, 2, 3, 4]
    assert card.shell_attributes == (4,)
    text = card.describe()
    assert "scales      length 1, mass 1, time 1, constants G " in text
    assert "rank 0 weight 1        kg m-3      layers [1, 2, 3]" in text
    assert "rank 4 weight 1 Voigt  kg m-1 s-2  layers [2, 3]" in text
    assert repr(export) == "ExportResult(shells3.mesh, 3 fields, referential delivery)"
    # a broken record is named
    bad = copy.deepcopy(card)
    bad.fields[1]["layers"] = [0]
    with pytest.raises(ValueError, match=r"fields\[1\].layers"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.constants["G"] = "big"
    with pytest.raises(ValueError, match="constants.G"):
        sc.validate_structure(bad)


def test_fields_are_chosen_by_name_and_the_order_can_differ(built, tmp_path):
    res, model = built["hollow2"]
    export = export_mfem(res, tmp_path / "rho", model=model, fields=["rho"],
                         order=0)
    assert list(export.field_paths) == ["rho"]
    assert [e["name"] for e in sc.read(export.manifest_path).fields] == ["rho"]
    mesh = load(export)
    gf = mfem.GridFunction(mesh, str(export.field_paths["rho"]))
    assert gf.FESpace().FEColl().Name() == "L2_2D_P0"
    assert gf.Size() == mesh.GetNE()
    with pytest.raises(KeyError, match="nope"):
        export_mfem(res, tmp_path / "bad", model=model, fields=["rho", "nope"])


def test_a_model_in_other_scales_records_them(built, tmp_path):
    res, _ = built["hollow2"]
    model = LayeredIsotropicElastic([0.5, 0.8, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                    vs=[1.0, 0.0],
                                    scales=Scales(length=1.0, mass=2.0, time=0.5))
    export = export_mfem(res, tmp_path / "scaled", model=model)
    card = sc.read(export.manifest_path)
    assert card.scales == {"length": 1.0, "mass": 2.0, "time": 0.5}
    assert card.constants["G"] == pytest.approx(G_SI / (0.5 * 4.0))
    assert {e["unit"] for e in card.fields} == {"1"}
    mesh = load(export)
    assert set(np.unique(read_back(mesh, export.field_paths["rho"]))) == {1.0, 2.0}


def test_a_model_on_another_skeleton_is_refused(built, tmp_path):
    res, _ = built["full3"]
    other = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                    vs=[1.0, 0.0])
    with pytest.raises(ValueError, match="skeleton"):
        export_mfem(res, tmp_path / "other", model=other)
    hollow = LayeredIsotropicElastic([0.5, 0.8, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                     vs=[1.0, 0.0])
    with pytest.raises(ValueError, match="skeleton"):
        export_mfem(res, tmp_path / "hollow", model=hollow)
    assert not (tmp_path / "other.mesh").exists()


def test_an_offset_mesh_is_refused(tmp_path):
    res = build_offset_mesh(tmp_path / "offset", inner_radius=0.4,
                            outer_radius=1.0, offset=0.3, sizing=COARSE,
                            dimension=2)
    model = LayeredIsotropicElastic([0.0, 0.4, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                    vs=[1.0, 0.0])
    with pytest.raises(ValueError, match="not built from a geometry"):
        export_mfem(res, tmp_path / "nope", model=model)
